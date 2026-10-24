#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute a bounded, independently verified tool plan for a real LMM request.

This is a control plane for operator-declared *read-only* tools.  It does not
invent a response, hidden chain of thought, tool result, or model capability.
An external model (or a human/operator) may propose a typed plan; this program
then validates the plan against immutable schemas, executes each trusted tool
without a shell, validates an independent verifier's attestation, and emits a
provenance-bound EvidenceBundle that can be passed to ``lmm_gateway.py``.

Every invocation is tied to one LMM request digest and one registry digest.
Only content-addressed, typed results that pass a distinct verifier are exposed
to the next model call.  On a malformed plan, tool failure, verifier failure,
input swap, or publication collision, the operation fails closed.

The subprocess boundary is deliberately narrow (minimal environment, private
files, no shell, timeout, bounded output).  It is not a replacement for an OS
sandbox, container, network policy, human authorization, or real tool-quality
evaluation; those remain deployment responsibilities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from lmm_contract import (
    EVIDENCE_SCHEMA,
    MAX_JSON_BYTES,
    MAX_TEXT_BYTES,
    EvidenceBundle,
    FileIdentity,
    GenerationRequest,
    LMMContractError,
    canonical_json_bytes,
    file_identity,
    require,
    require_exact_fields,
    require_unchanged,
    sha256_bytes,
    validate_identifier,
    validate_sha256,
    validate_unicode_text,
)


REGISTRY_SCHEMA = "vegai-lmm-tool-registry/v1"
VALUE_SCHEMA = "vegai-lmm-tool-value-schema/v1"
PLAN_SCHEMA = "vegai-lmm-tool-plan/v1"
INVOCATION_SCHEMA = "vegai-lmm-tool-invocation/v1"
RESULT_SCHEMA = "vegai-lmm-tool-result/v1"
VERIFICATION_SCHEMA = "vegai-lmm-tool-verification/v1"
REPORT_SCHEMA = "vegai-lmm-tool-execution-report/v1"

MAX_TOOLS = 64
MAX_INVOCATIONS = 64
MAX_SCHEMA_BYTES = 1 * 1024 * 1024
MAX_SCHEMA_DEPTH = 8
MAX_SCHEMA_NODES = 256
MAX_FIELDS_PER_OBJECT = 64
MAX_ARRAY_ITEMS = 256
MAX_COMMAND_ARGUMENTS = 128
MAX_COMMAND_BYTES = 64 * 1024
MAX_TIMEOUT_SECONDS = 900
# Every verified result becomes one EvidenceItem.text, so keep it within the
# shared LMM evidence text boundary before any output artifact is published.
MAX_RESULT_BYTES = MAX_TEXT_BYTES
MAX_IMPLEMENTATION_BYTES = 64 * 1024 * 1024
MAX_INVOCATION_IDENTIFIER_BYTES = 100
_REF_KEY = "$vegai_ref"
_EXECUTOR_PLACEHOLDERS = frozenset(("{implementation}", "{input}", "{output}"))
_VERIFIER_PLACEHOLDERS = frozenset(("{implementation}", "{input}", "{result}", "{output}"))


class ToolOrchestrationError(LMMContractError):
    """A tool registry, plan, process, verification, or output contract failed."""


@dataclass(frozen=True, slots=True)
class ValueSchema:
    schema_id: str
    root: dict[str, Any]
    identity: FileIdentity


@dataclass(frozen=True, slots=True)
class ProgramSpec:
    implementation: FileIdentity
    command: tuple[str, ...]
    independence_group: str


@dataclass(frozen=True, slots=True)
class ToolSpec:
    tool_id: str
    source: str
    license: str
    input_schema: ValueSchema
    output_schema: ValueSchema
    executor: ProgramSpec
    verifier: ProgramSpec
    timeout_seconds: int
    maximum_result_bytes: int
    max_invocations_per_plan: int


@dataclass(frozen=True, slots=True)
class ToolRegistry:
    registry_id: str
    identity: FileIdentity
    tools: dict[str, ToolSpec]


@dataclass(frozen=True, slots=True)
class Invocation:
    invocation_id: str
    tool_id: str
    input_template: dict[str, Any]
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class ToolPlan:
    plan_id: str
    identity: FileIdentity
    request_sha256: str
    registry_sha256: str
    invocations: tuple[Invocation, ...]


@dataclass(frozen=True, slots=True)
class InvocationOutcome:
    invocation_id: str
    tool: ToolSpec
    input_sha256: str
    result: dict[str, Any]
    result_sha256: str
    verification_sha256: str
    elapsed_milliseconds: int


def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse JSON's otherwise silent last-key-wins behavior."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ToolOrchestrationError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def parse_strict_json(payload: bytes, label: str) -> object:
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ToolOrchestrationError(f"cannot decode {label} as UTF-8: {error}") from error
    try:
        return json.loads(decoded, object_pairs_hook=reject_duplicate_json_keys)
    except json.JSONDecodeError as error:
        raise ToolOrchestrationError(f"cannot parse {label} as JSON: {error}") from error


def load_json_mapping(path: Path, label: str, *, maximum_bytes: int) -> tuple[dict[str, Any], FileIdentity]:
    identity = file_identity(path, label, maximum_bytes=maximum_bytes)
    try:
        payload = path.resolve().read_bytes()
    except OSError as error:
        raise ToolOrchestrationError(f"cannot read {label} {path}: {error}") from error
    value = parse_strict_json(payload, label)
    require(isinstance(value, Mapping), f"{label} must contain one JSON object")
    require_unchanged(path, identity, label)
    return dict(value), identity


def parse_bounded_int(value: object, field: str, minimum: int, maximum: int) -> int:
    require(type(value) is int and minimum <= value <= maximum,
            f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def parse_finite_number(value: object, field: str) -> float:
    require(type(value) in {int, float} and not isinstance(value, bool), f"{field} must be a finite number")
    parsed = float(value)
    require(math.isfinite(parsed), f"{field} must be finite")
    return parsed


def validate_invocation_identifier(value: object, field: str) -> str:
    identifier = validate_identifier(value, field)
    require(len(identifier.encode("ascii")) <= MAX_INVOCATION_IDENTIFIER_BYTES,
            f"{field} exceeds {MAX_INVOCATION_IDENTIFIER_BYTES} bytes")
    return identifier


def resolve_registry_member(value: object, base: Path, field: str, expected_sha256: object) -> FileIdentity:
    """Resolve a non-symlink registry member beneath the immutable registry root."""
    relative = validate_unicode_text(value, field, maximum_bytes=4 * 1024)
    member = Path(relative)
    require(not member.is_absolute() and ".." not in member.parts and "." not in member.parts,
            f"{field} must be a relative path beneath the registry")
    lexical = base
    for component in member.parts:
        lexical = lexical / component
        require(not lexical.is_symlink(), f"{field} must not traverse a symlink: {lexical}")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as error:
        raise ToolOrchestrationError(f"{field} escapes the registry directory") from error
    identity = file_identity(resolved, field, maximum_bytes=MAX_IMPLEMENTATION_BYTES)
    require(identity.sha256 == validate_sha256(expected_sha256, f"{field}.sha256"),
            f"{field} does not match its declared SHA-256")
    return identity


def validate_schema_node(node: object, field: str, depth: int, nodes: list[int]) -> None:
    require(depth <= MAX_SCHEMA_DEPTH, f"{field} exceeds maximum schema depth {MAX_SCHEMA_DEPTH}")
    nodes[0] += 1
    require(nodes[0] <= MAX_SCHEMA_NODES, f"value schema exceeds {MAX_SCHEMA_NODES} nodes")
    require(isinstance(node, Mapping), f"{field} must be an object")
    kind = node.get("type")
    require(isinstance(kind, str), f"{field}.type must be a string")
    if kind == "string":
        require_exact_fields(node, {"type", "min_bytes", "max_bytes"}, field)
        minimum = parse_bounded_int(node["min_bytes"], f"{field}.min_bytes", 0, MAX_RESULT_BYTES)
        maximum = parse_bounded_int(node["max_bytes"], f"{field}.max_bytes", 0, MAX_RESULT_BYTES)
        require(minimum <= maximum, f"{field}.min_bytes must not exceed max_bytes")
    elif kind == "integer":
        require_exact_fields(node, {"type", "minimum", "maximum"}, field)
        minimum = parse_bounded_int(node["minimum"], f"{field}.minimum", -(2**63), 2**63 - 1)
        maximum = parse_bounded_int(node["maximum"], f"{field}.maximum", -(2**63), 2**63 - 1)
        require(minimum <= maximum, f"{field}.minimum must not exceed maximum")
    elif kind == "number":
        require_exact_fields(node, {"type", "minimum", "maximum"}, field)
        minimum = parse_finite_number(node["minimum"], f"{field}.minimum")
        maximum = parse_finite_number(node["maximum"], f"{field}.maximum")
        require(minimum <= maximum, f"{field}.minimum must not exceed maximum")
    elif kind == "boolean":
        require_exact_fields(node, {"type"}, field)
    elif kind == "array":
        require_exact_fields(node, {"type", "max_items", "items"}, field)
        parse_bounded_int(node["max_items"], f"{field}.max_items", 0, MAX_ARRAY_ITEMS)
        validate_schema_node(node["items"], f"{field}.items", depth + 1, nodes)
    elif kind == "object":
        require_exact_fields(node, {"type", "fields", "required", "additional_properties"}, field)
        fields = node["fields"]
        require(isinstance(fields, Mapping), f"{field}.fields must be an object")
        require(len(fields) <= MAX_FIELDS_PER_OBJECT, f"{field}.fields exceeds {MAX_FIELDS_PER_OBJECT} entries")
        for name, child in fields.items():
            validate_identifier(name, f"{field}.fields key")
            validate_schema_node(child, f"{field}.fields.{name}", depth + 1, nodes)
        required = node["required"]
        require(isinstance(required, list), f"{field}.required must be an array")
        parsed_required = [validate_identifier(item, f"{field}.required[{index}]") for index, item in enumerate(required)]
        require(len(set(parsed_required)) == len(parsed_required), f"{field}.required contains duplicates")
        require(set(parsed_required).issubset(set(fields)), f"{field}.required references unknown fields")
        require(node["additional_properties"] is False,
                f"{field}.additional_properties must be false for a typed tool contract")
    else:
        raise ToolOrchestrationError(f"{field}.type is unsupported: {kind!r}")


def load_value_schema(value: object, base: Path, field: str) -> ValueSchema:
    require(isinstance(value, Mapping), f"{field} must be an object")
    require_exact_fields(value, {"path", "sha256"}, field)
    identity = resolve_registry_member(value["path"], base, field, value["sha256"])
    mapping, parsed_identity = load_json_mapping(Path(identity.path), field, maximum_bytes=MAX_SCHEMA_BYTES)
    require(parsed_identity == identity, f"{field} changed while it was parsed")
    require_exact_fields(mapping, {"schema_version", "id", "root"}, field)
    require(mapping["schema_version"] == VALUE_SCHEMA,
            f"{field}.schema_version must be {VALUE_SCHEMA!r}")
    schema_id = validate_identifier(mapping["id"], f"{field}.id")
    validate_schema_node(mapping["root"], f"{field}.root", 0, [0])
    return ValueSchema(schema_id=schema_id, root=dict(mapping["root"]), identity=identity)


def parse_command(value: object, field: str, placeholders: frozenset[str]) -> tuple[str, ...]:
    require(isinstance(value, list) and value, f"{field} must be a nonempty string array")
    require(len(value) <= MAX_COMMAND_ARGUMENTS, f"{field} exceeds {MAX_COMMAND_ARGUMENTS} arguments")
    rendered: list[str] = []
    total = 0
    for index, item in enumerate(value):
        token = validate_unicode_text(item, f"{field}[{index}]", maximum_bytes=8 * 1024)
        total += len(token.encode("utf-8"))
        rendered.append(token)
    require(total <= MAX_COMMAND_BYTES, f"{field} exceeds {MAX_COMMAND_BYTES} UTF-8 bytes")
    for placeholder in placeholders:
        require(rendered.count(placeholder) == 1, f"{field} must contain exactly one {placeholder}")
    for token in rendered:
        if token not in placeholders:
            require("{" not in token and "}" not in token,
                    f"{field} supports only complete declared placeholder arguments")
    return tuple(rendered)


def parse_program(value: object, base: Path, field: str, placeholders: frozenset[str]) -> ProgramSpec:
    require(isinstance(value, Mapping), f"{field} must be an object")
    require_exact_fields(value, {"implementation", "command", "independence_group"}, field)
    implementation_spec = value["implementation"]
    require(isinstance(implementation_spec, Mapping), f"{field}.implementation must be an object")
    require_exact_fields(implementation_spec, {"path", "sha256"}, f"{field}.implementation")
    implementation = resolve_registry_member(
        implementation_spec["path"], base, f"{field}.implementation", implementation_spec["sha256"]
    )
    return ProgramSpec(
        implementation=implementation,
        command=parse_command(value["command"], f"{field}.command", placeholders),
        independence_group=validate_identifier(value["independence_group"], f"{field}.independence_group"),
    )


def parse_tool(value: object, base: Path, field: str) -> ToolSpec:
    require(isinstance(value, Mapping), f"{field} must be an object")
    expected = {
        "id", "source", "license", "side_effect", "input_schema", "output_schema",
        "executor", "verifier", "timeout_seconds", "maximum_result_bytes", "max_invocations_per_plan",
    }
    require_exact_fields(value, expected, field)
    require(value["side_effect"] == "none", f"{field}.side_effect must be 'none'; mutating tools are intentionally unsupported")
    executor = parse_program(value["executor"], base, f"{field}.executor", _EXECUTOR_PLACEHOLDERS)
    verifier = parse_program(value["verifier"], base, f"{field}.verifier", _VERIFIER_PLACEHOLDERS)
    require(executor.implementation.sha256 != verifier.implementation.sha256,
            f"{field} executor and verifier must use distinct implementation hashes")
    require(executor.independence_group != verifier.independence_group,
            f"{field} executor and verifier must declare distinct independence groups")
    return ToolSpec(
        tool_id=validate_identifier(value["id"], f"{field}.id"),
        source=validate_unicode_text(value["source"], f"{field}.source", maximum_bytes=8 * 1024),
        license=validate_unicode_text(value["license"], f"{field}.license", maximum_bytes=8 * 1024),
        input_schema=load_value_schema(value["input_schema"], base, f"{field}.input_schema"),
        output_schema=load_value_schema(value["output_schema"], base, f"{field}.output_schema"),
        executor=executor,
        verifier=verifier,
        timeout_seconds=parse_bounded_int(value["timeout_seconds"], f"{field}.timeout_seconds", 1, MAX_TIMEOUT_SECONDS),
        maximum_result_bytes=parse_bounded_int(value["maximum_result_bytes"], f"{field}.maximum_result_bytes", 1, MAX_RESULT_BYTES),
        max_invocations_per_plan=parse_bounded_int(value["max_invocations_per_plan"], f"{field}.max_invocations_per_plan", 1, MAX_INVOCATIONS),
    )


def load_registry(path: Path) -> ToolRegistry:
    mapping, identity = load_json_mapping(path, "tool registry", maximum_bytes=MAX_JSON_BYTES)
    require_exact_fields(mapping, {"schema_version", "id", "tools"}, "tool registry")
    require(mapping["schema_version"] == REGISTRY_SCHEMA,
            f"unsupported tool registry schema_version: {mapping['schema_version']!r}")
    raw_tools = mapping["tools"]
    require(isinstance(raw_tools, list) and raw_tools, "tool registry.tools must be a nonempty array")
    require(len(raw_tools) <= MAX_TOOLS, f"tool registry exceeds {MAX_TOOLS} tools")
    tools: dict[str, ToolSpec] = {}
    base = Path(identity.path).parent
    for index, raw_tool in enumerate(raw_tools):
        tool = parse_tool(raw_tool, base, f"tool registry.tools[{index}]")
        require(tool.tool_id not in tools, f"tool registry repeats tool id {tool.tool_id!r}")
        tools[tool.tool_id] = tool
    return ToolRegistry(
        registry_id=validate_identifier(mapping["id"], "tool registry.id"),
        identity=identity,
        tools=tools,
    )


def validate_template_shape(value: object, field: str, depth: int = 0) -> None:
    require(depth <= MAX_SCHEMA_DEPTH + 2, f"{field} exceeds maximum JSON template depth")
    if value is None or type(value) in {str, int, float, bool}:
        if type(value) is float:
            require(math.isfinite(value), f"{field} must not contain a non-finite number")
        return
    if isinstance(value, list):
        require(len(value) <= MAX_ARRAY_ITEMS, f"{field} exceeds {MAX_ARRAY_ITEMS} items")
        for index, item in enumerate(value):
            validate_template_shape(item, f"{field}[{index}]", depth + 1)
        return
    require(isinstance(value, Mapping), f"{field} must be JSON-compatible")
    require(len(value) <= MAX_FIELDS_PER_OBJECT, f"{field} exceeds {MAX_FIELDS_PER_OBJECT} keys")
    if set(value) == {_REF_KEY}:
        require(isinstance(value[_REF_KEY], str), f"{field}.{_REF_KEY} must be a string")
        require(len(value[_REF_KEY].encode("utf-8")) <= 1_024, f"{field}.{_REF_KEY} is too long")
        return
    for key, item in value.items():
        require(isinstance(key, str), f"{field} keys must be strings")
        validate_template_shape(item, f"{field}.{key}", depth + 1)


def parse_invocation(value: object, index: int, registry: ToolRegistry) -> Invocation:
    field = f"tool plan.invocations[{index}]"
    require(isinstance(value, Mapping), f"{field} must be an object")
    require_exact_fields(value, {"id", "tool_id", "input", "idempotency_key"}, field)
    invocation_id = validate_invocation_identifier(value["id"], f"{field}.id")
    tool_id = validate_identifier(value["tool_id"], f"{field}.tool_id")
    require(tool_id in registry.tools, f"{field}.tool_id is absent from the immutable registry")
    require(isinstance(value["input"], Mapping), f"{field}.input must be an object")
    validate_template_shape(value["input"], f"{field}.input")
    return Invocation(
        invocation_id=invocation_id,
        tool_id=tool_id,
        input_template=dict(value["input"]),
        idempotency_key=validate_sha256(value["idempotency_key"], f"{field}.idempotency_key"),
    )


def load_plan(path: Path, registry: ToolRegistry, request: GenerationRequest) -> ToolPlan:
    mapping, identity = load_json_mapping(path, "tool plan", maximum_bytes=MAX_JSON_BYTES)
    require_exact_fields(mapping, {"schema_version", "id", "request_sha256", "registry_sha256", "invocations"}, "tool plan")
    require(mapping["schema_version"] == PLAN_SCHEMA, f"unsupported tool plan schema_version: {mapping['schema_version']!r}")
    request_sha256 = validate_sha256(mapping["request_sha256"], "tool plan.request_sha256")
    registry_sha256 = validate_sha256(mapping["registry_sha256"], "tool plan.registry_sha256")
    require(request_sha256 == request.digest(), "tool plan.request_sha256 does not bind the supplied LMM request")
    require(registry_sha256 == registry.identity.sha256, "tool plan.registry_sha256 does not bind the supplied registry")
    raw_invocations = mapping["invocations"]
    require(isinstance(raw_invocations, list) and raw_invocations, "tool plan.invocations must be a nonempty array")
    require(len(raw_invocations) <= MAX_INVOCATIONS, f"tool plan exceeds {MAX_INVOCATIONS} invocations")
    invocations = tuple(parse_invocation(item, index, registry) for index, item in enumerate(raw_invocations))
    ids = [item.invocation_id for item in invocations]
    keys = [item.idempotency_key for item in invocations]
    require(len(set(ids)) == len(ids), "tool plan invocation IDs must be unique")
    require(len(set(keys)) == len(keys), "tool plan idempotency keys must be unique")
    counts: dict[str, int] = {}
    for invocation in invocations:
        counts[invocation.tool_id] = counts.get(invocation.tool_id, 0) + 1
    for tool_id, count in counts.items():
        require(count <= registry.tools[tool_id].max_invocations_per_plan,
                f"tool plan invokes {tool_id!r} more than its declared per-plan limit")
    return ToolPlan(
        plan_id=validate_identifier(mapping["id"], "tool plan.id"),
        identity=identity,
        request_sha256=request_sha256,
        registry_sha256=registry_sha256,
        invocations=invocations,
    )


def decode_pointer_token(value: str, field: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character != "~":
            result.append(character)
            index += 1
            continue
        require(index + 1 < len(value) and value[index + 1] in {"0", "1"},
                f"{field} has an invalid JSON Pointer escape")
        result.append("~" if value[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


def resolve_reference(reference: str, outcomes: Mapping[str, InvocationOutcome], field: str) -> object:
    separator = reference.find("#")
    require(separator > 0, f"{field} must have form '<previous-invocation-id>#/json/pointer'")
    invocation_id = validate_invocation_identifier(reference[:separator], f"{field} invocation id")
    require(invocation_id in outcomes, f"{field} references an unknown, current, or future invocation")
    pointer = reference[separator + 1:]
    current: object = outcomes[invocation_id].result["output"]
    if pointer == "":
        return json.loads(canonical_json_bytes(current).decode("utf-8"))
    require(pointer.startswith("/"), f"{field} JSON Pointer must begin with '/' after '#'")
    for raw_token in pointer[1:].split("/"):
        token = decode_pointer_token(raw_token, field)
        if isinstance(current, Mapping):
            require(token in current, f"{field} JSON Pointer references an absent object key")
            current = current[token]
        elif isinstance(current, list):
            require(token.isdecimal() and (token == "0" or not token.startswith("0")),
                    f"{field} JSON Pointer array index is invalid")
            index = int(token)
            require(index < len(current), f"{field} JSON Pointer array index is out of range")
            current = current[index]
        else:
            raise ToolOrchestrationError(f"{field} JSON Pointer traverses a scalar value")
    return json.loads(canonical_json_bytes(current).decode("utf-8"))


def resolve_template(value: object, outcomes: Mapping[str, InvocationOutcome], field: str) -> object:
    if isinstance(value, Mapping):
        if set(value) == {_REF_KEY}:
            reference = value[_REF_KEY]
            assert isinstance(reference, str)
            return resolve_reference(reference, outcomes, f"{field}.{_REF_KEY}")
        return {key: resolve_template(item, outcomes, f"{field}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_template(item, outcomes, f"{field}[{index}]") for index, item in enumerate(value)]
    return value


def validate_value(value: object, schema: Mapping[str, Any], field: str) -> object:
    kind = schema["type"]
    if kind == "string":
        rendered = validate_unicode_text(value, field, maximum_bytes=int(schema["max_bytes"]), allow_empty=True)
        require(len(rendered.encode("utf-8")) >= int(schema["min_bytes"]), f"{field} is shorter than its contract")
        return rendered
    if kind == "integer":
        require(type(value) is int and int(schema["minimum"]) <= value <= int(schema["maximum"]),
                f"{field} is outside its integer contract")
        return value
    if kind == "number":
        parsed = parse_finite_number(value, field)
        require(float(schema["minimum"]) <= parsed <= float(schema["maximum"]),
                f"{field} is outside its number contract")
        return parsed if type(value) is float else value
    if kind == "boolean":
        require(type(value) is bool, f"{field} must be boolean")
        return value
    if kind == "array":
        require(isinstance(value, list) and len(value) <= int(schema["max_items"]), f"{field} violates its array contract")
        return [validate_value(item, schema["items"], f"{field}[{index}]") for index, item in enumerate(value)]
    assert kind == "object"
    require(isinstance(value, Mapping), f"{field} must be an object")
    fields = schema["fields"]
    unknown = sorted(str(key) for key in value if key not in fields)
    missing = sorted(set(schema["required"]).difference(value))
    require(not unknown, f"{field} contains unsupported fields: {', '.join(unknown)}")
    require(not missing, f"{field} is missing required fields: {', '.join(missing)}")
    return {
        key: validate_value(item, fields[key], f"{field}.{key}")
        for key, item in value.items()
    }


def write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def safe_environment() -> dict[str, str]:
    """Remove ambient credentials while retaining only basic executable discovery."""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "PYTHONUTF8": "1",
    }
    for name in ("SYSTEMROOT", "WINDIR"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def expand_command(command: Sequence[str], *, implementation: Path, input_path: Path, result_path: Path | None, output_path: Path) -> list[str]:
    replacements = {
        "{implementation}": str(implementation),
        "{input}": str(input_path),
        "{output}": str(output_path),
    }
    if result_path is not None:
        replacements["{result}"] = str(result_path)
    return [replacements.get(item, item) for item in command]


def run_program(command: Sequence[str], *, timeout_seconds: int, cwd: Path, label: str) -> int:
    """Run a registered program without a shell or retained child output."""
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(cwd),
            env=safe_environment(),
            close_fds=True,
            start_new_session=(os.name == "posix"),
        )
    except OSError as error:
        raise ToolOrchestrationError(f"cannot start {label}: {error}") from error
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
        else:
            process.kill()
        process.wait()
        raise ToolOrchestrationError(f"{label} exceeded its {timeout_seconds}-second timeout") from error


def load_private_mapping(path: Path, label: str, maximum_bytes: int) -> tuple[dict[str, Any], FileIdentity]:
    require(path.exists() and path.is_file() and not path.is_symlink(), f"{label} did not create a regular private output file")
    return load_json_mapping(path, label, maximum_bytes=maximum_bytes)


def parse_tool_result(value: Mapping[str, Any], invocation: Invocation, tool: ToolSpec) -> dict[str, Any]:
    require_exact_fields(value, {"schema_version", "id", "tool_id", "output"}, "tool result")
    require(value["schema_version"] == RESULT_SCHEMA, "tool result has unsupported schema_version")
    require(value["id"] == invocation.invocation_id, "tool result id does not match its invocation")
    require(value["tool_id"] == tool.tool_id, "tool result tool_id does not match its invocation")
    validated_output = validate_value(value["output"], tool.output_schema.root, "tool result.output")
    return {
        "schema_version": RESULT_SCHEMA,
        "id": invocation.invocation_id,
        "tool_id": tool.tool_id,
        "output": validated_output,
    }


def parse_verification(value: Mapping[str, Any], invocation: Invocation, tool: ToolSpec, result_sha256: str) -> None:
    require_exact_fields(value, {"schema_version", "id", "tool_id", "result_sha256", "verdict"}, "tool verification")
    require(value["schema_version"] == VERIFICATION_SCHEMA, "tool verification has unsupported schema_version")
    require(value["id"] == invocation.invocation_id, "tool verification id does not match its invocation")
    require(value["tool_id"] == tool.tool_id, "tool verification tool_id does not match its invocation")
    require(value["result_sha256"] == result_sha256, "tool verification does not bind the exact tool result")
    require(value["verdict"] == "pass", "independent tool verifier did not return verdict 'pass'")


def ensure_output_path(path: Path, label: str, protected: set[str]) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except OSError as error:
        raise ToolOrchestrationError(f"cannot resolve {label} path {path}: {error}") from error
    require(str(resolved) not in protected, f"{label} must not overwrite an immutable input")
    require(not resolved.exists(), f"{label} already exists; refusing to replace it")
    return resolved


def write_json_no_replace(path: Path, value: Mapping[str, Any], *, maximum_bytes: int = MAX_JSON_BYTES) -> FileIdentity:
    encoded = canonical_json_bytes(value) + b"\n"
    require(len(encoded) <= maximum_bytes, f"published output exceeds {maximum_bytes} bytes")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ToolOrchestrationError(f"refusing to replace existing output {path}") from error
    except OSError as error:
        raise ToolOrchestrationError(f"cannot create output {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return file_identity(path, "published output", maximum_bytes=MAX_JSON_BYTES)


def registered_identities(registry: ToolRegistry) -> tuple[FileIdentity, ...]:
    identities: list[FileIdentity] = [registry.identity]
    for tool in registry.tools.values():
        identities.extend((
            tool.input_schema.identity,
            tool.output_schema.identity,
            tool.executor.implementation,
            tool.verifier.implementation,
        ))
    return tuple(identities)


def ensure_inputs_unchanged(registry: ToolRegistry, plan: ToolPlan, request_identity: FileIdentity) -> None:
    require_unchanged(Path(request_identity.path), request_identity, "LMM request")
    require_unchanged(Path(plan.identity.path), plan.identity, "tool plan")
    for identity in registered_identities(registry):
        require_unchanged(Path(identity.path), identity, "tool registry member")


def execute(
    registry_path: Path,
    request_path: Path,
    plan_path: Path,
    evidence_output_path: Path,
    report_output_path: Path,
) -> dict[str, Any]:
    request_mapping, request_identity = load_json_mapping(request_path, "LMM request", maximum_bytes=MAX_JSON_BYTES)
    request = GenerationRequest.from_mapping(request_mapping)
    registry = load_registry(registry_path)
    plan = load_plan(plan_path, registry, request)

    protected = {request_identity.path, registry.identity.path, plan.identity.path}
    protected.update(identity.path for identity in registered_identities(registry))
    evidence_output = ensure_output_path(evidence_output_path, "tool evidence output", protected)
    report_output = ensure_output_path(report_output_path, "tool report output", protected | {str(evidence_output)})

    outcomes: dict[str, InvocationOutcome] = {}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vegai-tool-plan-") as temporary_name:
        temporary = Path(temporary_name)
        for invocation in plan.invocations:
            tool = registry.tools[invocation.tool_id]
            resolved = resolve_template(invocation.input_template, outcomes, f"tool plan invocation {invocation.invocation_id}.input")
            validated = validate_value(resolved, tool.input_schema.root, f"tool plan invocation {invocation.invocation_id}.input")
            require(isinstance(validated, Mapping), f"tool {tool.tool_id} input schema root must produce an object")
            input_payload: dict[str, Any] = {
                "schema_version": INVOCATION_SCHEMA,
                "id": invocation.invocation_id,
                "tool_id": tool.tool_id,
                "request_sha256": request.digest(),
                "idempotency_key": invocation.idempotency_key,
                "input": dict(validated),
            }
            input_path = temporary / f"{invocation.invocation_id}.input.json"
            executor_output_path = temporary / f"{invocation.invocation_id}.result.json"
            write_private_json(input_path, input_payload)
            began = time.monotonic()
            executor_code = run_program(
                expand_command(
                    tool.executor.command,
                    implementation=Path(tool.executor.implementation.path),
                    input_path=input_path,
                    result_path=None,
                    output_path=executor_output_path,
                ),
                timeout_seconds=tool.timeout_seconds,
                cwd=temporary,
                label=f"tool executor {tool.tool_id!r} for invocation {invocation.invocation_id!r}",
            )
            require(executor_code == 0, f"tool executor {tool.tool_id!r} exited with status {executor_code}")
            raw_result, _ = load_private_mapping(executor_output_path, "tool executor result", tool.maximum_result_bytes)
            result = parse_tool_result(raw_result, invocation, tool)
            result_sha256 = sha256_bytes(canonical_json_bytes(result))

            verifier_input = temporary / f"{invocation.invocation_id}.verify-input.json"
            verifier_result = temporary / f"{invocation.invocation_id}.verify-result.json"
            verifier_output = temporary / f"{invocation.invocation_id}.verification.json"
            write_private_json(verifier_input, input_payload)
            write_private_json(verifier_result, result)
            verifier_input_identity = file_identity(verifier_input, "verifier input", maximum_bytes=tool.maximum_result_bytes)
            verifier_result_identity = file_identity(verifier_result, "verifier result", maximum_bytes=tool.maximum_result_bytes)
            verifier_code = run_program(
                expand_command(
                    tool.verifier.command,
                    implementation=Path(tool.verifier.implementation.path),
                    input_path=verifier_input,
                    result_path=verifier_result,
                    output_path=verifier_output,
                ),
                timeout_seconds=tool.timeout_seconds,
                cwd=temporary,
                label=f"tool verifier {tool.tool_id!r} for invocation {invocation.invocation_id!r}",
            )
            require(verifier_code == 0, f"tool verifier {tool.tool_id!r} exited with status {verifier_code}")
            verification, verification_identity = load_private_mapping(verifier_output, "tool verification", tool.maximum_result_bytes)
            parse_verification(verification, invocation, tool, result_sha256)
            require_unchanged(verifier_input, verifier_input_identity, "verifier input")
            require_unchanged(verifier_result, verifier_result_identity, "verifier result")
            outcomes[invocation.invocation_id] = InvocationOutcome(
                invocation_id=invocation.invocation_id,
                tool=tool,
                input_sha256=sha256_bytes(canonical_json_bytes(input_payload)),
                result=result,
                result_sha256=result_sha256,
                verification_sha256=verification_identity.sha256,
                elapsed_milliseconds=max(0, round((time.monotonic() - began) * 1000)),
            )

    ensure_inputs_unchanged(registry, plan, request_identity)
    evidence_items: list[dict[str, object]] = []
    report_invocations: list[dict[str, object]] = []
    for invocation in plan.invocations:
        outcome = outcomes[invocation.invocation_id]
        evidence_id = f"tool-{invocation.invocation_id}"
        validate_identifier(evidence_id, "derived tool evidence id")
        output_text = canonical_json_bytes(outcome.result["output"]).decode("utf-8")
        evidence_items.append({
            "id": evidence_id,
            "text": output_text,
            "source": outcome.tool.source,
            "license": outcome.tool.license,
            "sha256": sha256_bytes(output_text.encode("utf-8")),
            "score": 1.0,
        })
        report_invocations.append({
            "id": invocation.invocation_id,
            "tool_id": outcome.tool.tool_id,
            "input_sha256": outcome.input_sha256,
            "result_sha256": outcome.result_sha256,
            "verification_sha256": outcome.verification_sha256,
            "executor_implementation_sha256": outcome.tool.executor.implementation.sha256,
            "verifier_implementation_sha256": outcome.tool.verifier.implementation.sha256,
            "elapsed_milliseconds": outcome.elapsed_milliseconds,
            "evidence_id": evidence_id,
        })
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA,
        "query_sha256": request.digest(),
        "index": {
            "kind": "vegai-lmm-tool-orchestrator/v1",
            "registry_sha256": registry.identity.sha256,
            "plan_sha256": plan.identity.sha256,
            "request_sha256": request.digest(),
        },
        "items": evidence_items,
    }
    EvidenceBundle.from_mapping(evidence)
    evidence_identity = write_json_no_replace(evidence_output, evidence)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "outcome": "completed",
        "registry": {
            "id": registry.registry_id,
            "identity": registry.identity.to_mapping(redact_path=True),
        },
        "plan": {
            "id": plan.plan_id,
            "identity": plan.identity.to_mapping(redact_path=True),
            "request_sha256": plan.request_sha256,
        },
        "request": {
            "identity": request_identity.to_mapping(redact_path=True),
            "logical_sha256": request.digest(),
        },
        "execution": {
            "strategy": "declared-order-typed-reference/v1",
            "side_effect_policy": "read-only tools only",
            "invocations": report_invocations,
            "total_elapsed_milliseconds": max(0, round((time.monotonic() - started) * 1000)),
        },
        "evidence_output": {
            "sha256": evidence_identity.sha256,
            "bytes": evidence_identity.bytes,
            "schema_version": EVIDENCE_SCHEMA,
        },
        "limitations": [
            "A verifier pass binds the declared tool result, not the factual correctness of an arbitrary external tool or source.",
            "This process boundary is not an OS sandbox; production deployments require container, network, identity, and authorization controls.",
            "The orchestrator does not generate a final answer, adjudicate model claims, or prove general reasoning capability.",
        ],
    }
    try:
        write_json_no_replace(report_output, report)
    except BaseException:
        try:
            if file_identity(evidence_output, "published tool evidence", maximum_bytes=MAX_JSON_BYTES) == evidence_identity:
                evidence_output.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    execute_parser = subcommands.add_parser("execute", help="execute a verified read-only tool plan and publish an EvidenceBundle")
    execute_parser.add_argument("--registry", required=True, type=Path)
    execute_parser.add_argument("--request", required=True, type=Path)
    execute_parser.add_argument("--plan", required=True, type=Path)
    execute_parser.add_argument("--evidence-output", required=True, type=Path)
    execute_parser.add_argument("--report-output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "execute":
            report = execute(
                arguments.registry,
                arguments.request,
                arguments.plan,
                arguments.evidence_output,
                arguments.report_output,
            )
            print(json.dumps({"outcome": report["outcome"], "invocations": len(report["execution"]["invocations"])}, sort_keys=True))
            return 0
        raise ToolOrchestrationError("unsupported command")
    except (ToolOrchestrationError, LMMContractError, OSError, ValueError) as error:
        print(f"LMM tool orchestrator: ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
