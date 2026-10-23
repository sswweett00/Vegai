#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run held-out, provenance-aware quality gates against a real LMM gateway.

This evaluator deliberately does not generate an answer template or use an LLM
judge. Each operator-owned JSONL case carries a real request, the evidence that
will be supplied to the backend, and deterministic requirements for
model-emitted citation/abstention markers. The evaluator invokes an explicit
``lmm_gateway.py --print-json`` command vector without a shell, retains no raw
prompt/response in its report, and can fail a release gate.

Passing proves only the configured model run obeyed the chosen held-out marker
contract.  It is not a proof of factuality, fluency, safety, universal language
quality, benchmark superiority, or multimodal understanding.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import hmac
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

from lmm_contract import (
    GENERATION_REPORT_SCHEMA,
    MAX_EVIDENCE_ITEMS,
    MAX_JSON_BYTES,
    MAX_TEXT_BYTES,
    EvidenceBundle,
    FileIdentity,
    GenerationRequest,
    LMMContractError,
    canonical_json_bytes,
    file_identity,
    parse_grounding_markers,
    require,
    require_unchanged,
    sha256_bytes,
    validate_identifier,
    validate_sha256,
    validate_unicode_text,
    write_json_atomic,
)


CASE_SCHEMA = "vegai-lmm-quality-case/v1"
REPORT_SCHEMA = "vegai-lmm-quality-report/v1"
MAX_CASE_LINE_BYTES = 4 * 1024 * 1024
MAX_CASESET_BYTES = 512 * 1024 * 1024
MAX_CASES = 10_000
MAX_HMAC_KEY_BYTES = 4 * 1024
MAX_GATEWAY_STDOUT_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 180.0
REQUEST_PLACEHOLDER = "{request}"
EVIDENCE_PLACEHOLDER = "{evidence}"


class LMMQualityEvaluationError(LMMContractError):
    """A held-out quality input, gateway run, or release gate is invalid."""


@dataclass(frozen=True, slots=True)
class QualityCase:
    number: int
    case_id: str
    request: GenerationRequest
    evidence: EvidenceBundle
    required_citation_ids: tuple[str, ...]
    allowed_citation_ids: tuple[str, ...]
    require_abstention: bool

    @property
    def evidence_ids(self) -> set[str]:
        return {item.document_id for item in self.evidence.items}


@dataclass(frozen=True, slots=True)
class GatewayObservation:
    response: str
    response_sha256: str
    response_bytes: int
    generation_report_sha256: str
    model_sha256: str
    model_manifest_sha256: str
    runtime_identity_sha256: str
    backend_name: str
    elapsed_seconds: float


def exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    unknown = sorted(str(key) for key in value if key not in expected)
    missing = sorted(expected.difference(value))
    require(not unknown, f"{field} contains unsupported fields: {', '.join(unknown)}")
    require(not missing, f"{field} is missing required fields: {', '.join(missing)}")


def parse_probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def parse_identifier_array(value: object, field: str) -> tuple[str, ...]:
    require(isinstance(value, list), f"{field} must be an array")
    require(len(value) <= MAX_EVIDENCE_ITEMS, f"{field} exceeds {MAX_EVIDENCE_ITEMS} items")
    parsed = tuple(validate_identifier(item, f"{field}[{index}]") for index, item in enumerate(value))
    require(len(set(parsed)) == len(parsed), f"{field} contains duplicate IDs")
    return parsed


def parse_case(value: object, number: int) -> QualityCase:
    require(isinstance(value, Mapping), f"quality case {number} must be an object")
    expected = {
        "schema_version",
        "id",
        "request",
        "evidence",
        "required_citation_ids",
        "allowed_citation_ids",
        "require_abstention",
    }
    exact_fields(value, expected, f"quality case {number}")
    require(value["schema_version"] == CASE_SCHEMA,
            f"quality case {number} has unsupported schema_version: {value['schema_version']!r}")
    case_id = validate_identifier(value["id"], f"quality case {number}.id")
    request = GenerationRequest.from_mapping(value["request"])
    evidence = EvidenceBundle.from_mapping(value["evidence"])
    required = parse_identifier_array(value["required_citation_ids"], f"quality case {number}.required_citation_ids")
    allowed = parse_identifier_array(value["allowed_citation_ids"], f"quality case {number}.allowed_citation_ids")
    require(type(value["require_abstention"]) is bool,
            f"quality case {number}.require_abstention must be boolean")
    evidence_ids = {item.document_id for item in evidence.items}
    require(set(required).issubset(allowed),
            f"quality case {number}.required_citation_ids must be a subset of allowed_citation_ids")
    require(set(allowed).issubset(evidence_ids),
            f"quality case {number}.allowed_citation_ids must be supplied by quality case evidence")
    if value["require_abstention"]:
        require(not required and not allowed,
                f"quality case {number} cannot require abstention and allow citations simultaneously")
    return QualityCase(
        number=number,
        case_id=case_id,
        request=request,
        evidence=evidence,
        required_citation_ids=required,
        allowed_citation_ids=allowed,
        require_abstention=value["require_abstention"],
    )


def iter_cases(path: Path, maximum_cases: int) -> Iterator[QualityCase]:
    """Read strict LF JSONL from one stable source snapshot."""
    require(maximum_cases > 0, "--max-cases must be positive")
    source = path.resolve()
    identity = file_identity(source, "quality case set", maximum_bytes=MAX_CASESET_BYTES)
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    identifiers: set[str] = set()
    yielded = 0
    expanded_bytes = 0
    try:
        with opener(source, "rb") as handle:
            number = 0
            limit = MAX_CASE_LINE_BYTES + 2
            while True:
                raw = handle.readline(limit)
                if not raw:
                    break
                number += 1
                has_lf = raw.endswith(b"\n")
                payload_bytes = len(raw) - 1 if has_lf else len(raw)
                overlong = payload_bytes > MAX_CASE_LINE_BYTES or (not has_lf and len(raw) == limit)
                if overlong and not has_lf:
                    while raw and not raw.endswith(b"\n"):
                        raw = handle.readline(limit)
                require(not overlong, f"quality case line {number} exceeds {MAX_CASE_LINE_BYTES} bytes")
                require(has_lf, f"quality case line {number} must terminate with LF")
                expanded_bytes += len(raw)
                require(expanded_bytes <= MAX_CASESET_BYTES,
                        f"quality case set exceeds {MAX_CASESET_BYTES} uncompressed bytes")
                raw = raw[:-1]
                require(not raw.endswith(b"\r"), f"quality case line {number} uses CRLF; strict LF is required")
                require(bool(raw.strip()), f"quality case line {number} is blank")
                try:
                    decoded = raw.decode("utf-8")
                    parsed = json.loads(decoded)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise LMMQualityEvaluationError(f"quality case line {number} is invalid JSON: {error}") from error
                case = parse_case(parsed, number)
                require(case.case_id not in identifiers, "quality case IDs must be unique")
                identifiers.add(case.case_id)
                yielded += 1
                require(yielded <= maximum_cases,
                        f"quality case set exceeds --max-cases={maximum_cases}")
                yield case
    except (OSError, EOFError) as error:
        raise LMMQualityEvaluationError(f"cannot read quality case set {source}: {error}") from error
    require_unchanged(source, identity, "quality case set")


def require_option_placeholder_binding(command: Sequence[str], option: str, placeholder: str) -> None:
    """Require a literal placeholder to be the sole value of one gateway option."""
    positions = [index for index, item in enumerate(command) if item == placeholder]
    require(len(positions) == 1, f"quality gateway command must contain exactly one literal {placeholder!r}")
    position = positions[0]
    require(position > 0 and command[position - 1] == option,
            f"quality gateway {placeholder!r} must be the direct value of {option}")
    require(not any(item.startswith(f"{option}=") for item in command),
            f"quality gateway command must not use an additional {option}=... option")
    require(command.count(option) == 1, f"quality gateway command must contain exactly one {option}")


def load_command(path: Path) -> tuple[list[str], FileIdentity]:
    identity = file_identity(path, "quality gateway command", maximum_bytes=MAX_JSON_BYTES)
    try:
        value = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LMMQualityEvaluationError(f"cannot parse quality gateway command {path}: {error}") from error
    require(isinstance(value, list) and value and all(isinstance(item, str) and item for item in value),
            "quality gateway command must be a nonempty JSON string array")
    command = list(value)
    for item in command:
        require(
            (REQUEST_PLACEHOLDER not in item or item == REQUEST_PLACEHOLDER)
            and (EVIDENCE_PLACEHOLDER not in item or item == EVIDENCE_PLACEHOLDER),
            "quality gateway placeholders must occupy complete command arguments",
        )
    require_option_placeholder_binding(command, "--request", REQUEST_PLACEHOLDER)
    require_option_placeholder_binding(command, "--evidence", EVIDENCE_PLACEHOLDER)
    require(command.count("--print-json") == 1,
            "quality gateway command must include exactly one --print-json so output provenance is machine-readable")
    require(not any(item == "--output" or item.startswith("--output=") for item in command),
            "quality gateway command must not include --output; evaluator owns its private temporary case files")
    require(not any(item == "--include-response-in-report" or item.startswith("--include-response-in-report=") for item in command),
            "quality gateway command must not include --include-response-in-report; evaluator avoids persisting held-out output")
    require_unchanged(path, identity, "quality gateway command")
    return command, identity


def load_hmac_key(path: Path | None) -> tuple[bytes | None, FileIdentity | None]:
    if path is None:
        return None, None
    identity = file_identity(path, "quality case HMAC key", maximum_bytes=MAX_HMAC_KEY_BYTES)
    try:
        with path.resolve().open("rb") as handle:
            value = handle.read(MAX_HMAC_KEY_BYTES + 1)
    except OSError as error:
        raise LMMQualityEvaluationError(f"cannot read quality case HMAC key {path}: {error}") from error
    require(bool(value), "quality case HMAC key must not be empty")
    require(len(value) <= MAX_HMAC_KEY_BYTES,
            f"quality case HMAC key exceeds {MAX_HMAC_KEY_BYTES} bytes")
    require_unchanged(path, identity, "quality case HMAC key")
    return value, identity


def private_case_id(case_id: str, key: bytes | None) -> str:
    encoded = case_id.encode("ascii")
    return hashlib.sha256(encoded).hexdigest() if key is None else hmac.new(key, encoded, hashlib.sha256).hexdigest()


def expand_command(command: Sequence[str], request_path: Path, evidence_path: Path) -> list[str]:
    return [
        str(request_path) if item == REQUEST_PLACEHOLDER else str(evidence_path) if item == EVIDENCE_PLACEHOLDER else item
        for item in command
    ]


def stable_runtime_identity(report: Mapping[str, Any]) -> tuple[str, str]:
    """Return redaction-safe digests that bind all non-case gateway dependencies."""
    identities = report.get("identities")
    require(isinstance(identities, Mapping), "quality gateway report has no identities object")
    stable_fields = (
        "gateway_tool",
        "model_manifest",
        "tokenizer",
        "asset_registry",
        "worker_command_configuration",
        "model_artifacts",
    )
    missing = [field for field in stable_fields if field not in identities]
    require(not missing, "quality gateway report identities is missing fields: " + ", ".join(missing))
    model_manifest = identities["model_manifest"]
    require(isinstance(model_manifest, Mapping), "quality gateway report model manifest identity is invalid")
    model_manifest_sha256 = validate_sha256(model_manifest.get("sha256"), "quality gateway model manifest sha256")
    stable = {field: identities[field] for field in stable_fields}
    return model_manifest_sha256, sha256_bytes(canonical_json_bytes(stable))


def run_gateway_case(
    command: Sequence[str],
    request_path: Path,
    evidence_path: Path,
    timeout_seconds: float,
) -> GatewayObservation:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            expand_command(command, request_path, evidence_path),
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise LMMQualityEvaluationError(f"quality gateway case exceeded {timeout_seconds} seconds") from error
    except OSError as error:
        raise LMMQualityEvaluationError(f"cannot launch quality gateway command: {error}") from error
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise LMMQualityEvaluationError(
            "quality gateway command failed; "
            f"stdout-sha256={sha256_bytes(completed.stdout)}, stderr-sha256={sha256_bytes(completed.stderr)}, "
            f"exit={completed.returncode}"
        )
    require(len(completed.stdout) <= MAX_GATEWAY_STDOUT_BYTES,
            f"quality gateway stdout exceeds {MAX_GATEWAY_STDOUT_BYTES} bytes")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LMMQualityEvaluationError(f"quality gateway did not emit UTF-8 --print-json output: {error}") from error
    require(isinstance(payload, Mapping), "quality gateway output must be an object")
    exact_fields(payload, {"response", "report"}, "quality gateway output")
    response = validate_unicode_text(payload["response"], "quality gateway response", maximum_bytes=MAX_TEXT_BYTES)
    report = payload["report"]
    require(isinstance(report, Mapping), "quality gateway report must be an object")
    require(report.get("schema_version") == GENERATION_REPORT_SCHEMA,
            "quality gateway report has an unsupported schema")
    report_response = report.get("response")
    require(isinstance(report_response, Mapping), "quality gateway report has no response object")
    expected_digest = sha256_bytes(response.encode("utf-8"))
    require(report_response.get("sha256") == expected_digest,
            "quality gateway response does not match its provenance report hash")
    model = report.get("model")
    backend = report.get("backend")
    grounding = report.get("grounding")
    require(isinstance(model, Mapping) and isinstance(backend, Mapping),
            "quality gateway report has no model/backend provenance")
    require(isinstance(grounding, Mapping), "quality gateway report has no grounding observation")
    observed_markers = parse_grounding_markers(response)
    require(grounding.get("citation_ids") == list(observed_markers.citation_ids)
            and grounding.get("abstention_markers") == observed_markers.abstention_markers
            and grounding.get("malformed_citation_marker") == observed_markers.malformed_citation_marker
            and grounding.get("malformed_abstention_marker") == observed_markers.malformed_abstention_marker
            and grounding.get("substantive_response_bytes") == observed_markers.substantive_response_bytes,
            "quality gateway grounding observation does not match its response")
    backend_name = backend.get("name")
    require(isinstance(backend_name, str) and backend_name,
            "quality gateway report backend.name is invalid")
    model_manifest_sha256, runtime_identity_sha256 = stable_runtime_identity(report)
    return GatewayObservation(
        response=response,
        response_sha256=expected_digest,
        response_bytes=len(response.encode("utf-8")),
        generation_report_sha256=sha256_bytes(canonical_json_bytes(report)),
        model_sha256=sha256_bytes(canonical_json_bytes(model)),
        model_manifest_sha256=model_manifest_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        backend_name=backend_name,
        elapsed_seconds=elapsed,
    )


def evaluate_case(case: QualityCase, observation: GatewayObservation, case_key: bytes | None) -> tuple[dict[str, object], dict[str, int]]:
    markers = parse_grounding_markers(observation.response)
    citations = set(markers.citation_ids)
    required = set(case.required_citation_ids)
    allowed = set(case.allowed_citation_ids)
    missing = sorted(required.difference(citations))
    invalid = sorted(citations.difference(allowed))
    failures: list[str] = []
    if markers.malformed_citation_marker:
        failures.append("malformed_citation_marker")
    if markers.malformed_abstention_marker:
        failures.append("malformed_abstention_marker")
    if markers.substantive_response_bytes == 0:
        failures.append("empty_response_body")
    if missing:
        failures.append("missing_required_citation")
    if invalid:
        failures.append("unallowed_citation")
    abstention_correct = False
    if case.require_abstention:
        abstention_correct = markers.abstention_markers == 1 and not citations
        if markers.abstention_markers != 1:
            failures.append("missing_or_repeated_abstention_marker")
        if citations:
            failures.append("citation_present_on_abstention_case")
    else:
        abstention_correct = markers.abstention_markers == 0
        if markers.abstention_markers:
            failures.append("unexpected_abstention_marker")
    result: dict[str, object] = {
        "case": private_case_id(case.case_id, case_key),
        "passed": not failures,
        "failure_codes": failures,
        "response": {
            "sha256": observation.response_sha256,
            "bytes": observation.response_bytes,
            "substantive_bytes": markers.substantive_response_bytes,
        },
        "grounding": {
            "citation_ids": list(markers.citation_ids),
            "missing_required_citation_ids": missing,
            "unallowed_citation_ids": invalid,
            "abstention_markers": markers.abstention_markers,
            "malformed_citation_marker": markers.malformed_citation_marker,
            "malformed_abstention_marker": markers.malformed_abstention_marker,
        },
        "generation": {
            "report_sha256": observation.generation_report_sha256,
            "model_sha256": observation.model_sha256,
            "model_manifest_sha256": observation.model_manifest_sha256,
            "runtime_identity_sha256": observation.runtime_identity_sha256,
            "backend": observation.backend_name,
            "elapsed_seconds": round(observation.elapsed_seconds, 9),
        },
    }
    counters = {
        "required": len(required),
        "required_cited": len(required.intersection(citations)),
        "invalid_case": int(bool(invalid or markers.malformed_citation_marker)),
        "abstention_expectation": 1,
        "abstention_correct": int(abstention_correct),
    }
    return result, counters


def rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def gate_report(
    case_results: Sequence[Mapping[str, object]],
    counters: Mapping[str, int],
    arguments: argparse.Namespace,
) -> tuple[dict[str, object], bool]:
    count = len(case_results)
    passed = sum(1 for result in case_results if result["passed"])
    pass_rate = rate(passed, count)
    citation_recall = rate(counters["required_cited"], counters["required"])
    invalid_rate = rate(counters["invalid_case"], count)
    abstention_accuracy = rate(counters["abstention_correct"], counters["abstention_expectation"])
    conditions: list[dict[str, object]] = [
        {
            "name": "case_pass_rate",
            "actual": pass_rate,
            "threshold": arguments.min_case_pass_rate,
            "applicable": True,
            "passed": bool(pass_rate is not None and pass_rate >= arguments.min_case_pass_rate),
        },
        {
            "name": "invalid_citation_case_rate",
            "actual": invalid_rate,
            "threshold": arguments.max_invalid_citation_rate,
            "applicable": True,
            "passed": bool(invalid_rate is not None and invalid_rate <= arguments.max_invalid_citation_rate),
        },
        {
            "name": "required_citation_recall",
            "actual": citation_recall,
            "threshold": arguments.min_citation_recall,
            "applicable": citation_recall is not None,
            "passed": citation_recall is None or citation_recall >= arguments.min_citation_recall,
        },
        {
            "name": "abstention_accuracy",
            "actual": abstention_accuracy,
            "threshold": arguments.min_abstention_accuracy,
            "applicable": abstention_accuracy is not None,
            "passed": abstention_accuracy is None or abstention_accuracy >= arguments.min_abstention_accuracy,
        },
    ]
    passed_gate = all(bool(condition["passed"]) for condition in conditions)
    metrics = {
        "cases": count,
        "passed_cases": passed,
        "case_pass_rate": pass_rate,
        "required_citations": counters["required"],
        "cited_required_citations": counters["required_cited"],
        "required_citation_recall": citation_recall,
        "invalid_citation_cases": counters["invalid_case"],
        "invalid_citation_case_rate": invalid_rate,
        "abstention_expectation_cases": counters["abstention_expectation"],
        "correct_abstention_expectations": counters["abstention_correct"],
        "abstention_accuracy": abstention_accuracy,
    }
    return {"passed": passed_gate, "conditions": conditions, "metrics": metrics}, passed_gate


def evaluate(arguments: argparse.Namespace) -> tuple[dict[str, object], bool]:
    evaluator_path = Path(__file__).resolve()
    cases_path = arguments.cases.resolve()
    command_path = arguments.gateway_command_json.resolve()
    evaluator_identity = file_identity(evaluator_path, "quality evaluator", maximum_bytes=MAX_JSON_BYTES)
    cases_identity = file_identity(cases_path, "quality case set", maximum_bytes=MAX_CASESET_BYTES)
    command, command_identity = load_command(command_path)
    case_key, key_identity = load_hmac_key(arguments.case_id_hmac_key)
    dependencies = [evaluator_path, cases_path, command_path]
    if arguments.case_id_hmac_key is not None:
        dependencies.append(arguments.case_id_hmac_key.resolve())
    for dependency in dependencies:
        require(arguments.report.resolve() != dependency.resolve(),
                f"--report must not overwrite immutable evaluation dependency {dependency}")

    results: list[dict[str, object]] = []
    counters = {
        "required": 0,
        "required_cited": 0,
        "invalid_case": 0,
        "abstention_expectation": 0,
        "abstention_correct": 0,
    }
    observations: list[GatewayObservation] = []
    with tempfile.TemporaryDirectory(prefix="vegai-lmm-quality-") as directory:
        workspace = Path(directory)
        for index, case in enumerate(iter_cases(cases_path, arguments.max_cases), start=1):
            request_path = workspace / f"case-{index:06d}-request.json"
            evidence_path = workspace / f"case-{index:06d}-evidence.json"
            write_json_atomic(request_path, case.request.to_mapping())
            write_json_atomic(evidence_path, case.evidence.to_mapping())
            observation = run_gateway_case(command, request_path, evidence_path, arguments.timeout_seconds)
            observations.append(observation)
            result, result_counters = evaluate_case(case, observation, case_key)
            results.append(result)
            for key, value in result_counters.items():
                counters[key] += value

    require(results, "quality case set is empty")
    # Bind a release report to the exact held-out corpus/configuration used for
    # every child call. The gateway individually binds model/request/evidence.
    require_unchanged(evaluator_path, evaluator_identity, "quality evaluator")
    require_unchanged(cases_path, cases_identity, "quality case set")
    require_unchanged(command_path, command_identity, "quality gateway command")
    if key_identity is not None and arguments.case_id_hmac_key is not None:
        require_unchanged(arguments.case_id_hmac_key, key_identity, "quality case HMAC key")
    model_manifest_hashes = {observation.model_manifest_sha256 for observation in observations}
    runtime_identity_hashes = {observation.runtime_identity_sha256 for observation in observations}
    backend_names = {observation.backend_name for observation in observations}
    require(len(model_manifest_hashes) == 1,
            "quality gateway model manifest identity changed across held-out cases")
    require(len(runtime_identity_hashes) == 1,
            "quality gateway runtime identity changed across held-out cases")
    require(len(backend_names) == 1,
            "quality gateway backend identity changed across held-out cases")
    gate, gate_passed = gate_report(results, counters, arguments)
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "dependencies": {
            "evaluator": evaluator_identity.to_mapping(),
            "cases": cases_identity.to_mapping(),
            "gateway_command": command_identity.to_mapping(),
            "case_id_hmac_key": None if key_identity is None else key_identity.to_mapping(redact_path=True),
        },
        "configuration": {
            "max_cases": arguments.max_cases,
            "timeout_seconds": arguments.timeout_seconds,
            "case_id_privacy": "hmac-sha256" if case_key is not None else "sha256",
            "run_provenance": {
                "model_manifest_sha256": next(iter(model_manifest_hashes)),
                "runtime_identity_sha256": next(iter(runtime_identity_hashes)),
                "backend": next(iter(backend_names)),
            },
            "thresholds": {
                "min_case_pass_rate": arguments.min_case_pass_rate,
                "min_citation_recall": arguments.min_citation_recall,
                "max_invalid_citation_rate": arguments.max_invalid_citation_rate,
                "min_abstention_accuracy": arguments.min_abstention_accuracy,
            },
        },
        "quality_gate": gate,
        "cases": results,
        "quality_boundary": (
            "This held-out marker contract measures supplied-evidence citation and explicit abstention behavior only. "
            "It does not prove factual correctness, fluency, safety, copyright status, broad multilingual quality, "
            "multimodal understanding, or superiority to another model."
        ),
    }
    write_json_atomic(arguments.report, report)
    return report, gate_passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="strict vegai-lmm-quality-case/v1 JSONL held-out corpus")
    parser.add_argument("--gateway-command-json", required=True, type=Path,
                        help="JSON argv with literal {request}, {evidence}, and --print-json")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-cases", type=parse_positive_int, default=MAX_CASES)
    parser.add_argument("--timeout-seconds", type=parse_positive_float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--case-id-hmac-key", type=Path,
                        help="optional local secret: report uses HMAC-SHA-256 instead of raw case-id SHA-256")
    parser.add_argument("--min-case-pass-rate", type=parse_probability, default=1.0)
    parser.add_argument("--min-citation-recall", type=parse_probability, default=1.0)
    parser.add_argument("--max-invalid-citation-rate", type=parse_probability, default=0.0)
    parser.add_argument("--min-abstention-accuracy", type=parse_probability, default=1.0)
    parser.add_argument("--fail-on-gate", action="store_true", help="exit 3 after writing the report when threshold gate fails")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        report, passed = evaluate(arguments)
        print(json.dumps({"report": report, "release_gate_passed": passed}, ensure_ascii=False, sort_keys=True))
        return 0 if passed or not arguments.fail_on_gate else 3
    except (LMMContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"lmm quality evaluation: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
