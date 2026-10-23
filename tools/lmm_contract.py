#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Strict contracts shared by Vegai's optional LMM integration plane.

The native Fortran HDC engine remains a deterministic symbolic/retrieval
runtime.  This module deliberately defines a separate, model-agnostic boundary
for a real external generative model runtime: Unicode/subword tokenization,
long-context packing, evidence provenance, and image/audio/video asset
references.  It contains no fabricated language model and no built-in model
weights.  A backend must supply an actual model or the gateway fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Mapping, Protocol, Sequence


REQUEST_SCHEMA = "vegai-lmm-request/v1"
MODEL_SCHEMA = "vegai-lmm-model/v1"
ASSET_SCHEMA = "vegai-multimodal-asset/v1"
EVIDENCE_SCHEMA = "vegai-lmm-evidence/v1"
WORKER_REQUEST_SCHEMA = "vegai-lmm-worker-request/v1"
WORKER_RESPONSE_SCHEMA = "vegai-lmm-worker-response/v1"
GENERATION_REPORT_SCHEMA = "vegai-lmm-generation-report/v1"

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_TEXT_BYTES = 1 * 1024 * 1024
MAX_MESSAGE_PARTS = 64
MAX_MESSAGES = 16_384
MAX_ASSETS = 1_024
MAX_EVIDENCE_ITEMS = 256
MAX_ASSET_CONTEXT_TOKENS = 2_000_000
MAX_MODEL_ARTIFACTS = 4_096
MAX_GENERATION_TOKENS = 65_536
MAX_IDENTIFIER_BYTES = 160
ABSTENTION_MARKER = "[[vegai-abstain]]"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_CITATION_MARKER = re.compile(r"\[\[vegai-source:([A-Za-z0-9][A-Za-z0-9._:-]{0,159})\]\]")
_ALLOWED_ROLES = frozenset(("system", "user", "assistant", "tool"))
_ALLOWED_MODALITIES = frozenset(("image", "audio", "video"))
_ALLOWED_ENGINES = frozenset(("openvino-genai", "openai-compatible", "worker"))
_ALLOWED_PROMPT_FORMATS = frozenset(("chatml-v1", "json-v1"))


class LMMContractError(RuntimeError):
    """A versioned LMM input, package, or output violates its contract."""


class LMMOptionalDependencyError(LMMContractError):
    """The selected backend/tokenizer needs an intentionally optional package."""


class LMMBackendError(LMMContractError):
    """A configured model backend could not produce a trustworthy result."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LMMContractError(message)


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LMMContractError(f"value is not canonical JSON: {error}") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_identifier(value: object, field: str) -> str:
    require(isinstance(value, str), f"{field} must be a string")
    require(bool(_IDENTIFIER.fullmatch(value)), f"{field} must match {_IDENTIFIER.pattern!r}")
    require(len(value.encode("ascii")) <= MAX_IDENTIFIER_BYTES, f"{field} exceeds {MAX_IDENTIFIER_BYTES} bytes")
    return value


def validate_sha256(value: object, field: str) -> str:
    require(isinstance(value, str) and bool(_HEX_64.fullmatch(value)), f"{field} must be a lowercase SHA-256 hex digest")
    return value


def validate_unicode_text(
    value: object,
    field: str,
    *,
    maximum_bytes: int = MAX_TEXT_BYTES,
    allow_empty: bool = False,
) -> str:
    require(isinstance(value, str), f"{field} must be a Unicode string")
    require(allow_empty or bool(value), f"{field} must not be empty")
    require("\r" not in value and "\x00" not in value, f"{field} must not contain CR or NUL")
    try:
        encoded = value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise LMMContractError(f"{field} contains an invalid Unicode surrogate") from error
    require(len(encoded) <= maximum_bytes, f"{field} exceeds {maximum_bytes} UTF-8 bytes")
    for character in value:
        codepoint = ord(character)
        if codepoint < 32 and character not in {"\n", "\t"}:
            raise LMMContractError(f"{field} contains a disallowed control character")
    return value


def validate_json_object(value: object, field: str, *, maximum_bytes: int = MAX_TEXT_BYTES) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{field} must be a JSON object")
    try:
        normalized = json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, LMMContractError) as error:
        raise LMMContractError(f"{field} is not JSON serializable: {error}") from error
    require(len(canonical_json_bytes(normalized)) <= maximum_bytes, f"{field} exceeds {maximum_bytes} bytes")
    return normalized


def require_known_fields(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    require(not unknown, f"{field} contains unsupported fields: {', '.join(unknown)}")


def require_exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    require_known_fields(value, expected, field)
    missing = sorted(expected.difference(value))
    require(not missing, f"{field} is missing required fields: {', '.join(missing)}")


def parse_nonnegative_int(value: object, field: str, maximum: int) -> int:
    require(type(value) is int and 0 <= value <= maximum, f"{field} must be an integer in [0, {maximum}]")
    return value


def parse_positive_int(value: object, field: str, maximum: int) -> int:
    require(type(value) is int and 1 <= value <= maximum, f"{field} must be an integer in [1, {maximum}]")
    return value


def parse_float_range(value: object, field: str, minimum: float, maximum: float, *, minimum_inclusive: bool = True) -> float:
    require(type(value) in {int, float} and not isinstance(value, bool), f"{field} must be a finite number")
    parsed = float(value)
    lower_ok = parsed >= minimum if minimum_inclusive else parsed > minimum
    require(math.isfinite(parsed) and lower_ok and parsed <= maximum,
            f"{field} must be in {'[' if minimum_inclusive else '('}{minimum}, {maximum}]")
    return parsed


def relative_package_path(value: object, field: str) -> str:
    require(isinstance(value, str) and value, f"{field} must be a nonempty relative POSIX path")
    require("\\" not in value and "\x00" not in value, f"{field} must use POSIX separators and contain no NUL")
    parsed = PurePosixPath(value)
    require(not parsed.is_absolute() and ".." not in parsed.parts and "." not in parsed.parts, f"{field} escapes its package root")
    return value


def resolve_package_path(root: Path, relative: str, field: str) -> Path:
    """Resolve one package member without allowing a symlink escape/swap.

    Hash identities are meaningful only for the path that was authorized.  A
    symlink inside an otherwise trusted model or media package could be swapped
    between validation and inference, so package manifests intentionally use
    plain directory/file paths rather than symlink indirection.
    """
    normalized = relative_package_path(relative, field)
    require(not root.is_symlink(), f"{field} package root must not be a symlink")
    resolved_root = root.resolve()
    lexical = resolved_root
    for component in PurePosixPath(normalized).parts:
        lexical = lexical / component
        require(not lexical.is_symlink(), f"{field} must not traverse a symlink: {lexical}")
    candidate = lexical.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise LMMContractError(f"{field} escapes package root") from error
    return candidate


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    sha256: str
    bytes: int

    def to_mapping(self, *, redact_path: bool = False) -> dict[str, object]:
        return {
            "path": "<redacted>" if redact_path else self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


def file_identity(path: Path, label: str, *, maximum_bytes: int | None = None) -> FileIdentity:
    resolved = path.resolve()
    require(resolved.is_file(), f"{label} is not a regular file: {resolved}")
    digest = hashlib.sha256()
    total = 0
    try:
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                total += len(chunk)
                if maximum_bytes is not None:
                    require(total <= maximum_bytes, f"{label} exceeds {maximum_bytes} bytes")
                digest.update(chunk)
    except OSError as error:
        raise LMMContractError(f"cannot hash {label} {resolved}: {error}") from error
    return FileIdentity(path=str(resolved), sha256=digest.hexdigest(), bytes=total)


def require_unchanged(path: Path, before: FileIdentity, label: str) -> FileIdentity:
    after = file_identity(path, label)
    require(after == before, f"{label} changed while the operation was running: {path}")
    return after


def read_json_object(path: Path, label: str, *, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    identity = file_identity(path, label, maximum_bytes=maximum_bytes)
    try:
        payload = path.resolve().read_bytes()
        decoded = payload.decode("utf-8")
        value = json.loads(decoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LMMContractError(f"cannot parse {label} {path}: {error}") from error
    require(isinstance(value, Mapping), f"{label} must contain one top-level JSON object")
    require_unchanged(path, identity, label)
    return dict(value)


def _fsync_directory(directory: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, os.O_RDONLY | directory_flag)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class ContentPart:
    kind: str
    text: str | None = None
    asset_id: str | None = None

    @classmethod
    def from_mapping(cls, value: object, field: str) -> "ContentPart":
        require(isinstance(value, Mapping), f"{field} must be an object")
        kind = value.get("type")
        require(kind in {"text", "asset"}, f"{field}.type must be 'text' or 'asset'")
        if kind == "text":
            require_exact_fields(value, {"type", "text"}, field)
            return cls(kind="text", text=validate_unicode_text(value["text"], f"{field}.text"))
        require_exact_fields(value, {"type", "asset_id"}, field)
        return cls(kind="asset", asset_id=validate_identifier(value["asset_id"], f"{field}.asset_id"))

    def to_mapping(self) -> dict[str, object]:
        if self.kind == "text":
            assert self.text is not None
            return {"type": "text", "text": self.text}
        assert self.asset_id is not None
        return {"type": "asset", "asset_id": self.asset_id}


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    message_id: str
    role: str
    content: tuple[ContentPart, ...]
    pinned: bool

    @classmethod
    def from_mapping(cls, value: object, field: str) -> "ConversationMessage":
        require(isinstance(value, Mapping), f"{field} must be an object")
        require_exact_fields(value, {"id", "role", "content", "pinned"}, field)
        message_id = validate_identifier(value["id"], f"{field}.id")
        role = value["role"]
        require(role in _ALLOWED_ROLES, f"{field}.role must be one of {', '.join(sorted(_ALLOWED_ROLES))}")
        require(type(value["pinned"]) is bool, f"{field}.pinned must be boolean")
        raw_content = value["content"]
        require(isinstance(raw_content, list) and raw_content, f"{field}.content must be a nonempty array")
        require(len(raw_content) <= MAX_MESSAGE_PARTS, f"{field}.content exceeds {MAX_MESSAGE_PARTS} parts")
        content = tuple(ContentPart.from_mapping(part, f"{field}.content[{index}]") for index, part in enumerate(raw_content))
        return cls(message_id=message_id, role=role, content=content, pinned=value["pinned"])

    def to_mapping(self) -> dict[str, object]:
        return {
            "id": self.message_id,
            "role": self.role,
            "content": [part.to_mapping() for part in self.content],
            "pinned": self.pinned,
        }

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: str
    messages: tuple[ConversationMessage, ...]
    max_new_tokens: int
    temperature: float
    top_p: float

    @classmethod
    def from_mapping(cls, value: object) -> "GenerationRequest":
        require(isinstance(value, Mapping), "LMM request must be an object")
        require_exact_fields(
            value,
            {"schema_version", "id", "messages", "max_new_tokens", "temperature", "top_p"},
            "LMM request",
        )
        require(value["schema_version"] == REQUEST_SCHEMA, f"unsupported LMM request schema: {value['schema_version']!r}")
        request_id = validate_identifier(value["id"], "LMM request.id")
        raw_messages = value["messages"]
        require(isinstance(raw_messages, list) and raw_messages, "LMM request.messages must be a nonempty array")
        require(len(raw_messages) <= MAX_MESSAGES, f"LMM request.messages exceeds {MAX_MESSAGES}")
        messages = tuple(ConversationMessage.from_mapping(message, f"LMM request.messages[{index}]") for index, message in enumerate(raw_messages))
        identifiers = [message.message_id for message in messages]
        require(len(set(identifiers)) == len(identifiers), "LMM request.messages contains duplicate ids")
        require(messages[-1].role == "user", "LMM request must end with a user message")
        max_new_tokens = parse_positive_int(value["max_new_tokens"], "LMM request.max_new_tokens", MAX_GENERATION_TOKENS)
        temperature = parse_float_range(value["temperature"], "LMM request.temperature", 0.0, 2.0)
        top_p = parse_float_range(value["top_p"], "LMM request.top_p", 0.0, 1.0, minimum_inclusive=False)
        return cls(
            request_id=request_id,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @classmethod
    def from_file(cls, path: Path) -> "GenerationRequest":
        return cls.from_mapping(read_json_object(path, "LMM request"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": REQUEST_SCHEMA,
            "id": self.request_id,
            "messages": [message.to_mapping() for message in self.messages],
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    kind: str
    relative_path: str | None = None
    sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: object) -> "TokenizerSpec":
        require(isinstance(value, Mapping), "LMM model.tokenizer must be an object")
        kind = value.get("kind")
        if kind == "utf8-byte":
            require_exact_fields(value, {"kind"}, "LMM model.tokenizer")
            return cls(kind="utf8-byte")
        require(kind == "tokenizers-json", "LMM model.tokenizer.kind must be 'utf8-byte' or 'tokenizers-json'")
        require_exact_fields(value, {"kind", "path", "sha256"}, "LMM model.tokenizer")
        return cls(
            kind="tokenizers-json",
            relative_path=relative_package_path(value["path"], "LMM model.tokenizer.path"),
            sha256=validate_sha256(value["sha256"], "LMM model.tokenizer.sha256"),
        )

    def to_mapping(self) -> dict[str, str]:
        if self.kind == "utf8-byte":
            return {"kind": self.kind}
        assert self.relative_path is not None and self.sha256 is not None
        return {"kind": self.kind, "path": self.relative_path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    relative_path: str
    sha256: str
    bytes: int

    @classmethod
    def from_mapping(cls, value: object, field: str) -> "ModelArtifact":
        require(isinstance(value, Mapping), f"{field} must be an object")
        require_exact_fields(value, {"path", "sha256", "bytes"}, field)
        return cls(
            relative_path=relative_package_path(value["path"], f"{field}.path"),
            sha256=validate_sha256(value["sha256"], f"{field}.sha256"),
            bytes=parse_positive_int(value["bytes"], f"{field}.bytes", 2**63 - 1),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.relative_path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class ModelManifest:
    model_id: str
    engine: str
    context_window_tokens: int
    max_output_tokens: int
    message_overhead_tokens: int
    reply_primer_tokens: int
    capabilities: dict[str, bool]
    tokenizer: TokenizerSpec
    artifacts: tuple[ModelArtifact, ...]
    prompt_format: str
    source: str
    license: str
    revision: str

    @classmethod
    def from_mapping(cls, value: object) -> "ModelManifest":
        require(isinstance(value, Mapping), "LMM model manifest must be an object")
        required = {
            "schema_version", "model_id", "engine", "context_window_tokens", "max_output_tokens",
            "message_overhead_tokens", "reply_primer_tokens", "capabilities", "tokenizer", "artifacts", "prompt_format",
            "source", "license", "revision",
        }
        require_exact_fields(value, required, "LMM model manifest")
        require(value["schema_version"] == MODEL_SCHEMA, f"unsupported LMM model schema: {value['schema_version']!r}")
        engine = value["engine"]
        require(engine in _ALLOWED_ENGINES, f"LMM model.engine must be one of {', '.join(sorted(_ALLOWED_ENGINES))}")
        context_window = parse_positive_int(value["context_window_tokens"], "LMM model.context_window_tokens", 16_000_000)
        max_output = parse_positive_int(value["max_output_tokens"], "LMM model.max_output_tokens", MAX_GENERATION_TOKENS)
        require(max_output < context_window, "LMM model.max_output_tokens must be below context_window_tokens")
        overhead = parse_nonnegative_int(value["message_overhead_tokens"], "LMM model.message_overhead_tokens", 1_024)
        primer = parse_nonnegative_int(value["reply_primer_tokens"], "LMM model.reply_primer_tokens", 1_024)
        raw_capabilities = value["capabilities"]
        require(isinstance(raw_capabilities, Mapping), "LMM model.capabilities must be an object")
        require_exact_fields(raw_capabilities, {"text", "image", "audio", "video"}, "LMM model.capabilities")
        capabilities: dict[str, bool] = {}
        for name in ("text", "image", "audio", "video"):
            require(type(raw_capabilities[name]) is bool, f"LMM model.capabilities.{name} must be boolean")
            capabilities[name] = raw_capabilities[name]
        require(capabilities["text"], "LMM model must support text")
        prompt_format = value["prompt_format"]
        require(prompt_format in _ALLOWED_PROMPT_FORMATS,
                f"LMM model.prompt_format must be one of {', '.join(sorted(_ALLOWED_PROMPT_FORMATS))}")
        raw_artifacts = value["artifacts"]
        require(isinstance(raw_artifacts, list) and len(raw_artifacts) <= MAX_MODEL_ARTIFACTS,
                f"LMM model.artifacts must be an array with at most {MAX_MODEL_ARTIFACTS} entries")
        artifacts = tuple(ModelArtifact.from_mapping(item, f"LMM model.artifacts[{index}]") for index, item in enumerate(raw_artifacts))
        artifact_paths = [artifact.relative_path for artifact in artifacts]
        require(len(set(artifact_paths)) == len(artifact_paths), "LMM model.artifacts repeats a path")
        require(engine != "openvino-genai" or bool(artifacts),
                "openvino-genai model manifests must bind at least one local model artifact")
        return cls(
            model_id=validate_identifier(value["model_id"], "LMM model.model_id"),
            engine=engine,
            context_window_tokens=context_window,
            max_output_tokens=max_output,
            message_overhead_tokens=overhead,
            reply_primer_tokens=primer,
            capabilities=capabilities,
            tokenizer=TokenizerSpec.from_mapping(value["tokenizer"]),
            artifacts=artifacts,
            prompt_format=prompt_format,
            source=validate_unicode_text(value["source"], "LMM model.source", maximum_bytes=8 * 1024),
            license=validate_unicode_text(value["license"], "LMM model.license", maximum_bytes=8 * 1024),
            revision=validate_unicode_text(value["revision"], "LMM model.revision", maximum_bytes=8 * 1024),
        )

    @classmethod
    def from_file(cls, path: Path) -> "ModelManifest":
        return cls.from_mapping(read_json_object(path, "LMM model manifest"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_SCHEMA,
            "model_id": self.model_id,
            "engine": self.engine,
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
            "message_overhead_tokens": self.message_overhead_tokens,
            "reply_primer_tokens": self.reply_primer_tokens,
            "capabilities": dict(self.capabilities),
            "tokenizer": self.tokenizer.to_mapping(),
            "artifacts": [artifact.to_mapping() for artifact in self.artifacts],
            "prompt_format": self.prompt_format,
            "source": self.source,
            "license": self.license,
            "revision": self.revision,
        }


@dataclass(frozen=True, slots=True)
class AssetRecord:
    asset_id: str
    modality: str
    mime_type: str
    relative_path: str
    sha256: str
    bytes: int
    context_tokens: int
    source: str
    license: str
    retrieved_at: str
    metadata: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: object, field: str = "multimodal asset") -> "AssetRecord":
        require(isinstance(value, Mapping), f"{field} must be an object")
        required = {
            "schema_version", "id", "modality", "mime_type", "path", "sha256", "bytes", "context_tokens",
            "source", "license", "retrieved_at", "metadata",
        }
        require_exact_fields(value, required, field)
        require(value["schema_version"] == ASSET_SCHEMA, f"{field}.schema_version is unsupported")
        modality = value["modality"]
        require(modality in _ALLOWED_MODALITIES, f"{field}.modality must be one of {', '.join(sorted(_ALLOWED_MODALITIES))}")
        mime_type = validate_unicode_text(value["mime_type"], f"{field}.mime_type", maximum_bytes=256)
        require(mime_type.startswith(f"{modality}/"), f"{field}.mime_type must begin with {modality!r}/")
        return cls(
            asset_id=validate_identifier(value["id"], f"{field}.id"),
            modality=modality,
            mime_type=mime_type,
            relative_path=relative_package_path(value["path"], f"{field}.path"),
            sha256=validate_sha256(value["sha256"], f"{field}.sha256"),
            bytes=parse_positive_int(value["bytes"], f"{field}.bytes", 2**63 - 1),
            context_tokens=parse_positive_int(value["context_tokens"], f"{field}.context_tokens", MAX_ASSET_CONTEXT_TOKENS),
            source=validate_unicode_text(value["source"], f"{field}.source", maximum_bytes=8 * 1024),
            license=validate_unicode_text(value["license"], f"{field}.license", maximum_bytes=8 * 1024),
            retrieved_at=validate_unicode_text(value["retrieved_at"], f"{field}.retrieved_at", maximum_bytes=1_024),
            metadata=validate_json_object(value["metadata"], f"{field}.metadata", maximum_bytes=64 * 1024),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": ASSET_SCHEMA,
            "id": self.asset_id,
            "modality": self.modality,
            "mime_type": self.mime_type,
            "path": self.relative_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "context_tokens": self.context_tokens,
            "source": self.source,
            "license": self.license,
            "retrieved_at": self.retrieved_at,
            "metadata": self.metadata,
        }

    def resolve(self, asset_root: Path) -> Path:
        return resolve_package_path(asset_root, self.relative_path, f"asset {self.asset_id}.path")


class AssetRegistry:
    """A strict, immutable-by-hash mapping from request asset ids to local files."""

    def __init__(self, assets: Mapping[str, AssetRecord]) -> None:
        self.assets = dict(assets)

    @classmethod
    def from_jsonl(cls, path: Path) -> "AssetRegistry":
        resolved = path.resolve()
        identity = file_identity(resolved, "asset registry", maximum_bytes=MAX_JSON_BYTES)
        assets: dict[str, AssetRecord] = {}
        try:
            with resolved.open("rb") as handle:
                for number, raw in _iter_bounded_jsonl_lines(handle, MAX_JSON_BYTES):
                    try:
                        mapping = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise LMMContractError(f"asset registry line {number} is invalid JSON: {error}") from error
                    asset = AssetRecord.from_mapping(mapping, f"asset registry line {number}")
                    require(asset.asset_id not in assets, f"asset registry repeats id {asset.asset_id!r}")
                    require(len(assets) < MAX_ASSETS, f"asset registry exceeds {MAX_ASSETS} entries")
                    assets[asset.asset_id] = asset
        except OSError as error:
            raise LMMContractError(f"cannot read asset registry {resolved}: {error}") from error
        require_unchanged(resolved, identity, "asset registry")
        return cls(assets)

    def to_jsonl_bytes(self) -> bytes:
        return b"".join(canonical_json_bytes(self.assets[asset_id].to_mapping()) + b"\n" for asset_id in sorted(self.assets))

    def required_for_messages(self, messages: Iterable[ConversationMessage]) -> dict[str, AssetRecord]:
        required: dict[str, AssetRecord] = {}
        for message in messages:
            for part in message.content:
                if part.kind == "asset":
                    assert part.asset_id is not None
                    require(part.asset_id in self.assets, f"message {message.message_id} references unknown asset {part.asset_id!r}")
                    required[part.asset_id] = self.assets[part.asset_id]
        return required

    def verify_required(self, asset_root: Path, messages: Iterable[ConversationMessage]) -> dict[str, tuple[AssetRecord, FileIdentity]]:
        verified: dict[str, tuple[AssetRecord, FileIdentity]] = {}
        for asset_id, record in self.required_for_messages(messages).items():
            target = record.resolve(asset_root)
            identity = file_identity(target, f"asset {asset_id}")
            require(identity.sha256 == record.sha256 and identity.bytes == record.bytes,
                    f"asset {asset_id} does not match its immutable manifest identity")
            verified[asset_id] = (record, identity)
        return verified


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    document_id: str
    text: str
    source: str
    license: str
    sha256: str
    score: float

    @classmethod
    def from_mapping(cls, value: object, field: str) -> "EvidenceItem":
        require(isinstance(value, Mapping), f"{field} must be an object")
        require_exact_fields(value, {"id", "text", "source", "license", "sha256", "score"}, field)
        score_value = value["score"]
        require(type(score_value) in {int, float} and not isinstance(score_value, bool), f"{field}.score must be finite")
        score = float(score_value)
        require(math.isfinite(score), f"{field}.score must be finite")
        text = validate_unicode_text(value["text"], f"{field}.text")
        digest = validate_sha256(value["sha256"], f"{field}.sha256")
        require(sha256_bytes(text.encode("utf-8")) == digest, f"{field}.sha256 does not bind text bytes")
        return cls(
            document_id=validate_identifier(value["id"], f"{field}.id"),
            text=text,
            source=validate_unicode_text(value["source"], f"{field}.source", maximum_bytes=8 * 1024),
            license=validate_unicode_text(value["license"], f"{field}.license", maximum_bytes=8 * 1024),
            sha256=digest,
            score=score,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "id": self.document_id,
            "text": self.text,
            "source": self.source,
            "license": self.license,
            "sha256": self.sha256,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    query_sha256: str
    index_identity: dict[str, Any]
    items: tuple[EvidenceItem, ...]

    @classmethod
    def from_mapping(cls, value: object) -> "EvidenceBundle":
        require(isinstance(value, Mapping), "evidence bundle must be an object")
        require_exact_fields(value, {"schema_version", "query_sha256", "index", "items"}, "evidence bundle")
        require(value["schema_version"] == EVIDENCE_SCHEMA, f"unsupported evidence schema: {value['schema_version']!r}")
        index_identity = validate_json_object(value["index"], "evidence bundle.index", maximum_bytes=64 * 1024)
        raw_items = value["items"]
        require(isinstance(raw_items, list), "evidence bundle.items must be an array")
        require(len(raw_items) <= MAX_EVIDENCE_ITEMS, f"evidence bundle exceeds {MAX_EVIDENCE_ITEMS} items")
        items = tuple(EvidenceItem.from_mapping(item, f"evidence bundle.items[{index}]") for index, item in enumerate(raw_items))
        identifiers = [item.document_id for item in items]
        require(len(set(identifiers)) == len(identifiers), "evidence bundle repeats document ids")
        return cls(
            query_sha256=validate_sha256(value["query_sha256"], "evidence bundle.query_sha256"),
            index_identity=index_identity,
            items=items,
        )

    @classmethod
    def from_file(cls, path: Path) -> "EvidenceBundle":
        return cls.from_mapping(read_json_object(path, "evidence bundle"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": EVIDENCE_SCHEMA,
            "query_sha256": self.query_sha256,
            "index": self.index_identity,
            "items": [item.to_mapping() for item in self.items],
        }

    def to_pinned_messages(self) -> tuple[ConversationMessage, ...]:
        messages: list[ConversationMessage] = []
        for item in self.items:
            # Delimit source material as data. This reduces accidental role
            # confusion but cannot replace a model-specific prompt-injection
            # defence or human source review.
            payload = (
                "<vegai-grounded-evidence id=\"" + item.document_id + "\" source=\"" + item.source + "\">\n"
                + item.text
                + "\n</vegai-grounded-evidence>"
            )
            messages.append(
                ConversationMessage(
                    message_id=f"evidence:{item.document_id}",
                    role="system",
                    content=(ContentPart(kind="text", text=payload),),
                    pinned=True,
                )
            )
        return tuple(messages)


@dataclass(frozen=True, slots=True)
class GroundingMarkers:
    """Machine-readable provenance markers observed in backend-owned text.

    The gateway never inserts these markers into a response. A model policy or
    fine-tune must emit ``[[vegai-source:<evidence-id>]]`` itself. The optional
    release/evaluation planes can then verify that the cited IDs were actually
    supplied as evidence. ``[[vegai-abstain]]`` is an explicit no-answer marker,
    not a claim that a model has solved factuality or uncertainty calibration.
    """

    citation_ids: tuple[str, ...]
    abstention_markers: int
    malformed_citation_marker: bool
    malformed_abstention_marker: bool
    substantive_response_bytes: int

    @property
    def abstained(self) -> bool:
        return self.abstention_markers > 0

    def to_mapping(self) -> dict[str, object]:
        return {
            "citation_ids": list(self.citation_ids),
            "abstention_markers": self.abstention_markers,
            "malformed_citation_marker": self.malformed_citation_marker,
            "malformed_abstention_marker": self.malformed_abstention_marker,
            "substantive_response_bytes": self.substantive_response_bytes,
        }


def parse_grounding_markers(text: str) -> GroundingMarkers:
    """Parse bounded citation/abstention tags without altering backend output."""
    validate_unicode_text(text, "response grounding text")
    citation_ids = tuple(dict.fromkeys(match.group(1) for match in _CITATION_MARKER.finditer(text)))
    require(len(citation_ids) <= MAX_EVIDENCE_ITEMS,
            f"response grounding text exceeds {MAX_EVIDENCE_ITEMS} distinct citation markers")
    malformed_citation = False
    start = 0
    while True:
        position = text.find("[[vegai-source:", start)
        if position < 0:
            break
        match = _CITATION_MARKER.match(text, position)
        if match is None:
            malformed_citation = True
            start = position + len("[[vegai-source:")
        else:
            start = match.end()
    malformed_abstention = False
    start = 0
    while True:
        position = text.find("[[vegai-abstain", start)
        if position < 0:
            break
        if text.startswith(ABSTENTION_MARKER, position):
            start = position + len(ABSTENTION_MARKER)
        else:
            malformed_abstention = True
            start = position + len("[[vegai-abstain")
    stripped = _CITATION_MARKER.sub("", text).replace(ABSTENTION_MARKER, "").strip()
    return GroundingMarkers(
        citation_ids=citation_ids,
        abstention_markers=text.count(ABSTENTION_MARKER),
        malformed_citation_marker=malformed_citation,
        malformed_abstention_marker=malformed_abstention,
        substantive_response_bytes=len(stripped.encode("utf-8")),
    )


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        ...


@dataclass(frozen=True, slots=True)
class UTF8ByteTokenizer:
    """Exact UTF-8 byte fallback for tests/bootstrap; not a semantic BPE model."""

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        try:
            return bytes(token_ids).decode("utf-8", "strict")
        except (ValueError, UnicodeDecodeError) as error:
            raise LMMContractError(f"invalid UTF-8 byte tokenizer sequence: {error}") from error


class TokenizersJSONTokenizer:
    """Fast Unicode/subword tokenizer backed by Hugging Face's Rust tokenizers."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)


def load_tokenizer(manifest: ModelManifest, manifest_path: Path) -> TokenCounter:
    if manifest.tokenizer.kind == "utf8-byte":
        return UTF8ByteTokenizer()
    assert manifest.tokenizer.relative_path is not None and manifest.tokenizer.sha256 is not None
    tokenizer_path = resolve_package_path(manifest_path.resolve().parent, manifest.tokenizer.relative_path, "tokenizer.path")
    identity = file_identity(tokenizer_path, "tokenizer")
    require(identity.sha256 == manifest.tokenizer.sha256, "tokenizer bytes do not match model manifest")
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise LMMOptionalDependencyError(
            "tokenizers-json requires the optional 'tokenizers' package; install requirements-lmm.txt"
        ) from error
    try:
        loaded = TokenizersJSONTokenizer(Tokenizer.from_file(str(tokenizer_path)))
    except Exception as error:  # Optional native extension raises package-specific errors.
        raise LMMContractError(f"cannot load tokenizer JSON {tokenizer_path}: {error}") from error
    require_unchanged(tokenizer_path, identity, "tokenizer")
    return loaded


def verify_model_artifacts(manifest: ModelManifest, model_root: Path) -> dict[str, FileIdentity]:
    """Hash every declared local model artifact before an inference release."""
    verified: dict[str, FileIdentity] = {}
    for artifact in manifest.artifacts:
        target = resolve_package_path(model_root, artifact.relative_path, f"model artifact {artifact.relative_path}")
        identity = file_identity(target, f"model artifact {artifact.relative_path}")
        require(identity.sha256 == artifact.sha256 and identity.bytes == artifact.bytes,
                f"model artifact {artifact.relative_path} does not match immutable model manifest")
        verified[artifact.relative_path] = identity
    return verified


def _iter_bounded_jsonl_lines(handle: Any, maximum_bytes: int) -> Iterable[tuple[int, bytes]]:
    number = 0
    limit = maximum_bytes + 2
    while True:
        raw = handle.readline(limit)
        if not raw:
            return
        number += 1
        has_lf = raw.endswith(b"\n")
        payload_bytes = len(raw) - 1 if has_lf else len(raw)
        overlong = payload_bytes > maximum_bytes or (not has_lf and len(raw) == limit)
        if overlong and not has_lf:
            while raw and not raw.endswith(b"\n"):
                raw = handle.readline(limit)
        require(not overlong, f"JSONL line {number} exceeds {maximum_bytes} bytes")
        require(has_lf, f"JSONL line {number} must terminate with LF")
        payload = raw[:-1]
        require(not payload.endswith(b"\r"), f"JSONL line {number} uses CRLF; strict LF is required")
        require(bool(payload.strip()), f"JSONL line {number} is blank")
        yield number, payload


def message_modalities(messages: Iterable[ConversationMessage], registry: AssetRegistry) -> set[str]:
    modalities: set[str] = set()
    for record in registry.required_for_messages(messages).values():
        modalities.add(record.modality)
    return modalities


def message_token_cost(message: ConversationMessage, tokenizer: TokenCounter, manifest: ModelManifest, registry: AssetRegistry) -> int:
    total = manifest.message_overhead_tokens
    for part in message.content:
        if part.kind == "text":
            assert part.text is not None
            total += tokenizer.count(part.text)
        else:
            assert part.asset_id is not None
            require(part.asset_id in registry.assets, f"message {message.message_id} references unknown asset {part.asset_id!r}")
            total += registry.assets[part.asset_id].context_tokens
    return total


@dataclass(frozen=True, slots=True)
class PackedContext:
    messages: tuple[ConversationMessage, ...]
    omitted_message_ids: tuple[str, ...]
    input_tokens: int
    reserved_output_tokens: int
    context_window_tokens: int

    def to_mapping(self, *, include_content: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "included_message_ids": [message.message_id for message in self.messages],
            "included_message_sha256": [message.digest() for message in self.messages],
            "omitted_message_ids": list(self.omitted_message_ids),
            "input_tokens": self.input_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "context_window_tokens": self.context_window_tokens,
        }
        if include_content:
            payload["messages"] = [message.to_mapping() for message in self.messages]
        return payload


def _conversation_units(messages: Sequence[ConversationMessage]) -> list[tuple[ConversationMessage, ...]]:
    """Keep user-led dialogue turns intact whenever old context is evicted."""
    units: list[tuple[ConversationMessage, ...]] = []
    current: list[ConversationMessage] = []
    for message in messages:
        if message.role == "user" and current:
            units.append(tuple(current))
            current = [message]
        else:
            current.append(message)
    if current:
        units.append(tuple(current))
    return units


def pack_context(
    request: GenerationRequest,
    manifest: ModelManifest,
    tokenizer: TokenCounter,
    registry: AssetRegistry | None = None,
    evidence: EvidenceBundle | None = None,
) -> PackedContext:
    registry = registry or AssetRegistry({})
    evidence_messages = evidence.to_pinned_messages() if evidence is not None else ()
    evidence_ids = {message.message_id for message in evidence_messages}
    request_ids = {message.message_id for message in request.messages}
    require(not evidence_ids.intersection(request_ids), "evidence message ids collide with request message ids")
    messages = tuple(evidence_messages) + request.messages
    require(len(messages) <= MAX_MESSAGES,
            f"request plus pinned evidence exceeds {MAX_MESSAGES} messages")
    required_modalities = message_modalities(messages, registry)
    for modality in required_modalities:
        require(manifest.capabilities[modality], f"model {manifest.model_id} does not declare {modality} capability")

    costs = {message.message_id: message_token_cost(message, tokenizer, manifest, registry) for message in messages}
    reserve = request.max_new_tokens + manifest.reply_primer_tokens
    require(reserve <= manifest.max_output_tokens + manifest.reply_primer_tokens,
            "request max_new_tokens exceeds model max_output_tokens")
    capacity = manifest.context_window_tokens - reserve
    require(capacity > 0, "model context window cannot reserve requested output")

    latest_user_id = request.messages[-1].message_id
    pinned_ids = {message.message_id for message in messages if message.pinned}
    pinned_ids.add(latest_user_id)
    selected_ids: set[str] = set(pinned_ids)
    fixed_tokens = sum(costs[message_id] for message_id in pinned_ids)
    require(fixed_tokens <= capacity,
            "pinned messages/evidence plus latest user message exceed available model context; reduce evidence or output budget")
    remaining = capacity - fixed_tokens

    # Scan complete old turns newest-first. A turn is selected atomically so an
    # assistant answer is not retained without the user turn that motivated it.
    units = _conversation_units(messages)
    for unit in reversed(units):
        unit_ids = {message.message_id for message in unit}
        if unit_ids.intersection(selected_ids):
            continue
        unit_tokens = sum(costs[message.message_id] for message in unit)
        if unit_tokens <= remaining:
            selected_ids.update(unit_ids)
            remaining -= unit_tokens

    selected = tuple(message for message in messages if message.message_id in selected_ids)
    omitted = tuple(message.message_id for message in messages if message.message_id not in selected_ids)
    input_tokens = sum(costs[message.message_id] for message in selected) + manifest.reply_primer_tokens
    require(input_tokens + request.max_new_tokens <= manifest.context_window_tokens,
            "internal context-budget invariant failed")
    return PackedContext(
        messages=selected,
        omitted_message_ids=omitted,
        input_tokens=input_tokens,
        reserved_output_tokens=request.max_new_tokens,
        context_window_tokens=manifest.context_window_tokens,
    )


def render_prompt(context: PackedContext, manifest: ModelManifest, registry: AssetRegistry | None = None) -> str:
    """Render a conservative model prompt; package owners own model-specific templates."""
    registry = registry or AssetRegistry({})
    if manifest.prompt_format == "json-v1":
        return canonical_json_bytes({"messages": [message.to_mapping() for message in context.messages]}).decode("utf-8")
    rendered: list[str] = []
    for message in context.messages:
        rendered.append(f"<|{message.role}|>")
        for part in message.content:
            if part.kind == "text":
                assert part.text is not None
                rendered.append(part.text)
            else:
                assert part.asset_id is not None
                record = registry.assets[part.asset_id]
                rendered.append(f"<vegai-asset id=\"{record.asset_id}\" modality=\"{record.modality}\" />")
        rendered.append("<|end|>")
    rendered.append("<|assistant|>")
    return "\n".join(rendered)


def build_worker_request(
    request: GenerationRequest,
    manifest: ModelManifest,
    context: PackedContext,
    registry: AssetRegistry,
    verified_assets: Mapping[str, tuple[AssetRecord, FileIdentity]],
) -> dict[str, object]:
    """Create a local-worker payload with verified asset paths, never shell args."""
    messages: list[dict[str, object]] = []
    for message in context.messages:
        content: list[dict[str, object]] = []
        for part in message.content:
            if part.kind == "text":
                assert part.text is not None
                content.append({"type": "text", "text": part.text})
            else:
                assert part.asset_id is not None
                record, identity = verified_assets[part.asset_id]
                content.append(
                    {
                        "type": "asset",
                        "id": record.asset_id,
                        "modality": record.modality,
                        "mime_type": record.mime_type,
                        "path": identity.path,
                        "sha256": identity.sha256,
                        "bytes": identity.bytes,
                        "context_tokens": record.context_tokens,
                    }
                )
        messages.append({"id": message.message_id, "role": message.role, "content": content, "pinned": message.pinned})
    return {
        "schema_version": WORKER_REQUEST_SCHEMA,
        "model": manifest.to_mapping(),
        "request": {
            "id": request.request_id,
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        },
        "context": context.to_mapping(include_content=False),
        "messages": messages,
    }


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    text: str
    finish_reason: str
    input_tokens: int | None
    output_tokens: int | None
    backend: str
    device: str | None

    @classmethod
    def from_mapping(cls, value: object) -> "GenerationResponse":
        require(isinstance(value, Mapping), "worker response must be an object")
        required = {"schema_version", "text", "finish_reason", "input_tokens", "output_tokens", "backend", "device"}
        require_exact_fields(value, required, "worker response")
        require(value["schema_version"] == WORKER_RESPONSE_SCHEMA,
                f"unsupported worker response schema: {value['schema_version']!r}")
        finish_reason = validate_identifier(value["finish_reason"], "worker response.finish_reason")
        backend = validate_identifier(value["backend"], "worker response.backend")
        device_value = value["device"]
        require(device_value is None or isinstance(device_value, str), "worker response.device must be string or null")
        device = None if device_value is None else validate_unicode_text(device_value, "worker response.device", maximum_bytes=1_024)
        for name in ("input_tokens", "output_tokens"):
            parsed = value[name]
            require(parsed is None or (type(parsed) is int and 0 <= parsed <= MAX_GENERATION_TOKENS * 256),
                    f"worker response.{name} must be a nonnegative bounded integer or null")
        return cls(
            text=validate_unicode_text(value["text"], "worker response.text"),
            finish_reason=finish_reason,
            input_tokens=value["input_tokens"],
            output_tokens=value["output_tokens"],
            backend=backend,
            device=device,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": WORKER_RESPONSE_SCHEMA,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "backend": self.backend,
            "device": self.device,
        }
