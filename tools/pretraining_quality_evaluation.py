#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Evaluate a real Vegai HDC checkpoint against held-out semantic-neighbour cases.

The HDC pretraining executable intentionally exposes a narrow, deterministic
``inspect --query-file`` interface: for each active token it returns the
nearest *self-excluded* semantic neighbour and a bipolar similarity.  This
module turns that interface into an operator-owned release-evaluation boundary.

It accepts a strict versioned case manifest and immutable threshold policy,
binds the evaluator/binary/checkpoint/case/policy bytes by SHA-256, invokes the
real compiled executable without a shell, and writes a privacy-conscious report
that contains keyed HMAC identifiers rather than raw case labels or tokens.
A failed threshold exits non-zero after the report is atomically published.

The measured capability is deliberately limited: top-1 semantic-neighbour
retrieval on the supplied held-out target-domain annotations.  Passing does not
prove general language understanding, factuality, safety, coding ability,
multimodal reasoning, or superiority over an LLM/LMM.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CASESET_SCHEMA = "vegai-pretraining-quality-cases/v1"
CASE_SCHEMA = "vegai-pretraining-quality-case/v1"
POLICY_SCHEMA = "vegai-pretraining-quality-policy/v1"
REPORT_SCHEMA = "vegai-pretraining-quality-report/v1"
TOKEN_PATTERN = re.compile(r"[a-z0-9_']{1,48}\Z")
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
MAX_CASE_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_POLICY_BYTES = 256 * 1024
MAX_PROVENANCE_TEXT_BYTES = 1_024
MAX_HMAC_KEY_BYTES = 4 * 1024
MAX_CASES = 4_096
MAX_NEIGHBOURS_PER_CASE = 128
# Artifact hashing/copying is streaming, but finite ceilings prevent accidental
# paths from exhausting evaluation storage. HDC checkpoint capacity is operator-selected.
MAX_BINARY_BYTES = 512 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
MAX_INSPECT_OUTPUT_BYTES = 16 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 120.0


class PretrainingQualityEvaluationError(RuntimeError):
    """Expected input, provenance, executable, or release-gate failure."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """A content and object identity held stable during one evaluation."""

    path: Path
    sha256: str
    bytes: int
    device: int
    inode: int

    def public(self) -> dict[str, object]:
        """Return provenance that does not disclose a local filesystem path."""
        return {"sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class QualityCase:
    """One exact, self-excluded semantic-neighbour expectation."""

    number: int
    case_id: str
    query_token: str
    acceptable_neighbours: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaseManifest:
    """Versioned operator assertion that all annotations are held out."""

    evaluation_id: str
    declared_split: str
    cases: tuple[QualityCase, ...]


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """All quality thresholds; there are no hidden evaluator defaults."""

    policy_id: str
    minimum_cases: int
    minimum_top1_accuracy: float
    minimum_top1_wilson_lower_bound_95: float
    minimum_mean_similarity: float
    minimum_case_similarity: float
    maximum_incorrect_cases: int

    def requirements(self) -> dict[str, int | float]:
        return {
            "minimum_cases": self.minimum_cases,
            "minimum_top1_accuracy": self.minimum_top1_accuracy,
            "minimum_top1_wilson_lower_bound_95": self.minimum_top1_wilson_lower_bound_95,
            "minimum_mean_similarity": self.minimum_mean_similarity,
            "minimum_case_similarity": self.minimum_case_similarity,
            "maximum_incorrect_cases": self.maximum_incorrect_cases,
        }


@dataclass(frozen=True, slots=True)
class EvaluationInputs:
    """Stable parsed case/policy/key inputs used by exactly one run."""

    cases: CaseManifest
    policy: QualityPolicy
    hmac_key: bytes
    case_manifest_identity: FileIdentity
    policy_identity: FileIdentity
    hmac_key_identity: FileIdentity


@dataclass(frozen=True, slots=True)
class InspectObservation:
    """One machine-parsed output block from ``vegai_pretrain inspect``."""

    query_token: str
    neighbour_token: str
    similarity: float


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PretrainingQualityEvaluationError(message)


def exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = sorted(str(key) for key in value if key not in expected)
    missing = sorted(expected.difference(value))
    require(not unknown, f"{label} contains unsupported fields: {', '.join(unknown)}")
    require(not missing, f"{label} is missing required fields: {', '.join(missing)}")


def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PretrainingQualityEvaluationError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def strict_json_loads(raw: bytes, label: str) -> object:
    try:
        decoded = raw.decode("utf-8")
        return json.loads(decoded, object_pairs_hook=reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PretrainingQualityEvaluationError(f"{label} is not valid UTF-8 JSON: {error}") from error


def regular_resolved_path(path: Path, label: str) -> Path:
    """Resolve an input once to its regular-file target for stable snapshots."""
    candidate = path.expanduser()
    try:
        # Input symlinks are supported for secret-manager/versioned artifact
        # layouts. Subsequent snapshots re-resolve the supplied path, so a
        # retargeted symlink still changes the identity and fails closed.
        resolved = candidate.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as error:
        raise PretrainingQualityEvaluationError(f"cannot access {label}: {candidate}: {error}") from error
    require(stat.S_ISREG(mode), f"{label} is not a regular file: {resolved}")
    return resolved


def read_snapshot(path: Path, label: str, maximum_bytes: int) -> tuple[bytes, FileIdentity]:
    """Read one bounded regular-file snapshot and detect leaf replacement races."""
    require(maximum_bytes > 0, "internal snapshot limit must be positive")
    resolved = regular_resolved_path(path, label)
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise PretrainingQualityEvaluationError(f"cannot open {label}: {resolved}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} changed to a non-regular file: {resolved}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            require(total <= maximum_bytes, f"{label} exceeds {maximum_bytes} bytes: {resolved}")
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise PretrainingQualityEvaluationError(f"cannot restat {label}: {resolved}: {error}") from error
    require(
        (before.st_dev, before.st_ino, before.st_size) == (after.st_dev, after.st_ino, after.st_size)
        and (before.st_dev, before.st_ino) == (current.st_dev, current.st_ino),
        f"{label} changed while its snapshot was read: {resolved}",
    )
    identity = FileIdentity(
        path=resolved,
        sha256=digest.hexdigest(),
        bytes=total,
        device=before.st_dev,
        inode=before.st_ino,
    )
    return b"".join(chunks), identity


def file_identity(path: Path, label: str, maximum_bytes: int = MAX_INSPECT_OUTPUT_BYTES) -> FileIdentity:
    """Hash a bounded dependency without retaining all of its bytes in memory."""
    resolved = regular_resolved_path(path, label)
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise PretrainingQualityEvaluationError(f"cannot open {label}: {resolved}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} changed to a non-regular file: {resolved}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            require(total <= maximum_bytes, f"{label} exceeds {maximum_bytes} bytes: {resolved}")
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise PretrainingQualityEvaluationError(f"cannot restat {label}: {resolved}: {error}") from error
    require(
        (before.st_dev, before.st_ino, before.st_size) == (after.st_dev, after.st_ino, after.st_size)
        and (before.st_dev, before.st_ino) == (current.st_dev, current.st_ino),
        f"{label} changed while it was fingerprinted: {resolved}",
    )
    return FileIdentity(resolved, digest.hexdigest(), total, before.st_dev, before.st_ino)


def require_unchanged(path: Path, before: FileIdentity, label: str, maximum_bytes: int = MAX_INSPECT_OUTPUT_BYTES) -> None:
    after = file_identity(path, label, maximum_bytes)
    require(after == before, f"{label} changed while evaluation was running: {before.path}")


def validate_identifier(value: object, label: str) -> str:
    require(isinstance(value, str) and bool(IDENTIFIER_PATTERN.fullmatch(value)),
            f"{label} must match {IDENTIFIER_PATTERN.pattern!r}")
    return value


def validate_token(value: object, label: str) -> str:
    require(isinstance(value, str) and bool(TOKEN_PATTERN.fullmatch(value)),
            f"{label} must be one lowercase Vegai tokenizer token (1..48 ASCII [a-z0-9_'])")
    return value


def validate_provenance_text(value: object, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be a nonempty string")
    require("\r" not in value and "\n" not in value, f"{label} must not contain CR/LF")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise PretrainingQualityEvaluationError(f"{label} is not valid Unicode text") from error
    require(len(encoded) <= MAX_PROVENANCE_TEXT_BYTES,
            f"{label} exceeds {MAX_PROVENANCE_TEXT_BYTES} UTF-8 bytes")
    return value


def parse_annotation_provenance(value: object) -> None:
    require(isinstance(value, Mapping), "quality case manifest.annotation_provenance must be an object")
    exact_fields(
        value, {"source", "license", "retrieved_at", "method"},
        "quality case manifest.annotation_provenance",
    )
    for field in ("source", "license", "retrieved_at", "method"):
        validate_provenance_text(value[field], f"quality case manifest.annotation_provenance.{field}")


def finite_number(value: object, label: str, minimum: float, maximum: float) -> float:
    require(type(value) in {int, float}, f"{label} must be a JSON number")
    parsed = float(value)
    require(math.isfinite(parsed) and minimum <= parsed <= maximum,
            f"{label} must be finite and in [{minimum}, {maximum}]")
    return parsed


def positive_integer(value: object, label: str, maximum: int) -> int:
    require(type(value) is int and 0 < value <= maximum, f"{label} must be an integer in [1, {maximum}]")
    return value


def nonnegative_integer(value: object, label: str, maximum: int) -> int:
    require(type(value) is int and 0 <= value <= maximum, f"{label} must be an integer in [0, {maximum}]")
    return value


def parse_quality_case(value: object, number: int) -> QualityCase:
    require(isinstance(value, Mapping), f"quality case {number} must be an object")
    exact_fields(value, {"schema_version", "id", "query_token", "acceptable_neighbours"}, f"quality case {number}")
    require(value["schema_version"] == CASE_SCHEMA,
            f"quality case {number} has unsupported schema_version: {value['schema_version']!r}")
    case_id = validate_identifier(value["id"], f"quality case {number}.id")
    query = validate_token(value["query_token"], f"quality case {number}.query_token")
    neighbours_value = value["acceptable_neighbours"]
    require(isinstance(neighbours_value, list), f"quality case {number}.acceptable_neighbours must be an array")
    require(0 < len(neighbours_value) <= MAX_NEIGHBOURS_PER_CASE,
            f"quality case {number}.acceptable_neighbours must contain 1..{MAX_NEIGHBOURS_PER_CASE} tokens")
    neighbours = tuple(
        validate_token(token, f"quality case {number}.acceptable_neighbours[{index}]")
        for index, token in enumerate(neighbours_value)
    )
    require(len(set(neighbours)) == len(neighbours), f"quality case {number} repeats an acceptable neighbour")
    require(query not in neighbours, f"quality case {number} must not accept its own query token")
    return QualityCase(number, case_id, query, tuple(sorted(neighbours)))


def parse_case_manifest(raw: bytes, maximum_cases: int) -> CaseManifest:
    require(maximum_cases > 0 and maximum_cases <= MAX_CASES,
            f"--max-cases must be an integer in [1, {MAX_CASES}]")
    value = strict_json_loads(raw, "quality case manifest")
    require(isinstance(value, Mapping), "quality case manifest must be an object")
    exact_fields(
        value,
        {"schema_version", "evaluation_id", "declared_split", "annotation_provenance", "cases"},
        "quality case manifest",
    )
    require(value["schema_version"] == CASESET_SCHEMA,
            f"quality case manifest has unsupported schema_version: {value['schema_version']!r}")
    evaluation_id = validate_identifier(value["evaluation_id"], "quality case manifest.evaluation_id")
    require(value["declared_split"] == "held-out",
            "quality case manifest.declared_split must exactly be 'held-out'")
    parse_annotation_provenance(value["annotation_provenance"])
    raw_cases = value["cases"]
    require(isinstance(raw_cases, list), "quality case manifest.cases must be an array")
    require(0 < len(raw_cases) <= maximum_cases,
            f"quality case manifest must contain 1..{maximum_cases} cases")
    cases = tuple(parse_quality_case(item, index + 1) for index, item in enumerate(raw_cases))
    ids = [case.case_id for case in cases]
    queries = [case.query_token for case in cases]
    require(len(set(ids)) == len(ids), "quality case manifest case IDs must be unique")
    require(len(set(queries)) == len(queries),
            "quality case manifest query tokens must be unique to prevent implicit case weighting")
    return CaseManifest(evaluation_id, "held-out", cases)


def parse_policy(raw: bytes) -> QualityPolicy:
    value = strict_json_loads(raw, "quality policy")
    require(isinstance(value, Mapping), "quality policy must be an object")
    exact_fields(
        value,
        {
            "schema_version", "policy_id", "minimum_cases", "minimum_top1_accuracy",
            "minimum_top1_wilson_lower_bound_95", "minimum_mean_similarity", "minimum_case_similarity",
            "maximum_incorrect_cases",
        },
        "quality policy",
    )
    require(value["schema_version"] == POLICY_SCHEMA,
            f"quality policy has unsupported schema_version: {value['schema_version']!r}")
    policy = QualityPolicy(
        policy_id=validate_identifier(value["policy_id"], "quality policy.policy_id"),
        minimum_cases=positive_integer(value["minimum_cases"], "quality policy.minimum_cases", MAX_CASES),
        minimum_top1_accuracy=finite_number(
            value["minimum_top1_accuracy"], "quality policy.minimum_top1_accuracy", 0.0, 1.0
        ),
        minimum_top1_wilson_lower_bound_95=finite_number(
            value["minimum_top1_wilson_lower_bound_95"],
            "quality policy.minimum_top1_wilson_lower_bound_95", 0.0, 1.0,
        ),
        minimum_mean_similarity=finite_number(
            value["minimum_mean_similarity"], "quality policy.minimum_mean_similarity", -1.0, 1.0
        ),
        minimum_case_similarity=finite_number(
            value["minimum_case_similarity"], "quality policy.minimum_case_similarity", -1.0, 1.0
        ),
        maximum_incorrect_cases=nonnegative_integer(
            value["maximum_incorrect_cases"], "quality policy.maximum_incorrect_cases", MAX_CASES
        ),
    )
    require(policy.maximum_incorrect_cases < policy.minimum_cases,
            "quality policy.maximum_incorrect_cases must be lower than quality policy.minimum_cases")
    return policy


def load_hmac_key(path: Path) -> tuple[bytes, FileIdentity]:
    raw, identity = read_snapshot(path, "case redaction key", MAX_HMAC_KEY_BYTES)
    require(bool(raw), "case redaction key must not be empty")
    return raw, identity


def load_evaluation_inputs(
    case_manifest_path: Path,
    policy_path: Path,
    redaction_key_path: Path,
    maximum_cases: int = MAX_CASES,
) -> EvaluationInputs:
    """Validate all operator inputs before any checkpoint inspection occurs."""
    case_raw, case_identity = read_snapshot(case_manifest_path, "quality case manifest", MAX_CASE_MANIFEST_BYTES)
    policy_raw, policy_identity = read_snapshot(policy_path, "quality policy", MAX_POLICY_BYTES)
    cases = parse_case_manifest(case_raw, maximum_cases)
    policy = parse_policy(policy_raw)
    key, key_identity = load_hmac_key(redaction_key_path)
    return EvaluationInputs(cases, policy, key, case_identity, policy_identity, key_identity)


def evaluator_binding(
    case_manifest_path: Path,
    policy_path: Path,
    redaction_key_path: Path,
    maximum_cases: int = MAX_CASES,
) -> dict[str, object]:
    """Return immutable non-secret policy identity for an orchestration fingerprint.

    This is intentionally public: the HMAC key is checked for existence and
    bounds but neither its bytes nor a hash usable for key-rotation correlation
    is persisted in a production-run manifest.
    """
    inputs = load_evaluation_inputs(case_manifest_path, policy_path, redaction_key_path, maximum_cases)
    evaluator = file_identity(Path(__file__), "quality evaluator tool", MAX_CASE_MANIFEST_BYTES)
    return {
        "mode": "held-out-semantic-neighbour/v1",
        "evaluator_source_sha256": evaluator.sha256,
        "case_manifest_sha256": inputs.case_manifest_identity.sha256,
        "case_manifest_bytes": inputs.case_manifest_identity.bytes,
        "policy_sha256": inputs.policy_identity.sha256,
        "policy_bytes": inputs.policy_identity.bytes,
        "case_count": len(inputs.cases.cases),
        "declared_split": inputs.cases.declared_split,
        "redaction": "HMAC-SHA-256-required",
        "requirements": inputs.policy.requirements(),
    }


def hmac_identifier(key: bytes, namespace: str, value: str) -> str:
    payload = namespace.encode("ascii") + b"\x00" + value.encode("ascii")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def canonical_neighbour_set_digest(key: bytes, neighbours: Sequence[str]) -> str:
    payload = json.dumps(list(neighbours), ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hmac.new(key, b"acceptable-neighbours\x00" + payload, hashlib.sha256).hexdigest()


def fsync_directory(directory: Path) -> None:
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


def scratch_directory(path: Path | None) -> Path | None:
    """Resolve an optional private-snapshot directory without publishing it."""
    if path is None:
        return None
    candidate = path.expanduser()
    try:
        resolved = candidate.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as error:
        raise PretrainingQualityEvaluationError(f"cannot access --scratch-dir {candidate}: {error}") from error
    require(stat.S_ISDIR(mode), f"--scratch-dir is not a directory: {resolved}")
    return resolved


def output_path(path: Path) -> Path:
    candidate = path.expanduser()
    require(candidate.name not in {"", ".", ".."}, "--output must name a report file")
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        parent = candidate.parent.resolve(strict=True)
    except OSError as error:
        raise PretrainingQualityEvaluationError(f"cannot create report directory for {candidate}: {error}") from error
    resolved = parent / candidate.name
    if resolved.exists():
        require(not resolved.is_symlink(), f"--output must not be an existing symlink: {resolved}")
        require(resolved.is_file(), f"--output is not a regular file: {resolved}")
    return resolved


def ensure_output_distinct(output: Path, dependencies: Sequence[tuple[str, Path]]) -> None:
    for label, dependency in dependencies:
        resolved = regular_resolved_path(dependency, label)
        same = output == resolved
        if output.exists():
            try:
                same = same or os.path.samefile(output, resolved)
            except OSError:
                pass
        require(not same, f"--output must not overwrite {label}: {output}")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_all(descriptor: int, payload: bytes) -> None:
    """Write a complete byte sequence even if a low-level write is partial."""
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while creating private evaluation snapshot")
        offset += written


def copy_private_snapshot(source: Path, identity: FileIdentity, destination: Path, label: str, mode: int) -> None:
    """Copy the exact initially fingerprinted bytes before an external exec.

    Rechecking a mutable path after a child exits detects most changes but does
    not prove which bytes the child actually opened.  The evaluator therefore
    executes a private binary snapshot and supplies a private checkpoint
    snapshot.  Both are verified while copying and live only in the temporary
    evaluation directory.
    """
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    try:
        source_descriptor = os.open(source, flags)
        destination_descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except OSError as error:
        try:
            os.close(source_descriptor)
        except (NameError, OSError):
            pass
        raise PretrainingQualityEvaluationError(f"cannot create private {label} snapshot: {error}") from error
    try:
        before = os.fstat(source_descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and (before.st_dev, before.st_ino, before.st_size) == (identity.device, identity.inode, identity.bytes),
            f"{label} changed before private snapshot creation: {source}",
        )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            require(total <= identity.bytes, f"{label} grew while private snapshot was created")
            digest.update(chunk)
            write_all(destination_descriptor, chunk)
        after = os.fstat(source_descriptor)
        os.fsync(destination_descriptor)
    except OSError as error:
        raise PretrainingQualityEvaluationError(f"cannot copy private {label} snapshot: {error}") from error
    finally:
        os.close(source_descriptor)
        os.close(destination_descriptor)
    require(
        (before.st_dev, before.st_ino, before.st_size) == (after.st_dev, after.st_ino, after.st_size)
        and total == identity.bytes and digest.hexdigest() == identity.sha256,
        f"{label} changed while private snapshot was created: {source}",
    )


def invoke_with_limited_output(command: Sequence[str], timeout_seconds: float) -> tuple[int, bytes, float]:
    """Capture a child stream without allowing an untrusted stdout file to grow.

    ``subprocess.run(..., stdout=file)`` can fill a filesystem before a later
    size check. A drain thread keeps at most the protocol cap in memory and
    terminates a child that exceeds it; the pipe provides back-pressure instead
    of a durable unbounded scratch file. This works on Windows too, where
    selector-based pipe polling is not portable.
    """
    try:
        process = subprocess.Popen(
            list(command), cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except OSError as error:
        raise PretrainingQualityEvaluationError(f"cannot start vegai_pretrain inspect: {error}") from error
    require(process.stdout is not None, "internal child stdout pipe is unavailable")
    chunks: list[bytes] = []
    captured_bytes = 0
    overflow = threading.Event()
    reader_errors: list[BaseException] = []

    def drain() -> None:
        nonlocal captured_bytes
        try:
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    return
                if captured_bytes + len(chunk) > MAX_INSPECT_OUTPUT_BYTES:
                    overflow.set()
                    try:
                        process.terminate()
                    except OSError:
                        pass
                    # Continue draining without retaining output so a child
                    # that is terminating cannot deadlock on a full pipe.
                    continue
                chunks.append(chunk)
                captured_bytes += len(chunk)
        except Exception as error:  # surfaced in the controlling thread
            reader_errors.append(error)

    reader = threading.Thread(target=drain, name="vegai-quality-output-drain", daemon=True)
    started = time.monotonic()
    deadline = started + timeout_seconds
    reader.start()
    timed_out = False
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            timed_out = True
            break
        try:
            # Short waits let an oversized-output signal terminate a hostile
            # child promptly rather than waiting for the whole inspect timeout.
            process.wait(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            if overflow.is_set():
                break
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    reader.join(timeout=5.0)
    if reader.is_alive():
        # The direct child is gone but a malformed descendant might retain the
        # inherited pipe. Closing our read end makes this a fail-closed error
        # without keeping a non-daemon thread alive at interpreter shutdown.
        process.stdout.close()
        reader.join(timeout=1.0)
    elapsed = time.monotonic() - started
    require(not reader.is_alive(), "vegai_pretrain inspect left its stdout pipe open after exit")
    if reader_errors:
        raise PretrainingQualityEvaluationError("cannot capture vegai_pretrain inspect output") from reader_errors[0]
    require(not timed_out, f"vegai_pretrain inspect exceeded --timeout-seconds={timeout_seconds:g}")
    require(not overflow.is_set(),
            f"vegai_pretrain inspect emitted more than {MAX_INSPECT_OUTPUT_BYTES} bytes")
    return process.returncode, b"".join(chunks), elapsed


def execute_inspect(
    binary: Path,
    binary_identity: FileIdentity,
    checkpoint: Path,
    checkpoint_identity: FileIdentity,
    cases: Sequence[QualityCase],
    timeout_seconds: float,
    scratch_dir: Path | None,
) -> tuple[bytes, float]:
    """Run the compiled batch-inspection interface without retaining raw output."""
    with tempfile.TemporaryDirectory(
        prefix="vegai-quality-inspect-", dir=None if scratch_dir is None else str(scratch_dir)
    ) as temporary:
        workspace = Path(temporary)
        # Preserve .exe on Windows; a suffixless private copy is not reliably
        # executable through CreateProcess even when the original was.
        private_binary = workspace / ("vegai_pretrain" + binary.suffix)
        private_checkpoint = workspace / "checkpoint.vgai"
        copy_private_snapshot(binary, binary_identity, private_binary, "pretraining executable", 0o700)
        copy_private_snapshot(checkpoint, checkpoint_identity, private_checkpoint, "checkpoint", 0o600)
        query_file = workspace / "queries.txt"
        query_bytes = b"".join(case.query_token.encode("ascii") + b"\n" for case in cases)
        with query_file.open("wb") as handle:
            handle.write(query_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        command = [str(private_binary), "inspect", str(private_checkpoint), "--query-file", str(query_file)]
        return_code, output, elapsed = invoke_with_limited_output(command, timeout_seconds)
    require(return_code == 0,
            f"vegai_pretrain inspect failed with exit code {return_code}; raw child output is intentionally not retained")
    return output, elapsed


def parse_similarity(value: str) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as error:
        raise PretrainingQualityEvaluationError("vegai_pretrain inspect emitted a nonnumeric bipolar similarity") from error
    require(math.isfinite(parsed) and -1.0 <= parsed <= 1.0,
            "vegai_pretrain inspect emitted an out-of-range bipolar similarity")
    return parsed


def parse_inspect_output(raw: bytes, expected_cases: Sequence[QualityCase]) -> tuple[InspectObservation, ...]:
    """Parse every output block exactly; no best-effort substring matching."""
    try:
        output = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise PretrainingQualityEvaluationError("vegai_pretrain inspect output must be ASCII") from error
    require("\r" not in output, "vegai_pretrain inspect output uses CR; strict LF output is required")
    lines = output.split("\n")
    blocks: list[InspectObservation] = []
    for index, line in enumerate(lines):
        if not line.startswith("query: "):
            continue
        require(index + 2 < len(lines), "truncated vegai_pretrain inspect query block")
        query = line[len("query: "):]
        neighbour_line = lines[index + 1]
        similarity_line = lines[index + 2]
        nearest_prefix = "  nearest semantic neighbour: "
        similarity_prefix = "  bipolar similarity:"
        require(neighbour_line.startswith(nearest_prefix),
                "vegai_pretrain inspect query block omitted nearest semantic neighbour")
        require(similarity_line.startswith(similarity_prefix),
                "vegai_pretrain inspect query block omitted bipolar similarity")
        neighbour = neighbour_line[len(nearest_prefix):]
        validate_token(query, "vegai_pretrain inspect query")
        validate_token(neighbour, "vegai_pretrain inspect nearest semantic neighbour")
        blocks.append(InspectObservation(query, neighbour, parse_similarity(similarity_line[len(similarity_prefix):])))
    require(len(blocks) == len(expected_cases),
            f"vegai_pretrain inspect produced {len(blocks)} query blocks for {len(expected_cases)} requested cases")
    expected_queries = tuple(case.query_token for case in expected_cases)
    observed_queries = tuple(block.query_token for block in blocks)
    require(observed_queries == expected_queries,
            "vegai_pretrain inspect query blocks were reordered, omitted, or substituted")
    return tuple(blocks)


def wilson_lower_bound_95(successes: int, total: int) -> float:
    """Conservative two-sided 95% Wilson lower bound for a Bernoulli hit rate."""
    require(0 <= successes <= total and total > 0, "internal Wilson inputs are invalid")
    z = 1.959963984540054
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    centre = (proportion + z_squared / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + z_squared / (4.0 * total)) / total
    ) / denominator
    return max(0.0, centre - radius)


def gate_failures(
    policy: QualityPolicy,
    case_count: int,
    top1_accuracy: float,
    top1_wilson_lower_bound_95: float,
    mean_similarity: float,
    minimum_similarity: float,
    incorrect_cases: int,
) -> list[str]:
    failures: list[str] = []
    if case_count < policy.minimum_cases:
        failures.append(f"case_count={case_count} is below minimum_cases={policy.minimum_cases}")
    if top1_accuracy < policy.minimum_top1_accuracy:
        failures.append(
            f"top1_accuracy={top1_accuracy:.9f} is below minimum_top1_accuracy={policy.minimum_top1_accuracy:.9f}"
        )
    if top1_wilson_lower_bound_95 < policy.minimum_top1_wilson_lower_bound_95:
        failures.append(
            "top1_wilson_lower_bound_95="
            f"{top1_wilson_lower_bound_95:.9f} is below minimum_top1_wilson_lower_bound_95="
            f"{policy.minimum_top1_wilson_lower_bound_95:.9f}"
        )
    if mean_similarity < policy.minimum_mean_similarity:
        failures.append(
            f"mean_similarity={mean_similarity:.9f} is below minimum_mean_similarity={policy.minimum_mean_similarity:.9f}"
        )
    if minimum_similarity < policy.minimum_case_similarity:
        failures.append(
            f"minimum_case_similarity={minimum_similarity:.9f} is below required {policy.minimum_case_similarity:.9f}"
        )
    if incorrect_cases > policy.maximum_incorrect_cases:
        failures.append(
            f"incorrect_cases={incorrect_cases} exceeds maximum_incorrect_cases={policy.maximum_incorrect_cases}"
        )
    return failures


def build_report(
    evaluator_identity: FileIdentity,
    binary_identity: FileIdentity,
    checkpoint_identity: FileIdentity,
    inputs: EvaluationInputs,
    observations: Sequence[InspectObservation],
    inspect_output: bytes,
    elapsed_seconds: float,
) -> dict[str, Any]:
    require(len(observations) == len(inputs.cases.cases), "internal inspection result count mismatch")
    results: list[dict[str, object]] = []
    similarities: list[float] = []
    correct = 0
    for case, observed in zip(inputs.cases.cases, observations, strict=True):
        is_correct = observed.neighbour_token in set(case.acceptable_neighbours)
        correct += int(is_correct)
        similarities.append(observed.similarity)
        results.append(
            {
                "case_number": case.number,
                "case_id_hmac_sha256": hmac_identifier(inputs.hmac_key, "case-id", case.case_id),
                "query_token_hmac_sha256": hmac_identifier(inputs.hmac_key, "query-token", case.query_token),
                "acceptable_neighbour_set_hmac_sha256": canonical_neighbour_set_digest(
                    inputs.hmac_key, case.acceptable_neighbours
                ),
                "observed_neighbour_hmac_sha256": hmac_identifier(
                    inputs.hmac_key, "observed-neighbour", observed.neighbour_token
                ),
                "top1_acceptable": is_correct,
                "bipolar_similarity": observed.similarity,
            }
        )
    total = len(results)
    incorrect = total - correct
    top1_accuracy = correct / total
    top1_wilson_lower_bound_95 = wilson_lower_bound_95(correct, total)
    mean_similarity = math.fsum(similarities) / total
    minimum_similarity = min(similarities)
    maximum_similarity = max(similarities)
    failures = gate_failures(
        inputs.policy, total, top1_accuracy, top1_wilson_lower_bound_95,
        mean_similarity, minimum_similarity, incorrect,
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "completed",
        "evaluator": evaluator_identity.public(),
        "provenance": {
            "pretrain_binary": binary_identity.public(),
            "checkpoint": checkpoint_identity.public(),
            "case_manifest": inputs.case_manifest_identity.public(),
            "policy": inputs.policy_identity.public(),
        },
        "redaction": {
            "algorithm": "HMAC-SHA-256",
            "key_material_persisted": False,
            "contract": "Raw case IDs, query tokens, expected neighbour labels, and observed neighbour tokens are omitted.",
        },
        "evaluation": {
            "evaluation_id_hmac_sha256": hmac_identifier(inputs.hmac_key, "evaluation-id", inputs.cases.evaluation_id),
            "declared_split": inputs.cases.declared_split,
            "case_count": total,
        },
        "metrics": {
            "correct_top1_cases": correct,
            "incorrect_top1_cases": incorrect,
            "top1_accuracy": top1_accuracy,
            "top1_wilson_lower_bound_95": top1_wilson_lower_bound_95,
            "mean_bipolar_similarity": mean_similarity,
            "minimum_bipolar_similarity": minimum_similarity,
            "maximum_bipolar_similarity": maximum_similarity,
        },
        "inspect": {
            "output_sha256": hashlib.sha256(inspect_output).hexdigest(),
            "output_bytes": len(inspect_output),
            "elapsed_seconds": elapsed_seconds,
            "contract": "One real vegai_pretrain inspect --query-file invocation; inspection time is not training throughput.",
        },
        "case_results": results,
        "quality_gate": {
            "passed": not failures,
            "requirements": inputs.policy.requirements(),
            "failures": failures,
            "scope": (
                "Only the operator-supplied held-out semantic-neighbour annotations and this exact checkpoint are measured; "
                "this is not evidence of general intelligence, factuality, safety, or LLM/LMM equivalence."
            ),
        },
    }


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 0 < parsed <= MAX_CASES:
        raise argparse.ArgumentTypeError(f"must be an integer in [1, {MAX_CASES}]")
    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--checkpoint", required=True, type=Path, help="immutable checkpoint to inspect")
    parser.add_argument("--cases", required=True, type=Path, help="strict JSON held-out semantic-neighbour case manifest")
    parser.add_argument("--policy", required=True, type=Path, help="strict JSON immutable quality threshold policy")
    parser.add_argument("--redaction-key-file", required=True, type=Path,
                        help="bounded local secret required for report HMAC redaction")
    parser.add_argument("--output", required=True, type=Path, help="atomic JSON quality report path")
    parser.add_argument("--scratch-dir", type=Path,
                        help="optional local directory for private binary/checkpoint snapshots; path is not reported")
    parser.add_argument("--max-cases", type=parse_positive_int, default=MAX_CASES,
                        help=f"hard case-count bound (default: {MAX_CASES})")
    parser.add_argument("--timeout-seconds", type=parse_positive_float, default=DEFAULT_TIMEOUT_SECONDS,
                        help=f"per-batch inspect timeout (default: {DEFAULT_TIMEOUT_SECONDS:g})")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        binary = regular_resolved_path(arguments.pretrain_bin, "pretraining executable")
        require(os.access(binary, os.X_OK), f"pretraining executable is not executable: {binary}")
        checkpoint = regular_resolved_path(arguments.checkpoint, "checkpoint")
        output = output_path(arguments.output)
        scratch = scratch_directory(arguments.scratch_dir)
        evaluator_source = Path(__file__).resolve()
        ensure_output_distinct(
            output,
            [
                ("quality evaluator tool", evaluator_source),
                ("pretraining executable", binary),
                ("checkpoint", checkpoint),
                ("quality case manifest", arguments.cases),
                ("quality policy", arguments.policy),
                ("case redaction key", arguments.redaction_key_file),
            ],
        )
        inputs = load_evaluation_inputs(arguments.cases, arguments.policy, arguments.redaction_key_file, arguments.max_cases)
        evaluator_identity = file_identity(evaluator_source, "quality evaluator tool", MAX_CASE_MANIFEST_BYTES)
        binary_identity = file_identity(binary, "pretraining executable", MAX_BINARY_BYTES)
        checkpoint_identity = file_identity(arguments.checkpoint, "checkpoint", MAX_CHECKPOINT_BYTES)
        output_bytes, elapsed = execute_inspect(
            binary, binary_identity, checkpoint, checkpoint_identity, inputs.cases.cases,
            arguments.timeout_seconds, scratch,
        )
        observations = parse_inspect_output(output_bytes, inputs.cases.cases)
        # Every source that defines a measurement, including the secret that
        # protects report identifiers, must remain byte/object stable through
        # the external process invocation.
        require_unchanged(evaluator_source, evaluator_identity, "quality evaluator tool", MAX_CASE_MANIFEST_BYTES)
        require_unchanged(binary, binary_identity, "pretraining executable", MAX_BINARY_BYTES)
        require_unchanged(arguments.checkpoint, checkpoint_identity, "checkpoint", MAX_CHECKPOINT_BYTES)
        require_unchanged(arguments.cases, inputs.case_manifest_identity, "quality case manifest", MAX_CASE_MANIFEST_BYTES)
        require_unchanged(arguments.policy, inputs.policy_identity, "quality policy", MAX_POLICY_BYTES)
        require_unchanged(arguments.redaction_key_file, inputs.hmac_key_identity, "case redaction key", MAX_HMAC_KEY_BYTES)
        report = build_report(
            evaluator_identity, binary_identity, checkpoint_identity, inputs, observations, output_bytes, elapsed
        )
        write_json_atomic(output, report)
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        if not report["quality_gate"]["passed"]:
            print("pretraining quality evaluation: QUALITY GATE FAILED", file=sys.stderr)
            return 3
        return 0
    except (PretrainingQualityEvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"pretraining quality evaluation: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
