#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Crash-resilient, append-only continuous pretraining controller for Vegai.

The controller consumes an ordered immutable-shard manifest and invokes the real
``vegai_pretrain train`` / ``vegai_pretrain continue`` commands.  It provides
operational losslessness for Vegai's ASCII ingest contract:

* each shard has a SHA-256 identity and strict preflight contract;
* every accepted shard token must be active (min_frequency=1) and yields no OOV;
* training occurs on a candidate checkpoint, never the durable generation;
* the checkpoint generation is finalized before an atomically replaced state
  journal marks the shard committed; and
* restart resumes from the last committed generation, so an interrupted shard
  is retried from the prior model rather than double-applied.

"Lossless" here describes input integrity and durable exactly-once shard
commit semantics.  Binary HDC bundling itself remains a lossy semantic encoder,
not a reversible archive of corpus text.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid
from typing import Any

from continuous_quality_gate import (
    ContinuousQualityGateError,
    REGRESSION_POLICY_SCHEMA,
    regression_policy_binding,
)
from corpus_manifest import (
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_TOKENS_PER_LINE,
    DEFAULT_TOKEN_LENGTH,
    PreflightStats,
    scan_corpus,
)
from pretraining_quality_evaluation import (
    MAX_CASES as QUALITY_MAX_CASES,
    MAX_CHECKPOINT_BYTES as QUALITY_MAX_CHECKPOINT_BYTES,
    PretrainingQualityEvaluationError,
    evaluator_binding,
    scratch_directory,
)
from run_9mib_pretraining import (
    MAX_TRAIN_STDERR_BYTES,
    MAX_TRAIN_STDOUT_BYTES,
    PretrainingRunError,
    run_bounded_child,
)
from training_resource_plan import ResourcePlanError, build_plan


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = "vegai-continuous-shard-manifest/v1"
STATE_SCHEMA = "vegai-continuous-state/v1"
STATE_FILENAME = "state.json"
GENERATIONS_DIRECTORY = "generations"
QUALITY_DIRECTORY = "quality"
QUALITY_REJECTIONS_DIRECTORY = "quality-rejections"
LOCK_FILENAME = ".continuous-pretrain.lock"
QUALITY_GUARD_SCHEMA = "vegai-continuous-quality-guard/v1"
QUALITY_EVALUATOR = ROOT / "tools" / "pretraining_quality_evaluation.py"
QUALITY_COMPARATOR = ROOT / "tools" / "continuous_quality_gate.py"
# app/pretrain.f90 receives corpus/checkpoint arguments in character(len=512).
MAX_PRETRAIN_CLI_PATH_BYTES = 512
HEX64 = set("0123456789abcdef")
MODES = {"legacy", "positional", "hybrid", "hierarchical", "coherent", "debiased", "resonant"}


class ContinuousPretrainError(RuntimeError):
    """Expected validation, integrity, or command failure."""


@dataclass(frozen=True)
class Shard:
    identity: str
    path: Path
    sha256: str
    byte_count: int
    records: int
    accepted_tokens: int
    preflight: dict[str, int]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuousPretrainError(message)


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in HEX64 for character in value.lower())


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def require_cli_path_fits(path: Path, label: str) -> None:
    try:
        encoded = os.fsencode(path)
    except (TypeError, UnicodeError) as error:
        raise ContinuousPretrainError(f"cannot encode {label} path for vegai_pretrain: {path}") from error
    require(
        len(encoded) <= MAX_PRETRAIN_CLI_PATH_BYTES,
        f"{label} path exceeds vegai_pretrain's {MAX_PRETRAIN_CLI_PATH_BYTES}-byte argument limit: {path}",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise ContinuousPretrainError(f"cannot hash file {path}: {error}") from error
    return digest.hexdigest()


def fsync_directory(directory: Path) -> None:
    """Best-effort directory durability for POSIX rename/write ordering."""

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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContinuousPretrainError(f"cannot load {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def resolve_inside(parent: Path, relative_path: str, label: str) -> Path:
    require(isinstance(relative_path, str) and relative_path, f"{label} path is missing")
    candidate = (parent / relative_path).resolve()
    try:
        candidate.relative_to(parent.resolve())
    except ValueError as error:
        raise ContinuousPretrainError(f"{label} escapes its state directory: {relative_path}") from error
    return candidate


def stats_to_dict(stats: PreflightStats) -> dict[str, int]:
    return {
        "accepted_tokens": stats.accepted_tokens,
        "crlf_records": stats.crlf_records,
        "dropped_tokens": stats.dropped_tokens,
        "maximum_record_bytes": stats.maximum_record_bytes,
        "non_ascii_bytes": stats.non_ascii_bytes,
        "overlong_records": stats.overlong_records,
        "records_with_tokenizer_truncation": stats.records_with_tokenizer_truncation,
    }


def validate_lossless_preflight(stats: PreflightStats, label: str) -> None:
    failures: list[str] = []
    if stats.overlong_records:
        failures.append(f"overlong_records={stats.overlong_records}")
    if stats.records_with_tokenizer_truncation:
        failures.append(f"tokenizer_truncation_records={stats.records_with_tokenizer_truncation}")
    if stats.dropped_tokens:
        failures.append(f"dropped_tokens={stats.dropped_tokens}")
    if stats.non_ascii_bytes:
        failures.append(f"non_ascii_bytes={stats.non_ascii_bytes}")
    if stats.crlf_records:
        failures.append(f"crlf_records={stats.crlf_records}")
    if stats.accepted_tokens == 0:
        failures.append("accepted_tokens=0")
    require(not failures, f"{label} violates the lossless ASCII shard contract: {', '.join(failures)}")


def scan_shard(path: Path) -> tuple[str, int, PreflightStats]:
    try:
        with path.open("rb") as handle:
            return scan_corpus(
                handle,
                DEFAULT_MAX_LINE_BYTES,
                DEFAULT_TOKEN_LENGTH,
                DEFAULT_MAX_TOKENS_PER_LINE,
            )
    except OSError as error:
        raise ContinuousPretrainError(f"cannot read shard {path}: {error}") from error


def load_manifest(path: Path, expected_sha256: str | None = None) -> list[Shard]:
    if expected_sha256 is not None:
        require(is_sha256(expected_sha256), "expected manifest SHA-256 is invalid")
        require(
            file_sha256(path) == expected_sha256.lower(),
            f"continuous manifest changed or has wrong digest: {path}",
        )
    manifest = load_json(path, "continuous shard manifest")
    if expected_sha256 is not None:
        require(
            file_sha256(path) == expected_sha256.lower(),
            f"continuous manifest changed while being read: {path}",
        )
    require(manifest.get("schema") == MANIFEST_SCHEMA, f"unsupported shard manifest schema: {manifest.get('schema')!r}")
    contract = manifest.get("vegai_ascii_ingest_contract")
    require(isinstance(contract, dict), "manifest omitted vegai_ascii_ingest_contract")
    require(
        contract.get("max_line_bytes") == DEFAULT_MAX_LINE_BYTES
        and contract.get("token_length_bytes") == DEFAULT_TOKEN_LENGTH
        and contract.get("max_tokens_per_line") == DEFAULT_MAX_TOKENS_PER_LINE
        and contract.get("require_ascii_only") is True
        and contract.get("require_lf_only") is True
        and contract.get("require_no_dropped_tokens") is True,
        "manifest ingest contract does not match lossless controller requirements",
    )
    raw_shards = manifest.get("shards")
    require(isinstance(raw_shards, list), "manifest shards must be an array")

    shards: list[Shard] = []
    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    for position, raw in enumerate(raw_shards, start=1):
        require(isinstance(raw, dict), f"manifest shard {position} must be an object")
        identity = raw.get("id")
        digest = raw.get("sha256")
        relative_path = raw.get("path")
        byte_count = raw.get("bytes")
        records = raw.get("physical_lf_records")
        preflight = raw.get("preflight")
        require(is_sha256(identity), f"manifest shard {position} id must be a SHA-256 identity")
        require(is_sha256(digest), f"manifest shard {position} has invalid SHA-256")
        require(identity.lower() == digest.lower(), f"manifest shard {position} id and SHA-256 disagree")
        require(isinstance(relative_path, str) and relative_path, f"manifest shard {position} has no path")
        require(isinstance(byte_count, int) and byte_count >= 0, f"manifest shard {position} has invalid bytes")
        require(isinstance(records, int) and records >= 0, f"manifest shard {position} has invalid record count")
        require(isinstance(preflight, dict), f"manifest shard {position} has no preflight")
        expected_preflight = {
            "accepted_tokens",
            "crlf_records",
            "dropped_tokens",
            "maximum_record_bytes",
            "non_ascii_bytes",
            "overlong_records",
            "records_with_tokenizer_truncation",
        }
        require(set(preflight) == expected_preflight, f"manifest shard {position} preflight fields are malformed")
        require(all(isinstance(preflight[key], int) and preflight[key] >= 0 for key in expected_preflight),
                f"manifest shard {position} preflight values are malformed")
        canonical_identity = identity.lower()
        canonical_digest = digest.lower()
        require(canonical_identity not in seen_ids, f"manifest repeats shard id {identity!r}")
        require(canonical_digest not in seen_digests, f"manifest repeats shard SHA-256 {digest}")
        seen_ids.add(canonical_identity)
        seen_digests.add(canonical_digest)
        shard = Shard(
            identity=canonical_identity,
            path=(path.parent / relative_path).resolve(),
            sha256=canonical_digest,
            byte_count=byte_count,
            records=records,
            accepted_tokens=preflight["accepted_tokens"],
            preflight={key: int(preflight[key]) for key in expected_preflight},
        )
        validate_lossless_preflight(
            PreflightStats(
                records=shard.records,
                accepted_tokens=shard.accepted_tokens,
                crlf_records=shard.preflight["crlf_records"],
                dropped_tokens=shard.preflight["dropped_tokens"],
                maximum_record_bytes=shard.preflight["maximum_record_bytes"],
                non_ascii_bytes=shard.preflight["non_ascii_bytes"],
                overlong_records=shard.preflight["overlong_records"],
                records_with_tokenizer_truncation=shard.preflight["records_with_tokenizer_truncation"],
            ),
            f"manifest shard {position}",
        )
        shards.append(shard)
    return shards


def verify_path_matches_shard(path: Path, shard: Shard, label: str) -> None:
    digest, byte_count, stats = scan_shard(path)
    require(digest == shard.sha256, f"{label} changed or has wrong digest: {path}")
    require(byte_count == shard.byte_count, f"{label} byte count changed: {path}")
    require(stats.records == shard.records, f"{label} record count changed: {path}")
    require(stats_to_dict(stats) == shard.preflight, f"{label} ingest preflight changed: {path}")
    validate_lossless_preflight(stats, str(path))


def verify_shard_bytes(shard: Shard) -> None:
    verify_path_matches_shard(shard.path, shard, "shard")


def regular_file_identity(path: Path, label: str, maximum_bytes: int = 32 * 1024 * 1024) -> dict[str, object]:
    """Hash a bounded control artifact without accepting symlink replacement."""

    try:
        before = path.stat()
        require(path.is_file() and not path.is_symlink(), f"{label} is not a regular non-symlink file: {path}")
        require(0 < before.st_size <= maximum_bytes, f"{label} is outside supported byte bound: {path}")
        digest = file_sha256(path)
        after = path.stat()
    except OSError as error:
        raise ContinuousPretrainError(f"cannot identify {label}: {error}") from error
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"{label} changed while its identity was calculated: {path}",
    )
    return {"sha256": digest, "bytes": before.st_size}


def validate_binary_identity(identity: object) -> dict[str, object]:
    require(isinstance(identity, dict) and set(identity) == {"sha256", "bytes"},
            "pretraining binary identity is malformed")
    require(is_sha256(identity.get("sha256")) and type(identity.get("bytes")) is int and identity["bytes"] > 0,
            "pretraining binary identity is invalid")
    return identity


def quality_guard_from_arguments(arguments: argparse.Namespace) -> dict[str, Any] | None:
    """Bind optional quality/regression evidence before a mutable train starts.

    The key is intentionally validated by the evaluator binding but never
    recorded (not even as a digest).  Report-row HMAC identifiers make a key
    rotation fail closed at comparison time rather than leaking key identity.
    """

    cases = getattr(arguments, "quality_cases", None)
    policy = getattr(arguments, "quality_policy", None)
    key = getattr(arguments, "quality_redaction_key_file", None)
    regression_policy = getattr(arguments, "quality_regression_policy", None)
    supplied = [cases is not None, policy is not None, key is not None, regression_policy is not None]
    if not any(supplied):
        require(getattr(arguments, "quality_scratch_dir", None) is None,
                "--quality-scratch-dir requires all --quality-* evidence inputs")
        return None
    require(all(supplied),
            "continuous quality protection requires --quality-cases, --quality-policy, "
            "--quality-redaction-key-file, and --quality-regression-policy together")
    maximum_cases = getattr(arguments, "quality_max_cases", QUALITY_MAX_CASES)
    timeout_seconds = getattr(arguments, "quality_timeout_seconds", 120.0)
    require(type(maximum_cases) is int and 0 < maximum_cases <= QUALITY_MAX_CASES,
            f"--quality-max-cases must be in [1, {QUALITY_MAX_CASES}]")
    require(type(timeout_seconds) in {int, float} and math.isfinite(float(timeout_seconds)) and float(timeout_seconds) > 0.0,
            "--quality-timeout-seconds must be finite and positive")
    try:
        evidence = evaluator_binding(cases, policy, key, maximum_cases)
    except PretrainingQualityEvaluationError as error:
        raise ContinuousPretrainError(f"quality evidence preflight failed: {error}") from error
    try:
        regression = regression_policy_binding(regression_policy)
    except ContinuousQualityGateError as error:
        raise ContinuousPretrainError(f"quality regression-policy preflight failed: {error}") from error
    evaluator_identity = regular_file_identity(QUALITY_EVALUATOR, "quality evaluator source")
    comparator_identity = regular_file_identity(QUALITY_COMPARATOR, "continuous quality comparator source")
    require(evaluator_identity["sha256"] == evidence["evaluator_source_sha256"],
            "quality evaluator identity disagrees with evaluator binding")
    return {
        "schema": QUALITY_GUARD_SCHEMA,
        "quality_evidence": evidence,
        "regression_policy": regression,
        "quality_comparator": comparator_identity,
        "maximum_cases": maximum_cases,
        "evaluation_timeout_seconds": float(timeout_seconds),
    }


def validate_quality_guard_binding(guard: object) -> dict[str, Any]:
    require(isinstance(guard, dict), "continuous quality guard is malformed")
    expected = {
        "schema", "quality_evidence", "regression_policy", "quality_comparator", "maximum_cases", "evaluation_timeout_seconds",
    }
    require(set(guard) == expected, "continuous quality guard has unknown or missing fields")
    require(guard.get("schema") == QUALITY_GUARD_SCHEMA, "continuous quality guard has unsupported schema")
    evidence = guard.get("quality_evidence")
    expected_evidence = {
        "mode", "evaluator_source_sha256", "case_manifest_sha256", "case_manifest_bytes", "policy_sha256", "policy_bytes",
        "case_count", "declared_split", "redaction", "requirements",
    }
    require(isinstance(evidence, dict) and set(evidence) == expected_evidence,
            "continuous quality guard evidence binding is malformed")
    require(evidence.get("mode") == "held-out-semantic-neighbour/v1", "continuous quality guard evidence mode is invalid")
    for field in ("evaluator_source_sha256", "case_manifest_sha256", "policy_sha256"):
        require(is_sha256(evidence.get(field)), f"continuous quality guard {field} is invalid")
    for field in ("case_manifest_bytes", "policy_bytes", "case_count"):
        require(type(evidence.get(field)) is int and evidence[field] > 0,
                f"continuous quality guard {field} is invalid")
    require(evidence.get("declared_split") == "held-out", "continuous quality guard must use held-out cases")
    require(evidence.get("redaction") == "HMAC-SHA-256-required", "continuous quality guard redaction contract is invalid")
    requirements = evidence.get("requirements")
    expected_requirements = {
        "minimum_cases", "minimum_top1_accuracy", "minimum_top1_wilson_lower_bound_95",
        "minimum_mean_similarity", "minimum_case_similarity", "maximum_incorrect_cases",
    }
    require(isinstance(requirements, dict) and set(requirements) == expected_requirements,
            "continuous quality guard requirements are malformed")
    require(type(requirements.get("minimum_cases")) is int and requirements["minimum_cases"] > 0
            and type(requirements.get("maximum_incorrect_cases")) is int and requirements["maximum_incorrect_cases"] >= 0,
            "continuous quality guard integer requirements are invalid")
    for field, minimum, maximum in (
        ("minimum_top1_accuracy", 0.0, 1.0),
        ("minimum_top1_wilson_lower_bound_95", 0.0, 1.0),
        ("minimum_mean_similarity", -1.0, 1.0),
        ("minimum_case_similarity", -1.0, 1.0),
    ):
        value = requirements.get(field)
        require(type(value) in {int, float} and math.isfinite(float(value)) and minimum <= float(value) <= maximum,
                f"continuous quality guard requirement {field} is invalid")

    regression = guard.get("regression_policy")
    expected_regression = {"schema_version", "policy_id", "sha256", "bytes", "requirements"}
    require(isinstance(regression, dict) and set(regression) == expected_regression,
            "continuous quality regression-policy binding is malformed")
    require(regression.get("schema_version") == REGRESSION_POLICY_SCHEMA,
            "continuous quality regression-policy schema is invalid")
    require(isinstance(regression.get("policy_id"), str) and regression["policy_id"],
            "continuous quality regression-policy identifier is invalid")
    require(is_sha256(regression.get("sha256")) and type(regression.get("bytes")) is int and regression["bytes"] > 0,
            "continuous quality regression-policy identity is invalid")
    expected_regression_requirements = {
        "maximum_top1_accuracy_drop", "maximum_top1_wilson_lower_bound_95_drop",
        "maximum_mean_bipolar_similarity_drop", "maximum_minimum_bipolar_similarity_drop",
        "maximum_incorrect_top1_cases_increase", "maximum_top1_hit_regressions",
        "maximum_case_similarity_drop", "maximum_case_similarity_regressions",
    }
    require(isinstance(regression.get("requirements"), dict)
            and set(regression["requirements"]) == expected_regression_requirements,
            "continuous quality regression-policy requirements are malformed")
    regression_requirements = regression["requirements"]
    for field in ("maximum_incorrect_top1_cases_increase", "maximum_top1_hit_regressions", "maximum_case_similarity_regressions"):
        require(type(regression_requirements.get(field)) is int and regression_requirements[field] >= 0,
                f"continuous quality regression-policy {field} is invalid")
    for field, maximum in (
        ("maximum_top1_accuracy_drop", 1.0),
        ("maximum_top1_wilson_lower_bound_95_drop", 1.0),
        ("maximum_mean_bipolar_similarity_drop", 2.0),
        ("maximum_minimum_bipolar_similarity_drop", 2.0),
        ("maximum_case_similarity_drop", 2.0),
    ):
        value = regression_requirements.get(field)
        require(type(value) in {int, float} and math.isfinite(float(value)) and 0.0 <= float(value) <= maximum,
                f"continuous quality regression-policy {field} is invalid")

    comparator = guard.get("quality_comparator")
    require(isinstance(comparator, dict) and set(comparator) == {"sha256", "bytes"}
            and is_sha256(comparator.get("sha256")) and type(comparator.get("bytes")) is int
            and comparator["bytes"] > 0,
            "continuous quality comparator identity is invalid")
    require(type(guard.get("maximum_cases")) is int and 0 < guard["maximum_cases"] <= QUALITY_MAX_CASES
            and evidence["case_count"] <= guard["maximum_cases"],
            "continuous quality maximum case bound is invalid")
    timeout = guard.get("evaluation_timeout_seconds")
    require(type(timeout) in {int, float} and math.isfinite(float(timeout)) and float(timeout) > 0.0,
            "continuous quality evaluation timeout is invalid")
    return guard


def quality_scratch_path(arguments: argparse.Namespace) -> Path:
    try:
        selected = scratch_directory(arguments.quality_scratch_dir)
        return (selected if selected is not None else Path(tempfile.gettempdir()).resolve(strict=True))
    except (OSError, PretrainingQualityEvaluationError) as error:
        raise ContinuousPretrainError(f"quality scratch preflight failed: {error}") from error


def quality_scratch_capacity(
    arguments: argparse.Namespace, guard: dict[str, Any], binary: Path, checkpoint: Path
) -> dict[str, int | str]:
    """Reserve private evaluator snapshot space before it copies a checkpoint."""

    try:
        checkpoint_bytes = checkpoint.stat().st_size
        binary_bytes = binary.stat().st_size
        scratch = quality_scratch_path(arguments)
        free_bytes = shutil.disk_usage(scratch).free
    except OSError as error:
        raise ContinuousPretrainError(f"cannot inspect quality scratch capacity: {error}") from error
    require(0 < checkpoint_bytes <= QUALITY_MAX_CHECKPOINT_BYTES,
            f"candidate checkpoint exceeds quality evaluator snapshot cap ({checkpoint_bytes} bytes)")
    required_bytes = checkpoint_bytes + binary_bytes + int(guard["maximum_cases"]) * 49 + 2 * 1024 * 1024
    require(free_bytes >= required_bytes,
            "quality scratch free space is below private snapshot requirement "
            f"({free_bytes} < {required_bytes} bytes)")
    return {
        "directory": str(scratch),
        "free_bytes": free_bytes,
        "required_bytes": required_bytes,
        "checkpoint_bytes_basis": checkpoint_bytes,
        "binary_bytes_basis": binary_bytes,
    }


def default_policy(arguments: argparse.Namespace) -> dict[str, Any]:
    policy = {
        "max_vocabulary": 20_000 if arguments.max_vocab is None else arguments.max_vocab,
        "min_frequency": 1 if arguments.min_frequency is None else arguments.min_frequency,
        "context_window": 4 if arguments.window is None else arguments.window,
        "position_shift_bits": 73 if arguments.position_shift is None else arguments.position_shift,
        "context_mode": "hierarchical" if arguments.context_mode is None else arguments.context_mode,
        "epochs_per_shard": 1 if arguments.epochs is None else arguments.epochs,
        "initial_plasticity_percent": 25 if arguments.initial_plasticity is None else arguments.initial_plasticity,
        "minimum_plasticity_percent": 1 if arguments.minimum_plasticity is None else arguments.minimum_plasticity,
        "plasticity_decay_events": 32 if arguments.decay_events is None else arguments.decay_events,
    }
    validate_policy(policy)
    return policy


def validate_policy(policy: dict[str, Any]) -> None:
    required = {
        "max_vocabulary",
        "min_frequency",
        "context_window",
        "position_shift_bits",
        "context_mode",
        "epochs_per_shard",
        "initial_plasticity_percent",
        "minimum_plasticity_percent",
        "plasticity_decay_events",
    }
    require(set(policy) == required, "continuous state policy has unknown or missing fields")
    numeric_fields = required - {"context_mode"}
    require(all(type(policy[key]) is int for key in numeric_fields), "continuous state policy numeric fields must be integers")
    require(type(policy["context_mode"]) is str, "continuous state context_mode must be a string")
    require(policy["max_vocabulary"] > 0, "max_vocabulary must be positive")
    require(policy["min_frequency"] == 1, "lossless continuous mode requires min_frequency=1")
    require(policy["context_window"] > 0, "context_window must be positive")
    require(policy["position_shift_bits"] != 0, "position_shift_bits must be nonzero")
    require(policy["context_mode"] in MODES, "context_mode must be legacy, positional, hybrid, hierarchical, coherent, debiased, or resonant")
    require(policy["epochs_per_shard"] > 0, "epochs_per_shard must be positive")
    require(0 < policy["minimum_plasticity_percent"] <= policy["initial_plasticity_percent"] <= 100,
            "plasticity percentages are invalid")
    require(policy["plasticity_decay_events"] > 0, "plasticity_decay_events must be positive")


def supplied_policy_overrides(arguments: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "max_vocab": "max_vocabulary",
        "min_frequency": "min_frequency",
        "window": "context_window",
        "position_shift": "position_shift_bits",
        "context_mode": "context_mode",
        "epochs": "epochs_per_shard",
        "initial_plasticity": "initial_plasticity_percent",
        "minimum_plasticity": "minimum_plasticity_percent",
        "decay_events": "plasticity_decay_events",
    }
    return {target: getattr(arguments, source) for source, target in mapping.items() if getattr(arguments, source) is not None}


def fresh_state(
    policy: dict[str, Any],
    quality_guard: dict[str, Any] | None = None,
    binary_identity: dict[str, object] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "generation": 0,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "policy": policy,
        "applied_shards": [],
    }
    if quality_guard is not None:
        state["quality_guard"] = validate_quality_guard_binding(quality_guard)
    if binary_identity is not None:
        state["pretrain_binary"] = validate_binary_identity(binary_identity)
    return state


def checkpoint_from_state(state_directory: Path, state: dict[str, Any]) -> Path:
    checkpoint = state.get("checkpoint")
    digest = state.get("checkpoint_sha256")
    require(isinstance(checkpoint, str) and checkpoint, "state omits committed checkpoint path")
    require(is_sha256(digest), "state omits committed checkpoint SHA-256")
    checkpoint_path = resolve_inside(state_directory, checkpoint, "checkpoint")
    require(checkpoint_path.is_file(), f"committed checkpoint is missing: {checkpoint_path}")
    actual = file_sha256(checkpoint_path)
    require(actual == digest, f"committed checkpoint digest mismatch: {checkpoint_path}")
    return checkpoint_path


def validate_state_journal(state_directory: Path, state: dict[str, Any]) -> None:
    """Verify every prior generation before using it as an exactly-once base."""

    quality_guard_value = state.get("quality_guard")
    quality_guard = None if quality_guard_value is None else validate_quality_guard_binding(quality_guard_value)
    applied = state["applied_shards"]
    for expected_generation, entry in enumerate(applied, start=1):
        require(isinstance(entry, dict), f"state applied shard {expected_generation} is malformed")
        require(is_sha256(entry.get("id")), f"state applied shard {expected_generation} has invalid id")
        require(is_sha256(entry.get("sha256")), f"state applied shard {expected_generation} has invalid SHA-256")
        require(entry["id"].lower() == entry["sha256"].lower(),
                f"state applied shard {expected_generation} has inconsistent identity")
        require(entry.get("generation") == expected_generation,
                f"state applied shard {expected_generation} has inconsistent generation")
        require(type(entry.get("bytes")) is int and entry["bytes"] >= 0,
                f"state applied shard {expected_generation} has invalid byte count")
        require(type(entry.get("physical_lf_records")) is int and entry["physical_lf_records"] >= 0,
                f"state applied shard {expected_generation} has invalid record count")
        require(type(entry.get("accepted_tokens")) is int and entry["accepted_tokens"] > 0,
                f"state applied shard {expected_generation} has invalid token count")
        entry_preflight = entry.get("preflight")
        expected_preflight = {
            "accepted_tokens",
            "crlf_records",
            "dropped_tokens",
            "maximum_record_bytes",
            "non_ascii_bytes",
            "overlong_records",
            "records_with_tokenizer_truncation",
        }
        require(isinstance(entry_preflight, dict) and set(entry_preflight) == expected_preflight,
                f"state applied shard {expected_generation} has malformed preflight")
        require(all(type(entry_preflight[key]) is int and entry_preflight[key] >= 0 for key in expected_preflight),
                f"state applied shard {expected_generation} has invalid preflight values")
        require(entry_preflight["accepted_tokens"] == entry["accepted_tokens"],
                f"state applied shard {expected_generation} token count disagrees with preflight")
        verified_metrics = entry.get("verified_metrics")
        expected_metrics = {
            "vocabulary_observed_tokens",
            "training_raw_tokens",
            "training_active_tokens",
            "training_oov_tokens",
        }
        require(isinstance(verified_metrics, dict) and set(verified_metrics) == expected_metrics,
                f"state applied shard {expected_generation} has malformed verified metrics")
        require(all(type(verified_metrics[key]) is int and verified_metrics[key] >= 0 for key in expected_metrics),
                f"state applied shard {expected_generation} has invalid verified metrics")
        expected_training_tokens = entry["accepted_tokens"] * state["policy"]["epochs_per_shard"]
        require(
            verified_metrics["vocabulary_observed_tokens"] == entry["accepted_tokens"]
            and verified_metrics["training_raw_tokens"] == expected_training_tokens
            and verified_metrics["training_active_tokens"] == expected_training_tokens
            and verified_metrics["training_oov_tokens"] == 0,
            f"state applied shard {expected_generation} verified metrics violate the lossless training contract",
        )
        entry_path = resolve_inside(state_directory, entry.get("checkpoint"), "applied checkpoint")
        require(entry_path.is_file(), f"applied checkpoint is missing: {entry_path}")
        require(is_sha256(entry.get("checkpoint_sha256")),
                f"state applied shard {expected_generation} has invalid checkpoint SHA-256")
        require(file_sha256(entry_path) == entry["checkpoint_sha256"],
                f"applied checkpoint digest mismatch: {entry_path}")
        validate_training_measurement(entry, expected_training_tokens)
        if quality_guard is None:
            require("quality" not in entry,
                    f"state applied shard {expected_generation} carries quality evidence without a state quality guard")
    if quality_guard is not None:
        validate_quality_anchor_sequence(state, state_directory, quality_guard)
    if applied:
        final = applied[-1]
        require(state.get("checkpoint") == final["checkpoint"]
                and state.get("checkpoint_sha256") == final["checkpoint_sha256"],
                "state final checkpoint does not match final journal generation")


def validate_training_measurement(entry: dict[str, Any], expected_tokens: int) -> None:
    """Validate train-only timing evidence when present in a new journal entry."""

    measurement = entry.get("training_measurement")
    if measurement is None:
        # Pre-quality-guard journals predate child timing evidence and remain
        # readable; new commits always carry the stronger contract below.
        return
    expected = {
        "scope", "start_boundary", "stop_boundary", "excluded", "train_process_wall_seconds",
        "raw_training_tokens_per_second", "active_training_tokens_per_second", "omp_num_threads",
        "stdout_sha256", "stdout_bytes", "stderr_sha256", "stderr_bytes",
    }
    require(isinstance(measurement, dict) and set(measurement) == expected,
            "state training measurement is malformed")
    require(measurement.get("scope") == "one vegai_pretrain train-or-continue child-process wall-clock interval",
            "state training measurement has an invalid scope")
    require(measurement.get("start_boundary") == "immediately before spawning vegai_pretrain",
            "state training measurement has an invalid start boundary")
    require(measurement.get("stop_boundary") == "immediately after vegai_pretrain exits",
            "state training measurement has an invalid stop boundary")
    excluded = measurement.get("excluded")
    require(isinstance(excluded, list) and all(isinstance(item, str) for item in excluded),
            "state training measurement exclusions are malformed")
    for required in ("compilation", "quality evaluation", "controller preflight", "CI orchestration"):
        require(required in excluded, f"state training measurement omits {required!r}")
    elapsed = measurement.get("train_process_wall_seconds")
    require(type(elapsed) in {int, float} and math.isfinite(float(elapsed)) and float(elapsed) > 0.0,
            "state training duration is invalid")
    require(type(measurement.get("omp_num_threads")) is int and measurement["omp_num_threads"] > 0,
            "state measurement OMP thread count is invalid")
    for field in ("stdout_sha256", "stderr_sha256"):
        require(is_sha256(measurement.get(field)), f"state measurement {field} is invalid")
    for field in ("stdout_bytes", "stderr_bytes"):
        require(type(measurement.get(field)) is int and measurement[field] >= 0,
                f"state measurement {field} is invalid")
    for field in ("raw_training_tokens_per_second", "active_training_tokens_per_second"):
        observed = measurement.get(field)
        require(type(observed) in {int, float} and math.isfinite(float(observed)) and float(observed) > 0.0,
                f"state measurement {field} is invalid")
        expected_rate = expected_tokens / float(elapsed)
        require(math.isclose(float(observed), round(expected_rate, 3), rel_tol=0.0, abs_tol=0.001),
                f"state measurement {field} disagrees with token count and duration")


def load_strict_json_snapshot(path: Path, label: str, maximum_bytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    """Read a small durable child report once; reject duplicate-key replacement."""

    try:
        before = path.stat()
        require(path.is_file() and not path.is_symlink() and 0 < before.st_size <= maximum_bytes,
                f"{label} is not a bounded regular file")
        raw = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise ContinuousPretrainError(f"cannot read {label}: {error}") from error
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        and len(raw) == before.st_size,
        f"{label} changed while it was read",
    )

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ContinuousPretrainError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContinuousPretrainError(f"{label} is malformed JSON: {error}") from error
    require(isinstance(parsed, dict), f"{label} must be a JSON object")
    return parsed


def validate_quality_evidence_entry(
    state_directory: Path,
    entry: dict[str, Any],
    quality_guard: dict[str, Any],
    expected_generation: int,
    expected_binary_sha256: str | None = None,
) -> None:
    """Bind each committed checkpoint to its immutable quality/regression evidence."""

    quality = entry.get("quality")
    expected = {
        "schema", "quality_report", "quality_report_sha256", "quality_report_bytes", "comparison_report",
        "comparison_report_sha256", "comparison_report_bytes", "anchor_generation", "anchor_checkpoint_sha256",
    }
    require(isinstance(quality, dict) and set(quality) == expected,
            f"state applied shard {expected_generation} has malformed quality evidence")
    require(quality.get("schema") == "vegai-continuous-quality-evidence/v1",
            f"state applied shard {expected_generation} has unsupported quality evidence schema")
    require(type(quality.get("anchor_generation")) is int and quality["anchor_generation"] == 1,
            f"state applied shard {expected_generation} has invalid quality anchor generation")
    require(is_sha256(quality.get("anchor_checkpoint_sha256")),
            f"state applied shard {expected_generation} has invalid quality anchor checkpoint digest")
    for label, path_field, digest_field, bytes_field in (
        ("quality report", "quality_report", "quality_report_sha256", "quality_report_bytes"),
        ("comparison report", "comparison_report", "comparison_report_sha256", "comparison_report_bytes"),
    ):
        path = resolve_inside(state_directory, quality.get(path_field), f"{label} path")
        require(path.is_file() and not path.is_symlink(),
                f"state applied shard {expected_generation} {label} is missing")
        require(is_sha256(quality.get(digest_field)),
                f"state applied shard {expected_generation} {label} digest is invalid")
        require(type(quality.get(bytes_field)) is int and quality[bytes_field] > 0,
                f"state applied shard {expected_generation} {label} byte count is invalid")
        require(path.stat().st_size == quality[bytes_field] and file_sha256(path) == quality[digest_field],
                f"state applied shard {expected_generation} {label} identity mismatch")
    quality_report_path = resolve_inside(state_directory, quality["quality_report"], "quality report path")
    comparison_report_path = resolve_inside(state_directory, quality["comparison_report"], "comparison report path")
    quality_report = load_strict_json_snapshot(quality_report_path, "committed quality report")
    comparison_report = load_strict_json_snapshot(comparison_report_path, "committed regression report")
    evidence = quality_guard["quality_evidence"]
    require(quality_report.get("schema_version") == "vegai-pretraining-quality-report/v1"
            and quality_report.get("status") == "completed",
            f"state applied shard {expected_generation} quality report schema/status is invalid")
    report_provenance = quality_report.get("provenance")
    report_gate = quality_report.get("quality_gate")
    report_evaluation = quality_report.get("evaluation")
    require(isinstance(report_provenance, dict) and isinstance(report_gate, dict) and isinstance(report_evaluation, dict)
            and report_gate.get("passed") is True
            and report_gate.get("requirements") == quality_guard["quality_evidence"]["requirements"]
            and report_evaluation.get("declared_split") == "held-out"
            and report_evaluation.get("case_count") == quality_guard["quality_evidence"]["case_count"],
            f"state applied shard {expected_generation} quality report is not an accepted absolute gate")
    # Report provenance is checked without raw held-out content. The state
    # separately binds the native executable identity before any child starts.
    binary_provenance = report_provenance.get("pretrain_binary")
    if expected_binary_sha256 is not None:
        require(isinstance(binary_provenance, dict) and binary_provenance.get("sha256") == expected_binary_sha256,
                f"state applied shard {expected_generation} quality report binary provenance is invalid")
    checkpoint_provenance = report_provenance.get("checkpoint")
    case_provenance = report_provenance.get("case_manifest")
    policy_provenance = report_provenance.get("policy")
    evaluator = quality_report.get("evaluator")
    require(isinstance(checkpoint_provenance, dict) and checkpoint_provenance.get("sha256") == entry.get("checkpoint_sha256"),
            f"state applied shard {expected_generation} quality report checkpoint provenance is invalid")
    require(isinstance(case_provenance, dict) and case_provenance.get("sha256") == evidence["case_manifest_sha256"],
            f"state applied shard {expected_generation} quality report case provenance is invalid")
    require(isinstance(policy_provenance, dict) and policy_provenance.get("sha256") == evidence["policy_sha256"],
            f"state applied shard {expected_generation} quality report policy provenance is invalid")
    require(isinstance(evaluator, dict) and evaluator.get("sha256") == evidence["evaluator_source_sha256"],
            f"state applied shard {expected_generation} quality report evaluator provenance is invalid")

    require(comparison_report.get("schema_version") == "vegai-continuous-quality-regression-report/v1"
            and comparison_report.get("status") == "passed" and comparison_report.get("failures") == [],
            f"state applied shard {expected_generation} regression report is not an accepted pass")
    baseline = comparison_report.get("baseline")
    candidate = comparison_report.get("candidate")
    regression = comparison_report.get("regression_policy")
    require(isinstance(baseline, dict) and isinstance(candidate, dict) and isinstance(regression, dict),
            f"state applied shard {expected_generation} regression report is malformed")
    require(candidate.get("checkpoint_sha256") == entry.get("checkpoint_sha256")
            and candidate.get("quality_report_sha256") == quality["quality_report_sha256"],
            f"state applied shard {expected_generation} regression candidate provenance is invalid")
    require(baseline.get("checkpoint_sha256") == quality["anchor_checkpoint_sha256"],
            f"state applied shard {expected_generation} regression anchor provenance is invalid")
    require(regression == quality_guard["regression_policy"],
            f"state applied shard {expected_generation} regression policy provenance is invalid")
    if expected_generation == 1:
        require(quality["anchor_checkpoint_sha256"] == entry.get("checkpoint_sha256"),
                "initial quality anchor does not bind the initial checkpoint")


def validate_quality_anchor_sequence(state: dict[str, Any], state_directory: Path, quality_guard: dict[str, Any]) -> None:
    applied = state["applied_shards"]
    if not applied:
        return
    first = applied[0]
    require(isinstance(first, dict) and is_sha256(first.get("checkpoint_sha256")),
            "continuous quality anchor checkpoint is unavailable")
    anchor_digest = first["checkpoint_sha256"]
    binary = state.get("pretrain_binary")
    binary_sha256 = binary.get("sha256") if isinstance(binary, dict) and is_sha256(binary.get("sha256")) else None
    for generation, entry in enumerate(applied, start=1):
        require(isinstance(entry, dict), f"continuous quality entry {generation} is malformed")
        validate_quality_evidence_entry(state_directory, entry, quality_guard, generation, binary_sha256)
        quality = entry["quality"]
        require(quality["anchor_checkpoint_sha256"] == anchor_digest,
                f"state applied shard {generation} quality evidence changed its immutable anchor")


def load_state(
    state_directory: Path,
    arguments: argparse.Namespace,
    requested_quality_guard: dict[str, Any] | None = None,
    requested_binary_identity: dict[str, object] | None = None,
    *,
    enforce_quality_guard: bool = False,
    enforce_binary_identity: bool = False,
) -> dict[str, Any]:
    state_path = state_directory / STATE_FILENAME
    if not state_path.exists():
        return fresh_state(default_policy(arguments), requested_quality_guard, requested_binary_identity)
    state = load_json(state_path, "continuous state")
    require(state.get("schema") == STATE_SCHEMA, f"unsupported continuous state schema: {state.get('schema')!r}")
    require(isinstance(state.get("generation"), int) and state["generation"] >= 0, "state generation is invalid")
    require(isinstance(state.get("applied_shards"), list), "state applied_shards is invalid")
    require(state["generation"] == len(state["applied_shards"]), "state generation does not match applied shard count")
    require(isinstance(state.get("policy"), dict), "state policy is invalid")
    validate_policy(state["policy"])
    stored_quality_guard = state.get("quality_guard")
    if stored_quality_guard is not None:
        stored_quality_guard = validate_quality_guard_binding(stored_quality_guard)
    if enforce_quality_guard:
        require(stored_quality_guard == requested_quality_guard,
                "continuous quality guard differs from the immutable state; supply the original quality evidence and policy")
    stored_binary_identity = state.get("pretrain_binary")
    if stored_binary_identity is not None:
        stored_binary_identity = validate_binary_identity(stored_binary_identity)
    if enforce_binary_identity:
        require(stored_binary_identity == requested_binary_identity,
                "pretraining executable differs from the immutable continuous state")
    overrides = supplied_policy_overrides(arguments)
    for key, value in overrides.items():
        require(value == state["policy"][key], f"state policy mismatch for {key}: {value!r} != {state['policy'][key]!r}")
    if state["generation"] == 0:
        require(state.get("checkpoint") is None and state.get("checkpoint_sha256") is None,
                "empty state must not point to a checkpoint")
    else:
        validate_state_journal(state_directory, state)
        checkpoint_from_state(state_directory, state)
    return state


def validate_manifest_prefix(state: dict[str, Any], shards: list[Shard]) -> None:
    applied = state["applied_shards"]
    require(len(applied) <= len(shards), "manifest removed a previously committed shard")
    for index, entry in enumerate(applied):
        require(isinstance(entry, dict), f"state applied shard {index} is malformed")
        shard = shards[index]
        require(
            entry.get("id") == shard.identity
            and entry.get("sha256") == shard.sha256
            and entry.get("bytes") == shard.byte_count
            and entry.get("physical_lf_records") == shard.records
            and entry.get("accepted_tokens") == shard.accepted_tokens
            and entry.get("preflight") == shard.preflight,
            f"manifest rewrote committed shard at position {index + 1}",
        )


def line_values(output: str, label: str) -> list[int]:
    import re

    return [int(value) for value in re.findall(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", output, re.MULTILINE)]


def require_values(output: str, label: str, expected: list[int]) -> None:
    actual = line_values(output, label)
    require(actual == expected, f"CLI {label!r} mismatch: expected {expected}, got {actual}\n{output}")


def validate_cli_output(output: str, shard: Shard, policy: dict[str, Any], generation: int, extending: bool) -> dict[str, int]:
    import re

    expected_phase = "vocabulary extension" if extending else "vocabulary"
    require(
        re.search(rf"^\s*phase:\s*{re.escape(expected_phase)}\s*$", output, re.MULTILINE) is not None,
        f"CLI omitted expected phase {expected_phase!r}\n{output}",
    )
    require(
        re.search(r"^\s*phase:\s*streaming HDC context learning\s*$", output, re.MULTILINE) is not None,
        f"CLI omitted training phase\n{output}",
    )
    epochs = policy["epochs_per_shard"]
    require_values(output, "corpus lines", [shard.records, shard.records * epochs])
    require_values(output, "observed tokens", [shard.accepted_tokens])
    require_values(output, "raw tokens", [shard.accepted_tokens * epochs])
    # There are two active-token lines: vocabulary active count then training active count.
    active_values = line_values(output, "active tokens")
    require(len(active_values) == 2 and active_values[1] == shard.accepted_tokens * epochs,
            f"CLI active-token coverage mismatch for {shard.path}: {active_values}\n{output}")
    require_values(output, "out-of-vocabulary tokens", [0])
    require_values(output, "completed epochs", [generation * epochs])
    require_values(output, "tokenizer truncations", [0, 0])
    require_values(output, "non-ASCII input bytes", [0, 0])
    require_values(output, "dropped overlong records", [0, 0])
    return {
        "vocabulary_observed_tokens": shard.accepted_tokens,
        "training_raw_tokens": shard.accepted_tokens * epochs,
        "training_active_tokens": shard.accepted_tokens * epochs,
        "training_oov_tokens": 0,
    }


def train_command(binary: Path, shard: Shard, candidate: Path, policy: dict[str, Any]) -> list[str]:
    return [
        str(binary),
        "train",
        str(shard.path),
        str(candidate),
        "--max-vocab",
        str(policy["max_vocabulary"]),
        "--min-frequency",
        str(policy["min_frequency"]),
        "--window",
        str(policy["context_window"]),
        "--position-shift",
        str(policy["position_shift_bits"]),
        "--context-mode",
        str(policy["context_mode"]),
        "--epochs",
        str(policy["epochs_per_shard"]),
        "--initial-plasticity",
        str(policy["initial_plasticity_percent"]),
        "--minimum-plasticity",
        str(policy["minimum_plasticity_percent"]),
        "--decay-events",
        str(policy["plasticity_decay_events"]),
        "--strict",
    ]


def continue_command(binary: Path, shard: Shard, candidate: Path, policy: dict[str, Any]) -> list[str]:
    return [
        str(binary),
        "continue",
        str(shard.path),
        str(candidate),
        "--epochs",
        str(policy["epochs_per_shard"]),
        "--strict",
    ]


def invoke(command: list[str], arguments: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Run one native training child with bounded I/O and a train-only timer."""

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(arguments.omp_threads)
    environment["OMP_PROC_BIND"] = "close"
    environment["OMP_PLACES"] = "cores"
    try:
        result = run_bounded_child(
            command,
            environment=environment,
            timeout_seconds=arguments.train_timeout_seconds,
            stdout_limit_bytes=MAX_TRAIN_STDOUT_BYTES,
            stderr_limit_bytes=MAX_TRAIN_STDERR_BYTES,
            label="vegai_pretrain train/continue",
        )
    except PretrainingRunError as error:
        raise ContinuousPretrainError(str(error)) from error
    try:
        output = result.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ContinuousPretrainError("vegai_pretrain emitted non-UTF-8 training output") from error
    require(result.returncode == 0,
            f"vegai_pretrain failed ({result.returncode}): {' '.join(command)}\n{output}")
    elapsed = round(result.child_wall_seconds, 9)
    require(elapsed > 0.0, "vegai_pretrain train child duration is not positive")
    return output, {
        "scope": "one vegai_pretrain train-or-continue child-process wall-clock interval",
        "start_boundary": "immediately before spawning vegai_pretrain",
        "stop_boundary": "immediately after vegai_pretrain exits",
        "excluded": [
            "compilation", "controller preflight", "corpus identity scans", "checkpoint publication",
            "quality evaluation", "report serialization", "CI orchestration",
        ],
        "train_process_wall_seconds": elapsed,
        "raw_training_tokens_per_second": None,
        "active_training_tokens_per_second": None,
        "omp_num_threads": arguments.omp_threads,
        "stdout_sha256": result.stdout_sha256,
        "stdout_bytes": result.stdout_bytes,
        "stderr_sha256": result.stderr_sha256,
        "stderr_bytes": result.stderr_bytes,
    }


def resource_preflight_for_shard(
    shard: Shard, candidate: Path, policy: dict[str, Any], arguments: argparse.Namespace
) -> dict[str, Any]:
    """Fail before the native process if RAM/checkpoint capacity is insufficient."""

    planner_arguments = SimpleNamespace(
        corpus=shard.path,
        checkpoint=candidate,
        max_vocab=policy["max_vocabulary"],
        context_mode=policy["context_mode"],
        omp_threads=arguments.omp_threads,
        reserve_fraction=arguments.reserve_fraction,
        require_accelerator=False,
    )
    try:
        plan = build_plan(planner_arguments)
    except (ResourcePlanError, OSError, ValueError) as error:
        raise ContinuousPretrainError(f"training resource preflight failed: {error}") from error
    verdict = plan.get("verdict")
    require(isinstance(verdict, dict) and verdict.get("ready") is True,
            "training resource preflight is not ready: " + "; ".join(str(value) for value in verdict.get("errors", [])))
    memory = plan.get("memory")
    checkpoint = plan.get("checkpoint")
    cpu = plan.get("cpu_execution")
    accelerators = plan.get("accelerators")
    hardware = plan.get("hardware")
    require(all(isinstance(value, dict) for value in (memory, checkpoint, cpu, accelerators, hardware)),
            "resource planner emitted an incomplete plan")
    return {
        "schema": plan.get("schema"),
        "ready": True,
        "memory": {
            "required_available_bytes": memory.get("required_available_bytes"),
            "host_available_bytes": memory.get("host_available_bytes"),
            "max_vocabulary": memory.get("max_vocabulary"),
            "reserve_fraction": memory.get("reserve_fraction"),
        },
        "checkpoint": {
            "estimated_max_bytes": checkpoint.get("estimated_max_bytes"),
            "free_bytes": checkpoint.get("storage", {}).get("free_bytes") if isinstance(checkpoint.get("storage"), dict) else None,
        },
        "cpu_execution": cpu.get("environment"),
        "accelerators": {
            "cpu_compute_used": accelerators.get("cpu_compute_used"),
            "gpu_compute_used": accelerators.get("gpu_compute_used"),
            "npu_compute_used": accelerators.get("npu_compute_used"),
        },
        "hardware": {
            "cpu_model": hardware.get("cpu", {}).get("model") if isinstance(hardware.get("cpu"), dict) else None,
            "operating_system": hardware.get("host", {}).get("operating_system") if isinstance(hardware.get("host"), dict) else None,
            "architecture": hardware.get("host", {}).get("architecture") if isinstance(hardware.get("host"), dict) else None,
        },
        "warnings": verdict.get("warnings"),
    }


def run_quality_evaluator(
    binary: Path,
    candidate: Path,
    output: Path,
    guard: dict[str, Any],
    arguments: argparse.Namespace,
) -> int:
    """Generate one bounded, redacted checkpoint-quality report."""

    scratch = quality_scratch_capacity(arguments, guard, binary, candidate)
    command = [
        sys.executable, str(QUALITY_EVALUATOR),
        "--pretrain-bin", str(binary), "--checkpoint", str(candidate),
        "--cases", str(arguments.quality_cases), "--policy", str(arguments.quality_policy),
        "--redaction-key-file", str(arguments.quality_redaction_key_file),
        "--output", str(output), "--max-cases", str(guard["maximum_cases"]),
        "--scratch-dir", str(scratch["directory"]),
        "--timeout-seconds", str(guard["evaluation_timeout_seconds"]),
    ]
    # The evaluator's timeout governs real inspect work. Its wrapper can need
    # bounded snapshot I/O as well, so give it a fixed administrative margin.
    outer_timeout = float(guard["evaluation_timeout_seconds"]) + 300.0
    try:
        child = run_bounded_child(
            command,
            environment=os.environ.copy(),
            timeout_seconds=outer_timeout,
            stdout_limit_bytes=MAX_TRAIN_STDOUT_BYTES,
            stderr_limit_bytes=MAX_TRAIN_STDERR_BYTES,
            label="pretraining quality evaluator",
        )
    except PretrainingRunError as error:
        raise ContinuousPretrainError(str(error)) from error
    require(child.returncode in {0, 3},
            f"quality evaluator failed with unexpected status {child.returncode}")
    require(output.is_file() and not output.is_symlink(),
            "quality evaluator did not atomically publish a regular report")
    return child.returncode


def run_quality_comparator(
    binary: Path,
    baseline_checkpoint: Path,
    candidate_checkpoint: Path,
    baseline_report: Path,
    candidate_report: Path,
    output: Path,
    guard: dict[str, Any],
    arguments: argparse.Namespace,
) -> int:
    command = [
        sys.executable, str(QUALITY_COMPARATOR),
        "--pretrain-bin", str(binary),
        "--baseline-checkpoint", str(baseline_checkpoint),
        "--candidate-checkpoint", str(candidate_checkpoint),
        "--baseline-report", str(baseline_report),
        "--candidate-report", str(candidate_report),
        "--cases", str(arguments.quality_cases),
        "--quality-policy", str(arguments.quality_policy),
        "--redaction-key-file", str(arguments.quality_redaction_key_file),
        "--regression-policy", str(arguments.quality_regression_policy),
        "--output", str(output), "--max-cases", str(guard["maximum_cases"]),
    ]
    try:
        child = run_bounded_child(
            command,
            environment=os.environ.copy(),
            timeout_seconds=300.0,
            stdout_limit_bytes=MAX_TRAIN_STDOUT_BYTES,
            stderr_limit_bytes=MAX_TRAIN_STDERR_BYTES,
            label="continuous quality comparator",
        )
    except PretrainingRunError as error:
        raise ContinuousPretrainError(str(error)) from error
    require(child.returncode in {0, 3},
            f"continuous quality comparator failed with unexpected status {child.returncode}")
    require(output.is_file() and not output.is_symlink(),
            "continuous quality comparator did not publish a regular report")
    return child.returncode


def publish_rejected_quality_evidence(
    state_directory: Path,
    generation: int,
    checkpoint_sha256: str,
    quality_stage: Path,
    comparison_stage: Path,
) -> None:
    """Keep redacted failure evidence without publishing a rejected checkpoint."""

    rejected = state_directory / QUALITY_REJECTIONS_DIRECTORY
    rejected.mkdir(parents=True, exist_ok=True)
    stem = f"generation-{generation:020d}-{checkpoint_sha256[:16]}"
    published: dict[str, str | None] = {"quality_report": None, "comparison_report": None}
    for stage, suffix, field in (
        (quality_stage, ".quality.json", "quality_report"),
        (comparison_stage, ".regression.json", "comparison_report"),
    ):
        if stage.is_file() and not stage.is_symlink():
            target = rejected / f"{stem}{suffix}"
            os.replace(stage, target)
            fsync_file(target)
            published[field] = str(target.relative_to(state_directory))
    atomic_write_json(
        rejected / f"{stem}.rejection.json",
        {
            "schema": "vegai-continuous-quality-rejection/v1",
            "generation": generation,
            "candidate_checkpoint_sha256": checkpoint_sha256,
            "quality_report": published["quality_report"],
            "comparison_report": published["comparison_report"],
            "decision": "rejected-before-checkpoint-publication",
        },
    )
    fsync_directory(rejected)


def quality_evidence_for_candidate(
    state_directory: Path,
    binary: Path,
    state: dict[str, Any],
    candidate: Path,
    checkpoint_sha256: str,
    generation: int,
    work_id: str,
    arguments: argparse.Namespace,
) -> dict[str, Any] | None:
    """Qualify a candidate before its checkpoint can become a journal generation."""

    guard_value = state.get("quality_guard")
    if guard_value is None:
        return None
    guard = validate_quality_guard_binding(guard_value)
    quality_directory = state_directory / QUALITY_DIRECTORY
    quality_directory.mkdir(parents=True, exist_ok=True)
    quality_stage = quality_directory / f".candidate-{work_id}.quality.json"
    comparison_stage = quality_directory / f".candidate-{work_id}.regression.json"
    final_stem = f"generation-{generation:020d}-{checkpoint_sha256[:16]}"
    final_quality = quality_directory / f"{final_stem}.quality.json"
    final_comparison = quality_directory / f"{final_stem}.regression.json"
    try:
        evaluator_status = run_quality_evaluator(binary, candidate, quality_stage, guard, arguments)
        if generation == 1:
            baseline_checkpoint = candidate
            baseline_report = quality_stage
            anchor_checkpoint_sha256 = checkpoint_sha256
        else:
            first = state["applied_shards"][0]
            require(isinstance(first, dict) and isinstance(first.get("quality"), dict),
                    "quality-guarded state omitted initial anchor evidence")
            first_quality = first["quality"]
            baseline_checkpoint = resolve_inside(state_directory, first.get("checkpoint"), "quality anchor checkpoint")
            baseline_report = resolve_inside(state_directory, first_quality.get("quality_report"), "quality anchor report")
            anchor_checkpoint_sha256 = str(first.get("checkpoint_sha256"))
        comparator_status = run_quality_comparator(
            binary, baseline_checkpoint, candidate, baseline_report, quality_stage, comparison_stage, guard, arguments
        )
        # Re-identify every policy/evaluator/comparator input after external
        # execution. The equality check includes no secret key material.
        require(quality_guard_from_arguments(arguments) == guard,
                "quality guard inputs changed while candidate quality was evaluated")
        if evaluator_status != 0 or comparator_status != 0:
            publish_rejected_quality_evidence(
                state_directory, generation, checkpoint_sha256, quality_stage, comparison_stage
            )
            raise ContinuousPretrainError(
                "candidate failed immutable absolute quality policy or continuation regression policy; "
                "checkpoint was rejected before publication"
            )
        os.replace(quality_stage, final_quality)
        os.replace(comparison_stage, final_comparison)
        fsync_file(final_quality)
        fsync_file(final_comparison)
        fsync_directory(quality_directory)
        return {
            "schema": "vegai-continuous-quality-evidence/v1",
            "quality_report": str(final_quality.relative_to(state_directory)),
            "quality_report_sha256": file_sha256(final_quality),
            "quality_report_bytes": final_quality.stat().st_size,
            "comparison_report": str(final_comparison.relative_to(state_directory)),
            "comparison_report_sha256": file_sha256(final_comparison),
            "comparison_report_bytes": final_comparison.stat().st_size,
            "anchor_generation": 1,
            "anchor_checkpoint_sha256": anchor_checkpoint_sha256,
        }
    finally:
        quality_stage.unlink(missing_ok=True)
        comparison_stage.unlink(missing_ok=True)


def commit_one_shard(
    state_directory: Path,
    binary: Path,
    state: dict[str, Any],
    shard: Shard,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    policy = state["policy"]
    generation = state["generation"] + 1
    generations = state_directory / GENERATIONS_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)

    # Scan the publisher's shard first, then take and validate a private byte
    # snapshot.  The child learns only that snapshot, so concurrent mutation of
    # a source path cannot turn a SHA-verified ledger entry into a mixed read.
    verify_shard_bytes(shard)
    expected_binary = state.get("pretrain_binary")
    if expected_binary is not None:
        require(regular_file_identity(binary, "pretraining executable") == expected_binary,
                "pretraining executable changed before native training")
    work_id = f"{os.getpid()}-{uuid.uuid4().hex}"
    shard_snapshot = generations / f".shard-{work_id}.txt"
    candidate = generations / f".candidate-{work_id}.vgai"
    try:
        shutil.copyfile(shard.path, shard_snapshot)
        fsync_file(shard_snapshot)
        verify_path_matches_shard(shard_snapshot, shard, "private shard snapshot")
        training_shard = replace(shard, path=shard_snapshot)
        require_cli_path_fits(training_shard.path, "private shard snapshot")
        require_cli_path_fits(candidate, "candidate checkpoint")
        resource_plan = resource_preflight_for_shard(training_shard, candidate, policy, arguments)
        if state["generation"] == 0:
            command = train_command(binary, training_shard, candidate, policy)
            extending = False
        else:
            shutil.copyfile(checkpoint_from_state(state_directory, state), candidate)
            fsync_file(candidate)
            command = continue_command(binary, training_shard, candidate, policy)
            extending = True
        output, measurement = invoke(command, arguments)
        metrics = validate_cli_output(output, training_shard, policy, generation, extending)
        elapsed = float(measurement["train_process_wall_seconds"])
        measurement["raw_training_tokens_per_second"] = round(metrics["training_raw_tokens"] / elapsed, 3)
        measurement["active_training_tokens_per_second"] = round(metrics["training_active_tokens"] / elapsed, 3)
        verify_shard_bytes(shard)
        require(candidate.is_file() and not candidate.is_symlink(),
                "pretraining command did not create a regular candidate checkpoint")
        fsync_file(candidate)
        checkpoint_sha256 = file_sha256(candidate)
        if expected_binary is not None:
            require(regular_file_identity(binary, "pretraining executable") == expected_binary,
                    "pretraining executable changed during native training")
        quality = quality_evidence_for_candidate(
            state_directory, binary, state, candidate, checkpoint_sha256, generation, work_id, arguments
        )
        final_name = f"generation-{generation:020d}-{checkpoint_sha256[:16]}.vgai"
        final_checkpoint = generations / final_name
        # Quality qualification runs before this rename. A failed candidate
        # therefore remains private and is removed below rather than becoming
        # an unreferenced but accidentally consumable checkpoint artifact.
        os.replace(candidate, final_checkpoint)
        fsync_directory(generations)

        next_state = deepcopy(state)
        next_state["generation"] = generation
        next_state["checkpoint"] = str(final_checkpoint.relative_to(state_directory))
        next_state["checkpoint_sha256"] = checkpoint_sha256
        entry: dict[str, Any] = {
            "id": shard.identity,
            "sha256": shard.sha256,
            "bytes": shard.byte_count,
            "physical_lf_records": shard.records,
            "accepted_tokens": shard.accepted_tokens,
            "preflight": shard.preflight,
            "generation": generation,
            "checkpoint": str(final_checkpoint.relative_to(state_directory)),
            "checkpoint_sha256": checkpoint_sha256,
            "verified_metrics": metrics,
            "training_measurement": measurement,
            "resource_preflight": resource_plan,
        }
        if quality is not None:
            entry["quality"] = quality
        next_state["applied_shards"].append(entry)
        # The state file is the commit marker.  If the process fails before this
        # atomic replace, the finalized generation is an ignored orphan and the
        # next invocation replays the shard from the previous stable checkpoint.
        atomic_write_json(state_directory / STATE_FILENAME, next_state)
        return next_state
    finally:
        candidate.unlink(missing_ok=True)
        shard_snapshot.unlink(missing_ok=True)


def cleanup_private_work(generations: Path) -> None:
    """Remove only unpublishable work files left by an interrupted controller."""

    removed = False
    for pattern in (".candidate-*.vgai", ".shard-*.txt"):
        for stale in generations.glob(pattern):
            if stale.is_file() or stale.is_symlink():
                stale.unlink()
                removed = True
    if removed:
        fsync_directory(generations)


def cleanup_private_quality_work(quality_directory: Path) -> None:
    """Remove never-published evaluator/comparator staging reports after a crash."""

    if not quality_directory.is_dir():
        return
    removed = False
    for pattern in (".candidate-*.quality.json", ".candidate-*.regression.json"):
        for stale in quality_directory.glob(pattern):
            if stale.is_file() or stale.is_symlink():
                stale.unlink()
                removed = True
    if removed:
        fsync_directory(quality_directory)


class ControllerLock:
    def __init__(self, path: Path, recover_stale: bool) -> None:
        self.path = path
        self.acquired = False
        if recover_stale:
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> "ControllerLock":
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise ContinuousPretrainError(
                f"continuous controller lock exists: {self.path}; inspect the owner or pass --recover-lock explicitly"
            ) from error
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            fsync_directory(self.path.parent)


def process_available(
    state_directory: Path,
    manifest_path: Path,
    binary: Path,
    state: dict[str, Any],
    maximum_shards: int | None,
    expected_manifest_sha256: str | None,
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    shards = load_manifest(manifest_path, expected_manifest_sha256)
    validate_manifest_prefix(state, shards)
    committed = 0
    while state["generation"] < len(shards) and (maximum_shards is None or committed < maximum_shards):
        shard = shards[state["generation"]]
        state = commit_one_shard(state_directory, binary, state, shard, arguments)
        committed += 1
        print(
            f"committed generation={state['generation']} shard={shard.identity[:16]} "
            f"tokens={shard.accepted_tokens} checkpoint_sha256={state['checkpoint_sha256']}"
        )
    return state, committed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--manifest", required=True, type=Path, help="immutable ordered shard manifest JSON")
    parser.add_argument("--state-dir", required=True, type=Path, help="controller state and generation directory")
    parser.add_argument(
        "--expect-manifest-sha256",
        help="optional immutable manifest digest; reject replacement before every processing pass",
    )
    parser.add_argument("--follow", action="store_true", help="poll an atomically updated append-only manifest forever")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="follow polling interval (default: 5)")
    parser.add_argument("--max-shards", type=int, help="commit at most this many currently pending shards")
    parser.add_argument("--recover-lock", action="store_true", help="remove a stale lock only after operator review")
    parser.add_argument("--omp-threads", type=int, default=1,
                        help="explicit OMP_NUM_THREADS for each native training child (default: 1)")
    parser.add_argument("--train-timeout-seconds", type=float, default=1200.0,
                        help="bounded native train/continue child timeout (default: 1200)")
    parser.add_argument("--reserve-fraction", type=float, default=0.25,
                        help="conservative RAM reservation fraction for each shard preflight (default: 0.25)")

    # Supplying all four inputs turns continual checkpoint publication into a
    # held-out absolute + anchor-regression quality transaction. These inputs
    # are immutable for a state; the HMAC key itself remains intentionally
    # absent from the durable journal.
    parser.add_argument("--quality-cases", type=Path, help="strict held-out semantic-neighbour case manifest")
    parser.add_argument("--quality-policy", type=Path, help="strict absolute held-out quality policy")
    parser.add_argument("--quality-redaction-key-file", type=Path,
                        help="local HMAC key used only to redact evaluator reports")
    parser.add_argument("--quality-regression-policy", type=Path,
                        help="immutable maximum-regression policy for continuous training")
    parser.add_argument("--quality-max-cases", type=int, default=QUALITY_MAX_CASES,
                        help=f"quality case safety bound (default: {QUALITY_MAX_CASES})")
    parser.add_argument("--quality-scratch-dir", type=Path,
                        help="private local directory for evaluator snapshots (default: OS temporary directory)")
    parser.add_argument("--quality-timeout-seconds", type=float, default=120.0,
                        help="per-inspect quality evaluator timeout (default: 120)")

    # These define a new state.  If a state already exists, any supplied value
    # must match its immutable policy rather than silently changing semantics.
    parser.add_argument("--max-vocab", type=int)
    parser.add_argument("--min-frequency", type=int)
    parser.add_argument("--window", type=int)
    parser.add_argument("--position-shift", type=int)
    parser.add_argument("--context-mode", choices=sorted(MODES))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--initial-plasticity", type=int)
    parser.add_argument("--minimum-plasticity", type=int)
    parser.add_argument("--decay-events", type=int)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        require(arguments.poll_seconds > 0.0, "--poll-seconds must be positive")
        require(arguments.max_shards is None or arguments.max_shards > 0, "--max-shards must be positive")
        require(arguments.omp_threads > 0, "--omp-threads must be positive")
        require(math.isfinite(arguments.train_timeout_seconds) and arguments.train_timeout_seconds > 0.0,
                "--train-timeout-seconds must be finite and positive")
        require(math.isfinite(arguments.reserve_fraction) and 0.0 <= arguments.reserve_fraction <= 4.0,
                "--reserve-fraction must be finite and in [0, 4]")
        binary = arguments.pretrain_bin.resolve()
        manifest_path = arguments.manifest.resolve()
        state_directory = arguments.state_dir.resolve()
        require(binary.is_file() and not binary.is_symlink() and os.access(binary, os.X_OK),
                f"pretraining executable is unusable: {binary}")
        require(manifest_path.is_file(), f"manifest does not exist: {manifest_path}")
        binary_identity = regular_file_identity(binary, "pretraining executable")
        quality_guard = quality_guard_from_arguments(arguments)
        state_directory.mkdir(parents=True, exist_ok=True)
        (state_directory / GENERATIONS_DIRECTORY).mkdir(parents=True, exist_ok=True)

        with ControllerLock(state_directory / LOCK_FILENAME, arguments.recover_lock):
            cleanup_private_work(state_directory / GENERATIONS_DIRECTORY)
            cleanup_private_quality_work(state_directory / QUALITY_DIRECTORY)
            state = load_state(
                state_directory, arguments, quality_guard, binary_identity,
                enforce_quality_guard=True, enforce_binary_identity=True,
            )
            while True:
                state, committed = process_available(
                    state_directory, manifest_path, binary, state, arguments.max_shards,
                    arguments.expect_manifest_sha256, arguments,
                )
                if not arguments.follow:
                    break
                if arguments.max_shards is not None and committed >= arguments.max_shards:
                    break
                time.sleep(arguments.poll_seconds)
        print(
            f"continuous pretraining: PASS generation={state['generation']} "
            f"committed_shards={len(state['applied_shards'])}"
        )
        return 0
    except KeyboardInterrupt:
        print("continuous pretraining: interrupted; durable state remains at the last committed generation", file=sys.stderr)
        return 130
    except (ContinuousPretrainError, OSError, ValueError) as error:
        print(f"continuous pretraining: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
