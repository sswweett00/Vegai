#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run auditable, exactly-once production HDC pretraining from a built dataset.

This controller composes Vegai's real ``continuous_pretrain.py`` transaction
engine with the content-addressed dataset built by ``production_dataset.py``.
It adds the operational controls commonly required around large-model training:

* immutable dataset/binary identity and resolved-policy provenance;
* an exact disk-backed train/validation record-leakage audit by default;
* RAM/checkpoint-space preflight before child training starts;
* a single-writer run lock plus atomic planned/result/failure reports;
* an optional operator-held-out semantic-neighbour checkpoint quality gate with
  HMAC-redacted report provenance; and
* a real shard-level exactly-once checkpoint ledger, never a simulated metric.

The scope is intentional: this orchestrates the repository's deterministic
HDC/VSA learner.  It does not claim to turn its packed-bit model into a
Transformer, multimodal foundation model, or distributed gradient engine.
Those need a different model/runtime contract rather than misleading flags.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

from continuous_pretrain import (
    ContinuousPretrainError,
    file_sha256,
    load_manifest,
    verify_shard_bytes,
)
from corpus_manifest import DEFAULT_MAX_LINE_BYTES, DEFAULT_MAX_TOKENS_PER_LINE, DEFAULT_TOKEN_LENGTH
from hardware_compatibility import collect_hardware_profile
from production_dataset import DATASET_SCHEMA, DIGEST_INDEX_SCHEMA, DatasetBuildError, iter_strict_lf_records
from pretraining_quality_evaluation import MAX_CASES as QUALITY_MAX_CASES
from pretraining_quality_evaluation import MAX_CHECKPOINT_BYTES as QUALITY_MAX_CHECKPOINT_BYTES
from pretraining_quality_evaluation import (
    PretrainingQualityEvaluationError,
    QualityPolicy,
    evaluator_binding,
    gate_failures,
    scratch_directory,
    wilson_lower_bound_95,
)
from training_resource_plan import build_plan


ROOT = Path(__file__).resolve().parents[1]
RUN_SCHEMA = "vegai-production-pretrain-run/v1"
RESULT_SCHEMA = "vegai-production-pretrain-result/v1"
LOCK_FILENAME = ".production-pretrain.lock"
RUN_FILENAME = "run.json"
RESULT_FILENAME = "result.json"
FAILURE_FILENAME = "failure.json"
QUALITY_REPORT_FILENAME = "quality-report.json"
STATE_FILENAME = "state.json"
BASIS_POINTS = 10_000
QUALITY_EVALUATOR = ROOT / "tools" / "pretraining_quality_evaluation.py"


class ProductionRunError(RuntimeError):
    """Expected preflight, provenance, or child-process failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionRunError(message)


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


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProductionRunError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def resolve_inside(parent: Path, relative: object, label: str) -> Path:
    require(isinstance(relative, str) and relative, f"{label} path is missing")
    candidate = (parent / relative).resolve()
    try:
        candidate.relative_to(parent.resolve())
    except ValueError as error:
        raise ProductionRunError(f"{label} escapes dataset directory: {relative}") from error
    return candidate


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def policy_from_arguments(arguments: argparse.Namespace, quality_gate: dict[str, Any]) -> dict[str, Any]:
    policy = {
        "max_vocabulary": arguments.max_vocab,
        "min_frequency": arguments.min_frequency,
        "context_window": arguments.window,
        "position_shift_bits": arguments.position_shift,
        "context_mode": arguments.context_mode,
        "epochs_per_shard": arguments.epochs,
        "initial_plasticity_percent": arguments.initial_plasticity,
        "minimum_plasticity_percent": arguments.minimum_plasticity,
        "plasticity_decay_events": arguments.decay_events,
    }
    if quality_gate["mode"] != "not-configured":
        # The quality evidence itself is part of the immutable run identity:
        # a resumed controller must not silently swap held-out annotations or
        # thresholds after training has begun. Secret redaction-key bytes are
        # intentionally excluded; they affect report privacy, not semantics.
        # Omitting an unconfigured gate preserves resumability of historical
        # v1 operational runs with their pre-existing training-only policy.
        policy["quality_gate"] = quality_gate
    require(policy["max_vocabulary"] > 0, "--max-vocab must be positive")
    require(policy["min_frequency"] == 1, "production continual path requires --min-frequency 1")
    require(policy["context_window"] > 0, "--window must be positive")
    require(policy["position_shift_bits"] != 0, "--position-shift must be nonzero")
    require(policy["context_mode"] in {"legacy", "positional", "hybrid", "hierarchical", "coherent"}, "invalid --context-mode")
    require(policy["epochs_per_shard"] > 0, "--epochs must be positive")
    require(0 < policy["minimum_plasticity_percent"] <= policy["initial_plasticity_percent"] <= 100,
            "plasticity policy is invalid")
    require(policy["plasticity_decay_events"] > 0, "--decay-events must be positive")
    return policy


def quality_gate_binding_from_arguments(arguments: argparse.Namespace) -> dict[str, Any]:
    """Validate optional held-out evidence before any mutable training action.

    The legacy production controller remains usable for operational checkpoint
    construction without a semantic release decision.  Supplying all three
    evidence inputs enables a fail-closed gate for a completed checkpoint; an
    explicit ``--require-quality-gate`` forbids the ungated mode.
    """
    supplied = [
        arguments.quality_cases is not None,
        arguments.quality_policy is not None,
        arguments.quality_redaction_key_file is not None,
    ]
    if not any(supplied):
        require(not arguments.require_quality_gate,
                "--require-quality-gate needs --quality-cases, --quality-policy, and --quality-redaction-key-file")
        require(arguments.quality_scratch_dir is None,
                "--quality-scratch-dir requires configured --quality-cases, --quality-policy, and --quality-redaction-key-file")
        return {"mode": "not-configured"}
    require(all(supplied),
            "quality evaluation requires --quality-cases, --quality-policy, and --quality-redaction-key-file together")
    require(math.isfinite(arguments.quality_timeout_seconds) and arguments.quality_timeout_seconds > 0.0,
            "--quality-timeout-seconds must be a finite positive number")
    require(0 < arguments.quality_max_cases <= QUALITY_MAX_CASES,
            f"--quality-max-cases must be in [1, {QUALITY_MAX_CASES}]")
    try:
        binding = evaluator_binding(
            arguments.quality_cases,
            arguments.quality_policy,
            arguments.quality_redaction_key_file,
            arguments.quality_max_cases,
        )
    except PretrainingQualityEvaluationError as error:
        raise ProductionRunError(f"quality-gate preflight failed: {error}") from error
    try:
        scratch = scratch_directory(arguments.quality_scratch_dir)
    except PretrainingQualityEvaluationError as error:
        raise ProductionRunError(f"quality scratch preflight failed: {error}") from error
    if scratch is None:
        try:
            scratch = Path(tempfile.gettempdir()).resolve(strict=True)
        except OSError as error:
            raise ProductionRunError(f"cannot resolve default quality scratch directory: {error}") from error
    require(scratch.is_dir(), f"quality scratch directory is not a directory: {scratch}")
    binding["maximum_cases"] = arguments.quality_max_cases
    binding["timeout_seconds"] = arguments.quality_timeout_seconds
    binding["scratch_directory"] = str(scratch)
    return binding


@dataclass(frozen=True)
class DatasetContract:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    payload: dict[str, Any]
    train_ledger: Path
    train_ledger_sha256: str
    training_unique_tokens_min_frequency_1: int
    validation_ledger: Path | None
    validation_ledger_sha256: str | None
    train_shards: list[Any]
    validation_shards: list[Any]


def require_sha256(value: object, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower()),
            f"{label} must be a SHA-256 hex digest")
    return value.lower()


def require_split_shards_inside(shards: list[Any], split_directory: Path, label: str) -> None:
    """Production packages cannot use ledger-relative paths to escape a split."""
    root = split_directory.resolve()
    for position, shard in enumerate(shards, start=1):
        try:
            shard.path.resolve().relative_to(root)
        except ValueError as error:
            raise ProductionRunError(
                f"{label} shard {position} escapes its immutable split directory: {shard.path}"
            ) from error
        require(shard.path.name == f"{shard.sha256}.txt",
                f"{label} shard {position} is not stored under its content-addressed SHA-256 filename")


def validate_split_metadata(metadata: dict[str, Any], shards: list[Any], label: str) -> None:
    """Bind dataset-level counters to signed-by-digest ledger contents."""
    records = sum(shard.records for shard in shards)
    tokens = sum(shard.accepted_tokens for shard in shards)
    require(type(metadata.get("records")) is int and metadata["records"] == records,
            f"{label} record total disagrees with its ledger")
    require(type(metadata.get("accepted_tokens")) is int and metadata["accepted_tokens"] == tokens,
            f"{label} accepted-token total disagrees with its ledger")
    require(type(metadata.get("shards")) is int and metadata["shards"] == len(shards),
            f"{label} shard count disagrees with its ledger")
    require_sha256(metadata.get("record_digest_stream_sha256"), f"{label} record digest stream")


def load_dataset_contract(path: Path) -> DatasetContract:
    manifest_path = path.resolve()
    manifest_sha256 = file_sha256(manifest_path)
    payload = read_json(manifest_path, "production dataset manifest")
    require(file_sha256(manifest_path) == manifest_sha256,
            f"production dataset manifest changed while being read: {manifest_path}")
    require(payload.get("schema") == DATASET_SCHEMA, f"unsupported dataset schema: {payload.get('schema')!r}")
    root = manifest_path.parent
    provenance = payload.get("declared_provenance")
    require(
        isinstance(provenance, dict)
        and all(isinstance(provenance.get(key), str) and provenance[key]
                for key in ("source", "license", "retrieved_at", "preprocessing")),
        "dataset omitted required source/license/retrieval/preprocessing provenance",
    )
    contract = payload.get("vegai_ascii_ingest_contract")
    require(
        isinstance(contract, dict)
        and contract.get("max_line_bytes") == DEFAULT_MAX_LINE_BYTES
        and contract.get("token_length_bytes") == DEFAULT_TOKEN_LENGTH
        and contract.get("max_tokens_per_line") == DEFAULT_MAX_TOKENS_PER_LINE
        and contract.get("require_ascii_only") is True
        and contract.get("require_lf_only") is True
        and contract.get("require_no_dropped_tokens") is True,
        "dataset is not the strict Vegai ASCII/LF ingest contract",
    )
    source = payload.get("source")
    require(isinstance(source, dict), "dataset omitted source snapshot metadata")
    require(isinstance(source.get("input_filename"), str) and source["input_filename"],
            "dataset source snapshot filename is malformed")
    require_sha256(source.get("sha256"), "source snapshot digest")
    require(all(type(source.get(key)) is int and source[key] >= 0 for key in (
        "bytes", "physical_lf_records", "empty_or_punctuation_records_filtered",
    )), "dataset source snapshot counters are malformed")
    # Rich JSON/Arrow/Parquet sources are projected once into canonical strict
    # text before packaging. Old TXT-only v1 packages have no adapter field and
    # remain valid; new packages bind raw source provenance to that text stream.
    input_adapter = payload.get("input_adapter")
    if input_adapter is not None:
        require(isinstance(input_adapter, dict), "dataset input_adapter must be an object")
        require(
            input_adapter.get("schema") == "vegai-training-data-adapter/v1"
            and input_adapter.get("source_format") in {"txt", "jsonl", "json", "parquet", "arrow"}
            and isinstance(input_adapter.get("contract"), str) and bool(input_adapter["contract"]),
            "dataset input_adapter schema, source format, or projection contract is malformed",
        )
        require_sha256(input_adapter.get("source_sha256"), "input-adapter raw source digest")
        require_sha256(input_adapter.get("canonical_hdc_text_sha256"), "input-adapter canonical text digest")
        require(
            input_adapter["canonical_hdc_text_sha256"] == source["sha256"]
            and type(input_adapter.get("source_bytes")) is int and input_adapter["source_bytes"] >= 0
            and type(input_adapter.get("canonical_hdc_text_bytes")) is int
            and input_adapter["canonical_hdc_text_bytes"] == source["bytes"],
            "dataset input_adapter does not bind the canonical strict text source snapshot",
        )
    deduplication = payload.get("deduplication")
    require(
        isinstance(deduplication, dict)
        and deduplication.get("mode") == "exact"
        and deduplication.get("identity") == "sha256(raw-record-without-lf)"
        and deduplication.get("sqlite_index_schema") == DIGEST_INDEX_SCHEMA
        and type(deduplication.get("duplicates_removed")) is int
        and deduplication["duplicates_removed"] >= 0
        and type(deduplication.get("retained_records")) is int
        and deduplication["retained_records"] > 0,
        "production pretraining requires exact raw-record SHA-256 deduplication",
    )
    split = payload.get("split")
    require(
        isinstance(split, dict)
        and split.get("algorithm") == "sha256-first-64-bits-mod-10000/v1"
        and type(split.get("validation_basis_points")) is int
        and 0 <= split["validation_basis_points"] < BASIS_POINTS
        and split.get("train_validation_record_identity_disjoint") is True,
        "dataset deterministic split contract is malformed",
    )
    train = payload.get("train")
    require(isinstance(train, dict), "dataset omitted training split")
    train_ledger = resolve_inside(root, train.get("ledger"), "training ledger")
    require(train_ledger.parent == (root / "train").resolve(),
            "training ledger must be train/ledger.json inside the dataset package")
    train_ledger_sha256 = require_sha256(train.get("ledger_sha256"), "training ledger digest")
    require(train_ledger.is_file(), f"training ledger does not exist: {train_ledger}")
    require(file_sha256(train_ledger) == train_ledger_sha256, "training ledger SHA-256 mismatch")
    try:
        train_shards = load_manifest(train_ledger, train_ledger_sha256)
    except ContinuousPretrainError as error:
        raise ProductionRunError(f"training ledger is invalid: {error}") from error
    require(train_shards, "training ledger contains no shards")
    require_split_shards_inside(train_shards, train_ledger.parent, "training")
    validate_split_metadata(train, train_shards, "training")
    training_unique_tokens = train.get("unique_tokens_min_frequency_1")
    require(type(training_unique_tokens) is int and 0 < training_unique_tokens <= train["accepted_tokens"],
            "training min-frequency-1 unique-token count is malformed")

    validation = payload.get("validation")
    validation_ledger: Path | None = None
    validation_ledger_sha256: str | None = None
    validation_shards: list[Any] = []
    if validation is not None:
        require(isinstance(validation, dict), "validation split must be null or an object")
        validation_ledger = resolve_inside(root, validation.get("ledger"), "validation ledger")
        require(validation_ledger.parent == (root / "validation").resolve(),
                "validation ledger must be validation/ledger.json inside the dataset package")
        validation_ledger_sha256 = require_sha256(validation.get("ledger_sha256"), "validation ledger digest")
        require(validation_ledger.is_file(), f"validation ledger does not exist: {validation_ledger}")
        require(file_sha256(validation_ledger) == validation_ledger_sha256, "validation ledger SHA-256 mismatch")
        try:
            validation_shards = load_manifest(validation_ledger, validation_ledger_sha256)
        except ContinuousPretrainError as error:
            raise ProductionRunError(f"validation ledger is invalid: {error}") from error
        require(validation_shards, "validation ledger contains no shards")
        require_split_shards_inside(validation_shards, validation_ledger.parent, "validation")
        validate_split_metadata(validation, validation_shards, "validation")

    # Ledger metadata/path containment is now checked.  Verify every actual
    # payload here as well, before the resource planner, leakage
    # scanner, or real child process can act on a rewritten corpus.
    try:
        for shard in [*train_shards, *validation_shards]:
            verify_shard_bytes(shard)
    except ContinuousPretrainError as error:
        raise ProductionRunError(f"dataset shard integrity verification failed: {error}") from error

    total_retained_records = train["records"] + (0 if validation is None else validation["records"])
    require(deduplication["retained_records"] == total_retained_records,
            "dataset retained-record accounting disagrees with split totals")
    require(
        source["physical_lf_records"] - source["empty_or_punctuation_records_filtered"]
        == deduplication["retained_records"] + deduplication["duplicates_removed"],
        "dataset source/deduplication record accounting is inconsistent",
    )
    require((split["validation_basis_points"] > 0) == (validation is not None),
            "dataset validation presence disagrees with split policy")

    return DatasetContract(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        payload=payload,
        train_ledger=train_ledger,
        train_ledger_sha256=train_ledger_sha256,
        training_unique_tokens_min_frequency_1=training_unique_tokens,
        validation_ledger=validation_ledger,
        validation_ledger_sha256=validation_ledger_sha256,
        train_shards=train_shards,
        validation_shards=validation_shards,
    )


def audit_exact_split_leakage(contract: DatasetContract, scratch_directory: Path) -> dict[str, Any]:
    """Prove split isolation and validate foundry record-stream commitments."""
    scratch_directory.mkdir(parents=True, exist_ok=True)
    database = scratch_directory / ".split-leakage.sqlite"
    connection: sqlite3.Connection | None = None
    train_records = 0
    validation_records = 0
    train_stream = hashlib.sha256()
    validation_stream = hashlib.sha256()
    try:
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE train_record (digest BLOB PRIMARY KEY)")
        connection.execute("CREATE TABLE validation_record (digest BLOB PRIMARY KEY)")
        for shard in contract.train_shards:
            for _, record in iter_strict_lf_records(shard.path):
                digest = hashlib.sha256(record).digest()
                try:
                    connection.execute("INSERT INTO train_record(digest) VALUES (?)", (digest,))
                except sqlite3.IntegrityError as error:
                    raise ProductionRunError("exact duplicate record found within the training split") from error
                train_stream.update(digest)
                train_records += 1
                if train_records % 4096 == 0:
                    connection.commit()
        connection.commit()
        expected_train = contract.payload["train"]
        require(train_records == expected_train["records"], "training record stream count changed after foundry publication")
        require(train_stream.hexdigest() == expected_train["record_digest_stream_sha256"],
                "training record identity stream changed after foundry publication")

        overlap = 0
        for shard in contract.validation_shards:
            for _, record in iter_strict_lf_records(shard.path):
                digest = hashlib.sha256(record).digest()
                try:
                    connection.execute("INSERT INTO validation_record(digest) VALUES (?)", (digest,))
                except sqlite3.IntegrityError as error:
                    raise ProductionRunError("exact duplicate record found within the validation split") from error
                validation_stream.update(digest)
                validation_records += 1
                if connection.execute("SELECT 1 FROM train_record WHERE digest = ?", (digest,)).fetchone():
                    overlap += 1
                if validation_records % 4096 == 0:
                    connection.commit()
        connection.commit()
        if contract.validation_shards:
            expected_validation = contract.payload["validation"]
            require(validation_records == expected_validation["records"],
                    "validation record stream count changed after foundry publication")
            require(validation_stream.hexdigest() == expected_validation["record_digest_stream_sha256"],
                    "validation record identity stream changed after foundry publication")
        require(overlap == 0, f"exact train/validation leakage audit found {overlap} shared record identity/identities")
        return {
            "mode": "exact-sha256-record-sqlite/v1" if contract.validation_shards else "exact-training-dedup-only/v1",
            "train_records": train_records,
            "validation_records": validation_records,
            "overlap_records": overlap,
        }
    except DatasetBuildError as error:
        raise ProductionRunError(f"cannot complete exact split-leakage audit: {error}") from error
    except sqlite3.Error as error:
        raise ProductionRunError(f"cannot complete exact split-leakage audit: {error}") from error
    finally:
        if connection is not None:
            connection.close()
        database.unlink(missing_ok=True)


def plan_resources(contract: DatasetContract, arguments: argparse.Namespace, checkpoint: Path) -> dict[str, Any]:
    first_shard = contract.train_shards[0].path
    namespace = SimpleNamespace(
        corpus=first_shard,
        checkpoint=checkpoint,
        max_vocab=arguments.max_vocab,
        omp_threads=arguments.omp_threads,
        reserve_fraction=arguments.reserve_fraction,
        require_accelerator=False,
    )
    try:
        plan = build_plan(namespace)
    except Exception as error:
        raise ProductionRunError(f"resource preflight failed: {error}") from error
    require(plan["verdict"]["ready"], "resource preflight is not ready: " + "; ".join(plan["verdict"]["errors"]))
    return plan


QUALITY_SCRATCH_METADATA_BYTES = 2 * 1024 * 1024


def quality_snapshot_required_bytes(checkpoint_bytes: int, binary_bytes: int, maximum_cases: int) -> int:
    """Conservative local scratch requirement for private evaluator snapshots."""
    require(checkpoint_bytes >= 0 and binary_bytes >= 0, "quality snapshot byte estimates must be nonnegative")
    require(0 < maximum_cases <= QUALITY_MAX_CASES, "quality maximum case bound is invalid")
    # Query rows are <=48 bytes plus LF. The inspect stream itself is bounded
    # in memory, so only the binary/checkpoint copies and modest directory/query
    # metadata need filesystem capacity.
    return checkpoint_bytes + binary_bytes + maximum_cases * 49 + QUALITY_SCRATCH_METADATA_BYTES


def quality_scratch_capacity(
    quality_binding: dict[str, Any], binary: Path, checkpoint_bytes: int
) -> dict[str, Any]:
    """Fail before a snapshot can exhaust the operator-selected local volume."""
    if quality_binding["mode"] == "not-configured":
        return {"mode": "not-configured"}
    scratch_value = quality_binding.get("scratch_directory")
    require(isinstance(scratch_value, str) and scratch_value,
            "quality binding omitted scratch directory")
    scratch = Path(scratch_value)
    require(scratch.is_dir(), f"quality scratch directory is unavailable: {scratch}")
    try:
        free_bytes = shutil.disk_usage(scratch).free
        binary_bytes = binary.stat().st_size
    except OSError as error:
        raise ProductionRunError(f"cannot inspect quality scratch capacity: {error}") from error
    required_bytes = quality_snapshot_required_bytes(
        checkpoint_bytes, binary_bytes, int(quality_binding["maximum_cases"])
    )
    require(free_bytes >= required_bytes,
            "quality scratch free space is below required private binary/checkpoint snapshot capacity "
            f"({free_bytes} < {required_bytes} bytes)")
    return {
        "mode": "private-snapshot/v1",
        "directory": str(scratch),
        "free_bytes": free_bytes,
        "required_bytes": required_bytes,
        "checkpoint_bytes_basis": checkpoint_bytes,
        "binary_bytes_basis": binary_bytes,
        "query_case_bound": quality_binding["maximum_cases"],
    }


def quality_scratch_preflight(
    quality_binding: dict[str, Any], binary: Path, resource_plan: dict[str, Any]
) -> dict[str, Any]:
    if quality_binding["mode"] == "not-configured":
        return {"mode": "not-configured"}
    checkpoint = resource_plan.get("checkpoint")
    require(isinstance(checkpoint, dict) and type(checkpoint.get("estimated_max_bytes")) is int,
            "resource plan omitted estimated checkpoint bytes for quality scratch preflight")
    require(checkpoint["estimated_max_bytes"] <= QUALITY_MAX_CHECKPOINT_BYTES,
            "estimated checkpoint exceeds the quality evaluator's private-snapshot cap "
            f"({checkpoint['estimated_max_bytes']} > {QUALITY_MAX_CHECKPOINT_BYTES} bytes)")
    return quality_scratch_capacity(quality_binding, binary, checkpoint["estimated_max_bytes"])


class RunLock:
    def __init__(self, path: Path, recover_stale: bool) -> None:
        self.path = path
        self.acquired = False
        # This is intentionally opt-in: PID files cannot prove that a process
        # on every OS/filesystem is dead, so automatic deletion would risk two
        # writers.  The operator must inspect a stale lock before recovery.
        if recover_stale:
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> "RunLock":
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise ProductionRunError(f"production run lock exists: {self.path}; another controller may be active") from error
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


def run_fingerprint(
    contract: DatasetContract, binary_sha256: str, policy: dict[str, Any], state_directory: Path
) -> dict[str, Any]:
    return {
        "dataset_manifest_sha256": contract.manifest_sha256,
        "training_ledger_sha256": contract.train_ledger_sha256,
        "validation_ledger_sha256": contract.validation_ledger_sha256,
        "pretrain_binary_sha256": binary_sha256,
        "state_directory": str(state_directory.resolve()),
        "policy": policy,
    }


def initialize_or_verify_run(
    run_directory: Path,
    allow_resume: bool,
    fingerprint: dict[str, Any],
    contract: DatasetContract,
    binary: Path,
    state_directory: Path,
    resource_plan: dict[str, Any],
    vocabulary_preflight: dict[str, int],
    quality_scratch_preflight: dict[str, Any],
) -> dict[str, Any]:
    run_path = run_directory / RUN_FILENAME
    if run_path.exists():
        require(allow_resume, f"run directory already has {RUN_FILENAME}; pass --resume-run only after reviewing it")
        current = read_json(run_path, "existing run manifest")
        require(current.get("schema") == RUN_SCHEMA, "existing run manifest has incompatible schema")
        require(current.get("fingerprint") == fingerprint,
                "existing run identity differs in dataset, binary, or immutable policy")
        require(current.get("vocabulary_preflight") == vocabulary_preflight,
                "existing run vocabulary-capacity preflight is inconsistent")
        return current
    require(not run_directory.exists() or not any(run_directory.iterdir()),
            f"run directory exists and is not empty: {run_directory}")
    require(not state_directory.exists() or not any(state_directory.iterdir()),
            f"new run refuses a nonempty continuous state directory: {state_directory}; use its matching --resume-run")
    run_directory.mkdir(parents=True, exist_ok=True)
    planned = {
        "schema": RUN_SCHEMA,
        "fingerprint": fingerprint,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository_revision": git_revision(),
        "dataset_manifest": str(contract.manifest_path),
        "training_ledger": str(contract.train_ledger),
        "validation_ledger": None if contract.validation_ledger is None else str(contract.validation_ledger),
        "pretrain_binary": str(binary),
        "state_directory": str(state_directory),
        "vocabulary_preflight": vocabulary_preflight,
        "resource_plan": resource_plan,
        "quality_scratch_preflight": quality_scratch_preflight,
        "hardware": collect_hardware_profile(),
        "contract": (
            "Shard-level exactly-once mutation is committed by continuous_pretrain.py state.json. "
            "This planned manifest is provenance, not a substitute for its generation journal."
        ),
    }
    atomic_write_json(run_path, planned)
    return planned


def continuous_command(
    binary: Path, ledger: Path, ledger_sha256: str, state_directory: Path, arguments: argparse.Namespace
) -> list[str]:
    command = [
        sys.executable, str(ROOT / "tools" / "continuous_pretrain.py"),
        "--pretrain-bin", str(binary), "--manifest", str(ledger), "--state-dir", str(state_directory),
        "--expect-manifest-sha256", ledger_sha256,
        "--max-vocab", str(arguments.max_vocab), "--min-frequency", str(arguments.min_frequency),
        "--window", str(arguments.window), "--position-shift", str(arguments.position_shift),
        "--context-mode", arguments.context_mode, "--epochs", str(arguments.epochs),
        "--initial-plasticity", str(arguments.initial_plasticity),
        "--minimum-plasticity", str(arguments.minimum_plasticity), "--decay-events", str(arguments.decay_events),
    ]
    if arguments.max_shards is not None:
        command.extend(["--max-shards", str(arguments.max_shards)])
    if arguments.recover_locks:
        command.append("--recover-lock")
    return command


def result_from_state(state_path: Path) -> dict[str, Any]:
    state = read_json(state_path, "continuous state")
    applied = state.get("applied_shards")
    require(isinstance(applied, list), "continuous state is malformed after training")
    total_tokens = 0
    for entry in applied:
        require(isinstance(entry, dict) and isinstance(entry.get("accepted_tokens"), int),
                "continuous state contains malformed applied shard")
        total_tokens += entry["accepted_tokens"]
    return {
        "generation": state.get("generation"),
        "committed_shards": len(applied),
        "accepted_training_tokens": total_tokens,
        "checkpoint": state.get("checkpoint"),
        "checkpoint_sha256": state.get("checkpoint_sha256"),
    }


def read_quality_report_snapshot(report_path: Path) -> tuple[dict[str, Any], str, int]:
    """Read a bounded report once and reject duplicate-key/replacement tricks."""
    try:
        before = report_path.stat()
        require(stat.S_ISREG(before.st_mode), "quality evaluator report is not a regular file")
        require(before.st_size <= 32 * 1024 * 1024,
                "quality evaluator report exceeds the 32 MiB production safety bound")
        raw = report_path.read_bytes()
        after = report_path.stat()
    except OSError as error:
        raise ProductionRunError(f"cannot read quality evaluator report {report_path}: {error}") from error
    require(
        (before.st_dev, before.st_ino, before.st_size) == (after.st_dev, after.st_ino, after.st_size)
        and len(raw) == before.st_size,
        "quality evaluator report changed while it was read",
    )

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ProductionRunError(f"quality evaluator report repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionRunError(f"quality evaluator report is malformed JSON: {error}") from error
    require(isinstance(parsed, dict), "quality evaluator report must be a JSON object")
    return parsed, hashlib.sha256(raw).hexdigest(), len(raw)


def report_number(value: object, label: str, minimum: float, maximum: float) -> float:
    require(type(value) in {int, float}, f"quality evaluator report {label} must be numeric")
    parsed = float(value)
    require(math.isfinite(parsed) and minimum <= parsed <= maximum,
            f"quality evaluator report {label} is outside its finite bounds")
    return parsed


def require_report_close(actual: object, expected: float, label: str) -> None:
    parsed = report_number(actual, label, -1.0, 1.0)
    require(math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1e-12),
            f"quality evaluator report {label} disagrees with its case results")


def validate_quality_report_measurement(
    report: dict[str, Any], quality_binding: dict[str, Any], gate: dict[str, Any]
) -> None:
    """Recompute the public aggregate gate from privacy-redacted case rows.

    The evaluator source is already SHA-bound, but its atomically published
    report is still treated as an untrusted child artifact.  This prevents a
    malformed/replaced report from claiming a pass merely by toggling its
    aggregate booleans or metrics.  The recomputation needs no HMAC key and
    therefore keeps raw held-out labels out of the production controller.
    """
    expected_requirements = quality_binding.get("requirements")
    require(isinstance(expected_requirements, dict), "quality binding omitted gate requirements")
    required_requirement_fields = {
        "minimum_cases", "minimum_top1_accuracy", "minimum_top1_wilson_lower_bound_95",
        "minimum_mean_similarity", "minimum_case_similarity", "maximum_incorrect_cases",
    }
    require(set(expected_requirements) == required_requirement_fields,
            "quality binding has malformed gate requirements")
    require(gate.get("requirements") == expected_requirements,
            "quality evaluator report requirements differ from the immutable run binding")

    evaluation = report.get("evaluation")
    results = report.get("case_results")
    metrics = report.get("metrics")
    require(isinstance(evaluation, dict) and isinstance(results, list) and isinstance(metrics, dict),
            "quality evaluator report omitted evaluation, case results, or metrics")
    expected_count = quality_binding.get("case_count")
    require(type(expected_count) is int and expected_count > 0,
            "quality binding has malformed case count")
    require(evaluation.get("declared_split") == "held-out",
            "quality evaluator report is not declared held-out")
    require(evaluation.get("case_count") == expected_count and len(results) == expected_count,
            "quality evaluator report case count differs from immutable run binding")

    correct = 0
    similarities: list[float] = []
    result_fields = {
        "case_number", "case_id_hmac_sha256", "query_token_hmac_sha256",
        "acceptable_neighbour_set_hmac_sha256", "observed_neighbour_hmac_sha256",
        "top1_acceptable", "bipolar_similarity",
    }
    for number, result in enumerate(results, start=1):
        require(isinstance(result, dict) and set(result) == result_fields,
                "quality evaluator report has malformed or non-private case result fields")
        require(type(result.get("case_number")) is int and result["case_number"] == number,
                "quality evaluator report case numbering is not contiguous")
        for field in (
            "case_id_hmac_sha256", "query_token_hmac_sha256",
            "acceptable_neighbour_set_hmac_sha256", "observed_neighbour_hmac_sha256",
        ):
            require_sha256(result.get(field), f"quality evaluator report {field}")
        require(type(result.get("top1_acceptable")) is bool,
                "quality evaluator report top1_acceptable must be boolean")
        correct += int(result["top1_acceptable"])
        similarities.append(report_number(result.get("bipolar_similarity"), "case similarity", -1.0, 1.0))

    total = len(results)
    incorrect = total - correct
    top1_accuracy = correct / total
    top1_wilson = wilson_lower_bound_95(correct, total)
    mean_similarity = math.fsum(similarities) / total
    minimum_similarity = min(similarities)
    maximum_similarity = max(similarities)
    require(type(metrics.get("correct_top1_cases")) is int and metrics["correct_top1_cases"] == correct,
            "quality evaluator report correct-case total disagrees with case results")
    require(type(metrics.get("incorrect_top1_cases")) is int and metrics["incorrect_top1_cases"] == incorrect,
            "quality evaluator report incorrect-case total disagrees with case results")
    require_report_close(metrics.get("top1_accuracy"), top1_accuracy, "top1_accuracy")
    require_report_close(metrics.get("top1_wilson_lower_bound_95"), top1_wilson, "top1_wilson_lower_bound_95")
    require_report_close(metrics.get("mean_bipolar_similarity"), mean_similarity, "mean_bipolar_similarity")
    require_report_close(metrics.get("minimum_bipolar_similarity"), minimum_similarity, "minimum_bipolar_similarity")
    require_report_close(metrics.get("maximum_bipolar_similarity"), maximum_similarity, "maximum_bipolar_similarity")

    policy = QualityPolicy(
        policy_id="production-bound-quality-policy",
        minimum_cases=expected_requirements["minimum_cases"],
        minimum_top1_accuracy=expected_requirements["minimum_top1_accuracy"],
        minimum_top1_wilson_lower_bound_95=expected_requirements["minimum_top1_wilson_lower_bound_95"],
        minimum_mean_similarity=expected_requirements["minimum_mean_similarity"],
        minimum_case_similarity=expected_requirements["minimum_case_similarity"],
        maximum_incorrect_cases=expected_requirements["maximum_incorrect_cases"],
    )
    failures = gate_failures(
        policy, total, top1_accuracy, top1_wilson, mean_similarity, minimum_similarity, incorrect
    )
    require(gate.get("failures") == failures,
            "quality evaluator report gate failures disagree with recomputed metrics")
    require(gate.get("passed") is (not failures),
            "quality evaluator report gate pass state disagrees with recomputed metrics")


def quality_report_summary(
    report_path: Path,
    quality_binding: dict[str, Any],
    binary_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Verify that a child evaluator's report binds this exact run candidate."""
    report, report_sha256, report_bytes = read_quality_report_snapshot(report_path)
    require(report.get("schema_version") == "vegai-pretraining-quality-report/v1",
            "quality evaluator report has an unsupported schema")
    require(report.get("status") == "completed", "quality evaluator report is not completed")
    evaluator = report.get("evaluator")
    provenance = report.get("provenance")
    gate = report.get("quality_gate")
    require(isinstance(evaluator, dict) and isinstance(provenance, dict) and isinstance(gate, dict),
            "quality evaluator report omitted evaluator, provenance, or quality gate")
    require(evaluator.get("sha256") == quality_binding["evaluator_source_sha256"],
            "quality evaluator report is bound to a different evaluator source")
    expected = {
        "pretrain_binary": binary_sha256,
        "checkpoint": checkpoint_sha256,
        "case_manifest": quality_binding["case_manifest_sha256"],
        "policy": quality_binding["policy_sha256"],
    }
    for label, digest in expected.items():
        value = provenance.get(label)
        require(isinstance(value, dict) and value.get("sha256") == digest,
                f"quality evaluator report has wrong {label} provenance")
    require(type(gate.get("passed")) is bool, "quality evaluator report has malformed quality_gate.passed")
    validate_quality_report_measurement(report, quality_binding, gate)
    return {
        "report_sha256": report_sha256,
        "report_bytes": report_bytes,
        "passed": gate["passed"],
    }


def run_quality_gate(
    binary: Path,
    binary_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    run_directory: Path,
    arguments: argparse.Namespace,
    quality_binding: dict[str, Any],
) -> dict[str, Any]:
    """Run and independently bind a privacy-safe held-out checkpoint gate.

    The production result intentionally records only report byte identity and
    public policy identities.  It does not copy evaluator stdout, case labels,
    token strings, or the HMAC secret path into a durable training manifest.
    """
    if quality_binding["mode"] == "not-configured":
        return {
            "status": "not-configured",
            "release_eligible": False,
            "contract": "No held-out semantic-neighbour release evidence was supplied for this run.",
        }
    report_path = run_directory / QUALITY_REPORT_FILENAME
    command = [
        sys.executable, str(QUALITY_EVALUATOR),
        "--pretrain-bin", str(binary), "--checkpoint", str(checkpoint),
        "--cases", str(arguments.quality_cases), "--policy", str(arguments.quality_policy),
        "--redaction-key-file", str(arguments.quality_redaction_key_file),
        "--output", str(report_path), "--max-cases", str(quality_binding.get("maximum_cases", QUALITY_MAX_CASES)),
        "--scratch-dir", str(quality_binding.get("scratch_directory", tempfile.gettempdir())),
        "--timeout-seconds", str(arguments.quality_timeout_seconds),
    ]
    try:
        checkpoint_bytes = checkpoint.stat().st_size
        require(checkpoint_bytes <= QUALITY_MAX_CHECKPOINT_BYTES,
                "final checkpoint exceeds the quality evaluator's private-snapshot cap "
                f"({checkpoint_bytes} > {QUALITY_MAX_CHECKPOINT_BYTES} bytes)")
        runtime_scratch = quality_scratch_capacity(quality_binding, binary, checkpoint_bytes)
        require(file_sha256(QUALITY_EVALUATOR) == quality_binding["evaluator_source_sha256"],
                "quality evaluator source changed after quality preflight")
        # The evaluator writes its privacy-safe report atomically. Its stdout
        # is deliberately not retained by this wrapper: even a malformed or
        # replaced child must not create an unbounded/sensitive durable log.
        child = subprocess.run(
            command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        require(file_sha256(QUALITY_EVALUATOR) == quality_binding["evaluator_source_sha256"],
                "quality evaluator source changed while quality gate was running")
        report_summary = quality_report_summary(report_path, quality_binding, binary_sha256, checkpoint_sha256)
    except (OSError, ValueError, json.JSONDecodeError, ProductionRunError) as error:
        # Child output can contain operator-specific diagnostics.  The durable
        # production result deliberately carries only this bounded category.
        return {
            "status": "execution-failed",
            "release_eligible": False,
            "reason": "evaluator-or-report-validation-failed",
            "report_published": report_path.is_file(),
        }
    if child.returncode == 0 and report_summary["passed"]:
        return {
            "status": "passed",
            "release_eligible": True,
            "evaluator_exit_code": 0,
            "private_snapshot_scratch": {
                "free_bytes": runtime_scratch["free_bytes"],
                "required_bytes": runtime_scratch["required_bytes"],
            },
            **report_summary,
        }
    if child.returncode == 3 and not report_summary["passed"]:
        return {
            "status": "failed",
            "release_eligible": False,
            "evaluator_exit_code": 3,
            "private_snapshot_scratch": {
                "free_bytes": runtime_scratch["free_bytes"],
                "required_bytes": runtime_scratch["required_bytes"],
            },
            **report_summary,
        }
    return {
        "status": "execution-failed",
        "release_eligible": False,
        "reason": "evaluator-exit-status-disagrees-with-validated-report",
        "evaluator_exit_code": child.returncode,
        "private_snapshot_scratch": {
            "free_bytes": runtime_scratch["free_bytes"],
            "required_bytes": runtime_scratch["required_bytes"],
        },
        **report_summary,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--dataset", required=True, type=Path, help="dataset.json emitted by production_dataset.py")
    parser.add_argument("--run-dir", required=True, type=Path, help="new or resumeable production run directory")
    parser.add_argument("--state-dir", type=Path, help="continuous state directory (default: RUN_DIR/state)")
    parser.add_argument("--resume-run", action="store_true", help="resume only when immutable run identity matches exactly")
    parser.add_argument(
        "--recover-locks", action="store_true",
        help="after operator review, remove stale production and continuous-controller locks before resuming",
    )
    parser.add_argument("--quality-cases", type=Path,
                        help="strict JSON held-out semantic-neighbour cases; supply with policy and redaction key")
    parser.add_argument("--quality-policy", type=Path,
                        help="strict JSON quality thresholds; supply with cases and redaction key")
    parser.add_argument("--quality-redaction-key-file", type=Path,
                        help="bounded local secret for HMAC-redacted quality-report identifiers")
    parser.add_argument("--quality-max-cases", type=int, default=QUALITY_MAX_CASES,
                        help=f"quality-case safety bound when evidence is configured (default: {QUALITY_MAX_CASES})")
    parser.add_argument("--quality-scratch-dir", type=Path,
                        help="local directory for private quality binary/checkpoint snapshots (default: OS temp directory)")
    parser.add_argument("--quality-timeout-seconds", type=float, default=120.0,
                        help="real checkpoint inspection timeout for an optional quality gate (default: 120)")
    parser.add_argument("--require-quality-gate", action="store_true",
                        help="reject an ungated run; requires all three --quality-* evidence inputs")
    parser.add_argument("--preflight-only", action="store_true", help="validate data/leakage/resources and write no training state")
    parser.add_argument("--skip-exact-leakage-audit", action="store_true", help="skip expensive exact held-out audit; recorded explicitly")
    parser.add_argument("--max-shards", type=int, help="commit at most this many pending immutable package shards")
    parser.add_argument("--omp-threads", type=int, help="OMP_NUM_THREADS for real child training")
    parser.add_argument("--reserve-fraction", type=float, default=0.25, help="resource-plan RAM reserve fraction (default: 0.25)")
    parser.add_argument("--max-vocab", type=int, default=20_000)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--position-shift", type=int, default=73)
    parser.add_argument("--context-mode", choices=("legacy", "positional", "hybrid", "hierarchical", "coherent"), default="hierarchical")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--initial-plasticity", type=int, default=25)
    parser.add_argument("--minimum-plasticity", type=int, default=1)
    parser.add_argument("--decay-events", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        require(arguments.max_shards is None or arguments.max_shards > 0, "--max-shards must be positive")
        require(arguments.omp_threads is None or arguments.omp_threads > 0, "--omp-threads must be positive")
        require(0.0 <= arguments.reserve_fraction <= 4.0, "--reserve-fraction must be between 0 and 4")
        quality_binding = quality_gate_binding_from_arguments(arguments)
        policy = policy_from_arguments(arguments, quality_binding)
        binary = arguments.pretrain_bin.resolve()
        require(binary.is_file(), f"pretraining executable does not exist: {binary}")
        require(os.access(binary, os.X_OK), f"pretraining executable is not executable: {binary}")
        contract = load_dataset_contract(arguments.dataset)
        require(policy["max_vocabulary"] >= contract.training_unique_tokens_min_frequency_1,
                "--max-vocab is below the exact min-frequency-1 training vocabulary requirement: "
                f"{policy['max_vocabulary']} < {contract.training_unique_tokens_min_frequency_1}")
        vocabulary_preflight = {
            "min_frequency": policy["min_frequency"],
            "exact_training_unique_tokens": contract.training_unique_tokens_min_frequency_1,
            "max_vocabulary": policy["max_vocabulary"],
            "capacity_margin_tokens": policy["max_vocabulary"] - contract.training_unique_tokens_min_frequency_1,
        }
        run_directory = arguments.run_dir.resolve()
        run_directory.parent.mkdir(parents=True, exist_ok=True)
        state_directory = arguments.state_dir.resolve() if arguments.state_dir is not None else run_directory / "state"
        binary_sha256 = file_sha256(binary)
        fingerprint = run_fingerprint(contract, binary_sha256, policy, state_directory)

        with RunLock(run_directory.parent / (run_directory.name + LOCK_FILENAME), arguments.recover_locks):
            # Keep audit scratch outside the durable continuous state so an
            # interrupted audit cannot look like a candidate generation.
            audit_scratch = run_directory.parent / ("." + run_directory.name + ".audit")
            shutil.rmtree(audit_scratch, ignore_errors=True)
            try:
                if arguments.skip_exact_leakage_audit:
                    leakage = {"mode": "skipped-by-operator", "train_records": None,
                               "validation_records": None, "overlap_records": None}
                else:
                    leakage = audit_exact_split_leakage(contract, audit_scratch)
            finally:
                shutil.rmtree(audit_scratch, ignore_errors=True)
            resource_plan = plan_resources(contract, arguments, state_directory / "planned-checkpoint.vgai")
            quality_scratch = quality_scratch_preflight(quality_binding, binary, resource_plan)
            if arguments.preflight_only:
                quality_preflight = (
                    {"status": "validated-not-run", "release_eligible": False, "binding": quality_binding}
                    if quality_binding["mode"] != "not-configured"
                    else {"status": "not-configured", "release_eligible": False}
                )
                print(json.dumps({"dataset": str(contract.manifest_path), "leakage_audit": leakage,
                                  "vocabulary_preflight": vocabulary_preflight,
                                  "resource_plan": resource_plan, "quality_scratch_preflight": quality_scratch,
                                  "quality_gate": quality_preflight,
                                  "ready": True}, ensure_ascii=False, sort_keys=True))
                return 0
            initialize_or_verify_run(
                run_directory, arguments.resume_run, fingerprint, contract, binary, state_directory,
                resource_plan, vocabulary_preflight, quality_scratch,
            )
            state_directory.mkdir(parents=True, exist_ok=True)
            state_path = state_directory / STATE_FILENAME
            before_summary = (
                result_from_state(state_path)
                if state_path.exists()
                else {"generation": 0, "committed_shards": 0, "accepted_training_tokens": 0}
            )
            command = continuous_command(
                binary, contract.train_ledger, contract.train_ledger_sha256, state_directory, arguments
            )
            environment = os.environ.copy()
            if arguments.omp_threads is not None:
                environment["OMP_NUM_THREADS"] = str(arguments.omp_threads)
                environment.setdefault("OMP_PROC_BIND", "close")
                environment.setdefault("OMP_PLACES", "cores")
            started = time.perf_counter()
            child = subprocess.run(
                command, cwd=ROOT, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            elapsed = time.perf_counter() - started
            if child.returncode != 0:
                failure = {
                    "schema": RESULT_SCHEMA,
                    "status": "failed",
                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "child_exit_code": child.returncode,
                    "elapsed_train_controller_seconds": elapsed,
                    "leakage_audit": leakage,
                    "vocabulary_preflight": vocabulary_preflight,
                    "command": command,
                    "child_output_tail": child.stdout[-8000:],
                }
                # ``result.json`` is the latest-attempt status as well as the
                # convenient success report.  Overwrite a prior success so a
                # failed resume cannot be mistaken for the current outcome;
                # keep the dedicated failure name for operators/log tooling.
                atomic_write_json(run_directory / FAILURE_FILENAME, failure)
                atomic_write_json(run_directory / RESULT_FILENAME, failure)
                raise ProductionRunError(f"continuous pretraining failed ({child.returncode}); see {run_directory / FAILURE_FILENAME}")
            summary = result_from_state(state_path)
            committed_this_invocation = summary["committed_shards"] - before_summary["committed_shards"]
            tokens_this_invocation = summary["accepted_training_tokens"] - before_summary["accepted_training_tokens"]
            require(committed_this_invocation >= 0 and tokens_this_invocation >= 0,
                    "continuous state regressed relative to the production run baseline")
            summary["committed_this_invocation"] = committed_this_invocation
            summary["accepted_training_tokens_this_invocation"] = tokens_this_invocation
            if quality_binding["mode"] != "not-configured" and summary["committed_shards"] < len(contract.train_shards):
                # A partial exactly-once invocation has a valid transactional
                # checkpoint, but it is not a candidate for the declared full
                # dataset release decision.  Do not turn bounded operational
                # batches into a misleading failed model-quality result.
                quality_gate = {
                    "status": "deferred",
                    "release_eligible": False,
                    "reason": "pending-immutable-training-shards",
                    "committed_shards": summary["committed_shards"],
                    "required_shards": len(contract.train_shards),
                    "binding": quality_binding,
                }
            else:
                checkpoint_name = summary.get("checkpoint")
                checkpoint_digest = summary.get("checkpoint_sha256")
                if quality_binding["mode"] == "not-configured":
                    quality_gate = {
                        "status": "not-configured",
                        "release_eligible": False,
                        "contract": "No held-out semantic-neighbour release evidence was supplied for this run.",
                    }
                else:
                    require(isinstance(checkpoint_name, str) and checkpoint_name,
                            "continuous state omitted final checkpoint name before quality evaluation")
                    require(isinstance(checkpoint_digest, str) and len(checkpoint_digest) == 64,
                            "continuous state omitted final checkpoint digest before quality evaluation")
                    checkpoint = (state_directory / checkpoint_name).resolve()
                    try:
                        checkpoint.relative_to(state_directory.resolve())
                    except ValueError as error:
                        raise ProductionRunError("continuous state checkpoint escapes state directory") from error
                    require(checkpoint.is_file(), "continuous state final checkpoint is missing before quality evaluation")
                    require(file_sha256(checkpoint) == checkpoint_digest,
                            "continuous state final checkpoint digest changed before quality evaluation")
                    quality_gate = run_quality_gate(
                        binary, binary_sha256, checkpoint, checkpoint_digest,
                        run_directory, arguments, quality_binding,
                    )
            result = {
                "schema": RESULT_SCHEMA,
                "status": "success",
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                # This timer is intentionally stopped before any held-out
                # inspection: it remains the continuous training controller
                # wall-clock interval, not evaluator/report overhead.
                "elapsed_train_controller_seconds": elapsed,
                "leakage_audit": leakage,
                "vocabulary_preflight": vocabulary_preflight,
                "quality_scratch_preflight": quality_scratch,
                "command": command,
                "child_output_sha256": hashlib.sha256(child.stdout.encode("utf-8")).hexdigest(),
                "child_output_tail": child.stdout[-8000:],
                "continuous_state": summary,
                "quality_gate": quality_gate,
            }
            if quality_gate["status"] in {"failed", "execution-failed"}:
                result["status"] = "failed"
                atomic_write_json(run_directory / FAILURE_FILENAME, result)
                atomic_write_json(run_directory / RESULT_FILENAME, result)
                raise ProductionRunError(
                    f"held-out semantic quality gate {quality_gate['status']}; see {run_directory / QUALITY_REPORT_FILENAME}"
                )
            atomic_write_json(run_directory / RESULT_FILENAME, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (ProductionRunError, ContinuousPretrainError, OSError, ValueError, sqlite3.Error) as error:
        print(f"production pretraining: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
