#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run auditable, exactly-once production HDC pretraining from a built dataset.

This controller composes Vegai's real ``continuous_pretrain.py`` transaction
engine with the content-addressed dataset built by ``production_dataset.py``.
It adds the operational controls commonly required around large-model training:

* immutable dataset/binary identity and resolved-policy provenance;
* an exact disk-backed train/validation record-leakage audit by default;
* RAM/checkpoint-space preflight before child training starts;
* a single-writer run lock plus atomic planned/result/failure reports; and
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
import os
from pathlib import Path
import sqlite3
import shutil
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
from training_resource_plan import build_plan


ROOT = Path(__file__).resolve().parents[1]
RUN_SCHEMA = "vegai-production-pretrain-run/v1"
RESULT_SCHEMA = "vegai-production-pretrain-result/v1"
LOCK_FILENAME = ".production-pretrain.lock"
RUN_FILENAME = "run.json"
RESULT_FILENAME = "result.json"
FAILURE_FILENAME = "failure.json"
STATE_FILENAME = "state.json"
BASIS_POINTS = 10_000


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


def policy_from_arguments(arguments: argparse.Namespace) -> dict[str, Any]:
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
        policy = policy_from_arguments(arguments)
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
            if arguments.preflight_only:
                print(json.dumps({"dataset": str(contract.manifest_path), "leakage_audit": leakage,
                                  "vocabulary_preflight": vocabulary_preflight,
                                  "resource_plan": resource_plan, "ready": True}, ensure_ascii=False, sort_keys=True))
                return 0
            initialize_or_verify_run(
                run_directory, arguments.resume_run, fingerprint, contract, binary, state_directory,
                resource_plan, vocabulary_preflight,
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
            result = {
                "schema": RESULT_SCHEMA,
                "status": "success",
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_train_controller_seconds": elapsed,
                "leakage_audit": leakage,
                "vocabulary_preflight": vocabulary_preflight,
                "command": command,
                "child_output_sha256": hashlib.sha256(child.stdout.encode("utf-8")).hexdigest(),
                "child_output_tail": child.stdout[-8000:],
                "continuous_state": summary,
            }
            atomic_write_json(run_directory / RESULT_FILENAME, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (ProductionRunError, ContinuousPretrainError, OSError, ValueError, sqlite3.Error) as error:
        print(f"production pretraining: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
