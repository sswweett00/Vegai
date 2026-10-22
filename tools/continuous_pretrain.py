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
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

from corpus_manifest import (
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_TOKENS_PER_LINE,
    DEFAULT_TOKEN_LENGTH,
    PreflightStats,
    scan_corpus,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = "vegai-continuous-shard-manifest/v1"
STATE_SCHEMA = "vegai-continuous-state/v1"
STATE_FILENAME = "state.json"
GENERATIONS_DIRECTORY = "generations"
LOCK_FILENAME = ".continuous-pretrain.lock"
# app/pretrain.f90 receives corpus/checkpoint arguments in character(len=512).
MAX_PRETRAIN_CLI_PATH_BYTES = 512
HEX64 = set("0123456789abcdef")
MODES = {"legacy", "positional", "hybrid"}


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


def load_manifest(path: Path) -> list[Shard]:
    manifest = load_json(path, "continuous shard manifest")
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


def default_policy(arguments: argparse.Namespace) -> dict[str, Any]:
    policy = {
        "max_vocabulary": 20_000 if arguments.max_vocab is None else arguments.max_vocab,
        "min_frequency": 1 if arguments.min_frequency is None else arguments.min_frequency,
        "context_window": 4 if arguments.window is None else arguments.window,
        "position_shift_bits": 73 if arguments.position_shift is None else arguments.position_shift,
        "context_mode": "hybrid" if arguments.context_mode is None else arguments.context_mode,
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
    require(policy["context_mode"] in MODES, "context_mode must be legacy, positional, or hybrid")
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


def fresh_state(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "generation": 0,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "policy": policy,
        "applied_shards": [],
    }


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
        entry_path = resolve_inside(state_directory, entry.get("checkpoint"), "applied checkpoint")
        require(entry_path.is_file(), f"applied checkpoint is missing: {entry_path}")
        require(is_sha256(entry.get("checkpoint_sha256")),
                f"state applied shard {expected_generation} has invalid checkpoint SHA-256")
        require(file_sha256(entry_path) == entry["checkpoint_sha256"],
                f"applied checkpoint digest mismatch: {entry_path}")
    if applied:
        final = applied[-1]
        require(state.get("checkpoint") == final["checkpoint"]
                and state.get("checkpoint_sha256") == final["checkpoint_sha256"],
                "state final checkpoint does not match final journal generation")


def load_state(state_directory: Path, arguments: argparse.Namespace) -> dict[str, Any]:
    state_path = state_directory / STATE_FILENAME
    if not state_path.exists():
        return fresh_state(default_policy(arguments))
    state = load_json(state_path, "continuous state")
    require(state.get("schema") == STATE_SCHEMA, f"unsupported continuous state schema: {state.get('schema')!r}")
    require(isinstance(state.get("generation"), int) and state["generation"] >= 0, "state generation is invalid")
    require(isinstance(state.get("applied_shards"), list), "state applied_shards is invalid")
    require(state["generation"] == len(state["applied_shards"]), "state generation does not match applied shard count")
    require(isinstance(state.get("policy"), dict), "state policy is invalid")
    validate_policy(state["policy"])
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


def invoke(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise ContinuousPretrainError(f"vegai_pretrain failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result.stdout


def commit_one_shard(state_directory: Path, binary: Path, state: dict[str, Any], shard: Shard) -> dict[str, Any]:
    policy = state["policy"]
    generation = state["generation"] + 1
    generations = state_directory / GENERATIONS_DIRECTORY
    generations.mkdir(parents=True, exist_ok=True)

    # Scan the publisher's shard first, then take and validate a private byte
    # snapshot.  The child learns only that snapshot, so concurrent mutation of
    # a source path cannot turn a SHA-verified ledger entry into a mixed read.
    verify_shard_bytes(shard)
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
        if state["generation"] == 0:
            command = train_command(binary, training_shard, candidate, policy)
            extending = False
        else:
            shutil.copyfile(checkpoint_from_state(state_directory, state), candidate)
            fsync_file(candidate)
            command = continue_command(binary, training_shard, candidate, policy)
            extending = True
        output = invoke(command)
        metrics = validate_cli_output(output, training_shard, policy, generation, extending)
        verify_shard_bytes(shard)
        require(candidate.is_file(), "pretraining command did not create candidate checkpoint")
        fsync_file(candidate)
        checkpoint_sha256 = file_sha256(candidate)
        final_name = f"generation-{generation:020d}-{checkpoint_sha256[:16]}.vgai"
        final_checkpoint = generations / final_name
        # An unjournaled final file can be left by a crash after its durable
        # rename.  It is not a commit marker, and replacing it makes replay of
        # the same shard idempotent instead of treating that safe orphan as an
        # error.
        os.replace(candidate, final_checkpoint)
        fsync_directory(generations)

        next_state = deepcopy(state)
        next_state["generation"] = generation
        next_state["checkpoint"] = str(final_checkpoint.relative_to(state_directory))
        next_state["checkpoint_sha256"] = checkpoint_sha256
        next_state["applied_shards"].append(
            {
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
            }
        )
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
) -> tuple[dict[str, Any], int]:
    shards = load_manifest(manifest_path)
    validate_manifest_prefix(state, shards)
    committed = 0
    while state["generation"] < len(shards) and (maximum_shards is None or committed < maximum_shards):
        shard = shards[state["generation"]]
        state = commit_one_shard(state_directory, binary, state, shard)
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
    parser.add_argument("--follow", action="store_true", help="poll an atomically updated append-only manifest forever")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="follow polling interval (default: 5)")
    parser.add_argument("--max-shards", type=int, help="commit at most this many currently pending shards")
    parser.add_argument("--recover-lock", action="store_true", help="remove a stale lock only after operator review")

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
        binary = arguments.pretrain_bin.resolve()
        manifest_path = arguments.manifest.resolve()
        state_directory = arguments.state_dir.resolve()
        require(binary.is_file(), f"pretraining executable does not exist: {binary}")
        require(manifest_path.is_file(), f"manifest does not exist: {manifest_path}")
        state_directory.mkdir(parents=True, exist_ok=True)
        (state_directory / GENERATIONS_DIRECTORY).mkdir(parents=True, exist_ok=True)

        with ControllerLock(state_directory / LOCK_FILENAME, arguments.recover_lock):
            cleanup_private_work(state_directory / GENERATIONS_DIRECTORY)
            state = load_state(state_directory, arguments)
            while True:
                state, committed = process_available(
                    state_directory, manifest_path, binary, state, arguments.max_shards
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
