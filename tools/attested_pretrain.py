#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed provenance wrapper for one direct Vegai HDC training run.

``vegai_pretrain`` is deliberately a small native training CLI. This companion
owns the operational contract for one *new* direct ``train`` invocation:

* a reviewed ``vegai-corpus-manifest/v1`` is required and re-verified before
  and after training;
* RAM/storage preflight is required by default;
* the compiled binary, corpus, manifest, configuration, optional OpenVINO
  worker/interpreter identities, child output hashes, checkpoint, and inspection
  result are bound into an atomic JSON attestation;
* training writes to a same-filesystem private checkpoint candidate first, so a
  corpus/binary/manifest integrity failure never publishes that candidate; and
* timing covers exactly one ``vegai_pretrain train`` child process.

The tool is intentionally not a replacement for ``production_pretrain.py`` or
``continuous_pretrain.py``. Those own immutable sharded datasets and
exactly-once generation journals. This wrapper gives a single reviewed corpus
run an equivalent bounded provenance/checkpoint publication boundary without
claiming a held-out semantic-quality result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

from corpus_manifest import DEFAULT_MAX_LINE_BYTES, DEFAULT_MAX_TOKENS_PER_LINE, DEFAULT_TOKEN_LENGTH, scan_corpus
from run_9mib_pretraining import (
    MAX_INSPECT_STDERR_BYTES,
    MAX_INSPECT_STDOUT_BYTES,
    MAX_TRAIN_STDERR_BYTES,
    MAX_TRAIN_STDOUT_BYTES,
    BoundedProcessResult,
    PretrainingRunError,
    atomic_json_write,
    file_identity,
    run_bounded_child,
)
from training_resource_plan import ResourcePlanError, build_plan


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "vegai-attested-pretraining-run/v1"
MANIFEST_SCHEMA = "vegai-corpus-manifest/v1"
_CONTEXT_MODES = ("legacy", "positional", "hybrid", "hierarchical", "coherent", "debiased", "resonant")
_CONTEXT_PROVIDERS = ("cpu", "openvino")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TRAIN_METRICS = (
    "completed epochs",
    "corpus lines",
    "raw tokens",
    "active tokens",
    "out-of-vocabulary tokens",
    "ordered context pairs",
    "prototype updates",
    "tokenizer truncations",
    "non-ASCII input bytes",
    "dropped overlong records",
)


class AttestedPretrainError(RuntimeError):
    """Raised when a run or attestation violates its publication contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttestedPretrainError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def regular_resolved(path: Path, label: str, *, executable: bool = False) -> Path:
    """Resolve an input only after rejecting a final symlink and non-file path."""
    require(not path.is_symlink(), f"refusing symlinked {label}: {path}")
    resolved = path.resolve()
    require(resolved.is_file(), f"{label} does not exist or is not a regular file: {resolved}")
    if executable:
        require(os.access(resolved, os.X_OK), f"{label} is not executable: {resolved}")
    return resolved


def resolve_command_executable(command: str, label: str) -> Path:
    """Resolve a no-argument executable command to the exact launched file."""
    text_field(command, label, nonempty=True)
    located = shutil.which(command)
    require(located is not None, f"{label} is not resolvable on PATH: {command}")
    resolved = Path(located).resolve()
    require(resolved.is_file(), f"{label} is not a regular file: {resolved}")
    require(os.access(resolved, os.X_OK), f"{label} is not executable: {resolved}")
    return resolved


def require_cli_path(path: Path, label: str) -> None:
    """Mirror app/pretrain.f90's fixed character(len=512) path ABI."""
    try:
        rendered = os.fsencode(path)
    except (TypeError, ValueError) as error:
        raise AttestedPretrainError(f"cannot encode {label} path: {path}") from error
    require(len(rendered) <= 512, f"{label} path exceeds vegai_pretrain's 512-byte CLI ABI: {path}")


def integer_field(value: object, label: str, *, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def text_field(value: object, label: str, *, nonempty: bool = False) -> str:
    require(isinstance(value, str), f"{label} must be a string")
    if nonempty:
        require(bool(value.strip()), f"{label} must not be empty")
    return value


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a strict corpus-manifest/v1 and return it with a stable identity."""
    manifest_path = regular_resolved(path, "corpus manifest")
    identity = file_identity(manifest_path)
    try:
        decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttestedPretrainError(f"cannot parse corpus manifest {manifest_path}: {error}") from error
    require(isinstance(decoded, dict), "corpus manifest must be a JSON object")
    require(decoded.get("schema") == MANIFEST_SCHEMA, f"unsupported corpus manifest schema: {decoded.get('schema')!r}")

    input_value = decoded.get("input")
    contract = decoded.get("vegai_ascii_ingest_contract")
    preflight = decoded.get("preflight")
    provenance = decoded.get("declared_provenance")
    require(isinstance(input_value, dict), "corpus manifest input must be an object")
    require(isinstance(contract, dict), "corpus manifest ingest contract must be an object")
    require(isinstance(preflight, dict), "corpus manifest preflight must be an object")
    require(isinstance(provenance, dict), "corpus manifest declared provenance must be an object")

    digest = text_field(input_value.get("sha256"), "manifest input.sha256", nonempty=True).lower()
    require(bool(_HEX64.fullmatch(digest)), "manifest input.sha256 must be lowercase SHA-256 hex")
    integer_field(input_value.get("bytes"), "manifest input.bytes")
    integer_field(input_value.get("physical_lf_records"), "manifest input.physical_lf_records")
    require(
        integer_field(contract.get("max_line_bytes"), "manifest max_line_bytes", minimum=1) == DEFAULT_MAX_LINE_BYTES,
        "manifest max_line_bytes is incompatible with the compiled Vegai ingest contract",
    )
    require(
        integer_field(contract.get("token_length_bytes"), "manifest token_length_bytes", minimum=1) == DEFAULT_TOKEN_LENGTH,
        "manifest token_length_bytes is incompatible with the compiled Vegai ingest contract",
    )
    require(
        integer_field(contract.get("max_tokens_per_line"), "manifest max_tokens_per_line", minimum=1)
        == DEFAULT_MAX_TOKENS_PER_LINE,
        "manifest max_tokens_per_line is incompatible with the compiled Vegai ingest contract",
    )
    for field in ("source", "license", "retrieved_at", "preprocessing"):
        text_field(provenance.get(field), f"manifest declared_provenance.{field}", nonempty=True)
    for field in (
        "overlong_records",
        "records_with_tokenizer_truncation",
        "accepted_tokens",
        "dropped_tokens",
        "maximum_record_bytes",
        "non_ascii_bytes",
        "crlf_records",
    ):
        integer_field(preflight.get(field), f"manifest preflight.{field}")
    return decoded, identity


def corpus_identity(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Stream-scan exactly against the locked manifest contract and detect races."""
    corpus = regular_resolved(path, "corpus")
    contract = manifest["vegai_ascii_ingest_contract"]
    before = corpus.stat()
    try:
        with corpus.open("rb") as handle:
            digest, byte_count, stats = scan_corpus(
                handle,
                int(contract["max_line_bytes"]),
                int(contract["token_length_bytes"]),
                int(contract["max_tokens_per_line"]),
            )
        after = corpus.stat()
    except OSError as error:
        raise AttestedPretrainError(f"cannot stream corpus {corpus}: {error}") from error
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"corpus changed while its identity was calculated: {corpus}",
    )
    return {
        "path": str(corpus),
        "bytes": byte_count,
        "sha256": digest,
        "physical_lf_records": stats.records,
        "preflight": {
            "overlong_records": stats.overlong_records,
            "records_with_tokenizer_truncation": stats.records_with_tokenizer_truncation,
            "accepted_tokens": stats.accepted_tokens,
            "dropped_tokens": stats.dropped_tokens,
            "maximum_record_bytes": stats.maximum_record_bytes,
            "non_ascii_bytes": stats.non_ascii_bytes,
            "crlf_records": stats.crlf_records,
        },
    }


def verify_corpus_against_manifest(corpus: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Bind content and strict no-loss preflight fields, not a path spelling."""
    expected = manifest["input"]
    require(corpus["sha256"] == expected["sha256"], "corpus SHA-256 differs from the reviewed manifest")
    require(corpus["bytes"] == expected["bytes"], "corpus byte count differs from the reviewed manifest")
    require(
        corpus["physical_lf_records"] == expected["physical_lf_records"],
        "corpus physical LF-record count differs from the reviewed manifest",
    )
    require(corpus["preflight"] == manifest["preflight"], "corpus preflight telemetry differs from the reviewed manifest")
    preflight = corpus["preflight"]
    require(preflight["overlong_records"] == 0, "attested training rejects overlong corpus records")
    require(preflight["records_with_tokenizer_truncation"] == 0, "attested training rejects tokenizer-truncated records")
    require(preflight["dropped_tokens"] == 0, "attested training rejects dropped corpus tokens")
    require(preflight["non_ascii_bytes"] == 0, "attested training rejects non-ASCII corpus bytes")
    require(preflight["crlf_records"] == 0, "attested training requires LF-only corpus records")


def parse_metrics(stdout: bytes) -> dict[str, int]:
    try:
        text = stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AttestedPretrainError("train output is not valid UTF-8") from error
    marker = "phase: streaming HDC context learning"
    pieces = text.split(marker)
    require(len(pieces) == 2, "train output must contain exactly one streaming-learning phase")
    metrics: dict[str, int] = {}
    for label in _TRAIN_METRICS:
        values = re.findall(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", pieces[1], flags=re.MULTILINE)
        require(len(values) == 1, f"train output must contain exactly one {label!r} metric")
        metrics[label] = int(values[0])
    return metrics


def validate_metrics(metrics: Mapping[str, int], corpus: Mapping[str, Any], epochs: int) -> None:
    expected_lines = int(corpus["physical_lf_records"]) * epochs
    expected_raw = int(corpus["preflight"]["accepted_tokens"]) * epochs
    require(metrics["completed epochs"] == epochs, "train completed an unexpected epoch count")
    require(metrics["corpus lines"] == expected_lines, "train corpus-line count differs from reviewed corpus identity")
    require(metrics["raw tokens"] == expected_raw, "train raw-token count differs from reviewed corpus preflight")
    require(
        metrics["active tokens"] + metrics["out-of-vocabulary tokens"] == metrics["raw tokens"],
        "train active/OOV token partition is inconsistent",
    )
    require(metrics["tokenizer truncations"] == 0, "strict attested train reported tokenizer truncation")
    require(metrics["non-ASCII input bytes"] == 0, "strict attested train reported non-ASCII bytes")
    require(metrics["dropped overlong records"] == 0, "strict attested train reported dropped overlong records")
    for label, value in metrics.items():
        require(value >= 0, f"train metric {label!r} is negative")


def metric_from_inspection(text: str, label: str) -> str:
    values = re.findall(rf"^\s*{re.escape(label)}:\s*(.*?)\s*$", text, flags=re.MULTILINE)
    require(len(values) == 1, f"checkpoint inspection must contain exactly one {label!r} field")
    return values[0]


def inspect_checkpoint(
    binary: Path,
    checkpoint: Path,
    configuration: Mapping[str, Any],
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[dict[str, Any], BoundedProcessResult]:
    result = run_bounded_child(
        [str(binary), "inspect", str(checkpoint)],
        environment=environment,
        timeout_seconds=min(timeout_seconds, 120.0),
        stdout_limit_bytes=MAX_INSPECT_STDOUT_BYTES,
        stderr_limit_bytes=MAX_INSPECT_STDERR_BYTES,
        label="vegai_pretrain inspect",
    )
    require(result.returncode == 0, f"checkpoint inspect exited with status {result.returncode}")
    try:
        text = result.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AttestedPretrainError("checkpoint inspection output is not valid UTF-8") from error
    require("Vegai HDC pretraining checkpoint" in text, "checkpoint inspect did not emit the expected model header")
    observed = {
        "vocabulary_capacity": int(metric_from_inspection(text, "vocabulary capacity")),
        "min_frequency": int(metric_from_inspection(text, "min frequency")),
        "context_window": int(metric_from_inspection(text, "context window")),
        "position_shift_bits": int(metric_from_inspection(text, "positional shift")),
        "context_mode": metric_from_inspection(text, "context mode"),
        "completed_epochs": int(metric_from_inspection(text, "completed epochs")),
        "plasticity_sampler": metric_from_inspection(text, "plasticity sampler"),
    }
    require(observed["vocabulary_capacity"] == configuration["max_vocabulary"], "checkpoint capacity differs from attested config")
    require(observed["min_frequency"] == configuration["min_frequency"], "checkpoint min frequency differs from attested config")
    require(observed["context_window"] == configuration["context_window"], "checkpoint window differs from attested config")
    require(
        observed["position_shift_bits"] == configuration["position_shift_bits"],
        "checkpoint position shift differs from attested config",
    )
    require(observed["context_mode"] == configuration["context_mode"], "checkpoint context mode differs from attested config")
    require(observed["completed_epochs"] == configuration["epochs"], "checkpoint completed epochs differ from attested config")
    require(observed["plasticity_sampler"] == "floyd-complement", "fresh attested train must use the v7 Floyd/complement lineage")
    return observed, result


def child_metadata(result: BoundedProcessResult) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout_bytes": result.stdout_bytes,
        "stdout_sha256": result.stdout_sha256,
        "stderr_bytes": result.stderr_bytes,
        "stderr_sha256": result.stderr_sha256,
    }


def fsync_regular_file(path: Path) -> None:
    """Best-effort file durability before a staged artifact is published."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    """Best-effort directory durability; unsupported hosts remain usable."""
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class AttestedOutputLocks:
    """Conservative exclusion for each resource in one checkpoint/report pair.

    Locking each final path, rather than only the ordered pair, prevents two
    invocations from racing when they accidentally reuse just a checkpoint or
    just a report path. A stale lock is never removed implicitly: an operator
    must explicitly ask for recovery after confirming that no writer owns it.
    """

    def __init__(self, checkpoint: Path, report: Path, *, recover: bool) -> None:
        self.entries = self.entries_for(checkpoint, report)
        self.recover = recover
        self._identities: dict[Path, tuple[int, int]] = {}

    @staticmethod
    def entries_for(checkpoint: Path, report: Path) -> list[tuple[Path, Path]]:
        entries: list[tuple[Path, Path]] = []
        for resource in {checkpoint, report}:
            encoded = str(resource).encode("utf-8", errors="surrogateescape")
            name = hashlib.sha256(encoded).hexdigest()[:24]
            entries.append((resource.parent / f".vegai-attested-output-{name}.lock", resource))
        # All contenders acquire in one stable order. O_EXCL means they never
        # wait holding the first resource lock, but ordering also keeps manual
        # diagnosis and future lock implementations straightforward.
        entries.sort(key=lambda entry: str(entry[0]))
        return entries

    def existing(self) -> list[Path]:
        return [lock for lock, _ in self.entries if lock.exists() or lock.is_symlink()]

    @staticmethod
    def _remove_stale(lock: Path, *, recover: bool) -> None:
        if not (lock.exists() or lock.is_symlink()):
            return
        if not recover:
            raise AttestedPretrainError(
                f"attested output lock exists: {lock}; inspect the owner or pass --recover-lock after review"
            )
        require(not lock.is_symlink(), f"refusing to recover a symlinked output lock: {lock}")
        require(lock.is_file(), f"refusing to recover a non-file output lock: {lock}")
        try:
            lock.unlink()
            fsync_directory(lock.parent)
        except OSError as error:
            raise AttestedPretrainError(f"cannot recover output lock {lock}: {error}") from error

    def _acquire_one(self, lock: Path, resource: Path) -> None:
        self._remove_stale(lock, recover=self.recover)
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise AttestedPretrainError(
                f"attested output lock was acquired concurrently: {lock}; retry only after the writer finishes"
            ) from error
        except OSError as error:
            raise AttestedPretrainError(f"cannot create attested output lock {lock}: {error}") from error
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(f"pid={os.getpid()}\n")
                handle.write(f"resource={resource}\n")
                handle.flush()
                os.fsync(handle.fileno())
            status = lock.stat()
            self._identities[lock] = (status.st_dev, status.st_ino)
            fsync_directory(lock.parent)
        except BaseException:
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def acquire(self) -> None:
        try:
            for lock, resource in self.entries:
                self._acquire_one(lock, resource)
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        for lock, _ in reversed(self.entries):
            identity = self._identities.pop(lock, None)
            if identity is None:
                continue
            try:
                status = lock.stat()
                if (status.st_dev, status.st_ino) == identity and not lock.is_symlink():
                    lock.unlink()
                    fsync_directory(lock.parent)
            except OSError:
                # A missing/replaced lock must not mask the original train failure.
                pass


def validate_output_path(path: Path, label: str, *, overwrite: bool) -> Path:
    require(not path.is_symlink(), f"refusing symlinked {label} output: {path}")
    resolved = path.resolve()
    if resolved.exists():
        require(resolved.is_file(), f"{label} output is not a regular file: {resolved}")
        require(overwrite, f"{label} output already exists (pass --overwrite to replace it): {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def stage_path(final_path: Path, label: str) -> Path:
    descriptor, rendered = tempfile.mkstemp(prefix=f".{final_path.name}.{label}.", suffix=".partial", dir=final_path.parent)
    os.close(descriptor)
    staged = Path(rendered)
    require(not staged.is_symlink(), f"internal {label} staging path is unexpectedly symlinked")
    return staged


def publish(staged_checkpoint: Path, checkpoint: Path, staged_report: Path, report: Path, *, overwrite: bool) -> None:
    """Publish checkpoint before the success report; report presence means checkpoint exists."""
    if not overwrite:
        require(not checkpoint.exists(), f"checkpoint output appeared during run: {checkpoint}")
        require(not report.exists(), f"report output appeared during run: {report}")
    require(staged_checkpoint.is_file() and not staged_checkpoint.is_symlink(), "staged checkpoint is invalid")
    require(staged_report.is_file() and not staged_report.is_symlink(), "staged report is invalid")
    os.replace(staged_checkpoint, checkpoint)
    fsync_directory(checkpoint.parent)
    os.replace(staged_report, report)
    fsync_directory(report.parent)


def preflight(arguments: argparse.Namespace, corpus: Path, checkpoint: Path) -> dict[str, Any]:
    if arguments.skip_resource_preflight:
        return {
            "status": "skipped-by-explicit-operator-option",
            "reason": "--skip-resource-preflight was supplied",
        }
    namespace = argparse.Namespace(
        corpus=corpus,
        checkpoint=checkpoint,
        max_vocab=arguments.max_vocab,
        context_mode=arguments.context_mode,
        omp_threads=arguments.omp_threads,
        reserve_fraction=arguments.reserve_fraction,
        require_accelerator=False,
    )
    try:
        plan = build_plan(namespace)
    except (ResourcePlanError, OSError, ValueError) as error:
        raise AttestedPretrainError(f"resource preflight failed: {error}") from error
    verdict = plan.get("verdict")
    require(isinstance(verdict, dict) and verdict.get("ready") is True, "resource preflight is not ready")
    return {"status": "ready", "plan": plan}


def configuration_from_arguments(
    arguments: argparse.Namespace,
    *,
    openvino_worker: Path | None = None,
    openvino_python: Path | None = None,
) -> dict[str, Any]:
    if arguments.context_provider == "openvino":
        require(openvino_worker is not None and openvino_python is not None,
                "OpenVINO configuration requires resolved worker and interpreter identities")
    return {
        "max_vocabulary": arguments.max_vocab,
        "min_frequency": arguments.min_frequency,
        "context_window": arguments.window,
        "position_shift_bits": arguments.position_shift,
        "context_mode": arguments.context_mode,
        "epochs": arguments.epochs,
        "initial_plasticity_percent": arguments.initial_plasticity,
        "minimum_plasticity_percent": arguments.minimum_plasticity,
        "plasticity_decay_events": arguments.decay_events,
        "strict_tokenization": True,
        "reference_context_kernel": arguments.reference_context_kernel,
        "parallel_context_build": not arguments.serial_context_build,
        "parallel_min_contexts": arguments.parallel_min_contexts,
        "vocabulary_lookup_cache": not arguments.no_lookup_cache,
        "context_provider": arguments.context_provider,
        "openvino": {
            "worker": str(openvino_worker) if openvino_worker is not None else None,
            "python": str(openvino_python) if openvino_python is not None else None,
            "work_dir": str(arguments.openvino_work_dir.resolve()) if arguments.context_provider == "openvino" else None,
            "device": arguments.openvino_device if arguments.context_provider == "openvino" else None,
            "min_contexts": arguments.openvino_min_contexts if arguments.context_provider == "openvino" else None,
            "timeout_milliseconds": arguments.openvino_timeout_ms if arguments.context_provider == "openvino" else None,
            "policy": arguments.openvino_policy if arguments.context_provider == "openvino" else None,
        },
    }


def build_train_command(binary: Path, corpus: Path, checkpoint: Path, configuration: Mapping[str, Any]) -> list[str]:
    command = [
        str(binary),
        "train",
        str(corpus),
        str(checkpoint),
        "--max-vocab", str(configuration["max_vocabulary"]),
        "--min-frequency", str(configuration["min_frequency"]),
        "--window", str(configuration["context_window"]),
        "--position-shift", str(configuration["position_shift_bits"]),
        "--context-mode", str(configuration["context_mode"]),
        "--epochs", str(configuration["epochs"]),
        "--initial-plasticity", str(configuration["initial_plasticity_percent"]),
        "--minimum-plasticity", str(configuration["minimum_plasticity_percent"]),
        "--decay-events", str(configuration["plasticity_decay_events"]),
        "--strict",
        "--parallel-min-contexts", str(configuration["parallel_min_contexts"]),
        "--context-provider", str(configuration["context_provider"]),
    ]
    command.append("--serial-context-build" if not configuration["parallel_context_build"] else "--parallel-context-build")
    command.append("--reference-context-kernel" if configuration["reference_context_kernel"] else "--lookup-cache")
    if not configuration["vocabulary_lookup_cache"]:
        command.append("--no-lookup-cache")
    if configuration["context_provider"] == "openvino":
        openvino = configuration["openvino"]
        command.extend([
            "--openvino-worker", str(openvino["worker"]),
            "--openvino-python", str(openvino["python"]),
            "--openvino-work-dir", str(openvino["work_dir"]),
            "--openvino-device", str(openvino["device"]),
            "--openvino-min-contexts", str(openvino["min_contexts"]),
            "--openvino-timeout-ms", str(openvino["timeout_milliseconds"]),
        ])
        command.append({
            "fallback-cpu": "--openvino-fallback-cpu",
            "required": "--openvino-required",
            "require-accelerator": "--openvino-require-accelerator",
        }[str(openvino["policy"])])
    return command


def validate_arguments(arguments: argparse.Namespace) -> None:
    require(arguments.max_vocab > 0, "--max-vocab must be positive")
    require(arguments.min_frequency > 0, "--min-frequency must be positive")
    require(arguments.window > 0, "--window must be positive")
    require(arguments.position_shift != 0, "--position-shift must be nonzero")
    require(arguments.epochs > 0, "--epochs must be positive")
    require(1 <= arguments.initial_plasticity <= 100, "--initial-plasticity must be in [1, 100]")
    require(1 <= arguments.minimum_plasticity <= arguments.initial_plasticity, "plasticity bounds are invalid")
    require(arguments.decay_events > 0, "--decay-events must be positive")
    require(1 <= arguments.parallel_min_contexts <= DEFAULT_MAX_TOKENS_PER_LINE,
            "--parallel-min-contexts is outside the supported record bound")
    require(arguments.omp_threads > 0, "--omp-threads must be positive")
    require(math.isfinite(arguments.timeout_seconds) and arguments.timeout_seconds > 0.0,
            "--timeout-seconds must be finite and positive")
    require(0.0 <= arguments.reserve_fraction <= 4.0, "--reserve-fraction must be in [0, 4]")
    if arguments.context_provider == "openvino":
        require(arguments.context_mode in {"positional", "hybrid", "hierarchical"},
                "OpenVINO immutable ABI does not support this context mode")
        require(arguments.openvino_worker.is_file(), "--openvino-worker must name a regular worker file")
        require(not arguments.openvino_worker.is_symlink(), "refusing symlinked --openvino-worker")
        require(arguments.openvino_work_dir.is_dir(), "--openvino-work-dir must be a directory")
        require(arguments.openvino_min_contexts > 0, "--openvino-min-contexts must be positive")
        require(arguments.openvino_timeout_ms > 0, "--openvino-timeout-ms must be positive")


def run_train(arguments: argparse.Namespace) -> dict[str, Any]:
    """Execute a staged direct train with exclusive checkpoint/report ownership."""
    validate_arguments(arguments)
    require(not arguments.checkpoint.is_symlink(), f"refusing symlinked checkpoint output: {arguments.checkpoint}")
    require(not arguments.report.is_symlink(), f"refusing symlinked report output: {arguments.report}")
    requested_checkpoint = arguments.checkpoint.resolve()
    requested_report = arguments.report.resolve()
    require(requested_checkpoint != requested_report, "checkpoint and report outputs must differ")
    require_cli_path(requested_checkpoint, "checkpoint")
    require_cli_path(requested_report, "report")
    try:
        requested_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        requested_report.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AttestedPretrainError(f"cannot create attested output parent directory: {error}") from error
    lock = AttestedOutputLocks(requested_checkpoint, requested_report, recover=arguments.recover_lock)
    lock.acquire()
    try:
        return _run_train_locked(arguments)
    finally:
        lock.release()


def _run_train_locked(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run the complete direct-train transaction while its output pair is locked."""
    binary = regular_resolved(arguments.pretrain_bin, "pretraining executable", executable=True)
    manifest, manifest_before = load_manifest(arguments.manifest)
    corpus = regular_resolved(arguments.corpus, "corpus")
    openvino_worker: Path | None = None
    openvino_python: Path | None = None
    openvino_worker_before: dict[str, Any] | None = None
    openvino_python_before: dict[str, Any] | None = None
    if arguments.context_provider == "openvino":
        openvino_worker = regular_resolved(arguments.openvino_worker, "OpenVINO worker")
        openvino_python = resolve_command_executable(arguments.openvino_python, "OpenVINO Python interpreter")
        require_cli_path(openvino_worker, "OpenVINO worker")
        require_cli_path(openvino_python, "OpenVINO Python interpreter")
        require_cli_path(arguments.openvino_work_dir.resolve(), "OpenVINO work directory")
        openvino_worker_before = file_identity(openvino_worker)
        openvino_python_before = file_identity(openvino_python)
    require_cli_path(binary, "pretraining executable")
    require_cli_path(corpus, "corpus")
    require_cli_path(arguments.checkpoint.resolve(), "checkpoint")
    checkpoint = validate_output_path(arguments.checkpoint, "checkpoint", overwrite=arguments.overwrite)
    report = validate_output_path(arguments.report, "report", overwrite=arguments.overwrite)
    require(checkpoint != report, "checkpoint and report outputs must differ")
    require(corpus != checkpoint and corpus != report, "corpus must not overlap checkpoint/report output")
    require(binary != checkpoint and binary != report, "binary must not overlap checkpoint/report output")
    require(arguments.manifest.resolve() != checkpoint and arguments.manifest.resolve() != report,
            "manifest must not overlap checkpoint/report output")

    corpus_before = corpus_identity(corpus, manifest)
    verify_corpus_against_manifest(corpus_before, manifest)
    binary_before = file_identity(binary)
    configuration = configuration_from_arguments(
        arguments, openvino_worker=openvino_worker, openvino_python=openvino_python
    )
    resource_preflight = preflight(arguments, corpus, checkpoint)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(arguments.omp_threads)
    environment["OMP_PROC_BIND"] = "close"
    environment["OMP_PLACES"] = "cores"

    staged_checkpoint: Path | None = None
    staged_report: Path | None = None
    try:
        staged_checkpoint = stage_path(checkpoint, "checkpoint")
        staged_report = stage_path(report, "report")
        require_cli_path(staged_checkpoint, "staged checkpoint")
        require_cli_path(staged_report, "staged report")
        command = build_train_command(binary, corpus, staged_checkpoint, configuration)
        result = run_bounded_child(
            command,
            environment=environment,
            timeout_seconds=arguments.timeout_seconds,
            stdout_limit_bytes=MAX_TRAIN_STDOUT_BYTES,
            stderr_limit_bytes=MAX_TRAIN_STDERR_BYTES,
            label="vegai_pretrain train",
        )
        require(result.returncode == 0, f"vegai_pretrain train exited with status {result.returncode}")
        metrics = parse_metrics(result.stdout)
        validate_metrics(metrics, corpus_before, arguments.epochs)
        require(staged_checkpoint.is_file() and not staged_checkpoint.is_symlink(), "train did not create a regular staged checkpoint")
        require(staged_checkpoint.stat().st_size > 0, "train produced an empty staged checkpoint")
        fsync_regular_file(staged_checkpoint)
        checkpoint_identity = file_identity(staged_checkpoint)
        checkpoint_identity["path"] = str(checkpoint)

        corpus_after = corpus_identity(corpus, manifest)
        manifest_after = file_identity(regular_resolved(arguments.manifest, "corpus manifest"))
        binary_after = file_identity(regular_resolved(binary, "pretraining executable", executable=True))
        require(corpus_before == corpus_after, "corpus changed during the train command")
        require(manifest_before == manifest_after, "manifest changed during the train command")
        require(binary_before == binary_after, "pretraining binary changed during the train command")
        if openvino_worker is not None and openvino_python is not None:
            require(openvino_worker_before is not None and openvino_python_before is not None,
                    "OpenVINO provenance identities are unavailable")
            require(file_identity(regular_resolved(openvino_worker, "OpenVINO worker")) == openvino_worker_before,
                    "OpenVINO worker changed during the train command")
            require(file_identity(regular_resolved(openvino_python, "OpenVINO Python interpreter", executable=True))
                    == openvino_python_before,
                    "OpenVINO Python interpreter changed during the train command")
        inspection, inspect_result = inspect_checkpoint(
            binary, staged_checkpoint, configuration, environment, arguments.timeout_seconds
        )
        elapsed = result.child_wall_seconds
        require(elapsed > 0.0, "train child duration is not positive")
        reported_elapsed = round(elapsed, 9)
        require(reported_elapsed > 0.0, "rounded train child duration is not positive")
        attested_inputs: dict[str, Any] = {
            "corpus": corpus_before,
            "corpus_manifest": {
                "identity": manifest_before,
                "declared_provenance": manifest["declared_provenance"],
                "ingest_contract": manifest["vegai_ascii_ingest_contract"],
            },
            "pretrain_binary": binary_before,
        }
        if openvino_worker_before is not None and openvino_python_before is not None:
            attested_inputs["openvino_worker"] = openvino_worker_before
            attested_inputs["openvino_python"] = openvino_python_before
        report_document: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "succeeded",
            "created_at": utc_now(),
            "provenance_limit": (
                "This report binds one deterministic HDC/VSA training execution and its data/binary/checkpoint identities. "
                "It is not evidence of language understanding, factual accuracy, general intelligence, safety, or held-out quality."
            ),
            "inputs": attested_inputs,
            "resource_preflight": resource_preflight,
            "configuration": configuration,
            "train": {
                "command": command,
                "environment": {
                    "OMP_NUM_THREADS": environment["OMP_NUM_THREADS"],
                    "OMP_PROC_BIND": environment["OMP_PROC_BIND"],
                    "OMP_PLACES": environment["OMP_PLACES"],
                },
                "metrics": metrics,
                "process": child_metadata(result),
            },
            "measurement": {
                "scope": "one vegai_pretrain train child-process wall-clock interval",
                "start_boundary": "immediately before spawning vegai_pretrain train",
                "stop_boundary": "immediately after vegai_pretrain train exits",
                "excluded": [
                    "corpus and manifest identity scans",
                    "resource preflight",
                    "binary hashing",
                    "checkpoint inspection",
                    "checkpoint/report publication",
                    "report serialization",
                    "compilation",
                    "test discovery",
                    "CI orchestration",
                ],
                "train_process_wall_seconds": reported_elapsed,
                "raw_training_tokens_per_second": round(metrics["raw tokens"] / reported_elapsed, 3),
                "active_training_tokens_per_second": round(metrics["active tokens"] / reported_elapsed, 3),
            },
            "checkpoint": checkpoint_identity,
            "inspection": {
                "fields": inspection,
                "process": child_metadata(inspect_result),
                "child_wall_seconds_not_in_train_measurement": round(inspect_result.child_wall_seconds, 9),
            },
            "integrity": {
                "corpus_unchanged_across_train": True,
                "manifest_unchanged_across_train": True,
                "binary_unchanged_across_train": True,
                "exclusive_output_lock_held_during_run": True,
                "checkpoint_was_staged_before_publication": True,
                "report_presence_means_checkpoint_publish_completed": True,
            },
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        }
        atomic_json_write(staged_report, report_document)
        fsync_regular_file(staged_report)
        publish(staged_checkpoint, checkpoint, staged_report, report, overwrite=arguments.overwrite)
        return report_document
    finally:
        for staged in (staged_checkpoint, staged_report):
            if staged is None:
                continue
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass


def require_mapping(value: object, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_exact_keys(value: Mapping[str, Any], label: str, expected: set[str]) -> None:
    actual = set(value)
    require(actual == expected, f"{label} fields differ from the {SCHEMA} contract: expected {sorted(expected)}, got {sorted(actual)}")


def validate_identity_record(value: Mapping[str, Any], label: str) -> None:
    require_exact_keys(value, label, {"path", "bytes", "sha256"})
    text_field(value.get("path"), f"{label}.path", nonempty=True)
    integer_field(value.get("bytes"), f"{label}.bytes")
    digest = text_field(value.get("sha256"), f"{label}.sha256", nonempty=True).lower()
    require(bool(_HEX64.fullmatch(digest)), f"{label}.sha256 must be lowercase SHA-256 hex")


def validate_corpus_identity_record(value: Mapping[str, Any], label: str) -> None:
    """Validate the richer streamed-corpus identity emitted by ``corpus_identity``."""
    require_exact_keys(value, label, {"path", "bytes", "sha256", "physical_lf_records", "preflight"})
    text_field(value.get("path"), f"{label}.path", nonempty=True)
    integer_field(value.get("bytes"), f"{label}.bytes")
    digest = text_field(value.get("sha256"), f"{label}.sha256", nonempty=True).lower()
    require(bool(_HEX64.fullmatch(digest)), f"{label}.sha256 must be lowercase SHA-256 hex")
    integer_field(value.get("physical_lf_records"), f"{label}.physical_lf_records")
    preflight = require_mapping(value.get("preflight"), f"{label}.preflight")
    require_exact_keys(
        preflight,
        f"{label}.preflight",
        {
            "overlong_records", "records_with_tokenizer_truncation", "accepted_tokens", "dropped_tokens",
            "maximum_record_bytes", "non_ascii_bytes", "crlf_records",
        },
    )
    for field in preflight:
        integer_field(preflight.get(field), f"{label}.preflight.{field}")


def validate_process_metadata(value: Mapping[str, Any], label: str) -> None:
    require_exact_keys(value, label, {"returncode", "stdout_bytes", "stdout_sha256", "stderr_bytes", "stderr_sha256"})
    require(integer_field(value.get("returncode"), f"{label}.returncode") == 0, f"{label}.returncode must be zero")
    integer_field(value.get("stdout_bytes"), f"{label}.stdout_bytes")
    integer_field(value.get("stderr_bytes"), f"{label}.stderr_bytes")
    for field in ("stdout_sha256", "stderr_sha256"):
        digest = text_field(value.get(field), f"{label}.{field}", nonempty=True).lower()
        require(bool(_HEX64.fullmatch(digest)), f"{label}.{field} must be lowercase SHA-256 hex")


def validate_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every v1 configuration field before using it to inspect a model."""
    expected = {
        "max_vocabulary", "min_frequency", "context_window", "position_shift_bits", "context_mode", "epochs",
        "initial_plasticity_percent", "minimum_plasticity_percent", "plasticity_decay_events", "strict_tokenization",
        "reference_context_kernel", "parallel_context_build", "parallel_min_contexts", "vocabulary_lookup_cache",
        "context_provider", "openvino",
    }
    require_exact_keys(value, "attestation configuration", expected)
    configuration = dict(value)
    integer_field(configuration.get("max_vocabulary"), "attestation configuration.max_vocabulary", minimum=1)
    integer_field(configuration.get("min_frequency"), "attestation configuration.min_frequency", minimum=1)
    integer_field(configuration.get("context_window"), "attestation configuration.context_window", minimum=1)
    position_shift = integer_field(configuration.get("position_shift_bits"), "attestation configuration.position_shift_bits")
    require(position_shift != 0, "attestation configuration.position_shift_bits must be nonzero")
    integer_field(configuration.get("epochs"), "attestation configuration.epochs", minimum=1)
    initial = integer_field(
        configuration.get("initial_plasticity_percent"), "attestation configuration.initial_plasticity_percent", minimum=1
    )
    minimum = integer_field(
        configuration.get("minimum_plasticity_percent"), "attestation configuration.minimum_plasticity_percent", minimum=1
    )
    require(initial <= 100 and minimum <= initial, "attestation configuration plasticity bounds are invalid")
    integer_field(configuration.get("plasticity_decay_events"), "attestation configuration.plasticity_decay_events", minimum=1)
    require(configuration.get("strict_tokenization") is True, "attested train must bind native strict tokenization")
    for field in ("reference_context_kernel", "parallel_context_build", "vocabulary_lookup_cache"):
        require(isinstance(configuration.get(field), bool), f"attestation configuration.{field} must be boolean")
    parallel_minimum = integer_field(
        configuration.get("parallel_min_contexts"), "attestation configuration.parallel_min_contexts", minimum=1
    )
    require(parallel_minimum <= DEFAULT_MAX_TOKENS_PER_LINE, "attestation parallel context threshold exceeds record bound")
    context_mode = text_field(configuration.get("context_mode"), "attestation configuration.context_mode", nonempty=True)
    require(context_mode in _CONTEXT_MODES, "attestation context mode is invalid")
    provider = text_field(configuration.get("context_provider"), "attestation configuration.context_provider", nonempty=True)
    require(provider in _CONTEXT_PROVIDERS, "attestation context provider is invalid")
    openvino = require_mapping(configuration.get("openvino"), "attestation configuration.openvino")
    require_exact_keys(
        openvino,
        "attestation configuration.openvino",
        {"worker", "python", "work_dir", "device", "min_contexts", "timeout_milliseconds", "policy"},
    )
    if provider == "cpu":
        require(all(item is None for item in openvino.values()), "CPU attestation must not carry OpenVINO settings")
    else:
        require(context_mode in {"positional", "hybrid", "hierarchical"}, "OpenVINO attestation uses an unsupported context mode")
        for field in ("worker", "python", "work_dir", "device"):
            text_field(openvino.get(field), f"attestation OpenVINO {field}", nonempty=True)
        integer_field(openvino.get("min_contexts"), "attestation OpenVINO min_contexts", minimum=1)
        integer_field(openvino.get("timeout_milliseconds"), "attestation OpenVINO timeout_milliseconds", minimum=1)
        require(
            text_field(openvino.get("policy"), "attestation OpenVINO policy", nonempty=True)
            in {"fallback-cpu", "required", "require-accelerator"},
            "attestation OpenVINO policy is invalid",
        )
    return configuration


def validate_measurement(value: Mapping[str, Any], metrics: Mapping[str, int]) -> None:
    expected = {
        "scope", "start_boundary", "stop_boundary", "excluded", "train_process_wall_seconds",
        "raw_training_tokens_per_second", "active_training_tokens_per_second",
    }
    require_exact_keys(value, "attestation measurement", expected)
    require(value.get("scope") == "one vegai_pretrain train child-process wall-clock interval",
            "attestation timing scope is not the train-only contract")
    require(value.get("start_boundary") == "immediately before spawning vegai_pretrain train",
            "attestation timing start boundary is invalid")
    require(value.get("stop_boundary") == "immediately after vegai_pretrain train exits",
            "attestation timing stop boundary is invalid")
    excluded = value.get("excluded")
    require(isinstance(excluded, list) and all(isinstance(item, str) for item in excluded),
            "attestation measurement exclusions must be a string list")
    for required_exclusion in ("compilation", "checkpoint inspection", "CI orchestration"):
        require(required_exclusion in excluded, f"attestation measurement omits required exclusion: {required_exclusion}")
    duration = value.get("train_process_wall_seconds")
    require(isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(float(duration)) and float(duration) > 0.0,
            "attestation train duration is invalid")
    for field, metric in (
        ("raw_training_tokens_per_second", metrics["raw tokens"]),
        ("active_training_tokens_per_second", metrics["active tokens"]),
    ):
        observed = value.get(field)
        require(isinstance(observed, (int, float)) and not isinstance(observed, bool) and math.isfinite(float(observed)),
                f"attestation {field} is invalid")
        expected_rate = round(metric / float(duration), 3)
        require(float(observed) == expected_rate, f"attestation {field} does not match bound metrics/duration")


def validate_staged_train_command(
    value: object,
    binary: Path,
    corpus: Path,
    checkpoint: Path,
    configuration: Mapping[str, Any],
) -> None:
    require(isinstance(value, list) and all(isinstance(item, str) and item for item in value),
            "attestation train command must be a nonempty string list")
    command = list(value)
    require(len(command) >= 4, "attestation train command is too short")
    staged_checkpoint = Path(command[3])
    require(command[0] == str(binary) and command[1] == "train" and command[2] == str(corpus),
            "attestation train command does not bind the current binary/corpus")
    require(staged_checkpoint != checkpoint and staged_checkpoint.parent == checkpoint.parent,
            "attestation train command did not use a same-directory private checkpoint")
    require(
        staged_checkpoint.name.startswith(f".{checkpoint.name}.checkpoint.") and staged_checkpoint.name.endswith(".partial"),
        "attestation train command staged-checkpoint name is invalid",
    )
    require(command == build_train_command(binary, corpus, staged_checkpoint, configuration),
            "attestation train command differs from its bound configuration")


def verify_report(arguments: argparse.Namespace) -> dict[str, Any]:
    """Re-validate every durable v1 claim that can be checked without retraining."""
    report_path = regular_resolved(arguments.report, "attestation report")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttestedPretrainError(f"cannot parse attestation report {report_path}: {error}") from error

    root = require_mapping(report, "attestation report")
    require_exact_keys(
        root,
        "attestation report",
        {
            "schema", "status", "created_at", "provenance_limit", "inputs", "resource_preflight", "configuration",
            "train", "measurement", "checkpoint", "inspection", "integrity", "runtime",
        },
    )
    require(root.get("schema") == SCHEMA, f"unsupported attestation schema: {root.get('schema')!r}")
    require(root.get("status") == "succeeded", "attestation report is not a successful run")
    text_field(root.get("created_at"), "attestation created_at", nonempty=True)
    text_field(root.get("provenance_limit"), "attestation provenance_limit", nonempty=True)

    configuration = validate_configuration(require_mapping(root.get("configuration"), "attestation configuration"))
    inputs = require_mapping(root.get("inputs"), "attestation inputs")
    expected_input_fields = {"corpus", "corpus_manifest", "pretrain_binary"}
    if configuration["context_provider"] == "openvino":
        expected_input_fields.update({"openvino_worker", "openvino_python"})
    require_exact_keys(inputs, "attestation inputs", expected_input_fields)
    corpus_record = require_mapping(inputs.get("corpus"), "attestation corpus")
    manifest_record = require_mapping(inputs.get("corpus_manifest"), "attestation corpus manifest")
    binary_record = require_mapping(inputs.get("pretrain_binary"), "attestation binary")
    openvino_worker_record = None
    openvino_python_record = None
    if configuration["context_provider"] == "openvino":
        openvino_worker_record = require_mapping(inputs.get("openvino_worker"), "attestation OpenVINO worker")
        openvino_python_record = require_mapping(inputs.get("openvino_python"), "attestation OpenVINO Python interpreter")
        validate_identity_record(openvino_worker_record, "attestation OpenVINO worker")
        validate_identity_record(openvino_python_record, "attestation OpenVINO Python interpreter")
    validate_corpus_identity_record(corpus_record, "attestation corpus")
    validate_identity_record(binary_record, "attestation binary")
    require_exact_keys(manifest_record, "attestation corpus manifest", {"identity", "declared_provenance", "ingest_contract"})
    manifest_identity_record = require_mapping(manifest_record.get("identity"), "attestation manifest identity")
    validate_identity_record(manifest_identity_record, "attestation manifest identity")

    checkpoint_record = require_mapping(root.get("checkpoint"), "attestation checkpoint")
    validate_identity_record(checkpoint_record, "attestation checkpoint")

    train = require_mapping(root.get("train"), "attestation train")
    require_exact_keys(train, "attestation train", {"command", "environment", "metrics", "process"})
    train_environment = require_mapping(train.get("environment"), "attestation train environment")
    require_exact_keys(train_environment, "attestation train environment", {"OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES"})
    omp_threads = text_field(train_environment.get("OMP_NUM_THREADS"), "attestation OMP_NUM_THREADS", nonempty=True)
    require(omp_threads.isascii() and omp_threads.isdecimal() and int(omp_threads) > 0,
            "attestation OMP_NUM_THREADS must be a positive decimal integer")
    require(train_environment.get("OMP_PROC_BIND") == "close", "attestation OMP_PROC_BIND differs from run contract")
    require(train_environment.get("OMP_PLACES") == "cores", "attestation OMP_PLACES differs from run contract")
    metrics_record = require_mapping(train.get("metrics"), "attestation train metrics")
    require_exact_keys(metrics_record, "attestation train metrics", set(_TRAIN_METRICS))
    metrics = {label: integer_field(metrics_record.get(label), f"attestation train metric {label!r}") for label in _TRAIN_METRICS}
    validate_process_metadata(require_mapping(train.get("process"), "attestation train process"), "attestation train process")

    measurement = require_mapping(root.get("measurement"), "attestation measurement")
    validate_measurement(measurement, metrics)

    inspection_record = require_mapping(root.get("inspection"), "attestation inspection")
    require_exact_keys(inspection_record, "attestation inspection", {"fields", "process", "child_wall_seconds_not_in_train_measurement"})
    stored_inspection = require_mapping(inspection_record.get("fields"), "attestation inspection fields")
    inspection_fields = {
        "vocabulary_capacity", "min_frequency", "context_window", "position_shift_bits", "context_mode",
        "completed_epochs", "plasticity_sampler",
    }
    require_exact_keys(stored_inspection, "attestation inspection fields", inspection_fields)
    for field in ("vocabulary_capacity", "min_frequency", "context_window", "position_shift_bits", "completed_epochs"):
        integer_field(stored_inspection.get(field), f"attestation inspection {field}", minimum=1)
    text_field(stored_inspection.get("context_mode"), "attestation inspection context_mode", nonempty=True)
    text_field(stored_inspection.get("plasticity_sampler"), "attestation inspection plasticity_sampler", nonempty=True)
    validate_process_metadata(require_mapping(inspection_record.get("process"), "attestation inspection process"),
                              "attestation inspection process")
    inspect_seconds = inspection_record.get("child_wall_seconds_not_in_train_measurement")
    require(isinstance(inspect_seconds, (int, float)) and not isinstance(inspect_seconds, bool) and
            math.isfinite(float(inspect_seconds)) and float(inspect_seconds) > 0.0,
            "attestation inspection duration is invalid")

    integrity = require_mapping(root.get("integrity"), "attestation integrity")
    integrity_fields = {
        "corpus_unchanged_across_train", "manifest_unchanged_across_train", "binary_unchanged_across_train",
        "exclusive_output_lock_held_during_run", "checkpoint_was_staged_before_publication",
        "report_presence_means_checkpoint_publish_completed",
    }
    require_exact_keys(integrity, "attestation integrity", integrity_fields)
    require(all(integrity.get(field) is True for field in integrity_fields),
            "attestation integrity flags must all be true")

    resource_preflight = require_mapping(root.get("resource_preflight"), "attestation resource preflight")
    resource_status = resource_preflight.get("status")
    if resource_status == "ready":
        require_exact_keys(resource_preflight, "ready resource preflight", {"status", "plan"})
        plan = require_mapping(resource_preflight.get("plan"), "attestation resource plan")
        require(plan.get("schema") == "vegai-training-resource-plan/v1", "attestation resource plan schema is invalid")
    elif resource_status == "skipped-by-explicit-operator-option":
        require_exact_keys(resource_preflight, "skipped resource preflight", {"status", "reason"})
        text_field(resource_preflight.get("reason"), "attestation resource-preflight bypass reason", nonempty=True)
    else:
        raise AttestedPretrainError("attestation resource preflight status is invalid")

    runtime = require_mapping(root.get("runtime"), "attestation runtime")
    require_exact_keys(runtime, "attestation runtime", {"python", "platform"})
    text_field(runtime.get("python"), "attestation runtime.python", nonempty=True)
    text_field(runtime.get("platform"), "attestation runtime.platform", nonempty=True)

    corpus_path = regular_resolved(
        Path(text_field(corpus_record.get("path"), "attestation corpus path", nonempty=True)), "corpus"
    )
    manifest_path = regular_resolved(
        Path(text_field(manifest_identity_record.get("path"), "attestation manifest path", nonempty=True)),
        "corpus manifest",
    )
    binary_path = regular_resolved(
        Path(text_field(binary_record.get("path"), "attestation binary path", nonempty=True)),
        "pretraining executable", executable=True,
    )
    openvino_worker_path: Path | None = None
    openvino_python_path: Path | None = None
    if openvino_worker_record is not None and openvino_python_record is not None:
        openvino_worker_path = regular_resolved(
            Path(text_field(openvino_worker_record.get("path"), "attestation OpenVINO worker path", nonempty=True)),
            "OpenVINO worker",
        )
        openvino_python_path = regular_resolved(
            Path(text_field(openvino_python_record.get("path"), "attestation OpenVINO Python path", nonempty=True)),
            "OpenVINO Python interpreter", executable=True,
        )
        openvino = configuration["openvino"]
        require(str(openvino_worker_path) == openvino["worker"],
                "attested OpenVINO worker identity does not match configuration")
        require(str(openvino_python_path) == openvino["python"],
                "attested OpenVINO Python identity does not match configuration")
    checkpoint_path = regular_resolved(
        Path(text_field(checkpoint_record.get("path"), "attestation checkpoint path", nonempty=True)),
        "checkpoint",
    )
    output_locks = AttestedOutputLocks(checkpoint_path, report_path, recover=arguments.recover_lock)
    active_locks = output_locks.existing()
    if active_locks and not arguments.recover_lock:
        raise AttestedPretrainError(
            f"attested output lock is present; verification refuses a potentially in-progress publication: {active_locks[0]}"
        )
    if active_locks:
        # Acquire/release after explicit recovery so an immediately competing
        # writer is still detected by O_EXCL rather than silently ignored.
        output_locks.acquire()
        output_locks.release()
        active_locks = output_locks.existing()
        if active_locks:
            raise AttestedPretrainError(f"could not clear attested output lock: {active_locks[0]}")
    validate_staged_train_command(train.get("command"), binary_path, corpus_path, checkpoint_path, configuration)

    manifest, manifest_identity = load_manifest(manifest_path)
    current_corpus = corpus_identity(corpus_path, manifest)
    verify_corpus_against_manifest(current_corpus, manifest)
    current_binary = file_identity(binary_path)
    current_checkpoint = file_identity(checkpoint_path)
    require(current_corpus == corpus_record, "current corpus identity differs from attestation")
    require(current_binary == binary_record, "current binary identity differs from attestation")
    require(current_checkpoint == checkpoint_record, "current checkpoint identity differs from attestation")
    if openvino_worker_path is not None and openvino_python_path is not None:
        require(openvino_worker_record is not None and openvino_python_record is not None,
                "OpenVINO attestation records are unavailable")
        require(file_identity(openvino_worker_path) == openvino_worker_record,
                "current OpenVINO worker identity differs from attestation")
        require(file_identity(openvino_python_path) == openvino_python_record,
                "current OpenVINO Python interpreter identity differs from attestation")
    require(manifest_identity == manifest_identity_record, "current manifest identity differs from attestation")
    require(manifest_record.get("declared_provenance") == manifest["declared_provenance"],
            "attested manifest provenance differs from current reviewed manifest")
    require(manifest_record.get("ingest_contract") == manifest["vegai_ascii_ingest_contract"],
            "attested manifest ingest contract differs from current reviewed manifest")
    validate_metrics(metrics, current_corpus, int(configuration["epochs"]))

    environment = {
        "OMP_NUM_THREADS": "1",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    observed, inspect_result = inspect_checkpoint(binary_path, checkpoint_path, configuration, environment, 120.0)
    require(observed == stored_inspection, "current checkpoint inspection differs from attested inspection fields")
    return {
        "schema": "vegai-attested-pretraining-verification/v1",
        "status": "verified",
        "verified_at": utc_now(),
        "report": str(report_path),
        "checkpoint_sha256": current_checkpoint["sha256"],
        "corpus_sha256": current_corpus["sha256"],
        "inspection": observed,
        "inspect_process": child_metadata(inspect_result),
    }

def add_common_train_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--corpus", required=True, type=Path, help="strict LF/ASCII corpus bound to --manifest")
    parser.add_argument("--manifest", required=True, type=Path, help="reviewed vegai-corpus-manifest/v1 for --corpus")
    parser.add_argument("--checkpoint", required=True, type=Path, help="published checkpoint output path")
    parser.add_argument("--report", required=True, type=Path, help="published JSON attestation output path")
    parser.add_argument("--max-vocab", type=int, default=20_000, help="fixed vocabulary capacity (default: 20000)")
    parser.add_argument("--min-frequency", type=int, default=2, help="active vocabulary threshold (default: 2)")
    parser.add_argument("--window", type=int, default=4, help="context window (default: 4)")
    parser.add_argument("--position-shift", type=int, default=73, help="relative position shift (default: 73)")
    parser.add_argument("--context-mode", choices=_CONTEXT_MODES, default="hierarchical")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs (default: 1)")
    parser.add_argument("--initial-plasticity", type=int, default=25)
    parser.add_argument("--minimum-plasticity", type=int, default=1)
    parser.add_argument("--decay-events", type=int, default=32)
    parser.add_argument("--reference-context-kernel", action="store_true", help="attest the exact regression kernel path")
    scheduler = parser.add_mutually_exclusive_group()
    scheduler.add_argument("--parallel-context-build", dest="serial_context_build", action="store_false", help="use adaptive OpenMP context build (default)")
    scheduler.add_argument("--serial-context-build", dest="serial_context_build", action="store_true", help="force serial context build")
    parser.set_defaults(serial_context_build=False)
    parser.add_argument("--parallel-min-contexts", type=int, default=32)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--lookup-cache", dest="no_lookup_cache", action="store_false", help="enable exact-token lookup cache (default)")
    cache.add_argument("--no-lookup-cache", dest="no_lookup_cache", action="store_true", help="disable exact-token lookup cache")
    parser.set_defaults(no_lookup_cache=False)
    parser.add_argument("--omp-threads", type=int, default=1, help="OMP_NUM_THREADS for the timed train child (default: 1)")
    parser.add_argument("--timeout-seconds", type=float, default=1200.0, help="timed train timeout (default: 1200)")
    parser.add_argument("--reserve-fraction", type=float, default=0.25, help="resource-plan RAM reserve fraction (default: 0.25)")
    parser.add_argument("--skip-resource-preflight", action="store_true", help="explicitly record a resource-preflight bypass; intended only for isolated tests")
    parser.add_argument("--overwrite", action="store_true", help="replace existing checkpoint/report only after a successful staged run")
    parser.add_argument(
        "--recover-lock",
        action="store_true",
        help="remove a stale same-output lock only after operator review; never use while another writer may run",
    )
    parser.add_argument("--context-provider", choices=_CONTEXT_PROVIDERS, default="cpu")
    parser.add_argument("--openvino-worker", type=Path, default=ROOT / "tools" / "openvino_context_worker.py")
    parser.add_argument("--openvino-python", default="python3")
    parser.add_argument("--openvino-work-dir", type=Path, default=ROOT)
    parser.add_argument("--openvino-device", default="auto")
    parser.add_argument("--openvino-min-contexts", type=int, default=64)
    parser.add_argument("--openvino-timeout-ms", type=int, default=120_000)
    parser.add_argument("--openvino-policy", choices=("fallback-cpu", "required", "require-accelerator"), default="fallback-cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="run one staged, manifest-bound direct train")
    add_common_train_options(train)
    verify = commands.add_parser("verify", help="re-verify a published attestation without retraining")
    verify.add_argument("--report", required=True, type=Path, help="successful JSON attestation to verify")
    verify.add_argument(
        "--recover-lock",
        action="store_true",
        help="remove stale output locks only after operator review, then verify; never use while a writer may run",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        if arguments.command == "train":
            report = run_train(arguments)
        elif arguments.command == "verify":
            report = verify_report(arguments)
        else:
            raise AttestedPretrainError(f"unsupported command: {arguments.command!r}")
    except (AttestedPretrainError, PretrainingRunError, ResourcePlanError, OSError, ValueError) as error:
        print(f"attested pretraining: FAIL: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
