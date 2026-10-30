#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preflight, run, and attest one bounded synthetic-fixture pretraining command.

The elapsed value in the report is deliberately scoped to one
``vegai_pretrain train`` child-process invocation: it starts immediately before
spawn and stops when that child exits. Corpus generation, streaming manifest
verification, resource planning, binary hashing, checkpoint inspection, report
writing, compilation, test discovery, and CI overhead are outside that metric.

This runner accepts only the provenance-bound synthetic fixture created by
``synthetic_pretraining_fixture.py``. It verifies inputs before and after
training, uses bounded child I/O and timeout handling, and writes a report with
corpus/binary/checkpoint SHA-256 identities. A successful run verifies plumbing
and deterministic HDC training only; it does not establish language, factual,
or general-model quality.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

from synthetic_pretraining_fixture import FixtureError, sha256_file, verify_fixture
from training_resource_plan import ResourcePlanError, build_plan


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "vegai-9mib-pretraining-run/v1"
MAX_TRAIN_STDOUT_BYTES = 1_000_000
MAX_TRAIN_STDERR_BYTES = 1_000_000
MAX_INSPECT_STDOUT_BYTES = 128 * 1024
MAX_INSPECT_STDERR_BYTES = 128 * 1024
METRIC_LABELS = (
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


class PretrainingRunError(RuntimeError):
    """Raised when a pretraining run cannot satisfy its fail-closed contract."""


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    """Captured bounded output and an invocation-only child wall-clock interval."""

    returncode: int
    stdout: bytes
    stdout_sha256: str
    stdout_bytes: int
    stderr_sha256: str
    stderr_bytes: int
    child_wall_seconds: float


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PretrainingRunError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json_write(path: Path, document: Mapping[str, Any]) -> None:
    """Write one report atomically so readers never accept a partial attestation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def file_identity(path: Path) -> dict[str, Any]:
    """Hash a stable regular file without retaining its payload in memory."""

    try:
        before = path.stat()
    except OSError as error:
        raise PretrainingRunError(f"cannot stat required file {path}: {error}") from error
    require(path.is_file(), f"required path is not a regular file: {path}")
    try:
        digest = sha256_file(path)
        after = path.stat()
    except OSError as error:
        raise PretrainingRunError(f"cannot hash required file {path}: {error}") from error
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while its identity was calculated: {path}",
    )
    return {
        "path": str(path),
        "bytes": before.st_size,
        "sha256": digest,
    }


def _terminate_process_group(process: subprocess.Popen[bytes], *, force: bool) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif force:
            process.kill()
        else:
            process.terminate()
    except OSError:
        try:
            if force:
                process.kill()
            else:
                process.terminate()
        except OSError:
            pass


def run_bounded_child(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
    label: str,
) -> BoundedProcessResult:
    """Run a no-shell child with bounded pipes and a child-invocation timer.

    The stopwatch starts immediately before ``Popen`` and stops immediately
    after its process exits. Reader shutdown and output parsing happen later,
    so those controller-only operations cannot inflate the reported train time.
    """

    require(timeout_seconds > 0.0 and math.isfinite(timeout_seconds), f"{label} timeout must be finite and positive")
    require(stdout_limit_bytes > 0 and stderr_limit_bytes > 0, f"{label} output caps must be positive")
    require(command and all(isinstance(item, str) and item for item in command), f"{label} command is invalid")
    started_ns = time.perf_counter_ns()
    try:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            list(command),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
            start_new_session=(os.name == "posix"),
        )
    except (OSError, ValueError) as error:
        raise PretrainingRunError(f"cannot launch {label}: {error}") from error
    require(process.stdout is not None and process.stderr is not None, f"{label} pipes are unavailable")

    stdout_chunks: list[bytes] = []
    stdout_bytes = 0
    stderr_bytes = 0
    stdout_digest = hashlib.sha256()
    stderr_digest = hashlib.sha256()
    stdout_overflow = threading.Event()
    stderr_overflow = threading.Event()
    reader_errors: list[BaseException] = []
    counter_lock = threading.Lock()

    def drain_stdout() -> None:
        nonlocal stdout_bytes
        try:
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    return
                with counter_lock:
                    next_size = stdout_bytes + len(chunk)
                    if next_size <= stdout_limit_bytes:
                        stdout_chunks.append(chunk)
                        stdout_bytes = next_size
                        stdout_digest.update(chunk)
                    else:
                        stdout_overflow.set()
                        _terminate_process_group(process, force=False)
        except (OSError, ValueError) as error:
            reader_errors.append(error)

    def drain_stderr() -> None:
        nonlocal stderr_bytes
        try:
            while True:
                chunk = process.stderr.read(64 * 1024)
                if not chunk:
                    return
                with counter_lock:
                    stderr_bytes += len(chunk)
                    stderr_digest.update(chunk)
                    if stderr_bytes > stderr_limit_bytes:
                        stderr_overflow.set()
                        _terminate_process_group(process, force=False)
        except (OSError, ValueError) as error:
            reader_errors.append(error)

    stdout_reader = threading.Thread(target=drain_stdout, name=f"{label}-stdout", daemon=True)
    stderr_reader = threading.Thread(target=drain_stderr, name=f"{label}-stderr", daemon=True)
    stdout_reader.start()
    stderr_reader.start()
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            timed_out = True
            _terminate_process_group(process, force=False)
            break
        try:
            process.wait(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue
    if process.poll() is None:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process, force=True)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired as error:
                raise PretrainingRunError(f"{label} did not terminate after timeout/cap enforcement") from error
    stopped_ns = time.perf_counter_ns()

    # The process has exited. Drain threads should finish quickly; a leaked
    # descendant pipe is rejected rather than allowing a controller hang.
    for reader, stream in ((stdout_reader, process.stdout), (stderr_reader, process.stderr)):
        reader.join(timeout=5.0)
        if reader.is_alive():
            try:
                stream.close()
            except OSError:
                pass
            reader.join(timeout=1.0)
    if stdout_reader.is_alive() or stderr_reader.is_alive():
        _terminate_process_group(process, force=True)
        raise PretrainingRunError(f"{label} left an output pipe open after process exit")
    if reader_errors:
        raise PretrainingRunError(f"cannot drain {label} output") from reader_errors[0]
    if timed_out:
        raise PretrainingRunError(f"{label} exceeded its {timeout_seconds:g}-second timeout")
    if stdout_overflow.is_set():
        raise PretrainingRunError(f"{label} stdout exceeded its {stdout_limit_bytes}-byte cap")
    if stderr_overflow.is_set():
        raise PretrainingRunError(f"{label} stderr exceeded its {stderr_limit_bytes}-byte cap")
    if process.returncode is None:
        raise PretrainingRunError(f"{label} did not report an exit status")
    return BoundedProcessResult(
        returncode=process.returncode,
        stdout=b"".join(stdout_chunks),
        stdout_sha256=stdout_digest.hexdigest(),
        stdout_bytes=stdout_bytes,
        stderr_sha256=stderr_digest.hexdigest(),
        stderr_bytes=stderr_bytes,
        child_wall_seconds=(stopped_ns - started_ns) / 1_000_000_000,
    )


def parse_training_metrics(stdout: bytes) -> dict[str, int]:
    try:
        text = stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PretrainingRunError("pretraining stdout is not valid UTF-8") from error
    marker = "phase: streaming HDC context learning"
    matches = text.split(marker)
    require(len(matches) == 2, "pretraining output must contain exactly one streaming-learning phase")
    training_output = matches[1]
    metrics: dict[str, int] = {}
    for label in METRIC_LABELS:
        found = re.findall(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", training_output, flags=re.MULTILINE)
        require(len(found) == 1, f"pretraining output must contain exactly one {label!r} metric")
        metrics[label] = int(found[0])
    return metrics


def validate_training_metrics(metrics: Mapping[str, int], fixture: Mapping[str, Any], epochs: int) -> None:
    records = fixture["records"]
    expected_tokens = int(records["raw_tokens"]) * epochs
    expected_lines = int(records["physical_records"]) * epochs
    require(metrics["completed epochs"] == epochs, "pretraining did not complete the requested number of epochs")
    require(metrics["corpus lines"] == expected_lines, "pretraining corpus-line count differs from manifest")
    require(metrics["raw tokens"] == expected_tokens, "pretraining raw-token count differs from manifest")
    require(metrics["active tokens"] == expected_tokens, "fixture training unexpectedly dropped active tokens")
    require(metrics["out-of-vocabulary tokens"] == 0, "fixture training unexpectedly emitted out-of-vocabulary tokens")
    require(metrics["prototype updates"] == expected_tokens, "fixture training must update each active center once")
    require(metrics["ordered context pairs"] > 0, "fixture training produced no ordered context pairs")
    require(metrics["tokenizer truncations"] == 0, "fixture tokenization unexpectedly truncated input")
    require(metrics["non-ASCII input bytes"] == 0, "fixture scanner unexpectedly found non-ASCII bytes")
    require(metrics["dropped overlong records"] == 0, "fixture scanner unexpectedly dropped records")


def preflight_plan(arguments: argparse.Namespace, corpus: Path, checkpoint: Path) -> dict[str, Any]:
    if arguments.skip_resource_preflight:
        return {
            "status": "skipped-by-explicit-operator-option",
            "reason": "--skip-resource-preflight was supplied",
        }
    plan_arguments = argparse.Namespace(
        corpus=corpus,
        checkpoint=checkpoint,
        max_vocab=arguments.max_vocab,
        context_mode=arguments.context_mode,
        omp_threads=arguments.omp_threads,
        reserve_fraction=arguments.reserve_fraction,
        require_accelerator=False,
    )
    try:
        plan = build_plan(plan_arguments)
    except (OSError, ResourcePlanError, ValueError) as error:
        raise PretrainingRunError(f"resource preflight failed: {error}") from error
    verdict = plan.get("verdict")
    require(isinstance(verdict, dict) and verdict.get("ready") is True, "resource preflight is not ready")
    return {"status": "ready", "plan": plan}


def prepare_output_path(path: Path, *, label: str, overwrite: bool) -> Path:
    # Check the operator spelling before resolve(): Path.resolve() follows a
    # final symlink and would otherwise conceal a symlinked output target.
    require(not path.is_symlink(), f"refusing symlinked {label} output: {path}")
    resolved = path.resolve()
    if resolved.exists():
        require(not resolved.is_dir(), f"{label} output is a directory: {resolved}")
        require(overwrite, f"{label} output already exists (pass --overwrite to replace it): {resolved}")
        try:
            resolved.unlink()
        except OSError as error:
            raise PretrainingRunError(f"cannot remove prior {label} output {resolved}: {error}") from error
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def child_metadata(result: BoundedProcessResult) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout_bytes": result.stdout_bytes,
        "stdout_sha256": result.stdout_sha256,
        "stderr_bytes": result.stderr_bytes,
        "stderr_sha256": result.stderr_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--corpus", required=True, type=Path, help="synthetic fixture corpus")
    parser.add_argument("--manifest", required=True, type=Path, help="fixture manifest bound to --corpus")
    parser.add_argument("--checkpoint", required=True, type=Path, help="new checkpoint path")
    parser.add_argument("--report", required=True, type=Path, help="new JSON attestation path")
    parser.add_argument("--max-vocab", type=int, default=1024, help="vocabulary capacity supplied to train (default: 1024)")
    parser.add_argument("--min-frequency", type=int, default=1, help="minimum token frequency supplied to train (default: 1)")
    parser.add_argument("--window", type=int, default=2, help="context window supplied to train (default: 2)")
    parser.add_argument("--position-shift", type=int, default=41, help="position shift supplied to train (default: 41)")
    parser.add_argument(
        "--context-mode",
        choices=("legacy", "positional", "hybrid", "hierarchical", "coherent", "debiased", "resonant"),
        default="hierarchical",
        help="HDC context mode supplied to train (default: hierarchical)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="complete corpus epochs (default: 1)")
    parser.add_argument("--initial-plasticity", type=int, default=25, help="initial Hebbian plasticity percent (default: 25)")
    parser.add_argument("--minimum-plasticity", type=int, default=1, help="minimum Hebbian plasticity percent (default: 1)")
    parser.add_argument("--decay-events", type=int, default=32, help="plasticity decay interval (default: 32)")
    parser.add_argument("--omp-threads", type=int, default=2, help="OMP_NUM_THREADS for the one train child (default: 2)")
    parser.add_argument("--timeout-seconds", type=float, default=1200.0, help="hard train-child timeout (default: 1200)")
    parser.add_argument("--reserve-fraction", type=float, default=0.25, help="resource-plan reserve fraction (default: 0.25)")
    parser.add_argument("--skip-resource-preflight", action="store_true", help="record an explicit preflight bypass (for isolated contract tests only)")
    parser.add_argument("--overwrite", action="store_true", help="replace a prior report/checkpoint at the declared output paths")
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    require(arguments.max_vocab > 0, "--max-vocab must be positive")
    require(arguments.min_frequency > 0, "--min-frequency must be positive")
    require(arguments.window > 0, "--window must be positive")
    require(arguments.position_shift > 0, "--position-shift must be positive")
    require(arguments.epochs > 0, "--epochs must be positive")
    require(1 <= arguments.initial_plasticity <= 100, "--initial-plasticity must be in [1, 100]")
    require(1 <= arguments.minimum_plasticity <= arguments.initial_plasticity, "plasticity bounds are invalid")
    require(arguments.decay_events > 0, "--decay-events must be positive")
    require(arguments.omp_threads > 0, "--omp-threads must be positive")
    require(0.0 <= arguments.reserve_fraction <= 4.0, "--reserve-fraction must be in [0, 4]")
    require(arguments.timeout_seconds > 0.0 and math.isfinite(arguments.timeout_seconds), "--timeout-seconds must be finite and positive")


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    """Execute one fully attested run, raising before writing a success report on violation."""

    validate_arguments(arguments)
    binary = arguments.pretrain_bin.resolve()
    corpus = arguments.corpus.resolve()
    manifest = arguments.manifest.resolve()
    checkpoint_candidate = arguments.checkpoint.resolve()
    report_candidate = arguments.report.resolve()
    # Validate all input/output relationships before an --overwrite path can
    # remove anything. In particular, an operator typo must never delete the
    # corpus or manifest that is about to be attested.
    require(corpus != checkpoint_candidate and corpus != report_candidate, "corpus must not overlap checkpoint/report output")
    require(manifest != checkpoint_candidate and manifest != report_candidate, "manifest must not overlap checkpoint/report output")
    require(binary != checkpoint_candidate and binary != report_candidate, "pretraining binary must not overlap checkpoint/report output")
    require(checkpoint_candidate != report_candidate, "checkpoint and report outputs must differ")
    checkpoint = prepare_output_path(arguments.checkpoint, label="checkpoint", overwrite=arguments.overwrite)
    # Validation/cleanup side effects only: main() owns the final report path.
    prepare_output_path(arguments.report, label="report", overwrite=arguments.overwrite)
    require(binary.is_file(), f"pretraining executable does not exist or is not a regular file: {binary}")
    require(os.access(binary, os.X_OK), f"pretraining executable is not executable: {binary}")

    fixture = verify_fixture(corpus, manifest)
    records = fixture["records"]
    contract = fixture["training_contract"]
    require(arguments.max_vocab >= int(records["unique_tokens"]), "--max-vocab is below the manifest vocabulary cardinality")
    require(
        arguments.min_frequency <= int(records["minimum_token_frequency"]),
        "--min-frequency exceeds the fixture's manifest minimum token frequency",
    )
    probes = contract["probes"]
    require(isinstance(probes, list) and probes and isinstance(probes[0], str), "fixture does not declare a usable probe token")
    probe = probes[0]
    corpus_before = file_identity(corpus)
    manifest_before = file_identity(manifest)
    binary_identity = file_identity(binary)
    resource_preflight = preflight_plan(arguments, corpus, checkpoint)

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(arguments.omp_threads)
    environment["OMP_PROC_BIND"] = "close"
    environment["OMP_PLACES"] = "cores"
    command = [
        str(binary),
        "train",
        str(corpus),
        str(checkpoint),
        "--max-vocab",
        str(arguments.max_vocab),
        "--min-frequency",
        str(arguments.min_frequency),
        "--window",
        str(arguments.window),
        "--position-shift",
        str(arguments.position_shift),
        "--context-mode",
        arguments.context_mode,
        "--epochs",
        str(arguments.epochs),
        "--initial-plasticity",
        str(arguments.initial_plasticity),
        "--minimum-plasticity",
        str(arguments.minimum_plasticity),
        "--decay-events",
        str(arguments.decay_events),
        "--strict",
    ]
    result = run_bounded_child(
        command,
        environment=environment,
        timeout_seconds=arguments.timeout_seconds,
        stdout_limit_bytes=MAX_TRAIN_STDOUT_BYTES,
        stderr_limit_bytes=MAX_TRAIN_STDERR_BYTES,
        label="vegai_pretrain train",
    )
    require(result.returncode == 0, f"vegai_pretrain train exited with status {result.returncode}")
    metrics = parse_training_metrics(result.stdout)
    validate_training_metrics(metrics, fixture, arguments.epochs)
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "train command did not create a nonempty checkpoint")
    require(not checkpoint.is_symlink(), "train command produced a symlinked checkpoint")
    checkpoint_identity = file_identity(checkpoint)
    corpus_after = file_identity(corpus)
    manifest_after = file_identity(manifest)
    binary_after = file_identity(binary)
    require(corpus_before == corpus_after, "corpus changed during the train command")
    require(manifest_before == manifest_after, "manifest changed during the train command")
    require(binary_identity == binary_after, "pretraining binary changed during the train command")

    # Inspection is an integrity/readability check after the measured child;
    # its elapsed time is retained separately and never added to train time.
    inspect_command = [str(binary), "inspect", str(checkpoint), "--query", probe]
    inspect_result = run_bounded_child(
        inspect_command,
        environment=environment,
        timeout_seconds=min(arguments.timeout_seconds, 120.0),
        stdout_limit_bytes=MAX_INSPECT_STDOUT_BYTES,
        stderr_limit_bytes=MAX_INSPECT_STDERR_BYTES,
        label="vegai_pretrain inspect",
    )
    require(inspect_result.returncode == 0, f"vegai_pretrain inspect exited with status {inspect_result.returncode}")
    try:
        inspect_text = inspect_result.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PretrainingRunError("checkpoint inspection output is not valid UTF-8") from error
    require(f"query: {probe}" in inspect_text, "checkpoint inspection did not resolve the manifest probe")
    require("nearest semantic neighbour:" in inspect_text, "checkpoint inspection did not emit a nearest-neighbour result")

    elapsed = result.child_wall_seconds
    require(elapsed > 0.0, "train child wall-clock duration is not positive")
    return {
        "schema": SCHEMA,
        "status": "succeeded",
        "created_at": utc_now(),
        "fixture": {
            "verification": fixture,
            "provenance_limit": (
                "Synthetic structural fixture only; this report is not evidence of factual, linguistic, domain, safety, "
                "or general-model capability."
            ),
        },
        "inputs": {
            "corpus": corpus_before,
            "manifest": manifest_before,
            "pretrain_binary": binary_identity,
        },
        "resource_preflight": resource_preflight,
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
                "fixture generation",
                "fixture and manifest verification",
                "resource preflight",
                "binary hashing",
                "checkpoint inspection",
                "report formatting and write",
                "compilation",
                "test discovery",
                "CI orchestration",
            ],
            "train_process_wall_seconds": round(elapsed, 9),
            "raw_training_tokens_per_second": round(metrics["raw tokens"] / elapsed, 3),
            "active_training_tokens_per_second": round(metrics["active tokens"] / elapsed, 3),
        },
        "checkpoint": checkpoint_identity,
        "inspection": {
            "probe": probe,
            "process": child_metadata(inspect_result),
            "child_wall_seconds_not_in_train_measurement": round(inspect_result.child_wall_seconds, 9),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    report_path = arguments.report.resolve()
    failure_report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "created_at": utc_now(),
        "failure": None,
    }
    try:
        report = run(arguments)
    except (FixtureError, PretrainingRunError, OSError, ValueError) as error:
        failure_report["failure"] = {"type": type(error).__name__, "message": str(error)}
        # A failure report is best effort. Do not hide the originating failure
        # if its destination is unavailable or intentionally refused.
        try:
            if not report_path.exists():
                atomic_json_write(report_path, failure_report)
        except OSError:
            pass
        print(f"9 MiB pretraining run: FAIL: {error}", file=sys.stderr)
        return 2
    atomic_json_write(report_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
