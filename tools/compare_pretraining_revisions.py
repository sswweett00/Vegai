#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure two Vegai pretraining revisions on identical real train processes.

This is a controlled architecture benchmark, not a general corpus-performance
claim.  It deliberately contains (1) many short records and (2) dense repeated
centers, because those are where per-record staging wipes/OpenMP launches and
quadratic source-order planning used to be measurable bottlenecks.  Every sample
starts immediately before one ``vegai_pretrain train`` subprocess and stops when
that subprocess exits.  Fixture creation, both binary builds, checkpoint byte
comparison, report serialization, and harness work are outside timed samples.

By default both supplied binaries must implement the same checkpoint
semantics, and the tool requires byte-identical checkpoint output for each
workload so an apparent speedup cannot hide a source-order or tie-breaking
change. An explicit versioned-algorithm-transition contract is available for a
reviewed checkpoint-lineage migration: both binaries must still be internally
deterministic, their fresh checkpoint bytes must differ, and the candidate must
byte-exactly resume a real baseline checkpoint. The timing ratio is reported
only as a descriptive cross-lineage measurement, not a semantic-equivalence
performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

from hardware_compatibility import collect_hardware_profile, compact_hardware_summary


ROOT = Path(__file__).resolve().parents[1]
SHORT_TOKEN_COUNT = 8
SHORT_RECORDS = 256
SHORT_EPOCHS = 2
SHORT_WINDOW = 2
REPEATED_TOKEN_COUNT = 8
REPEATED_TOKENS_PER_RECORD = 1024
REPEATED_RECORDS = 8
REPEATED_EPOCHS = 2
REPEATED_WINDOW = 1


class RevisionComparisonError(RuntimeError):
    """Raised for an invalid binary, failed train process, or contract violation."""


CHECKPOINT_CONTRACT_IDENTICAL = "byte-identical"
CHECKPOINT_CONTRACT_VERSIONED_TRANSITION = "versioned-algorithm-transition"
_CHECKPOINT_CONTRACTS = frozenset((CHECKPOINT_CONTRACT_IDENTICAL, CHECKPOINT_CONTRACT_VERSIONED_TRANSITION))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RevisionComparisonError(message)


def validate_checkpoint_contract(name: str, before_checkpoint: bytes, after_checkpoint: bytes, contract: str) -> tuple[bool, str]:
    """Validate the explicit relation required between two deterministic runs."""
    require(contract in _CHECKPOINT_CONTRACTS, f"unsupported checkpoint contract: {contract}")
    byte_identical = before_checkpoint == after_checkpoint
    if contract == CHECKPOINT_CONTRACT_IDENTICAL:
        require(byte_identical, f"{name}: before/after checkpoint bytes differ")
        return True, "byte-identical"
    # A caller must consciously opt in to this looser cross-version relation.
    # Requiring a difference catches an accidental missing format/algorithm bump.
    require(not byte_identical, f"{name}: versioned algorithm transition did not change checkpoint bytes")
    return False, "intentional-versioned-algorithm-transition"


def metric(output: str, label: str) -> int:
    values = re.findall(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", output, re.MULTILINE)
    require(len(values) == 1, f"expected exactly one {label!r} metric, got {values!r}\n{output}")
    return int(values[0])


def write_short_records(path: Path) -> tuple[int, int, int, int]:
    tokens = [f"short{index}" for index in range(SHORT_TOKEN_COUNT)]
    with path.open("w", encoding="ascii", newline="\n") as destination:
        for record in range(SHORT_RECORDS):
            shift = record % SHORT_TOKEN_COUNT
            destination.write(" ".join(tokens[shift:] + tokens[:shift]) + "\n")
    training_tokens = SHORT_RECORDS * SHORT_TOKEN_COUNT * SHORT_EPOCHS
    return training_tokens, SHORT_WINDOW, SHORT_EPOCHS, SHORT_TOKEN_COUNT


def write_repeated_centers(path: Path) -> tuple[int, int, int, int]:
    tokens = [f"repeat{index}" for index in range(REPEATED_TOKEN_COUNT)]
    record_tokens = [tokens[index % REPEATED_TOKEN_COUNT] for index in range(REPEATED_TOKENS_PER_RECORD)]
    with path.open("w", encoding="ascii", newline="\n") as destination:
        for record in range(REPEATED_RECORDS):
            shift = record % REPEATED_TOKEN_COUNT
            rotated = record_tokens[shift:] + record_tokens[:shift]
            destination.write(" ".join(rotated) + "\n")
    training_tokens = REPEATED_RECORDS * REPEATED_TOKENS_PER_RECORD * REPEATED_EPOCHS
    return training_tokens, REPEATED_WINDOW, REPEATED_EPOCHS, REPEATED_TOKEN_COUNT


def run_train(
    binary: Path,
    corpus: Path,
    checkpoint: Path,
    token_count: int,
    window: int,
    epochs: int,
) -> tuple[str, float]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "2"
    environment["OMP_DYNAMIC"] = "FALSE"
    command = [
        str(binary),
        "train",
        str(corpus),
        str(checkpoint),
        "--max-vocab",
        str(token_count),
        "--min-frequency",
        "1",
        "--window",
        str(window),
        "--position-shift",
        "41",
        "--context-mode",
        "hierarchical",
        "--epochs",
        str(epochs),
        "--initial-plasticity",
        "25",
        "--minimum-plasticity",
        "1",
        "--decay-events",
        "8",
        "--strict",
        "--parallel-context-build",
    ]
    started_ns = time.perf_counter_ns()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000
    if result.returncode != 0:
        raise RevisionComparisonError(f"train failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result.stdout, elapsed_seconds


def run_resume(binary: Path, corpus: Path, checkpoint: Path) -> str:
    """Resume one epoch in-place outside timing, for legacy-lineage proof."""
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "2"
    environment["OMP_DYNAMIC"] = "FALSE"
    command = [
        str(binary),
        "resume",
        str(corpus),
        str(checkpoint),
        "--epochs",
        "1",
        "--strict",
        "--parallel-context-build",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RevisionComparisonError(f"resume failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result.stdout


def validate_learning_phase(output: str, training_tokens: int) -> None:
    require("phase: streaming HDC context learning" in output, "training output omitted learning phase")
    _, training = output.split("phase: streaming HDC context learning", maxsplit=1)
    require(metric(training, "raw tokens") == training_tokens, "unexpected training-token count")
    require(metric(training, "active tokens") == training_tokens, "fixture produced inactive tokens")
    require(metric(training, "out-of-vocabulary tokens") == 0, "fixture produced OOV tokens")
    require(metric(training, "prototype updates") == training_tokens, "fixture unexpectedly skipped center updates")


def validate(output: str, training_tokens: int) -> None:
    require("phase: vocabulary" in output, "training output omitted vocabulary phase")
    validate_learning_phase(output, training_tokens)


def verify_legacy_resume_compatibility(
    name: str,
    corpus: Path,
    before_binary: Path,
    after_binary: Path,
    initial_checkpoint: bytes,
    workdir: Path,
    one_epoch_tokens: int,
) -> dict[str, object]:
    """Prove that the current binary resumes a historical checkpoint byte-exactly."""
    baseline_checkpoint = workdir / f"{name}-baseline-legacy-resume.vgai"
    candidate_checkpoint = workdir / f"{name}-candidate-legacy-resume.vgai"
    baseline_checkpoint.write_bytes(initial_checkpoint)
    candidate_checkpoint.write_bytes(initial_checkpoint)
    baseline_output = run_resume(before_binary, corpus, baseline_checkpoint)
    candidate_output = run_resume(after_binary, corpus, candidate_checkpoint)
    validate_learning_phase(baseline_output, one_epoch_tokens)
    validate_learning_phase(candidate_output, one_epoch_tokens)
    baseline_resumed = baseline_checkpoint.read_bytes()
    candidate_resumed = candidate_checkpoint.read_bytes()
    require(baseline_resumed == candidate_resumed,
            f"{name}: candidate did not byte-exactly continue the historical checkpoint")
    return {
        "initial_checkpoint_sha256": hashlib.sha256(initial_checkpoint).hexdigest(),
        "resumed_checkpoint_sha256": hashlib.sha256(candidate_resumed).hexdigest(),
        "resumed_checkpoint_byte_identical_to_baseline": True,
        "resume_epochs": 1,
    }


def render_measurement(samples: list[float], checkpoint_bytes: bytes, training_tokens: int) -> dict[str, Any]:
    median_seconds = statistics.median(samples)
    return {
        "repetitions": len(samples),
        "samples_seconds": [round(sample, 9) for sample in samples],
        "median_seconds": round(median_seconds, 9),
        "training_tokens": training_tokens,
        "training_tokens_per_second": round(training_tokens / median_seconds, 3),
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
    }


def compare_workload(
    name: str,
    fixture_writer: Any,
    before_binary: Path,
    after_binary: Path,
    repetitions: int,
    workdir: Path,
    checkpoint_contract: str,
) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    corpus = workdir / f"{name}.txt"
    training_tokens, window, epochs, token_count = fixture_writer(corpus)

    # Prime both executables and corpus pages outside the samples.  Timed pairs
    # then alternate ordering, reducing cache/thermal bias toward the revision
    # that happened to run second.
    for label, binary in (("before", before_binary), ("after", after_binary)):
        output, _ = run_train(binary, corpus, workdir / f"{label}-warmup.vgai", token_count, window, epochs)
        validate(output, training_tokens)

    samples: dict[str, list[float]] = {"before": [], "after": []}
    checkpoints: dict[str, bytes] = {}
    binaries = {"before": before_binary, "after": after_binary}
    for repetition in range(repetitions):
        order = ("before", "after") if repetition % 2 == 0 else ("after", "before")
        for label in order:
            checkpoint = workdir / f"{label}-{repetition}.vgai"
            output, elapsed_seconds = run_train(
                binaries[label], corpus, checkpoint, token_count, window, epochs
            )
            validate(output, training_tokens)
            require(checkpoint.is_file(), f"{label} run did not create a checkpoint")
            checkpoint_bytes = checkpoint.read_bytes()
            if label in checkpoints:
                require(checkpoints[label] == checkpoint_bytes, f"{name}: {label} is not self-deterministic")
            checkpoints[label] = checkpoint_bytes
            samples[label].append(elapsed_seconds)

    before_checkpoint = checkpoints["before"]
    after_checkpoint = checkpoints["after"]
    checkpoint_byte_identical, checkpoint_relation = validate_checkpoint_contract(
        name, before_checkpoint, after_checkpoint, checkpoint_contract
    )
    legacy_resume_compatibility: dict[str, object] | None = None
    if checkpoint_contract == CHECKPOINT_CONTRACT_VERSIONED_TRANSITION:
        legacy_resume_compatibility = verify_legacy_resume_compatibility(
            name,
            corpus,
            before_binary,
            after_binary,
            before_checkpoint,
            workdir,
            training_tokens // epochs,
        )
    before = render_measurement(samples["before"], before_checkpoint, training_tokens)
    after = render_measurement(samples["after"], after_checkpoint, training_tokens)
    return {
        "name": name,
        "workload": {
            "records": SHORT_RECORDS if name == "short_records" else REPEATED_RECORDS,
            "tokens_per_record": SHORT_TOKEN_COUNT if name == "short_records" else REPEATED_TOKENS_PER_RECORD,
            "epochs": epochs,
            "window": window,
            "training_tokens_per_run": training_tokens,
            "context_mode": "hierarchical",
            "purpose": (
                "many short records exercise bounded reset, no staging wipe, hierarchy threshold, and adaptive scheduling"
                if name == "short_records"
                else "repeated centers exercise source-order experience planning without quadratic backward scans"
            ),
        },
        "before": before,
        "after": after,
        "checkpoint_contract": checkpoint_contract,
        "checkpoint_relation": checkpoint_relation,
        "checkpoint_byte_identical": checkpoint_byte_identical,
        "legacy_resume_compatibility": legacy_resume_compatibility,
        "after_runtime_ratio_vs_before": round(before["median_seconds"] / after["median_seconds"], 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-pretrain-bin", required=True, type=Path, help="compiled baseline vegai_pretrain")
    parser.add_argument("--after-pretrain-bin", required=True, type=Path, help="compiled optimized vegai_pretrain")
    parser.add_argument("--repetitions", type=int, default=3, help="real train processes per binary/workload (default: 3)")
    parser.add_argument(
        "--checkpoint-contract",
        choices=sorted(_CHECKPOINT_CONTRACTS),
        default=CHECKPOINT_CONTRACT_IDENTICAL,
        help=(
            "required before/after checkpoint relation (default: byte-identical); "
            "versioned-algorithm-transition is only for an explicitly reviewed lineage migration"
        ),
    )
    parser.add_argument("--output", type=Path, help="optional JSON output file; stdout remains the report")
    arguments = parser.parse_args()
    try:
        before_binary = arguments.before_pretrain_bin.resolve()
        after_binary = arguments.after_pretrain_bin.resolve()
        require(before_binary.is_file(), f"baseline executable does not exist: {before_binary}")
        require(after_binary.is_file(), f"optimized executable does not exist: {after_binary}")
        require(arguments.repetitions > 0, "--repetitions must be positive")
        with tempfile.TemporaryDirectory(prefix="vegai-revision-comparison-") as temporary:
            workdir = Path(temporary)
            workloads = [
                compare_workload(
                    "short_records",
                    write_short_records,
                    before_binary,
                    after_binary,
                    arguments.repetitions,
                    workdir / "short",
                    arguments.checkpoint_contract,
                ),
                compare_workload(
                    "repeated_centers",
                    write_repeated_centers,
                    before_binary,
                    after_binary,
                    arguments.repetitions,
                    workdir / "repeated",
                    arguments.checkpoint_contract,
                ),
            ]
    except (RevisionComparisonError, OSError, subprocess.SubprocessError) as error:
        print(f"pretraining revision comparison: FAIL: {error}", file=sys.stderr)
        return 1

    report: dict[str, Any] = {
        "schema": "vegai-pretraining-revision-comparison/v2",
        "hardware": collect_hardware_profile(),
        "timing_scope": (
            "one vegai_pretrain train process per sample; compilation, fixture generation, checkpoint comparison, "
            "report serialization, and harness work excluded"
        ),
        "omp": {"OMP_NUM_THREADS": 2, "OMP_DYNAMIC": "FALSE"},
        "before_binary": str(before_binary),
        "after_binary": str(after_binary),
        "checkpoint_contract": arguments.checkpoint_contract,
        "workloads": workloads,
        "all_checkpoint_contracts_satisfied": True,
        "all_checkpoints_byte_identical": all(workload["checkpoint_byte_identical"] for workload in workloads),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print("pretraining revision comparison hardware: " + compact_hardware_summary(report["hardware"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
