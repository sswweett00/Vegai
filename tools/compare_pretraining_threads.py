#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure deterministic OpenMP context construction with real train processes.

Each timing sample starts immediately before one ``vegai_pretrain train`` child
and stops after it exits. Corpus fixture creation, compilation, checkpoint byte
comparison, JSON serialization, and this harness are excluded from elapsed
samples. The generated workload contains long records so the CPU-only OpenMP
context-construction stage has meaningful independent centers to schedule.

The script does not claim a speedup threshold: CPU topology, memory bandwidth,
record length, OpenMP runtime, thermals, and storage all affect scaling. It does
require byte-identical checkpoints between thread counts, which prevents a
throughput result from hiding a changed learning schedule.
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
TOKEN_COUNT = 128
TOKENS_PER_RECORD = TOKEN_COUNT * 2
RECORDS = 16
EPOCHS = 2
WINDOW = 4


class ComparisonError(RuntimeError):
    """Expected benchmark contract failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def expected_context_pairs() -> int:
    per_record = sum(min(WINDOW, index) + min(WINDOW, TOKENS_PER_RECORD - 1 - index) for index in range(TOKENS_PER_RECORD))
    return per_record * RECORDS * EPOCHS


def metric(output: str, label: str) -> int:
    values = re.findall(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", output, re.MULTILINE)
    require(len(values) == 1, f"expected exactly one {label!r} metric, got {values!r}\n{output}")
    return int(values[0])


def write_fixture(path: Path) -> tuple[int, int]:
    tokens = [f"ctx{index:03d}" for index in range(TOKEN_COUNT)]
    with path.open("w", encoding="ascii", newline="\n") as destination:
        for record in range(RECORDS):
            shift = record % TOKEN_COUNT
            rotated = tokens[shift:] + tokens[:shift]
            destination.write(" ".join((*rotated, *rotated)) + "\n")
    return RECORDS * TOKENS_PER_RECORD * EPOCHS, expected_context_pairs()


def run_train(binary: Path, corpus: Path, checkpoint: Path, threads: int) -> tuple[str, float]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    command = [
        str(binary),
        "train",
        str(corpus),
        str(checkpoint),
        "--max-vocab",
        str(TOKEN_COUNT),
        "--min-frequency",
        "1",
        "--window",
        str(WINDOW),
        "--position-shift",
        "41",
        "--context-mode",
        "hierarchical",
        "--epochs",
        str(EPOCHS),
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
        raise ComparisonError(f"train failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result.stdout, elapsed_seconds


def validate(output: str, training_tokens: int, context_pairs: int) -> None:
    require("phase: vocabulary" in output, "training output omitted vocabulary phase")
    require("phase: streaming HDC context learning" in output, "training output omitted learning phase")
    _, training = output.split("phase: streaming HDC context learning", maxsplit=1)
    require(metric(training, "completed epochs") == EPOCHS, "unexpected completed epoch count")
    require(metric(training, "raw tokens") == training_tokens, "unexpected training token count")
    require(metric(training, "active tokens") == training_tokens, "long fixture produced inactive tokens")
    require(metric(training, "out-of-vocabulary tokens") == 0, "long fixture produced OOV tokens")
    require(metric(training, "ordered context pairs") == context_pairs, "unexpected ordered context-pair count")
    require(metric(training, "prototype updates") == training_tokens, "unexpected prototype updates")


def benchmark(binary: Path, corpus: Path, threads: int, repetitions: int, workdir: Path, training_tokens: int, context_pairs: int) -> tuple[dict[str, Any], Path]:
    samples: list[float] = []
    checkpoint = workdir / f"threads-{threads}-last.vgai"
    for repetition in range(repetitions):
        candidate = workdir / f"threads-{threads}-{repetition}.vgai"
        output, elapsed_seconds = run_train(binary, corpus, candidate, threads)
        validate(output, training_tokens, context_pairs)
        require(candidate.is_file(), f"thread-{threads} run did not create checkpoint")
        samples.append(elapsed_seconds)
        checkpoint = candidate
    median_seconds = statistics.median(samples)
    return {
        "omp_threads": threads,
        "repetitions": repetitions,
        "samples_seconds": [round(sample, 9) for sample in samples],
        "median_seconds": round(median_seconds, 9),
        "training_tokens": training_tokens,
        "training_tokens_per_second": round(training_tokens / median_seconds, 3),
    }, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--parallel-threads", type=int, default=2, help="OMP threads in the comparison arm (default: 2)")
    parser.add_argument("--repetitions", type=int, default=3, help="real train processes per thread arm (default: 3)")
    parser.add_argument("--output", type=Path, help="optional JSON output file; stdout remains JSON")
    arguments = parser.parse_args()
    try:
        binary = arguments.pretrain_bin.resolve()
        require(binary.is_file(), f"pretraining executable does not exist: {binary}")
        require(arguments.parallel_threads > 1, "--parallel-threads must exceed one")
        require(arguments.repetitions > 0, "--repetitions must be positive")
        with tempfile.TemporaryDirectory(prefix="vegai-thread-comparison-") as temporary:
            workdir = Path(temporary)
            corpus = workdir / "long-record-context-fixture.txt"
            training_tokens, context_pairs = write_fixture(corpus)
            serial, serial_checkpoint = benchmark(binary, corpus, 1, arguments.repetitions, workdir, training_tokens, context_pairs)
            parallel, parallel_checkpoint = benchmark(
                binary, corpus, arguments.parallel_threads, arguments.repetitions, workdir, training_tokens, context_pairs
            )
            serial_bytes = serial_checkpoint.read_bytes()
            parallel_bytes = parallel_checkpoint.read_bytes()
            require(serial_bytes == parallel_bytes, "OpenMP thread counts produced different checkpoint bytes")
            report: dict[str, Any] = {
                "schema": "vegai-pretraining-thread-comparison/v1",
                "hardware": collect_hardware_profile(),
                "timing_scope": (
                    "one vegai_pretrain train process per sample; compilation, fixture generation, checkpoint comparison, "
                    "and harness work excluded"
                ),
                "workload": {
                    "records": RECORDS,
                    "tokens_per_record": TOKENS_PER_RECORD,
                    "epochs": EPOCHS,
                    "window": WINDOW,
                    "training_tokens_per_run": training_tokens,
                    "context_pairs_per_run": context_pairs,
                    "context_mode": "hierarchical",
                },
                "single_thread": serial,
                "parallel": parallel,
                "checkpoint_byte_identical": True,
                "single_to_parallel_speedup": round(serial["median_seconds"] / parallel["median_seconds"], 4),
                "single_checkpoint_sha256": hashlib.sha256(serial_bytes).hexdigest(),
                "parallel_checkpoint_sha256": hashlib.sha256(parallel_bytes).hexdigest(),
            }
    except (ComparisonError, OSError, subprocess.SubprocessError) as error:
        print(f"pretraining thread comparison: FAIL: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print("pretraining thread comparison hardware: " + compact_hardware_summary(report["hardware"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
