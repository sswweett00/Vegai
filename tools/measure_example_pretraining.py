#!/usr/bin/env python3
"""Time one complete vegai_pretrain train command for the curated fixture.

The timer starts immediately before spawning vegai_pretrain and stops immediately
when that process exits. It deliberately excludes compiler configuration,
compilation, test discovery, and report formatting. The measured command still
includes its required vocabulary scan, streaming HDC learning, and checkpoint
write, because all three occur inside the train subcommand.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "examples" / "corpus" / "mini_turkish_ascii_pretrain.txt"
EXPECTED_RAW_TOKENS_PER_EPOCH = 1_024
EXPECTED_CONTEXT_PAIRS_PER_EPOCH = 3_328


def read_metric(output: str, label: str) -> int:
    matches = re.findall(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", output, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one '{label}' metric, found {len(matches)}")
    return int(matches[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--checkpoint", required=True, type=Path, help="temporary checkpoint output path")
    parser.add_argument("--epochs", type=int, default=2, help="complete fixture epochs to train (default: 2)")
    parser.add_argument("--omp-threads", type=int, default=1, help="OMP_NUM_THREADS for the child command")
    arguments = parser.parse_args()

    if arguments.epochs <= 0:
        parser.error("--epochs must be positive")
    if arguments.omp_threads <= 0:
        parser.error("--omp-threads must be positive")

    binary = arguments.pretrain_bin.resolve()
    checkpoint = arguments.checkpoint.resolve()
    if not binary.is_file():
        parser.error(f"pretraining executable does not exist: {binary}")
    if not CORPUS.is_file():
        parser.error(f"fixture corpus does not exist: {CORPUS}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(binary),
        "train",
        str(CORPUS),
        str(checkpoint),
        "--max-vocab",
        "128",
        "--min-frequency",
        "8",
        "--window",
        "2",
        "--position-shift",
        "41",
        "--epochs",
        str(arguments.epochs),
        "--initial-plasticity",
        "25",
        "--minimum-plasticity",
        "1",
        "--decay-events",
        "8",
        "--strict",
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(arguments.omp_threads)

    started_ns = time.perf_counter_ns()
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False)
    elapsed_ns = time.perf_counter_ns() - started_ns
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        print(f"vegai_pretrain exited with status {result.returncode}", file=sys.stderr)
        return result.returncode

    try:
        _, training_output = result.stdout.split("phase: streaming HDC context learning", maxsplit=1)
        completed_epochs = read_metric(training_output, "completed epochs")
        raw_tokens = read_metric(training_output, "raw tokens")
        active_tokens = read_metric(training_output, "active tokens")
        context_pairs = read_metric(training_output, "ordered context pairs")
        prototype_updates = read_metric(training_output, "prototype updates")
    except (RuntimeError, ValueError) as error:
        print(result.stdout, end="", file=sys.stderr)
        print(f"invalid pretraining output: {error}", file=sys.stderr)
        return 1

    expected_raw_tokens = EXPECTED_RAW_TOKENS_PER_EPOCH * arguments.epochs
    expected_context_pairs = EXPECTED_CONTEXT_PAIRS_PER_EPOCH * arguments.epochs
    if (
        completed_epochs != arguments.epochs
        or raw_tokens != expected_raw_tokens
        or active_tokens != expected_raw_tokens
        or context_pairs != expected_context_pairs
        or prototype_updates != expected_raw_tokens
    ):
        print(result.stdout, end="", file=sys.stderr)
        print("unexpected fixture training counters", file=sys.stderr)
        return 1
    if not checkpoint.is_file():
        print("pretraining command did not create its checkpoint", file=sys.stderr)
        return 1

    elapsed_seconds = elapsed_ns / 1_000_000_000
    report = {
        "measurement": "single vegai_pretrain train process wall clock",
        "compile_or_test_time_included": False,
        "epochs": completed_epochs,
        "training_tokens": raw_tokens,
        "active_training_tokens": active_tokens,
        "context_pairs": context_pairs,
        "prototype_updates": prototype_updates,
        "omp_threads": arguments.omp_threads,
        "elapsed_seconds": round(elapsed_seconds, 9),
        "training_tokens_per_second": round(raw_tokens / elapsed_seconds, 3),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
