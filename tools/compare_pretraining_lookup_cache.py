#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare the exact vocabulary lookup cache with real train child processes.

Every sample starts immediately before one ``vegai_pretrain train`` subprocess
and stops immediately after that child exits. Fixture creation, compilation,
checkpoint hashing/comparison, report generation, and Python harness work are
outside the timed interval. The workload intentionally reuses one fixed-width
active token many times so it measures the bounded lookup-cache hot path rather
than claiming a universal natural-corpus speedup.

The cache-on and cache-off arms must write byte-identical checkpoints. Cache
hits additionally require the complete padded token text as well as its
fingerprint, so this timing comparison cannot accept a faster symbol-merge
shortcut as an optimization.
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
RECORDS = 64
TOKENS_PER_RECORD = 64
EPOCHS = 2
WINDOW = 2
TOKEN = "lookuphot"


class ComparisonError(RuntimeError):
    """Expected benchmark-contract failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def metric(output: str, label: str) -> int:
    values = re.findall(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", output, re.MULTILINE)
    require(len(values) == 1, f"expected exactly one {label!r} metric, got {values!r}\n{output}")
    return int(values[0])


def logical_metric(output: str, label: str) -> bool:
    values = re.findall(rf"^\s*{re.escape(label)}:\s*([TF])\s*$", output, re.MULTILINE)
    require(len(values) == 1, f"expected exactly one {label!r} logical metric, got {values!r}\n{output}")
    return values[0] == "T"


def expected_context_pairs() -> int:
    per_record = sum(
        min(WINDOW, index) + min(WINDOW, TOKENS_PER_RECORD - 1 - index)
        for index in range(TOKENS_PER_RECORD)
    )
    return per_record * RECORDS * EPOCHS


def write_fixture(path: Path) -> tuple[int, int]:
    line = " ".join([TOKEN] * TOKENS_PER_RECORD)
    with path.open("w", encoding="ascii", newline="\n") as destination:
        for _ in range(RECORDS):
            destination.write(line + "\n")
    return RECORDS * TOKENS_PER_RECORD * EPOCHS, expected_context_pairs()


def run_train(binary: Path, corpus: Path, checkpoint: Path, cache_enabled: bool) -> tuple[str, float]:
    command = [
        str(binary),
        "train",
        str(corpus),
        str(checkpoint),
        "--max-vocab",
        "8",
        "--min-frequency",
        "1",
        "--window",
        str(WINDOW),
        "--position-shift",
        "41",
        "--context-mode",
        "hybrid",
        "--epochs",
        str(EPOCHS),
        "--initial-plasticity",
        "25",
        "--minimum-plasticity",
        "1",
        "--decay-events",
        "8",
        "--strict",
        "--serial-context-build",
        "--lookup-cache" if cache_enabled else "--no-lookup-cache",
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["OMP_DYNAMIC"] = "FALSE"
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


def validate(output: str, cache_enabled: bool, training_tokens: int, context_pairs: int) -> tuple[int, int]:
    require("phase: vocabulary" in output, "training output omitted vocabulary phase")
    require("phase: streaming HDC context learning" in output, "training output omitted learning phase")
    _, training = output.split("phase: streaming HDC context learning", maxsplit=1)
    require(metric(training, "completed epochs") == EPOCHS, "unexpected completed epoch count")
    require(metric(training, "raw tokens") == training_tokens, "unexpected training token count")
    require(metric(training, "active tokens") == training_tokens, "lookup fixture produced inactive tokens")
    require(metric(training, "out-of-vocabulary tokens") == 0, "lookup fixture produced OOV tokens")
    require(metric(training, "ordered context pairs") == context_pairs, "unexpected ordered context-pair count")
    require(metric(training, "prototype updates") == training_tokens, "unexpected prototype update count")
    require(metric(training, "CPU serial context batches") == RECORDS * EPOCHS, "fixture unexpectedly used parallel contexts")
    require(metric(training, "CPU parallel context batches") == 0, "serial lookup fixture selected parallel contexts")
    require(logical_metric(training, "vocabulary lookup cache enabled") == cache_enabled,
            "cache-enabled output flag disagrees with comparison arm")
    hits = metric(training, "vocabulary lookup cache hits")
    misses = metric(training, "vocabulary lookup cache misses")
    if cache_enabled:
        require(hits > 0 and misses > 0, "cached arm did not exercise both lookup cache miss and hit paths")
        require(hits + misses == training_tokens, "cached lookup telemetry does not partition raw token lookups")
    else:
        require(hits == 0 and misses == 0, "uncached arm reported cache activity")
    return hits, misses


def append_sample(arm: dict[str, Any], elapsed_seconds: float, hits: int, misses: int, checkpoint: Path) -> None:
    arm["samples_seconds"].append(elapsed_seconds)
    arm["cache_hits"].append(hits)
    arm["cache_misses"].append(misses)
    arm["checkpoints"].append(checkpoint)


def finalize_arm(arm: dict[str, Any], training_tokens: int) -> dict[str, Any]:
    samples = arm["samples_seconds"]
    require(samples, "timing arm had no subprocess samples")
    median_seconds = statistics.median(samples)
    return {
        "cache_enabled": arm["cache_enabled"],
        "repetitions": len(samples),
        "samples_seconds": [round(sample, 9) for sample in samples],
        "median_seconds": round(median_seconds, 9),
        "training_tokens": training_tokens,
        "training_tokens_per_second": round(training_tokens / median_seconds, 3),
        "cache_hits_per_run": arm["cache_hits"],
        "cache_misses_per_run": arm["cache_misses"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--repetitions", type=int, default=3, help="real train processes per cache arm (default: 3)")
    parser.add_argument("--output", type=Path, help="optional JSON output file; stdout remains JSON")
    arguments = parser.parse_args()

    try:
        binary = arguments.pretrain_bin.resolve()
        require(binary.is_file(), f"pretraining executable does not exist: {binary}")
        require(arguments.repetitions > 0, "--repetitions must be positive")
        with tempfile.TemporaryDirectory(prefix="vegai-lookup-cache-comparison-") as temporary:
            workdir = Path(temporary)
            corpus = workdir / "repeated-lookup-fixture.txt"
            training_tokens, context_pairs = write_fixture(corpus)
            uncached: dict[str, Any] = {
                "cache_enabled": False,
                "samples_seconds": [],
                "cache_hits": [],
                "cache_misses": [],
                "checkpoints": [],
            }
            cached: dict[str, Any] = {
                "cache_enabled": True,
                "samples_seconds": [],
                "cache_hits": [],
                "cache_misses": [],
                "checkpoints": [],
            }
            # Alternate arm order to avoid always crediting process/cache warmup
            # to one arm. Each sample remains one isolated real child process.
            for repetition in range(arguments.repetitions):
                arms = ((cached, True), (uncached, False)) if repetition % 2 else ((uncached, False), (cached, True))
                for arm, enabled in arms:
                    checkpoint = workdir / f"{'cache' if enabled else 'no-cache'}-{repetition}.vgai"
                    output, elapsed_seconds = run_train(binary, corpus, checkpoint, enabled)
                    hits, misses = validate(output, enabled, training_tokens, context_pairs)
                    require(checkpoint.is_file(), "train arm did not create its checkpoint")
                    append_sample(arm, elapsed_seconds, hits, misses, checkpoint)

            uncached_bytes = uncached["checkpoints"][-1].read_bytes()
            cached_bytes = cached["checkpoints"][-1].read_bytes()
            require(uncached_bytes == cached_bytes, "cache toggle produced different checkpoint bytes")
            no_cache_report = finalize_arm(uncached, training_tokens)
            cache_report = finalize_arm(cached, training_tokens)
            report: dict[str, Any] = {
                "schema": "vegai-pretraining-lookup-cache-comparison/v1",
                "hardware": collect_hardware_profile(),
                "timing_scope": (
                    "one vegai_pretrain train child-process wall-clock interval per sample; compilation, fixture generation, "
                    "checkpoint comparison, report serialization, and harness work excluded"
                ),
                "workload": {
                    "records": RECORDS,
                    "tokens_per_record": TOKENS_PER_RECORD,
                    "unique_active_tokens": 1,
                    "epochs": EPOCHS,
                    "window": WINDOW,
                    "training_tokens_per_run": training_tokens,
                    "context_pairs_per_run": context_pairs,
                    "context_mode": "hybrid",
                    "scheduler": "serial",
                },
                "without_cache": no_cache_report,
                "with_cache": cache_report,
                "checkpoint_byte_identical": True,
                "without_cache_checkpoint_sha256": hashlib.sha256(uncached_bytes).hexdigest(),
                "with_cache_checkpoint_sha256": hashlib.sha256(cached_bytes).hexdigest(),
                "cache_speedup_vs_no_cache": round(
                    no_cache_report["median_seconds"] / cache_report["median_seconds"], 4
                ),
            }
    except (ComparisonError, OSError, subprocess.SubprocessError) as error:
        print(f"pretraining lookup-cache comparison: FAIL: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print("pretraining lookup-cache comparison hardware: " + compact_hardware_summary(report["hardware"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
