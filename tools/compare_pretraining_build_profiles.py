#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure portable versus host-native Vegai pretraining binaries.

Each retained timing sample covers exactly one real ``vegai_pretrain train``
process. Compilation, profile selection, hardware discovery, fixture setup,
checkpoint byte comparison, and JSON rendering are outside the measured window.
The pair is accepted only when portable and native binaries emit byte-identical
checkpoints for the same deterministic corpus/configuration.

This benchmark compares build profiles on one host; it is not a claim that a
native binary is portable, that all CPUs have equal performance, or that any
GPU/iGPU is used for compute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any

from compare_pretraining_modes import (
    BASE_CONTEXT_PAIRS,
    BASE_EPOCHS,
    BASE_TRAINING_TOKENS,
    CORPUS,
    ROOT,
    run,
    train_command,
    validate_train_output,
)
from hardware_compatibility import collect_hardware_profile, compact_hardware_summary


SCHEMA = "vegai-pretraining-build-profile-comparison/v1"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_profile(
    binary: Path,
    label: str,
    repetitions: int,
    workdir: Path,
    environment: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    samples: list[float] = []
    checkpoint = workdir / f"{label}-last.vgai"
    for repetition in range(repetitions):
        candidate = workdir / f"{label}-{repetition}.vgai"
        output, elapsed_seconds = run(
            train_command(binary, CORPUS, candidate, "hybrid", BASE_EPOCHS),
            environment,
        )
        validate_train_output(
            output,
            epochs=BASE_EPOCHS,
            training_tokens=BASE_TRAINING_TOKENS,
            context_pairs=BASE_CONTEXT_PAIRS,
        )
        if not candidate.is_file():
            fail(f"{label} run did not write a checkpoint")
        samples.append(elapsed_seconds)
        checkpoint = candidate

    median_seconds = statistics.median(samples)
    return (
        {
            "binary": str(binary),
            "binary_sha256": sha256_file(binary),
            "repetitions": repetitions,
            "samples_seconds": [round(value, 9) for value in samples],
            "median_seconds": round(median_seconds, 9),
            "training_tokens": BASE_TRAINING_TOKENS,
            "training_tokens_per_second": round(BASE_TRAINING_TOKENS / median_seconds, 3),
        },
        checkpoint,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-pretrain-bin", required=True, type=Path, help="portable-profile executable")
    parser.add_argument("--native-pretrain-bin", required=True, type=Path, help="host-native-profile executable")
    parser.add_argument("--repetitions", type=int, default=3, help="train processes per profile (default: 3)")
    parser.add_argument("--omp-threads", type=int, default=1, help="OMP_NUM_THREADS for each train process")
    parser.add_argument("--output", type=Path, help="optional JSON output file; stdout remains JSON")
    arguments = parser.parse_args()

    try:
        portable_binary = arguments.portable_pretrain_bin.resolve()
        native_binary = arguments.native_pretrain_bin.resolve()
        if not portable_binary.is_file():
            fail(f"portable executable does not exist: {portable_binary}")
        if not native_binary.is_file():
            fail(f"native executable does not exist: {native_binary}")
        if portable_binary == native_binary:
            fail("portable and native executables must be distinct files")
        if not CORPUS.is_file():
            fail(f"fixture corpus does not exist: {CORPUS}")
        if arguments.repetitions <= 0:
            fail("--repetitions must be positive")
        if arguments.omp_threads <= 0:
            fail("--omp-threads must be positive")

        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(arguments.omp_threads)
        hardware = collect_hardware_profile()
        with tempfile.TemporaryDirectory(prefix="vegai-build-profile-comparison-") as temporary:
            workdir = Path(temporary)
            portable, portable_checkpoint = benchmark_profile(
                portable_binary, "portable", arguments.repetitions, workdir, environment
            )
            native, native_checkpoint = benchmark_profile(
                native_binary, "native", arguments.repetitions, workdir, environment
            )
            portable_checkpoint_bytes = portable_checkpoint.read_bytes()
            native_checkpoint_bytes = native_checkpoint.read_bytes()
            if portable_checkpoint_bytes != native_checkpoint_bytes:
                fail("portable and native profiles produced different checkpoint bytes")
            portable_checkpoint_sha256 = hashlib.sha256(portable_checkpoint_bytes).hexdigest()
            native_checkpoint_sha256 = hashlib.sha256(native_checkpoint_bytes).hexdigest()
            report: dict[str, Any] = {
                "schema": SCHEMA,
                "timing_scope": (
                    "one vegai_pretrain train process per sample; compilation, hardware discovery, fixture setup, "
                    "checkpoint comparison, and harness work excluded"
                ),
                "hardware": hardware,
                "workload": {
                    "corpus": str(CORPUS.relative_to(ROOT)),
                    "context_mode": "hybrid",
                    "epochs": BASE_EPOCHS,
                    "training_tokens_per_run": BASE_TRAINING_TOKENS,
                    "context_pairs_per_run": BASE_CONTEXT_PAIRS,
                    "omp_threads": arguments.omp_threads,
                },
                "portable": portable,
                "native": native,
                "checkpoint_byte_identical": True,
                "portable_checkpoint_sha256": portable_checkpoint_sha256,
                "native_checkpoint_sha256": native_checkpoint_sha256,
                "native_speedup_vs_portable": round(
                    portable["median_seconds"] / native["median_seconds"], 4
                ),
            }
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Pretraining build-profile comparison: FAIL: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(
        "Pretraining build-profile comparison: PASS: "
        f"{compact_hardware_summary(report['hardware'])}; "
        f"checkpoint_bytes_identical={report['checkpoint_byte_identical']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
