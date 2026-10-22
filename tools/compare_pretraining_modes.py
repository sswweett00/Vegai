#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare legacy, fast positional, and hybrid HDC pretraining modes.

The benchmark runs complete ``vegai_pretrain train`` processes with a fixed,
repository-authored corpus and times only those child processes.  It also asks
the executable for every fixture token's nearest semantic neighbour, so its
quality diagnostics use the real checkpoint and CLI instead of a Python model.

This is a controlled in-corpus semantic-cohesion diagnostic, not a natural
language benchmark or a production throughput promise.  It is intended to
make the optimization and hybrid-context trade-off inspectable in CI.
"""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples" / "corpus" / "mini_turkish_ascii_pretrain.txt"
EPOCHS = 2
TRAINING_TOKENS = 2_048
CONTEXT_PAIRS = 6_656

CLUSTERS: tuple[tuple[str, ...], ...] = (
    ("okul", "ogrenci", "ogretmen", "ders", "kitap", "sinif", "odev", "bilgi"),
    ("hastane", "doktor", "hemsire", "hasta", "tedavi", "ilac", "saglik", "bakim"),
    ("orman", "agac", "nehir", "dag", "toprak", "kus", "doga", "iklim"),
    ("bilgisayar", "yazilim", "kod", "veri", "ag", "sunucu", "sistem", "teknoloji"),
    ("pazar", "satici", "alici", "urun", "fiyat", "para", "ticaret", "musteri"),
    ("arac", "yol", "surucu", "yolcu", "durak", "sefer", "ulasim", "trafik"),
    ("ciftci", "tarla", "tohum", "sulama", "hasat", "mahsul", "koy", "tarim"),
    ("mahalle", "komsu", "aile", "destek", "toplanti", "dayanisma", "park", "belediye"),
)
TOKEN_TO_CLUSTER = {token: index for index, cluster in enumerate(CLUSTERS) for token in cluster}
TOKENS = tuple(token for cluster in CLUSTERS for token in cluster)
MODES = ("legacy", "positional", "hybrid")


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def metric(output: str, label: str) -> int:
    matches = re.findall(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", output, re.MULTILINE)
    require(len(matches) == 1, f"expected one {label!r} metric, got {matches!r}\n{output}")
    return int(matches[0])


def run(command: list[str], environment: dict[str, str]) -> tuple[str, float]:
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
        fail(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result.stdout, elapsed_seconds


def train_command(binary: Path, checkpoint: Path, mode: str) -> list[str]:
    return [
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
        "--context-mode",
        mode,
        "--epochs",
        str(EPOCHS),
        "--initial-plasticity",
        "25",
        "--minimum-plasticity",
        "1",
        "--decay-events",
        "8",
        "--strict",
    ]


def validate_train_output(output: str) -> None:
    require("phase: vocabulary" in output, "training output omitted vocabulary phase")
    require("phase: streaming HDC context learning" in output, "training output omitted learning phase")
    _, training = output.split("phase: streaming HDC context learning", maxsplit=1)
    require(metric(training, "completed epochs") == EPOCHS, "unexpected completed epoch count")
    require(metric(training, "raw tokens") == TRAINING_TOKENS, "unexpected training token count")
    require(metric(training, "active tokens") == TRAINING_TOKENS, "unexpected active training token count")
    require(metric(training, "out-of-vocabulary tokens") == 0, "fixture produced OOV training tokens")
    require(metric(training, "ordered context pairs") == CONTEXT_PAIRS, "unexpected context-pair count")
    require(metric(training, "prototype updates") == TRAINING_TOKENS, "unexpected prototype-update count")


def benchmark_mode(
    binary: Path,
    mode: str,
    repetitions: int,
    workdir: Path,
    environment: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    samples: list[float] = []
    checkpoint = workdir / f"{mode}-last.vgai"
    for repetition in range(repetitions):
        candidate = workdir / f"{mode}-{repetition}.vgai"
        output, elapsed_seconds = run(train_command(binary, candidate, mode), environment)
        validate_train_output(output)
        require(candidate.is_file(), f"{mode} run did not write a checkpoint")
        samples.append(elapsed_seconds)
        checkpoint = candidate

    median_seconds = statistics.median(samples)
    return (
        {
            "repetitions": repetitions,
            "samples_seconds": [round(value, 9) for value in samples],
            "median_seconds": round(median_seconds, 9),
            "training_tokens": TRAINING_TOKENS,
            "training_tokens_per_second": round(TRAINING_TOKENS / median_seconds, 3),
        },
        checkpoint,
    )


def inspect_top1_quality(
    binary: Path,
    checkpoint: Path,
    mode: str,
    query_file: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    output, _ = run([str(binary), "inspect", str(checkpoint), "--query-file", str(query_file)], environment)
    require(f"context mode: {mode}" in output, f"checkpoint mode mismatch for {mode}")
    blocks = re.findall(
        r"^query:\s*(\S+)\s*\n\s*nearest semantic neighbour:\s*(\S+)\s*"
        r"\n\s*bipolar similarity:\s*(-?\d+\.\d+)\s*$",
        output,
        re.MULTILINE,
    )
    require(len(blocks) == len(TOKENS), f"expected {len(TOKENS)} batch query results, got {len(blocks)}\n{output}")

    matches: list[tuple[str, str, float]] = []
    for token, neighbour, rendered_similarity in blocks:
        require(token in TOKEN_TO_CLUSTER, f"unexpected query {token!r}")
        require(neighbour in TOKEN_TO_CLUSTER, f"unexpected neighbour {neighbour!r} for {token!r}")
        matches.append((token, neighbour, float(rendered_similarity)))
    require(tuple(entry[0] for entry in matches) == TOKENS, "batch query output reordered or omitted fixture tokens")

    correct = [entry for entry in matches if TOKEN_TO_CLUSTER[entry[0]] == TOKEN_TO_CLUSTER[entry[1]]]
    similarities = [entry[2] for entry in matches]
    return {
        "queries": len(matches),
        "same_cluster_top1_hits": len(correct),
        "same_cluster_top1_percent": round(100.0 * len(correct) / len(matches), 3),
        "mean_top1_similarity": round(statistics.fmean(similarities), 6),
        "minimum_top1_similarity": round(min(similarities), 6),
        "maximum_top1_similarity": round(max(similarities), 6),
        "top1_neighbours": [[token, neighbour, round(similarity, 5)] for token, neighbour, similarity in matches],
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--repetitions", type=int, default=3, help="complete train processes per mode (default: 3)")
    parser.add_argument("--omp-threads", type=int, default=1, help="OMP_NUM_THREADS for each child (default: 1)")
    parser.add_argument("--output", type=Path, help="optional JSON output file; stdout remains JSON")
    arguments = parser.parse_args()

    try:
        binary = arguments.pretrain_bin.resolve()
        require(binary.is_file(), f"pretraining executable does not exist: {binary}")
        require(CORPUS.is_file(), f"fixture corpus does not exist: {CORPUS}")
        require(arguments.repetitions > 0, "--repetitions must be positive")
        require(arguments.omp_threads > 0, "--omp-threads must be positive")
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(arguments.omp_threads)

        with tempfile.TemporaryDirectory(prefix="vegai-mode-comparison-") as temporary:
            workdir = Path(temporary)
            query_file = workdir / "fixture-queries.txt"
            query_file.write_text("\n".join(TOKENS) + "\n", encoding="ascii")
            speed: dict[str, dict[str, Any]] = {}
            checkpoints: dict[str, Path] = {}
            for mode in MODES:
                speed[mode], checkpoints[mode] = benchmark_mode(
                    binary, mode, arguments.repetitions, workdir, environment
                )
            quality = {
                mode: inspect_top1_quality(binary, checkpoints[mode], mode, query_file, environment)
                for mode in MODES
            }

            legacy_pairs = quality["legacy"]["top1_neighbours"]
            require(
                quality["positional"]["top1_neighbours"] == legacy_pairs,
                "fast positional encoder diverged from legacy nearest-neighbour output",
            )
            require(
                quality["hybrid"]["same_cluster_top1_hits"] >= quality["legacy"]["same_cluster_top1_hits"],
                "hybrid encoder regressed controlled top-1 cluster retrieval",
            )
            semantic_gain = (
                quality["hybrid"]["mean_top1_similarity"]
                - quality["positional"]["mean_top1_similarity"]
            )
            require(
                semantic_gain > 0.01,
                f"hybrid semantic cohesion gain was unexpectedly small: {semantic_gain:.6f}",
            )

            report: dict[str, Any] = {
                "schema": "vegai-pretraining-mode-comparison/v1",
                "workload": {
                    "corpus": str(CORPUS.relative_to(ROOT)),
                    "epochs": EPOCHS,
                    "training_tokens_per_mode_run": TRAINING_TOKENS,
                    "context_pairs_per_mode_run": CONTEXT_PAIRS,
                    "omp_threads": arguments.omp_threads,
                    "timing_scope": "single vegai_pretrain train process; compilation and harness work excluded",
                },
                "speed": speed,
                "quality": quality,
                "comparison": {
                    "positional_speedup_vs_legacy": round(
                        speed["legacy"]["median_seconds"] / speed["positional"]["median_seconds"], 4
                    ),
                    "hybrid_speedup_vs_legacy": round(
                        speed["legacy"]["median_seconds"] / speed["hybrid"]["median_seconds"], 4
                    ),
                    "hybrid_mean_top1_similarity_gain_vs_positional": round(semantic_gain, 6),
                    "hybrid_mean_top1_similarity_relative_gain_percent": round(
                        100.0 * semantic_gain / quality["positional"]["mean_top1_similarity"], 3
                    ),
                    "positional_neighbour_outputs_match_legacy": True,
                },
            }
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Pretraining mode comparison: FAIL: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
