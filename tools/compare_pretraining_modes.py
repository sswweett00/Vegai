#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare exact fused HDC kernels and controlled context encoders.

Every elapsed sample in this report is one complete ``vegai_pretrain train``
child process.  Compilation, test setup, fixture construction, checkpoint
inspection, JSON rendering, and the Python harness are deliberately outside the
timed interval.  The train process itself necessarily includes its vocabulary
scan, streaming learning epochs, and checkpoint write.

The semantic measurements are deliberately narrow controlled diagnostics over
repository-authored, disjoint topical token sets.  They are useful regression
signals for this HDC/VSA implementation; they are not claims about natural
language intelligence, held-out task accuracy, or production throughput.
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
CORPUS = ROOT / "examples" / "corpus" / "mini_turkish_ascii_pretrain.txt"
BASE_EPOCHS = 2
BASE_TRAINING_TOKENS = 2_048
BASE_CONTEXT_PAIRS = 6_656
BASE_MODES = ("legacy", "positional", "hybrid", "resonant")
LONG_CONTEXT_MODES = ("hybrid", "hierarchical", "coherent")
LONG_RECORDS_PER_CLUSTER = 16
LONG_EPOCHS = 2

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
    """Run one command; callers only retain elapsed time for train commands."""
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


def train_command(
    binary: Path,
    corpus: Path,
    checkpoint: Path,
    mode: str,
    epochs: int,
    *,
    reference_context_kernel: bool = False,
) -> list[str]:
    command = [
        str(binary),
        "train",
        str(corpus),
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
        str(epochs),
        "--initial-plasticity",
        "25",
        "--minimum-plasticity",
        "1",
        "--decay-events",
        "8",
        "--strict",
    ]
    if reference_context_kernel:
        command.append("--reference-context-kernel")
    return command


def validate_train_output(
    output: str,
    *,
    epochs: int,
    training_tokens: int,
    context_pairs: int,
) -> None:
    require("phase: vocabulary" in output, "training output omitted vocabulary phase")
    require("phase: streaming HDC context learning" in output, "training output omitted learning phase")
    _, training = output.split("phase: streaming HDC context learning", maxsplit=1)
    require(metric(training, "completed epochs") == epochs, "unexpected completed epoch count")
    require(metric(training, "raw tokens") == training_tokens, "unexpected training token count")
    require(metric(training, "active tokens") == training_tokens, "unexpected active training token count")
    require(metric(training, "out-of-vocabulary tokens") == 0, "fixture produced OOV training tokens")
    require(metric(training, "ordered context pairs") == context_pairs, "unexpected context-pair count")
    require(metric(training, "prototype updates") == training_tokens, "unexpected prototype-update count")


def benchmark_train(
    binary: Path,
    corpus: Path,
    mode: str,
    epochs: int,
    training_tokens: int,
    context_pairs: int,
    repetitions: int,
    workdir: Path,
    environment: dict[str, str],
    *,
    label: str,
    reference_context_kernel: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Time only real train subprocesses and retain the final checkpoint."""
    samples: list[float] = []
    checkpoint = workdir / f"{label}-last.vgai"
    for repetition in range(repetitions):
        candidate = workdir / f"{label}-{repetition}.vgai"
        output, elapsed_seconds = run(
            train_command(
                binary,
                corpus,
                candidate,
                mode,
                epochs,
                reference_context_kernel=reference_context_kernel,
            ),
            environment,
        )
        validate_train_output(
            output,
            epochs=epochs,
            training_tokens=training_tokens,
            context_pairs=context_pairs,
        )
        require(candidate.is_file(), f"{label} run did not write a checkpoint")
        samples.append(elapsed_seconds)
        checkpoint = candidate

    median_seconds = statistics.median(samples)
    return (
        {
            "repetitions": repetitions,
            "samples_seconds": [round(value, 9) for value in samples],
            "median_seconds": round(median_seconds, 9),
            "training_tokens": training_tokens,
            "training_tokens_per_second": round(training_tokens / median_seconds, 3),
            "reference_context_kernel": reference_context_kernel,
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
    """Use the real checkpoint/query path, but never include it in timing."""
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


def write_long_context_fixture(path: Path) -> tuple[int, int, int]:
    """Create an explicit long-record controlled topical diagnostic fixture.

    Each record repeats the eight tokens of exactly one disjoint cluster, making
    sixteen active tokens per record.  It intentionally crosses the hierarchical
    encoder's twelve-token threshold but contains no hidden linguistic claim.
    """
    record_token_count = 2 * len(CLUSTERS[0])
    context_pairs_per_record = sum(
        min(2, index) + min(2, record_token_count - 1 - index)
        for index in range(record_token_count)
    )
    records = len(CLUSTERS) * LONG_RECORDS_PER_CLUSTER
    with path.open("w", encoding="ascii", newline="\n") as destination:
        for cluster in CLUSTERS:
            record = " ".join((*cluster, *cluster))
            for _ in range(LONG_RECORDS_PER_CLUSTER):
                destination.write(record + "\n")
    tokens_per_epoch = records * record_token_count
    return tokens_per_epoch * LONG_EPOCHS, records, context_pairs_per_record * records * LONG_EPOCHS


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--repetitions", type=int, default=3, help="complete train processes per variant (default: 3)")
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

            # Historical encoder comparison, retained so the positional
            # regression oracle and hybrid cohesion diagnostic remain visible.
            speed: dict[str, dict[str, Any]] = {}
            checkpoints: dict[str, Path] = {}
            for mode in BASE_MODES:
                speed[mode], checkpoints[mode] = benchmark_train(
                    binary,
                    CORPUS,
                    mode,
                    BASE_EPOCHS,
                    BASE_TRAINING_TOKENS,
                    BASE_CONTEXT_PAIRS,
                    arguments.repetitions,
                    workdir,
                    environment,
                    label=f"base-{mode}",
                )
            quality = {
                mode: inspect_top1_quality(binary, checkpoints[mode], mode, query_file, environment)
                for mode in BASE_MODES
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
            hybrid_semantic_gain = (
                quality["hybrid"]["mean_top1_similarity"]
                - quality["positional"]["mean_top1_similarity"]
            )
            require(
                hybrid_semantic_gain > 0.01,
                f"hybrid semantic cohesion gain was unexpectedly small: {hybrid_semantic_gain:.6f}",
            )
            # RESONANT's first epoch is exactly HYBRID; its second epoch uses
            # only an immutable semantic snapshot captured before streaming
            # updates begin. Report the controlled retrieval delta separately
            # from speed so the evidence remains a narrow fixture diagnostic,
            # not a general-intelligence claim.
            resonant_semantic_gain = (
                quality["resonant"]["mean_top1_similarity"]
                - quality["hybrid"]["mean_top1_similarity"]
            )

            # A true before/after optimization comparison: only the hidden
            # implementation switch differs.  The emitted checkpoints must be
            # byte-identical, proving bit-exact semantics rather than merely
            # matching a qualitative neighbour metric.
            reference_speed, reference_checkpoint = benchmark_train(
                binary,
                CORPUS,
                "hybrid",
                BASE_EPOCHS,
                BASE_TRAINING_TOKENS,
                BASE_CONTEXT_PAIRS,
                arguments.repetitions,
                workdir,
                environment,
                label="reference-hybrid",
                reference_context_kernel=True,
            )
            fused_checkpoint = checkpoints["hybrid"]
            reference_checksum = checkpoint_sha256(reference_checkpoint)
            fused_checksum = checkpoint_sha256(fused_checkpoint)
            require(
                reference_checkpoint.read_bytes() == fused_checkpoint.read_bytes(),
                "fused and reference hybrid kernels produced different checkpoint bytes",
            )

            # A separate long-record, controlled diagnostic for the bounded
            # hierarchical channel.  The regular fixture has eight-token
            # records and deliberately stays below this channel's threshold.
            long_corpus = workdir / "long-context-controlled.txt"
            long_training_tokens, long_records, long_context_pairs = write_long_context_fixture(long_corpus)
            long_speed: dict[str, dict[str, Any]] = {}
            long_checkpoints: dict[str, Path] = {}
            for mode in LONG_CONTEXT_MODES:
                long_speed[mode], long_checkpoints[mode] = benchmark_train(
                    binary,
                    long_corpus,
                    mode,
                    LONG_EPOCHS,
                    long_training_tokens,
                    long_context_pairs,
                    arguments.repetitions,
                    workdir,
                    environment,
                    label=f"long-{mode}",
                )
            long_quality = {
                mode: inspect_top1_quality(binary, long_checkpoints[mode], mode, query_file, environment)
                for mode in LONG_CONTEXT_MODES
            }
            hierarchical_similarity_gain = (
                long_quality["hierarchical"]["mean_top1_similarity"]
                - long_quality["hybrid"]["mean_top1_similarity"]
            )
            require(
                long_quality["hierarchical"]["same_cluster_top1_hits"]
                >= long_quality["hybrid"]["same_cluster_top1_hits"],
                "hierarchical mode regressed controlled long-record top-1 cohesion",
            )
            require(
                hierarchical_similarity_gain > 0.002,
                "hierarchical long-record cohesion gain was unexpectedly small: "
                f"{hierarchical_similarity_gain:.6f}",
            )
            coherent_similarity_gain = (
                long_quality["coherent"]["mean_top1_similarity"]
                - long_quality["hybrid"]["mean_top1_similarity"]
            )
            require(
                long_quality["coherent"]["same_cluster_top1_hits"]
                >= long_quality["hybrid"]["same_cluster_top1_hits"],
                "coherent cross-fold mode regressed controlled long-record top-1 cohesion",
            )

            report: dict[str, Any] = {
                "schema": "vegai-pretraining-mode-comparison/v5",
                "hardware": collect_hardware_profile(),
                "timing_scope": (
                    "one vegai_pretrain train process per sample; compilation, CI setup, fixture construction, "
                    "inspection, and harness work excluded"
                ),
                "workload": {
                    "corpus": str(CORPUS.relative_to(ROOT)),
                    "epochs": BASE_EPOCHS,
                    "training_tokens_per_mode_run": BASE_TRAINING_TOKENS,
                    "context_pairs_per_mode_run": BASE_CONTEXT_PAIRS,
                    "omp_threads": arguments.omp_threads,
                },
                "speed": speed,
                "quality": quality,
                "fused_kernel": {
                    "mode": "hybrid",
                    "reference": reference_speed,
                    "fused": speed["hybrid"],
                    "checkpoint_byte_identical": True,
                    "reference_checkpoint_sha256": reference_checksum,
                    "fused_checkpoint_sha256": fused_checksum,
                    "fused_speedup_vs_reference": round(
                        reference_speed["median_seconds"] / speed["hybrid"]["median_seconds"], 4
                    ),
                },
                "long_context_diagnostic": {
                    "records": long_records,
                    "tokens_per_record": 2 * len(CLUSTERS[0]),
                    "hierarchical_min_active_tokens": 12,
                    "epochs": LONG_EPOCHS,
                    "training_tokens_per_mode_run": long_training_tokens,
                    "context_pairs_per_mode_run": long_context_pairs,
                    "speed": long_speed,
                    "quality": long_quality,
                    "hierarchical_mean_top1_similarity_gain_vs_hybrid": round(
                        hierarchical_similarity_gain, 6
                    ),
                    "coherent_mean_top1_similarity_gain_vs_hybrid": round(
                        coherent_similarity_gain, 6
                    ),
                    "hierarchical_speedup_vs_hybrid": round(
                        long_speed["hybrid"]["median_seconds"]
                        / long_speed["hierarchical"]["median_seconds"],
                        4,
                    ),
                    "coherent_speedup_vs_hybrid": round(
                        long_speed["hybrid"]["median_seconds"]
                        / long_speed["coherent"]["median_seconds"],
                        4,
                    ),
                },
                "comparison": {
                    "positional_speedup_vs_legacy": round(
                        speed["legacy"]["median_seconds"] / speed["positional"]["median_seconds"], 4
                    ),
                    "hybrid_speedup_vs_legacy": round(
                        speed["legacy"]["median_seconds"] / speed["hybrid"]["median_seconds"], 4
                    ),
                    "hybrid_mean_top1_similarity_gain_vs_positional": round(hybrid_semantic_gain, 6),
                    "resonant_mean_top1_similarity_gain_vs_hybrid": round(resonant_semantic_gain, 6),
                    "resonant_same_cluster_top1_hit_delta_vs_hybrid": (
                        quality["resonant"]["same_cluster_top1_hits"]
                        - quality["hybrid"]["same_cluster_top1_hits"]
                    ),
                    "hybrid_mean_top1_similarity_relative_gain_percent": round(
                        100.0 * hybrid_semantic_gain / quality["positional"]["mean_top1_similarity"], 3
                    ),
                    "positional_neighbour_outputs_match_legacy": True,
                    "fused_checkpoint_byte_identical_to_reference": True,
                    "fused_hybrid_speedup_vs_reference": round(
                        reference_speed["median_seconds"] / speed["hybrid"]["median_seconds"], 4
                    ),
                    "hierarchical_long_record_mean_top1_similarity_gain_vs_hybrid": round(
                        hierarchical_similarity_gain, 6
                    ),
                    "coherent_long_record_mean_top1_similarity_gain_vs_hybrid": round(
                        coherent_similarity_gain, 6
                    ),
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
    print("Pretraining mode comparison hardware: " + compact_hardware_summary(report["hardware"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
