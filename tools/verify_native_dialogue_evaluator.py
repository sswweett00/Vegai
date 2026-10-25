#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the all-Fortran stateless dialogue batch evaluator end to end.

The fixture intentionally checks retrieval and abstention accounting only. It
is not a generation, factuality, or natural-conversation benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != expected:
        raise AssertionError(
            f"command returned {completed.returncode}, expected {expected}: {' '.join(command)}\n{completed.stdout}"
        )
    return completed


def train_command(binary: Path, corpus: Path, checkpoint: Path) -> list[str]:
    return [
        str(binary), "train", str(corpus), str(checkpoint),
        "--max-vocab", "64", "--min-frequency", "1", "--window", "2", "--position-shift", "41",
        "--context-mode", "hierarchical", "--epochs", "2", "--initial-plasticity", "25",
        "--minimum-plasticity", "1", "--decay-events", "8", "--strict",
    ]


def evaluator_command(binary: Path, checkpoint: Path, knowledge: Path, evaluation: Path, *options: str) -> list[str]:
    return [
        str(binary), "evaluate", str(checkpoint), str(knowledge), str(evaluation),
        "--max-entries", "8", "--max-cases", "8", "--min-similarity", "0.99", "--min-margin", "0.0",
        *options,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--native-eval-bin", required=True, type=Path, help="compiled vegai_dialogue_eval executable")
    arguments = parser.parse_args()
    pretrain = arguments.pretrain_bin.resolve()
    evaluator = arguments.native_eval_bin.resolve()

    try:
        require(pretrain.is_file(), f"pretraining executable does not exist: {pretrain}")
        require(evaluator.is_file(), f"native evaluator executable does not exist: {evaluator}")
        with tempfile.TemporaryDirectory(prefix="vegai-native-dialogue-evaluator-") as temporary_name:
            workspace = Path(temporary_name)
            corpus = workspace / "corpus.txt"
            corpus.write_text(
                "\n".join(
                    (
                        "okul ders kitap ogrenci",
                        "okul ders ogretmen kitap",
                        "hastane doktor hasta tedavi",
                        "hastane doktor hemsire ilac",
                        "bilgisayar yazilim kod veri",
                        "bilgisayar yazilim sunucu sistem",
                        "tarim mahsul ciftci tohum",
                    )
                ) + "\n",
                encoding="ascii", newline="\n",
            )
            checkpoint = workspace / "dialogue.vgai"
            trained = run(train_command(pretrain, corpus, checkpoint))
            require("phase: streaming HDC context learning" in trained.stdout, "real HDC training did not run")

            knowledge = workspace / "knowledge.tsv"
            knowledge.write_text(
                "\n".join(
                    (
                        "okul ders\tokul ders operator yaniti",
                        "hastane doktor\thastane doktor operator yaniti",
                        "bilgisayar yazilim\tbilgisayar yazilim operator yaniti",
                    )
                ) + "\n",
                encoding="ascii", newline="\n",
            )
            evaluation = workspace / "evaluation.tsv"
            evaluation.write_text(
                "\n".join(
                    (
                        "exact\tokul ders\tokul ders",
                        "abstain\ttarim\t",
                        "oov\ttamamenbilinmeyen\t",
                    )
                ) + "\n",
                encoding="ascii", newline="\n",
            )
            measured = run(evaluator_command(evaluator, checkpoint, knowledge, evaluation))
            report = json.loads(measured.stdout)
            require(report.get("schema") == "vegai-native-dialogue-evaluation-report/v1",
                    f"wrong native evaluator report schema: {report}")
            require(report.get("contract") == "stateless native Fortran TSV evaluation; literal retrieval only",
                    "native report did not preserve its non-generative contract")
            identities = report.get("identities", {})
            require(
                all(isinstance(identities.get(name, {}).get("sha256"), str)
                    and len(identities[name]["sha256"]) == 64
                    and identities[name].get("bytes", 0) > 0
                    for name in ("checkpoint", "knowledge", "evaluation")),
                f"native report did not bind checkpoint/knowledge/evaluation identities: {report}",
            )
            metrics = report.get("metrics", {})
            require(
                metrics.get("total_cases") == 3
                and metrics.get("expected_grounded_cases") == 1
                and metrics.get("expected_abstain_cases") == 2
                and metrics.get("correct_grounded_prompt_cases") == 1
                and metrics.get("true_abstain_cases") == 1
                and metrics.get("false_grounded_cases") == 0
                and metrics.get("oov_rejected_abstain_cases") == 1,
                f"native evaluator accounting is wrong: {report}",
            )
            require("okul ders" not in measured.stdout and "operator yaniti" not in measured.stdout,
                    "native report leaked raw prompt/response text")

            # Bounded measurement must fail rather than silently accepting a
            # prefix, which prevents accidental source-order benchmark drift.
            too_small = subprocess.run(
                evaluator_command(evaluator, checkpoint, knowledge, evaluation, "--max-cases", "2"),
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="strict",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            require(too_small.returncode != 0 and "exceeds --max-cases" in too_small.stdout,
                    f"native evaluator silently accepted an over-limit case stream:\n{too_small.stdout}")
    except (AssertionError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"native dialogue evaluator verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("native dialogue evaluator verification: PASS (stateless Fortran batch retrieval/abstention accounting, bounded TSV)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
