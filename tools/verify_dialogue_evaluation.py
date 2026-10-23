#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Integration verification for held-out grounded dialogue quality gates.

This invokes real compiled Vegai binaries, then verifies that the evaluator
binds its dependencies, uses keyed held-out prompt identifiers without copying
prompt text, rejects malformed case schema, and returns a nonzero quality-gate
status for a deliberately wrong expectation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from dialogue_evaluation import CASE_SCHEMA, REPORT_SCHEMA, iter_cases


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools" / "dialogue_evaluation.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
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


def evaluator_command(
    chat: Path, checkpoint: Path, knowledge: Path, evaluation: Path, report: Path, *gates: str
) -> list[str]:
    return [
        sys.executable, str(EVALUATOR), "--chat-bin", str(chat), "--checkpoint", str(checkpoint),
        "--knowledge", str(knowledge), "--evaluation", str(evaluation), "--output", str(report),
        "--max-entries", "8", "--min-similarity", "0.99", "--min-margin", "0.0", *gates,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    parser.add_argument("--chat-bin", required=True, type=Path, help="compiled vegai_chat executable")
    arguments = parser.parse_args()
    pretrain = arguments.pretrain_bin.resolve()
    chat = arguments.chat_bin.resolve()

    try:
        require(pretrain.is_file(), f"pretraining executable does not exist: {pretrain}")
        require(chat.is_file(), f"chat executable does not exist: {chat}")
        fixture = ROOT / "examples" / "dialogue" / "mini_turkish_hdc_evaluation.jsonl"
        fixture_cases = list(iter_cases(fixture, 100))
        require(len(fixture_cases) == 9, "checked-in dialogue evaluation fixture count changed")
        require(sum(case.expected_prompt is None for case in fixture_cases) == 1,
                "checked-in dialogue evaluation fixture lost its abstention case")

        with tempfile.TemporaryDirectory(prefix="vegai-dialogue-evaluation-") as temporary:
            workspace = Path(temporary)
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
            evaluation = workspace / "heldout.jsonl"
            evaluation.write_text(
                "\n".join(
                    (
                        json.dumps({"schema_version": CASE_SCHEMA, "id": "education", "prompt": "okul ders", "expected_prompt": "okul ders"}),
                        json.dumps({"schema_version": CASE_SCHEMA, "id": "health", "prompt": "hastane doktor", "expected_prompt": "hastane doktor"}),
                        json.dumps({"schema_version": CASE_SCHEMA, "id": "abstain", "prompt": "tarim", "expected_prompt": None}),
                    )
                ) + "\n",
                encoding="ascii", newline="\n",
            )
            report_path = workspace / "report.json"
            hmac_key = workspace / "case-hmac-key.bin"
            hmac_key.write_bytes(b"synthetic-test-only-dialogue-report-key")
            passed = run(
                evaluator_command(
                    chat, checkpoint, knowledge, evaluation, report_path,
                    "--case-hmac-key-file", str(hmac_key),
                    "--min-grounded-prompt-accuracy", "1", "--min-grounded-precision", "1",
                    "--min-grounded-recall", "1", "--min-abstain-precision", "1", "--min-abstain-recall", "1",
                )
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            require(report.get("schema") == REPORT_SCHEMA, "evaluation report schema is wrong")
            require("evaluator_tool" in report.get("identities", {}),
                    "evaluation report did not bind the evaluator implementation")
            require("case_hmac_key" in report.get("identities", {}),
                    "keyed case identifiers were not provenance-bound")
            require(report.get("identities", {}).get("case_hmac_key", {}).get("path") == "<redacted-case-hmac-key>",
                    "case HMAC key location leaked into the report")
            require(report.get("evaluation_contract", {}).get("case_digest_algorithm") == "hmac-sha256",
                    "sensitive held-out report did not use keyed case identifiers")
            require(report.get("quality_gate", {}).get("passed") is True, "valid held-out quality gate did not pass")
            metrics = report.get("metrics", {})
            require(
                metrics.get("correct_grounded_prompt_cases") == 2
                and metrics.get("true_abstain_cases") == 1
                and metrics.get("false_grounded_cases") == 0
                and metrics.get("false_abstain_cases") == 0,
                f"held-out metrics are inconsistent: {report}",
            )
            require(report.get("policy", {}).get("history_turns") == 0, "evaluation was not stateless")
            rendered_report = report_path.read_text(encoding="utf-8")
            require("okul ders" not in rendered_report and "operator yaniti" not in rendered_report,
                    "privacy-preserving report copied a held-out prompt or answer")
            require("quality_gate" in passed.stdout, "evaluator did not render a report to stdout")

            unsafe_output = subprocess.run(
                evaluator_command(chat, checkpoint, knowledge, evaluation, evaluation),
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            require(unsafe_output.returncode == 2 and "must not overwrite the evaluation set" in unsafe_output.stdout,
                    "evaluator allowed report output to overwrite an immutable input")

            wrong = workspace / "wrong.jsonl"
            wrong.write_text(
                json.dumps({"schema_version": CASE_SCHEMA, "id": "wrong", "prompt": "okul ders", "expected_prompt": "hastane doktor"}) + "\n",
                encoding="ascii", newline="\n",
            )
            wrong_report = workspace / "wrong-report.json"
            rejected_quality = run(
                evaluator_command(
                    chat, checkpoint, knowledge, wrong, wrong_report,
                    "--min-grounded-prompt-accuracy", "1", "--max-wrong-grounded-prompt-cases", "0",
                ),
                expected=3,
            )
            wrong_payload = json.loads(wrong_report.read_text(encoding="utf-8"))
            require(wrong_payload.get("quality_gate", {}).get("passed") is False,
                    "wrong held-out expectation did not leave a failed report")
            require(wrong_payload.get("quality_gate", {}).get("requirements", {}).get("max_wrong_grounded_prompt_cases") == 0,
                    "count-based incorrect-grounding gate was not provenance-bound")
            require("QUALITY GATE FAILED" in rejected_quality.stdout,
                    "quality-gate failure was not explicit")

            unsafe_abstention = workspace / "unsafe-abstention.jsonl"
            unsafe_abstention.write_text(
                json.dumps({"schema_version": CASE_SCHEMA, "id": "unsafe", "prompt": "okul ders", "expected_prompt": None}) + "\n",
                encoding="ascii", newline="\n",
            )
            unsafe_report = workspace / "unsafe-abstention-report.json"
            unsafe_gate = run(
                evaluator_command(
                    chat, checkpoint, knowledge, unsafe_abstention, unsafe_report,
                    "--max-false-grounded-cases", "0",
                ),
                expected=3,
            )
            unsafe_payload = json.loads(unsafe_report.read_text(encoding="utf-8"))
            require(unsafe_payload.get("metrics", {}).get("false_grounded_cases") == 1,
                    "expected-abstention grounding was not counted as unsafe")
            require(unsafe_payload.get("quality_gate", {}).get("passed") is False
                    and "false_grounded_cases=1 exceeds permitted 0" in unsafe_gate.stdout,
                    "unsafe abstention behavior did not fail the count gate")

            malformed = workspace / "malformed.jsonl"
            malformed.write_text('{"id":"bad","prompt":"okul ders","expected_prompt":null,"unknown":1}\n', encoding="ascii")
            malformed_result = subprocess.run(
                evaluator_command(chat, checkpoint, knowledge, malformed, workspace / "malformed-report.json"),
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            require(malformed_result.returncode == 2 and "unsupported fields" in malformed_result.stdout,
                    f"malformed evaluation schema was accepted:\n{malformed_result.stdout}")
    except (AssertionError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"dialogue evaluation verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("dialogue evaluation verification: PASS (held-out grounding/abstention metrics, provenance, privacy, strict schema, release gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
