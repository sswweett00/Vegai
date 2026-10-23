#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise Vegai's compiled, data-driven grounded dialogue retrieval path.

The verifier trains a real deterministic HDC checkpoint, loads TSV knowledge
bases supplied at runtime, and proves that answers change with TSV content—not
source-code response branches.  It also checks grounded abstention, strict
unknown-token rejection, malformed-knowledge rejection, and the interactive
REPL path.  This is retrieval over an HDC checkpoint, not an LLM quality test.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(command: list[str], expected: int = 0, input_text: str | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}: {' '.join(command)}\n{result.stdout}"
        )
    return result.stdout


def rejected(command: list[str], input_text: str | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    if result.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(command)}\n{result.stdout}")
    return result.stdout


def train_command(binary: Path, corpus: Path, checkpoint: Path) -> list[str]:
    return [
        str(binary),
        "train",
        str(corpus),
        str(checkpoint),
        "--max-vocab",
        "64",
        "--min-frequency",
        "1",
        "--window",
        "2",
        "--position-shift",
        "41",
        "--context-mode",
        "hierarchical",
        "--epochs",
        "2",
        "--initial-plasticity",
        "25",
        "--minimum-plasticity",
        "1",
        "--decay-events",
        "8",
        "--strict",
    ]


def validate_checked_in_fixture(fixture: Path) -> None:
    """Keep the documented operator-editable fixture inside the loader contract."""
    raw = fixture.read_bytes()
    require(raw.endswith(b"\n"), f"checked-in dialogue fixture must end in LF: {fixture}")
    require(b"\r" not in raw, f"checked-in dialogue fixture must not contain CR: {fixture}")
    require(all(byte <= 127 for byte in raw), f"checked-in dialogue fixture must be ASCII: {fixture}")

    prompts: set[bytes] = set()
    rows = raw.splitlines()
    require(bool(rows), f"checked-in dialogue fixture must contain rows: {fixture}")
    for number, row in enumerate(rows, start=1):
        require(
            all(byte == 9 or 32 <= byte <= 126 for byte in row),
            f"fixture row {number} contains a non-printable ASCII byte",
        )
        require(row.count(b"\t") == 1, f"fixture row {number} must contain exactly one TAB")
        prompt, response = row.split(b"\t")
        require(prompt and response, f"fixture row {number} has an empty prompt or response")
        require(len(prompt) <= 1024, f"fixture row {number} exceeds the prompt limit")
        require(len(response) <= 2048, f"fixture row {number} exceeds the response limit")
        require(prompt not in prompts, f"fixture row {number} repeats an operator prompt")
        prompts.add(prompt)


def ask_command(chat: Path, checkpoint: Path, knowledge: Path, prompt: str, minimum_similarity: str = "0.90") -> list[str]:
    return [
        str(chat),
        "ask",
        str(checkpoint),
        str(knowledge),
        "--prompt",
        prompt,
        "--max-entries",
        "8",
        "--history-turns",
        "0",
        "--min-similarity",
        minimum_similarity,
        "--min-margin",
        "0.0",
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
        require(chat.is_file(), f"dialogue executable does not exist: {chat}")
        checked_in_fixture = ROOT / "examples" / "dialogue" / "mini_turkish_hdc_dialogue.tsv"
        validate_checked_in_fixture(checked_in_fixture)
        with tempfile.TemporaryDirectory(prefix="vegai-dialogue-") as temporary:
            workspace = Path(temporary)
            corpus = workspace / "dialogue-train.txt"
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
                )
                + "\n",
                encoding="ascii",
                newline="\n",
            )
            checkpoint = workspace / "dialogue.vgai"
            train_output = run(train_command(pretrain, corpus, checkpoint))
            require("phase: streaming HDC context learning" in train_output, "real HDC training did not run")
            require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "training did not publish a checkpoint")

            knowledge = workspace / "knowledge.tsv"
            knowledge.write_text(
                "\n".join(
                    (
                        "okul ders\tokul ders kitap ogrenci hakkinda bilgi",
                        "hastane doktor\thastane doktor hasta tedavi hakkinda bilgi",
                        "bilgisayar yazilim\tbilgisayar yazilim kod veri hakkinda bilgi",
                    )
                )
                + "\n",
                encoding="ascii",
                newline="\n",
            )
            first = run(ask_command(chat, checkpoint, knowledge, "okul ders"))
            require("query known tokens: 2" in first, f"known-token accounting is wrong\n{first}")
            require("query ignored tokens: 0" in first, f"ignored-token accounting is wrong\n{first}")
            require("candidates examined: 3" in first, f"all bounded entries were not examined\n{first}")
            require("grounded: true" in first, f"exact knowledge prompt did not ground\n{first}")
            require("matched prompt: okul ders" in first, f"retrieval did not disclose matched source prompt\n{first}")
            require(
                "answer: okul ders kitap ogrenci hakkinda bilgi" in first,
                f"runtime did not return the supplied knowledge response\n{first}",
            )

            # Changing only operator-owned TSV content must change the rendered
            # answer.  This prevents a source-coded answer branch from passing.
            replacement_knowledge = workspace / "replacement.tsv"
            replacement_knowledge.write_text(
                "\n".join(
                    (
                        "okul ders\tyeni kaynak cevabi veri dosyasindan gelir",
                        "hastane doktor\thastane doktor hasta tedavi hakkinda bilgi",
                        "bilgisayar yazilim\tbilgisayar yazilim kod veri hakkinda bilgi",
                    )
                )
                + "\n",
                encoding="ascii",
                newline="\n",
            )
            replacement = run(ask_command(chat, checkpoint, replacement_knowledge, "okul ders"))
            require("grounded: true" in replacement, f"replacement data failed to ground\n{replacement}")
            require(
                "answer: yeni kaynak cevabi veri dosyasindan gelir" in replacement
                and "answer: okul ders kitap ogrenci hakkinda bilgi" not in replacement,
                f"answer was not controlled solely by TSV content\n{replacement}",
            )

            # A known checkpoint term that has no sufficiently similar KB row
            # must abstain rather than emit a fabricated conversational reply.
            abstained = run(ask_command(chat, checkpoint, knowledge, "tarim", minimum_similarity="0.99"))
            require("grounded: false" in abstained and "answer:" not in abstained, f"ungrounded query did not abstain\n{abstained}")

            unknown = rejected(ask_command(chat, checkpoint, knowledge, "bilinmeyen"))
            require("too few active checkpoint vocabulary tokens" in unknown, f"unknown prompt was not rejected\n{unknown}")

            oversized_prompt = "okul ders" + " " * (1025 - len("okul ders"))
            oversized_output = rejected(ask_command(chat, checkpoint, knowledge, oversized_prompt))
            require("Usage:" in oversized_output, f"oversized command prompt was silently truncated\n{oversized_output}")
            oversized_repl = rejected(
                [
                    str(chat),
                    "repl",
                    str(checkpoint),
                    str(knowledge),
                    "--max-entries",
                    "8",
                    "--history-turns",
                    "2",
                ],
                input_text=oversized_prompt + "\n",
            )
            require("bounded byte contract" in oversized_repl, f"oversized REPL record was silently truncated\n{oversized_repl}")

            malformed = workspace / "malformed.tsv"
            malformed.write_text("row without a tab\n", encoding="ascii", newline="\n")
            malformed_output = rejected(ask_command(chat, checkpoint, malformed, "okul ders"))
            require("prompt TAB response pair" in malformed_output, f"malformed TSV was not rejected\n{malformed_output}")

            crlf_knowledge = workspace / "crlf.tsv"
            crlf_knowledge.write_bytes(b"okul ders\tokul ders kaynak cevabi\r\n")
            crlf_output = rejected(ask_command(chat, checkpoint, crlf_knowledge, "okul ders"))
            require("strict ASCII text contract" in crlf_output, f"CRLF knowledge record was not rejected\n{crlf_output}")

            repl = run(
                [
                    str(chat),
                    "repl",
                    str(checkpoint),
                    str(knowledge),
                    "--max-entries",
                    "8",
                    "--history-turns",
                    "2",
                    "--min-similarity",
                    "0.90",
                    "--min-margin",
                    "0.0",
                ],
                input_text="okul ders\n/quit\n",
            )
            require("knowledge entries: 3" in repl, f"REPL did not load the runtime knowledge base\n{repl}")
            require("grounded: true" in repl and "answer: okul ders kitap ogrenci hakkinda bilgi" in repl,
                    f"REPL did not use the same grounded retrieval path\n{repl}")

            # The checked-in documentation fixture must also load through the
            # compiled strict loader against its paired example checkpoint.
            fixture_checkpoint = workspace / "checked-in-fixture.vgai"
            fixture_corpus = ROOT / "examples" / "corpus" / "mini_turkish_ascii_pretrain.txt"
            fixture_training = run(train_command(pretrain, fixture_corpus, fixture_checkpoint))
            require("active tokens: 64" in fixture_training, "fixture corpus did not retain its expected vocabulary")
            fixture_answer = run(ask_command(chat, fixture_checkpoint, checked_in_fixture, "okul ders"))
            require("knowledge entries: 8" in fixture_answer, f"checked-in fixture did not load all rows\n{fixture_answer}")
            require("grounded: true" in fixture_answer, f"checked-in fixture did not ground its exact prompt\n{fixture_answer}")
            require(
                "answer: okul ders kitap ogrenci ogretmen ile ilgili bilgi" in fixture_answer,
                f"checked-in fixture response was not emitted verbatim\n{fixture_answer}",
            )
    except (AssertionError, OSError) as error:
        print(f"dialogue retrieval verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("dialogue retrieval verification: PASS (runtime TSV answers, fixture, grounding, abstention, strict input, REPL)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
