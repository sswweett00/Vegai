#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the repository-authored Turkish ASCII HDC pretraining fixture.

Without --pretrain-bin this validates corpus composition and its checked-in
manifest. With a Vegai CLI binary it additionally runs deterministic train,
checkpoint, inspect, and resume integration checks in a temporary directory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from corpus_manifest import (
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_TOKENS_PER_LINE,
    DEFAULT_TOKEN_LENGTH,
    scan_corpus,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPOSITORY_ROOT / "examples/corpus/mini_turkish_ascii_pretrain.txt"
MANIFEST_PATH = REPOSITORY_ROOT / "examples/corpus/mini_turkish_ascii_pretrain.manifest.json"

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

EXPECTED_SHA256 = "886f754c1cf65676c5d35837cc323b9c14f467a5b8e7f0be7db2097328faa362"
EXPECTED_BYTES = 6544
EXPECTED_RECORDS = 128
EXPECTED_MAXIMUM_RECORD_BYTES = 58
EXPECTED_UNIQUE_TOKENS = 64
EXPECTED_TOKEN_OCCURRENCES = 16
EXPECTED_TOKENS_PER_RECORD = 8
EXPECTED_RAW_TOKENS = 1024
EXPECTED_CONTEXT_PAIRS_PER_RECORD = 26
EXPECTED_CONTEXT_PAIRS_PER_EPOCH = EXPECTED_RECORDS * EXPECTED_CONTEXT_PAIRS_PER_RECORD


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def line_values(output: str, label: str) -> list[int]:
    pattern = re.compile(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", re.MULTILINE)
    return [int(match.group(1)) for match in pattern.finditer(output)]


def require_values(output: str, label: str, expected: list[int]) -> None:
    actual = line_values(output, label)
    require(actual == expected, f"{label!r}: expected {expected}, got {actual}\n{output}")


def verify_manifest_and_structure() -> None:
    require(CORPUS_PATH.is_file(), f"missing corpus fixture: {CORPUS_PATH}")
    require(MANIFEST_PATH.is_file(), f"missing corpus manifest: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    with CORPUS_PATH.open("rb") as corpus_file:
        digest, byte_count, stats = scan_corpus(
            corpus_file,
            DEFAULT_MAX_LINE_BYTES,
            DEFAULT_TOKEN_LENGTH,
            DEFAULT_MAX_TOKENS_PER_LINE,
        )

    require(manifest["schema"] == "vegai-corpus-manifest/v1", "unexpected manifest schema")
    require(
        manifest["declared_provenance"]["source"] == "Vegai repository-authored synthetic fixture"
        and manifest["declared_provenance"]["license"] == "MIT",
        "fixture provenance declaration is wrong",
    )
    require(digest == EXPECTED_SHA256, "fixture content SHA-256 changed unexpectedly")
    require(byte_count == EXPECTED_BYTES, "fixture byte count changed unexpectedly")
    require(manifest["input"]["sha256"] == digest, "checked-in manifest SHA-256 is stale")
    require(manifest["input"]["bytes"] == byte_count, "checked-in manifest byte count is stale")
    require(
        manifest["input"]["physical_lf_records"] == stats.records == EXPECTED_RECORDS,
        "physical record count is wrong",
    )
    require(stats.maximum_record_bytes == EXPECTED_MAXIMUM_RECORD_BYTES, "maximum record size is wrong")
    expected_preflight = {
        "accepted_tokens": EXPECTED_RAW_TOKENS,
        "crlf_records": 0,
        "dropped_tokens": 0,
        "maximum_record_bytes": EXPECTED_MAXIMUM_RECORD_BYTES,
        "non_ascii_bytes": 0,
        "overlong_records": 0,
        "records_with_tokenizer_truncation": 0,
    }
    require(manifest["preflight"] == expected_preflight, "checked-in manifest preflight is stale")
    require(stats.accepted_tokens == EXPECTED_RAW_TOKENS, "accepted token count is wrong")
    require(stats.dropped_tokens == 0, "fixture must not drop tokens")
    require(stats.records_with_tokenizer_truncation == 0, "fixture must not truncate tokens")
    require(stats.overlong_records == 0, "fixture must not contain overlong records")
    require(stats.non_ascii_bytes == 0, "fixture must remain ASCII-safe")
    require(stats.crlf_records == 0, "fixture must use LF records")

    lines = CORPUS_PATH.read_text(encoding="ascii").splitlines()
    require(len(lines) == EXPECTED_RECORDS, "text line count is wrong")
    all_tokens: list[str] = []
    for cluster_index, cluster in enumerate(CLUSTERS):
        cluster_lines = lines[cluster_index * 16 : (cluster_index + 1) * 16]
        require(len(cluster_lines) == 16, f"cluster {cluster_index} line count is wrong")
        for line in cluster_lines:
            tokens = line.split()
            require(len(tokens) == EXPECTED_TOKENS_PER_RECORD, "record token cardinality is wrong")
            require(set(tokens) == set(cluster), "record crosses or omits a semantic cluster")
            all_tokens.extend(tokens)
        cluster_frequencies = Counter(token for line in cluster_lines for token in line.split())
        require(
            set(cluster_frequencies) == set(cluster)
            and all(count == EXPECTED_TOKEN_OCCURRENCES for count in cluster_frequencies.values()),
            f"cluster {cluster_index} frequencies are not balanced",
        )

    frequencies = Counter(all_tokens)
    require(len(frequencies) == EXPECTED_UNIQUE_TOKENS, "unique vocabulary size is wrong")
    require(len(all_tokens) == EXPECTED_RAW_TOKENS, "raw token count is wrong")
    require(
        all(count == EXPECTED_TOKEN_OCCURRENCES for count in frequencies.values()),
        "global token frequencies are not balanced",
    )


def run_command(command: list[str], environment: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return result.stdout


def run_rejected_command(command: list[str], environment: dict[str, str]) -> str:
    """Require the compiled CLI to fail rather than testing a Python surrogate."""
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(command)}\n{result.stdout}")
    return result.stdout


def train_command(binary: Path, checkpoint: Path) -> list[str]:
    return [
        str(binary),
        "train",
        str(CORPUS_PATH),
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
        "2",
        "--initial-plasticity",
        "25",
        "--minimum-plasticity",
        "1",
        "--decay-events",
        "8",
        "--strict",
        "--query",
        "okul",
    ]


def verify_train_output(output: str) -> None:
    require("phase: vocabulary" in output, "training output omitted vocabulary phase")
    require("phase: streaming HDC context learning" in output, "training output omitted learning phase")
    require_values(output, "corpus lines", [EXPECTED_RECORDS, EXPECTED_RECORDS * 2])
    require_values(output, "observed tokens", [EXPECTED_RAW_TOKENS])
    require_values(output, "unique tokens", [EXPECTED_UNIQUE_TOKENS])
    require_values(output, "active tokens", [EXPECTED_UNIQUE_TOKENS, EXPECTED_RAW_TOKENS * 2])
    require_values(output, "completed epochs", [2])
    require_values(output, "raw tokens", [EXPECTED_RAW_TOKENS * 2])
    require_values(output, "out-of-vocabulary tokens", [0])
    require_values(output, "ordered context pairs", [EXPECTED_CONTEXT_PAIRS_PER_EPOCH * 2])
    require_values(output, "prototype updates", [EXPECTED_RAW_TOKENS * 2])
    require_values(output, "tokenizer truncations", [0, 0])
    require_values(output, "non-ASCII input bytes", [0, 0])
    require_values(output, "dropped overlong records", [0, 0])
    require("query: okul" in output, "training query did not execute")
    require("nearest semantic neighbour:" in output, "training query did not return a neighbour")


def verify_cli(binary: Path) -> None:
    require(binary.is_file(), f"pretraining executable does not exist: {binary}")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"

    with tempfile.TemporaryDirectory(prefix="vegai-example-pretrain-") as temp_directory:
        temporary_path = Path(temp_directory)
        checkpoint_a = temporary_path / "example-a.vgai"
        checkpoint_b = temporary_path / "example-b.vgai"

        first_output = run_command(train_command(binary, checkpoint_a), environment)
        second_output = run_command(train_command(binary, checkpoint_b), environment)
        verify_train_output(first_output)
        verify_train_output(second_output)
        require(checkpoint_a.is_file() and checkpoint_b.is_file(), "training did not write checkpoints")
        require(
            checkpoint_a.read_bytes() == checkpoint_b.read_bytes(),
            "identical corpus/config runs produced different checkpoints",
        )

        inspect_output = run_command(
            [str(binary), "inspect", str(checkpoint_a), "--query", "teknoloji"], environment
        )
        require("Vegai HDC pretraining checkpoint" in inspect_output, "checkpoint inspect failed")
        require_values(inspect_output, "unique tokens", [EXPECTED_UNIQUE_TOKENS])
        require_values(inspect_output, "active tokens", [EXPECTED_UNIQUE_TOKENS])
        require_values(inspect_output, "min frequency", [8])
        require_values(inspect_output, "context window", [2])
        require_values(inspect_output, "positional shift", [41])
        require("context mode: hierarchical" in inspect_output, "checkpoint did not preserve default hierarchical context mode")
        require_values(inspect_output, "completed epochs", [2])
        require_values(inspect_output, "total prototype updates", [EXPECTED_RAW_TOKENS * 2])
        require("query: teknoloji" in inspect_output, "inspect query did not execute")

        # Exercise the real CLI's checkpoint checksum boundary, not only the
        # in-process Fortran API: flipping a trailing checksum bit must reject
        # inspection before it could be trusted as a resume base.
        tampered_checkpoint = temporary_path / "example-tampered.vgai"
        tampered_bytes = checkpoint_b.read_bytes()
        require(len(tampered_bytes) > 0, "checkpoint is unexpectedly empty before tamper test")
        tampered_checkpoint.write_bytes(tampered_bytes[:-1] + bytes((tampered_bytes[-1] ^ 1,)))
        tampered_output = run_rejected_command(
            [str(binary), "inspect", str(tampered_checkpoint)], environment
        )
        require(
            "checkpoint schema or integrity failure" in tampered_output,
            f"tampered checkpoint was not rejected as an integrity failure\n{tampered_output}",
        )

        resume_output = run_command(
            [
                str(binary),
                "resume",
                str(CORPUS_PATH),
                str(checkpoint_a),
                "--epochs",
                "1",
                "--strict",
                "--query",
                "okul",
            ],
            environment,
        )
        require_values(resume_output, "completed epochs", [3])
        require_values(resume_output, "corpus lines", [EXPECTED_RECORDS])
        require_values(resume_output, "raw tokens", [EXPECTED_RAW_TOKENS])
        require_values(resume_output, "active tokens", [EXPECTED_RAW_TOKENS])
        require_values(resume_output, "out-of-vocabulary tokens", [0])
        require_values(resume_output, "ordered context pairs", [EXPECTED_CONTEXT_PAIRS_PER_EPOCH])
        require_values(resume_output, "prototype updates", [EXPECTED_RAW_TOKENS])
        require_values(resume_output, "tokenizer truncations", [0])
        require_values(resume_output, "non-ASCII input bytes", [0])
        require_values(resume_output, "dropped overlong records", [0])
        require("query: okul" in resume_output, "resume query did not execute")

        resumed_inspect_output = run_command([str(binary), "inspect", str(checkpoint_a)], environment)
        require_values(resumed_inspect_output, "completed epochs", [3])
        require_values(
            resumed_inspect_output,
            "total prototype updates",
            [EXPECTED_RAW_TOKENS * 3],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrain-bin",
        type=Path,
        help="optional vegai_pretrain binary for full CLI/checkpoint integration coverage",
    )
    arguments = parser.parse_args()

    try:
        verify_manifest_and_structure()
        if arguments.pretrain_bin is not None:
            verify_cli(arguments.pretrain_bin.resolve())
    except (AssertionError, OSError, json.JSONDecodeError) as error:
        print(f"Example corpus verification: FAIL: {error}", file=sys.stderr)
        return 1

    if arguments.pretrain_bin is None:
        print("Example corpus structural verification: PASS")
    else:
        print("Example corpus + HDC pretraining integration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
