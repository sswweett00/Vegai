#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Contract verification for the audited Turkish dialogue corpus builder."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools" / "build_turkish_dialogue_retrieval_corpus.py"
PACKAGE_SCHEMA = "vegai-turkish-dialogue-retrieval-package/v1"
EVALUATION_SCHEMA = "vegai-dialogue-evaluation-case/v1"
REVISION = "0123456789abcdef0123456789abcdef01234567"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def record(conversation_id: str, split: str, messages: list[tuple[str, str]]) -> str:
    return json.dumps(
        {
            "conversation_id": conversation_id,
            "messages": [{"role": role, "content": content} for role, content in messages],
            "split": split,
            "synthetic": True,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"


def command(train: Path, validation: Path, test: Path, output: Path) -> list[str]:
    return [
        sys.executable, str(BUILDER),
        "--train-jsonl", str(train),
        "--validation-jsonl", str(validation),
        "--test-jsonl", str(test),
        "--output-dir", str(output),
        "--source-repository", "https://github.com/example/licensed-turkish-dialogue",
        "--source-revision", REVISION,
        "--source-license", "CC-BY-4.0",
        "--attribution", "Synthetic Turkish dialogue fixture, CC BY 4.0",
        "--source-nature", "synthetic-test-only",
    ]


def run(arguments: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != expected:
        raise AssertionError(f"builder returned {completed.returncode}, expected {expected}\n{completed.stdout}")
    return completed


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="vegai-turkish-dialogue-corpus-") as temporary_name:
            temporary = Path(temporary_name)
            train = temporary / "train.jsonl"
            validation = temporary / "validation.jsonl"
            test = temporary / "test.jsonl"
            train.write_text(
                record(
                    "train-001", "train",
                    [
                        ("user", "Öğrenci kaydı nasıl yapılır?"),
                        ("assistant", "Önce öğrenci işleri birimine başvurun."),
                        ("user", "Kimlik belgesi gerekli mi?"),
                        ("assistant", "Evet, geçerli kimliğinizi yanınıza alın."),
                    ],
                )
                + record(
                    "train-002", "train",
                    [
                        ("user", "Öğrenci kaydı nasıl yapılır?"),
                        ("assistant", "Önce öğrenci işleri birimine başvurun."),
                        ("user", "Ders programını nereden görürüm?"),
                        ("assistant", "Ders programını öğrenci sisteminden görebilirsiniz."),
                    ],
                ),
                encoding="utf-8", newline="\n",
            )
            validation.write_text(
                record(
                    "validation-001", "validation",
                    [
                        ("user", "Kütüphane kartı nereden alınır?"),
                        ("assistant", "Danışmaya kimliğinizle başvurabilirsiniz."),
                    ],
                ),
                encoding="utf-8", newline="\n",
            )
            test.write_text(
                record(
                    "test-001", "test",
                    [
                        ("user", "Otobüs saatlerini nasıl öğrenebilirim?"),
                        ("assistant", "Belediyenin uygulamasını kontrol edin."),
                    ],
                ),
                encoding="utf-8", newline="\n",
            )
            output = temporary / "package"
            completed = run(command(train, validation, test, output))
            summary = json.loads(completed.stdout)
            require(summary["schema"] == PACKAGE_SCHEMA and summary["status"] == "ready",
                    "builder did not emit a ready package summary")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            require(manifest["schema"] == PACKAGE_SCHEMA, "package manifest schema is wrong")
            counts = manifest["counts"]
            require(counts["train_pairs_raw"] == 4 and counts["train_knowledge_rows_unique_canonical_prompt"] == 3,
                    f"canonical duplicate policy is wrong: {counts}")
            require(counts["train_identical_canonical_prompt_duplicates_dropped"] == 1,
                    "identical train prompt/response was not reported as deduplicated")
            require(counts["train_exact_retrieval_contract_cases"] == 3
                    and counts["train_exact_retrieval_evidence_sample_cases"] == 3
                    and counts["test_source_disjoint_abstention_evidence_sample_cases"] == 1,
                    "non-held-out/source-disjoint bounded evidence sample counts are wrong")
            require(counts["validation_abstention_cases"] == 1 and counts["test_abstention_cases"] == 1,
                    "source-disjoint abstention cases are wrong")
            knowledge = (output / "knowledge_train.tsv").read_text(encoding="utf-8")
            require("Öğrenci kaydı nasıl yapılır?\tÖnce öğrenci işleri birimine başvurun." in knowledge,
                    "UTF-8 literal response row was not preserved")
            projected = (output / "pretrain_train_ascii.txt").read_bytes()
            require(projected.isascii() and b"ogrenci kaydi nasil yapilir" in projected,
                    "native ASCII training projection is not Turkish folded")
            exact_cases = [json.loads(line) for line in
                           (output / "train_exact_retrieval_contract.jsonl").read_text(encoding="utf-8").splitlines()]
            require(len(exact_cases) == 3 and all(case["prompt"] == case["expected_prompt"] for case in exact_cases),
                    "train exact retrieval package contract is not explicitly scoped")
            native_exact = (output / "train_exact_retrieval_contract.tsv").read_text(encoding="utf-8").splitlines()
            native_validation = (output / "validation_source_disjoint_abstention.tsv").read_text(encoding="utf-8").splitlines()
            native_test = (output / "test_source_disjoint_abstention.tsv").read_text(encoding="utf-8").splitlines()
            require(len(native_exact) == 3 and all(row.count("\t") == 2 and row.rsplit("\t", 1)[1]
                    for row in native_exact),
                    "native exact-evaluation TSV does not preserve explicit expected prompts")
            require(len(native_validation) == 1 and len(native_test) == 1
                    and native_validation[0].count("\t") == 2 and native_test[0].count("\t") == 2
                    and native_validation[0].endswith("\t") and native_test[0].endswith("\t"),
                    "native source-disjoint TSV does not preserve empty expected-abstention fields")
            for identity_name in (
                "train_exact_retrieval_contract_native_tsv",
                "validation_source_disjoint_abstention_native_tsv",
                "test_source_disjoint_abstention_native_tsv",
            ):
                require(identity_name in manifest["outputs"],
                        f"manifest lacks native batch evaluation output identity: {identity_name}")
            exact_sample = [json.loads(line) for line in
                            (output / "train_exact_retrieval_evidence_sample.jsonl").read_text(encoding="utf-8").splitlines()]
            require(len(exact_sample) == 3 and {case["id"] for case in exact_sample} == {case["id"] for case in exact_cases},
                    "hash-ranked exact evidence sample is not a deterministic complete small-set subset")
            validation_cases = [json.loads(line) for line in
                                (output / "validation_source_disjoint_abstention.jsonl").read_text(encoding="utf-8").splitlines()]
            require(len(validation_cases) == 1 and validation_cases[0]["schema_version"] == EVALUATION_SCHEMA
                    and validation_cases[0]["expected_prompt"] is None,
                    "validation output does not make the abstention-only label explicit")
            require(manifest["split_isolation"]["all_source_conversation_ids_unique_across_train_validation_test"] is True,
                    "manifest does not bind source split isolation")
            require("generation_claim" in manifest["native_runtime_contract"]
                    and manifest["native_runtime_contract"]["generation_claim"] == "none; this package supports grounded retrieval only",
                    "builder overclaimed generative capability")

            # An existing package is immutable: rerunning must not silently
            # replace hashes/data after an operator has reviewed the manifest.
            rerun = run(command(train, validation, test, output), expected=2)
            require("refuse to overwrite immutable package" in rerun.stdout,
                    "builder overwrote an existing immutable package")

            invalid = temporary / "invalid-test.jsonl"
            invalid.write_text(
                record("train-001", "test", [("user", "aynı kimlik"), ("assistant", "yanıt")]),
                encoding="utf-8", newline="\n",
            )
            rejected = run(command(train, validation, invalid, temporary / "invalid-package"), expected=2)
            require("conversation_id crosses source splits" in rejected.stdout,
                    "cross-split source identity reuse was accepted")
    except (AssertionError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"turkish dialogue corpus verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("turkish dialogue corpus verification: PASS (UTF-8 rows, native projection, provenance, split isolation, immutable package)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
