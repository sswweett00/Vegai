#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the Turkish corpus builder + epoch-1 runner contracts locally.

Builds a tiny self-authored MediaWiki XML fixture (MIT, no external data),
runs the real corpus builder twice, and checks: filter funnel accounting,
document-level split stability, exact dedupe, output hashes recomputed from
disk, run-to-run determinism, record bounds for the native BPE trainer, and
the runner's scale-projection/output-parsing pure functions.  It does not
need the Fortran binary; set VEGAI_NEURAL_BIN to additionally execute a
micro end-to-end native run when available.
"""
from __future__ import annotations

import bz2
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent

GOOD_PARAGRAPH = (
    "Trabzon, Karadeniz kıyısında yer alan ve tarihi ticaret yolları üzerinde bulunan bir liman şehridir. "
    "Şehrin ekonomisi uzun yıllar boyunca deniz ticareti, fındık üretimi ve balıkçılık ile şekillenmiştir. "
    "Bölgede yağışların yıl boyunca düzenli dağılması, tarım alanlarının verimliliğini artırır ve ormanların "
    "yoğunluğunu korur. Tarihi yapılar arasında surlar, köprüler ve eski konaklar bulunur; bu yapılar şehrin "
    "farklı dönemlerine ait mimari özellikleri birlikte gösterir. Üniversite ve araştırma kurumları, genç "
    "nüfusun eğitim olanaklarını genişletir ve bölgesel kalkınmaya katkı sağlar."
)
SECOND_PARAGRAPH = (
    "Deniz seviyesinden yükselen tepeler, kıyı şeridi boyunca dar vadiler oluşturur. Bu vadilerde akan dereler, "
    "yerleşimlerin su gereksinimini karşılar ve tarım için uygun mikro iklimler yaratır. Kış aylarında iç "
    "kesimlerde kar yağışı görülürken kıyı kesimlerinde hava genellikle daha ılımandır. Bölge mutfağında karalahana, "
    "hamsi ve mısır unu ile hazırlanan yemekler önemli yer tutar. Ulaşım ağı, sahil yolu ile iç bölgeleri birbirine "
    "bağlayan geçitler üzerinden sağlanır."
)


def wiki_page(page_id: int, title: str, text: str, namespace: int = 0) -> str:
    return (
        f"<page><title>{title}</title><ns>{namespace}</ns><id>{page_id}</id>"
        f"<revision><id>{page_id}1</id><text xml:space=\"preserve\">{text}</text></revision></page>"
    )


def build_fixture(path: Path) -> None:
    good = "{{Bilgi kutusu|ad=Örnek}}\n" + GOOD_PARAGRAPH + "\n\n" + SECOND_PARAGRAPH + \
        "\n\n== Kaynakça ==\n<ref>Örnek kaynak</ref>\n[[Kategori:Örnek]]"
    pages = [
        wiki_page(101, "Örnek Şehir", good),
        wiki_page(102, "Kopya Şehir", good),  # exact duplicate text -> duplicate document
        wiki_page(103, "Kısa Sayfa", "Kısa bir metin."),
        wiki_page(104, "İletişim", GOOD_PARAGRAPH + " Bize ornek@ornek.org adresinden ulaşın. " + SECOND_PARAGRAPH),
        wiki_page(105, "Sayılar", " ".join(str(i) for i in range(600))),
        wiki_page(106, "Tartışma:Örnek", good, namespace=1),
        wiki_page(107, "İkinci Şehir",
                  SECOND_PARAGRAPH + "\n\n" +
                  "Ankara, Türkiye'nin başkenti ve en kalabalık ikinci şehridir. " +
                  GOOD_PARAGRAPH.replace("Trabzon, Karadeniz kıyısında yer alan ve tarihi ticaret yolları üzerinde bulunan bir liman şehridir",
                                         "İç Anadolu'nun ortasında yer alan ve Cumhuriyet döneminde hızla büyüyen bir yönetim merkezidir")),
    ]
    xml = "<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.11/\">" + "".join(pages) + "</mediawiki>"
    path.write_bytes(bz2.compress(xml.encode("utf-8")))


def expect(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"ok - {label}")


def sha256_file(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            hasher.update(block)
            size += len(block)
    return size, hasher.hexdigest()


def run_builder(fixture: Path, output: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools/build_turkish_wiki_corpus.py"),
         "--wiki-xml-bz2", str(fixture), "--output-dir", str(output),
         "--retrieved-at", "fixture-2026-10-25", "--min-chars", "700",
         "--tokenizer-sample-bytes", "1500", "--train-byte-budget", "2000"],
        capture_output=True, text=True, check=False,
    )
    expect(completed.returncode == 0, f"builder succeeds (stderr: {completed.stderr[-400:]})")
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_neural_turkish_epoch1", ROOT / "tools/run_neural_turkish_epoch1.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vegai-99b-verify-") as temporary:
        tmp = Path(temporary)
        fixture = tmp / "fixture.xml.bz2"
        build_fixture(fixture)
        manifest_a = run_builder(fixture, tmp / "a")
        manifest_b = run_builder(fixture, tmp / "b")

        expect(manifest_a == manifest_b, "two builder runs are manifest-identical (deterministic)")
        funnel = manifest_a["funnel"]
        expect(funnel["source_documents"] == 6, "namespace-1 page excluded before funnel accounting")
        expect(funnel["duplicate_documents"] == 1, "exact duplicate document removed once")
        expect(funnel["rejected"].get("too_short", 0) >= 1, "short document rejected")
        expect(funnel["rejected"].get("direct_pii_pattern", 0) == 1, "email document rejected")
        expect(sum(1 for key in ("digit_heavy", "low_alpha_ratio", "word_length_anomaly", "function_word_anomaly")
                   if funnel["rejected"].get(key)) >= 1, "digit page rejected by a numeric quality filter")
        expect(funnel["accepted_documents"] >= 2, "good Turkish documents accepted")
        expect(funnel["duplicate_paragraphs"] >= 1, "repeated paragraph deduped across documents")

        for key, meta in manifest_a["outputs"].items():
            size, digest = sha256_file(tmp / "a" / meta["path"])
            expect(size == meta["bytes"] and digest == meta["sha256"], f"manifest hash matches disk for {key}")
            if meta["records"]:
                records = (tmp / "a" / meta["path"]).read_bytes().split(b"\n")
                expect(all(len(record) < 8192 for record in records), f"{key} records fit native BPE bound")
        expect(manifest_a["outputs"]["train_epoch"]["bytes"] <= 2000, "epoch slice respects byte budget")
        expect(manifest_a["outputs"]["tokenizer_sample"]["bytes"] <= 1500, "tokenizer sample respects byte budget")
        expect(manifest_a["target"]["program_token_target"] == 99_000_000_000, "99B target recorded as target only")

        runner = load_runner_module()
        projection = runner.project_99b(1_000_000.0, 5_000_000)
        expect(abs(projection["projected_single_host_seconds_for_99b"] - 99_000.0) < 1e-9,
               "99B projection arithmetic is tokens / measured rate")
        expect(projection["is_projection_not_a_result"] is True, "projection explicitly labelled non-result")
        parsed = runner.parse_prefixed_json('noise\nNEURAL_EVALUATE_JSON: {"mean_nll": 2.5}\n', "NEURAL_EVALUATE_JSON:")
        expect(parsed["mean_nll"] == 2.5, "prefixed JSON parsing works")
        generation = runner.extract_generation("x\nNEURAL_GENERATION_BEGIN\nmerhaba dünya\nNEURAL_GENERATION_END\n")
        expect(generation == "merhaba dünya", "generation marker extraction is verbatim")

        binary = os.environ.get("VEGAI_NEURAL_BIN", "")
        if binary:
            run_dir = tmp / "run"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
                 "--neural-bin", binary, "--corpus-dir", str(tmp / "a"), "--run-dir", str(run_dir),
                 "--vocab", "300", "--width", "16", "--heads", "2", "--window", "8", "--ffn", "32",
                 "--warmup-updates", "10", "--generation-tokens", "4"],
                capture_output=True, text=True, check=False, timeout=1200,
            )
            expect(completed.returncode == 0, f"micro native epoch-1 run succeeds (stderr: {completed.stderr[-600:]})")
            evidence = json.loads((run_dir / "evidence.json").read_text(encoding="utf-8"))
            expect(evidence["exact_epochs"] == 1, "micro run is exactly one epoch")
            expect(len(evidence["generations"]) == len(runner.DEFAULT_PROMPTS), "prompt suite completed")

            chunk_run_dir = tmp / "run-chunks"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
                 "--neural-bin", binary, "--corpus-dir", str(tmp / "a"), "--run-dir", str(chunk_run_dir),
                 "--vocab", "300", "--width", "16", "--heads", "2", "--window", "8", "--ffn", "32",
                 "--warmup-updates", "10", "--generation-tokens", "4", "--chunks", "2"],
                capture_output=True, text=True, check=False, timeout=1200,
            )
            expect(completed.returncode == 0,
                   f"chunked micro epoch-1 run succeeds (stderr: {completed.stderr[-600:]})")
            chunk_evidence = json.loads((chunk_run_dir / "evidence.json").read_text(encoding="utf-8"))
            expect(chunk_evidence["exact_epochs"] == 1, "chunked micro run is exactly one epoch")
            expect(len(chunk_evidence["training_chunks"]) >= 1, "chunked run reports its chunks")
            expect(chunk_evidence["training_epoch"]["prediction_tokens"]
                   == evidence["training_epoch"]["prediction_tokens"],
                   "chunked and single-call modes consume identical prediction tokens")
            expect(chunk_evidence["training_epoch"]["source_records_completed"]
                   == chunk_evidence["corpus"]["manifest"]["outputs"]["train_epoch"]["records"],
                   "chunked run consumes every epoch source record exactly once")
        else:
            print("skip - VEGAI_NEURAL_BIN unset; native micro run not executed")
    print("verify_neural_turkish_epoch1: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
