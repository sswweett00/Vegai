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
from unittest.mock import patch

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


def page_id_for_split_bucket(title: str, bucket: int) -> int:
    modulus = 200
    for page_id in range(1, 1_000_000):
        identity = f"vegai-trwiki-split-v2\0{page_id}\0{title}".encode("utf-8")
        actual = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % modulus
        if actual == bucket:
            return page_id
    raise AssertionError(f"could not find a fixture page ID for split bucket {bucket}")


def build_fixture(path: Path) -> None:
    good = "{{Bilgi kutusu|ad=Örnek}}\n" + GOOD_PARAGRAPH + "\n\n" + SECOND_PARAGRAPH + \
        "\n\n== Kaynakça ==\n<ref>Örnek kaynak</ref>\n[[Kategori:Örnek]]"
    train_title = "Örnek Şehir"
    validation_title = "İkinci Şehir"
    test_title = "Üçüncü Şehir"
    pages = [
        wiki_page(page_id_for_split_bucket(train_title, 2), train_title, good),
        wiki_page(page_id_for_split_bucket(validation_title, 1), validation_title,
                  SECOND_PARAGRAPH + "\n\n" +
                  "Ankara, Türkiye'nin başkenti ve en kalabalık ikinci şehridir. " +
                  GOOD_PARAGRAPH.replace("Trabzon, Karadeniz kıyısında yer alan ve tarihi ticaret yolları üzerinde bulunan bir liman şehridir",
                                         "İç Anadolu'nun ortasında yer alan ve Cumhuriyet döneminde hızla büyüyen bir yönetim merkezidir")),
        wiki_page(page_id_for_split_bucket(test_title, 0), test_title,
                  GOOD_PARAGRAPH.replace("Trabzon", "Giresun", 1) + "\n\n" +
                  SECOND_PARAGRAPH.replace("Kış aylarında", "Bahar aylarında")),
        wiki_page(102, "Kopya Şehir", good),  # exact duplicate text -> duplicate document
        wiki_page(103, "Kısa Sayfa", "Kısa bir metin."),
        wiki_page(104, "İletişim", GOOD_PARAGRAPH + " Bize ornek@ornek.org adresinden ulaşın. " + SECOND_PARAGRAPH),
        wiki_page(105, "Sayılar", " ".join(str(i) for i in range(600))),
        wiki_page(106, "Tartışma:Örnek", good, namespace=1),
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


def run_builder(fixture: Path, output: Path, max_docs: int = 0) -> dict:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools/build_turkish_wiki_corpus.py"),
         "--wiki-xml-bz2", str(fixture), "--output-dir", str(output),
         "--retrieved-at", "fixture-2026-10-25", "--min-chars", "700", "--max-docs", str(max_docs),
         "--tokenizer-sample-bytes", "1500", "--train-byte-budget", "2000"],
        capture_output=True, text=True, check=False,
    )
    expect(completed.returncode == 0, f"builder succeeds (stderr: {completed.stderr[-400:]})")
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_neural_turkish_epoch1", ROOT / "tools/run_neural_turkish_epoch1.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_corpus_builder_module():
    spec = importlib.util.spec_from_file_location("build_turkish_wiki_corpus", ROOT / "tools/build_turkish_wiki_corpus.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vegai-99b-verify-") as temporary:
        tmp = Path(temporary)
        fixture = tmp / "fixture.xml.bz2"
        build_fixture(fixture)
        manifest_a = run_builder(fixture, tmp / "a")
        manifest_b = run_builder(fixture, tmp / "b")
        capped_manifest = run_builder(fixture, tmp / "capped", max_docs=3)

        expect(manifest_a == manifest_b, "two builder runs are manifest-identical (deterministic)")
        expect(capped_manifest["funnel"]["source_documents"] == 3
               and capped_manifest["funnel"]["accepted_documents"] == 3,
               "document cap reports exactly the three documents it processes")
        expect(manifest_a["schema"] == "vegai-turkish-wiki-corpus/v4"
               and "evenly-spaced whole-record sample" in manifest_a["byte_budget_selection"],
               "manifest versions and attests the stratified byte-budget sampling contract")
        undersized_dir = tmp / "undersized"
        undersized = subprocess.run(
            [sys.executable, str(ROOT / "tools/build_turkish_wiki_corpus.py"),
             "--wiki-xml-bz2", str(fixture), "--output-dir", str(undersized_dir),
             "--retrieved-at", "fixture-undersized", "--min-chars", "700", "--max-docs", "1"],
            capture_output=True, text=True, check=False,
        )
        expect(undersized.returncode != 0 and "empty validation or held-out test" in undersized.stderr,
               "builder rejects a corpus without validation and final-test records")
        expect(not (undersized_dir / "manifest.json").exists(),
               "failed split construction publishes no corpus manifest")
        funnel = manifest_a["funnel"]
        expect(funnel["source_documents"] == 7, "namespace-1 page excluded before funnel accounting")
        expect(funnel["duplicate_documents"] == 1, "exact duplicate document removed once")
        expect(funnel["rejected"].get("too_short", 0) >= 1, "short document rejected")
        expect(funnel["rejected"].get("direct_pii_pattern", 0) == 1, "email document rejected")
        expect(sum(1 for key in ("digit_heavy", "low_alpha_ratio", "word_length_anomaly", "function_word_anomaly")
                   if funnel["rejected"].get(key)) >= 1, "digit page rejected by a numeric quality filter")
        expect(funnel["accepted_documents"] == 3, "three unique good Turkish documents accepted")
        expect(funnel["train_documents"] >= 1 and funnel["validation_documents"] >= 1
               and funnel["heldout_documents"] >= 1, "train/validation/test splits are all non-empty and document-level")
        expect(manifest_a["outputs"]["validation"]["records"] > 0
               and manifest_a["outputs"]["heldout"]["records"] > 0,
               "validation and final held-out partitions contain records")
        expect(manifest_a["outputs"]["validation"]["sha256"] != manifest_a["outputs"]["heldout"]["sha256"],
               "validation and final held-out outputs are distinct")
        expect(manifest_a["split"]["rule"] ==
               "sha256(vegai-trwiki-split-v2|page_id|title) mod 200: 0 -> heldout_test, 1 -> validation, 2..199 -> train",
               "manifest records the stable three-way document split rule")
        expect(manifest_a["split"]["selection_uses"] == "validation only"
               and manifest_a["split"]["final_reporting_uses"] == "heldout test only",
               "manifest separates checkpoint selection from final test reporting")
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
        default_args = runner.build_parser().parse_args([
            "--neural-bin", "unused-neural-bin", "--corpus-dir", "unused-corpus", "--run-dir", "unused-run",
        ])
        runner.validate_arguments(default_args)
        expect((default_args.vocab, default_args.min_pair_frequency, default_args.width, default_args.heads,
                default_args.width // default_args.heads, default_args.window, default_args.ffn)
               == (4096, 8, 96, 4, 24, 64, 384),
               "default model shape matches requested vocab/heads/window/FFN profile")
        effective_workflow_args = runner.argparse.Namespace(**vars(default_args))
        effective_workflow_args.lr = 0.003
        effective_workflow_args.schedule = "cosine"
        effective_workflow_args.warmup_updates = 4000
        effective_workflow_args.total_updates = 52000000
        runner.validate_arguments(effective_workflow_args)
        expect(effective_workflow_args.window == 64 and effective_workflow_args.width // effective_workflow_args.heads == 24,
               "requested 64-window cosine configuration passes model and scheduler compatibility checks")
        builder = load_corpus_builder_module()
        streamed_pages = list(builder.iter_wiki_documents(fixture))
        expect(len(streamed_pages) == 7 and streamed_pages[0].title == "Örnek Şehir",
               "namespace-tolerant XML streaming retains complete pages while clearing parsed nodes")
        builder_defaults = builder.parse_args(["--output-dir", str(tmp / "builder-defaults")])
        expect(builder_defaults.tokenizer_sample_bytes == 3_000_000,
               "corpus builder default tokenizer sample matches requested three megabytes")
        stratified_source = tmp / "stratified-source.txt"
        stratified_destination = tmp / "stratified-sample.txt"
        stratified_source.write_bytes(b"".join(f"{index:04d}\n".encode("ascii") for index in range(100)))
        source_reads = 0
        original_path_open = Path.open

        def count_source_reads(path: Path, *args: object, **kwargs: object):
            nonlocal source_reads
            if path == stratified_source and args and args[0] == "rb":
                source_reads += 1
            return original_path_open(path, *args, **kwargs)

        with patch.object(Path, "open", count_source_reads):
            stratified_meta, second_sample_meta, full_copy_meta = builder.write_byte_budget_slices(
                stratified_source,
                [stratified_destination, tmp / "second-sample.txt", tmp / "full-copy.txt"],
                [100, 200, 0],
            )
        sampled_indices = [int(line) for line in stratified_destination.read_text(encoding="ascii").splitlines()]
        second_sample_indices = [int(line) for line in (tmp / "second-sample.txt").read_text(encoding="ascii").splitlines()]
        expect(stratified_meta["bytes"] == 100 and stratified_meta["records"] == 20
               and sampled_indices[0] > 0 and sampled_indices[-1] >= 95
               and all(b - a >= 4 for a, b in zip(sampled_indices, sampled_indices[1:])),
               "byte-budget samples are evenly distributed across the full source, not a prefix")
        expect(second_sample_meta["bytes"] == 200 and second_sample_meta["records"] == 40
               and full_copy_meta["bytes"] == 500 and full_copy_meta["records"] == 100
               and source_reads == 2,
               "multiple stratified budgets share one count pass and one output pass")
        expect(runner.validation_improves({"mean_nll": 1.9}, {"mean_nll": 2.0}),
               "strictly lower validation NLL selects a checkpoint")
        expect(not runner.validation_improves({"mean_nll": 2.0}, {"mean_nll": 2.0}),
               "validation ties preserve the earlier checkpoint")
        source_checkpoint = tmp / "source-checkpoint.bin"
        selected_checkpoint = tmp / "selected-checkpoint.bin"
        source_checkpoint.write_bytes(b"checkpoint-v1")
        selected_checkpoint.write_bytes(b"incumbent")
        runner.atomic_copy(source_checkpoint, selected_checkpoint)
        expect(selected_checkpoint.read_bytes() == b"checkpoint-v1",
               "selected checkpoint publication atomically replaces the incumbent")
        expect(not selected_checkpoint.with_name(selected_checkpoint.name + ".tmp").exists(),
               "atomic checkpoint publication leaves no temporary artifact")
        try:
            runner.atomic_copy(tmp / "missing-checkpoint.bin", selected_checkpoint)
        except OSError:
            pass
        else:
            raise AssertionError("missing source should abort checkpoint publication")
        expect(selected_checkpoint.read_bytes() == b"checkpoint-v1",
               "failed checkpoint publication preserves the previous selected artifact")
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
            expect(evidence["config"]["head_width"] == 8 and evidence["config"]["window"] == 8,
                   "evidence records effective attention-head width and local window")
            expect(evidence["config"]["train_byte_budget"] == 2000
                   and evidence["config"]["tokenizer_sample_bytes"] == 1500
                   and evidence["config"]["max_docs"] == 0,
                   "evidence records corpus-building caps and uncapped-document setting")
            expect(evidence["selected_epoch"] in (0, 1), "validation selects baseline or trained checkpoint")
            expect(evidence["baseline_validation"]["prediction_tokens"] > 0
                   and evidence["epoch_validation"]["prediction_tokens"] > 0,
                   "validation is scored before and after training")
            expect(evidence["heldout_test"]["prediction_tokens"] > 0,
                   "final held-out test is scored on the selected checkpoint")
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
