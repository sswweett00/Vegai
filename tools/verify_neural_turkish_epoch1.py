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
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tracemalloc
from unittest.mock import patch

import build_turkish_wiki_corpus as corpus_builder

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
        # SHA-256 split bucket zero: exercise genuine nonempty held-out data.
        wiki_page(283, "Deneme Kenti", GOOD_PARAGRAPH.replace("Trabzon", "Eskişehir") + "\n\n" +
                  SECOND_PARAGRAPH.replace("Deniz seviyesinden", "Dağların eteklerinden")),
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


def verify_checkpoint_cli_overrides(binary: str, tokenizer: Path, corpus: Path, root: Path) -> None:
    """Exercise resume/chunk paths against identical saved checkpoint bytes."""
    shape = ["--width", "16", "--heads", "2", "--window", "8", "--ffn", "32"]
    saved = [
        *shape, "--seed", "47", "--optimizer", "adamw", "--lr", "0.001",
        "--schedule", "invsqrt", "--warmup-updates", "7", "--total-updates", "50",
        "--min-lr-ratio", "0.2", "--beta1", "0.8", "--beta2", "0.95", "--adam-eps", "1e-7",
        "--grad-clip", "3.0", "--embedding-limit", "7.0", "--label-smoothing", "0.03",
        "--gradient-accumulation", "4", "--weight-decay", "0.006", "--max-tokens-per-record", "1024",
    ]
    baseline = root / "checkpoint-base.vgnr"

    def native(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([binary, *arguments], capture_output=True, text=True, check=False, timeout=120)

    result = native("init-checkpoint", str(tokenizer), str(baseline), *saved)
    expect(result.returncode == 0, f"non-default checkpoint initializes (stderr: {result.stderr[-400:]})")
    tokenizer_before = tokenizer.read_bytes()
    os.utime(tokenizer, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    tokenizer_mtime = tokenizer.stat().st_mtime_ns

    def compare(name: str, command: str, left_options: list[str], right_options: list[str]) -> None:
        left, right = root / f"{name}-left.vgnr", root / f"{name}-right.vgnr"
        shutil.copyfile(baseline, left)
        shutil.copyfile(baseline, right)
        prefix = [command, str(corpus)]
        left_result = native(*prefix, str(left), str(tokenizer), "1", *left_options)
        right_result = native(*prefix, str(right), str(tokenizer), "1", *right_options)
        expect(left_result.returncode == 0 and right_result.returncode == 0,
               f"{name} CLI calls succeed (stderr: {(left_result.stderr + right_result.stderr)[-400:]})")
        expect(left.read_bytes() == right.read_bytes(), f"{name} checkpoint and optimizer state are byte-identical")

    compare("chunk-shape-only", "train-chunk", [], ["--width", "16"])
    compare("resume-shape-only", "resume", [], ["--width", "16"])
    changes = ["--lr", "0.002", "--label-smoothing", "0.05", "--warmup-updates", "11",
               "--gradient-accumulation", "2"]
    compare("chunk-selected-overrides", "train-chunk", changes, [*saved, *changes])
    compare("chunk-sgd-default-decay", "train-chunk", ["--optimizer", "sgd"],
            ["--optimizer", "sgd", "--weight-decay", "0"])
    trained = root / "chunk-shape-only-left.vgnr"
    trained_before = trained.read_bytes()
    result = native("train-chunk", str(corpus), str(trained), str(tokenizer), "1", "--optimizer", "nova")
    expect(result.returncode != 0 and "cannot change the optimizer" in result.stderr
           and trained.read_bytes() == trained_before,
           "changing the optimizer on a trained checkpoint is rejected before replacement")
    invalid = root / "checkpoint-invalid.vgnr"
    shutil.copyfile(baseline, invalid)
    before = invalid.read_bytes()
    result = native("train-chunk", str(corpus), str(invalid), str(tokenizer), "1", "--width", "32")
    expect(result.returncode != 0 and invalid.read_bytes() == before,
           "a mismatched requested shape is rejected without replacing the checkpoint")
    result = native("train-chunk", str(corpus), str(invalid), str(tokenizer), "1",
                    "--optimizer", "sgd", "--weight-decay", "0.01")
    expect(result.returncode != 0 and invalid.read_bytes() == before,
           "invalid SGD decay override is rejected without replacing the checkpoint")
    too_large = root / "overflow-rate.vgnr"
    result = native("train", str(corpus), str(too_large), str(tokenizer), "1", *shape,
                    "--optimizer", "nova", "--lr", "1e300", "--weight-decay", "0")
    expect(result.returncode > 0 and "SIGFPE" not in result.stderr and not too_large.exists(),
           "native CLI rejects out-of-range real32 learning rates without FPE or checkpoint creation")
    trained_fresh = root / "new-train.vgnr"
    result = native("train", str(corpus), str(trained_fresh), str(tokenizer), "1", *shape)
    expect(result.returncode == 0, "new training accepts the prebuilt tokenizer")
    expect(tokenizer.read_bytes() == tokenizer_before and tokenizer.stat().st_mtime_ns == tokenizer_mtime,
           "train/resume/chunk never rewrite an unchanged prebuilt tokenizer")
    new_tokenizer, new_checkpoint = root / "fallback-new.bpe", root / "fallback-new.vgnr"
    result = native("train-chunk", str(corpus), str(new_checkpoint), str(new_tokenizer), "1", *shape)
    expect(result.returncode == 0 and new_tokenizer.is_file() and new_checkpoint.is_file(),
           "a missing tokenizer is still initialized and saved for a new chunked run")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vegai-99b-verify-") as temporary:
        tmp = Path(temporary)
        fixture = tmp / "fixture.xml.bz2"
        build_fixture(fixture)
        manifest_a = run_builder(fixture, tmp / "a")
        manifest_b = run_builder(fixture, tmp / "b")
        builder_command = [sys.executable, str(ROOT / "tools/build_turkish_wiki_corpus.py"),
                           "--wiki-xml-bz2", str(fixture)]
        corpus_files = ("train.txt", "heldout.txt", "train_epoch.txt", "tokenizer_sample.txt", "manifest.json")
        before = {name: (tmp / "a" / name).read_bytes() for name in corpus_files}
        retry = subprocess.run(builder_command + ["--output-dir", str(tmp / "a")],
                               capture_output=True, text=True, check=False)
        expect(retry.returncode == 2 and all((tmp / "a" / name).read_bytes() == content
                                            for name, content in before.items())
               and not (tmp / "a" / ".vegai-turkish-corpus.lock").exists(),
               "builder refuses to overwrite an existing corpus")
        locked = tmp / "locked-corpus"
        locked.mkdir()
        (locked / ".vegai-turkish-corpus.lock").write_text("other owner")
        result = subprocess.run(builder_command + ["--output-dir", str(locked)],
                                capture_output=True, text=True, check=False)
        expect(result.returncode == 2 and (locked / ".vegai-turkish-corpus.lock").read_text() == "other owner"
               and len(list(locked.iterdir())) == 1, "builder preserves an active or stale corpus lock")
        for index, bad in enumerate((["--max-docs", "-1"], ["--train-byte-budget", "-1"],
                                     ["--tokenizer-sample-bytes", "-1"], ["--max-record-chars", "0"])):
            destination = tmp / f"invalid-builder-{index}"
            result = subprocess.run(builder_command + ["--output-dir", str(destination), *bad],
                                    capture_output=True, text=True, check=False)
            expect(result.returncode == 2 and not destination.exists(), f"invalid corpus budget {bad} fails before I/O")
        capped = tmp / "capped"
        result = subprocess.run(builder_command + ["--output-dir", str(capped), "--max-docs", "2"],
                                capture_output=True, text=True, check=False)
        capped_funnel = json.loads((capped / "manifest.json").read_text())["funnel"] if result.returncode == 0 else {}
        expect(capped_funnel.get("source_documents") == 2
               and capped_funnel["accepted_documents"] + capped_funnel["rejected_total"]
                   + capped_funnel["duplicate_documents"] == 2,
               "source-document budget counts only documents actually processed")
        peaks = []
        for pages in (5_000, 50_000):
            large_fixture = tmp / f"{pages}.xml.bz2"
            with bz2.open(large_fixture, "wb") as stream:
                stream.write(b"<mediawiki>")
                for index in range(pages):
                    stream.write((f"<page><title>T{index}</title><ns>0</ns><id>{index}</id>"
                                  "<revision><text>data</text></revision></page>").encode())
                stream.write(b"</mediawiki>")
            tracemalloc.start()
            try:
                parsed = sum(1 for _ in corpus_builder.iter_wiki_documents(large_fixture))
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            expect(parsed == pages, f"streamed XML retains all {pages:,} pages")
            peaks.append(peak)
        expect(peaks[1] < peaks[0] + 2 * 1024 * 1024,
               "completed XML page elements do not accumulate with document count")
        text_source = tmp / "text-source"
        text_source.mkdir()
        (text_source / "long.txt").write_text("Başlık" * 1600 + "\n" + GOOD_PARAGRAPH + "\n\n" + SECOND_PARAGRAPH,
                                             encoding="utf-8")
        text_builder = [sys.executable, str(ROOT / "tools/build_turkish_wiki_corpus.py"), "--text-dir", str(text_source)]
        nested = subprocess.run(text_builder + ["--output-dir", str(text_source / "output")],
                                capture_output=True, text=True, check=False)
        expect(nested.returncode == 2 and not (text_source / "output").exists(),
               "recursive text input cannot include its own corpus output")
        too_long_dir = tmp / "too-long-corpus"
        too_long = subprocess.run(text_builder + ["--output-dir", str(too_long_dir)],
                                  capture_output=True, text=True, check=False)
        expect(too_long.returncode == 2 and "8191-byte limit" in too_long.stderr
               and not (too_long_dir / "manifest.json").exists(),
               "oversized native records fail instead of silently dropping document chunks")
        changing_dump = tmp / "changing.xml.bz2"
        shutil.copyfile(fixture, changing_dump)
        hashing_output = tmp / "changed-hash-corpus"
        hashing_args = corpus_builder.parse_args(["--wiki-xml-bz2", str(changing_dump),
                                                  "--output-dir", str(hashing_output)])
        real_hash = corpus_builder.sha256_file

        def mutate_during_hash(path):
            digest = real_hash(path)
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
            return digest

        with patch.object(corpus_builder, "sha256_file", mutate_during_hash):
            try:
                corpus_builder.build_corpus(hashing_args)
            except corpus_builder.CorpusError as error:
                expect("changed" in str(error), "dump mutation during hashing fails before corpus files")
            else:
                raise AssertionError("dump mutation during hashing was accepted")
        expect(not (hashing_output / "train.txt").exists()
               and not (hashing_output / ".vegai-turkish-corpus.lock").exists(),
               "hash-time mutation releases the lock before writing training data")
        changed_output = tmp / "changed-source-corpus"
        changed_args = corpus_builder.parse_args(["--wiki-xml-bz2", str(changing_dump),
                                                  "--output-dir", str(changed_output)])
        real_iter = corpus_builder.iter_wiki_documents

        def mutate_during_read(path):
            for index, document in enumerate(real_iter(path)):
                if index == 0:
                    info = path.stat()
                    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
                yield document

        with patch.object(corpus_builder, "iter_wiki_documents", mutate_during_read):
            try:
                corpus_builder.build_corpus(changed_args)
            except corpus_builder.CorpusError as error:
                expect("changed" in str(error), "dump mutation is reported explicitly")
            else:
                raise AssertionError("mutating source dump produced a successful corpus")
        expect(not (changed_output / "manifest.json").exists()
               and not (changed_output / ".vegai-turkish-corpus.lock").exists(),
               "changed source leaves no successful manifest and releases its lock")

        expect(manifest_a == manifest_b, "two builder runs are manifest-identical (deterministic)")
        funnel = manifest_a["funnel"]
        expect(funnel["source_documents"] == 7, "namespace-1 page excluded before funnel accounting")
        expect(funnel["heldout_documents"] >= 1 and manifest_a["outputs"]["heldout"]["records"] > 0,
               "fixture has a real nonempty held-out split")
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
        key = "vegai-trwiki-split-v1\0" + "283" + "\0" + "Deneme Kenti"
        expected_bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % 200
        expect(expected_bucket == 0 and "first 8 bytes" in manifest_a["split"]["rule"]
               and "original_title" in manifest_a["split"]["rule"],
               "split manifest describes the actual SHA-256 byte selection and input")

        runner = load_runner_module()
        text_fixture = tmp / "valid-text-source"
        text_fixture.mkdir()
        title = "Deneme Kenti"
        held_name = next(f"source-{index}.txt" for index in range(2_000)
                         if corpus_builder.is_heldout(corpus_builder.Document(f"source-{index}.txt", title, "")))
        train_name = next(f"source-{index}.txt" for index in range(2_000)
                          if not corpus_builder.is_heldout(corpus_builder.Document(f"source-{index}.txt", title, "")))
        (text_fixture / held_name).write_text(title + "\n" + GOOD_PARAGRAPH.replace("Trabzon", "Eskişehir") +
                                              "\n\n" + SECOND_PARAGRAPH.replace("Deniz seviyesinden", "Dağların eteklerinden"),
                                              encoding="utf-8")
        (text_fixture / train_name).write_text(title + "\n" + GOOD_PARAGRAPH + "\n\n" + SECOND_PARAGRAPH,
                                               encoding="utf-8")
        text_corpus = tmp / "valid-text-corpus"
        text_result = subprocess.run([sys.executable, str(ROOT / "tools/build_turkish_wiki_corpus.py"),
                                      "--text-dir", str(text_fixture), "--output-dir", str(text_corpus)],
                                     capture_output=True, text=True, check=False)
        expect(text_result.returncode == 0, f"operator text-directory corpus builds: {text_result.stderr[-200:]}")
        text_manifest = json.loads((text_corpus / "manifest.json").read_text(encoding="utf-8"))
        runner.corpus_inputs(text_manifest, text_corpus)
        expect(text_manifest["source"]["kind"] == "operator-text-dir" and text_manifest["source"]["sha256"] is None
               and text_manifest["outputs"]["heldout"]["records"] > 0,
               "operator text-directory provenance and held-out inputs validate")
        publication = tmp / "publication"
        publication.mkdir()
        real_replace = os.replace

        def fail_evidence_replace(source, destination):
            if Path(destination).name == "evidence.json":
                raise OSError("simulated evidence publication failure")
            real_replace(source, destination)

        with patch.object(runner.os, "replace", side_effect=fail_evidence_replace):
            try:
                runner.publish_results(publication, "complete results\n", '{"status":"completed"}\n')
            except OSError:
                pass
            else:
                raise AssertionError("simulated evidence publication failure was swallowed")
        expect(not list(publication.iterdir()), "failed evidence publication leaves no completion report or temp file")
        runner.publish_results(publication, "complete results\n", '{"status":"completed"}\n')
        expect((publication / "RESULTS.md").read_text() == "complete results\n"
               and json.loads((publication / "evidence.json").read_text())["status"] == "completed",
               "complete results and evidence publish successfully")
        projection = runner.project_99b(1_000_000.0, 5_000_000)
        expect(abs(projection["projected_single_host_seconds_for_99b"] - 99_000.0) < 1e-9,
               "99B projection arithmetic is tokens / measured rate")
        expect(projection["is_projection_not_a_result"] is True, "projection explicitly labelled non-result")
        extreme_projection = runner.project_99b(5e-324, 1)
        expect(extreme_projection["projection_out_of_float_range"]
               and "projected_single_host_seconds_for_99b" not in extreme_projection
               and "Infinity" not in json.dumps(extreme_projection, allow_nan=False),
               "unrepresentable duration projection stays strict JSON and is explicitly omitted")
        safe_cell = runner.markdown_cell("<tag>|`\nnext")
        expect(safe_cell == "&lt;tag&gt;&#124;&#96; next", "untrusted report text cannot break Markdown cells")
        parsed = runner.parse_prefixed_json('noise\nNEURAL_EVALUATE_JSON: {"mean_nll": 2.5}\n', "NEURAL_EVALUATE_JSON:")
        expect(parsed["mean_nll"] == 2.5, "prefixed JSON parsing works")
        generation = runner.extract_generation("x\nNEURAL_GENERATION_BEGIN\nmerhaba dünya\nNEURAL_GENERATION_END\n")
        expect(generation == "merhaba dünya", "generation marker extraction is verbatim")
        generated = "\nmerhaba\nNEURAL_GENERATION_END\nson satır\n"
        framed = "NEURAL_GENERATION_BEGIN\n" + generated + "\nNEURAL_GENERATION_END\n"
        expect(runner.extract_generation(framed) == generated,
               "generated newlines and embedded marker text are not truncated")
        for malformed in ('{"mean_nll":NaN}', '{"mean_nll":1e999}', '{"mean_nll":1e-9999}',
                          '{"x":' + '9' * 129 + '}', '{"x":1,"x":2}', '{"x":"\\ud800"}', '[]'):
            try:
                runner.parse_prefixed_json("NEURAL_EVALUATE_JSON: " + malformed, "NEURAL_EVALUATE_JSON:")
            except runner.RunError:
                pass
            else:
                raise AssertionError(f"ambiguous or nonfinite native JSON accepted: {malformed}")
        expect(runner.strict_json_object('{"x":0e-9999,"y":5e-324}', "finite JSON")["y"] > 0,
               "zero and representable subnormal JSON numbers remain valid")
        expect(True, "native JSON rejects duplicates, nonfinite/underflowed numbers, and non-object payloads")
        expect(runner.extract_generation("NEURAL_GENERATION_BEGIN\nabc\r\nNEURAL_GENERATION_END\n") == "abc\r",
               "generation framing preserves a generated trailing carriage return")
        good_eval = {"schema_version": "vegai-neural-evaluate/v1", "eligible_records": 2,
                     "prediction_tokens": 3, "updates_completed_at_checkpoint": 0,
                     "skipped_records": 0, "tokenizer_truncations": 0,
                     "mean_nll": 1.0, "perplexity": None, "top1_accuracy": 0.2, "top5_accuracy": 0.7}
        runner.validate_evaluation(good_eval)
        for changes in ({"eligible_records": 0}, {"prediction_tokens": 0},
                        {"prediction_tokens": 1}, {"tokenizer_truncations": 1},
                        {"mean_nll": float("inf")}, {"top1_accuracy": 0.9, "top5_accuracy": 0.2},
                        {"prediction_tokens": True}):
            try:
                runner.validate_evaluation({**good_eval, **changes})
            except runner.RunError:
                pass
            else:
                raise AssertionError(f"invalid native evaluation accepted: {changes}")
        runner.validate_evaluation_budget(good_eval, 2)
        try:
            runner.validate_evaluation_budget(good_eval, 3)
        except runner.RunError:
            pass
        else:
            raise AssertionError("held-out record count inconsistent with the budget was accepted")
        expect(True, "zero-budget and inconsistent held-out evaluations fail closed")
        expect(runner.safe_perplexity(0.0) == 1.0 and runner.safe_perplexity(700.0) is None
               and 'null' in json.dumps({"perplexity": runner.safe_perplexity(1000.0)}, allow_nan=False),
               "extreme chunk NLL yields strict JSON null rather than Infinity")
        for bad_path in (Path("x" * 1025), Path("file ")):
            try:
                runner.require_native_path(bad_path, "test")
            except runner.RunError:
                pass
            else:
                raise AssertionError("native path truncation/trim hazard was accepted")
        expect(True, "runner rejects native path truncation and trailing-blank hazards")

        expect(manifest_a["outputs"]["heldout"]["records"] > 0, "fixture has a real held-out split")
        _paths, snapshots = runner.corpus_inputs(manifest_a, tmp / "a")
        runner.verify_stamps(snapshots)
        invalid_manifests = []
        for kind in ("empty", "identical", "traversal", "boolean", "bad-hash", "funnel-count",
                     "source", "source-hash", "target", "schema"):
            invalid = copy.deepcopy(manifest_a)
            if kind == "empty":
                invalid["outputs"]["heldout"]["records"] = 0
            elif kind == "identical":
                invalid["outputs"]["heldout"] = copy.deepcopy(invalid["outputs"]["train_epoch"])
            elif kind == "traversal":
                invalid["outputs"]["heldout"]["path"] = "../outside.txt"
            elif kind == "boolean":
                invalid["outputs"]["heldout"]["records"] = True
            elif kind == "bad-hash":
                invalid["outputs"]["heldout"]["sha256"] = "0" * 64
            elif kind == "funnel-count":
                invalid["funnel"]["rejected_total"] += 1
            elif kind == "source":
                invalid["source"]["license"] = " "
            elif kind == "source-hash":
                invalid["source"]["sha256"] = None
            elif kind == "target":
                invalid["target"]["program_token_target"] = 99
            else:
                invalid["schema"] = "unsupported"
            invalid_manifests.append((kind, invalid))
        for kind, invalid in invalid_manifests:
            try:
                runner.corpus_inputs(invalid, tmp / "a")
            except runner.RunError:
                pass
            else:
                raise AssertionError(f"invalid manifest accepted: {kind}")
        expect(True, "empty, identical, escaping, malformed and mismatched corpus manifests fail closed")
        bad_corpus = tmp / "bad-corpus"
        shutil.copytree(tmp / "a", bad_corpus)
        invalid = copy.deepcopy(manifest_a)
        invalid["outputs"]["heldout"]["records"] = 0
        (bad_corpus / "manifest.json").write_text(json.dumps(invalid), encoding="utf-8")
        bad_run = tmp / "bad-heldout-run"
        rejected_heldout = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
             "--neural-bin", "/nonexistent/vegai_neural_pretrain", "--corpus-dir", str(bad_corpus),
             "--run-dir", str(bad_run)], capture_output=True, text=True, timeout=30, check=False)
        expect(rejected_heldout.returncode == 1 and "nonempty" in rejected_heldout.stderr
               and (bad_run / "failure.json").exists() and not (bad_run / "tokenizer.bpe").exists()
               and not (bad_run / "evidence.json").exists(),
               "empty held-out data stops the complete runner before training or success evidence")
        # Normal file replacement/tampering between preflight and evidence is
        # detected by cheap metadata checks without another full-array scan.
        changed_root = tmp / "changed-corpus"
        shutil.copytree(tmp / "a", changed_root)
        paths, stamps = runner.corpus_inputs(manifest_a, changed_root)
        paths["heldout"].write_bytes(paths["heldout"].read_bytes() + b"x")
        try:
            runner.verify_stamps(stamps)
        except runner.RunError:
            pass
        else:
            raise AssertionError("changed held-out file was not detected")
        expect(True, "routine corpus changes after preflight are detected")
        text, elapsed = runner.run_child([sys.executable, "-c", "print('bounded native output')"], 5)
        expect(text.strip() == "bounded native output" and elapsed >= 0, "native wrapper preserves normal output")
        try:
            runner.run_child([sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'\\xff')"], 5)
        except runner.RunError as error:
            expect("UTF-8" in str(error), "invalid UTF-8 structured output is rejected")
        else:
            raise AssertionError("invalid UTF-8 structured output was silently replaced")
        try:
            runner.run_child([sys.executable, "-c", "import sys;sys.stderr.write('useful diagnostic');sys.exit(7)"], 5)
        except runner.RunError as error:
            expect("rc=7" in str(error) and "useful diagnostic" in str(error), "native error retains a bounded stderr tail")
        else:
            raise AssertionError("native wrapper accepted a failed command")
        original_stdout_limit = runner.NATIVE_STDOUT_LIMIT
        try:
            runner.NATIVE_STDOUT_LIMIT = 1024
            try:
                runner.run_child([sys.executable, "-c", "import sys,time;print('x'*2048,flush=True);time.sleep(10)"], 5)
            except runner.RunError as error:
                expect("stdout_limit" in str(error), "native wrapper fails closed on unbounded child output")
            else:
                raise AssertionError("native wrapper accepted output over its cap")
        finally:
            runner.NATIVE_STDOUT_LIMIT = original_stdout_limit
        try:
            runner.run_child([sys.executable, "-c", "import time;time.sleep(10)"], 0.15)
        except runner.RunError as error:
            expect("timeout" in str(error), "native wrapper enforces a total deadline")
        else:
            raise AssertionError("native wrapper accepted a timed-out child")

        # The native SGD branch has no decoupled decay term, so the runner must
        # refuse --optimizer sgd with a non-zero --weight-decay instead of
        # recording regularisation that never runs.  This exits at argument
        # parsing time and therefore needs no compiled binary or corpus.
        rejected = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
             "--neural-bin", "/nonexistent/vegai_neural_pretrain",
             "--corpus-dir", str(tmp / "unused"), "--run-dir", str(tmp / "unused-run"),
             "--optimizer", "sgd", "--weight-decay", "0.01"],
            capture_output=True, text=True, check=False,
        )
        expect(rejected.returncode == 2 and "requires --optimizer nova or adamw" in rejected.stderr,
               "runner rejects weight decay on the SGD optimizer instead of ignoring it")

        # The schedule horizons are token-unit values and are divided by the
        # accumulation factor, so a budget that collapses below the warmup must
        # be rejected before any corpus work happens (exit 2 from argparse).
        collapsed = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
             "--neural-bin", "/nonexistent/vegai_neural_pretrain",
             "--corpus-dir", str(tmp / "unused"), "--run-dir", str(tmp / "unused-run"),
             "--gradient-accumulation", "32", "--warmup-updates", "20", "--total-updates", "40"],
            capture_output=True, text=True, check=False,
        )
        expect(collapsed.returncode == 2 and "must exceed --warmup-updates" in collapsed.stderr,
               "runner rejects a token-unit cosine budget that collapses below the warmup in step units")
        bad_accumulation = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
             "--neural-bin", "/nonexistent/vegai_neural_pretrain",
             "--corpus-dir", str(tmp / "unused"), "--run-dir", str(tmp / "unused-run"),
             "--gradient-accumulation", "0"],
            capture_output=True, text=True, check=False,
        )
        expect(bad_accumulation.returncode == 2, "runner rejects a zero gradient accumulation factor")

        # These invalid settings used to fail only after a tokenizer had been
        # trained (or silently change meaning). They must fail before creating
        # a run directory or invoking a native binary.
        invalid_options = [
            ["--lr", "nan"], ["--lr", "inf"], ["--lr", "1e308", "--weight-decay", "1e308"],
            ["--lr", "1e300", "--weight-decay", "0"], ["--lr", "1e-300"],
            ["--adam-eps", "1e-300"], ["--adam-eps", "1e300"],
            ["--optimizer", "adamw", "--lr", "1e30", "--beta1", "0.9999999999999998", "--weight-decay", "0"],
            ["--optimizer", "nova", "--lr", "1e20", "--adam-eps", "1e-30", "--weight-decay", "0"],
            ["--beta1", "1"], ["--adam-eps", "0"], ["--label-smoothing", "0.5"],
            ["--width", "15", "--heads", "2"],
            ["--width", "16777218", "--heads", "1", "--ffn", "16777218"], ["--window", "4097"],
            ["--gradient-accumulation", "513"], ["--chunks", "0"],
            ["--generation-tokens", "513"], ["--train-timeout-seconds", "0"], ["--threads", "0"],
            ["--schedule", "cosine", "--total-updates", "0"], ["--warmup-updates", "-1"],
        ]
        for index, options in enumerate(invalid_options):
            unused = tmp / f"invalid-preflight-{index}"
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
                 "--neural-bin", "/nonexistent/vegai_neural_pretrain",
                 "--corpus-dir", str(tmp / "a"), "--run-dir", str(unused), *options],
                capture_output=True, text=True, check=False,
            )
            expect(result.returncode == 2 and not unused.exists(),
                   f"invalid config {options} fails before directory or tokenizer work")

        existing = tmp / "existing-run"
        existing.mkdir()
        checkpoint, tokenizer = existing / "checkpoint.vgnr", existing / "tokenizer.bpe"
        checkpoint.write_bytes(b"valuable prior checkpoint")
        tokenizer.write_bytes(b"valuable prior tokenizer")
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
             "--neural-bin", "/nonexistent/vegai_neural_pretrain",
             "--corpus-dir", str(tmp / "a"), "--run-dir", str(existing)],
            capture_output=True, text=True, check=False,
        )
        expect(result.returncode == 1 and "already contains output" in result.stderr
               and checkpoint.read_bytes() == b"valuable prior checkpoint"
               and tokenizer.read_bytes() == b"valuable prior tokenizer"
               and not (existing / "failure.json").exists() and not (existing / runner.RUN_LOCK).exists(),
               "existing run artifacts are preserved without a failure report or leftover lock")
        locked = tmp / "locked-run"
        locked.mkdir()
        (locked / runner.RUN_LOCK).write_text("another runner", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
             "--neural-bin", "/nonexistent/vegai_neural_pretrain",
             "--corpus-dir", str(tmp / "a"), "--run-dir", str(locked)],
            capture_output=True, text=True, check=False,
        )
        expect(result.returncode == 1 and "locked" in result.stderr
               and (locked / runner.RUN_LOCK).read_text(encoding="utf-8") == "another runner"
               and not (locked / "failure.json").exists(), "an active or stale lock is never replaced")
        empty_prompts = tmp / "empty-prompts.txt"
        empty_prompts.write_text("\n", encoding="utf-8")
        prompt_run = tmp / "bad-prompts-run"
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
             "--neural-bin", "/nonexistent/vegai_neural_pretrain",
             "--corpus-dir", str(tmp / "a"), "--run-dir", str(prompt_run), "--prompts-file", str(empty_prompts)],
            capture_output=True, text=True, check=False,
        )
        expect(result.returncode == 1 and "prompts must be nonempty" in result.stderr
               and not (prompt_run / "tokenizer.bpe").exists() and not (prompt_run / runner.RUN_LOCK).exists(),
               "bad prompt files fail before training and release the run lock")
        huge_prompts = tmp / "huge-prompts.txt"
        huge_prompts.write_bytes(b"x" * (runner.MAX_PROMPTS_BYTES + 1))
        huge_run = tmp / "huge-prompts-run"
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
             "--neural-bin", "/nonexistent/vegai_neural_pretrain", "--corpus-dir", str(tmp / "a"),
             "--run-dir", str(huge_run), "--prompts-file", str(huge_prompts)],
            capture_output=True, text=True, check=False, timeout=30)
        expect(result.returncode == 1 and "exceeds 1 MiB" in result.stderr
               and not (huge_run / "tokenizer.bpe").exists(),
               "oversized prompt files fail before training")

        binary = os.environ.get("VEGAI_NEURAL_BIN", "")
        if binary:
            # A nonempty file can still contain no eligible evaluation tokens.
            # Exercise the real native baseline, not only manifest metadata.
            zero_eval_corpus = tmp / "zero-eval-corpus"
            shutil.copytree(tmp / "a", zero_eval_corpus)
            zero_eval_manifest = copy.deepcopy(manifest_a)
            meta = zero_eval_manifest["outputs"]["heldout"]
            (zero_eval_corpus / meta["path"]).write_bytes(b"\n\n")
            meta.update(bytes=2, records=2, approx_words=0, sha256=hashlib.sha256(b"\n\n").hexdigest())
            (zero_eval_corpus / "manifest.json").write_text(json.dumps(zero_eval_manifest), encoding="utf-8")
            zero_eval_run = tmp / "zero-eval-run"
            zero_eval = subprocess.run(
                [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"), "--neural-bin", binary,
                 "--corpus-dir", str(zero_eval_corpus), "--run-dir", str(zero_eval_run), "--vocab", "300",
                 "--width", "16", "--heads", "2", "--window", "8", "--ffn", "32", "--threads", "1"],
                capture_output=True, text=True, check=False, timeout=120)
            expect(zero_eval.returncode == 1 and "eligible_records" in zero_eval.stderr
                   and not (zero_eval_run / "evidence.json").exists(),
                   "native baseline with zero eligible held-out records rejects success evidence")
            inspect = subprocess.run(
                [binary, "evaluate", str(zero_eval_run / "checkpoint.vgnr"),
                 str(zero_eval_run / "tokenizer.bpe"), str(tmp / "a" / manifest_a["outputs"]["heldout"]["path"])],
                capture_output=True, text=True, check=False, timeout=60)
            expect(inspect.returncode == 0 and
                   runner.parse_prefixed_json(inspect.stdout, "NEURAL_EVALUATE_JSON:")["updates_completed_at_checkpoint"] == 0,
                   "zero-budget baseline fails before applying any training update")
            # Long positional paths used to be truncated silently; a trailing
            # blank used to be stripped and could target a different file.
            path_tokenizer = tmp / "path-guard.bpe"
            too_long = str(tmp / ("x" * 1100))
            result = subprocess.run([binary, "init-checkpoint", str(path_tokenizer), too_long],
                                    capture_output=True, text=True, check=False, timeout=30)
            expect(result.returncode == 2 and "truncated" in result.stderr and not path_tokenizer.exists(),
                   "native positional path truncation fails before creating a tokenizer")
            result = subprocess.run([binary, "init-checkpoint", str(path_tokenizer) + " ", str(tmp / "path-guard.vgnr")],
                                    capture_output=True, text=True, check=False, timeout=30)
            expect(result.returncode == 2 and "trailing-blank" in result.stderr and not path_tokenizer.exists(),
                   "native path trailing blanks cannot redirect writes to a different file")
            guarded_file = tmp / "path-collision.txt"
            original = b"original corpus or checkpoint\n"
            guarded_file.write_bytes(original)
            collision_cases = [
                ["tokenizer-train-fast", str(guarded_file), str(guarded_file), "300"],
                ["init-checkpoint", str(guarded_file), str(guarded_file)],
                ["train", str(guarded_file), str(guarded_file), str(path_tokenizer), "1"],
                ["train-chunk", str(guarded_file), str(path_tokenizer), str(guarded_file), "1"],
            ]
            for argv in collision_cases:
                result = subprocess.run([binary, *argv], capture_output=True, text=True, check=False, timeout=30)
                expect(result.returncode == 2 and "must be distinct" in result.stderr
                       and guarded_file.read_bytes() == original and not path_tokenizer.exists(),
                       f"native {argv[0]} preserves an exact input/output path collision")
            # The native CLI must reject unrepresentable or ambiguous real
            # options before creating even the fallback tokenizer. This also
            # catches SIGFPE from the runtime's decimal conversion in a trap build.
            for index, value in enumerate(("16,17", "16/", "/", "2*16", "+", "16 17", "2147483648")):
                tokenizer = tmp / f"invalid-int-{index}.bpe"
                checkpoint = tmp / f"invalid-int-{index}.vgnr"
                result = subprocess.run([binary, "init-checkpoint", str(tokenizer), str(checkpoint),
                                         "--width", value], capture_output=True, text=True, check=False, timeout=30)
                expect(result.returncode == 2 and not tokenizer.exists() and not checkpoint.exists(),
                       f"native integer option rejects ambiguous or out-of-range value {value!r}")
            # int64 minimum and maximum must remain valid, rather than being
            # accidentally rejected by a positive-magnitude overflow check.
            for index, seed in enumerate(("-9223372036854775808", "+9223372036854775807")):
                tokenizer = tmp / f"valid-seed-{index}.bpe"
                checkpoint = tmp / f"valid-seed-{index}.vgnr"
                result = subprocess.run([binary, "init-checkpoint", str(tokenizer), str(checkpoint),
                                         "--width", "16", "--heads", "2", "--ffn", "32", "--seed", seed],
                                        capture_output=True, text=True, check=False, timeout=60)
                expect(result.returncode == 0, f"native accepts boundary int64 seed {seed}")
            invalid_chunk = subprocess.run([binary, "train-chunk", str(guarded_file),
                                            str(tmp / "bad-count.vgnr"), str(path_tokenizer), "1,2"],
                                           capture_output=True, text=True, check=False, timeout=30)
            expect(invalid_chunk.returncode == 2 and not path_tokenizer.exists()
                   and not (tmp / "bad-count.vgnr").exists(),
                   "native positional record count rejects list-directed suffix before creating output")
            for index, seed in enumerate(("9223372036854775808", "-9223372036854775809", "/", "1,2")):
                tokenizer = tmp / f"invalid-seed-{index}.bpe"
                checkpoint = tmp / f"invalid-seed-{index}.vgnr"
                result = subprocess.run([binary, "init-checkpoint", str(tokenizer), str(checkpoint), "--seed", seed],
                                        capture_output=True, text=True, check=False, timeout=30)
                expect(result.returncode == 2 and not tokenizer.exists() and not checkpoint.exists(),
                       f"native rejects malformed or out-of-range int64 seed {seed!r}")
            invalid_reals = [
                ("--lr", "nan"), ("--lr", "inf"), ("--lr", "1e309"), ("--lr", "-1e309"),
                ("--weight-decay", "1e-9999"), ("--beta1", "nan"), ("--lr", "0e-9999"),
                ("--lr", "0.1,0.2"), ("--lr", "0.1/"), ("--lr", "1e-"),
                ("--lr", "0." + "0" * 300 + "1"),
            ]
            for index, (option, value) in enumerate(invalid_reals):
                tokenizer = tmp / f"invalid-real-{index}.bpe"
                checkpoint = tmp / f"invalid-real-{index}.vgnr"
                result = subprocess.run(
                    [binary, "init-checkpoint", str(tokenizer), str(checkpoint), option, value],
                    capture_output=True, text=True, check=False, timeout=30,
                )
                expect(result.returncode == 2 and not tokenizer.exists() and not checkpoint.exists(),
                       f"native {option} rejects invalid real {value[:24]!r} before output (exit {result.returncode})")
            # Quotes, backslashes and U+2028 are legal POSIX path bytes and
            # must remain valid JSON rather than corrupt the native report.
            run_dir = tmp / ('run-"quote-\\backslash-\u2028' if os.name == "posix" else "run")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
                 "--neural-bin", binary, "--corpus-dir", str(tmp / "a"), "--run-dir", str(run_dir),
                 "--vocab", "300", "--width", "16", "--heads", "2", "--window", "8", "--ffn", "32",
                 "--warmup-updates", "10", "--generation-tokens", "4", "--threads", "1", "--omp-wait-policy", "passive"],
                capture_output=True, text=True, check=False, timeout=1200,
            )
            expect(completed.returncode == 0, f"micro native epoch-1 run succeeds (stderr: {completed.stderr[-600:]})")
            evidence = json.loads((run_dir / "evidence.json").read_text(encoding="utf-8"))
            expect(evidence["exact_epochs"] == 1, "micro run is exactly one epoch")
            expect(evidence["config"]["threads"] == 1 and evidence["config"]["omp_wait_policy"] == "PASSIVE",
                   "native runner records the requested thread count and wait policy")
            expect(len(evidence["generations"]) == len(runner.DEFAULT_PROMPTS), "prompt suite completed")
            expect(evidence["baseline_heldout"]["prediction_tokens"] > 0
                   and evidence["final_heldout"]["prediction_tokens"] == evidence["baseline_heldout"]["prediction_tokens"]
                   and evidence["config"]["max_eval_records"] == 0,
                   "real held-out measurements use and record the same positive token budget")
            expect(evidence["init"]["checkpoint"] == str(run_dir / "checkpoint.vgnr")
                   and evidence["training_complete"]["tokenizer"] == str(run_dir / "tokenizer.bpe"),
                   "native JSON safely round-trips escaped run paths")
            byte_tokenizer, byte_checkpoint = tmp / "byte-model.bpe", tmp / "byte-model.vgnr"
            byte_init = subprocess.run(
                [binary, "init-checkpoint", str(byte_tokenizer), str(byte_checkpoint),
                 "--width", "16", "--heads", "2", "--window", "8", "--ffn", "32"],
                capture_output=True, text=True, check=False, timeout=60)
            expect(byte_init.returncode == 0, "byte-fallback generation fixture initializes")
            generated_prompt = subprocess.run(
                [binary, "generate", str(byte_checkpoint), str(byte_tokenizer), "a ", "1", "7"],
                capture_output=True, text=False, check=False, timeout=60)
            generated_report = runner.parse_prefixed_json(generated_prompt.stdout.decode("utf-8", errors="replace"),
                                                           "NEURAL_GENERATE_JSON:")
            expect(generated_prompt.returncode == 0 and generated_report["prompt_tokens"] == 2,
                   "native generation preserves prompt trailing spaces")
            epoch_path = tmp / "a" / manifest_a["outputs"]["train_epoch"]["path"]
            verify_checkpoint_cli_overrides(binary, run_dir / "tokenizer.bpe", epoch_path, tmp)

            chunk_run_dir = tmp / "run-chunks"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
                 "--neural-bin", binary, "--corpus-dir", str(tmp / "a"), "--run-dir", str(chunk_run_dir),
                 "--vocab", "300", "--width", "16", "--heads", "2", "--window", "8", "--ffn", "32",
                 "--warmup-updates", "10", "--generation-tokens", "4", "--chunks", "2", "--threads", "1",
                 "--omp-wait-policy", "passive"],
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
            expect((chunk_run_dir / "checkpoint.vgnr").read_bytes() == (run_dir / "checkpoint.vgnr").read_bytes(),
                   "one-pass and chunked AdamW runs produce byte-identical checkpoints")
            expect(not (run_dir / runner.RUN_LOCK).exists() and not (chunk_run_dir / runner.RUN_LOCK).exists(),
                   "successful runs release their directory locks")

            nova_run_dir = tmp / "run-nova"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "tools/run_neural_turkish_epoch1.py"),
                 "--neural-bin", binary, "--corpus-dir", str(tmp / "a"), "--run-dir", str(nova_run_dir),
                 "--vocab", "300", "--width", "16", "--heads", "2", "--window", "8", "--ffn", "32",
                 "--warmup-updates", "10", "--generation-tokens", "4", "--optimizer", "nova",
                 "--weight-decay", "0.007", "--max-eval-records", "1"],
                capture_output=True, text=True, check=False, timeout=1200,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_WAIT_POLICY": "PASSIVE"},
            )
            expect(completed.returncode == 0,
                   f"NOVA micro run accepts its decay (stderr: {completed.stderr[-600:]})")
            nova_evidence = json.loads((nova_run_dir / "evidence.json").read_text(encoding="utf-8"))
            expect(nova_evidence["config"]["optimizer"] == "nova"
                   and nova_evidence["config"]["weight_decay"] == 0.007
                   and nova_evidence["config"]["weight_decay_scope"] == "nova embeddings only",
                   "runner binds NOVA and its effective decay in evidence")
            expect(nova_evidence["config"]["threads"] == 1 and nova_evidence["config"]["omp_wait_policy"] == "PASSIVE",
                   "runner honors valid inherited OpenMP settings when no explicit override is given")
            expect(nova_evidence["config"]["max_eval_records"] == 1
                   and nova_evidence["baseline_heldout"]["eligible_records"] == 1
                   and nova_evidence["final_heldout"]["eligible_records"] == 1,
                   "bounded held-out record selection is recorded and applied equally")
        else:
            print("skip - VEGAI_NEURAL_BIN unset; native micro run not executed")
    print("verify_neural_turkish_epoch1: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
