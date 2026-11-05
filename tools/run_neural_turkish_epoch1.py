#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run exactly one native neural pretraining epoch on a governed Turkish corpus.

Orchestrates only the native Fortran ``vegai_neural_pretrain`` binary:

1. ``tokenizer-train-fast`` on the manifest's tokenizer sample;
2. ``init-checkpoint`` + ``evaluate`` for the untrained held-out baseline;
3. ``train`` for exactly **one epoch** over the manifest's ``train_epoch``
   slice (the slice definition and its hash are part of the evidence);
4. ``evaluate`` on the frozen document-disjoint held-out split;
5. ``generate`` for a fixed Turkish prompt suite, captured verbatim.

The report distinguishes three numbers that are often conflated: tokens
actually trained in this run, tokens accepted by the full corpus pipeline,
and the 99-billion-token program target.  The 99B completion time below is
an arithmetic projection from the *measured* native throughput on this
host, labelled as a projection -- never as an achieved result.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any

from bounded_process import ProcessExecutionError, run_process

SCHEMA = "vegai-neural-turkish-epoch1/v1"
TARGET_TOKENS_99B = 99_000_000_000
# A generated token can decode to at most 8192 bytes, with at most 512 tokens.
# The cap therefore allows a full native generation and its JSON/markers.
NATIVE_STDOUT_LIMIT = 8 * 1024 * 1024
NATIVE_STDERR_LIMIT = 2 * 1024 * 1024
NATIVE_STDERR_TAIL = 8 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_PROMPTS_BYTES = 1024 * 1024
MAX_PROMPTS = 1000

DEFAULT_PROMPTS = [
    "Türkiye'nin başkenti",
    "Trabzon Karadeniz kıyısında",
    "Yapay zeka verilerden",
    "Bilim insanları deney",
    "Türkçe eklemeli bir dil",
    "İstanbul Boğazı'nın iki yakasında",
    "Osmanlı İmparatorluğu'nun başkenti",
    "Fotosentez sürecinde bitkiler",
    "Karadeniz Bölgesi'nde yağış",
    "Bir romanın ilk cümlesi şöyleydi:",
]


class RunError(Exception):
    pass


# Never reuse a run directory implicitly: its last completed checkpoint may be
# the only recoverable result of a long/failed run. The exclusive lock only
# coordinates other instances of this runner, not arbitrary external writers.
RUN_OUTPUTS = ("tokenizer.bpe", "checkpoint.vgnr", "chunks.json", "evidence.json", "RESULTS.md", "failure.json")
RUN_LOCK = ".vegai-neural-epoch1.lock"


def claim_run_directory(run_dir: Path) -> tuple[int, Path]:
    lock_path = run_dir / RUN_LOCK
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RunError(f"run directory is locked: {lock_path}; verify the previous process has stopped") from exc
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        conflicts = [name for name in RUN_OUTPUTS if os.path.lexists(run_dir / name)]
        if conflicts:
            raise RunError(f"run directory already contains output(s): {', '.join(conflicts)}; choose a fresh directory")
        return fd, lock_path
    except BaseException:
        os.close(fd)
        lock_path.unlink(missing_ok=True)
        raise


def release_run_directory(fd: int, lock_path: Path) -> None:
    os.close(fd)
    try:
        lock_path.unlink(missing_ok=True)
    except OSError as exc:
        print(f"warning: could not remove run lock {lock_path}: {exc}", file=sys.stderr)


def publish_text(path: Path, text: str) -> None:
    """Publish one complete metadata file; this is not a power-loss fsync guarantee."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            fd = -1
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def publish_results(run_dir: Path, results_text: str, evidence_text: str) -> None:
    results_path = run_dir / "RESULTS.md"
    publish_text(results_path, results_text)
    try:
        publish_text(run_dir / "evidence.json", evidence_text)
    except BaseException:
        try:
            results_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            print(f"warning: could not remove incomplete RESULTS.md: {cleanup_error}", file=sys.stderr)
        raise


def sha256_file(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
            size += len(block)
    return {"path": str(path), "bytes": size, "sha256": hasher.hexdigest()}


def require_native_path(path: Path, label: str) -> None:
    value = str(path)
    if not value or value.endswith(" ") or len(value.encode("utf-8")) > 1024:
        raise RunError(f"{label} must fit a native 1024-byte path and cannot end with a blank")


def file_stamp(path: Path, label: str) -> tuple[int, int, int, int, int]:
    """Cheap change detection; not an adversarial file attestation."""
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RunError(f"{label} must be a regular file: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def verify_stamps(snapshots: list[tuple[Path, str, tuple[int, int, int, int, int]]]) -> None:
    for path, label, before in snapshots:
        if file_stamp(path, label) != before:
            raise RunError(f"{label} changed during the run: {path}")


def corpus_inputs(manifest: dict[str, Any], corpus_dir: Path) -> tuple[dict[str, Path], list[tuple[Path, str, tuple[int, int, int, int, int]]]]:
    """Validate builder metadata without an extra full-corpus scan."""
    if manifest.get("schema") != "vegai-turkish-wiki-corpus/v2":
        raise RunError("unsupported corpus manifest schema")
    outputs = manifest.get("outputs")
    funnel = manifest.get("funnel")
    if not isinstance(outputs, dict) or not isinstance(funnel, dict) or not isinstance(manifest.get("source"), dict):
        raise RunError("corpus manifest is missing output, funnel or source metadata")
    for key in ("source_documents", "accepted_documents", "rejected_total", "duplicate_documents",
                "duplicate_paragraphs", "train_documents", "heldout_documents"):
        value = funnel.get(key)
        if type(value) is not int or not 0 <= value <= 2**63 - 1:
            raise RunError(f"corpus funnel {key} must be a nonnegative int64")
    rejected = funnel.get("rejected")
    if (not isinstance(rejected, dict)
            or any(not isinstance(k, str) or not k or type(v) is not int or not 0 <= v <= 2**63 - 1
                   for k, v in rejected.items())
            or sum(rejected.values()) != funnel["rejected_total"]
            or funnel["train_documents"] + funnel["heldout_documents"] != funnel["accepted_documents"]
            or funnel["accepted_documents"] + funnel["rejected_total"] + funnel["duplicate_documents"]
               > funnel["source_documents"]):
        raise RunError("corpus funnel counts are inconsistent")
    source = manifest["source"]
    if any(not isinstance(source.get(key), str) or not source[key].strip() for key in ("file", "kind", "license")):
        raise RunError("corpus source file, kind and license must be nonempty strings")
    source_hash = source.get("sha256")
    if ("sha256" not in source or (source_hash is not None and
            (not isinstance(source_hash, str) or len(source_hash) != 64
             or any(c not in "0123456789abcdef" for c in source_hash)))):
        raise RunError("corpus source SHA-256 must be null or a lowercase digest")
    if ((source["kind"] == "trwiki-pages-articles-xml-bz2" and source_hash is None)
            or (source["kind"] == "operator-text-dir" and source_hash is not None)
            or source["kind"] not in ("trwiki-pages-articles-xml-bz2", "operator-text-dir")):
        raise RunError("corpus source kind and SHA-256 are inconsistent")
    target = manifest.get("target")
    if (not isinstance(target, dict) or type(target.get("program_token_target")) is not int
            or target["program_token_target"] != TARGET_TOKENS_99B):
        raise RunError("corpus manifest has an invalid 99B program target")
    root = corpus_dir.resolve(strict=True)
    if not root.is_dir():
        raise RunError("corpus directory is not a directory")
    paths: dict[str, Path] = {}
    snapshots: list[tuple[Path, str, tuple[int, int, int, int, int]]] = []
    hashes: dict[str, str] = {}
    for key in ("train", "heldout", "train_epoch", "tokenizer_sample"):
        meta = outputs.get(key)
        if not isinstance(meta, dict):
            raise RunError(f"corpus output {key} is missing")
        raw = meta.get("path")
        if (not isinstance(raw, str) or not raw or "\\" in raw or raw.startswith("/")
                or any(part in ("", ".", "..") for part in raw.split("/"))):
            raise RunError(f"corpus output {key} must be a relative path inside the corpus directory")
        for field in ("bytes", "records", "approx_words"):
            value = meta.get(field)
            if type(value) is not int or not 0 <= value <= 2**63 - 1:
                raise RunError(f"corpus output {key} {field} must be a nonnegative int64")
        if meta["bytes"] == 0 or meta["records"] == 0:
            raise RunError(f"corpus output {key} must be nonempty; held-out evidence cannot use zero records")
        expected_hash = meta.get("sha256")
        if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                or any(c not in "0123456789abcdef" for c in expected_hash)):
            raise RunError(f"corpus output {key} has an invalid SHA-256")
        path = (root / raw).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RunError(f"corpus output {key} escapes its directory") from error
        label = f"corpus output {key}"
        before = file_stamp(path, label)
        if before[2] != meta["bytes"]:
            raise RunError(f"{label} does not match manifest bytes")
        actual = sha256_file(path)
        if file_stamp(path, label) != before or actual["sha256"] != expected_hash:
            raise RunError(f"{label} changed or does not match manifest hash")
        paths[key], hashes[key] = path, actual["sha256"]
        snapshots.append((path, label, before))
    # The builder uses document-level splits; at least reject identical whole
    # files here. This does not prove independent record/document disjointness.
    if any(hashes["heldout"] == hashes[key] or os.path.samefile(paths["heldout"], paths[key])
           for key in ("train", "train_epoch", "tokenizer_sample")):
        raise RunError("held-out corpus is identical to a training/tokenizer input")
    return paths, snapshots


def strict_json_object(text: str, label: str) -> dict[str, Any]:
    """Reject ambiguous objects and non-finite JSON numbers, including nested values."""
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def bounded_integer(value: str) -> int:
        if len(value) > 128:
            raise ValueError("JSON integer is too long")
        return int(value)

    def finite_float(value: str) -> float:
        if len(value) > 128:
            raise ValueError("JSON number is too long")
        result = float(value)
        mantissa = value.lower().split("e", 1)[0]
        if not math.isfinite(result) or (result == 0.0 and any(c in "123456789" for c in mantissa)):
            raise ValueError("JSON number is outside the finite nonzero range")
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value}")

    try:
        value = json.loads(text, object_pairs_hook=unique_object, parse_int=bounded_integer,
                           parse_constant=reject_constant, parse_float=finite_float)
    except (ValueError, RecursionError) as error:
        raise RunError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise RunError(f"{label} must be a JSON object")
    pending: list[Any] = [value]
    try:
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                item.encode("utf-8")  # reject unpaired escaped surrogates
            elif isinstance(item, dict):
                pending.extend(item.keys())
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
    except UnicodeEncodeError as error:
        raise RunError(f"{label} contains invalid Unicode") from error
    return value


def parse_prefixed_json(stdout: str, prefix: str) -> dict[str, Any]:
    # Generated text is arbitrary data and can itself contain protocol-looking
    # lines; the metadata line is always before the begin marker.
    if prefix == "NEURAL_GENERATE_JSON:":
        stdout = stdout.split("NEURAL_GENERATION_BEGIN", 1)[0]
    lines = [line[len(prefix):].strip() for line in stdout.split("\n") if line.startswith(prefix)]
    if len(lines) != 1:
        raise RunError(f"expected exactly one {prefix} line in native output")
    return strict_json_object(lines[0], prefix)


def metric_count(report: dict[str, Any], key: str, *, minimum: int = 0) -> int:
    value = report.get(key)
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise RunError(f"native metric {key} must be an integer in [{minimum}, 2**63-1]")
    return value


def metric_number(report: dict[str, Any], key: str, *, minimum: float = 0.0, maximum: float = math.inf) -> float:
    value = report.get(key)
    if type(value) not in (int, float):
        raise RunError(f"native metric {key} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise RunError(f"native metric {key} is out of range") from error
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise RunError(f"native metric {key} is out of range")
    return number


def safe_perplexity(mean_nll: float) -> float | None:
    if not math.isfinite(mean_nll) or mean_nll < 0:
        raise RunError("mean NLL must be finite and nonnegative")
    return math.exp(mean_nll) if mean_nll < 700.0 else None


def validate_evaluation(report: dict[str, Any]) -> None:
    if report.get("schema_version") != "vegai-neural-evaluate/v1":
        raise RunError("unexpected native evaluation schema")
    eligible = metric_count(report, "eligible_records", minimum=1)
    tokens = metric_count(report, "prediction_tokens", minimum=1)
    skipped = metric_count(report, "skipped_records")
    truncations = metric_count(report, "tokenizer_truncations")
    if tokens < eligible or truncations > skipped:
        raise RunError("native evaluation record/token counters are inconsistent")
    metric_count(report, "updates_completed_at_checkpoint")
    metric_number(report, "mean_nll")
    top1 = metric_number(report, "top1_accuracy", maximum=1.0)
    top5 = metric_number(report, "top5_accuracy", maximum=1.0)
    if top1 > top5:
        raise RunError("native top-1 accuracy exceeds top-5 accuracy")
    if report.get("perplexity") is not None:
        metric_number(report, "perplexity", minimum=1.0)


def validate_evaluation_budget(report: dict[str, Any], expected_records: int) -> None:
    validate_evaluation(report)
    if metric_count(report, "eligible_records") + metric_count(report, "skipped_records") != expected_records:
        raise RunError("native held-out evaluation record count does not match manifest/budget")


def run_child(argv: list[str], timeout: float, env_extra: dict[str, str] | None = None) -> tuple[str, float]:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        completed = run_process(
            argv, cwd=Path.cwd(), timeout_seconds=timeout, environment=env,
            stdout_limit_bytes=NATIVE_STDOUT_LIMIT, stderr_limit_bytes=NATIVE_STDERR_LIMIT,
            stderr_tail_limit_bytes=NATIVE_STDERR_TAIL, label="neural native command",
        )
    except ProcessExecutionError as error:
        raise RunError(f"native command failed ({error.kind}): {error}") from error
    try:
        replace = completed.returncode != 0 or (len(argv) > 1 and argv[1] == "generate")
        stdout = completed.stdout.decode("utf-8", errors="replace" if replace else "strict")
    except UnicodeDecodeError as error:
        raise RunError("native structured output is not valid UTF-8") from error
    stderr = completed.stderr_tail.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise RunError(
            f"native command failed rc={completed.returncode}: {' '.join(argv)}\n"
            f"stdout tail:\n{stdout[-3000:]}\nstderr tail:\n{stderr[-3000:]}"
        )
    return stdout, completed.elapsed_seconds


def extract_generation(stdout: str) -> str:
    begin_marker, end_marker = "NEURAL_GENERATION_BEGIN", "NEURAL_GENERATION_END"
    begin, end = stdout.find(begin_marker), stdout.rfind(end_marker)
    if begin < 0 or end < 0 or end <= begin:
        raise RunError("generation markers missing")
    start = begin + len(begin_marker)
    if stdout[start:start + 2] == "\r\n":
        newline = "\r\n"
    elif stdout[start:start + 1] == "\n":
        newline = "\n"
    else:
        raise RunError("generation begin marker is not a complete line")
    start += len(newline)
    if stdout[end + len(end_marker):].strip("\r\n"):
        raise RunError("unexpected data after generation end marker")
    body = stdout[start:end]
    # The native writer emits one protocol newline after the decoded payload;
    # remove exactly that one, not newlines belonging to generated text.
    if body.endswith(newline):
        return body[:-len(newline)]
    raise RunError("generation end marker is not on a separate line")


def project_99b(prediction_tokens_per_second: float, trained_prediction_tokens: int) -> dict[str, Any]:
    """Arithmetic projection only; assumptions are spelled out in the report."""
    result: dict[str, Any] = {
        "program_target_tokens": TARGET_TOKENS_99B,
        "trained_prediction_tokens_this_run": trained_prediction_tokens,
        "fraction_of_target_this_run": (
            trained_prediction_tokens / TARGET_TOKENS_99B if TARGET_TOKENS_99B else 0.0
        ),
        "measured_prediction_tokens_per_second": prediction_tokens_per_second,
        "assumptions": [
            "single host identical to this run, same model shape and vocabulary",
            "constant throughput at the measured epoch rate (no data stalls, no failures)",
            "99B counts BPE prediction tokens, matching the trainer metric",
            "excludes corpus acquisition, filtering, tokenizer training and evaluation",
        ],
        "is_projection_not_a_result": True,
    }
    if prediction_tokens_per_second and prediction_tokens_per_second > 0:
        seconds = TARGET_TOKENS_99B / prediction_tokens_per_second
        if math.isfinite(seconds):
            result["projected_single_host_seconds_for_99b"] = seconds
            result["projected_single_host_days_for_99b"] = seconds / 86_400
            result["projected_single_host_years_for_99b"] = seconds / (86_400 * 365.25)
        else:
            result["projection_out_of_float_range"] = True
    return result


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 0.001 or abs(value) >= 1e9):
            return f"{value:.6g}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_cell(text: str) -> str:
    # The report is a readable view; evidence.json retains the decoded text.
    clean = "".join(" " if c in "\r\n" else (f"\\x{ord(c):02x}" if ord(c) < 32 or ord(c) == 127 else c)
                    for c in text)
    return html.escape(clean, quote=False).replace("|", "&#124;").replace("`", "&#96;")


def render_results_md(evidence: dict[str, Any]) -> str:
    corpus = evidence["corpus"]
    funnel = corpus["manifest"]["funnel"]
    outputs = corpus["manifest"]["outputs"]
    baseline = evidence["baseline_heldout"]
    final = evidence["final_heldout"]
    epoch = evidence["training_epoch"]
    projection = evidence["scale_projection"]
    lines: list[str] = []
    lines.append("# Türkçe neural pretraining — epoch 1 sonucu (99B programı)")
    lines.append("")
    lines.append(
        "> Bu kayıt gerçek bir native Fortran koşusunun kanıtıdır. 99 milyar token, bu koşunun "
        "sonucu değil program hedefidir; aşağıda eğitilen gerçek token sayısı, tam boru hattının "
        "kabul ettiği gerçek veri ve ölçülen hızdan türetilen projeksiyon birbirinden açıkça ayrılır."
    )
    lines.append("")
    lines.append("## Veri (gerçek, kaynak-izlenebilir)")
    lines.append("")
    source = corpus["manifest"]["source"]
    lines.append(f"- Kaynak: `{markdown_cell(source['file'])}` ({markdown_cell(source['kind'])}), "
                 f"lisans: {markdown_cell(source['license'])}")
    lines.append(f"- Kaynak SHA-256: `{source['sha256']}`")
    lines.append(f"- Kaynak belge: {funnel['source_documents']:,}; kabul: {funnel['accepted_documents']:,}; "
                 f"ret: {funnel['rejected_total']:,}; yinelenen belge: {funnel['duplicate_documents']:,}; "
                 f"yinelenen paragraf: {funnel['duplicate_paragraphs']:,}")
    lines.append(f"- Ret hunisi: `{markdown_cell(json.dumps(funnel['rejected'], ensure_ascii=False, sort_keys=True))}`")
    for name in ("train", "heldout", "train_epoch", "tokenizer_sample"):
        meta = outputs[name]
        lines.append(f"- `{markdown_cell(meta['path'])}`: {meta['bytes']:,} bayt, {meta['records']:,} kayıt, "
                     f"~{meta['approx_words']:,} kelime, SHA-256 `{meta['sha256']}`")
    lines.append("")
    lines.append("## Model ve eğitim (epoch = 1)")
    lines.append("")
    cfg = evidence["config"]
    actual_vocab = evidence.get("tokenizer", {}).get("actual_vocabulary_size", cfg['vocab'])
    lines.append(
        f"- Mimari: byte-BPE vocab {actual_vocab:,} (istenen {cfg['vocab']:,}), genişlik {cfg['width']}, başlık {cfg['heads']}, "
        f"pencere {cfg['window']}, FFN {cfg['ffn']} (bağlı-çıktı katmanı eğitilir; dikkat/retention çekirdeği donuktur)"
    )
    lines.append(
        f"- Optimize edici: {cfg['optimizer']}, tepe LR {cfg['lr']}, zamanlama {cfg['schedule']} "
        f"(warmup {cfg['warmup_updates']:,}), label smoothing {cfg['label_smoothing']}, seed {cfg['seed']}"
    )
    lines.append(
        f"- Ağırlık azalması (weight decay): {cfg['weight_decay']} ({cfg.get('weight_decay_scope', 'n/a')}); "
        f"AdamW beta1 {cfg.get('adam_beta1', 'n/a')}, beta2 {cfg.get('adam_beta2', 'n/a')}, "
        f"eps {cfg.get('adam_epsilon', 'n/a')}"
    )
    lines.append(
        f"- Güncelleme birikimi: her {cfg.get('gradient_accumulation', 1)} tahmin tokenında bir optimize edici adımı "
        f"(adım cinsinden warmup {cfg['warmup_updates']:,}, horizon {cfg['total_updates']:,}); "
        f"tied embedding ve AdamW momentleri token başına değil batch başına bir kez akıtılır"
    )
    lines.append(
        f"- Epoch: {epoch['eligible_records']:,} kayıt, {epoch['prediction_tokens']:,} tahmin tokenı, "
        f"{fmt(epoch['epoch_wall_seconds'])} sn, {fmt(epoch['prediction_tokens_per_second'])} token/sn"
    )
    lines.append(f"- Eğitim içi ortalama NLL: {fmt(epoch['mean_nll'])} (online, güncelleme öncesi dağılım)")
    if epoch.get("chunks"):
        lines.append(
            f"- Eğitim {epoch['chunks']} checkpoint'li parçada koşuldu (aynı online sıra/config; "
            f"süre: {epoch['timing_basis']})")
    lines.append("")
    lines.append("## Held-out sonuç (manifestin belirttiği bölme)")
    lines.append("")
    lines.append("| Ölçüm | Eğitimsiz başlangıç | Epoch 1 sonrası |")
    lines.append("|---|---:|---:|")
    for label, key in (("Ortalama NLL", "mean_nll"), ("Perplexity", "perplexity"),
                       ("Top-1 doğruluk", "top1_accuracy"), ("Top-5 doğruluk", "top5_accuracy")):
        lines.append(f"| {label} | {fmt(baseline.get(key))} | {fmt(final.get(key))} |")
    lines.append(f"| Held-out tahmin tokenı | {baseline['prediction_tokens']:,} | {final['prediction_tokens']:,} |")
    limit = cfg.get("max_eval_records", 0)
    lines.append(f"- Kayıt limiti: {limit:,} (0 = tüm held-out kayıtları); "
                 f"uygun: {baseline['eligible_records']:,}, atlanan: {baseline['skipped_records']:,}.")
    lines.append("")
    lines.append("## Prompt → Response (tablo görünümü)")
    lines.append("")
    lines.append("Model bir talimat modeli değildir; yanıtlar seçilip düzenlenmemiş nedensel devam üretimidir. "
                 "Tabloda satır sonları ve özel karakterler gösterim için dönüştürülür; çözümlenmiş metin "
                 "evidence.json içindedir. Geçersiz UTF-8 baytları değiştirme karakteriyle gösterilir.")
    lines.append("")
    lines.append("| # | Prompt | Response |")
    lines.append("|---:|---|---|")
    for index, item in enumerate(evidence["generations"], start=1):
        response = markdown_cell(item["response"])
        prompt = markdown_cell(item["prompt"])
        lines.append(f"| {index} | {prompt} | {response} |")
    lines.append("")
    lines.append("## 99 milyar ölçeği — dürüst projeksiyon")
    lines.append("")
    lines.append(f"- Bu koşuda eğitilen: **{projection['trained_prediction_tokens_this_run']:,} token** "
                 f"(hedefin {fmt(projection['fraction_of_target_this_run'] * 100)}%'si)")
    if "projected_single_host_days_for_99b" in projection:
        lines.append(
            f"- Ölçülen hızla tek hostta 99B token: ~{fmt(projection['projected_single_host_days_for_99b'])} gün "
            f"(~{fmt(projection['projected_single_host_years_for_99b'])} yıl) — bu bir projeksiyondur, sonuç değildir."
        )
    elif projection.get("projection_out_of_float_range"):
        lines.append("- Ölçülen hız için süre projeksiyonu sayı aralığı dışında; süre verilmedi.")
    for assumption in projection["assumptions"]:
        lines.append(f"- Varsayım: {assumption}")
    lines.append("")
    lines.append("## Sınırlar")
    lines.append("")
    for limitation in evidence["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-bin", required=True)
    parser.add_argument("--corpus-dir", required=True, help="directory written by build_turkish_wiki_corpus.py")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--min-pair-frequency", type=int, default=8)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--ffn", type=int, default=384)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--optimizer", default="adamw", choices=("nova", "adamw", "sgd"))
    parser.add_argument("--schedule", default="invsqrt", choices=("constant", "invsqrt", "cosine"))
    # The two schedule horizons below are expressed in *prediction tokens*, the
    # unit the corpus estimate is available in (bytes/token).  They are divided
    # by --gradient-accumulation because the native scheduler counts optimizer
    # steps, and one step now consumes a whole batch.
    parser.add_argument("--warmup-updates", type=int, default=2000,
                        help="warmup in prediction tokens (converted to optimizer steps)")
    parser.add_argument("--total-updates", type=int, default=0,
                        help="cosine horizon in prediction tokens (estimate from measured bytes/token); "
                             "0 = omit")
    parser.add_argument("--gradient-accumulation", type=int, default=32,
                        help="prediction tokens averaged into one optimizer step (1 = exact per-token "
                             "updates).  The update loops over the tied embedding and both AdamW "
                             "moments once per batch instead of once per token, which is where the "
                             "epoch time went; 32-64 is the measured sweet spot for this shape.")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    # NOVA and AdamW decay embeddings; SGD has no decay term.
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="NOVA/AdamW embedding decay (default 0.01); SGD requires 0")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--generation-seed", type=int, default=7)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--max-eval-records", type=int, default=0, help="0 = full held-out split")
    parser.add_argument("--threads", type=int, help="OpenMP threads; default: positive OMP_NUM_THREADS or host CPU count")
    parser.add_argument("--omp-wait-policy", choices=("active", "passive"),
                        help="OpenMP worker wait policy; default: valid OMP_WAIT_POLICY or active")
    parser.add_argument("--tokenizer-timeout-seconds", type=int, default=7200)
    parser.add_argument("--train-timeout-seconds", type=int, default=18000,
                        help="timeout per native train call (per chunk when --chunks > 1)")
    parser.add_argument("--chunks", type=int, default=1,
                        help="split the epoch into N train-chunk calls; checkpoint writes are in place "
                             "and are not crash-atomic")
    parser.add_argument("--evaluate-timeout-seconds", type=int, default=7200)
    parser.add_argument("--generation-timeout-seconds", type=int, default=600)
    parser.add_argument("--prompts-file", help="optional UTF-8 file, one prompt per line (max 1 MiB / 1000 prompts)")
    args = parser.parse_args(argv)
    # Fail before tokenization or directory creation: native config errors can
    # otherwise waste an expensive tokenizer pass and overwrite prior outputs.
    max_i32, max_i64 = 2**31 - 1, 2**63 - 1
    if not 256 <= args.vocab <= max_i32 or not 1 <= args.min_pair_frequency <= max_i32:
        parser.error("--vocab must be >= 256 and --min-pair-frequency must be positive int32")
    # Match NRD_MAX_WIDTH: beyond this conservative bound int64 resource
    # estimates cannot be guaranteed for all supported vocabulary/FFN sizes.
    if (args.width < 2 or args.width > 16_777_216 or args.heads < 1 or args.heads > max_i32
            or args.width % args.heads or (args.width // args.heads) % 2
            or args.ffn < args.width or args.ffn > max_i32 or not 1 <= args.window <= 4096):
        parser.error("invalid model dimensions: width 2..16777216, width/heads needs an even head width, "
                     "ffn >= width, window 1..4096")
    if not 1 <= args.gradient_accumulation <= 512:
        parser.error("--gradient-accumulation must be in 1..512")
    if not 1 <= args.chunks <= max_i32:
        parser.error("--chunks must be a positive int32")
    if args.threads is not None and not 1 <= args.threads <= max_i32:
        parser.error("--threads must be a positive int32")
    env_threads = os.environ.get("OMP_NUM_THREADS", "").strip()
    inherited_threads = int(env_threads) if env_threads.isascii() and env_threads.isdecimal() and len(env_threads) <= 10 else 0
    threads = args.threads or (inherited_threads if 1 <= inherited_threads <= max_i32 else (os.cpu_count() or 2))
    env_wait_policy = os.environ.get("OMP_WAIT_POLICY", "").strip().upper()
    wait_policy = (args.omp_wait_policy.upper() if args.omp_wait_policy else
                   (env_wait_policy if env_wait_policy in {"ACTIVE", "PASSIVE"} else "ACTIVE"))
    if not 1 <= args.generation_tokens <= 512 or not 0 <= args.max_eval_records <= max_i64:
        parser.error("--generation-tokens must be 1..512 and --max-eval-records a nonnegative int64")
    if not 0 <= args.warmup_updates <= max_i64 or not 0 <= args.total_updates <= max_i64:
        parser.error("schedule horizons must be nonnegative int64 values")
    if not -(2**63) <= args.seed <= max_i64 or not -(2**63) <= args.generation_seed <= max_i64:
        parser.error("seeds must be int64 values")
    if any(value <= 0 for value in (args.tokenizer_timeout_seconds, args.train_timeout_seconds,
                                     args.evaluate_timeout_seconds, args.generation_timeout_seconds)):
        parser.error("timeouts must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be positive and finite")
    if (not math.isfinite(args.beta1) or not 0 <= args.beta1 < 1
            or not math.isfinite(args.beta2) or not 0 <= args.beta2 < 1
            or not math.isfinite(args.adam_eps) or args.adam_eps <= 0):
        parser.error("--beta1/--beta2 must be finite in [0, 1), and --adam-eps must be positive and finite")
    # Native optimizer arithmetic uses real32 parameters. Reject scalar
    # narrowing/first-step overflow before spending time on tokenization.
    real32_min, real32_max = float.fromhex("0x1p-126"), float.fromhex("0x1.fffffep+127")
    if not real32_min <= args.lr <= real32_max:
        parser.error("--lr must fit the normal positive real32 range")
    if args.optimizer in {"adamw", "nova"}:
        if not real32_min <= args.adam_eps <= real32_max:
            parser.error("--adam-eps must fit the normal positive real32 range")
        if args.optimizer == "adamw" and args.lr >= real32_max * (1.0 - args.beta1):
            parser.error("--lr overflows real32 after the AdamW first-step bias correction")
        if args.optimizer == "nova" and args.adam_eps < 1.0 and args.lr >= real32_max * args.adam_eps:
            parser.error("--lr / --adam-eps can overflow real32 for the NOVA step")
    if (not math.isfinite(args.label_smoothing) or not 0 <= args.label_smoothing < 0.5
            or not math.isfinite(args.min_lr_ratio) or not 0 <= args.min_lr_ratio <= 1):
        parser.error("--label-smoothing must be in [0, 0.5), and --min-lr-ratio in [0, 1]")
    if args.optimizer == "sgd":
        if args.weight_decay not in (None, 0.0):
            # parser.error exits with a clean usage message; the native CLI
            # rejects the same combination, so fail here rather than after the
            # tokenizer and corpus work has already been spent.
            parser.error("--weight-decay requires --optimizer nova or adamw: the SGD update has no decay term")
        args.weight_decay = 0.0
    elif args.weight_decay is None:
        args.weight_decay = 0.01
    if (not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or args.lr * args.weight_decay >= 1):
        parser.error("--weight-decay must be nonnegative and finite, with lr * weight_decay < 1")

    # Token-unit horizons -> optimizer-step horizons.  Keeping the conversion
    # here means the workflow can keep expressing budgets in tokens while the
    # native trainer only ever sees steps.
    warmup_steps = max(1, args.warmup_updates // args.gradient_accumulation) if args.warmup_updates else 0
    total_steps = args.total_updates // args.gradient_accumulation if args.total_updates else 0
    if (args.total_updates or args.schedule == "cosine") and total_steps <= warmup_steps:
        parser.error("--total-updates must exceed --warmup-updates in optimizer steps "
                     "(raise the budget or lower --warmup-updates)")

    corpus_dir = Path(args.corpus_dir)
    run_dir = Path(args.run_dir)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        lock_fd, lock_path = claim_run_directory(run_dir)
    except (OSError, RunError) as error:
        # Do not write failure.json into a conflicting or locked run directory.
        print(json.dumps({"schema": SCHEMA, "status": "failed", "error": str(error)}, ensure_ascii=False),
              file=sys.stderr)
        return 1
    manifest_path = corpus_dir / "manifest.json"
    try:
        # Resolve inside the protected region so broken paths cannot strand the lock.
        try:
            binary = str(Path(args.neural_bin).resolve())
        except RuntimeError as error:
            raise RunError(f"cannot resolve native binary: {error}") from error
        prompts = DEFAULT_PROMPTS
        if args.prompts_file:
            prompt_path = Path(args.prompts_file)
            prompt_stamp = file_stamp(prompt_path, "generation prompt file")
            with prompt_path.open("rb") as stream:
                prompt_bytes = stream.read(MAX_PROMPTS_BYTES + 1)
            if len(prompt_bytes) > MAX_PROMPTS_BYTES:
                raise RunError("generation prompt file exceeds 1 MiB")
            if file_stamp(prompt_path, "generation prompt file") != prompt_stamp:
                raise RunError("generation prompt file changed while being read")
            prompts = [line.strip() for line in prompt_bytes.decode("utf-8").splitlines() if line.strip()]
        if (not prompts or len(prompts) > MAX_PROMPTS
                or any("\x00" in prompt or len(prompt.encode("utf-8")) > 8192 for prompt in prompts)):
            raise RunError("generation prompts must be nonempty, at most 1000 lines and at most 8192 UTF-8 bytes each")
        if args.generation_seed + len(prompts) - 1 > max_i64:
            raise RunError("generation seed plus prompt index exceeds int64")
        manifest_stamp = file_stamp(manifest_path, "corpus manifest")
        with manifest_path.open("rb") as stream:
            manifest_bytes = stream.read(MAX_MANIFEST_BYTES + 1)
        if file_stamp(manifest_path, "corpus manifest") != manifest_stamp:
            raise RunError("corpus manifest changed while being read")
        if len(manifest_bytes) > MAX_MANIFEST_BYTES:
            raise RunError("corpus manifest exceeds 16 MiB")
        try:
            manifest_text = manifest_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise RunError("corpus manifest is not UTF-8") from error
        manifest = strict_json_object(manifest_text, "corpus manifest")
        input_paths, snapshots = corpus_inputs(manifest, corpus_dir)
        snapshots.append((manifest_path, "corpus manifest", manifest_stamp))
        manifest_identity = {"path": str(manifest_path), "bytes": len(manifest_bytes),
                             "sha256": hashlib.sha256(manifest_bytes).hexdigest()}
        binary_path = Path(binary)
        binary_stamp = file_stamp(binary_path, "native binary")
        if not os.access(binary_path, os.X_OK):
            raise RunError("native binary is not executable")
        binary_identity = sha256_file(binary_path)
        if file_stamp(binary_path, "native binary") != binary_stamp:
            raise RunError("native binary changed while being hashed")
        snapshots.append((binary_path, "native binary", binary_stamp))

        tokenizer_path = run_dir / "tokenizer.bpe"
        checkpoint_path = run_dir / "checkpoint.vgnr"
        for label, path in [("tokenizer", tokenizer_path), ("checkpoint", checkpoint_path), *input_paths.items()]:
            require_native_path(path, label)
        model_options = [
            "--width", str(args.width), "--heads", str(args.heads), "--window", str(args.window),
            "--ffn", str(args.ffn), "--seed", str(args.seed), "--lr", repr(args.lr),
            "--optimizer", args.optimizer, "--schedule", args.schedule,
            "--warmup-updates", str(warmup_steps),
            "--label-smoothing", repr(args.label_smoothing),
            "--min-lr-ratio", repr(args.min_lr_ratio),
            "--beta1", repr(args.beta1), "--beta2", repr(args.beta2), "--adam-eps", repr(args.adam_eps),
            "--gradient-accumulation", str(args.gradient_accumulation),
        ]
        if args.optimizer in {"nova", "adamw"}:
            model_options += ["--weight-decay", repr(args.weight_decay)]
        if total_steps:
            model_options += ["--total-updates", str(total_steps)]

        stdout, tokenizer_wall = run_child(
            [binary, "tokenizer-train-fast", str(input_paths["tokenizer_sample"]),
             str(tokenizer_path), str(args.vocab), str(args.min_pair_frequency)],
            timeout=args.tokenizer_timeout_seconds,
        )
        tokenizer_report = parse_prefixed_json(stdout, "BPE_TRAIN_JSON:")
        if tokenizer_report.get("schema_version") != "vegai-byte-bpe-train-fast/v1":
            raise RunError("unexpected native tokenizer schema")
        actual_vocab = metric_count(tokenizer_report, "actual_vocabulary_size", minimum=256)
        if (metric_count(tokenizer_report, "target_vocabulary_size") != args.vocab or actual_vocab > args.vocab
                or metric_count(tokenizer_report, "merges_added") != actual_vocab - 256):
            raise RunError("native tokenizer reported inconsistent vocabulary counts")
        metric_count(tokenizer_report, "records_scanned", minimum=1)
        metric_count(tokenizer_report, "retained_input_bytes", minimum=1)
        snapshots.append((tokenizer_path, "trained tokenizer", file_stamp(tokenizer_path, "trained tokenizer")))

        stdout, _ = run_child(
            [binary, "init-checkpoint", str(tokenizer_path), str(checkpoint_path), *model_options],
            timeout=args.evaluate_timeout_seconds,
        )
        init_report = parse_prefixed_json(stdout, "NEURAL_INIT_JSON:")
        if (init_report.get("schema_version") != "vegai-neural-init-checkpoint/v1"
                or init_report.get("tokenizer") != str(tokenizer_path)
                or init_report.get("checkpoint") != str(checkpoint_path)
                or metric_count(init_report, "vocabulary_size") != actual_vocab
                or metric_count(init_report, "width") != args.width
                or metric_count(init_report, "heads") != args.heads
                or metric_count(init_report, "local_window") != args.window
                or metric_count(init_report, "ffn_width") != args.ffn):
            raise RunError("native checkpoint initialization does not match the requested model")

        heldout_path = str(input_paths["heldout"])
        heldout_records = manifest["outputs"]["heldout"]["records"]
        expected_eval_records = min(heldout_records, args.max_eval_records) if args.max_eval_records else heldout_records
        # Both choices are configurable and recorded: active waiting can
        # reduce wake latency but can also consume CPU/energy while waiting.
        omp_env = {"OMP_NUM_THREADS": str(threads), "OMP_WAIT_POLICY": wait_policy}

        def evaluate() -> dict[str, Any]:
            command = [binary, "evaluate", str(checkpoint_path), str(tokenizer_path), heldout_path]
            if args.max_eval_records:
                command.append(str(args.max_eval_records))
            text, _wall = run_child(command, timeout=args.evaluate_timeout_seconds, env_extra=omp_env)
            report = parse_prefixed_json(text, "NEURAL_EVALUATE_JSON:")
            validate_evaluation_budget(report, expected_eval_records)
            return report

        baseline = evaluate()
        if metric_count(baseline, "updates_completed_at_checkpoint") != 0:
            raise RunError("untrained baseline unexpectedly contains optimizer updates")

        verify_stamps(snapshots)
        epoch_file = str(input_paths["train_epoch"])
        train_chunks: list[dict[str, Any]] = []
        if args.chunks <= 1:
            stdout, train_wall = run_child(
                [binary, "train", epoch_file,
                 str(checkpoint_path), str(tokenizer_path), "1", *model_options],
                timeout=args.train_timeout_seconds,
                env_extra=omp_env,
            )
            epoch_report = parse_prefixed_json(stdout, "NEURAL_PRETRAIN_EPOCH_JSON:")
            complete_report = parse_prefixed_json(stdout, "NEURAL_PRETRAIN_COMPLETE_JSON:")
        else:
            # Chunked single epoch: repeated train-chunk calls advance the
            # checkpoint's own source-record cursor with the same options and
            # online order as one `train` call, saving after every chunk.
            # Saves currently replace the checkpoint in place: an interruption
            # during a write can also damage the previous checkpoint. The
            # progress report is not an atomic transaction with the checkpoint.
            epoch_records = int(manifest["outputs"]["train_epoch"]["records"])
            chunk_records = max(1, -(-epoch_records // args.chunks))
            train_wall = 0.0
            completed_records = 0
            previous_updates = 0
            guard = 0
            while completed_records < epoch_records:
                guard += 1
                if guard > args.chunks + 3:
                    raise RunError(
                        f"chunked training stalled at {completed_records}/{epoch_records} source records")
                stdout, chunk_wall = run_child(
                    [binary, "train-chunk", epoch_file,
                     str(checkpoint_path), str(tokenizer_path), str(chunk_records), *model_options],
                    timeout=args.train_timeout_seconds,
                    env_extra=omp_env,
                )
                chunk = parse_prefixed_json(stdout, "NEURAL_PRETRAIN_CHUNK_JSON:")
                if chunk.get("schema_version") != "vegai-neural-pretraining-chunk/v1":
                    raise RunError("unexpected native chunk schema")
                start_record = metric_count(chunk, "start_source_record")
                end_record = metric_count(chunk, "source_records_completed")
                updates = metric_count(chunk, "updates_completed")
                if (start_record != completed_records or not completed_records < end_record <= epoch_records
                        or end_record - start_record > chunk_records or updates < previous_updates
                        or metric_count(chunk, "eligible_records") > end_record - start_record):
                    raise RunError("native chunk reported inconsistent record/update cursors")
                metric_count(chunk, "prediction_tokens")
                metric_number(chunk, "mean_nll")
                top1 = metric_number(chunk, "top1_accuracy", maximum=1.0)
                top5 = metric_number(chunk, "top5_accuracy", maximum=1.0)
                if top1 > top5:
                    raise RunError("native chunk top-1 accuracy exceeds top-5")
                chunk["controller_wall_seconds"] = chunk_wall
                train_chunks.append(chunk)
                (run_dir / "chunks.json").write_text(
                    json.dumps(train_chunks, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                train_wall += chunk_wall
                completed_records, previous_updates = end_record, updates
            total_tokens = sum(int(c["prediction_tokens"]) for c in train_chunks)
            total_eligible = sum(int(c["eligible_records"]) for c in train_chunks)

            def weighted(key: str) -> float:
                if total_tokens <= 0:
                    return 0.0
                return sum(float(c[key]) * int(c["prediction_tokens"]) for c in train_chunks) / total_tokens

            mean_nll = weighted("mean_nll")
            epoch_report = {
                "schema_version": "vegai-neural-pretraining-chunk-aggregate/v1",
                "mode": "tied-output-cross-entropy-adapter/v1",
                "epoch_in_invocation": 1,
                "chunks": len(train_chunks),
                "optimizer": args.optimizer,
                "schedule": args.schedule,
                "learning_rate": args.lr,
                "label_smoothing": args.label_smoothing,
                "eligible_records": total_eligible,
                "skipped_records_derived": completed_records - total_eligible,
                "source_records_completed": completed_records,
                "prediction_tokens": total_tokens,
                "mean_nll": mean_nll,
                "perplexity": safe_perplexity(mean_nll),
                "top1_accuracy": weighted("top1_accuracy"),
                "top5_accuracy": weighted("top5_accuracy"),
                "epoch_wall_seconds": train_wall,
                "prediction_tokens_per_second": (total_tokens / train_wall) if train_wall > 0 else 0.0,
                "timing_basis": "runner wall per chunk including checkpoint save/load between chunks",
                "updates_completed": int(train_chunks[-1]["updates_completed"]) if train_chunks else 0,
            }
            complete_report = {
                "schema_version": "vegai-neural-pretraining-chunk-aggregate/v1",
                "epochs_completed_in_invocation": 1,
                "chunks": len(train_chunks),
                "updates_completed": epoch_report["updates_completed"],
                "prediction_tokens": total_tokens,
                "tokens_seen": total_tokens + total_eligible,
                "records_completed": total_eligible,
                "source_records_completed": completed_records,
            }
        if metric_count(epoch_report, "epoch_in_invocation", minimum=1) != 1:
            raise RunError("native trainer did not report exactly epoch 1")
        token_count = metric_count(epoch_report, "prediction_tokens", minimum=1)
        eligible_records = metric_count(epoch_report, "eligible_records", minimum=1)
        metric_number(epoch_report, "mean_nll")
        top1 = metric_number(epoch_report, "top1_accuracy", maximum=1.0)
        top5 = metric_number(epoch_report, "top5_accuracy", maximum=1.0)
        if top1 > top5:
            raise RunError("native training top-1 accuracy exceeds top-5")
        metric_number(epoch_report, "prediction_tokens_per_second")
        updates = metric_count(complete_report, "updates_completed", minimum=1)
        epoch_records = manifest["outputs"]["train_epoch"]["records"]
        if (metric_count(complete_report, "source_records_completed") != epoch_records
                or metric_count(complete_report, "records_completed") != eligible_records
                or metric_count(complete_report, "tokens_seen") != token_count + eligible_records
                or not eligible_records <= token_count or updates > token_count):
            raise RunError("native training did not consume the advertised epoch records/tokens")
        if args.chunks <= 1:
            skipped = metric_count(epoch_report, "skipped_records")
            if skipped + eligible_records != epoch_records or metric_count(epoch_report, "tokenizer_truncations") > skipped:
                raise RunError("native training skip/truncation counters are inconsistent")
        verify_stamps(snapshots)
        snapshots.append((checkpoint_path, "trained checkpoint", file_stamp(checkpoint_path, "trained checkpoint")))

        final = evaluate()
        if (any(metric_count(final, field) != metric_count(baseline, field)
                for field in ("prediction_tokens", "eligible_records", "skipped_records", "tokenizer_truncations"))
                or metric_count(final, "updates_completed_at_checkpoint") != updates):
            raise RunError("held-out budget or checkpoint update count changed unexpectedly")

        generations = []
        for index, prompt in enumerate(prompts):
            text, _wall = run_child(
                [binary, "generate", str(checkpoint_path), str(tokenizer_path), prompt,
                 str(args.generation_tokens), str(args.generation_seed + index)],
                timeout=args.generation_timeout_seconds, env_extra=omp_env,
            )
            report = parse_prefixed_json(text, "NEURAL_GENERATE_JSON:")
            if (report.get("schema_version") != "vegai-neural-generate/v1"
                    or metric_count(report, "prompt_tokens", minimum=1) > 8192
                    or metric_count(report, "generated_tokens", minimum=1) != args.generation_tokens
                    or type(report.get("initial_seed")) is not int
                    or report["initial_seed"] != args.generation_seed + index):
                raise RunError("native generation report does not match the requested budget or seed")
            response = extract_generation(text)
            generations.append({
                "prompt": prompt,
                "response": response,
                "response_contains_replacement_character": "\ufffd" in response,
                "seed": args.generation_seed + index,
                "generated_tokens_reported": report["generated_tokens"],
            })

        verify_stamps(snapshots)
        tokenizer_identity = sha256_file(tokenizer_path)
        checkpoint_identity = sha256_file(checkpoint_path)
        verify_stamps(snapshots)
        evidence: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "completed",
            "exact_epochs": 1,
            "native_binary": binary_identity,
            "corpus": {
                "manifest_path": str(manifest_path),
                "manifest_file": manifest_identity,
                "manifest": manifest,
            },
            "config": {
                "vocab": args.vocab,
                "min_pair_frequency": args.min_pair_frequency,
                "width": args.width,
                "heads": args.heads,
                "window": args.window,
                "ffn": args.ffn,
                "lr": args.lr,
                "optimizer": args.optimizer,
                "schedule": args.schedule,
                "warmup_updates": warmup_steps,
                "total_updates": total_steps,
                "warmup_updates_token_units": args.warmup_updates,
                "total_updates_token_units": args.total_updates,
                "gradient_accumulation": args.gradient_accumulation,
                "min_lr_ratio": args.min_lr_ratio,
                "label_smoothing": args.label_smoothing,
                "weight_decay": args.weight_decay,
                "weight_decay_scope": (f"{args.optimizer} embeddings only" if args.optimizer != "sgd"
                                       else "not applicable (sgd)"),
                "adam_beta1": args.beta1,
                "adam_beta2": args.beta2,
                "adam_epsilon": args.adam_eps,
                "seed": args.seed,
                "chunks": args.chunks,
                "max_eval_records": args.max_eval_records,
                "generation_tokens": args.generation_tokens,
                "generation_seed": args.generation_seed,
                "threads": threads,
                "omp_wait_policy": wait_policy,
            },
            "tokenizer": {**tokenizer_report, "file": tokenizer_identity,
                          "controller_wall_seconds": tokenizer_wall},
            "init": init_report,
            "baseline_heldout": baseline,
            "training_epoch": epoch_report,
            "training_chunks": train_chunks,
            "training_complete": complete_report,
            "training_controller_wall_seconds": train_wall,
            "final_heldout": final,
            "checkpoint": checkpoint_identity,
            "generations": generations,
            "scale_projection": project_99b(
                float(epoch_report["prediction_tokens_per_second"]),
                int(epoch_report["prediction_tokens"]),
            ),
            "limitations": [
                "Yalnız bağlı çıktı katmanı eğitilir; dikkat/retention/FFN parametreleri donuktur. Bu tam BPTT değildir.",
                "Epoch korpusu, manifestte hash ile tanımlı train_epoch dilimidir; tam kabul edilen korpus daha büyüktür ve train-chunk ile devam edilebilir.",
                "Üretimler nedensel devamdır; talimat takip, olgusal doğruluk veya genel Türkçe asistan kalitesi iddia edilmez.",
                "Held-out, manifestin belirttiği belge-ayrık bölmedir; koşucu tüm kayıtların gerçekten ayrık olduğunu bağımsız kanıtlamaz.",
                "Girdi stat kontrolleri olağan değişiklikleri yakalar; eşzamanlı kötü niyetli değişmezlik garantisi değildir.",
                "Üretimde geçersiz UTF-8 baytları değiştirme karakteriyle gösterilir; bu ham bayt kaydı değildir.",
                "Checkpoint yerinde yazılır; kesintiye veya güç kaybına dayanıklı atomik yedek değildir.",
                "99B tamamlama süresi ölçülen hızdan aritmetik projeksiyondur; dağıtık GPU eğitimi bu repoda yoktur.",
            ],
        }
        # Render/serialize before publication. Publish the completed evidence
        # last so a partial report cannot be mistaken for a completed run.
        evidence_text = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        results_text = render_results_md(evidence)
        publish_results(run_dir, results_text, evidence_text)
    except (RunError, OSError, KeyError, ValueError) as error:
        failure = {"schema": SCHEMA, "status": "failed", "error": str(error)}
        (run_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        release_run_directory(lock_fd, lock_path)
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
