#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run one native neural pretraining epoch with validation checkpoint selection.

Orchestrates only the native Fortran ``vegai_neural_pretrain`` binary:

1. ``tokenizer-train-fast`` on the manifest's training-only tokenizer sample;
2. ``init-checkpoint`` and an untrained no-update validation baseline;
3. ``train`` for exactly **one epoch** over the manifest's ``train_epoch``
   slice (the slice definition and its hash are part of the evidence);
4. select the initial or trained checkpoint by validation NLL only;
5. evaluate the selected checkpoint once on the document-disjoint held-out test;
6. ``generate`` from that selected checkpoint for a fixed prompt suite.

The report distinguishes three numbers that are often conflated: tokens
actually trained in this run, tokens accepted by the full corpus pipeline,
and the 99-billion-token program target.  The 99B completion time below is
an arithmetic projection from the *measured* native throughput on this
host, labelled as a projection -- never as an achieved result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

SCHEMA = "vegai-neural-turkish-epoch1/v2"
TARGET_TOKENS_99B = 99_000_000_000

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


def sha256_file(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
            size += len(block)
    return {"path": str(path), "bytes": size, "sha256": hasher.hexdigest()}


def parse_prefixed_json(stdout: str, prefix: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix):].strip())
    raise RunError(f"missing {prefix} line in native output:\n{stdout[-2000:]}")


def validation_improves(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    """Strict validation-NLL comparison; ties keep the simpler/earlier checkpoint."""
    candidate_nll = float(candidate.get("mean_nll", float("nan")))
    incumbent_nll = float(incumbent.get("mean_nll", float("nan")))
    if not math.isfinite(candidate_nll) or not math.isfinite(incumbent_nll):
        raise RunError("checkpoint selection requires finite validation NLL values")
    return candidate_nll < incumbent_nll


def atomic_copy(source: Path, destination: Path) -> None:
    """Publish a selected checkpoint atomically without destroying the incumbent on copy failure."""
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run_child(argv: list[str], timeout: int, env_extra: dict[str, str] | None = None) -> tuple[str, float]:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    started = time.monotonic()
    completed = subprocess.run(argv, capture_output=True, timeout=timeout, check=False, env=env)
    elapsed = time.monotonic() - started
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise RunError(
            f"native command failed rc={completed.returncode}: {' '.join(argv)}\n"
            f"stdout tail:\n{stdout[-3000:]}\nstderr tail:\n{stderr[-3000:]}"
        )
    return stdout, elapsed


def extract_generation(stdout: str) -> str:
    begin = stdout.find("NEURAL_GENERATION_BEGIN")
    end = stdout.find("NEURAL_GENERATION_END")
    if begin < 0 or end < 0 or end <= begin:
        raise RunError(f"generation markers missing:\n{stdout[-2000:]}")
    body = stdout[begin + len("NEURAL_GENERATION_BEGIN"):end]
    return body.strip("\n")


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
        result["projected_single_host_seconds_for_99b"] = seconds
        result["projected_single_host_days_for_99b"] = seconds / 86_400
        result["projected_single_host_years_for_99b"] = seconds / (86_400 * 365.25)
    return result


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if value != 0 and (abs(value) < 0.001 or abs(value) >= 1e9):
            return f"{value:.6g}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def render_results_md(evidence: dict[str, Any]) -> str:
    corpus = evidence["corpus"]
    funnel = corpus["manifest"]["funnel"]
    outputs = corpus["manifest"]["outputs"]
    final = evidence["heldout_test"]
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
    lines.append(f"- Kaynak: `{source.get('file')}` ({source.get('kind')}), lisans: {source.get('license')}")
    lines.append(f"- Kaynak SHA-256: `{source.get('sha256')}`")
    lines.append(f"- Kaynak belge: {funnel['source_documents']:,}; kabul: {funnel['accepted_documents']:,}; "
                 f"ret: {funnel['rejected_total']:,}; yinelenen belge: {funnel['duplicate_documents']:,}; "
                 f"yinelenen paragraf: {funnel['duplicate_paragraphs']:,}")
    lines.append(f"- Ret hunisi: `{json.dumps(funnel['rejected'], ensure_ascii=False, sort_keys=True)}`")
    for name in ("train", "validation", "heldout", "train_epoch", "tokenizer_sample"):
        meta = outputs[name]
        lines.append(f"- `{meta['path']}`: {meta['bytes']:,} bayt, {meta['records']:,} kayıt, "
                     f"~{meta['approx_words']:,} kelime, SHA-256 `{meta['sha256']}`")
    cfg = evidence["config"]
    max_docs_text = f"{cfg['max_docs']:,}" if cfg["max_docs"] > 0 else "sınırsız"
    train_budget_text = f"{cfg['train_byte_budget']:,} bayt" if cfg["train_byte_budget"] > 0 else "tüm train split'i"
    lines.append(
        f"- Veri sınırları: belge üst sınırı {max_docs_text}; epoch bütçesi {train_budget_text} "
        f"(gerçek epoch dilimi {cfg['actual_train_epoch_bytes']:,} bayt); tokenizer örnek üst sınırı "
        f"{cfg['tokenizer_sample_bytes']:,} bayt (gerçek örnek {cfg['actual_tokenizer_sample_bytes']:,} bayt)."
    )
    lines.append("")
    lines.append("## Model ve eğitim (epoch = 1)")
    lines.append("")
    actual_vocab = evidence.get("tokenizer", {}).get("actual_vocabulary_size", cfg['vocab'])
    lines.append(
        f"- Mimari: byte-BPE vocab {actual_vocab:,} (istenen {cfg['vocab']:,}; minimum pair frequency "
        f"{cfg['min_pair_frequency']}), width {cfg['width']}, attention heads {cfg['heads']} "
        f"(head width {cfg['head_width']}), local window {cfg['window']}, FFN {cfg['ffn']} "
        "(bağlı-çıktı katmanı eğitilir; dikkat/retention çekirdeği donuktur)"
    )
    lines.append(
        f"- Optimize edici: {cfg['update_system']}, tepe LR {cfg['lr']}, zamanlama {cfg['schedule']} "
        f"(warmup {cfg['warmup_updates']:,}), label smoothing {cfg['label_smoothing']}, seed {cfg['seed']}"
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
    lines.append(f"- Validation NLL: başlangıç {fmt(evidence['baseline_validation']['mean_nll'])}; "
                 f"epoch 1 sonrası {fmt(evidence['epoch_validation']['mean_nll'])}; "
                 f"seçilen checkpoint epoch {evidence['selected_epoch']} "
                 "(0 = eğitimsiz başlangıç; NLL iyileşmezse daha kötü checkpoint seçilmez).")
    lines.append("")
    lines.append("## Held-out test sonucu (belge-düzeyinde ayrık, seçimde kullanılmadı)")
    lines.append("")
    lines.append("| Ölçüm | Validation-seçilmiş checkpoint (held-out test) |")
    lines.append("|---|---:|")
    for label, key in (("Ortalama NLL", "mean_nll"), ("Perplexity", "perplexity"),
                       ("Top-1 doğruluk", "top1_accuracy"), ("Top-5 doğruluk", "top5_accuracy")):
        lines.append(f"| {label} | {fmt(final.get(key))} |")
    lines.append(f"| Held-out tahmin tokenı | {final['prediction_tokens']:,} |")
    lines.append("")
    lines.append("## Prompt → Response (aynen, düzenlenmeden)")
    lines.append("")
    lines.append("Model bir talimat modeli değildir; aşağıdaki yanıtlar nedensel devam üretimidir ve "
                 "daha iyi görünsün diye seçilip düzenlenmemiştir.")
    lines.append("")
    lines.append("| # | Prompt | Response |")
    lines.append("|---:|---|---|")
    for index, item in enumerate(evidence["generations"], start=1):
        response = item["response"].replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {index} | `{item['prompt']}` | `{response}` |")
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
    for assumption in projection["assumptions"]:
        lines.append(f"- Varsayım: {assumption}")
    lines.append("")
    lines.append("## Sınırlar")
    lines.append("")
    for limitation in evidence["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def validate_arguments(args: argparse.Namespace) -> None:
    if args.vocab < 256 or args.min_pair_frequency < 1:
        raise RunError("vocab must be at least the byte alphabet and minimum pair frequency positive")
    if args.width < 2 or args.heads < 1 or args.width % args.heads != 0:
        raise RunError("model width must be positive and divisible by the attention-head count")
    head_width = args.width // args.heads
    if head_width % 2 != 0:
        raise RunError("per-head width must be even for the decoder's paired rotary features")
    if args.window < 1 or args.window > 4096:
        raise RunError("window must be in [1, 4096]")
    if args.ffn < args.width:
        raise RunError("FFN width must be at least model width")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise RunError("learning rate must be finite and positive")
    if args.warmup_updates < 0 or args.total_updates < 0:
        raise RunError("warmup and total updates must be non-negative")
    if args.schedule == "cosine" and args.total_updates <= args.warmup_updates:
        raise RunError("cosine schedule requires total updates greater than warmup updates")
    if not math.isfinite(args.min_lr_ratio) or not 0.0 <= args.min_lr_ratio <= 1.0:
        raise RunError("minimum LR ratio must be finite and in [0, 1]")
    if not math.isfinite(args.label_smoothing) or not 0.0 <= args.label_smoothing < 0.5:
        raise RunError("label smoothing must be finite and in [0, 0.5)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise RunError("weight decay must be finite and non-negative")
    if args.chunks < 1 or args.max_eval_records < 0:
        raise RunError("chunks must be positive and max-eval-records must be non-negative")
    if args.generation_tokens < 1 or args.generation_tokens > 512:
        raise RunError("generation-tokens must be in [1, 512]")
    for label in ("tokenizer_timeout_seconds", "train_timeout_seconds",
                  "evaluate_timeout_seconds", "generation_timeout_seconds"):
        if getattr(args, label) <= 0:
            raise RunError(f"{label.replace('_', '-')} must be positive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-bin", required=True)
    parser.add_argument("--corpus-dir", required=True, help="directory written by build_turkish_wiki_corpus.py")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--min-pair-frequency", type=int, default=8)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--ffn", type=int, default=384)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--schedule", default="invsqrt", choices=("constant", "invsqrt", "cosine"))
    parser.add_argument("--warmup-updates", type=int, default=2000)
    parser.add_argument("--total-updates", type=int, default=0,
                        help="cosine horizon in updates (estimate from measured bytes/token); 0 = omit")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--generation-seed", type=int, default=7)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--max-eval-records", type=int, default=0, help="0 = full validation and held-out splits")
    parser.add_argument("--tokenizer-timeout-seconds", type=int, default=7200)
    parser.add_argument("--train-timeout-seconds", type=int, default=18000,
                        help="timeout per native train call (per chunk when --chunks > 1)")
    parser.add_argument("--chunks", type=int, default=1,
                        help="split the single epoch into N train-chunk calls with a checkpoint "
                             "save after each; identical online order/config as a single train call, "
                             "but a timeout loses at most the in-flight chunk")
    parser.add_argument("--evaluate-timeout-seconds", type=int, default=7200)
    parser.add_argument("--generation-timeout-seconds", type=int, default=600)
    parser.add_argument("--prompts-file", help="optional UTF-8 file, one prompt per line")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_dir / "manifest.json"
    binary = str(Path(args.neural_bin).resolve())
    try:
        validate_arguments(args)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "vegai-turkish-wiki-corpus/v4":
            raise RunError("corpus manifest schema must provide the v4 stratified byte-budget sample and document-level split")
        split = manifest.get("split", {})
        expected_split_rule = (
            "sha256(vegai-trwiki-split-v2|page_id|title) mod 200: "
            "0 -> heldout_test, 1 -> validation, 2..199 -> train"
        )
        if split.get("level") != "source-document" or split.get("rule") != expected_split_rule or \
                split.get("selection_uses") != "validation only" or \
                split.get("final_reporting_uses") != "heldout test only":
            raise RunError("corpus manifest does not attest the required document-level validation/test split")
        corpus_root = corpus_dir.resolve()
        output_paths: dict[str, Path] = {}
        for key in ("train", "validation", "heldout", "train_epoch", "tokenizer_sample"):
            meta = manifest["outputs"][key]
            output_path = (corpus_root / meta["path"]).resolve()
            if output_path.parent != corpus_root or not output_path.is_file():
                raise RunError(f"corpus output {key} must be an existing direct child of the corpus directory")
            output_paths[key] = output_path
            actual = sha256_file(output_path)
            if actual["sha256"] != meta["sha256"] or actual["bytes"] != meta["bytes"]:
                raise RunError(f"corpus output {key} does not match manifest hash/bytes")
        train_meta = manifest["outputs"]["train"]
        validation_meta = manifest["outputs"]["validation"]
        heldout_meta = manifest["outputs"]["heldout"]
        partitions = (train_meta, validation_meta, heldout_meta)
        if any(int(meta["records"]) <= 0 for meta in partitions):
            raise RunError("train, validation, and final held-out partitions must all contain records")
        if len({meta["path"] for meta in partitions}) != 3:
            raise RunError("train, validation, and final held-out paths must be distinct")
        if len({meta["sha256"] for meta in partitions}) != 3:
            raise RunError("train, validation, and final held-out partitions must have distinct content hashes")
        if int(manifest["outputs"]["train_epoch"]["records"]) <= 0:
            raise RunError("the configured training epoch slice is empty")

        tokenizer_path = run_dir / "tokenizer.bpe"
        checkpoint_path = run_dir / "checkpoint.vgnr"
        best_checkpoint_path = run_dir / "best_checkpoint.vgnr"
        model_options = [
            "--width", str(args.width), "--heads", str(args.heads), "--window", str(args.window),
            "--ffn", str(args.ffn), "--seed", str(args.seed), "--lr", repr(args.lr),
            "--schedule", args.schedule,
            "--warmup-updates", str(args.warmup_updates),
            "--label-smoothing", repr(args.label_smoothing), "--weight-decay", repr(args.weight_decay),
            "--min-lr-ratio", repr(args.min_lr_ratio),
        ]
        if args.total_updates:
            model_options += ["--total-updates", str(args.total_updates)]

        stdout, tokenizer_wall = run_child(
            [binary, "tokenizer-train-fast", str(output_paths["tokenizer_sample"]),
             str(tokenizer_path), str(args.vocab), str(args.min_pair_frequency)],
            timeout=args.tokenizer_timeout_seconds,
        )
        tokenizer_report = parse_prefixed_json(stdout, "BPE_TRAIN_JSON:")

        stdout, _ = run_child(
            [binary, "init-checkpoint", str(tokenizer_path), str(checkpoint_path), *model_options],
            timeout=args.evaluate_timeout_seconds,
        )
        init_report = parse_prefixed_json(stdout, "NEURAL_INIT_JSON:")

        validation_path = str(output_paths["validation"])
        heldout_path = str(output_paths["heldout"])
        # Per-token OpenMP regions are microseconds apart; the default wait
        # policy puts workers to sleep between them and wake latency then
        # dominates (measured in probe A/Bs 40301975287/40305596215).  An
        # active wait policy keeps the bounded thread team spinning for the
        # duration of these dedicated training/evaluation processes.
        omp_env = {"OMP_NUM_THREADS": str(os.cpu_count() or 2), "OMP_WAIT_POLICY": "ACTIVE"}

        def evaluate(selected_checkpoint: Path, corpus_path: str) -> dict[str, Any]:
            command = [binary, "evaluate", str(selected_checkpoint), str(tokenizer_path), corpus_path]
            if args.max_eval_records:
                command.append(str(args.max_eval_records))
            text, _wall = run_child(command, timeout=args.evaluate_timeout_seconds, env_extra=omp_env)
            report = parse_prefixed_json(text, "NEURAL_EVALUATE_JSON:")
            if int(report.get("prediction_tokens", 0)) <= 0:
                raise RunError(f"evaluation has no prediction tokens: {corpus_path}")
            if not math.isfinite(float(report.get("mean_nll", float("nan")))):
                raise RunError(f"evaluation mean NLL is non-finite: {corpus_path}")
            return report

        baseline_validation = evaluate(checkpoint_path, validation_path)
        best_validation = baseline_validation
        best_epoch = 0
        atomic_copy(checkpoint_path, best_checkpoint_path)

        epoch_file = str(output_paths["train_epoch"])
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
            # online order as one `train` call (v3 checkpoints round-trip
            # optimizer state bit-exactly), saving a checkpoint after every
            # chunk.  A subprocess timeout therefore loses only the chunk in
            # flight, and chunks.json in the run dir shows exact progress.
            epoch_records = int(manifest["outputs"]["train_epoch"]["records"])
            chunk_records = max(1, -(-epoch_records // args.chunks))
            train_wall = 0.0
            completed_records = 0
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
                chunk["controller_wall_seconds"] = chunk_wall
                train_chunks.append(chunk)
                (run_dir / "chunks.json").write_text(
                    json.dumps(train_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
                train_wall += chunk_wall
                completed_records = int(chunk["source_records_completed"])
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
                "update_system": "apex",
                "schedule": args.schedule,
                "learning_rate": args.lr,
                "label_smoothing": args.label_smoothing,
                "eligible_records": total_eligible,
                "skipped_records_derived": completed_records - total_eligible,
                "source_records_completed": completed_records,
                "prediction_tokens": total_tokens,
                "mean_nll": mean_nll,
                "perplexity": math.exp(mean_nll) if mean_nll < 700 else float("inf"),
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
            }
        if int(epoch_report["epoch_in_invocation"]) != 1:
            raise RunError("native trainer did not report exactly epoch 1")

        epoch_validation = evaluate(checkpoint_path, validation_path)
        if validation_improves(epoch_validation, best_validation):
            atomic_copy(checkpoint_path, best_checkpoint_path)
            best_validation = epoch_validation
            best_epoch = 1

        # Held-out scores are never consulted by the selector; this final
        # evaluation uses the checkpoint chosen solely on validation NLL.
        final = evaluate(best_checkpoint_path, heldout_path)

        prompts = DEFAULT_PROMPTS
        if args.prompts_file:
            prompts = [line.strip() for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        generations = []
        for index, prompt in enumerate(prompts):
            text, _wall = run_child(
                [binary, "generate", str(best_checkpoint_path), str(tokenizer_path), prompt,
                 str(args.generation_tokens), str(args.generation_seed + index)],
                timeout=args.generation_timeout_seconds,
            )
            report = parse_prefixed_json(text, "NEURAL_GENERATE_JSON:")
            generations.append({
                "prompt": prompt,
                "response": extract_generation(text),
                "seed": args.generation_seed + index,
                "generated_tokens_reported": report.get("generated_tokens"),
            })

        evidence: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "completed",
            "exact_epochs": 1,
            "native_binary": sha256_file(Path(binary)),
            "corpus": {
                "manifest_path": str(manifest_path),
                "manifest_file": sha256_file(manifest_path),
                "manifest": manifest,
            },
            "config": {
                "max_docs": manifest["filters"]["max_docs"],
                "train_byte_budget": manifest["train_byte_budget"],
                "actual_train_epoch_bytes": manifest["outputs"]["train_epoch"]["bytes"],
                "tokenizer_sample_bytes": manifest["tokenizer_sample_bytes"],
                "actual_tokenizer_sample_bytes": manifest["outputs"]["tokenizer_sample"]["bytes"],
                "vocab": args.vocab,
                "min_pair_frequency": args.min_pair_frequency,
                "width": args.width,
                "heads": args.heads,
                "head_width": args.width // args.heads,
                "window": args.window,
                "ffn": args.ffn,
                "lr": args.lr,
                "update_system": "apex",
                "schedule": args.schedule,
                "warmup_updates": args.warmup_updates,
                "total_updates": args.total_updates,
                "min_lr_ratio": args.min_lr_ratio,
                "label_smoothing": args.label_smoothing,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "chunks": args.chunks,
            },
            "tokenizer": {**tokenizer_report, "file": sha256_file(tokenizer_path),
                          "controller_wall_seconds": tokenizer_wall},
            "init": init_report,
            "baseline_validation": baseline_validation,
            "epoch_validation": epoch_validation,
            "best_validation": best_validation,
            "selected_epoch": best_epoch,
            "training_epoch": epoch_report,
            "training_chunks": train_chunks,
            "training_complete": complete_report,
            "training_controller_wall_seconds": train_wall,
            "heldout_test": final,
            "checkpoint": sha256_file(best_checkpoint_path),
            "last_checkpoint": sha256_file(checkpoint_path),
            "selected_checkpoint": str(best_checkpoint_path),
            "generations": generations,
            "scale_projection": project_99b(
                float(epoch_report["prediction_tokens_per_second"]),
                int(epoch_report["prediction_tokens"]),
            ),
            "limitations": [
                "Yalnız bağlı çıktı katmanı eğitilir; dikkat/retention/FFN parametreleri donuktur. Bu tam BPTT değildir.",
                "Epoch korpusu, manifestte hash ile tanımlı train_epoch dilimidir; tam kabul edilen korpus daha büyüktür ve train-chunk ile devam edilebilir.",
                "Üretimler nedensel devamdır; talimat takip, olgusal doğruluk veya genel Türkçe asistan kalitesi iddia edilmez.",
                "Validation bölümü checkpoint seçimi için kullanılır; held-out test bölümü yalnız seçimden sonra raporlanır.",
                "Held-out test aynı dump içinden belge-ayrık bölmedir; bağımsız insan-anotasyonlu benchmark değildir.",
                "99B tamamlama süresi ölçülen hızdan aritmetik projeksiyondur; dağıtık GPU eğitimi bu repoda yoktur.",
            ],
        }
        (run_dir / "evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_dir / "RESULTS.md").write_text(render_results_md(evidence), encoding="utf-8")
    except (RunError, OSError, KeyError, ValueError, subprocess.TimeoutExpired) as error:
        failure = {"schema": SCHEMA, "status": "failed", "error": str(error)}
        (run_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
