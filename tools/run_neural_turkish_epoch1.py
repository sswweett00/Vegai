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
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

SCHEMA = "vegai-neural-turkish-epoch1/v1"
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
    lines.append(f"- Kaynak: `{source.get('file')}` ({source.get('kind')}), lisans: {source.get('license')}")
    lines.append(f"- Kaynak SHA-256: `{source.get('sha256')}`")
    lines.append(f"- Kaynak belge: {funnel['source_documents']:,}; kabul: {funnel['accepted_documents']:,}; "
                 f"ret: {funnel['rejected_total']:,}; yinelenen belge: {funnel['duplicate_documents']:,}; "
                 f"yinelenen paragraf: {funnel['duplicate_paragraphs']:,}")
    lines.append(f"- Ret hunisi: `{json.dumps(funnel['rejected'], ensure_ascii=False, sort_keys=True)}`")
    for name in ("train", "heldout", "train_epoch", "tokenizer_sample"):
        meta = outputs[name]
        lines.append(f"- `{meta['path']}`: {meta['bytes']:,} bayt, {meta['records']:,} kayıt, "
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
    lines.append("## Held-out sonuç (belge-düzeyinde ayrık, eğitimde görülmedi)")
    lines.append("")
    lines.append("| Ölçüm | Eğitimsiz başlangıç | Epoch 1 sonrası |")
    lines.append("|---|---:|---:|")
    for label, key in (("Ortalama NLL", "mean_nll"), ("Perplexity", "perplexity"),
                       ("Top-1 doğruluk", "top1_accuracy"), ("Top-5 doğruluk", "top5_accuracy")):
        lines.append(f"| {label} | {fmt(baseline.get(key))} | {fmt(final.get(key))} |")
    lines.append(f"| Held-out tahmin tokenı | {baseline['prediction_tokens']:,} | {final['prediction_tokens']:,} |")
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
    parser.add_argument("--optimizer", default="adamw", choices=("adamw", "sgd"))
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
    # AdamW-only decoupled weight decay; the legacy SGD update has no decay
    # term, so an explicit value there is rejected instead of silently ignored.
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="AdamW decoupled weight decay (adamw only; default 0.01 for adamw, 0 for sgd)")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--generation-seed", type=int, default=7)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--max-eval-records", type=int, default=0, help="0 = full held-out split")
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
    args = parser.parse_args(argv)
    if args.gradient_accumulation < 1:
        parser.error("--gradient-accumulation must be >= 1")
    if args.optimizer == "sgd":
        if args.weight_decay not in (None, 0.0):
            # parser.error exits with a clean usage message; the native CLI
            # rejects the same combination, so fail here rather than after the
            # tokenizer and corpus work has already been spent.
            parser.error("--weight-decay requires --optimizer adamw: the SGD update has no decay term")
        args.weight_decay = 0.0
    elif args.weight_decay is None:
        args.weight_decay = 0.01

    # Token-unit horizons -> optimizer-step horizons.  Keeping the conversion
    # here means the workflow can keep expressing budgets in tokens while the
    # native trainer only ever sees steps.
    warmup_steps = max(1, args.warmup_updates // args.gradient_accumulation) if args.warmup_updates else 0
    total_steps = args.total_updates // args.gradient_accumulation if args.total_updates else 0
    if args.total_updates and total_steps <= warmup_steps:
        parser.error("--total-updates must exceed --warmup-updates in optimizer steps "
                     "(raise the budget or lower --warmup-updates)")

    corpus_dir = Path(args.corpus_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_dir / "manifest.json"
    binary = str(Path(args.neural_bin).resolve())
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("train", "heldout", "train_epoch", "tokenizer_sample"):
            meta = manifest["outputs"][key]
            actual = sha256_file(corpus_dir / meta["path"])
            if actual["sha256"] != meta["sha256"] or actual["bytes"] != meta["bytes"]:
                raise RunError(f"corpus output {key} does not match manifest hash/bytes")

        tokenizer_path = run_dir / "tokenizer.bpe"
        checkpoint_path = run_dir / "checkpoint.vgnr"
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
        if args.optimizer == "adamw":
            model_options += ["--weight-decay", repr(args.weight_decay)]
        if total_steps:
            model_options += ["--total-updates", str(total_steps)]

        stdout, tokenizer_wall = run_child(
            [binary, "tokenizer-train-fast", str(corpus_dir / manifest["outputs"]["tokenizer_sample"]["path"]),
             str(tokenizer_path), str(args.vocab), str(args.min_pair_frequency)],
            timeout=args.tokenizer_timeout_seconds,
        )
        tokenizer_report = parse_prefixed_json(stdout, "BPE_TRAIN_JSON:")

        stdout, _ = run_child(
            [binary, "init-checkpoint", str(tokenizer_path), str(checkpoint_path), *model_options],
            timeout=args.evaluate_timeout_seconds,
        )
        init_report = parse_prefixed_json(stdout, "NEURAL_INIT_JSON:")

        heldout_path = str(corpus_dir / manifest["outputs"]["heldout"]["path"])
        # Per-token OpenMP regions are microseconds apart; the default wait
        # policy puts workers to sleep between them and wake latency then
        # dominates (measured in probe A/Bs 40301975287/40305596215).  An
        # active wait policy keeps the bounded thread team spinning for the
        # duration of these dedicated training/evaluation processes.
        omp_env = {"OMP_NUM_THREADS": str(os.cpu_count() or 2), "OMP_WAIT_POLICY": "ACTIVE"}

        def evaluate() -> dict[str, Any]:
            command = [binary, "evaluate", str(checkpoint_path), str(tokenizer_path), heldout_path]
            if args.max_eval_records:
                command.append(str(args.max_eval_records))
            text, _wall = run_child(command, timeout=args.evaluate_timeout_seconds, env_extra=omp_env)
            return parse_prefixed_json(text, "NEURAL_EVALUATE_JSON:")

        baseline = evaluate()

        epoch_file = str(corpus_dir / manifest["outputs"]["train_epoch"]["path"])
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
                "optimizer": args.optimizer,
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

        final = evaluate()

        prompts = DEFAULT_PROMPTS
        if args.prompts_file:
            prompts = [line.strip() for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        generations = []
        for index, prompt in enumerate(prompts):
            text, _wall = run_child(
                [binary, "generate", str(checkpoint_path), str(tokenizer_path), prompt,
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
                "weight_decay_scope": "adamw embeddings only" if args.optimizer == "adamw" else "not applicable (sgd)",
                "adam_beta1": args.beta1,
                "adam_beta2": args.beta2,
                "adam_epsilon": args.adam_eps,
                "seed": args.seed,
                "chunks": args.chunks,
            },
            "tokenizer": {**tokenizer_report, "file": sha256_file(tokenizer_path),
                          "controller_wall_seconds": tokenizer_wall},
            "init": init_report,
            "baseline_heldout": baseline,
            "training_epoch": epoch_report,
            "training_chunks": train_chunks,
            "training_complete": complete_report,
            "training_controller_wall_seconds": train_wall,
            "final_heldout": final,
            "checkpoint": sha256_file(checkpoint_path),
            "generations": generations,
            "scale_projection": project_99b(
                float(epoch_report["prediction_tokens_per_second"]),
                int(epoch_report["prediction_tokens"]),
            ),
            "limitations": [
                "Yalnız bağlı çıktı katmanı eğitilir; dikkat/retention/FFN parametreleri donuktur. Bu tam BPTT değildir.",
                "Epoch korpusu, manifestte hash ile tanımlı train_epoch dilimidir; tam kabul edilen korpus daha büyüktür ve train-chunk ile devam edilebilir.",
                "Üretimler nedensel devamdır; talimat takip, olgusal doğruluk veya genel Türkçe asistan kalitesi iddia edilmez.",
                "Held-out ölçümü aynı dump içinden belge-ayrık bölmedir; bağımsız insan-anotasyonlu benchmark değildir.",
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
