#!/usr/bin/env python3
"""Measured A/B of mini-batch gradient accumulation for the native neural trainer.

The tied-output update streams the tied embedding, both AdamW moment arrays and
the two bias moment arrays once per prediction token.  At the production shape
(vocabulary 4096, width 96) that is ~17 MB of L3-resident traffic per token, and
it dominated the epoch wall clock.  `--gradient-accumulation N` averages the
gradients of N prediction tokens into one optimizer step, so the same state is
streamed once per N tokens.

This benchmark exists so the speed claim is a measurement and not a promise: it
trains two byte-identical synthetic corpora with the same seed, shape and
schedule, once per-token (N=1) and once batched, and reports

  * prediction tokens/s for both,
  * held-out NLL / perplexity for both (quality at equal tokens), and
  * NLL improvement per second of training (quality at equal wall clock).

Gates (`--min-speedup`, `--max-nll-regression`, `--min-quality-rate-ratio`) make
it usable as a CI regression check; the JSON/Markdown artifacts are the evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA = "vegai-neural-batch-throughput/v1"

# Deterministic Turkish-flavoured synthetic text.  The point is a controlled,
# dependency-free corpus with realistic BPE merge structure, not a language
# model benchmark: both arms see byte-identical data.
STEMS = (
    "deniz", "kıyı", "bulut", "yağmur", "şehir", "sokak", "kitap", "defter",
    "öğrenci", "öğretmen", "sabah", "akşam", "yolculuk", "tren", "istasyon",
    "bahçe", "ağaç", "çiçek", "rüzgar", "güneş", "toprak", "köprü", "nehir",
    "orman", "kuş", "balık", "kaya", "dalga", "liman", "gemi", "ışık", "gölge",
)
SUFFIXES = ("", "ler", "lar", "de", "da", "den", "dan", "in", "ın", "e", "a")
CONNECTORS = (" ve ", " ama ", " çünkü ", ", ", " ki ", "; ", ". ", " - ")


def build_corpus(path: Path, target_bytes: int, seed: int) -> dict[str, Any]:
    """Writes a deterministic UTF-8 record-per-line corpus of about target_bytes."""
    rng = random.Random(seed)
    written = 0
    records = 0
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        while written < target_bytes:
            words = []
            for _ in range(rng.randint(4, 14)):
                words.append(rng.choice(STEMS) + rng.choice(SUFFIXES))
                words.append(rng.choice(CONNECTORS))
            line = "".join(words).strip() + "\n"
            payload = line.encode("utf-8")
            handle.write(line)
            digest.update(payload)
            written += len(payload)
            records += 1
    return {"path": str(path), "bytes": written, "records": records, "sha256": digest.hexdigest()}


def split_heldout(corpus_path: Path, heldout_path: Path, train_path: Path, heldout_every: int) -> dict[str, Any]:
    """Document-disjoint split: every nth record goes to held-out."""
    train_digest = hashlib.sha256()
    heldout_digest = hashlib.sha256()
    train_bytes = heldout_bytes = 0
    train_records = heldout_records = 0
    with corpus_path.open("r", encoding="utf-8") as source, \
            train_path.open("w", encoding="utf-8", newline="\n") as train, \
            heldout_path.open("w", encoding="utf-8", newline="\n") as heldout:
        for index, line in enumerate(source):
            payload = line.encode("utf-8")
            if index % heldout_every == 0:
                heldout.write(line)
                heldout_digest.update(payload)
                heldout_bytes += len(payload)
                heldout_records += 1
            else:
                train.write(line)
                train_digest.update(payload)
                train_bytes += len(payload)
                train_records += 1
    return {
        "train": {"path": str(train_path), "bytes": train_bytes, "records": train_records,
                  "sha256": train_digest.hexdigest()},
        "heldout": {"path": str(heldout_path), "bytes": heldout_bytes, "records": heldout_records,
                    "sha256": heldout_digest.hexdigest()},
    }


def parse_prefixed_json(stdout: str, prefix: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix):].strip())
    raise RuntimeError(f"missing {prefix.strip()} in native output:\n{stdout[-4000:]}")


def run_child(command: list[str], timeout: int, env_extra: dict[str, str] | None = None) -> tuple[str, float]:
    environment = dict(os.environ)
    if env_extra:
        environment.update(env_extra)
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment, check=False)
    wall = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
        )
    return completed.stdout, wall


def arm_name(accumulation: int) -> str:
    return "per-token" if accumulation == 1 else f"batch-{accumulation}"


def train_arm(binary: str, tokenizer_path: Path, checkpoint_path: Path, train_path: Path, heldout_path: Path,
              args: argparse.Namespace, accumulation: int) -> dict[str, Any]:
    """Trains one arm from a freshly initialized checkpoint and measures nothing but behavior."""
    model_options = [
        "--width", str(args.width), "--heads", str(args.heads), "--window", str(args.window),
        "--ffn", str(args.ffn), "--seed", str(args.seed), "--lr", repr(args.lr),
        "--optimizer", args.optimizer, "--schedule", args.schedule,
        "--warmup-updates", str(args.warmup_updates), "--total-updates", str(args.total_updates),
        "--label-smoothing", repr(args.label_smoothing),
        "--gradient-accumulation", str(accumulation),
    ]
    # Same wait policy the governed epoch runner exports: per-token OpenMP
    # regions are microseconds apart, so a sleeping worker team spends more
    # time waking up than working.  Applied to both arms identically.
    omp_env = {"OMP_NUM_THREADS": str(args.threads), "OMP_WAIT_POLICY": "ACTIVE"}

    # `train` (not `resume`) always starts from a fresh, seed-determined model,
    # so both arms share identical initial parameters and only the update
    # grouping differs.
    train_stdout, train_wall = run_child(
        [binary, "train", str(train_path), str(checkpoint_path), str(tokenizer_path), "1", *model_options],
        timeout=args.timeout, env_extra=omp_env,
    )
    epoch = parse_prefixed_json(train_stdout, "NEURAL_PRETRAIN_EPOCH_JSON:")
    complete = parse_prefixed_json(train_stdout, "NEURAL_PRETRAIN_COMPLETE_JSON:")
    evaluate_stdout, _ = run_child(
        [binary, "evaluate", str(checkpoint_path), str(tokenizer_path), str(heldout_path)],
        timeout=args.timeout, env_extra=omp_env,
    )
    heldout = parse_prefixed_json(evaluate_stdout, "NEURAL_EVALUATE_JSON:")
    tokens = int(epoch["prediction_tokens"])
    return {
        "arm": arm_name(accumulation),
        "gradient_accumulation": accumulation,
        "epoch": epoch,
        "training_complete": complete,
        "heldout": heldout,
        "controller_wall_seconds": train_wall,
        "reported_tokens_per_second": float(epoch["prediction_tokens_per_second"]),
        "controller_tokens_per_second": (tokens / train_wall) if train_wall > 0 else 0.0,
        "nll_improvement_per_second": (
            (float(args.baseline_nll) - float(heldout["mean_nll"])) / train_wall if train_wall > 0 else 0.0
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable summary of one report (also used as the CI step summary)."""
    lines = [f"### Neural update A/B: {report.get('status')}", ""]
    shape = report.get("shape", {})
    if shape:
        lines.append(
            f"- shape: vocab {shape.get('vocab')}, width {shape.get('width')}, heads {shape.get('heads')}, "
            f"window {shape.get('window')}, ffn {shape.get('ffn')}, {shape.get('optimizer')}/{shape.get('schedule')}, "
            f"{shape.get('threads')} threads, {shape.get('corpus_bytes')} corpus bytes per arm"
        )
    arms = report.get("arms", {})
    if arms:
        lines += ["", "| Arm | prediction tok/s | held-out NLL | held-out ppl | train wall (s) |",
                  "|---|---:|---:|---:|---:|"]
        for name, arm in arms.items():
            lines.append(
                f"| {name} | {float(arm['reported_tokens_per_second']):.1f} | "
                f"{float(arm['heldout']['mean_nll']):.6f} | {float(arm['heldout']['perplexity']):.3f} | "
                f"{float(arm['controller_wall_seconds']):.1f} |"
            )
    comparison = report.get("comparison", {})
    if comparison:
        lines += [
            "",
            f"- speedup: **{comparison['speedup_tokens_per_second']:.3f}x**",
            f"- held-out NLL delta (batched - per-token): "
            f"{comparison['heldout_nll_delta_batched_minus_per_token']:+.6f}",
            f"- NLL improvement per second ratio: **{comparison['nll_improvement_per_second_ratio']:.3f}x**",
        ]
    for failure in report.get("gate_failures", []):
        lines.append(f"- gate failure: {failure}")
    if report.get("error"):
        lines.append(f"- error: {report['error']}")
    return "\n".join(lines) + "\n"


def headline(report: dict[str, Any]) -> str:
    """One-line, API-readable restatement of the measurement."""
    arms = report.get("arms", {})
    comparison = report.get("comparison", {})
    if not arms or not comparison:
        return ""
    measured = "; ".join(
        f"{name}: {float(arm['reported_tokens_per_second']):.1f} tok/s, "
        f"NLL {float(arm['heldout']['mean_nll']):.6f}"
        for name, arm in arms.items()
    )
    return (
        f"speedup {comparison['speedup_tokens_per_second']:.3f}x; "
        f"held-out NLL delta {comparison['heldout_nll_delta_batched_minus_per_token']:+.6f}; "
        f"quality-per-second {comparison['nll_improvement_per_second_ratio']:.3f}x; {measured}"
    )


def publish(report: dict[str, Any]) -> None:
    """Writes the step summary and a check annotation when running in CI."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as handle:
                handle.write(render_markdown(report))
        except OSError:
            pass
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    text = headline(report)
    if report.get("status") == "passed" and text:
        print(f"::notice title=Neural update A/B (passed)::{text}")
    else:
        detail = text or str(report.get("error", "unknown failure"))
        print(f"::error title=Neural update A/B (failed)::{detail[:1800]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neural-bin", help="path to the built vegai_neural_pretrain binary (not needed with --report)")
    parser.add_argument("--work-dir", help="default: a temporary directory")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--corpus-bytes", type=int, default=1_000_000)
    parser.add_argument("--heldout-every", type=int, default=9, help="every nth record is held-out")
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--ffn", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", default="adamw", choices=("adamw", "sgd"))
    parser.add_argument("--schedule", default="invsqrt", choices=("constant", "invsqrt", "cosine"))
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--total-updates", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--accumulation", type=int, default=32, help="batch arm accumulation factor")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--min-speedup", type=float, default=1.5,
                        help="gate: required measured tokens/s ratio of the batched arm")
    parser.add_argument("--max-nll-regression", type=float, default=0.05,
                        help="gate: batched held-out NLL may exceed the per-token arm by at most this")
    parser.add_argument("--min-quality-rate-ratio", type=float, default=1.5,
                        help="gate: required ratio of NLL-improvement-per-second (quality per wall clock)")
    parser.add_argument("--output", help="write the JSON report here")
    parser.add_argument("--report", help="render an existing report (no training) and publish it")
    parser.add_argument("--summary", action="store_true",
                        help="with --report: write the Markdown summary to $GITHUB_STEP_SUMMARY and annotate")
    args = parser.parse_args(argv)
    if args.report:
        existing = json.loads(Path(args.report).read_text(encoding="utf-8"))
        print(render_markdown(existing), end="")
        if args.summary:
            publish(existing)
        return 0 if existing.get("status") == "passed" else 1
    if not args.report and not args.neural_bin:
        parser.error("--neural-bin is required unless --report renders an existing measurement")
    if not args.report and args.accumulation < 2:
        parser.error("--accumulation must be >= 2 (the gate compares against the per-token path)")

    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="vegai-throughput-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    binary = str(Path(args.neural_bin).resolve())
    report: dict[str, Any] = {"schema": SCHEMA, "status": "running", "shape": {
        "vocab": args.vocab, "width": args.width, "heads": args.heads, "window": args.window,
        "ffn": args.ffn, "optimizer": args.optimizer, "schedule": args.schedule,
        "lr": args.lr, "threads": args.threads, "corpus_bytes": args.corpus_bytes,
    }, "gates": {"min_speedup": args.min_speedup, "max_nll_regression": args.max_nll_regression,
                 "min_quality_rate_ratio": args.min_quality_rate_ratio}}
    exit_code = 0
    try:
        corpus_path = work_dir / "corpus.txt"
        corpus = build_corpus(corpus_path, args.corpus_bytes, args.seed)
        split = split_heldout(corpus_path, work_dir / "heldout.txt", work_dir / "train.txt", args.heldout_every)
        report["corpus"] = {"full": corpus, **split}

        tokenizer_path = work_dir / "tokenizer.bpe"
        run_child([binary, "tokenizer-train-fast", str(work_dir / "train.txt"), str(tokenizer_path),
                   str(args.vocab), "2"], timeout=args.timeout)
        report["tokenizer"] = {"path": str(tokenizer_path), "bytes": tokenizer_path.stat().st_size}

        # Baseline held-out NLL of the untrained model, shared by both arms.
        baseline_checkpoint = work_dir / "baseline.vgnr"
        run_child([binary, "init-checkpoint", str(tokenizer_path), str(baseline_checkpoint),
                   "--width", str(args.width), "--heads", str(args.heads), "--window", str(args.window),
                   "--ffn", str(args.ffn), "--seed", str(args.seed)], timeout=args.timeout)
        baseline_stdout, _ = run_child(
            [binary, "evaluate", str(baseline_checkpoint), str(tokenizer_path), str(work_dir / "heldout.txt")],
            timeout=args.timeout,
        )
        baseline = parse_prefixed_json(baseline_stdout, "NEURAL_EVALUATE_JSON:")
        args.baseline_nll = float(baseline["mean_nll"])
        report["baseline_heldout"] = baseline

        # Both arms train from the same seed-determined initialization; the
        # corpus, tokenizer, shape, learning rate and schedule are identical.
        arms = {}
        for position, accumulation in enumerate((1, args.accumulation)):
            checkpoint = work_dir / f"arm{position}.vgnr"
            arms[arm_name(accumulation)] = train_arm(
                binary, tokenizer_path, checkpoint, work_dir / "train.txt", work_dir / "heldout.txt",
                args, accumulation,
            )
        report["arms"] = arms

        per_token = arms["per-token"]
        batched = arms[arm_name(args.accumulation)]
        speedup = (batched["reported_tokens_per_second"] / per_token["reported_tokens_per_second"]
                   if per_token["reported_tokens_per_second"] else 0.0)
        nll_delta = float(batched["heldout"]["mean_nll"]) - float(per_token["heldout"]["mean_nll"])
        quality_rate = (batched["nll_improvement_per_second"] / per_token["nll_improvement_per_second"]
                        if per_token["nll_improvement_per_second"] > 0 else 0.0)
        report["comparison"] = {
            "speedup_tokens_per_second": speedup,
            "heldout_nll_delta_batched_minus_per_token": nll_delta,
            "heldout_perplexity_per_token": math.exp(float(per_token["heldout"]["mean_nll"])),
            "heldout_perplexity_batched": math.exp(float(batched["heldout"]["mean_nll"])),
            "nll_improvement_per_second_ratio": quality_rate,
        }
        failures = []
        if speedup < args.min_speedup:
            failures.append(f"speedup {speedup:.3f}x < gate {args.min_speedup:.3f}x")
        if nll_delta > args.max_nll_regression:
            failures.append(f"held-out NLL regression {nll_delta:+.4f} > gate {args.max_nll_regression:.4f}")
        if quality_rate < args.min_quality_rate_ratio:
            failures.append(f"quality-per-second ratio {quality_rate:.3f}x < gate {args.min_quality_rate_ratio:.3f}x")
        report["status"] = "passed" if not failures else "failed"
        report["gate_failures"] = failures
        if failures:
            exit_code = 1
    except (RuntimeError, OSError, KeyError, ValueError, subprocess.TimeoutExpired) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        exit_code = 1
    finally:
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        publish(report)
        print(render_markdown(report), end="")
        print(payload, end="")
        if not args.keep_work_dir and not args.work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
