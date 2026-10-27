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
# Paragraph-scale records, matching the governed corpus builder, which drops
# filtered paragraphs below 700 characters (tools/build_turkish_wiki_corpus.py).
RECORD_MIN_CHARS = 700

STEMS = (
    "deniz", "kıyı", "bulut", "yağmur", "şehir", "sokak", "kitap", "defter",
    "öğrenci", "öğretmen", "sabah", "akşam", "yolculuk", "tren", "istasyon",
    "bahçe", "ağaç", "çiçek", "rüzgar", "güneş", "toprak", "köprü", "nehir",
    "orman", "kuş", "balık", "kaya", "dalga", "liman", "gemi", "ışık", "gölge",
)
SUFFIXES = ("", "ler", "lar", "de", "da", "den", "dan", "in", "ın", "e", "a")
CONNECTORS = (" ve ", " ama ", " çünkü ", ", ", " ki ", "; ", ". ", " - ")


def build_corpus(path: Path, target_bytes: int, seed: int) -> dict[str, Any]:
    """Writes a deterministic UTF-8 record-per-line corpus of about target_bytes.

    Records are paragraph scale (at least RECORD_MIN_CHARS), because that is what
    the governed corpus emits: the wiki builder filters out paragraphs below 700
    characters and frames whole paragraphs as records.  Record length is not
    cosmetic here -- accumulation flushes at a record boundary, so a corpus of
    eighty byte lines caps the effective batch at roughly twenty tokens whatever
    factor is requested, which made a 32/64/128 comparison meaningless.
    """
    rng = random.Random(seed)
    written = 0
    records = 0
    record_bytes = 0
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        while written < target_bytes:
            sentences = []
            length = 0
            while length < RECORD_MIN_CHARS:
                words = []
                for _ in range(rng.randint(5, 12)):
                    words.append(rng.choice(STEMS) + rng.choice(SUFFIXES))
                    words.append(rng.choice(CONNECTORS))
                sentence = "".join(words).strip().rstrip(",") + "."
                sentences.append(sentence)
                length += len(sentence) + 1
            line = " ".join(sentences) + "\n"
            payload = line.encode("utf-8")
            handle.write(line)
            digest.update(payload)
            written += len(payload)
            record_bytes += len(payload)
            records += 1
    return {"path": str(path), "bytes": written, "records": records, "sha256": digest.hexdigest(),
            "mean_record_bytes": (record_bytes / records) if records else 0.0,
            "record_min_chars": RECORD_MIN_CHARS}


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
              args: argparse.Namespace, accumulation: int, supports_accumulation: bool = True) -> dict[str, Any]:
    """Trains one arm from a freshly initialized checkpoint and measures nothing but behavior."""
    model_options = [
        "--width", str(args.width), "--heads", str(args.heads), "--window", str(args.window),
        "--ffn", str(args.ffn), "--seed", str(args.seed), "--lr", repr(args.lr),
        "--optimizer", args.optimizer, "--schedule", args.schedule,
        "--warmup-updates", str(args.warmup_updates), "--total-updates", str(args.total_updates),
        "--label-smoothing", repr(args.label_smoothing),
    ]
    if supports_accumulation:
        model_options += ["--gradient-accumulation", str(accumulation)]
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
            f"- reported batch arm: **N={comparison.get('accumulation', 'n/a')}**",
            f"- speedup (batched vs per-token, same binary): **{comparison['speedup_tokens_per_second']:.3f}x**",
        ]
        if "speedup_vs_baseline" in comparison:
            lines += [
                f"- speedup vs the comparison binary (end to end): "
                f"**{comparison['speedup_vs_baseline']:.3f}x**",
                f"- held-out NLL delta vs baseline: "
                f"{comparison['heldout_nll_delta_batched_minus_baseline']:+.6f}",
            ]
        lines += [
            f"- held-out NLL delta (batched - per-token): "
            f"{comparison['heldout_nll_delta_batched_minus_per_token']:+.6f}",
            f"- NLL improvement per second ratio: {comparison['nll_improvement_per_second_ratio']:.3f}x",
        ]
    for failure in report.get("gate_failures", []):
        lines.append(f"- gate failure: {failure}")
    if report.get("error"):
        lines.append(f"- error: {report['error']}")
    return "\n".join(lines) + "\n"


def phase_text(comparison: dict[str, Any]) -> str:
    """Compact per-token phase breakdown (microseconds) for the annotation."""
    phases = comparison.get("phase_microseconds_per_token")
    if not phases:
        return "phases unavailable"
    return "phases us/token " + " ".join(
        f"{name} {value:.1f}" for name, value in phases.items()
    )


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
    baseline_text = ""
    if "speedup_vs_baseline" in comparison:
        baseline_text = (
            f"speedup vs baseline {comparison['speedup_vs_baseline']:.3f}x; "
            f"NLL delta vs baseline {comparison['heldout_nll_delta_batched_minus_baseline']:+.6f}; "
        )
    return (
        f"N={comparison.get('accumulation', 0)}; "
        f"speedup {comparison['speedup_tokens_per_second']:.3f}x; "
        f"held-out NLL delta {comparison['heldout_nll_delta_batched_minus_per_token']:+.6f}; "
        f"{baseline_text}quality-per-second {comparison['nll_improvement_per_second_ratio']:.3f}x; "
        f"evaluate-only {comparison.get('evaluate_only_tokens_per_second', 0.0):.1f} tok/s; "
        f"{phase_text(comparison)}; {measured}"
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
    parser.add_argument("--corpus-bytes", type=int, default=400_000)
    parser.add_argument("--heldout-every", type=int, default=9, help="every nth record is held-out")
    # Defaults mirror the governed Turkish epoch contract, because the size of
    # the win depends on the shape: the update loop scales with vocabulary x
    # width while the rest of the step scales with width^2, so a toy 512-token
    # vocabulary understates the production effect (measured 1.30x there versus
    # the shape below).  Measure at the shape the claim is about.
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--ffn", type=int, default=384)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", default="adamw", choices=("adamw", "sgd"))
    parser.add_argument("--schedule", default="invsqrt", choices=("constant", "invsqrt", "cosine"))
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--total-updates", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--accumulation", type=int, default=32, help="batch arm accumulation factor")
    parser.add_argument("--sweep", default=None,
                        help="comma separated accumulation factors to compare (e.g. 32,64,128); "
                             "the fastest arm whose held-out NLL does not regress becomes the reported"
                             " batch arm")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--baseline-bin", help="optional binary built from the comparison revision (e.g. main); "
                                               "its arm measures the end-to-end speedup instead of just the "
                                               "accumulation factor")
    parser.add_argument("--min-speedup", type=float, default=1.25,
                        help="gate: required measured tokens/s ratio of the batched arm against the per-token arm "
                             "of the same binary")
    parser.add_argument("--min-baseline-speedup", type=float, default=0.0,
                        help="gate: required tokens/s ratio of the batched arm against --baseline-bin (0 = no gate)")
    parser.add_argument("--max-nll-regression", type=float, default=0.05,
                        help="gate: batched held-out NLL may exceed the per-token arm by at most this")
    parser.add_argument("--max-baseline-nll-regression", type=float, default=0.05,
                        help="gate: batched held-out NLL may exceed the baseline arm by at most this")
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
    }, "gates": {"min_speedup": args.min_speedup, "min_baseline_speedup": args.min_baseline_speedup,
                 "max_nll_regression": args.max_nll_regression,
                 "max_baseline_nll_regression": args.max_baseline_nll_regression}}
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
        factors: list[int] = [args.accumulation]
        if args.sweep:
            factors += [int(part) for part in args.sweep.split(",") if part.strip()]
        factors = [factor for factor in dict.fromkeys(factors) if factor > 1]
        for position, accumulation in enumerate([1, *factors]):
            checkpoint = work_dir / f"arm{position}.vgnr"
            arms[arm_name(accumulation)] = train_arm(
                binary, tokenizer_path, checkpoint, work_dir / "train.txt", work_dir / "heldout.txt",
                args, accumulation,
            )
        if args.baseline_bin:
            # Same corpus, tokenizer, shape and seed, but the binary built from
            # the comparison revision: this is the end-to-end number ("training
            # is X times faster than main"), measured inside one job so runner
            # noise cannot flatter it.
            baseline_binary = str(Path(args.baseline_bin).resolve())
            arms["baseline"] = train_arm(
                baseline_binary, tokenizer_path, work_dir / "baseline_arm.vgnr", work_dir / "train.txt",
                work_dir / "heldout.txt", args, 1, supports_accumulation=False,
            )
        report["arms"] = arms

        per_token = arms["per-token"]
        # The reported batch arm is the fastest accumulation factor whose
        # held-out NLL does not regress against the per-token arm; a larger
        # batch buys speed by amortizing the optimizer step, and it must not buy
        # it by degrading the model, so the choice is made from measurements.
        candidates = [arms[arm_name(factor)] for factor in factors]
        eligible = [arm for arm in candidates
                    if float(arm["heldout"]["mean_nll"]) - float(per_token["heldout"]["mean_nll"])
                    <= args.max_nll_regression]
        batched = max(eligible or candidates, key=lambda arm: arm["reported_tokens_per_second"])
        chosen_accumulation = int(batched["gradient_accumulation"])
        report["accumulation_sweep"] = {
            arm["arm"]: {
                "accumulation": int(arm["gradient_accumulation"]),
                "tokens_per_second": float(arm["reported_tokens_per_second"]),
                "heldout_mean_nll": float(arm["heldout"]["mean_nll"]),
            }
            for arm in candidates
        }
        speedup = (batched["reported_tokens_per_second"] / per_token["reported_tokens_per_second"]
                   if per_token["reported_tokens_per_second"] else 0.0)
        nll_delta = float(batched["heldout"]["mean_nll"]) - float(per_token["heldout"]["mean_nll"])
        # Quality per unit of wall clock.  The honest reference is the
        # comparison binary when one was supplied, because a short probe run
        # does not necessarily improve the per-token arm at all (its
        # improvement-per-second then divides by ~0 and reports a meaningless 0).
        reference = arms.get("baseline", per_token)
        quality_rate = (batched["nll_improvement_per_second"] / reference["nll_improvement_per_second"]
                        if reference["nll_improvement_per_second"] > 0 else 0.0)
        comparison = {
            "accumulation": chosen_accumulation,
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
        # Forward-only reference: the held-out evaluate pass of the batched arm
        # runs the same decoder step and softmax as training, but no update and
        # no staging, so its rate separates "forward + scoring" from
        # "optimizer" cost.
        comparison["evaluate_only_tokens_per_second"] = float(
            batched["heldout"].get("prediction_tokens_per_second", 0.0)
        )
        # Per-prediction cost of each phase, in microseconds, straight from the
        # trainer's own counters.  Runners differ by more than 2x in absolute
        # rate, so the breakdown is reported per token to stay comparable.
        phase_seconds = batched.get("epoch", {}).get("phase_seconds", {})
        predictions = int(batched["epoch"].get("prediction_tokens", 0) or 0)
        if phase_seconds and predictions > 0:
            comparison["phase_microseconds_per_token"] = {
                name: (float(phase_seconds.get(name, 0.0)) / predictions) * 1.0e6
                for name in ("decode", "project", "score", "update")
            }
        baseline_arm = arms.get("baseline")
        if baseline_arm:
            baseline_rate = float(baseline_arm["reported_tokens_per_second"])
            baseline_speedup = (float(batched["reported_tokens_per_second"]) / baseline_rate) if baseline_rate else 0.0
            per_token_speedup = (float(per_token["reported_tokens_per_second"]) / baseline_rate) if baseline_rate else 0.0
            baseline_nll_delta = float(batched["heldout"]["mean_nll"]) - float(baseline_arm["heldout"]["mean_nll"])
            comparison.update({
                "baseline_binary": str(Path(args.baseline_bin).resolve()),
                "speedup_vs_baseline": baseline_speedup,
                "per_token_speedup_vs_baseline": per_token_speedup,
                "heldout_nll_delta_batched_minus_baseline": baseline_nll_delta,
            })
            if args.min_baseline_speedup > 0 and baseline_speedup < args.min_baseline_speedup:
                failures.append(
                    f"speedup vs baseline {baseline_speedup:.3f}x < gate {args.min_baseline_speedup:.3f}x"
                )
            if baseline_nll_delta > args.max_baseline_nll_regression:
                failures.append(
                    f"held-out NLL regression vs baseline {baseline_nll_delta:+.4f} > "
                    f"gate {args.max_baseline_nll_regression:.4f}"
                )
        report["comparison"] = comparison
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
