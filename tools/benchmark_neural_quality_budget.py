#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare learning rates or binaries at equal native-training budgets.

The self-generated Turkish-flavoured records are split before tokenizer
training. Both arms use the same training records, held-out records, tokenizer,
initial model seed, shape, optimizer, accumulation factor and schedule. The
report checks prediction-token and optimizer-step counts before comparing NLL.
For equal-rate binary A/B, optional checkpoint identity proves byte equality
for these runs. A separate v5-only mode checks every byte except the cumulative
NLL statistic, which can change when the scoring formula is corrected.
Timings are observations, not a cost guarantee; this is not an independent
natural-language quality evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
from typing import Any

from benchmark_neural_batch_throughput import (
    build_corpus,
    parse_prefixed_json,
    run_child,
    split_heldout,
    train_arm,
)

SCHEMA = "vegai-neural-equal-budget-quality/v1"
# Pass these explicitly rather than silently depending on a binary's defaults.
FIXED_OPTIONS = {
    "beta1": 0.9,
    "beta2": 0.999,
    "adam_eps": 1.0e-8,
    "grad_clip": 4.0,
    "embedding_limit": 8.0,
    "min_lr_ratio": 0.1,
    "max_tokens_per_record": 8192,
}
WEIGHT_DECAY = 0.01
MIN_PAIR_FREQUENCY = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_are_identical(left: Path, right: Path) -> bool:
    """Compare without loading potentially large checkpoints into memory."""
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            chunk = first.read(1024 * 1024)
            if chunk != second.read(1024 * 1024):
                return False
            if not chunk:
                return True


def v5_state_identity_excluding_cumulative_nll(left: Path, right: Path) -> tuple[bool, bool]:
    """Compare a v5 stream bytewise, allowing only its eight-byte NLL to differ.

    The native stream layout in neural_checkpoint_save has 192 header bytes
    before state%cumulative_nll. Validate v5's dimensions, optimizer flag and
    exact payload length before comparing. Reads are bounded to 1 MiB. Return
    (state-identical, whole-file-identical); this is not a general file validator.
    """
    with left.open("rb") as first, right.open("rb") as second:
        first_header, second_header = first.read(204), second.read(204)
        if (len(first_header) != 204 or len(second_header) != 204
                or first_header[:8] != b"VGNR0001" or second_header[:8] != b"VGNR0001"
                or first_header[:192] != second_header[:192] or first_header[200:] != second_header[200:]):
            return False, False
        if first_header[8:12] == b"\x05\x00\x00\x00":
            endian = "<"
        elif first_header[8:12] == b"\x00\x00\x00\x05":
            endian = ">"
        else:
            return False, False
        vocab, width, heads, window, ffn = struct.unpack_from(endian + "5i", first_header, 12)
        flag = struct.unpack_from(endian + "i", first_header, 200)[0]
        if (vocab < 2 or width < 2 or heads < 1 or width % heads or (width // heads) % 2
                or not 1 <= window <= 4096 or ffn < width or flag not in (0, 1)):
            return False, False
        parameter_count = width * vocab + vocab + 2 * width + 4 * width * width + 3 * width * ffn + heads
        expected_length = 204 + 4 * parameter_count
        if flag == 1:
            first_optimizer, second_optimizer = first.read(4), second.read(4)
            if len(first_optimizer) != 4 or first_optimizer != second_optimizer:
                return False, False
            optimizer = struct.unpack(endian + "i", first_optimizer)[0]
            if optimizer == 2:  # AdamW: two full embedding and two bias moments.
                expected_length += 4 + 8 * (width * vocab + vocab)
            elif optimizer == 3:  # NOVA: velocity plus one energy per vocabulary column.
                expected_length += 4 + 4 * (width * vocab + 2 * vocab)
            else:
                return False, False
        if left.stat().st_size != expected_length or right.stat().st_size != expected_length:
            return False, False
        whole_identical = first_header[192:200] == second_header[192:200]
        while True:
            first_chunk, second_chunk = first.read(1024 * 1024), second.read(1024 * 1024)
            if first_chunk != second_chunk:
                return False, False
            if not first_chunk:
                return True, whole_identical


def require_disjoint_records(train: Path, heldout: Path) -> None:
    """Fail closed if identical synthetic records straddle the split."""
    with train.open("rb") as stream:
        train_digests = {hashlib.sha256(line).digest() for line in stream}
    with heldout.open("rb") as stream:
        if any(hashlib.sha256(line).digest() in train_digests for line in stream):
            raise ValueError("an identical record occurs in both train and held-out splits")


def compare_arms(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Fail rather than present a quality comparison with unequal budgets."""
    ref_train, cand_train = reference["epoch"], candidate["epoch"]
    ref_done, cand_done = reference["training_complete"], candidate["training_complete"]
    ref_eval, cand_eval = reference["heldout"], candidate["heldout"]
    fields = (
        (ref_train["prediction_tokens"], cand_train["prediction_tokens"], "training prediction tokens"),
        (ref_done["updates_completed"], cand_done["updates_completed"], "optimizer updates"),
        (ref_eval["prediction_tokens"], cand_eval["prediction_tokens"], "held-out prediction tokens"),
    )
    for left, right, name in fields:
        if type(left) is not int or type(right) is not int or left <= 0 or left != right:
            raise ValueError(f"unequal or empty {name}: {left!r} vs {right!r}")
    values = (ref_eval["mean_nll"], cand_eval["mean_nll"],
              reference["controller_wall_seconds"], candidate["controller_wall_seconds"])
    if not all(type(value) in (int, float) for value in values):
        raise ValueError("non-numeric NLL or training time")
    try:
        ref_nll, cand_nll, ref_wall, cand_wall = map(float, values)
    except OverflowError as exc:
        raise ValueError("non-finite NLL or training time") from exc
    if not all(math.isfinite(value) for value in (ref_nll, cand_nll, ref_wall, cand_wall)):
        raise ValueError("non-finite NLL or training time")
    if ref_nll < 0.0 or cand_nll < 0.0 or ref_wall <= 0.0 or cand_wall <= 0.0:
        raise ValueError("negative NLL or non-positive training time")
    compared = {
        "train_prediction_tokens": ref_train["prediction_tokens"],
        "optimizer_updates": ref_done["updates_completed"],
        "heldout_prediction_tokens": ref_eval["prediction_tokens"],
        "reference_nll": ref_nll,
        "candidate_nll": cand_nll,
        "candidate_minus_reference_nll": cand_nll - ref_nll,
        "reference_train_seconds": ref_wall,
        "candidate_train_seconds": cand_wall,
        "candidate_over_reference_time": cand_wall / ref_wall,
    }
    # The native epoch report may provide phase timings; include them only
    # when both arms have finite, non-negative numeric values.
    ref_phases, cand_phases = ref_train.get("phase_seconds"), cand_train.get("phase_seconds")
    if ref_phases is not None or cand_phases is not None:
        if not isinstance(ref_phases, dict) or not isinstance(cand_phases, dict):
            raise ValueError("both arms must provide valid phase timing objects")
        phases = {}
        for name in ("decode", "project", "score", "update"):
            left, right = ref_phases.get(name), cand_phases.get(name)
            if type(left) not in (int, float) or type(right) not in (int, float):
                raise ValueError(f"invalid {name} phase timing")
            try:
                left, right = float(left), float(right)
            except OverflowError as exc:
                raise ValueError(f"invalid {name} phase timing") from exc
            if not math.isfinite(left) or not math.isfinite(right) or left < 0 or right < 0:
                raise ValueError(f"invalid {name} phase timing")
            phases[name] = {"reference_seconds": left, "candidate_seconds": right}
        compared["phase_seconds"] = phases
    return compared


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-bin", required=True, help="candidate vegai_neural_pretrain executable")
    parser.add_argument("--reference-bin", help="optional reference binary (defaults to the candidate binary)")
    parser.add_argument("--require-checkpoint-identity", action="store_true",
                        help="fail unless equal-LR arms produce byte-identical checkpoints")
    parser.add_argument("--require-model-state-identity", action="store_true",
                        help="v5 only: require every checkpoint byte except cumulative NLL to match")
    parser.add_argument("--reference-lr", type=positive_float, default=0.025)
    parser.add_argument("--candidate-lr", type=positive_float, default=0.003)
    parser.add_argument("--seeds", default="21,42,73", help="comma-separated corpus/model seeds")
    parser.add_argument("--corpus-bytes", type=int, default=50_000)
    parser.add_argument("--heldout-every", type=int, default=9)
    parser.add_argument("--vocab", type=int, default=320)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--ffn", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--optimizer", choices=("nova", "adamw", "sgd"), default="nova")
    parser.add_argument("--schedule", choices=("constant", "invsqrt", "cosine"), default="constant")
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--total-updates", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)
    try:
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    except ValueError:
        parser.error("--seeds must contain comma-separated integers")
    if not seeds or len(seeds) > 32 or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**63 for seed in seeds):
        parser.error("--seeds requires 1..32 distinct nonnegative int64 values")
    if not (5_000 <= args.corpus_bytes <= 10_000_000 and 2 <= args.heldout_every <= 1000):
        parser.error("corpus bytes must be 5000..10000000 and heldout-every must be 2..1000")
    if min(args.vocab, args.width, args.heads, args.window, args.ffn, args.accumulation, args.threads, args.timeout) < 1:
        parser.error("shape, accumulation, threads, and timeout must be positive")
    if args.accumulation > 512 or args.width % args.heads:
        parser.error("accumulation must be <= 512 and width must be divisible by heads")
    if not math.isfinite(args.label_smoothing) or not 0 <= args.label_smoothing < 0.5:
        parser.error("label smoothing must be in [0, 0.5)")
    if args.warmup_updates < 0 or args.total_updates < 0 or (
        args.schedule == "cosine" and args.total_updates <= args.warmup_updates
    ):
        parser.error("invalid schedule horizons")
    if (args.require_checkpoint_identity or args.require_model_state_identity) and args.reference_lr != args.candidate_lr:
        parser.error("checkpoint/model-state identity needs equal learning rates")
    binary = Path(args.neural_bin).resolve()
    reference_binary = Path(args.reference_bin).resolve() if args.reference_bin else binary
    if not binary.is_file() or not reference_binary.is_file():
        parser.error("--neural-bin and --reference-bin must name existing executable files")
    if args.output and any(args.output.resolve() == candidate or
                           (args.output.exists() and args.output.samefile(candidate))
                           for candidate in (binary, reference_binary)):
        parser.error("--output must not overwrite a native executable")

    rows: list[dict[str, Any]] = []
    try:
        binary_sha256 = sha256_file(binary)
        reference_binary_sha256 = sha256_file(reference_binary)
        with tempfile.TemporaryDirectory(prefix="vegai-quality-budget-") as directory:
            root = Path(directory)
            for ordinal, seed in enumerate(seeds):
                work = root / str(seed)
                work.mkdir()
                corpus = build_corpus(work / "full.txt", args.corpus_bytes, seed)
                split = split_heldout(work / "full.txt", work / "heldout.txt", work / "train.txt", args.heldout_every)
                require_disjoint_records(work / "train.txt", work / "heldout.txt")
                tokenizer = work / "tokenizer.bpe"
                run_child([str(binary), "tokenizer-train-fast", str(work / "train.txt"), str(tokenizer),
                           str(args.vocab), str(MIN_PAIR_FREQUENCY)], args.timeout)
                initial = work / "initial.vgnr"
                model_options = ["--width", str(args.width), "--heads", str(args.heads), "--window", str(args.window),
                                 "--ffn", str(args.ffn), "--seed", str(seed)]
                run_child([str(binary), "init-checkpoint", str(tokenizer), str(initial), *model_options], args.timeout)
                initial_stdout, _ = run_child([str(binary), "evaluate", str(initial), str(tokenizer),
                                               str(work / "heldout.txt")], args.timeout,
                                              {"OMP_NUM_THREADS": str(args.threads), "OMP_WAIT_POLICY": "ACTIVE"})
                baseline = parse_prefixed_json(initial_stdout, "NEURAL_EVALUATE_JSON:")
                arm_args = copy.copy(args)
                arm_args.seed = seed
                arm_args.baseline_nll = float(baseline["mean_nll"])
                arms: dict[str, dict[str, Any]] = {}
                # Alternate order to reduce, not eliminate, load/order bias.
                order = ("reference", "candidate") if ordinal % 2 == 0 else ("candidate", "reference")
                fixed_options: list[str] = []
                for key, value in FIXED_OPTIONS.items():
                    fixed_options.extend(("--" + key.replace("_", "-"), str(value)))
                if args.optimizer != "sgd":
                    fixed_options.extend(("--weight-decay", str(WEIGHT_DECAY)))
                for name in order:
                    arm_args.lr = args.reference_lr if name == "reference" else args.candidate_lr
                    arm_binary = reference_binary if name == "reference" else binary
                    arms[name] = train_arm(str(arm_binary), tokenizer, work / f"{name}.vgnr", work / "train.txt",
                                           work / "heldout.txt", arm_args, args.accumulation,
                                           extra_options=fixed_options)
                compared = compare_arms(arms["reference"], arms["candidate"])
                if args.require_checkpoint_identity:
                    if not files_are_identical(work / "reference.vgnr", work / "candidate.vgnr"):
                        raise ValueError(f"checkpoints differ for seed {seed}")
                    compared["checkpoint_sha256"] = sha256_file(work / "reference.vgnr")
                    compared["checkpoint_byte_identical"] = True
                if args.require_model_state_identity:
                    state_identical, whole_identical = v5_state_identity_excluding_cumulative_nll(
                        work / "reference.vgnr", work / "candidate.vgnr")
                    if not state_identical:
                        raise ValueError(f"v5 model/optimizer state differs for seed {seed}")
                    compared["checkpoint_state_identical_except_cumulative_nll"] = True
                    compared["checkpoint_byte_identical"] = whole_identical
                rows.append({
                    "seed": seed,
                    "order": list(order),
                    "corpus_sha256": corpus["sha256"],
                    "train_sha256": split["train"]["sha256"],
                    "heldout_sha256": split["heldout"]["sha256"],
                    "tokenizer_sha256": sha256_file(tokenizer),
                    "initial_heldout_nll": arm_args.baseline_nll,
                    **compared,
                })
            if sha256_file(binary) != binary_sha256 or sha256_file(reference_binary) != reference_binary_sha256:
                raise ValueError("a native binary changed during the comparison")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"quality benchmark failed: {error}", file=sys.stderr)
        return 1

    report = {
        "schema": SCHEMA,
        "binary_sha256": binary_sha256,
        "reference_binary_sha256": reference_binary_sha256,
        "configuration": {
            "require_checkpoint_identity": args.require_checkpoint_identity,
            "require_model_state_identity": args.require_model_state_identity,
            "reference_lr": args.reference_lr, "candidate_lr": args.candidate_lr, "optimizer": args.optimizer,
            "schedule": args.schedule, "warmup_updates": args.warmup_updates, "total_updates": args.total_updates,
            "label_smoothing": args.label_smoothing, "gradient_accumulation": args.accumulation,
            "vocab": args.vocab, "width": args.width, "heads": args.heads, "window": args.window,
            "ffn": args.ffn, "threads": args.threads, "requested_corpus_bytes": args.corpus_bytes,
            "heldout_every": args.heldout_every, "min_pair_frequency": MIN_PAIR_FREQUENCY,
            "weight_decay": WEIGHT_DECAY if args.optimizer != "sgd" else 0.0,
            **FIXED_OPTIONS,
        },
        "runs": rows,
        "summary": {
            "median_candidate_minus_reference_nll": statistics.median(row["candidate_minus_reference_nll"] for row in rows),
            "candidate_better_seeds": sum(row["candidate_nll"] < row["reference_nll"] for row in rows),
            "median_candidate_over_reference_time": statistics.median(row["candidate_over_reference_time"] for row in rows),
        },
        "limitations": [
            "Synthetic record-disjoint microbenchmark, not a real-world language or safety evaluation.",
            "Equal tokens and updates do not prove equal energy, peak memory or cost on other hardware.",
            "A learning rate better on these seeds need not generalize to other datasets or schedules.",
            "The native trainer updates tied output embeddings only, not full backpropagation through the core.",
        ],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        try:
            args.output.write_text(payload, encoding="utf-8")
        except OSError as error:
            print(f"cannot write quality report: {error}", file=sys.stderr)
            return 1
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
