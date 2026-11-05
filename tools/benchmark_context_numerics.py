#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Interleaved, equal-token/update context-engine numerical-safety comparison.

The native bench command reports training NLL to six decimal places. Identical
printed NLL is not proof of identical parameters, and timing on shared hardware
is not a cost or energy guarantee. Both binaries receive identical arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def integer_list(text: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or len(values) > 32 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected 1..32 distinct integers")
    return values


def run(binary: Path, seed: int, tokens: int, chunk: int, timeout: int) -> dict[str, object]:
    command = [str(binary), "bench", "--tokens", str(tokens), "--train-chunk-tokens", str(chunk),
               "--task", "markov", "--width", "32", "--heads", "2", "--ffn", "64", "--vocab", "256",
               "--local-window", "8", "--memory-scales", "2", "--memory-slots", "16", "--seed", str(seed)]
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_WAIT_POLICY="ACTIVE")
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    if result.returncode != 0:
        raise ValueError(f"native benchmark failed ({result.returncode}): {result.stderr[-2000:]}")
    prefix = "CONTEXT_BENCH_JSON: "
    lines = [line[len(prefix):] for line in result.stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError("native benchmark must produce exactly one machine-readable result")
    data = json.loads(lines[0])
    if not isinstance(data, dict):
        raise ValueError("benchmark result is not an object")
    for name in ("training_tokens_per_s", "training_nll"):
        value = data.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid benchmark field: {name}")
    if data["training_tokens_per_s"] == 0:
        raise ValueError("zero measured training throughput")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bin", type=Path, required=True)
    parser.add_argument("--candidate-bin", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--chunks", type=integer_list, default=[16, 256])
    parser.add_argument("--seeds", type=integer_list, default=[21, 42, 73, 101, 137])
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 64 <= args.tokens <= 1_000_000 or any(not 1 <= chunk <= 8192 or args.tokens % chunk for chunk in args.chunks):
        parser.error("tokens must be 64..1000000 and divisible by every chunk (1..8192)")
    if any(not 0 <= seed < 2**63 for seed in args.seeds) or args.timeout < 1:
        parser.error("seeds must fit nonnegative int64 and timeout must be positive")
    reference, candidate = args.reference_bin.resolve(), args.candidate_bin.resolve()
    if not reference.is_file() or not candidate.is_file():
        parser.error("both binaries must be existing files")
    if args.output and any(args.output.resolve() == binary or
                           (args.output.exists() and args.output.samefile(binary)) for binary in (reference, candidate)):
        parser.error("output must not overwrite either binary")
    rows = []
    try:
        for chunk in args.chunks:
            for ordinal, seed in enumerate(args.seeds):
                order = ("reference", "candidate") if ordinal % 2 == 0 else ("candidate", "reference")
                binaries = {"reference": reference, "candidate": candidate}
                results = {name: run(binaries[name], seed, args.tokens, chunk, args.timeout) for name in order}
                old, new = results["reference"], results["candidate"]
                for field in ("horizon", "parameters", "state_bytes", "forward_ops_per_token"):
                    if old.get(field) != new.get(field):
                        raise ValueError(f"unequal native resource field {field}")
                rows.append({"chunk": chunk, "seed": seed, "order": list(order), "tokens": args.tokens,
                             "updates": args.tokens // chunk,
                             "reference_train_tokens_per_s": old["training_tokens_per_s"],
                             "candidate_train_tokens_per_s": new["training_tokens_per_s"],
                             "candidate_over_reference_time": old["training_tokens_per_s"] / new["training_tokens_per_s"],
                             "reference_training_nll_6dp": old["training_nll"],
                             "candidate_training_nll_6dp": new["training_nll"],
                             "parameters": old["parameters"], "state_bytes": old["state_bytes"]})
        summary = {}
        for chunk in args.chunks:
            ratios = [float(row["candidate_over_reference_time"]) for row in rows if row["chunk"] == chunk]
            summary[str(chunk)] = {"median_time_ratio": statistics.median(ratios),
                                   "min_time_ratio": min(ratios), "max_time_ratio": max(ratios)}
        report = {"schema": "vegai-context-numeric-guard-ab/v1", "reference_sha256": digest(reference),
                  "candidate_sha256": digest(candidate), "shape": {"width": 32, "heads": 2, "ffn": 64,
                  "vocabulary": 256, "local_window": 8, "memory_scales": 2, "memory_slots": 16},
                  "task": "markov", "threads": 1, "runs": rows, "summary_by_chunk": summary,
                  "limitations": ["Synthetic training NLL is printed to six decimals, not parameter identity.",
                                  "Timings are noisy and do not establish cost, energy or general quality guarantees."]}
        output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"context numerical benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
