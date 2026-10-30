#!/usr/bin/env python3
"""Summarise the fixed-budget context quality ladder.

Input: one CONTEXT_QUALITY_JSON row per line (as printed by
`vegai_context_engine quality`).  Rows are grouped by (task, memory_scales) and
summarised over seeds: mean, standard deviation and the per-task delta against
the 4-scale baseline.

Three output modes exist so the same numbers can be shown in a job summary, in
a check-run notice (single line, length limited) and as a recall-only view:

    summarise_context_ladder.py ladder.txt              # markdown table
    summarise_context_ladder.py ladder.txt --compact     # one notice line
    summarise_context_ladder.py ladder.txt --recall      # recall line

Only the standard library is used: the CI job must not need extra packages.
"""

from __future__ import annotations

import collections
import json
import math
import statistics
import sys

TASK_NAMES = {1: "markov", 2: "needle"}   # matches FBC_TASK_MARKOV / FBC_TASK_NEEDLE
BASELINE_SCALES = 4

# The budget axis that must be identical for a comparison to mean anything.
# `parameters` and `state_bytes` are deliberately *not* in this list: they are
# expected to grow with the scale count (scale_logit has one entry per scale,
# and the recurrent state has one matrix per scale per head).
BUDGET_KEYS = ("horizon", "local_window", "memory_slots", "train_tokens", "eval_tokens", "distance")


def load(path: str) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def group(rows: list[dict]) -> dict[tuple[str, int, int], list[dict]]:
    """One group per (task, distance, scale count).

    The distance belongs to the key: a needle task at 256 tokens and the same
    task at 1024 are different problems and must not be averaged together.
    """
    grouped: dict[tuple[str, int, int], list[dict]] = collections.defaultdict(list)
    for row in rows:
        task = TASK_NAMES.get(row["task"], str(row["task"]))
        grouped[(task, int(row.get("distance", 0)), int(row["memory_scales"]))].append(row)
    return grouped


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation, ignoring non-finite entries.

    `statistics` raises on NaN rather than propagating it, and a task without
    masked answer positions has no recall to average, so finite filtering is
    part of the contract rather than a nicety.
    """
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    mean = statistics.mean(finite)
    deviation = statistics.stdev(finite) if len(finite) > 1 else 0.0
    return mean, deviation


def summarise(grouped) -> dict[tuple[str, int], dict]:
    summary = {}
    for key, runs in grouped.items():
        nll_mean, nll_sd = mean_sd([r["eval_nll"] for r in runs])
        recall_mean, recall_sd = mean_sd([r["recall"] for r in runs])
        summary[key] = {
            "answers": runs[0]["answers"],
            "parameters": runs[0]["parameters"],
            "state_bytes": runs[0]["state_bytes"],
            "nll_mean": nll_mean,
            "nll_sd": nll_sd,
            "recall_mean": recall_mean,
            "recall_sd": recall_sd,
            "seeds": len(runs),
        }
    return summary


def recall_text(row: dict) -> str:
    """Markov has no masked answer positions, so a recall figure there is noise."""
    if row["answers"] <= 0 or not math.isfinite(row["recall_mean"]):
        return "n/a"
    return f"{row['recall_mean']:.4f}"


def delta(summary, task: str, distance: int, scales: int) -> str:
    base = summary.get((task, distance, BASELINE_SCALES))
    current = summary.get((task, distance, scales))
    if base is None or current is None or scales == BASELINE_SCALES:
        return "baseline"
    change = current["nll_mean"] - base["nll_mean"]
    # A change smaller than the seed-to-seed spread is not a result.
    spread = max(base["nll_sd"], current["nll_sd"])
    verdict = "better" if change < -spread else "worse" if change > spread else "within noise"
    return f"{change:+.4f} ({verdict})"


def print_markdown(summary, grouped) -> None:
    print()
    print("## Memory-scale ladder (held-out, identical budget)")

    # Per-group, not global: different task/distance groups are meant to differ.
    mismatched = [
        key for key, runs in grouped.items()
        if len({tuple(run[k] for k in BUDGET_KEYS) for run in runs}) > 1
    ]
    if mismatched:
        print()
        print("> WARNING: the runs do not share one budget; the comparison is invalid.")
    print()
    print("Parameter and state-byte columns grow by construction: one extra decay")
    print("scale adds one `scale_logit` entry and one recurrent matrix per head.")
    print()
    print("| task | distance | scales | params | state bytes | held-out NLL | sd "
          "| delta vs 4 | recall |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key in sorted(summary):
        task, distance, scales = key
        row = summary[key]
        print(
            f"| {task} | {distance} | {scales} | {row['parameters']} | {row['state_bytes']} "
            f"| {row['nll_mean']:.4f} | {row['nll_sd']:.4f} | {delta(summary, task, distance, scales)} "
            f"| {recall_text(row)} |"
        )
    print()


def print_compact(summary) -> None:
    parts = []
    for key in sorted(summary):
        task, distance, scales = key
        row = summary[key]
        recall = recall_text(row)
        parts.append(
            f"{task}@{distance}/s{scales}:nll={row['nll_mean']:.5f}(sd{row['nll_sd']:.5f},"
            f"{row['seeds']}seeds) recall={recall}(n={row['answers']})"
        )
    print(" ".join(parts))


def print_recall(summary) -> None:
    parts = []
    for key in sorted(summary):
        task, distance, scales = key
        row = summary[key]
        if recall_text(row) == "n/a":
            parts.append(f"{task}@{distance}/s{scales}:n/a")
        else:
            parts.append(f"{task}@{distance}/s{scales}:{row['recall_mean']:.4f}"
                         f"(sd{row['recall_sd']:.4f},n={row['answers']})")
    print(" ".join(parts))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    grouped = group(load(argv[1]))
    summary = summarise(grouped)
    if "--compact" in argv[2:]:
        print_compact(summary)
    elif "--recall" in argv[2:]:
        print_recall(summary)
    else:
        print_markdown(summary, grouped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
