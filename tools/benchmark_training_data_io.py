#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure only Vegai training-data reader throughput on a caller-provided file.

This is intentionally a reader benchmark, not an HDC training benchmark. Each
sample times one full ``iter_records`` traversal after the process has started;
fixture construction, package installation, JSON rendering, and harness setup
are outside the sample. Throughput samples deliberately run *without*
``tracemalloc`` instrumentation. One separate, untimed traversal reports
Python-managed peak allocations only; Arrow's native buffers are intentionally
called out separately rather than mislabeled as total RSS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

from training_data_io import TrainingDataError, infer_format, iter_records


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="existing dataset file to read")
    parser.add_argument("--format", choices=("auto", "jsonl", "json", "txt", "parquet", "arrow"), default="auto")
    parser.add_argument("--compression", choices=("auto", "none", "gzip"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured complete reader traversals before samples (default: 1)")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.repetitions <= 0 or arguments.batch_size <= 0 or arguments.warmup < 0:
        print("training-data benchmark: FAIL: --repetitions/--batch-size must be positive and --warmup nonnegative", file=sys.stderr)
        return 2
    try:
        physical_bytes = arguments.input.stat().st_size
        format_name = infer_format(arguments.input, arguments.format)
        samples: list[float] = []
        expected_count: int | None = None
        # Warm-up is deliberately a complete streaming traversal so codec and
        # Arrow initialization/page-cache effects are not mislabeled as the
        # steady-state median. Report cold-cache measurements separately by
        # passing --warmup 0 in a fresh controlled environment.
        for _ in range(arguments.warmup):
            warm_count = sum(1 for _record in iter_records(
                arguments.input, format=format_name, compression=arguments.compression, batch_size=arguments.batch_size
            ))
            if expected_count is None:
                expected_count = warm_count
            elif warm_count != expected_count:
                raise TrainingDataError("reader count changed across identical benchmark warm-ups")
        for _ in range(arguments.repetitions):
            started = time.perf_counter_ns()
            count = sum(1 for _record in iter_records(
                arguments.input, format=format_name, compression=arguments.compression, batch_size=arguments.batch_size
            ))
            elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
            if expected_count is None:
                expected_count = count
            elif count != expected_count:
                raise TrainingDataError("reader count changed across identical benchmark repetitions")
            samples.append(elapsed)

        # Keep allocation instrumentation outside throughput samples. This is
        # one extra full stream, not a process RSS claim.
        tracemalloc.start()
        memory_count = sum(1 for _record in iter_records(
            arguments.input, format=format_name, compression=arguments.compression, batch_size=arguments.batch_size
        ))
        _current, memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if memory_count != expected_count:
            raise TrainingDataError("reader count changed during untimed memory traversal")
        median = statistics.median(samples)
        report = {
            "schema": "vegai-training-data-reader-benchmark/v1",
            "timing_scope": (
                "one complete streaming iter_records traversal per measured sample; fixture construction, installation, "
                "JSON rendering, harness setup, configured warm-up traversals, and tracemalloc instrumentation excluded"
            ),
            "input": {"path": str(arguments.input), "physical_bytes": physical_bytes, "format": format_name},
            "configuration": {
                "batch_size": arguments.batch_size,
                "warmup_repetitions_excluded": arguments.warmup,
                "repetitions": arguments.repetitions,
            },
            "records": expected_count,
            "samples_seconds": [round(sample, 9) for sample in samples],
            "median_seconds": round(median, 9),
            "records_per_second": round(expected_count / median, 3) if median else None,
            "physical_mib_per_second": round(physical_bytes / median / (1024 * 1024), 3) if median else None,
            "python_tracemalloc_peak_bytes": memory_peak,
            "memory_measurement_scope": "one additional untimed full reader traversal with tracemalloc enabled",
            "memory_note": (
                "tracemalloc excludes Arrow native allocator buffers and OS page cache; use host RSS/cgroup tooling "
                "alongside this report for a deployment memory limit."
            ),
        }
    except (OSError, TrainingDataError, ValueError) as error:
        print(f"training-data benchmark: FAIL: {error}", file=sys.stderr)
        return 1
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
