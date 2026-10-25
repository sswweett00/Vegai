#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure real LMM gateway child-process latency without setup/compile time.

The command configuration must invoke ``lmm_gateway.py --print-json`` against
an actual selected backend.  Warmups are excluded from samples.  The tool does
not invent token/s or accelerator claims: it reports backend-provided output
counts when available and records the complete child wall time, including model
load if the configured command performs a cold load.  For resident throughput,
benchmark a persistent OpenAI-compatible/worker service rather than a process
that reloads a model each sample.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

from lmm_contract import LMMContractError, FileIdentity, file_identity, require, require_unchanged, write_json_atomic
from lmm_process import run_bounded_process


DEFAULT_TIMEOUT_SECONDS = 300.0
MAX_GATEWAY_STDOUT_BYTES = 8 * 1024 * 1024
MAX_GATEWAY_STDERR_BYTES = 1 * 1024 * 1024


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def load_command(path: Path) -> tuple[list[str], FileIdentity]:
    identity = file_identity(path, "benchmark command configuration")
    try:
        value = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LMMContractError(f"cannot parse benchmark command configuration: {error}") from error
    require(isinstance(value, list) and value and all(isinstance(item, str) and item for item in value),
            "benchmark command configuration must be a nonempty JSON string array")
    require("--print-json" in value, "benchmark command must include --print-json for token accounting")
    require_unchanged(path, identity, "benchmark command configuration")
    return list(value), identity


def run_sample(command: list[str], timeout_seconds: float) -> dict[str, float | int | None]:
    completed = run_bounded_process(
        command,
        cwd=Path(__file__).resolve().parents[1],
        timeout_seconds=timeout_seconds,
        stdout_limit_bytes=MAX_GATEWAY_STDOUT_BYTES,
        stderr_limit_bytes=MAX_GATEWAY_STDERR_BYTES,
        label="benchmark child",
    )
    elapsed = completed.elapsed_seconds
    if completed.returncode != 0:
        raise LMMContractError(
            f"benchmark child exited {completed.returncode}; stderr-sha256={completed.stderr_sha256}; "
            f"stderr-bytes={completed.stderr_bytes}"
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        report = payload["report"]
        response = report["response"]
        output_tokens = response["output_tokens"]
        input_tokens = response["input_tokens"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise LMMContractError("benchmark child did not emit lmm_gateway --print-json output") from error
    require(type(output_tokens) is int and output_tokens >= 0, "gateway output_tokens is invalid")
    require(type(input_tokens) is int and input_tokens >= 0, "gateway input_tokens is invalid")
    return {
        "elapsed_seconds": round(elapsed, 9),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tokens_per_second": None if output_tokens == 0 else round(output_tokens / elapsed, 9),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-json", required=True, type=Path, help="JSON string-array invoking lmm_gateway.py --print-json")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=parse_positive_int, default=3)
    parser.add_argument("--timeout-seconds", type=parse_positive_float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        require(arguments.warmups >= 0, "--warmups must be nonnegative")
        command, command_identity = load_command(arguments.command_json)
        require(arguments.output.resolve() != arguments.command_json.resolve(),
                "--output must not overwrite benchmark command configuration")
        for _ in range(arguments.warmups):
            run_sample(command, arguments.timeout_seconds)
        samples = [run_sample(command, arguments.timeout_seconds) for _ in range(arguments.repetitions)]
        require_unchanged(arguments.command_json, command_identity, "benchmark command configuration")
        elapsed = [float(sample["elapsed_seconds"]) for sample in samples]
        token_rates = [float(sample["output_tokens_per_second"]) for sample in samples if sample["output_tokens_per_second"] is not None]
        report: dict[str, Any] = {
            "schema_version": "vegai-lmm-gateway-benchmark/v1",
            "methodology": {
                "timed_unit": "one complete configured lmm_gateway child process",
                "excluded": "warmups, command construction, report aggregation, and Python benchmark harness work",
                "included": "the configured backend's model/context load and generation work; use a resident service command for warm throughput",
                "accelerator_truth": "device/provider fields are backend reports, not an independent accelerator execution proof",
            },
            "command_configuration": command_identity.to_mapping(),
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
            "samples": samples,
            "latency_seconds": {
                "min": round(min(elapsed), 9),
                "median": round(statistics.median(elapsed), 9),
                "max": round(max(elapsed), 9),
            },
            "output_tokens_per_second": None if not token_rates else {
                "min": round(min(token_rates), 9),
                "median": round(statistics.median(token_rates), 9),
                "max": round(max(token_rates), 9),
            },
        }
        write_json_atomic(arguments.output, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except (LMMContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"lmm benchmark: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
