#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify exact CPU/OpenVINO context-provider checkpoint equivalence.

The test is intentionally optional: when OpenVINO is absent it reports a skip
and succeeds so portable CMake/Make users do not acquire a vendor runtime. CI
installs the optional requirement for an actual bridge test. A target-host GPU
or NPU can use the same trainer CLI with ``--openvino-require-accelerator``.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "openvino_context_worker.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(command: list[str], label: str) -> tuple[str, float]:
    # Timing scope is one real vegai_pretrain child process only. Corpus
    # construction, Python comparison and checkpoint byte comparison are out of
    # scope, matching the repository's benchmark contract.
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise AssertionError(f"{label} failed ({result.returncode})\n{' '.join(command)}\n{result.stdout}")
    return result.stdout, elapsed


def metric(output: str, label: str) -> int:
    prefix = f"  {label}:"
    for line in output.splitlines():
        if line.startswith(prefix):
            return int(line.split(":", maxsplit=1)[1].strip())
    raise AssertionError(f"missing metric {label!r}\n{output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="optional JSON report containing train-child timings")
    arguments = parser.parse_args()
    try:
        try:
            importlib.import_module("openvino")
            importlib.import_module("numpy")
        except ImportError:
            print("OpenVINO context provider verification: SKIP (optional runtime absent)")
            return 0
        require(arguments.pretrain_bin.is_file(), f"pretrain binary does not exist: {arguments.pretrain_bin}")

        with tempfile.TemporaryDirectory(prefix="vegai-openvino-provider-") as temporary:
            directory = Path(temporary)
            corpus = directory / "long-records.txt"
            # Two long records exercise both 16-item hybrid local contexts and
            # 19-item hierarchical contexts, including even seeded ties.
            corpus.write_text(
                "ankara bogaz firtina liman deniz dag yayla hamsi cayi sokak okul ev bahce pazar kitap\n"
                "liman deniz hamsi cayi ankara bogaz firtina dag yayla sokak okul ev bahce pazar kitap\n",
                encoding="ascii",
            )
            cpu_checkpoint = directory / "cpu.vgai"
            provider_checkpoint = directory / "openvino.vgai"
            common = [
                str(arguments.pretrain_bin), "train", str(corpus), "", "--max-vocab", "64", "--min-frequency", "1",
                "--window", "4", "--context-mode", "hierarchical", "--epochs", "1", "--serial-context-build",
            ]
            cpu_command = common.copy()
            cpu_command[3] = str(cpu_checkpoint)
            _, cpu_seconds = run(cpu_command, "CPU reference training")

            # CPU is not an accelerator for this packed-bit workload.  In
            # normal fallback mode an explicit CPU policy must therefore stay
            # on the native fused/OpenMP implementation rather than launching
            # Python/OpenVINO.  This is the production speed path; the strict
            # provider arm below remains an exact bridge correctness check.
            bypass_checkpoint = directory / "native-cpu-bypass.vgai"
            bypass_command = common.copy()
            bypass_command[3] = str(bypass_checkpoint)
            bypass_command.extend(
                [
                    "--context-provider", "openvino", "--openvino-worker", str(WORKER),
                    "--openvino-python", sys.executable, "--openvino-device", "CPU",
                    "--openvino-work-dir", str(directory), "--openvino-min-contexts", "1", "--openvino-fallback-cpu",
                ]
            )
            bypass_output, bypass_seconds = run(bypass_command, "native CPU bypass training")
            require(metric(bypass_output, "OpenVINO-dispatched contexts") == 0, "CPU bypass dispatched OpenVINO contexts")
            require(metric(bypass_output, "OpenVINO resident worker sessions") == 0, "CPU bypass launched a worker")
            require(metric(bypass_output, "OpenVINO worker requests") == 0, "CPU bypass sent a worker request")
            require(metric(bypass_output, "OpenVINO native-CPU bypass contexts") == 30, "CPU bypass did not cover all contexts")
            require(metric(bypass_output, "OpenVINO CPU-fallback contexts") == 0, "CPU bypass was reported as a worker failure")
            require("OpenVINO execution devices: CPU=F GPU=F NPU=F" in bypass_output, "CPU bypass made a false device-execution claim")
            require(cpu_checkpoint.read_bytes() == bypass_checkpoint.read_bytes(), "CPU bypass checkpoint differs from CPU reference")
            # This tiny fixture is deliberately startup-sensitive.  The bound
            # allows normal host jitter while guarding against a regression to
            # a per-line Python worker launch in the production CPU path.
            require(bypass_seconds <= cpu_seconds * 3.0, "native CPU bypass regressed beyond the bounded overhead guard")

            provider_command = common.copy()
            provider_command[3] = str(provider_checkpoint)
            provider_command.extend(
                [
                    "--context-provider", "openvino", "--openvino-worker", str(WORKER),
                    "--openvino-python", sys.executable, "--openvino-device", "CPU",
                    "--openvino-work-dir", str(directory), "--openvino-min-contexts", "1", "--openvino-required",
                ]
            )
            output, provider_seconds = run(provider_command, "OpenVINO context-provider training")
            dispatched_contexts = metric(output, "OpenVINO-dispatched contexts")
            worker_sessions = metric(output, "OpenVINO resident worker sessions")
            worker_requests = metric(output, "OpenVINO worker requests")
            require(dispatched_contexts > 0, "worker never received a context")
            require(worker_sessions == 1, f"expected one resident worker session, got {worker_sessions}")
            require(worker_requests == 2, f"expected one worker request per corpus record, got {worker_requests}")
            require(metric(output, "OpenVINO native-CPU bypass contexts") == 0, "strict provider unexpectedly bypassed OpenVINO")
            require(metric(output, "OpenVINO CPU-fallback contexts") == 0, "provider unexpectedly used CPU fallback")
            require("OpenVINO execution devices: CPU=T GPU=F NPU=F" in output, "provider execution truth was not CPU")
            require(cpu_checkpoint.read_bytes() == provider_checkpoint.read_bytes(), "CPU and OpenVINO checkpoints differ")
            report = {
                "schema": "vegai-openvino-context-provider-verification/v3",
                "runtime": "OpenVINO CPU plugin",
                "workload": {"records": 2, "active_tokens": 30, "context_mode": "hierarchical", "window": 4},
                "cpu_reference_train_seconds": cpu_seconds,
                "native_cpu_bypass_train_seconds": bypass_seconds,
                "native_cpu_bypass_to_cpu_time_ratio": bypass_seconds / cpu_seconds if cpu_seconds > 0.0 else None,
                "openvino_context_provider_train_seconds": provider_seconds,
                "openvino_to_cpu_time_ratio": provider_seconds / cpu_seconds if cpu_seconds > 0.0 else None,
                "openvino_dispatched_contexts": dispatched_contexts,
                "openvino_resident_worker_sessions": worker_sessions,
                "openvino_worker_requests": worker_requests,
                "checkpoint_byte_identical": True,
            }
            if arguments.report is not None:
                arguments.report.parent.mkdir(parents=True, exist_ok=True)
                arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (AssertionError, OSError) as error:
        print(f"OpenVINO context provider verification: FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "OpenVINO context provider verification: PASS "
        f"(native CPU bypass + OpenVINO graph contexts, byte-identical checkpoints; "
        f"cpu_train_seconds={cpu_seconds:.6f}; native_cpu_bypass_seconds={bypass_seconds:.6f}; "
        f"openvino_context_provider_train_seconds={provider_seconds:.6f}; "
        f"strict_provider_ratio={provider_seconds / cpu_seconds:.4f}x)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
