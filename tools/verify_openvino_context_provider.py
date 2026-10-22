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
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "openvino_context_worker.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(command: list[str], label: str) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise AssertionError(f"{label} failed ({result.returncode})\n{' '.join(command)}\n{result.stdout}")
    return result.stdout


def metric(output: str, label: str) -> int:
    prefix = f"  {label}:"
    for line in output.splitlines():
        if line.startswith(prefix):
            return int(line.split(":", maxsplit=1)[1].strip())
    raise AssertionError(f"missing metric {label!r}\n{output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", type=Path, required=True)
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
            run(cpu_command, "CPU reference training")

            provider_command = common.copy()
            provider_command[3] = str(provider_checkpoint)
            provider_command.extend(
                [
                    "--context-provider", "openvino", "--openvino-worker", str(WORKER),
                    "--openvino-python", sys.executable, "--openvino-device", "CPU",
                    "--openvino-work-dir", str(directory), "--openvino-min-contexts", "1", "--openvino-required",
                ]
            )
            output = run(provider_command, "OpenVINO context-provider training")
            require(metric(output, "OpenVINO-dispatched contexts") > 0, "worker never received a context")
            require(metric(output, "OpenVINO CPU-fallback contexts") == 0, "provider unexpectedly used CPU fallback")
            require("OpenVINO execution devices: CPU=T GPU=F NPU=F" in output, "provider execution truth was not CPU")
            require(cpu_checkpoint.read_bytes() == provider_checkpoint.read_bytes(), "CPU and OpenVINO checkpoints differ")
    except (AssertionError, OSError) as error:
        print(f"OpenVINO context provider verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("OpenVINO context provider verification: PASS (OpenVINO graph contexts, byte-identical checkpoint)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
