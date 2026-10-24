#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a small, real portable-binary interoperability smoke check.

The checker intentionally uses only stdlib subprocess/path APIs so it can run
unchanged on Linux, macOS, and Windows.  It validates that a CMake-built
``hds_engine`` and ``vegai_pretrain`` executable can be launched on the current
host, can create a checkpoint, and can inspect that checkpoint using the same
binary.  It is a correctness/launcher check, not a throughput benchmark and
not a checkpoint interchange claim across compilers or operating systems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples" / "corpus" / "toy_hdc_corpus.txt"
SCHEMA = "vegai-portable-runtime-smoke/v1"


class PortableRuntimeError(RuntimeError):
    """Expected portable runtime verification failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PortableRuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, timeout_seconds: float) -> str:
    environment = os.environ.copy()
    # A one-thread smoke path works both with a linked OpenMP runtime and the
    # serial CMake preset.  It also avoids treating the smoke as a host-speed
    # measurement.
    environment["OMP_NUM_THREADS"] = "1"
    environment["OMP_DYNAMIC"] = "FALSE"
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
    except OSError as error:
        raise PortableRuntimeError(f"cannot launch {' '.join(command)!r}: {error}") from error
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        raise PortableRuntimeError(
            f"portable runtime command exceeded {timeout_seconds:g} seconds: {' '.join(command)!r}\n{output[-4000:]}"
        ) from error
    if result.returncode != 0:
        raise PortableRuntimeError(
            f"portable runtime command failed with status {result.returncode}: {' '.join(command)!r}\n{result.stdout[-8000:]}"
        )
    return result.stdout


def resolve_binary(path: Path, label: str) -> Path:
    binary = path.resolve()
    require(binary.is_file(), f"{label} binary does not exist: {binary}")
    return binary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-bin", required=True, type=Path, help="CMake-built hds_engine executable")
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="CMake-built vegai_pretrain executable")
    parser.add_argument("--timeout-seconds", type=float, default=90.0, help="timeout per child process (default: 90)")
    parser.add_argument("--output", type=Path, help="optional JSON evidence path; stdout remains JSON")
    arguments = parser.parse_args()

    try:
        require(arguments.timeout_seconds > 0.0, "--timeout-seconds must be positive")
        engine = resolve_binary(arguments.engine_bin, "hds_engine")
        pretrain = resolve_binary(arguments.pretrain_bin, "vegai_pretrain")
        require(CORPUS.is_file(), f"repository smoke corpus is missing: {CORPUS}")

        engine_output = run([str(engine)], timeout_seconds=arguments.timeout_seconds)
        require("VERIFICATION RESULT: PASS" in engine_output, "hds_engine did not report its verification pass marker")

        help_output = run([str(pretrain), "help"], timeout_seconds=arguments.timeout_seconds)
        require("Vegai deterministic HDC/VSA corpus pretraining" in help_output, "vegai_pretrain help marker is missing")

        with tempfile.TemporaryDirectory(prefix="vegai-portable-runtime-") as temporary:
            checkpoint = Path(temporary) / "portable-smoke.vgai"
            training_output = run(
                [
                    str(pretrain),
                    "train",
                    str(CORPUS),
                    str(checkpoint),
                    "--max-vocab",
                    "64",
                    "--min-frequency",
                    "2",
                    "--window",
                    "2",
                    "--position-shift",
                    "17",
                    "--context-mode",
                    "hierarchical",
                    "--epochs",
                    "1",
                    "--strict",
                    "--query",
                    "dog",
                ],
                timeout_seconds=arguments.timeout_seconds,
            )
            require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "training did not create a nonempty checkpoint")
            require("phase: streaming HDC context learning" in training_output, "training output omitted learning-phase marker")
            require("completed epochs: 1" in training_output, "training did not complete exactly one smoke epoch")
            inspect_output = run(
                [str(pretrain), "inspect", str(checkpoint), "--query", "dog"],
                timeout_seconds=arguments.timeout_seconds,
            )
            require("Vegai HDC pretraining checkpoint" in inspect_output, "checkpoint inspect marker is missing")
            checkpoint_sha256 = sha256_file(checkpoint)
            checkpoint_bytes = checkpoint.stat().st_size

        report: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "passed",
            "host": {
                "operating_system": platform.system() or "unknown",
                "architecture": platform.machine() or "unknown",
            },
            "binaries": {
                "hds_engine": {"path": str(engine), "sha256": sha256_file(engine)},
                "vegai_pretrain": {"path": str(pretrain), "sha256": sha256_file(pretrain)},
            },
            "checkpoint": {"bytes": checkpoint_bytes, "sha256": checkpoint_sha256},
            "contract": (
                "same-host portable launch, train, and inspect only; this report does not assert cross-OS, "
                "cross-architecture, or cross-compiler checkpoint interchange"
            ),
        }
    except (PortableRuntimeError, OSError, ValueError) as error:
        print(f"portable runtime verification: FAIL: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print("portable runtime verification: PASS (same-host engine/train/inspect contract)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
