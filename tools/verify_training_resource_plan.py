#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Contract test for the bounded RAM/SSD/OpenMP training preflight."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PLANNER = ROOT / "tools" / "training_resource_plan.py"
CORPUS = ROOT / "examples" / "corpus" / "mini_turkish_ascii_pretrain.txt"


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="vegai-resource-plan-") as temporary:
            directory = Path(temporary)
            report_path = directory / "plan.json"
            checkpoint = directory / "checkpoint.vgai"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PLANNER),
                    "--corpus",
                    str(CORPUS),
                    "--checkpoint",
                    str(checkpoint),
                    "--max-vocab",
                    "128",
                    "--omp-threads",
                    "2",
                    "--format",
                    "json",
                    "--output",
                    str(report_path),
                    "--require-ready",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode != 0:
                fail(f"resource planner failed ({result.returncode})\n{result.stdout}")
            plan = json.loads(report_path.read_text(encoding="utf-8"))
            if plan.get("schema") != "vegai-training-resource-plan/v1":
                fail(f"unexpected planner schema: {plan.get('schema')!r}")
            if plan.get("corpus", {}).get("path") != str(CORPUS.resolve()):
                fail("planner did not preserve the resolved corpus path")
            memory = plan.get("memory")
            if not isinstance(memory, dict) or memory.get("max_vocabulary") != 128:
                fail("planner omitted requested vocabulary capacity")
            if not isinstance(memory.get("required_available_bytes"), int) or memory["required_available_bytes"] <= 0:
                fail("planner did not calculate a positive RAM reservation")
            if memory["required_available_bytes"] <= memory.get("model_payload_bytes", 0):
                fail("planner reservation omitted vocabulary/staging/headroom")
            checkpoint_plan = plan.get("checkpoint", {})
            if checkpoint_plan.get("path") != str(checkpoint.resolve()):
                fail("planner did not preserve checkpoint path")
            if not isinstance(checkpoint_plan.get("estimated_max_bytes"), int) or checkpoint_plan["estimated_max_bytes"] <= 0:
                fail("planner did not calculate checkpoint storage")
            environment = plan.get("cpu_execution", {}).get("environment", {})
            if environment.get("OMP_PROC_BIND") != "close" or environment.get("OMP_PLACES") != "cores":
                fail("planner omitted affinity-safe OpenMP recommendations")
            if environment.get("OMP_NUM_THREADS") not in {"1", "2"}:
                fail(f"planner emitted unexpected OMP thread recommendation: {environment.get('OMP_NUM_THREADS')!r}")
            accelerators = plan.get("accelerators", {})
            if accelerators.get("cpu_compute_used") is not True:
                fail("planner did not identify the active CPU backend")
            if accelerators.get("gpu_compute_used") is not False or accelerators.get("npu_compute_used") is not False:
                fail("planner falsely claimed GPU/NPU training dispatch")

            # A request for an unavailable accelerator must fail rather than
            # silently fall back while presenting an accelerator claim.
            rejected = subprocess.run(
                [sys.executable, str(PLANNER), "--corpus", str(CORPUS), "--require-accelerator"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if rejected.returncode == 0 or "NOT READY" not in rejected.stdout:
                fail(f"planner did not fail closed for accelerator request\n{rejected.stdout}")
    except (AssertionError, OSError, json.JSONDecodeError) as error:
        print(f"training resource plan verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("training resource plan verification: PASS (bounded RAM/storage plan, static accelerator fail-closed truth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
