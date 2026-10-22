#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify that the optional OpenVINO provider reports capability truthfully."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tools" / "openvino_provider.py"


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="vegai-openvino-probe-") as temporary:
            report_path = Path(temporary) / "probe.json"
            result = subprocess.run(
                [sys.executable, str(PROBE), "--format", "json", "--output", str(report_path)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode != 0:
                fail(f"optional OpenVINO probe unexpectedly failed ({result.returncode})\n{result.stdout}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("schema") != "vegai-openvino-provider-probe/v2":
                fail(f"unexpected OpenVINO probe schema: {report.get('schema')!r}")
            if type(report.get("openvino_installed")) is not bool:
                fail("OpenVINO probe did not report installation state")
            if not isinstance(report.get("available_devices"), list):
                fail("OpenVINO probe did not return a device list")
            dispatch = report.get("training_dispatch")
            if not isinstance(dispatch, dict):
                fail("OpenVINO probe omitted training dispatch truth object")
            if dispatch.get("cpu_compute_used") is not True:
                fail("OpenVINO probe did not preserve CPU training truth")
            if dispatch.get("gpu_compute_used") is not False or dispatch.get("npu_compute_used") is not False:
                fail("OpenVINO probe falsely claimed GPU/NPU training dispatch")
            if dispatch.get("hdc_context_offload_integrated") is not False:
                fail("OpenVINO probe falsely claimed full HDC context offload")

            if not report["openvino_installed"]:
                rejected = subprocess.run(
                    [sys.executable, str(PROBE), "--require-openvino"],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if rejected.returncode == 0 or "FAIL" not in rejected.stdout:
                    fail(f"OpenVINO required-mode did not fail closed\n{rejected.stdout}")
            else:
                probe = report.get("uint32_lane_hdc_bundle_probe")
                if not isinstance(probe, dict) or not probe.get("attempted"):
                    fail("installed OpenVINO did not attempt its uint32-lane HDC primitive probe")
                if probe.get("lane_count") != 512 or probe.get("native_word_count") != 256:
                    fail("OpenVINO HDC probe reported an incompatible packed-vector ABI")
                if not isinstance(probe.get("cases"), list) or not probe["cases"]:
                    fail("installed OpenVINO did not exercise its HDC bundle cases")
    except (AssertionError, OSError, json.JSONDecodeError) as error:
        print(f"OpenVINO provider verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("OpenVINO provider verification: PASS (optional provider, no false GPU/NPU training claim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
