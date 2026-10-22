#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the optional OpenVINO context-worker protocol and fail-closed path."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "openvino_context_worker.py"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
from openvino_provider import HDC_LANES, hdc_bundle_reference  # noqa: E402


INPUT_MAGIC = b"VGHDCI01"
OUTPUT_MAGIC = b"VGHDCO01"


def fail(message: str) -> None:
    raise AssertionError(message)


def openvino_available() -> bool:
    try:
        importlib.import_module("openvino")
        importlib.import_module("numpy")
    except ImportError:
        return False
    return True


def write_request(path: Path, numpy: object, groups: list[tuple[int, object, object]]) -> None:
    with path.open("wb") as stream:
        stream.write(INPUT_MAGIC)
        stream.write(struct.pack("<4i", 1, len(groups), HDC_LANES, 0))
        for item_count, items, ties in groups:
            context_count = int(items.shape[0])
            stream.write(struct.pack("<2i", item_count, context_count))
            fortran_items = numpy.transpose(items, (2, 1, 0))
            stream.write(fortran_items.ravel(order="F").astype("<u4", copy=False).tobytes())
            fortran_ties = numpy.transpose(ties, (1, 0))
            stream.write(fortran_ties.ravel(order="F").astype("<u4", copy=False).tobytes())


def read_and_verify_response(path: Path, numpy: object, groups: list[tuple[int, object, object]]) -> None:
    with path.open("rb") as stream:
        if stream.read(8) != OUTPUT_MAGIC:
            fail("worker response magic mismatch")
        version, group_count, lane_count, execution_mask = struct.unpack("<4i", stream.read(16))
        if (version, group_count, lane_count) != (1, len(groups), HDC_LANES):
            fail(f"worker response header mismatch: {(version, group_count, lane_count)!r}")
        if execution_mask <= 0 or execution_mask & ~7:
            fail(f"worker response has invalid execution mask {execution_mask}")
        for item_count, items, ties in groups:
            response_items, context_count = struct.unpack("<2i", stream.read(8))
            if (response_items, context_count) != (item_count, int(items.shape[0])):
                fail("worker response group header mismatch")
            raw = numpy.frombuffer(stream.read(context_count * HDC_LANES * 4), dtype="<u4")
            if raw.size != context_count * HDC_LANES:
                fail("worker response context payload is truncated")
            got = numpy.transpose(raw.reshape((HDC_LANES, context_count), order="F"), (1, 0))
            expected = numpy.array(
                [hdc_bundle_reference(numpy, items[index], ties[index]) for index in range(context_count)],
                dtype=numpy.uint32,
            )
            if not numpy.array_equal(got, expected):
                fail(f"worker response diverged from HDC oracle for {item_count}-item group")
        if stream.read(1) != b"":
            fail("worker response contains trailing bytes")


def main() -> int:
    try:
        help_result = subprocess.run(
            [sys.executable, str(WORKER), "--help"], cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        if help_result.returncode != 0 or "OpenVINO" not in help_result.stdout:
            fail(f"worker help failed\n{help_result.stdout}")

        if not openvino_available():
            rejected = subprocess.run(
                [sys.executable, str(WORKER), "--input", "missing.bin", "--output", "missing-out.bin"],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            if rejected.returncode == 0 or "FAIL" not in rejected.stdout:
                fail(f"worker did not fail closed without optional OpenVINO\n{rejected.stdout}")
            print("OpenVINO context worker verification: PASS (optional runtime absent; fail closed)")
            return 0

        numpy = importlib.import_module("numpy")
        with tempfile.TemporaryDirectory(prefix="vegai-openvino-worker-") as temporary:
            directory = Path(temporary)
            input_path = directory / "request.bin"
            output_path = directory / "response.bin"
            report_path = directory / "report.json"
            generator = numpy.random.default_rng(20261022)
            groups = []
            for item_count, context_count in ((16, 3), (19, 2), (2, 4), (5, 1)):
                items = generator.integers(0, 2**32, size=(context_count, item_count, HDC_LANES), dtype=numpy.uint32)
                ties = generator.integers(0, 2**32, size=(context_count, HDC_LANES), dtype=numpy.uint32)
                groups.append((item_count, items, ties))
            write_request(input_path, numpy, groups)
            result = subprocess.run(
                [
                    sys.executable, str(WORKER), "--input", str(input_path), "--output", str(output_path),
                    "--device", "CPU", "--report", str(report_path), "--quiet",
                ],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            if result.returncode != 0:
                fail(f"CPU worker failed\n{result.stdout}")
            read_and_verify_response(output_path, numpy, groups)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("execution_devices") != ["CPU"]:
                fail(f"unexpected CPU worker execution report: {report!r}")
    except (AssertionError, OSError, struct.error, json.JSONDecodeError) as error:
        print(f"OpenVINO context worker verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("OpenVINO context worker verification: PASS (exact uint32-lane protocol and CPU graph)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
