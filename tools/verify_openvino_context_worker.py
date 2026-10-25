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
import time


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


def read_and_verify_response(
    path: Path,
    numpy: object,
    groups: list[tuple[int, object, object]],
    expected_contexts: list[object] | None = None,
) -> None:
    """Validate wire contexts; optional expected values test neutral padding."""
    with path.open("rb") as stream:
        if stream.read(8) != OUTPUT_MAGIC:
            fail("worker response magic mismatch")
        version, group_count, lane_count, execution_mask = struct.unpack("<4i", stream.read(16))
        if (version, group_count, lane_count) != (1, len(groups), HDC_LANES):
            fail(f"worker response header mismatch: {(version, group_count, lane_count)!r}")
        if execution_mask <= 0 or execution_mask & ~7:
            fail(f"worker response has invalid execution mask {execution_mask}")
        for group_index, (item_count, items, ties) in enumerate(groups):
            response_items, context_count = struct.unpack("<2i", stream.read(8))
            if (response_items, context_count) != (item_count, int(items.shape[0])):
                fail("worker response group header mismatch")
            raw = numpy.frombuffer(stream.read(context_count * HDC_LANES * 4), dtype="<u4")
            if raw.size != context_count * HDC_LANES:
                fail("worker response context payload is truncated")
            got = numpy.transpose(raw.reshape((HDC_LANES, context_count), order="F"), (1, 0))
            if expected_contexts is None:
                expected = numpy.array(
                    [hdc_bundle_reference(numpy, items[index], ties[index]) for index in range(context_count)],
                    dtype=numpy.uint32,
                )
            else:
                expected = expected_contexts[group_index]
            if not numpy.array_equal(got, expected):
                fail(f"worker response diverged from HDC oracle for {item_count}-item group")
        if stream.read(1) != b"":
            fail("worker response contains trailing bytes")


def neutral_pad_contexts(numpy: object, actual_items: object, ties: object, wire_item_count: int) -> object:
    """Mirror Fortran's (tie, NOT tie) neutral-majority padding contract."""
    context_count, actual_item_count, lane_count = actual_items.shape
    if wire_item_count < actual_item_count or (wire_item_count - actual_item_count) % 2:
        fail("invalid neutral-padding test shape")
    padded = numpy.zeros((context_count, wire_item_count, lane_count), dtype=numpy.uint32)
    padded[:, :actual_item_count, :] = actual_items
    for padding_index in range(actual_item_count, wire_item_count, 2):
        padded[:, padding_index, :] = ties
        padded[:, padding_index + 1, :] = numpy.bitwise_not(ties)
    return padded


def wait_for(path: Path, failure: Path, process: subprocess.Popen[str], label: str) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if path.exists():
            return
        if failure.exists():
            fail(f"resident worker {label} failed: {failure.read_text(encoding='utf-8', errors='replace')}")
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            fail(f"resident worker exited while waiting for {label} ({process.returncode})\\n{stdout}\\n{stderr}")
        time.sleep(0.001)
    fail(f"timed out waiting for resident worker {label}")


def verify_resident_session(directory: Path, numpy: object, groups: list[tuple[int, object, object]]) -> None:
    """Exercise one server across two requests and prove its graph-cache reuse."""
    session = directory / "session"
    report_path = directory / "resident-report.json"
    process = subprocess.Popen(
        [
            sys.executable, str(WORKER), "--serve", str(session), "--device", "CPU",
            "--report", str(report_path), "--quiet",
        ],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        wait_for(session / "ready.bin", session / "startup.error", process, "ready marker")
        for request_index in range(2):
            write_request(session / "request.bin", numpy, groups)
            (session / "request.ready").write_bytes(b"1\\n")
            wait_for(session / "response.ready", session / "response.error", process, f"response {request_index}")
            read_and_verify_response(session / "response.bin", numpy, groups)
            (session / "response.ready").unlink()
            (session / "response.bin").unlink()
            (session / "request.bin").unlink()
        (session / "shutdown.ready").write_bytes(b"1\\n")
        wait_for(session / "stopped.ready", session / "response.error", process, "shutdown")
        (session / "stopped.ack").write_bytes(b"1\\n")
        process.wait(timeout=5)
        if process.returncode != 0:
            stdout, stderr = process.communicate(timeout=1)
            fail(f"resident worker stopped unsuccessfully ({process.returncode})\\n{stdout}\\n{stderr}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cache = report.get("cache", {})
        if cache.get("requests_completed") != 2 or cache.get("compile_cache_misses") != len(groups):
            fail(f"resident worker did not retain the expected compiled graphs: {report!r}")
        if cache.get("compile_cache_hits", 0) < len(groups):
            fail(f"resident worker did not reuse compiled graphs: {report!r}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


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

            # The Fortran provider normalizes variable actual cardinalities to
            # one odd and one even compiled shape with neutral tie/complement
            # pairs.  Verify that this optimization remains bit-identical to
            # majority over the original, unpadded contexts.
            normalized_groups = []
            normalized_expected = []
            for wire_item_count, actual_counts in ((19, (11, 13, 15, 17, 19)), (18, (10, 12, 14, 16, 18))):
                ties = generator.integers(0, 2**32, size=(len(actual_counts), HDC_LANES), dtype=numpy.uint32)
                padded_contexts = []
                expected_contexts = []
                for context_index, actual_item_count in enumerate(actual_counts):
                    actual = generator.integers(
                        0, 2**32, size=(1, actual_item_count, HDC_LANES), dtype=numpy.uint32,
                    )
                    padded_contexts.append(neutral_pad_contexts(numpy, actual, ties[context_index:context_index + 1], wire_item_count)[0])
                    expected_contexts.append(hdc_bundle_reference(numpy, actual[0], ties[context_index]))
                normalized_groups.append((wire_item_count, numpy.stack(padded_contexts), ties))
                normalized_expected.append(numpy.array(expected_contexts, dtype=numpy.uint32))
            normalized_input = directory / "normalized-request.bin"
            normalized_output = directory / "normalized-response.bin"
            write_request(normalized_input, numpy, normalized_groups)
            normalized_result = subprocess.run(
                [
                    sys.executable, str(WORKER), "--input", str(normalized_input), "--output", str(normalized_output),
                    "--device", "CPU", "--quiet",
                ],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            if normalized_result.returncode != 0:
                fail(f"neutral-padding CPU worker failed\\n{normalized_result.stdout}")
            read_and_verify_response(normalized_output, numpy, normalized_groups, normalized_expected)

            # The resident service must compile each shape once and retain it
            # across independent record requests.
            verify_resident_session(directory, numpy, groups)
    except (AssertionError, OSError, struct.error, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(f"OpenVINO context worker verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("OpenVINO context worker verification: PASS (exact uint32-lane protocol and CPU graph)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
