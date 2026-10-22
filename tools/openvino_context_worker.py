#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute batched immutable Vegai HDC context bundles through OpenVINO.

This is the optional process boundary used by the Fortran trainer's OpenVINO
context provider. It deliberately handles only immutable, pre-positioned HDC
items and returns finalized context vectors; source-ordered prototype/Hebbian
commits remain in Fortran. The binary protocol is little-endian and uses an
explicit uint32-lane ABI so it never relies on a provider's uint64 bitwise
semantics.

Single-request input layout, all little-endian:

    8 bytes  magic VGHDCI01
    4*int32  version=1, group_count, lane_count=512, reserved=0
    repeated group_count times:
      2*int32 item_count, context_count
      uint32  items(512, item_count, context_count), Fortran-column order
      uint32  ties(512, context_count), Fortran-column order

Output replaces items/ties with finalized uint32 contexts(512, context_count)
and uses magic VGHDCO01. Its final header int32 is an execution-device bitmask
(CPU=1, GPU=2, NPU=4); group order is preserved.

``--serve SESSION_DIR`` turns the worker into a resident, file-signalled
service.  It imports OpenVINO, creates one Core, and retains a dynamic-batch
compiled graph cache for the life of a ``vegai_pretrain train`` process.
Fortran publishes a closed request file followed by ``request.ready``; the
worker responds with an atomically replaced response file followed by
``response.ready``.  This removes the former Python/OpenVINO startup and
per-line graph-compile costs without making the portable Fortran binary depend
on OpenVINO.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
from typing import Any, BinaryIO

from openvino_provider import (
    HDC_BITACC_PLANES,
    HDC_LANES,
    HDC_WORDS,
    build_u32_bundle_model,
    device_kind,
    import_openvino,
    select_device,
)


INPUT_MAGIC = b"VGHDCI01"
OUTPUT_MAGIC = b"VGHDCO01"
PROTOCOL_VERSION = 1
HEADER_FORMAT = "<4i"
GROUP_HEADER_FORMAT = "<2i"
MAX_GROUPS = 1024
MAX_ITEM_COUNT = (1 << HDC_BITACC_PLANES) - 1

# The server-ready header is deliberately binary so the Fortran host can verify
# that the optional process really initialized without acquiring a JSON parser.
SESSION_READY_MAGIC = b"VGHDCW01"
SESSION_READY_VERSION = 1
SESSION_READY_FORMAT = "<4i"
SESSION_REQUEST_NAME = "request.bin"
SESSION_REQUEST_READY_NAME = "request.ready"
SESSION_REQUEST_PROCESSING_NAME = "request.processing"
SESSION_RESPONSE_NAME = "response.bin"
SESSION_RESPONSE_READY_NAME = "response.ready"
SESSION_ERROR_NAME = "response.error"
SESSION_SHUTDOWN_NAME = "shutdown.ready"
SESSION_STOPPED_NAME = "stopped.ready"
SESSION_STOPPED_ACK_NAME = "stopped.ack"
SESSION_REPORT_NAME = "worker-report.json"
POLL_SECONDS = 0.001
DEFAULT_IDLE_SECONDS = 300.0


class WorkerError(RuntimeError):
    """A recoverable worker/protocol failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkerError(message)


def read_exact(stream: BinaryIO, byte_count: int, label: str) -> bytes:
    data = stream.read(byte_count)
    require(len(data) == byte_count, f"truncated {label}: expected {byte_count} bytes, got {len(data)}")
    return data


def read_ints(stream: BinaryIO, count: int, label: str) -> tuple[int, ...]:
    return struct.unpack("<" + "i" * count, read_exact(stream, count * 4, label))


def write_ints(stream: BinaryIO, *values: int) -> None:
    stream.write(struct.pack("<" + "i" * len(values), *values))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish a complete small control file with a same-directory replace."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise WorkerError(f"cannot write {path}: {error}") from error


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def touch_marker(path: Path) -> None:
    atomic_write_bytes(path, b"1\n")


def read_request(path: Path, numpy: Any) -> list[tuple[int, Any, Any]]:
    """Read groups as C-order [contexts, items, lanes] / [contexts, lanes]."""
    try:
        with path.open("rb") as stream:
            require(read_exact(stream, len(INPUT_MAGIC), "input magic") == INPUT_MAGIC, "invalid input magic")
            version, group_count, lane_count, reserved = read_ints(stream, 4, "input header")
            require(version == PROTOCOL_VERSION, f"unsupported input protocol version {version}")
            require(0 <= group_count <= MAX_GROUPS, f"invalid group count {group_count}")
            require(lane_count == HDC_LANES, f"expected {HDC_LANES} lanes, got {lane_count}")
            require(reserved == 0, "input reserved header field must be zero")
            groups: list[tuple[int, Any, Any]] = []
            for group_index in range(group_count):
                item_count, context_count = read_ints(stream, 2, f"group {group_index} header")
                require(1 <= item_count <= MAX_ITEM_COUNT, f"group {group_index}: invalid item count {item_count}")
                require(context_count > 0, f"group {group_index}: invalid context count {context_count}")
                item_values = context_count * item_count * HDC_LANES
                raw_items = numpy.frombuffer(
                    read_exact(stream, item_values * 4, f"group {group_index} items"), dtype="<u4"
                )
                # Fortran writes (lane, item, context): lane is contiguous.
                fortran_items = raw_items.reshape((HDC_LANES, item_count, context_count), order="F")
                items = numpy.transpose(fortran_items, (2, 1, 0)).copy()
                tie_values = context_count * HDC_LANES
                raw_ties = numpy.frombuffer(
                    read_exact(stream, tie_values * 4, f"group {group_index} tie breakers"), dtype="<u4"
                )
                fortran_ties = raw_ties.reshape((HDC_LANES, context_count), order="F")
                ties = numpy.transpose(fortran_ties, (1, 0)).copy()
                groups.append((item_count, items, ties))
            require(stream.read(1) == b"", "input contains trailing bytes")
            return groups
    except OSError as error:
        raise WorkerError(f"cannot read input {path}: {error}") from error


class OpenVINOContextExecutor:
    """One Core plus a bounded, dynamic-batch compiled-model cache.

    A model's semantic shape is determined by its number of bundle items, not
    by the number of contexts on a corpus line.  Compiling with a dynamic batch
    dimension therefore makes a 19-item model reusable for every long record.
    The cache is deliberately resident only for one train process: it contains
    no model state and cannot influence checkpoint contents.
    """

    def __init__(self, openvino: Any, numpy: Any, selected_device: str, require_accelerator: bool) -> None:
        self.openvino = openvino
        self.numpy = numpy
        self.selected_device = selected_device
        self.require_accelerator = require_accelerator
        self.core = openvino.Core()
        self.compiled_by_item_count: dict[int, Any] = {}
        self.execution_devices_by_item_count: dict[int, list[str]] = {}
        self.compile_cache_hits = 0
        self.compile_cache_misses = 0
        self.requests_completed = 0
        self.contexts_completed = 0

    def compiled_model(self, item_count: int) -> tuple[Any, list[str]]:
        compiled = self.compiled_by_item_count.get(item_count)
        if compiled is not None:
            self.compile_cache_hits += 1
            return compiled, self.execution_devices_by_item_count[item_count]
        model = build_u32_bundle_model(self.openvino, self.numpy, item_count, batch_size=None)
        try:
            compiled = self.core.compile_model(model, self.selected_device)
            devices = [str(device) for device in compiled.get_property("EXECUTION_DEVICES")]
        except Exception as error:
            raise WorkerError(f"OpenVINO {item_count}-item graph compile failed: {error}") from error
        if self.require_accelerator:
            require(
                any(device_kind(device) in {"GPU", "NPU"} for device in devices),
                f"device policy fell back to CPU ({', '.join(devices) or 'none'})",
            )
        self.compiled_by_item_count[item_count] = compiled
        self.execution_devices_by_item_count[item_count] = devices
        self.compile_cache_misses += 1
        return compiled, devices

    def execute_groups(self, groups: list[tuple[int, Any, Any]]) -> tuple[list[tuple[int, Any]], list[str]]:
        """Run every group while preserving wire order and exact uint32 lanes."""
        results: list[tuple[int, Any]] = []
        execution_devices: set[str] = set()
        for group_index, (item_count, items, ties) in enumerate(groups):
            context_count = int(items.shape[0])
            compiled, devices = self.compiled_model(item_count)
            try:
                output = next(iter(compiled([items, ties.reshape(context_count, 1, HDC_LANES)]).values()))
            except Exception as error:
                raise WorkerError(f"OpenVINO group {group_index} ({item_count} items) inference failed: {error}") from error
            require(output.shape == (context_count, 1, HDC_LANES), f"group {group_index}: unexpected output shape {output.shape}")
            require(str(output.dtype) == "uint32", f"group {group_index}: unexpected output dtype {output.dtype}")
            execution_devices.update(devices)
            results.append((item_count, output[:, 0, :].copy()))
            self.contexts_completed += context_count
        self.requests_completed += 1
        return results, sorted(execution_devices)

    def report(self) -> dict[str, Any]:
        return {
            "compiled_item_counts": sorted(self.compiled_by_item_count),
            "compile_cache_hits": self.compile_cache_hits,
            "compile_cache_misses": self.compile_cache_misses,
            "requests_completed": self.requests_completed,
            "contexts_completed": self.contexts_completed,
        }


def execute_groups(
    openvino: Any,
    numpy: Any,
    groups: list[tuple[int, Any, Any]],
    selected_device: str,
    require_accelerator: bool,
) -> tuple[list[tuple[int, Any]], list[str]]:
    """Compatibility one-shot executor used by direct worker invocations."""
    return OpenVINOContextExecutor(openvino, numpy, selected_device, require_accelerator).execute_groups(groups)


def execution_device_mask(devices: list[str]) -> int:
    """Encode observed execution plugins for Fortran's truthful runtime report."""
    mask = 0
    for device in devices:
        kind = device_kind(device)
        if kind == "CPU":
            mask |= 1
        elif kind == "GPU":
            mask |= 2
        elif kind == "NPU":
            mask |= 4
    return mask


def selected_device_mask(selected_device: str) -> int:
    """Return a definite startup mask only for a single concrete plugin."""
    if ":" in selected_device or "," in selected_device:
        return 0
    return execution_device_mask([selected_device])


def write_response(path: Path, groups: list[tuple[int, Any]], execution_devices: list[str], numpy: Any) -> None:
    """Write output contexts using the Fortran (lane, context) layout."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(OUTPUT_MAGIC)
            write_ints(stream, PROTOCOL_VERSION, len(groups), HDC_LANES, execution_device_mask(execution_devices))
            for item_count, contexts in groups:
                context_count = int(contexts.shape[0])
                require(contexts.shape == (context_count, HDC_LANES), "internal context output shape mismatch")
                write_ints(stream, item_count, context_count)
                fortran_contexts = numpy.transpose(contexts, (1, 0))
                stream.write(fortran_contexts.ravel(order="F").astype("<u4", copy=False).tobytes())
        temporary.replace(path)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise WorkerError(f"cannot write output {path}: {error}") from error


def worker_report(
    selected: str,
    available: list[str],
    execution_devices: list[str],
    groups: list[tuple[int, Any]],
    executor: OpenVINOContextExecutor | None = None,
    persistent_session: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "vegai-openvino-context-worker/v2",
        "selected_device": selected,
        "available_devices": available,
        "execution_devices": execution_devices,
        "groups": [
            {"item_count": item_count, "context_count": int(contexts.shape[0])}
            for item_count, contexts in groups
        ],
        "uint32_lane_count": HDC_LANES,
        "native_hdc_word_count": HDC_WORDS,
        "persistent_session": persistent_session,
    }
    if executor is not None:
        report["cache"] = executor.report()
    return report


def write_session_ready(path: Path, selected: str) -> None:
    payload = SESSION_READY_MAGIC + struct.pack(
        SESSION_READY_FORMAT,
        SESSION_READY_VERSION,
        selected_device_mask(selected),
        0,
        0,
    )
    atomic_write_bytes(path, payload)


def claim_request(ready_path: Path, processing_path: Path) -> bool:
    """Atomically claim a Fortran-published request marker if one is present."""
    try:
        os.replace(ready_path, processing_path)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise WorkerError(f"cannot claim session request: {error}") from error
    return True


def read_session_error(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "worker request failed without a readable diagnostic"


def serve_session(
    session_directory: Path,
    openvino: Any,
    numpy: Any,
    selected: str,
    available: list[str],
    require_accelerator: bool,
    report_path: Path | None,
    idle_seconds: float,
) -> int:
    """Run one persistent cache-owning service for a single Fortran train call."""
    try:
        session_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise WorkerError(f"session directory already exists: {session_directory}") from error
    except OSError as error:
        raise WorkerError(f"cannot create session directory {session_directory}: {error}") from error

    ready_path = session_directory / "ready.bin"
    failure_path = session_directory / "startup.error"
    request_path = session_directory / SESSION_REQUEST_NAME
    request_ready_path = session_directory / SESSION_REQUEST_READY_NAME
    request_processing_path = session_directory / SESSION_REQUEST_PROCESSING_NAME
    response_path = session_directory / SESSION_RESPONSE_NAME
    response_ready_path = session_directory / SESSION_RESPONSE_READY_NAME
    response_error_path = session_directory / SESSION_ERROR_NAME
    shutdown_path = session_directory / SESSION_SHUTDOWN_NAME
    stopped_path = session_directory / SESSION_STOPPED_NAME
    stopped_ack_path = session_directory / SESSION_STOPPED_ACK_NAME

    try:
        executor = OpenVINOContextExecutor(openvino, numpy, selected, require_accelerator)
        # Fail the server before declaring it ready when an explicitly requested
        # accelerator policy has already resolved to a concrete CPU plugin.
        if require_accelerator and selected_device_mask(selected) == 1:
            raise WorkerError("device policy selected CPU while an accelerator was required")
        write_session_ready(ready_path, selected)
    except WorkerError as error:
        atomic_write_text(failure_path, str(error) + "\n")
        return 2

    last_activity = time.monotonic()
    last_devices: list[str] = []
    last_groups: list[tuple[int, Any]] = []
    terminal_reason = "shutdown"
    try:
        while True:
            if shutdown_path.exists():
                terminal_reason = "shutdown"
                break
            if claim_request(request_ready_path, request_processing_path):
                last_activity = time.monotonic()
                try:
                    # A previous response must have been consumed before the
                    # single in-flight protocol allows another request.
                    require(not response_ready_path.exists(), "previous response marker was not consumed")
                    response_error_path.unlink(missing_ok=True)
                    groups = read_request(request_path, numpy)
                    response, devices = executor.execute_groups(groups)
                    write_response(response_path, response, devices, numpy)
                    touch_marker(response_ready_path)
                    last_devices = devices
                    last_groups = response
                except WorkerError as error:
                    atomic_write_text(response_error_path, str(error) + "\n")
                finally:
                    request_processing_path.unlink(missing_ok=True)
                continue
            if time.monotonic() - last_activity >= idle_seconds:
                terminal_reason = "idle timeout"
                atomic_write_text(response_error_path, "worker session expired while idle\n")
                break
            time.sleep(POLL_SECONDS)
    finally:
        report = worker_report(selected, available, last_devices, last_groups, executor, persistent_session=True)
        report["terminal_reason"] = terminal_reason
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Keep the session directory long enough for the Fortran side to observe
        # an acknowledged shutdown; the worker removes it after acknowledgement
        # or a short grace period so normal runs do not leak SSD artifacts.
        try:
            touch_marker(stopped_path)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not stopped_ack_path.exists():
                time.sleep(POLL_SECONDS)
        finally:
            try:
                shutil.rmtree(session_directory)
            except OSError:
                pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="binary provider request from Vegai Fortran trainer")
    parser.add_argument("--output", type=Path, help="binary finalized-context response")
    parser.add_argument("--serve", type=Path, help="persistent file-signalled session directory")
    parser.add_argument("--device", default="auto", help="OpenVINO device policy or device ID (default: auto)")
    parser.add_argument(
        "--require-accelerator",
        action="store_true",
        help="reject a GPU/NPU policy that compiles or executes on CPU fallback",
    )
    parser.add_argument("--report", type=Path, help="optional JSON execution report")
    parser.add_argument(
        "--idle-seconds", type=float, default=DEFAULT_IDLE_SECONDS,
        help="persistent server idle expiry in seconds (default: %(default)s)",
    )
    parser.add_argument("--quiet", action="store_true", help="write no success JSON to stdout (for the Fortran provider)")
    arguments = parser.parse_args()
    try:
        one_shot = arguments.input is not None or arguments.output is not None
        require(not (one_shot and arguments.serve is not None), "choose either --input/--output or --serve")
        require(arguments.serve is not None or (arguments.input is not None and arguments.output is not None),
                "--serve or both --input and --output are required")
        require(arguments.idle_seconds > 0.0, "--idle-seconds must be positive")
        openvino, numpy = import_openvino()
        require(openvino is not None and numpy is not None, "OpenVINO Python package is not installed")
        available = [str(device) for device in openvino.Core().available_devices]
        selected = select_device(available, arguments.device)
        if arguments.serve is not None:
            return serve_session(
                arguments.serve, openvino, numpy, selected, available,
                arguments.require_accelerator, arguments.report, arguments.idle_seconds,
            )

        assert arguments.input is not None and arguments.output is not None
        request = read_request(arguments.input, numpy)
        executor = OpenVINOContextExecutor(openvino, numpy, selected, arguments.require_accelerator)
        response, execution_devices = executor.execute_groups(request)
        write_response(arguments.output, response, execution_devices, numpy)
        report = worker_report(selected, available, execution_devices, response, executor)
        if arguments.report is not None:
            arguments.report.parent.mkdir(parents=True, exist_ok=True)
            arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not arguments.quiet:
            print(json.dumps(report, sort_keys=True))
    except WorkerError as error:
        # A failed server needs a discoverable marker because its parent only
        # has an asynchronous process launch.  Never turn an optional runtime
        # failure into a false successful CPU/GPU/NPU execution claim.
        if arguments.serve is not None:
            try:
                arguments.serve.mkdir(parents=True, exist_ok=True)
                atomic_write_text(arguments.serve / "startup.error", str(error) + "\n")
            except WorkerError:
                pass
        print(f"OpenVINO context worker: FAIL: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
