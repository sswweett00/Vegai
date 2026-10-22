#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute batched immutable Vegai HDC context bundles through OpenVINO.

This is the optional process boundary used by the Fortran trainer's OpenVINO
context provider. It deliberately handles only immutable, pre-positioned HDC
items and returns finalized context vectors; source-ordered prototype/Hebbian
commits remain in Fortran. The binary protocol is little-endian and uses an
explicit uint32-lane ABI so it never relies on a provider's uint64 bitwise
semantics.

Input layout, all little-endian:

    8 bytes  magic VGHDCI01
    4*int32  version=1, group_count, lane_count=512, reserved=0
    repeated group_count times:
      2*int32 item_count, context_count
      uint32  items(512, item_count, context_count), Fortran-column order
      uint32  ties(512, context_count), Fortran-column order

Output replaces items/ties with finalized uint32 contexts(512, context_count)
and uses magic VGHDCO01. Its final header int32 is an execution-device bitmask
(CPU=1, GPU=2, NPU=4); group order is preserved. The worker fails closed on
malformed files or a requested accelerator that falls back to CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
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


def execute_groups(
    openvino: Any,
    numpy: Any,
    groups: list[tuple[int, Any, Any]],
    selected_device: str,
    require_accelerator: bool,
) -> tuple[list[tuple[int, Any]], list[str]]:
    """Compile each shape once, run it, and preserve group ordering."""
    core = openvino.Core()
    results: list[tuple[int, Any]] = []
    execution_devices: set[str] = set()
    for group_index, (item_count, items, ties) in enumerate(groups):
        context_count = int(items.shape[0])
        model = build_u32_bundle_model(openvino, numpy, item_count, context_count)
        try:
            compiled = core.compile_model(model, selected_device)
            devices = [str(device) for device in compiled.get_property("EXECUTION_DEVICES")]
            output = next(iter(compiled([items, ties.reshape(context_count, 1, HDC_LANES)]).values()))
        except Exception as error:
            raise WorkerError(f"OpenVINO group {group_index} ({item_count} items) failed: {error}") from error
        require(output.shape == (context_count, 1, HDC_LANES), f"group {group_index}: unexpected output shape {output.shape}")
        require(str(output.dtype) == "uint32", f"group {group_index}: unexpected output dtype {output.dtype}")
        if require_accelerator:
            require(
                any(device_kind(device) in {"GPU", "NPU"} for device in devices),
                f"group {group_index}: device policy fell back to CPU ({', '.join(devices) or 'none'})",
            )
        execution_devices.update(devices)
        results.append((item_count, output[:, 0, :].copy()))
    return results, sorted(execution_devices)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="binary provider request from Vegai Fortran trainer")
    parser.add_argument("--output", required=True, type=Path, help="binary finalized-context response")
    parser.add_argument("--device", default="auto", help="OpenVINO device policy or device ID (default: auto)")
    parser.add_argument(
        "--require-accelerator",
        action="store_true",
        help="reject a GPU/NPU policy that compiles or executes on CPU fallback",
    )
    parser.add_argument("--report", type=Path, help="optional JSON execution report")
    parser.add_argument("--quiet", action="store_true", help="write no success JSON to stdout (for the Fortran provider)")
    arguments = parser.parse_args()
    try:
        openvino, numpy = import_openvino()
        require(openvino is not None and numpy is not None, "OpenVINO Python package is not installed")
        available = [str(device) for device in openvino.Core().available_devices]
        selected = select_device(available, arguments.device)
        request = read_request(arguments.input, numpy)
        response, execution_devices = execute_groups(
            openvino, numpy, request, selected, arguments.require_accelerator
        )
        write_response(arguments.output, response, execution_devices, numpy)
        report = {
            "schema": "vegai-openvino-context-worker/v1",
            "selected_device": selected,
            "available_devices": available,
            "execution_devices": execution_devices,
            "groups": [
                {"item_count": item_count, "context_count": int(contexts.shape[0])}
                for item_count, contexts in response
            ],
            "uint32_lane_count": HDC_LANES,
            "native_hdc_word_count": HDC_WORDS,
        }
        if arguments.report is not None:
            arguments.report.parent.mkdir(parents=True, exist_ok=True)
            arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not arguments.quiet:
            print(json.dumps(report, sort_keys=True))
    except WorkerError as error:
        print(f"OpenVINO context worker: FAIL: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
