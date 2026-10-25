#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe an optional Intel OpenVINO device provider for Vegai HDC primitives.

OpenVINO is an optional provider, not a dependency of the portable Fortran
binary. This tool inventories CPU/GPU/NPU plugins and runs two real device
graphs:

* a deliberately adversarial direct ``uint64`` XOR diagnostic; and
* exact 32-bit-lane HDC bundle/finalize graphs for the default full-window
  hybrid (16 items) and hierarchical (19 items) context shapes.

The lane graph is a prospective tensor ABI: every Fortran int64 bit pattern is
split explicitly into low/high uint32 lanes. Its HDC operations are only
per-bit boolean operations, so they preserve the packed 16,384-dimensional
semantics without relying on a provider's uint64 implementation.

This is a readiness gate, not a cosmetic adapter detector. The checked-in
Fortran provider uses this uint32-lane ABI for batched immutable context
construction while retaining authoritative source-ordered CPU commits. The
probe itself is not a train run: it cannot claim GPU/NPU dispatch until
``vegai_pretrain`` reports a worker execution mask and checkpoint equality is
verified.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any


SCHEMA = "vegai-openvino-provider-probe/v2"
PREFERRED_DEVICE_ORDER = ("GPU", "NPU", "CPU")
HDC_WORDS = 256
HDC_LANES = HDC_WORDS * 2
HDC_BITACC_PLANES = 11
# context_window=4 has 8 neighbors. Hybrid contributes a position and topic
# item for each neighbor (16); hierarchical can add three record-summary items.
HDC_BUNDLE_CASES = (16, 19)


class ProviderError(RuntimeError):
    """Expected optional-provider configuration error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProviderError(message)


def import_openvino() -> tuple[Any, Any] | tuple[None, None]:
    """Import optional OpenVINO and NumPy without contaminating CPU-only use."""
    try:
        return importlib.import_module("openvino"), importlib.import_module("numpy")
    except ImportError:
        return None, None


def device_info(core: Any, device: str) -> dict[str, Any]:
    information: dict[str, Any] = {"device": device}
    for property_name, output_name in (
        ("FULL_DEVICE_NAME", "full_name"),
        ("OPTIMIZATION_CAPABILITIES", "optimization_capabilities"),
    ):
        try:
            value = core.get_property(device, property_name)
        except Exception:
            continue
        if isinstance(value, (list, tuple)):
            information[output_name] = [str(item) for item in value]
        else:
            information[output_name] = str(value)
    return information


def device_kind(device: str) -> str:
    """Normalize OpenVINO device IDs such as GPU.0 to their plugin kind."""
    return device.split(".", maxsplit=1)[0].upper()


def devices_of_kind(available: list[str], kind: str) -> list[str]:
    return [device for device in available if device_kind(device) == kind]


def select_device(available: list[str], requested: str) -> str:
    if requested != "auto":
        return requested
    # Some runtimes expose GPU.0 / NPU.0 rather than generic plugin IDs. Keep
    # the actual Core-reported identifier so selection and later truth reporting
    # refer to the same physical plugin.
    preferred = [
        device
        for kind in PREFERRED_DEVICE_ORDER
        for device in devices_of_kind(available, kind)
    ]
    # AUTO is a preference policy, not evidence that all devices execute the
    # same graph simultaneously. Avoid an AUTO string on CPU-only installs,
    # where some OpenVINO distributions do not bundle unavailable plugins.
    if len(preferred) == 1:
        return preferred[0]
    if preferred:
        return "AUTO:" + ",".join(preferred)
    return "CPU"


def failed_probe(error: str) -> dict[str, Any]:
    return {
        "attempted": True,
        "compiled": False,
        "execution_devices": [],
        "output_bit_identical": False,
        "word_dtype": None,
        "error": error,
    }


def run_direct_uint64_xor_diagnostic(openvino: Any, numpy: Any, selected_device: str) -> dict[str, Any]:
    """Detect whether a provider preserves all 64 packed bits in direct XOR.

    This diagnostic intentionally contains mixed high/low 32-bit patterns.
    All-zero and all-one-only vectors can hide providers that silently lower
    uint64 bitwise operations to 32 bits.
    """
    left = openvino.opset13.parameter([2, 8], openvino.Type.u64, name="left_words")
    right = openvino.opset13.parameter([2, 8], openvino.Type.u64, name="right_words")
    bound = openvino.opset13.bitwise_xor(left, right)
    model = openvino.Model([openvino.opset13.result(bound)], [left, right], "vegai_direct_uint64_xor_diagnostic")
    try:
        compiled = openvino.Core().compile_model(model, selected_device)
        execution_devices = [str(item) for item in compiled.get_property("EXECUTION_DEVICES")]
        left_values = numpy.array(
            [
                [0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0x1234567812345678, 0x8000000000000001,
                 0x0000000100000000, 0xDEADBEEFCAFEBABE, 0xAAAAAAAA55555555, 0x0123456789ABCDEF],
                [0x7FFFFFFF80000000, 0x1111111122222222, 0xFEDCBA9876543210, 0x00000000FFFFFFFF,
                 0xFFFFFFFF00000000, 0x13579BDF2468ACE0, 0x0102030405060708, 0x8877665544332211],
            ],
            dtype=numpy.uint64,
        )
        right_values = numpy.array(
            [
                [0xFFFFFFFFFFFFFFFF, 0x0000000000000000, 0xFEDCBA98FEDCBA98, 0x7FFFFFFFFFFFFFFF,
                 0xFFFFFFFF00000000, 0x0123456789ABCDEF, 0x55555555AAAAAAAA, 0xF0E1D2C3B4A59687],
                [0x800000007FFFFFFF, 0x3333333344444444, 0x0123456789ABCDEF, 0xFFFFFFFF00000000,
                 0x00000000FFFFFFFF, 0xFEDCBA9876543210, 0x0807060504030201, 0x1122334455667788],
            ],
            dtype=numpy.uint64,
        )
        expected = numpy.bitwise_xor(left_values, right_values)
        output = next(iter(compiled([left_values, right_values]).values()))
        return {
            "attempted": True,
            "compiled": True,
            "execution_devices": execution_devices,
            "output_bit_identical": bool(numpy.array_equal(output, expected)),
            "word_dtype": str(output.dtype),
            "error": None,
        }
    except Exception as error:
        return failed_probe(str(error))


def build_u32_bundle_model(openvino: Any, numpy: Any, item_count: int, batch_size: int | None = 1) -> Any:
    """Build exact bit-sliced HDC bundle/finalize for pre-permuted lane vectors.

    The graph receives ``[batch, item_count, 512]`` uint32 items and
    ``[batch, 1, 512]`` seeded tie breakers.  ``batch_size=None`` creates a
    dynamic-batch graph.  That form is important for the resident worker: a
    compiled graph is then reusable for every record having the same item
    cardinality, rather than being recompiled for every record length.

    It is algebraically the same carry-plane counter and majority finalization
    as ``hv_bitacc_*`` in ``src/hyper_types.f90``. The input items are already
    positioned; this makes the graph's ABI explicit and lets a future batched
    provider choose whether rotations happen on host or device.
    """
    require(1 <= item_count <= (1 << HDC_BITACC_PLANES) - 1, f"invalid HDC item count {item_count}")
    if batch_size is not None:
        require(batch_size > 0, f"invalid HDC batch size {batch_size}")
    ops = openvino.opset13
    batch_dimension = -1 if batch_size is None else batch_size
    words = ops.parameter([batch_dimension, item_count, HDC_LANES], openvino.Type.u32, name="prepositioned_u32_lanes")
    tie_breaker = ops.parameter([batch_dimension, 1, HDC_LANES], openvino.Type.u32, name="tie_breaker_u32_lanes")
    item_axis = ops.constant(numpy.array(1, dtype=numpy.int64))
    if batch_size is None:
        # A dynamic-shape Constant cannot express the batch axis.  Deriving a
        # zero tensor from the first immutable item retains exact uint32 bit
        # semantics and has the desired [batch, 1, lanes] shape.
        first_item = ops.gather(words, ops.constant(numpy.array([0], dtype=numpy.int64)), item_axis)
        zero = ops.bitwise_xor(first_item, first_item)
    else:
        zero = ops.constant(numpy.zeros((batch_size, 1, HDC_LANES), dtype=numpy.uint32))
    planes = [zero for _ in range(HDC_BITACC_PLANES)]

    for item_index in range(item_count):
        index = ops.constant(numpy.array([item_index], dtype=numpy.int64))
        carry = ops.gather(words, index, item_axis)
        for plane_index in range(HDC_BITACC_PLANES):
            previous = planes[plane_index]
            planes[plane_index] = ops.bitwise_xor(previous, carry)
            carry = ops.bitwise_and(previous, carry)

    # Build all-ones by NOT(zero), rather than embedding 0xffffffff as a
    # constant. This also catches provider constant-folding/unsigned-width bugs
    # in the exact graph we validate.
    equal_mask = ops.bitwise_not(zero)
    greater_mask = zero
    threshold = item_count // 2
    for plane_index in range(HDC_BITACC_PLANES - 1, -1, -1):
        if threshold & (1 << plane_index):
            equal_mask = ops.bitwise_and(equal_mask, planes[plane_index])
        else:
            greater_mask = ops.bitwise_or(greater_mask, ops.bitwise_and(equal_mask, planes[plane_index]))
            equal_mask = ops.bitwise_and(equal_mask, ops.bitwise_not(planes[plane_index]))
    if item_count % 2 == 0:
        result = ops.bitwise_or(greater_mask, ops.bitwise_and(equal_mask, tie_breaker))
    else:
        result = greater_mask
    return openvino.Model([ops.result(result)], [words, tie_breaker], f"vegai_hdc_bundle_{item_count}_batch_{batch_size}")


def hdc_bundle_reference(numpy: Any, items: Any, tie_breaker: Any) -> Any:
    """Pure NumPy oracle for the uint32-lane graph (same bit-sliced algorithm)."""
    planes = numpy.zeros((HDC_BITACC_PLANES, HDC_LANES), dtype=numpy.uint32)
    for item in items:
        carry = item
        for plane_index in range(HDC_BITACC_PLANES):
            previous = planes[plane_index].copy()
            planes[plane_index] = numpy.bitwise_xor(previous, carry)
            carry = numpy.bitwise_and(previous, carry)

    equal_mask = numpy.bitwise_not(numpy.zeros(HDC_LANES, dtype=numpy.uint32))
    greater_mask = numpy.zeros(HDC_LANES, dtype=numpy.uint32)
    threshold = len(items) // 2
    for plane_index in range(HDC_BITACC_PLANES - 1, -1, -1):
        if threshold & (1 << plane_index):
            equal_mask = numpy.bitwise_and(equal_mask, planes[plane_index])
        else:
            greater_mask = numpy.bitwise_or(greater_mask, numpy.bitwise_and(equal_mask, planes[plane_index]))
            equal_mask = numpy.bitwise_and(equal_mask, numpy.bitwise_not(planes[plane_index]))
    if len(items) % 2 == 0:
        return numpy.bitwise_or(greater_mask, numpy.bitwise_and(equal_mask, tie_breaker))
    return greater_mask


def deterministic_lane_inputs(numpy: Any, item_count: int) -> tuple[Any, Any]:
    """Generate high-bit-rich reproducible lane vectors without a corpus dependency."""
    # Use unsigned arithmetic with a fixed generator instead of a small-value
    # fixture, so sign-bit and carry-plane errors are observable.
    generator = numpy.random.default_rng(0x5E6A1)
    items = generator.integers(0, 2**32, size=(item_count, HDC_LANES), dtype=numpy.uint32)
    tie_breaker = generator.integers(0, 2**32, size=(HDC_LANES,), dtype=numpy.uint32)
    return items, tie_breaker


def verify_lane_codec(numpy: Any) -> bool:
    """Verify explicit int64-bit-pattern <-> low/high uint32 lane ordering."""
    words = numpy.array(
        [
            0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0x1234567812345678, 0x8000000000000001,
            0x0000000100000000, 0xDEADBEEFCAFEBABE, 0xAAAAAAAA55555555, 0x0123456789ABCDEF,
        ],
        dtype=numpy.uint64,
    )
    low = numpy.bitwise_and(words, numpy.uint64(0xFFFFFFFF)).astype(numpy.uint32)
    high = numpy.right_shift(words, numpy.uint64(32)).astype(numpy.uint32)
    lanes = numpy.empty(words.size * 2, dtype=numpy.uint32)
    lanes[0::2] = low
    lanes[1::2] = high
    rebuilt = numpy.bitwise_or(
        lanes[0::2].astype(numpy.uint64),
        numpy.left_shift(lanes[1::2].astype(numpy.uint64), numpy.uint64(32)),
    )
    return bool(numpy.array_equal(rebuilt, words))


def run_u32_bundle_case(openvino: Any, numpy: Any, selected_device: str, item_count: int) -> dict[str, Any]:
    try:
        model = build_u32_bundle_model(openvino, numpy, item_count)
        compiled = openvino.Core().compile_model(model, selected_device)
        execution_devices = [str(item) for item in compiled.get_property("EXECUTION_DEVICES")]
        items, tie_breaker = deterministic_lane_inputs(numpy, item_count)
        expected = hdc_bundle_reference(numpy, items, tie_breaker).reshape(1, 1, HDC_LANES)
        output = next(iter(compiled([items.reshape(1, item_count, HDC_LANES), tie_breaker.reshape(1, 1, HDC_LANES)]).values()))
        return {
            "item_count": item_count,
            "attempted": True,
            "compiled": True,
            "execution_devices": execution_devices,
            "output_bit_identical": bool(numpy.array_equal(output, expected)),
            "word_dtype": str(output.dtype),
            "error": None,
        }
    except Exception as error:
        result = failed_probe(str(error))
        result["item_count"] = item_count
        return result


def run_u32_hdc_bundle_probe(openvino: Any, numpy: Any, selected_device: str) -> dict[str, Any]:
    cases = [run_u32_bundle_case(openvino, numpy, selected_device, item_count) for item_count in HDC_BUNDLE_CASES]
    execution_devices = sorted({device for case in cases for device in case["execution_devices"]})
    compiled = all(case["compiled"] for case in cases)
    exact = compiled and all(case["output_bit_identical"] for case in cases)
    errors = [case["error"] for case in cases if case["error"]]
    return {
        "attempted": True,
        "compiled": compiled,
        "output_bit_identical": exact,
        "execution_devices": execution_devices,
        "lane_dtype": "uint32",
        "lane_count": HDC_LANES,
        "native_word_count": HDC_WORDS,
        "bitaccumulator_planes": HDC_BITACC_PLANES,
        "lane_codec_roundtrip_bit_identical": verify_lane_codec(numpy),
        "cases": cases,
        "error": "; ".join(errors) if errors else None,
    }


def unavailable_report() -> dict[str, Any]:
    unavailable = {
        "attempted": False,
        "compiled": False,
        "execution_devices": [],
        "output_bit_identical": False,
        "word_dtype": None,
        "error": "OpenVINO Python package is not installed",
    }
    return {
        "schema": SCHEMA,
        "openvino_installed": False,
        "runtime_version": None,
        "available_devices": [],
        "selected_device": None,
        "devices": [],
        "direct_uint64_xor_diagnostic": unavailable,
        "uint32_lane_hdc_bundle_probe": {
            "attempted": False,
            "compiled": False,
            "output_bit_identical": False,
            "execution_devices": [],
            "lane_dtype": "uint32",
            "lane_count": HDC_LANES,
            "native_word_count": HDC_WORDS,
            "bitaccumulator_planes": HDC_BITACC_PLANES,
            "lane_codec_roundtrip_bit_identical": False,
            "cases": [],
            "error": unavailable["error"],
        },
        "tensor_abi": {
            "status": "unavailable",
            "fortran_word_to_openvino_lanes": "lane[2*w]=low32(word[w]); lane[2*w+1]=high32(word[w])",
            "reason": "OpenVINO is not installed",
        },
        "training_dispatch": {
            "cpu_compute_used": True,
            "gpu_compute_used": False,
            "npu_compute_used": False,
            "openvino_context_provider_implemented": True,
            "openvino_context_provider_available": False,
            "probe_is_not_a_train_run": True,
            "contract": (
                "The Fortran/OpenVINO immutable-context provider is implemented but this optional runtime is unavailable. "
                "The portable CPU trainer remains usable; no GPU/NPU training fallback was silently claimed."
            ),
        },
    }


def build_report(arguments: argparse.Namespace) -> dict[str, Any]:
    openvino, numpy = import_openvino()
    if openvino is None or numpy is None:
        return unavailable_report()

    core = openvino.Core()
    available = [str(device) for device in core.available_devices]
    selected = select_device(available, arguments.device)
    direct_u64 = run_direct_uint64_xor_diagnostic(openvino, numpy, selected)
    lane_bundle = run_u32_hdc_bundle_probe(openvino, numpy, selected)
    return {
        "schema": SCHEMA,
        "openvino_installed": True,
        "runtime_version": str(openvino.get_version()),
        "available_devices": available,
        "selected_device": selected,
        "devices": [device_info(core, device) for device in available],
        "direct_uint64_xor_diagnostic": direct_u64,
        "uint32_lane_hdc_bundle_probe": lane_bundle,
        "tensor_abi": {
            "status": "candidate-validated-by-probe" if lane_bundle["output_bit_identical"] else "not-validated",
            "fortran_word_to_openvino_lanes": "lane[2*w]=low32(word[w]); lane[2*w+1]=high32(word[w])",
            "vector_shape": [HDC_LANES],
            "context_input_shape": ["items", HDC_LANES],
            "rotation_contract": "input items are already position-permuted; no host byte reinterpretation is allowed",
            "checkpoint_contract": "not integrated: full serial-vs-provider checkpoint-byte equality is still required",
        },
        "training_dispatch": {
            "cpu_compute_used": True,
            "gpu_compute_used": False,
            "npu_compute_used": False,
            "openvino_context_provider_implemented": True,
            "openvino_context_provider_available": bool(lane_bundle["output_bit_identical"]),
            "probe_is_not_a_train_run": True,
            "contract": (
                "The checked-in Fortran provider sends immutable pre-positioned contexts to the uint32-lane worker and "
                "keeps source-ordered prototype commits on CPU. This probe is not a train run: GPU/NPU fields remain false "
                "until vegai_pretrain reports actual worker execution and checkpoint equality is verified."
            ),
        },
    }


def render_text(report: dict[str, Any]) -> str:
    lines = ["Vegai optional OpenVINO provider probe", f"OpenVINO installed: {report['openvino_installed']}"]
    if report["openvino_installed"]:
        lines.extend(
            [
                f"Runtime version: {report['runtime_version']}",
                "Available devices: " + (", ".join(report["available_devices"]) or "none"),
                f"Selected device policy: {report['selected_device']}",
            ]
        )
        for device in report["devices"]:
            properties = ", ".join(f"{key}={value}" for key, value in sorted(device.items()))
            lines.append("  - " + properties)

    direct = report["direct_uint64_xor_diagnostic"]
    lane = report["uint32_lane_hdc_bundle_probe"]
    lines.extend(
        [
            f"direct uint64 XOR compiled: {direct['compiled']}",
            f"direct uint64 XOR bit-identical: {direct['output_bit_identical']}",
            f"uint32-lane HDC bundle compiled: {lane['compiled']}",
            f"uint32-lane HDC bundle bit-identical: {lane['output_bit_identical']}",
            f"u32 lane codec bit-identical: {lane['lane_codec_roundtrip_bit_identical']}",
            "bundle execution devices: " + (", ".join(lane["execution_devices"]) or "none"),
        ]
    )
    for case in lane["cases"]:
        lines.append(
            f"  - {case['item_count']} items: compiled={case['compiled']} exact={case['output_bit_identical']}"
        )
    if direct["error"]:
        lines.append("direct uint64 diagnostic detail: " + direct["error"])
    if lane["error"]:
        lines.append("u32 bundle probe detail: " + lane["error"])
    dispatch = report["training_dispatch"]
    lines.extend(
        [
            "probe train-dispatch claim: none (run vegai_pretrain for actual CPU/GPU/NPU flags)",
            "contract: " + dispatch["contract"],
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="auto",
        help="OpenVINO device or policy, e.g. auto, GPU, NPU, CPU, AUTO:GPU,NPU,CPU (default: auto)",
    )
    parser.add_argument("--require-openvino", action="store_true", help="fail if optional OpenVINO Python package is absent")
    parser.add_argument("--require-gpu", action="store_true", help="fail unless OpenVINO exposes a GPU device")
    parser.add_argument("--require-npu", action="store_true", help="fail unless OpenVINO exposes an NPU device")
    parser.add_argument(
        "--require-hdc-probe",
        "--require-xor-probe",
        dest="require_hdc_probe",
        action="store_true",
        help="fail unless exact uint32-lane HDC bundle/finalize cases compile and match their CPU oracle",
    )
    parser.add_argument(
        "--require-hdc-accelerator",
        action="store_true",
        help="fail unless an exact HDC bundle case actually executes on GPU or NPU (not CPU fallback)",
    )
    parser.add_argument(
        "--require-direct-uint64",
        action="store_true",
        help="fail unless the diagnostic direct uint64 XOR preserves every bit; not required by the u32 lane ABI",
    )
    parser.add_argument("--format", choices=("json", "text"), default="text", help="rendering format")
    parser.add_argument("--output", type=Path, help="optional report file; stdout remains the report")
    arguments = parser.parse_args()
    try:
        report = build_report(arguments)
        if arguments.require_openvino:
            require(report["openvino_installed"], "OpenVINO package is required but unavailable")
        if arguments.require_gpu:
            require(bool(devices_of_kind(report["available_devices"], "GPU")), "OpenVINO GPU device is required but unavailable")
        if arguments.require_npu:
            require(bool(devices_of_kind(report["available_devices"], "NPU")), "OpenVINO NPU device is required but unavailable")
        lane_probe = report["uint32_lane_hdc_bundle_probe"]
        if arguments.require_hdc_probe:
            require(
                lane_probe["compiled"] and lane_probe["output_bit_identical"] and lane_probe["lane_codec_roundtrip_bit_identical"],
                "selected provider failed exact uint32-lane HDC bundle/finalize probe",
            )
        if arguments.require_hdc_accelerator:
            accelerator_devices = [
                device for device in lane_probe["execution_devices"] if device_kind(device) in {"GPU", "NPU"}
            ]
            require(
                lane_probe["compiled"] and lane_probe["output_bit_identical"] and bool(accelerator_devices),
                "exact HDC bundle did not execute on GPU or NPU (CPU fallback is not accepted)",
            )
        if arguments.require_direct_uint64:
            direct = report["direct_uint64_xor_diagnostic"]
            require(direct["compiled"] and direct["output_bit_identical"], "selected provider failed exact direct uint64 XOR diagnostic")
    except ProviderError as error:
        print(f"OpenVINO provider probe: FAIL: {error}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) if arguments.format == "json" else render_text(report)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
