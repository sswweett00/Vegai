#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preflight a bounded Vegai HDC pretraining run against RAM, CPU, and storage.

This planner never reads the corpus into Python memory and never launches
training.  It estimates the resident Fortran model payload from ``--max-vocab``,
checks free space for the checkpoint path, and emits safe OpenMP placement
recommendations.  It is intentionally advisory rather than a hardware control
plane: an operating system owns RAM channel interleaving, NUMA page placement,
and SSD scheduling.

GPU and NPU adapters are included only as inventory.  The checked-in Vegai
binary currently trains on CPU and does not claim accelerator dispatch merely
because an adapter was detected.  ``--require-accelerator`` therefore fails
closed until a separately validated GPU/NPU provider is integrated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from hardware_compatibility import collect_hardware_profile, compact_hardware_summary


SCHEMA = "vegai-training-resource-plan/v1"
# Exact packed-HDC model planes: two 16,384-bit vectors plus one int64
# experience counter per vocabulary capacity slot.
MODEL_BYTES_PER_SLOT = 2 * 16_384 // 8 + 8
# Current vocabulary arrays on common Fortran ABIs are approximately: 48-byte
# token, int64 frequency, logical active, plus a 2N+1 hash table containing a
# logical, int64 key and int32 slot.  A deliberately larger allowance covers
# logical representation, descriptors and allocator alignment without calling
# it an exact ABI guarantee.
VOCABULARY_ALLOWANCE_BYTES_PER_SLOT = 256
# `train_one_epoch` allocates a bounded 256 x 1024 int64 context staging area
# for deterministic OpenMP construction.  It does not grow with corpus bytes.
CONTEXT_STAGING_BYTES = 256 * 1024 * 8
CHECKPOINT_BYTES_PER_SLOT = 48 + 8 + 8 + 2 * 16_384 // 8 + 8
CHECKPOINT_FIXED_BYTES = 64 * 1024


class ResourcePlanError(RuntimeError):
    """Raised for invalid operator input, not for ordinary unknown telemetry."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResourcePlanError(message)


def bytes_to_mib(value: int | None) -> float | None:
    return None if value is None else round(value / (1024 * 1024), 3)


def estimate_memory(max_vocabulary: int, reserve_fraction: float) -> dict[str, int]:
    model_payload = max_vocabulary * MODEL_BYTES_PER_SLOT
    vocabulary_allowance = max_vocabulary * VOCABULARY_ALLOWANCE_BYTES_PER_SLOT
    resident_before_reserve = model_payload + vocabulary_allowance + CONTEXT_STAGING_BYTES
    reserve = int(resident_before_reserve * reserve_fraction)
    return {
        "model_payload_bytes": model_payload,
        "vocabulary_allowance_bytes": vocabulary_allowance,
        "context_staging_bytes": CONTEXT_STAGING_BYTES,
        "reserve_bytes": reserve,
        "required_available_bytes": resident_before_reserve + reserve,
    }


def estimate_checkpoint(max_vocabulary: int) -> int:
    return CHECKPOINT_FIXED_BYTES + max_vocabulary * CHECKPOINT_BYTES_PER_SLOT


def existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.exists() and candidate.is_dir():
        return candidate
    if candidate.exists():
        return candidate.parent
    for parent in (candidate, *candidate.parents):
        if parent.exists():
            return parent if parent.is_dir() else parent.parent
    return Path.cwd()


def storage_probe(path: Path) -> dict[str, Any]:
    location = existing_parent(path)
    try:
        stat = os.statvfs(location)
        free_bytes = stat.f_bavail * stat.f_frsize
        total_bytes = stat.f_blocks * stat.f_frsize
    except (AttributeError, OSError):
        free_bytes = None
        total_bytes = None
    return {
        "path": str(path),
        "probe_directory": str(location),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "filesystem_type": "not-inferred-portably",
        "ssd_note": (
            "Vegai uses ordinary streaming file I/O and works on SSD, HDD, network, or removable filesystems. "
            "This portable preflight reports free space; it does not infer medium type or promise SSD throughput."
        ),
    }


def recommended_threads(profile: dict[str, Any], requested: int | None) -> int:
    topology = profile["cpu"].get("topology", {})
    physical = topology.get("physical_cores")
    logical = profile["cpu"].get("logical_processors")
    available = physical if isinstance(physical, int) and physical > 0 else logical
    if not isinstance(available, int) or available < 1:
        available = 1
    if requested is None:
        return available
    return min(requested, available)


def build_plan(arguments: argparse.Namespace) -> dict[str, Any]:
    corpus = arguments.corpus.resolve()
    require(corpus.is_file(), f"corpus does not exist or is not a regular file: {corpus}")
    require(arguments.max_vocab > 0, "--max-vocab must be positive")
    require(0.0 <= arguments.reserve_fraction <= 4.0, "--reserve-fraction must be between 0 and 4")
    require(arguments.omp_threads is None or arguments.omp_threads > 0, "--omp-threads must be positive")
    checkpoint = arguments.checkpoint.resolve() if arguments.checkpoint else corpus.with_suffix(".vgai")

    profile = collect_hardware_profile()
    memory_estimate = estimate_memory(arguments.max_vocab, arguments.reserve_fraction)
    checkpoint_estimate = estimate_checkpoint(arguments.max_vocab)
    storage = storage_probe(checkpoint)
    available_memory = profile["memory"].get("available_bytes")
    free_storage = storage["free_bytes"]
    thread_count = recommended_threads(profile, arguments.omp_threads)

    warnings: list[str] = []
    errors: list[str] = []
    if not isinstance(available_memory, int):
        warnings.append("available RAM could not be queried; compare the conservative estimate manually before training")
    elif available_memory < memory_estimate["required_available_bytes"]:
        errors.append(
            "available RAM is below the conservative model/staging reservation "
            f"({available_memory} < {memory_estimate['required_available_bytes']} bytes)"
        )
    if not isinstance(free_storage, int):
        warnings.append("checkpoint filesystem free space could not be queried")
    elif free_storage < checkpoint_estimate:
        errors.append(
            "checkpoint filesystem free space is below the estimated checkpoint size "
            f"({free_storage} < {checkpoint_estimate} bytes)"
        )
    if profile["memory"].get("memory_channel_configuration") not in {"dual-channel", "quad-channel"}:
        warnings.append(
            "RAM channel topology is not a portable software control. Use matched DIMMs and firmware documentation "
            "to confirm dual-/multi-channel interleaving; Vegai uses normal OS memory allocation."
        )
    if profile["cpu"].get("topology", {}).get("numa_nodes") not in {None, 1}:
        warnings.append(
            "multiple NUMA nodes were detected; benchmark OMP_PROC_BIND/OMP_PLACES on the target host and avoid "
            "assuming one placement policy is universally fastest"
        )
    if profile["gpu_adapters"] or profile["npu_adapters"]:
        warnings.append(
            "GPU/NPU adapters were detected, but the current checked-in training backend remains CPU-only; "
            "inventory is not accelerator dispatch"
        )
    if arguments.require_accelerator:
        errors.append(
            "--require-accelerator cannot be satisfied: no validated GPU/NPU training provider is part of this binary"
        )

    ready = not errors
    return {
        "schema": SCHEMA,
        "corpus": {
            "path": str(corpus),
            "bytes": corpus.stat().st_size,
            "streaming_contract": "corpus is streamed line by line; corpus byte size is not added to the resident model estimate",
        },
        "checkpoint": {
            "path": str(checkpoint),
            "estimated_max_bytes": checkpoint_estimate,
            "storage": storage,
        },
        "memory": {
            "max_vocabulary": arguments.max_vocab,
            "reserve_fraction": arguments.reserve_fraction,
            **memory_estimate,
            "host_available_bytes": available_memory,
            "host_total_bytes": profile["memory"].get("total_bytes"),
            "estimate_contract": (
                "conservative planning estimate, not an allocation guarantee; Fortran logical size, allocator overhead, "
                "OpenMP runtime, page cache, other processes, and checkpoint tooling can change RSS"
            ),
        },
        "cpu_execution": {
            "recommended_omp_threads": thread_count,
            "environment": {
                "OMP_NUM_THREADS": str(thread_count),
                "OMP_PROC_BIND": "close",
                "OMP_PLACES": "cores",
            },
            "contract": (
                "OpenMP parallelizes immutable per-record context construction; source-ordered prototype commits preserve "
                "checkpoint bytes. Small records or I/O-bound corpora may not speed up, so measure on the target host."
            ),
        },
        "accelerators": {
            "gpu_adapters_detected": len(profile["gpu_adapters"]),
            "npu_adapters_detected": len(profile["npu_adapters"]),
            "cpu_compute_used": True,
            "gpu_compute_used": False,
            "npu_compute_used": False,
            "contract": profile["execution_backend"]["accelerator_dispatch_contract"],
        },
        "hardware": profile,
        "verdict": {
            "ready": ready,
            "errors": errors,
            "warnings": warnings,
        },
    }


def render_text(plan: dict[str, Any]) -> str:
    memory = plan["memory"]
    checkpoint = plan["checkpoint"]
    storage = checkpoint["storage"]
    cpu = plan["cpu_execution"]
    verdict = plan["verdict"]
    lines = [
        "Vegai training resource plan",
        "Hardware: " + compact_hardware_summary(plan["hardware"]),
        f"Corpus: {plan['corpus']['path']} ({plan['corpus']['bytes']} bytes, streaming)",
        f"Max vocabulary: {memory['max_vocabulary']}",
        f"Estimated model payload: {memory['model_payload_bytes']} bytes ({bytes_to_mib(memory['model_payload_bytes'])} MiB)",
        f"Estimated vocabulary allowance: {memory['vocabulary_allowance_bytes']} bytes",
        f"OpenMP context staging: {memory['context_staging_bytes']} bytes",
        f"Conservative required available RAM: {memory['required_available_bytes']} bytes "
        f"({bytes_to_mib(memory['required_available_bytes'])} MiB)",
        f"Host-reported available RAM: {memory['host_available_bytes']} bytes",
        f"Checkpoint target: {checkpoint['path']}",
        f"Estimated maximum checkpoint: {checkpoint['estimated_max_bytes']} bytes ({bytes_to_mib(checkpoint['estimated_max_bytes'])} MiB)",
        f"Checkpoint filesystem free bytes: {storage['free_bytes']}",
        f"Recommended OpenMP: OMP_NUM_THREADS={cpu['environment']['OMP_NUM_THREADS']} "
        f"OMP_PROC_BIND={cpu['environment']['OMP_PROC_BIND']} OMP_PLACES={cpu['environment']['OMP_PLACES']}",
        "GPU/NPU execution: CPU active; GPU/NPU dispatch is not active in this binary.",
        f"Verdict: {'READY' if verdict['ready'] else 'NOT READY'}",
    ]
    for message in verdict["errors"]:
        lines.append("ERROR: " + message)
    for message in verdict["warnings"]:
        lines.append("WARNING: " + message)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path, help="input corpus to be streamed")
    parser.add_argument("--checkpoint", type=Path, help="planned output checkpoint path (default: corpus with .vgai suffix)")
    parser.add_argument("--max-vocab", type=int, default=20_000, help="same capacity passed to vegai_pretrain (default: 20000)")
    parser.add_argument("--omp-threads", type=int, help="cap recommendation to this many CPU threads")
    parser.add_argument(
        "--reserve-fraction",
        type=float,
        default=0.25,
        help="additional conservative RAM reservation as fraction of model/staging estimate (default: 0.25)",
    )
    parser.add_argument("--require-ready", action="store_true", help="exit nonzero when RAM/storage checks fail")
    parser.add_argument(
        "--require-accelerator",
        action="store_true",
        help="fail closed because this CPU binary has no validated GPU/NPU training provider",
    )
    parser.add_argument("--format", choices=("json", "text"), default="text", help="rendering format")
    parser.add_argument("--output", type=Path, help="optional report path; stdout remains the report")
    arguments = parser.parse_args()
    try:
        plan = build_plan(arguments)
    except (OSError, ResourcePlanError, ValueError) as error:
        print(f"training resource plan: FAIL: {error}", file=sys.stderr)
        return 2

    rendered = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) if arguments.format == "json" else render_text(plan)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    if (arguments.require_ready or arguments.require_accelerator) and not plan["verdict"]["ready"]:
        print("training resource plan: NOT READY", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
