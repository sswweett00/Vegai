#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit a privacy-conscious Vegai hardware/OS compatibility profile.

Vegai's current execution backend is portable Fortran CPU code.  It does not
load CUDA, OpenCL, oneAPI, Metal, Vulkan, vendor NPU libraries, or GPU drivers.
Consequently an Intel/AMD/Apple integrated GPU is not a runtime dependency: the
program remains usable on its host CPU whether an iGPU is present, absent, or
has a proprietary/display-only driver.

The report records observable host facts for reproducible benchmark metadata.
It intentionally avoids host names, serial numbers, MAC addresses, and full
raw environment dumps.  GPU entries are inventory information, not a claim that
Vegai dispatches compute work to them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
from typing import Any, Iterable


SCHEMA = "vegai-hardware-compatibility/v2"
# These are the operating systems exercised by the repository's portable
# build/runtime workflow. Other systems retain a best-effort profile rather
# than being silently represented as tested platforms.
RECOGNIZED_OPERATING_SYSTEMS = {"Linux", "Darwin", "Windows"}
DISPLAY_PATTERNS = ("vga compatible controller", "3d controller", "display controller")


def _normalise_architecture(value: str) -> str:
    """Return a stable architecture family without assuming a vendor ISA."""
    key = value.strip().casefold().replace("-", "_")
    aliases = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "x64": "x86_64",
        "i386": "x86_32",
        "i486": "x86_32",
        "i586": "x86_32",
        "i686": "x86_32",
        "x86": "x86_32",
        "aarch64": "aarch64",
        "arm64": "aarch64",
        "armv8l": "aarch64",
        "ppc64le": "ppc64le",
        "ppc64": "ppc64",
        "s390x": "s390x",
        "riscv64": "riscv64",
    }
    return aliases.get(key, "unknown")


def _read_text(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return value or None


def _command_text(command: list[str], timeout_seconds: float = 3.0) -> str | None:
    """Best-effort command helper; absent optional inventory tools are normal."""
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def _first_cpuinfo_value(names: Iterable[str]) -> str | None:
    requested = {name.casefold() for name in names}
    value = _read_text(Path("/proc/cpuinfo"))
    if value is None:
        return None
    for line in value.splitlines():
        if ":" not in line:
            continue
        key, candidate = (part.strip() for part in line.split(":", 1))
        if key.casefold() in requested and candidate:
            return candidate
    return None


def _linux_cpu_model() -> str | None:
    candidate = _first_cpuinfo_value(("model name", "hardware", "cpu"))
    if candidate is not None:
        return candidate
    # ARM commonly reports its model under Processor; x86 instead uses this
    # field as a numeric logical-CPU index, which is not a model name.
    candidate = _first_cpuinfo_value(("processor",))
    if candidate is not None and not candidate.isdecimal():
        return candidate

    rendered = _command_text(["lscpu", "--json"])
    if rendered is not None:
        try:
            payload = json.loads(rendered)
        except json.JSONDecodeError:
            payload = {}
        for entry in payload.get("lscpu", []):
            field = str(entry.get("field", "")).strip().rstrip(":").casefold()
            if field in {"model name", "model", "hardware"}:
                value = str(entry.get("data", "")).strip()
                if value:
                    return value
    return None


def _windows_powershell_json(script: str) -> Any | None:
    for executable in ("powershell", "powershell.exe", "pwsh"):
        rendered = _command_text([executable, "-NoProfile", "-Command", script])
        if rendered is None:
            continue
        try:
            return json.loads(rendered)
        except json.JSONDecodeError:
            continue
    return None


def _windows_cpu_model() -> str | None:
    payload = _windows_powershell_json(
        "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name | ConvertTo-Json -Compress"
    )
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    return None


def _darwin_cpu_model() -> str | None:
    return _command_text(["sysctl", "-n", "machdep.cpu.brand_string"]) or _command_text(
        ["sysctl", "-n", "hw.model"]
    )


def _normalise_adapter(entry: dict[str, Any]) -> dict[str, str]:
    return {key: str(value).strip() for key, value in entry.items() if value is not None and str(value).strip()}


def _linux_gpu_adapters() -> list[dict[str, str]]:
    adapters: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    # sysfs is available even on minimal Linux images where lspci is absent.
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if re.fullmatch(r"card[0-9]+", card.name) is None:
            continue
        device = card / "device"
        if not device.exists():
            continue
        vendor = _read_text(device / "vendor")
        device_id = _read_text(device / "device")
        uevent = _read_text(device / "uevent")
        driver = None
        if uevent is not None:
            for line in uevent.splitlines():
                if line.startswith("DRIVER="):
                    driver = line.removeprefix("DRIVER=")
                    break
        entry = _normalise_adapter(
            {
                "source": "linux-drm",
                "adapter": card.name,
                "vendor_id": vendor,
                "device_id": device_id,
                "driver": driver,
            }
        )
        key = tuple(sorted(entry.items()))
        if entry and key not in seen:
            adapters.append(entry)
            seen.add(key)

    pci = _command_text(["lspci", "-nn"])
    if pci is not None:
        for line in pci.splitlines():
            if not any(pattern in line.casefold() for pattern in DISPLAY_PATTERNS):
                continue
            entry = _normalise_adapter({"source": "lspci", "description": line})
            key = tuple(sorted(entry.items()))
            if key not in seen:
                adapters.append(entry)
                seen.add(key)
    return adapters


def _windows_gpu_adapters() -> list[dict[str, str]]:
    payload = _windows_powershell_json(
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterCompatibility,DriverVersion | ConvertTo-Json -Compress"
    )
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    adapters: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        entry = _normalise_adapter(
            {
                "source": "win32-video-controller",
                "name": item.get("Name"),
                "vendor": item.get("AdapterCompatibility"),
                "driver": item.get("DriverVersion"),
            }
        )
        if entry:
            adapters.append(entry)
    return adapters


def _darwin_gpu_adapters() -> list[dict[str, str]]:
    rendered = _command_text(["system_profiler", "SPDisplaysDataType", "-json"], timeout_seconds=8.0)
    if rendered is None:
        return []
    try:
        payload = json.loads(rendered)
    except json.JSONDecodeError:
        return []
    adapters: list[dict[str, str]] = []
    for item in payload.get("SPDisplaysDataType", []):
        if not isinstance(item, dict):
            continue
        entry = _normalise_adapter(
            {
                "source": "system-profiler",
                "name": item.get("_name"),
                "model": item.get("sppci_model"),
                "vendor": item.get("spdisplays_vendor"),
                "metal": item.get("spdisplays_metal"),
            }
        )
        if entry:
            adapters.append(entry)
    return adapters



def _coerce_positive_int(value: object) -> int | None:
    try:
        rendered = str(value).strip().replace(",", "")
        candidate = int(rendered)
    except (TypeError, ValueError):
        return None
    return candidate if candidate > 0 else None


def _linux_lscpu_fields() -> dict[str, str]:
    rendered = _command_text(["lscpu", "--json"])
    if rendered is None:
        return {}
    try:
        payload = json.loads(rendered)
    except json.JSONDecodeError:
        return {}
    values: dict[str, str] = {}
    for entry in payload.get("lscpu", []):
        if not isinstance(entry, dict):
            continue
        field = str(entry.get("field", "")).strip().rstrip(":").casefold()
        value = str(entry.get("data", "")).strip()
        if field and value:
            values[field] = value
    return values


def _linux_cpu_topology() -> dict[str, Any]:
    fields = _linux_lscpu_fields()
    logical = _coerce_positive_int(fields.get("cpu(s)")) or os.cpu_count()
    cores_per_socket = _coerce_positive_int(fields.get("core(s) per socket"))
    sockets = _coerce_positive_int(fields.get("socket(s)"))
    threads_per_core = _coerce_positive_int(fields.get("thread(s) per core"))
    physical_cores = None
    if cores_per_socket is not None and sockets is not None:
        physical_cores = cores_per_socket * sockets
    numa_directories = [
        node
        for node in Path("/sys/devices/system/node").glob("node[0-9]*")
        if re.fullmatch(r"node[0-9]+", node.name)
    ]
    numa_nodes = len(numa_directories) or _coerce_positive_int(fields.get("numa node(s)"))
    return {
        "logical_processors": logical,
        "physical_cores": physical_cores,
        "sockets": sockets,
        "threads_per_core": threads_per_core,
        "numa_nodes": numa_nodes,
        "source": "lscpu/sysfs",
    }


def _windows_cpu_topology() -> dict[str, Any]:
    payload = _windows_powershell_json(
        "Get-CimInstance Win32_Processor | "
        "Measure-Object -Property NumberOfCores -Sum | "
        "Select-Object @{N='physical_cores';E={$_.Sum}} | ConvertTo-Json -Compress"
    )
    physical = _coerce_positive_int(payload.get("physical_cores")) if isinstance(payload, dict) else None
    return {
        "logical_processors": os.cpu_count(),
        "physical_cores": physical,
        "sockets": None,
        "threads_per_core": None,
        "numa_nodes": None,
        "source": "Win32_Processor",
    }


def _darwin_cpu_topology() -> dict[str, Any]:
    return {
        "logical_processors": _coerce_positive_int(_command_text(["sysctl", "-n", "hw.logicalcpu"])) or os.cpu_count(),
        "physical_cores": _coerce_positive_int(_command_text(["sysctl", "-n", "hw.physicalcpu"])),
        "sockets": None,
        "threads_per_core": None,
        "numa_nodes": None,
        "source": "sysctl",
    }


def _linux_memory() -> dict[str, Any]:
    values: dict[str, int] = {}
    source = _read_text(Path("/proc/meminfo"))
    if source is not None:
        for line in source.splitlines():
            match = re.match(r"^([A-Za-z_]+):\s*(\d+)\s*kB$", line)
            if match:
                values[match.group(1)] = int(match.group(2)) * 1024
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
        "memory_channel_configuration": "not-detected-without-privileged-firmware-inventory",
        "source": "/proc/meminfo",
    }


def _windows_memory() -> dict[str, Any]:
    payload = _windows_powershell_json(
        "Get-CimInstance Win32_OperatingSystem | "
        "Select-Object TotalVisibleMemorySize,FreePhysicalMemory,TotalVirtualMemorySize,FreeVirtualMemory | "
        "ConvertTo-Json -Compress"
    )
    def kibibytes(field: str) -> int | None:
        if not isinstance(payload, dict):
            return None
        value = _coerce_positive_int(payload.get(field))
        return None if value is None else value * 1024
    return {
        "total_bytes": kibibytes("TotalVisibleMemorySize"),
        "available_bytes": kibibytes("FreePhysicalMemory"),
        "swap_total_bytes": kibibytes("TotalVirtualMemorySize"),
        "swap_free_bytes": kibibytes("FreeVirtualMemory"),
        "memory_channel_configuration": "not-detected-without-privileged-firmware-inventory",
        "source": "Win32_OperatingSystem",
    }


def _darwin_memory() -> dict[str, Any]:
    return {
        "total_bytes": _coerce_positive_int(_command_text(["sysctl", "-n", "hw.memsize"])),
        "available_bytes": None,
        "swap_total_bytes": None,
        "swap_free_bytes": None,
        "memory_channel_configuration": "not-exposed-by-portable-sysctl-query",
        "source": "sysctl",
    }


def _linux_npu_adapters() -> list[dict[str, str]]:
    adapters: list[dict[str, str]] = []
    for accelerator in sorted(Path("/sys/class/accel").glob("accel[0-9]*")):
        if re.fullmatch(r"accel[0-9]+", accelerator.name) is None:
            continue
        entry = _normalise_adapter({"source": "linux-accel", "adapter": accelerator.name})
        if entry:
            adapters.append(entry)
    pci = _command_text(["lspci", "-nn"])
    if pci is not None:
        patterns = ("neural", "npu", "ai accelerator", "processing accelerators")
        for line in pci.splitlines():
            if any(pattern in line.casefold() for pattern in patterns):
                entry = _normalise_adapter({"source": "lspci", "description": line})
                if entry and entry not in adapters:
                    adapters.append(entry)
    return adapters


def _windows_npu_adapters() -> list[dict[str, str]]:
    payload = _windows_powershell_json(
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object {$_.Name -match 'NPU|Neural|AI Accelerator'} | "
        "Select-Object Name,Manufacturer,Status | ConvertTo-Json -Compress"
    )
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    adapters: list[dict[str, str]] = []
    for item in payload:
        if isinstance(item, dict):
            entry = _normalise_adapter(
                {"source": "win32-pnp", "name": item.get("Name"), "vendor": item.get("Manufacturer"), "status": item.get("Status")}
            )
            if entry:
                adapters.append(entry)
    return adapters


def _darwin_npu_adapters() -> list[dict[str, str]]:
    # Apple Silicon's Neural Engine is not exposed as a portable, separately
    # dispatchable adapter in system_profiler.  Do not infer a provider merely
    # from the presence of an Apple SoC.
    return []


def collect_hardware_profile() -> dict[str, Any]:
    """Collect stable, best-effort host metadata without requiring drivers.

    Inventory tells an operator what is physically visible.  It never changes
    Vegai's execution backend by itself: actual accelerator dispatch requires a
    separately validated provider and is reported as disabled until then.
    """
    operating_system = platform.system() or "unknown"
    raw_architecture = platform.machine() or "unknown"
    architecture_family = _normalise_architecture(raw_architecture)
    pointer_bits = struct.calcsize("P") * 8
    if operating_system == "Linux":
        cpu_model = _linux_cpu_model()
        gpu_adapters = _linux_gpu_adapters()
        npu_adapters = _linux_npu_adapters()
        cpu_topology = _linux_cpu_topology()
        memory = _linux_memory()
    elif operating_system == "Windows":
        cpu_model = _windows_cpu_model()
        gpu_adapters = _windows_gpu_adapters()
        npu_adapters = _windows_npu_adapters()
        cpu_topology = _windows_cpu_topology()
        memory = _windows_memory()
    elif operating_system == "Darwin":
        cpu_model = _darwin_cpu_model()
        gpu_adapters = _darwin_gpu_adapters()
        npu_adapters = _darwin_npu_adapters()
        cpu_topology = _darwin_cpu_topology()
        memory = _darwin_memory()
    else:
        cpu_model = None
        gpu_adapters = []
        npu_adapters = []
        cpu_topology = {
            "logical_processors": os.cpu_count(),
            "physical_cores": None,
            "sockets": None,
            "threads_per_core": None,
            "numa_nodes": None,
            "source": "unknown",
        }
        memory = {
            "total_bytes": None,
            "available_bytes": None,
            "swap_total_bytes": None,
            "swap_free_bytes": None,
            "memory_channel_configuration": "unknown",
            "source": "unknown",
        }

    logical_processors = cpu_topology.get("logical_processors") or os.cpu_count()
    return {
        "schema": SCHEMA,
        "host": {
            "operating_system": operating_system,
            "operating_system_release": platform.release() or "unknown",
            "architecture": raw_architecture,
            "architecture_family": architecture_family,
            "pointer_bits": pointer_bits,
            "byte_order": sys.byteorder,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "model": cpu_model or platform.processor() or "unknown",
            "logical_processors": logical_processors,
            "topology": cpu_topology,
        },
        "memory": memory,
        "gpu_adapters": gpu_adapters,
        "npu_adapters": npu_adapters,
        "execution_backend": {
            "kind": "portable-cpu",
            "cpu_compute_used": True,
            "gpu_compute_used": False,
            "npu_compute_used": False,
            "gpu_runtime_required": False,
            "npu_runtime_required": False,
            "integrated_gpu_contract": (
                "No GPU compute API is loaded. Integrated graphics can coexist with Vegai; "
                "execution remains on the host CPU without a vendor GPU runtime."
            ),
            "accelerator_dispatch_contract": (
                "Detected GPU/NPU adapters are inventory only. The current binary does not silently dispatch "
                "training work to them; a separately validated provider is required before compute use is claimed."
            ),
        },
        "portability": {
            "recognized_operating_system": operating_system in RECOGNIZED_OPERATING_SYSTEMS,
            "portable_cpu_build_candidate": architecture_family != "unknown" and pointer_bits >= 32,
            "architecture_contract": (
                "The CPU backend uses standard Fortran integer/bit operations and contains no mandatory x86, ARM, "
                "CUDA, Metal, OpenCL, or vendor GPU instruction path. A target still needs its own Fortran 2023-capable "
                "toolchain build; an inventory report is not a successful compilation proof."
            ),
            "byte_order_contract": (
                "Training arithmetic does not reinterpret host byte layouts. Stream-unformatted checkpoints remain local "
                "to a compiler/ABI/endianness family, and the optional OpenVINO lane worker rejects unsupported endian ABI."
            ),
            "portable_build_contract": (
                "The portable profile emits no host-specific ISA flags. Build separately on each target "
                "architecture with a Fortran 2023-capable compiler."
            ),
            "native_build_contract": (
                "The native profile may emit host-only instructions and must not be distributed as a universal binary."
            ),
            "checkpoint_contract": (
                "Stream-unformatted checkpoints remain compiler/ABI/endianness-local; rebuild/retrain or retain "
                "matching toolchain metadata when moving across incompatible platforms."
            ),
        },
    }

def compact_hardware_summary(profile: dict[str, Any]) -> str:
    host = profile["host"]
    cpu = profile["cpu"]
    adapters = profile["gpu_adapters"]
    return (
        f"os={host['operating_system']} {host['operating_system_release']}; "
        f"arch={host['architecture']} ({host.get('architecture_family')}, {host.get('pointer_bits')}-bit, "
        f"{host.get('byte_order')}-endian); cpu={cpu['model']}; logical_cpus={cpu['logical_processors']}; "
        f"physical_cores={cpu['topology'].get('physical_cores')}; gpu_adapters={len(adapters)}; "
        f"npu_adapters={len(profile['npu_adapters'])}; backend=portable-cpu"
    )


def render_text(profile: dict[str, Any]) -> str:
    host = profile["host"]
    cpu = profile["cpu"]
    backend = profile["execution_backend"]
    lines = [
        "Vegai hardware compatibility profile",
        f"OS: {host['operating_system']} {host['operating_system_release']}",
        f"Architecture: {host['architecture']} ({host.get('architecture_family')}, {host.get('pointer_bits')}-bit, {host.get('byte_order')}-endian)",
        f"CPU: {cpu['model']}",
        f"Logical processors: {cpu['logical_processors']}",
        f"Physical cores (best effort): {cpu['topology'].get('physical_cores')}",
        f"NUMA nodes (best effort): {cpu['topology'].get('numa_nodes')}",
        f"RAM total bytes: {profile['memory'].get('total_bytes')}",
        f"RAM available bytes: {profile['memory'].get('available_bytes')}",
        "RAM channel note: " + str(profile['memory'].get('memory_channel_configuration')),
        f"Execution backend: {backend['kind']}",
        "CPU compute used: yes",
        "GPU compute used: no",
        "NPU compute used: no",
        f"GPU adapters detected: {len(profile['gpu_adapters'])}",
        f"NPU adapters detected: {len(profile['npu_adapters'])}",
    ]
    for adapter in profile["gpu_adapters"]:
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(adapter.items()))
        lines.append(f"  GPU - {rendered}")
    for adapter in profile["npu_adapters"]:
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(adapter.items()))
        lines.append(f"  NPU - {rendered}")
    lines.extend(
        [
            "Integrated GPU policy: " + backend["integrated_gpu_contract"],
            "Accelerator dispatch policy: " + backend["accelerator_dispatch_contract"],
            "Portable CPU build candidate: " + str(profile["portability"].get("portable_cpu_build_candidate")),
            "Architecture policy: " + profile["portability"]["architecture_contract"],
            "Portable build policy: " + profile["portability"]["portable_build_contract"],
            "Checkpoint policy: " + profile["portability"]["checkpoint_contract"],
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "text"), default="text", help="rendering format")
    parser.add_argument("--output", type=Path, help="optional output file; stdout remains the report")
    parser.add_argument(
        "--require-recognized-os",
        action="store_true",
        help="fail if the host is not Linux, macOS, or Windows",
    )
    arguments = parser.parse_args()

    profile = collect_hardware_profile()
    if arguments.require_recognized_os and not profile["portability"]["recognized_operating_system"]:
        print(f"Unsupported OS for the documented portability contract: {profile['host']['operating_system']}", file=sys.stderr)
        return 2

    if arguments.format == "json":
        rendered = json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        rendered = render_text(profile)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
