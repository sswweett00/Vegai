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
import subprocess
import sys
from typing import Any, Iterable


SCHEMA = "vegai-hardware-compatibility/v1"
RECOGNIZED_OPERATING_SYSTEMS = {"Linux", "Darwin", "Windows"}
DISPLAY_PATTERNS = ("vga compatible controller", "3d controller", "display controller")


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


def collect_hardware_profile() -> dict[str, Any]:
    """Collect stable, best-effort host metadata without requiring drivers."""
    operating_system = platform.system() or "unknown"
    if operating_system == "Linux":
        cpu_model = _linux_cpu_model()
        gpu_adapters = _linux_gpu_adapters()
    elif operating_system == "Windows":
        cpu_model = _windows_cpu_model()
        gpu_adapters = _windows_gpu_adapters()
    elif operating_system == "Darwin":
        cpu_model = _darwin_cpu_model()
        gpu_adapters = _darwin_gpu_adapters()
    else:
        cpu_model = None
        gpu_adapters = []

    return {
        "schema": SCHEMA,
        "host": {
            "operating_system": operating_system,
            "operating_system_release": platform.release() or "unknown",
            "architecture": platform.machine() or "unknown",
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "model": cpu_model or platform.processor() or "unknown",
            "logical_processors": os.cpu_count(),
        },
        "gpu_adapters": gpu_adapters,
        "execution_backend": {
            "kind": "portable-cpu",
            "gpu_compute_used": False,
            "gpu_runtime_required": False,
            "integrated_gpu_contract": (
                "No GPU compute API is loaded. Integrated graphics can coexist with Vegai; "
                "execution remains on the host CPU without a vendor GPU runtime."
            ),
        },
        "portability": {
            "recognized_operating_system": operating_system in RECOGNIZED_OPERATING_SYSTEMS,
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
        f"arch={host['architecture']}; cpu={cpu['model']}; logical_cpus={cpu['logical_processors']}; "
        f"gpu_adapters={len(adapters)}; backend=portable-cpu"
    )


def render_text(profile: dict[str, Any]) -> str:
    host = profile["host"]
    cpu = profile["cpu"]
    backend = profile["execution_backend"]
    lines = [
        "Vegai hardware compatibility profile",
        f"OS: {host['operating_system']} {host['operating_system_release']}",
        f"Architecture: {host['architecture']}",
        f"CPU: {cpu['model']}",
        f"Logical processors: {cpu['logical_processors']}",
        f"Execution backend: {backend['kind']}",
        "GPU compute used: no",
        f"GPU adapters detected: {len(profile['gpu_adapters'])}",
    ]
    for adapter in profile["gpu_adapters"]:
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(adapter.items()))
        lines.append(f"  - {rendered}")
    lines.extend(
        [
            "Integrated GPU policy: " + backend["integrated_gpu_contract"],
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
