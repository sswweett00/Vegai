#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Contract test for the portable hardware/OS metadata reporter."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REPORTER = ROOT / "tools" / "hardware_compatibility.py"


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="vegai-hardware-profile-") as temporary:
            output = Path(temporary) / "profile.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPORTER),
                    "--format",
                    "json",
                    "--require-recognized-os",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode != 0:
                fail(f"hardware reporter failed ({result.returncode})\n{result.stdout}")
            if not output.is_file():
                fail("hardware reporter did not create requested JSON output")
            profile = json.loads(output.read_text(encoding="utf-8"))
            if profile.get("schema") != "vegai-hardware-compatibility/v2":
                fail(f"unexpected profile schema: {profile.get('schema')!r}")
            host = profile.get("host")
            cpu = profile.get("cpu")
            backend = profile.get("execution_backend")
            portability = profile.get("portability")
            if not isinstance(host, dict) or not isinstance(cpu, dict) or not isinstance(backend, dict):
                fail("hardware profile omitted required host/cpu/backend objects")
            if not isinstance(portability, dict) or not portability.get("recognized_operating_system"):
                fail("hardware profile did not recognize the current CI operating system")
            if not isinstance(host.get("operating_system"), str) or not host["operating_system"]:
                fail("hardware profile omitted operating system")
            if not isinstance(host.get("architecture"), str) or not host["architecture"]:
                fail("hardware profile omitted architecture")
            if not isinstance(host.get("architecture_family"), str) or not host["architecture_family"]:
                fail("hardware profile omitted normalized architecture family")
            if not isinstance(host.get("pointer_bits"), int) or host["pointer_bits"] <= 0:
                fail("hardware profile omitted pointer-width metadata")
            if host.get("byte_order") not in {"little", "big"}:
                fail("hardware profile omitted byte-order metadata")
            if portability.get("portable_cpu_build_candidate") is not True:
                fail("hardware profile did not expose a portable CPU build candidate for current CI architecture")
            if not isinstance(portability.get("architecture_contract"), str) or not portability["architecture_contract"]:
                fail("hardware profile omitted architecture portability contract")
            if not isinstance(cpu.get("model"), str) or not cpu["model"]:
                fail("hardware profile omitted CPU model fallback")
            if backend.get("kind") != "portable-cpu":
                fail(f"unexpected backend kind: {backend.get('kind')!r}")
            if backend.get("cpu_compute_used") is not True:
                fail("portable backend did not identify active CPU compute")
            if backend.get("gpu_compute_used") is not False or backend.get("gpu_runtime_required") is not False:
                fail("portable backend incorrectly claims GPU compute/runtime use")
            if backend.get("npu_compute_used") is not False or backend.get("npu_runtime_required") is not False:
                fail("portable backend incorrectly claims NPU compute/runtime use")
            if not isinstance(profile.get("gpu_adapters"), list) or not isinstance(profile.get("npu_adapters"), list):
                fail("GPU/NPU inventory is not a list")
            topology = cpu.get("topology")
            memory = profile.get("memory")
            if not isinstance(topology, dict) or not isinstance(memory, dict):
                fail("hardware profile omitted CPU topology or memory object")
            if topology.get("logical_processors") is not None and not isinstance(topology["logical_processors"], int):
                fail("hardware profile logical processor topology is malformed")
            if not isinstance(memory.get("memory_channel_configuration"), str):
                fail("hardware profile omitted RAM channel discovery policy")

            # The distribution default must stay free of host-only ISA flags;
            # native tuning belongs to the explicit opt-in profile only.
            makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
            cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
            if "OPT_FLAGS ?= -O3" not in makefile:
                fail("Make portable default no longer uses the architecture-neutral O3 profile")
            if "NATIVE_OPT_FLAGS ?= -O3 -march=native -mtune=native" not in makefile:
                fail("Make native profile no longer declares host tuning separately")
            if 'set(VEGAI_CPU_TARGET "portable"' not in cmake:
                fail("CMake CPU target does not default to portable")
            if 'VEGAI_CPU_TARGET STREQUAL "native"' not in cmake:
                fail("CMake no longer exposes an opt-in native CPU profile")
    except (AssertionError, OSError, json.JSONDecodeError) as error:
        print(f"hardware compatibility verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("hardware compatibility verification: PASS (portable CPU backend, RAM/topology metadata, optional GPU/NPU inventory)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
