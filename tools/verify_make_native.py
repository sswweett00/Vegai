#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the GNU Make graph with a real GNU Fortran compiler in a temp tree."""

from __future__ import annotations

import argparse
from collections import Counter
import errno
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from verify_make_build import source_groups


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_compiled_sources(output: str, expected: list[str], description: str) -> None:
    # The real Make recipes print each compile command. Compare multisets so
    # common modules built once in each library are counted twice.
    actual = re.findall(r' -c "src/([a-z_][a-z0-9_]*)\.f90"', output)
    require(Counter(actual) == Counter(expected),
            f"{description}: expected {sorted(expected)}, got {sorted(actual)}")


def verify_fifo(root: Path) -> None:
    if not hasattr(os, "mkfifo"):
        return
    fifo = root / "stream.fifo"
    os.mkfifo(fifo)
    process = subprocess.Popen([str(root / "build" / "vegai_utf8_stream_tests"), "--fifo", str(fifo)],
                               cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError as error:
                if error.errno != errno.ENXIO or process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("FIFO reader did not open") from error
                time.sleep(0.01)
        with os.fdopen(writer, "wb", buffering=0) as stream:
            stream.write(b"abc\x00def\nxy")
        stdout, stderr = process.communicate(timeout=10)
        require(process.returncode == 0 and "PASS: non-seekable" in stdout,
                f"FIFO contract failed: {stdout}\n{stderr}")
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        fifo.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--standard", default="f2018", help="GNU Fortran -std value (default f2018)")
    args = parser.parse_args()
    require(os.name == "posix" and shutil.which("make") and shutil.which("ar"), "requires POSIX make and ar")
    require(shutil.which(args.compiler) is not None, f"Fortran compiler not found: {args.compiler}")
    groups = source_groups(ROOT)

    with tempfile.TemporaryDirectory(prefix="vegai-make-native-") as temporary:
        root = Path(temporary)
        shutil.copy2(ROOT / "Makefile", root / "Makefile")
        for directory in ("src", "app", "test"):
            shutil.copytree(ROOT / directory, root / directory)

        def run_make(*targets: str, optimization: str = "-O2", link_flags: str = "") -> str:
            command = ["make", "--no-print-directory", "-j4", *targets, f"FC={args.compiler}",
                       f"OPT_FLAGS={optimization}", f"FORTRAN_STANDARD={args.standard}",
                       "OPENMP_FLAGS=", f"LDFLAGS={link_flags}"]
            result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=180)
            require(result.returncode == 0, f"native make failed:\n{result.stdout[-10000:]}\n{result.stderr[-10000:]}")
            return result.stdout

        first = run_make("context-engine", "prompt-engine", "utf8-stream-tests", "sparse-tests")
        require("PASS: buffered UTF-8 record stream contracts" in first, "native stream contract did not pass")
        require("PASS: all sparse-layer contracts hold" in first, "optimized native sparse contracts did not pass")
        require_compiled_sources(first, groups["HDS_OBJ_DIR"] + groups["CONTEXT_OBJ_DIR"],
                                 "native clean parallel library build")
        verify_fifo(root)
        no_op = run_make("context-engine", "prompt-engine", "utf8-stream-tests")
        require(' -c "src/' not in no_op and "app/context_engine.f90" not in no_op,
                "unchanged native Make build rebuilt sources")

        (root / "build" / "mod-context" / "fixed_budget_backward.mod").unlink()
        repaired = run_make("context-engine", "prompt-engine")
        require((root / "build" / "mod-context" / "fixed_budget_backward.mod").is_file(), "missing native module was not repaired")
        require_compiled_sources(repaired, groups["CONTEXT_OBJ_DIR"], "native module repair")

        changed = run_make("context-engine", optimization="-O0 -g -fcheck=all")
        require_compiled_sources(changed, groups["CONTEXT_OBJ_DIR"], "changed compiler flags")
        same = run_make("context-engine", optimization="-O0 -g -fcheck=all")
        require(' -c "src/' not in same and "app/context_engine.f90" not in same,
                "unchanged native compiler flags rebuilt the target")
        relinked = run_make("context-engine", optimization="-O0 -g -fcheck=all", link_flags="-g")
        require(' -c "src/' not in relinked and "app/context_engine.f90" in relinked,
                "native link-only flag change did not relink without recompiling")

    print(f"native Make verification: PASS (compiler={args.compiler}, standard={args.standard})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"native Make verification: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
