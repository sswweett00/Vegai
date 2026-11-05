#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify Make's Fortran dependency graph, recovery and configuration changes.

A fake compiler checks module ordering; the real ar builds the archives. This
is not a substitute for actual Fortran compilation or runtime tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]

# Check the actual executable identities, not an opaque link count: adding a
# new default target must update this contract instead of silently masking a
# missing or duplicate link.
HDS_APPLICATIONS = {
    "hds_engine", "vegai_pretrain", "vegai_causal_lm", "vegai_neural_lm",
    "vegai_neural_pretrain", "vegai_field", "vegai_chat", "vegai_dialogue_eval",
    "vegai_control", "vegai_corpus_overlap", "vegai_ud_text_extract", "vegai_sparse_layer_bench",
}
CONTEXT_APPLICATIONS = {"vegai_context_engine", "vegai_prompt_engine", "vegai_natural_eval"}
TEST_APPLICATIONS = {"vegai_tests", "vegai_utf8_stream_tests", "vegai_sparse_layer_tests", "vegai_context_tests"}

FAKE_COMPILER = r'''#!/usr/bin/env python3
import json, os, re, sys
from pathlib import Path
args = sys.argv[1:]
log = os.environ['FAKE_COMPILER_LOG']
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(fd, (json.dumps(args) + '\n').encode())
os.close(fd)
module_dir = next(Path(arg[2:]) for arg in args if arg.startswith('-J'))
sources = [Path(arg) for arg in args if arg.lower().endswith('.f90')]
module_pattern = re.compile(r'^\s*module\s+(?!procedure\b|function\b|subroutine\b)([a-z_]\w*)', re.I | re.M)
use_pattern = re.compile(r'^\s*use(?:\s*,\s*non_intrinsic)?\s*(?:::)?\s*([a-z_]\w*)', re.I | re.M)
root = Path(os.environ['FAKE_SOURCE_ROOT'])
local_modules = set()
for file in (root / 'src').glob('*.f90'):
    local_modules.update(match.lower() for match in module_pattern.findall(file.read_text()))
local_modules.add('test_support')
for source in sources:
    content = source.read_text()
    for module in use_pattern.findall(content):
        module = module.lower()
        if module in local_modules and not (module_dir / (module + '.mod')).exists():
            raise SystemExit('missing module ' + module + ' for ' + str(source))
    for module in module_pattern.findall(content):
        (module_dir / (module.lower() + '.mod')).write_text(str(source))
    if '-c' in args:
        target = Path(args[args.index('-o') + 1])
        target.write_text('fake object for ' + str(source))
        if source.name == os.environ.get('FAKE_FAIL_SOURCE'):
            raise SystemExit('simulated compiler failure after writing object')
if '-c' not in args:
    target = Path(args[args.index('-o') + 1])
    target.write_text('#!/bin/sh\nexit 0\n')
    target.chmod(0o755)
'''


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def source_groups(root: Path) -> dict[str, list[str]]:
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    return {
        "HDS_OBJ_DIR": re.findall(r"src/([a-z_][a-z0-9_]*)\.f90", makefile.split("SOURCES :=", 1)[1].split("APP :=", 1)[0]),
        "CONTEXT_OBJ_DIR": re.findall(
            r"src/([a-z_][a-z0-9_]*)\.f90", makefile.split("CONTEXT_SOURCES :=", 1)[1].split("CONTEXT_APP :=", 1)[0]
        ),
    }


def verify_module_graph(root: Path) -> dict[str, list[str]]:
    """A clean parallel build must not rely on accidentally pre-existing .mod files."""
    for directory in ("src", "app", "test", "tools"):
        for source in sorted((root / directory).glob("*.f90")):
            for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
                require(len(line.encode("utf-8")) <= 132,
                        f"{source.relative_to(root)}:{number} exceeds the portable 132-byte free-form limit")
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    groups = source_groups(root)
    require(all(groups.values()), "could not find both source lists")
    all_modules = {name for names in groups.values() for name in names}
    module_pattern = re.compile(r"^\s*module\s+(?!procedure\b|function\b|subroutine\b)([a-z_]\w*)", re.I | re.M)
    use_pattern = re.compile(r"^\s*use(?:\s*,\s*non_intrinsic)?\s*(?:::)?\s*([a-z_]\w*)", re.I | re.M)
    for variable, names in groups.items():
        local = set(names)
        edge_pattern = re.compile(rf"^\$\({variable}\)/([a-z_][a-z0-9_]*)\.o:\s*(.*)$", re.M)
        ref_pattern = re.compile(rf"\$\({variable}\)/([a-z_][a-z0-9_]*)\.o")
        edges: dict[str, set[str]] = {}
        for producer, dependencies in edge_pattern.findall(makefile):
            edges.setdefault(producer, set()).update(ref_pattern.findall(dependencies))
        require(set(edges).issubset(local), f"unknown {variable} rule(s): {set(edges) - local}")
        for name in names:
            content = (root / "src" / f"{name}.f90").read_text(encoding="utf-8")
            definitions = [item.lower() for item in module_pattern.findall(content)]
            require(definitions == [name], f"{name}.f90 must define exactly its same-named module")
            used = {item.lower() for item in use_pattern.findall(content)} & all_modules
            require(used.issubset(local), f"{name}.f90 uses a module from a different library")
            used.discard(name)
            require(edges.get(name, set()) == used,
                    f"{name}.f90 Make USE edges differ: expected {sorted(used)}, got {sorted(edges.get(name, set()))}")
    return groups


def compiled_sources(invocations: list[list[str]]) -> set[str]:
    return {
        Path(arg).stem
        for args in invocations if "-c" in args
        for arg in args if arg.lower().endswith(".f90")
    }


def require_links(invocations: list[list[str]], expected: set[str], description: str) -> None:
    targets = [Path(args[args.index("-o") + 1]).name for args in invocations if "-c" not in args]
    require(set(targets) == expected and len(targets) == len(expected),
            f"{description}: expected {sorted(expected)}, got {sorted(targets)}")


def main() -> int:
    groups = verify_module_graph(ROOT)
    if os.name != "posix" or not shutil.which("make") or not shutil.which("ar"):
        print("make build verification: SKIP compiler exercise (requires POSIX make and ar); source graph PASS")
        return 0
    with tempfile.TemporaryDirectory(prefix="vegai-make-test-") as temporary:
        root = Path(temporary)
        shutil.copy2(ROOT / "Makefile", root / "Makefile")
        for directory in ("src", "app", "test"):
            shutil.copytree(ROOT / directory, root / directory)
        compiler = root / "fake-fortran"
        compiler.write_text(FAKE_COMPILER, encoding="utf-8")
        compiler.chmod(0o755)
        log = root / "compiler-log.jsonl"
        env = dict(os.environ, FAKE_COMPILER_LOG=str(log), FAKE_SOURCE_ROOT=str(root))

        def run_make(*targets: str, fail_source: str = "", expect_failure: bool = False) -> list[list[str]]:
            before = len(log.read_text().splitlines()) if log.exists() else 0
            command = ["make", "--no-print-directory", "-j8", *targets, f"FC={compiler}"]
            completed = subprocess.run(command, cwd=root, env=dict(env, FAKE_FAIL_SOURCE=fail_source),
                                       text=True, capture_output=True, timeout=60)
            require((completed.returncode != 0) == expect_failure,
                    f"unexpected make status {completed.returncode}:\n{completed.stdout}\n{completed.stderr}")
            after = log.read_text().splitlines() if log.exists() else []
            return [json.loads(line) for line in after[before:]]

        first = run_make("all", "test", "sparse-tests", "context-tests")
        expected = set(groups["HDS_OBJ_DIR"] + groups["CONTEXT_OBJ_DIR"])
        require(compiled_sources(first) == expected,
                f"library source mismatch: missing={expected - compiled_sources(first)}, extra={compiled_sources(first) - expected}")
        # portable_cli is intentionally compiled into both independently usable
        # libraries, with separate module/object directories. Count objects, not
        # unique source names, and still reject a duplicate build of one object.
        object_targets = [args[args.index("-o") + 1] for args in first if "-c" in args]
        require(len(object_targets) == sum(map(len, groups.values())) and len(set(object_targets)) == len(object_targets),
                "a library object was missing or compiled more than once")
        require_links(first, HDS_APPLICATIONS | CONTEXT_APPLICATIONS | TEST_APPLICATIONS,
                      "clean build application/test links")
        require(not run_make("all"), "unchanged make all unexpectedly recompiled or relinked")

        # Only the changed module and its transitive USE consumers are rebuilt.
        # GNU Make 3.81 (macOS) compares second-resolution timestamps.
        time.sleep(1.05)
        (root / "src" / "hdc_pretraining.f90").touch()
        updated = run_make("all")
        require(compiled_sources(updated) == {"hdc_pretraining", "dialogue_retrieval", "fortran_quality_gate", "fortran_control"},
                f"incorrect HDS incremental rebuild: {compiled_sources(updated)}")
        require_links(updated, HDS_APPLICATIONS, "HDS incremental rebuild links")

        # An individual missing .mod must repair the affected library even if
        # its directory, archive and application binaries still exist.
        (root / "build" / "mod-context" / "fixed_budget_backward.mod").unlink()
        repaired = run_make("context-engine", "prompt-engine")
        require(compiled_sources(repaired) == set(groups["CONTEXT_OBJ_DIR"]), "individual missing module was not repaired")
        require(not run_make("context-engine", "prompt-engine"), "repair did not leave a stable no-op build")
        shutil.rmtree(root / "build" / "mod-context")
        repaired = run_make("context-engine", "prompt-engine")
        require(compiled_sources(repaired) == set(groups["CONTEXT_OBJ_DIR"]), "missing module directory was not repaired")

        # A failed compiler can leave a partial .o; make must delete it and the
        # next invocation must rebuild it and dependent modules successfully.
        # GNU Make 3.81 (macOS) compares second-resolution timestamps.
        time.sleep(1.05)
        (root / "src" / "turkish_text.f90").touch()
        run_make("all", fail_source="turkish_text.f90", expect_failure=True)
        require(not (root / "build" / "obj-hds" / "turkish_text.o").exists(), "failed compile retained a partial object")
        recovered = run_make("all")
        require(compiled_sources(recovered) == {"turkish_text", "dialogue_retrieval"}, "failed compile was not recovered")

        # Compiler flags trigger object rebuilds; link-only changes relink but
        # do not recompile library modules. Quoting survives the config stamp.
        flags = "FFLAGS=-O0 -DVALUE='a b'"
        time.sleep(1.05)
        changed = run_make("all", flags)
        require(compiled_sources(changed) == expected, "changed compiler flags reused stale objects")
        require(not run_make("all", flags), "identical compiler flags did not reuse the build")
        time.sleep(1.05)
        link_changed = run_make("all", flags, "LDFLAGS=-Wl,--as-needed")
        require(not compiled_sources(link_changed), "link-only flag change unnecessarily recompiled modules")
        require_links(link_changed, HDS_APPLICATIONS | CONTEXT_APPLICATIONS, "link-only flag change")
        require(not run_make("all", flags, "LDFLAGS=-Wl,--as-needed"), "identical link flags did not reuse the build")
    print("make build verification: PASS (USE graph, parallel/incremental builds, module repair, failure recovery, flags)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.SubprocessError) as error:
        print(f"make build verification: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
