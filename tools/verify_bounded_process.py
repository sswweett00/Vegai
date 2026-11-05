#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regression tests for process deadlines, pipe lifecycle and bounded I/O."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from unittest.mock import patch

import bounded_process
from bounded_process import ProcessExecutionError, run_process
from lmm_process import BoundedProcessError, run_bounded_process
from pretraining_quality_evaluation import PretrainingQualityEvaluationError, invoke_with_limited_output
from run_9mib_pretraining import PretrainingRunError, run_bounded_child


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(code: str, *, timeout: float = 5, stdout_limit: int = 1024, stderr_limit: int = 1024,
        stdin: bytes | None = None):
    return run_process(
        [sys.executable, "-c", code], cwd=ROOT, timeout_seconds=timeout,
        stdout_limit_bytes=stdout_limit, stderr_limit_bytes=stderr_limit,
        label="fixture", stdin_bytes=stdin,
    )


def expect_failure(kind: str, code: str, **options: object) -> None:
    started = time.monotonic()
    try:
        run(code, **options)  # type: ignore[arg-type]
    except ProcessExecutionError as error:
        require(error.kind == kind, f"expected {kind}, got {error.kind}: {error}")
    else:
        raise AssertionError(f"expected {kind} failure")
    require(time.monotonic() - started < 6, f"{kind} enforcement did not finish promptly")


def main() -> int:
    result = run("import sys; sys.stdout.buffer.write(b'a'*1024); sys.stderr.write('secret')")
    require(result.returncode == 0 and result.stdout == b"a" * 1024 and result.stdout_bytes == 1024,
            "exact stdout boundary or output content changed")
    require(result.stdout_sha256 == hashlib.sha256(result.stdout).hexdigest(), "stdout digest changed")
    require(result.stderr_bytes == 6 and result.stderr_sha256 == hashlib.sha256(b"secret").hexdigest()
            and result.stderr_tail == b"", "stderr digest/count or default non-retention changed")
    diagnostic = run_process(
        [sys.executable, "-c", "import sys;sys.stderr.write('secret-tail')"], cwd=ROOT,
        timeout_seconds=5, stdout_limit_bytes=1024, stderr_limit_bytes=1024,
        stderr_tail_limit_bytes=4, label="diagnostic fixture",
    )
    require(diagnostic.stderr_tail == b"tail" and diagnostic.stderr_bytes == 11,
            "opt-in stderr tail was not bounded or accurate")
    tail_accumulator = bounded_process._Output(8, 32, "tail chunks", stderr_tail_limit=4)
    for part in (b"a", b"bc", b"defgh"):
        tail_accumulator.add("stderr", part)
    require(bytes(tail_accumulator.stderr_tail) == b"efgh", "stderr tail failed across chunk boundaries")
    try:
        run_process([sys.executable, "-c", "pass"], cwd=ROOT, timeout_seconds=5,
                    stdout_limit_bytes=8, stderr_limit_bytes=8, stderr_tail_limit_bytes=9, label="invalid tail")
    except ProcessExecutionError as error:
        require(error.kind == "invalid", "invalid stderr tail did not fail closed")
    else:
        raise AssertionError("stderr tail exceeded its output cap")
    require(0 <= result.child_wall_seconds <= result.elapsed_seconds, "child timing includes post-exit work")

    # A slow producer can deliver a large number of tiny chunks. The retained
    # stdout container must not incur per-chunk object/list overhead.
    accumulator = bounded_process._Output(100_000, 100, "tiny chunks")
    for _ in range(100_000):
        accumulator.add("stdout", b"x")
    require(len(accumulator.stdout_buffer) == 100_000 and sys.getsizeof(accumulator.stdout_buffer) < 200_000,
            "tiny stdout chunks caused excessive capture overhead")

    # The child fills stdout before it reads stdin. Sequential writing/reading
    # would deadlock once both OS pipes fill.
    payload = b"request" * 100_000
    response_size = 200_000
    result = run(
        "import sys; sys.stdout.buffer.write(b'x'*200000); sys.stdout.buffer.flush(); "
        "data=sys.stdin.buffer.read(); sys.stderr.write(str(len(data)))",
        stdin=payload, stdout_limit=response_size, stderr_limit=100,
    )
    require(result.stdout == b"x" * response_size, "bidirectional I/O lost stdout")
    require(result.stderr_sha256 == hashlib.sha256(str(len(payload)).encode()).hexdigest(),
            "bidirectional I/O did not deliver all stdin")

    # A cap smaller than the read buffer must trigger immediately, rather than
    # waiting for a 64 KiB buffered read or incorrectly reporting a timeout.
    expect_failure("stdout_limit", "import sys,time; sys.stdout.write('x'*2048); sys.stdout.flush(); time.sleep(15)")
    expect_failure("stderr_limit", "import sys,time; sys.stderr.write('x'*2048); sys.stderr.flush(); time.sleep(15)")
    expect_failure("timeout", "import time; time.sleep(15)", timeout=0.15)
    expect_failure("input", "import os; os.close(0); print('ok')", stdin=b"x" * 4_000_000)
    nonzero = run("import os,sys; os.close(0); sys.exit(7)", stdin=b"x" * 4_000_000)
    require(nonzero.returncode == 7, "nonzero child exit was hidden by a broken stdin pipe")

    for invalid in (0.0, -1.0, math.nan, math.inf, -math.inf, True, 10**1000):
        expect_failure("invalid", "pass", timeout=invalid)
    base = dict(command=[sys.executable, "-c", "pass"], cwd=ROOT, timeout_seconds=5,
                stdout_limit_bytes=8, stderr_limit_bytes=8, label="invalid arguments")
    for override in ({"command": "abc"}, {"command": iter(["abc"])}, {"stdout_limit_bytes": True},
                     {"stderr_limit_bytes": True}, {"stdin_limit_bytes": True},
                     {"merge_stderr": 1}, {"isolate_process_group": 1}):
        with patch.object(bounded_process.subprocess, "Popen", side_effect=AssertionError("spawned invalid command")):
            try:
                run_process(**{**base, **override})
            except ProcessExecutionError as error:
                require(error.kind == "invalid", f"invalid arguments got {error.kind}: {override}")
            else:
                raise AssertionError(f"invalid process arguments were accepted: {override}")
    with patch.object(bounded_process.threading.Thread, "start", side_effect=RuntimeError("no threads")), \
         patch.object(bounded_process, "_terminate_process_group", wraps=bounded_process._terminate_process_group) as cleanup:
        expect_failure("launch", "import time; time.sleep(15)")
        require(cleanup.called, "watcher startup failure left an unmonitored child")
    original_start = bounded_process.threading.Thread.start
    starts = [0]

    def fail_io_worker(thread):
        starts[0] += 1
        if starts[0] == 2:  # watcher starts; first I/O worker cannot
            raise RuntimeError("no I/O threads")
        return original_start(thread)

    with patch.object(bounded_process.threading.Thread, "start", fail_io_worker), \
         patch.object(bounded_process, "_pump_posix", bounded_process._pump_threaded), \
         patch.object(bounded_process, "_terminate_process_group", wraps=bounded_process._terminate_process_group) as cleanup:
        expect_failure("launch", "import time; time.sleep(15)")
        require(cleanup.called, "I/O worker startup failure left an unmonitored child")

    # Exercise the portable unbuffered-thread implementation on POSIX too.
    # Windows selects this path automatically.
    if os.name == "posix":
        with patch.object(bounded_process, "_pump_posix", bounded_process._pump_threaded):
            fallback = run("import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())", stdin=b"fallback")
            require(fallback.stdout == b"fallback", "portable pipe pump changed successful I/O")

    with tempfile.TemporaryDirectory(prefix="vegai-process-tests-") as temporary:
        root = Path(temporary)
        if os.name == "posix":
            # Child exit does not pause the total deadline while descendants
            # keep the pipes open. Test both native and threaded pumps.
            short_parent = (
                "import subprocess,sys; "
                "subprocess.Popen([sys.executable,'-c','import time;time.sleep(3)']); "
                "print('parent exited',flush=True)"
            )
            for threaded in (False, True):
                context = (patch.object(bounded_process, "_pump_posix", bounded_process._pump_threaded)
                           if threaded else nullcontext())
                with context:
                    started = time.monotonic()
                    expect_failure("timeout", short_parent, timeout=0.15)
                    require(time.monotonic() - started < 1.0,
                            "post-exit pipe grace extended the requested total deadline")
            # A trusted leaf can remain in its enclosing controller group so
            # outer cancellation reaches it. Its local timeout must kill only
            # that leaf, never the caller's process group.
            shared = run_process(
                [sys.executable, "-c", "import os; print(os.getpgrp())"], cwd=ROOT,
                timeout_seconds=3, stdout_limit_bytes=128, stderr_limit_bytes=128,
                label="shared group fixture", isolate_process_group=False,
            )
            require(int(shared.stdout) == os.getpgrp(), "trusted leaf left the enclosing process group")
            try:
                run_process(
                    [sys.executable, "-c", "import time; time.sleep(15)"], cwd=ROOT,
                    timeout_seconds=0.2, stdout_limit_bytes=128, stderr_limit_bytes=128,
                    label="shared group timeout", isolate_process_group=False,
                )
            except ProcessExecutionError as error:
                require(error.kind == "timeout", "shared group leaf did not time out")
            else:
                raise AssertionError("shared group leaf exceeded its timeout")

            nested_marker = root / "nested-survived"
            nested_leaf = (
                "import pathlib,time; time.sleep(1.1); "
                f"pathlib.Path({str(nested_marker)!r}).write_text('survived')"
            )
            nested_controller = (
                f"import sys; sys.path.insert(0,{str(ROOT / 'tools')!r}); "
                "from pathlib import Path; from bounded_process import run_process; "
                f"run_process([sys.executable,'-c',{nested_leaf!r}],cwd=Path({str(ROOT)!r}), "
                "timeout_seconds=5,stdout_limit_bytes=128,stderr_limit_bytes=128, "
                "label='nested leaf',isolate_process_group=False)"
            )
            expect_failure("timeout", nested_controller, timeout=0.35)
            time.sleep(1.1)
            require(not nested_marker.exists(), "outer timeout left the shared-group leaf running")

        pidfile = root / "pid"
        marker = root / "descendant-survived"
        grandchild = (
            "import pathlib,sys,time; time.sleep(1.3); "
            "pathlib.Path(sys.argv[1]).write_text('still running'); time.sleep(15)"
        )
        parent = (
            "import pathlib,subprocess,sys; "
            f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r},{str(marker)!r}]); "
            f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); print('parent exited',flush=True)"
        )
        try:
            expect_failure("pipe_open", parent, timeout=4)
            require(pidfile.exists(), "descendant fixture did not start")
            if os.name == "posix":
                # Parent has already exited: group termination must still kill
                # its pipe-holding descendant, not just poll the leader.
                time.sleep(1.0)
                require(not marker.exists(), "descendant survived process-group cleanup")
        finally:
            if pidfile.exists():
                try:
                    os.kill(int(pidfile.read_text()), signal.SIGTERM)
                except OSError:
                    pass

        if os.name == "posix":
            # A malicious descendant can escape the process group. It must not
            # make the caller hang; explicitly stop it after this fixture.
            escaped_pid = root / "escaped-pid"
            escaped_parent = (
                "import pathlib,subprocess,sys; "
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(15)'],start_new_session=True); "
                f"pathlib.Path({str(escaped_pid)!r}).write_text(str(p.pid))"
            )
            try:
                expect_failure("pipe_open", escaped_parent, timeout=4)
            finally:
                if escaped_pid.exists():
                    try:
                        os.kill(int(escaped_pid.read_text()), signal.SIGKILL)
                    except OSError:
                        pass

    # Every public wrapper preserves its own exception contract.
    try:
        run_bounded_process([sys.executable, "-c", "import os;os.close(0)"], cwd=ROOT,
                            timeout_seconds=3, stdout_limit_bytes=1024, stderr_limit_bytes=1024,
                            label="LMM", stdin_bytes=b"x" * 4_000_000)
    except BoundedProcessError as error:
        require("complete request" in str(error), "LMM wrapper lost its input diagnostic")
    else:
        raise AssertionError("LMM wrapper accepted partially delivered input")
    try:
        run_bounded_child([sys.executable, "-c", "import time;time.sleep(10)"], environment=os.environ,
                          timeout_seconds=0.15, stdout_limit_bytes=1024, stderr_limit_bytes=1024, label="training")
    except PretrainingRunError as error:
        require("exceeded" in str(error), "training wrapper lost its timeout diagnostic")
    else:
        raise AssertionError("training wrapper accepted a timed-out child")
    try:
        invoke_with_limited_output([sys.executable, "-c", "import time;time.sleep(10)"], 0.15)
    except PretrainingQualityEvaluationError as error:
        require("exceeded --timeout-seconds" in str(error), "quality wrapper lost its timeout diagnostic")
    else:
        raise AssertionError("quality wrapper accepted a timed-out child")
    rc, merged, _ = invoke_with_limited_output(
        [sys.executable, "-c", "import sys; sys.stdout.write('out'); sys.stderr.write('err')"], 3
    )
    require(rc == 0 and b"out" in merged and b"err" in merged, "quality inspect lost merged output")

    print("bounded process verification: PASS (timeouts, caps, bidirectional I/O, incomplete stdin, descendants, wrapper contracts)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, ProcessExecutionError) as error:
        print(f"bounded process verification: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
