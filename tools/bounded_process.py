#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded, no-shell process execution shared by the control planes.

On POSIX, nonblocking pipes are multiplexed in the calling thread.  This is
important: closing a buffered pipe from another thread can itself block forever
if a descendant inherited its other end.  Windows uses unbuffered daemon I/O
threads instead, because its pipe handles cannot be used with selectors.  A
stuck Windows descendant can leave a daemon thread/handle until it closes the
pipe; it cannot keep the caller waiting indefinitely.  POSIX process groups are
terminated on failures, even if their original leader has already exited.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import threading
import time
from typing import Mapping, Sequence


# A direct child has exited, but there may still be buffered bytes to drain.
# An inherited pipe that does not reach EOF within this grace is an error.
_POST_EXIT_GRACE = 0.5
_TERMINATE_GRACE = 0.25
_KILL_GRACE = 0.5
_POLL_INTERVAL = 0.05
_CHUNK_BYTES = 64 * 1024


class ProcessExecutionError(RuntimeError):
    """An invocation could not satisfy its time, I/O or lifecycle contract."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stdout_sha256: str
    stdout_bytes: int
    stderr_sha256: str
    stderr_bytes: int
    # Only spawn through direct-child exit; excludes subsequent pipe draining.
    child_wall_seconds: float
    # Includes spawn and pipe draining, for callers reporting total latency.
    elapsed_seconds: float
    # Opt-in only: existing callers retain no stderr content by default.
    stderr_tail: bytes = b""


class _Output:
    def __init__(self, stdout_limit: int, stderr_limit: int, label: str, stderr_tail_limit: int = 0) -> None:
        self.stdout_limit = stdout_limit
        self.stderr_limit = stderr_limit
        self.label = label
        # A bytearray bounds storage overhead even when a child produces many
        # tiny writes; a list of one-byte chunks can dwarf the protocol cap.
        self.stdout_buffer = bytearray()
        self.stdout_bytes = 0
        self.stderr_bytes = 0
        self.stdout_hash = hashlib.sha256()
        self.stderr_hash = hashlib.sha256()
        self.stderr_tail_limit = stderr_tail_limit
        self.stderr_tail = bytearray()

    def add(self, name: str, chunk: bytes) -> None:
        if name == "stdout":
            if self.stdout_bytes + len(chunk) > self.stdout_limit:
                raise ProcessExecutionError("stdout_limit", f"{self.label} stdout exceeds {self.stdout_limit} bytes")
            self.stdout_buffer.extend(chunk)
            self.stdout_bytes += len(chunk)
            self.stdout_hash.update(chunk)
        else:
            if self.stderr_bytes + len(chunk) > self.stderr_limit:
                raise ProcessExecutionError("stderr_limit", f"{self.label} stderr exceeds {self.stderr_limit} bytes")
            self.stderr_bytes += len(chunk)
            self.stderr_hash.update(chunk)
            if self.stderr_tail_limit:
                # Bound diagnostics independently of the total stderr cap.
                self.stderr_tail.extend(chunk[-self.stderr_tail_limit:])
                excess = len(self.stderr_tail) - self.stderr_tail_limit
                if excess > 0:
                    del self.stderr_tail[:excess]


def _terminate_process_group(process: subprocess.Popen[bytes], isolated: bool = True) -> None:
    """Bound termination; signal the group only when this runner created it."""
    def send(force: bool) -> None:
        try:
            if os.name == "posix" and isolated:
                # Do not skip this just because poll() says the leader exited:
                # descendants can still be alive in the original process group.
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif process.poll() is None:
                process.kill() if force else process.terminate()
        except OSError:
            # The group may have exited already. Still try to reap/kill the
            # direct child if it is alive (e.g. on a non-POSIX platform).
            try:
                if process.poll() is None:
                    process.kill() if force else process.terminate()
            except OSError:
                pass

    send(False)
    try:
        process.wait(timeout=_TERMINATE_GRACE)
    except subprocess.TimeoutExpired:
        pass
    # A leader can exit on SIGTERM while an inherited-pipe holder ignores it.
    # Always send SIGKILL to the group as well, not just to a live leader.
    send(True)
    try:
        process.wait(timeout=_KILL_GRACE)
    except subprocess.TimeoutExpired as error:
        raise ProcessExecutionError("termination", "child did not terminate after timeout/cap enforcement") from error


def _close(stream: object) -> None:
    try:
        stream.close()  # type: ignore[attr-defined]
    except (OSError, ValueError):
        pass


def _pump_posix(
    process: subprocess.Popen[bytes],
    output: _Output,
    stdin_bytes: bytes | None,
    done: threading.Event,
    deadline: float,
    label: str,
    timeout_seconds: float,
) -> None:
    """Pump nonblocking POSIX pipes without any blocking reader/writer thread."""
    assert process.stdout is not None
    delivered = 0
    input_failed = False
    post_exit_deadline: float | None = None
    with selectors.DefaultSelector() as selector:
        def register(stream: object, events: int, name: str) -> None:
            os.set_blocking(stream.fileno(), False)  # type: ignore[attr-defined]
            selector.register(stream, events, name)

        def finish(stream: object) -> None:
            selector.unregister(stream)
            _close(stream)

        register(process.stdout, selectors.EVENT_READ, "stdout")
        if process.stderr is not None:
            register(process.stderr, selectors.EVENT_READ, "stderr")
        if process.stdin is not None:
            if stdin_bytes:
                register(process.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                _close(process.stdin)
        view = memoryview(stdin_bytes) if stdin_bytes is not None else None

        while True:
            now = time.monotonic()
            if now >= deadline:
                raise ProcessExecutionError("timeout", f"{label} exceeded {timeout_seconds:g} seconds")
            if done.is_set():
                if post_exit_deadline is None:
                    post_exit_deadline = min(deadline, now + _POST_EXIT_GRACE)
                if not selector.get_map():
                    break
                if now >= post_exit_deadline:
                    raise ProcessExecutionError("pipe_open", f"{label} left an I/O pipe open after exit")

            remaining = (post_exit_deadline if post_exit_deadline is not None else deadline) - now
            wait = min(_POLL_INTERVAL, max(0.0, remaining))
            if not selector.get_map():
                # Once the pipes close, wake on process exit rather than
                # adding a polling interval to every short-lived invocation.
                done.wait(wait)
                continue
            for key, _ in selector.select(wait):
                stream, name = key.fileobj, key.data
                if name == "stdin":
                    assert view is not None
                    try:
                        written = os.write(stream.fileno(), view[delivered:delivered + _CHUNK_BYTES])
                    except BlockingIOError:
                        continue
                    except OSError:
                        input_failed = True
                        finish(stream)
                        continue
                    if written <= 0:
                        input_failed = True
                        finish(stream)
                        continue
                    delivered += written
                    if delivered == len(view):
                        finish(stream)
                else:
                    try:
                        chunk = os.read(stream.fileno(), _CHUNK_BYTES)
                    except BlockingIOError:
                        continue
                    except OSError as error:
                        raise ProcessExecutionError("output", f"cannot drain {label} output") from error
                    if not chunk:
                        finish(stream)
                    else:
                        output.add(name, chunk)

    if process.returncode == 0 and (input_failed or (view is not None and delivered != len(view))):
        raise ProcessExecutionError("input", f"cannot deliver complete request to {label}")


def _pump_threaded(
    process: subprocess.Popen[bytes],
    output: _Output,
    stdin_bytes: bytes | None,
    done: threading.Event,
    deadline: float,
    label: str,
    timeout_seconds: float,
) -> None:
    """Portable fallback: workers own unbuffered pipes; the caller never blocks closing them."""
    assert process.stdout is not None
    errors: list[ProcessExecutionError] = []
    wake = threading.Event()
    delivered = 0
    input_failed = False

    def read(name: str, stream: object) -> None:
        try:
            while True:
                chunk = stream.read(_CHUNK_BYTES)  # type: ignore[attr-defined]
                if not chunk:
                    break
                output.add(name, chunk)
        except ProcessExecutionError as error:
            errors.append(error)
            wake.set()
        except (OSError, ValueError) as error:
            errors.append(ProcessExecutionError("output", f"cannot drain {label} output"))
            errors[-1].__cause__ = error
            wake.set()
        finally:
            _close(stream)
            wake.set()

    def write(stream: object) -> None:
        nonlocal delivered, input_failed
        assert stdin_bytes is not None
        view = memoryview(stdin_bytes)
        try:
            while delivered < len(view):
                count = stream.write(view[delivered:delivered + _CHUNK_BYTES])  # type: ignore[attr-defined]
                if not count:
                    input_failed = True
                    break
                delivered += count
        except (OSError, ValueError):
            input_failed = True
        finally:
            _close(stream)
            wake.set()

    threads = [threading.Thread(target=read, args=("stdout", process.stdout), daemon=True)]
    streams = [process.stdout]
    if process.stderr is not None:
        threads.append(threading.Thread(target=read, args=("stderr", process.stderr), daemon=True))
        streams.append(process.stderr)
    if process.stdin is not None:
        threads.append(threading.Thread(target=write, args=(process.stdin,), daemon=True))
        streams.append(process.stdin)
    started = 0
    try:
        for thread in threads:
            thread.start()
            started += 1
    except (OSError, RuntimeError) as error:
        # Close only handles with no owner. The outer lifecycle handler then
        # kills the child; workers that did start close their handles on EOF.
        for stream in streams[started:]:
            _close(stream)
        raise ProcessExecutionError("launch", f"cannot start {label} I/O workers: {error}") from error

    post_exit_deadline: float | None = None
    while True:
        if errors:
            raise errors[0]
        now = time.monotonic()
        if now >= deadline:
            raise ProcessExecutionError("timeout", f"{label} exceeded {timeout_seconds:g} seconds")
        if done.is_set():
            if post_exit_deadline is None:
                post_exit_deadline = min(deadline, now + _POST_EXIT_GRACE)
            if all(not thread.is_alive() for thread in threads):
                break
            if now >= post_exit_deadline:
                raise ProcessExecutionError("pipe_open", f"{label} left an I/O pipe open after exit")
        remaining = (post_exit_deadline if post_exit_deadline is not None else deadline) - now
        wake.wait(min(_POLL_INTERVAL, max(0.0, remaining)))
        wake.clear()

    if errors:
        raise errors[0]
    if process.returncode == 0 and (input_failed or (stdin_bytes is not None and delivered != len(stdin_bytes))):
        raise ProcessExecutionError("input", f"cannot deliver complete request to {label}")


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
    label: str,
    stdin_bytes: bytes | None = None,
    stdin_limit_bytes: int | None = None,
    environment: Mapping[str, str] | None = None,
    merge_stderr: bool = False,
    isolate_process_group: bool = True,
    stderr_tail_limit_bytes: int = 0,
) -> ProcessResult:
    """Execute a process with bounded retained output, input and wall time.

    stderr is hashed rather than retained by default; a caller can opt into a
    bounded tail or deliberately merge it into stdout. A zero-exit child that
    closes stdin before all request bytes have been written is rejected.
    Nonzero exits remain available to callers. The deadline starts before
    Popen and covers pipe draining, but a Popen call itself cannot be
    interrupted here; failure cleanup can add bounded termination grace.
    With isolation disabled only the direct child is killed; this is for trusted
    leaf helpers that must remain in an enclosing controller's POSIX group.
    """
    if type(isolate_process_group) is not bool or type(merge_stderr) is not bool:
        raise ProcessExecutionError("invalid", f"{label} process-group isolation and stderr merging must be boolean")
    try:
        valid_timeout = type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds) and timeout_seconds > 0
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise ProcessExecutionError("invalid", f"{label} timeout must be finite and positive")
    if (not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command
            or any(not isinstance(item, str) or not item for item in command)):
        raise ProcessExecutionError("invalid", f"{label} command must be a nonempty argument sequence")
    if (type(stdout_limit_bytes) is not int or type(stderr_limit_bytes) is not int
            or stdout_limit_bytes <= 0 or stderr_limit_bytes <= 0):
        raise ProcessExecutionError("invalid", f"{label} output limits must be positive integers")
    if (type(stderr_tail_limit_bytes) is not int or stderr_tail_limit_bytes < 0
            or stderr_tail_limit_bytes > stderr_limit_bytes or (merge_stderr and stderr_tail_limit_bytes)):
        raise ProcessExecutionError("invalid", f"{label} stderr tail limit is invalid")
    if stdin_limit_bytes is not None and (type(stdin_limit_bytes) is not int or stdin_limit_bytes <= 0):
        raise ProcessExecutionError("invalid", f"{label} input limit must be a positive integer")
    if stdin_bytes is not None and not isinstance(stdin_bytes, bytes):
        raise ProcessExecutionError("invalid", f"{label} request must be bytes")
    if stdin_limit_bytes is not None and stdin_bytes is not None and len(stdin_bytes) > stdin_limit_bytes:
        raise ProcessExecutionError("invalid", f"{label} request exceeds {stdin_limit_bytes} bytes")

    started_ns = time.perf_counter_ns()
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(command), cwd=cwd,
            stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            env=None if environment is None else dict(environment),
            start_new_session=(os.name == "posix" and isolate_process_group), bufsize=0,
        )
    except (OSError, ValueError) as error:
        raise ProcessExecutionError("launch", f"cannot launch {label}: {error}") from error

    done = threading.Event()
    stopped_ns: list[int] = []

    def watch_child() -> None:
        process.wait()
        stopped_ns.append(time.perf_counter_ns())
        done.set()

    try:
        threading.Thread(target=watch_child, name="vegai-child-wait", daemon=True).start()
    except (OSError, RuntimeError) as error:
        # Thread exhaustion after Popen must not leave an unmonitored child or
        # its pipe handles running indefinitely.
        try:
            _terminate_process_group(process, isolated=isolate_process_group)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    _close(stream)
        raise ProcessExecutionError("launch", f"cannot monitor {label}: {error}") from error
    output = _Output(stdout_limit_bytes, stderr_limit_bytes, label, stderr_tail_limit_bytes)
    try:
        pump = _pump_posix if os.name == "posix" else _pump_threaded
        pump(process, output, stdin_bytes, done, started + timeout_seconds, label, timeout_seconds)
    except BaseException:
        _terminate_process_group(process, isolated=isolate_process_group)
        raise
    finally:
        if os.name == "posix":
            # No thread owns these unbuffered, nonblocking handles on POSIX.
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    _close(stream)
        # On Windows each I/O thread closes its own handle. Never close one in
        # the caller while its reader/writer may be blocked in the kernel.

    assert process.returncode is not None and stopped_ns
    return ProcessResult(
        returncode=process.returncode,
        stdout=bytes(output.stdout_buffer),
        stdout_sha256=output.stdout_hash.hexdigest(),
        stdout_bytes=output.stdout_bytes,
        stderr_sha256=output.stderr_hash.hexdigest(),
        stderr_bytes=output.stderr_bytes,
        child_wall_seconds=(stopped_ns[0] - started_ns) / 1_000_000_000,
        # Same monotonic clock as child_wall_seconds: Windows mono-
        # tonic() has ~15.6 ms resolution and could otherwise report
        # a total duration shorter than the measured child lifetime.
        elapsed_seconds=(time.perf_counter_ns() - started_ns) / 1_000_000_000,
        stderr_tail=bytes(output.stderr_tail),
    )
