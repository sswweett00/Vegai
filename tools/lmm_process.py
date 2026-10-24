#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded no-shell child-process execution for Vegai's LMM control plane.

Gateway, quality, routing, and benchmark commands can invoke operator-provided
processes.  Capturing their stdout/stderr with ``subprocess.run`` would retain
unbounded, potentially sensitive bytes in controller memory.  This helper keeps
only a declared stdout protocol budget, streams stderr into a digest, enforces a
timeout, and tears down a POSIX process group when possible.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Mapping, Sequence

from lmm_contract import LMMContractError


class BoundedProcessError(LMMContractError):
    """A child could not complete within its bounded I/O/timeout contract."""


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    returncode: int
    stdout: bytes
    stderr_sha256: str
    stderr_bytes: int
    elapsed_seconds: float


def run_bounded_process(
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
) -> BoundedProcessResult:
    """Run one command without retaining unbounded child output.

    ``stdout`` is retained only because callers need to parse a bounded machine
    protocol.  ``stderr`` is never retained: it is represented by SHA-256 and a
    byte count on nonzero exit.  If a descendant keeps an inherited pipe open,
    closing the controller end yields a fail-closed error rather than a hang.
    """
    if timeout_seconds <= 0.0:
        raise BoundedProcessError(f"{label} timeout must be positive")
    if stdout_limit_bytes <= 0 or stderr_limit_bytes <= 0:
        raise BoundedProcessError(f"{label} output limits must be positive")
    if stdin_limit_bytes is not None and stdin_limit_bytes <= 0:
        raise BoundedProcessError(f"{label} input limit must be positive")
    if stdin_limit_bytes is not None and stdin_bytes is not None and len(stdin_bytes) > stdin_limit_bytes:
        raise BoundedProcessError(f"{label} request exceeds {stdin_limit_bytes} bytes")
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=None if environment is None else dict(environment),
            start_new_session=(os.name == "posix"),
        )
    except (OSError, ValueError) as error:
        raise BoundedProcessError(f"cannot launch {label}: {error}") from error
    if process.stdout is None or process.stderr is None or (stdin_bytes is not None and process.stdin is None):
        try:
            process.kill()
            process.wait()
        except OSError:
            pass
        raise BoundedProcessError(f"internal {label} pipes are unavailable")

    stdout_chunks: list[bytes] = []
    stdout_bytes = 0
    stderr_digest = hashlib.sha256()
    stderr_bytes = 0
    stdout_overflow = threading.Event()
    stderr_overflow = threading.Event()
    reader_errors: list[BaseException] = []
    writer_errors: list[BaseException] = []

    def stop_process(*, force: bool = False) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                process.kill()
            else:
                process.terminate()
        except OSError:
            try:
                if force:
                    process.kill()
                else:
                    process.terminate()
            except OSError:
                pass

    def drain_stdout() -> None:
        nonlocal stdout_bytes
        try:
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    return
                if stdout_bytes + len(chunk) > stdout_limit_bytes:
                    stdout_overflow.set()
                    stop_process()
                    # Drain without retaining output so shutdown cannot block
                    # on a full pipe.
                    continue
                stdout_chunks.append(chunk)
                stdout_bytes += len(chunk)
        except (OSError, ValueError) as error:
            reader_errors.append(error)

    def drain_stderr() -> None:
        nonlocal stderr_bytes
        try:
            while True:
                chunk = process.stderr.read(64 * 1024)
                if not chunk:
                    return
                stderr_bytes += len(chunk)
                stderr_digest.update(chunk)
                if stderr_bytes > stderr_limit_bytes:
                    stderr_overflow.set()
                    stop_process()
        except (OSError, ValueError) as error:
            reader_errors.append(error)

    def write_stdin() -> None:
        assert stdin_bytes is not None and process.stdin is not None
        try:
            process.stdin.write(stdin_bytes)
            process.stdin.flush()
        except BrokenPipeError:
            # A child that exits nonzero is reported below without retaining its
            # raw diagnostic; a successful child with a broken input pipe is
            # rejected after process completion.
            pass
        except (OSError, ValueError) as error:
            writer_errors.append(error)
        finally:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass

    stdout_reader = threading.Thread(target=drain_stdout, name="vegai-lmm-child-stdout", daemon=True)
    stderr_reader = threading.Thread(target=drain_stderr, name="vegai-lmm-child-stderr", daemon=True)
    writer = (
        threading.Thread(target=write_stdin, name="vegai-lmm-child-stdin", daemon=True)
        if stdin_bytes is not None else None
    )
    started = time.monotonic()
    stdout_reader.start()
    stderr_reader.start()
    if writer is not None:
        writer.start()
    deadline = started + timeout_seconds
    timed_out = False
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            timed_out = True
            break
        try:
            process.wait(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            if stdout_overflow.is_set() or stderr_overflow.is_set():
                break
    termination_timed_out = False
    if process.poll() is None:
        stop_process()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            stop_process(force=True)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # Never turn a declared timeout/cap into an unbounded controller
                # wait. This can occur only for an uninterruptible OS process;
                # close our pipe ends below and surface a fail-closed error.
                termination_timed_out = True
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except (OSError, ValueError):
                            pass

    for reader, stream in ((stdout_reader, process.stdout), (stderr_reader, process.stderr)):
        reader.join(timeout=5.0)
        if reader.is_alive():
            try:
                stream.close()
            except OSError:
                pass
            reader.join(timeout=1.0)
    if writer is not None:
        writer.join(timeout=1.0)
        if writer.is_alive():
            try:
                assert process.stdin is not None
                process.stdin.close()
            except (OSError, ValueError):
                pass
            writer.join(timeout=1.0)
    elapsed = time.monotonic() - started

    if termination_timed_out:
        raise BoundedProcessError(f"{label} did not terminate after timeout/cap enforcement")
    if stdout_reader.is_alive() or stderr_reader.is_alive() or (writer is not None and writer.is_alive()):
        raise BoundedProcessError(f"{label} left an I/O pipe open after exit")
    if reader_errors:
        raise BoundedProcessError(f"cannot drain {label} output") from reader_errors[0]
    if writer_errors and process.returncode == 0:
        raise BoundedProcessError(f"cannot deliver request to {label}") from writer_errors[0]
    if timed_out:
        raise BoundedProcessError(f"{label} exceeded {timeout_seconds:g} seconds")
    if stdout_overflow.is_set():
        raise BoundedProcessError(f"{label} stdout exceeds {stdout_limit_bytes} bytes")
    if stderr_overflow.is_set():
        raise BoundedProcessError(f"{label} stderr exceeds {stderr_limit_bytes} bytes")
    if process.returncode is None:
        raise BoundedProcessError(f"{label} did not report an exit status")
    return BoundedProcessResult(
        returncode=process.returncode,
        stdout=b"".join(stdout_chunks),
        stderr_sha256=stderr_digest.hexdigest(),
        stderr_bytes=stderr_bytes,
        elapsed_seconds=elapsed,
    )
