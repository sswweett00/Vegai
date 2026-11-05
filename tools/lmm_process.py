#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded no-shell child-process execution for Vegai's LMM control plane.

Only a declared stdout protocol budget is retained; stderr is represented by a
SHA-256 digest and count. Process-group termination and I/O deadlines are
shared with the pretraining controllers rather than implemented separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from bounded_process import ProcessExecutionError, run_process
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
    isolate_process_group: bool = True,
) -> BoundedProcessResult:
    """Run one command without retaining unbounded child output or diagnostics."""
    try:
        result = run_process(
            command, cwd=cwd, timeout_seconds=timeout_seconds,
            stdout_limit_bytes=stdout_limit_bytes, stderr_limit_bytes=stderr_limit_bytes,
            label=label, stdin_bytes=stdin_bytes, stdin_limit_bytes=stdin_limit_bytes,
            environment=environment, isolate_process_group=isolate_process_group,
        )
    except ProcessExecutionError as error:
        raise BoundedProcessError(str(error)) from error
    return BoundedProcessResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr_sha256=result.stderr_sha256,
        stderr_bytes=result.stderr_bytes,
        elapsed_seconds=result.elapsed_seconds,
    )
