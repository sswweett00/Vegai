#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only integrity/status audit for a Vegai continuous-pretraining state.

This tool never starts ``vegai_pretrain``, writes a checkpoint, changes a
journal, or removes a lock.  It turns the controller's internal integrity
contracts into an operator-facing health report:

* load and validate every committed checkpoint/journal generation;
* verify that committed state remains an ordered prefix of the immutable
  manifest;
* optionally rescan every current shard payload and strict preflight; and
* expose a best-effort local controller-lock observation without treating a PID
  file as authority to remove a lock.

Use it before handoff, resume, backup/restore, retention work, or a held-out
production evaluation.  It verifies HDC transaction integrity; it is not a
semantic-quality or LLM/LMM evaluation tool.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from continuous_pretrain import (
    ContinuousPretrainError,
    GENERATIONS_DIRECTORY,
    LOCK_FILENAME,
    STATE_FILENAME,
    checkpoint_from_state,
    file_sha256,
    load_manifest,
    load_state,
    validate_manifest_prefix,
    verify_shard_bytes,
)


STATUS_SCHEMA = "vegai-continuous-pretraining-status/v1"


class StatusError(RuntimeError):
    """Expected read-only audit or CLI contract failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StatusError(message)


def no_policy_overrides() -> SimpleNamespace:
    """Match the controller's optional policy-override surface exactly."""
    return SimpleNamespace(
        max_vocab=None,
        min_frequency=None,
        window=None,
        position_shift=None,
        context_mode=None,
        epochs=None,
        initial_plasticity=None,
        minimum_plasticity=None,
        decay_events=None,
    )


def lock_report(lock_path: Path) -> dict[str, Any]:
    """Report a lock conservatively; never infer that it is safe to remove."""
    present = lock_path.exists()
    note = "No lock is present." if not present else "Lock exists; inspect ownership before any recovery."
    report: dict[str, Any] = {
        "path": str(lock_path),
        "present": present,
        "pid": None,
        "pid_observable_alive": None,
        "note": note,
    }
    if not present:
        return report

    try:
        content = lock_path.read_text(encoding="ascii", errors="replace")[:256]
    except OSError as error:
        report["note"] = f"Lock exists but cannot be read: {error}"
        return report

    for line in content.splitlines():
        if not line.startswith("pid="):
            continue
        try:
            pid = int(line.removeprefix("pid="))
        except ValueError:
            report["note"] = "Lock exists with an unparseable PID; do not remove it automatically."
            return report
        if pid <= 0:
            report["note"] = "Lock exists with a non-positive PID; do not remove it automatically."
            return report
        report["pid"] = pid
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            report["pid_observable_alive"] = False
            report["note"] = "PID is not observable locally; manual stale-lock review is still required."
        except PermissionError:
            report["pid_observable_alive"] = None
            report["note"] = "PID visibility is permission-limited; manual stale-lock review is required."
        except (OSError, ValueError, NotImplementedError) as error:
            report["pid_observable_alive"] = None
            report["note"] = f"PID liveness could not be checked ({error}); manual review is required."
        else:
            report["pid_observable_alive"] = True
            report["note"] = "PID is observable locally; do not use recovery while it may own the controller."
        return report

    report["note"] = "Lock exists without a PID line; do not remove it automatically."
    return report


def inspect_state(
    manifest_path: Path,
    state_directory: Path,
    expected_manifest_sha256: str | None,
    verify_shards: bool,
    require_complete: bool,
    require_unlocked: bool,
) -> dict[str, Any]:
    require(manifest_path.is_file(), f"manifest does not exist: {manifest_path}")

    # Pin a stable manifest read even when the operator did not supply a
    # release digest.  ``load_manifest`` hashes before and after parsing when
    # given an expected digest, closing the ordinary atomic-replace race.
    observed_manifest_sha256 = file_sha256(manifest_path)
    if expected_manifest_sha256 is not None:
        require(
            observed_manifest_sha256 == expected_manifest_sha256.lower(),
            f"manifest does not match --expect-manifest-sha256: {manifest_path}",
        )
    manifest_digest = expected_manifest_sha256.lower() if expected_manifest_sha256 else observed_manifest_sha256
    shards = load_manifest(manifest_path, manifest_digest)

    state_path = state_directory / STATE_FILENAME
    state_was_published = state_path.is_file()
    state_digest_before = file_sha256(state_path) if state_was_published else None
    state = load_state(state_directory, no_policy_overrides())
    validate_manifest_prefix(state, shards)
    state_was_published_after = state_path.is_file()
    require(
        state_was_published == state_was_published_after,
        "continuous state journal appeared or disappeared while being audited; "
        "retry after controller activity settles",
    )
    state_digest_after = file_sha256(state_path) if state_was_published_after else None
    require(
        state_digest_before == state_digest_after,
        "continuous state journal changed while being audited; retry after controller activity settles",
    )

    verified_shards = 0
    if verify_shards:
        for shard in shards:
            verify_shard_bytes(shard)
            verified_shards += 1

    generation = state["generation"]
    applied = state["applied_shards"]
    require(isinstance(generation, int) and isinstance(applied, list), "validated state unexpectedly changed shape")
    pending = len(shards) - generation
    require(pending >= 0, "manifest has fewer shards than committed state")
    require(not require_complete or pending == 0,
            f"continuous state is not complete: {pending} manifest shard(s) remain pending")

    latest_checkpoint: dict[str, Any] | None = None
    if generation > 0:
        checkpoint_path = checkpoint_from_state(state_directory, state)
        latest_checkpoint = {
            "path": str(checkpoint_path),
            "sha256": state["checkpoint_sha256"],
            "bytes": checkpoint_path.stat().st_size,
        }

    lock = lock_report(state_directory / LOCK_FILENAME)
    require(not require_unlocked or not lock["present"],
            f"continuous controller lock is present: {lock['path']}")

    total_tokens = sum(entry["accepted_tokens"] for entry in applied)
    total_records = sum(entry["physical_lf_records"] for entry in applied)
    private_work = state_directory / GENERATIONS_DIRECTORY
    stale_private_candidates = 0
    if private_work.is_dir():
        stale_private_candidates = sum(
            1 for pattern in (".candidate-*.vgai", ".shard-*.txt") for _ in private_work.glob(pattern)
        )

    return {
        "schema": STATUS_SCHEMA,
        "state_initialized": state_was_published,
        "state_directory": str(state_directory),
        "state_journal": None if not state_was_published else {
            "path": str(state_path),
            "sha256": state_digest_after,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_digest,
            "shards": len(shards),
        },
        "policy": state["policy"],
        "progress": {
            "generation": generation,
            "committed_shards": len(applied),
            "pending_shards": pending,
            "committed_physical_lf_records": total_records,
            "committed_accepted_tokens": total_tokens,
            "complete": pending == 0,
        },
        "latest_checkpoint": latest_checkpoint,
        "verification": {
            "state_journal_and_all_committed_checkpoints": True,
            "manifest_prefix": True,
            "shard_payloads": "verified" if verify_shards else "metadata-only",
            "verified_shards": verified_shards,
            "stale_private_candidate_files": stale_private_candidates,
        },
        "controller_lock": lock,
    }


def print_text(report: dict[str, Any]) -> None:
    progress = report["progress"]
    manifest = report["manifest"]
    verification = report["verification"]
    print("continuous pretraining status: PASS")
    print(f"state initialized: {report['state_initialized']}")
    print(f"manifest: {manifest['path']}")
    print(f"manifest SHA-256: {manifest['sha256']}")
    print(
        "progress: "
        f"generation={progress['generation']} committed_shards={progress['committed_shards']} "
        f"pending_shards={progress['pending_shards']} complete={progress['complete']}"
    )
    print(
        "committed data: "
        f"records={progress['committed_physical_lf_records']} tokens={progress['committed_accepted_tokens']}"
    )
    checkpoint = report["latest_checkpoint"]
    if checkpoint is not None:
        print(f"latest checkpoint: {checkpoint['path']}")
        print(f"latest checkpoint SHA-256: {checkpoint['sha256']}")
    print(
        "shard payload verification: "
        f"{verification['shard_payloads']} ({verification['verified_shards']} shard(s))"
    )
    print(f"stale private candidates: {verification['stale_private_candidate_files']}")
    lock = report["controller_lock"]
    print(f"controller lock present: {lock['present']}")
    print(f"controller lock note: {lock['note']}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="ordered immutable shard manifest")
    parser.add_argument("--state-dir", required=True, type=Path, help="continuous controller state directory")
    parser.add_argument(
        "--expect-manifest-sha256",
        help="optional immutable manifest digest; reject replacement during this read-only audit",
    )
    parser.add_argument(
        "--verify-shards", action="store_true",
        help="rescan every manifest shard payload and strict preflight (expensive for large corpora)",
    )
    parser.add_argument(
        "--require-complete", action="store_true",
        help="fail unless every manifest shard has a committed generation",
    )
    parser.add_argument(
        "--require-unlocked", action="store_true",
        help="fail if a controller lock currently exists; this never deletes it",
    )
    parser.add_argument("--format", choices=("json", "text"), default="json", help="report format (default: json)")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        manifest_path = arguments.manifest.resolve()
        state_directory = arguments.state_dir.resolve()
        report = inspect_state(
            manifest_path,
            state_directory,
            arguments.expect_manifest_sha256,
            arguments.verify_shards,
            arguments.require_complete,
            arguments.require_unlocked,
        )
    except (StatusError, ContinuousPretrainError, OSError, ValueError) as error:
        print(f"continuous pretraining status: FAIL: {error}", file=sys.stderr)
        return 1

    if arguments.format == "json":
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
