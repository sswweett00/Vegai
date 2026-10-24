#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""End-to-end integrity test for Vegai's continuous pretraining ledger.

This uses the compiled CLI, rather than emulating HDC updates.  It proves a
small two-shard run supports vocabulary growth, refuses a rewritten pending
shard, commits each accepted shard once, and remains byte-identical on an
idempotent restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_TOOL = ROOT / "tools" / "continuous_shard_manifest.py"
CONTROLLER = ROOT / "tools" / "continuous_pretrain.py"
STATUS_TOOL = ROOT / "tools" / "continuous_pretrain_status.py"


def fail(message: str) -> None:
    raise AssertionError(message)


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != expected:
        fail(
            f"command returned {completed.returncode}, expected {expected}: {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed


def load_state(state_path: Path) -> dict[str, object]:
    return json.loads(state_path.read_text(encoding="utf-8"))


def controller_command(binary: Path, manifest: Path, state_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(CONTROLLER),
        "--pretrain-bin",
        str(binary),
        "--manifest",
        str(manifest),
        "--state-dir",
        str(state_dir),
    ]


def status_command(manifest: Path, state_dir: Path, manifest_digest: str) -> list[str]:
    return [
        sys.executable,
        str(STATUS_TOOL),
        "--manifest",
        str(manifest),
        "--state-dir",
        str(state_dir),
        "--expect-manifest-sha256",
        manifest_digest,
        "--format",
        "json",
    ]


def parse_status(output: str) -> dict[str, object]:
    try:
        report = json.loads(output)
    except json.JSONDecodeError as error:
        fail(f"status tool did not emit JSON: {error}\n{output}")
    if not isinstance(report, dict):
        fail(f"status tool did not emit a JSON object: {report!r}")
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    binary = arguments.pretrain_bin.resolve()
    if not binary.is_file():
        print(f"continuous pretraining verification: FAIL: executable does not exist: {binary}", file=sys.stderr)
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="vegai-continuous-") as temporary:
            workspace = Path(temporary)
            shard_a = workspace / "shard-a.txt"
            shard_b = workspace / "shard-b.txt"
            shard_a_bytes = b"alpha beta\nbeta alpha\n"
            shard_b_bytes = b"beta gamma\ngamma beta\n"
            shard_a.write_bytes(shard_a_bytes)
            shard_b.write_bytes(shard_b_bytes)
            manifest = workspace / "ledger.json"
            state_dir = workspace / "state"

            run(
                [
                    sys.executable,
                    str(MANIFEST_TOOL),
                    str(shard_a),
                    str(shard_b),
                    "--output",
                    str(manifest),
                    "--source",
                    "continuous integration fixture",
                    "--license",
                    "MIT",
                    "--preprocessing",
                    "none",
                ]
            )

            manifest_bytes = manifest.read_bytes()
            manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
            initial = controller_command(binary, manifest, state_dir) + [
                "--expect-manifest-sha256",
                manifest_digest,
                "--max-vocab",
                "16",
                "--min-frequency",
                "1",
                "--window",
                "2",
                "--context-mode",
                "debiased",
                "--epochs",
                "1",
                "--max-shards",
                "1",
            ]
            first = run(initial)
            if "committed generation=1" not in first.stdout:
                fail(f"initial shard was not committed\n{first.stdout}")
            state_path = state_dir / "state.json"
            before_rewrite = state_path.read_bytes()
            state_one = load_state(state_path)
            if state_one.get("generation") != 1 or len(state_one.get("applied_shards", [])) != 1:
                fail(f"first state does not contain exactly one commit: {state_one}")
            if state_one.get("policy", {}).get("context_mode") != "debiased":
                fail(f"continuous policy did not preserve debiased mode: {state_one}")

            # A pinned immutable release rejects even a syntactically valid
            # whitespace-only manifest replacement before it reads a shard.
            manifest.write_bytes(manifest_bytes + b" ")
            pinned_replacement = run(
                controller_command(binary, manifest, state_dir) + [
                    "--expect-manifest-sha256", manifest_digest
                ], expected=1
            )
            if "manifest changed or has wrong digest" not in pinned_replacement.stdout:
                fail(f"pinned manifest replacement was not diagnosed\n{pinned_replacement.stdout}")
            if state_path.read_bytes() != before_rewrite:
                fail("pinned manifest replacement modified the durable state")
            manifest.write_bytes(manifest_bytes)

            # A pending shard must remain immutable under the SHA identity in
            # the ledger.  Its mutation is rejected before the real CLI sees it.
            shard_b.write_bytes(b"beta rewritten\nrewritten beta\n")
            status_rejected = run(status_command(manifest, state_dir, manifest_digest) + ["--verify-shards"], expected=1)
            if "wrong digest" not in status_rejected.stdout:
                fail(f"status tool did not diagnose the rewritten pending shard\n{status_rejected.stdout}")
            rejected = run(controller_command(binary, manifest, state_dir), expected=1)
            if "wrong digest" not in rejected.stdout:
                fail(f"pending-shard mutation was not diagnosed as an integrity failure\n{rejected.stdout}")
            if state_path.read_bytes() != before_rewrite:
                fail("rejected rewritten shard modified the durable state")
            shard_b.write_bytes(shard_b_bytes)

            second = run(controller_command(binary, manifest, state_dir))
            if "committed generation=2" not in second.stdout:
                fail(f"second shard was not committed\n{second.stdout}")
            committed_state = load_state(state_path)
            applied = committed_state.get("applied_shards")
            if committed_state.get("generation") != 2 or not isinstance(applied, list) or len(applied) != 2:
                fail(f"final state does not contain exactly two commits: {committed_state}")
            if [entry.get("verified_metrics", {}).get("training_oov_tokens") for entry in applied] != [0, 0]:
                fail(f"continuous run reported OOV loss: {applied}")
            if [entry.get("accepted_tokens") for entry in applied] != [4, 4]:
                fail(f"continuous run lost accepted token accounting: {applied}")
            if [entry.get("sha256") for entry in applied] != [
                hashlib.sha256(shard_a_bytes).hexdigest(),
                hashlib.sha256(shard_b_bytes).hexdigest(),
            ]:
                fail(f"journal did not preserve ordered shard identities: {applied}")

            checkpoint = state_dir / str(committed_state["checkpoint"])
            inspect = run([str(binary), "inspect", str(checkpoint), "--query", "gamma"])
            if "query: gamma" not in inspect.stdout or "completed epochs: 2" not in inspect.stdout:
                fail(f"continued checkpoint does not contain vocabulary growth and two epochs\n{inspect.stdout}")

            # A crash can leave only private candidate/snapshot names.  A
            # restart removes those non-durable artifacts but must not replay
            # a shard or alter either committed byte stream.
            generations = state_dir / "generations"
            orphan_candidate = generations / ".candidate-crash.vgai"
            orphan_snapshot = generations / ".shard-crash.txt"
            orphan_candidate.write_bytes(b"not a committed checkpoint")
            orphan_snapshot.write_bytes(b"not a committed shard snapshot\n")
            state_before_recovery = state_path.read_bytes()
            checkpoint_before_recovery = checkpoint.read_bytes()
            recovered = run(controller_command(binary, manifest, state_dir))
            if "committed generation=" in recovered.stdout:
                fail(f"recovery unexpectedly replayed a shard\n{recovered.stdout}")
            if orphan_candidate.exists() or orphan_snapshot.exists():
                fail("restart did not remove stale private candidate artifacts")
            if state_path.read_bytes() != state_before_recovery or checkpoint.read_bytes() != checkpoint_before_recovery:
                fail("private-artifact recovery changed the committed model or journal")

            # Status reports a lock conservatively and never removes it.  A
            # release gate can require the controller to be unlocked.
            lock_path = state_dir / ".continuous-pretrain.lock"
            lock_path.write_text(f"pid={os.getpid()}\n", encoding="ascii")
            locked_status = run(status_command(manifest, state_dir, manifest_digest) + ["--require-unlocked"], expected=1)
            if "lock is present" not in locked_status.stdout or not lock_path.exists():
                fail(f"status lock gate was not conservative\n{locked_status.stdout}")
            lock_path.unlink()

            complete_status = run(
                status_command(manifest, state_dir, manifest_digest)
                + ["--verify-shards", "--require-complete", "--require-unlocked"]
            )
            status_report = parse_status(complete_status.stdout)
            progress = status_report.get("progress")
            verification = status_report.get("verification")
            controller_lock = status_report.get("controller_lock")
            if (
                status_report.get("schema") != "vegai-continuous-pretraining-status/v1"
                or status_report.get("state_initialized") is not True
                or not isinstance(progress, dict)
                or progress.get("generation") != 2
                or progress.get("committed_shards") != 2
                or progress.get("pending_shards") != 0
                or progress.get("committed_accepted_tokens") != 8
                or progress.get("complete") is not True
                or not isinstance(verification, dict)
                or verification.get("state_journal_and_all_committed_checkpoints") is not True
                or verification.get("manifest_prefix") is not True
                or verification.get("shard_payloads") != "verified"
                or verification.get("verified_shards") != 2
                or verification.get("stale_private_candidate_files") != 0
                or not isinstance(controller_lock, dict)
                or controller_lock.get("present") is not False
            ):
                fail(f"complete health report is inconsistent: {status_report}")

            # The journal's verified CLI coverage metrics are durable evidence,
            # not decorative JSON.  Their relation to the committed shard and
            # immutable policy is checked before status or resume can proceed.
            state_before_metric_tamper = state_path.read_bytes()
            metric_tamper = json.loads(state_before_metric_tamper)
            metric_tamper["applied_shards"][0]["verified_metrics"]["training_oov_tokens"] = 1
            state_path.write_text(json.dumps(metric_tamper), encoding="utf-8")
            tampered_metrics_status = run(status_command(manifest, state_dir, manifest_digest), expected=1)
            if "verified metrics violate the lossless training contract" not in tampered_metrics_status.stdout:
                fail(f"status did not reject tampered journal metrics\n{tampered_metrics_status.stdout}")
            state_path.write_bytes(state_before_metric_tamper)

            # A status check validates every historical committed checkpoint,
            # so a damaged old generation is rejected before an operator can
            # mistake the state for resume-safe.  Restore the fixture bytes to
            # continue exercising idempotency and prefix rejection.
            checkpoint_original = checkpoint.read_bytes()
            checkpoint.write_bytes(b"!" + checkpoint_original[1:])
            tampered_status = run(status_command(manifest, state_dir, manifest_digest), expected=1)
            if "applied checkpoint digest mismatch" not in tampered_status.stdout:
                fail(f"status did not reject a tampered committed checkpoint\n{tampered_status.stdout}")
            checkpoint.write_bytes(checkpoint_original)

            # A normal restart after commit is a true no-op: no replay and no
            # new checkpoint/state bytes are emitted.
            state_before_restart = state_path.read_bytes()
            checkpoint_before_restart = checkpoint.read_bytes()
            restarted = run(controller_command(binary, manifest, state_dir))
            if "committed generation=" in restarted.stdout:
                fail(f"idempotent restart unexpectedly applied a shard\n{restarted.stdout}")
            if state_path.read_bytes() != state_before_restart or checkpoint.read_bytes() != checkpoint_before_restart:
                fail("idempotent restart changed the committed model or journal")

            # Accepted history is a manifest prefix, not merely an unordered
            # set.  Reordering it must be rejected without changing state.
            run([sys.executable, str(MANIFEST_TOOL), str(shard_b), str(shard_a), "--output", str(manifest)])
            reordered = run(controller_command(binary, manifest, state_dir), expected=1)
            if "rewrote committed shard" not in reordered.stdout:
                fail(f"manifest prefix rewrite was not rejected\n{reordered.stdout}")
            if state_path.read_bytes() != state_before_restart:
                fail("rejected manifest-prefix rewrite modified the durable state")

            # The manifest authoring tool itself rejects input that would lose
            # non-ASCII bytes under the strict controller contract.
            invalid = workspace / "invalid-nonascii.txt"
            invalid.write_bytes(b"alpha \xc3\xa7\n")
            invalid_manifest = workspace / "invalid.json"
            invalid_result = run(
                [sys.executable, str(MANIFEST_TOOL), str(invalid), "--output", str(invalid_manifest)], expected=1
            )
            if "non-ASCII" not in invalid_result.stdout:
                fail(f"non-ASCII strict-contract rejection was not reported\n{invalid_result.stdout}")

            invalid_crlf = workspace / "invalid-crlf.txt"
            invalid_crlf.write_bytes(b"alpha\r\nbeta\r\n")
            crlf_result = run(
                [sys.executable, str(MANIFEST_TOOL), str(invalid_crlf), "--output", str(workspace / "invalid-crlf.json")],
                expected=1,
            )
            if "CRLF" not in crlf_result.stdout:
                fail(f"CRLF strict-contract rejection was not reported\n{crlf_result.stdout}")

            # Refuse a destructive authoring typo rather than replacing a
            # source shard with JSON before it can be learned.
            self_overwrite = workspace / "self-overwrite.txt"
            self_overwrite_bytes = b"alpha beta\n"
            self_overwrite.write_bytes(self_overwrite_bytes)
            overwrite_result = run(
                [sys.executable, str(MANIFEST_TOOL), str(self_overwrite), "--output", str(self_overwrite)],
                expected=1,
            )
            if "must not overwrite" not in overwrite_result.stdout or self_overwrite.read_bytes() != self_overwrite_bytes:
                fail(f"input-shard self-overwrite protection failed\n{overwrite_result.stdout}")
    except (AssertionError, OSError, json.JSONDecodeError) as error:
        print(f"continuous pretraining verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("continuous pretraining verification: PASS (immutable shards, checkpoint health audit, exactly-once recovery, idempotent restart)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
