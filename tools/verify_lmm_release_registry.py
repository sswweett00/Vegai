#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synthetic end-to-end verification for the local authoritative release ledger.

The fixture creates fake model/evaluation artifacts only to exercise contracts.
It is not a model-quality, telemetry-integrity, traffic-enforcement, signing, or
production-readiness claim.  It verifies that governor artifacts can be
serialized under a single local CAS head without leaking raw telemetry IDs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from verify_lmm_release_governor import (
    canonical,
    make_portfolio_fixture,
    policy_mapping,
    sha,
    write_batch,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
GOVERNOR = ROOT / "tools" / "lmm_release_governor.py"
REGISTRY = ROOT / "tools" / "lmm_release_registry.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(command: list[str], expected: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"command exit {result.returncode}, expected {expected}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def governor(*arguments: str | Path) -> list[str]:
    return [sys.executable, str(GOVERNOR), *(str(argument) for argument in arguments)]


def registry(*arguments: str | Path) -> list[str]:
    return [sys.executable, str(REGISTRY), *(str(argument) for argument in arguments)]


def parsed_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    value = json.loads(result.stdout)
    require(isinstance(value, dict), "CLI output is not an object")
    return value


def initial_governor_artifacts(policy: Path, state: Path, decision: Path) -> dict[str, object]:
    return parsed_stdout(run(governor(
        "init", "--policy", policy, "--state-output", state, "--decision-output", decision,
    ), expected=0))


def advance_governor(policy: Path, prior_state: Path, batch: Path, state: Path, decision: Path) -> dict[str, object]:
    return parsed_stdout(run(governor(
        "advance", "--policy", policy, "--state", prior_state, "--observations", batch,
        "--state-output", state, "--decision-output", decision,
    ), expected=0))


def record(policy: Path, state: Path, decision: Path, ledger: Path, expected_head: str) -> dict[str, object]:
    return parsed_stdout(run(registry(
        "record", "--policy", policy, "--state", state, "--decision", decision,
        "--ledger", ledger, "--expected-state-sha256", expected_head,
    ), expected=0))


def main() -> int:
    require(GOVERNOR.is_file(), f"missing {GOVERNOR}")
    require(REGISTRY.is_file(), f"missing {REGISTRY}")
    with tempfile.TemporaryDirectory(prefix="vegai-release-registry-") as temporary_name:
        workspace = Path(temporary_name)
        portfolio, task, _ = make_portfolio_fixture(workspace)
        policy = workspace / "policy.json"
        write_json(policy, policy_mapping(portfolio, task))
        ledger = workspace / "authoritative-release.jsonl"

        state0 = workspace / "state-000.json"
        decision0 = workspace / "decision-000.json"
        initial = initial_governor_artifacts(policy, state0, decision0)
        require(initial["outcome"] == "shadow_insufficient_evidence", "fixture did not begin in shadow")
        registered0 = record(policy, state0, decision0, ledger, "GENESIS")
        require(registered0["sequence"] == 0 and registered0["head_state_sha256"] == sha(state0),
                "initial state was not registered as authoritative head")
        require(registered0["traffic_directive"]["mode"] == "shadow-mirror-only",
                "initial authoritative directive is not shadow-only")
        first_ledger_sha = sha(ledger)
        privacy_text = ledger.read_text(encoding="utf-8")
        require("shadow-baseline-observation-0" not in privacy_text and "replication-0" not in privacy_text,
                "authoritative registry leaked raw telemetry identifiers")
        require(str(workspace) not in privacy_text, "authoritative registry leaked local artifact paths")

        inspected0 = parsed_stdout(run(registry("inspect", "--policy", policy, "--ledger", ledger), expected=0))
        require(inspected0["events"] == 1 and inspected0["head_state_sha256"] == sha(state0),
                "registry inspection has wrong initial head")

        # Shadow-disabled policies begin directly at bounded canary; the
        # registry must accept that distinct but policy-derived GENESIS shape.
        direct_policy = workspace / "direct-canary-policy.json"
        direct_policy_mapping = policy_mapping(portfolio, task)
        direct_policy_mapping["id"] = "synthetic-direct-canary-release"
        direct_policy_mapping["shadow_basis_points"] = 0
        direct_policy_mapping["shadow_minimum_observations_per_arm"] = 0
        write_json(direct_policy, direct_policy_mapping)
        direct_state = workspace / "direct-canary-state-000.json"
        direct_decision = workspace / "direct-canary-decision-000.json"
        initial_governor_artifacts(direct_policy, direct_state, direct_decision)
        direct_ledger = workspace / "direct-canary-ledger.jsonl"
        direct_registered = record(direct_policy, direct_state, direct_decision, direct_ledger, "GENESIS")
        require(direct_registered["phase"] == "canary" and
                direct_registered["traffic_directive"]["mode"] == "bounded-canary",
                "registry rejected or misrepresented shadow-disabled initial canary state")

        duplicate = run(registry(
            "record", "--policy", policy, "--state", state0, "--decision", decision0,
            "--ledger", ledger, "--expected-state-sha256", "GENESIS",
        ), expected=2)
        require("head conflict" in duplicate.stderr, "registry accepted stale initial CAS head")
        require(sha(ledger) == first_ledger_sha, "stale CAS attempt changed authoritative ledger")

        state0_mapping = json.loads(state0.read_text(encoding="utf-8"))
        shadow_batch = workspace / "shadow-pass.jsonl"
        write_batch(
            shadow_batch, sha(policy), state0_mapping["baseline_identity_sha256"], state0_mapping["candidate_identity_sha256"],
            stage="shadow", count=40, candidate_health=1.0, candidate_cost=150.0, candidate_latency=150.0,
        )
        state1 = workspace / "state-001-canary.json"
        decision1 = workspace / "decision-001-canary.json"
        shadow_pass = advance_governor(policy, state0, shadow_batch, state1, decision1)
        require(shadow_pass["outcome"] == "shadow_pass", "fixture did not produce shadow pass")
        registered1 = record(policy, state1, decision1, ledger, sha(state0))
        require(registered1["sequence"] == 1 and registered1["phase"] == "canary" and
                registered1["traffic_directive"]["mode"] == "bounded-canary",
                "shadow pass was not atomically accepted as canary head")
        second_ledger_sha = sha(ledger)

        # An advanced decision must bind the same redacted observation-batch
        # identity which the immutable state consumed.  This prevents a valid
        # state from being paired with a decision that claims unrelated input.
        provenance_ledger = workspace / "provenance-ledger.jsonl"
        record(policy, state0, decision0, provenance_ledger, "GENESIS")
        forged_batch_decision = workspace / "forged-batch-decision.json"
        forged_batch_mapping = json.loads(decision1.read_text(encoding="utf-8"))
        forged_batch_mapping["dependencies"]["observation_batch"] = {
            "path": "<redacted>", "sha256": "0" * 64, "bytes": 1,
        }
        write_json(forged_batch_decision, forged_batch_mapping)
        forged_batch = run(registry(
            "record", "--policy", policy, "--state", state1, "--decision", forged_batch_decision,
            "--ledger", provenance_ledger, "--expected-state-sha256", sha(state0),
        ), expected=2)
        require("observation batch" in forged_batch.stderr,
                "registry accepted a decision with a mismatched consumed observation batch")
        provenance_inspection = parsed_stdout(run(
            registry("inspect", "--policy", policy, "--ledger", provenance_ledger), expected=0,
        ))
        require(provenance_inspection["events"] == 1 and provenance_inspection["head_state_sha256"] == sha(state0),
                "rejected observation-batch mismatch changed authoritative head")

        # A separately built valid child of state0 must not race the already
        # registered state1 child through the local compare-and-swap boundary.
        competing_batch = workspace / "shadow-competing.jsonl"
        write_batch(
            competing_batch, sha(policy), state0_mapping["baseline_identity_sha256"], state0_mapping["candidate_identity_sha256"],
            stage="shadow", count=40, candidate_health=1.0, candidate_cost=151.0, candidate_latency=150.0,
        )
        competing_state = workspace / "state-001-competing.json"
        competing_decision = workspace / "decision-001-competing.json"
        advance_governor(policy, state0, competing_batch, competing_state, competing_decision)
        competing = run(registry(
            "record", "--policy", policy, "--state", competing_state, "--decision", competing_decision,
            "--ledger", ledger, "--expected-state-sha256", sha(state0),
        ), expected=2)
        require("head conflict" in competing.stderr, "registry accepted a competing stale child state")
        require(sha(ledger) == second_ledger_sha, "competing branch changed authoritative ledger")

        canary_batch = workspace / "canary-pass.jsonl"
        write_batch(
            canary_batch, sha(policy), state0_mapping["baseline_identity_sha256"], state0_mapping["candidate_identity_sha256"],
            stage="canary", count=40, candidate_health=1.0, candidate_cost=150.0, candidate_latency=150.0,
        )
        state2 = workspace / "state-002-promoted.json"
        decision2 = workspace / "decision-002-promoted.json"
        promoted = advance_governor(policy, state1, canary_batch, state2, decision2)
        require(promoted["outcome"] == "promote", "fixture did not produce conservative promotion")
        registered2 = record(policy, state2, decision2, ledger, sha(state1))
        require(registered2["phase"] == "promoted" and
                registered2["traffic_directive"]["candidate_basis_points"] == 10_000,
                "authoritative registry did not retain promoted directive")
        inspected2 = parsed_stdout(run(registry("inspect", "--policy", policy, "--ledger", ledger), expected=0))
        require(inspected2["events"] == 3 and inspected2["outcome"] == "promote" and
                inspected2["head_state_sha256"] == sha(state2),
                "authoritative promoted head inspection is inconsistent")

        # Exercise the other terminal transition in a separate ledger.  A
        # confident canary regression must serialize to baseline-only rollback,
        # not bypass the authoritative predecessor/CAS contract.
        rollback_ledger = workspace / "rollback-ledger.jsonl"
        record(policy, state0, decision0, rollback_ledger, "GENESIS")
        record(policy, state1, decision1, rollback_ledger, sha(state0))
        rollback_batch = workspace / "canary-rollback.jsonl"
        write_batch(
            rollback_batch, sha(policy), state0_mapping["baseline_identity_sha256"],
            state0_mapping["candidate_identity_sha256"], stage="canary", count=800,
            candidate_health=0.95, candidate_cost=300.0, candidate_latency=300.0,
        )
        rollback_state = workspace / "state-002-rolled-back.json"
        rollback_decision = workspace / "decision-002-rolled-back.json"
        rolled_back = advance_governor(policy, state1, rollback_batch, rollback_state, rollback_decision)
        require(rolled_back["outcome"] == "rollback", "fixture did not produce confident canary rollback")
        registered_rollback = record(policy, rollback_state, rollback_decision, rollback_ledger, sha(state1))
        require(registered_rollback["phase"] == "rolled_back" and
                registered_rollback["traffic_directive"]["mode"] == "baseline-only-rollback",
                "authoritative registry did not retain rollback directive")

        # A forged decision cannot smuggle a traffic directive into a fresh
        # ledger even when it references a valid governor state.
        forged_ledger = workspace / "forged-ledger.jsonl"
        forged_decision = workspace / "forged-decision.json"
        forged_mapping = json.loads(decision0.read_text(encoding="utf-8"))
        forged_mapping["traffic_directive"]["shadow_candidate_basis_points"] = 999
        write_json(forged_decision, forged_mapping)
        forged = run(registry(
            "record", "--policy", policy, "--state", state0, "--decision", forged_decision,
            "--ledger", forged_ledger, "--expected-state-sha256", "GENESIS",
        ), expected=2)
        require("traffic directive" in forged.stderr, "registry accepted forged traffic directive")
        require(not forged_ledger.exists(), "forged decision created an authoritative ledger")

        # A stale local lock is never silently removed by a second writer.
        locked_ledger = workspace / "locked-ledger.jsonl"
        lock = locked_ledger.with_name(f".{locked_ledger.name}.lock")
        lock.write_text("pid=manual-review\n", encoding="ascii", newline="\n")
        locked = run(registry(
            "record", "--policy", policy, "--state", state0, "--decision", decision0,
            "--ledger", locked_ledger, "--expected-state-sha256", "GENESIS", "--lock-timeout-seconds", "0",
        ), expected=2)
        require("timed out waiting for release registry lock" in locked.stderr,
                "registry silently removed an existing lock")
        require(lock.exists() and not locked_ledger.exists(), "lock failure changed registry files")
        lock.unlink()

        # Hash-chain corruption is rejected by inspection before a controller
        # receives a supposed authoritative head.
        clean_ledger = workspace / "clean-authoritative-release.jsonl"
        clean_text = ledger.read_text(encoding="utf-8")
        clean_ledger.write_text(clean_text, encoding="utf-8", newline="\n")
        crlf_ledger = workspace / "crlf-authoritative-release.jsonl"
        crlf_ledger.write_bytes(clean_text.replace("\n", "\r\n").encode("utf-8"))
        crlf = run(registry("inspect", "--policy", policy, "--ledger", crlf_ledger), expected=2)
        require("CRLF" in crlf.stderr, "registry accepted CRLF ledger records")
        noncanonical_ledger = workspace / "noncanonical-authoritative-release.jsonl"
        noncanonical = clean_text.replace(',"sequence":', ', "sequence":', 1)
        require(noncanonical != clean_text, "noncanonical fixture did not mutate ledger")
        noncanonical_ledger.write_text(noncanonical, encoding="utf-8", newline="\n")
        noncanonical_result = run(registry("inspect", "--policy", policy, "--ledger", noncanonical_ledger), expected=2)
        require("canonical JSON" in noncanonical_result.stderr,
                "registry accepted a noncanonical but semantically identical ledger event")
        corrupted = clean_text.replace('"phase":"promoted"', '"phase":"canary"', 1)
        require(corrupted != clean_text, "corruption fixture did not mutate ledger")
        ledger.write_text(corrupted, encoding="utf-8", newline="\n")
        corruption = run(registry("inspect", "--policy", policy, "--ledger", ledger), expected=2)
        require("event_sha256" in corruption.stderr, "hash-chain-corrupted ledger was accepted")

        # A hash chain is deliberately not presented as a signature.  Even if a
        # writer recomputes the final digest after a semantic mutation, inspect
        # must still reject a phase/outcome/directive contradiction.
        semantic_ledger = workspace / "rehashed-semantic-corruption.jsonl"
        semantic_events = [json.loads(line) for line in clean_text.splitlines()]
        semantic_events[-1]["outcome"] = "canary_pending"
        semantic_payload = dict(semantic_events[-1])
        del semantic_payload["event_sha256"]
        semantic_events[-1]["event_sha256"] = hashlib.sha256(canonical(semantic_payload)).hexdigest()
        semantic_ledger.write_bytes(b"".join(canonical(event) + b"\n" for event in semantic_events))
        semantic = run(registry("inspect", "--policy", policy, "--ledger", semantic_ledger), expected=2)
        require("canary outcome is inconsistent" in semantic.stderr,
                "registry accepted a rehashed but semantically inconsistent transition")

        wrong_policy = workspace / "wrong-policy.json"
        wrong_policy_mapping = policy_mapping(portfolio, task)
        wrong_policy_mapping["id"] = "other-release"
        write_json(wrong_policy, wrong_policy_mapping)
        policy_mismatch = run(registry("inspect", "--policy", wrong_policy, "--ledger", clean_ledger), expected=2)
        require("policy identity" in policy_mismatch.stderr or "release_id" in policy_mismatch.stderr,
                "registry accepted an unrelated immutable policy")

    print("LMM release registry verification: PASS (authoritative CAS head, semantic transition, batch/directive binding, strict-LF privacy, lock and tamper gates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
