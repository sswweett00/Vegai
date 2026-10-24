#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synthetic contract verification for immutable LMM canary release governance.

This test builds two local fake model packages and portfolio/evaluation metadata.
The model files, reports, and telemetry values are protocol fixtures only. They
do not represent an actual model quality measurement or a claim of production
readiness. The verifier exercises immutable portfolio binding, state chaining,
time-uniform confidence decisions, promotion, rollback, privacy, replay,
tamper, terminal-state, and no-replace gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

from lmm_contract import canonical_json_bytes, file_identity
from lmm_release_governor import MetricPolicy, confidence_radius


ROOT = Path(__file__).resolve().parents[1]
GOVERNOR = ROOT / "tools" / "lmm_release_governor.py"
GATEWAY = ROOT / "tools" / "lmm_gateway.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical(value) + b"\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def run(command: list[str], expected: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"command exit {result.returncode}, expected {expected}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def model_manifest(model_id: str, revision: str) -> dict[str, object]:
    return {
        "schema_version": "vegai-lmm-model/v1",
        "model_id": model_id,
        "engine": "worker",
        "context_window_tokens": 4096,
        "max_output_tokens": 512,
        "message_overhead_tokens": 4,
        "reply_primer_tokens": 2,
        "capabilities": {"text": True, "image": False, "audio": False, "video": False},
        "tokenizer": {"kind": "utf8-byte"},
        "artifacts": [],
        "prompt_format": "chatml-v1",
        "source": "synthetic release governor contract fixture",
        "license": "MIT",
        "revision": revision,
    }


def runtime_identity(model: Path, worker_config: Path) -> str:
    stable = {
        "gateway_tool": file_identity(GATEWAY, "gateway").to_mapping(),
        "model_manifest": file_identity(model, "model").to_mapping(),
        "tokenizer": None,
        "asset_registry": None,
        "worker_command_configuration": file_identity(worker_config, "worker config").to_mapping(),
        "structured_response_contract": None,
        "model_artifacts": {},
    }
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def quality_report(model_sha256: str, command_sha256: str, runtime_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "vegai-lmm-quality-report/v1",
        "dependencies": {"gateway_command": {"sha256": command_sha256}},
        "configuration": {
            "run_provenance": {
                "model_manifest_sha256": model_sha256,
                "runtime_identity_sha256": runtime_sha256,
                "backend": "synthetic-worker",
            }
        },
        "quality_gate": {"passed": True},
    }


def evaluation_card(card_id: str, model_sha: str, quality_sha: str, quality: float) -> dict[str, object]:
    return {
        "schema_version": "vegai-lmm-evaluation-card/v1",
        "id": card_id,
        "cost_unit": "synthetic-microunits",
        "model_manifest_sha256": model_sha,
        "quality_report_sha256": quality_sha,
        "measurements": [
            {
                "domain": "coding",
                "suite_id": "synthetic-release-heldout",
                "suite_report_sha256": hash_text(card_id + "-suite"),
                "cases": 250,
                "quality_lower_bound": quality,
                "reliability_lower_bound": 0.90,
                "safety_lower_bound": 0.95,
                "p95_latency_ms": 220,
                "cost_microunits_per_1k_output_tokens": 40,
            }
        ],
    }


def routing_task() -> dict[str, object]:
    return {
        "schema_version": "vegai-lmm-routing-task/v1",
        "id": "release-coding-task",
        "domain": "coding",
        "data_residency": "TR",
        "minimum_safety_tier": 2,
        "strategy": "single-best",
        "minimum_independent_candidates": 1,
        "required_modalities": ["text"],
        "require_evidence": True,
        "minimum_context_window_tokens": 1024,
        "minimum_quality_lower_bound": 0.80,
        "minimum_reliability_lower_bound": 0.85,
        "minimum_safety_lower_bound": 0.90,
        "max_p95_latency_ms": 500,
        "max_cost_microunits_per_1k_output_tokens": 100,
    }


def make_portfolio_fixture(workspace: Path) -> tuple[Path, Path, Path]:
    worker_config = workspace / "worker-command.json"
    write_json(worker_config, [sys.executable, "-c", "import sys; sys.exit(0)"])
    task_path = workspace / "task.json"
    write_json(task_path, routing_task())

    candidates: list[dict[str, object]] = []
    for name, quality in (("baseline", 0.88), ("candidate", 0.93)):
        model = workspace / f"{name}-model.json"
        write_json(model, model_manifest(f"{name}-model", f"{name}-r1"))
        command = workspace / f"{name}-gateway.json"
        write_json(
            command,
            [
                sys.executable,
                str(GATEWAY),
                "--model", str(model),
                "--request", "{request}",
                "--evidence", "{evidence}",
                "--worker-command-json", str(worker_config),
                "--grounding-mode", "evidence-or-abstain",
                "--print-json",
            ],
        )
        report = workspace / f"{name}-quality.json"
        write_json(
            report,
            quality_report(
                sha(model),
                sha(command),
                runtime_identity(model, worker_config),
            ),
        )
        card = workspace / f"{name}-card.json"
        write_json(card, evaluation_card(f"{name}-card", sha(model), sha(report), quality))
        candidates.append(
            {
                "id": f"{name}-candidate",
                "domains": ["coding"],
                "independence_group": f"{name}-group",
                "data_residencies": ["TR"],
                "safety_tier": 3,
                "grounding_mode": "evidence-or-abstain",
                "model_manifest": model.name,
                "gateway_command_json": command.name,
                "evaluation_card": card.name,
                "quality_report": report.name,
            }
        )
    portfolio = workspace / "portfolio.json"
    write_json(
        portfolio,
        {
            "schema_version": "vegai-lmm-portfolio/v1",
            "id": "synthetic-release-portfolio",
            "cost_unit": "synthetic-microunits",
            "candidates": candidates,
        },
    )
    return portfolio, task_path, workspace / "candidate-model.json"


def policy_mapping(portfolio: Path, task: Path) -> dict[str, object]:
    health = {
        "direction": "maximize",
        "lower_bound": 0.95,
        "upper_bound": 1.0,
        "promotion_limit": 0.97,
        "rollback_limit": 0.96,
        "maximum_regression_vs_baseline": 0.03,
    }
    performance = {
        "direction": "minimize",
        "lower_bound": 100.0,
        "upper_bound": 300.0,
        "promotion_limit": 250.0,
        "rollback_limit": 290.0,
        "maximum_regression_vs_baseline": 160.0,
    }
    return {
        "schema_version": "vegai-lmm-release-policy/v1",
        "id": "synthetic-coding-release",
        "task": {"path": task.name, "sha256": sha(task)},
        "baseline": {
            "portfolio": {"path": portfolio.name, "sha256": sha(portfolio)},
            "candidate_id": "baseline-candidate",
        },
        "candidate": {
            "portfolio": {"path": portfolio.name, "sha256": sha(portfolio)},
            "candidate_id": "candidate-candidate",
        },
        "shadow_basis_points": 250,
        "shadow_minimum_observations_per_arm": 40,
        "canary_basis_points": 500,
        "minimum_observations_per_arm": 40,
        "confidence_delta": 0.05,
        "metrics": [
            {"id": "quality", "kind": "quality", **health},
            {"id": "reliability", "kind": "reliability", **health},
            {"id": "safety", "kind": "safety", **health},
            {"id": "cost", "kind": "cost", **performance},
            {"id": "p95-latency", "kind": "latency", **performance},
        ],
    }


def observation(
    policy_sha: str,
    arm_identity: str,
    stage: str,
    arm: str,
    index: int,
    health: float,
    cost: float,
    latency: float,
) -> dict[str, object]:
    return {
        "schema_version": "vegai-lmm-release-observation/v1",
        "id": f"{stage}-{arm}-observation-{index}",
        "replication_id": f"{stage}-{arm}-replication-{index}",
        "policy_sha256": policy_sha,
        "stage": stage,
        "arm": arm,
        "arm_identity_sha256": arm_identity,
        "metrics": {
            "quality": health,
            "reliability": health,
            "safety": health,
            "cost": cost,
            "p95-latency": latency,
        },
    }


def write_batch(
    path: Path,
    policy_sha: str,
    baseline_sha: str,
    candidate_sha: str,
    *,
    stage: str,
    count: int,
    candidate_health: float,
    candidate_cost: float,
    candidate_latency: float,
) -> None:
    lines: list[bytes] = []
    for index in range(count):
        lines.append(canonical(observation(policy_sha, baseline_sha, stage, "baseline", index, 1.0, 150.0, 150.0)) + b"\n")
    for index in range(count):
        lines.append(canonical(observation(
            policy_sha, candidate_sha, stage, "candidate", index,
            candidate_health, candidate_cost, candidate_latency,
        )) + b"\n")
    path.write_bytes(b"".join(lines))


def governor(*arguments: str) -> list[str]:
    return [sys.executable, str(GOVERNOR), *arguments]


def init_release(policy: Path, state: Path, decision: Path) -> subprocess.CompletedProcess[str]:
    return run(
        governor("init", "--policy", str(policy), "--state-output", str(state), "--decision-output", str(decision)),
        expected=0,
    )


def advance_release(policy: Path, state: Path, observations: Path, state_output: Path, decision_output: Path) -> subprocess.CompletedProcess[str]:
    return run(
        governor(
            "advance", "--policy", str(policy), "--state", str(state), "--observations", str(observations),
            "--state-output", str(state_output), "--decision-output", str(decision_output),
        ),
        expected=0,
    )


def main() -> int:
    require(GOVERNOR.is_file(), f"missing {GOVERNOR}")
    # The policy parser permits positive finite subnormal deltas. Validate that
    # the time-uniform Hoeffding sequence handles that legal boundary in the
    # log domain rather than overflowing its intermediate union-bound product.
    extreme_metric = MetricPolicy(
        metric_id="bounded",
        kind="custom",
        direction="maximize",
        lower_bound=0.0,
        upper_bound=1.0,
        promotion_limit=0.0,
        rollback_limit=1.0,
        maximum_regression_vs_baseline=1.0,
    )
    extreme_policy = SimpleNamespace(
        metrics=(extreme_metric,),
        confidence_delta=float.fromhex("0x0.0000000000001p-1022"),
    )
    extreme_radius = confidence_radius(extreme_policy, extreme_metric, 1)
    require(math.isfinite(extreme_radius) and extreme_radius > 0.0,
            "subnormal release confidence delta keeps a finite radius")
    ordinary_policy = SimpleNamespace(metrics=(extreme_metric,), confidence_delta=0.05)
    ordinary_radius = confidence_radius(ordinary_policy, extreme_metric, 128)
    ordinary_reference = math.sqrt(math.log((2.0 * 2.0 * 1.0 * 128.0 * 129.0) / 0.05) / (2.0 * 128.0))
    require(math.isclose(ordinary_radius, ordinary_reference, rel_tol=1.0e-15, abs_tol=0.0),
            "log-domain release confidence radius preserves the Hoeffding formula")

    with tempfile.TemporaryDirectory(prefix="vegai-release-governor-") as temporary_name:
        workspace = Path(temporary_name)
        portfolio, task, _ = make_portfolio_fixture(workspace)
        policy = workspace / "policy.json"
        write_json(policy, policy_mapping(portfolio, task))

        initial = workspace / "state-000.json"
        initial_decision = workspace / "decision-000.json"
        init = init_release(policy, initial, initial_decision)
        require('"phase": "shadow"' in init.stdout, "initial release is not in shadow phase")
        require('"outcome": "shadow_insufficient_evidence"' in init.stdout, "initial release did not fail closed to shadow")
        initial_state = json.loads(initial.read_text(encoding="utf-8"))
        initial_decision_mapping = json.loads(initial_decision.read_text(encoding="utf-8"))
        require(initial_state["previous_state_sha256"] == "GENESIS", "initial state has wrong chain head")
        require(initial_state["phase"] == "shadow" and initial_state["active_stage"] == "shadow",
                "initial state phase/active stage is wrong")
        require(initial_decision_mapping["dependencies"]["governor"] == {
            "path": "<redacted>", "sha256": sha(GOVERNOR), "bytes": GOVERNOR.stat().st_size,
        }, "initial decision did not bind/redact the governor implementation identity")
        require(initial_decision_mapping["traffic_directive"] == {
            "mode": "shadow-mirror-only",
            "candidate_basis_points": 0,
            "baseline_basis_points": 10000,
            "shadow_candidate_basis_points": 250,
            "enforcement": "external-deployment-platform-required",
        }, "initial decision did not emit a bounded shadow-only directive")

        collision = run(
            governor("init", "--policy", str(policy), "--state-output", str(initial),
                     "--decision-output", str(workspace / "collision-decision.json")),
            expected=2,
        )
        require("already exists" in collision.stderr, "initial state overwrite was accepted")

        shadow_batch = workspace / "shadow-observations.jsonl"
        write_batch(
            shadow_batch, sha(policy), initial_state["baseline_identity_sha256"], initial_state["candidate_identity_sha256"],
            stage="shadow", count=40, candidate_health=1.0, candidate_cost=150.0, candidate_latency=150.0,
        )
        canary_initial_state = workspace / "state-001-canary.json"
        shadow_pass_decision = workspace / "decision-shadow-pass.json"
        shadow_pass = advance_release(policy, initial, shadow_batch, canary_initial_state, shadow_pass_decision)
        require('"outcome": "shadow_pass"' in shadow_pass.stdout, "conservative shadow gate did not pass")
        canary_initial_mapping = json.loads(canary_initial_state.read_text(encoding="utf-8"))
        shadow_pass_mapping = json.loads(shadow_pass_decision.read_text(encoding="utf-8"))
        require(canary_initial_mapping["phase"] == "canary" and canary_initial_mapping["active_stage"] == "canary",
                "passed shadow did not enter canary phase")
        require(shadow_pass_mapping["traffic_directive"]["mode"] == "bounded-canary" and
                shadow_pass_mapping["traffic_directive"]["candidate_basis_points"] == 500 and
                shadow_pass_mapping["traffic_directive"]["shadow_candidate_basis_points"] == 0,
                "shadow pass did not emit bounded canary directive")
        require(shadow_pass_mapping["state"]["observations"]["shadow"] == {"baseline": 40, "candidate": 40},
                "shadow aggregate observation counts are wrong")

        canary_batch = workspace / "canary-promotion-observations.jsonl"
        write_batch(
            canary_batch, sha(policy), initial_state["baseline_identity_sha256"], initial_state["candidate_identity_sha256"],
            stage="canary", count=40, candidate_health=1.0, candidate_cost=150.0, candidate_latency=150.0,
        )
        promoted_state = workspace / "state-002-promoted.json"
        promote_decision = workspace / "decision-promote.json"
        promoted = advance_release(policy, canary_initial_state, canary_batch, promoted_state, promote_decision)
        require('"outcome": "promote"' in promoted.stdout, "conservative canary promotion did not occur")
        promoted_mapping = json.loads(promoted_state.read_text(encoding="utf-8"))
        decision_mapping = json.loads(promote_decision.read_text(encoding="utf-8"))
        require(promoted_mapping["phase"] == "promoted", "promotion did not produce terminal promoted state")
        require(decision_mapping["traffic_directive"]["candidate_basis_points"] == 10000 and
                decision_mapping["traffic_directive"]["shadow_candidate_basis_points"] == 0,
                "promotion did not emit full candidate traffic directive")
        require(decision_mapping["state"]["observations"]["canary"] == {"baseline": 40, "candidate": 40},
                "canary aggregate observation counts are wrong")
        rendered_decision = json.dumps(decision_mapping, ensure_ascii=False, sort_keys=True)
        require("shadow-baseline-observation-0" not in rendered_decision and "canary-candidate-observation-0" not in rendered_decision,
                "raw observation IDs leaked into production decision")
        require("replication-0" not in rendered_decision, "raw replication IDs leaked into production decision")
        require(decision_mapping["dependencies"]["policy"]["path"] == "<redacted>", "policy path leaked into decision")

        terminal = run(
            governor(
                "advance", "--policy", str(policy), "--state", str(promoted_state), "--observations", str(canary_batch),
                "--state-output", str(workspace / "illegal-terminal-state.json"),
                "--decision-output", str(workspace / "illegal-terminal-decision.json"),
            ),
            expected=2,
        )
        require("terminal promoted/rolled_back" in terminal.stderr, "terminal promoted state accepted another transition")

        inspect_output = workspace / "decision-inspect-promoted.json"
        inspected = run(
            governor("inspect", "--policy", str(policy), "--state", str(promoted_state), "--decision-output", str(inspect_output)),
            expected=0,
        )
        require('"current_evidence_outcome": "promote"' in inspected.stdout, "promoted state inspection is inconsistent")
        require(json.loads(inspect_output.read_text(encoding="utf-8"))["outcome"] == "promote",
                "aggregate-only inspection decision is inconsistent")

        rollback_initial = workspace / "state-rollback-000.json"
        rollback_initial_decision = workspace / "decision-rollback-000.json"
        init_release(policy, rollback_initial, rollback_initial_decision)
        rollback_initial_mapping = json.loads(rollback_initial.read_text(encoding="utf-8"))
        rollback_shadow_batch = workspace / "rollback-shadow-observations.jsonl"
        write_batch(
            rollback_shadow_batch, sha(policy), rollback_initial_mapping["baseline_identity_sha256"],
            rollback_initial_mapping["candidate_identity_sha256"], stage="shadow", count=40,
            candidate_health=1.0, candidate_cost=150.0, candidate_latency=150.0,
        )
        rollback_canary_initial = workspace / "state-rollback-001-canary.json"
        advance_release(policy, rollback_initial, rollback_shadow_batch, rollback_canary_initial,
                        workspace / "decision-rollback-shadow-pass.json")
        rollback_batch = workspace / "rollback-canary-observations.jsonl"
        write_batch(
            rollback_batch, sha(policy), rollback_initial_mapping["baseline_identity_sha256"],
            rollback_initial_mapping["candidate_identity_sha256"], stage="canary", count=800,
            candidate_health=0.95, candidate_cost=300.0, candidate_latency=300.0,
        )
        rollback_state = workspace / "state-rollback-002.json"
        rollback_decision = workspace / "decision-rollback.json"
        rolled_back = advance_release(policy, rollback_canary_initial, rollback_batch, rollback_state, rollback_decision)
        require('"outcome": "rollback"' in rolled_back.stdout, "confident canary rollback did not occur")
        rollback_decision_mapping = json.loads(rollback_decision.read_text(encoding="utf-8"))
        require(rollback_decision_mapping["traffic_directive"]["candidate_basis_points"] == 0 and
                rollback_decision_mapping["traffic_directive"]["shadow_candidate_basis_points"] == 0,
                "rollback did not remove candidate and shadow traffic")

        replay_initial = workspace / "state-replay-000.json"
        init_release(policy, replay_initial, workspace / "decision-replay-000.json")
        replay_mapping = json.loads(replay_initial.read_text(encoding="utf-8"))
        replay_batch = workspace / "replay-shadow-observations.jsonl"
        write_batch(
            replay_batch, sha(policy), replay_mapping["baseline_identity_sha256"], replay_mapping["candidate_identity_sha256"],
            stage="shadow", count=20, candidate_health=1.0, candidate_cost=150.0, candidate_latency=150.0,
        )
        replay_pending = workspace / "state-replay-001-pending.json"
        replay = advance_release(policy, replay_initial, replay_batch, replay_pending, workspace / "decision-replay-001.json")
        require('"outcome": "shadow_insufficient_evidence"' in replay.stdout, "replay fixture did not stay shadow-pending")
        replay_again = run(
            governor(
                "advance", "--policy", str(policy), "--state", str(replay_pending), "--observations", str(replay_batch),
                "--state-output", str(workspace / "state-replay-002.json"),
                "--decision-output", str(workspace / "decision-replay-002.json"),
            ),
            expected=2,
        )
        require("already consumed" in replay_again.stderr, "replayed telemetry batch was accepted")

        malformed = workspace / "malformed-observations.jsonl"
        malformed.write_text(
            '{"schema_version":"vegai-lmm-release-observation/v1","schema_version":"vegai-lmm-release-observation/v1"}\n',
            encoding="utf-8", newline="\n",
        )
        malformed_failure = run(
            governor(
                "advance", "--policy", str(policy), "--state", str(initial), "--observations", str(malformed),
                "--state-output", str(workspace / "malformed-state.json"), "--decision-output", str(workspace / "malformed-decision.json"),
            ), expected=2,
        )
        require("duplicate key" in malformed_failure.stderr, "duplicate telemetry JSON key was accepted")

        forged_batch = workspace / "forged-observations.jsonl"
        forged = observation(sha(policy), initial_state["baseline_identity_sha256"], "shadow", "baseline", 999, 1.0, 150.0, 150.0)
        forged["arm_identity_sha256"] = "0" * 64
        forged_batch.write_bytes(canonical(forged) + b"\n")
        forged_failure = run(
            governor(
                "advance", "--policy", str(policy), "--state", str(initial), "--observations", str(forged_batch),
                "--state-output", str(workspace / "forged-state.json"), "--decision-output", str(workspace / "forged-decision.json"),
            ), expected=2,
        )
        require("does not match the declared baseline deployment" in forged_failure.stderr,
                "forged arm deployment identity was accepted")

        wrong_policy_batch = workspace / "wrong-policy-observations.jsonl"
        wrong_policy = observation("0" * 64, initial_state["baseline_identity_sha256"], "shadow", "baseline", 1000,
                                   1.0, 150.0, 150.0)
        wrong_policy_batch.write_bytes(canonical(wrong_policy) + b"\n")
        wrong_policy_failure = run(
            governor(
                "advance", "--policy", str(policy), "--state", str(initial), "--observations", str(wrong_policy_batch),
                "--state-output", str(workspace / "wrong-policy-state.json"),
                "--decision-output", str(workspace / "wrong-policy-decision.json"),
            ), expected=2,
        )
        require("does not match immutable release policy" in wrong_policy_failure.stderr,
                "telemetry for another immutable policy was accepted")

        impossible_state = workspace / "impossible-promoted-state.json"
        impossible_mapping = json.loads(initial.read_text(encoding="utf-8"))
        impossible_mapping["phase"] = "promoted"
        impossible_mapping["active_stage"] = "canary"
        impossible_mapping["last_decision"] = "promote"
        write_json(impossible_state, impossible_mapping)
        impossible_failure = run(
            governor("inspect", "--policy", str(policy), "--state", str(impossible_state),
                     "--decision-output", str(workspace / "impossible-decision.json")),
            expected=2,
        )
        require("initial release state" in impossible_failure.stderr or "shadow gate" in impossible_failure.stderr or
                "promoted state" in impossible_failure.stderr,
                "evidence-inconsistent terminal state was accepted")

        wrong_governor_state = workspace / "wrong-governor-state.json"
        wrong_governor_mapping = json.loads(initial.read_text(encoding="utf-8"))
        wrong_governor_mapping["governor_sha256"] = "0" * 64
        write_json(wrong_governor_state, wrong_governor_mapping)
        wrong_governor_failure = run(
            governor("inspect", "--policy", str(policy), "--state", str(wrong_governor_state),
                     "--decision-output", str(workspace / "wrong-governor-decision.json")),
            expected=2,
        )
        require("governor implementation identity" in wrong_governor_failure.stderr,
                "state from another governor implementation was accepted")

        no_shadow_policy = workspace / "no-shadow-policy.json"
        no_shadow_policy_mapping = policy_mapping(portfolio, task)
        no_shadow_policy_mapping["id"] = "synthetic-coding-release-no-shadow"
        no_shadow_policy_mapping["shadow_basis_points"] = 0
        no_shadow_policy_mapping["shadow_minimum_observations_per_arm"] = 0
        write_json(no_shadow_policy, no_shadow_policy_mapping)
        no_shadow_state = workspace / "no-shadow-state-000.json"
        no_shadow_decision = workspace / "no-shadow-decision-000.json"
        no_shadow_init = init_release(no_shadow_policy, no_shadow_state, no_shadow_decision)
        require('"phase": "canary"' in no_shadow_init.stdout,
                "shadow-disabled policy did not begin directly in canary")
        no_shadow_decision_mapping = json.loads(no_shadow_decision.read_text(encoding="utf-8"))
        require(no_shadow_decision_mapping["traffic_directive"]["mode"] == "bounded-canary" and
                no_shadow_decision_mapping["traffic_directive"]["candidate_basis_points"] == 500 and
                no_shadow_decision_mapping["traffic_directive"]["shadow_candidate_basis_points"] == 0,
                "shadow-disabled policy did not emit a bounded canary directive")

        alias_portfolio = workspace / "alias-portfolio.json"
        alias_portfolio_mapping = json.loads(portfolio.read_text(encoding="utf-8"))
        baseline_alias = dict(alias_portfolio_mapping["candidates"][0])
        baseline_alias["id"] = "baseline-alias"
        alias_portfolio_mapping["candidates"].append(baseline_alias)
        write_json(alias_portfolio, alias_portfolio_mapping)
        alias_policy = workspace / "alias-policy.json"
        alias_policy_mapping = policy_mapping(alias_portfolio, task)
        alias_policy_mapping["candidate"]["candidate_id"] = "baseline-alias"
        write_json(alias_policy, alias_policy_mapping)
        alias_failure = run(
            governor("init", "--policy", str(alias_policy), "--state-output", str(workspace / "alias-state.json"),
                     "--decision-output", str(workspace / "alias-decision.json")),
            expected=2,
        )
        require("actual immutable model/gateway/runtime deployment identity" in alias_failure.stderr,
                "same baseline deployment under a different candidate label was accepted")

        bad_policy = workspace / "bad-policy.json"
        bad_policy_mapping = policy_mapping(portfolio, task)
        bad_policy_mapping["metrics"] = bad_policy_mapping["metrics"][:4]
        write_json(bad_policy, bad_policy_mapping)
        bad_policy_failure = run(
            governor("init", "--policy", str(bad_policy), "--state-output", str(workspace / "bad-state.json"),
                     "--decision-output", str(workspace / "bad-decision.json")),
            expected=2,
        )
        require("must include quality, reliability, safety, cost, and latency" in bad_policy_failure.stderr,
                "policy missing a required production metric kind was accepted")

        tampered_model = workspace / "candidate-model.json"
        tampered_model.write_text('{"tampered":true}\n', encoding="utf-8", newline="\n")
        tamper_failure = run(
            governor("inspect", "--policy", str(policy), "--state", str(initial),
                     "--decision-output", str(workspace / "tampered-decision.json")),
            expected=2,
        )
        require("does not match" in tamper_failure.stderr or "unsupported" in tamper_failure.stderr,
                "tampered immutable candidate manifest was accepted")

    print("LMM release governor verification: PASS (immutable tool/portfolio provenance, shadow/canary gates, promotion, rollback, privacy, replay/tamper and policy-completeness gates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
