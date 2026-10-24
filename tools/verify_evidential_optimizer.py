#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""End-to-end verification for the evidence-gated algorithm/optimizer selector.

Fixtures are ephemeral, synthetic protocol values only.  They neither train a
model nor encode prompts, answers, language facts, or production capability
claims.  The test checks provenance binding, confidence-gated multi-objective
selection, deterministic reports, no-replace publication, and negative gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "evidential_optimizer.py"
sys.path.insert(0, str(ROOT / "tools"))
from evidential_optimizer import MetricPolicy, hoeffding_radius


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(
        completed.returncode == expected,
        f"expected exit {expected}, got {completed.returncode}: {' '.join(command)}\n{completed.stdout}",
    )
    return completed


def benchmark() -> dict[str, str]:
    return {
        "suite_id": "synthetic-heldout-suite",
        "suite_revision_sha256": digest("suite revision"),
        "protocol_sha256": digest("measurement protocol"),
        "split": "heldout",
    }


def candidate_identity(label: str) -> dict[str, str]:
    return {
        "algorithm_config_sha256": digest(label + " algorithm config"),
        "optimizer_config_sha256": digest(label + " optimizer config"),
        "training_contract_sha256": digest(label + " training contract"),
        "model_artifact_sha256": digest(label + " model artifact"),
    }


def policy() -> dict[str, Any]:
    return {
        "schema_version": "vegai-evidential-optimizer-policy/v1",
        "id": "synthetic-optimizer-policy",
        "benchmark": benchmark(),
        "baseline_candidate_id": "baseline-system",
        "confidence_delta": 0.05,
        "minimum_observations_per_candidate": 128,
        "selection_strategy": "robust-pareto-lexicographic/v1",
        "metrics": [
            {
                "id": "quality",
                "direction": "maximize",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "acceptance_limit": 0.7,
                "minimum_improvement_vs_baseline": 0.05,
                "dominance_margin": 0.0,
            },
            {
                "id": "latency_ms",
                "direction": "minimize",
                "lower_bound": 1.0,
                "upper_bound": 1000.0,
                "acceptance_limit": 300.0,
                "minimum_improvement_vs_baseline": 10.0,
                "dominance_margin": 0.0,
            },
        ],
        "priority_metrics": ["quality", "latency_ms"],
    }


def candidates() -> dict[str, Any]:
    strong = candidate_identity("adaptive-evidence")
    baseline = candidate_identity("baseline")
    return {
        "schema_version": "vegai-evidential-optimizer-candidates/v1",
        "id": "synthetic-algorithm-optimizer-candidates",
        "benchmark": benchmark(),
        "candidates": [
            {
                "id": "baseline-system",
                "algorithm": {"family": "baseline-architecture", "config_sha256": baseline["algorithm_config_sha256"]},
                "optimizer": {"family": "baseline-optimizer", "config_sha256": baseline["optimizer_config_sha256"]},
                "training_contract_sha256": baseline["training_contract_sha256"],
                "model_artifact_sha256": baseline["model_artifact_sha256"],
            },
            {
                "id": "adaptive-evidence-engine",
                "algorithm": {"family": "evidence-routed-hybrid", "config_sha256": strong["algorithm_config_sha256"]},
                "optimizer": {"family": "trust-region-meta-optimizer", "config_sha256": strong["optimizer_config_sha256"]},
                "training_contract_sha256": strong["training_contract_sha256"],
                "model_artifact_sha256": strong["model_artifact_sha256"],
            },
        ],
    }


def observation(candidate_id: str, identity: dict[str, str], index: int, quality: float, latency: float) -> dict[str, Any]:
    return {
        "schema_version": "vegai-evidential-optimizer-observation/v1",
        "candidate_id": candidate_id,
        "run_id": f"{candidate_id}-run-{index:03d}",
        "replication_id": f"replication-{index:03d}",
        "benchmark": benchmark(),
        "candidate_identity": identity,
        "execution_report_sha256": digest(f"{candidate_id} execution {index}"),
        "metrics": {"quality": quality, "latency_ms": latency},
    }


def write_observations(path: Path, *, strong_quality: float = 0.92, strong_latency: float = 105.0) -> None:
    baseline = candidate_identity("baseline")
    strong = candidate_identity("adaptive-evidence")
    records: list[dict[str, Any]] = []
    for index in range(128):
        # A deterministic two-point variation validates aggregation rather than
        # depending on a single repeated scalar fixture.
        records.append(observation("baseline-system", baseline, index, 0.21 + 0.01 * (index % 2), 800.0 + 5.0 * (index % 2)))
        records.append(observation("adaptive-evidence-engine", strong, index, strong_quality + 0.01 * (index % 2), strong_latency + 5.0 * (index % 2)))
    path.write_text("".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
                    encoding="utf-8", newline="\n")


def command(policy_path: Path, candidates_path: Path, observations_path: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(TOOL),
        "select",
        "--policy",
        str(policy_path),
        "--candidates",
        str(candidates_path),
        "--observations",
        str(observations_path),
        "--output",
        str(output),
    ]


def main() -> int:
    try:
        # A valid subnormal confidence budget used to overflow the intermediate
        # ratio before log() was applied. The log-domain formulation must retain
        # a finite conservative radius rather than rejecting valid policy input.
        extreme_metric = MetricPolicy(
            metric_id="bounded",
            direction="maximize",
            lower_bound=0.0,
            upper_bound=1.0,
            acceptance_limit=0.0,
            minimum_improvement_vs_baseline=0.0,
            dominance_margin=0.0,
        )
        extreme_radius = hoeffding_radius(extreme_metric, 1, 128, 16, float.fromhex("0x0.0000000000001p-1022"))
        require(math.isfinite(extreme_radius) and extreme_radius > 0.0,
                "subnormal optimizer confidence delta keeps a finite radius")
        ordinary_radius = hoeffding_radius(extreme_metric, 128, 2, 1, 0.05)
        ordinary_reference = math.sqrt(math.log((2.0 * 2.0 * 1.0) / 0.05) / (2.0 * 128.0))
        require(math.isclose(ordinary_radius, ordinary_reference, rel_tol=1.0e-15, abs_tol=0.0),
                "log-domain optimizer confidence radius preserves the Hoeffding formula")

        with tempfile.TemporaryDirectory(prefix="vegai-evidential-optimizer-") as temporary_name:
            workspace = Path(temporary_name)
            policy_path = workspace / "policy.json"
            candidates_path = workspace / "candidates.json"
            observations_path = workspace / "observations.jsonl"
            write_json(policy_path, policy())
            write_json(candidates_path, candidates())
            write_observations(observations_path)

            first_output = workspace / "decision-one.json"
            first = run(command(policy_path, candidates_path, observations_path, first_output))
            require(json.loads(first.stdout)["selected"] == "adaptive-evidence-engine", "selector stdout did not identify certified candidate")
            require(first_output.is_file(), "successful selection did not publish decision")
            first_payload = json.loads(first_output.read_text(encoding="utf-8"))
            require(first_payload["schema_version"] == "vegai-evidential-optimizer-decision/v1", "wrong decision schema")
            require(first_payload["outcome"] == "selected", "decision did not report selected outcome")
            require(first_payload["selected"]["id"] == "adaptive-evidence-engine", "wrong candidate selected")
            require(first_payload["certified_robust_pareto_frontier"] == ["adaptive-evidence-engine"], "wrong robust frontier")
            require(
                first_payload["selected"]["identity"]["optimizer"]["family"] == "trust-region-meta-optimizer",
                "decision did not bind selected optimizer identity",
            )
            selected_candidate = next(item for item in first_payload["candidates"] if item["id"] == "adaptive-evidence-engine")
            require(selected_candidate["certified_improvement_over_baseline"], "selected candidate lacks certified baseline improvement")
            require(selected_candidate["metrics"]["quality"]["passes_acceptance_limit"], "quality lower bound did not pass gate")
            require(selected_candidate["metrics"]["latency_ms"]["passes_acceptance_limit"], "latency upper bound did not pass gate")
            # The source observation IDs and metric samples remain private; only
            # aggregate evidence and immutable file identities are published.
            rendered = first_output.read_text(encoding="utf-8")
            require("adaptive-evidence-engine-run-000" not in rendered, "decision leaked raw run identifiers")
            require("replication-000" not in rendered, "decision leaked raw replication identifiers")

            second_output = workspace / "decision-two.json"
            run(command(policy_path, candidates_path, observations_path, second_output))
            require(first_output.read_bytes() == second_output.read_bytes(), "identical evidence did not produce byte-identical decision reports")

            collision = run(command(policy_path, candidates_path, observations_path, first_output), expected=2)
            require("refusing to overwrite" in collision.stdout, "selector overwrote existing immutable decision")

            mismatched_records = observations_path.read_text(encoding="utf-8").splitlines()
            altered = json.loads(mismatched_records[0])
            altered["candidate_identity"]["optimizer_config_sha256"] = digest("forged optimizer")
            mismatched_records[0] = json.dumps(altered, sort_keys=True, separators=(",", ":"))
            mismatch_path = workspace / "mismatched-observations.jsonl"
            mismatch_path.write_text("\n".join(mismatched_records) + "\n", encoding="utf-8", newline="\n")
            mismatch = run(command(policy_path, candidates_path, mismatch_path, workspace / "mismatch.json"), expected=2)
            require("does not match candidate manifest" in mismatch.stdout, "forged candidate identity was accepted")

            duplicate_policy = workspace / "duplicate-policy.json"
            duplicate_policy.write_text(
                '{"schema_version":"vegai-evidential-optimizer-policy/v1",'
                '"schema_version":"vegai-evidential-optimizer-policy/v1"}\n',
                encoding="utf-8",
                newline="\n",
            )
            duplicate = run(command(duplicate_policy, candidates_path, observations_path, workspace / "duplicate.json"), expected=2)
            require("duplicate key" in duplicate.stdout, "duplicate JSON object key was accepted")

            partial_records = observations_path.read_text(encoding="utf-8").splitlines()
            partial_records.pop()
            partial_path = workspace / "partial-observations.jsonl"
            partial_path.write_text("\n".join(partial_records) + "\n", encoding="utf-8", newline="\n")
            partial_output = workspace / "partial-must-not-publish.json"
            partial = run(command(policy_path, candidates_path, partial_path, partial_output), expected=3)
            require("every declared candidate requires independent held-out evidence" in partial.stdout,
                    "selector ignored a candidate with insufficient observations")
            require(not partial_output.exists(), "incomplete candidate evidence published a decision")

            close_path = workspace / "close-observations.jsonl"
            write_observations(close_path, strong_quality=0.30, strong_latency=800.0)
            gate_output = workspace / "must-not-publish.json"
            gate = run(command(policy_path, candidates_path, close_path, gate_output), expected=3)
            require("GATE FAILED" in gate.stdout, "non-dominating evidence did not fail gate")
            require(not gate_output.exists(), "failed gate published a deployment decision")
    except (AssertionError, OSError, json.JSONDecodeError) as error:
        print(f"evidential optimizer verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("evidential optimizer verification: PASS (provenance, confidence bounds, Pareto gate, no-replace publication)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
