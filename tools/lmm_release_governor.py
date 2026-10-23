#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Govern a production LMM shadow/canary release from evidence-bound telemetry.

A model that is competitive in a target domain needs more than a successful
offline gateway call: immutable model/evaluation provenance, consented shadow
checks, a bounded canary, and an auditable promotion/rollback decision. This
stdlib control plane binds an already-qualified portfolio candidate and baseline
to a routing task, consumes privacy-conscious aggregate observations, and emits
an immutable hash-chained release state plus a traffic directive.

It neither trains a foundation model, routes user traffic, fabricates telemetry,
nor declares broad intelligence. A deployment platform must enforce the emitted
traffic directive, isolate tenants/secrets/network, collect trustworthy metrics,
and perform real held-out/red-team evaluation. Any malformed provenance,
telemetry, state transition, or inconclusive evidence fails closed to the
existing rollout stage rather than silently promoting a candidate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from lmm_contract import (
    MAX_JSON_BYTES,
    FileIdentity,
    LMMContractError,
    canonical_json_bytes,
    file_identity,
    require,
    require_exact_fields,
    require_unchanged,
    sha256_bytes,
    validate_identifier,
    validate_sha256,
    validate_unicode_text,
)
from lmm_portfolio import CandidateRuntime, RoutingTask, load_candidate_runtime, parse_portfolio, parse_task


POLICY_SCHEMA = "vegai-lmm-release-policy/v1"
OBSERVATION_SCHEMA = "vegai-lmm-release-observation/v1"
STATE_SCHEMA = "vegai-lmm-release-state/v1"
DECISION_SCHEMA = "vegai-lmm-release-decision/v1"
GENESIS = "GENESIS"

MAX_METRICS = 16
MAX_OBSERVATIONS_PER_BATCH = 10_000
MAX_TOTAL_OBSERVATIONS = 50_000
MAX_OBSERVATION_BATCH_BYTES = 64 * 1024 * 1024
MAX_OBSERVATION_LINE_BYTES = 256 * 1024
MAX_CONSUMED_BATCHES = 1_024
MAX_CANARY_BASIS_POINTS = 5_000
MAX_METRIC_ABSOLUTE_VALUE = 1.0e12
MAX_IDENTIFIER_BYTES = 100
_ARM_NAMES = ("baseline", "candidate")
_STAGES = ("shadow", "canary")
_PHASES = frozenset(("shadow", "canary", "promoted", "rolled_back"))
_DIRECTIONS = frozenset(("maximize", "minimize"))
_METRIC_KINDS = frozenset(("quality", "reliability", "safety", "cost", "latency", "custom"))
_REQUIRED_METRIC_KINDS = frozenset(("quality", "reliability", "safety", "cost", "latency"))


class ReleaseGovernorError(LMMContractError):
    """A release policy, model provenance, telemetry, or state contract failed."""


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    metric_id: str
    kind: str
    direction: str
    lower_bound: float
    upper_bound: float
    promotion_limit: float
    rollback_limit: float
    maximum_regression_vs_baseline: float


@dataclass(frozen=True, slots=True)
class ArmRuntime:
    arm_name: str
    portfolio_identity: FileIdentity
    candidate: CandidateRuntime
    deployment_identity_sha256: str
    identity_sha256: str

    def identity_mapping(self) -> dict[str, object]:
        return {
            "portfolio_sha256": self.portfolio_identity.sha256,
            "candidate_id": self.candidate.spec.candidate_id,
            "model_manifest_sha256": self.candidate.manifest_identity.sha256,
            "gateway_command_sha256": self.candidate.command.identity.sha256,
            "evaluation_card_sha256": self.candidate.card_identity.sha256,
            "heldout_quality_report_sha256": self.candidate.quality_report_identity.sha256,
            "heldout_runtime_identity_sha256": self.candidate.quality_report_runtime_identity_sha256,
            "deployment_identity_sha256": self.deployment_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReleasePolicy:
    policy_id: str
    identity: FileIdentity
    governor_identity: FileIdentity
    task: RoutingTask
    task_identity: FileIdentity
    baseline: ArmRuntime
    candidate: ArmRuntime
    shadow_basis_points: int
    shadow_minimum_observations_per_arm: int
    canary_basis_points: int
    minimum_observations_per_arm: int
    confidence_delta: float
    metrics: tuple[MetricPolicy, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id_sha256: str
    replication_sha256: str
    stage: str
    arm: str
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class ReleaseState:
    release_id: str
    policy_sha256: str
    governor_sha256: str
    task_sha256: str
    baseline_identity_sha256: str
    candidate_identity_sha256: str
    phase: str
    active_stage: str
    sequence: int
    previous_state_sha256: str
    stage_aggregates: dict[str, dict[str, dict[str, object]]]
    seen_observation_id_sha256: tuple[str, ...]
    seen_replication_sha256: tuple[str, ...]
    consumed_batches: tuple[dict[str, object], ...]
    last_decision: str


def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseGovernorError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def parse_strict_json(payload: bytes, label: str) -> object:
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseGovernorError(f"cannot decode {label} as UTF-8: {error}") from error
    try:
        return json.loads(decoded, object_pairs_hook=reject_duplicate_json_keys)
    except json.JSONDecodeError as error:
        raise ReleaseGovernorError(f"cannot parse {label} as JSON: {error}") from error


def load_json_mapping(path: Path, label: str, *, maximum_bytes: int = MAX_JSON_BYTES) -> tuple[dict[str, Any], FileIdentity]:
    identity = file_identity(path, label, maximum_bytes=maximum_bytes)
    try:
        payload = path.resolve().read_bytes()
    except OSError as error:
        raise ReleaseGovernorError(f"cannot read {label} {path}: {error}") from error
    value = parse_strict_json(payload, label)
    require(isinstance(value, Mapping), f"{label} must contain one JSON object")
    require_unchanged(path, identity, label)
    return dict(value), identity


def parse_bounded_int(value: object, field: str, minimum: int, maximum: int) -> int:
    require(type(value) is int and minimum <= value <= maximum,
            f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def parse_finite_number(value: object, field: str) -> float:
    require(type(value) in {int, float} and not isinstance(value, bool), f"{field} must be a finite number")
    parsed = float(value)
    require(math.isfinite(parsed) and abs(parsed) <= MAX_METRIC_ABSOLUTE_VALUE,
            f"{field} must be finite and within ±{MAX_METRIC_ABSOLUTE_VALUE:g}")
    return parsed


def parse_probability(value: object, field: str, *, open_interval: bool = False) -> float:
    parsed = parse_finite_number(value, field)
    if open_interval:
        require(0.0 < parsed < 1.0, f"{field} must be in (0, 1)")
    else:
        require(0.0 <= parsed <= 1.0, f"{field} must be in [0, 1]")
    return parsed


def stable_float(value: float) -> float:
    require(math.isfinite(value), "internal numeric value is not finite")
    return float(format(value, ".17g"))


def resolve_declared_file(value: object, base: Path, field: str, expected_sha256: object) -> FileIdentity:
    rendered = validate_unicode_text(value, field, maximum_bytes=8 * 1024)
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = base / candidate
    require(not candidate.is_symlink(), f"{field} must not be a symlink")
    identity = file_identity(candidate, field, maximum_bytes=MAX_JSON_BYTES)
    require(identity.sha256 == validate_sha256(expected_sha256, f"{field}.sha256"),
            f"{field} does not match its declared SHA-256")
    return identity


def parse_file_reference(value: object, base: Path, field: str) -> FileIdentity:
    require(isinstance(value, Mapping), f"{field} must be an object")
    require_exact_fields(value, {"path", "sha256"}, field)
    return resolve_declared_file(value["path"], base, field, value["sha256"])


def metric_from_mapping(value: object, index: int) -> MetricPolicy:
    field = f"release policy.metrics[{index}]"
    require(isinstance(value, Mapping), f"{field} must be an object")
    require_exact_fields(
        value,
        {
            "id", "kind", "direction", "lower_bound", "upper_bound", "promotion_limit",
            "rollback_limit", "maximum_regression_vs_baseline",
        },
        field,
    )
    kind = value["kind"]
    require(isinstance(kind, str) and kind in _METRIC_KINDS,
            f"{field}.kind must be one of {', '.join(sorted(_METRIC_KINDS))}")
    direction = value["direction"]
    require(isinstance(direction, str) and direction in _DIRECTIONS,
            f"{field}.direction must be one of {', '.join(sorted(_DIRECTIONS))}")
    expected_direction = "maximize" if kind in {"quality", "reliability", "safety"} else "minimize"
    if kind != "custom":
        require(direction == expected_direction,
                f"{field}.direction must be {expected_direction!r} for {kind!r}")
    lower = parse_finite_number(value["lower_bound"], f"{field}.lower_bound")
    upper = parse_finite_number(value["upper_bound"], f"{field}.upper_bound")
    promotion = parse_finite_number(value["promotion_limit"], f"{field}.promotion_limit")
    rollback = parse_finite_number(value["rollback_limit"], f"{field}.rollback_limit")
    regression = parse_finite_number(value["maximum_regression_vs_baseline"], f"{field}.maximum_regression_vs_baseline")
    require(lower < upper, f"{field}.lower_bound must be less than upper_bound")
    require(lower <= promotion <= upper, f"{field}.promotion_limit must be within metric bounds")
    require(lower <= rollback <= upper, f"{field}.rollback_limit must be within metric bounds")
    require(regression >= 0.0 and regression <= upper - lower,
            f"{field}.maximum_regression_vs_baseline must be within metric range")
    if direction == "maximize":
        require(rollback <= promotion,
                f"{field}.rollback_limit must not exceed promotion_limit for a maximize metric")
    else:
        require(rollback >= promotion,
                f"{field}.rollback_limit must not be below promotion_limit for a minimize metric")
    return MetricPolicy(
        metric_id=validate_identifier(value["id"], f"{field}.id"),
        kind=kind,
        direction=direction,
        lower_bound=lower,
        upper_bound=upper,
        promotion_limit=promotion,
        rollback_limit=rollback,
        maximum_regression_vs_baseline=regression,
    )


def validate_candidate_for_task(runtime: CandidateRuntime, task: RoutingTask, arm_name: str) -> None:
    """Apply the static portion of portfolio eligibility before live rollout."""
    prefix = f"release {arm_name} candidate {runtime.spec.candidate_id!r}"
    require(task.strategy == "single-best" and task.minimum_independent_candidates == 1,
            "release policy task must use single-best with one selected candidate")
    require(task.domain in runtime.spec.domains, f"{prefix} does not declare the release task domain")
    require(task.data_residency in runtime.spec.data_residencies,
            f"{prefix} does not declare the release task data residency")
    require(runtime.spec.safety_tier >= task.minimum_safety_tier,
            f"{prefix} safety tier is below the release task requirement")
    supported_modalities = {name for name, supported in runtime.manifest.capabilities.items() if supported}
    require(task.required_modalities.issubset(supported_modalities),
            f"{prefix} does not support all release task modalities")
    require(runtime.manifest.context_window_tokens >= task.minimum_context_window_tokens,
            f"{prefix} context window is below the release task requirement")
    if task.require_evidence:
        require(runtime.spec.grounding_mode != "off", f"{prefix} disables required evidence grounding")
    measurement = runtime.card.measurements.get(task.domain)
    require(measurement is not None, f"{prefix} has no target-domain evaluation measurement")
    assert measurement is not None
    require(measurement.quality_lower_bound >= task.minimum_quality_lower_bound,
            f"{prefix} measured quality lower bound is below the release task requirement")
    require(measurement.reliability_lower_bound >= task.minimum_reliability_lower_bound,
            f"{prefix} measured reliability lower bound is below the release task requirement")
    require(measurement.safety_lower_bound >= task.minimum_safety_lower_bound,
            f"{prefix} measured safety lower bound is below the release task requirement")
    require(measurement.p95_latency_ms <= task.max_p95_latency_ms,
            f"{prefix} measured p95 latency exceeds the release task limit")
    require(measurement.cost_microunits_per_1k_output_tokens <= task.max_cost_microunits_per_1k_output_tokens,
            f"{prefix} measured output-token cost exceeds the release task limit")


def load_arm(value: object, base: Path, task: RoutingTask, arm_name: str) -> ArmRuntime:
    field = f"release policy.{arm_name}"
    require(isinstance(value, Mapping), f"{field} must be an object")
    require_exact_fields(value, {"portfolio", "candidate_id"}, field)
    portfolio_reference = parse_file_reference(value["portfolio"], base, f"{field}.portfolio")
    portfolio, portfolio_identity = parse_portfolio(Path(portfolio_reference.path))
    require(portfolio_identity == portfolio_reference, f"{field}.portfolio changed while being parsed")
    candidate_id = validate_identifier(value["candidate_id"], f"{field}.candidate_id")
    matches = [candidate for candidate in portfolio.candidates if candidate.candidate_id == candidate_id]
    require(len(matches) == 1, f"{field}.candidate_id is absent from its portfolio")
    runtime = load_candidate_runtime(matches[0], portfolio.cost_unit)
    validate_candidate_for_task(runtime, task, arm_name)
    identity_mapping = {
        "portfolio_sha256": portfolio_identity.sha256,
        "candidate_id": runtime.spec.candidate_id,
        "model_manifest_sha256": runtime.manifest_identity.sha256,
        "gateway_command_sha256": runtime.command.identity.sha256,
        "evaluation_card_sha256": runtime.card_identity.sha256,
        "heldout_quality_report_sha256": runtime.quality_report_identity.sha256,
        "heldout_runtime_identity_sha256": runtime.quality_report_runtime_identity_sha256,
    }
    deployment_mapping = {
        "model_manifest_sha256": runtime.manifest_identity.sha256,
        "gateway_command_sha256": runtime.command.identity.sha256,
        "heldout_runtime_identity_sha256": runtime.quality_report_runtime_identity_sha256,
    }
    deployment_identity_sha256 = sha256_bytes(canonical_json_bytes(deployment_mapping))
    identity_mapping["deployment_identity_sha256"] = deployment_identity_sha256
    return ArmRuntime(
        arm_name=arm_name,
        portfolio_identity=portfolio_identity,
        candidate=runtime,
        deployment_identity_sha256=deployment_identity_sha256,
        identity_sha256=sha256_bytes(canonical_json_bytes(identity_mapping)),
    )


def load_policy(path: Path) -> ReleasePolicy:
    mapping, identity = load_json_mapping(path, "release policy")
    expected = {
        "schema_version", "id", "task", "baseline", "candidate", "shadow_basis_points",
        "shadow_minimum_observations_per_arm", "canary_basis_points", "minimum_observations_per_arm",
        "confidence_delta", "metrics",
    }
    require_exact_fields(mapping, expected, "release policy")
    require(mapping["schema_version"] == POLICY_SCHEMA,
            f"unsupported release policy schema_version: {mapping['schema_version']!r}")
    base = Path(identity.path).parent
    task_reference = parse_file_reference(mapping["task"], base, "release policy.task")
    task, task_identity = parse_task(Path(task_reference.path))
    require(task_identity == task_reference, "release policy.task changed while being parsed")
    baseline = load_arm(mapping["baseline"], base, task, "baseline")
    candidate = load_arm(mapping["candidate"], base, task, "candidate")
    require(candidate.identity_sha256 != baseline.identity_sha256,
            "release policy candidate must differ from baseline by immutable arm identity")
    require(candidate.deployment_identity_sha256 != baseline.deployment_identity_sha256,
            "release policy candidate must differ from baseline by actual immutable model/gateway/runtime deployment identity")
    raw_metrics = mapping["metrics"]
    require(isinstance(raw_metrics, list) and raw_metrics, "release policy.metrics must be a nonempty array")
    require(len(raw_metrics) <= MAX_METRICS, f"release policy.metrics exceeds {MAX_METRICS} metrics")
    metrics = tuple(metric_from_mapping(item, index) for index, item in enumerate(raw_metrics))
    metric_ids = [metric.metric_id for metric in metrics]
    require(len(set(metric_ids)) == len(metric_ids), "release policy.metrics contains duplicate IDs")
    metric_kinds = [metric.kind for metric in metrics]
    require(_REQUIRED_METRIC_KINDS.issubset(metric_kinds),
            "release policy.metrics must include quality, reliability, safety, cost, and latency metrics")
    require(all(metric_kinds.count(kind) == 1 for kind in _REQUIRED_METRIC_KINDS),
            "release policy.metrics must contain exactly one metric for each required kind")
    shadow_basis_points = parse_bounded_int(
        mapping["shadow_basis_points"], "release policy.shadow_basis_points", 0, 10_000
    )
    shadow_minimum_observations = parse_bounded_int(
        mapping["shadow_minimum_observations_per_arm"],
        "release policy.shadow_minimum_observations_per_arm", 0, MAX_TOTAL_OBSERVATIONS,
    )
    require((shadow_basis_points == 0) == (shadow_minimum_observations == 0),
            "release policy shadow basis points and shadow minimum observations must either both be zero or both be positive")
    canary_minimum_observations = parse_bounded_int(
        mapping["minimum_observations_per_arm"],
        "release policy.minimum_observations_per_arm",
        1,
        MAX_TOTAL_OBSERVATIONS,
    )
    require(2 * (shadow_minimum_observations + canary_minimum_observations) <= MAX_TOTAL_OBSERVATIONS,
            "release policy shadow plus canary minimum observations exceed the immutable total observation budget")
    return ReleasePolicy(
        policy_id=validate_identifier(mapping["id"], "release policy.id"),
        identity=identity,
        governor_identity=file_identity(Path(__file__), "LMM release governor", maximum_bytes=MAX_JSON_BYTES),
        task=task,
        task_identity=task_identity,
        baseline=baseline,
        candidate=candidate,
        shadow_basis_points=shadow_basis_points,
        shadow_minimum_observations_per_arm=shadow_minimum_observations,
        canary_basis_points=parse_bounded_int(
            mapping["canary_basis_points"], "release policy.canary_basis_points", 1, MAX_CANARY_BASIS_POINTS
        ),
        minimum_observations_per_arm=canary_minimum_observations,
        confidence_delta=parse_probability(mapping["confidence_delta"], "release policy.confidence_delta", open_interval=True),
        metrics=metrics,
    )


def policy_metric_ids(policy: ReleasePolicy) -> set[str]:
    return {metric.metric_id for metric in policy.metrics}


def empty_arm_aggregates(policy: ReleasePolicy) -> dict[str, dict[str, object]]:
    return {
        arm: {
            "observations": 0,
            "metric_sums": {metric.metric_id: 0.0 for metric in policy.metrics},
        }
        for arm in _ARM_NAMES
    }


def initial_stage_aggregates(policy: ReleasePolicy) -> dict[str, dict[str, dict[str, object]]]:
    return {stage: empty_arm_aggregates(policy) for stage in _STAGES}


def initial_stage(policy: ReleasePolicy) -> str:
    return "shadow" if policy.shadow_minimum_observations_per_arm else "canary"


def initial_state_mapping(policy: ReleasePolicy) -> dict[str, object]:
    stage = initial_stage(policy)
    return {
        "schema_version": STATE_SCHEMA,
        "release_id": policy.policy_id,
        "policy_sha256": policy.identity.sha256,
        "governor_sha256": policy.governor_identity.sha256,
        "task_sha256": policy.task_identity.sha256,
        "baseline_identity_sha256": policy.baseline.identity_sha256,
        "candidate_identity_sha256": policy.candidate.identity_sha256,
        "phase": stage,
        "active_stage": stage,
        "sequence": 0,
        "previous_state_sha256": GENESIS,
        "stage_aggregates": initial_stage_aggregates(policy),
        "seen_observation_id_sha256": [],
        "seen_replication_sha256": [],
        "consumed_batches": [],
        "last_decision": "initialized",
    }


def parse_aggregate(value: object, policy: ReleasePolicy, stage: str, arm: str) -> dict[str, object]:
    field = f"release state.stage_aggregates.{stage}.{arm}"
    require(isinstance(value, Mapping), f"{field} must be an object")
    require_exact_fields(value, {"observations", "metric_sums"}, field)
    observations = parse_bounded_int(value["observations"], f"{field}.observations", 0, MAX_TOTAL_OBSERVATIONS)
    sums = value["metric_sums"]
    require(isinstance(sums, Mapping), f"{field}.metric_sums must be an object")
    require(set(sums) == policy_metric_ids(policy), f"{field}.metric_sums does not match immutable policy metrics")
    parsed_sums: dict[str, float] = {}
    for metric in policy.metrics:
        total = parse_finite_number(sums[metric.metric_id], f"{field}.metric_sums.{metric.metric_id}")
        lower_total = observations * metric.lower_bound
        upper_total = observations * metric.upper_bound
        tolerance = max(1.0, abs(lower_total), abs(upper_total)) * 1.0e-10
        require(lower_total - tolerance <= total <= upper_total + tolerance,
                f"{field}.metric_sums.{metric.metric_id} is outside the bounded aggregate range")
        parsed_sums[metric.metric_id] = total
    return {"observations": observations, "metric_sums": parsed_sums}


def parse_stage_aggregates(value: object, policy: ReleasePolicy) -> dict[str, dict[str, dict[str, object]]]:
    require(isinstance(value, Mapping) and set(value) == set(_STAGES),
            "release state.stage_aggregates must contain shadow and canary")
    parsed: dict[str, dict[str, dict[str, object]]] = {}
    for stage in _STAGES:
        arms = value[stage]
        require(isinstance(arms, Mapping) and set(arms) == set(_ARM_NAMES),
                f"release state.stage_aggregates.{stage} must contain baseline and candidate")
        parsed[stage] = {arm: parse_aggregate(arms[arm], policy, stage, arm) for arm in _ARM_NAMES}
    return parsed


def parse_digest_array(value: object, field: str, maximum: int) -> tuple[str, ...]:
    require(isinstance(value, list), f"{field} must be an array")
    require(len(value) <= maximum, f"{field} exceeds {maximum} entries")
    parsed = tuple(validate_sha256(item, f"{field}[{index}]") for index, item in enumerate(value))
    require(tuple(sorted(parsed)) == parsed and len(set(parsed)) == len(parsed),
            f"{field} must be sorted and unique")
    return parsed


def parse_consumed_batches(value: object) -> tuple[dict[str, object], ...]:
    require(isinstance(value, list), "release state.consumed_batches must be an array")
    require(len(value) <= MAX_CONSUMED_BATCHES, f"release state.consumed_batches exceeds {MAX_CONSUMED_BATCHES} entries")
    parsed: list[dict[str, object]] = []
    for index, item in enumerate(value):
        field = f"release state.consumed_batches[{index}]"
        require(isinstance(item, Mapping), f"{field} must be an object")
        require_exact_fields(item, {"stage", "sha256", "bytes"}, field)
        stage = item["stage"]
        require(isinstance(stage, str) and stage in _STAGES, f"{field}.stage must be shadow or canary")
        parsed.append({
            "stage": stage,
            "sha256": validate_sha256(item["sha256"], f"{field}.sha256"),
            "bytes": parse_bounded_int(item["bytes"], f"{field}.bytes", 1, MAX_OBSERVATION_BATCH_BYTES),
        })
    digests = [str(item["sha256"]) for item in parsed]
    require(len(set(digests)) == len(digests), "release state.consumed_batches contains duplicate batch identities")
    return tuple(parsed)


def stage_observation_count(state: ReleaseState, stage: str) -> int:
    return sum(int(state.stage_aggregates[stage][arm]["observations"]) for arm in _ARM_NAMES)


def state_observation_count(state: ReleaseState) -> int:
    return sum(stage_observation_count(state, stage) for stage in _STAGES)


def state_from_mapping(value: object, policy: ReleasePolicy) -> ReleaseState:
    require(isinstance(value, Mapping), "release state must be an object")
    expected = {
        "schema_version", "release_id", "policy_sha256", "governor_sha256", "task_sha256", "baseline_identity_sha256",
        "candidate_identity_sha256", "phase", "active_stage", "sequence", "previous_state_sha256",
        "stage_aggregates", "seen_observation_id_sha256", "seen_replication_sha256", "consumed_batches",
        "last_decision",
    }
    require_exact_fields(value, expected, "release state")
    require(value["schema_version"] == STATE_SCHEMA, "release state has unsupported schema_version")
    release_id = validate_identifier(value["release_id"], "release state.release_id")
    require(release_id == policy.policy_id, "release state.release_id does not match immutable policy")
    policy_sha = validate_sha256(value["policy_sha256"], "release state.policy_sha256")
    governor_sha = validate_sha256(value["governor_sha256"], "release state.governor_sha256")
    task_sha = validate_sha256(value["task_sha256"], "release state.task_sha256")
    baseline_sha = validate_sha256(value["baseline_identity_sha256"], "release state.baseline_identity_sha256")
    candidate_sha = validate_sha256(value["candidate_identity_sha256"], "release state.candidate_identity_sha256")
    require(policy_sha == policy.identity.sha256, "release state policy identity does not match supplied policy")
    require(governor_sha == policy.governor_identity.sha256,
            "release state governor implementation identity does not match supplied governor")
    require(task_sha == policy.task_identity.sha256, "release state task identity does not match supplied policy")
    require(baseline_sha == policy.baseline.identity_sha256, "release state baseline identity does not match supplied policy")
    require(candidate_sha == policy.candidate.identity_sha256, "release state candidate identity does not match supplied policy")
    phase = value["phase"]
    require(isinstance(phase, str) and phase in _PHASES, "release state.phase is unsupported")
    active_stage = value["active_stage"]
    require(isinstance(active_stage, str) and active_stage in _STAGES,
            "release state.active_stage must be shadow or canary")
    sequence = parse_bounded_int(value["sequence"], "release state.sequence", 0, MAX_CONSUMED_BATCHES)
    previous = value["previous_state_sha256"]
    if previous != GENESIS:
        validate_sha256(previous, "release state.previous_state_sha256")
    stage_aggregates = parse_stage_aggregates(value["stage_aggregates"], policy)
    seen_observation = parse_digest_array(value["seen_observation_id_sha256"], "release state.seen_observation_id_sha256", MAX_TOTAL_OBSERVATIONS)
    seen_replication = parse_digest_array(value["seen_replication_sha256"], "release state.seen_replication_sha256", MAX_TOTAL_OBSERVATIONS)
    last_decision = value["last_decision"]
    valid_decisions = {
        "initialized", "shadow_insufficient_evidence", "shadow_pending", "shadow_pass",
        "canary_insufficient_evidence", "canary_pending", "promote", "rollback",
    }
    require(isinstance(last_decision, str) and last_decision in valid_decisions,
            "release state.last_decision is unsupported")
    state = ReleaseState(
        release_id=release_id,
        policy_sha256=policy_sha,
        governor_sha256=governor_sha,
        task_sha256=task_sha,
        baseline_identity_sha256=baseline_sha,
        candidate_identity_sha256=candidate_sha,
        phase=phase,
        active_stage=active_stage,
        sequence=sequence,
        previous_state_sha256=previous,
        stage_aggregates=stage_aggregates,
        seen_observation_id_sha256=seen_observation,
        seen_replication_sha256=seen_replication,
        consumed_batches=parse_consumed_batches(value["consumed_batches"]),
        last_decision=last_decision,
    )
    observations_total = state_observation_count(state)
    require(len(seen_observation) == observations_total and len(seen_replication) == observations_total,
            "release state replay-protection identities do not match aggregate observation count")
    require(sequence == len(state.consumed_batches),
            "release state sequence must equal immutable consumed-batch count")
    if sequence == 0:
        require(previous == GENESIS, "initial release state must use the GENESIS predecessor")
        require(observations_total == 0, "initial release state must not contain observations")
        require(phase == initial_stage(policy) and active_stage == initial_stage(policy),
                "initial release state must begin at the policy-selected rollout stage")
        require(last_decision == "initialized", "initial release state must have initialized decision")
    else:
        require(previous != GENESIS, "noninitial release state must reference an immutable predecessor")
    validate_state_semantics(policy, state)
    return state


def load_state(path: Path, policy: ReleasePolicy) -> tuple[ReleaseState, FileIdentity]:
    value, identity = load_json_mapping(path, "release state")
    return state_from_mapping(value, policy), identity


def observation_from_mapping(value: object, number: int, policy: ReleasePolicy) -> Observation:
    field = f"release observation {number}"
    require(isinstance(value, Mapping), f"{field} must be an object")
    require_exact_fields(value, {"schema_version", "id", "replication_id", "policy_sha256", "stage", "arm", "arm_identity_sha256", "metrics"}, field)
    require(value["schema_version"] == OBSERVATION_SCHEMA, f"{field} has unsupported schema_version")
    observation_id = validate_identifier(value["id"], f"{field}.id")
    require(len(observation_id.encode("ascii")) <= MAX_IDENTIFIER_BYTES, f"{field}.id exceeds {MAX_IDENTIFIER_BYTES} bytes")
    replication_id = validate_identifier(value["replication_id"], f"{field}.replication_id")
    require(len(replication_id.encode("ascii")) <= MAX_IDENTIFIER_BYTES,
            f"{field}.replication_id exceeds {MAX_IDENTIFIER_BYTES} bytes")
    require(validate_sha256(value["policy_sha256"], f"{field}.policy_sha256") == policy.identity.sha256,
            f"{field}.policy_sha256 does not match immutable release policy")
    stage = value["stage"]
    require(isinstance(stage, str) and stage in _STAGES, f"{field}.stage must be shadow or canary")
    arm = value["arm"]
    require(isinstance(arm, str) and arm in _ARM_NAMES, f"{field}.arm must be baseline or candidate")
    expected_arm_identity = policy.baseline.identity_sha256 if arm == "baseline" else policy.candidate.identity_sha256
    require(validate_sha256(value["arm_identity_sha256"], f"{field}.arm_identity_sha256") == expected_arm_identity,
            f"{field}.arm_identity_sha256 does not match the declared {arm} deployment")
    metrics = value["metrics"]
    require(isinstance(metrics, Mapping) and set(metrics) == policy_metric_ids(policy),
            f"{field}.metrics does not exactly match immutable policy metrics")
    parsed_metrics: dict[str, float] = {}
    for metric in policy.metrics:
        measurement = parse_finite_number(metrics[metric.metric_id], f"{field}.metrics.{metric.metric_id}")
        require(metric.lower_bound <= measurement <= metric.upper_bound,
                f"{field}.metrics.{metric.metric_id} is outside declared metric bounds")
        parsed_metrics[metric.metric_id] = measurement
    return Observation(
        observation_id_sha256=sha256_bytes(observation_id.encode("ascii")),
        replication_sha256=sha256_bytes(f"{arm}\x00{replication_id}".encode("ascii")),
        stage=stage,
        arm=arm,
        metrics=parsed_metrics,
    )


def iter_observations(path: Path, policy: ReleasePolicy) -> tuple[list[Observation], FileIdentity]:
    source = path.resolve()
    identity = file_identity(source, "release observation batch", maximum_bytes=MAX_OBSERVATION_BATCH_BYTES)
    observations: list[Observation] = []
    ids: set[str] = set()
    replications: set[str] = set()
    try:
        with source.open("rb") as handle:
            number = 0
            limit = MAX_OBSERVATION_LINE_BYTES + 2
            while True:
                raw = handle.readline(limit)
                if not raw:
                    break
                number += 1
                has_lf = raw.endswith(b"\n")
                payload_bytes = len(raw) - 1 if has_lf else len(raw)
                overlong = payload_bytes > MAX_OBSERVATION_LINE_BYTES or (not has_lf and len(raw) == limit)
                if overlong and not has_lf:
                    while raw and not raw.endswith(b"\n"):
                        raw = handle.readline(limit)
                require(not overlong, f"release observation line {number} exceeds {MAX_OBSERVATION_LINE_BYTES} bytes")
                require(has_lf, f"release observation line {number} must terminate with LF")
                raw = raw[:-1]
                require(not raw.endswith(b"\r"), f"release observation line {number} uses CRLF; strict LF is required")
                require(bool(raw.strip()), f"release observation line {number} is blank")
                observation = observation_from_mapping(parse_strict_json(raw, f"release observation line {number}"), number, policy)
                require(observation.observation_id_sha256 not in ids,
                        "release observation batch contains duplicate observation IDs")
                require(observation.replication_sha256 not in replications,
                        "release observation batch contains duplicate arm/replication IDs")
                ids.add(observation.observation_id_sha256)
                replications.add(observation.replication_sha256)
                observations.append(observation)
                require(len(observations) <= MAX_OBSERVATIONS_PER_BATCH,
                        f"release observation batch exceeds {MAX_OBSERVATIONS_PER_BATCH} records")
    except OSError as error:
        raise ReleaseGovernorError(f"cannot read release observation batch {source}: {error}") from error
    require(observations, "release observation batch must contain at least one observation")
    require_unchanged(source, identity, "release observation batch")
    return observations, identity


def confidence_radius(policy: ReleasePolicy, metric: MetricPolicy, observations: int) -> float:
    require(observations > 0, "confidence radius requires at least one observation")
    # For every arm/metric and every possible sequential sample size n, allocate
    # delta / (arms * metrics * n * (n + 1)).  The n(n+1) series sums to one,
    # so this is a conservative time-uniform two-sided Hoeffding confidence
    # sequence rather than a fixed-n interval reused under optional stopping.
    multiplier = 2.0 * len(_ARM_NAMES) * len(policy.metrics) * observations * (observations + 1) / policy.confidence_delta
    return (metric.upper_bound - metric.lower_bound) * math.sqrt(math.log(multiplier) / (2.0 * observations))


def stage_minimum_observations(policy: ReleasePolicy, stage: str) -> int:
    require(stage in _STAGES, "internal release stage is unsupported")
    return policy.shadow_minimum_observations_per_arm if stage == "shadow" else policy.minimum_observations_per_arm


def summarize_arm(policy: ReleasePolicy, state: ReleaseState, stage: str, arm: str) -> dict[str, dict[str, object]]:
    aggregate = state.stage_aggregates[stage][arm]
    observations = int(aggregate["observations"])
    sums = aggregate["metric_sums"]
    assert isinstance(sums, Mapping)
    summaries: dict[str, dict[str, object]] = {}
    for metric in policy.metrics:
        if observations == 0:
            summaries[metric.metric_id] = {
                "observations": 0,
                "mean": None,
                "lower_confidence_bound": None,
                "upper_confidence_bound": None,
                "time_uniform_hoeffding_radius": None,
            }
            continue
        mean = float(sums[metric.metric_id]) / observations
        radius = confidence_radius(policy, metric, observations)
        summaries[metric.metric_id] = {
            "observations": observations,
            "mean": stable_float(mean),
            "lower_confidence_bound": stable_float(max(metric.lower_bound, mean - radius)),
            "upper_confidence_bound": stable_float(min(metric.upper_bound, mean + radius)),
            "time_uniform_hoeffding_radius": stable_float(radius),
        }
    return summaries


def classify_evidence(policy: ReleasePolicy, state: ReleaseState, stage: str) -> tuple[str, dict[str, dict[str, object]]]:
    """Classify one immutable rollout stage without changing traffic state."""
    require(stage in _STAGES, "internal release stage is unsupported")
    baseline = summarize_arm(policy, state, stage, "baseline")
    candidate = summarize_arm(policy, state, stage, "candidate")
    baseline_count = int(state.stage_aggregates[stage]["baseline"]["observations"])
    candidate_count = int(state.stage_aggregates[stage]["candidate"]["observations"])
    sample_ready = baseline_count >= stage_minimum_observations(policy, stage) and candidate_count >= stage_minimum_observations(policy, stage)
    details: dict[str, dict[str, object]] = {}
    any_rollback = False
    all_promotion = sample_ready
    for metric in policy.metrics:
        baseline_summary = baseline[metric.metric_id]
        candidate_summary = candidate[metric.metric_id]
        if not sample_ready:
            details[metric.metric_id] = {
                "promotion_absolute": False,
                "non_regression_vs_baseline": False,
                "rollback_absolute": False,
                "rollback_regression_vs_baseline": False,
            }
            continue
        baseline_lcb = float(baseline_summary["lower_confidence_bound"])
        baseline_ucb = float(baseline_summary["upper_confidence_bound"])
        candidate_lcb = float(candidate_summary["lower_confidence_bound"])
        candidate_ucb = float(candidate_summary["upper_confidence_bound"])
        if metric.direction == "maximize":
            promotion_absolute = candidate_lcb >= metric.promotion_limit
            non_regression = candidate_lcb >= baseline_ucb - metric.maximum_regression_vs_baseline
            rollback_absolute = candidate_ucb < metric.rollback_limit
            rollback_regression = candidate_ucb < baseline_lcb - metric.maximum_regression_vs_baseline
        else:
            promotion_absolute = candidate_ucb <= metric.promotion_limit
            non_regression = candidate_ucb <= baseline_lcb + metric.maximum_regression_vs_baseline
            rollback_absolute = candidate_lcb > metric.rollback_limit
            rollback_regression = candidate_lcb > baseline_ucb + metric.maximum_regression_vs_baseline
        details[metric.metric_id] = {
            "promotion_absolute": promotion_absolute,
            "non_regression_vs_baseline": non_regression,
            "rollback_absolute": rollback_absolute,
            "rollback_regression_vs_baseline": rollback_regression,
        }
        any_rollback = any_rollback or rollback_absolute or rollback_regression
        all_promotion = all_promotion and promotion_absolute and non_regression
    if not sample_ready:
        return "insufficient_evidence", details
    if any_rollback:
        return "rollback", details
    if all_promotion:
        return "pass", details
    return "pending", details


def stage_outcome(stage: str, evidence_outcome: str) -> str:
    require(stage in _STAGES, "internal release stage is unsupported")
    require(evidence_outcome in {"insufficient_evidence", "pending"},
            "internal nonterminal evidence outcome is unsupported")
    return f"{stage}_{evidence_outcome}"


def validate_state_semantics(policy: ReleasePolicy, state: ReleaseState) -> None:
    """Reject a hand-crafted terminal/transition state inconsistent with evidence.

    This checks the deterministic content of a state.  Authenticity and durable
    ordering of separately published immutable artifacts remain responsibilities
    of the signed registry/transactional deployment controller described in the
    public contract.
    """
    shadow_total = stage_observation_count(state, "shadow")
    canary_total = stage_observation_count(state, "canary")
    if policy.shadow_minimum_observations_per_arm == 0:
        require(shadow_total == 0, "shadow-disabled policy must not contain shadow observations")
    elif state.active_stage == "shadow":
        require(canary_total == 0, "shadow stage must complete before canary observations are accepted")
    else:
        shadow_evidence, _ = classify_evidence(policy, state, "shadow")
        require(shadow_evidence == "pass", "canary/terminal state requires a passed immutable shadow gate")

    evidence_outcome, _ = classify_evidence(policy, state, state.active_stage)
    if state.phase == "shadow":
        require(state.active_stage == "shadow", "shadow phase must have shadow active stage")
        if state.sequence == 0:
            return
        require(state.last_decision == stage_outcome("shadow", evidence_outcome),
                "shadow state decision does not match its immutable evidence")
        return
    if state.phase == "canary":
        require(state.active_stage == "canary", "canary phase must have canary active stage")
        if state.sequence == 0:
            return
        if state.last_decision == "shadow_pass":
            require(policy.shadow_minimum_observations_per_arm > 0,
                    "shadow_pass transition is invalid when the immutable policy disables shadow rollout")
            require(canary_total == 0, "shadow_pass transition must not already contain canary observations")
            return
        require(state.last_decision == stage_outcome("canary", evidence_outcome),
                "canary state decision does not match its immutable evidence")
        return
    if state.phase == "promoted":
        require(state.active_stage == "canary" and state.last_decision == "promote",
                "promoted state must be a canary promotion")
        require(evidence_outcome == "pass", "promoted state does not satisfy the canary promotion gate")
        return
    require(state.phase == "rolled_back", "internal release phase is unsupported")
    require(state.last_decision == "rollback", "rolled_back state must have a rollback decision")
    require(evidence_outcome == "rollback", "rolled_back state does not satisfy the active-stage rollback gate")


def state_to_mapping(state: ReleaseState) -> dict[str, object]:
    return {
        "schema_version": STATE_SCHEMA,
        "release_id": state.release_id,
        "policy_sha256": state.policy_sha256,
        "governor_sha256": state.governor_sha256,
        "task_sha256": state.task_sha256,
        "baseline_identity_sha256": state.baseline_identity_sha256,
        "candidate_identity_sha256": state.candidate_identity_sha256,
        "phase": state.phase,
        "active_stage": state.active_stage,
        "sequence": state.sequence,
        "previous_state_sha256": state.previous_state_sha256,
        "stage_aggregates": state.stage_aggregates,
        "seen_observation_id_sha256": list(state.seen_observation_id_sha256),
        "seen_replication_sha256": list(state.seen_replication_sha256),
        "consumed_batches": list(state.consumed_batches),
        "last_decision": state.last_decision,
    }


def advance_state(policy: ReleasePolicy, previous: ReleaseState, previous_identity: FileIdentity, observations: Sequence[Observation], batch_identity: FileIdentity) -> tuple[ReleaseState, str, dict[str, dict[str, object]]]:
    require(previous.phase in _STAGES, "terminal promoted/rolled_back release states cannot consume more observations")
    stage = previous.active_stage
    require(previous.phase == stage, "release state phase and active stage are inconsistent")
    require(len(previous.consumed_batches) < MAX_CONSUMED_BATCHES, "release state reached its consumed batch limit")
    existing_batches = {str(item["sha256"]) for item in previous.consumed_batches}
    require(batch_identity.sha256 not in existing_batches, "release observation batch was already consumed by this state chain")
    existing_ids = set(previous.seen_observation_id_sha256)
    existing_replications = set(previous.seen_replication_sha256)
    for observation in observations:
        require(observation.stage == stage,
                f"release observation stage {observation.stage!r} does not match active {stage!r} rollout stage")
        require(observation.observation_id_sha256 not in existing_ids,
                "release observation ID was already consumed by this state chain")
        require(observation.replication_sha256 not in existing_replications,
                "release arm/replication ID was already consumed by this state chain")
    require(state_observation_count(previous) + len(observations) <= MAX_TOTAL_OBSERVATIONS,
            f"release state would exceed {MAX_TOTAL_OBSERVATIONS} total observations")

    stage_aggregates: dict[str, dict[str, dict[str, object]]] = {
        rollout_stage: {
            arm: {
                "observations": int(previous.stage_aggregates[rollout_stage][arm]["observations"]),
                "metric_sums": dict(previous.stage_aggregates[rollout_stage][arm]["metric_sums"]),
            }
            for arm in _ARM_NAMES
        }
        for rollout_stage in _STAGES
    }
    for arm in _ARM_NAMES:
        arm_observations = sorted(
            (item for item in observations if item.arm == arm), key=lambda item: item.observation_id_sha256
        )
        if not arm_observations:
            continue
        sums = stage_aggregates[stage][arm]["metric_sums"]
        assert isinstance(sums, dict)
        for metric in policy.metrics:
            values = [observation.metrics[metric.metric_id] for observation in arm_observations]
            sums[metric.metric_id] = stable_float(math.fsum([float(sums[metric.metric_id]), *values]))
        stage_aggregates[stage][arm]["observations"] = int(stage_aggregates[stage][arm]["observations"]) + len(arm_observations)

    provisional = ReleaseState(
        release_id=previous.release_id,
        policy_sha256=previous.policy_sha256,
        governor_sha256=previous.governor_sha256,
        task_sha256=previous.task_sha256,
        baseline_identity_sha256=previous.baseline_identity_sha256,
        candidate_identity_sha256=previous.candidate_identity_sha256,
        phase=stage,
        active_stage=stage,
        sequence=previous.sequence + 1,
        previous_state_sha256=previous_identity.sha256,
        stage_aggregates=stage_aggregates,
        seen_observation_id_sha256=tuple(sorted((*previous.seen_observation_id_sha256, *(item.observation_id_sha256 for item in observations)))),
        seen_replication_sha256=tuple(sorted((*previous.seen_replication_sha256, *(item.replication_sha256 for item in observations)))),
        consumed_batches=(*previous.consumed_batches, {"stage": stage, "sha256": batch_identity.sha256, "bytes": batch_identity.bytes}),
        last_decision=stage_outcome(stage, "pending"),
    )
    evidence_outcome, details = classify_evidence(policy, provisional, stage)
    if evidence_outcome == "pass" and stage == "shadow":
        phase = "canary"
        active_stage = "canary"
        outcome = "shadow_pass"
    elif evidence_outcome == "pass":
        phase = "promoted"
        active_stage = "canary"
        outcome = "promote"
    elif evidence_outcome == "rollback":
        phase = "rolled_back"
        active_stage = stage
        outcome = "rollback"
    else:
        phase = stage
        active_stage = stage
        outcome = stage_outcome(stage, evidence_outcome)
    state = ReleaseState(
        release_id=provisional.release_id,
        policy_sha256=provisional.policy_sha256,
        governor_sha256=provisional.governor_sha256,
        task_sha256=provisional.task_sha256,
        baseline_identity_sha256=provisional.baseline_identity_sha256,
        candidate_identity_sha256=provisional.candidate_identity_sha256,
        phase=phase,
        active_stage=active_stage,
        sequence=provisional.sequence,
        previous_state_sha256=provisional.previous_state_sha256,
        stage_aggregates=provisional.stage_aggregates,
        seen_observation_id_sha256=provisional.seen_observation_id_sha256,
        seen_replication_sha256=provisional.seen_replication_sha256,
        consumed_batches=provisional.consumed_batches,
        last_decision=outcome,
    )
    # Reparse through the public state contract before publishing any artifact.
    state_from_mapping(state_to_mapping(state), policy)
    return state, outcome, details


def metric_policy_mapping(metric: MetricPolicy) -> dict[str, object]:
    return {
        "id": metric.metric_id,
        "kind": metric.kind,
        "direction": metric.direction,
        "lower_bound": stable_float(metric.lower_bound),
        "upper_bound": stable_float(metric.upper_bound),
        "promotion_limit": stable_float(metric.promotion_limit),
        "rollback_limit": stable_float(metric.rollback_limit),
        "maximum_regression_vs_baseline": stable_float(metric.maximum_regression_vs_baseline),
    }


def traffic_directive(policy: ReleasePolicy, state: ReleaseState, outcome: str) -> dict[str, object]:
    if outcome == "promote":
        candidate_basis_points = 10_000
        shadow_basis_points = 0
        mode = "candidate-promoted"
    elif outcome == "rollback":
        candidate_basis_points = 0
        shadow_basis_points = 0
        mode = "baseline-only-rollback"
    elif state.phase == "shadow":
        candidate_basis_points = 0
        shadow_basis_points = policy.shadow_basis_points
        mode = "shadow-mirror-only"
    else:
        candidate_basis_points = policy.canary_basis_points
        shadow_basis_points = 0
        mode = "bounded-canary"
    return {
        "mode": mode,
        "candidate_basis_points": candidate_basis_points,
        "baseline_basis_points": 10_000 - candidate_basis_points,
        "shadow_candidate_basis_points": shadow_basis_points,
        "enforcement": "external-deployment-platform-required",
    }


def decision_mapping(
    policy: ReleasePolicy,
    prior_identity: FileIdentity | None,
    state_identity: FileIdentity | None,
    state: ReleaseState,
    outcome: str,
    metric_verdicts: Mapping[str, Mapping[str, object]],
    batch_identity: FileIdentity | None,
) -> dict[str, object]:
    stage_summaries = {
        stage: {
            "baseline": summarize_arm(policy, state, stage, "baseline"),
            "candidate": summarize_arm(policy, state, stage, "candidate"),
        }
        for stage in _STAGES
    }
    return {
        "schema_version": DECISION_SCHEMA,
        "release_id": policy.policy_id,
        "outcome": outcome,
        "phase": state.phase,
        "active_stage": state.active_stage,
        "traffic_directive": traffic_directive(policy, state, outcome),
        "dependencies": {
            "governor": policy.governor_identity.to_mapping(redact_path=True),
            "policy": policy.identity.to_mapping(redact_path=True),
            "task": policy.task_identity.to_mapping(redact_path=True),
            "baseline": policy.baseline.identity_mapping(),
            "candidate": policy.candidate.identity_mapping(),
            "prior_state": None if prior_identity is None else prior_identity.to_mapping(redact_path=True),
            "new_state": None if state_identity is None else state_identity.to_mapping(redact_path=True),
            "observation_batch": None if batch_identity is None else batch_identity.to_mapping(redact_path=True),
        },
        "policy": {
            "shadow_basis_points": policy.shadow_basis_points,
            "shadow_minimum_observations_per_arm": policy.shadow_minimum_observations_per_arm,
            "canary_basis_points": policy.canary_basis_points,
            "minimum_observations_per_arm": policy.minimum_observations_per_arm,
            "confidence_delta": stable_float(policy.confidence_delta),
            "confidence_method": "time-uniform-two-sided-hoeffding-v1",
            "radius_formula": "range * sqrt(log((2 * arms * metrics * n * (n + 1)) / delta) / (2 * n))",
            "metrics": [metric_policy_mapping(metric) for metric in policy.metrics],
        },
        "aggregates": {
            "stages": stage_summaries,
            "active_stage_metric_verdicts": {identifier: dict(value) for identifier, value in sorted(metric_verdicts.items())},
        },
        "state": {
            "sequence": state.sequence,
            "previous_state_sha256": state.previous_state_sha256,
            "observations": {
                stage: {
                    arm: int(state.stage_aggregates[stage][arm]["observations"])
                    for arm in _ARM_NAMES
                }
                for stage in _STAGES
            },
            "consumed_batches": len(state.consumed_batches),
        },
        "assumptions_and_limits": [
            "Each arm/replication observation is an independent, bounded measurement from the declared shadow or canary allocation.",
            "A shadow directive must mirror only consented/authorized traffic in an isolated external deployment platform; it does not expose candidate output to an end user.",
            "The time-uniform bound supports repeated state inspections only under the declared bounded-observation contract; telemetry integrity remains an operator responsibility.",
            "Live SLO evidence does not replace frozen held-out quality, safety, red-team, factuality, or human-preference evaluation.",
            "The traffic directive is not traffic enforcement, an OS sandbox, provider weight attestation, or a proof of general intelligence.",
        ],
    }


def output_path(path: Path, label: str, protected: set[str]) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except OSError as error:
        raise ReleaseGovernorError(f"cannot resolve {label} {path}: {error}") from error
    require(str(resolved) not in protected, f"{label} must not overwrite an immutable input")
    require(not resolved.exists(), f"{label} already exists; refusing to replace it")
    return resolved


def fsync_directory(path: Path) -> None:
    """Persist a no-replace directory entry where the host filesystem supports it."""
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(path), os.O_RDONLY | directory_flag)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_json_no_replace(path: Path, value: Mapping[str, Any]) -> FileIdentity:
    encoded = canonical_json_bytes(value) + b"\n"
    require(len(encoded) <= MAX_JSON_BYTES, f"published release artifact exceeds {MAX_JSON_BYTES} bytes")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ReleaseGovernorError(f"refusing to replace existing release artifact {path}") from error
    except OSError as error:
        raise ReleaseGovernorError(f"cannot create release artifact {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        fsync_directory(path.parent)
        raise
    fsync_directory(path.parent)
    return file_identity(path, "published release artifact", maximum_bytes=MAX_JSON_BYTES)


def arm_identities(policy: ReleasePolicy) -> tuple[FileIdentity, ...]:
    identities: list[FileIdentity] = [policy.task_identity]
    for arm in (policy.baseline, policy.candidate):
        identities.extend((
            arm.portfolio_identity,
            arm.candidate.manifest_identity,
            arm.candidate.command.identity,
            arm.candidate.card_identity,
            arm.candidate.quality_report_identity,
        ))
    return tuple(identities)


def recheck_policy(policy: ReleasePolicy) -> None:
    require_unchanged(Path(policy.governor_identity.path), policy.governor_identity, "LMM release governor")
    require_unchanged(Path(policy.identity.path), policy.identity, "release policy")
    for identity in arm_identities(policy):
        require_unchanged(Path(identity.path), identity, "release policy dependency")


def current_state_outcome(policy: ReleasePolicy, state: ReleaseState) -> tuple[str, dict[str, dict[str, object]]]:
    """Render the externally actionable outcome implied by an already-valid state."""
    evidence_outcome, verdicts = classify_evidence(policy, state, state.active_stage)
    if state.phase == "promoted":
        return "promote", verdicts
    if state.phase == "rolled_back":
        return "rollback", verdicts
    return stage_outcome(state.active_stage, evidence_outcome), verdicts


def delete_state_if_unchanged(path: Path, identity: FileIdentity) -> None:
    """Best-effort all-or-nothing cleanup when paired decision publication fails."""
    try:
        if file_identity(path, "published release state", maximum_bytes=MAX_JSON_BYTES) == identity:
            path.unlink(missing_ok=True)
    except (OSError, LMMContractError):
        pass


def init_release(arguments: argparse.Namespace) -> dict[str, object]:
    policy = load_policy(arguments.policy)
    protected = {policy.identity.path, *(identity.path for identity in arm_identities(policy))}
    state_output = output_path(arguments.state_output, "--state-output", protected)
    decision_output = output_path(arguments.decision_output, "--decision-output", protected | {str(state_output)})
    state = state_from_mapping(initial_state_mapping(policy), policy)
    recheck_policy(policy)
    state_identity = write_json_no_replace(state_output, state_to_mapping(state))
    initial_outcome, initial_verdicts = current_state_outcome(policy, state)
    decision = decision_mapping(policy, None, state_identity, state, initial_outcome, initial_verdicts, None)
    try:
        decision_identity = write_json_no_replace(decision_output, decision)
    except BaseException:
        delete_state_if_unchanged(state_output, state_identity)
        raise
    print(json.dumps({
        "release_id": policy.policy_id,
        "outcome": initial_outcome,
        "phase": state.phase,
        "state_sha256": state_identity.sha256,
        "decision_sha256": decision_identity.sha256,
    }, sort_keys=True))
    return decision


def advance_release(arguments: argparse.Namespace) -> dict[str, object]:
    policy = load_policy(arguments.policy)
    previous, previous_identity = load_state(arguments.state, policy)
    observations, batch_identity = iter_observations(arguments.observations, policy)
    protected = {
        policy.identity.path,
        previous_identity.path,
        batch_identity.path,
        *(identity.path for identity in arm_identities(policy)),
    }
    state_output = output_path(arguments.state_output, "--state-output", protected)
    decision_output = output_path(arguments.decision_output, "--decision-output", protected | {str(state_output)})
    state, outcome, verdicts = advance_state(policy, previous, previous_identity, observations, batch_identity)
    recheck_policy(policy)
    require_unchanged(Path(previous_identity.path), previous_identity, "prior release state")
    require_unchanged(Path(batch_identity.path), batch_identity, "release observation batch")
    state_identity = write_json_no_replace(state_output, state_to_mapping(state))
    decision = decision_mapping(policy, previous_identity, state_identity, state, outcome, verdicts, batch_identity)
    try:
        decision_identity = write_json_no_replace(decision_output, decision)
    except BaseException:
        delete_state_if_unchanged(state_output, state_identity)
        raise
    print(json.dumps({
        "release_id": policy.policy_id,
        "outcome": outcome,
        "phase": state.phase,
        "sequence": state.sequence,
        "state_sha256": state_identity.sha256,
        "decision_sha256": decision_identity.sha256,
    }, sort_keys=True))
    return decision


def inspect_release(arguments: argparse.Namespace) -> dict[str, object]:
    policy = load_policy(arguments.policy)
    state, state_identity = load_state(arguments.state, policy)
    protected = {policy.identity.path, state_identity.path, *(identity.path for identity in arm_identities(policy))}
    decision_output = output_path(arguments.decision_output, "--decision-output", protected)
    outcome, verdicts = current_state_outcome(policy, state)
    recheck_policy(policy)
    require_unchanged(Path(state_identity.path), state_identity, "release state")
    decision = decision_mapping(policy, state_identity, None, state, outcome, verdicts, None)
    decision_identity = write_json_no_replace(decision_output, decision)
    print(json.dumps({
        "release_id": policy.policy_id,
        "phase": state.phase,
        "current_evidence_outcome": outcome,
        "sequence": state.sequence,
        "decision_sha256": decision_identity.sha256,
    }, sort_keys=True))
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="publish immutable initial rollout state and traffic directive")
    init.add_argument("--policy", required=True, type=Path)
    init.add_argument("--state-output", required=True, type=Path)
    init.add_argument("--decision-output", required=True, type=Path)
    advance = commands.add_parser("advance", help="consume one immutable telemetry batch and publish next state/decision")
    advance.add_argument("--policy", required=True, type=Path)
    advance.add_argument("--state", required=True, type=Path)
    advance.add_argument("--observations", required=True, type=Path)
    advance.add_argument("--state-output", required=True, type=Path)
    advance.add_argument("--decision-output", required=True, type=Path)
    inspect = commands.add_parser("inspect", help="recompute and publish an immutable aggregate-only audit directive")
    inspect.add_argument("--policy", required=True, type=Path)
    inspect.add_argument("--state", required=True, type=Path)
    inspect.add_argument("--decision-output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "init":
            init_release(arguments)
        elif arguments.command == "advance":
            advance_release(arguments)
        elif arguments.command == "inspect":
            inspect_release(arguments)
        else:
            raise ReleaseGovernorError("unsupported command")
        return 0
    except (ReleaseGovernorError, LMMContractError, OSError, ValueError) as error:
        print(f"LMM release governor: ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
