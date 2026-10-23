#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Select an AI algorithm/optimizer only from bounded held-out evidence.

This is an *optimizer governor*, not a fabricated universal optimizer or an
LLM-training implementation.  It compares operator-declared algorithm and
optimizer configurations using repeated, provenance-bound held-out measurements.
A candidate is selected only when conservative Hoeffding confidence bounds show
that it improves every declared objective over a pinned baseline.  Otherwise it
fails closed and emits no deployment decision.

No prompts, answers, model weights, training data, or hyperparameter values are
invented or written into the report.  Candidate configuration hashes bind the
real external training configuration while keeping sensitive configuration data
outside this control-plane report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


POLICY_SCHEMA = "vegai-evidential-optimizer-policy/v1"
CANDIDATES_SCHEMA = "vegai-evidential-optimizer-candidates/v1"
OBSERVATION_SCHEMA = "vegai-evidential-optimizer-observation/v1"
DECISION_SCHEMA = "vegai-evidential-optimizer-decision/v1"

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_OBSERVATIONS_BYTES = 64 * 1024 * 1024
MAX_CANDIDATES = 128
MAX_METRICS = 16
MAX_OBSERVATIONS = 100_000
MAX_IDENTIFIER_BYTES = 128
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_DIRECTIONS = frozenset(("maximize", "minimize"))
_SELECTION_STRATEGY = "robust-pareto-lexicographic/v1"


class EvidentialOptimizerError(ValueError):
    """An input, provenance, schema, or publication contract was violated."""


class OptimizerGateFailure(EvidentialOptimizerError):
    """Evidence was valid but did not certify a candidate for selection."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    sha256: str
    bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class BenchmarkIdentity:
    suite_id: str
    suite_revision_sha256: str
    protocol_sha256: str
    split: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "suite_id": self.suite_id,
            "suite_revision_sha256": self.suite_revision_sha256,
            "protocol_sha256": self.protocol_sha256,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    algorithm_family: str
    algorithm_config_sha256: str
    optimizer_family: str
    optimizer_config_sha256: str
    training_contract_sha256: str
    model_artifact_sha256: str

    def observation_mapping(self) -> dict[str, str]:
        return {
            "algorithm_config_sha256": self.algorithm_config_sha256,
            "optimizer_config_sha256": self.optimizer_config_sha256,
            "training_contract_sha256": self.training_contract_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
        }

    def report_mapping(self) -> dict[str, object]:
        return {
            "algorithm": {
                "family": self.algorithm_family,
                "config_sha256": self.algorithm_config_sha256,
            },
            "optimizer": {
                "family": self.optimizer_family,
                "config_sha256": self.optimizer_config_sha256,
            },
            "training_contract_sha256": self.training_contract_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    identity: CandidateIdentity


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    metric_id: str
    direction: str
    lower_bound: float
    upper_bound: float
    acceptance_limit: float
    minimum_improvement_vs_baseline: float
    dominance_margin: float


@dataclass(frozen=True, slots=True)
class OptimizerPolicy:
    policy_id: str
    benchmark: BenchmarkIdentity
    baseline_candidate_id: str
    confidence_delta: float
    minimum_observations_per_candidate: int
    metrics: tuple[MetricPolicy, ...]
    priority_metrics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    candidate_id: str
    run_id: str
    replication_id: str
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class MetricSummary:
    observations: int
    mean: float
    hoeffding_radius: float
    lower_confidence_bound: float
    upper_confidence_bound: float
    passes_acceptance_limit: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "mean": stable_float(self.mean),
            "hoeffding_radius": stable_float(self.hoeffding_radius),
            "lower_confidence_bound": stable_float(self.lower_confidence_bound),
            "upper_confidence_bound": stable_float(self.upper_confidence_bound),
            "passes_acceptance_limit": self.passes_acceptance_limit,
        }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidentialOptimizerError(message)


def stable_float(value: float) -> float:
    """Render a deterministic, finite report value without hiding precision policy."""
    require(math.isfinite(value), "internal report value is not finite")
    return float(format(value, ".12g"))


def validate_identifier(value: object, field: str) -> str:
    require(isinstance(value, str), f"{field} must be a string")
    rendered = value.encode("utf-8")
    require(0 < len(rendered) <= MAX_IDENTIFIER_BYTES, f"{field} has invalid UTF-8 byte length")
    require(_IDENTIFIER_PATTERN.fullmatch(value) is not None, f"{field} must be an ASCII identifier")
    return value


def validate_sha256(value: object, field: str) -> str:
    require(isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None,
            f"{field} must be a lowercase SHA-256 hex digest")
    return value


def parse_finite_number(value: object, field: str) -> float:
    require(type(value) in {int, float} and not isinstance(value, bool), f"{field} must be a finite number")
    parsed = float(value)
    require(math.isfinite(parsed), f"{field} must be finite")
    return parsed


def parse_bounded_int(value: object, field: str, minimum: int, maximum: int) -> int:
    require(type(value) is int and minimum <= value <= maximum,
            f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    missing = sorted(expected.difference(actual))
    unknown = sorted(str(key) for key in actual.difference(expected))
    require(not missing, f"{field} is missing fields: {', '.join(missing)}")
    require(not unknown, f"{field} contains unsupported fields: {', '.join(unknown)}")


def canonical_json_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return (rendered + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EvidentialOptimizerError(f"cannot canonicalize report JSON: {error}") from error


def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvidentialOptimizerError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def parse_strict_json(rendered: str, label: str) -> object:
    try:
        return json.loads(rendered, object_pairs_hook=reject_duplicate_json_keys)
    except json.JSONDecodeError as error:
        raise EvidentialOptimizerError(f"cannot parse {label} as JSON: {error}") from error


def read_bounded_file(path: Path, label: str, maximum_bytes: int) -> tuple[bytes, FileIdentity]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        require(resolved.is_file(), f"{label} is not a regular file: {resolved}")
        with resolved.open("rb") as source:
            payload = source.read(maximum_bytes + 1)
    except OSError as error:
        raise EvidentialOptimizerError(f"cannot read {label} {path}: {error}") from error
    require(len(payload) <= maximum_bytes, f"{label} exceeds {maximum_bytes} bytes")
    identity = FileIdentity(
        path=str(resolved),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )
    return payload, identity


def require_unchanged(identity: FileIdentity, label: str, maximum_bytes: int) -> None:
    _, current = read_bounded_file(Path(identity.path), label, maximum_bytes)
    require(current == identity, f"{label} changed while the selection was being evaluated")


def load_json_mapping(path: Path, label: str, maximum_bytes: int = MAX_JSON_BYTES) -> tuple[dict[str, Any], FileIdentity]:
    payload, identity = read_bounded_file(path, label, maximum_bytes)
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidentialOptimizerError(f"cannot decode {label} as UTF-8: {error}") from error
    value = parse_strict_json(decoded, label)
    require(isinstance(value, Mapping), f"{label} must contain one JSON object")
    return dict(value), identity


def parse_benchmark(value: object, field: str) -> BenchmarkIdentity:
    require(isinstance(value, Mapping), f"{field} must be an object")
    exact_fields(value, {"suite_id", "suite_revision_sha256", "protocol_sha256", "split"}, field)
    split = validate_identifier(value["split"], f"{field}.split")
    require(split == "heldout", f"{field}.split must be heldout")
    return BenchmarkIdentity(
        suite_id=validate_identifier(value["suite_id"], f"{field}.suite_id"),
        suite_revision_sha256=validate_sha256(value["suite_revision_sha256"], f"{field}.suite_revision_sha256"),
        protocol_sha256=validate_sha256(value["protocol_sha256"], f"{field}.protocol_sha256"),
        split=split,
    )


def parse_metric_policy(value: object, field: str) -> MetricPolicy:
    require(isinstance(value, Mapping), f"{field} must be an object")
    expected = {
        "id",
        "direction",
        "lower_bound",
        "upper_bound",
        "acceptance_limit",
        "minimum_improvement_vs_baseline",
        "dominance_margin",
    }
    exact_fields(value, expected, field)
    metric_id = validate_identifier(value["id"], f"{field}.id")
    direction = value["direction"]
    require(isinstance(direction, str) and direction in _ALLOWED_DIRECTIONS,
            f"{field}.direction must be maximize or minimize")
    lower_bound = parse_finite_number(value["lower_bound"], f"{field}.lower_bound")
    upper_bound = parse_finite_number(value["upper_bound"], f"{field}.upper_bound")
    require(lower_bound < upper_bound, f"{field} requires lower_bound < upper_bound")
    acceptance_limit = parse_finite_number(value["acceptance_limit"], f"{field}.acceptance_limit")
    require(lower_bound <= acceptance_limit <= upper_bound,
            f"{field}.acceptance_limit must be within declared metric bounds")
    improvement = parse_finite_number(value["minimum_improvement_vs_baseline"],
                                      f"{field}.minimum_improvement_vs_baseline")
    dominance_margin = parse_finite_number(value["dominance_margin"], f"{field}.dominance_margin")
    metric_range = upper_bound - lower_bound
    require(0.0 <= improvement <= metric_range,
            f"{field}.minimum_improvement_vs_baseline must be within metric range")
    require(0.0 <= dominance_margin <= metric_range,
            f"{field}.dominance_margin must be within metric range")
    return MetricPolicy(
        metric_id=metric_id,
        direction=direction,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        acceptance_limit=acceptance_limit,
        minimum_improvement_vs_baseline=improvement,
        dominance_margin=dominance_margin,
    )


def load_policy(path: Path) -> tuple[OptimizerPolicy, FileIdentity]:
    value, identity = load_json_mapping(path, "optimizer policy")
    expected = {
        "schema_version",
        "id",
        "benchmark",
        "baseline_candidate_id",
        "confidence_delta",
        "minimum_observations_per_candidate",
        "selection_strategy",
        "metrics",
        "priority_metrics",
    }
    exact_fields(value, expected, "optimizer policy")
    require(value["schema_version"] == POLICY_SCHEMA, "optimizer policy has unsupported schema_version")
    confidence_delta = parse_finite_number(value["confidence_delta"], "optimizer policy.confidence_delta")
    require(0.0 < confidence_delta < 1.0, "optimizer policy.confidence_delta must be in (0, 1)")
    minimum_observations = parse_bounded_int(
        value["minimum_observations_per_candidate"],
        "optimizer policy.minimum_observations_per_candidate",
        2,
        MAX_OBSERVATIONS,
    )
    require(value["selection_strategy"] == _SELECTION_STRATEGY,
            f"optimizer policy.selection_strategy must be {_SELECTION_STRATEGY}")
    raw_metrics = value["metrics"]
    require(isinstance(raw_metrics, list) and 1 <= len(raw_metrics) <= MAX_METRICS,
            f"optimizer policy.metrics must contain 1 to {MAX_METRICS} metrics")
    metrics = tuple(parse_metric_policy(item, f"optimizer policy.metrics[{index}]")
                    for index, item in enumerate(raw_metrics))
    metric_ids = tuple(metric.metric_id for metric in metrics)
    require(len(set(metric_ids)) == len(metric_ids), "optimizer policy.metrics contains duplicate IDs")
    raw_priority = value["priority_metrics"]
    require(isinstance(raw_priority, list) and len(raw_priority) == len(metrics),
            "optimizer policy.priority_metrics must rank every declared metric once")
    priority = tuple(validate_identifier(item, f"optimizer policy.priority_metrics[{index}]")
                     for index, item in enumerate(raw_priority))
    require(set(priority) == set(metric_ids) and len(set(priority)) == len(priority),
            "optimizer policy.priority_metrics must be a duplicate-free permutation of metric IDs")
    return OptimizerPolicy(
        policy_id=validate_identifier(value["id"], "optimizer policy.id"),
        benchmark=parse_benchmark(value["benchmark"], "optimizer policy.benchmark"),
        baseline_candidate_id=validate_identifier(value["baseline_candidate_id"], "optimizer policy.baseline_candidate_id"),
        confidence_delta=confidence_delta,
        minimum_observations_per_candidate=minimum_observations,
        metrics=metrics,
        priority_metrics=priority,
    ), identity


def parse_component(value: object, field: str) -> tuple[str, str]:
    require(isinstance(value, Mapping), f"{field} must be an object")
    exact_fields(value, {"family", "config_sha256"}, field)
    return (
        validate_identifier(value["family"], f"{field}.family"),
        validate_sha256(value["config_sha256"], f"{field}.config_sha256"),
    )


def load_candidates(path: Path) -> tuple[BenchmarkIdentity, tuple[Candidate, ...], FileIdentity]:
    value, identity = load_json_mapping(path, "optimizer candidates")
    exact_fields(value, {"schema_version", "id", "benchmark", "candidates"}, "optimizer candidates")
    require(value["schema_version"] == CANDIDATES_SCHEMA, "optimizer candidates has unsupported schema_version")
    validate_identifier(value["id"], "optimizer candidates.id")
    raw_candidates = value["candidates"]
    require(isinstance(raw_candidates, list) and 2 <= len(raw_candidates) <= MAX_CANDIDATES,
            f"optimizer candidates must contain 2 to {MAX_CANDIDATES} entries")
    candidates: list[Candidate] = []
    for index, raw in enumerate(raw_candidates):
        field = f"optimizer candidates.candidates[{index}]"
        require(isinstance(raw, Mapping), f"{field} must be an object")
        exact_fields(raw, {"id", "algorithm", "optimizer", "training_contract_sha256", "model_artifact_sha256"}, field)
        algorithm_family, algorithm_config = parse_component(raw["algorithm"], f"{field}.algorithm")
        optimizer_family, optimizer_config = parse_component(raw["optimizer"], f"{field}.optimizer")
        candidates.append(Candidate(
            candidate_id=validate_identifier(raw["id"], f"{field}.id"),
            identity=CandidateIdentity(
                algorithm_family=algorithm_family,
                algorithm_config_sha256=algorithm_config,
                optimizer_family=optimizer_family,
                optimizer_config_sha256=optimizer_config,
                training_contract_sha256=validate_sha256(raw["training_contract_sha256"],
                                                          f"{field}.training_contract_sha256"),
                model_artifact_sha256=validate_sha256(raw["model_artifact_sha256"],
                                                       f"{field}.model_artifact_sha256"),
            ),
        ))
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    require(len(set(candidate_ids)) == len(candidate_ids), "optimizer candidates contains duplicate IDs")
    return parse_benchmark(value["benchmark"], "optimizer candidates.benchmark"), tuple(candidates), identity


def parse_observation(
    value: object,
    line_number: int,
    policy: OptimizerPolicy,
    candidates: Mapping[str, Candidate],
) -> Observation:
    field = f"optimizer observations line {line_number}"
    require(isinstance(value, Mapping), f"{field} must be an object")
    expected = {
        "schema_version",
        "candidate_id",
        "run_id",
        "replication_id",
        "benchmark",
        "candidate_identity",
        "execution_report_sha256",
        "metrics",
    }
    exact_fields(value, expected, field)
    require(value["schema_version"] == OBSERVATION_SCHEMA, f"{field} has unsupported schema_version")
    candidate_id = validate_identifier(value["candidate_id"], f"{field}.candidate_id")
    require(candidate_id in candidates, f"{field} references unknown candidate {candidate_id}")
    require(parse_benchmark(value["benchmark"], f"{field}.benchmark") == policy.benchmark,
            f"{field}.benchmark does not exactly match policy benchmark identity")
    identity_value = value["candidate_identity"]
    require(isinstance(identity_value, Mapping), f"{field}.candidate_identity must be an object")
    exact_fields(identity_value, set(candidates[candidate_id].identity.observation_mapping()), f"{field}.candidate_identity")
    parsed_identity = {
        name: validate_sha256(component, f"{field}.candidate_identity.{name}")
        for name, component in identity_value.items()
    }
    require(parsed_identity == candidates[candidate_id].identity.observation_mapping(),
            f"{field}.candidate_identity does not match candidate manifest")
    validate_sha256(value["execution_report_sha256"], f"{field}.execution_report_sha256")
    metrics = value["metrics"]
    require(isinstance(metrics, Mapping), f"{field}.metrics must be an object")
    expected_metric_ids = {metric.metric_id for metric in policy.metrics}
    exact_fields(metrics, expected_metric_ids, f"{field}.metrics")
    parsed_metrics: dict[str, float] = {}
    for metric in policy.metrics:
        observation_value = parse_finite_number(metrics[metric.metric_id], f"{field}.metrics.{metric.metric_id}")
        require(metric.lower_bound <= observation_value <= metric.upper_bound,
                f"{field}.metrics.{metric.metric_id} is outside declared bounded range")
        parsed_metrics[metric.metric_id] = observation_value
    return Observation(
        candidate_id=candidate_id,
        run_id=validate_identifier(value["run_id"], f"{field}.run_id"),
        replication_id=validate_identifier(value["replication_id"], f"{field}.replication_id"),
        metrics=parsed_metrics,
    )


def load_observations(path: Path, policy: OptimizerPolicy, candidates: Mapping[str, Candidate]) -> tuple[tuple[Observation, ...], FileIdentity]:
    payload, identity = read_bounded_file(path, "optimizer observations", MAX_OBSERVATIONS_BYTES)
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidentialOptimizerError(f"optimizer observations is not UTF-8: {error}") from error
    observations: list[Observation] = []
    run_ids: set[str] = set()
    replication_ids: set[tuple[str, str]] = set()
    for line_number, line in enumerate(decoded.splitlines(), start=1):
        require(len(line.encode("utf-8")) <= MAX_JSON_BYTES, f"optimizer observations line {line_number} is too large")
        require(line.strip(), f"optimizer observations line {line_number} is blank")
        value = parse_strict_json(line, f"optimizer observations line {line_number}")
        observation = parse_observation(value, line_number, policy, candidates)
        require(observation.run_id not in run_ids, f"optimizer observations duplicate run_id {observation.run_id}")
        replication_key = (observation.candidate_id, observation.replication_id)
        require(replication_key not in replication_ids,
                f"optimizer observations duplicate replication_id for {observation.candidate_id}: {observation.replication_id}")
        run_ids.add(observation.run_id)
        replication_ids.add(replication_key)
        observations.append(observation)
        require(len(observations) <= MAX_OBSERVATIONS, f"optimizer observations exceeds {MAX_OBSERVATIONS} records")
    require(observations, "optimizer observations must contain at least one record")
    return tuple(observations), identity


def hoeffding_radius(metric: MetricPolicy, observations: int, candidate_count: int, metric_count: int, delta: float) -> float:
    """Bonferroni-adjusted two-sided Hoeffding radius for bounded observations."""
    require(observations > 0 and candidate_count > 0 and metric_count > 0, "invalid confidence cardinality")
    intervals = candidate_count * metric_count
    radius = (metric.upper_bound - metric.lower_bound) * math.sqrt(
        math.log((2.0 * intervals) / delta) / (2.0 * float(observations))
    )
    require(math.isfinite(radius) and radius >= 0.0, "confidence radius is not finite")
    return radius


def summarize_candidate(
    candidate: Candidate,
    grouped: Mapping[str, Sequence[Observation]],
    policy: OptimizerPolicy,
    candidate_count: int,
) -> tuple[dict[str, MetricSummary], bool]:
    observations = grouped.get(candidate.candidate_id, ())
    if len(observations) < policy.minimum_observations_per_candidate:
        return {}, False
    summaries: dict[str, MetricSummary] = {}
    for metric in policy.metrics:
        values = [observation.metrics[metric.metric_id] for observation in observations]
        mean = math.fsum(values) / float(len(values))
        radius = hoeffding_radius(metric, len(values), candidate_count, len(policy.metrics), policy.confidence_delta)
        lower = max(metric.lower_bound, mean - radius)
        upper = min(metric.upper_bound, mean + radius)
        if metric.direction == "maximize":
            passes = lower >= metric.acceptance_limit
        else:
            passes = upper <= metric.acceptance_limit
        summaries[metric.metric_id] = MetricSummary(
            observations=len(values),
            mean=mean,
            hoeffding_radius=radius,
            lower_confidence_bound=lower,
            upper_confidence_bound=upper,
            passes_acceptance_limit=passes,
        )
    return summaries, True


def certified_improves_over_baseline(
    candidate: Mapping[str, MetricSummary],
    baseline: Mapping[str, MetricSummary],
    policy: OptimizerPolicy,
) -> bool:
    for metric in policy.metrics:
        candidate_metric = candidate[metric.metric_id]
        baseline_metric = baseline[metric.metric_id]
        if metric.direction == "maximize":
            if candidate_metric.lower_confidence_bound < baseline_metric.upper_confidence_bound + metric.minimum_improvement_vs_baseline:
                return False
        else:
            if candidate_metric.upper_confidence_bound > baseline_metric.lower_confidence_bound - metric.minimum_improvement_vs_baseline:
                return False
    return True


def certified_dominates(
    first: Mapping[str, MetricSummary],
    second: Mapping[str, MetricSummary],
    policy: OptimizerPolicy,
) -> bool:
    strictly_better = False
    for metric in policy.metrics:
        first_metric = first[metric.metric_id]
        second_metric = second[metric.metric_id]
        if metric.direction == "maximize":
            required = second_metric.upper_confidence_bound + metric.dominance_margin
            if first_metric.lower_confidence_bound < required:
                return False
            strictly_better = strictly_better or first_metric.lower_confidence_bound > required
        else:
            required = second_metric.lower_confidence_bound - metric.dominance_margin
            if first_metric.upper_confidence_bound > required:
                return False
            strictly_better = strictly_better or first_metric.upper_confidence_bound < required
    return strictly_better


def priority_key(candidate_id: str, summaries: Mapping[str, MetricSummary], policy: OptimizerPolicy) -> tuple[float | str, ...]:
    key: list[float | str] = []
    metrics_by_id = {metric.metric_id: metric for metric in policy.metrics}
    for metric_id in policy.priority_metrics:
        metric = metrics_by_id[metric_id]
        summary = summaries[metric_id]
        # Higher conservative lower bound is better for maximization; smaller
        # conservative upper bound is better for minimization.  Candidate ID is
        # a final deterministic tie-breaker, never an implicit quality signal.
        key.append(-summary.lower_confidence_bound if metric.direction == "maximize" else summary.upper_confidence_bound)
    key.append(candidate_id)
    return tuple(key)


def report_candidate(
    candidate: Candidate,
    observations: int,
    summaries: Mapping[str, MetricSummary],
    enough_observations: bool,
    improves_baseline: bool,
) -> dict[str, object]:
    rendered: dict[str, object] = {
        "id": candidate.candidate_id,
        "identity": candidate.identity.report_mapping(),
        "observations": observations,
        "minimum_observations_met": enough_observations,
        "metrics": {metric_id: summary.to_mapping() for metric_id, summary in sorted(summaries.items())},
        "certified_improvement_over_baseline": improves_baseline,
    }
    return rendered


def ensure_distinct_output(output_path: Path, identities: Sequence[FileIdentity]) -> Path:
    try:
        resolved = output_path.expanduser().resolve(strict=False)
    except OSError as error:
        raise EvidentialOptimizerError(f"cannot resolve output path {output_path}: {error}") from error
    input_paths = {identity.path for identity in identities}
    require(str(resolved) not in input_paths, "decision output must not overwrite an immutable input")
    return resolved


def write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise EvidentialOptimizerError(f"refusing to overwrite existing decision output: {path}") from error
    except OSError as error:
        raise EvidentialOptimizerError(f"cannot create decision output {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
    except OSError as error:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise EvidentialOptimizerError(f"cannot write decision output {path}: {error}") from error


def select(policy_path: Path, candidates_path: Path, observations_path: Path, output_path: Path) -> dict[str, Any]:
    policy, policy_identity = load_policy(policy_path)
    candidates_benchmark, candidates, candidates_identity = load_candidates(candidates_path)
    require(candidates_benchmark == policy.benchmark,
            "candidate manifest benchmark identity does not exactly match policy benchmark")
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    require(policy.baseline_candidate_id in candidates_by_id,
            "policy baseline_candidate_id is absent from candidate manifest")
    observations, observations_identity = load_observations(observations_path, policy, candidates_by_id)

    grouped: dict[str, list[Observation]] = {candidate.candidate_id: [] for candidate in candidates}
    for observation in observations:
        grouped[observation.candidate_id].append(observation)

    summaries_by_id: dict[str, dict[str, MetricSummary]] = {}
    enough_by_id: dict[str, bool] = {}
    for candidate in candidates:
        summaries, enough = summarize_candidate(candidate, grouped, policy, len(candidates))
        summaries_by_id[candidate.candidate_id] = summaries
        enough_by_id[candidate.candidate_id] = enough

    insufficient = [
        f"{candidate.candidate_id}={len(grouped[candidate.candidate_id])}"
        for candidate in candidates
        if not enough_by_id[candidate.candidate_id]
    ]
    if insufficient:
        raise OptimizerGateFailure(
            "every declared candidate requires independent held-out evidence; insufficient observations: "
            + ", ".join(insufficient)
        )

    baseline_summaries = summaries_by_id[policy.baseline_candidate_id]

    improvements: dict[str, bool] = {}
    eligible_ids: list[str] = []
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        if candidate_id == policy.baseline_candidate_id or not enough_by_id[candidate_id]:
            improvements[candidate_id] = False
            continue
        summaries = summaries_by_id[candidate_id]
        passes_limits = all(summary.passes_acceptance_limit for summary in summaries.values())
        improves = passes_limits and certified_improves_over_baseline(summaries, baseline_summaries, policy)
        improvements[candidate_id] = improves
        if improves:
            eligible_ids.append(candidate_id)

    if not eligible_ids:
        counts = ", ".join(f"{candidate.candidate_id}={len(grouped[candidate.candidate_id])}" for candidate in candidates)
        raise OptimizerGateFailure(
            "no non-baseline candidate certified an all-objective confidence-bounded improvement over the baseline "
            f"(observation counts: {counts})"
        )

    frontier_ids = [
        candidate_id
        for candidate_id in eligible_ids
        if not any(
            other_id != candidate_id and certified_dominates(
                summaries_by_id[other_id], summaries_by_id[candidate_id], policy
            )
            for other_id in eligible_ids
        )
    ]
    require(frontier_ids, "internal optimizer frontier is empty")
    selected_id = min(frontier_ids, key=lambda item: priority_key(item, summaries_by_id[item], policy))

    identity_set = (policy_identity, candidates_identity, observations_identity)
    for identity, label, limit in (
        (policy_identity, "optimizer policy", MAX_JSON_BYTES),
        (candidates_identity, "optimizer candidates", MAX_JSON_BYTES),
        (observations_identity, "optimizer observations", MAX_OBSERVATIONS_BYTES),
    ):
        require_unchanged(identity, label, limit)
    output = ensure_distinct_output(output_path, identity_set)

    candidate_reports = [
        report_candidate(
            candidate,
            len(grouped[candidate.candidate_id]),
            summaries_by_id[candidate.candidate_id],
            enough_by_id[candidate.candidate_id],
            improvements.get(candidate.candidate_id, False),
        )
        for candidate in sorted(candidates, key=lambda item: item.candidate_id)
    ]
    metrics_report = [
        {
            "id": metric.metric_id,
            "direction": metric.direction,
            "lower_bound": stable_float(metric.lower_bound),
            "upper_bound": stable_float(metric.upper_bound),
            "acceptance_limit": stable_float(metric.acceptance_limit),
            "minimum_improvement_vs_baseline": stable_float(metric.minimum_improvement_vs_baseline),
            "dominance_margin": stable_float(metric.dominance_margin),
        }
        for metric in policy.metrics
    ]
    decision: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA,
        "outcome": "selected",
        "policy": {
            "id": policy.policy_id,
            "identity": policy_identity.to_mapping(),
            "benchmark": policy.benchmark.to_mapping(),
            "baseline_candidate_id": policy.baseline_candidate_id,
            "confidence_delta": stable_float(policy.confidence_delta),
            "minimum_observations_per_candidate": policy.minimum_observations_per_candidate,
            "selection_strategy": _SELECTION_STRATEGY,
            "priority_metrics": list(policy.priority_metrics),
            "metrics": metrics_report,
        },
        "candidate_manifest": candidates_identity.to_mapping(),
        "observations": observations_identity.to_mapping(),
        "confidence_method": {
            "name": "bonferroni-adjusted-two-sided-hoeffding/v1",
            "intervals": len(candidates) * len(policy.metrics),
            "formula": "range * sqrt(log((2 * candidates * metrics) / delta) / (2 * observations))",
            "assumptions": [
                "Each declared replication is an independent bounded held-out measurement.",
                "Benchmark suite revision and protocol are immutable and identical across candidates.",
                "The analysis is pre-registered and one-shot; adaptive repeated held-out peeking requires a separate sequential-testing policy.",
                "Confidence bounds do not prove general intelligence, external-world factuality, or future deployment performance.",
            ],
        },
        "candidates": candidate_reports,
        "certified_robust_pareto_frontier": sorted(frontier_ids),
        "selected": {
            "id": selected_id,
            "reason": "all-metric baseline improvement certified; selected from robust Pareto frontier by declared lexicographic priority",
            "identity": candidates_by_id[selected_id].identity.report_mapping(),
        },
    }
    write_json_no_replace(output, decision)
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    select_parser = subcommands.add_parser("select", help="certify one algorithm/optimizer configuration from held-out evidence")
    select_parser.add_argument("--policy", required=True, type=Path)
    select_parser.add_argument("--candidates", required=True, type=Path)
    select_parser.add_argument("--observations", required=True, type=Path)
    select_parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "select":
            decision = select(arguments.policy, arguments.candidates, arguments.observations, arguments.output)
            print(json.dumps({"outcome": decision["outcome"], "selected": decision["selected"]["id"]},
                             ensure_ascii=False, sort_keys=True))
            return 0
        raise EvidentialOptimizerError("unsupported command")
    except OptimizerGateFailure as error:
        print(f"evidential optimizer: GATE FAILED: {error}", file=sys.stderr)
        return 3
    except (EvidentialOptimizerError, OSError, ValueError) as error:
        print(f"evidential optimizer: ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
