#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Route real LMM generation to the best measured eligible domain expert.

A portfolio is not a fabricated universal super-model.  It is a fail-closed
system layer that selects an operator-declared model only after checking its
immutable manifest, a passed citation/abstention quality report, target-domain
evaluation card, modality/context limits, residency, safety tier, latency and
cost constraints.  It can also produce an independent multi-provider panel
plan for an operator-owned adjudication workflow.

The tool never writes a prompt, response template, hidden chain of thought, or
answer branch.  ``generate`` passes the caller's real request/evidence unchanged
in private temporary files to the selected configured gateway command.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from lmm_contract import (
    GENERATION_REPORT_SCHEMA,
    MAX_JSON_BYTES,
    MAX_TEXT_BYTES,
    AssetRegistry,
    EvidenceBundle,
    FileIdentity,
    GenerationRequest,
    LMMContractError,
    ModelManifest,
    canonical_json_bytes,
    file_identity,
    require,
    require_unchanged,
    sha256_bytes,
    validate_identifier,
    validate_sha256,
    validate_unicode_text,
    write_json_atomic,
)
from lmm_gateway_template import (
    GROUNDING_MODES,
    GatewayCommandTemplate,
    direct_option_value,
    expand_gateway_command,
    load_gateway_command_template,
    require_model_argument,
)


PORTFOLIO_SCHEMA = "vegai-lmm-portfolio/v1"
EVALUATION_CARD_SCHEMA = "vegai-lmm-evaluation-card/v1"
TASK_SCHEMA = "vegai-lmm-routing-task/v1"
DECISION_SCHEMA = "vegai-lmm-route-decision/v1"
GENERATION_SCHEMA = "vegai-lmm-portfolio-generation-report/v1"
QUALITY_REPORT_SCHEMA = "vegai-lmm-quality-report/v1"

MAX_CANDIDATES = 64
MAX_DOMAINS_PER_CANDIDATE = 64
MAX_RESIDENCIES_PER_CANDIDATE = 32
MAX_PANEL_CANDIDATES = 8
MAX_GATEWAY_STDOUT_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 180.0
_RUNTIME_IDENTITY_FIELDS = (
    "gateway_tool",
    "model_manifest",
    "tokenizer",
    "asset_registry",
    "worker_command_configuration",
    "model_artifacts",
)
_ALLOWED_MODALITIES = frozenset(("text", "image", "audio", "video"))
_ALLOWED_STRATEGIES = frozenset(("single-best", "independent-panel"))


class LMMPortfolioError(LMMContractError):
    """A portfolio, evaluation card, routing constraint, or dispatch failed."""


@dataclass(frozen=True, slots=True)
class EvaluationMeasurement:
    domain: str
    suite_id: str
    suite_report_sha256: str
    cases: int
    quality_lower_bound: float
    reliability_lower_bound: float
    safety_lower_bound: float
    p95_latency_ms: int
    cost_microunits_per_1k_output_tokens: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "suite_id": self.suite_id,
            "suite_report_sha256": self.suite_report_sha256,
            "cases": self.cases,
            "quality_lower_bound": self.quality_lower_bound,
            "reliability_lower_bound": self.reliability_lower_bound,
            "safety_lower_bound": self.safety_lower_bound,
            "p95_latency_ms": self.p95_latency_ms,
            "cost_microunits_per_1k_output_tokens": self.cost_microunits_per_1k_output_tokens,
        }


@dataclass(frozen=True, slots=True)
class EvaluationCard:
    card_id: str
    cost_unit: str
    model_manifest_sha256: str
    quality_report_sha256: str
    measurements: dict[str, EvaluationMeasurement]


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    domains: tuple[str, ...]
    independence_group: str
    data_residencies: tuple[str, ...]
    safety_tier: int
    grounding_mode: str
    model_manifest_path: Path
    gateway_command_path: Path
    evaluation_card_path: Path
    quality_report_path: Path


@dataclass(frozen=True, slots=True)
class Portfolio:
    portfolio_id: str
    cost_unit: str
    candidates: tuple[CandidateSpec, ...]


@dataclass(frozen=True, slots=True)
class RoutingTask:
    task_id: str
    domain: str
    data_residency: str
    minimum_safety_tier: int
    strategy: str
    minimum_independent_candidates: int
    required_modalities: frozenset[str]
    require_evidence: bool
    minimum_context_window_tokens: int
    minimum_quality_lower_bound: float
    minimum_reliability_lower_bound: float
    minimum_safety_lower_bound: float
    max_p95_latency_ms: int
    max_cost_microunits_per_1k_output_tokens: int


@dataclass(frozen=True, slots=True)
class CandidateRuntime:
    spec: CandidateSpec
    manifest: ModelManifest
    manifest_identity: FileIdentity
    command: GatewayCommandTemplate
    card: EvaluationCard
    card_identity: FileIdentity
    quality_report_identity: FileIdentity
    quality_report_runtime_identity_sha256: str


@dataclass(frozen=True, slots=True)
class EligibleCandidate:
    runtime: CandidateRuntime
    measurement: EvaluationMeasurement


@dataclass(frozen=True, slots=True)
class RoutingState:
    router_identity: FileIdentity
    portfolio: Portfolio
    portfolio_identity: FileIdentity
    task: RoutingTask
    task_identity: FileIdentity
    request: GenerationRequest
    request_identity: FileIdentity
    evidence: EvidenceBundle
    evidence_identity: FileIdentity
    required_request_modalities: frozenset[str]
    asset_registry_identity: FileIdentity | None
    runtimes: tuple[CandidateRuntime, ...]


def exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    unknown = sorted(str(key) for key in value if key not in expected)
    missing = sorted(expected.difference(value))
    require(not unknown, f"{field} contains unsupported fields: {', '.join(unknown)}")
    require(not missing, f"{field} is missing required fields: {', '.join(missing)}")


def parse_probability(value: object, field: str) -> float:
    require(type(value) in {int, float} and not isinstance(value, bool), f"{field} must be a finite number in [0, 1]")
    parsed = float(value)
    require(math.isfinite(parsed) and 0.0 <= parsed <= 1.0, f"{field} must be a finite number in [0, 1]")
    return parsed


def parse_bounded_int(value: object, field: str, minimum: int, maximum: int) -> int:
    require(type(value) is int and minimum <= value <= maximum,
            f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def parse_identifier_list(value: object, field: str, maximum: int) -> tuple[str, ...]:
    require(isinstance(value, list) and value, f"{field} must be a nonempty array")
    require(len(value) <= maximum, f"{field} exceeds {maximum} values")
    parsed = tuple(validate_identifier(item, f"{field}[{index}]") for index, item in enumerate(value))
    require(len(set(parsed)) == len(parsed), f"{field} contains duplicate values")
    return parsed


def parse_modalities(value: object, field: str) -> frozenset[str]:
    require(isinstance(value, list) and value, f"{field} must be a nonempty array")
    require(len(value) <= len(_ALLOWED_MODALITIES), f"{field} has too many modalities")
    parsed: list[str] = []
    for index, item in enumerate(value):
        require(isinstance(item, str) and item in _ALLOWED_MODALITIES,
                f"{field}[{index}] must be one of {', '.join(sorted(_ALLOWED_MODALITIES))}")
        parsed.append(item)
    require(len(set(parsed)) == len(parsed), f"{field} contains duplicate modalities")
    result = frozenset(parsed)
    require("text" in result, f"{field} must include text")
    return result


def resolve_declared_path(value: object, base: Path, field: str) -> Path:
    rendered = validate_unicode_text(value, field, maximum_bytes=16 * 1024)
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def load_json_mapping(path: Path, label: str, *, maximum_bytes: int = MAX_JSON_BYTES) -> tuple[dict[str, Any], FileIdentity]:
    identity = file_identity(path, label, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LMMPortfolioError(f"cannot parse {label} {path}: {error}") from error
    require(isinstance(value, Mapping), f"{label} must contain one JSON object")
    require_unchanged(path, identity, label)
    return dict(value), identity


def parse_measurement(value: object, field: str) -> EvaluationMeasurement:
    require(isinstance(value, Mapping), f"{field} must be an object")
    expected = {
        "domain",
        "suite_id",
        "suite_report_sha256",
        "cases",
        "quality_lower_bound",
        "reliability_lower_bound",
        "safety_lower_bound",
        "p95_latency_ms",
        "cost_microunits_per_1k_output_tokens",
    }
    exact_fields(value, expected, field)
    return EvaluationMeasurement(
        domain=validate_identifier(value["domain"], f"{field}.domain"),
        suite_id=validate_identifier(value["suite_id"], f"{field}.suite_id"),
        suite_report_sha256=validate_sha256(value["suite_report_sha256"], f"{field}.suite_report_sha256"),
        cases=parse_bounded_int(value["cases"], f"{field}.cases", 1, 2**63 - 1),
        quality_lower_bound=parse_probability(value["quality_lower_bound"], f"{field}.quality_lower_bound"),
        reliability_lower_bound=parse_probability(value["reliability_lower_bound"], f"{field}.reliability_lower_bound"),
        safety_lower_bound=parse_probability(value["safety_lower_bound"], f"{field}.safety_lower_bound"),
        p95_latency_ms=parse_bounded_int(value["p95_latency_ms"], f"{field}.p95_latency_ms", 1, 2**63 - 1),
        cost_microunits_per_1k_output_tokens=parse_bounded_int(
            value["cost_microunits_per_1k_output_tokens"],
            f"{field}.cost_microunits_per_1k_output_tokens",
            0,
            2**63 - 1,
        ),
    )


def parse_evaluation_card(path: Path) -> tuple[EvaluationCard, FileIdentity]:
    value, identity = load_json_mapping(path, "portfolio evaluation card")
    expected = {"schema_version", "id", "cost_unit", "model_manifest_sha256", "quality_report_sha256", "measurements"}
    exact_fields(value, expected, "portfolio evaluation card")
    require(value["schema_version"] == EVALUATION_CARD_SCHEMA,
            f"unsupported evaluation card schema: {value['schema_version']!r}")
    raw_measurements = value["measurements"]
    require(isinstance(raw_measurements, list) and raw_measurements, "portfolio evaluation card.measurements must be nonempty")
    require(len(raw_measurements) <= MAX_DOMAINS_PER_CANDIDATE,
            f"portfolio evaluation card.measurements exceeds {MAX_DOMAINS_PER_CANDIDATE}")
    measurements: dict[str, EvaluationMeasurement] = {}
    for index, raw in enumerate(raw_measurements):
        measurement = parse_measurement(raw, f"portfolio evaluation card.measurements[{index}]")
        require(measurement.domain not in measurements,
                "portfolio evaluation card has multiple measurements for one domain")
        measurements[measurement.domain] = measurement
    return EvaluationCard(
        card_id=validate_identifier(value["id"], "portfolio evaluation card.id"),
        cost_unit=validate_identifier(value["cost_unit"], "portfolio evaluation card.cost_unit"),
        model_manifest_sha256=validate_sha256(value["model_manifest_sha256"], "portfolio evaluation card.model_manifest_sha256"),
        quality_report_sha256=validate_sha256(value["quality_report_sha256"], "portfolio evaluation card.quality_report_sha256"),
        measurements=measurements,
    ), identity


def parse_candidate(value: object, index: int, base: Path) -> CandidateSpec:
    field = f"portfolio candidates[{index}]"
    require(isinstance(value, Mapping), f"{field} must be an object")
    expected = {
        "id",
        "domains",
        "independence_group",
        "data_residencies",
        "safety_tier",
        "grounding_mode",
        "model_manifest",
        "gateway_command_json",
        "evaluation_card",
        "quality_report",
    }
    exact_fields(value, expected, field)
    grounding_mode = value["grounding_mode"]
    require(isinstance(grounding_mode, str) and grounding_mode in GROUNDING_MODES,
            f"{field}.grounding_mode must be one of {', '.join(sorted(GROUNDING_MODES))}")
    return CandidateSpec(
        candidate_id=validate_identifier(value["id"], f"{field}.id"),
        domains=parse_identifier_list(value["domains"], f"{field}.domains", MAX_DOMAINS_PER_CANDIDATE),
        independence_group=validate_identifier(value["independence_group"], f"{field}.independence_group"),
        data_residencies=parse_identifier_list(value["data_residencies"], f"{field}.data_residencies", MAX_RESIDENCIES_PER_CANDIDATE),
        safety_tier=parse_bounded_int(value["safety_tier"], f"{field}.safety_tier", 0, 4),
        grounding_mode=grounding_mode,
        model_manifest_path=resolve_declared_path(value["model_manifest"], base, f"{field}.model_manifest"),
        gateway_command_path=resolve_declared_path(value["gateway_command_json"], base, f"{field}.gateway_command_json"),
        evaluation_card_path=resolve_declared_path(value["evaluation_card"], base, f"{field}.evaluation_card"),
        quality_report_path=resolve_declared_path(value["quality_report"], base, f"{field}.quality_report"),
    )


def parse_portfolio(path: Path) -> tuple[Portfolio, FileIdentity]:
    value, identity = load_json_mapping(path, "LMM portfolio")
    expected = {"schema_version", "id", "cost_unit", "candidates"}
    exact_fields(value, expected, "LMM portfolio")
    require(value["schema_version"] == PORTFOLIO_SCHEMA, f"unsupported portfolio schema: {value['schema_version']!r}")
    raw_candidates = value["candidates"]
    require(isinstance(raw_candidates, list) and raw_candidates, "LMM portfolio.candidates must be nonempty")
    require(len(raw_candidates) <= MAX_CANDIDATES, f"LMM portfolio.candidates exceeds {MAX_CANDIDATES}")
    candidates = tuple(parse_candidate(raw, index, path.resolve().parent) for index, raw in enumerate(raw_candidates))
    identifiers = [candidate.candidate_id for candidate in candidates]
    require(len(set(identifiers)) == len(identifiers), "LMM portfolio.candidates repeats an id")
    return Portfolio(
        portfolio_id=validate_identifier(value["id"], "LMM portfolio.id"),
        cost_unit=validate_identifier(value["cost_unit"], "LMM portfolio.cost_unit"),
        candidates=candidates,
    ), identity


def parse_task(path: Path) -> tuple[RoutingTask, FileIdentity]:
    value, identity = load_json_mapping(path, "LMM routing task")
    expected = {
        "schema_version",
        "id",
        "domain",
        "data_residency",
        "minimum_safety_tier",
        "strategy",
        "minimum_independent_candidates",
        "required_modalities",
        "require_evidence",
        "minimum_context_window_tokens",
        "minimum_quality_lower_bound",
        "minimum_reliability_lower_bound",
        "minimum_safety_lower_bound",
        "max_p95_latency_ms",
        "max_cost_microunits_per_1k_output_tokens",
    }
    exact_fields(value, expected, "LMM routing task")
    require(value["schema_version"] == TASK_SCHEMA, f"unsupported routing task schema: {value['schema_version']!r}")
    strategy = value["strategy"]
    require(isinstance(strategy, str) and strategy in _ALLOWED_STRATEGIES,
            f"LMM routing task.strategy must be one of {', '.join(sorted(_ALLOWED_STRATEGIES))}")
    panel_count = parse_bounded_int(
        value["minimum_independent_candidates"],
        "LMM routing task.minimum_independent_candidates",
        1,
        MAX_PANEL_CANDIDATES,
    )
    require((strategy == "single-best" and panel_count == 1) or (strategy == "independent-panel" and panel_count >= 2),
            "single-best requires one candidate; independent-panel requires at least two independent candidates")
    require(type(value["require_evidence"]) is bool, "LMM routing task.require_evidence must be boolean")
    return RoutingTask(
        task_id=validate_identifier(value["id"], "LMM routing task.id"),
        domain=validate_identifier(value["domain"], "LMM routing task.domain"),
        data_residency=validate_identifier(value["data_residency"], "LMM routing task.data_residency"),
        minimum_safety_tier=parse_bounded_int(value["minimum_safety_tier"], "LMM routing task.minimum_safety_tier", 0, 4),
        strategy=strategy,
        minimum_independent_candidates=panel_count,
        required_modalities=parse_modalities(value["required_modalities"], "LMM routing task.required_modalities"),
        require_evidence=value["require_evidence"],
        minimum_context_window_tokens=parse_bounded_int(
            value["minimum_context_window_tokens"],
            "LMM routing task.minimum_context_window_tokens",
            1,
            16_000_000,
        ),
        minimum_quality_lower_bound=parse_probability(
            value["minimum_quality_lower_bound"], "LMM routing task.minimum_quality_lower_bound"
        ),
        minimum_reliability_lower_bound=parse_probability(
            value["minimum_reliability_lower_bound"], "LMM routing task.minimum_reliability_lower_bound"),
        minimum_safety_lower_bound=parse_probability(value["minimum_safety_lower_bound"], "LMM routing task.minimum_safety_lower_bound"),
        max_p95_latency_ms=parse_bounded_int(value["max_p95_latency_ms"], "LMM routing task.max_p95_latency_ms", 1, 2**63 - 1),
        max_cost_microunits_per_1k_output_tokens=parse_bounded_int(
            value["max_cost_microunits_per_1k_output_tokens"],
            "LMM routing task.max_cost_microunits_per_1k_output_tokens",
            0,
            2**63 - 1,
        ),
    ), identity


def parse_quality_report(
    path: Path,
    expected_model_manifest_sha256: str,
    expected_report_sha256: str,
    expected_gateway_command_sha256: str,
) -> tuple[FileIdentity, str]:
    value, identity = load_json_mapping(path, "candidate held-out quality report")
    require(identity.sha256 == expected_report_sha256,
            "candidate held-out quality report does not match the evaluation card quality_report_sha256")
    require(value.get("schema_version") == QUALITY_REPORT_SCHEMA,
            "candidate held-out quality report has an unsupported schema")
    dependencies = value.get("dependencies")
    require(isinstance(dependencies, Mapping), "candidate held-out quality report has no dependency provenance")
    quality_command = dependencies.get("gateway_command")
    require(isinstance(quality_command, Mapping),
            "candidate held-out quality report has no gateway command identity")
    require(validate_sha256(quality_command.get("sha256"), "candidate held-out quality report gateway command sha256")
            == expected_gateway_command_sha256,
            "candidate held-out quality report was produced with another gateway command")
    quality_gate = value.get("quality_gate")
    configuration = value.get("configuration")
    require(isinstance(quality_gate, Mapping) and quality_gate.get("passed") is True,
            "candidate held-out quality report did not pass its release gate")
    require(isinstance(configuration, Mapping), "candidate held-out quality report has no configuration")
    run_provenance = configuration.get("run_provenance")
    require(isinstance(run_provenance, Mapping), "candidate held-out quality report has no run provenance")
    model_manifest_sha256 = validate_sha256(
        run_provenance.get("model_manifest_sha256"),
        "candidate held-out quality report run_provenance.model_manifest_sha256",
    )
    require(model_manifest_sha256 == expected_model_manifest_sha256,
            "candidate held-out quality report was produced by another model manifest")
    runtime_identity_sha256 = validate_sha256(
        run_provenance.get("runtime_identity_sha256"),
        "candidate held-out quality report run_provenance.runtime_identity_sha256",
    )
    return identity, runtime_identity_sha256


def load_candidate_runtime(spec: CandidateSpec, cost_unit: str) -> CandidateRuntime:
    manifest_identity = file_identity(spec.model_manifest_path, f"portfolio candidate {spec.candidate_id} model manifest", maximum_bytes=MAX_JSON_BYTES)
    manifest = ModelManifest.from_file(spec.model_manifest_path)
    require_unchanged(spec.model_manifest_path, manifest_identity, f"portfolio candidate {spec.candidate_id} model manifest")
    command = load_gateway_command_template(spec.gateway_command_path, label=f"portfolio candidate {spec.candidate_id} gateway command")
    require(command.grounding_mode == spec.grounding_mode,
            f"portfolio candidate {spec.candidate_id} grounding_mode does not match its gateway command")
    require_model_argument(command, spec.model_manifest_path, label=f"portfolio candidate {spec.candidate_id} gateway command")
    card, card_identity = parse_evaluation_card(spec.evaluation_card_path)
    require(card.cost_unit == cost_unit,
            f"portfolio candidate {spec.candidate_id} evaluation card cost_unit differs from the portfolio")
    require(card.model_manifest_sha256 == manifest_identity.sha256,
            f"portfolio candidate {spec.candidate_id} evaluation card binds another model manifest")
    quality_report_identity, runtime_identity_sha256 = parse_quality_report(
        spec.quality_report_path,
        manifest_identity.sha256,
        card.quality_report_sha256,
        command.identity.sha256,
    )
    return CandidateRuntime(
        spec=spec,
        manifest=manifest,
        manifest_identity=manifest_identity,
        command=command,
        card=card,
        card_identity=card_identity,
        quality_report_identity=quality_report_identity,
        quality_report_runtime_identity_sha256=runtime_identity_sha256,
    )


def load_request(path: Path) -> tuple[GenerationRequest, FileIdentity]:
    identity = file_identity(path, "routing generation request", maximum_bytes=MAX_JSON_BYTES)
    request = GenerationRequest.from_file(path)
    require_unchanged(path, identity, "routing generation request")
    return request, identity


def load_evidence(path: Path) -> tuple[EvidenceBundle, FileIdentity]:
    identity = file_identity(path, "routing evidence bundle", maximum_bytes=MAX_JSON_BYTES)
    evidence = EvidenceBundle.from_file(path)
    require_unchanged(path, identity, "routing evidence bundle")
    return evidence, identity


def request_modalities(request: GenerationRequest, asset_registry: AssetRegistry | None) -> frozenset[str]:
    modalities = {"text"}
    has_assets = any(part.kind == "asset" for message in request.messages for part in message.content)
    require(not has_assets or asset_registry is not None,
            "routing a request with media assets requires --asset-registry")
    if asset_registry is not None:
        modalities.update(record.modality for record in asset_registry.required_for_messages(request.messages).values())
    return frozenset(modalities)


def load_state(arguments: argparse.Namespace) -> RoutingState:
    router_path = Path(__file__).resolve()
    router_identity = file_identity(router_path, "portfolio router tool", maximum_bytes=MAX_JSON_BYTES)
    portfolio, portfolio_identity = parse_portfolio(arguments.portfolio.resolve())
    task, task_identity = parse_task(arguments.task.resolve())
    request, request_identity = load_request(arguments.request.resolve())
    evidence, evidence_identity = load_evidence(arguments.evidence.resolve())
    asset_registry_identity: FileIdentity | None = None
    registry: AssetRegistry | None = None
    if arguments.asset_registry is not None:
        asset_registry_identity = file_identity(arguments.asset_registry, "routing asset registry", maximum_bytes=MAX_JSON_BYTES)
        registry = AssetRegistry.from_jsonl(arguments.asset_registry)
        require_unchanged(arguments.asset_registry, asset_registry_identity, "routing asset registry")
    actual_modalities = request_modalities(request, registry)
    require(actual_modalities.issubset(task.required_modalities),
            "routing task.required_modalities does not cover modalities referenced by the generation request")
    if task.require_evidence:
        require(bool(evidence.items), "routing task requires evidence but the supplied evidence bundle is empty")
    runtimes = tuple(load_candidate_runtime(candidate, portfolio.cost_unit) for candidate in portfolio.candidates)
    return RoutingState(
        router_identity=router_identity,
        portfolio=portfolio,
        portfolio_identity=portfolio_identity,
        task=task,
        task_identity=task_identity,
        request=request,
        request_identity=request_identity,
        evidence=evidence,
        evidence_identity=evidence_identity,
        required_request_modalities=actual_modalities,
        asset_registry_identity=asset_registry_identity,
        runtimes=runtimes,
    )


def eligibility_reasons(state: RoutingState, runtime: CandidateRuntime) -> tuple[EvaluationMeasurement | None, list[str]]:
    task = state.task
    spec = runtime.spec
    measurement = runtime.card.measurements.get(task.domain)
    reasons: list[str] = []
    if task.domain not in spec.domains:
        reasons.append("domain_not_declared")
    if task.data_residency not in spec.data_residencies:
        reasons.append("data_residency_not_declared")
    if spec.safety_tier < task.minimum_safety_tier:
        reasons.append("safety_tier_below_requirement")
    if not task.required_modalities.issubset({name for name, supported in runtime.manifest.capabilities.items() if supported}):
        reasons.append("model_missing_required_modality")
    if runtime.manifest.context_window_tokens < task.minimum_context_window_tokens:
        reasons.append("context_window_below_requirement")
    if runtime.manifest.max_output_tokens < state.request.max_new_tokens:
        reasons.append("max_output_tokens_below_request")
    if task.require_evidence and spec.grounding_mode == "off":
        reasons.append("evidence_task_requires_active_grounding")
    if spec.grounding_mode == "require-citations" and not state.evidence.items:
        reasons.append("require_citations_mode_needs_nonempty_evidence")
    if measurement is None:
        reasons.append("no_target_domain_evaluation_measurement")
        return None, reasons
    if measurement.quality_lower_bound < task.minimum_quality_lower_bound:
        reasons.append("quality_lower_bound_below_requirement")
    if measurement.reliability_lower_bound < task.minimum_reliability_lower_bound:
        reasons.append("reliability_lower_bound_below_requirement")
    if measurement.safety_lower_bound < task.minimum_safety_lower_bound:
        reasons.append("safety_lower_bound_below_requirement")
    if measurement.p95_latency_ms > task.max_p95_latency_ms:
        reasons.append("p95_latency_above_requirement")
    if measurement.cost_microunits_per_1k_output_tokens > task.max_cost_microunits_per_1k_output_tokens:
        reasons.append("cost_above_requirement")
    return measurement, reasons


def ranking_key(candidate: EligibleCandidate) -> tuple[float, float, float, int, int, str]:
    measurement = candidate.measurement
    return (
        -measurement.quality_lower_bound,
        -measurement.reliability_lower_bound,
        -measurement.safety_lower_bound,
        measurement.p95_latency_ms,
        measurement.cost_microunits_per_1k_output_tokens,
        candidate.runtime.spec.candidate_id,
    )


def candidate_summary(runtime: CandidateRuntime, measurement: EvaluationMeasurement | None, reasons: Sequence[str], *, selected: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "id": runtime.spec.candidate_id,
        "independence_group": runtime.spec.independence_group,
        "selected": selected,
        "reasons": list(reasons),
        "model": {
            "id": runtime.manifest.model_id,
            "revision": runtime.manifest.revision,
            "manifest_sha256": runtime.manifest_identity.sha256,
            "capabilities": dict(runtime.manifest.capabilities),
            "context_window_tokens": runtime.manifest.context_window_tokens,
            "max_output_tokens": runtime.manifest.max_output_tokens,
        },
        "gateway_command": runtime.command.identity.to_mapping(),
        "grounding_mode": runtime.spec.grounding_mode,
        "evaluation_card": runtime.card_identity.to_mapping(),
        "heldout_quality_report": runtime.quality_report_identity.to_mapping(),
        "heldout_runtime_identity_sha256": runtime.quality_report_runtime_identity_sha256,
    }
    if measurement is not None:
        result["target_domain_measurement"] = measurement.to_mapping()
    return result


def select_candidates(state: RoutingState) -> tuple[tuple[EligibleCandidate, ...], list[dict[str, object]]]:
    eligible: list[EligibleCandidate] = []
    observations: list[tuple[CandidateRuntime, EvaluationMeasurement | None, list[str]]] = []
    for runtime in state.runtimes:
        measurement, reasons = eligibility_reasons(state, runtime)
        observations.append((runtime, measurement, reasons))
        if measurement is not None and not reasons:
            eligible.append(EligibleCandidate(runtime=runtime, measurement=measurement))
    eligible.sort(key=ranking_key)
    require(eligible, "no portfolio candidate meets the routing task hard constraints and measured thresholds")
    if state.task.strategy == "single-best":
        selected = tuple(eligible[:1])
    else:
        selected_list: list[EligibleCandidate] = []
        independence_groups: set[str] = set()
        for candidate in eligible:
            group = candidate.runtime.spec.independence_group
            if group in independence_groups:
                continue
            selected_list.append(candidate)
            independence_groups.add(group)
            if len(selected_list) == state.task.minimum_independent_candidates:
                break
        require(len(selected_list) == state.task.minimum_independent_candidates,
                "not enough independently grouped eligible portfolio candidates for the requested panel")
        selected = tuple(selected_list)
    selected_ids = {candidate.runtime.spec.candidate_id for candidate in selected}
    summaries: list[dict[str, object]] = []
    for runtime, measurement, reasons in observations:
        selected_here = runtime.spec.candidate_id in selected_ids
        summary_reasons = list(reasons)
        if not selected_here and not summary_reasons:
            if state.task.strategy == "independent-panel" and runtime.spec.independence_group in {
                item.runtime.spec.independence_group for item in selected
            }:
                summary_reasons.append("not_selected_duplicate_independence_group")
            else:
                summary_reasons.append("not_selected_after_deterministic_ranking")
        summaries.append(candidate_summary(runtime, measurement, summary_reasons, selected=selected_here))
    return selected, summaries


def state_identities(state: RoutingState) -> tuple[FileIdentity, ...]:
    identities: list[FileIdentity] = [
        state.router_identity,
        state.portfolio_identity,
        state.task_identity,
        state.request_identity,
        state.evidence_identity,
    ]
    if state.asset_registry_identity is not None:
        identities.append(state.asset_registry_identity)
    for runtime in state.runtimes:
        identities.extend((
            runtime.manifest_identity,
            runtime.command.identity,
            runtime.card_identity,
            runtime.quality_report_identity,
        ))
    return tuple(identities)


def command_execution_path(value: str) -> Path:
    """Resolve one path exactly as the selected command will from repository root."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[1] / candidate
    return candidate.resolve()


def output_protection_paths(state: RoutingState) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return files and package roots that a portfolio report must never replace.

    The gateway has its own output-collision checks, but a portfolio owns the
    final report path and can otherwise publish it after a successful model run.
    Protecting model/asset roots also protects their members, rather than only
    the manifest files already present in the route-decision provenance.
    """
    files = [Path(identity.path).resolve() for identity in state_identities(state)]
    roots: list[Path] = []
    for runtime in state.runtimes:
        command = runtime.command.command
        for item in command:
            if item.startswith("-") or item in {"{request}", "{evidence}"}:
                continue
            candidate = command_execution_path(item)
            if candidate.is_file():
                files.append(candidate)
        for option in ("--model-root", "--asset-root"):
            raw_path = direct_option_value(command, option, f"portfolio candidate {runtime.spec.candidate_id} gateway command")
            if raw_path is not None:
                roots.append(command_execution_path(raw_path))
        if runtime.manifest.tokenizer.kind == "tokenizers-json":
            assert runtime.manifest.tokenizer.relative_path is not None
            # The tokenizer is package-relative to its manifest, not to model-root.
            files.append((runtime.spec.model_manifest_path.parent / runtime.manifest.tokenizer.relative_path).resolve())
    return tuple(files), tuple(roots)


def ensure_output_safe(path: Path, state: RoutingState) -> None:
    output = path.resolve()
    files, roots = output_protection_paths(state)
    for dependency in files:
        require(output != dependency,
                f"--output must not overwrite immutable routing dependency {dependency}")
    for root in roots:
        try:
            output.relative_to(root)
        except ValueError:
            continue
        raise LMMPortfolioError(f"--output must not be published inside immutable package root {root}")


def recheck_state(state: RoutingState) -> None:
    labels = (
        "portfolio router tool",
        "LMM portfolio",
        "LMM routing task",
        "routing generation request",
        "routing evidence bundle",
    )
    for identity, label in zip(state_identities(state)[:5], labels, strict=True):
        require_unchanged(Path(identity.path), identity, label)
    if state.asset_registry_identity is not None:
        require_unchanged(Path(state.asset_registry_identity.path), state.asset_registry_identity, "routing asset registry")
    for runtime in state.runtimes:
        require_unchanged(Path(runtime.manifest_identity.path), runtime.manifest_identity,
                          f"portfolio candidate {runtime.spec.candidate_id} model manifest")
        require_unchanged(Path(runtime.command.identity.path), runtime.command.identity,
                          f"portfolio candidate {runtime.spec.candidate_id} gateway command")
        require_unchanged(Path(runtime.card_identity.path), runtime.card_identity,
                          f"portfolio candidate {runtime.spec.candidate_id} evaluation card")
        require_unchanged(Path(runtime.quality_report_identity.path), runtime.quality_report_identity,
                          f"portfolio candidate {runtime.spec.candidate_id} held-out quality report")


def build_decision(state: RoutingState, selected: Sequence[EligibleCandidate], candidates: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "schema_version": DECISION_SCHEMA,
        "dependencies": {
            "router_tool": state.router_identity.to_mapping(),
            "portfolio": state.portfolio_identity.to_mapping(),
            "routing_task": state.task_identity.to_mapping(),
            "request": state.request_identity.to_mapping(),
            "evidence": state.evidence_identity.to_mapping(),
            "asset_registry": None if state.asset_registry_identity is None else state.asset_registry_identity.to_mapping(),
        },
        "task": {
            "id": state.task.task_id,
            "domain": state.task.domain,
            "data_residency": state.task.data_residency,
            "strategy": state.task.strategy,
            "minimum_independent_candidates": state.task.minimum_independent_candidates,
            "required_modalities": sorted(state.task.required_modalities),
            "request_modalities": sorted(state.required_request_modalities),
            "require_evidence": state.task.require_evidence,
            "thresholds": {
                "minimum_safety_tier": state.task.minimum_safety_tier,
                "minimum_context_window_tokens": state.task.minimum_context_window_tokens,
                "minimum_quality_lower_bound": state.task.minimum_quality_lower_bound,
                "minimum_reliability_lower_bound": state.task.minimum_reliability_lower_bound,
                "minimum_safety_lower_bound": state.task.minimum_safety_lower_bound,
                "max_p95_latency_ms": state.task.max_p95_latency_ms,
                "max_cost_microunits_per_1k_output_tokens": state.task.max_cost_microunits_per_1k_output_tokens,
            },
        },
        "portfolio": {"id": state.portfolio.portfolio_id, "cost_unit": state.portfolio.cost_unit},
        "selected": [
            candidate_summary(item.runtime, item.measurement, (), selected=True)
            for item in selected
        ],
        "candidates": list(candidates),
        "quality_boundary": (
            "Selection proves only that the chosen candidate meets operator-declared, provenance-bound routing constraints "
            "and a passed held-out citation/abstention report. It does not prove broad factuality, semantic reasoning, "
            "safety, benchmark superiority, or universal intelligence."
        ),
    }


def prepare_routing(arguments: argparse.Namespace) -> tuple[RoutingState, tuple[EligibleCandidate, ...], dict[str, object]]:
    state = load_state(arguments)
    ensure_output_safe(arguments.output, state)
    selected, candidates = select_candidates(state)
    recheck_state(state)
    decision = build_decision(state, selected, candidates)
    return state, selected, decision


def route(arguments: argparse.Namespace) -> int:
    _, selected, decision = prepare_routing(arguments)
    write_json_atomic(arguments.output, decision)
    print(json.dumps({"report": decision, "selected_candidates": len(selected)}, ensure_ascii=False, sort_keys=True))
    return 0


def copy_private_input(source: Path, identity: FileIdentity, destination: Path, label: str) -> None:
    try:
        shutil.copyfile(source, destination)
    except OSError as error:
        raise LMMPortfolioError(f"cannot prepare private {label} input: {error}") from error
    copied = file_identity(destination, f"private {label} input", maximum_bytes=MAX_JSON_BYTES)
    require(copied.sha256 == identity.sha256 and copied.bytes == identity.bytes,
            f"{label} changed while preparing the private gateway input")


def gateway_runtime_identity(identities: Mapping[str, Any]) -> str:
    """Match the stable gateway dependency digest recorded by quality evaluation."""
    missing = [field for field in _RUNTIME_IDENTITY_FIELDS if field not in identities]
    require(not missing,
            "selected gateway report identities is missing runtime fields: " + ", ".join(missing))
    stable = {field: identities[field] for field in _RUNTIME_IDENTITY_FIELDS}
    return sha256_bytes(canonical_json_bytes(stable))


def parse_gateway_result(
    stdout: bytes,
    selected: EligibleCandidate,
    request_identity: FileIdentity,
    evidence_identity: FileIdentity,
    evidence: EvidenceBundle,
    asset_registry_identity: FileIdentity | None,
) -> tuple[str, Mapping[str, Any], str]:
    require(len(stdout) <= MAX_GATEWAY_STDOUT_BYTES,
            f"selected gateway stdout exceeds {MAX_GATEWAY_STDOUT_BYTES} bytes")
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LMMPortfolioError(f"selected gateway did not emit UTF-8 --print-json output: {error}") from error
    require(isinstance(payload, Mapping) and set(payload) == {"response", "report"},
            "selected gateway output must contain exactly response and report")
    response = validate_unicode_text(payload["response"], "selected gateway response", maximum_bytes=MAX_TEXT_BYTES)
    report = payload["report"]
    require(isinstance(report, Mapping) and report.get("schema_version") == GENERATION_REPORT_SCHEMA,
            "selected gateway report has an unsupported schema")
    response_mapping = report.get("response")
    require(isinstance(response_mapping, Mapping), "selected gateway report has no response provenance")
    response_sha256 = sha256_bytes(response.encode("utf-8"))
    require(response_mapping.get("sha256") == response_sha256,
            "selected gateway response does not match its provenance hash")
    identities = report.get("identities")
    grounding = report.get("grounding")
    require(isinstance(identities, Mapping) and isinstance(grounding, Mapping),
            "selected gateway report has no immutable model or grounding provenance")
    report_request = identities.get("request")
    report_evidence = identities.get("evidence")
    require(isinstance(report_request, Mapping) and isinstance(report_evidence, Mapping),
            "selected gateway report has no request/evidence provenance")
    require(validate_sha256(report_request.get("sha256"), "selected gateway report request sha256") == request_identity.sha256,
            "selected gateway did not use the routed immutable request")
    require(validate_sha256(report_evidence.get("sha256"), "selected gateway report evidence sha256") == evidence_identity.sha256,
            "selected gateway did not use the routed immutable evidence bundle")
    report_asset_registry = identities.get("asset_registry")
    if asset_registry_identity is None:
        require(report_asset_registry is None,
                "selected gateway used an asset registry that was not bound by the routing request")
    else:
        require(isinstance(report_asset_registry, Mapping),
                "selected gateway report has no asset registry provenance")
        require(validate_sha256(report_asset_registry.get("sha256"), "selected gateway report asset registry sha256")
                == asset_registry_identity.sha256,
                "selected gateway did not use the routed immutable asset registry")
    report_evidence_ids = report.get("evidence_document_ids")
    require(report_evidence_ids == [item.document_id for item in evidence.items],
            "selected gateway evidence document IDs differ from the routed evidence bundle")
    model_manifest = identities.get("model_manifest")
    require(isinstance(model_manifest, Mapping), "selected gateway report has no model manifest identity")
    report_model_sha256 = validate_sha256(model_manifest.get("sha256"), "selected gateway report model manifest sha256")
    require(report_model_sha256 == selected.runtime.manifest_identity.sha256,
            "selected gateway generated with a model manifest other than the routed candidate")
    require(gateway_runtime_identity(identities) == selected.runtime.quality_report_runtime_identity_sha256,
            "selected gateway runtime dependencies differ from the held-out quality report")
    require(grounding.get("mode") == selected.runtime.spec.grounding_mode,
            "selected gateway grounding mode differs from the routed candidate")
    return response, report, response_sha256


def generate(arguments: argparse.Namespace) -> int:
    state, selected, decision = prepare_routing(arguments)
    require(state.task.strategy == "single-best" and len(selected) == 1,
            "generate supports only single-best routing; use route for an independent-panel plan and operator-owned adjudication")
    choice = selected[0]
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vegai-lmm-portfolio-") as directory:
        workspace = Path(directory)
        private_request = workspace / "request.json"
        private_evidence = workspace / "evidence.json"
        copy_private_input(arguments.request.resolve(), state.request_identity, private_request, "request")
        copy_private_input(arguments.evidence.resolve(), state.evidence_identity, private_evidence, "evidence")
        try:
            completed = subprocess.run(
                expand_gateway_command(choice.runtime.command, private_request, private_evidence),
                cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=arguments.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise LMMPortfolioError(f"selected gateway exceeded {arguments.timeout_seconds} seconds") from error
        except OSError as error:
            raise LMMPortfolioError(f"cannot launch selected gateway: {error}") from error
    elapsed_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise LMMPortfolioError(
            "selected gateway failed; "
            f"stdout-sha256={sha256_bytes(completed.stdout)}, stderr-sha256={sha256_bytes(completed.stderr)}, "
            f"exit={completed.returncode}"
        )
    response, gateway_report, response_sha256 = parse_gateway_result(
        completed.stdout,
        choice,
        state.request_identity,
        state.evidence_identity,
        state.evidence,
        state.asset_registry_identity,
    )
    recheck_state(state)
    report: dict[str, object] = {
        "schema_version": GENERATION_SCHEMA,
        "routing_decision": decision,
        "generation": {
            "candidate_id": choice.runtime.spec.candidate_id,
            "gateway_report_sha256": sha256_bytes(canonical_json_bytes(gateway_report)),
            "model_manifest_sha256": choice.runtime.manifest_identity.sha256,
            "runtime_identity_sha256": choice.runtime.quality_report_runtime_identity_sha256,
            "backend": gateway_report.get("backend", {}).get("name") if isinstance(gateway_report.get("backend"), Mapping) else None,
            "elapsed_seconds": round(elapsed_seconds, 9),
        },
        "response": {
            "sha256": response_sha256,
            "bytes": len(response.encode("utf-8")),
            "text": response if arguments.include_response_in_report else None,
        },
        "quality_boundary": (
            "A routed generation is a real response from the selected configured model, not a synthetic answer. "
            "The routing/evidence contracts do not prove factuality, safety, or universal superiority."
        ),
    }
    write_json_atomic(arguments.output, report)
    if arguments.print_json:
        print(json.dumps({"response": response, "report": report}, ensure_ascii=False, sort_keys=True))
    else:
        print(response)
    return 0


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--portfolio", required=True, type=Path, help="strict vegai-lmm-portfolio/v1 JSON object")
    parser.add_argument("--task", required=True, type=Path, help="strict vegai-lmm-routing-task/v1 JSON object")
    parser.add_argument("--request", required=True, type=Path, help="real vegai-lmm-request/v1 JSON object")
    parser.add_argument("--evidence", required=True, type=Path, help="real vegai-lmm-evidence/v1 bundle; may be empty only when task permits it")
    parser.add_argument("--asset-registry", type=Path, help="required when the request references image/audio/video asset IDs")
    parser.add_argument("--output", required=True, type=Path, help="atomic decision/generation report path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    route_parser = commands.add_parser("route", help="emit a provenance-bound best-model or independent-panel decision")
    add_common_arguments(route_parser)
    generate_parser = commands.add_parser("generate", help="route then invoke one selected model without fabricating an answer")
    add_common_arguments(generate_parser)
    generate_parser.add_argument("--timeout-seconds", type=parse_positive_float, default=DEFAULT_TIMEOUT_SECONDS)
    generate_parser.add_argument("--include-response-in-report", action="store_true",
                                 help="response text is sensitive and omitted from the report by default")
    generate_parser.add_argument("--print-json", action="store_true", help="emit response plus report JSON instead of response text")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        return route(arguments) if arguments.command == "route" else generate(arguments)
    except (LMMContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"lmm portfolio: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
