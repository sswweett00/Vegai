#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed held-out regression comparison for continuous HDC pretraining.

A normal held-out quality gate answers whether one checkpoint meets an absolute
policy.  It cannot answer whether a later continual-learning checkpoint forgot
behaviour that an earlier checkpoint had.  This tool validates two independently
produced ``pretraining_quality_evaluation.py`` reports, recomputes their public
metrics from HMAC-redacted case rows, and applies an immutable regression
policy.  It deliberately writes a completed *failed* comparison report when a
quality regression is observed so a controller can retain audit evidence while
refusing to commit/publish the candidate checkpoint.

It does not train, select labels, or claim general model quality.  Meaningful
results require operator-owned held-out annotations and a regression policy
calibrated for the target domain.  Repository synthetic fixtures only exercise
the control path and a controlled topology diagnostic.
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
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

from pretraining_quality_evaluation import (
    MAX_CASES,
    PretrainingQualityEvaluationError,
    QualityPolicy,
    evaluator_binding,
    gate_failures,
    wilson_lower_bound_95,
)


ROOT = Path(__file__).resolve().parents[1]
REGRESSION_POLICY_SCHEMA = "vegai-continuous-quality-regression-policy/v1"
REPORT_SCHEMA = "vegai-continuous-quality-regression-report/v1"
QUALITY_REPORT_SCHEMA = "vegai-pretraining-quality-report/v1"
MAX_REPORT_BYTES = 32 * 1024 * 1024
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EPSILON = 1.0e-12


class ContinuousQualityGateError(RuntimeError):
    """Raised for invalid evidence, policy, or comparison invocation."""


@dataclass(frozen=True)
class RegressionPolicy:
    """Explicit maximum degradation tolerances for one frozen case set."""

    policy_id: str
    maximum_top1_accuracy_drop: float
    maximum_top1_wilson_lower_bound_95_drop: float
    maximum_mean_bipolar_similarity_drop: float
    maximum_minimum_bipolar_similarity_drop: float
    maximum_incorrect_top1_cases_increase: int
    maximum_top1_hit_regressions: int
    maximum_case_similarity_drop: float
    maximum_case_similarity_regressions: int

    def requirements(self) -> dict[str, object]:
        return {
            "maximum_top1_accuracy_drop": self.maximum_top1_accuracy_drop,
            "maximum_top1_wilson_lower_bound_95_drop": self.maximum_top1_wilson_lower_bound_95_drop,
            "maximum_mean_bipolar_similarity_drop": self.maximum_mean_bipolar_similarity_drop,
            "maximum_minimum_bipolar_similarity_drop": self.maximum_minimum_bipolar_similarity_drop,
            "maximum_incorrect_top1_cases_increase": self.maximum_incorrect_top1_cases_increase,
            "maximum_top1_hit_regressions": self.maximum_top1_hit_regressions,
            "maximum_case_similarity_drop": self.maximum_case_similarity_drop,
            "maximum_case_similarity_regressions": self.maximum_case_similarity_regressions,
        }


@dataclass(frozen=True)
class ReportSummary:
    """Trusted public summary derived from one evaluator report."""

    report_sha256: str
    report_bytes: int
    checkpoint_sha256: str
    quality_gate_passed: bool
    evaluation_id_hmac_sha256: str
    case_count: int
    metrics: dict[str, object]
    cases: tuple[dict[str, object], ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuousQualityGateError(message)


def require_sha256(value: object, label: str) -> str:
    require(isinstance(value, str) and bool(HEX64.fullmatch(value.lower())), f"{label} must be SHA-256 hex")
    return value.lower()


def finite_number(value: object, label: str, minimum: float, maximum: float) -> float:
    require(type(value) in {int, float}, f"{label} must be numeric")
    parsed = float(value)
    require(math.isfinite(parsed) and minimum <= parsed <= maximum,
            f"{label} must be finite and in [{minimum:g}, {maximum:g}]")
    return parsed


def nonnegative_integer(value: object, label: str, maximum: int = MAX_CASES) -> int:
    require(type(value) is int and 0 <= value <= maximum, f"{label} must be an integer in [0, {maximum}]")
    return value


def strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ContinuousQualityGateError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContinuousQualityGateError(f"cannot parse {label}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def read_snapshot(path: Path, label: str, maximum_bytes: int) -> tuple[bytes, str, int]:
    try:
        before = path.stat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(), f"{label} is not a regular non-symlink file")
        require(0 < before.st_size <= maximum_bytes, f"{label} is outside byte bound 1..{maximum_bytes}")
        raw = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise ContinuousQualityGateError(f"cannot read {label} {path}: {error}") from error
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        and len(raw) == before.st_size,
        f"{label} changed while it was read",
    )
    return raw, hashlib.sha256(raw).hexdigest(), len(raw)


def parse_regression_policy(value: Mapping[str, object]) -> RegressionPolicy:
    """Strictly parse an already snapshotted policy object."""

    expected = {
        "schema_version",
        "policy_id",
        "maximum_top1_accuracy_drop",
        "maximum_top1_wilson_lower_bound_95_drop",
        "maximum_mean_bipolar_similarity_drop",
        "maximum_minimum_bipolar_similarity_drop",
        "maximum_incorrect_top1_cases_increase",
        "maximum_top1_hit_regressions",
        "maximum_case_similarity_drop",
        "maximum_case_similarity_regressions",
    }
    require(set(value) == expected,
            f"quality regression policy fields differ: expected {sorted(expected)}, got {sorted(value)}")
    require(value["schema_version"] == REGRESSION_POLICY_SCHEMA,
            f"unsupported quality regression policy schema {value['schema_version']!r}")
    policy_id = value["policy_id"]
    require(isinstance(policy_id, str) and bool(IDENTIFIER.fullmatch(policy_id)),
            "quality regression policy.policy_id is invalid")
    return RegressionPolicy(
        policy_id=policy_id,
        maximum_top1_accuracy_drop=finite_number(
            value["maximum_top1_accuracy_drop"], "maximum_top1_accuracy_drop", 0.0, 1.0
        ),
        maximum_top1_wilson_lower_bound_95_drop=finite_number(
            value["maximum_top1_wilson_lower_bound_95_drop"],
            "maximum_top1_wilson_lower_bound_95_drop", 0.0, 1.0,
        ),
        maximum_mean_bipolar_similarity_drop=finite_number(
            value["maximum_mean_bipolar_similarity_drop"],
            "maximum_mean_bipolar_similarity_drop", 0.0, 2.0,
        ),
        maximum_minimum_bipolar_similarity_drop=finite_number(
            value["maximum_minimum_bipolar_similarity_drop"],
            "maximum_minimum_bipolar_similarity_drop", 0.0, 2.0,
        ),
        maximum_incorrect_top1_cases_increase=nonnegative_integer(
            value["maximum_incorrect_top1_cases_increase"], "maximum_incorrect_top1_cases_increase"
        ),
        maximum_top1_hit_regressions=nonnegative_integer(
            value["maximum_top1_hit_regressions"], "maximum_top1_hit_regressions"
        ),
        maximum_case_similarity_drop=finite_number(
            value["maximum_case_similarity_drop"], "maximum_case_similarity_drop", 0.0, 2.0
        ),
        maximum_case_similarity_regressions=nonnegative_integer(
            value["maximum_case_similarity_regressions"], "maximum_case_similarity_regressions"
        ),
    )


def load_regression_policy(path: Path) -> RegressionPolicy:
    raw, _, _ = read_snapshot(path, "quality regression policy", 1024 * 1024)
    return parse_regression_policy(strict_json(raw, "quality regression policy"))


def regression_policy_binding(path: Path) -> dict[str, object]:
    """Validate and identify a policy before a controller starts mutable work."""

    raw, digest, byte_count = read_snapshot(path, "quality regression policy", 1024 * 1024)
    policy = parse_regression_policy(strict_json(raw, "quality regression policy"))
    return {
        "schema_version": REGRESSION_POLICY_SCHEMA,
        "policy_id": policy.policy_id,
        "sha256": digest,
        "bytes": byte_count,
        "requirements": policy.requirements(),
    }


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            except OSError:
                pass
            finally:
                os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def report_number(value: object, label: str, minimum: float, maximum: float) -> float:
    return finite_number(value, f"quality evaluator report {label}", minimum, maximum)


def require_close(actual: object, expected: float, label: str) -> None:
    observed = report_number(actual, label, -1.0, 1.0)
    require(math.isclose(observed, expected, rel_tol=0.0, abs_tol=EPSILON),
            f"quality evaluator report {label} disagrees with case rows")


def read_validated_report(
    path: Path,
    quality_binding: Mapping[str, object],
    binary_sha256: str,
    checkpoint_sha256: str,
) -> ReportSummary:
    raw, report_digest, report_bytes = read_snapshot(path, "quality evaluator report", MAX_REPORT_BYTES)
    report = strict_json(raw, "quality evaluator report")
    require(report.get("schema_version") == QUALITY_REPORT_SCHEMA,
            "quality evaluator report has an unsupported schema")
    require(report.get("status") == "completed", "quality evaluator report is not completed")
    evaluator = report.get("evaluator")
    provenance = report.get("provenance")
    evaluation = report.get("evaluation")
    gate = report.get("quality_gate")
    metrics = report.get("metrics")
    rows = report.get("case_results")
    require(all(isinstance(item, dict) for item in (evaluator, provenance, evaluation, gate, metrics)),
            "quality evaluator report omitted required object sections")
    require(isinstance(rows, list), "quality evaluator report omitted case_results")
    assert isinstance(evaluator, dict) and isinstance(provenance, dict)
    assert isinstance(evaluation, dict) and isinstance(gate, dict) and isinstance(metrics, dict)

    require(evaluator.get("sha256") == quality_binding.get("evaluator_source_sha256"),
            "quality evaluator report is bound to a different evaluator")
    expected_provenance = {
        "pretrain_binary": binary_sha256,
        "checkpoint": checkpoint_sha256,
        "case_manifest": quality_binding.get("case_manifest_sha256"),
        "policy": quality_binding.get("policy_sha256"),
    }
    for name, expected_digest in expected_provenance.items():
        value = provenance.get(name)
        require(isinstance(value, dict) and value.get("sha256") == expected_digest,
                f"quality evaluator report has wrong {name} provenance")

    expected_requirements = quality_binding.get("requirements")
    expected_cases = quality_binding.get("case_count")
    require(isinstance(expected_requirements, dict) and type(expected_cases) is int and expected_cases > 0,
            "quality binding is malformed")
    require(gate.get("requirements") == expected_requirements,
            "quality evaluator report requirements differ from immutable binding")
    require(evaluation.get("declared_split") == "held-out", "quality evaluator report is not held-out")
    require(evaluation.get("case_count") == expected_cases and len(rows) == expected_cases,
            "quality evaluator report case count differs from immutable binding")
    evaluation_hmac = require_sha256(evaluation.get("evaluation_id_hmac_sha256"), "evaluation ID HMAC")

    result_fields = {
        "case_number", "case_id_hmac_sha256", "query_token_hmac_sha256",
        "acceptable_neighbour_set_hmac_sha256", "observed_neighbour_hmac_sha256",
        "top1_acceptable", "bipolar_similarity",
    }
    trusted_rows: list[dict[str, object]] = []
    correct = 0
    similarities: list[float] = []
    for number, row in enumerate(rows, start=1):
        require(isinstance(row, dict) and set(row) == result_fields,
                "quality evaluator report has malformed/private case row")
        require(type(row.get("case_number")) is int and row["case_number"] == number,
                "quality evaluator report case numbering is not contiguous")
        identifiers: dict[str, object] = {}
        for field in (
            "case_id_hmac_sha256", "query_token_hmac_sha256",
            "acceptable_neighbour_set_hmac_sha256", "observed_neighbour_hmac_sha256",
        ):
            identifiers[field] = require_sha256(row.get(field), f"quality evaluator report {field}")
        acceptable = row.get("top1_acceptable")
        require(type(acceptable) is bool, "quality evaluator report top1_acceptable must be boolean")
        similarity = report_number(row.get("bipolar_similarity"), "case similarity", -1.0, 1.0)
        correct += int(acceptable)
        similarities.append(similarity)
        trusted_rows.append({
            "case_number": number,
            **identifiers,
            "top1_acceptable": acceptable,
            "bipolar_similarity": similarity,
        })

    total = len(trusted_rows)
    incorrect = total - correct
    top1_accuracy = correct / total
    top1_wilson = wilson_lower_bound_95(correct, total)
    mean_similarity = math.fsum(similarities) / total
    minimum_similarity = min(similarities)
    maximum_similarity = max(similarities)
    require(type(metrics.get("correct_top1_cases")) is int and metrics["correct_top1_cases"] == correct,
            "quality evaluator report correct-case total disagrees with case rows")
    require(type(metrics.get("incorrect_top1_cases")) is int and metrics["incorrect_top1_cases"] == incorrect,
            "quality evaluator report incorrect-case total disagrees with case rows")
    require_close(metrics.get("top1_accuracy"), top1_accuracy, "top1_accuracy")
    require_close(metrics.get("top1_wilson_lower_bound_95"), top1_wilson, "top1_wilson_lower_bound_95")
    require_close(metrics.get("mean_bipolar_similarity"), mean_similarity, "mean_bipolar_similarity")
    require_close(metrics.get("minimum_bipolar_similarity"), minimum_similarity, "minimum_bipolar_similarity")
    require_close(metrics.get("maximum_bipolar_similarity"), maximum_similarity, "maximum_bipolar_similarity")

    policy = QualityPolicy(
        policy_id="comparison-bound-quality-policy",
        minimum_cases=expected_requirements["minimum_cases"],
        minimum_top1_accuracy=expected_requirements["minimum_top1_accuracy"],
        minimum_top1_wilson_lower_bound_95=expected_requirements["minimum_top1_wilson_lower_bound_95"],
        minimum_mean_similarity=expected_requirements["minimum_mean_similarity"],
        minimum_case_similarity=expected_requirements["minimum_case_similarity"],
        maximum_incorrect_cases=expected_requirements["maximum_incorrect_cases"],
    )
    failures = gate_failures(policy, total, top1_accuracy, top1_wilson, mean_similarity, minimum_similarity, incorrect)
    require(gate.get("failures") == failures and gate.get("passed") is (not failures),
            "quality evaluator report gate result disagrees with recomputed metrics")
    return ReportSummary(
        report_sha256=report_digest,
        report_bytes=report_bytes,
        checkpoint_sha256=checkpoint_sha256,
        quality_gate_passed=gate["passed"],
        evaluation_id_hmac_sha256=evaluation_hmac,
        case_count=total,
        metrics={
            "correct_top1_cases": correct,
            "incorrect_top1_cases": incorrect,
            "top1_accuracy": top1_accuracy,
            "top1_wilson_lower_bound_95": top1_wilson,
            "mean_bipolar_similarity": mean_similarity,
            "minimum_bipolar_similarity": minimum_similarity,
            "maximum_bipolar_similarity": maximum_similarity,
        },
        cases=tuple(trusted_rows),
    )


def regular_checkpoint_digest(path: Path, label: str) -> str:
    try:
        before = path.stat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(), f"{label} is not a regular non-symlink file")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
    except OSError as error:
        raise ContinuousQualityGateError(f"cannot hash {label}: {error}") from error
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"{label} changed while it was hashed",
    )
    return digest.hexdigest()


def comparison_document(
    baseline: ReportSummary,
    candidate: ReportSummary,
    policy: RegressionPolicy,
    regression_binding: Mapping[str, object],
) -> dict[str, object]:
    require(baseline.case_count == candidate.case_count,
            "baseline/candidate reports have different case counts")
    require(baseline.evaluation_id_hmac_sha256 == candidate.evaluation_id_hmac_sha256,
            "baseline/candidate reports use different redaction/evaluation namespaces")
    failures: list[str] = []
    if not baseline.quality_gate_passed:
        failures.append("baseline checkpoint did not pass its immutable absolute held-out quality policy")
    if not candidate.quality_gate_passed:
        failures.append("candidate checkpoint did not pass its immutable absolute held-out quality policy")

    for index, (before, after) in enumerate(zip(baseline.cases, candidate.cases, strict=True), start=1):
        for identifier in (
            "case_number", "case_id_hmac_sha256", "query_token_hmac_sha256", "acceptable_neighbour_set_hmac_sha256",
        ):
            require(before[identifier] == after[identifier],
                    f"baseline/candidate case identity differs at position {index} ({identifier})")

    baseline_metrics = baseline.metrics
    candidate_metrics = candidate.metrics
    accuracy_drop = float(baseline_metrics["top1_accuracy"]) - float(candidate_metrics["top1_accuracy"])
    wilson_drop = (
        float(baseline_metrics["top1_wilson_lower_bound_95"])
        - float(candidate_metrics["top1_wilson_lower_bound_95"])
    )
    mean_similarity_drop = (
        float(baseline_metrics["mean_bipolar_similarity"])
        - float(candidate_metrics["mean_bipolar_similarity"])
    )
    minimum_similarity_drop = (
        float(baseline_metrics["minimum_bipolar_similarity"])
        - float(candidate_metrics["minimum_bipolar_similarity"])
    )
    incorrect_increase = (
        int(candidate_metrics["incorrect_top1_cases"]) - int(baseline_metrics["incorrect_top1_cases"])
    )
    top1_regressions = sum(
        1
        for before, after in zip(baseline.cases, candidate.cases, strict=True)
        if before["top1_acceptable"] is True and after["top1_acceptable"] is False
    )
    case_similarity_drops = [
        float(before["bipolar_similarity"]) - float(after["bipolar_similarity"])
        for before, after in zip(baseline.cases, candidate.cases, strict=True)
    ]
    case_similarity_regressions = sum(
        1 for drop in case_similarity_drops if drop > policy.maximum_case_similarity_drop + EPSILON
    )
    maximum_observed_case_similarity_drop = max(case_similarity_drops)

    limits = {
        "top1_accuracy_drop": policy.maximum_top1_accuracy_drop,
        "top1_wilson_lower_bound_95_drop": policy.maximum_top1_wilson_lower_bound_95_drop,
        "mean_bipolar_similarity_drop": policy.maximum_mean_bipolar_similarity_drop,
        "minimum_bipolar_similarity_drop": policy.maximum_minimum_bipolar_similarity_drop,
        "incorrect_top1_cases_increase": policy.maximum_incorrect_top1_cases_increase,
        "top1_hit_regressions": policy.maximum_top1_hit_regressions,
        "case_similarity_drop": policy.maximum_case_similarity_drop,
        "case_similarity_regressions": policy.maximum_case_similarity_regressions,
    }
    observed = {
        "top1_accuracy_drop": accuracy_drop,
        "top1_wilson_lower_bound_95_drop": wilson_drop,
        "mean_bipolar_similarity_drop": mean_similarity_drop,
        "minimum_bipolar_similarity_drop": minimum_similarity_drop,
        "incorrect_top1_cases_increase": incorrect_increase,
        "top1_hit_regressions": top1_regressions,
        "maximum_case_similarity_drop": maximum_observed_case_similarity_drop,
        "case_similarity_regressions": case_similarity_regressions,
    }
    checks: Sequence[tuple[str, float | int, float | int]] = (
        ("top1_accuracy_drop", accuracy_drop, policy.maximum_top1_accuracy_drop),
        ("top1_wilson_lower_bound_95_drop", wilson_drop, policy.maximum_top1_wilson_lower_bound_95_drop),
        ("mean_bipolar_similarity_drop", mean_similarity_drop, policy.maximum_mean_bipolar_similarity_drop),
        ("minimum_bipolar_similarity_drop", minimum_similarity_drop, policy.maximum_minimum_bipolar_similarity_drop),
        ("incorrect_top1_cases_increase", incorrect_increase, policy.maximum_incorrect_top1_cases_increase),
        ("top1_hit_regressions", top1_regressions, policy.maximum_top1_hit_regressions),
        ("case_similarity_regressions", case_similarity_regressions, policy.maximum_case_similarity_regressions),
    )
    for name, actual, maximum in checks:
        if float(actual) > float(maximum) + EPSILON:
            failures.append(f"{name}={actual} exceeds allowed {maximum}")

    return {
        "schema_version": REPORT_SCHEMA,
        "status": "failed" if failures else "passed",
        "baseline": {
            "checkpoint_sha256": baseline.checkpoint_sha256,
            "quality_report_sha256": baseline.report_sha256,
            "quality_report_bytes": baseline.report_bytes,
            "quality_gate_passed": baseline.quality_gate_passed,
            "metrics": baseline.metrics,
        },
        "candidate": {
            "checkpoint_sha256": candidate.checkpoint_sha256,
            "quality_report_sha256": candidate.report_sha256,
            "quality_report_bytes": candidate.report_bytes,
            "quality_gate_passed": candidate.quality_gate_passed,
            "metrics": candidate.metrics,
        },
        "regression_policy": dict(regression_binding),
        "comparison": {
            "case_count": baseline.case_count,
            "observed": observed,
            "allowed": limits,
            "case_identity_alignment": "HMAC-redacted case/query/acceptable-set identities match in source order",
        },
        "failures": failures,
        "scope": (
            "Comparison covers only the same immutable held-out semantic-neighbour case set and policy. "
            "It is a continuation-regression signal, not evidence of general intelligence, factuality, safety, "
            "language competence, or LLM/LMM equivalence."
        ),
    }


def compare(arguments: argparse.Namespace) -> dict[str, object]:
    require(0 < arguments.max_cases <= MAX_CASES, f"--max-cases must be in [1, {MAX_CASES}]")
    regression_binding = regression_policy_binding(arguments.regression_policy)
    regression_policy = load_regression_policy(arguments.regression_policy)
    require(
        regression_binding["requirements"] == regression_policy.requirements(),
        "quality regression policy changed while it was read",
    )
    try:
        binding = evaluator_binding(
            arguments.cases, arguments.quality_policy, arguments.redaction_key_file, arguments.max_cases
        )
    except PretrainingQualityEvaluationError as error:
        raise ContinuousQualityGateError(f"quality evidence preflight failed: {error}") from error
    binary_digest = regular_checkpoint_digest(arguments.pretrain_bin, "pretraining executable")
    baseline_digest = regular_checkpoint_digest(arguments.baseline_checkpoint, "baseline checkpoint")
    candidate_digest = regular_checkpoint_digest(arguments.candidate_checkpoint, "candidate checkpoint")
    baseline = read_validated_report(arguments.baseline_report, binding, binary_digest, baseline_digest)
    candidate = read_validated_report(arguments.candidate_report, binding, binary_digest, candidate_digest)
    # Bind a comparison to the current exact policy bytes after the report work
    # as well; a mutable operator path cannot silently alter tolerances mid-run.
    current_binding = regression_policy_binding(arguments.regression_policy)
    require(current_binding == regression_binding, "quality regression policy changed during comparison")
    return comparison_document(baseline, candidate, regression_policy, regression_binding)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--quality-policy", required=True, type=Path)
    parser.add_argument("--redaction-key-file", required=True, type=Path)
    parser.add_argument("--regression-policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=MAX_CASES)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        report = compare(arguments)
        atomic_write_json(arguments.output, report)
    except (ContinuousQualityGateError, OSError, ValueError) as error:
        print(f"continuous quality gate: FAIL: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    if report["status"] != "passed":
        print("continuous quality gate: QUALITY REGRESSION", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
