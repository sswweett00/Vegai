#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Calibrate a retrieval abstention score floor on validation data only.

The program invokes the native Fortran batch evaluator against (1) a literal
train-key integrity suite at the strictest candidate and (2) a source-disjoint
*validation* negative set at each predeclared score candidate. It chooses the
lowest validation-safe score floor, retaining the most potential retrieval
coverage subject to the explicit leakage cap. A test set is deliberately not an
input to this tool and cannot influence selection.

This is calibration of an abstention/retrieval safety policy. It does not label
unseen answers correct, train a causal decoder, or establish factuality or
natural conversation quality.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


REPORT_SCHEMA = "vegai-native-dialogue-evaluation-report/v1"
POLICY_SCHEMA = "vegai-dialogue-abstention-calibration/v1"


class CalibrationError(RuntimeError):
    """Expected invalid input, evaluator execution, or no-safe-policy error."""


@dataclass(frozen=True)
class CandidateResult:
    score_floor: float
    validation_report: dict[str, Any]
    validation_identity: dict[str, Any]
    false_grounded_rate: float
    nonfabrication_rate_including_oov: float
    eligible: bool
    rejection_reasons: tuple[str, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def sha256_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            byte_count += len(block)
    return {"path": path.name, "sha256": digest.hexdigest(), "bytes": byte_count}


def bounded_unit_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a decimal score: {value!r}") from error
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"score must be in [0, 1], got {value!r}")
    return parsed


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value!r}")
    return parsed


def run_native_evaluation(
    evaluator: Path,
    checkpoint: Path,
    knowledge: Path,
    evaluation: Path,
    *,
    output: Path,
    max_entries: int,
    max_cases: int,
    min_similarity: float,
    min_margin: float,
    semantic_weight: float,
    lexical_weight: float,
    score_floor: float,
    hybrid_margin_floor: float,
) -> dict[str, Any]:
    command = [
        str(evaluator), "evaluate", str(checkpoint), str(knowledge), str(evaluation),
        "--max-entries", str(max_entries), "--max-cases", str(max_cases),
        "--min-similarity", format(min_similarity, ".9g"), "--min-margin", format(min_margin, ".9g"),
        "--semantic-weight", format(semantic_weight, ".9g"), "--lexical-weight", format(lexical_weight, ".9g"),
        "--min-hybrid-score", format(score_floor, ".9g"),
        "--min-hybrid-margin", format(hybrid_margin_floor, ".9g"),
    ]
    completed = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output.write_text(completed.stdout, encoding="utf-8", newline="\n")
    if completed.returncode != 0:
        raise CalibrationError(
            f"native evaluator failed for score floor {score_floor:.9g} (exit {completed.returncode}); "
            f"see {output.name}: {completed.stdout[-1000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise CalibrationError(f"native evaluator did not emit valid JSON for score {score_floor:.9g}: {error}") from error
    validate_native_report(payload, output.name)
    identities = payload.get("identities")
    require(isinstance(identities, dict), f"{output.name}: native report has no identities object")
    for name, source_path in (("checkpoint", checkpoint), ("knowledge", knowledge), ("evaluation", evaluation)):
        expected_identity = sha256_identity(source_path)
        # Native reports intentionally avoid filesystem paths; the controller's
        # decision separately records basename/hash/size identities. Compare
        # the common cryptographic identity rather than requiring a path leak.
        expected_native_identity = {
            "sha256": expected_identity["sha256"],
            "bytes": expected_identity["bytes"],
        }
        reported_identity = identities.get(name)
        require(reported_identity == expected_native_identity,
                f"{output.name}: native report {name} identity does not bind the supplied input")
    return payload


def validate_native_report(payload: object, label: str) -> dict[str, Any]:
    require(isinstance(payload, dict), f"{label}: native report is not an object")
    require(payload.get("schema") == REPORT_SCHEMA, f"{label}: unexpected native report schema")
    metrics = payload.get("metrics")
    require(isinstance(metrics, dict), f"{label}: native report has no metrics object")
    required_counts = (
        "total_cases", "expected_grounded_cases", "expected_abstain_cases", "actual_grounded_cases",
        "rendered_abstain_cases", "correct_grounded_prompt_cases", "wrong_grounded_prompt_cases",
        "false_abstain_cases", "false_grounded_cases", "true_abstain_cases",
        "oov_rejected_abstain_cases", "oov_rejected_expected_grounded_cases",
    )
    for key in required_counts:
        value = metrics.get(key)
        require(isinstance(value, int) and value >= 0, f"{label}: invalid nonnegative count {key!r}")
    require(
        metrics["expected_grounded_cases"] + metrics["expected_abstain_cases"] == metrics["total_cases"],
        f"{label}: expected class counts do not cover total cases",
    )
    return payload


def evaluate_candidate(
    score_floor: float,
    report: dict[str, Any],
    report_identity: dict[str, Any],
    *,
    maximum_false_grounded_rate: float,
) -> CandidateResult:
    metrics = report["metrics"]
    expected_abstain = metrics["expected_abstain_cases"]
    reasons: list[str] = []
    if expected_abstain <= 0:
        reasons.append("validation set has no expected-abstention cases")
        false_grounded_rate = 1.0
        nonfabrication_rate = 0.0
    else:
        false_grounded_rate = metrics["false_grounded_cases"] / expected_abstain
        nonfabrication_rate = (
            metrics["true_abstain_cases"] + metrics["oov_rejected_abstain_cases"]
        ) / expected_abstain
        if false_grounded_rate > maximum_false_grounded_rate:
            reasons.append(
                f"validation false-grounded rate {false_grounded_rate:.12g} exceeds "
                f"cap {maximum_false_grounded_rate:.12g}"
            )
    if metrics["expected_grounded_cases"] != 0:
        reasons.append("validation set unexpectedly contains grounded-answer labels")
    return CandidateResult(
        score_floor=score_floor,
        validation_report=report,
        validation_identity=report_identity,
        false_grounded_rate=false_grounded_rate,
        nonfabrication_rate_including_oov=nonfabrication_rate,
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def validate_strict_train_contract(report: dict[str, Any], required_accuracy: float) -> dict[str, object]:
    metrics = report["metrics"]
    expected = metrics["expected_grounded_cases"]
    require(expected > 0, "strict train exact suite has no expected grounded cases")
    require(metrics["expected_abstain_cases"] == 0, "strict train exact suite contains abstention cases")
    accuracy = metrics["correct_grounded_prompt_cases"] / expected
    require(
        accuracy >= required_accuracy,
        f"strict train exact grounded-prompt accuracy {accuracy:.12g} is below required {required_accuracy:.12g}",
    )
    require(metrics["wrong_grounded_prompt_cases"] == 0 and metrics["false_abstain_cases"] == 0,
            "strict train exact suite contains a wrong grounding or false abstention")
    return {
        "expected_grounded_cases": expected,
        "correct_grounded_prompt_cases": metrics["correct_grounded_prompt_cases"],
        "grounded_prompt_accuracy": accuracy,
        "required_minimum_accuracy": required_accuracy,
    }


def select_candidate(candidates: Sequence[CandidateResult]) -> CandidateResult:
    eligible = [candidate for candidate in candidates if candidate.eligible]
    require(eligible, "no predeclared score candidate met the validation false-grounding cap")
    # Lower score retains more candidate retrievals, so use it only after it
    # met the held-out validation safety cap. Ties are deterministic.
    return min(eligible, key=lambda candidate: candidate.score_floor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-bin", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--knowledge", required=True, type=Path)
    parser.add_argument("--train-evaluation", required=True, type=Path,
                        help="native TSV with literal train-key integrity cases")
    parser.add_argument("--validation-evaluation", required=True, type=Path,
                        help="native TSV with source-disjoint expected-abstention cases")
    parser.add_argument("--report-directory", required=True, type=Path,
                        help="new directory for artifact-bound native reports")
    parser.add_argument("--output", required=True, type=Path, help="new calibration decision JSON")
    parser.add_argument("--score-candidate", action="append", type=bounded_unit_float, required=True,
                        help="predeclared hybrid score floor; repeat for every candidate")
    parser.add_argument("--hybrid-margin-floor", type=bounded_unit_float, required=True)
    parser.add_argument("--maximum-validation-false-grounded-rate", type=bounded_unit_float, required=True)
    parser.add_argument("--minimum-train-exact-accuracy", type=bounded_unit_float, default=1.0)
    parser.add_argument("--max-entries", type=positive_int, default=5000)
    parser.add_argument("--max-train-cases", type=positive_int, default=10000)
    parser.add_argument("--max-validation-cases", type=positive_int, default=10000)
    parser.add_argument("--min-similarity", type=float, default=-1.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--semantic-weight", type=bounded_unit_float, required=True)
    parser.add_argument("--lexical-weight", type=bounded_unit_float, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        require(arguments.evaluator_bin.is_file(), f"evaluator binary does not exist: {arguments.evaluator_bin}")
        for path, label in (
            (arguments.checkpoint, "checkpoint"),
            (arguments.knowledge, "knowledge TSV"),
            (arguments.train_evaluation, "train evaluation TSV"),
            (arguments.validation_evaluation, "validation evaluation TSV"),
        ):
            require(path.is_file(), f"{label} does not exist: {path}")
        require(arguments.semantic_weight + arguments.lexical_weight > 0.0,
                "semantic and lexical weights must have a positive sum")
        require(not arguments.output.exists(), f"refuse to overwrite calibration output: {arguments.output}")
        require(not arguments.report_directory.exists(),
                f"refuse to overwrite calibration report directory: {arguments.report_directory}")
        candidates = sorted(set(arguments.score_candidate))
        require(candidates, "at least one distinct --score-candidate is required")
        strict_score = max(candidates)
        arguments.report_directory.mkdir(parents=True, exist_ok=False)

        common = {
            "evaluator": arguments.evaluator_bin,
            "checkpoint": arguments.checkpoint,
            "knowledge": arguments.knowledge,
            "max_entries": arguments.max_entries,
            "min_similarity": arguments.min_similarity,
            "min_margin": arguments.min_margin,
            "semantic_weight": arguments.semantic_weight,
            "lexical_weight": arguments.lexical_weight,
            "hybrid_margin_floor": arguments.hybrid_margin_floor,
        }
        strict_train_path = arguments.report_directory / "train-exact-at-strictest-score.json"
        strict_train = run_native_evaluation(
            evaluation=arguments.train_evaluation,
            output=strict_train_path,
            max_cases=arguments.max_train_cases,
            score_floor=strict_score,
            **common,
        )
        strict_train_contract = validate_strict_train_contract(strict_train, arguments.minimum_train_exact_accuracy)

        candidate_results: list[CandidateResult] = []
        for score in candidates:
            report_path = arguments.report_directory / f"validation-score-{score:.6f}.json"
            validation_report = run_native_evaluation(
                evaluation=arguments.validation_evaluation,
                output=report_path,
                max_cases=arguments.max_validation_cases,
                score_floor=score,
                **common,
            )
            candidate_results.append(
                evaluate_candidate(
                    score,
                    validation_report,
                    sha256_identity(report_path),
                    maximum_false_grounded_rate=arguments.maximum_validation_false_grounded_rate,
                )
            )
        selected = select_candidate(candidate_results)
        decision = {
            "schema": POLICY_SCHEMA,
            "contract": {
                "selection_data": "source-disjoint validation expected-abstention cases only; no test set input",
                "scope": "retrieval abstention/leakage safety calibration only; not answer relevance, factuality, naturalness, or generation",
                "selection_rule": "lowest predeclared score floor meeting validation false-grounding cap after strict exact train contract",
            },
            "inputs": {
                "checkpoint": sha256_identity(arguments.checkpoint),
                "knowledge": sha256_identity(arguments.knowledge),
                "train_evaluation": sha256_identity(arguments.train_evaluation),
                "validation_evaluation": sha256_identity(arguments.validation_evaluation),
            },
            "policy_family": {
                "semantic_weight": arguments.semantic_weight,
                "lexical_weight": arguments.lexical_weight,
                "minimum_similarity": arguments.min_similarity,
                "minimum_margin": arguments.min_margin,
                "minimum_hybrid_margin": arguments.hybrid_margin_floor,
                "maximum_validation_false_grounded_rate": arguments.maximum_validation_false_grounded_rate,
            },
            "strict_train_exact_contract": {
                **strict_train_contract,
                "score_floor": strict_score,
                "report": sha256_identity(strict_train_path),
            },
            "candidates": [
                {
                    "score_floor": item.score_floor,
                    "validation_report": item.validation_identity,
                    "validation_false_grounded_rate": item.false_grounded_rate,
                    "validation_nonfabrication_rate_including_oov_rejections": item.nonfabrication_rate_including_oov,
                    "eligible": item.eligible,
                    "rejection_reasons": list(item.rejection_reasons),
                }
                for item in candidate_results
            ],
            "selected_policy": {
                "minimum_hybrid_score": selected.score_floor,
                "minimum_hybrid_margin": arguments.hybrid_margin_floor,
                "validation_false_grounded_rate": selected.false_grounded_rate,
                "validation_nonfabrication_rate_including_oov_rejections": selected.nonfabrication_rate_including_oov,
                "validation_report": selected.validation_identity,
            },
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    except (CalibrationError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"dialogue abstention calibration: ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
