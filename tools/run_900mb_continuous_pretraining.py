#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the exact-decimal 900 MB -> 90 MB continual-pretraining experiment.

This is the repository's large controlled fixture path for the user-facing
sequence:

1. stream a deterministic exact ``900MB = 900,000,000`` byte initial corpus;
2. run one real native ``vegai_pretrain train`` child through the immutable
   continuous controller;
3. stream an immutable exact ``90MB = 90,000,000`` byte continuation shard;
4. run one real native ``vegai_pretrain continue`` child; and
5. compare the frozen held-out topology annotations for the 900 MB anchor and
   the final 990 MB checkpoint before the continuation checkpoint is published.

The datasets are intentionally synthetic, algorithmically generated ASCII
fixtures.  This measures controller integrity, train-only throughput and a
controlled semantic-neighbour topology signal; it is *not* evidence of natural
language competence, factual knowledge, production domain quality, or general
AI capability.  Replace the fixture and its permissive absolute policy with
licensed domain data and independently governed held-out annotations before
making a domain-quality decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from synthetic_pretraining_fixture import FixtureConfig, FixtureError, generate_fixture, token_for


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "tools" / "continuous_pretrain.py"
MANIFEST_TOOL = ROOT / "tools" / "continuous_shard_manifest.py"
SCHEMA = "vegai-900mb-continuous-pretraining-experiment/v1"
INITIAL_BYTES = 900_000_000
CONTINUATION_BYTES = 90_000_000
CLUSTERS = 8
TOKENS_PER_CLUSTER = 16
TOKENS_PER_RECORD = 64
VOCABULARY = CLUSTERS * TOKENS_PER_CLUSTER
QUALITY_CASES = CLUSTERS + 1


class ExperimentError(RuntimeError):
    """Expected controlled-experiment setup or child failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentError(f"cannot read {label}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def run(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as error:
        raise ExperimentError(f"cannot start {label}: {error}") from error
    if result.returncode != 0:
        raise ExperimentError(f"{label} failed with status {result.returncode}\n{result.stdout[-16_384:]}")
    return result


def quality_case_manifest() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    all_tokens = [token_for(cluster, token) for cluster in range(CLUSTERS) for token in range(TOKENS_PER_CLUSTER)]
    # The sentinel guarantees that the absolute quality policy verifies a live
    # inspect/checkpoint path without pretending this synthetic topology has a
    # calibrated production threshold. The remaining cases measure cluster
    # retrieval and are protected by the regression limits below.
    sentinel_query = token_for(0, 0)
    cases.append({
        "schema_version": "vegai-pretraining-quality-case/v1",
        "id": "synthetic-reachability-sentinel",
        "query_token": sentinel_query,
        "acceptable_neighbours": [token for token in all_tokens if token != sentinel_query],
    })
    for cluster in range(CLUSTERS):
        query_index = 1 if cluster == 0 else 0
        query = token_for(cluster, query_index)
        cases.append({
            "schema_version": "vegai-pretraining-quality-case/v1",
            "id": f"synthetic-cluster-{cluster:03d}",
            "query_token": query,
            "acceptable_neighbours": [
                token_for(cluster, token)
                for token in range(TOKENS_PER_CLUSTER)
                if token != query_index
            ],
        })
    return {
        "schema_version": "vegai-pretraining-quality-cases/v1",
        "evaluation_id": "synthetic-900mb-continuation-topology-v1",
        "declared_split": "held-out",
        "annotation_provenance": {
            "source": "algorithmic clustered-fixture topology; annotations withheld from native train input",
            "license": "not applicable; no third-party text or labels",
            "retrieved_at": "generated with this immutable controlled experiment",
            "method": "cluster-membership semantic-neighbour topology diagnostic/v1",
        },
        "cases": cases,
    }


def quality_policy() -> dict[str, Any]:
    # This gate is deliberately permissive except for the reachability
    # sentinel. The regression policy, not a fabricated absolute claim,
    # determines whether continuation can publish a candidate fixture model.
    return {
        "schema_version": "vegai-pretraining-quality-policy/v1",
        "policy_id": "synthetic-900mb-structural-absolute-v1",
        "minimum_cases": QUALITY_CASES,
        "minimum_top1_accuracy": 0.0,
        "minimum_top1_wilson_lower_bound_95": 0.0,
        "minimum_mean_similarity": -1.0,
        "minimum_case_similarity": -1.0,
        "maximum_incorrect_cases": QUALITY_CASES - 1,
    }


def regression_policy() -> dict[str, Any]:
    # These explicit tolerances make an observed decline visible in the report
    # and fail closed only for material forgetting on the frozen nine cases.
    return {
        "schema_version": "vegai-continuous-quality-regression-policy/v1",
        "policy_id": "synthetic-900mb-topology-regression-v1",
        "maximum_top1_accuracy_drop": 1.0 / QUALITY_CASES,
        "maximum_top1_wilson_lower_bound_95_drop": 0.15,
        "maximum_mean_bipolar_similarity_drop": 0.10,
        "maximum_minimum_bipolar_similarity_drop": 0.25,
        "maximum_incorrect_top1_cases_increase": 1,
        "maximum_top1_hit_regressions": 1,
        "maximum_case_similarity_drop": 0.20,
        "maximum_case_similarity_regressions": 1,
    }


def fixture_report(path: Path, manifest: Path, result: dict[str, Any], expected_bytes: int) -> dict[str, Any]:
    corpus = result.get("corpus")
    records = result.get("records")
    require(isinstance(corpus, dict) and isinstance(records, dict), "fixture generator emitted an incomplete report")
    require(corpus.get("bytes") == expected_bytes and path.stat().st_size == expected_bytes,
            f"fixture does not have exact expected decimal byte size {expected_bytes}")
    return {
        "path": path.name,
        "manifest": manifest.name,
        "manifest_sha256": sha256_file(manifest),
        "bytes": corpus["bytes"],
        "sha256": corpus["sha256"],
        "raw_tokens": records["raw_tokens"],
        "physical_records": records["physical_records"],
        "unique_tokens": records["unique_tokens"],
    }


def controller_command(
    binary: Path,
    manifest: Path,
    state: Path,
    manifest_sha256: str,
    cases: Path,
    absolute_policy: Path,
    key: Path,
    regression: Path,
    arguments: argparse.Namespace,
    *,
    max_shards: int | None,
) -> list[str]:
    command = [
        sys.executable, str(CONTROLLER),
        "--pretrain-bin", str(binary), "--manifest", str(manifest), "--state-dir", str(state),
        "--expect-manifest-sha256", manifest_sha256,
        "--max-vocab", str(VOCABULARY), "--min-frequency", "1", "--window", "1",
        "--position-shift", "41", "--context-mode", "hybrid", "--epochs", "1",
        "--initial-plasticity", "25", "--minimum-plasticity", "1", "--decay-events", "32",
        "--omp-threads", str(arguments.omp_threads), "--train-timeout-seconds", str(arguments.train_timeout_seconds),
        "--reserve-fraction", str(arguments.reserve_fraction),
        "--quality-cases", str(cases), "--quality-policy", str(absolute_policy),
        "--quality-redaction-key-file", str(key), "--quality-regression-policy", str(regression),
        "--quality-max-cases", str(QUALITY_CASES), "--quality-timeout-seconds", str(arguments.quality_timeout_seconds),
    ]
    if arguments.quality_scratch_dir is not None:
        command.extend(["--quality-scratch-dir", str(arguments.quality_scratch_dir)])
    if max_shards is not None:
        command.extend(["--max-shards", str(max_shards)])
    return command


def phase_summary(entry: dict[str, Any]) -> dict[str, Any]:
    measurement = entry.get("training_measurement")
    require(isinstance(measurement, dict), "committed generation omitted train-only measurement")
    return {
        "generation": entry.get("generation"),
        "checkpoint": entry.get("checkpoint"),
        "checkpoint_sha256": entry.get("checkpoint_sha256"),
        "accepted_tokens": entry.get("accepted_tokens"),
        "training_measurement": measurement,
        "resource_preflight": entry.get("resource_preflight"),
        "quality": entry.get("quality"),
    }


def outcome_report(
    run_dir: Path,
    fixtures: dict[str, Any],
    state_path: Path,
    *,
    status: str,
    failure: str | None = None,
) -> dict[str, Any]:
    state = read_json(state_path, "continuous state") if state_path.is_file() else None
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "dataset_units": {
            "initial": "decimal MB; 900MB = 900,000,000 physical bytes",
            "continuous": "decimal MB; 90MB = 90,000,000 physical bytes",
        },
        "fixtures": fixtures,
        "timing_scope": (
            "Each phase's train_process_wall_seconds measures only its native vegai_pretrain train/continue child. "
            "Fixture generation, manifests, resource checks, evaluator inspection, report work, compilation and CI are excluded."
        ),
        "limitations": (
            "Synthetic deterministic topology fixture only: not evidence of natural-language, factual, domain, safety, "
            "general intelligence, LLM, or LMM quality. Use licensed corpus and independently governed held-out annotations "
            "for a production quality claim."
        ),
        "failure": failure,
    }
    if not isinstance(state, dict):
        payload["continuous_state"] = None
        return payload
    applied = state.get("applied_shards")
    payload["continuous_state"] = {
        "generation": state.get("generation"),
        "checkpoint": state.get("checkpoint"),
        "checkpoint_sha256": state.get("checkpoint_sha256"),
        "pretrain_binary": state.get("pretrain_binary"),
    }
    if not isinstance(applied, list) or not applied:
        return payload
    payload["initial_pretraining"] = phase_summary(applied[0])
    if len(applied) >= 2:
        payload["continuous_pretraining"] = phase_summary(applied[1])
        quality = applied[1].get("quality") if isinstance(applied[1], dict) else None
        if isinstance(quality, dict):
            comparison_path = state_path.parent / str(quality.get("comparison_report", ""))
            if comparison_path.is_file():
                payload["quality_regression"] = read_json(comparison_path, "continuation regression report")
    rejected = state_path.parent / "quality-rejections"
    if rejected.is_dir():
        rejection_files = sorted(rejected.glob("*.rejection.json"))
        if rejection_files:
            payload["quality_rejection"] = read_json(rejection_files[-1], "quality rejection report")
            regression_files = sorted(rejected.glob("*.regression.json"))
            if regression_files:
                payload["quality_regression"] = read_json(regression_files[-1], "rejected continuation regression report")
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled real vegai_pretrain executable")
    parser.add_argument("--run-dir", required=True, type=Path, help="new empty ignored runtime directory")
    parser.add_argument("--report", type=Path, help="final experiment report (default: RUN_DIR/experiment-report.json)")
    parser.add_argument("--omp-threads", type=int, default=2, help="native OMP threads (default: 2)")
    parser.add_argument("--train-timeout-seconds", type=float, default=2400.0,
                        help="per native train/continue timeout (default: 2400)")
    parser.add_argument("--reserve-fraction", type=float, default=0.25)
    parser.add_argument("--quality-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--quality-scratch-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    run_dir = arguments.run_dir.resolve()
    report_path = (arguments.report or run_dir / "experiment-report.json").resolve()
    key_path: Path | None = None
    try:
        require(arguments.omp_threads > 0, "--omp-threads must be positive")
        require(arguments.train_timeout_seconds > 0.0, "--train-timeout-seconds must be positive")
        require(0.0 <= arguments.reserve_fraction <= 4.0, "--reserve-fraction must be in [0, 4]")
        binary = arguments.pretrain_bin.resolve()
        require(binary.is_file() and os.access(binary, os.X_OK), f"unusable pretraining executable: {binary}")
        require(not run_dir.exists() or not any(run_dir.iterdir()),
                f"--run-dir must be new/empty for the immutable 900MB experiment: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        corpus_dir = run_dir / "corpora"
        initial = corpus_dir / "synthetic_initial_900MB.txt"
        initial_manifest = corpus_dir / "synthetic_initial_900MB.manifest.json"
        continuation = corpus_dir / "synthetic_continuous_90MB.txt"
        continuation_manifest = corpus_dir / "synthetic_continuous_90MB.manifest.json"
        config_initial = FixtureConfig(INITIAL_BYTES, CLUSTERS, TOKENS_PER_CLUSTER, TOKENS_PER_RECORD)
        config_continuation = FixtureConfig(CONTINUATION_BYTES, CLUSTERS, TOKENS_PER_CLUSTER, TOKENS_PER_RECORD)
        initial_result = generate_fixture(initial, initial_manifest, config_initial, overwrite=False)
        continuation_result = generate_fixture(continuation, continuation_manifest, config_continuation, overwrite=False)
        fixtures = {
            "initial_900MB": fixture_report(initial, initial_manifest, initial_result, INITIAL_BYTES),
            "continuous_90MB": fixture_report(continuation, continuation_manifest, continuation_result, CONTINUATION_BYTES),
            "configuration": {
                "clusters": CLUSTERS, "tokens_per_cluster": TOKENS_PER_CLUSTER,
                "tokens_per_record": TOKENS_PER_RECORD, "vocabulary": VOCABULARY,
            },
        }

        ledger = run_dir / "continuous-ledger.json"
        run([
            sys.executable, str(MANIFEST_TOOL), str(initial), str(continuation), "--output", str(ledger),
            "--source", "deterministic synthetic 900MB then 90MB topology fixture",
            "--license", "not applicable; algorithmically generated no third-party text",
            "--preprocessing", "clustered-cyclic-ascii-topology/v1; strict ASCII LF",
        ], "immutable continuous shard manifest authoring")
        ledger_digest = sha256_file(ledger)

        cases = run_dir / "quality-cases.json"
        absolute_policy = run_dir / "quality-policy.json"
        regression = run_dir / "quality-regression-policy.json"
        atomic_json(cases, quality_case_manifest())
        atomic_json(absolute_policy, quality_policy())
        atomic_json(regression, regression_policy())
        key_path = run_dir / ".quality-redaction.key"
        key_path.write_bytes(secrets.token_bytes(32))
        os.chmod(key_path, 0o600)

        state = run_dir / "state"
        try:
            run(controller_command(
                binary, ledger, state, ledger_digest, cases, absolute_policy, key_path, regression, arguments,
                max_shards=1,
            ), "900MB initial native pretraining")
            run(controller_command(
                binary, ledger, state, ledger_digest, cases, absolute_policy, key_path, regression, arguments,
                max_shards=None,
            ), "90MB native continuous pretraining and regression gate")
        except ExperimentError as error:
            outcome = outcome_report(run_dir, fixtures, state / "state.json", status="failed-or-quality-rejected", failure=str(error))
            atomic_json(report_path, outcome)
            print(json.dumps(outcome, ensure_ascii=True, sort_keys=True))
            return 3
        outcome = outcome_report(run_dir, fixtures, state / "state.json", status="succeeded")
        continuous_state = outcome.get("continuous_state")
        require(isinstance(continuous_state, dict) and continuous_state.get("generation") == 2,
                "controller did not commit exactly the 900MB anchor plus 90MB continuation")
        regression_report = outcome.get("quality_regression")
        require(isinstance(regression_report, dict) and regression_report.get("status") == "passed",
                "continuous checkpoint was committed without a passing quality regression report")
        atomic_json(report_path, outcome)
        print(json.dumps(outcome, ensure_ascii=True, sort_keys=True))
        return 0
    except (ExperimentError, FixtureError, OSError, ValueError) as error:
        failure = {
            "schema": SCHEMA,
            "status": "setup-or-execution-failed",
            "failure": str(error),
            "limitations": "No general/model-quality claim is available from a failed controlled synthetic experiment.",
        }
        try:
            atomic_json(report_path, failure)
        except OSError:
            pass
        print(f"900MB continuous pretraining: FAIL: {error}", file=sys.stderr)
        return 2
    finally:
        if key_path is not None:
            key_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
