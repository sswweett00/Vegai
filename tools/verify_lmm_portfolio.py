#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline integration verification for domain-expert LMM portfolio routing.

Synthetic workers are only protocol fixtures: they prove that routing invokes
the selected configured gateway and preserves provenance.  They do not claim a
real model's domain quality or intelligence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from lmm_contract import canonical_json_bytes, file_identity


ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_TOOL = ROOT / "tools" / "lmm_portfolio.py"
GATEWAY = ROOT / "tools" / "lmm_gateway.py"


def require(condition: bool, message: str) -> None:
    if not condition:
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
        raise AssertionError(
            f"command returned {completed.returncode}, expected {expected}: {' '.join(command)}\n{completed.stdout}"
        )
    return completed


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_manifest(model_id: str, revision: str) -> dict[str, object]:
    return {
        "schema_version": "vegai-lmm-model/v1",
        "model_id": model_id,
        "engine": "worker",
        "context_window_tokens": 512,
        "max_output_tokens": 128,
        "message_overhead_tokens": 4,
        "reply_primer_tokens": 2,
        "capabilities": {"text": True, "image": False, "audio": False, "video": False},
        "tokenizer": {"kind": "utf8-byte"},
        "artifacts": [],
        "prompt_format": "chatml-v1",
        "source": "repository-local synthetic portfolio fixture",
        "license": "MIT",
        "revision": revision,
    }


def quality_report(
    model_sha256: str,
    backend: str,
    runtime_identity_sha256: str,
    gateway_command_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "vegai-lmm-quality-report/v1",
        "dependencies": {"gateway_command": {"sha256": gateway_command_sha256}},
        "configuration": {
            "run_provenance": {
                "model_manifest_sha256": model_sha256,
                "runtime_identity_sha256": runtime_identity_sha256,
                "backend": backend,
            }
        },
        "quality_gate": {"passed": True},
    }


def generation_runtime_identity(model: Path, worker_command: Path) -> str:
    stable = {
        "gateway_tool": file_identity(GATEWAY, "gateway tool").to_mapping(),
        "model_manifest": file_identity(model, "model manifest").to_mapping(),
        "tokenizer": None,
        "asset_registry": None,
        "worker_command_configuration": file_identity(worker_command, "worker command").to_mapping(),
        "structured_response_contract": None,
        "model_artifacts": {},
    }
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def evaluation_card(card_id: str, model_sha256: str, quality_report_sha256: str, quality: float, latency: int, cost: int) -> dict[str, object]:
    return {
        "schema_version": "vegai-lmm-evaluation-card/v1",
        "id": card_id,
        "cost_unit": "synthetic-microunits",
        "model_manifest_sha256": model_sha256,
        "quality_report_sha256": quality_report_sha256,
        "measurements": [
            {
                "domain": "coding",
                "suite_id": "synthetic-coding-heldout",
                "suite_report_sha256": hashlib.sha256(f"{card_id}-suite".encode("ascii")).hexdigest(),
                "cases": 200,
                "quality_lower_bound": quality,
                "reliability_lower_bound": 0.90,
                "safety_lower_bound": 0.94,
                "p95_latency_ms": latency,
                "cost_microunits_per_1k_output_tokens": cost,
            }
        ],
    }


def task(strategy: str, candidates: int, minimum_quality: float = 0.80) -> dict[str, object]:
    return {
        "schema_version": "vegai-lmm-routing-task/v1",
        "id": f"coding-{strategy}",
        "domain": "coding",
        "data_residency": "TR",
        "minimum_safety_tier": 2,
        "strategy": strategy,
        "minimum_independent_candidates": candidates,
        "required_modalities": ["text"],
        "require_evidence": True,
        "minimum_context_window_tokens": 256,
        "minimum_quality_lower_bound": minimum_quality,
        "minimum_reliability_lower_bound": 0.85,
        "minimum_safety_lower_bound": 0.90,
        "max_p95_latency_ms": 500,
        "max_cost_microunits_per_1k_output_tokens": 100,
    }


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="vegai-lmm-portfolio-") as temporary:
            workspace = Path(temporary)
            request = workspace / "request.json"
            request_text = "Bu kod görevi için kaynaklı bir açıklama üret."
            write_json(
                request,
                {
                    "schema_version": "vegai-lmm-request/v1",
                    "id": "portfolio-coding-request",
                    "messages": [
                        {
                            "id": "policy",
                            "role": "system",
                            "pinned": True,
                            "content": [{"type": "text", "text": "Kaynak ID markerini model kendisi üretmelidir."}],
                        },
                        {
                            "id": "question",
                            "role": "user",
                            "pinned": False,
                            "content": [{"type": "text", "text": request_text}],
                        },
                    ],
                    "max_new_tokens": 64,
                    "temperature": 0,
                    "top_p": 1,
                },
            )
            evidence_text = "Kod incelemesinde sınır kontrolleri zorunludur."
            evidence = workspace / "evidence.json"
            write_json(
                evidence,
                {
                    "schema_version": "vegai-lmm-evidence/v1",
                    "query_sha256": digest_text("coding query"),
                    "index": {"schema": "synthetic", "revision": "v1"},
                    "items": [
                        {
                            "id": "turkish",
                            "text": evidence_text,
                            "source": "synthetic-test",
                            "license": "MIT",
                            "sha256": digest_text(evidence_text),
                            "score": 1.0,
                        }
                    ],
                },
            )

            high_model = workspace / "high-model.json"
            general_model = workspace / "general-model.json"
            write_json(high_model, model_manifest("coding-expert", "high-v1"))
            write_json(general_model, model_manifest("general-expert", "general-v1"))
            high_identity = file_identity(high_model, "high model")
            general_identity = file_identity(general_model, "general model")

            worker = workspace / "worker.py"
            worker.write_text(
                """import json, sys
+payload = json.load(sys.stdin)
+model = payload['model']['model_id']
+evidence = [message for message in payload['messages'] if message['id'].startswith('evidence:')]
+assert any(message['id'] == 'evidence:turkish' for message in evidence)
+prefix = 'Coding expert selected.' if model == 'coding-expert' else 'General expert selected.'
+print(json.dumps({
+  'schema_version': 'vegai-lmm-worker-response/v1',
+  'text': prefix + ' [[vegai-source:turkish]]',
+  'finish_reason': 'stop',
+  'input_tokens': 9,
+  'output_tokens': 7,
+  'backend': model,
+  'device': None
+}))
+""".replace("\n+", "\n"),
                encoding="utf-8",
                newline="\n",
            )
            worker_command = workspace / "worker-command.json"
            write_json(worker_command, [sys.executable, str(worker)])

            high_command = workspace / "high-gateway-command.json"
            general_command = workspace / "general-gateway-command.json"
            for command, model in ((high_command, high_model), (general_command, general_model)):
                write_json(
                    command,
                    [
                        sys.executable,
                        str(GATEWAY),
                        "--model",
                        str(model),
                        "--request",
                        "{request}",
                        "--evidence",
                        "{evidence}",
                        "--worker-command-json",
                        str(worker_command),
                        "--grounding-mode",
                        "evidence-or-abstain",
                        "--print-json",
                    ],
                )

            high_quality_report = workspace / "high-quality-report.json"
            general_quality_report = workspace / "general-quality-report.json"
            write_json(
                high_quality_report,
                quality_report(
                    high_identity.sha256,
                    "coding-expert",
                    generation_runtime_identity(high_model, worker_command),
                    file_identity(high_command, "high gateway command").sha256,
                ),
            )
            write_json(
                general_quality_report,
                quality_report(
                    general_identity.sha256,
                    "general-expert",
                    generation_runtime_identity(general_model, worker_command),
                    file_identity(general_command, "general gateway command").sha256,
                ),
            )
            high_card = workspace / "high-card.json"
            general_card = workspace / "general-card.json"
            write_json(
                high_card,
                evaluation_card(
                    "coding-expert-card",
                    high_identity.sha256,
                    file_identity(high_quality_report, "high quality report").sha256,
                    0.96,
                    180,
                    45,
                ),
            )
            write_json(
                general_card,
                evaluation_card(
                    "general-expert-card",
                    general_identity.sha256,
                    file_identity(general_quality_report, "general quality report").sha256,
                    0.84,
                    120,
                    25,
                ),
            )
            portfolio = workspace / "portfolio.json"
            write_json(
                portfolio,
                {
                    "schema_version": "vegai-lmm-portfolio/v1",
                    "id": "synthetic-domain-portfolio",
                    "cost_unit": "synthetic-microunits",
                    "candidates": [
                        {
                            "id": "general-expert",
                            "domains": ["coding"],
                            "independence_group": "provider-general",
                            "data_residencies": ["TR"],
                            "safety_tier": 3,
                            "grounding_mode": "evidence-or-abstain",
                            "model_manifest": str(general_model),
                            "gateway_command_json": str(general_command),
                            "evaluation_card": str(general_card),
                            "quality_report": str(general_quality_report),
                        },
                        {
                            "id": "coding-expert",
                            "domains": ["coding"],
                            "independence_group": "provider-coding",
                            "data_residencies": ["TR"],
                            "safety_tier": 3,
                            "grounding_mode": "evidence-or-abstain",
                            "model_manifest": str(high_model),
                            "gateway_command_json": str(high_command),
                            "evaluation_card": str(high_card),
                            "quality_report": str(high_quality_report),
                        },
                    ],
                },
            )

            single_task = workspace / "single-task.json"
            write_json(single_task, task("single-best", 1))
            route_report = workspace / "route-report.json"
            route_result = run(
                [
                    sys.executable,
                    str(PORTFOLIO_TOOL),
                    "route",
                    "--portfolio",
                    str(portfolio),
                    "--task",
                    str(single_task),
                    "--request",
                    str(request),
                    "--evidence",
                    str(evidence),
                    "--output",
                    str(route_report),
                ]
            )
            require(json.loads(route_result.stdout)["selected_candidates"] == 1, "single-best router selected an unexpected number of models")
            route_payload = json.loads(route_report.read_text(encoding="utf-8"))
            require(route_payload["selected"][0]["id"] == "coding-expert", "router did not choose the highest measured eligible coding model")
            require(request_text not in route_report.read_text(encoding="utf-8"), "route decision exposed raw user request text")
            require(route_payload["selected"][0]["heldout_quality_report"]["sha256"] == file_identity(high_quality_report, "high report").sha256,
                    "route decision did not bind the selected held-out quality report")

            collision = run(
                [
                    sys.executable,
                    str(PORTFOLIO_TOOL),
                    "route",
                    "--portfolio",
                    str(portfolio),
                    "--task",
                    str(single_task),
                    "--request",
                    str(request),
                    "--evidence",
                    str(evidence),
                    "--output",
                    str(worker_command),
                ],
                expected=2,
            )
            require("must not overwrite immutable routing dependency" in collision.stdout,
                    "router allowed its report to overwrite a configured worker command")

            generation_report = workspace / "generation-report.json"
            generated = run(
                [
                    sys.executable,
                    str(PORTFOLIO_TOOL),
                    "generate",
                    "--portfolio",
                    str(portfolio),
                    "--task",
                    str(single_task),
                    "--request",
                    str(request),
                    "--evidence",
                    str(evidence),
                    "--output",
                    str(generation_report),
                    "--print-json",
                ]
            )
            generated_payload = json.loads(generated.stdout)
            require(generated_payload["response"].startswith("Coding expert selected."),
                    "generate did not invoke the routed coding expert gateway")
            generation_payload = json.loads(generation_report.read_text(encoding="utf-8"))
            require(generation_payload["generation"]["candidate_id"] == "coding-expert"
                    and generation_payload["response"]["text"] is None,
                    "generation report lost selected-candidate provenance or retained response text by default")
            require(request_text not in generation_report.read_text(encoding="utf-8"),
                    "generation report exposed raw request content")

            panel_task = workspace / "panel-task.json"
            write_json(panel_task, task("independent-panel", 2))
            panel_report = workspace / "panel-report.json"
            run(
                [
                    sys.executable,
                    str(PORTFOLIO_TOOL),
                    "route",
                    "--portfolio",
                    str(portfolio),
                    "--task",
                    str(panel_task),
                    "--request",
                    str(request),
                    "--evidence",
                    str(evidence),
                    "--output",
                    str(panel_report),
                ]
            )
            panel_payload = json.loads(panel_report.read_text(encoding="utf-8"))
            require([item["id"] for item in panel_payload["selected"]] == ["coding-expert", "general-expert"],
                    "independent panel did not preserve deterministic measured ranking")
            require(len({item["independence_group"] for item in panel_payload["selected"]}) == 2,
                    "independent panel did not enforce independently grouped candidates")

            impossible_task = workspace / "impossible-task.json"
            write_json(impossible_task, task("single-best", 1, minimum_quality=0.99))
            impossible_output = workspace / "must-not-exist.json"
            impossible = run(
                [
                    sys.executable,
                    str(PORTFOLIO_TOOL),
                    "route",
                    "--portfolio",
                    str(portfolio),
                    "--task",
                    str(impossible_task),
                    "--request",
                    str(request),
                    "--evidence",
                    str(evidence),
                    "--output",
                    str(impossible_output),
                ],
                expected=2,
            )
            require("no portfolio candidate" in impossible.stdout and not impossible_output.exists(),
                    "router published an unsafe fallback when no candidate met the quality threshold")

            mismatched_command = workspace / "mismatched-model-command.json"
            write_json(
                mismatched_command,
                [
                    sys.executable,
                    str(GATEWAY),
                    "--model",
                    str(general_model),
                    "--request",
                    "{request}",
                    "--evidence",
                    "{evidence}",
                    "--worker-command-json",
                    str(worker_command),
                    "--grounding-mode",
                    "evidence-or-abstain",
                    "--print-json",
                ],
            )
            malformed_portfolio = workspace / "mismatched-portfolio.json"
            bad_payload = json.loads(portfolio.read_text(encoding="utf-8"))
            bad_payload["candidates"][1]["gateway_command_json"] = str(mismatched_command)
            write_json(malformed_portfolio, bad_payload)
            mismatch = run(
                [
                    sys.executable,
                    str(PORTFOLIO_TOOL),
                    "route",
                    "--portfolio",
                    str(malformed_portfolio),
                    "--task",
                    str(single_task),
                    "--request",
                    str(request),
                    "--evidence",
                    str(evidence),
                    "--output",
                    str(workspace / "mismatch-output.json"),
                ],
                expected=2,
            )
            require("does not match" in mismatch.stdout,
                    "router accepted a gateway command bound to a different model manifest")

            unmeasured_command = workspace / "high-gateway-command-unmeasured.json"
            unmeasured_payload = json.loads(high_command.read_text(encoding="utf-8"))
            unmeasured_payload.extend(["--timeout-seconds", "7"])
            write_json(unmeasured_command, unmeasured_payload)
            unmeasured_portfolio = workspace / "unmeasured-command-portfolio.json"
            unmeasured_portfolio_payload = json.loads(portfolio.read_text(encoding="utf-8"))
            unmeasured_portfolio_payload["candidates"][1]["gateway_command_json"] = str(unmeasured_command)
            write_json(unmeasured_portfolio, unmeasured_portfolio_payload)
            unmeasured = run(
                [
                    sys.executable,
                    str(PORTFOLIO_TOOL),
                    "route",
                    "--portfolio",
                    str(unmeasured_portfolio),
                    "--task",
                    str(single_task),
                    "--request",
                    str(request),
                    "--evidence",
                    str(evidence),
                    "--output",
                    str(workspace / "unmeasured-output.json"),
                ],
                expected=2,
            )
            require("another gateway command" in unmeasured.stdout,
                    "router accepted a gateway command that its held-out report did not measure")

            drift_worker_command = workspace / "worker-command-drift.json"
            drift_worker_command.write_text(
                "[\n  " + json.dumps(sys.executable) + ",\n  " + json.dumps(str(worker)) + "\n]\n",
                encoding="utf-8",
                newline="\n",
            )
            drift_command = workspace / "high-gateway-command-drift.json"
            write_json(
                drift_command,
                [
                    sys.executable,
                    str(GATEWAY),
                    "--model",
                    str(high_model),
                    "--request",
                    "{request}",
                    "--evidence",
                    "{evidence}",
                    "--worker-command-json",
                    str(drift_worker_command),
                    "--grounding-mode",
                    "evidence-or-abstain",
                    "--print-json",
                ],
            )
            # The synthetic report deliberately claims the old measured runtime
            # while binding the new command. Routing may load it, but generate
            # must compare the actual gateway report and reject the drift.
            drift_quality_report = workspace / "high-runtime-drift-quality-report.json"
            write_json(
                drift_quality_report,
                quality_report(
                    high_identity.sha256,
                    "coding-expert",
                    generation_runtime_identity(high_model, worker_command),
                    file_identity(drift_command, "drift gateway command").sha256,
                ),
            )
            drift_card = workspace / "high-runtime-drift-card.json"
            write_json(
                drift_card,
                evaluation_card(
                    "coding-expert-runtime-drift-card",
                    high_identity.sha256,
                    file_identity(drift_quality_report, "drift quality report").sha256,
                    0.96,
                    180,
                    45,
                ),
            )
            drift_portfolio = workspace / "runtime-drift-portfolio.json"
            drift_payload = json.loads(portfolio.read_text(encoding="utf-8"))
            drift_payload["candidates"][1]["gateway_command_json"] = str(drift_command)
            drift_payload["candidates"][1]["evaluation_card"] = str(drift_card)
            drift_payload["candidates"][1]["quality_report"] = str(drift_quality_report)
            write_json(drift_portfolio, drift_payload)
            drift_output = workspace / "runtime-drift-output.json"
            runtime_drift = run(
                [
                    sys.executable,
                    str(PORTFOLIO_TOOL),
                    "generate",
                    "--portfolio",
                    str(drift_portfolio),
                    "--task",
                    str(single_task),
                    "--request",
                    str(request),
                    "--evidence",
                    str(evidence),
                    "--output",
                    str(drift_output),
                ],
                expected=2,
            )
            require("runtime dependencies differ" in runtime_drift.stdout and not drift_output.exists(),
                    "generate accepted gateway runtime dependencies that differ from held-out quality evaluation")
    except (AssertionError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"lmm portfolio verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("lmm portfolio verification: PASS (domain routing, evidence/runtime gate, independent panel, provenance, privacy, output safety, fail-closed dispatch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
