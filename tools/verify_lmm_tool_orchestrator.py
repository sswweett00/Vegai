#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synthetic end-to-end contract test for the verified LMM tool plane.

The fixture creates independent arithmetic executor/verifier programs in a
private temporary directory.  It proves control-plane mechanics only: typed
plans, request/registry binding, predecessor references, independent result
verification, privacy-conscious publication, no-replace outputs, and negative
gates.  It does not claim a real model, a broadly correct calculator, or a
production OS sandbox.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from lmm_contract import GenerationRequest


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = ROOT / "tools" / "lmm_tool_orchestrator.py"
REGISTRY_SCHEMA = "vegai-lmm-tool-registry/v1"
VALUE_SCHEMA = "vegai-lmm-tool-value-schema/v1"
PLAN_SCHEMA = "vegai-lmm-tool-plan/v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical(value) + b"\n")


def run(command: list[str], expected: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"command exit {result.returncode}, expected {expected}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def schema(schema_id: str, root: dict[str, Any]) -> dict[str, object]:
    return {"schema_version": VALUE_SCHEMA, "id": schema_id, "root": root}


def arithmetic_executor_source() -> str:
    return '''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
with Path(args.input).open(encoding="utf-8") as source:
    invocation = json.load(source)
values = invocation["input"]
result = {
    "schema_version": "vegai-lmm-tool-result/v1",
    "id": invocation["id"],
    "tool_id": invocation["tool_id"],
    "output": {"value": values["left"] + values["right"]},
}
with Path(args.output).open("w", encoding="utf-8", newline="\\n") as destination:
    json.dump(result, destination, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    destination.write("\\n")
'''


def arithmetic_verifier_source(verdict: str = "pass", *, mutate_result: bool = False) -> str:
    mutation = ""
    if mutate_result:
        mutation = 'Path(args.result).write_text("{}\\n", encoding="utf-8", newline="\\n")'
    return f'''#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--result", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
with Path(args.input).open(encoding="utf-8") as source:
    invocation = json.load(source)
with Path(args.result).open(encoding="utf-8") as source:
    result = json.load(source)
if result["output"]["value"] != invocation["input"]["left"] + invocation["input"]["right"]:
    raise SystemExit(9)
digest = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
{mutation}
verification = {{
    "schema_version": "vegai-lmm-tool-verification/v1",
    "id": invocation["id"],
    "tool_id": invocation["tool_id"],
    "result_sha256": digest,
    "verdict": "{verdict}",
}}
with Path(args.output).open("w", encoding="utf-8", newline="\\n") as destination:
    json.dump(verification, destination, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    destination.write("\\n")
'''


def create_registry(workspace: Path, verifier_name: str = "verifier.py", *, forged_executor_hash: bool = False) -> Path:
    input_schema_path = workspace / "arithmetic-input.json"
    output_schema_path = workspace / "arithmetic-output.json"
    executor_path = workspace / "executor.py"
    verifier_path = workspace / verifier_name
    write_json(
        input_schema_path,
        schema(
            "arithmetic-input",
            {
                "type": "object",
                "fields": {
                    "left": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
                    "right": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
                    "label": {"type": "string", "min_bytes": 1, "max_bytes": 128},
                },
                "required": ["left", "right"],
                "additional_properties": False,
            },
        ),
    )
    write_json(
        output_schema_path,
        schema(
            "arithmetic-output",
            {
                "type": "object",
                "fields": {"value": {"type": "integer", "minimum": -2000000, "maximum": 2000000}},
                "required": ["value"],
                "additional_properties": False,
            },
        ),
    )
    executor_path.write_text(arithmetic_executor_source(), encoding="utf-8", newline="\n")
    if not verifier_path.exists():
        verifier_path.write_text(arithmetic_verifier_source(), encoding="utf-8", newline="\n")
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "id": "synthetic-arithmetic-registry",
        "tools": [
            {
                "id": "checked-arithmetic",
                "source": "synthetic independent arithmetic fixture",
                "license": "MIT",
                "side_effect": "none",
                "input_schema": {"path": input_schema_path.name, "sha256": digest_path(input_schema_path)},
                "output_schema": {"path": output_schema_path.name, "sha256": digest_path(output_schema_path)},
                "executor": {
                    "implementation": {
                        "path": executor_path.name,
                        "sha256": "0" * 64 if forged_executor_hash else digest_path(executor_path),
                    },
                    "command": [sys.executable, "{implementation}", "--input", "{input}", "--output", "{output}"],
                    "independence_group": "executor-implementation",
                },
                "verifier": {
                    "implementation": {"path": verifier_path.name, "sha256": digest_path(verifier_path)},
                    "command": [
                        sys.executable, "{implementation}", "--input", "{input}", "--result", "{result}", "--output", "{output}",
                    ],
                    "independence_group": "verifier-implementation",
                },
                "timeout_seconds": 10,
                "maximum_result_bytes": 16384,
                "max_invocations_per_plan": 4,
            }
        ],
    }
    registry_path = workspace / "registry.json"
    write_json(registry_path, registry)
    return registry_path


def create_request(workspace: Path) -> tuple[Path, str]:
    request_mapping: dict[str, object] = {
        "schema_version": "vegai-lmm-request/v1",
        "id": "tool-plan-request",
        "messages": [
            {
                "id": "system-policy",
                "role": "system",
                "content": [{"type": "text", "text": "Use only verified tool evidence."}],
                "pinned": True,
            },
            {
                "id": "user-question",
                "role": "user",
                "content": [{"type": "text", "text": "Compute a governed value."}],
                "pinned": False,
            },
        ],
        "max_new_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    request_path = workspace / "request.json"
    write_json(request_path, request_mapping)
    return request_path, GenerationRequest.from_mapping(request_mapping).digest()


def create_plan(
    workspace: Path,
    request_sha256: str,
    registry_path: Path,
    *,
    name: str = "plan.json",
    bind_request_sha256: str | None = None,
    forward_reference: bool = False,
) -> Path:
    second_left: object = {"$vegai_ref": "first#/value"}
    if forward_reference:
        second_left = {"$vegai_ref": "second#/value"}
    plan = {
        "schema_version": PLAN_SCHEMA,
        "id": "synthetic-arithmetic-plan",
        "request_sha256": bind_request_sha256 or request_sha256,
        "registry_sha256": digest_path(registry_path),
        "invocations": [
            {
                "id": "first",
                "tool_id": "checked-arithmetic",
                "input": {"left": 2, "right": 3, "label": "Türkçe doğrulanmış araç verisi"},
                "idempotency_key": hashlib.sha256(b"first invocation").hexdigest(),
            },
            {
                "id": "second",
                "tool_id": "checked-arithmetic",
                "input": {"left": second_left, "right": 7, "label": "ikinci adım"},
                "idempotency_key": hashlib.sha256(b"second invocation").hexdigest(),
            },
        ],
    }
    path = workspace / name
    write_json(path, plan)
    return path


def command(registry: Path, request: Path, plan: Path, evidence: Path, report: Path) -> list[str]:
    return [
        sys.executable,
        str(ORCHESTRATOR),
        "execute",
        "--registry", str(registry),
        "--request", str(request),
        "--plan", str(plan),
        "--evidence-output", str(evidence),
        "--report-output", str(report),
    ]


def main() -> int:
    require(ORCHESTRATOR.is_file(), f"missing {ORCHESTRATOR}")
    with tempfile.TemporaryDirectory(prefix="vegai-tool-orchestrator-test-") as temporary_name:
        workspace = Path(temporary_name)
        registry = create_registry(workspace)
        request, request_digest = create_request(workspace)
        plan = create_plan(workspace, request_digest, registry)
        evidence = workspace / "evidence.json"
        report = workspace / "report.json"

        completed = run(command(registry, request, plan, evidence, report), expected=0)
        require('"invocations": 2' in completed.stdout, "success summary did not report two invocations")
        evidence_mapping = json.loads(evidence.read_text(encoding="utf-8"))
        report_mapping = json.loads(report.read_text(encoding="utf-8"))
        require(evidence_mapping["query_sha256"] == request_digest, "evidence did not bind the request digest")
        require([item["id"] for item in evidence_mapping["items"]] == ["tool-first", "tool-second"],
                "evidence item IDs are not deterministic")
        values = [json.loads(item["text"])["value"] for item in evidence_mapping["items"]]
        require(values == [5, 12], "typed predecessor reference or checked arithmetic result is wrong")
        require(report_mapping["outcome"] == "completed", "report outcome is wrong")
        require(report_mapping["registry"]["identity"]["path"] == "<redacted>", "registry path leaked into report")
        require(report_mapping["plan"]["identity"]["path"] == "<redacted>", "plan path leaked into report")
        require(report_mapping["request"]["identity"]["path"] == "<redacted>", "request path leaked into report")
        rendered_report = json.dumps(report_mapping, ensure_ascii=False, sort_keys=True)
        require('"left"' not in rendered_report and '"right"' not in rendered_report,
                "raw tool arguments leaked into execution report")
        require('"value"' not in rendered_report, "raw tool result leaked into execution report")
        require("Türkçe doğrulanmış araç verisi" not in rendered_report,
                "raw Unicode tool argument leaked into execution report")

        collision = run(command(registry, request, plan, evidence, report), expected=2)
        require("already exists" in collision.stderr, "existing output was not fail-closed")

        wrong_binding_evidence = workspace / "wrong-binding-evidence.json"
        wrong_binding_report = workspace / "wrong-binding-report.json"
        wrong_binding_plan = create_plan(
            workspace, request_digest, registry, name="wrong-binding-plan.json", bind_request_sha256="0" * 64
        )
        wrong_binding = run(command(registry, request, wrong_binding_plan, wrong_binding_evidence, wrong_binding_report), expected=2)
        require("does not bind the supplied LMM request" in wrong_binding.stderr, "forged request binding was accepted")
        require(not wrong_binding_evidence.exists() and not wrong_binding_report.exists(),
                "forged request binding published an artifact")

        forward_evidence = workspace / "forward-evidence.json"
        forward_report = workspace / "forward-report.json"
        forward_plan = create_plan(workspace, request_digest, registry, name="forward-plan.json", forward_reference=True)
        forward = run(command(registry, request, forward_plan, forward_evidence, forward_report), expected=2)
        require("current, or future invocation" in forward.stderr, "forward tool-output reference was accepted")
        require(not forward_evidence.exists() and not forward_report.exists(), "invalid reference published an artifact")

        failing_workspace = workspace / "failing-verifier"
        failing_workspace.mkdir()
        bad_verifier_path = failing_workspace / "bad-verifier.py"
        bad_verifier_path.write_text(arithmetic_verifier_source("fail"), encoding="utf-8", newline="\n")
        failing_registry = create_registry(failing_workspace, verifier_name=bad_verifier_path.name)
        failing_plan = create_plan(failing_workspace, request_digest, failing_registry, name="failing-verifier-plan.json")
        failing_evidence = failing_workspace / "failing-verifier-evidence.json"
        failing_report = failing_workspace / "failing-verifier-report.json"
        failing = run(command(failing_registry, request, failing_plan, failing_evidence, failing_report), expected=2)
        require("did not return verdict 'pass'" in failing.stderr, "failed independent verifier was accepted")
        require(not failing_evidence.exists() and not failing_report.exists(), "failed verifier published an artifact")

        mutating_workspace = workspace / "mutating-verifier"
        mutating_workspace.mkdir()
        mutating_verifier_path = mutating_workspace / "mutating-verifier.py"
        mutating_verifier_path.write_text(
            arithmetic_verifier_source(mutate_result=True), encoding="utf-8", newline="\n"
        )
        mutating_registry = create_registry(mutating_workspace, verifier_name=mutating_verifier_path.name)
        mutating_plan = create_plan(mutating_workspace, request_digest, mutating_registry, name="mutating-verifier-plan.json")
        mutating_evidence = mutating_workspace / "mutating-verifier-evidence.json"
        mutating_report = mutating_workspace / "mutating-verifier-report.json"
        mutating = run(command(mutating_registry, request, mutating_plan, mutating_evidence, mutating_report), expected=2)
        require("verifier result changed" in mutating.stderr, "verifier result mutation was accepted")
        require(not mutating_evidence.exists() and not mutating_report.exists(), "mutating verifier published an artifact")

        forged_workspace = workspace / "forged-registry"
        forged_workspace.mkdir()
        forged_registry = create_registry(forged_workspace, forged_executor_hash=True)
        forged_plan = create_plan(forged_workspace, request_digest, forged_registry, name="forged-registry-plan.json")
        forged_evidence = forged_workspace / "forged-evidence.json"
        forged_report = forged_workspace / "forged-report.json"
        forged = run(command(forged_registry, request, forged_plan, forged_evidence, forged_report), expected=2)
        require("does not match its declared SHA-256" in forged.stderr, "forged implementation identity was accepted")
        require(not forged_evidence.exists() and not forged_report.exists(), "forged registry published an artifact")

        duplicate_plan = workspace / "duplicate-plan.json"
        duplicate_plan.write_text(
            '{"schema_version":"vegai-lmm-tool-plan/v1","schema_version":"vegai-lmm-tool-plan/v1"}\n',
            encoding="utf-8",
            newline="\n",
        )
        duplicate = run(
            command(registry, request, duplicate_plan, workspace / "duplicate-evidence.json", workspace / "duplicate-report.json"),
            expected=2,
        )
        require("duplicate key" in duplicate.stderr, "duplicate JSON key was accepted")

    print("LMM tool orchestrator verification: PASS (typed plan, references, independent verification, provenance, privacy, fail-closed gates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
