#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Adversarial verification for held-out HDC checkpoint quality evaluation.

The verifier uses a tiny deterministic inspect fixture for strict schema,
privacy, failed-gate, and TOCTOU checks, then invokes the supplied compiled
``vegai_pretrain`` binary to ensure the evaluator consumes an actual HDC
checkpoint through its public batch-inspection contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from pretraining_quality_evaluation import evaluator_binding
from production_pretrain import ProductionRunError, quality_report_summary


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools" / "pretraining_quality_evaluation.py"


def fail(message: str) -> None:
    raise AssertionError(message)


def run(
    command: list[str],
    expected: int = 0,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    child_environment = os.environ.copy()
    if environment is not None:
        child_environment.update(environment)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=child_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != expected:
        fail(
            f"command returned {completed.returncode}, expected {expected}: {' '.join(command)}\n{completed.stdout}"
        )
    return completed


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cases(*, wrong_expectation: bool = False) -> dict[str, object]:
    first_expected = ["omega"] if wrong_expectation else ["beta"]
    return {
        "schema_version": "vegai-pretraining-quality-cases/v1",
        "evaluation_id": "quality-verifier-held-out-v1",
        "declared_split": "held-out",
        "annotation_provenance": {
            "source": "repository verifier fixture",
            "license": "MIT",
            "retrieved_at": "source-controlled verifier fixture",
            "method": "deterministic contract fixture/v1",
        },
        "cases": [
            {
                "schema_version": "vegai-pretraining-quality-case/v1",
                "id": "case-alpha",
                "query_token": "alpha",
                "acceptable_neighbours": first_expected,
            },
            {
                "schema_version": "vegai-pretraining-quality-case/v1",
                "id": "case-gamma",
                "query_token": "gamma",
                "acceptable_neighbours": ["delta"],
            },
        ],
    }


def policy() -> dict[str, object]:
    return {
        "schema_version": "vegai-pretraining-quality-policy/v1",
        "policy_id": "quality-verifier-policy-v1",
        "minimum_cases": 2,
        "minimum_top1_accuracy": 1.0,
        "minimum_top1_wilson_lower_bound_95": 0.0,
        "minimum_mean_similarity": 0.1,
        "minimum_case_similarity": 0.1,
        "maximum_incorrect_cases": 0,
    }


def evaluator_command(
    binary: Path,
    checkpoint: Path,
    case_path: Path,
    policy_path: Path,
    key: Path,
    output: Path,
    scratch: Path | None = None,
    timeout_seconds: str = "15",
) -> list[str]:
    command = [
        sys.executable, str(EVALUATOR),
        "--pretrain-bin", str(binary), "--checkpoint", str(checkpoint),
        "--cases", str(case_path), "--policy", str(policy_path),
        "--redaction-key-file", str(key), "--output", str(output), "--timeout-seconds", timeout_seconds,
    ]
    if scratch is not None:
        command.extend(["--scratch-dir", str(scratch)])
    return command


def write_fixture_binary(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "queries = Path(sys.argv[-1]).read_text(encoding='ascii').splitlines()\n"
        "if os.environ.get('VEGAI_QUALITY_VERIFIER_MUTATE_KEY'):\n"
        "    Path(os.environ['VEGAI_QUALITY_VERIFIER_MUTATE_KEY']).write_bytes(b'replaced-key')\n"
        "answers = {'alpha': ('beta', 0.75000), 'gamma': ('delta', 0.50000)}\n"
        "for query in queries:\n"
        "    neighbour, similarity = answers[query]\n"
        "    print('query: ' + query)\n"
        "    print('  nearest semantic neighbour: ' + neighbour)\n"
        "    print('  bipolar similarity: %8.5f' % similarity)\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def verify_fake_contract(workspace: Path) -> None:
    binary = workspace / "fixture-inspect"
    checkpoint = workspace / "fixture-checkpoint.vgai"
    case_path = workspace / "cases.json"
    policy_path = workspace / "policy.json"
    key = workspace / "redaction.key"
    report = workspace / "report.json"
    write_fixture_binary(binary)
    checkpoint.write_bytes(b"not-a-real-checkpoint-but-a-stable-fixture\n")
    write_json(case_path, cases())
    write_json(policy_path, policy())
    key.write_bytes(b"quality-verifier-local-secret\n")

    completed = run(evaluator_command(binary, checkpoint, case_path, policy_path, key, report, scratch=workspace))
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        fail(f"quality evaluator did not emit one JSON report: {error}\n{completed.stdout}")
    if payload.get("schema_version") != "vegai-pretraining-quality-report/v1":
        fail(f"quality evaluator report has wrong schema: {payload}")
    if payload.get("quality_gate", {}).get("passed") is not True:
        fail(f"expected deterministic fixture quality gate to pass: {payload}")
    if payload.get("metrics", {}).get("top1_accuracy") != 1.0:
        fail(f"deterministic fixture did not report exact top-1 accuracy: {payload}")
    serialized = report.read_text(encoding="utf-8")
    for secret_text in ("alpha", "beta", "gamma", "delta", "case-alpha", "case-gamma"):
        if secret_text in serialized:
            fail(f"privacy report retained raw held-out content {secret_text!r}: {serialized}")
    if "case_id_hmac_sha256" not in serialized or "observed_neighbour_hmac_sha256" not in serialized:
        fail("privacy report omitted HMAC-bound per-case audit identifiers")

    # The production wrapper independently recomputes public aggregate metrics
    # from redacted case rows. A report-file replacement cannot obtain a pass by
    # changing only a boolean or summary number after the evaluator exits.
    binding = evaluator_binding(case_path, policy_path, key, maximum_cases=2)
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    quality_report_summary(report, binding, binary_sha256, checkpoint_sha256)
    tampered = json.loads(report.read_text(encoding="utf-8"))
    tampered["metrics"]["top1_accuracy"] = 0.0
    write_json(report, tampered)
    try:
        quality_report_summary(report, binding, binary_sha256, checkpoint_sha256)
    except ProductionRunError:
        pass
    else:
        fail("production wrapper accepted a quality report whose aggregate metric disagreed with case rows")
    # Restore the source evaluator artifact because later negative cases use
    # their own output paths but the fake fixture remains an audit reference.
    write_json(report, payload)

    # Two perfect cases are not enough evidence for a high 95% Wilson lower
    # bound. This tests that a policy can prevent raw-accuracy-only release.
    wilson_policy = workspace / "wilson-policy.json"
    wilson_report = workspace / "wilson-report.json"
    strict_wilson_policy = policy()
    strict_wilson_policy["minimum_top1_wilson_lower_bound_95"] = 0.90
    write_json(wilson_policy, strict_wilson_policy)
    run(evaluator_command(binary, checkpoint, case_path, wilson_policy, key, wilson_report), expected=3)
    wilson_payload = json.loads(wilson_report.read_text(encoding="utf-8"))
    if not any("wilson" in failure for failure in wilson_payload.get("quality_gate", {}).get("failures", [])):
        fail(f"Wilson lower-bound policy was not enforced: {wilson_payload}")

    failed_cases = workspace / "failed-cases.json"
    failed_report = workspace / "failed-report.json"
    write_json(failed_cases, cases(wrong_expectation=True))
    run(evaluator_command(binary, checkpoint, failed_cases, policy_path, key, failed_report), expected=3)
    failed_payload = json.loads(failed_report.read_text(encoding="utf-8"))
    gate = failed_payload.get("quality_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not False or not gate.get("failures"):
        fail(f"intentional wrong expected neighbour did not create a published failed gate: {failed_payload}")

    overflow_binary = workspace / "overflow-inspect"
    overflow_binary.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stdout.write('x' * (17 * 1024 * 1024))\n",
        encoding="utf-8",
    )
    overflow_binary.chmod(0o700)
    overflow_report = workspace / "overflow-report.json"
    overflow = run(
        evaluator_command(overflow_binary, checkpoint, case_path, policy_path, key, overflow_report), expected=2
    )
    if "emitted more than" not in overflow.stdout or overflow_report.exists():
        fail(f"oversized inspect stdout was not bounded/fail-closed\n{overflow.stdout}")

    sleeping_binary = workspace / "sleeping-inspect"
    sleeping_binary.write_text(
        "#!/usr/bin/env python3\n"
        "import time\n"
        "time.sleep(2.0)\n",
        encoding="utf-8",
    )
    sleeping_binary.chmod(0o700)
    timeout_report = workspace / "timeout-report.json"
    timed_out = run(
        evaluator_command(
            sleeping_binary, checkpoint, case_path, policy_path, key, timeout_report, timeout_seconds="0.1"
        ),
        expected=2,
    )
    if "exceeded --timeout-seconds" not in timed_out.stdout or timeout_report.exists():
        fail(f"timed-out inspect was not fail-closed\n{timed_out.stdout}")

    # Mutating the redaction secret during the external inspection call must be
    # detected rather than yielding a report with an ambiguous HMAC namespace.
    key.write_bytes(b"quality-verifier-local-secret\n")
    raced_report = workspace / "raced-report.json"
    raced = run(
        evaluator_command(binary, checkpoint, case_path, policy_path, key, raced_report),
        expected=2,
        environment={"VEGAI_QUALITY_VERIFIER_MUTATE_KEY": str(key)},
    )
    if "changed while evaluation was running" not in raced.stdout:
        fail(f"redaction-key TOCTOU mutation was not diagnosed\n{raced.stdout}")
    if raced_report.exists():
        fail("TOCTOU-rejected evaluation published a report")

    duplicate_key_cases = workspace / "duplicate-key-cases.json"
    duplicate_key_cases.write_text(
        '{"schema_version":"vegai-pretraining-quality-cases/v1",'
        '"schema_version":"vegai-pretraining-quality-cases/v1"}\n',
        encoding="ascii",
    )
    key.write_bytes(b"quality-verifier-local-secret\n")
    duplicate = run(
        evaluator_command(binary, checkpoint, duplicate_key_cases, policy_path, key, workspace / "duplicate.json"), expected=2
    )
    if "repeats key" not in duplicate.stdout:
        fail(f"duplicate-key case manifest was not rejected strictly\n{duplicate.stdout}")

    no_provenance_cases = workspace / "no-provenance-cases.json"
    no_provenance = cases()
    no_provenance.pop("annotation_provenance")
    write_json(no_provenance_cases, no_provenance)
    missing_provenance = run(
        evaluator_command(binary, checkpoint, no_provenance_cases, policy_path, key, workspace / "no-provenance.json"),
        expected=2,
    )
    if "annotation_provenance" not in missing_provenance.stdout:
        fail(f"case annotation provenance was not required\n{missing_provenance.stdout}")


def verify_real_checkpoint(binary: Path, workspace: Path) -> None:
    corpus = workspace / "real-corpus.txt"
    checkpoint = workspace / "real-checkpoint.vgai"
    # The case labels intentionally accept every other vocabulary item.  This
    # is not a semantic-quality claim; it only proves that the release tool
    # consumes a real HDC checkpoint and the public self-excluded inspect ABI.
    corpus.write_text(
        "alpha beta gamma\nalpha beta gamma\nalpha beta gamma\n"
        "delta epsilon zeta\ndelta epsilon zeta\ndelta epsilon zeta\n",
        encoding="ascii",
    )
    run(
        [
            str(binary), "train", str(corpus), str(checkpoint), "--max-vocab", "16", "--min-frequency", "1",
            "--window", "2", "--position-shift", "37", "--context-mode", "hierarchical", "--epochs", "1",
            "--initial-plasticity", "25", "--minimum-plasticity", "1", "--decay-events", "8", "--strict",
        ]
    )
    real_cases = {
        "schema_version": "vegai-pretraining-quality-cases/v1",
        "evaluation_id": "real-checkpoint-interface-v1",
        "declared_split": "held-out",
        "annotation_provenance": {
            "source": "repository verifier fixture",
            "license": "MIT",
            "retrieved_at": "source-controlled verifier fixture",
            "method": "real checkpoint inspect ABI fixture/v1",
        },
        "cases": [
            {
                "schema_version": "vegai-pretraining-quality-case/v1",
                "id": "real-alpha",
                "query_token": "alpha",
                "acceptable_neighbours": ["beta", "gamma", "delta", "epsilon", "zeta"],
            },
            {
                "schema_version": "vegai-pretraining-quality-case/v1",
                "id": "real-delta",
                "query_token": "delta",
                "acceptable_neighbours": ["alpha", "beta", "gamma", "epsilon", "zeta"],
            },
        ],
    }
    real_policy = {
        "schema_version": "vegai-pretraining-quality-policy/v1",
        "policy_id": "real-checkpoint-interface-policy-v1",
        "minimum_cases": 2,
        "minimum_top1_accuracy": 1.0,
        "minimum_top1_wilson_lower_bound_95": 0.0,
        "minimum_mean_similarity": -1.0,
        "minimum_case_similarity": -1.0,
        "maximum_incorrect_cases": 0,
    }
    case_path = workspace / "real-cases.json"
    policy_path = workspace / "real-policy.json"
    key = workspace / "real.key"
    report = workspace / "real-report.json"
    write_json(case_path, real_cases)
    write_json(policy_path, real_policy)
    key.write_bytes(b"real-checkpoint-hmac-secret\n")
    run(evaluator_command(binary, checkpoint, case_path, policy_path, key, report))
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("quality_gate", {}).get("passed") is not True:
        fail(f"real compiled checkpoint did not satisfy interface fixture gate: {payload}")
    if payload.get("provenance", {}).get("checkpoint", {}).get("bytes", 0) <= 0:
        fail(f"real evaluator report omitted checkpoint provenance: {payload}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    arguments = parser.parse_args()
    binary = arguments.pretrain_bin.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        print(f"pretraining quality evaluation verification: FAIL: unusable executable {binary}", file=sys.stderr)
        return 2
    try:
        with tempfile.TemporaryDirectory(prefix="vegai-pretraining-quality-") as temporary:
            workspace = Path(temporary)
            fake = workspace / "fake"
            fake.mkdir()
            verify_fake_contract(fake)
            real = workspace / "real"
            real.mkdir()
            verify_real_checkpoint(binary, real)
    except (AssertionError, OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as error:
        print(f"pretraining quality evaluation verification: FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "pretraining quality evaluation verification: PASS "
        "(strict schema, HMAC privacy, recomputed production report metrics, bounded inspect I/O, failed gate, TOCTOU rejection, real checkpoint inspect ABI)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
