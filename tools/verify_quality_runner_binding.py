#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify quality-evaluator runner provenance and immutable-output guards."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import continuous_pretrain as continuous
import continuous_quality_gate as comparator
import pretraining_quality_evaluation as evaluator
import production_pretrain as production
from verify_continuous_quality_regression import quality_inputs


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rejected(action, error_type, message: str) -> None:
    try:
        action()
    except error_type:
        return
    raise AssertionError(message)


def verify(workspace: Path) -> None:
    cases, policy, regression, key = quality_inputs(workspace, strict_regression=True)
    binary = Path(sys.executable).resolve()
    checkpoint = workspace / "checkpoint.vgai"
    checkpoint.write_bytes(b"quality binding fixture\n")
    helper = workspace / "runner.py"
    original_helper = evaluator.PROCESS_RUNNER_SOURCE.read_bytes()
    helper.write_bytes(original_helper)
    arguments = SimpleNamespace(
        quality_cases=cases, quality_policy=policy, quality_redaction_key_file=key,
        quality_regression_policy=regression, quality_max_cases=2,
        quality_timeout_seconds=10.0, quality_scratch_dir=workspace,
    )

    # The binding must change when the imported helper changes, independently
    # of the evaluator file. Use a disposable copy so the repository is safe.
    with patch.object(evaluator, "PROCESS_RUNNER_SOURCE", helper):
        binding = evaluator.evaluator_binding(cases, policy, key, maximum_cases=2)
        guard = continuous.quality_guard_from_arguments(arguments)
        require(guard is not None, "quality guard was not configured")
        continuous.validate_quality_guard_binding(guard)
        helper.write_bytes(original_helper + b"\n# mutated\n")
        changed_binding = evaluator.evaluator_binding(cases, policy, key, maximum_cases=2)
        require(binding["process_runner_sha256"] != changed_binding["process_runner_sha256"],
                "quality preflight did not bind process-runner content")
        require(binding["evaluator_source_sha256"] == changed_binding["evaluator_source_sha256"],
                "runner mutation incorrectly changed the evaluator identity")
        helper.write_bytes(original_helper)

    # Old v1 guards remain readable for status/audit, but a newly computed
    # guard contains the runner hash and therefore cannot equal a legacy one.
    legacy_guard = deepcopy(guard)
    legacy_guard["quality_evidence"].pop("process_runner_sha256")
    continuous.validate_quality_guard_binding(legacy_guard)
    require(legacy_guard != guard, "legacy guard could resume under a new runner binding")
    malformed = deepcopy(guard)
    malformed["quality_evidence"]["process_runner_sha256"] = "not-a-sha256"
    rejected(lambda: continuous.validate_quality_guard_binding(malformed), continuous.ContinuousPretrainError,
             "malformed runner hash was accepted in a guard")

    inputs = evaluator.load_evaluation_inputs(cases, policy, key, maximum_cases=2)
    binary_identity = evaluator.file_identity(binary, "fixture binary", evaluator.MAX_BINARY_BYTES)
    checkpoint_identity = evaluator.file_identity(checkpoint, "fixture checkpoint", evaluator.MAX_CHECKPOINT_BYTES)
    evaluator_identity = evaluator.file_identity(Path(evaluator.__file__), "evaluator", evaluator.MAX_CASE_MANIFEST_BYTES)
    runner_identity = evaluator.file_identity(helper, "runner", evaluator.MAX_CASE_MANIFEST_BYTES)
    observations = [
        evaluator.InspectObservation("alpha", "beta", 0.5),
        evaluator.InspectObservation("delta", "alpha", 0.5),
    ]
    payload = evaluator.build_report(
        evaluator_identity, runner_identity, binary_identity, checkpoint_identity,
        inputs, observations, b"synthetic inspect output", 0.1,
    )
    report = workspace / "report.json"

    def write_report(value: dict) -> None:
        report.write_text(json.dumps(value), encoding="utf-8")

    def production_summary(current_binding: dict) -> None:
        production.quality_report_summary(report, current_binding, binary_identity.sha256, checkpoint_identity.sha256)

    def comparison_summary(current_binding: dict) -> None:
        comparator.read_validated_report(report, current_binding, binary_identity.sha256, checkpoint_identity.sha256)

    write_report(payload)
    production_summary(binding)
    comparison_summary(binding)

    def committed_entry(current_guard: dict) -> dict:
        report_bytes = report.read_bytes()
        report_sha = hashlib.sha256(report_bytes).hexdigest()
        comparison_path = workspace / "comparison.json"
        comparison_payload = {
            "schema_version": "vegai-continuous-quality-regression-report/v1",
            "status": "passed", "failures": [],
            "baseline": {"checkpoint_sha256": checkpoint_identity.sha256},
            "candidate": {"checkpoint_sha256": checkpoint_identity.sha256, "quality_report_sha256": report_sha},
            "regression_policy": current_guard["regression_policy"],
        }
        comparison_path.write_text(json.dumps(comparison_payload), encoding="utf-8")
        comparison_bytes = comparison_path.read_bytes()
        return {
            "checkpoint_sha256": checkpoint_identity.sha256,
            "quality": {
                "schema": "vegai-continuous-quality-evidence/v1",
                "quality_report": report.name, "quality_report_sha256": report_sha,
                "quality_report_bytes": len(report_bytes), "comparison_report": comparison_path.name,
                "comparison_report_sha256": hashlib.sha256(comparison_bytes).hexdigest(),
                "comparison_report_bytes": len(comparison_bytes), "anchor_generation": 1,
                "anchor_checkpoint_sha256": checkpoint_identity.sha256,
            },
        }

    entry = committed_entry(guard)
    continuous.validate_quality_evidence_entry(workspace, entry, guard, 1, binary_identity.sha256)
    tampered_payload = deepcopy(payload)
    tampered_payload["provenance"]["process_runner"]["sha256"] = "0" * 64
    write_report(tampered_payload)
    entry = committed_entry(guard)  # Rehash so provenance, not the byte check, rejects it.
    rejected(lambda: continuous.validate_quality_evidence_entry(workspace, entry, guard, 1, binary_identity.sha256),
             continuous.ContinuousPretrainError, "continuous state accepted a report bound to another runner")
    write_report(payload)
    for label, current in (("changed source", changed_binding),):
        rejected(lambda: production_summary(current), production.ProductionRunError,
                 f"production accepted {label} runner provenance")
        rejected(lambda: comparison_summary(current), comparator.ContinuousQualityGateError,
                 f"comparison accepted {label} runner provenance")
    without_runner = deepcopy(payload)
    without_runner["provenance"].pop("process_runner")
    write_report(without_runner)
    rejected(lambda: production_summary(binding), production.ProductionRunError,
             "production accepted a report missing runner provenance")
    rejected(lambda: comparison_summary(binding), comparator.ContinuousQualityGateError,
             "comparison accepted a report missing runner provenance")
    legacy_binding = dict(binding)
    legacy_binding.pop("process_runner_sha256")
    production_summary(legacy_binding)
    comparison_summary(legacy_binding)
    legacy_entry = committed_entry(legacy_guard)
    continuous.validate_quality_evidence_entry(workspace, legacy_entry, legacy_guard, 1, binary_identity.sha256)

    # Check both controllers reject a helper changed before the subprocess and
    # one changed while it runs. Mock only the child and scratch sizing.
    scratch = {"directory": str(workspace)}
    with patch.object(continuous, "PROCESS_RUNNER_SOURCE", helper), \
         patch.object(continuous, "quality_scratch_capacity", return_value=scratch), \
         patch.object(continuous, "run_bounded_child") as child:
        helper.write_bytes(original_helper + b"\n# before\n")
        rejected(lambda: continuous.run_quality_evaluator(binary, checkpoint, report, guard, arguments),
                 continuous.ContinuousPretrainError, "continuous controller accepted a changed runner")
        require(not child.called, "continuous controller started a child after runner mutation")
        helper.write_bytes(original_helper)

        def mutate_continuous(*args, **kwargs):
            helper.write_bytes(original_helper + b"\n# during\n")
            return SimpleNamespace(returncode=0)

        child.side_effect = mutate_continuous
        rejected(lambda: continuous.run_quality_evaluator(binary, checkpoint, report, guard, arguments),
                 continuous.ContinuousPretrainError, "continuous controller ignored runner mutation during evaluation")
        require(child.called, "continuous runner-mutation fixture did not execute")

    with patch.object(production, "PROCESS_RUNNER_SOURCE", helper), \
         patch.object(production, "quality_scratch_capacity", return_value=scratch), \
         patch.object(production.subprocess, "run") as child:
        helper.write_bytes(original_helper + b"\n# before\n")
        result = production.run_quality_gate(binary, binary_identity.sha256, checkpoint,
                                             checkpoint_identity.sha256, workspace, arguments, binding)
        require(result["status"] == "execution-failed" and not child.called,
                "production controller started a child after runner mutation")
        helper.write_bytes(original_helper)

        def mutate_production(*args, **kwargs):
            helper.write_bytes(original_helper + b"\n# during\n")
            return SimpleNamespace(returncode=0)

        child.side_effect = mutate_production
        result = production.run_quality_gate(binary, binary_identity.sha256, checkpoint,
                                             checkpoint_identity.sha256, workspace, arguments, binding)
        require(result["status"] == "execution-failed" and child.called,
                "production controller ignored runner mutation during evaluation")
        helper.unlink()
        child.reset_mock(side_effect=True)
        result = production.run_quality_gate(binary, binary_identity.sha256, checkpoint,
                                             checkpoint_identity.sha256, workspace, arguments, binding)
        require(result["status"] == "execution-failed" and not child.called,
                "production controller did not fail closed for a missing runner")
        helper.write_bytes(original_helper)
        child.side_effect = subprocess.TimeoutExpired("quality-evaluator", 310.0)
        result = production.run_quality_gate(binary, binary_identity.sha256, checkpoint,
                                             checkpoint_identity.sha256, workspace, arguments, binding)
        require(result["status"] == "execution-failed" and child.call_args.kwargs["timeout"] == 310.0,
                "production quality wrapper did not enforce its administrative timeout")

    # Run a copied evaluator so an output-overwrite regression cannot destroy
    # the working tree. It must reject a report path naming its runner.
    copied_dir = workspace / "copied-tool"
    copied_dir.mkdir()
    copied_evaluator = copied_dir / "pretraining_quality_evaluation.py"
    copied_runner = copied_dir / "bounded_process.py"
    shutil.copyfile(evaluator.__file__, copied_evaluator)
    shutil.copyfile(evaluator.PROCESS_RUNNER_SOURCE, copied_runner)
    copied_bytes = copied_runner.read_bytes()
    command = [
        sys.executable, str(copied_evaluator), "--pretrain-bin", str(binary),
        "--checkpoint", str(checkpoint), "--cases", str(cases), "--policy", str(policy),
        "--redaction-key-file", str(key), "--output", str(copied_runner),
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    require(completed.returncode == 2 and "must not overwrite bounded process runner" in completed.stdout,
            f"runner output collision was not rejected: {completed.stdout}")
    require(copied_runner.read_bytes() == copied_bytes, "output collision overwrote the runner")

    # The comparison CLI also writes atomically, so it must not replace any
    # checkpoint, report, policy, secret, or source dependency with its output.
    copied_comparator = copied_dir / "continuous_quality_gate.py"
    shutil.copyfile(comparator.__file__, copied_comparator)
    comparator_payload = deepcopy(payload)
    comparator_payload["evaluator"] = evaluator.file_identity(
        copied_evaluator, "copied evaluator", evaluator.MAX_CASE_MANIFEST_BYTES
    ).public()
    comparator_payload["provenance"]["process_runner"] = evaluator.file_identity(
        copied_runner, "copied runner", evaluator.MAX_CASE_MANIFEST_BYTES
    ).public()
    copied_binary = workspace / "binary-copy"
    shutil.copyfile(binary, copied_binary)
    comparator_payload["provenance"]["pretrain_binary"] = evaluator.file_identity(
        copied_binary, "copied binary", evaluator.MAX_BINARY_BYTES
    ).public()
    baseline_report = workspace / "baseline-report.json"
    candidate_report = workspace / "candidate-report.json"
    baseline_report.write_text(json.dumps(comparator_payload), encoding="utf-8")
    candidate_report.write_text(json.dumps(comparator_payload), encoding="utf-8")
    comparison_output = workspace / "comparison-output.json"
    comparison_command = [
        sys.executable, str(copied_comparator), "--pretrain-bin", str(copied_binary),
        "--baseline-checkpoint", str(checkpoint), "--candidate-checkpoint", str(checkpoint),
        "--baseline-report", str(baseline_report), "--candidate-report", str(candidate_report),
        "--cases", str(cases), "--quality-policy", str(policy), "--redaction-key-file", str(key),
        "--regression-policy", str(regression), "--output", str(comparison_output),
    ]
    completed = subprocess.run(comparison_command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=15)
    require(completed.returncode == 0 and json.loads(comparison_output.read_text())["status"] == "passed",
            f"valid comparator fixture failed: {completed.stdout}")
    dependencies = (copied_comparator, copied_evaluator, copied_runner, copied_binary, checkpoint, baseline_report,
                    candidate_report, cases, policy, key, regression)
    for dependency in dependencies:
        before = dependency.read_bytes()
        comparison_command[-1] = str(dependency)
        completed = subprocess.run(comparison_command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=15)
        require(completed.returncode == 2 and "must not overwrite" in completed.stdout and
                dependency.read_bytes() == before,
                f"comparator output collision was not rejected for {dependency.name}: {completed.stdout}")
    alias = workspace / "checkpoint-hardlink"
    try:
        os.link(checkpoint, alias)
    except OSError:
        pass  # Some filesystems or Windows policies do not allow hard links.
    else:
        comparison_command[-1] = str(alias)
        completed = subprocess.run(comparison_command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=15)
        require(completed.returncode == 2 and "must not overwrite" in completed.stdout and
                os.path.samefile(alias, checkpoint),
                f"comparator accepted a hard-linked checkpoint output: {completed.stdout}")

    if os.name == "posix":
        # Exercise an actual child mutating the copied helper after import. No
        # native Fortran binary is needed and the real source is never touched.
        fake = workspace / "mutating-inspect"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import os, pathlib, sys\n"
            "p = pathlib.Path(os.environ['VEGAI_TEST_RUNNER'])\n"
            "p.write_bytes(p.read_bytes() + b'\\n# changed during inspect\\n')\n"
            "for q in pathlib.Path(sys.argv[-1]).read_text().splitlines():\n"
            "    n = 'beta' if q == 'alpha' else 'alpha'\n"
            "    print('query: ' + q)\n"
            "    print('  nearest semantic neighbour: ' + n)\n"
            "    print('  bipolar similarity:  0.50000')\n",
            encoding="utf-8",
        )
        fake.chmod(0o700)
        output = workspace / "mutated-report.json"
        # sys.executable may already be resolved and equal to binary. Replace
        # the option value, not the first matching string (the interpreter).
        command[command.index("--pretrain-bin") + 1] = str(fake)
        command[command.index("--output") + 1] = str(output)
        environment = os.environ.copy()
        environment["VEGAI_TEST_RUNNER"] = str(copied_runner)
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=15, env=environment)
        require(completed.returncode == 2 and "bounded process runner changed while evaluation" in completed.stdout,
                f"runner mutation during inspect was not rejected: {completed.stdout}")
        require(not output.exists(), "a report was published after runner mutation")


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="vegai-quality-runner-") as temporary:
            verify(Path(temporary))
    except (AssertionError, OSError, ValueError) as error:
        print(f"quality runner binding verification: FAIL: {error}", file=sys.stderr)
        return 1
    print("quality runner binding verification: PASS (identity, legacy audit, report validation, mutation, output collision)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
