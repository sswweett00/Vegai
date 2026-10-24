#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verification for revision-benchmark checkpoint contract gates.

The subprocess fixture is a protocol emulator, not a model. Real train-process
timing remains the responsibility of compare_pretraining_revisions.py and CI;
this verifier checks that a reviewed versioned migration cannot silently weaken
the default byte-identity contract or omit baseline-checkpoint continuation.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "compare_pretraining_revisions.py"
sys.path.insert(0, str(ROOT / "tools"))
from compare_pretraining_revisions import (  # noqa: E402
    CHECKPOINT_CONTRACT_IDENTICAL,
    CHECKPOINT_CONTRACT_VERSIONED_TRANSITION,
    RevisionComparisonError,
    validate_checkpoint_contract,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_rejected(before: bytes, after: bytes, contract: str, label: str) -> None:
    try:
        validate_checkpoint_contract(label, before, after, contract)
    except RevisionComparisonError:
        return
    raise AssertionError(f"{label}: invalid checkpoint relation was accepted")


def write_fake_pretrain(path: Path, fresh_checkpoint: bytes) -> None:
    """Create a deterministic CLI emulator for harness-flow verification."""
    payload = f'''#!/usr/bin/env python3
from pathlib import Path
import sys

command, corpus_name, checkpoint_name = sys.argv[1:4]
corpus = Path(corpus_name)
checkpoint = Path(checkpoint_name)
epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 1
tokens = len(corpus.read_text(encoding="ascii").split()) * epochs
if command == "train":
    checkpoint.write_bytes({fresh_checkpoint!r})
    print("phase: vocabulary")
elif command == "resume":
    if checkpoint.read_bytes() != b"v5-fresh":
        raise SystemExit("unexpected legacy checkpoint")
    checkpoint.write_bytes(b"v5-resumed")
else:
    raise SystemExit("unexpected command")
print("phase: streaming HDC context learning")
print("  raw tokens: " + str(tokens))
print("  active tokens: " + str(tokens))
print("  out-of-vocabulary tokens: 0")
print("  prototype updates: " + str(tokens))
'''
    path.write_text(payload, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def verify_migration_subprocess_flow() -> None:
    with tempfile.TemporaryDirectory(prefix="vegai-revision-contract-") as temporary_name:
        workspace = Path(temporary_name)
        baseline = workspace / "baseline-pretrain.py"
        candidate = workspace / "candidate-pretrain.py"
        output = workspace / "report.json"
        write_fake_pretrain(baseline, b"v5-fresh")
        write_fake_pretrain(candidate, b"v7-fresh")
        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--before-pretrain-bin",
                str(baseline),
                "--after-pretrain-bin",
                str(candidate),
                "--repetitions",
                "1",
                "--checkpoint-contract",
                CHECKPOINT_CONTRACT_VERSIONED_TRANSITION,
                "--output",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        require(completed.returncode == 0, f"migration harness fixture failed:\n{completed.stdout}")
        report = json.loads(output.read_text(encoding="utf-8"))
        require(report["schema"] == "vegai-pretraining-revision-comparison/v2", "migration report schema is stale")
        require(report["all_checkpoint_contracts_satisfied"], "migration report did not mark contract success")
        require(not report["all_checkpoints_byte_identical"], "migration report incorrectly claimed cross-lineage equality")
        for workload in report["workloads"]:
            resume = workload["legacy_resume_compatibility"]
            require(workload["checkpoint_relation"] == "intentional-versioned-algorithm-transition",
                    "migration workload did not record its explicit checkpoint relation")
            require(isinstance(resume, dict) and resume["resumed_checkpoint_byte_identical_to_baseline"],
                    "migration workload omitted baseline checkpoint resume proof")


def main() -> int:
    identical, relation = validate_checkpoint_contract(
        "equal", b"checkpoint", b"checkpoint", CHECKPOINT_CONTRACT_IDENTICAL
    )
    require(identical and relation == "byte-identical", "normal contract did not preserve byte identity")
    expect_rejected(b"before", b"after", CHECKPOINT_CONTRACT_IDENTICAL, "normal-different")

    identical, relation = validate_checkpoint_contract(
        "migration", b"v5 payload", b"v7 payload", CHECKPOINT_CONTRACT_VERSIONED_TRANSITION
    )
    require(not identical and relation == "intentional-versioned-algorithm-transition",
            "versioned migration contract did not identify the explicit transition")
    expect_rejected(b"same", b"same", CHECKPOINT_CONTRACT_VERSIONED_TRANSITION, "migration-equal")
    expect_rejected(b"before", b"after", "unknown-contract", "unknown")
    verify_migration_subprocess_flow()

    print("pretraining revision comparison verification: PASS (default byte identity, explicit migration, baseline resume proof)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
