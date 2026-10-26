#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Native and adversarial verification for continual quality-regression gates.

The real compiled binary subtest proves a frozen held-out evaluator runs before
both a first and a continued checkpoint are journaled.  A separate hermetic
fake-native subtest deliberately forgets two annotations; it proves that the
candidate checkpoint stays private, the state remains at the 900-style anchor,
and redacted rejection evidence is retained.  The fixtures are contract tests,
not semantic-capability benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "tools" / "continuous_pretrain.py"
MANIFEST_TOOL = ROOT / "tools" / "continuous_shard_manifest.py"
STATUS_TOOL = ROOT / "tools" / "continuous_pretrain_status.py"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def run(command: list[str], *, expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if result.returncode != expected:
        raise VerificationError(
            f"command returned {result.returncode}, expected {expected}: {' '.join(command)}\n{result.stdout}"
        )
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quality_inputs(workspace: Path, *, strict_regression: bool) -> tuple[Path, Path, Path, Path]:
    cases = workspace / "cases.json"
    absolute = workspace / "absolute-policy.json"
    regression = workspace / "regression-policy.json"
    key = workspace / "redaction.key"
    write_json(cases, {
        "schema_version": "vegai-pretraining-quality-cases/v1",
        "evaluation_id": "continuous-quality-verifier-v1",
        "declared_split": "held-out",
        "annotation_provenance": {
            "source": "repository continuous-quality verifier fixture",
            "license": "MIT",
            "retrieved_at": "source-controlled verifier fixture",
            "method": "deterministic native/controller contract test/v1",
        },
        "cases": [
            {
                "schema_version": "vegai-pretraining-quality-case/v1",
                "id": "case-alpha",
                "query_token": "alpha",
                "acceptable_neighbours": ["beta", "gamma", "delta", "epsilon", "zeta"],
            },
            {
                "schema_version": "vegai-pretraining-quality-case/v1",
                "id": "case-delta",
                "query_token": "delta",
                "acceptable_neighbours": ["alpha", "beta", "gamma", "epsilon", "zeta"],
            },
        ],
    })
    write_json(absolute, {
        "schema_version": "vegai-pretraining-quality-policy/v1",
        "policy_id": "continuous-quality-verifier-absolute-v1",
        "minimum_cases": 2,
        "minimum_top1_accuracy": 1.0,
        "minimum_top1_wilson_lower_bound_95": 0.0,
        "minimum_mean_similarity": -1.0,
        "minimum_case_similarity": -1.0,
        "maximum_incorrect_cases": 0,
    })
    if strict_regression:
        requirements: dict[str, object] = {
            "maximum_top1_accuracy_drop": 0.0,
            "maximum_top1_wilson_lower_bound_95_drop": 0.0,
            "maximum_mean_bipolar_similarity_drop": 0.0,
            "maximum_minimum_bipolar_similarity_drop": 0.0,
            "maximum_incorrect_top1_cases_increase": 0,
            "maximum_top1_hit_regressions": 0,
            "maximum_case_similarity_drop": 0.0,
            "maximum_case_similarity_regressions": 0,
        }
    else:
        requirements = {
            "maximum_top1_accuracy_drop": 1.0,
            "maximum_top1_wilson_lower_bound_95_drop": 1.0,
            "maximum_mean_bipolar_similarity_drop": 2.0,
            "maximum_minimum_bipolar_similarity_drop": 2.0,
            "maximum_incorrect_top1_cases_increase": 2,
            "maximum_top1_hit_regressions": 2,
            "maximum_case_similarity_drop": 2.0,
            "maximum_case_similarity_regressions": 2,
        }
    write_json(regression, {
        "schema_version": "vegai-continuous-quality-regression-policy/v1",
        "policy_id": "continuous-quality-verifier-regression-v1",
        **requirements,
    })
    key.write_bytes(b"continuous-quality-verifier-secret\n")
    return cases, absolute, regression, key


def manifest(workspace: Path, *, fake: bool) -> Path:
    first = workspace / "first.txt"
    second = workspace / "second.txt"
    if fake:
        first.write_text("alpha beta\ngamma delta\n", encoding="ascii")
        second.write_text("delta gamma\nbeta alpha\n", encoding="ascii")
    else:
        first.write_text(("alpha beta gamma delta epsilon zeta\n") * 8, encoding="ascii")
        second.write_text(("zeta epsilon delta gamma beta alpha\n") * 8, encoding="ascii")
    ledger = workspace / "ledger.json"
    run([
        sys.executable, str(MANIFEST_TOOL), str(first), str(second), "--output", str(ledger),
        "--source", "continuous quality verifier", "--license", "MIT", "--preprocessing", "none",
    ])
    return ledger


def controller_command(
    binary: Path,
    ledger: Path,
    state: Path,
    cases: Path,
    absolute: Path,
    regression: Path,
    key: Path,
) -> list[str]:
    return [
        sys.executable, str(CONTROLLER), "--pretrain-bin", str(binary), "--manifest", str(ledger),
        "--state-dir", str(state), "--max-vocab", "16", "--min-frequency", "1", "--window", "1",
        "--position-shift", "41", "--context-mode", "hybrid", "--epochs", "1",
        "--initial-plasticity", "25", "--minimum-plasticity", "1", "--decay-events", "8",
        "--omp-threads", "1", "--train-timeout-seconds", "60", "--quality-cases", str(cases),
        "--quality-policy", str(absolute), "--quality-redaction-key-file", str(key),
        "--quality-regression-policy", str(regression), "--quality-max-cases", "2",
        "--quality-scratch-dir", str(state.parent), "--quality-timeout-seconds", "30",
    ]


def write_forgetting_fake(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "mode = args[0]\n"
        "if mode in ('train', 'continue'):\n"
        "    corpus = pathlib.Path(args[1]); checkpoint = pathlib.Path(args[2])\n"
        "    epochs = int(args[args.index('--epochs') + 1])\n"
        "    lines = corpus.read_bytes().splitlines(); records = len(lines); tokens = sum(len(line.split()) for line in lines)\n"
        "    checkpoint.write_bytes(b'baseline' if mode == 'train' else b'forgotten')\n"
        "    print('phase: vocabulary' if mode == 'train' else 'phase: vocabulary extension')\n"
        "    print(f'  corpus lines: {records}')\n"
        "    print(f'  observed tokens: {tokens}')\n"
        "    print(f'  active tokens: {tokens}')\n"
        "    print('  tokenizer truncations: 0'); print('  non-ASCII input bytes: 0'); print('  dropped overlong records: 0')\n"
        "    print('phase: streaming HDC context learning')\n"
        "    print(f\"  completed epochs: {epochs if mode == 'train' else 2 * epochs}\")\n"
        "    print(f'  corpus lines: {records * epochs}'); print(f'  raw tokens: {tokens * epochs}'); print(f'  active tokens: {tokens * epochs}')\n"
        "    print('  out-of-vocabulary tokens: 0'); print('  tokenizer truncations: 0'); print('  non-ASCII input bytes: 0'); print('  dropped overlong records: 0')\n"
        "elif mode == 'inspect':\n"
        "    checkpoint = pathlib.Path(args[1]); queries = pathlib.Path(args[args.index('--query-file') + 1])\n"
        "    forgotten = checkpoint.read_bytes() == b'forgotten'\n"
        "    for query in queries.read_text(encoding='ascii').splitlines():\n"
        "        if forgotten: neighbour, similarity = 'alpha', '0.10000'\n"
        "        elif query == 'alpha': neighbour, similarity = 'beta', '0.50000'\n"
        "        else: neighbour, similarity = 'alpha', '0.50000'\n"
        "        print(f'query: {query}'); print(f'  nearest semantic neighbour: {neighbour}'); print(f'  bipolar similarity: {similarity}')\n"
        "else:\n"
        "    raise SystemExit(2)\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def verify_native_success(binary: Path, workspace: Path) -> None:
    ledger = manifest(workspace, fake=False)
    cases, absolute, regression, key = quality_inputs(workspace, strict_regression=False)
    state = workspace / "state"
    run(controller_command(binary, ledger, state, cases, absolute, regression, key))
    journal = json.loads((state / "state.json").read_text(encoding="utf-8"))
    applied = journal.get("applied_shards")
    require(journal.get("generation") == 2 and isinstance(applied, list) and len(applied) == 2,
            f"native quality-gated controller did not commit two generations: {journal}")
    for entry in applied:
        quality = entry.get("quality")
        require(isinstance(quality, dict), f"native quality-gated journal omitted evidence: {entry}")
        comparison = state / str(quality.get("comparison_report"))
        payload = json.loads(comparison.read_text(encoding="utf-8"))
        require(payload.get("status") == "passed", f"native quality comparison did not pass: {payload}")
    status = run([
        sys.executable, str(STATUS_TOOL), "--manifest", str(ledger), "--state-dir", str(state),
        "--require-complete", "--format", "json",
    ])
    status_payload = json.loads(status.stdout)
    require(status_payload.get("quality", {}).get("mode") == "held-out-anchor-regression/v1",
            f"status omitted quality regression state: {status_payload}")
    # State validation hashes the durable reports; this catches a post-commit
    # report replacement before an operator can mistake the state as qualified.
    quality_path = state / str(applied[1]["quality"]["quality_report"])
    original = quality_path.read_bytes()
    quality_path.write_bytes(b"!" + original[1:])
    tampered = run([
        sys.executable, str(STATUS_TOOL), "--manifest", str(ledger), "--state-dir", str(state), "--format", "json",
    ], expected=1)
    require("quality report identity mismatch" in tampered.stdout,
            f"status did not reject a tampered committed quality report\n{tampered.stdout}")


def verify_forgetting_rejection(workspace: Path) -> None:
    fake = workspace / "forgetting-fake.py"
    write_forgetting_fake(fake)
    ledger = manifest(workspace, fake=True)
    cases, absolute, regression, key = quality_inputs(workspace, strict_regression=True)
    state = workspace / "state"
    rejected = run(controller_command(fake, ledger, state, cases, absolute, regression, key), expected=1)
    require("checkpoint was rejected before publication" in rejected.stdout,
            f"controller did not diagnose regression rejection\n{rejected.stdout}")
    journal = json.loads((state / "state.json").read_text(encoding="utf-8"))
    require(journal.get("generation") == 1, f"regressed candidate was committed: {journal}")
    generations = list((state / "generations").glob("generation-*.vgai"))
    require(len(generations) == 1, "rejected candidate left a published generation checkpoint")
    rejection = sorted((state / "quality-rejections").glob("*.rejection.json"))
    comparison = sorted((state / "quality-rejections").glob("*.regression.json"))
    require(len(rejection) == 1 and len(comparison) == 1,
            "rejected candidate did not retain redacted quality regression evidence")
    comparison_payload = json.loads(comparison[0].read_text(encoding="utf-8"))
    require(comparison_payload.get("status") == "failed" and comparison_payload.get("failures"),
            f"rejection comparison did not record failures: {comparison_payload}")

    # Changing the immutable regression policy is rejected before a retry can
    # run a native child. This protects against simply loosening a threshold
    # after observed forgetting.
    changed = json.loads(regression.read_text(encoding="utf-8"))
    changed["maximum_top1_accuracy_drop"] = 1.0
    write_json(regression, changed)
    mismatch = run(controller_command(fake, ledger, state, cases, absolute, regression, key), expected=1)
    require("quality guard differs from the immutable state" in mismatch.stdout,
            f"regression-policy swap was not rejected\n{mismatch.stdout}")

    # Native executable identity is equally immutable. A mutation is rejected
    # before it can interpret the pending shard with another training ABI.
    fake.write_text(fake.read_text(encoding="utf-8") + "\n# mutated\n", encoding="utf-8")
    os.chmod(fake, 0o700)
    unchanged_policy = json.loads((workspace / "regression-policy.json").read_text(encoding="utf-8"))
    unchanged_policy["maximum_top1_accuracy_drop"] = 0.0
    write_json(regression, unchanged_policy)
    binary_mismatch = run(controller_command(fake, ledger, state, cases, absolute, regression, key), expected=1)
    require("pretraining executable differs" in binary_mismatch.stdout,
            f"binary identity mutation was not rejected\n{binary_mismatch.stdout}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path)
    arguments = parser.parse_args()
    binary = arguments.pretrain_bin.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        print(f"continuous quality regression verification: FAIL: unusable binary {binary}", file=sys.stderr)
        return 2
    try:
        with tempfile.TemporaryDirectory(prefix="vegai-continuous-quality-native-") as temporary:
            verify_native_success(binary, Path(temporary))
        with tempfile.TemporaryDirectory(prefix="vegai-continuous-quality-rejection-") as temporary:
            verify_forgetting_rejection(Path(temporary))
    except (VerificationError, OSError, json.JSONDecodeError) as error:
        print(f"continuous quality regression verification: FAIL: {error}", file=sys.stderr)
        return 1
    print("continuous quality regression verification: PASS (native anchor comparison, immutable evidence, forgetting rejection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
