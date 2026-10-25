#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify validation-only score-floor selection with a deterministic fake ABI.

The native evaluator is covered separately. This verifier holds the controller
contract to account: the lowest predeclared candidate that meets a validation
leakage cap is selected, while a test set is neither accepted nor consulted.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CALIBRATOR = ROOT / "tools" / "calibrate_dialogue_abstention_policy.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="vegai-dialogue-calibration-") as temporary_name:
            workspace = Path(temporary_name)
            checkpoint = workspace / "model.vgai"
            knowledge = workspace / "knowledge.tsv"
            train = workspace / "train.tsv"
            validation = workspace / "validation.tsv"
            for path in (checkpoint, knowledge, train, validation):
                path.write_bytes(b"fixture\n")
            fake = workspace / "fake-native-evaluator.py"
            fake.write_text(
                """#!{python}
import hashlib
import json
import pathlib
import sys
args = sys.argv[1:]
evaluation = args[3]
score = float(args[args.index('--min-hybrid-score') + 1])
def identity(path):
    data = pathlib.Path(path).read_bytes()
    return {{'path': pathlib.Path(path).name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}}
if evaluation.endswith('train.tsv'):
    metrics = {{
      'total_cases': 2, 'expected_grounded_cases': 2, 'expected_abstain_cases': 0,
      'actual_grounded_cases': 2, 'rendered_abstain_cases': 0,
      'correct_grounded_prompt_cases': 2, 'wrong_grounded_prompt_cases': 0,
      'false_abstain_cases': 0, 'false_grounded_cases': 0, 'true_abstain_cases': 0,
      'oov_rejected_abstain_cases': 0, 'oov_rejected_expected_grounded_cases': 0,
    }}
else:
    false_grounded = 5 if score < 0.70 else (1 if score < 0.80 else 0)
    metrics = {{
      'total_cases': 100, 'expected_grounded_cases': 0, 'expected_abstain_cases': 100,
      'actual_grounded_cases': false_grounded, 'rendered_abstain_cases': 100 - false_grounded,
      'correct_grounded_prompt_cases': 0, 'wrong_grounded_prompt_cases': 0,
      'false_abstain_cases': 0, 'false_grounded_cases': false_grounded,
      'true_abstain_cases': 100 - false_grounded,
      'oov_rejected_abstain_cases': 0, 'oov_rejected_expected_grounded_cases': 0,
    }}
print(json.dumps({{
  'schema': 'vegai-native-dialogue-evaluation-report/v1',
  'identities': {{'checkpoint': identity(args[1]), 'knowledge': identity(args[2]), 'evaluation': identity(evaluation)}},
  'metrics': metrics,
}}))
""".format(python=sys.executable),
                encoding="utf-8",
                newline="\n",
            )
            fake.chmod(0o755)
            report_dir = workspace / "reports"
            output = workspace / "decision.json"
            command = [
                sys.executable, str(CALIBRATOR),
                "--evaluator-bin", str(fake), "--checkpoint", str(checkpoint), "--knowledge", str(knowledge),
                "--train-evaluation", str(train), "--validation-evaluation", str(validation),
                "--report-directory", str(report_dir), "--output", str(output),
                "--score-candidate", "0.65", "--score-candidate", "0.75", "--score-candidate", "0.85",
                "--hybrid-margin-floor", "0.03", "--maximum-validation-false-grounded-rate", "0.01",
                "--semantic-weight", "0.25", "--lexical-weight", "0.75",
            ]
            completed = subprocess.run(
                command, cwd=ROOT, text=True, encoding="utf-8", errors="strict",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            require(completed.returncode == 0, f"calibrator failed:\n{completed.stdout}")
            decision = json.loads(output.read_text(encoding="utf-8"))
            require(decision.get("schema") == "vegai-dialogue-abstention-calibration/v1",
                    f"wrong calibration schema: {decision}")
            selected = decision.get("selected_policy", {})
            require(selected.get("minimum_hybrid_score") == 0.75,
                    f"calibrator did not choose lowest validation-safe score: {decision}")
            candidates = {candidate["score_floor"]: candidate for candidate in decision.get("candidates", [])}
            require(candidates[0.65]["eligible"] is False and candidates[0.75]["eligible"] is True
                    and candidates[0.85]["eligible"] is True,
                    f"candidate eligibility does not follow validation leakage cap: {candidates}")
            require("test" not in decision.get("inputs", {}) and "no test set input" in decision["contract"]["selection_data"],
                    "calibration decision did not make test isolation explicit")
            require((report_dir / "train-exact-at-strictest-score.json").is_file()
                    and (report_dir / "validation-score-0.650000.json").is_file(),
                    "calibration did not retain artifact-bound native reports")
    except (AssertionError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"dialogue abstention calibration verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("dialogue abstention calibration verification: PASS (validation-only lowest-safe score policy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
