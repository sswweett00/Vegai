#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Small-byte contract test for the exact-900MB continuous-run orchestrator.

The production runner deliberately has fixed decimal 900,000,000 and 90,000,000
byte constants. This verifier temporarily substitutes tiny byte budgets inside
one Python process, while retaining the identical fixture/ledger/controller/
quality-report code path and a hermetic native-CLI ABI fixture. The separate
manual GitHub workflow owns the real 900 MB + 90 MB execution.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile

import run_900mb_continuous_pretraining as runner


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def write_fake_native(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, re, sys\n"
        "args = sys.argv[1:]\n"
        "mode = args[0]\n"
        "if mode in ('train', 'continue'):\n"
        "    corpus = pathlib.Path(args[1]); checkpoint = pathlib.Path(args[2])\n"
        "    epochs = int(args[args.index('--epochs') + 1])\n"
        "    lines = corpus.read_bytes().splitlines(); records = len(lines); tokens = sum(len(line.split()) for line in lines)\n"
        "    checkpoint.write_bytes(b'baseline' if mode == 'train' else checkpoint.read_bytes() + b'continued')\n"
        "    print('phase: vocabulary' if mode == 'train' else 'phase: vocabulary extension')\n"
        "    print(f'  corpus lines: {records}'); print(f'  observed tokens: {tokens}'); print(f'  active tokens: {tokens}')\n"
        "    print('  tokenizer truncations: 0'); print('  non-ASCII input bytes: 0'); print('  dropped overlong records: 0')\n"
        "    print('phase: streaming HDC context learning')\n"
        "    print(f\"  completed epochs: {epochs if mode == 'train' else 2 * epochs}\")\n"
        "    print(f'  corpus lines: {records * epochs}'); print(f'  raw tokens: {tokens * epochs}'); print(f'  active tokens: {tokens * epochs}')\n"
        "    print('  out-of-vocabulary tokens: 0'); print('  tokenizer truncations: 0'); print('  non-ASCII input bytes: 0'); print('  dropped overlong records: 0')\n"
        "elif mode == 'inspect':\n"
        "    query_file = pathlib.Path(args[args.index('--query-file') + 1])\n"
        "    for query in query_file.read_text(encoding='ascii').splitlines():\n"
        "        match = re.fullmatch(r'(.*_token_)([0-9]+)', query)\n"
        "        neighbour = match.group(1) + f'{(int(match.group(2)) + 1) % 16:03d}'\n"
        "        print(f'query: {query}'); print(f'  nearest semantic neighbour: {neighbour}'); print('  bipolar similarity: 0.50000')\n"
        "else:\n"
        "    raise SystemExit(2)\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def main() -> int:
    original = (runner.INITIAL_BYTES, runner.CONTINUATION_BYTES, runner.TOKENS_PER_RECORD)
    original_argv = sys.argv[:]
    try:
        with tempfile.TemporaryDirectory(prefix="vegai-900mb-continuation-contract-") as temporary:
            workspace = Path(temporary)
            binary = workspace / "fake-native.py"
            run_dir = workspace / "run"
            write_fake_native(binary)
            # Exercise the full implementation with sizes that retain the
            # exact-byte generator and two immutable shard phases but are
            # appropriate for regular CTest.
            runner.INITIAL_BYTES = 512 * 1024
            runner.CONTINUATION_BYTES = 128 * 1024
            runner.TOKENS_PER_RECORD = 16
            sys.argv = [
                "run_900mb_continuous_pretraining.py", "--pretrain-bin", str(binary), "--run-dir", str(run_dir),
                "--omp-threads", "1", "--train-timeout-seconds", "60", "--quality-timeout-seconds", "20",
            ]
            # The production runner prints its complete JSON evidence for CI;
            # retain only this verifier's concise PASS/FAIL output in CTest.
            with contextlib.redirect_stdout(io.StringIO()):
                status = runner.main()
            require(status == 0, f"small-byte controlled runner returned {status}")
            report = json.loads((run_dir / "experiment-report.json").read_text(encoding="utf-8"))
            require(report.get("status") == "succeeded", f"runner report is not successful: {report}")
            fixtures = report.get("fixtures", {})
            require(fixtures.get("initial_900MB", {}).get("bytes") == 512 * 1024,
                    "runner did not preserve exact initial fixture byte count")
            require(fixtures.get("continuous_90MB", {}).get("bytes") == 128 * 1024,
                    "runner did not preserve exact continuation fixture byte count")
            require(report.get("continuous_state", {}).get("generation") == 2,
                    "runner did not commit initial plus continuation generations")
            regression = report.get("quality_regression", {})
            require(regression.get("status") == "passed", "runner did not retain passing regression evidence")
            for phase in ("initial_pretraining", "continuous_pretraining"):
                measurement = report.get(phase, {}).get("training_measurement", {})
                require(
                    measurement.get("scope") == "one vegai_pretrain train-or-continue child-process wall-clock interval"
                    and measurement.get("train_process_wall_seconds", 0) > 0
                    and measurement.get("raw_training_tokens_per_second", 0) > 0,
                    f"{phase} did not retain train-only timing: {measurement}",
                )
            require(not (run_dir / ".quality-redaction.key").exists(),
                    "runner retained HMAC redaction key in uploadable runtime directory")
    except (VerificationError, OSError, json.JSONDecodeError) as error:
        print(f"900MB continuous pretraining contract: FAIL: {error}", file=sys.stderr)
        return 1
    finally:
        runner.INITIAL_BYTES, runner.CONTINUATION_BYTES, runner.TOKENS_PER_RECORD = original
        sys.argv = original_argv
    print("900MB continuous pretraining contract: PASS (exact-size topology, native controller sequence, regression evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
