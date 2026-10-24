#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the 9-MiB fixture runner contract against a real compiled Fortran CLI.

CTest uses a deliberately small (256 KiB) instance of the same provenance-bound
fixture schema so ordinary native verification stays bounded. The dedicated
GitHub workflow performs the full exact-9-MiB run; this test proves that the
runner's train/inspect/metric bindings are wired to the actual Fortran binary,
not only to the fake executable used by the stdlib contract verifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from synthetic_pretraining_fixture import FixtureConfig, FixtureError, generate_fixture, sha256_file


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "tools" / "run_9mib_pretraining.py"


class VerificationError(RuntimeError):
    """Raised for a real-binary integration contract failure."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    arguments = parser.parse_args()
    binary = arguments.pretrain_bin.resolve()
    if not binary.is_file():
        print(f"verify 9 MiB runner real binary: FAIL: executable does not exist: {binary}", file=sys.stderr)
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="vegai-9mib-real-binary-") as temporary:
            root = Path(temporary)
            corpus = root / "fixture.txt"
            manifest = root / "fixture.manifest.json"
            fixture = generate_fixture(
                corpus,
                manifest,
                FixtureConfig(
                    target_bytes=256 * 1024,
                    clusters=8,
                    tokens_per_cluster=8,
                    tokens_per_record=8,
                ),
                overwrite=False,
            )
            checkpoint = root / "fixture.vgai"
            report = root / "run.json"
            command = [
                sys.executable,
                str(RUNNER),
                "--pretrain-bin",
                str(binary),
                "--corpus",
                str(corpus),
                "--manifest",
                str(manifest),
                "--checkpoint",
                str(checkpoint),
                "--report",
                str(report),
                "--max-vocab",
                str(fixture["records"]["unique_tokens"]),
                "--min-frequency",
                "1",
                "--window",
                "2",
                "--context-mode",
                "hierarchical",
                "--epochs",
                "1",
                "--omp-threads",
                "1",
                "--timeout-seconds",
                "180",
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            check(
                result.returncode == 0,
                f"real runner returned {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            stdout_report = json.loads(result.stdout)
            report_document = json.loads(report.read_text(encoding="utf-8"))
            check(stdout_report == report_document, "persisted report differs from runner stdout")
            check(report_document["status"] == "succeeded", "real binary did not yield succeeded report")
            check(report_document["resource_preflight"]["status"] == "ready", "normal resource preflight was not ready")
            check(report_document["inputs"]["pretrain_binary"]["sha256"] == sha256_file(binary), "binary hash binding is wrong")
            check(report_document["checkpoint"]["sha256"] == sha256_file(checkpoint), "checkpoint hash binding is wrong")
            check(report_document["fixture"]["verification"]["corpus"]["bytes"] == 256 * 1024, "fixture size binding is wrong")
            check(
                report_document["train"]["metrics"]["raw tokens"] == fixture["records"]["raw_tokens"],
                "real binary raw-token metric does not match manifest",
            )
            check(report_document["train"]["metrics"]["out-of-vocabulary tokens"] == 0, "real binary emitted OOV fixture tokens")
            check(report_document["measurement"]["train_process_wall_seconds"] > 0.0, "real binary elapsed metric is not positive")
    except (OSError, ValueError, json.JSONDecodeError, FixtureError, VerificationError) as error:
        print(f"verify 9 MiB runner real binary: FAIL: {error}", file=sys.stderr)
        return 1
    print("verify 9 MiB runner real binary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
