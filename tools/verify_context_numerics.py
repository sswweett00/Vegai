#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline contracts for the context equal-budget comparison helper."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from benchmark_context_numerics import main, run


def main_test() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        old, new, output = root / "old", root / "new", root / "report.json"
        old.write_bytes(b"reference")
        new.write_bytes(b"candidate")
        shape = {"horizon": 8192, "parameters": 23317, "state_bytes": 21120,
                 "forward_ops_per_token": 50560, "training_nll": 2.5}

        def fake_run(binary: Path, seed: int, tokens: int, chunk: int, timeout: int) -> dict[str, object]:
            assert tokens == 512 and chunk in (16, 256) and timeout > 0
            return dict(shape, training_tokens_per_s=200 if binary == new else 100)

        args = ["--reference-bin", str(old), "--candidate-bin", str(new), "--tokens", "512",
                "--seeds", "21,42", "--output", str(output)]
        with patch("benchmark_context_numerics.run", side_effect=fake_run):
            assert main(args) == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert len(report["runs"]) == 4
        assert report["runs"][0]["updates"] == 32
        assert report["runs"][1]["order"] == ["candidate", "reference"]
        assert report["summary_by_chunk"]["16"]["median_time_ratio"] == 0.5

        for extra in (["--tokens", "513"], ["--seeds", "1,1"], ["--output", str(old)]):
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    main(args + extra)
                except SystemExit as exc:
                    assert exc.code == 2
                else:
                    raise AssertionError(f"invalid benchmark options were accepted: {extra}")
        assert old.read_bytes() == b"reference"

        def malformed(stdout: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

        for payload in ("[]", '{"training_tokens_per_s": 0, "training_nll": 1}',
                        '{"training_tokens_per_s": 1, "training_nll": NaN}'):
            with patch("benchmark_context_numerics.subprocess.run", return_value=malformed("CONTEXT_BENCH_JSON: " + payload)):
                try:
                    run(new, 21, 512, 16, 30)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"invalid native result was accepted: {payload}")

        binary = os.environ.get("VEGAI_CONTEXT_BIN")
        if binary:
            for value in ("nan", "inf", "-inf", "1e309", "-1e309", "1e-9999", "0.1,0.2",
                          "0.1/", "1e-", "+", "0." + "0" * 300 + "1"):
                result = subprocess.run([binary, "plan", "--lr", value], capture_output=True, text=True, timeout=30)
                assert result.returncode == 2 and "usage error" in result.stderr, (value, result.returncode, result.stderr)
            for value in (".003", "+0.003", "3e-3", "3D-3", " 0.003 "):
                result = subprocess.run([binary, "plan", "--lr", value], capture_output=True, text=True, timeout=30)
                assert result.returncode == 0, (value, result.returncode, result.stderr)
            for option in ("--seed", "--eval-seed"):
                for value in ("1,2", "1/", "/", "2*1", "+", "1 2",
                              "9223372036854775808", "-9223372036854775809", "1" + " " * 256):
                    result = subprocess.run([binary, "plan", option, value], capture_output=True, text=True, timeout=30)
                    assert result.returncode == 2, (option, value, result.returncode, result.stderr)
                for value in ("  +1 ", "-9223372036854775808", "9223372036854775807"):
                    result = subprocess.run([binary, "plan", option, value], capture_output=True, text=True, timeout=30)
                    assert result.returncode == 0, (option, value, result.returncode, result.stderr)
            checkpoint = root / "do-not-overwrite.vgfb"
            checkpoint.write_bytes(b"valuable checkpoint")
            result = subprocess.run([binary, "train", "--checkpoint", str(checkpoint) + " "],
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 2 and checkpoint.read_bytes() == b"valuable checkpoint", result.stderr
        else:
            print("skip - VEGAI_CONTEXT_BIN unset; native CLI numeric cases not executed")
    print("context numeric equal-budget comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
