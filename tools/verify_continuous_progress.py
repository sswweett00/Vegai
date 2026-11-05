#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify bounded production-controller progress without changing state commits."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import continuous_pretrain as continuous
import production_pretrain as production
from verify_continuous_quality_regression import (
    controller_command, manifest, quality_inputs, run, write_forgetting_fake,
)
from verify_production_pretraining import (
    build_command, output_json, run as run_production, run_command, select_fixture_rows,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def exercise_progress(count: int, quiet: bool, maximum: int | None = None) -> tuple[str, int]:
    shard = SimpleNamespace(identity="a" * 64, accepted_tokens=5)
    shards = [shard] * count
    state = {"generation": 0, "checkpoint_sha256": "b" * 64}
    arguments = SimpleNamespace(quiet_progress=quiet)

    def commit(_directory, _binary, previous, _shard, _arguments):
        return {**previous, "generation": previous["generation"] + 1}

    captured = io.StringIO()
    with patch.object(continuous, "load_manifest", return_value=shards), \
         patch.object(continuous, "validate_manifest_prefix"), \
         patch.object(continuous, "commit_one_shard", side_effect=commit), \
         redirect_stdout(captured):
        result, committed = continuous.process_available(
            Path("unused-state"), Path("unused-manifest"), Path("unused-binary"),
            state, maximum, None, arguments,
        )
    require(result["generation"] == committed, "progress mode changed committed generation accounting")
    return captured.getvalue(), committed


def main() -> int:
    text, committed = exercise_progress(3, quiet=False)
    require(committed == 3 and text.count("committed generation=") == 3,
            "default controller progress contract changed")
    text, committed = exercise_progress(20_000, quiet=True)
    require(committed == 20_000 and text == "", "quiet progress grew with the number of shards")
    text, committed = exercise_progress(5, quiet=True, maximum=2)
    require(committed == 2 and text == "", "quiet progress changed the shard limit")

    args = SimpleNamespace(
        max_vocab=16, min_frequency=1, window=1, position_shift=1, context_mode="hybrid", epochs=1,
        initial_plasticity=25, minimum_plasticity=1, decay_events=8, reserve_fraction=0.25,
        omp_threads=1, max_shards=None, recover_locks=False,
    )
    command = production.continuous_command(Path("binary"), Path("ledger"), "a" * 64, Path("state"), args)
    require("--quiet-progress" in command, "production still captures an unbounded line per shard")

    if os.name == "posix":
        # Exercise the real controller transaction with the existing fake
        # executable; the second candidate must still fail the quality guard.
        with tempfile.TemporaryDirectory(prefix="vegai-quiet-progress-") as temporary:
            workspace = Path(temporary)
            fake = workspace / "fake-inspect"
            write_forgetting_fake(fake)
            ledger = manifest(workspace, fake=True)
            cases, absolute, regression, key = quality_inputs(workspace, strict_regression=True)
            state = workspace / "state"
            result = run(controller_command(fake, ledger, state, cases, absolute, regression, key)
                         + ["--quiet-progress"], expected=1)
            require("committed generation=" not in result.stdout and
                    "checkpoint was rejected before publication" in result.stdout,
                    "quiet controller lost its failure diagnostic or printed per-shard progress")
            require((state / "state.json").is_file(), "quiet controller failed to commit its baseline state")
        with tempfile.TemporaryDirectory(prefix="vegai-production-progress-") as temporary:
            workspace = Path(temporary)
            train, validation = select_fixture_rows()
            source = workspace / "source.txt"
            source.write_text("\n".join([*train, *validation, train[0]]) + "\n", encoding="ascii")
            dataset = workspace / "dataset"
            run_production(build_command(source, dataset))
            fake = workspace / "fake-pretrain"
            write_forgetting_fake(fake)
            result = output_json(run_production(run_command(fake, dataset / "dataset.json", workspace / "run")))
            require(result.get("status") == "success" and
                    result.get("continuous_state", {}).get("committed_this_invocation") == 2 and
                    "committed generation=" not in result.get("child_output_tail", ""),
                    "production did not commit both shards with bounded progress output")
    print("continuous progress verification: PASS (constant-size captured progress, default output, shard limit, guard)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        import sys
        print(f"continuous progress verification: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
