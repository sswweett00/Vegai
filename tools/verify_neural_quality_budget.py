#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dependency-free contract tests for the equal-budget quality comparison."""
from __future__ import annotations

import contextlib
import copy
import io
import os
import struct
from pathlib import Path
import tempfile

from benchmark_neural_quality_budget import (
    compare_arms, files_are_identical, main as benchmark_main, require_disjoint_records,
    v5_state_identity_excluding_cumulative_nll,
)


def main() -> int:
    reference = {
        "epoch": {"prediction_tokens": 100},
        "training_complete": {"updates_completed": 10},
        "heldout": {"prediction_tokens": 20, "mean_nll": 3.0},
        "controller_wall_seconds": 2.0,
    }
    candidate = copy.deepcopy(reference)
    candidate["heldout"]["mean_nll"] = 2.5
    result = compare_arms(reference, candidate)
    assert result["candidate_minus_reference_nll"] == -0.5
    assert result["candidate_over_reference_time"] == 1.0
    reference["epoch"]["phase_seconds"] = dict(decode=0.5, project=0.4, score=0.3, update=0.2)
    candidate["epoch"]["phase_seconds"] = dict(decode=0.4, project=0.4, score=0.2, update=0.1)
    phases = compare_arms(reference, candidate)["phase_seconds"]
    assert phases["score"] == {"reference_seconds": 0.3, "candidate_seconds": 0.2}
    for value in (float("nan"), -0.1, True, 10**400):
        invalid = copy.deepcopy(candidate)
        invalid["epoch"]["phase_seconds"]["score"] = value
        try:
            compare_arms(reference, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid phase timing was accepted: {value!r}")
    asymmetric = copy.deepcopy(candidate)
    del asymmetric["epoch"]["phase_seconds"]
    try:
        compare_arms(reference, asymmetric)
    except ValueError:
        pass
    else:
        raise AssertionError("a missing phase timing object was accepted")
    for section, field in (("epoch", "prediction_tokens"), ("training_complete", "updates_completed"),
                           ("heldout", "prediction_tokens")):
        unequal = copy.deepcopy(candidate)
        unequal[section][field] += 1
        try:
            compare_arms(reference, unequal)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unequal {section}.{field} was accepted")
    for section, field, value in (("epoch", "prediction_tokens", 100.0),
                                  ("training_complete", "updates_completed", True),
                                  ("heldout", "mean_nll", float("nan")),
                                  ("heldout", "mean_nll", 10**400),
                                  ("heldout", "mean_nll", -0.5),
                                  ("heldout", "mean_nll", True)):
        invalid = copy.deepcopy(candidate)
        invalid[section][field] = value
        try:
            compare_arms(reference, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {section}.{field} was accepted: {value!r}")
    with tempfile.TemporaryDirectory() as directory:
        train, heldout = Path(directory) / "train", Path(directory) / "heldout"
        train.write_bytes(b"first\nsecond\n")
        heldout.write_bytes(b"third\n")
        require_disjoint_records(train, heldout)
        heldout.write_bytes(b"second\n")
        try:
            require_disjoint_records(train, heldout)
        except ValueError:
            pass
        else:
            raise AssertionError("a duplicate train/held-out record was accepted")
        left, right = Path(directory) / "left", Path(directory) / "right"
        left.write_bytes(b"x" * (1024 * 1024 + 3))
        right.write_bytes(left.read_bytes())
        assert files_are_identical(left, right)
        right.write_bytes(right.read_bytes()[:-1] + b"y")
        assert not files_are_identical(left, right)
        # Valid v5 headers may differ only in the NLL bytes. Any changed
        # parameter, optimizer state, header, truncation or format is rejected.
        vocab, width, heads, window, ffn = 1024, 256, 4, 8, 256
        parameters = width * vocab + vocab + 2 * width + 4 * width * width + 3 * width * ffn + heads
        fixture = bytearray(204 + 4 * parameters)
        fixture[:12] = b"VGNR0001\x05\x00\x00\x00"
        struct.pack_into("<5i", fixture, 12, vocab, width, heads, window, ffn)
        left.write_bytes(fixture)
        right.write_bytes(fixture)
        assert v5_state_identity_excluding_cumulative_nll(left, right) == (True, True)
        fixture[192] = 1
        right.write_bytes(fixture)
        assert v5_state_identity_excluding_cumulative_nll(left, right) == (True, False)
        fixture[-1] = 1
        right.write_bytes(fixture)
        assert v5_state_identity_excluding_cumulative_nll(left, right) == (False, False)
        fixture[-1] = 0
        fixture[191] = 1
        right.write_bytes(fixture)
        assert v5_state_identity_excluding_cumulative_nll(left, right) == (False, False)
        fixture[191] = 0
        right.write_bytes(fixture[:-1])
        assert v5_state_identity_excluding_cumulative_nll(left, right) == (False, False)
        fixture[8] = 4
        right.write_bytes(fixture)
        assert v5_state_identity_excluding_cumulative_nll(left, right) == (False, False)
        right.write_bytes(b"VGNR0001")
        assert v5_state_identity_excluding_cumulative_nll(left, right) == (False, False)
        binary = Path(directory) / "native"
        binary.write_bytes(b"do not overwrite")
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                benchmark_main(["--neural-bin", str(binary), "--require-checkpoint-identity"])
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError("checkpoint identity was accepted with unequal learning rates")
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                benchmark_main(["--neural-bin", str(binary), "--require-model-state-identity"])
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError("model identity was accepted with unequal learning rates")
        collisions = [binary]
        hardlink = Path(directory) / "report-link"
        try:
            os.link(binary, hardlink)
        except (OSError, NotImplementedError):
            pass  # Filesystems without hard links still test the direct collision.
        else:
            collisions.append(hardlink)
        for output in collisions:
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    benchmark_main(["--neural-bin", str(binary), "--output", str(output)])
                except SystemExit as exc:
                    assert exc.code == 2
                else:
                    raise AssertionError("an output/executable collision was accepted")
            assert binary.read_bytes() == b"do not overwrite"
    print("equal-budget neural quality comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
