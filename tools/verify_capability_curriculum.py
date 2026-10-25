#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline integration test for source-driven capability curriculum packaging.

All text and annotations here are temporary protocol fixtures. They prove no
reasoning, coding, factuality, or model-quality result; real curriculum data,
licenses, and held-out targets remain operator supplied.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from capability_curriculum import CapabilityCurriculumError, parse_policy
from continuous_pretrain import load_manifest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "capability_curriculum.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != expected:
        raise AssertionError(
            f"command returned {completed.returncode}, expected {expected}: {' '.join(command)}\n{completed.stdout}"
        )
    return completed


def policy(*, heldout_minimum: int = 1) -> dict[str, object]:
    return {
        "schema_version": "vegai-capability-curriculum-policy/v1",
        "id": "synthetic-capability-policy",
        "provenance": {
            "source": "repository-local protocol fixture",
            "license": "MIT",
            "retrieved_at": "source-controlled fixture",
            "preprocessing": "none",
        },
        "annotations": {
            "capabilities_path": ["curriculum", "skills"],
            "difficulty_path": ["curriculum", "difficulty"],
            "require_record_source": True,
        },
        "splits": {"validation_basis_points": 0, "test_basis_points": 0},
        "assignment": {
            "mode": "first-match",
            "reject_multiple_matches": True,
            "require_every_annotated_record": True,
        },
        "stages": [
            {
                "id": "reasoning-stage",
                "required_capabilities": ["reasoning"],
                "minimum_difficulty": 0.0,
                "maximum_difficulty": 0.49,
                "minimum_train_records": 1,
                "minimum_heldout_records": heldout_minimum,
                "minimum_distinct_train_sources": 1,
                "minimum_distinct_heldout_sources": 1,
            },
            {
                "id": "coding-stage",
                "required_capabilities": ["coding"],
                "minimum_difficulty": 0.5,
                "maximum_difficulty": 1.0,
                "minimum_train_records": 1,
                "minimum_heldout_records": heldout_minimum,
                "minimum_distinct_train_sources": 1,
                "minimum_distinct_heldout_sources": 1,
            },
        ],
    }


def record(record_id: str, text: str, split: str, source: str, skill: str, difficulty: float) -> dict[str, object]:
    return {
        "schema_version": "vegai-training-record/v1",
        "id": record_id,
        "text": text,
        "split": split,
        "source": source,
        "metadata": {"curriculum": {"skills": [skill], "difficulty": difficulty}},
    }


def build_command(input_path: Path, policy_path: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(TOOL),
        "build",
        str(input_path),
        "--policy",
        str(policy_path),
        "--output-dir",
        str(output_dir),
    ]


def verify_command(package: Path, source: Path | None = None, policy_path: Path | None = None) -> list[str]:
    command = [sys.executable, str(TOOL), "verify", "--package", str(package)]
    if source is not None:
        command.extend(("--source", str(source)))
    if policy_path is not None:
        command.extend(("--policy", str(policy_path)))
    return command


def verify_optional_columnar_adapters(workspace: Path, policy_path: Path, records: list[dict[str, object]]) -> bool:
    """Exercise compiler delegation to optional Parquet/Arrow readers when present."""
    try:
        import pyarrow  # noqa: F401
        from training_data_io import RecordWriter
    except ImportError:
        return False
    for format_name, suffix in (("parquet", ".parquet"), ("arrow", ".arrow")):
        source = workspace / f"records{suffix}"
        with RecordWriter(source, format=format_name, compression="none") as writer:
            writer.write_many(records)
        package = workspace / f"{format_name}-package"
        report = json.loads(run(build_command(source, policy_path, package)).stdout)
        require(report["input"]["format"] == format_name,
                f"compiler did not retain the selected {format_name} input format")
        verified = json.loads(run(verify_command(package, source, policy_path)).stdout)
        require(verified["source_policy_replayed"] and verified["replayed_files"] == 4,
                f"compiler did not deterministically replay {format_name} input")
    return True


def main() -> int:
    columnar_checked = False
    try:
        template, _ = parse_policy(ROOT / "examples" / "capability_curriculum" / "policy.template.json")
        require(template.policy_id == "replace-with-approved-policy-id" and len(template.stages) == 1,
                "repository curriculum policy template is not schema-valid")
        with tempfile.TemporaryDirectory(prefix="vegai-capability-curriculum-") as temporary:
            workspace = Path(temporary)
            input_path = workspace / "records.jsonl"
            policy_path = workspace / "policy.json"
            records = [
                record("reasoning-train", "premise invariant conclusion trace", "train", "source-reasoning-train", "reasoning", 0.20),
                record("reasoning-validation", "counterexample premise proof trace", "validation", "source-reasoning-heldout", "reasoning", 0.20),
                record("coding-train", "function parse input boundary test", "train", "source-coding-train", "coding", 0.80),
                record("coding-test", "function compile error output test", "test", "source-coding-heldout", "coding", 0.80),
                # Exact duplicate: acceptable only because its split and
                # capability/difficulty declaration match the first record.
                record("reasoning-duplicate", "premise invariant conclusion trace", "train", "source-reasoning-train", "reasoning", 0.20),
            ]
            input_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
                encoding="utf-8",
                newline="\n",
            )
            write_json(policy_path, policy())
            package = workspace / "package"
            built = run(build_command(input_path, policy_path, package))
            report = json.loads(built.stdout)
            require(report["schema_version"] == "vegai-capability-curriculum/v1", "compiler emitted an unexpected package schema")
            require(report["records"]["duplicates_removed"] == 1 and report["records"]["unique"] == 4,
                    "compiler did not perform exact cross-stage deduplication")
            require(report["capacity_preflight"]["minimum_continual_max_vocabulary"] > 0,
                    "compiler did not calculate a strict selected-training vocabulary capacity")
            require(report["heldout"]["raw_text_persisted"] is False,
                    "compiler claimed to persist raw held-out text")
            require(report["input"]["path"] == input_path.name and report["policy"]["path"] == policy_path.name
                    and report["ledger"]["path"] == "ledger.json",
                    "portable package report exposed a temporary or absolute local pathname")
            collision = run(build_command(input_path, policy_path, package), expected=2)
            require("already exists" in collision.stdout,
                    "compiler permitted overwrite of an immutable published package directory")
            if os.name != "nt":
                dangling_output = workspace / "dangling-package"
                os.symlink("unpublished-target", dangling_output)
                dangling = run(build_command(input_path, policy_path, dangling_output), expected=2)
                require("already exists" in dangling.stdout and dangling_output.is_symlink(),
                        "compiler followed a dangling output symlink instead of rejecting the collision")
                dangling_output.unlink()

            reasoning = package / "shards" / "001-reasoning-stage.txt"
            coding = package / "shards" / "002-coding-stage.txt"
            require(reasoning.read_text(encoding="ascii") == "premise invariant conclusion trace\n",
                    "reasoning stage contains data other than selected source text")
            require(coding.read_text(encoding="ascii") == "function parse input boundary test\n",
                    "coding stage contains data other than selected source text")
            package_text = (package / "curriculum.json").read_text(encoding="utf-8")
            require("counterexample premise proof trace" not in package_text and "metadata" not in reasoning.read_text(encoding="ascii"),
                    "curriculum report/corpus retained held-out text or metadata injection")

            ledger_path = package / "ledger.json"
            ledger_digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
            loaded = load_manifest(ledger_path, ledger_digest)
            require(len(loaded) == 2 and [shard.path.name for shard in loaded] == [reasoning.name, coding.name],
                    "curriculum ledger is not compatible with the continuous pretraining controller")
            verified = json.loads(run(verify_command(package, input_path, policy_path)).stdout)
            require(
                verified["verified_stages"] == 2
                and verified["source_rechecked"]
                and verified["policy_rechecked"]
                and verified["source_policy_replayed"]
                and verified["replayed_files"] == 4,
                "curriculum verification lost identity, stage coverage, or full source/policy replay",
            )
            relocated_source = workspace / "relocated-source.jsonl"
            relocated_policy = workspace / "relocated-policy.json"
            relocated_source.write_bytes(input_path.read_bytes())
            relocated_policy.write_bytes(policy_path.read_bytes())
            relocated = json.loads(run(verify_command(package, relocated_source, relocated_policy)).stdout)
            require(relocated["source_policy_replayed"] and relocated["replayed_files"] == 4,
                    "full replay depended on an operator-local input or policy pathname")
            columnar_checked = verify_optional_columnar_adapters(workspace, policy_path, records)

            # The package records the resolved reader codec rather than an
            # extension-dependent `auto` choice, so a governed gzip source can
            # be relocated without losing deterministic replay.
            compressed_input = workspace / "records.jsonl.gz"
            compressed_input.write_bytes(gzip.compress(input_path.read_bytes(), mtime=0))
            compressed_package = workspace / "compressed-package"
            compressed_report = json.loads(run(build_command(compressed_input, policy_path, compressed_package)).stdout)
            require(compressed_report["input"]["compression"] == "gzip",
                    "compiler did not bind the resolved gzip input codec")
            relocated_compressed = workspace / "relocated-compressed.payload"
            relocated_compressed.write_bytes(compressed_input.read_bytes())
            compressed_verified = json.loads(run(verify_command(compressed_package, relocated_compressed, policy_path)).stdout)
            require(compressed_verified["source_policy_replayed"] and compressed_verified["replayed_files"] == 4,
                    "full replay failed after relocating a byte-identical gzip source")

            # A package JSON field has no self-signature.  Supplying both
            # original inputs must therefore replay and reject altered
            # coverage/digest reporting, not merely hash the source files.
            curriculum_path = package / "curriculum.json"
            original_curriculum = curriculum_path.read_text(encoding="utf-8")
            changed_curriculum = json.loads(original_curriculum)
            changed_curriculum["heldout"]["record_digest_stream_sha256"] = "0" * 64
            write_json(curriculum_path, changed_curriculum)
            report_tamper = run(verify_command(package, input_path, policy_path), expected=2)
            require("source/policy replay differs" in report_tamper.stdout,
                    "source/policy replay accepted a self-reported held-out digest rewrite")
            curriculum_path.write_text(original_curriculum, encoding="utf-8", newline="\n")

            insufficient_policy = workspace / "insufficient-policy.json"
            write_json(insufficient_policy, policy(heldout_minimum=2))
            insufficient_output = workspace / "insufficient-package"
            insufficient = run(build_command(input_path, insufficient_policy, insufficient_output), expected=2)
            require("below policy minimum" in insufficient.stdout and not insufficient_output.exists(),
                    "compiler published a curriculum despite inadequate held-out coverage")

            # Validation and test records from the same declared source count
            # once for a stage's held-out source-diversity gate.
            diversity_policy = policy()
            diversity_policy["stages"] = [dict(diversity_policy["stages"][0])]
            diversity_policy["stages"][0]["minimum_distinct_heldout_sources"] = 2
            diversity_policy_path = workspace / "diversity-policy.json"
            write_json(diversity_policy_path, diversity_policy)
            diversity_input = workspace / "diversity.jsonl"
            diversity_records = [
                record("diversity-train", "premise branch invariant trace", "train", "diversity-train-source", "reasoning", 0.20),
                record("diversity-validation", "premise branch counterexample trace", "validation", "shared-heldout-source", "reasoning", 0.20),
                record("diversity-test", "premise branch conclusion trace", "test", "shared-heldout-source", "reasoning", 0.20),
            ]
            diversity_input.write_text(
                "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in diversity_records),
                encoding="utf-8",
                newline="\n",
            )
            diversity_output = workspace / "diversity-package"
            diversity = run(build_command(diversity_input, diversity_policy_path, diversity_output), expected=2)
            require("held-out sources" in diversity.stdout and not diversity_output.exists(),
                    "compiler double-counted one source across validation and test coverage")

            # TXT remains a supported generic rich-data adapter input, but it
            # cannot be silently accepted as a curriculum because it carries
            # neither metadata annotations nor record provenance.
            text_only_input = workspace / "text-only.txt"
            text_only_input.write_text("plain text without governed annotations\n", encoding="ascii", newline="\n")
            text_only_output = workspace / "text-only-package"
            text_only = run(build_command(text_only_input, policy_path, text_only_output), expected=2)
            require("metadata annotations" in text_only.stdout and not text_only_output.exists(),
                    "compiler silently accepted text-only input without curriculum metadata")

            conflicting_input = workspace / "conflicting.jsonl"
            conflicting_records = list(records)
            conflicting_records.append(
                record("conflicting-duplicate", "premise invariant conclusion trace", "train", "source-conflict", "reasoning", 0.30)
            )
            conflicting_input.write_text(
                "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in conflicting_records),
                encoding="utf-8",
                newline="\n",
            )
            conflicting_output = workspace / "conflicting-package"
            conflict = run(build_command(conflicting_input, policy_path, conflicting_output), expected=2)
            require("conflicting capability/difficulty annotations" in conflict.stdout and not conflicting_output.exists(),
                    "compiler accepted an exact duplicate with conflicting supervision metadata")

            split_conflicting_input = workspace / "split-conflicting.jsonl"
            split_conflicting_records = list(records)
            split_conflicting_records.append(
                record("split-conflicting-duplicate", "premise invariant conclusion trace", "test", "source-conflict", "reasoning", 0.20)
            )
            split_conflicting_input.write_text(
                "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in split_conflicting_records),
                encoding="utf-8",
                newline="\n",
            )
            split_conflicting_output = workspace / "split-conflicting-package"
            split_conflict = run(build_command(split_conflicting_input, policy_path, split_conflicting_output), expected=2)
            require("conflicting train/validation/test split" in split_conflict.stdout and not split_conflicting_output.exists(),
                    "compiler accepted an exact duplicate across train/held-out split boundaries")

            # Packages are closed layouts: an unbound file cannot ride beside
            # otherwise valid data and evade a release/provenance audit.
            unexpected = package / "unbound-payload.txt"
            unexpected.write_text("unexpected\n", encoding="ascii", newline="\n")
            layout_tamper = run(verify_command(package), expected=2)
            require("missing, unbound, or unexpected files" in layout_tamper.stdout,
                    "verification accepted an unbound package payload")
            unexpected.unlink()

            # Verify must detect any immutable stage payload replacement rather
            # than trusting package JSON counters alone.
            coding.write_text("rewritten payload\n", encoding="ascii", newline="\n")
            tampered = run(verify_command(package), expected=2)
            require("mismatch" in tampered.stdout and "curriculum shard" in tampered.stdout,
                    "read-only curriculum verification accepted a rewritten stage payload")
    except (AssertionError, CapabilityCurriculumError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"capability curriculum verification: FAIL: {error}", file=sys.stderr)
        return 1

    columnar_status = "checked" if columnar_checked else "skipped (optional PyArrow unavailable)"
    print(
        "capability curriculum verification: PASS "
        "(source-driven stages, held-out coverage, exact dedupe, text-only projection, controller ledger, "
        f"provenance/replay/tamper rejection, optional Parquet/Arrow={columnar_status})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
