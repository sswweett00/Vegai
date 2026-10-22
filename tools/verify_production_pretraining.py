#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""End-to-end verification for the production dataset and run-control boundary.

Unlike a mock orchestration test, this invokes the compiled ``vegai_pretrain``
through the immutable-shard controller.  It proves deterministic packaging,
exact deduplication, held-out leakage rejection, immutable payload rejection,
atomic run provenance, exactly-once state reuse, and a real checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DATASET_TOOL = ROOT / "tools" / "production_dataset.py"
RUN_TOOL = ROOT / "tools" / "production_pretrain.py"
STATUS_TOOL = ROOT / "tools" / "continuous_pretrain_status.py"


def fail(message: str) -> None:
    raise AssertionError(message)


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    if completed.returncode != expected:
        fail(
            f"command returned {completed.returncode}, expected {expected}: {' '.join(command)}\n{completed.stdout}"
        )
    return completed


def output_json(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        fail(f"production run did not emit one JSON result: {error}\n{completed.stdout}")


def build_command(source: Path, output: Path) -> list[str]:
    return [
        sys.executable, str(DATASET_TOOL), "--input", str(source), "--output-dir", str(output),
        "--source", "Vegai deterministic production-control verification fixture",
        "--license", "MIT", "--retrieved-at", "source-controlled verification fixture",
        "--preprocessing", "ASCII-only synthetic token rows/v1",
        "--validation-basis-points", "5000", "--max-shard-tokens", "3", "--max-shard-bytes", "8193",
    ]


def run_command(
    binary: Path, dataset: Path, run_directory: Path, resume: bool = False, max_vocabulary: int = 64
) -> list[str]:
    command = [
        sys.executable, str(RUN_TOOL), "--pretrain-bin", str(binary), "--dataset", str(dataset),
        "--run-dir", str(run_directory), "--max-vocab", str(max_vocabulary), "--min-frequency", "1",
        "--window", "2", "--position-shift", "37", "--context-mode", "hierarchical",
        "--epochs", "1", "--initial-plasticity", "25", "--minimum-plasticity", "1",
        "--decay-events", "8", "--omp-threads", "1",
    ]
    if resume:
        command.append("--resume-run")
    return command


def status_command(ledger: Path, state_directory: Path, ledger_sha256: str) -> list[str]:
    return [
        sys.executable, str(STATUS_TOOL), "--manifest", str(ledger), "--state-dir", str(state_directory),
        "--expect-manifest-sha256", ledger_sha256, "--verify-shards", "--require-complete",
        "--require-unlocked", "--format", "json",
    ]


def select_fixture_rows() -> tuple[list[str], list[str]]:
    train: list[str] = []
    validation: list[str] = []
    for index in range(10_000):
        row = f"topic{index} context{index} relation{index}"
        digest = hashlib.sha256(row.encode("ascii")).digest()
        if int.from_bytes(digest[:8], "big") % 10_000 < 5_000:
            if len(validation) < 2:
                validation.append(row)
        elif len(train) < 2:
            train.append(row)
        if len(train) == 2 and len(validation) == 2:
            return train, validation
    fail("could not construct deterministic two-split fixture")


def files_under(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    arguments = parser.parse_args()
    binary = arguments.pretrain_bin.resolve()
    if not binary.is_file():
        print(f"production pretraining verification: FAIL: executable does not exist: {binary}", file=sys.stderr)
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="vegai-production-pretrain-") as temporary:
            workspace = Path(temporary)
            train, validation = select_fixture_rows()
            source = workspace / "source.txt"
            # The repeated first train record must be removed exactly once, and
            # no record identity may cross the hash-defined held-out split.
            source.write_text("\n".join([train[0], validation[0], train[1], validation[1], train[0]]) + "\n", encoding="ascii")
            dataset_a = workspace / "dataset-a"
            dataset_b = workspace / "dataset-b"
            run(build_command(source, dataset_a))
            run(build_command(source, dataset_b))
            if files_under(dataset_a) != files_under(dataset_b):
                fail("identical source and contract did not produce byte-identical immutable dataset outputs")

            dataset = json.loads((dataset_a / "dataset.json").read_text(encoding="utf-8"))
            if dataset.get("deduplication", {}).get("duplicates_removed") != 1:
                fail(f"exact deduplication accounting is wrong: {dataset}")
            if dataset.get("train", {}).get("records") != 2 or dataset.get("validation", {}).get("records") != 2:
                fail(f"deterministic train/validation assignment is wrong: {dataset}")
            if dataset.get("train", {}).get("unique_tokens_min_frequency_1") != 6:
                fail(f"exact min-frequency-1 training vocabulary preflight is wrong: {dataset}")
            if dataset.get("split", {}).get("train_validation_record_identity_disjoint") is not True:
                fail(f"dataset omitted held-out isolation contract: {dataset}")

            capacity_rejected = run(
                run_command(binary, dataset_a / "dataset.json", workspace / "capacity-run", max_vocabulary=5)
                + ["--preflight-only"], expected=1
            )
            if "max-vocab is below" not in capacity_rejected.stdout:
                fail(f"unsafe vocabulary capacity was not rejected before training\n{capacity_rejected.stdout}")

            run_directory = workspace / "production-run"
            first = output_json(run(run_command(binary, dataset_a / "dataset.json", run_directory)))
            if first.get("status") != "success":
                fail(f"production run did not report success: {first}")
            if first.get("continuous_state", {}).get("committed_this_invocation") != 2:
                fail(f"production result did not report both first-run shard commits: {first}")
            leakage = first.get("leakage_audit")
            if not isinstance(leakage, dict) or leakage.get("mode") != "exact-sha256-record-sqlite/v1" or leakage.get("overlap_records") != 0:
                fail(f"exact held-out leakage audit was not recorded: {first}")
            state = json.loads((run_directory / "state" / "state.json").read_text(encoding="utf-8"))
            if state.get("generation") != 2 or len(state.get("applied_shards", [])) != 2:
                fail(f"real continuous controller did not commit each training shard once: {state}")
            checkpoint = run_directory / "state" / str(state["checkpoint"])
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                fail("production controller did not publish a nonempty final checkpoint")
            if not (run_directory / "run.json").is_file() or not (run_directory / "result.json").is_file():
                fail("production controller omitted atomic provenance/result manifests")

            train_metadata = dataset.get("train")
            if not isinstance(train_metadata, dict) or not isinstance(train_metadata.get("ledger_sha256"), str):
                fail(f"production dataset omitted the train-ledger release digest: {dataset}")
            health = output_json(
                run(
                    status_command(
                        dataset_a / "train" / "ledger.json",
                        run_directory / "state",
                        train_metadata["ledger_sha256"],
                    )
                )
            )
            health_progress = health.get("progress")
            health_verification = health.get("verification")
            health_lock = health.get("controller_lock")
            if (
                health.get("schema") != "vegai-continuous-pretraining-status/v1"
                or health.get("state_initialized") is not True
                or not isinstance(health_progress, dict)
                or health_progress.get("generation") != 2
                or health_progress.get("pending_shards") != 0
                or health_progress.get("complete") is not True
                or not isinstance(health_verification, dict)
                or health_verification.get("state_journal_and_all_committed_checkpoints") is not True
                or health_verification.get("manifest_prefix") is not True
                or health_verification.get("shard_payloads") != "verified"
                or health_verification.get("verified_shards") != 2
                or not isinstance(health_lock, dict)
                or health_lock.get("present") is not False
            ):
                fail(f"production state health report is inconsistent: {health}")

            state_before_resume = (run_directory / "state" / "state.json").read_bytes()
            second = output_json(run(run_command(binary, dataset_a / "dataset.json", run_directory, resume=True)))
            if (
                second.get("continuous_state", {}).get("committed_shards") != 2
                or second.get("continuous_state", {}).get("committed_this_invocation") != 0
            ):
                fail(f"resume did not retain exactly-once committed history: {second}")
            if (run_directory / "state" / "state.json").read_bytes() != state_before_resume:
                fail("idempotent production resume changed durable continuous state")

            # A mutable source payload is caught by the continuous-ledger SHA
            # verification before the real training executable can consume it.
            tampered = workspace / "tampered"
            shutil.copytree(dataset_a, tampered)
            train_shard = next((tampered / "train" / "shards").glob("*.txt"))
            train_shard.write_bytes(b"rewritten payload\n")
            rejected_payload = run(
                run_command(binary, tampered / "dataset.json", workspace / "tampered-run") + ["--preflight-only"], expected=1
            )
            if "wrong digest" not in rejected_payload.stdout:
                fail(f"tampered immutable payload was not diagnosed by SHA verification\n{rejected_payload.stdout}")

            # Construct a syntactically valid package that retains the split
            # directory layout and every relevant SHA/count commitment, but
            # intentionally places one train record into validation.  This
            # proves the record-level audit rather than merely a path check.
            leaky = workspace / "leaky"
            shutil.copytree(dataset_a, leaky)
            leaky_document = json.loads((leaky / "dataset.json").read_text(encoding="utf-8"))
            train_ledger = json.loads((leaky / "train" / "ledger.json").read_text(encoding="utf-8"))
            validation_ledger_path = leaky / "validation" / "ledger.json"
            validation_ledger = json.loads(validation_ledger_path.read_text(encoding="utf-8"))
            train_entry = dict(train_ledger["shards"][0])
            train_payload = leaky / "train" / train_entry["path"]
            validation_payload = leaky / "validation" / "shards" / f"{train_entry['sha256']}.txt"
            original_validation_payload = leaky / "validation" / validation_ledger["shards"][0]["path"]
            original_validation_payload.unlink()
            shutil.copyfile(train_payload, validation_payload)
            train_entry["path"] = str(Path("shards") / validation_payload.name)
            validation_ledger["shards"] = [train_entry]
            validation_ledger_path.write_text(json.dumps(validation_ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            record = validation_payload.read_bytes().rstrip(b"\n")
            record_digest_stream = hashlib.sha256(hashlib.sha256(record).digest()).hexdigest()
            leaky_document["validation"].update(
                {
                    "ledger": "validation/ledger.json",
                    "ledger_sha256": hashlib.sha256(validation_ledger_path.read_bytes()).hexdigest(),
                    "records": 1,
                    "accepted_tokens": train_entry["preflight"]["accepted_tokens"],
                    "shards": 1,
                    "record_digest_stream_sha256": record_digest_stream,
                }
            )
            # Make the adversarial metadata internally consistent too; the
            # source file itself is not shipped in an immutable package, so
            # only the exact raw-record split audit can catch this condition.
            leaky_document["deduplication"]["retained_records"] = 3
            leaky_document["deduplication"]["duplicates_removed"] = 2
            (leaky / "dataset.json").write_text(json.dumps(leaky_document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rejected_leakage = run(
                run_command(binary, leaky / "dataset.json", workspace / "leaky-run") + ["--preflight-only"], expected=1
            )
            if "leakage audit" not in rejected_leakage.stdout:
                fail(f"train/validation overlap was not rejected by exact audit\n{rejected_leakage.stdout}")

            invalid = workspace / "invalid-utf8.txt"
            invalid.write_bytes(b"alpha \xc3\xa7\n")
            invalid_output = workspace / "invalid-dataset"
            rejected_unicode = run(build_command(invalid, invalid_output), expected=1)
            if "non-ASCII" not in rejected_unicode.stdout or invalid_output.exists():
                fail(f"non-ASCII corpus was not fail-closed/atomically unpublished\n{rejected_unicode.stdout}")

            invalid_cr = workspace / "invalid-cr.txt"
            invalid_cr.write_bytes(b"alpha\rbeta\n")
            invalid_cr_output = workspace / "invalid-cr-dataset"
            rejected_cr = run(build_command(invalid_cr, invalid_cr_output), expected=1)
            if "CR byte" not in rejected_cr.stdout or invalid_cr_output.exists():
                fail(f"non-LF corpus was not fail-closed/atomically unpublished\n{rejected_cr.stdout}")
    except (AssertionError, OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        print(f"production pretraining verification: FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "production pretraining verification: PASS "
        "(deterministic foundry, exact dedupe/leakage gate, real exactly-once HDC run, read-only state health audit, provenance, idempotent resume)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
