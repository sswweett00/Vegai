#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the real-binary direct-training attestation boundary.

The production and continuous controllers own immutable sharded datasets and
exactly-once journals.  This independent verifier covers the complementary
single-corpus path: a reviewed generic corpus manifest, strict native training,
staged checkpoint/report publication, post-publication verification, and
fail-closed tamper/failed-overwrite behaviour.

It deliberately uses a tiny synthetic ASCII fixture only to test the control
contract.  It makes no semantic-quality or general-capability claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_TOOL = ROOT / "tools" / "corpus_manifest.py"
ATTESTED_TOOL = ROOT / "tools" / "attested_pretrain.py"


class VerificationError(RuntimeError):
    """Raised when a direct-run provenance invariant is not met."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, expected_status: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != expected_status:
        raise VerificationError(
            f"command returned {completed.returncode}, expected {expected_status}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def expect_rejection(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    check(
        completed.returncode != 0,
        f"{description} was accepted unexpectedly\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
    )
    return completed


def make_manifest(corpus: Path, manifest: Path) -> None:
    run(
        [
            sys.executable,
            str(MANIFEST_TOOL),
            str(corpus),
            "--output",
            str(manifest),
            "--source",
            "Vegai direct-attestation native verification fixture",
            "--license",
            "MIT",
            "--retrieved-at",
            "source-controlled verification fixture",
            "--preprocessing",
            "ASCII token rows/v1",
            "--strict-contract",
        ]
    )


def train_command(binary: Path, corpus: Path, manifest: Path, checkpoint: Path, report: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(ATTESTED_TOOL),
        "train",
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
        "16",
        "--min-frequency",
        "1",
        "--window",
        "2",
        "--position-shift",
        "37",
        "--context-mode",
        "hierarchical",
        "--epochs",
        "1",
        "--omp-threads",
        "1",
        "--timeout-seconds",
        "180",
        *extra,
    ]


def verify_command(report: Path, *extra: str) -> list[str]:
    return [sys.executable, str(ATTESTED_TOOL), "verify", "--report", str(report), *extra]


def parse_json(output: str, label: str) -> dict[str, Any]:
    try:
        document = json.loads(output)
    except json.JSONDecodeError as error:
        raise VerificationError(f"{label} did not emit one JSON object: {error}\n{output}") from error
    check(isinstance(document, dict), f"{label} did not emit a JSON object")
    return document


def output_lock_path(resource: Path) -> Path:
    resolved = resource.resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8", errors="surrogateescape")).hexdigest()[:24]
    return resolved.parent / f".vegai-attested-output-{digest}.lock"


def staged_artifacts(workspace: Path) -> list[Path]:
    return sorted(
        path
        for path in workspace.iterdir()
        if path.name.startswith(".checkpoint.vgai.checkpoint.")
        or path.name.startswith(".attestation.json.report.")
        or path.name.startswith(".vegai-attested-output-")
    )


def verify_successful_run(binary: Path, workspace: Path) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    corpus = workspace / "corpus.txt"
    # Every token occurs multiple times, so max-vocab=16/min-frequency=1
    # yields a lossless active-token training pass without relying on a semantic
    # result from this small structural fixture.
    corpus.write_text(
        "alpha beta gamma delta\n"
        "beta gamma delta alpha\n"
        "gamma delta alpha beta\n"
        "delta alpha beta gamma\n",
        encoding="ascii",
        newline="\n",
    )
    manifest = workspace / "corpus.manifest.json"
    make_manifest(corpus, manifest)
    checkpoint = workspace / "checkpoint.vgai"
    report = workspace / "attestation.json"

    completed = run(train_command(binary, corpus, manifest, checkpoint, report))
    stdout_report = parse_json(completed.stdout, "attested train")
    persisted_report = json.loads(report.read_text(encoding="utf-8"))
    check(stdout_report == persisted_report, "attested train stdout differs from atomically published report")
    check(persisted_report.get("schema") == "vegai-attested-pretraining-run/v1", "attestation schema changed")
    check(persisted_report.get("status") == "succeeded", "attested real-binary run did not succeed")
    check(persisted_report["resource_preflight"]["status"] == "ready", "resource preflight was not required/ready")
    check(persisted_report["configuration"]["strict_tokenization"] is True, "native strict ingest was not bound")
    check(persisted_report["train"]["command"][1] == "train", "attestation did not time a native train command")
    check("--strict" in persisted_report["train"]["command"], "native train command omitted strict ingest")
    check(
        persisted_report["train"]["command"][3] != str(checkpoint.resolve()),
        "native train wrote directly to the published checkpoint path",
    )
    check(
        persisted_report["integrity"]["exclusive_output_lock_held_during_run"] is True,
        "report did not bind exclusive checkpoint/report ownership",
    )
    check(
        persisted_report["integrity"]["checkpoint_was_staged_before_publication"] is True,
        "report did not bind staged checkpoint publication",
    )
    check(
        persisted_report["integrity"]["report_presence_means_checkpoint_publish_completed"] is True,
        "report did not bind publication ordering",
    )
    check(persisted_report["checkpoint"]["sha256"] == sha256(checkpoint), "checkpoint SHA binding is wrong")
    check(
        persisted_report["inputs"]["pretrain_binary"]["sha256"] == sha256(binary),
        "native binary SHA binding is wrong",
    )
    expected_tokens = 16
    metrics = persisted_report["train"]["metrics"]
    check(metrics["raw tokens"] == expected_tokens, "native raw-token metric differs from reviewed corpus")
    check(metrics["active tokens"] == expected_tokens, "native active-token metric differs from fixture")
    check(metrics["out-of-vocabulary tokens"] == 0, "lossless fixture unexpectedly produced OOV tokens")
    check(persisted_report["measurement"]["train_process_wall_seconds"] > 0.0, "train-only duration is not positive")
    check(
        persisted_report["measurement"]["scope"] == "one vegai_pretrain train child-process wall-clock interval",
        "measurement scope changed",
    )
    check(not staged_artifacts(workspace), "successful publication left a private stage/lock artifact")

    verified = parse_json(run(verify_command(report)).stdout, "attestation verification")
    check(verified.get("schema") == "vegai-attested-pretraining-verification/v1", "verification schema changed")
    check(verified.get("status") == "verified", "published attestation did not verify")
    check(verified.get("checkpoint_sha256") == sha256(checkpoint), "verification checkpoint identity is wrong")
    return corpus, manifest, checkpoint, report, persisted_report


def verify_tamper_rejection(corpus: Path, manifest: Path, checkpoint: Path, report: Path) -> None:
    corpus_bytes = corpus.read_bytes()
    corpus.write_bytes(corpus_bytes + b"extra\n")
    expect_rejection(verify_command(report), "modified corpus against published attestation")
    corpus.write_bytes(corpus_bytes)

    manifest_bytes = manifest.read_bytes()
    modified_manifest = json.loads(manifest_bytes)
    modified_manifest["declared_provenance"]["source"] = "tampered provenance label"
    manifest.write_text(json.dumps(modified_manifest, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    expect_rejection(verify_command(report), "modified reviewed manifest against published attestation")
    manifest.write_bytes(manifest_bytes)

    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(checkpoint_bytes + b"corruption")
    expect_rejection(verify_command(report), "modified checkpoint against published attestation")
    checkpoint.write_bytes(checkpoint_bytes)

    modified_report = json.loads(report.read_text(encoding="utf-8"))
    modified_report["inspection"]["fields"]["context_window"] = 999
    tampered_report = report.with_name("tampered-attestation.json")
    tampered_report.write_text(json.dumps(modified_report, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    expect_rejection(verify_command(tampered_report), "modified stored inspection result")
    tampered_report.unlink()

    verified = parse_json(run(verify_command(report)).stdout, "restored-attestation verification")
    check(verified.get("status") == "verified", "restored inputs no longer verify")


def verify_failed_overwrite_preserves_published_pair(
    binary: Path, corpus: Path, manifest: Path, checkpoint: Path, report: Path
) -> None:
    checkpoint_before = checkpoint.read_bytes()
    report_before = report.read_bytes()
    # This executes the real native child with a capacity below the reviewed
    # corpus's four distinct tokens. The native vocabulary pass must fail, and
    # the wrapper must leave both already published artifacts untouched despite
    # an explicit overwrite request.
    # A crash leaves a conservative lock. A second writer must refuse it rather
    # than assume the first process is dead; recovery is explicit and still
    # must preserve the previous published pair when the new native run fails.
    stale_lock = output_lock_path(checkpoint)
    stale_lock.write_text("pid=stale-contract-fixture\n", encoding="ascii", newline="\n")
    expect_rejection(verify_command(report), "verification while an output lock is present")
    alternate_report = report.with_name("alternate-attestation.json")
    # Deliberately change only the report path: the shared checkpoint resource
    # must still make this a conflict, rather than locking just an ordered pair.
    locked = expect_rejection(
        train_command(binary, corpus, manifest, checkpoint, alternate_report, "--max-vocab", "2", "--overwrite"),
        "unreviewed stale direct-run lock on a shared checkpoint",
    )
    check("lock" in locked.stderr.lower(), "stale lock was not diagnosed")
    rejected = expect_rejection(
        train_command(
            binary,
            corpus,
            manifest,
            checkpoint,
            report,
            "--max-vocab",
            "2",
            "--overwrite",
            "--recover-lock",
        ),
        "failed overwriting native train after explicit stale-lock recovery",
    )
    check("FAIL" in rejected.stderr, "failed overwrite did not emit a wrapper diagnostic")
    check(checkpoint.read_bytes() == checkpoint_before, "failed overwrite replaced published checkpoint")
    check(report.read_bytes() == report_before, "failed overwrite replaced published report")
    check(not staged_artifacts(checkpoint.parent), "failed overwrite left a private stage/lock artifact")

    # Read-only verification can explicitly recover a stale lock as well; it
    # must not require another training attempt merely to inspect a valid pair.
    stale_report_lock = output_lock_path(report)
    stale_report_lock.write_text("pid=stale-verify-fixture\n", encoding="ascii", newline="\n")
    recovered = parse_json(run(verify_command(report, "--recover-lock")).stdout, "stale-lock recovered verification")
    check(recovered.get("status") == "verified", "explicit verification lock recovery did not verify")
    check(not staged_artifacts(checkpoint.parent), "verification lock recovery left a private artifact")


def write_openvino_provenance_fake(path: Path) -> None:
    """Write a hermetic CLI for the optional-worker provenance branch only.

    The real-binary portion of this verifier proves native train/inspect. This
    small fake exists solely because the CTest environment does not promise an
    OpenVINO runtime/device while the wrapper still needs a deterministic test
    that hashes and re-verifies an optional worker/interpreter identity.
    """
    path.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys

arguments = sys.argv[1:]
if arguments[0] == 'train':
    corpus = pathlib.Path(arguments[1])
    checkpoint = pathlib.Path(arguments[2])
    max_vocabulary = arguments[arguments.index('--max-vocab') + 1]
    min_frequency = arguments[arguments.index('--min-frequency') + 1]
    window = arguments[arguments.index('--window') + 1]
    shift = arguments[arguments.index('--position-shift') + 1]
    mode = arguments[arguments.index('--context-mode') + 1]
    epochs = int(arguments[arguments.index('--epochs') + 1])
    rows = corpus.read_text(encoding='ascii').splitlines()
    tokens = sum(len(row.split()) for row in rows) * epochs
    checkpoint.write_text(f'{max_vocabulary} {min_frequency} {window} {shift} {mode} {epochs}', encoding='ascii')
    print('phase: streaming HDC context learning')
    for label, value in (
        ('completed epochs', epochs),
        ('corpus lines', len(rows) * epochs),
        ('raw tokens', tokens),
        ('active tokens', tokens),
        ('out-of-vocabulary tokens', 0),
        ('ordered context pairs', tokens),
        ('prototype updates', tokens),
        ('tokenizer truncations', 0),
        ('non-ASCII input bytes', 0),
        ('dropped overlong records', 0),
    ):
        print(f'  {label}: {value}')
elif arguments[0] == 'inspect':
    capacity, minimum, window, shift, mode, epochs = pathlib.Path(arguments[1]).read_text(encoding='ascii').split()
    print('Vegai HDC pretraining checkpoint')
    for label, value in (
        ('vocabulary capacity', capacity),
        ('min frequency', minimum),
        ('context window', window),
        ('positional shift', shift),
        ('context mode', mode),
        ('plasticity sampler', 'floyd-complement'),
        ('completed epochs', epochs),
    ):
        print(f'  {label}: {value}')
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
        newline="\n",
    )
    path.chmod(0o755)


def verify_openvino_provenance_contract(workspace: Path) -> None:
    """Cover optional worker/interpreter identity binding without a device claim."""
    branch = workspace / "openvino-provenance"
    branch.mkdir()
    fake_binary = branch / "fake-pretrain.py"
    worker = branch / "worker.py"
    write_openvino_provenance_fake(fake_binary)
    worker.write_text("# deterministic provenance-only worker fixture\n", encoding="ascii", newline="\n")
    corpus = branch / "corpus.txt"
    corpus.write_text("red blue\nblue red\n", encoding="ascii", newline="\n")
    manifest = branch / "corpus.manifest.json"
    make_manifest(corpus, manifest)
    checkpoint = branch / "checkpoint.vgai"
    report = branch / "attestation.json"
    command = train_command(
        fake_binary,
        corpus,
        manifest,
        checkpoint,
        report,
        "--skip-resource-preflight",
        "--context-provider",
        "openvino",
        "--openvino-worker",
        str(worker),
        "--openvino-python",
        sys.executable,
        "--openvino-work-dir",
        str(branch),
        "--openvino-device",
        "CPU",
        "--openvino-min-contexts",
        "1",
        "--openvino-policy",
        "fallback-cpu",
    )
    attestation = parse_json(run(command).stdout, "OpenVINO provenance mock train")
    inputs = attestation["inputs"]
    check(inputs["openvino_worker"]["sha256"] == sha256(worker), "OpenVINO worker identity is not bound")
    interpreter = Path(sys.executable).resolve()
    check(inputs["openvino_python"]["sha256"] == sha256(interpreter), "OpenVINO interpreter identity is not bound")
    check(
        attestation["configuration"]["openvino"]["python"] == str(interpreter),
        "OpenVINO Python command was not resolved to the attested executable",
    )
    check(parse_json(run(verify_command(report)).stdout, "OpenVINO provenance mock verify").get("status") == "verified",
          "OpenVINO provenance mock did not verify")

    worker_bytes = worker.read_bytes()
    worker.write_bytes(worker_bytes + b"# mutation\n")
    expect_rejection(verify_command(report), "modified OpenVINO worker against published attestation")
    worker.write_bytes(worker_bytes)
    check(parse_json(run(verify_command(report)).stdout, "restored OpenVINO provenance verify").get("status") == "verified",
          "restored OpenVINO worker did not verify")
    check(not staged_artifacts(branch), "OpenVINO provenance mock left a private stage/lock artifact")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", required=True, type=Path, help="compiled vegai_pretrain executable")
    arguments = parser.parse_args()
    binary = arguments.pretrain_bin.resolve()
    if not binary.is_file():
        print(f"attested pretraining verification: FAIL: executable does not exist: {binary}", file=sys.stderr)
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="vegai-attested-pretrain-") as temporary:
            workspace = Path(temporary)
            corpus, manifest, checkpoint, report, _ = verify_successful_run(binary, workspace)
            verify_tamper_rejection(corpus, manifest, checkpoint, report)
            verify_failed_overwrite_preserves_published_pair(binary, corpus, manifest, checkpoint, report)
            verify_openvino_provenance_contract(workspace)
    except (OSError, ValueError, json.JSONDecodeError, VerificationError) as error:
        print(f"attested pretraining verification: FAIL: {error}", file=sys.stderr)
        return 1
    print("attested pretraining verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
