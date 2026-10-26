#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent stdlib-only contract checks for the exact-9-MiB run path.

This verifier tests deterministic exact-size generation, manifest/tamper
rejection, bounded runner report lineage, output separation, and a deliberately
malformed train protocol using a temporary executable. It does not claim to
train a semantic model: the real compiled-binary 9-MiB run is owned by the
separate GitHub Actions workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from run_9mib_pretraining import build_parser, format_percent_tenths, parse_percent_tenths
from synthetic_pretraining_fixture import (
    DEFAULT_TARGET_BYTES,
    FixtureConfig,
    FixtureError,
    generate_fixture,
    sha256_file,
    verify_fixture,
)


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
FIXTURE_TOOL = ROOT / "tools" / "synthetic_pretraining_fixture.py"
RUNNER_TOOL = ROOT / "tools" / "run_9mib_pretraining.py"


class VerificationError(RuntimeError):
    """Raised for an unmet independent contract assertion."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def verify_parameter_configuration() -> None:
    check(parse_percent_tenths("11") == 110, "integer plasticity percent must map to tenths exactly")
    check(parse_percent_tenths("11.0") == 110, "decimal whole percentage must parse exactly")
    check(parse_percent_tenths("0.6") == 6, "sub-percent minimum plasticity must parse exactly")
    check(format_percent_tenths(6) == "0.6", "sub-percent report formatting must preserve one decimal")
    try:
        parse_percent_tenths("0.60")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise VerificationError("plasticity parser must reject precision beyond one decimal place")
    args = build_parser().parse_args(
        ["--pretrain-bin", "bin", "--corpus", "corpus", "--manifest", "manifest", "--checkpoint", "ckpt", "--report", "report"]
    )
    check((args.max_vocab, args.min_frequency, args.window, args.epochs) == (32768, 3, 6, 2),
          "canonical 9 MiB capacity/frequency/window/epoch settings changed")
    check((args.initial_plasticity, args.minimum_plasticity, args.decay_events) == (110, 6, 3000),
          "canonical fixed-point plasticity settings changed")
    check(args.omp_threads is None, "OMP thread count must default to detected physical cores")


def run_checked(command: list[str], *, expected_status: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != expected_status:
        raise VerificationError(
            f"command returned {result.returncode}, expected {expected_status}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def write_fake_pretrain(path: Path, *, malformed_training_output: bool = False) -> None:
    """Write a hermetic executable that obeys just enough CLI contract for the runner."""

    training_lines = """    print('phase: vocabulary')
    print(f'  corpus lines: {records}')
    print('phase: streaming HDC context learning')
    print(f'  completed epochs: {epochs}')
    print(f'  corpus lines: {records * epochs}')
    print(f'  raw tokens: {tokens * epochs}')
    print(f'  active tokens: {tokens * epochs}')
    print('  out-of-vocabulary tokens: 0')
    print(f'  ordered context pairs: {max(1, tokens * epochs)}')
    print(f'  prototype updates: {tokens * epochs}')
    print('  tokenizer truncations: 0')
    print('  non-ASCII input bytes: 0')
    print('  dropped overlong records: 0')"""
    if malformed_training_output:
        training_lines = """    print('phase: streaming HDC context learning')
    print(f'  completed epochs: {epochs}')"""
    script = f'''#!/usr/bin/env python3
import json
import pathlib
import sys
from decimal import Decimal

def option(name, default):
    return arguments[arguments.index(name) + 1] if name in arguments else default

def percent(value):
    tenths = int(Decimal(value) * 10)
    return f"{{tenths // 10}}.{{tenths % 10}}"

arguments = sys.argv[1:]
if len(arguments) < 1:
    raise SystemExit(2)
if arguments[0] == "train":
    corpus = pathlib.Path(arguments[1])
    checkpoint = pathlib.Path(arguments[2])
    epochs = int(arguments[arguments.index("--epochs") + 1])
    lines = corpus.read_bytes().splitlines()
    records = len(lines)
    tokens = sum(len(line.split()) for line in lines)
    config = {{
        "max_vocab": int(option("--max-vocab", 32768)),
        "min_frequency": int(option("--min-frequency", 3)),
        "window": int(option("--window", 6)),
        "position_shift": int(option("--position-shift", 41)),
        "context_mode": option("--context-mode", "hierarchical"),
        "epochs": epochs,
        "initial": percent(option("--initial-plasticity", "11.0")),
        "minimum": percent(option("--minimum-plasticity", "0.6")),
        "decay": int(option("--decay-events", 3000)),
    }}
    checkpoint.write_text(json.dumps(config), encoding="utf-8")
{training_lines}
elif arguments[0] == "inspect":
    probe = arguments[arguments.index("--query") + 1]
    config = json.loads(pathlib.Path(arguments[1]).read_text(encoding="utf-8"))
    print("Vegai HDC pretraining checkpoint")
    print(f"  vocabulary capacity: {{config['max_vocab']}}")
    print(f"  min frequency: {{config['min_frequency']}}")
    print(f"  context window: {{config['window']}}")
    print(f"  positional shift: {{config['position_shift']}}")
    print(f"  context mode: {{config['context_mode']}}")
    print(f"  initial Hebbian plasticity (%): {{config['initial']}}")
    print(f"  minimum Hebbian plasticity (%): {{config['minimum']}}")
    print(f"  plasticity decay events: {{config['decay']}}")
    print(f"  completed epochs: {{config['epochs']}}")
    print(f"query: {{probe}}")
    print("  nearest semantic neighbour: vgai9m_fixture_cluster_000_token_001")
    print("  bipolar similarity:  0.00000")
else:
    raise SystemExit(2)
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def verify_exact_fixture(temp: Path) -> tuple[Path, Path, dict[str, object]]:
    first_dir = temp / "first"
    second_dir = temp / "second"
    first_corpus = first_dir / "fixture.txt"
    first_manifest = first_dir / "fixture.manifest.json"
    second_corpus = second_dir / "fixture.txt"
    second_manifest = second_dir / "fixture.manifest.json"

    # Exercise the operator-facing CLI for the full exact 9 MiB contract.
    generated = run_checked(
        [
            PYTHON,
            str(FIXTURE_TOOL),
            "generate",
            "--output",
            str(first_corpus),
            "--manifest",
            str(first_manifest),
            "--target-bytes",
            "9MiB",
        ]
    )
    generated_report = json.loads(generated.stdout)
    check(first_corpus.stat().st_size == DEFAULT_TARGET_BYTES, "generated corpus is not exactly 9 MiB")
    check(generated_report["corpus"]["bytes"] == DEFAULT_TARGET_BYTES, "CLI did not report exact 9 MiB")
    check(generated_report["records"]["unique_tokens"] == 1024, "default fixture vocabulary cardinality changed")
    check(generated_report["records"]["maximum_line_bytes"] <= 8192, "default fixture violates Fortran line bound")

    verified = run_checked(
        [PYTHON, str(FIXTURE_TOOL), "verify", "--corpus", str(first_corpus), "--manifest", str(first_manifest)]
    )
    verified_report = json.loads(verified.stdout)
    check(verified_report["corpus"]["sha256"] == generated_report["corpus"]["sha256"], "CLI verification hash changed")

    # Same basename/configuration in another directory must reproduce not only
    # corpus bytes but the path-independent deterministic manifest bytes.
    second = generate_fixture(second_corpus, second_manifest, config=_default_config(), overwrite=False)
    check(sha256_file(first_corpus) == sha256_file(second_corpus), "same fixture configuration is not byte deterministic")
    check(first_manifest.read_bytes() == second_manifest.read_bytes(), "same fixture manifest is not byte deterministic")
    check(second["corpus"]["bytes"] == DEFAULT_TARGET_BYTES, "API generation lost exact target size")
    return first_corpus, first_manifest, generated_report


def _default_config() -> FixtureConfig:
    return FixtureConfig()


def verify_tamper_rejection(temp: Path, corpus: Path, manifest: Path) -> None:
    tamper_dir = temp / "tampered"
    tamper_dir.mkdir()
    tampered_corpus = tamper_dir / corpus.name
    tampered_manifest = tamper_dir / manifest.name
    shutil.copyfile(corpus, tampered_corpus)
    shutil.copyfile(manifest, tampered_manifest)
    with tampered_corpus.open("r+b") as handle:
        original = handle.read(1)
        check(original == b"v", "fixture first byte invariant changed unexpectedly")
        handle.seek(0)
        handle.write(b"x")
        handle.flush()
    try:
        verify_fixture(tampered_corpus, tampered_manifest)
    except FixtureError:
        pass
    else:
        raise VerificationError("tampered corpus was accepted against original manifest")


def verify_runner_contract(temp: Path) -> None:
    run_dir = temp / "runner"
    corpus = run_dir / "small.txt"
    manifest = run_dir / "small.manifest.json"
    fixture = generate_fixture(
        corpus,
        manifest,
        config=FixtureConfig(
            target_bytes=256 * 1024,
            clusters=8,
            tokens_per_cluster=8,
            tokens_per_record=8,
        ),
        overwrite=False,
    )
    fake = run_dir / "fake_pretrain.py"
    write_fake_pretrain(fake)
    checkpoint = run_dir / "checkpoint.vgai"
    report = run_dir / "run.json"
    runner = run_checked(
        [
            PYTHON,
            str(RUNNER_TOOL),
            "--pretrain-bin",
            str(fake),
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
            "--epochs",
            "2",
            "--skip-resource-preflight",
            "--timeout-seconds",
            "30",
        ]
    )
    stdout_report = json.loads(runner.stdout)
    report_document = json.loads(report.read_text(encoding="utf-8"))
    check(stdout_report == report_document, "runner stdout and persisted report differ")
    check(report_document["schema"] == "vegai-9mib-pretraining-run/v2", "runner report schema changed")
    check(report_document["status"] == "succeeded", "fake valid pretraining was not successful")
    check(report_document["fixture"]["verification"]["corpus"]["bytes"] == 256 * 1024, "report lost corpus identity")
    check(report_document["inputs"]["corpus"]["sha256"] == sha256_file(corpus), "report corpus hash is wrong")
    check(report_document["checkpoint"]["sha256"] == sha256_file(checkpoint), "report checkpoint hash is wrong")
    check(report_document["train"]["command"][1] == "train", "measured command is not train")
    check(report_document["measurement"]["scope"] == "one vegai_pretrain train child-process wall-clock interval", "measurement scope changed")
    check(report_document["measurement"]["train_process_wall_seconds"] > 0.0, "train duration is not positive")
    check(
        report_document["train"]["metrics"]["raw tokens"] == fixture["records"]["raw_tokens"] * 2,
        "runner did not bind token metric to fixture and epoch count",
    )
    check(
        report_document["inspection"]["child_wall_seconds_not_in_train_measurement"] > 0.0,
        "checkpoint inspection must be recorded outside measured train interval",
    )
    check(report_document["resource_preflight"]["status"] == "skipped-by-explicit-operator-option", "bypass was not explicit")
    configuration = report_document["configuration"]
    physical = configuration["hardware_profile"]["cpu"]["topology"]["physical_cores"]
    check(1 <= configuration["omp_threads"] <= physical, "runner did not cap OpenMP threads to physical cores")
    check(configuration["context_provider"] == "cpu", "default run must attest native CPU provider")
    check(configuration["parallel_min_contexts"] == (16 if configuration["omp_threads"] > 1 else 32),
          "adaptive OpenMP record-batch threshold does not match physical thread count")

    # Even with --overwrite, an output alias typo must be rejected before it
    # can remove the input corpus.
    corpus_hash_before_alias_attempt = sha256_file(corpus)
    alias_attempt = run_checked(
        [
            PYTHON,
            str(RUNNER_TOOL),
            "--pretrain-bin",
            str(fake),
            "--corpus",
            str(corpus),
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(run_dir / "alias-attempt.vgai"),
            "--report",
            str(corpus),
            "--max-vocab",
            str(fixture["records"]["unique_tokens"]),
            "--skip-resource-preflight",
            "--overwrite",
        ],
        expected_status=2,
    )
    check("must not overlap" in alias_attempt.stderr, "output/input alias was not rejected before execution")
    check(sha256_file(corpus) == corpus_hash_before_alias_attempt, "output alias attempt modified input corpus")

    # A zero-exit executable with an incomplete protocol cannot be passed off
    # as training. The runner must fail and persist a failed, non-success report.
    malformed = run_dir / "malformed_pretrain.py"
    write_fake_pretrain(malformed, malformed_training_output=True)
    failed_report = run_dir / "failed.json"
    failed = run_checked(
        [
            PYTHON,
            str(RUNNER_TOOL),
            "--pretrain-bin",
            str(malformed),
            "--corpus",
            str(corpus),
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(run_dir / "failed.vgai"),
            "--report",
            str(failed_report),
            "--max-vocab",
            str(fixture["records"]["unique_tokens"]),
            "--skip-resource-preflight",
        ],
        expected_status=2,
    )
    check("FAIL:" in failed.stderr, "runner failure was not surfaced on stderr")
    failure_document = json.loads(failed_report.read_text(encoding="utf-8"))
    check(failure_document["status"] == "failed", "malformed training protocol produced a success report")
    check(failure_document["failure"]["type"] == "PretrainingRunError", "failure report lacks typed violation")


def main() -> int:
    try:
        verify_parameter_configuration()
        with tempfile.TemporaryDirectory(prefix="vegai-9mib-contract-") as temporary:
            temp = Path(temporary)
            corpus, manifest, _ = verify_exact_fixture(temp)
            verify_tamper_rejection(temp, corpus, manifest)
            verify_runner_contract(temp)
    except (OSError, ValueError, json.JSONDecodeError, FixtureError, VerificationError) as error:
        print(f"verify 9 MiB pretraining: FAIL: {error}", file=sys.stderr)
        return 1
    print("verify 9 MiB pretraining: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
