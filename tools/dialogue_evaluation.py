#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Evaluate and release-gate grounded Vegai dialogue retrieval.

This tool evaluates the compiled ``vegai_chat ask`` contract against a
versioned JSONL held-out set.  It intentionally measures whether a query is
correctly grounded to an operator-owned knowledge-base *prompt* or correctly
abstained from; it does not pretend that HDC retrieval is an LLM benchmark or
that a response's prose is independently factual.

The report binds the evaluator, chat binary, checkpoint, knowledge base,
evaluation set, and stateless threshold policy by SHA-256. Per-case output
contains SHA-256 or optional keyed HMAC-SHA-256 prompt identifiers rather than
raw text, so a release report remains useful without unnecessarily copying
held-out prompts into another artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence


CASE_SCHEMA = "vegai-dialogue-evaluation-case/v1"
REPORT_SCHEMA = "vegai-dialogue-evaluation-report/v1"
MAX_CASE_LINE_BYTES = 16 * 1024
MAX_CASE_ID_BYTES = 256
MAX_PROMPT_BYTES = 1024
MAX_HMAC_KEY_BYTES = 4096
DEFAULT_MAX_CASES = 10_000
DEFAULT_TIMEOUT_SECONDS = 30.0


class DialogueEvaluationError(RuntimeError):
    """Expected input, execution, parsing, or quality-gate failure."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Byte identity of one immutable evaluation dependency."""

    path: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One held-out query: expected prompt or explicit expected abstention."""

    number: int
    case_id: str
    prompt: str
    expected_prompt: str | None

    def canonical_bytes(self) -> bytes:
        payload = {
            "id": self.case_id,
            "prompt": self.prompt,
            "expected_prompt": self.expected_prompt,
        }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"


@dataclass(frozen=True, slots=True)
class ChatObservation:
    """Strictly parsed machine facts emitted by one ``vegai_chat ask`` call."""

    grounded: bool
    matched_prompt: str | None
    known_tokens: int
    ignored_tokens: int
    candidates_examined: int
    similarity: float
    runner_up_similarity: float
    margin: float


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DialogueEvaluationError(message)


def text_digest(value: str | None, hmac_key: bytes | None) -> str | None:
    """Return a reproducible SHA-256 or keyed HMAC-SHA-256 text identifier."""
    if value is None:
        return None
    encoded = value.encode("ascii")
    if hmac_key is None:
        return hashlib.sha256(encoded).hexdigest()
    return hmac.new(hmac_key, encoded, hashlib.sha256).hexdigest()


def load_hmac_key(path: Path) -> tuple[bytes, FileIdentity]:
    """Read a bounded local secret and its identity from the same byte snapshot."""
    resolved = path.resolve()
    require(resolved.is_file(), f"case HMAC key is not a regular file: {resolved}")
    try:
        with resolved.open("rb") as handle:
            value = handle.read(MAX_HMAC_KEY_BYTES + 1)
    except OSError as error:
        raise DialogueEvaluationError(f"cannot read case HMAC key {resolved}: {error}") from error
    require(bool(value), "case HMAC key must not be empty")
    require(len(value) <= MAX_HMAC_KEY_BYTES, f"case HMAC key exceeds {MAX_HMAC_KEY_BYTES} bytes")
    return value, FileIdentity(str(resolved), hashlib.sha256(value).hexdigest(), len(value))


def file_identity(path: Path, label: str) -> FileIdentity:
    resolved = path.resolve()
    require(resolved.is_file(), f"{label} is not a regular file: {resolved}")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise DialogueEvaluationError(f"cannot fingerprint {label} {resolved}: {error}") from error
    return FileIdentity(str(resolved), digest.hexdigest(), byte_count)


def require_unchanged(path: Path, before: FileIdentity, label: str) -> FileIdentity:
    after = file_identity(path, label)
    require(after == before, f"{label} changed while evaluation was running: {path}")
    return after


def fsync_directory(directory: Path) -> None:
    """Best-effort directory durability after an atomic report replacement."""
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, os.O_RDONLY | directory_flag)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_ascii_text(value: object, field: str, maximum_bytes: int, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    require(isinstance(value, str) and value, f"{field} must be a nonempty string" + (" or null" if allow_none else ""))
    require("\r" not in value and "\n" not in value, f"{field} must not contain CR/LF")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise DialogueEvaluationError(f"{field} must be printable ASCII to match vegai_chat") from error
    require(len(encoded) <= maximum_bytes, f"{field} exceeds {maximum_bytes} bytes")
    require(all(32 <= byte <= 126 for byte in encoded), f"{field} contains a non-printable ASCII byte")
    return value


def _iter_bounded_lines(handle: Any, maximum_bytes: int) -> Iterator[tuple[int, bytes, bool]]:
    """Yield LF records without allocating an unbounded hostile JSONL line."""
    number = 0
    limit = maximum_bytes + 2
    while True:
        raw = handle.readline(limit)
        if not raw:
            return
        number += 1
        has_lf = raw.endswith(b"\n")
        payload_bytes = len(raw) - 1 if has_lf else len(raw)
        overlong = payload_bytes > maximum_bytes or (not has_lf and len(raw) == limit)
        if overlong and not has_lf:
            while raw and not raw.endswith(b"\n"):
                raw = handle.readline(limit)
        yield number, raw if not overlong else b"", overlong


def parse_case_mapping(value: object, number: int) -> EvaluationCase:
    require(isinstance(value, Mapping), f"evaluation record {number} must be a JSON object")
    allowed = {"schema_version", "id", "prompt", "expected_prompt"}
    unknown = sorted(str(key) for key in value.keys() if key not in allowed)
    require(not unknown, f"evaluation record {number} has unsupported fields: {', '.join(unknown)}")
    require("schema_version" in value, f"evaluation record {number} must contain schema_version")
    version = value["schema_version"]
    require(version == CASE_SCHEMA, f"evaluation record {number} has unsupported schema_version: {version!r}")
    require("id" in value and "prompt" in value and "expected_prompt" in value,
            f"evaluation record {number} must contain id, prompt, and expected_prompt")
    case_id = validate_ascii_text(value["id"], f"evaluation record {number}.id", MAX_CASE_ID_BYTES)
    prompt = validate_ascii_text(value["prompt"], f"evaluation record {number}.prompt", MAX_PROMPT_BYTES)
    expected = validate_ascii_text(
        value["expected_prompt"], f"evaluation record {number}.expected_prompt", MAX_PROMPT_BYTES, allow_none=True
    )
    assert case_id is not None and prompt is not None
    return EvaluationCase(number=number, case_id=case_id, prompt=prompt, expected_prompt=expected)


def iter_cases(path: Path, max_cases: int) -> Iterator[EvaluationCase]:
    """Strictly stream versioned JSONL cases; CRLF and blank rows are rejected."""
    require(max_cases > 0, "--max-cases must be positive")
    try:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rb") as handle:
            ids: set[str] = set()
            yielded = 0
            for number, raw, overlong in _iter_bounded_lines(handle, MAX_CASE_LINE_BYTES):
                require(not overlong, f"evaluation case line {number} exceeds {MAX_CASE_LINE_BYTES} bytes")
                require(raw.endswith(b"\n"), f"evaluation case line {number} must end in LF")
                raw = raw[:-1]
                require(not raw.endswith(b"\r"), f"evaluation case line {number} uses CRLF; strict LF is required")
                require(bool(raw.strip()), f"evaluation case line {number} is blank")
                try:
                    decoded = raw.decode("utf-8")
                    parsed = json.loads(decoded)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise DialogueEvaluationError(f"invalid JSONL evaluation case {number}: {error}") from error
                case = parse_case_mapping(parsed, number)
                require(case.case_id not in ids, f"evaluation case id is duplicated: {case.case_id!r}")
                ids.add(case.case_id)
                yielded += 1
                require(yielded <= max_cases, f"evaluation exceeds configured --max-cases={max_cases}")
                yield case
    except (OSError, EOFError) as error:
        raise DialogueEvaluationError(f"cannot read evaluation set {path}: {error}") from error


def parse_int(value: str, field: str) -> int:
    try:
        parsed = int(value.strip())
    except ValueError as error:
        raise DialogueEvaluationError(f"vegai_chat emitted invalid {field}: {value!r}") from error
    require(parsed >= 0, f"vegai_chat emitted negative {field}")
    return parsed


def parse_float(value: str, field: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as error:
        raise DialogueEvaluationError(f"vegai_chat emitted invalid {field}: {value!r}") from error
    require(math.isfinite(parsed) and minimum <= parsed <= maximum,
            f"vegai_chat emitted out-of-range {field}: {value!r}")
    return parsed


def output_fields(output: str, prefix: str) -> list[str]:
    return [line[len(prefix):] for line in output.splitlines() if line.startswith(prefix)]


def exactly_one_field(output: str, prefix: str, label: str) -> str:
    values = output_fields(output, prefix)
    require(len(values) == 1, f"vegai_chat output must contain exactly one {label} field")
    return values[0]


def parse_chat_output(output: str) -> ChatObservation:
    """Treat CLI text as a versioned parse contract rather than best effort."""
    known = parse_int(exactly_one_field(output, "query known tokens: ", "query-known-token"), "query known tokens")
    ignored = parse_int(exactly_one_field(output, "query ignored tokens: ", "query-ignored-token"), "query ignored tokens")
    candidates = parse_int(exactly_one_field(output, "candidates examined: ", "candidate-count"), "candidates examined")
    require(candidates > 0, "vegai_chat evaluated no knowledge-base candidates")
    similarity = parse_float(
        exactly_one_field(output, "best bipolar similarity: ", "best-similarity"), "best similarity", -1.0, 1.0
    )
    runner_up = parse_float(
        exactly_one_field(output, "runner-up bipolar similarity: ", "runner-up-similarity"),
        "runner-up similarity", -1.0, 1.0,
    )
    margin = parse_float(exactly_one_field(output, "grounding margin: ", "grounding-margin"), "grounding margin", 0.0, 2.0)
    grounded_value = exactly_one_field(output, "grounded: ", "grounded")
    require(grounded_value in {"true", "false"}, f"vegai_chat emitted invalid grounded value: {grounded_value!r}")
    matched = output_fields(output, "matched prompt: ")
    answers = output_fields(output, "answer: ")
    grounded = grounded_value == "true"
    if grounded:
        require(len(matched) == 1 and len(answers) == 1, "grounded chat output omitted or duplicated source answer fields")
        matched_prompt = validate_ascii_text(matched[0], "vegai_chat matched prompt", MAX_PROMPT_BYTES)
        assert matched_prompt is not None
    else:
        require(not matched and not answers, "abstained chat output must not expose a matched prompt or answer")
        matched_prompt = None
    return ChatObservation(
        grounded=grounded,
        matched_prompt=matched_prompt,
        known_tokens=known,
        ignored_tokens=ignored,
        candidates_examined=candidates,
        similarity=similarity,
        runner_up_similarity=runner_up,
        margin=margin,
    )


def normalized_float(value: float) -> str:
    """Use a stable CLI spelling and reject non-finite caller policy values."""
    require(math.isfinite(value), "threshold values must be finite")
    return format(value, ".9g")


def chat_command(arguments: argparse.Namespace, case: EvaluationCase) -> list[str]:
    command = [
        str(arguments.chat_bin.resolve()),
        "ask",
        str(arguments.checkpoint.resolve()),
        str(arguments.knowledge.resolve()),
        "--prompt",
        case.prompt,
        "--max-entries",
        str(arguments.max_entries),
        # Evaluation is deliberately stateless so case order cannot change a
        # score through conversational history.
        "--history-turns",
        "0",
        "--min-known-tokens",
        str(arguments.min_known_tokens),
        "--min-similarity",
        normalized_float(arguments.min_similarity),
        "--min-margin",
        normalized_float(arguments.min_margin),
    ]
    if arguments.require_all_known:
        command.append("--require-all-known")
    return command


def run_case(arguments: argparse.Namespace, case: EvaluationCase) -> ChatObservation:
    environment = os.environ.copy()
    # Release evaluation needs a stable evaluation contract. The underlying
    # exact retrieval is deterministic, and one thread avoids environmental
    # ambiguity in checkpoint/query setup telemetry.
    environment["OMP_NUM_THREADS"] = "1"
    try:
        completed = subprocess.run(
            chat_command(arguments, case),
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            text=True,
            encoding="utf-8",
            errors="strict",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=arguments.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise DialogueEvaluationError(f"case {case.case_id!r} exceeded {arguments.timeout_seconds} seconds") from error
    except OSError as error:
        raise DialogueEvaluationError(f"cannot launch vegai_chat for case {case.case_id!r}: {error}") from error
    if completed.returncode != 0:
        # The child may have emitted a TSV-sourced response before failing.
        # Bind diagnostics without copying held-out or operator-owned text into
        # an otherwise privacy-conscious release log.
        output = completed.stdout or ""
        output_digest = hashlib.sha256(output.encode("utf-8")).hexdigest()
        raise DialogueEvaluationError(
            f"vegai_chat failed for case {case.case_id!r} with exit {completed.returncode}; "
            f"captured-output-sha256={output_digest}, bytes={len(output.encode('utf-8'))}"
        )
    return parse_chat_output(completed.stdout)


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 9)


def classify(case: EvaluationCase, observed: ChatObservation) -> str:
    if case.expected_prompt is None:
        return "true_abstain" if not observed.grounded else "false_grounded"
    if not observed.grounded:
        return "false_abstain"
    return "correct_grounded_prompt" if observed.matched_prompt == case.expected_prompt else "wrong_grounded_prompt"


def build_report(
    arguments: argparse.Namespace,
    cases: Sequence[EvaluationCase],
    identities: Mapping[str, FileIdentity],
    hmac_key: bytes | None,
) -> dict[str, Any]:
    metrics_count = {
        "total_cases": len(cases),
        "expected_grounded_cases": 0,
        "expected_abstain_cases": 0,
        "actual_grounded_cases": 0,
        "actual_abstain_cases": 0,
        "correct_grounded_prompt_cases": 0,
        "wrong_grounded_prompt_cases": 0,
        "false_abstain_cases": 0,
        "false_grounded_cases": 0,
        "true_abstain_cases": 0,
    }
    results: list[dict[str, Any]] = []
    stream_digest = hmac.new(hmac_key, digestmod=hashlib.sha256) if hmac_key is not None else hashlib.sha256()
    for case in cases:
        stream_digest.update(case.canonical_bytes())
        observed = run_case(arguments, case)
        outcome = classify(case, observed)
        if case.expected_prompt is None:
            metrics_count["expected_abstain_cases"] += 1
        else:
            metrics_count["expected_grounded_cases"] += 1
        if observed.grounded:
            metrics_count["actual_grounded_cases"] += 1
        else:
            metrics_count["actual_abstain_cases"] += 1
        metrics_count[f"{outcome}_cases"] += 1
        results.append(
            {
                "case_id": case.case_id,
                "prompt_digest": text_digest(case.prompt, hmac_key),
                "expected_prompt_digest": text_digest(case.expected_prompt, hmac_key),
                "grounded": observed.grounded,
                "matched_prompt_digest": text_digest(observed.matched_prompt, hmac_key),
                "outcome": outcome,
                "query_known_tokens": observed.known_tokens,
                "query_ignored_tokens": observed.ignored_tokens,
                "candidates_examined": observed.candidates_examined,
                "best_bipolar_similarity": round(observed.similarity, 9),
                "runner_up_bipolar_similarity": round(observed.runner_up_similarity, 9),
                "grounding_margin": round(observed.margin, 9),
            }
        )

    metrics: dict[str, int | float | None] = dict(metrics_count)
    metrics["grounded_prompt_accuracy"] = ratio(
        metrics_count["correct_grounded_prompt_cases"], metrics_count["expected_grounded_cases"]
    )
    metrics["grounded_precision"] = ratio(
        metrics_count["correct_grounded_prompt_cases"], metrics_count["actual_grounded_cases"]
    )
    metrics["grounded_recall"] = ratio(
        metrics_count["correct_grounded_prompt_cases"], metrics_count["expected_grounded_cases"]
    )
    metrics["abstain_precision"] = ratio(
        metrics_count["true_abstain_cases"], metrics_count["actual_abstain_cases"]
    )
    metrics["abstain_recall"] = ratio(
        metrics_count["true_abstain_cases"], metrics_count["expected_abstain_cases"]
    )
    metrics["grounded_coverage"] = ratio(metrics_count["actual_grounded_cases"], metrics_count["total_cases"])

    return {
        "schema": REPORT_SCHEMA,
        "evaluation_contract": {
            "case_schema": CASE_SCHEMA,
            "case_stream_digest": stream_digest.hexdigest(),
            "case_digest_algorithm": "hmac-sha256" if hmac_key is not None else "sha256",
            "case_order": "source JSONL order; stateless evaluation; OMP_NUM_THREADS=1",
            "privacy": (
                "per-case report stores keyed HMAC-SHA-256 prompt identifiers, not raw held-out prompts or responses"
                if hmac_key is not None
                else "per-case report stores SHA-256 prompt identifiers, not raw held-out prompts or responses"
            ),
        },
        "identities": {
            label: (
                {"path": "<redacted-case-hmac-key>", "sha256": identity.sha256, "bytes": identity.bytes}
                if label == "case_hmac_key"
                else identity.to_dict()
            )
            for label, identity in identities.items()
        },
        "policy": {
            "max_entries": arguments.max_entries,
            "history_turns": 0,
            "min_known_tokens": arguments.min_known_tokens,
            "require_all_known": arguments.require_all_known,
            "minimum_similarity": arguments.min_similarity,
            "minimum_margin": arguments.min_margin,
            "per_case_timeout_seconds": arguments.timeout_seconds,
        },
        "metrics": metrics,
        "cases": results,
    }


def gate_requirements(arguments: argparse.Namespace) -> dict[str, int | float | None]:
    """Serialize release thresholds so a report is self-describing."""
    return {
        "min_grounded_prompt_accuracy": arguments.min_grounded_prompt_accuracy,
        "min_grounded_precision": arguments.min_grounded_precision,
        "min_grounded_recall": arguments.min_grounded_recall,
        "min_abstain_precision": arguments.min_abstain_precision,
        "min_abstain_recall": arguments.min_abstain_recall,
        "min_grounded_coverage": arguments.min_grounded_coverage,
        "max_wrong_grounded_prompt_cases": arguments.max_wrong_grounded_prompt_cases,
        "max_false_grounded_cases": arguments.max_false_grounded_cases,
        "max_false_abstain_cases": arguments.max_false_abstain_cases,
    }


def gate_failures(report: Mapping[str, Any], arguments: argparse.Namespace) -> list[str]:
    metrics = report["metrics"]
    assert isinstance(metrics, Mapping)
    requested = {
        "grounded_prompt_accuracy": arguments.min_grounded_prompt_accuracy,
        "grounded_precision": arguments.min_grounded_precision,
        "grounded_recall": arguments.min_grounded_recall,
        "abstain_precision": arguments.min_abstain_precision,
        "abstain_recall": arguments.min_abstain_recall,
        "grounded_coverage": arguments.min_grounded_coverage,
    }
    failures: list[str] = []
    for name, threshold in requested.items():
        if threshold is None:
            continue
        value = metrics.get(name)
        if not isinstance(value, (int, float)):
            failures.append(f"{name} is undefined because its denominator has no cases; cannot satisfy >= {threshold}")
        elif value < threshold:
            failures.append(f"{name}={value:.9f} is below required {threshold:.9f}")
    maximums = {
        "wrong_grounded_prompt_cases": arguments.max_wrong_grounded_prompt_cases,
        "false_grounded_cases": arguments.max_false_grounded_cases,
        "false_abstain_cases": arguments.max_false_abstain_cases,
    }
    for name, maximum in maximums.items():
        if maximum is None:
            continue
        value = metrics.get(name)
        assert isinstance(value, int)
        if value > maximum:
            failures.append(f"{name}={value} exceeds permitted {maximum}")
    return failures


def parse_fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite fraction in [0, 1]") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite fraction in [0, 1]")
    return parsed


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-bin", required=True, type=Path, help="compiled vegai_chat executable")
    parser.add_argument("--checkpoint", required=True, type=Path, help="immutable HDC checkpoint used for every case")
    parser.add_argument("--knowledge", required=True, type=Path, help="operator TSV knowledge base used for every case")
    parser.add_argument("--evaluation", required=True, type=Path, help="strict JSONL held-out case file (.jsonl or .jsonl.gz)")
    parser.add_argument("--case-hmac-key-file", type=Path,
                        help="optional bounded secret file; use HMAC-SHA-256 rather than reversible raw prompt hashes")
    parser.add_argument("--output", type=Path, help="optional atomic JSON report path")
    parser.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES)
    parser.add_argument("--max-entries", type=int, default=4096, help="same bounded index capacity passed to vegai_chat")
    parser.add_argument("--min-known-tokens", type=int, default=1)
    parser.add_argument("--require-all-known", action="store_true")
    parser.add_argument("--min-similarity", type=float, default=0.05)
    parser.add_argument("--min-margin", type=float, default=0.01)
    parser.add_argument("--timeout-seconds", type=parse_positive_float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--min-grounded-prompt-accuracy", type=parse_fraction)
    parser.add_argument("--min-grounded-precision", type=parse_fraction)
    parser.add_argument("--min-grounded-recall", type=parse_fraction)
    parser.add_argument("--min-abstain-precision", type=parse_fraction)
    parser.add_argument("--min-abstain-recall", type=parse_fraction)
    parser.add_argument("--min-grounded-coverage", type=parse_fraction)
    parser.add_argument("--max-wrong-grounded-prompt-cases", type=parse_nonnegative_int)
    parser.add_argument("--max-false-grounded-cases", type=parse_nonnegative_int)
    parser.add_argument("--max-false-abstain-cases", type=parse_nonnegative_int)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    require(arguments.max_cases > 0, "--max-cases must be positive")
    require(arguments.max_entries > 0, "--max-entries must be positive")
    require(arguments.min_known_tokens > 0, "--min-known-tokens must be positive")
    require(math.isfinite(arguments.min_similarity) and -1.0 <= arguments.min_similarity <= 1.0,
            "--min-similarity must be finite and in [-1, 1]")
    require(math.isfinite(arguments.min_margin) and 0.0 <= arguments.min_margin <= 2.0,
            "--min-margin must be finite and in [0, 2]")
    require(os.access(arguments.chat_bin.resolve(), os.X_OK),
            f"chat executable is not executable: {arguments.chat_bin}")
    if arguments.output is not None:
        output = arguments.output.resolve()
        dependencies: list[tuple[str, Path]] = [
            ("evaluator tool", Path(__file__)),
            ("chat executable", arguments.chat_bin),
            ("checkpoint", arguments.checkpoint),
            ("knowledge base", arguments.knowledge),
            ("evaluation set", arguments.evaluation),
        ]
        if arguments.case_hmac_key_file is not None:
            dependencies.append(("case HMAC key", arguments.case_hmac_key_file))
        for label, dependency in dependencies:
            same_file = output == dependency.resolve()
            if output.exists():
                try:
                    same_file = same_file or os.path.samefile(output, dependency)
                except OSError:
                    pass
            require(not same_file, f"--output must not overwrite the {label}: {output}")


def main() -> int:
    arguments = parse_arguments()
    try:
        validate_arguments(arguments)
        evaluator_source = Path(__file__).resolve()
        identities = {
            "evaluator_tool": file_identity(evaluator_source, "evaluator tool"),
            "chat_binary": file_identity(arguments.chat_bin, "chat executable"),
            "checkpoint": file_identity(arguments.checkpoint, "checkpoint"),
            "knowledge_base": file_identity(arguments.knowledge, "knowledge base"),
            "evaluation_set": file_identity(arguments.evaluation, "evaluation set"),
        }
        hmac_key: bytes | None = None
        immutable_dependencies: list[tuple[str, str, Path]] = [
            ("evaluator tool", "evaluator_tool", evaluator_source),
            ("chat executable", "chat_binary", arguments.chat_bin),
            ("checkpoint", "checkpoint", arguments.checkpoint),
            ("knowledge base", "knowledge_base", arguments.knowledge),
            ("evaluation set", "evaluation_set", arguments.evaluation),
        ]
        if arguments.case_hmac_key_file is not None:
            hmac_key, identities["case_hmac_key"] = load_hmac_key(arguments.case_hmac_key_file)
            immutable_dependencies.append(("case HMAC key", "case_hmac_key", arguments.case_hmac_key_file))
        cases = list(iter_cases(arguments.evaluation, arguments.max_cases))
        require(cases, "evaluation set contains no cases")
        report = build_report(arguments, cases, identities, hmac_key)
        # The source bytes that define every observed result must stay stable
        # through the entire serial evaluation transaction.
        for label, identity_key, path in immutable_dependencies:
            require_unchanged(path, identities[identity_key], label)
        failures = gate_failures(report, arguments)
        report["quality_gate"] = {
            "passed": not failures,
            "requirements": gate_requirements(arguments),
            "failures": failures,
            "contract": (
                "Thresholds assess only this versioned held-out grounded-retrieval set. They do not establish "
                "general language understanding, factual truth, safety, or LMM equivalence."
            ),
        }
        if arguments.output is not None:
            write_json_atomic(arguments.output, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        if failures:
            print("dialogue evaluation: QUALITY GATE FAILED: " + "; ".join(failures), file=sys.stderr)
            return 3
        return 0
    except (DialogueEvaluationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"dialogue evaluation: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
