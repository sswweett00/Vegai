#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a provenance-bound 9M-token Turkish Wikipedia HDC experiment.

The native model, checkpoint, semantic inspection, and grounded dialogue
retrieval are all performed by the compiled Fortran executables.  This small
stdlib-only controller deliberately has a narrower job: it turns an already
retrieved open corpus into the strict ASCII/LF input contract, records every
content identity, measures only the native training child, and collects the
operator-visible evidence without presenting retrieval as free-form generation.

It never downloads a corpus.  The workflow that invokes it owns retrieval from
an explicit public URL, so source selection and network access cannot be hidden
inside a preprocessing helper.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "vegai-turkish-wikipedia-9m-experiment/v1"
CORPUS_SCHEMA = "vegai-turkish-wikipedia-ascii-corpus/v1"
TARGET_TOKENS_DEFAULT = 9_000_000
VOCABULARY_SIZE_DEFAULT = 8_192
EPOCHS_DEFAULT = 2
TOKENS_PER_RECORD = 128
MAX_NATIVE_RECORD_BYTES = 8_192
TOKEN_MAX_BYTES = 48
TOKEN_PATTERN = re.compile(r"[a-z0-9_']{1,48}\Z")
ASCII_TOKEN_FIND = re.compile(r"[A-Za-z0-9_']+")
SEMANTIC_PRIORITY = (
    "turkiye",
    "istanbul",
    "ankara",
    "tarih",
    "devlet",
    "universite",
    "belediye",
    "sanat",
    "futbol",
    "kitap",
    "dil",
    "muzik",
    "film",
    "savas",
    "egitim",
    "bilim",
    "edebiyat",
    "sehir",
    "uluslararasi",
    "cumhuriyet",
)
STOP_TOKENS = frozenset(
    {
        "bir",
        "ve",
        "bu",
        "da",
        "de",
        "ile",
        "icin",
        "olarak",
        "olan",
        "da",
        "ya",
        "veya",
        "daha",
        "sonra",
        "gibi",
        "en",
        "ise",
        "ile",
        "nin",
        "na",
        "ne",
        "den",
        "dan",
        "in",
        "birinci",
    }
)

# Turkish characters are handled before Unicode decomposition.  The native
# tokenizer is intentionally ASCII-only today; this explicit, versioned
# transliteration preserves a consistent lexical projection rather than letting
# UTF-8 bytes become accidental token boundaries.  Other accented Latin forms
# are decomposed and their combining marks are dropped below.
TURKISH_ASCII_TRANSLITERATION = str.maketrans(
    {
        "Ç": "C",
        "ç": "c",
        "Ğ": "G",
        "ğ": "g",
        "İ": "I",
        "ı": "i",
        "Ö": "O",
        "ö": "o",
        "Ş": "S",
        "ş": "s",
        "Ü": "U",
        "ü": "u",
        "’": "'",
        "‘": "'",
        "`": "'",
        "´": "'",
    }
)


class ExperimentError(RuntimeError):
    """An expected contract, provenance, or evidence failure."""


@dataclass(frozen=True)
class ProcessResult:
    command: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentError(message)


def sha256_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise ExperimentError(f"cannot hash {path}: {error}") from error
    return {"path": path.name, "bytes": byte_count, "sha256": digest.hexdigest()}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def git_revision(repository: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ExperimentError(f"cannot resolve source revision in {repository}: {error}") from error
    if completed.returncode != 0:
        raise ExperimentError(f"cannot resolve source revision in {repository}: {completed.stderr.strip()}")
    revision = completed.stdout.strip()
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "source revision is not a full Git SHA-1")
    return revision


def license_identity(repository: Path) -> dict[str, Any]:
    candidates = ("LICENSE", "LICENCE", "LICENSE.md", "LICENCE.md")
    for name in candidates:
        candidate = repository / name
        if candidate.is_file():
            return sha256_file(candidate)
    raise ExperimentError(f"source repository has no recognized license file: {repository}")


def normalize_tokens(text: str) -> list[str]:
    """Apply the documented tr-ascii-v1 projection and native token grammar."""
    transliterated = text.translate(TURKISH_ASCII_TRANSLITERATION)
    decomposed = unicodedata.normalize("NFKD", transliterated)
    ascii_projection = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    tokens = ASCII_TOKEN_FIND.findall(ascii_projection)
    # The regex has no upper length bound because an individual source token can
    # be longer than Vegai's fixed key.  Drop it rather than silently truncate.
    return [token for token in tokens if len(token) <= TOKEN_MAX_BYTES]


def normalize_display_text(text: str) -> str:
    """Produce a printable ASCII TSV field without interpreting its contents."""
    transliterated = text.translate(TURKISH_ASCII_TRANSLITERATION)
    decomposed = unicodedata.normalize("NFKD", transliterated)
    ascii_projection = decomposed.encode("ascii", "ignore").decode("ascii")
    printable = "".join(character if 32 <= ord(character) <= 126 and character != "\t" else " " for character in ascii_projection)
    return " ".join(printable.split())


def iter_raw_lines(path: Path) -> Iterable[tuple[int, bytes, str]]:
    try:
        with path.open("rb") as handle:
            for number, raw in enumerate(handle, start=1):
                try:
                    yield number, raw, raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError as error:
                    raise ExperimentError(f"source line {number} is not valid UTF-8: {error}") from error
    except OSError as error:
        raise ExperimentError(f"cannot read Turkish Wikipedia source split {path}: {error}") from error


def scan_source_vocabulary(source_raw: Path) -> tuple[Counter[str], dict[str, Any]]:
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    source_lines = 0
    raw_tokens = 0
    for source_lines, raw, decoded in iter_raw_lines(source_raw):
        digest.update(raw)
        tokens = normalize_tokens(decoded)
        raw_tokens += len(tokens)
        counts.update(tokens)
    require(source_lines > 0, "Turkish Wikipedia raw source has no physical records")
    require(raw_tokens > 0 and counts, "Turkish Wikipedia raw source yielded no native-compatible tokens")
    return counts, {
        "path": source_raw.name,
        "bytes": source_raw.stat().st_size,
        "sha256": digest.hexdigest(),
        "physical_records": source_lines,
        "normalised_token_occurrences": raw_tokens,
        "normalised_unique_tokens": len(counts),
    }


def emit_segment(
    output: Any,
    segment: list[str],
    remaining: int,
) -> tuple[int, int]:
    """Write in-source-order contiguous selected lexical segments only.

    A dropped, out-of-selected-vocabulary token is a hard record boundary.  We
    intentionally do not concatenate words across a filtered source token,
    avoiding an invented local context in the HDC learning stream.
    """
    emitted = 0
    records = 0
    cursor = 0
    # A token without an active neighbour is correctly counted as an active
    # lexical occurrence by the native reader, but cannot receive an HDC
    # context/prototype update.  Preserve the stronger experiment contract by
    # emitting only pairs (or longer even-sized chunks); isolated selected
    # source terms remain an explicit filter boundary rather than inflated
    # "trained token" accounting.  The target is required to be even below.
    while cursor < len(segment) and emitted < remaining:
        take = min(TOKENS_PER_RECORD, len(segment) - cursor, remaining - emitted)
        if take % 2 != 0:
            take = take - 1
        if take < 2:
            exit
        record = " ".join(segment[cursor : cursor + take])
        encoded = record.encode("ascii", "strict")
        require(len(encoded) <= MAX_NATIVE_RECORD_BYTES, "generated native record exceeds 8,192 bytes")
        output.write(record + "\n")
        cursor += take
        emitted += take
        records += 1
    return emitted, records


def validate_generated_corpus(corpus: Path, expected_tokens: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    bytes_seen = 0
    records = 0
    tokens = 0
    maximum_record_bytes = 0
    try:
        with corpus.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                bytes_seen += len(raw)
                require(raw.endswith(b"\n"), "generated corpus has a non-LF-terminated record")
                record = raw[:-1]
                require(b"\r" not in record, "generated corpus contains a CR byte")
                require(all(byte < 128 for byte in record), "generated corpus contains a non-ASCII byte")
                require(len(record) <= MAX_NATIVE_RECORD_BYTES, "generated corpus record exceeds native limit")
                rendered = record.decode("ascii", "strict")
                fields = rendered.split(" ")
                require(len(fields) >= 2 and all(TOKEN_PATTERN.fullmatch(item) is not None for item in fields),
                        "generated corpus contains an invalid token or a contextless one-token record")
                records += 1
                tokens += len(fields)
                maximum_record_bytes = max(maximum_record_bytes, len(record))
    except OSError as error:
        raise ExperimentError(f"cannot validate generated corpus {corpus}: {error}") from error
    require(tokens == expected_tokens, f"generated corpus has {tokens} native tokens, expected {expected_tokens}")
    return {
        "path": corpus.name,
        "bytes": bytes_seen,
        "sha256": digest.hexdigest(),
        "physical_lf_records": records,
        "native_token_count": tokens,
        "maximum_record_bytes": maximum_record_bytes,
        "non_ascii_bytes": 0,
        "crlf_records": 0,
        "overlong_records": 0,
        "tokenizer_truncations": 0,
    }


def prepare_corpus(
    source_raw: Path,
    output: Path,
    selected_vocabulary_output: Path,
    *,
    target_tokens: int,
    vocabulary_size: int,
) -> dict[str, Any]:
    require(target_tokens >= 9_000_000, "target token count must not be below 9,000,000")
    require(target_tokens % 2 == 0, "target token count must be even so every emitted token has a context neighbour")
    require(vocabulary_size > 11, "selected vocabulary must contain more than eleven terms")
    counts, source_identity = scan_source_vocabulary(source_raw)
    require(len(counts) >= vocabulary_size,
            f"source has only {len(counts)} normalized unique tokens, below requested vocabulary {vocabulary_size}")
    ranked_vocabulary = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:vocabulary_size]
    selected = frozenset(token for token, _ in ranked_vocabulary)
    require(len(selected) == vocabulary_size, "frequency selection did not produce a unique vocabulary")

    output.parent.mkdir(parents=True, exist_ok=True)
    selected_vocabulary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp")
    source_records_consumed = 0
    source_tokens_considered = 0
    retained_source_tokens = 0
    emitted_tokens = 0
    emitted_records = 0
    try:
        with temporary_output.open("w", encoding="ascii", newline="\n") as handle:
            for source_records_consumed, _raw, decoded in iter_raw_lines(source_raw):
                segment: list[str] = []
                for token in normalize_tokens(decoded):
                    source_tokens_considered += 1
                    if token in selected:
                        segment.append(token)
                        retained_source_tokens += 1
                        continue
                    if segment:
                        written_tokens, written_records = emit_segment(handle, segment, target_tokens - emitted_tokens)
                        emitted_tokens += written_tokens
                        emitted_records += written_records
                        segment.clear()
                        if emitted_tokens == target_tokens:
                            break
                if emitted_tokens == target_tokens:
                    break
                if segment:
                    written_tokens, written_records = emit_segment(handle, segment, target_tokens - emitted_tokens)
                    emitted_tokens += written_tokens
                    emitted_records += written_records
                if emitted_tokens == target_tokens:
                    break
    except OSError as error:
        raise ExperimentError(f"cannot write normalized corpus {output}: {error}") from error
    finally:
        if emitted_tokens != target_tokens and temporary_output.exists():
            temporary_output.unlink(missing_ok=True)
    require(emitted_tokens == target_tokens,
            f"selected real-source tokens ended at {emitted_tokens}, below requested {target_tokens}")
    os.replace(temporary_output, output)

    selected_text = "".join(f"{token}\t{frequency}\n" for token, frequency in ranked_vocabulary)
    atomic_text(selected_vocabulary_output, selected_text)
    corpus_identity = validate_generated_corpus(output, target_tokens)
    require(corpus_identity["physical_lf_records"] == emitted_records,
            "corpus record accounting differs from emitted output")
    return {
        "schema": CORPUS_SCHEMA,
        "source_raw": source_identity,
        "normalization": {
            "id": "tr-ascii-v1",
            "description": (
                "Turkish C/c, G/g, dotted/dotless I/i, O/o, S/s, and U/u characters are explicitly "
                "transliterated; Unicode is NFKD decomposed; remaining non-ASCII code points and combining "
                "marks are removed; A-Z is lowercase; the native token grammar retains A-Z a-z 0-9 _ apostrophe."
            ),
            "long_tokens_dropped": True,
        },
        "selection": {
            "method": "full-source frequency rank descending then ASCII lexical tie-break",
            "selected_vocabulary_size": vocabulary_size,
            "selected_vocabulary_file": selected_vocabulary_output.name,
            "selected_vocabulary_sha256": sha256_file(selected_vocabulary_output)["sha256"],
            "filtered_token_is_record_boundary": True,
            "contextless_singleton_segments_dropped": True,
            "minimum_tokens_per_output_record": 2,
            "maximum_tokens_per_output_record": TOKENS_PER_RECORD,
            "source_records_consumed_until_target": source_records_consumed,
            "source_normalised_tokens_considered_until_target": source_tokens_considered,
            "selected_source_tokens_seen_until_target": retained_source_tokens,
        },
        "output": corpus_identity,
    }


def run_child(command: Sequence[str], *, timeout_seconds: float, cwd: Path) -> ProcessResult:
    started = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ExperimentError(f"child exceeded the {timeout_seconds:g}-second timeout: {' '.join(command)}") from error
    except OSError as error:
        raise ExperimentError(f"cannot launch child {' '.join(command)}: {error}") from error
    stopped = time.perf_counter_ns()
    return ProcessResult(
        command=tuple(command),
        returncode=completed.returncode,
        elapsed_seconds=(stopped - started) / 1_000_000_000,
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def write_process_transcript(run_dir: Path, stem: str, result: ProcessResult) -> None:
    atomic_text(run_dir / f"{stem}.stdout.txt", result.stdout)
    atomic_text(run_dir / f"{stem}.stderr.txt", result.stderr)
    atomic_text(run_dir / f"{stem}.command.txt", " ".join(result.command) + "\n")


def require_success(result: ProcessResult, label: str) -> None:
    if result.returncode != 0:
        stderr = result.stderr.strip().replace("\n", " ")
        raise ExperimentError(f"{label} exited {result.returncode}: {stderr[:800]}")


def parse_training_metrics(stdout: str) -> dict[str, int]:
    marker = "phase: streaming HDC context learning"
    require(stdout.count(marker) == 1, "native train output must have exactly one streaming-learning phase")
    tail = stdout.split(marker, 1)[1]
    labels = (
        "completed epochs",
        "corpus lines",
        "raw tokens",
        "active tokens",
        "out-of-vocabulary tokens",
        "ordered context pairs",
        "prototype updates",
        "tokenizer truncations",
        "non-ASCII input bytes",
        "dropped overlong records",
        "resonant frozen-snapshot contexts",
        "resonant epoch snapshot bytes",
    )
    metrics: dict[str, int] = {}
    for label in labels:
        found = re.findall(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", tail, flags=re.MULTILINE)
        require(len(found) == 1, f"native train output must contain exactly one {label!r} metric")
        metrics[label] = int(found[0])
    return metrics


def parse_active_vocabulary(path: Path) -> dict[str, int]:
    active: dict[str, int] = {}
    try:
        for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
            fields = raw.split("\t")
            require(len(fields) == 2, f"active vocabulary line {line_number} is not one TSV pair")
            token, frequency_text = fields
            require(TOKEN_PATTERN.fullmatch(token) is not None, f"active vocabulary token is invalid: {token!r}")
            frequency = int(frequency_text)
            require(frequency > 0 and token not in active, f"active vocabulary entry is invalid or duplicate: {token!r}")
            active[token] = frequency
    except (OSError, UnicodeError, ValueError) as error:
        raise ExperimentError(f"cannot parse active vocabulary export {path}: {error}") from error
    require(active, "active vocabulary export is empty")
    return active


def select_semantic_queries(active: Mapping[str, int], count: int = 11) -> list[str]:
    selected: list[str] = []
    for token in SEMANTIC_PRIORITY:
        if token in active and token not in selected:
            selected.append(token)
            if len(selected) == count:
                return selected
    fallback = sorted(active, key=lambda token: (-active[token], token))
    for token in fallback:
        if token in selected or token in STOP_TOKENS or len(token) < 4 or token.isdigit():
            continue
        selected.append(token)
        if len(selected) == count:
            return selected
    raise ExperimentError("fewer than eleven meaningful active tokens are available for semantic probes")


def parse_semantic_results(stdout: str, expected_queries: Sequence[str]) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"^query:\s*(?P<query>[a-z0-9_']+)\s*$\n"
        r"^\s*nearest semantic neighbour:\s*(?P<neighbour>[a-z0-9_']+)\s*$\n"
        r"^\s*bipolar similarity:\s*(?P<similarity>-?\d+(?:\.\d+)?)\s*$",
        flags=re.MULTILINE,
    )
    matches = list(pattern.finditer(stdout))
    require(len(matches) == len(expected_queries),
            f"semantic inspect yielded {len(matches)} complete results, expected {len(expected_queries)}")
    results: list[dict[str, Any]] = []
    for expected, match in zip(expected_queries, matches, strict=True):
        query = match.group("query")
        require(query == expected, f"semantic inspect order/query mismatch: expected {expected!r}, got {query!r}")
        results.append(
            {
                "query": query,
                "nearest_semantic_neighbour": match.group("neighbour"),
                "bipolar_similarity": float(match.group("similarity")),
            }
        )
    return results


def load_faq_candidates(faq_file: Path, active: Mapping[str, int], required: int = 11) -> list[dict[str, Any]]:
    """Select reproducible, fully or nearly-covered external FAQ prompts.

    The TSV answer remains a normalized literal from the independent FAQ data.
    A fallback in-vocabulary prompt projection is only used if fewer than
    ``required`` fully covered questions exist; the report labels it explicitly
    so a compact HDC query is never confused with an untouched source question.
    """
    try:
        lines = faq_file.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as error:
        raise ExperimentError(f"cannot read FAQ dataset {faq_file}: {error}") from error

    parsed: list[dict[str, Any]] = []
    for row_number, line in enumerate(lines, start=1):
        pieces = line.split("|", 2)
        if len(pieces) != 3:
            continue
        source_url, source_question, source_answer = (piece.strip() for piece in pieces)
        question_tokens = normalize_tokens(source_question)
        if not source_url or len(question_tokens) < 3:
            continue
        known_tokens = [token for token in question_tokens if token in active]
        response = normalize_display_text(source_answer)
        if len(known_tokens) < 3 or not response or len(response) > 2_048:
            continue
        parsed.append(
            {
                "source_row": row_number,
                "source_url": source_url,
                "source_question_ascii": normalize_display_text(source_question),
                "source_answer_ascii": response,
                "question_tokens": question_tokens,
                "known_tokens": known_tokens,
                "coverage": len(known_tokens) / len(question_tokens),
            }
        )

    selected: list[dict[str, Any]] = []
    used_signatures: set[tuple[str, ...]] = set()
    used_rows: set[int] = set()
    # First preserve a source question's full normalized token sequence.  The
    # subsequent thresholds are deterministic fallbacks, not relevance tuning.
    for threshold in (1.0, 0.85, 0.70):
        for candidate in parsed:
            if len(selected) >= required:
                break
            if candidate["source_row"] in used_rows or candidate["coverage"] < threshold:
                continue
            if threshold == 1.0:
                prompt_tokens = candidate["question_tokens"]
                projection = "full-normalized-source-question"
            else:
                prompt_tokens = candidate["known_tokens"]
                projection = f"in-vocabulary-projection-coverage-at-least-{threshold:.2f}"
            if len(prompt_tokens) > 48:
                continue
            prompt = " ".join(prompt_tokens)
            if len(prompt) > 1_024:
                continue
            signature = tuple(sorted(prompt_tokens))
            if signature in used_signatures:
                continue
            selected.append(
                {
                    "ordinal": len(selected) + 1,
                    "prompt": prompt,
                    "prompt_projection": projection,
                    "source_row": candidate["source_row"],
                    "source_url": candidate["source_url"],
                    "source_question_ascii": candidate["source_question_ascii"],
                    "response": candidate["source_answer_ascii"],
                    "source_prompt_token_count": len(candidate["question_tokens"]),
                    "active_prompt_token_count": len(prompt_tokens),
                    "active_prompt_coverage": candidate["coverage"],
                }
            )
            used_rows.add(candidate["source_row"])
            used_signatures.add(signature)
        if len(selected) >= required:
            break
    require(len(selected) == required,
            f"independent FAQ source supplied only {len(selected)} eligible in-vocabulary dialogue entries")
    return selected


def parse_chat_result(stdout: str, expected_prompt: str, expected_response: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    require(fields.get("grounded") == "true", f"dialogue prompt {expected_prompt!r} abstained")
    require(fields.get("matched prompt") == expected_prompt,
            f"dialogue matched wrong TSV prompt for {expected_prompt!r}: {fields.get('matched prompt')!r}")
    require(fields.get("answer") == expected_response,
            f"dialogue returned non-verbatim response for {expected_prompt!r}")
    try:
        return {
            "query_known_tokens": int(fields["query known tokens"]),
            "query_ignored_tokens": int(fields["query ignored tokens"]),
            "candidates_examined": int(fields["candidates examined"]),
            "best_bipolar_similarity": float(fields["best bipolar similarity"]),
            "runner_up_bipolar_similarity": float(fields["runner-up bipolar similarity"]),
            "grounding_margin": float(fields["grounding margin"]),
            "matched_prompt": fields["matched prompt"],
            "answer": fields["answer"],
            "grounded": True,
        }
    except (KeyError, ValueError) as error:
        raise ExperimentError(f"dialogue output is missing a numeric grounding field: {error}") from error


def copy_evidence_files(run_dir: Path, files: Sequence[Path], report: Path, checkpoint: Path) -> None:
    evidence = run_dir / "evidence"
    evidence.mkdir(exist_ok=True)
    for source in (*files, report):
        require(source.is_file(), f"expected evidence file is missing: {source}")
        shutil.copy2(source, evidence / source.name)
    checkpoint_identity = sha256_file(checkpoint)
    atomic_text(evidence / "checkpoint.sha256", f"{checkpoint_identity['sha256']}  {checkpoint.name}\n")


def build_provenance(arguments: argparse.Namespace, raw_identity: Mapping[str, Any]) -> dict[str, Any]:
    retrieved_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    source_repo = arguments.source_repo.resolve()
    faq_repo = arguments.faq_repo.resolve()
    archive = arguments.source_archive.resolve()
    faq_file = faq_repo / "dataset_v1.txt"
    require(source_repo.is_dir(), f"source metadata repository is unavailable: {source_repo}")
    require(faq_repo.is_dir(), f"FAQ source repository is unavailable: {faq_repo}")
    require(archive.is_file(), f"downloaded source archive is unavailable: {archive}")
    require(faq_file.is_file(), f"FAQ data file is unavailable: {faq_file}")
    return {
        "Turkish_Wikipedia_training_source": {
            "retrieved_at_utc": retrieved_at_utc,
            "dataset_project_url": arguments.dataset_url,
            "dataset_project_revision": git_revision(source_repo),
            "declared_license": arguments.dataset_license,
            "license_file": license_identity(source_repo),
            "archive_url": arguments.source_archive_url,
            "archive": sha256_file(archive),
            "raw_training_split": dict(raw_identity),
        },
        "separate_grounded_dialogue_source": {
            "retrieved_at_utc": retrieved_at_utc,
            "dataset_project_url": arguments.faq_url,
            "dataset_project_revision": git_revision(faq_repo),
            "declared_license": arguments.faq_license,
            "license_file": license_identity(faq_repo),
            "dataset_file": sha256_file(faq_file),
            "format": "source-webpage-url|question|answer; selected answers are ASCII-normalized literal TSV fields",
        },
    }


def run_experiment(arguments: argparse.Namespace) -> dict[str, Any]:
    run_dir = arguments.run_dir.resolve()
    require(not run_dir.exists() or not any(run_dir.iterdir()), f"run directory must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    require(arguments.pretrain_bin.is_file() and os.access(arguments.pretrain_bin, os.X_OK),
            f"pretraining executable is not usable: {arguments.pretrain_bin}")
    require(arguments.chat_bin.is_file() and os.access(arguments.chat_bin, os.X_OK),
            f"chat executable is not usable: {arguments.chat_bin}")
    require(arguments.source_root.is_dir(), f"extracted Turkish Wikipedia source root is unavailable: {arguments.source_root}")
    require(arguments.epochs >= 2, "this RESONANT evidence run requires at least two epochs")
    require(arguments.omp_threads > 0, "OMP thread count must be positive")

    candidates = sorted(arguments.source_root.rglob("*train*.raw"))
    require(len(candidates) == 1,
            f"expected exactly one raw training split matching *train*.raw, found {len(candidates)}")
    source_raw = candidates[0]
    corpus = run_dir / "turkish_wikipedia_9m_native_ascii.txt"
    selected_vocabulary = run_dir / "selected-source-vocabulary.tsv"
    corpus_manifest = prepare_corpus(
        source_raw,
        corpus,
        selected_vocabulary,
        target_tokens=arguments.target_tokens,
        vocabulary_size=arguments.vocabulary_size,
    )
    provenance = build_provenance(arguments, corpus_manifest["source_raw"])
    provenance_path = run_dir / "provenance.json"
    corpus_manifest_path = run_dir / "corpus-manifest.json"
    atomic_json(provenance_path, provenance)
    atomic_json(corpus_manifest_path, corpus_manifest)

    checkpoint = run_dir / "turkish_wikipedia_9m_resonant.vgai"
    training_command = (
        str(arguments.pretrain_bin.resolve()),
        "train",
        str(corpus),
        str(checkpoint),
        "--max-vocab",
        str(arguments.vocabulary_size),
        "--min-frequency",
        "1",
        "--window",
        "2",
        "--position-shift",
        "41",
        "--context-mode",
        "resonant",
        "--epochs",
        str(arguments.epochs),
        "--strict",
        "--parallel-context-build",
        "--parallel-min-contexts",
        "32",
    )
    environment_threads_before = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(arguments.omp_threads)
    try:
        training = run_child(training_command, timeout_seconds=arguments.train_timeout_seconds, cwd=Path.cwd())
    finally:
        if environment_threads_before is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = environment_threads_before
    write_process_transcript(run_dir, "native-training", training)
    require_success(training, "native Fortran training")
    require(checkpoint.is_file(), "native train command reported success without a checkpoint")
    metrics = parse_training_metrics(training.stdout)
    expected_training_tokens = arguments.target_tokens * arguments.epochs
    expected_training_records = int(corpus_manifest["output"]["physical_lf_records"]) * arguments.epochs
    require(metrics["completed epochs"] == arguments.epochs, "native train completed an unexpected epoch count")
    require(metrics["corpus lines"] == expected_training_records, "native train record count differs from corpus manifest")
    require(metrics["raw tokens"] == expected_training_tokens, "native train raw-token count differs from corpus manifest")
    require(metrics["active tokens"] == expected_training_tokens,
            "some native corpus tokens were not actual active training updates")
    require(metrics["out-of-vocabulary tokens"] == 0, "selected native corpus emitted out-of-vocabulary tokens")
    require(metrics["prototype updates"] == expected_training_tokens, "not every active native token updated a prototype")
    require(metrics["tokenizer truncations"] == 0 and metrics["non-ASCII input bytes"] == 0 and
            metrics["dropped overlong records"] == 0, "strict native input contract telemetry is nonzero")
    require(metrics["resonant frozen-snapshot contexts"] > 0, "RESONANT feedback did not construct any snapshot contexts")

    active_vocabulary = run_dir / "active-vocabulary.tsv"
    export_command = (
        str(arguments.pretrain_bin.resolve()),
        "inspect",
        str(checkpoint),
        "--active-vocabulary-file",
        str(active_vocabulary),
    )
    active_export = run_child(export_command, timeout_seconds=arguments.inspect_timeout_seconds, cwd=Path.cwd())
    write_process_transcript(run_dir, "active-vocabulary-export", active_export)
    require_success(active_export, "native active-vocabulary export")
    active = parse_active_vocabulary(active_vocabulary)
    require(len(active) <= arguments.vocabulary_size, "active vocabulary exceeds configured capacity")

    semantic_queries = select_semantic_queries(active)
    semantic_query_file = run_dir / "semantic-queries.txt"
    atomic_text(semantic_query_file, "\n".join(semantic_queries) + "\n")
    semantic_command = (
        str(arguments.pretrain_bin.resolve()),
        "inspect",
        str(checkpoint),
        "--query-file",
        str(semantic_query_file),
    )
    semantic = run_child(semantic_command, timeout_seconds=arguments.inspect_timeout_seconds, cwd=Path.cwd())
    write_process_transcript(run_dir, "semantic-retrieval", semantic)
    require_success(semantic, "native semantic retrieval")
    semantic_results = parse_semantic_results(semantic.stdout, semantic_queries)
    semantic_results_path = run_dir / "semantic-retrieval-results.json"
    atomic_json(semantic_results_path, {"queries": semantic_results})

    faq_file = arguments.faq_repo.resolve() / "dataset_v1.txt"
    dialogue_entries = load_faq_candidates(faq_file, active)
    knowledge_tsv = run_dir / "grounded-dialogue-knowledge.tsv"
    atomic_text(knowledge_tsv, "".join(f"{entry['prompt']}\t{entry['response']}\n" for entry in dialogue_entries))
    knowledge_provenance_path = run_dir / "grounded-dialogue-knowledge-provenance.json"
    atomic_json(
        knowledge_provenance_path,
        {
            "schema": "vegai-grounded-dialogue-knowledge-selection/v1",
            "knowledge_tsv": sha256_file(knowledge_tsv),
            "source": provenance["separate_grounded_dialogue_source"],
            "selection": dialogue_entries,
            "retrieval_contract": "Responses are loaded from this TSV and returned verbatim only; vegai_chat is not a free-form text generator.",
        },
    )

    dialogue_results: list[dict[str, Any]] = []
    dialogue_output_paths: list[Path] = []
    for entry in dialogue_entries:
        dialogue_command = (
            str(arguments.chat_bin.resolve()),
            "ask",
            str(checkpoint),
            str(knowledge_tsv),
            "--prompt",
            str(entry["prompt"]),
            "--max-entries",
            str(len(dialogue_entries)),
            "--history-turns",
            "0",
            "--min-known-tokens",
            "1",
            "--min-similarity",
            "0.95",
            "--min-margin",
            "0.01",
            "--require-all-known",
        )
        dialogue = run_child(dialogue_command, timeout_seconds=arguments.inspect_timeout_seconds, cwd=Path.cwd())
        stem = f"grounded-dialogue-{entry['ordinal']:02d}"
        write_process_transcript(run_dir, stem, dialogue)
        require_success(dialogue, f"grounded dialogue {entry['ordinal']}")
        parsed_dialogue = parse_chat_result(dialogue.stdout, str(entry["prompt"]), str(entry["response"]))
        dialogue_results.append({**entry, **parsed_dialogue})
        dialogue_output_paths.extend(
            [run_dir / f"{stem}.stdout.txt", run_dir / f"{stem}.stderr.txt", run_dir / f"{stem}.command.txt"]
        )

    dialogue_results_path = run_dir / "grounded-dialogue-results.tsv"
    dialogue_rows = [
        "ordinal\tprompt\tmatched_prompt\tbest_bipolar_similarity\trunner_up_bipolar_similarity\tgrounding_margin\tgrounded\tanswer\tsource_url\tsource_row\n"
    ]
    for result in dialogue_results:
        dialogue_rows.append(
            f"{result['ordinal']}\t{result['prompt']}\t{result['matched_prompt']}\t"
            f"{result['best_bipolar_similarity']:.5f}\t{result['runner_up_bipolar_similarity']:.5f}\t"
            f"{result['grounding_margin']:.5f}\ttrue\t{result['answer']}\t{result['source_url']}\t"
            f"{result['source_row']}\n"
        )
    atomic_text(dialogue_results_path, "".join(dialogue_rows))

    checkpoint_identity = sha256_file(checkpoint)
    report = {
        "schema": SCHEMA,
        "status": "succeeded",
        "scope": {
            "native_model": "Fortran HDC/VSA distributional checkpoint with RESONANT epoch-frozen semantic feedback",
            "not_claimed": [
                "free-form LLM generation",
                "held-out natural-language accuracy",
                "factual correctness beyond the explicitly cited TSV source",
                "general intelligence or general language understanding",
            ],
        },
        "provenance": provenance,
        "corpus": corpus_manifest,
        "training": {
            "command": list(training.command),
            "omp_num_threads": arguments.omp_threads,
            "train_process_wall_seconds": training.elapsed_seconds,
            "raw_training_tokens_per_second": expected_training_tokens / training.elapsed_seconds,
            "metrics": metrics,
            "timing_scope": "Only the one native vegai_pretrain train child process; archive retrieval, decompression, normalization, vocabulary selection, compilation, inspection, TSV construction, chat retrieval, hashing, and CI overhead are excluded.",
        },
        "checkpoint": checkpoint_identity,
        "active_vocabulary": {
            "file": sha256_file(active_vocabulary),
            "active_entries": len(active),
        },
        "semantic_retrieval": {
            "selection": "Priority list of named Turkish probe tokens active in this exact checkpoint, then deterministic high-frequency non-stopword fallback.",
            "results": semantic_results,
            "interpretation": "These are checkpoint self-excluded nearest-neighbour similarities, not generated responses or held-out semantic accuracy.",
        },
        "grounded_dialogue": {
            "knowledge_tsv": sha256_file(knowledge_tsv),
            "selection_provenance": sha256_file(knowledge_provenance_path),
            "policy": {
                "min_similarity": 0.95,
                "min_margin": 0.01,
                "require_all_prompt_tokens_known": True,
                "history_turns": 0,
            },
            "results": dialogue_results,
            "interpretation": "Each answer is the exact ASCII-normalized response field supplied by the separately sourced TSV; the Fortran chat executable retrieves or abstains and does not generate prose.",
        },
    }
    report_path = run_dir / "experiment-report.json"
    atomic_json(report_path, report)

    evidence_files = [
        provenance_path,
        corpus_manifest_path,
        selected_vocabulary,
        active_vocabulary,
        semantic_query_file,
        semantic_results_path,
        knowledge_tsv,
        knowledge_provenance_path,
        dialogue_results_path,
        run_dir / "native-training.command.txt",
        run_dir / "native-training.stdout.txt",
        run_dir / "native-training.stderr.txt",
        run_dir / "active-vocabulary-export.command.txt",
        run_dir / "active-vocabulary-export.stdout.txt",
        run_dir / "active-vocabulary-export.stderr.txt",
        run_dir / "semantic-retrieval.command.txt",
        run_dir / "semantic-retrieval.stdout.txt",
        run_dir / "semantic-retrieval.stderr.txt",
        *dialogue_output_paths,
    ]
    copy_evidence_files(run_dir, evidence_files, report_path, checkpoint)
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", type=Path, required=True, help="compiled native vegai_pretrain executable")
    parser.add_argument("--chat-bin", type=Path, required=True, help="compiled native vegai_chat executable")
    parser.add_argument("--source-root", type=Path, required=True, help="extracted trwiki-67 archive root")
    parser.add_argument("--source-repo", type=Path, required=True, help="checked-out tdd-ai/trwiki-67 metadata repository")
    parser.add_argument("--source-archive", type=Path, required=True, help="downloaded trwiki-67 archive")
    parser.add_argument("--faq-repo", type=Path, required=True, help="checked-out Aliumutcan/Turkish_NLP_FAQ_DATASET")
    parser.add_argument("--run-dir", type=Path, required=True, help="empty mutable experiment directory")
    parser.add_argument("--dataset-url", default="https://github.com/tdd-ai/trwiki-67")
    parser.add_argument("--dataset-license", default="CC-BY-SA-4.0")
    parser.add_argument("--source-archive-url", default="https://zenodo.org/record/5213891/files/trwiki-67.tar.bz2?download=1")
    parser.add_argument("--faq-url", default="https://github.com/Aliumutcan/Turkish_NLP_FAQ_DATASET")
    parser.add_argument("--faq-license", default="MIT")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS_DEFAULT)
    parser.add_argument("--vocabulary-size", type=int, default=VOCABULARY_SIZE_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--omp-threads", type=int, default=2)
    parser.add_argument("--train-timeout-seconds", type=float, default=3_600.0)
    parser.add_argument("--inspect-timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        require(arguments.train_timeout_seconds > 0 and arguments.inspect_timeout_seconds > 0,
                "all child timeouts must be positive")
        report = run_experiment(arguments)
    except ExperimentError as error:
        run_dir = arguments.run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(run_dir / "experiment-failure.json", {"schema": SCHEMA, "status": "failed", "error": str(error)})
        print(f"Turkish Wikipedia 9M experiment: FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "report": str(arguments.run_dir / "experiment-report.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
