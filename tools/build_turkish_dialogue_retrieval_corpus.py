#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an auditable Turkish dialogue-retrieval package from pinned JSONL.

This is *data orchestration*, not the Vegai model path: the resulting training
corpus is consumed by the native Fortran ``vegai_pretrain`` executable and the
UTF-8 TSV knowledge base by native ``vegai_chat``.  The tool never downloads a
corpus, guesses a licence, or mixes source splits.  An operator supplies local,
immutable train/validation/test JSONL snapshots and explicit provenance.

The generated package contains:

* ``knowledge_train.tsv``: UTF-8 user-prompt -> assistant-response retrieval
  rows from the training split only;
* ``pretrain_train_ascii.txt``: deterministic ASCII-projected dialogue turns
  suitable for the existing native HDC tokenizer;
* ``train_exact_retrieval_contract.jsonl``: non-held-out literal key lookup
  integrity cases (not a generalization metric), plus a deterministic bounded
  hash-ranked evidence sample;
* ``validation_source_disjoint_abstention.jsonl`` and
  ``test_source_disjoint_abstention.jsonl``: frozen source-disjoint negative
  retrieval sets.  They measure false grounding/abstention behaviour only;
  they do not mislabel unseen responses as known correct answers;
* corresponding strict three-column ``.tsv`` projections: bounded native
  Fortran batch-evaluation inputs that preserve an empty expected-prompt field
  for expected abstention without exposing a JSON parser to model runtime;
* ``manifest.json``: source/output SHA-256 identities, counts, split isolation,
  transformation version, licence and attribution supplied by the operator.

It deliberately does not call the output a natural-language-generation corpus.
A response returned by Vegai remains literal retrieved TSV data; a causal model
would need a separate native training/evaluation contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence


PACKAGE_SCHEMA = "vegai-turkish-dialogue-retrieval-package/v1"
EVALUATION_SCHEMA = "vegai-dialogue-evaluation-case/v1"
TRANSFORMATION = "turkish-dialogue-jsonl-to-native-retrieval/v2"
MAX_SOURCE_LINE_BYTES = 64 * 1024
MAX_DIALOGUE_TEXT_BYTES = 1024
MAX_DIALOGUE_TURNS = 64
DEFAULT_EVIDENCE_SAMPLE_CASES = 128

# Match the explicit native Turkish boundary projection. Python is used here
# only to prepare an immutable corpus; native Fortran validates it again at
# index-load/query time.
TURKISH_ASCII_TRANSLITERATION = str.maketrans(
    {
        "Â": "A", "â": "a", "Ç": "C", "ç": "c", "Ğ": "G", "ğ": "g", "İ": "I", "ı": "i",
        "Î": "I", "î": "i", "Ö": "O", "ö": "o", "Ş": "S", "ş": "s", "Ș": "S", "ș": "s",
        "Û": "U", "û": "u", "Ü": "U", "ü": "u",
        "\u00a0": " ", "\u2018": "'", "\u2019": "'", "\u2013": "-", "\u2014": "-",
    }
)
ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]+")
REVISION_RE = re.compile(r"[0-9a-f]{40}")


class DialogueCorpusError(RuntimeError):
    """Expected provenance, schema, content, or publication failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DialogueCorpusError(message)


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise DialogueCorpusError(f"cannot read {path}: {error}") from error
    return digest.hexdigest(), byte_count


def file_identity(path: Path, label: str) -> dict[str, object]:
    resolved = path.resolve()
    require(resolved.is_file(), f"{label} is not a regular file: {resolved}")
    digest, byte_count = sha256_file(resolved)
    return {"path": str(resolved), "sha256": digest, "bytes": byte_count}


def fsync_directory(directory: Path) -> None:
    """Best-effort directory durability after a successful package rename."""
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


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_dialogue_text(value: object, field: str) -> str:
    require(isinstance(value, str) and value, f"{field} must be a nonempty string")
    require("\r" not in value and "\n" not in value and "\t" not in value,
            f"{field} must not contain CR/LF/TAB")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise DialogueCorpusError(f"{field} is not strict UTF-8") from error
    require(len(encoded) <= MAX_DIALOGUE_TEXT_BYTES,
            f"{field} exceeds native dialogue bound {MAX_DIALOGUE_TEXT_BYTES} UTF-8 bytes")
    require(all(ord(character) >= 32 and not 127 <= ord(character) <= 159 for character in value),
            f"{field} contains a control character")
    return value


def canonicalize_for_native(text: str) -> str:
    """Return the tokenizer-visible ASCII projection, not a display rewrite."""
    translated = text.translate(TURKISH_ASCII_TRANSLITERATION).lower()
    return " ".join(ASCII_TOKEN_RE.findall(translated))


def iter_bounded_jsonl(path: Path, split_name: str) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Strictly stream UTF-8/LF JSONL without unbounded hostile line reads."""
    resolved = path.resolve()
    require(resolved.is_file(), f"{split_name} JSONL is not a regular file: {resolved}")
    try:
        with resolved.open("rb") as handle:
            number = 0
            while True:
                raw = handle.readline(MAX_SOURCE_LINE_BYTES + 2)
                if not raw:
                    return
                number += 1
                require(raw.endswith(b"\n"), f"{split_name} line {number} must end with LF")
                require(len(raw) - 1 <= MAX_SOURCE_LINE_BYTES,
                        f"{split_name} line {number} exceeds {MAX_SOURCE_LINE_BYTES} bytes")
                require(not raw[:-1].endswith(b"\r"), f"{split_name} line {number} uses CRLF")
                try:
                    parsed = json.loads(raw[:-1].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise DialogueCorpusError(f"{split_name} line {number} is not strict UTF-8 JSON: {error}") from error
                require(isinstance(parsed, Mapping), f"{split_name} line {number} must be a JSON object")
                yield number, parsed
    except OSError as error:
        raise DialogueCorpusError(f"cannot read {split_name} JSONL {resolved}: {error}") from error


def parse_conversation(record: Mapping[str, Any], split_name: str, line_number: int) -> tuple[str, list[tuple[str, str]]]:
    context = f"{split_name} line {line_number}"
    conversation_id = record.get("conversation_id")
    require(isinstance(conversation_id, str) and conversation_id and "\t" not in conversation_id and "\n" not in conversation_id,
            f"{context}.conversation_id must be a nonempty single-line string")
    try:
        encoded_id = conversation_id.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise DialogueCorpusError(f"{context}.conversation_id must be printable ASCII") from error
    require(len(encoded_id) <= 256 and all(32 <= byte <= 126 for byte in encoded_id),
            f"{context}.conversation_id must be printable ASCII and <= 256 bytes")
    declared_split = record.get("split")
    require(declared_split == split_name,
            f"{context}.split={declared_split!r} does not match selected split {split_name!r}")
    messages = record.get("messages")
    require(isinstance(messages, list) and 2 <= len(messages) <= MAX_DIALOGUE_TURNS and len(messages) % 2 == 0,
            f"{context}.messages must be an even list of 2..{MAX_DIALOGUE_TURNS} entries")
    pairs: list[tuple[str, str]] = []
    for position, message in enumerate(messages):
        require(isinstance(message, Mapping), f"{context}.messages[{position}] must be an object")
        expected_role = "user" if position % 2 == 0 else "assistant"
        require(message.get("role") == expected_role,
                f"{context}.messages[{position}].role must be {expected_role!r}")
        content = validate_dialogue_text(message.get("content"), f"{context}.messages[{position}].content")
        if position % 2 == 0:
            prompt = content
        else:
            pairs.append((prompt, content))
    return conversation_id, pairs


def load_split(path: Path, split_name: str, globally_seen_ids: set[str]) -> list[tuple[str, int, str, str]]:
    """Load one source split as (conversation id, turn number, prompt, response)."""
    pairs: list[tuple[str, int, str, str]] = []
    records = 0
    for number, record in iter_bounded_jsonl(path, split_name):
        conversation_id, conversation_pairs = parse_conversation(record, split_name, number)
        require(conversation_id not in globally_seen_ids,
                f"conversation_id crosses source splits or repeats: {conversation_id!r}")
        globally_seen_ids.add(conversation_id)
        records += 1
        for turn_number, (prompt, response) in enumerate(conversation_pairs, start=1):
            pairs.append((conversation_id, turn_number, prompt, response))
    require(records > 0 and pairs, f"{split_name} source split has no usable conversations/pairs")
    return pairs


def render_knowledge_train(pairs: Sequence[tuple[str, int, str, str]]) -> tuple[bytes, int, int]:
    """Deduplicate canonical prompt keys without selecting a response by code."""
    seen: dict[str, tuple[str, str]] = {}
    lines: list[str] = []
    duplicate_identical = 0
    for conversation_id, turn_number, prompt, response in pairs:
        canonical = canonicalize_for_native(prompt)
        require(canonical, f"train {conversation_id} turn {turn_number} becomes empty under native ASCII projection")
        existing = seen.get(canonical)
        if existing is not None:
            previous_prompt, previous_response = existing
            if previous_response != response:
                raise DialogueCorpusError(
                    "train canonical prompt has conflicting literal responses: "
                    f"{conversation_id} turn {turn_number}; existing prompt={previous_prompt!r}"
                )
            duplicate_identical += 1
            continue
        seen[canonical] = (prompt, response)
        lines.append(f"{prompt}\t{response}")
    require(lines, "train source produces no unique knowledge rows")
    return ("\n".join(lines) + "\n").encode("utf-8"), len(lines), duplicate_identical


def render_pretraining_ascii(pairs: Sequence[tuple[str, int, str, str]]) -> tuple[bytes, int, int]:
    """Emit bounded ASCII records from both speakers, preserving no raw response text here."""
    records: list[str] = []
    skipped = 0
    for _, _, prompt, response in pairs:
        for utterance in (prompt, response):
            canonical = canonicalize_for_native(utterance)
            tokens = canonical.split()
            if len(tokens) < 2:
                skipped += 1
                continue
            # The source dialog line bound plus collapse-only projection makes
            # this safely below the native 8,192-byte record bound. Retain the
            # native token maximum as an explicit package contract anyway.
            if len(tokens) > 128:
                tokens = tokens[:128]
            record = " ".join(tokens)
            require(record.isascii() and 0 < len(record.encode("ascii")) <= 8192,
                    "internal error: ASCII pretraining projection violates native byte bound")
            records.append(record)
    require(records, "train source produces no pretraining records")
    return ("\n".join(records) + "\n").encode("ascii"), len(records), skipped


def render_train_exact_retrieval_evaluation(knowledge: bytes) -> bytes:
    """Render a non-held-out package-integrity suite for literal train keys.

    Its only purpose is to prove that the published TSV and checkpoint can
    retrieve the operator-supplied key exactly. It is intentionally named
    ``train_exact`` rather than treated as a generalization metric.
    """
    lines: list[str] = []
    for number, line in enumerate(knowledge.decode("utf-8").splitlines(), start=1):
        prompt, separator, _ = line.partition("\t")
        require(bool(separator) and prompt, "internal error: rendered knowledge row lacks prompt TAB response")
        case = {
            "schema_version": EVALUATION_SCHEMA,
            "id": f"train-exact:{number}",
            "prompt": prompt,
            "expected_prompt": prompt,
        }
        lines.append(json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    require(lines, "internal error: rendered knowledge has no exact retrieval cases")
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_abstention_evaluation(split_name: str, pairs: Sequence[tuple[str, int, str, str]]) -> bytes:
    """Render source-disjoint prompts with explicit null expected grounding."""
    lines: list[str] = []
    seen_prompts: set[str] = set()
    for conversation_id, turn_number, prompt, _ in pairs:
        # Exact duplicate held-out prompts add no test information and make
        # case count unstable across a source's harmless repeated renderings.
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        case = {
            "schema_version": EVALUATION_SCHEMA,
            "id": f"{split_name}:{conversation_id}:turn:{turn_number}",
            "prompt": prompt,
            "expected_prompt": None,
        }
        lines.append(json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    require(lines, f"{split_name} source produces no abstention evaluation cases")
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_native_evaluation_tsv(evaluation: bytes) -> bytes:
    """Render strict native ``id<TAB>prompt<TAB>expected-prompt`` cases.

    JSONL remains the human/audit-facing contract. This bounded TSV projection
    allows the all-Fortran batch evaluator to load the native checkpoint and
    index once rather than spawning a chat process for every query. An empty
    third column denotes the explicit expected-abstention label. The builder's
    prior source validation rejects control characters, so no user text can
    introduce an extra TSV field or physical record.
    """
    rows: list[str] = []
    for raw_line in evaluation.decode("utf-8").splitlines():
        case = json.loads(raw_line)
        case_id = case["id"]
        prompt = case["prompt"]
        expected = case["expected_prompt"]
        require(isinstance(case_id, str) and isinstance(prompt, str),
                "internal error: evaluation case is missing string id/prompt")
        require(expected is None or isinstance(expected, str),
                "internal error: evaluation expected_prompt must be string or null")
        require("\t" not in case_id and "\t" not in prompt and "\n" not in case_id and "\n" not in prompt,
                "internal error: evaluation TSV projection has a delimiter in id/prompt")
        if expected is not None:
            require("\t" not in expected and "\n" not in expected,
                    "internal error: evaluation TSV projection has a delimiter in expected prompt")
        rows.append(f"{case_id}\t{prompt}\t{expected or ''}")
    require(rows, "internal error: cannot render empty native evaluation TSV")
    return ("\n".join(rows) + "\n").encode("utf-8")


def deterministic_evidence_sample(evaluation: bytes, maximum_cases: int) -> bytes:
    """Select a fixed, source-order-independent bounded evidence subset.

    ``dialogue_evaluation.py`` deliberately rejects a file larger than its
    configured maximum rather than silently measuring a prefix. Hash-ranking
    records gives a reproducible mixed sample while the full frozen set remains
    in the package for a later exhaustive/native batch evaluation.
    """
    require(maximum_cases > 0, "evidence sample case limit must be positive")
    lines = evaluation.decode("utf-8").splitlines()
    require(lines, "internal error: cannot sample an empty evaluation set")
    ranked = sorted(
        ((hashlib.sha256(line.encode("utf-8")).digest(), line) for line in lines),
        key=lambda item: item[0],
    )
    selected = [line for _, line in ranked[:maximum_cases]]
    return ("\n".join(selected) + "\n").encode("utf-8")


def output_identity(path: Path) -> dict[str, object]:
    digest, byte_count = sha256_file(path)
    return {"path": path.name, "sha256": digest, "bytes": byte_count}


def build_package(arguments: argparse.Namespace) -> dict[str, object]:
    output = arguments.output_dir.resolve()
    require(not output.exists(), f"output directory already exists; refuse to overwrite immutable package: {output}")
    require(REVISION_RE.fullmatch(arguments.source_revision) is not None,
            "--source-revision must be a full lowercase 40-hex Git commit")
    require(arguments.source_repository.startswith("https://github.com/"),
            "--source-repository must be an immutable GitHub repository URL")
    require(bool(arguments.source_license.strip()), "--source-license must not be blank")
    require(bool(arguments.attribution.strip()), "--attribution must not be blank")
    require(bool(arguments.source_nature.strip()), "--source-nature must not be blank")
    require(arguments.evidence_sample_cases > 0, "--evidence-sample-cases must be positive")

    source_identities = {
        "train": file_identity(arguments.train_jsonl, "train JSONL"),
        "validation": file_identity(arguments.validation_jsonl, "validation JSONL"),
        "test": file_identity(arguments.test_jsonl, "test JSONL"),
    }
    global_ids: set[str] = set()
    train_pairs = load_split(arguments.train_jsonl, "train", global_ids)
    validation_pairs = load_split(arguments.validation_jsonl, "validation", global_ids)
    test_pairs = load_split(arguments.test_jsonl, "test", global_ids)

    knowledge, unique_rows, duplicates = render_knowledge_train(train_pairs)
    pretraining, pretraining_records, skipped_records = render_pretraining_ascii(train_pairs)
    train_exact_evaluation = render_train_exact_retrieval_evaluation(knowledge)
    validation_evaluation = render_abstention_evaluation("validation", validation_pairs)
    test_evaluation = render_abstention_evaluation("test", test_pairs)
    train_exact_sample = deterministic_evidence_sample(train_exact_evaluation, arguments.evidence_sample_cases)
    test_abstention_sample = deterministic_evidence_sample(test_evaluation, arguments.evidence_sample_cases)
    train_exact_native_evaluation = render_native_evaluation_tsv(train_exact_evaluation)
    validation_native_evaluation = render_native_evaluation_tsv(validation_evaluation)
    test_native_evaluation = render_native_evaluation_tsv(test_evaluation)

    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=staging_parent))
    try:
        knowledge_path = staging / "knowledge_train.tsv"
        pretraining_path = staging / "pretrain_train_ascii.txt"
        train_exact_path = staging / "train_exact_retrieval_contract.jsonl"
        train_exact_sample_path = staging / "train_exact_retrieval_evidence_sample.jsonl"
        train_exact_native_path = staging / "train_exact_retrieval_contract.tsv"
        validation_path = staging / "validation_source_disjoint_abstention.jsonl"
        validation_native_path = staging / "validation_source_disjoint_abstention.tsv"
        test_path = staging / "test_source_disjoint_abstention.jsonl"
        test_native_path = staging / "test_source_disjoint_abstention.tsv"
        test_sample_path = staging / "test_source_disjoint_abstention_evidence_sample.jsonl"
        write_bytes_atomic(knowledge_path, knowledge)
        write_bytes_atomic(pretraining_path, pretraining)
        write_bytes_atomic(train_exact_path, train_exact_evaluation)
        write_bytes_atomic(train_exact_sample_path, train_exact_sample)
        write_bytes_atomic(train_exact_native_path, train_exact_native_evaluation)
        write_bytes_atomic(validation_path, validation_evaluation)
        write_bytes_atomic(validation_native_path, validation_native_evaluation)
        write_bytes_atomic(test_path, test_evaluation)
        write_bytes_atomic(test_native_path, test_native_evaluation)
        write_bytes_atomic(test_sample_path, test_abstention_sample)

        outputs = {
            "knowledge_train": output_identity(knowledge_path),
            "pretrain_train_ascii": output_identity(pretraining_path),
            "train_exact_retrieval_contract": output_identity(train_exact_path),
            "train_exact_retrieval_evidence_sample": output_identity(train_exact_sample_path),
            "train_exact_retrieval_contract_native_tsv": output_identity(train_exact_native_path),
            "validation_source_disjoint_abstention": output_identity(validation_path),
            "validation_source_disjoint_abstention_native_tsv": output_identity(validation_native_path),
            "test_source_disjoint_abstention": output_identity(test_path),
            "test_source_disjoint_abstention_native_tsv": output_identity(test_native_path),
            "test_source_disjoint_abstention_evidence_sample": output_identity(test_sample_path),
        }
        manifest: dict[str, object] = {
            "schema": PACKAGE_SCHEMA,
            "transformation": TRANSFORMATION,
            "native_runtime_contract": {
                "training": "native Fortran vegai_pretrain over strict ASCII projection",
                "retrieval": "native Fortran vegai_chat over literal UTF-8 TSV response rows",
                "generation_claim": "none; this package supports grounded retrieval only",
            },
            "source": {
                "repository": arguments.source_repository,
                "revision": arguments.source_revision,
                "license": arguments.source_license,
                "attribution": arguments.attribution,
                "synthetic_or_human_status": arguments.source_nature,
                "files": source_identities,
            },
            "split_isolation": {
                "unit": "conversation_id",
                "all_source_conversation_ids_unique_across_train_validation_test": True,
                "validation_and_test_outputs_are_explicit_source_disjoint_abstention_sets": True,
                "warning": (
                    "null expected prompts measure unsafe false grounding and correct abstention only; "
                    "they are not labels for unseen answer correctness"
                ),
            },
            "counts": {
                "train_pairs_raw": len(train_pairs),
                "train_knowledge_rows_unique_canonical_prompt": unique_rows,
                "train_identical_canonical_prompt_duplicates_dropped": duplicates,
                "train_pretraining_records": pretraining_records,
                "train_exact_retrieval_contract_cases": train_exact_evaluation.count(b"\n"),
                "evidence_sample_cases_requested": arguments.evidence_sample_cases,
                "train_exact_retrieval_evidence_sample_cases": train_exact_sample.count(b"\n"),
                "test_source_disjoint_abstention_evidence_sample_cases": test_abstention_sample.count(b"\n"),
                "train_utterances_under_two_tokens_skipped": skipped_records,
                "validation_pairs_raw": len(validation_pairs),
                "test_pairs_raw": len(test_pairs),
                "validation_abstention_cases": validation_evaluation.count(b"\n"),
                "test_abstention_cases": test_evaluation.count(b"\n"),
            },
            "outputs": outputs,
        }
        manifest_path = staging / "manifest.json"
        write_bytes_atomic(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.replace(staging, output)
        fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", required=True, type=Path)
    parser.add_argument("--test-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-repository", required=True,
                        help="canonical GitHub repository URL; no downloader is embedded")
    parser.add_argument("--source-revision", required=True, help="full pinned lowercase 40-hex Git revision")
    parser.add_argument("--source-license", required=True, help="declared source SPDX/licence identifier")
    parser.add_argument("--attribution", required=True, help="attribution text required when the source licence requires it")
    parser.add_argument("--source-nature", required=True,
                        help="e.g. synthetic-ai-assisted or human-authored; recorded rather than inferred")
    parser.add_argument("--evidence-sample-cases", type=int, default=DEFAULT_EVIDENCE_SAMPLE_CASES,
                        help="deterministic hash-ranked case count for bounded evaluator evidence files")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        manifest = build_package(arguments)
    except DialogueCorpusError as error:
        print(f"turkish dialogue corpus build: ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"schema": PACKAGE_SCHEMA, "status": "ready", "counts": manifest["counts"], "outputs": manifest["outputs"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
