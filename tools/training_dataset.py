#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CLI for Vegai's schema-checked multi-format training-data boundary.

Use ``schema`` to publish the logical record contract, ``validate`` to inspect
corruption and HDC compatibility, ``convert`` for streaming format conversion,
and ``prepare`` for exact text dedupe plus deterministic train/validation/test
splits.  The command never downloads, transliterates, or silently injects labels
into HDC text.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Any, Iterable, Iterator

from training_data_io import (
    DEFAULT_BATCH_SIZE,
    ExactTextDeduplicator,
    OptionalDependencyError,
    ReadStats,
    RecordIssue,
    RecordWriter,
    SchemaValidationError,
    TrainingDataError,
    TrainingRecord,
    arrow_record_schema,
    cleaned_record,
    infer_format,
    iter_records,
    split_for_text_digest,
    training_record_json_schema,
    validate_hdc_record,
    write_hdc_corpus,
)


PREPARED_DATASET_SCHEMA = "vegai-training-prepared-dataset/v1"
FORMAT_SUFFIX = {
    "jsonl": ".jsonl",
    "json": ".json",
    "txt": ".txt",
    "parquet": ".parquet",
    "arrow": ".arrow",
}


def fail(message: str) -> None:
    raise TrainingDataError(message)


def rendered_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def fsync_directory(directory: Path) -> None:
    """Best-effort rename durability on POSIX; unsupported hosts remain usable."""
    flag = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, os.O_RDONLY | flag)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def record_stream(arguments: argparse.Namespace, issues: list[RecordIssue], stats: ReadStats) -> Iterator[TrainingRecord]:
    policy = "collect" if arguments.drop_invalid else "raise"
    yield from iter_records(
        arguments.input,
        format=arguments.input_format,
        compression=arguments.input_compression,
        batch_size=arguments.batch_size,
        error_policy=policy,
        issues=issues,
        max_issues=arguments.max_issues,
        stats=stats,
    )


def source_identity(path: Path) -> dict[str, Any]:
    """Fingerprint a local source without materializing it in memory."""
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise TrainingDataError(f"cannot fingerprint input {path}: {error}") from error
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": byte_count}


def require_unchanged_source(path: Path, before: dict[str, Any]) -> dict[str, Any]:
    """Reject a mutable input whose published provenance cannot be trustworthy."""
    after = source_identity(path)
    if after != before:
        fail(f"input changed while it was being processed: {path}")
    return after


def _text_projection(records: Iterable[TrainingRecord]) -> Iterator[TrainingRecord]:
    for record in records:
        yield TrainingRecord(text=record.text)


def command_schema(arguments: argparse.Namespace) -> int:
    if arguments.kind == "json":
        sys.stdout.write(rendered_json(training_record_json_schema()))
    else:
        schema = arrow_record_schema()
        print(schema)
    return 0


def command_validate(arguments: argparse.Namespace) -> int:
    source_before = source_identity(arguments.input)
    issues: list[RecordIssue] = []
    stats = ReadStats()
    deduplicator_directory = tempfile.TemporaryDirectory(prefix="vegai-training-data-validate-")
    deduplicator = ExactTextDeduplicator(Path(deduplicator_directory.name) / "quality.sqlite")
    hdc_violations = 0
    empty_records = 0
    try:
        for record in record_stream(arguments, issues, stats):
            if not record.text:
                empty_records += 1
            if arguments.hdc_strict:
                try:
                    validate_hdc_record(record)
                except SchemaValidationError as error:
                    hdc_violations += 1
                    if not arguments.drop_invalid:
                        raise TrainingDataError(f"HDC strict validation failed: {error}") from error
                    if len(issues) < arguments.max_issues:
                        issues.append(RecordIssue(str(arguments.input), stats.records_yielded, f"HDC strict: {error}"))
                    continue
            split = record.split or split_for_text_digest(
                record.text_digest(), arguments.validation_basis_points, arguments.test_basis_points
            )
            deduplicator.accept(record, split)
        quality = deduplicator.report()
    finally:
        deduplicator.close()
        deduplicator_directory.cleanup()

    source = require_unchanged_source(arguments.input, source_before)
    payload = {
        "schema": "vegai-training-data-validation/v1",
        "input": source,
        "format": infer_format(arguments.input, arguments.input_format),
        "records": {
            "valid": stats.records_yielded,
            "issues": stats.issues + hdc_violations,
            "empty_text": empty_records,
            "maximum_record_bytes": stats.maximum_record_bytes,
        },
        "quality": quality.to_dict(),
        "hdc_strict": arguments.hdc_strict,
        "issues": [issue.to_dict() for issue in issues],
        "issues_truncated": stats.issues + hdc_violations > len(issues),
    }
    sys.stdout.write(rendered_json(payload))
    return 0


def command_convert(arguments: argparse.Namespace) -> int:
    if arguments.input.resolve() == arguments.output.resolve():
        fail("input and output must be different paths")
    source_before = source_identity(arguments.input)
    issues: list[RecordIssue] = []
    stats = ReadStats()
    records: Iterable[TrainingRecord] = record_stream(arguments, issues, stats)
    if arguments.hdc_strict:
        # Native corpus export is intentionally separate from generic TXT.  It
        # validates the exact ASCII/token contract and never serializes labels.
        # Keep its completed output private until the source fingerprint is
        # rechecked, so a mutable input cannot publish a falsely-provenanced
        # conversion.
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{arguments.output.name}.source-check-", suffix=".tmp", dir=arguments.output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            report = write_hdc_corpus(temporary, records, strict=True, atomic=True)
            source = require_unchanged_source(arguments.input, source_before)
            os.replace(temporary, arguments.output)
            fsync_directory(arguments.output.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        payload = {
            "schema": "vegai-training-data-conversion/v1",
            "input": source,
            "output": {
                "path": str(arguments.output),
                "records": report.records,
                "bytes_written": report.bytes_written,
                "sha256": report.sha256,
            },
            "output_format": "hdc-text",
            "issues": [issue.to_dict() for issue in issues],
            "input_stats": asdict(stats),
        }
    else:
        output_format = infer_format(arguments.output, arguments.output_format)
        if output_format == "txt":
            if not arguments.text_only:
                fail("TXT conversion is lossy for rich fields; pass --text-only explicitly or choose a rich format")
            records = _text_projection(records)
        writer = RecordWriter(
            arguments.output,
            format=output_format,
            compression=arguments.output_compression,
            batch_size=arguments.batch_size,
            atomic=True,
        )
        with writer:
            writer.write_many(records)
            # ``RecordWriter.__exit__`` atomically publishes only after this
            # block succeeds. Recheck while its completed file is still private.
            source = require_unchanged_source(arguments.input, source_before)
        payload = {
            "schema": "vegai-training-data-conversion/v1",
            "input": source,
            "output": {
                "path": str(arguments.output),
                "format": output_format,
                "records": writer.record_count,
                "compression": writer.selected_compression,
            },
            "issues": [issue.to_dict() for issue in issues],
            "input_stats": asdict(stats),
        }
    sys.stdout.write(rendered_json(payload))
    return 0


def _output_file(directory: Path, split: str, output_format: str, compression: str) -> Path:
    suffix = FORMAT_SUFFIX[output_format]
    if compression == "gzip" and output_format in {"jsonl", "json", "txt"}:
        suffix += ".gz"
    return directory / f"{split}{suffix}"


def resolved_output_compression(requested: str | None) -> str:
    """Persist the actual default rather than the ambiguous CLI token ``auto``."""
    return "none" if requested in (None, "auto") else requested


def command_prepare(arguments: argparse.Namespace) -> int:
    """Produce a new atomically-published split dataset from one input stream."""
    output = arguments.output_dir.resolve()
    if output.exists():
        fail(f"output directory already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_before = source_identity(arguments.input)
    output_format = infer_format(Path(f"placeholder{FORMAT_SUFFIX[arguments.output_format]}"), arguments.output_format)
    if output_format == "txt" and not arguments.text_only:
        fail("TXT prepared datasets are text-only; pass --text-only explicitly or choose a rich output format")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    issues: list[RecordIssue] = []
    stats = ReadStats()
    empty_dropped = 0
    cleaned_changed = 0
    hdc_rejected = 0
    deduplicator: ExactTextDeduplicator | None = None
    try:
        deduplicator = ExactTextDeduplicator(staging / ".exact-text-dedup.sqlite")
        output_compression = resolved_output_compression(arguments.output_compression)
        writer_paths = {
            split: _output_file(staging, split, output_format, output_compression)
            for split in ("train", "validation", "test")
        }
        with contextlib.ExitStack() as stack:
            writers = {
                split: stack.enter_context(
                    RecordWriter(
                        path,
                        format=output_format,
                        compression=output_compression,
                        batch_size=arguments.batch_size,
                        atomic=True,
                    )
                )
                for split, path in writer_paths.items()
            }
            for record in record_stream(arguments, issues, stats):
                cleaned = cleaned_record(
                    record,
                    strip_outer_whitespace=arguments.strip_outer_whitespace,
                    drop_empty=arguments.drop_empty,
                )
                if cleaned is None:
                    empty_dropped += 1
                    continue
                if cleaned.text != record.text:
                    cleaned_changed += 1
                if arguments.hdc_strict:
                    try:
                        validate_hdc_record(cleaned)
                    except SchemaValidationError as error:
                        hdc_rejected += 1
                        if not arguments.drop_invalid:
                            raise TrainingDataError(f"HDC strict validation failed: {error}") from error
                        if len(issues) < arguments.max_issues:
                            issues.append(RecordIssue(str(arguments.input), stats.records_yielded, f"HDC strict: {error}"))
                        continue
                split = split_for_text_digest(
                    cleaned.text_digest(), arguments.validation_basis_points, arguments.test_basis_points
                )
                if not deduplicator.accept(cleaned, split):
                    continue
                output_record = replace(cleaned, split=split)
                if output_format == "txt":
                    output_record = TrainingRecord(text=output_record.text)
                writers[split].write(output_record)
        quality = deduplicator.report()
        if arguments.reject_label_conflicts and quality.label_conflicts:
            fail(f"exact duplicate text had {quality.label_conflicts} conflicting labels")
        source = require_unchanged_source(arguments.input, source_before)
        dataset = {
            "schema": PREPARED_DATASET_SCHEMA,
            "source": source,
            "input_format": infer_format(arguments.input, arguments.input_format),
            "output_format": output_format,
            "output_compression": output_compression,
            "split": {
                "algorithm": "sha256-utf8-text-first-64-bits-mod-10000/v1",
                "validation_basis_points": arguments.validation_basis_points,
                "test_basis_points": arguments.test_basis_points,
                "contract": (
                    "Exact text duplicates are removed before assignment. Split is deterministic by UTF-8 text SHA-256; "
                    "label balance is audited, not silently resampled."
                ),
            },
            "cleaning": {
                "strip_outer_whitespace": arguments.strip_outer_whitespace,
                "drop_empty": arguments.drop_empty,
                "changed_records": cleaned_changed,
                "empty_records_dropped": empty_dropped,
                "hdc_strict": arguments.hdc_strict,
                "hdc_records_rejected": hdc_rejected,
            },
            "quality": quality.to_dict(),
            "files": {
                split: {
                    "path": str(path.relative_to(staging)),
                    "records": writers[split].record_count,
                    "sha256": source_identity(path)["sha256"],
                    "bytes": source_identity(path)["bytes"],
                }
                for split, path in writer_paths.items()
            },
            "invalid_records": {
                "input_parse_or_schema_issues": stats.issues,
                "stored_issue_examples": [issue.to_dict() for issue in issues],
                "examples_truncated": stats.issues + hdc_rejected > len(issues),
                "drop_invalid_enabled": arguments.drop_invalid,
            },
        }
        write_json_atomic(staging / "dataset.json", dataset)
        deduplicator.close()
        deduplicator = None
        (staging / ".exact-text-dedup.sqlite").unlink(missing_ok=True)
        os.replace(staging, output)
        fsync_directory(output.parent)
    except BaseException:
        if deduplicator is not None:
            deduplicator.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    sys.stdout.write(rendered_json(dataset))
    return 0


def add_common_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="input dataset file")
    parser.add_argument("--input-format", choices=("auto", "jsonl", "json", "txt", "parquet", "arrow"), default="auto")
    parser.add_argument("--input-compression", choices=("auto", "none", "gzip"), default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--drop-invalid", action="store_true", help="continue after malformed records and report them; default is fail-closed")
    parser.add_argument("--max-issues", type=int, default=100, help="maximum stored issue examples (default: 100)")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    schema = subcommands.add_parser("schema", help="print logical JSON Schema or Arrow schema")
    schema.add_argument("--kind", choices=("json", "arrow"), default="json")
    schema.set_defaults(handler=command_schema)

    validate = subcommands.add_parser("validate", help="stream-validate one dataset and report quality telemetry")
    add_common_input_arguments(validate)
    validate.add_argument("--hdc-strict", action="store_true", help="also require the native ASCII/LF/tokenizer contract")
    validate.add_argument("--validation-basis-points", type=int, default=1000)
    validate.add_argument("--test-basis-points", type=int, default=1000)
    validate.set_defaults(handler=command_validate)

    convert = subcommands.add_parser("convert", help="stream-convert one dataset without loading it all into memory")
    add_common_input_arguments(convert)
    convert.add_argument("output", type=Path, help="target dataset file")
    convert.add_argument("--output-format", choices=("auto", "jsonl", "json", "txt", "parquet", "arrow"), default="auto")
    convert.add_argument("--output-compression", choices=("auto", "none", "gzip", "zstd", "snappy", "lz4", "brotli"), default="auto")
    convert.add_argument("--text-only", action="store_true", help="explicitly project rich records to text for TXT output")
    convert.add_argument("--hdc-strict", action="store_true", help="export strict native HDC text rather than a generic rich format")
    convert.set_defaults(handler=command_convert)

    prepare = subcommands.add_parser(
        "prepare", help="exact-deduplicate, audit labels, and hash-split a stream into an atomic dataset directory"
    )
    add_common_input_arguments(prepare)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--output-format", choices=("jsonl", "json", "txt", "parquet", "arrow"), default="jsonl")
    prepare.add_argument("--output-compression", choices=("auto", "none", "gzip", "zstd", "snappy", "lz4", "brotli"), default="auto")
    prepare.add_argument("--text-only", action="store_true", help="allow the intentional rich-to-text projection for TXT output")
    prepare.add_argument("--strip-outer-whitespace", action="store_true", help="apply this explicit cleaning transform before dedupe/split")
    prepare.add_argument("--drop-empty", action="store_true", help="drop empty text after optional stripping and count it")
    prepare.add_argument("--hdc-strict", action="store_true", help="reject records incompatible with native HDC text export")
    prepare.add_argument("--reject-label-conflicts", action="store_true", help="fail if duplicate text appears with different labels")
    prepare.add_argument("--validation-basis-points", type=int, default=1000, help="held-out validation share in 1/10000 units")
    prepare.add_argument("--test-basis-points", type=int, default=1000, help="held-out test share in 1/10000 units")
    prepare.set_defaults(handler=command_prepare)

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if hasattr(arguments, "batch_size") and arguments.batch_size <= 0:
        print("training dataset: FAIL: --batch-size must be positive", file=sys.stderr)
        return 2
    if hasattr(arguments, "max_issues") and arguments.max_issues < 0:
        print("training dataset: FAIL: --max-issues must be nonnegative", file=sys.stderr)
        return 2
    try:
        return arguments.handler(arguments)
    except (TrainingDataError, OptionalDependencyError, OSError, ValueError, sqlite3.Error) as error:
        print(f"training dataset: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
