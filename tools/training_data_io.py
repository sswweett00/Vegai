#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Streaming, schema-checked training-record I/O for Vegai.

This module is deliberately an adapter *around* Vegai's native HDC corpus
contract, not a replacement for it.  The Fortran learner consumes one strict
ASCII/LF text record per physical line.  Rich records can therefore retain
labels and provenance while they move through JSONL, JSON, Parquet, or Arrow,
but ``write_hdc_corpus`` intentionally exports only ``text`` after an explicit
strict-contract validation step.

Supported logical formats
-------------------------
``jsonl``
    One record object per UTF-8 line.  Streaming and stdlib-only.
``json``
    One top-level JSON array of record objects.  Parsed incrementally; the
    reader never calls ``json.load`` on the whole document.
``txt``
    One UTF-8 text value per LF-delimited line.  It is a text-only projection:
    id/label/split/source/metadata cannot be silently discarded on write.
``parquet`` / ``arrow``
    Rich columnar records with the explicit Arrow schema below.  These require
    the optional ``pyarrow`` dependency listed in ``requirements-dataio.txt``.

The public API is intentionally generator based.  A caller can process files
larger than RAM as long as an individual record and configured batch fit in
memory.  For multiple independent files, ``iter_records_many`` offers bounded
parallel I/O with intentionally nondeterministic inter-file arrival order;
source-order-sensitive HDC pipelines should use one worker or sort/materialize
first.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import sqlite3
import tempfile
import threading
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence


RECORD_SCHEMA_VERSION = "vegai-training-record/v1"
TEXT_SCHEMA_VERSION = "vegai-training-text/v1"
SUPPORTED_FORMATS = ("jsonl", "json", "txt", "parquet", "arrow")
TEXT_FORMATS = frozenset(("jsonl", "json", "txt"))
COLUMNAR_FORMATS = frozenset(("parquet", "arrow"))
VALID_SPLITS = frozenset(("train", "validation", "test"))
DEFAULT_BATCH_SIZE = 4096
DEFAULT_CHUNK_CHARS = 64 * 1024
DEFAULT_MAX_RECORD_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_METADATA_BYTES = 64 * 1024

InputFormat = Literal["jsonl", "json", "txt", "parquet", "arrow"]
ErrorPolicy = Literal["raise", "collect", "skip"]


class TrainingDataError(RuntimeError):
    """Base class for expected training-data contract failures."""


class SchemaValidationError(TrainingDataError):
    """A logical record does not satisfy the versioned record schema."""


class DataReadError(TrainingDataError):
    """Input bytes cannot be decoded, parsed, or mapped to a record."""


class DataWriteError(TrainingDataError):
    """A target cannot represent a record or cannot be published safely."""


class OptionalDependencyError(TrainingDataError):
    """A selected columnar format needs pyarrow but it is not installed."""


@dataclass(frozen=True, slots=True)
class RecordIssue:
    """A bounded, privacy-safe diagnostic for one malformed logical record."""

    path: str
    record_number: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "record_number": self.record_number, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    """Versioned logical record shared by every supported rich format.

    ``text`` is the only field consumed by the native HDC trainer.  Optional
    fields are retained by rich formats so an upstream quality workflow can
    audit labels, deterministic split assignment, source provenance, and
    arbitrary JSON metadata without injecting them into training text.
    """

    text: str
    id: str | None = None
    label: str | None = None
    split: str | None = None
    source: str | None = None
    metadata: dict[str, Any] | None = None
    schema_version: str = RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Keep direct dataclass construction as strict as parsed mappings."""
        if self.schema_version != RECORD_SCHEMA_VERSION:
            raise SchemaValidationError(
                f"record schema_version must be {RECORD_SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        if not isinstance(self.text, str):
            raise SchemaValidationError("record.text must be a string")
        if "\n" in self.text or "\r" in self.text:
            raise SchemaValidationError("record.text must contain one logical record only; CR/LF is not allowed")
        for name in ("id", "label", "source"):
            raw = getattr(self, name)
            if raw is not None and (not isinstance(raw, str) or not raw):
                raise SchemaValidationError(f"record.{name} must be a nonempty string or null")
        if self.split is not None and self.split not in VALID_SPLITS:
            allowed_splits = ", ".join(sorted(VALID_SPLITS))
            raise SchemaValidationError(f"record.split must be one of {allowed_splits} or null")
        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise SchemaValidationError("record.metadata must be a JSON object or null")
            try:
                rendered_metadata = json.dumps(
                    self.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                normalized_metadata = json.loads(rendered_metadata)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise SchemaValidationError(f"record.metadata is not JSON-serializable: {error}") from error
            if len(rendered_metadata.encode("utf-8")) > DEFAULT_MAX_METADATA_BYTES:
                raise SchemaValidationError(
                    f"record.metadata exceeds {DEFAULT_MAX_METADATA_BYTES} UTF-8 bytes; split metadata upstream"
                )
            object.__setattr__(self, "metadata", normalized_metadata)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingRecord":
        """Validate one JSON-like mapping without silently dropping fields."""
        if not isinstance(value, Mapping):
            raise SchemaValidationError("record must be a JSON object / mapping")
        allowed = {"schema_version", "id", "text", "label", "split", "source", "metadata"}
        unknown = sorted(str(key) for key in value.keys() if key not in allowed)
        if unknown:
            raise SchemaValidationError(f"record contains unsupported fields: {', '.join(unknown)}")
        schema_version = value.get("schema_version", RECORD_SCHEMA_VERSION)
        if schema_version != RECORD_SCHEMA_VERSION:
            raise SchemaValidationError(
                f"record schema_version must be {RECORD_SCHEMA_VERSION!r}, got {schema_version!r}"
            )
        if "text" not in value:
            raise SchemaValidationError("record is missing required string field 'text'")
        text = value["text"]
        if not isinstance(text, str):
            raise SchemaValidationError("record.text must be a string")
        if "\n" in text or "\r" in text:
            raise SchemaValidationError("record.text must contain one logical record only; CR/LF is not allowed")

        optional: dict[str, str | None] = {}
        for name in ("id", "label", "source"):
            raw = value.get(name)
            if raw is not None and (not isinstance(raw, str) or not raw):
                raise SchemaValidationError(f"record.{name} must be a nonempty string or null")
            optional[name] = raw
        split = value.get("split")
        if split is not None and split not in VALID_SPLITS:
            allowed_splits = ", ".join(sorted(VALID_SPLITS))
            raise SchemaValidationError(f"record.split must be one of {allowed_splits} or null")

        metadata = value.get("metadata")
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise SchemaValidationError("record.metadata must be a JSON object or null")
            try:
                # Canonical round-trip both proves JSON compatibility and breaks
                # aliases to caller-owned mutable mappings.
                rendered_metadata = json.dumps(
                    metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                normalized_metadata = json.loads(rendered_metadata)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise SchemaValidationError(f"record.metadata is not JSON-serializable: {error}") from error
            if len(rendered_metadata.encode("utf-8")) > DEFAULT_MAX_METADATA_BYTES:
                raise SchemaValidationError(
                    f"record.metadata exceeds {DEFAULT_MAX_METADATA_BYTES} UTF-8 bytes; split metadata upstream"
                )
            metadata = normalized_metadata

        return cls(
            text=text,
            id=optional["id"],
            label=optional["label"],
            split=split,
            source=optional["source"],
            metadata=metadata,
            schema_version=schema_version,
        )

    def to_mapping(self, *, include_nulls: bool = False) -> dict[str, Any]:
        """Render the canonical JSON-object representation."""
        payload: dict[str, Any] = {"schema_version": self.schema_version, "text": self.text}
        for name in ("id", "label", "split", "source", "metadata"):
            value = getattr(self, name)
            if include_nulls or value is not None:
                payload[name] = value
        return payload

    def text_digest(self) -> str:
        """Stable SHA-256 identity used for exact text dedupe and split assignment."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ReadStats:
    """Streaming read telemetry; no record payload is retained."""

    records_yielded: int = 0
    issues: int = 0
    maximum_record_bytes: int = 0


@dataclass(frozen=True, slots=True)
class WriteReport:
    path: Path
    format: str
    records: int
    compression: str


@dataclass(frozen=True, slots=True)
class HdcTextReport:
    """Canonical corpus-export accounting for the native Fortran contract."""

    path: Path
    records: int
    bytes_written: int
    sha256: str


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Exact-dedup and label/split telemetry from a SQLite-backed inspection."""

    records_seen: int
    unique_text_records: int
    duplicate_text_records: int
    label_conflicts: int
    label_counts: dict[str, int]
    split_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_seen": self.records_seen,
            "unique_text_records": self.unique_text_records,
            "duplicate_text_records": self.duplicate_text_records,
            "label_conflicts": self.label_conflicts,
            "label_counts": self.label_counts,
            "split_counts": self.split_counts,
            "label_imbalance": label_imbalance(self.label_counts),
        }


def _require_positive_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _require_pyarrow() -> tuple[Any, Any, Any]:
    """Import optional pyarrow lazily so JSON/TXT need no third-party package."""
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq
    except ImportError as error:
        raise OptionalDependencyError(
            "Parquet and Arrow require optional pyarrow; install with "
            "python3 -m pip install -r requirements-dataio.txt"
        ) from error
    return pa, ipc, pq


def arrow_record_schema() -> Any:
    """Return the strict Arrow schema used by Parquet and Arrow IPC records."""
    pa, _, _ = _require_pyarrow()
    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("id", pa.string(), nullable=True),
            pa.field("text", pa.string(), nullable=False),
            pa.field("label", pa.string(), nullable=True),
            pa.field("split", pa.string(), nullable=True),
            pa.field("source", pa.string(), nullable=True),
            # A canonical JSON string avoids an Arrow map-type compatibility
            # trap while preserving arbitrary metadata exactly across formats.
            pa.field("metadata_json", pa.string(), nullable=True),
        ]
    )


def training_record_json_schema() -> dict[str, Any]:
    """JSON Schema published by the CLI for non-Arrow producer validation."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://vegai.local/schemas/training-record/v1",
        "title": "Vegai training record",
        "type": "object",
        "additionalProperties": False,
        "required": ["text"],
        "properties": {
            "schema_version": {"const": RECORD_SCHEMA_VERSION},
            "id": {"type": ["string", "null"], "minLength": 1},
            "text": {"type": "string", "pattern": "^[^\\r\\n]*$"},
            "label": {"type": ["string", "null"], "minLength": 1},
            "split": {"type": ["string", "null"], "enum": ["train", "validation", "test", None]},
            "source": {"type": ["string", "null"], "minLength": 1},
            "metadata": {"type": ["object", "null"]},
        },
    }


def infer_format(path: Path, declared_format: str = "auto") -> str:
    """Resolve a format without looking at payload bytes or loading the file."""
    if declared_format != "auto":
        if declared_format not in SUPPORTED_FORMATS:
            raise TrainingDataError(f"unsupported format {declared_format!r}; choose from {', '.join(SUPPORTED_FORMATS)}")
        return declared_format
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes.pop()
    suffix = suffixes[-1] if suffixes else ""
    mapping = {
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".json": "json",
        ".txt": "txt",
        ".text": "txt",
        ".parquet": "parquet",
        ".arrow": "arrow",
        ".feather": "arrow",
        ".ipc": "arrow",
    }
    resolved = mapping.get(suffix)
    if resolved is None:
        raise TrainingDataError(
            f"cannot infer format for {path}; use --input-format/--output-format explicitly"
        )
    return resolved


def _text_compression(path: Path, compression: str | None) -> str:
    if compression in (None, "auto"):
        return "gzip" if path.suffix.lower() == ".gz" else "none"
    if compression not in ("none", "gzip"):
        raise TrainingDataError("JSONL/JSON/TXT support only compression=none or gzip")
    return compression


@contextlib.contextmanager
def _open_binary_reader(path: Path, compression: str | None) -> Iterator[Any]:
    selected = _text_compression(path, compression)
    try:
        if selected == "gzip":
            with gzip.open(path, "rb") as handle:
                yield handle
        else:
            with path.open("rb") as handle:
                yield handle
    except (OSError, EOFError) as error:
        raise DataReadError(f"cannot read {path}: {error}") from error


@contextlib.contextmanager
def _open_text_writer(path: Path, compression: str) -> Iterator[Any]:
    try:
        if compression == "gzip":
            # Do not let a random atomic temporary filename or wall-clock mtime
            # leak into the gzip header. Payload bytes are reproducible for a
            # fixed Python/zlib implementation and logical record stream.
            with path.open("wb") as raw_handle:
                with gzip.GzipFile(fileobj=raw_handle, mode="wb", filename="", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                        yield handle
        else:
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                yield handle
    except OSError as error:
        raise DataWriteError(f"cannot write {path}: {error}") from error


def _record_issue(
    *,
    path: Path,
    number: int,
    error: BaseException,
    policy: ErrorPolicy,
    issues: list[RecordIssue] | None,
    max_issues: int,
    stats: ReadStats,
) -> None:
    issue = RecordIssue(str(path), number, str(error))
    stats.issues += 1
    if issues is not None and len(issues) < max_issues:
        issues.append(issue)
    if policy == "raise":
        raise DataReadError(f"{issue.path}: record {issue.record_number}: {issue.reason}") from error


def _decode_record_bytes(path: Path, number: int, raw: bytes, max_record_bytes: int) -> str:
    if len(raw) > max_record_bytes:
        raise DataReadError(f"record exceeds configured {max_record_bytes} byte limit")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataReadError(f"invalid UTF-8: {error}") from error


def _iter_bounded_binary_lines(handle: Any, max_record_bytes: int) -> Iterator[tuple[int, bytes, bool]]:
    """Yield physical lines without allocating an unbounded hostile record.

    ``for line in handle`` can materialize an arbitrarily large line before a
    schema check sees it.  The bounded ``readline`` path reports an over-limit
    line, drains its remainder, and then lets a collect/skip caller continue at
    the following physical record.
    """
    number = 0
    # One extra byte distinguishes a legal max-byte payload plus LF from an
    # overlong payload.  A final unterminated line is allowed at this generic
    # boundary; strict native-HDC export enforces terminal LF in its output.
    limit = max_record_bytes + 2
    while True:
        raw = handle.readline(limit)
        if not raw:
            return
        number += 1
        has_newline = raw.endswith(b"\n")
        payload_length = len(raw) - 1 if has_newline else len(raw)
        over_limit = payload_length > max_record_bytes or (not has_newline and len(raw) == limit)
        if over_limit and not has_newline:
            # Do not retain the rest of an oversized physical record.  Each
            # bounded read is discarded until the next record boundary/EOF.
            while raw and not raw.endswith(b"\n"):
                raw = handle.readline(limit)
        yield number, raw if not over_limit else b"", over_limit


def _iter_jsonl_records(
    path: Path,
    *,
    compression: str | None,
    policy: ErrorPolicy,
    issues: list[RecordIssue] | None,
    max_issues: int,
    stats: ReadStats,
    max_record_bytes: int,
) -> Iterator[TrainingRecord]:
    with _open_binary_reader(path, compression) as handle:
        for number, raw, over_limit in _iter_bounded_binary_lines(handle, max_record_bytes):
            # A single LF is blank and therefore an invalid rich record; never
            # treat malformed input as a silently absent training example.
            raw = raw[:-1] if raw.endswith(b"\n") else raw
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            try:
                if over_limit:
                    raise DataReadError(f"record exceeds configured {max_record_bytes} byte limit")
                decoded = _decode_record_bytes(path, number, raw, max_record_bytes)
                if not decoded.strip():
                    raise SchemaValidationError("blank JSONL line")
                parsed = json.loads(decoded)
                record = TrainingRecord.from_mapping(parsed)
            except (DataReadError, SchemaValidationError, json.JSONDecodeError, TypeError, ValueError) as error:
                _record_issue(
                    path=path, number=number, error=error, policy=policy, issues=issues,
                    max_issues=max_issues, stats=stats,
                )
                continue
            stats.records_yielded += 1
            stats.maximum_record_bytes = max(stats.maximum_record_bytes, len(raw))
            yield record


def _iter_txt_records(
    path: Path,
    *,
    compression: str | None,
    policy: ErrorPolicy,
    issues: list[RecordIssue] | None,
    max_issues: int,
    stats: ReadStats,
    max_record_bytes: int,
) -> Iterator[TrainingRecord]:
    with _open_binary_reader(path, compression) as handle:
        for number, raw, over_limit in _iter_bounded_binary_lines(handle, max_record_bytes):
            raw = raw[:-1] if raw.endswith(b"\n") else raw
            # Do not normalize a trailing CR.  TrainingRecord rejects it so a
            # CRLF corpus is visible to the caller instead of silently changed.
            try:
                if over_limit:
                    raise DataReadError(f"record exceeds configured {max_record_bytes} byte limit")
                text = _decode_record_bytes(path, number, raw, max_record_bytes)
                record = TrainingRecord(text=text, schema_version=RECORD_SCHEMA_VERSION)
            except (DataReadError, SchemaValidationError, TypeError, ValueError) as error:
                _record_issue(
                    path=path, number=number, error=error, policy=policy, issues=issues,
                    max_issues=max_issues, stats=stats,
                )
                continue
            stats.records_yielded += 1
            stats.maximum_record_bytes = max(stats.maximum_record_bytes, len(raw))
            yield record


def _iter_json_array_values(
    path: Path, compression: str | None, chunk_chars: int, max_record_bytes: int
) -> Iterator[tuple[int, Any]]:
    """Incrementally parse a top-level JSON array without ``json.load``.

    The standard library has no event parser.  This state machine uses
    ``JSONDecoder.raw_decode`` and retains only the unconsumed suffix plus one
    currently decoded object.  It intentionally supports a JSON array only;
    wrapper objects such as ``{"records": [...]}`` are not stream-friendly and
    are rejected rather than materialized silently.
    """
    if chunk_chars <= 0 or max_record_bytes <= 0:
        raise ValueError("json chunk size and maximum record bytes must be positive")
    selected = _text_compression(path, compression)
    try:
        opener: Any
        if selected == "gzip":
            opener = gzip.open(path, "rt", encoding="utf-8", newline="")
        else:
            opener = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise DataReadError(f"cannot read {path}: {error}") from error

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    state = "array-start"
    number = 0

    def append_chunk(*, compact: bool) -> bool:
        nonlocal buffer, position, eof
        if compact and position:
            buffer = buffer[position:]
            position = 0
        piece = opener.read(chunk_chars)
        if not piece:
            eof = True
            return False
        buffer += piece
        # UTF-8 character count is never greater than byte count, so this is a
        # safe early bound for a configured byte limit. One extra source chunk
        # permits discovery of a delimiter after a maximum-size value.
        if len(buffer) > max_record_bytes + chunk_chars:
            raise DataReadError(
                f"{path}: one JSON value or delimiter exceeds configured {max_record_bytes} byte limit; split records upstream"
            )
        return True

    def skip_whitespace() -> None:
        nonlocal position
        while position < len(buffer) and buffer[position] in " \t\r\n":
            position += 1

    try:
        while True:
            skip_whitespace()
            if position >= len(buffer):
                if eof:
                    if state == "done":
                        return
                    raise DataReadError(f"{path}: unexpected EOF in JSON top-level array")
                append_chunk(compact=True)
                continue

            if state == "array-start":
                if buffer[position] != "[":
                    raise DataReadError(f"{path}: JSON input must be one top-level array")
                position += 1
                state = "first-or-end"
                continue

            if state in ("first-or-end", "value-required"):
                skip_whitespace()
                if position >= len(buffer):
                    if eof:
                        raise DataReadError(f"{path}: unexpected EOF after JSON array delimiter")
                    append_chunk(compact=True)
                    continue
                if buffer[position] == "]":
                    if state == "value-required":
                        raise DataReadError(f"{path}: trailing comma in JSON array")
                    position += 1
                    state = "done"
                    continue
                # Discard prior decoded elements before an incomplete current
                # value asks for another source chunk. Without this compaction,
                # a long top-level array could retain every preceding record.
                if position:
                    buffer = buffer[position:]
                    position = 0
                start = 0
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, start)
                    except json.JSONDecodeError as error:
                        if eof:
                            raise DataReadError(f"{path}: invalid JSON array element {number + 1}: {error}") from error
                        append_chunk(compact=False)
                        start = position
                        continue
                    # A scalar number at a chunk boundary can look complete.
                    # Do not yield until the following delimiter is available.
                    probe = end
                    while True:
                        while probe < len(buffer) and buffer[probe] in " \t\r\n":
                            probe += 1
                        if probe < len(buffer) or eof:
                            break
                        append_chunk(compact=False)
                    if probe >= len(buffer):
                        raise DataReadError(f"{path}: JSON array element {number + 1} lacks a closing delimiter")
                    if buffer[probe] not in ",]":
                        raise DataReadError(f"{path}: JSON array element {number + 1} has no comma or closing bracket")
                    position = end
                    number += 1
                    yield number, value
                    state = "delimiter"
                    break
                continue

            if state == "delimiter":
                skip_whitespace()
                if position >= len(buffer):
                    if eof:
                        raise DataReadError(f"{path}: JSON array lacks closing bracket")
                    append_chunk(compact=True)
                    continue
                delimiter = buffer[position]
                position += 1
                if delimiter == ",":
                    state = "value-required"
                elif delimiter == "]":
                    state = "done"
                else:
                    raise DataReadError(f"{path}: expected comma or closing bracket after array element")
                continue

            if state == "done":
                skip_whitespace()
                if position < len(buffer):
                    raise DataReadError(f"{path}: trailing bytes after JSON top-level array")
                if eof:
                    return
                append_chunk(compact=True)
                continue

            raise AssertionError(f"unknown JSON parser state {state}")
    except UnicodeDecodeError as error:
        raise DataReadError(f"{path}: invalid UTF-8 JSON input: {error}") from error
    except (OSError, EOFError) as error:
        raise DataReadError(f"{path}: cannot read compressed JSON input: {error}") from error
    finally:
        opener.close()


def _iter_json_records(
    path: Path,
    *,
    compression: str | None,
    policy: ErrorPolicy,
    issues: list[RecordIssue] | None,
    max_issues: int,
    stats: ReadStats,
    max_record_bytes: int,
) -> Iterator[TrainingRecord]:
    for number, value in _iter_json_array_values(path, compression, DEFAULT_CHUNK_CHARS, max_record_bytes):
        try:
            # Validate a bounded representation before a pathological metadata
            # object can reach a writer or quality index.
            rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if len(rendered.encode("utf-8")) > max_record_bytes:
                raise DataReadError(f"JSON element exceeds configured {max_record_bytes} byte limit")
            record = TrainingRecord.from_mapping(value)
        except (DataReadError, SchemaValidationError, TypeError, ValueError, json.JSONDecodeError) as error:
            _record_issue(
                path=path, number=number, error=error, policy=policy, issues=issues,
                max_issues=max_issues, stats=stats,
            )
            continue
        stats.records_yielded += 1
        stats.maximum_record_bytes = max(stats.maximum_record_bytes, len(rendered.encode("utf-8")))
        yield record


def _validate_arrow_schema(schema: Any, path: Path) -> None:
    expected = arrow_record_schema()
    if not schema.equals(expected, check_metadata=False):
        raise DataReadError(
            f"{path}: Arrow schema mismatch; expected fields "
            "schema_version,id,text,label,split,source,metadata_json with the documented string types"
        )


def _record_from_arrow_row(
    path: Path, number: int, row: Mapping[str, Any], max_record_bytes: int
) -> TrainingRecord:
    text_value = row.get("text")
    if isinstance(text_value, str) and len(text_value.encode("utf-8")) > max_record_bytes:
        raise DataReadError(f"{path}: Arrow record {number} exceeds configured {max_record_bytes} byte limit")
    metadata_json = row.get("metadata_json")
    metadata: dict[str, Any] | None
    if metadata_json is None:
        metadata = None
    elif not isinstance(metadata_json, str):
        raise DataReadError(f"{path}: Arrow record {number} metadata_json must be a string or null")
    else:
        try:
            parsed_metadata = json.loads(metadata_json)
        except json.JSONDecodeError as error:
            raise DataReadError(f"{path}: Arrow record {number} has invalid metadata_json: {error}") from error
        if not isinstance(parsed_metadata, dict):
            raise DataReadError(f"{path}: Arrow record {number} metadata_json must decode to an object")
        metadata = parsed_metadata
    return TrainingRecord.from_mapping(
        {
            "schema_version": row.get("schema_version"),
            "id": row.get("id"),
            "text": text_value,
            "label": row.get("label"),
            "split": row.get("split"),
            "source": row.get("source"),
            "metadata": metadata,
        }
    )


def _iter_parquet_records(
    path: Path,
    *,
    batch_size: int,
    policy: ErrorPolicy,
    issues: list[RecordIssue] | None,
    max_issues: int,
    stats: ReadStats,
    max_record_bytes: int,
) -> Iterator[TrainingRecord]:
    _require_positive_batch_size(batch_size)
    _, _, pq = _require_pyarrow()
    parquet: Any | None = None
    try:
        parquet = pq.ParquetFile(str(path))
        _validate_arrow_schema(parquet.schema_arrow, path)
        number = 0
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                number += 1
                try:
                    record = _record_from_arrow_row(path, number, row, max_record_bytes)
                except (TrainingDataError, SchemaValidationError, TypeError, ValueError) as error:
                    _record_issue(
                        path=path, number=number, error=error, policy=policy, issues=issues,
                        max_issues=max_issues, stats=stats,
                    )
                    continue
                stats.records_yielded += 1
                stats.maximum_record_bytes = max(stats.maximum_record_bytes, len(record.text.encode("utf-8")))
                yield record
    except (OSError, TrainingDataError):
        raise
    except Exception as error:  # Arrow error classes vary by release.
        raise DataReadError(f"cannot read Parquet {path}: {error}") from error
    finally:
        if parquet is not None:
            with contextlib.suppress(Exception):
                parquet.close()


def _iter_arrow_records(
    path: Path,
    *,
    batch_size: int,
    policy: ErrorPolicy,
    issues: list[RecordIssue] | None,
    max_issues: int,
    stats: ReadStats,
    max_record_bytes: int,
) -> Iterator[TrainingRecord]:
    _require_positive_batch_size(batch_size)
    pa, ipc, _ = _require_pyarrow()
    reader: Any | None = None
    source: Any | None = None
    try:
        # File IPC permits memory mapping. Only failure to recognize the IPC
        # *container* falls back to stream IPC; a recognized file with a wrong
        # schema remains a schema error instead of being obscured by fallback.
        source = pa.memory_map(str(path), "r")
        try:
            reader = ipc.open_file(source)
        except Exception:
            source.close()
            source = pa.input_stream(str(path))
            reader = ipc.open_stream(source)
            batches = iter(reader)
        else:
            batches = (reader.get_batch(index) for index in range(reader.num_record_batches))
        _validate_arrow_schema(reader.schema, path)
        number = 0
        for batch in batches:
            # An IPC producer can legally create a much larger record batch
            # than Vegai's configured Python batch. Slice before ``to_pylist``
            # so row object materialization remains bounded by ``batch_size``.
            for offset in range(0, batch.num_rows, batch_size):
                for row in batch.slice(offset, batch_size).to_pylist():
                    number += 1
                    try:
                        record = _record_from_arrow_row(path, number, row, max_record_bytes)
                    except (TrainingDataError, SchemaValidationError, TypeError, ValueError) as error:
                        _record_issue(
                            path=path, number=number, error=error, policy=policy, issues=issues,
                            max_issues=max_issues, stats=stats,
                        )
                        continue
                    stats.records_yielded += 1
                    stats.maximum_record_bytes = max(stats.maximum_record_bytes, len(record.text.encode("utf-8")))
                    yield record
    except (OSError, TrainingDataError):
        raise
    except Exception as error:
        raise DataReadError(f"cannot read Arrow IPC {path}: {error}") from error
    finally:
        if reader is not None and hasattr(reader, "close"):
            with contextlib.suppress(Exception):
                reader.close()
        if source is not None:
            with contextlib.suppress(Exception):
                source.close()


def iter_records(
    path: Path | str,
    *,
    format: str = "auto",
    compression: str | None = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    error_policy: ErrorPolicy = "raise",
    issues: list[RecordIssue] | None = None,
    max_issues: int = 100,
    stats: ReadStats | None = None,
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
) -> Iterator[TrainingRecord]:
    """Lazily yield schema-valid records from one supported file.

    ``error_policy='raise'`` is the default to protect training provenance.
    ``collect`` or ``skip`` continues after malformed logical records; callers
    must make that lossy policy explicit and should persist the resulting issue
    count.  Container-level corruption (broken Parquet/Arrow footer, JSON
    syntax outside a record) always raises because it cannot be localized safely.
    """
    source = Path(path)
    resolved = infer_format(source, format)
    if error_policy not in ("raise", "collect", "skip"):
        raise ValueError("error_policy must be raise, collect, or skip")
    if max_issues < 0 or max_record_bytes <= 0:
        raise ValueError("max_issues must be nonnegative and max_record_bytes must be positive")
    _require_positive_batch_size(batch_size)
    active_stats = stats if stats is not None else ReadStats()
    if resolved == "jsonl":
        yield from _iter_jsonl_records(
            source, compression=compression, policy=error_policy, issues=issues,
            max_issues=max_issues, stats=active_stats, max_record_bytes=max_record_bytes,
        )
    elif resolved == "json":
        yield from _iter_json_records(
            source, compression=compression, policy=error_policy, issues=issues,
            max_issues=max_issues, stats=active_stats, max_record_bytes=max_record_bytes,
        )
    elif resolved == "txt":
        yield from _iter_txt_records(
            source, compression=compression, policy=error_policy, issues=issues,
            max_issues=max_issues, stats=active_stats, max_record_bytes=max_record_bytes,
        )
    elif resolved == "parquet":
        if compression not in (None, "auto", "none"):
            raise TrainingDataError("Parquet compression is embedded in the file; do not pass a reader compression option")
        yield from _iter_parquet_records(
            source, batch_size=batch_size, policy=error_policy, issues=issues,
            max_issues=max_issues, stats=active_stats, max_record_bytes=max_record_bytes,
        )
    elif resolved == "arrow":
        if compression not in (None, "auto", "none"):
            raise TrainingDataError("Arrow IPC compression is embedded in the file; do not pass a reader compression option")
        yield from _iter_arrow_records(
            source, batch_size=batch_size, policy=error_policy, issues=issues,
            max_issues=max_issues, stats=active_stats, max_record_bytes=max_record_bytes,
        )
    else:  # pragma: no cover - infer_format and type checking guard this.
        raise AssertionError(f"unhandled format {resolved}")


def iter_record_batches(records: Iterable[TrainingRecord], batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[list[TrainingRecord]]:
    """Group a record stream without retaining more than one configured batch."""
    _require_positive_batch_size(batch_size)
    batch: list[TrainingRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _arrow_rows(records: Sequence[TrainingRecord]) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    for record in records:
        metadata_json = (
            json.dumps(record.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if record.metadata is not None
            else None
        )
        rows.append(
            {
                "schema_version": record.schema_version,
                "id": record.id,
                "text": record.text,
                "label": record.label,
                "split": record.split,
                "source": record.source,
                "metadata_json": metadata_json,
            }
        )
    return rows


def _validate_text_projection(record: TrainingRecord) -> None:
    if any(getattr(record, field) is not None for field in ("id", "label", "split", "source", "metadata")):
        raise DataWriteError(
            "TXT is a text-only HDC-compatible projection and cannot preserve id/label/split/source/metadata; "
            "use JSONL/JSON/Parquet/Arrow or explicitly project records to text first"
        )


class RecordWriter:
    """Atomic, streaming writer shared by all five formats.

    The destination is replaced only after the final delimiter/footer has been
    written and the temporary payload is fsync'd.  A caller can therefore use
    this writer in a data preparation transaction without publishing malformed
    half-JSON or incomplete Parquet/Arrow footers.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        format: str = "auto",
        compression: str | None = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        atomic: bool = True,
    ) -> None:
        self.path = Path(path)
        self.format = infer_format(self.path, format)
        self.compression = compression
        self.batch_size = batch_size
        self.atomic = atomic
        _require_positive_batch_size(batch_size)
        self._temporary: Path | None = None
        self._handle: Any | None = None
        self._writer: Any | None = None
        self._first_json_record = True
        self._batch: list[TrainingRecord] = []
        self._records = 0
        self._selected_compression = "none"
        self._opened = False

    @property
    def record_count(self) -> int:
        """Number of accepted logical records written in this transaction."""
        return self._records

    @property
    def selected_compression(self) -> str:
        """Resolved codec after format/path ``auto`` policy is applied."""
        return self._selected_compression

    def __enter__(self) -> "RecordWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.atomic:
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
            os.close(descriptor)
            self._temporary = Path(temporary_name)
        else:
            self._temporary = self.path
        target = self._temporary
        assert target is not None
        try:
            if self.format in TEXT_FORMATS:
                self._selected_compression = _text_compression(self.path, self.compression)
                self._handle_context = _open_text_writer(target, self._selected_compression)
                self._handle = self._handle_context.__enter__()
                if self.format == "json":
                    self._handle.write("[")
            elif self.format == "parquet":
                self._selected_compression = "none" if self.compression in (None, "auto", "none") else str(self.compression)
                if self._selected_compression not in {"none", "zstd", "snappy", "gzip", "brotli", "lz4"}:
                    raise DataWriteError("Parquet compression must be none, zstd, snappy, gzip, brotli, or lz4")
                pa, _, pq = _require_pyarrow()
                self._writer = pq.ParquetWriter(
                    target, arrow_record_schema(), compression=None if self._selected_compression == "none" else self._selected_compression,
                    use_dictionary=True,
                    write_statistics=True,
                )
                self._pa = pa
            elif self.format == "arrow":
                self._selected_compression = "none" if self.compression in (None, "auto", "none") else str(self.compression)
                if self._selected_compression not in {"none", "zstd", "lz4"}:
                    raise DataWriteError("Arrow IPC compression must be none, zstd, or lz4")
                pa, ipc, _ = _require_pyarrow()
                self._handle = pa.OSFile(str(target), "wb")
                options = ipc.IpcWriteOptions(
                    compression=None if self._selected_compression == "none" else self._selected_compression
                )
                self._writer = ipc.new_file(self._handle, arrow_record_schema(), options=options)
                self._pa = pa
            else:  # pragma: no cover
                raise AssertionError(f"unsupported writer format {self.format}")
            self._opened = True
            return self
        except BaseException:
            self.abort()
            raise

    def write(self, record: TrainingRecord | Mapping[str, Any]) -> None:
        if not self._opened:
            raise DataWriteError("RecordWriter must be used as a context manager")
        normalized = record if isinstance(record, TrainingRecord) else TrainingRecord.from_mapping(record)
        if self.format == "jsonl":
            assert self._handle is not None
            self._handle.write(json.dumps(normalized.to_mapping(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        elif self.format == "json":
            assert self._handle is not None
            if not self._first_json_record:
                self._handle.write(",")
            self._handle.write(json.dumps(normalized.to_mapping(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            self._first_json_record = False
        elif self.format == "txt":
            _validate_text_projection(normalized)
            assert self._handle is not None
            self._handle.write(normalized.text + "\n")
        else:
            self._batch.append(normalized)
            if len(self._batch) >= self.batch_size:
                self._flush_columnar_batch()
        self._records += 1

    def write_many(self, records: Iterable[TrainingRecord | Mapping[str, Any]]) -> None:
        for record in records:
            self.write(record)

    def _flush_columnar_batch(self) -> None:
        if not self._batch:
            return
        assert self._writer is not None
        rows = _arrow_rows(self._batch)
        record_batch = self._pa.RecordBatch.from_pylist(rows, schema=arrow_record_schema())
        if self.format == "parquet":
            self._writer.write_batch(record_batch)
        else:
            self._writer.write_batch(record_batch)
        self._batch = []

    def close(self) -> WriteReport:
        if not self._opened:
            raise DataWriteError("writer is not open")
        target = self._temporary
        assert target is not None
        try:
            if self.format == "json":
                assert self._handle is not None
                self._handle.write("]\n")
            if self.format in COLUMNAR_FORMATS:
                self._flush_columnar_batch()
                assert self._writer is not None
                self._writer.close()
                self._writer = None
                # ParquetWriter owns its file descriptor. Arrow IPC owns a
                # separate OSFile sink that must be closed explicitly.
                if self._handle is not None:
                    self._handle.close()
                    self._handle = None
                _fsync_file(target)
            else:
                assert self._handle is not None
                self._handle.flush()
                self._handle_context.__exit__(None, None, None)
                # gzip writes its footer on close, so sync only after the
                # context has finalized the compressed member.
                _fsync_file(target)
                self._handle = None
            if self.atomic:
                os.replace(target, self.path)
                _fsync_directory(self.path.parent)
            self._opened = False
            return WriteReport(self.path, self.format, self.record_count, self.selected_compression)
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        """Close best-effort and remove unpublished temporary output."""
        with contextlib.suppress(Exception):
            if self._writer is not None:
                self._writer.close()
        with contextlib.suppress(Exception):
            if self._handle is not None:
                self._handle.close()
        with contextlib.suppress(Exception):
            if hasattr(self, "_handle_context") and self._handle_context is not None:
                self._handle_context.__exit__(None, None, None)
        if self.atomic and self._temporary is not None:
            self._temporary.unlink(missing_ok=True)
        self._opened = False
        self._writer = None
        self._handle = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is not None:
            self.abort()
            return False
        self.close()
        return False


def write_records(
    path: Path | str,
    records: Iterable[TrainingRecord | Mapping[str, Any]],
    *,
    format: str = "auto",
    compression: str | None = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    atomic: bool = True,
) -> WriteReport:
    """Write an iterable in bounded batches and atomically publish one file."""
    writer = RecordWriter(path, format=format, compression=compression, batch_size=batch_size, atomic=atomic)
    with writer:
        writer.write_many(records)
    # ``__exit__`` already closed; construct the report from durable state.
    return WriteReport(writer.path, writer.format, writer.record_count, writer.selected_compression)


def _fsync_file(path: Path) -> None:
    """Best-effort payload durability after an external columnar writer closes."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync on POSIX; harmless no-op on unsupported OSes."""
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


def iter_records_many(
    paths: Sequence[Path | str],
    *,
    workers: int = 1,
    format: str = "auto",
    compression: str | None = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    error_policy: ErrorPolicy = "raise",
    queue_batches: int = 2,
) -> Iterator[TrainingRecord]:
    """Read independent files concurrently with bounded queue backpressure.

    Arrival order across files is intentionally not deterministic when
    ``workers > 1``.  This avoids buffering whole early files simply to restore
    order. ``queue_batches`` bounds queued records to
    ``workers * queue_batches * batch_size``. Use ``workers=1`` for a
    source-order-sensitive training ledger.
    """
    if workers <= 0 or queue_batches <= 0:
        raise ValueError("workers and queue_batches must be positive")
    _require_positive_batch_size(batch_size)
    normalized_paths = [Path(path) for path in paths]
    if workers == 1 or len(normalized_paths) <= 1:
        for source in normalized_paths:
            yield from iter_records(
                source, format=format, compression=compression, batch_size=batch_size, error_policy=error_policy
            )
        return

    output: queue.Queue[tuple[str, int, Any]] = queue.Queue(maxsize=max(workers, 1) * queue_batches * batch_size)
    stop = threading.Event()

    def publish(event: tuple[str, int, Any]) -> bool:
        """Respect cancellation while a bounded consumer queue is full."""
        while not stop.is_set():
            try:
                output.put(event, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def worker(index: int, source: Path) -> None:
        try:
            for record in iter_records(
                source, format=format, compression=compression, batch_size=batch_size, error_policy=error_policy
            ):
                if not publish(("record", index, record)):
                    return
        except BaseException as error:  # carried to the consumer thread
            publish(("error", index, error))
        finally:
            publish(("done", index, None))

    threads = [threading.Thread(target=worker, args=(index, source), daemon=True) for index, source in enumerate(normalized_paths)]
    # Start at most the requested concurrency; launch a replacement only after
    # a worker completes to retain the bounded worker count.
    active: dict[int, threading.Thread] = {}
    next_index = 0
    completed = 0
    try:
        while next_index < len(threads) and len(active) < workers:
            active[next_index] = threads[next_index]
            threads[next_index].start()
            next_index += 1
        while completed < len(threads):
            kind, index, payload = output.get()
            if kind == "record":
                yield payload
            elif kind == "error":
                stop.set()
                raise payload
            elif kind == "done":
                completed += 1
                active.pop(index, None)
                if next_index < len(threads):
                    active[next_index] = threads[next_index]
                    threads[next_index].start()
                    next_index += 1
            else:  # pragma: no cover
                raise AssertionError(f"unknown parallel reader event {kind}")
    finally:
        stop.set()
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=1)


def validate_hdc_record(record: TrainingRecord) -> None:
    """Fail closed against the exact native HDC text-ingest limits.

    This function does not normalize Unicode, CRLF, tokens, or labels.  It
    detects violations before a rich-format dataset is exported as text, so a
    caller cannot mistakenly believe the native ASCII tokenizer trained on data
    that it would drop or reinterpret.
    """
    try:
        raw = record.text.encode("ascii")
    except UnicodeEncodeError as error:
        raise SchemaValidationError("HDC export requires ASCII text; preprocess Unicode upstream") from error
    if len(raw) > 8192:
        raise SchemaValidationError("HDC export record exceeds 8192-byte physical line limit")
    # Import lazily to keep the generic JSON/Arrow I/O module reusable while
    # still mirroring the repository's authoritative Python preflight helper.
    try:
        from corpus_manifest import DEFAULT_MAX_TOKENS_PER_LINE, DEFAULT_TOKEN_LENGTH, PreflightStats, inspect_short_record
    except ImportError as error:  # pragma: no cover - tools execution always has this sibling.
        raise TrainingDataError("cannot import Vegai corpus contract helper") from error
    stats = PreflightStats(records=1, maximum_record_bytes=len(raw))
    inspect_short_record(raw, stats, DEFAULT_TOKEN_LENGTH, DEFAULT_MAX_TOKENS_PER_LINE)
    if stats.non_ascii_bytes or stats.overlong_records or stats.records_with_tokenizer_truncation or stats.dropped_tokens:
        raise SchemaValidationError("HDC export record violates tokenizer token/count/ASCII limits")


def write_hdc_corpus(
    path: Path | str,
    records: Iterable[TrainingRecord | Mapping[str, Any]],
    *,
    strict: bool = True,
    atomic: bool = True,
) -> HdcTextReport:
    """Export only ``text`` as strict LF corpus records for ``vegai_pretrain``.

    Labels and metadata are intentionally never appended or tokenized here.
    Known labels can be injected only by an explicit upstream, versioned weak
    supervision policy; automatic label leakage would change model semantics.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path
    if atomic:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
    else:
        temporary = destination
    digest = hashlib.sha256()
    records_written = 0
    bytes_written = 0
    try:
        with temporary.open("wb") as handle:
            for number, value in enumerate(records, start=1):
                record = value if isinstance(value, TrainingRecord) else TrainingRecord.from_mapping(value)
                try:
                    if strict:
                        validate_hdc_record(record)
                    raw = record.text.encode("ascii" if strict else "utf-8")
                except (UnicodeEncodeError, TrainingDataError, SchemaValidationError) as error:
                    raise DataWriteError(f"HDC export record {number}: {error}") from error
                payload = raw + b"\n"
                handle.write(payload)
                digest.update(payload)
                records_written += 1
                bytes_written += len(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if atomic:
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        return HdcTextReport(destination, records_written, bytes_written, digest.hexdigest())
    except BaseException:
        if atomic:
            temporary.unlink(missing_ok=True)
        raise


def split_for_text_digest(text_digest: str, validation_basis_points: int, test_basis_points: int) -> str:
    """Streaming deterministic split assignment with approximate label stratification.

    Hashing record identity rather than source position ensures that shuffled
    input and repeated conversion runs make the same assignment.  It is
    statistically label-preserving only when labels are not correlated with the
    digest; ``inspect_quality`` should therefore audit each split afterward.
    """
    if validation_basis_points < 0 or test_basis_points < 0 or validation_basis_points + test_basis_points >= 10_000:
        raise ValueError("validation/test basis points must be nonnegative and sum to less than 10000")
    bucket = int.from_bytes(bytes.fromhex(text_digest)[:8], byteorder="big") % 10_000
    if bucket < test_basis_points:
        return "test"
    if bucket < test_basis_points + validation_basis_points:
        return "validation"
    return "train"


class ExactTextDeduplicator:
    """SQLite-backed exact text dedupe with label-conflict visibility.

    The table is disk backed so a large corpus does not require a Python set of
    every text digest.  Duplicates are identified by UTF-8 text SHA-256 because
    that is the field consumed by HDC training and the field that could leak
    between held-out splits.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.connection = sqlite3.connect(path)
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute(
                "CREATE TABLE seen (digest TEXT PRIMARY KEY, label TEXT, split TEXT NOT NULL)"
            )
            self.connection.execute("CREATE TABLE labels (label TEXT PRIMARY KEY, count INTEGER NOT NULL)")
            self.connection.execute("CREATE TABLE splits (split TEXT PRIMARY KEY, count INTEGER NOT NULL)")
            self.connection.commit()
        except sqlite3.Error as error:
            raise TrainingDataError(f"cannot initialize exact text dedupe index {path}: {error}") from error
        self.records_seen = 0
        self.unique_records = 0
        self.duplicates = 0
        self.label_conflicts = 0

    def accept(self, record: TrainingRecord, split: str) -> bool:
        self.records_seen += 1
        digest = record.text_digest()
        label = record.label
        try:
            existing = self.connection.execute("SELECT label FROM seen WHERE digest = ?", (digest,)).fetchone()
            if existing is not None:
                self.duplicates += 1
                if existing[0] != label:
                    self.label_conflicts += 1
                return False
            self.connection.execute("INSERT INTO seen(digest, label, split) VALUES (?, ?, ?)", (digest, label, split))
            if label is not None:
                self.connection.execute(
                    "INSERT INTO labels(label, count) VALUES (?, 1) "
                    "ON CONFLICT(label) DO UPDATE SET count = count + 1",
                    (label,),
                )
            self.connection.execute(
                "INSERT INTO splits(split, count) VALUES (?, 1) "
                "ON CONFLICT(split) DO UPDATE SET count = count + 1",
                (split,),
            )
            self.unique_records += 1
            if self.unique_records % 4096 == 0:
                self.connection.commit()
            return True
        except sqlite3.Error as error:
            raise TrainingDataError(f"exact text dedupe index failure: {error}") from error

    def report(self) -> QualityReport:
        try:
            self.connection.commit()
            labels = {str(label): int(count) for label, count in self.connection.execute("SELECT label, count FROM labels ORDER BY label")}
            splits = {str(split): int(count) for split, count in self.connection.execute("SELECT split, count FROM splits ORDER BY split")}
        except sqlite3.Error as error:
            raise TrainingDataError(f"cannot aggregate quality index: {error}") from error
        return QualityReport(
            records_seen=self.records_seen,
            unique_text_records=self.unique_records,
            duplicate_text_records=self.duplicates,
            label_conflicts=self.label_conflicts,
            label_counts=labels,
            split_counts=splits,
        )

    def close(self) -> None:
        self.connection.close()


def label_imbalance(label_counts: Mapping[str, int]) -> dict[str, float | int | None]:
    """Return transparent class-balance metrics; no resampling is implied."""
    if not label_counts:
        return {"classes": 0, "min_count": None, "max_count": None, "max_to_min_ratio": None, "entropy_bits": None}
    counts = list(label_counts.values())
    total = sum(counts)
    # Explicit log2 keeps output portable and avoids hidden numpy/pandas deps.
    import math

    entropy = -sum((count / total) * math.log2(count / total) for count in counts)
    minimum = min(counts)
    maximum = max(counts)
    return {
        "classes": len(counts),
        "min_count": minimum,
        "max_count": maximum,
        "max_to_min_ratio": round(maximum / minimum, 6) if minimum else None,
        "entropy_bits": round(entropy, 6),
    }


def inspect_quality(records: Iterable[TrainingRecord], *, sqlite_path: Path | None = None) -> QualityReport:
    """Consume a stream once and report exact dedupe/imbalance diagnostics."""
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if sqlite_path is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="vegai-training-quality-")
        sqlite_path = Path(temporary_directory.name) / "quality.sqlite"
    deduplicator = ExactTextDeduplicator(sqlite_path)
    try:
        for record in records:
            split = record.split or split_for_text_digest(record.text_digest(), 1000, 1000)
            deduplicator.accept(record, split)
        return deduplicator.report()
    finally:
        deduplicator.close()
        if temporary_directory is not None:
            temporary_directory.cleanup()


def cleaned_record(record: TrainingRecord, *, strip_outer_whitespace: bool, drop_empty: bool) -> TrainingRecord | None:
    """Apply only explicit, auditable cleaning transformations."""
    text = record.text.strip() if strip_outer_whitespace else record.text
    if not text and drop_empty:
        return None
    if text == record.text:
        return record
    return replace(record, text=text)
