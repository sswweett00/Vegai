#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a deterministic, deduplicated, content-addressed Vegai shard dataset.

Large-model training systems need a data boundary that is stricter than a file
list: records must have a known ingest contract, duplicate handling must be
reproducible, train/validation isolation must not depend on operator ordering,
and a crash must not publish a half-built dataset.  This tool supplies those
properties for Vegai's existing bounded ASCII tokenizer and continual-training
ledger.

It deliberately does *not* silently transliterate or discard UTF-8 text.  A
production owner must run a documented, reversible/traceable preprocessing
stage before this builder, then record that stage through ``--preprocessing``
and record the immutable source retrieval/version event through ``--retrieved-at``.
That preserves the project's fail-closed corpus contract instead of pretending
that an ASCII tokenizer learned arbitrary Unicode correctly.

Output layout (atomically published as one directory):

    DATASET/
      dataset.json                 immutable production data contract
      train/ledger.json            append-only-compatible training ledger
      train/shards/SHA256.txt      LF/ASCII strict content-addressed payloads
      validation/ledger.json       optional held-out deterministic split
      validation/shards/SHA256.txt

The exact-record SHA-256 index is SQLite-backed while building, allowing
bounded-RAM deduplication of large local corpora.  It is private staging state
and is never published with the dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import uuid
from typing import Any, BinaryIO, Iterator

from corpus_manifest import (
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_TOKENS_PER_LINE,
    DEFAULT_TOKEN_LENGTH,
    PreflightStats,
    inspect_short_record,
    is_token_byte,
)
from training_data_io import TrainingDataError, infer_format, iter_records, write_hdc_corpus


DATASET_SCHEMA = "vegai-production-dataset/v1"
LEDGER_SCHEMA = "vegai-continuous-shard-manifest/v1"
DIGEST_INDEX_SCHEMA = "vegai-exact-record-dedup/v1"
BASIS_POINTS = 10_000
CHUNK_BYTES = 1024 * 1024


class DatasetBuildError(RuntimeError):
    """Expected data-contract or publication failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetBuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(CHUNK_BYTES):
                digest.update(block)
    except OSError as error:
        raise DatasetBuildError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def fsync_directory(directory: Path) -> None:
    """Best-effort directory durability on POSIX without harming Windows."""
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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
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


def preflight_dict(stats: PreflightStats) -> dict[str, int]:
    return {
        "accepted_tokens": stats.accepted_tokens,
        "crlf_records": stats.crlf_records,
        "dropped_tokens": stats.dropped_tokens,
        "maximum_record_bytes": stats.maximum_record_bytes,
        "non_ascii_bytes": stats.non_ascii_bytes,
        "overlong_records": stats.overlong_records,
        "records_with_tokenizer_truncation": stats.records_with_tokenizer_truncation,
    }


def add_stats(destination: PreflightStats, source: PreflightStats) -> None:
    destination.records += source.records
    destination.overlong_records += source.overlong_records
    destination.records_with_tokenizer_truncation += source.records_with_tokenizer_truncation
    destination.accepted_tokens += source.accepted_tokens
    destination.dropped_tokens += source.dropped_tokens
    destination.maximum_record_bytes = max(destination.maximum_record_bytes, source.maximum_record_bytes)
    destination.non_ascii_bytes += source.non_ascii_bytes
    destination.crlf_records += source.crlf_records


def record_violation(stats: PreflightStats) -> str | None:
    if stats.overlong_records:
        return "overlong record"
    if stats.records_with_tokenizer_truncation:
        return "token exceeds the tokenizer's bounded record/token capacity"
    if stats.dropped_tokens:
        return "dropped tokenizer token"
    if stats.non_ascii_bytes:
        return "non-ASCII byte"
    if stats.crlf_records:
        return "CR byte / CRLF record"
    return None


def strict_record_stats(record: bytes) -> PreflightStats:
    """Mirror Fortran's tokenizer on one already-bounded LF-free record."""
    stats = PreflightStats(records=1, maximum_record_bytes=len(record))
    # The continuous controller exposes a CRLF counter, but this foundry makes
    # the stronger production promise "LF-only": an embedded bare CR is also
    # rejected instead of becoming an undocumented tokenizer delimiter.
    if b"\r" in record:
        stats.crlf_records = 1
    if len(record) > DEFAULT_MAX_LINE_BYTES:
        stats.overlong_records = 1
        return stats
    inspect_short_record(record, stats, DEFAULT_TOKEN_LENGTH, DEFAULT_MAX_TOKENS_PER_LINE)
    return stats


def canonical_record_tokens(record: bytes) -> Iterator[bytes]:
    """Yield strict-contract tokens in the tokenizer's ASCII lowercase form."""
    token = bytearray()
    for value in record:
        if is_token_byte(value):
            token.append(value + 32 if 65 <= value <= 90 else value)
        elif token:
            yield bytes(token)
            token.clear()
    if token:
        yield bytes(token)


def snapshot_source(source: Path, destination: Path) -> tuple[str, int]:
    """Take a private byte snapshot so packaged bytes match recorded identity."""
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            while block := reader.read(CHUNK_BYTES):
                writer.write(block)
                digest.update(block)
                byte_count += len(block)
            writer.flush()
            os.fsync(writer.fileno())
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise DatasetBuildError(f"cannot snapshot source corpus {source}: {error}") from error
    return digest.hexdigest(), byte_count


def materialize_structured_input_as_hdc_text(
    source: Path,
    destination: Path,
    input_format: str,
) -> tuple[str, int, dict[str, Any]]:
    """Project a rich input stream into the only corpus form native HDC accepts.

    The adapter is deliberately fail-closed: it streams schema-valid records,
    validates every ``text`` value against the same strict ASCII/token limits,
    then writes exactly one LF-terminated canonical text record. Labels,
    metadata and source fields remain in the rich source layer and are never
    appended to HDC text implicitly.  The returned provenance binds both raw
    source bytes and the canonical corpus used by the existing foundry.
    """
    raw_digest = sha256_file(source)
    try:
        raw_bytes = source.stat().st_size
    except OSError as error:
        raise DatasetBuildError(f"cannot stat structured source {source}: {error}") from error
    try:
        report = write_hdc_corpus(
            destination,
            iter_records(source, format=input_format, error_policy="raise"),
            strict=True,
            atomic=False,
        )
    except TrainingDataError as error:
        raise DatasetBuildError(f"structured input adapter rejected {source}: {error}") from error
    try:
        unchanged = sha256_file(source) == raw_digest and source.stat().st_size == raw_bytes
    except OSError as error:
        raise DatasetBuildError(f"cannot recheck structured source {source}: {error}") from error
    require(unchanged, f"source corpus changed while adapting: {source}")
    return report.sha256, report.bytes_written, {
        "schema": "vegai-training-data-adapter/v1",
        "source_format": input_format,
        "source_sha256": raw_digest,
        "source_bytes": raw_bytes,
        "canonical_hdc_text_sha256": report.sha256,
        "canonical_hdc_text_bytes": report.bytes_written,
        "contract": (
            "Only schema-valid record.text values are exported. Labels, split, source, and metadata are not "
            "silently injected into native HDC training text."
        ),
    }


def iter_strict_lf_records(path: Path, label: Path | None = None) -> Iterator[tuple[int, bytes]]:
    """Yield bounded LF records without buffering a hostile oversized line."""
    display_path = path if label is None else label
    try:
        with path.open("rb") as handle:
            number = 0
            while True:
                # Need one byte beyond the largest legal payload and one LF byte
                # to distinguish a legal 8192-byte record from an overlong one.
                piece = handle.readline(DEFAULT_MAX_LINE_BYTES + 2)
                if not piece:
                    return
                number += 1
                if not piece.endswith(b"\n"):
                    # Either EOF without a terminal LF or a record too long for
                    # the bounded read.  Drain the latter before failing so the
                    # exception does not leave a misleading partially processed
                    # source in a caller that catches it.
                    while piece and not piece.endswith(b"\n"):
                        piece = handle.readline(DEFAULT_MAX_LINE_BYTES + 2)
                    raise DatasetBuildError(
                        f"{display_path}: record {number} lacks a terminal LF or exceeds {DEFAULT_MAX_LINE_BYTES} bytes"
                    )
                record = piece[:-1]
                if len(record) > DEFAULT_MAX_LINE_BYTES:
                    raise DatasetBuildError(f"{display_path}: record {number} exceeds {DEFAULT_MAX_LINE_BYTES} bytes")
                yield number, record
    except OSError as error:
        raise DatasetBuildError(f"cannot read source corpus {display_path}: {error}") from error


def split_is_validation(record_digest: bytes, validation_basis_points: int) -> bool:
    if validation_basis_points == 0:
        return False
    bucket = int.from_bytes(record_digest[:8], byteorder="big", signed=False) % BASIS_POINTS
    return bucket < validation_basis_points


@dataclass
class DigestStream:
    """Order-sensitive audit digest over retained record identities."""

    count: int = 0
    _digest: Any = field(default_factory=hashlib.sha256)

    def add(self, digest: bytes) -> None:
        self._digest.update(digest)
        self.count += 1

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class ShardWriter:
    """Write one split as strict, fsync'd content-addressed shards."""

    def __init__(self, root: Path, max_bytes: int, max_tokens: int) -> None:
        self.root = root
        self.shard_directory = root / "shards"
        self.shard_directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_tokens = max_tokens
        self.entries: list[dict[str, Any]] = []
        self.total = PreflightStats()
        self.record_digests = DigestStream()
        self._handle: BinaryIO | None = None
        self._temporary: Path | None = None
        self._stats = PreflightStats()
        self._digest = hashlib.sha256()
        self._bytes = 0
        self._sequence = 0

    def _open(self) -> None:
        require(self._handle is None, "internal shard writer state error")
        self._sequence += 1
        self._temporary = self.shard_directory / f".shard-{self._sequence:08d}.{uuid.uuid4().hex}.tmp"
        self._handle = self._temporary.open("xb")
        self._stats = PreflightStats()
        self._digest = hashlib.sha256()
        self._bytes = 0

    def _close(self) -> None:
        if self._handle is None:
            return
        assert self._temporary is not None
        require(self._stats.accepted_tokens > 0, "attempted to publish a zero-token shard")
        digest = self._digest.hexdigest()
        final = self.shard_directory / f"{digest}.txt"
        require(not final.exists(), f"content-addressed shard collision at {final}")
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._temporary.replace(final)
            fsync_directory(self.shard_directory)
        finally:
            if not self._handle.closed:
                self._handle.close()
            self._temporary.unlink(missing_ok=True)
        self.entries.append(
            {
                "id": digest,
                "path": str(Path("shards") / final.name),
                "sha256": digest,
                "bytes": self._bytes,
                "physical_lf_records": self._stats.records,
                "preflight": preflight_dict(self._stats),
            }
        )
        add_stats(self.total, self._stats)
        self._handle = None
        self._temporary = None

    def add(self, record: bytes, stats: PreflightStats, digest: bytes) -> None:
        require(stats.accepted_tokens > 0, "zero-token records are not shardable")
        wire_bytes = len(record) + 1
        require(wire_bytes <= self.max_bytes, "single valid record exceeds --max-shard-bytes")
        require(stats.accepted_tokens <= self.max_tokens, "single valid record exceeds --max-shard-tokens")
        if self._handle is None:
            self._open()
        elif self._stats.records > 0 and (
            self._bytes + wire_bytes > self.max_bytes
            or self._stats.accepted_tokens + stats.accepted_tokens > self.max_tokens
        ):
            self._close()
            self._open()
        assert self._handle is not None
        payload = record + b"\n"
        self._handle.write(payload)
        self._digest.update(payload)
        self._bytes += len(payload)
        add_stats(self._stats, stats)
        self.record_digests.add(digest)

    def finish(self) -> None:
        self._close()

    def ledger_payload(self, provenance: dict[str, str]) -> dict[str, Any]:
        require(self.entries, f"{self.root.name} split has no accepted records")
        return {
            "schema": LEDGER_SCHEMA,
            "declared_provenance": provenance,
            "vegai_ascii_ingest_contract": {
                "max_line_bytes": DEFAULT_MAX_LINE_BYTES,
                "token_length_bytes": DEFAULT_TOKEN_LENGTH,
                "max_tokens_per_line": DEFAULT_MAX_TOKENS_PER_LINE,
                "require_ascii_only": True,
                "require_lf_only": True,
                "require_no_dropped_tokens": True,
            },
            "shards": self.entries,
        }


def create_digest_index(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE seen_record (digest BLOB PRIMARY KEY, split TEXT NOT NULL)")
        # Continual production mode fixes min_frequency=1, so this disk-backed
        # set is an exact upper bound and readiness check for max_vocabulary.
        connection.execute("CREATE TABLE train_token (token BLOB PRIMARY KEY)")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("schema", DIGEST_INDEX_SCHEMA))
        connection.commit()
        return connection
    except sqlite3.Error as error:
        raise DatasetBuildError(f"cannot create exact deduplication index: {error}") from error


def insert_unique_record(connection: sqlite3.Connection, digest: bytes, split: str) -> bool:
    try:
        connection.execute("INSERT INTO seen_record(digest, split) VALUES (?, ?)", (digest, split))
        return True
    except sqlite3.IntegrityError:
        return False
    except sqlite3.Error as error:
        raise DatasetBuildError(f"deduplication index failure: {error}") from error


def insert_train_tokens(connection: sqlite3.Connection, record: bytes) -> None:
    try:
        connection.executemany(
            "INSERT OR IGNORE INTO train_token(token) VALUES (?)",
            ((token,) for token in canonical_record_tokens(record)),
        )
    except sqlite3.Error as error:
        raise DatasetBuildError(f"training vocabulary index failure: {error}") from error


def training_unique_tokens(connection: sqlite3.Connection) -> int:
    try:
        row = connection.execute("SELECT COUNT(*) FROM train_token").fetchone()
    except sqlite3.Error as error:
        raise DatasetBuildError(f"cannot count training vocabulary index: {error}") from error
    require(row is not None and isinstance(row[0], int) and row[0] > 0,
            "training vocabulary index is empty")
    return row[0]


def relative_sha256(path: Path, root: Path) -> tuple[str, str]:
    return str(path.relative_to(root)), sha256_file(path)


def build_dataset(arguments: argparse.Namespace) -> dict[str, Any]:
    source = arguments.input.resolve()
    output = arguments.output_dir.resolve()
    require(source.is_file(), f"input corpus is not a regular file: {source}")
    try:
        input_format = infer_format(source, arguments.input_format)
    except TrainingDataError as error:
        raise DatasetBuildError(str(error)) from error
    require(not output.exists(), f"output directory already exists; refusing to overwrite: {output}")
    require(arguments.max_shard_bytes > 0, "--max-shard-bytes must be positive")
    require(arguments.max_shard_tokens > 0, "--max-shard-tokens must be positive")
    require(0 <= arguments.validation_basis_points < BASIS_POINTS,
            f"--validation-basis-points must be in [0, {BASIS_POINTS - 1}]")
    require(arguments.deduplicate == "exact", "production datasets require --deduplicate exact")
    require(arguments.source.strip(), "--source is required for production provenance")
    require(arguments.license.strip(), "--license is required for production provenance")
    require(arguments.retrieved_at.strip(), "--retrieved-at is required for production provenance")
    require(arguments.preprocessing.strip(), "--preprocessing is required; do not silently normalize corpus text")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    digest_index: sqlite3.Connection | None = None
    try:
        # Never package directly from a caller-controlled path. Uncompressed
        # TXT keeps its historic byte snapshot path; a .txt.gz source is
        # explicitly decompressed/revalidated through the same adapter as rich
        # formats. Rich inputs are streamed into the strict LF/ASCII text shape
        # that the native Fortran trainer and the existing foundry consume.
        source_snapshot = staging / ".source-snapshot.txt"
        if input_format == "txt" and source.suffix.lower() != ".gz":
            source_digest, source_bytes = snapshot_source(source, source_snapshot)
            input_adapter = {
                "schema": "vegai-training-data-adapter/v1",
                "source_format": "txt",
                "source_sha256": source_digest,
                "source_bytes": source_bytes,
                "canonical_hdc_text_sha256": source_digest,
                "canonical_hdc_text_bytes": source_bytes,
                "contract": "Strict TXT input is already the canonical native HDC corpus stream.",
            }
        else:
            source_digest, source_bytes, input_adapter = materialize_structured_input_as_hdc_text(
                source, source_snapshot, input_format
            )
        digest_index = create_digest_index(staging / ".exact-record-dedup.sqlite")
        provenance = {
            "source": arguments.source,
            "license": arguments.license,
            "retrieved_at": arguments.retrieved_at,
            "preprocessing": arguments.preprocessing,
        }
        train = ShardWriter(staging / "train", arguments.max_shard_bytes, arguments.max_shard_tokens)
        validation = (
            ShardWriter(staging / "validation", arguments.max_shard_bytes, arguments.max_shard_tokens)
            if arguments.validation_basis_points > 0
            else None
        )

        assert digest_index is not None
        source_records = 0
        source_empty_records = 0
        source_stats = PreflightStats()
        duplicate_records = 0
        accepted_records = 0
        for record_number, record in iter_strict_lf_records(source_snapshot, source):
            source_records += 1
            stats = strict_record_stats(record)
            violation = record_violation(stats)
            if violation is not None:
                raise DatasetBuildError(f"{source}: record {record_number} violates strict ingest contract ({violation})")
            add_stats(source_stats, stats)
            if stats.accepted_tokens == 0:
                # Empty/punctuation-only records teach nothing and would make a
                # zero-token shard invalid.  This explicit quality filter is
                # counted in dataset metadata rather than silently hidden.
                source_empty_records += 1
                continue
            digest = hashlib.sha256(record).digest()
            is_validation = split_is_validation(digest, arguments.validation_basis_points)
            split = "validation" if is_validation else "train"
            assert digest_index is not None
            if not insert_unique_record(digest_index, digest, split):
                duplicate_records += 1
                continue
            accepted_records += 1
            if is_validation:
                assert validation is not None
                validation.add(record, stats, digest)
            else:
                train.add(record, stats, digest)
                insert_train_tokens(digest_index, record)
            if accepted_records % 4096 == 0:
                digest_index.commit()
        digest_index.commit()
        train.finish()
        if validation is not None:
            validation.finish()

        # A normal source mutation is rejected rather than silently assigning
        # the caller's path a digest for different bytes.  The private snapshot
        # remains the immutable truth used for package construction.
        require(
            sha256_file(source) == input_adapter["source_sha256"]
            and source.stat().st_size == input_adapter["source_bytes"],
            f"source corpus changed while packaging: {source}",
        )
        require(train.entries, "deduplication/filtering left no training records")
        unique_training_tokens = training_unique_tokens(digest_index)
        if validation is not None:
            require(validation.entries, "validation split received no accepted records; adjust --validation-basis-points")

        train_ledger = staging / "train" / "ledger.json"
        atomic_write_json(train_ledger, train.ledger_payload(provenance))
        validation_metadata: dict[str, Any] | None = None
        if validation is not None:
            validation_ledger = staging / "validation" / "ledger.json"
            atomic_write_json(validation_ledger, validation.ledger_payload(provenance))
            validation_relative, validation_digest = relative_sha256(validation_ledger, staging)
            validation_metadata = {
                "ledger": validation_relative,
                "ledger_sha256": validation_digest,
                "records": validation.total.records,
                "accepted_tokens": validation.total.accepted_tokens,
                "shards": len(validation.entries),
                "record_digest_stream_sha256": validation.record_digests.hexdigest(),
            }

        train_relative, train_digest = relative_sha256(train_ledger, staging)
        dataset = {
            "schema": DATASET_SCHEMA,
            "declared_provenance": provenance,
            "source": {
                "input_filename": source.name,
                # This is the immutable strict-LF corpus snapshot actually
                # consumed by the existing HDC foundry. The raw structured
                # source identity remains in input_adapter below.
                "sha256": source_digest,
                "bytes": source_bytes,
                "physical_lf_records": source_records,
                "empty_or_punctuation_records_filtered": source_empty_records,
                "strict_preflight": preflight_dict(source_stats),
            },
            "input_adapter": input_adapter,
            "deduplication": {
                "mode": arguments.deduplicate,
                "identity": "sha256(raw-record-without-lf)",
                "duplicates_removed": duplicate_records,
                "retained_records": accepted_records,
                "sqlite_index_schema": DIGEST_INDEX_SCHEMA,
            },
            "split": {
                "algorithm": "sha256-first-64-bits-mod-10000/v1",
                "validation_basis_points": arguments.validation_basis_points,
                "train_validation_record_identity_disjoint": True,
                "contract": (
                    "Exact duplicate identities are removed before deterministic split assignment. "
                    "The builder writes each retained digest to exactly one split."
                ),
            },
            "vegai_ascii_ingest_contract": {
                "max_line_bytes": DEFAULT_MAX_LINE_BYTES,
                "token_length_bytes": DEFAULT_TOKEN_LENGTH,
                "max_tokens_per_line": DEFAULT_MAX_TOKENS_PER_LINE,
                "require_ascii_only": True,
                "require_lf_only": True,
                "require_no_dropped_tokens": True,
            },
            "train": {
                "ledger": train_relative,
                "ledger_sha256": train_digest,
                "records": train.total.records,
                "accepted_tokens": train.total.accepted_tokens,
                "unique_tokens_min_frequency_1": unique_training_tokens,
                "shards": len(train.entries),
                "record_digest_stream_sha256": train.record_digests.hexdigest(),
            },
            "validation": validation_metadata,
        }
        atomic_write_json(staging / "dataset.json", dataset)
        digest_index.close()
        digest_index = None
        (staging / ".exact-record-dedup.sqlite").unlink(missing_ok=True)
        source_snapshot.unlink(missing_ok=True)
        fsync_directory(staging / "train")
        if validation is not None:
            fsync_directory(staging / "validation")
        fsync_directory(staging)
        os.replace(staging, output)
        fsync_directory(output.parent)
        return dataset
    except BaseException:
        if digest_index is not None:
            digest_index.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path,
        help="local strict TXT corpus or schema-valid JSONL/JSON/Parquet/Arrow source to package",
    )
    parser.add_argument(
        "--input-format", choices=("auto", "jsonl", "json", "txt", "parquet", "arrow"), default="auto",
        help="source format (default: infer from extension; rich inputs are explicitly projected through text)",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="new immutable dataset directory to publish")
    parser.add_argument("--source", required=True, help="dataset/source identifier or URI recorded in immutable provenance")
    parser.add_argument("--license", required=True, help="license/usage identifier recorded in immutable provenance")
    parser.add_argument("--retrieved-at", required=True, help="source retrieval/version timestamp or immutable registry event")
    parser.add_argument("--preprocessing", required=True, help="versioned upstream normalization/tokenization contract")
    parser.add_argument(
        "--max-shard-bytes", type=int, default=64 * 1024 * 1024,
        help="maximum published shard payload bytes; never splits a physical record (default: 64 MiB)",
    )
    parser.add_argument(
        "--max-shard-tokens", type=int, default=1_000_000,
        help="maximum accepted tokenizer tokens per shard; never splits a physical record (default: 1000000)",
    )
    parser.add_argument(
        "--validation-basis-points", type=int, default=100,
        help="deterministic held-out record share in 1/10000 units; 100 means 1%%, 0 disables validation (default: 100)",
    )
    parser.add_argument(
        "--deduplicate", choices=("exact",), default="exact",
        help="required exact SHA-256 raw-record deduplication mode (default: exact)",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        dataset = build_dataset(arguments)
    except (DatasetBuildError, OSError, ValueError, sqlite3.Error) as error:
        print(f"production dataset: FAIL: {error}", file=sys.stderr)
        return 1
    validation = dataset["validation"]
    print(
        "production dataset: PASS "
        f"train_records={dataset['train']['records']} train_tokens={dataset['train']['accepted_tokens']} "
        f"train_unique_tokens={dataset['train']['unique_tokens_min_frequency_1']} "
        f"train_shards={dataset['train']['shards']} duplicates_removed={dataset['deduplication']['duplicates_removed']} "
        f"validation_records={0 if validation is None else validation['records']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
