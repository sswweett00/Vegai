#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and query a provenance-bound Unicode knowledge plane for LMM backends.

This is not a claim that Vegai has universal world knowledge.  It turns
operator-supplied, licensed documents into a deterministic SQLite FTS5 index
and emits explicit evidence bundles that an actual generative backend can
consume.  Every answer still requires backend-specific factual/safety
validation; this tool only makes source selection auditable and refreshable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
from typing import Any, Iterator, Mapping
import unicodedata

from lmm_contract import (
    EVIDENCE_SCHEMA,
    MAX_EVIDENCE_ITEMS,
    MAX_JSON_BYTES,
    MAX_TEXT_BYTES,
    EvidenceBundle,
    EvidenceItem,
    FileIdentity,
    LMMContractError,
    canonical_json_bytes,
    file_identity,
    require,
    require_unchanged,
    validate_identifier,
    validate_json_object,
    validate_unicode_text,
    write_json_atomic,
)


DOCUMENT_SCHEMA = "vegai-lmm-document/v1"
INDEX_SCHEMA = "vegai-lmm-knowledge-index/v1"
MAX_DOCUMENT_LINE_BYTES = 4 * 1024 * 1024
MAX_DOCUMENTS = 10_000_000


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    document_id: str
    text: str
    source: str
    license: str
    retrieved_at: str
    metadata: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: object, field: str) -> "KnowledgeDocument":
        require(isinstance(value, Mapping), f"{field} must be an object")
        required = {"schema_version", "id", "text", "source", "license", "retrieved_at", "metadata"}
        unknown = sorted(str(key) for key in value if key not in required)
        require(not unknown, f"{field} has unsupported fields: {', '.join(unknown)}")
        missing = sorted(required.difference(value))
        require(not missing, f"{field} is missing required fields: {', '.join(missing)}")
        require(value["schema_version"] == DOCUMENT_SCHEMA, f"{field}.schema_version is unsupported")
        return cls(
            document_id=validate_identifier(value["id"], f"{field}.id"),
            text=validate_unicode_text(value["text"], f"{field}.text", maximum_bytes=MAX_TEXT_BYTES),
            source=validate_unicode_text(value["source"], f"{field}.source", maximum_bytes=8 * 1024),
            license=validate_unicode_text(value["license"], f"{field}.license", maximum_bytes=8 * 1024),
            retrieved_at=validate_unicode_text(value["retrieved_at"], f"{field}.retrieved_at", maximum_bytes=1_024),
            metadata=validate_json_object(value["metadata"], f"{field}.metadata", maximum_bytes=64 * 1024),
        )

    def digest(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def iter_bounded_jsonl(path: Path, maximum_documents: int) -> Iterator[KnowledgeDocument]:
    require(maximum_documents > 0, "maximum_documents must be positive")
    source = path.resolve()
    identity = file_identity(source, "knowledge source", maximum_bytes=MAX_JSON_BYTES * 512)
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    count = 0
    try:
        with opener(source, "rb") as handle:
            number = 0
            limit = MAX_DOCUMENT_LINE_BYTES + 2
            while True:
                raw = handle.readline(limit)
                if not raw:
                    break
                number += 1
                has_lf = raw.endswith(b"\n")
                payload_bytes = len(raw) - 1 if has_lf else len(raw)
                overlong = payload_bytes > MAX_DOCUMENT_LINE_BYTES or (not has_lf and len(raw) == limit)
                if overlong and not has_lf:
                    while raw and not raw.endswith(b"\n"):
                        raw = handle.readline(limit)
                require(not overlong, f"knowledge source line {number} exceeds {MAX_DOCUMENT_LINE_BYTES} bytes")
                require(has_lf, f"knowledge source line {number} must terminate with LF")
                raw = raw[:-1]
                require(not raw.endswith(b"\r"), f"knowledge source line {number} uses CRLF; strict LF is required")
                require(bool(raw.strip()), f"knowledge source line {number} is blank")
                try:
                    mapping = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise LMMContractError(f"knowledge source line {number} is invalid JSON: {error}") from error
                document = KnowledgeDocument.from_mapping(mapping, f"knowledge source line {number}")
                count += 1
                require(count <= maximum_documents, f"knowledge source exceeds maximum_documents={maximum_documents}")
                yield document
    except (OSError, EOFError) as error:
        raise LMMContractError(f"cannot read knowledge source {source}: {error}") from error
    require_unchanged(source, identity, "knowledge source")


def fsync_directory(directory: Path) -> None:
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


def normalize_search_text(value: str) -> str:
    """Match SQLite unicode61's diacritic-insensitive behavior deterministically."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(character for character in decomposed if not unicodedata.combining(character))
    return unicodedata.normalize("NFC", without_marks)


def is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def cjk_terms(value: str) -> list[str]:
    """Add single-codepoint/bigram search tokens for scripts lacking spaces."""
    terms: list[str] = []
    run: list[str] = []
    def flush() -> None:
        if not run:
            return
        terms.extend(run)
        terms.extend("".join(run[index:index + 2]) for index in range(len(run) - 1))
        run.clear()
    for character in value:
        if is_cjk(character):
            run.append(character)
        else:
            flush()
    flush()
    return terms


def indexable_text(value: str) -> str:
    normalized = normalize_search_text(value)
    # The original normalized text keeps whitespace-delimited languages fast;
    # appended CJK terms make Japanese/Chinese/Korean substrings retrievable by
    # unicode61 without pretending it is a full morphology analyzer.
    return normalized + " " + " ".join(cjk_terms(normalized))


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE index_metadata (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE documents (
            id TEXT PRIMARY KEY NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            license TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            text_sha256 TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE VIRTUAL TABLE document_fts USING fts5(
            id UNINDEXED,
            text,
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )


def build_index(arguments: argparse.Namespace) -> dict[str, object]:
    source_identity = file_identity(arguments.input, "knowledge source", maximum_bytes=MAX_JSON_BYTES * 512)
    target = arguments.output.resolve()
    require(target != arguments.input.resolve(), "knowledge index output must not overwrite source JSONL")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".sqlite.tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    documents = 0
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            try:
                create_schema(connection)
            except sqlite3.OperationalError as error:
                raise LMMContractError(
                    "SQLite FTS5 with unicode61 is required for the knowledge index; use a Python/SQLite build with FTS5"
                ) from error
            metadata = {
                "schema_version": INDEX_SCHEMA,
                "source_sha256": source_identity.sha256,
                "source_bytes": str(source_identity.bytes),
                "tokenizer": "sqlite-fts5-unicode61-remove_diacritics=2+vegai-unicode-normalized-cjk-bigram",
            }
            connection.executemany("INSERT INTO index_metadata(key, value) VALUES (?, ?)", sorted(metadata.items()))
            with connection:
                for document in iter_bounded_jsonl(arguments.input, arguments.max_documents):
                    # The SQLite primary key performs duplicate detection without
                    # retaining every id in Python RAM (important for multi-
                    # million-document streaming knowledge packages).
                    try:
                        connection.execute(
                            "INSERT INTO documents(id, text, source, license, retrieved_at, metadata_json, text_sha256) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                document.document_id,
                                document.text,
                                document.source,
                                document.license,
                                document.retrieved_at,
                                canonical_json_bytes(document.metadata).decode("utf-8"),
                                document.digest(),
                            ),
                        )
                    except sqlite3.IntegrityError as error:
                        raise LMMContractError(f"knowledge source repeats id {document.document_id!r}") from error
                    connection.execute("INSERT INTO document_fts(id, text) VALUES (?, ?)", (document.document_id, indexable_text(document.text)))
                    documents += 1
            require(documents > 0, "knowledge source contained no documents")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            require(integrity == ("ok",), f"SQLite integrity check failed: {integrity}")
            connection.commit()
        finally:
            connection.close()
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # The JSONL reader proves stability while scanning. This second check
        # closes the interval between the build command's initial provenance
        # capture and publication, so a stale source identity can never be
        # attached to a newly published index.
        require_unchanged(arguments.input, source_identity, "knowledge source")
        os.replace(temporary, target)
        fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    output_identity = file_identity(target, "published knowledge index")
    return {
        "schema_version": "vegai-lmm-knowledge-build-report/v1",
        "documents": documents,
        "source": source_identity.to_mapping(),
        "index": output_identity.to_mapping(),
        "tokenizer": "sqlite-fts5-unicode61-remove_diacritics=2+vegai-unicode-normalized-cjk-bigram",
    }


def query_terms(value: str) -> list[str]:
    normalized = normalize_search_text(value)
    words = [term for term in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE) if term and not any(is_cjk(c) for c in term)]
    terms = words + cjk_terms(normalized)
    # Preserve deterministic order while removing duplicate terms, because an
    # AND query should not be inflated by repeated user wording.
    unique = list(dict.fromkeys(terms))
    require(unique, "query has no Unicode word tokens")
    require(len(unique) <= 64, "query has more than 64 lexical terms")
    return unique


def fts_query(value: str) -> str:
    # Quote each token so the user cannot inject FTS operators into a grounded
    # query policy. AND semantics favor evidence covering all supplied terms.
    return " AND ".join('"' + term.replace('"', '""') + '"' for term in query_terms(value))


def query_index(arguments: argparse.Namespace) -> dict[str, object]:
    query = validate_unicode_text(arguments.query, "--query", maximum_bytes=MAX_TEXT_BYTES)
    index_path = arguments.index.resolve()
    before = file_identity(index_path, "knowledge index")
    uri = f"{index_path.as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            metadata_rows = connection.execute("SELECT key, value FROM index_metadata").fetchall()
            metadata = {str(row["key"]): str(row["value"]) for row in metadata_rows}
            require(metadata.get("schema_version") == INDEX_SCHEMA, "knowledge index schema is unsupported")
            rows = connection.execute(
                """
                SELECT documents.id, documents.text, documents.source, documents.license, documents.text_sha256,
                       -bm25(document_fts) AS score
                FROM document_fts
                JOIN documents ON documents.id = document_fts.id
                WHERE document_fts MATCH ?
                ORDER BY score DESC, documents.id ASC
                LIMIT ?
                """,
                (fts_query(query), arguments.limit),
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise LMMContractError(f"cannot query knowledge index {index_path}: {error}") from error
    require_unchanged(index_path, before, "knowledge index")
    items = tuple(
        EvidenceItem(
            document_id=str(row["id"]),
            text=str(row["text"]),
            source=str(row["source"]),
            license=str(row["license"]),
            sha256=str(row["text_sha256"]),
            score=float(row["score"]),
        )
        for row in rows
    )
    bundle = EvidenceBundle(
        query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
        index_identity=before.to_mapping(),
        items=items,
    )
    payload = bundle.to_mapping()
    if arguments.output is not None:
        require(arguments.output.resolve() != index_path, "evidence output must not overwrite the knowledge index")
        write_json_atomic(arguments.output, payload)
    return payload


def verify_index(arguments: argparse.Namespace) -> dict[str, object]:
    path = arguments.index.resolve()
    before = file_identity(path, "knowledge index")
    uri = f"{path.as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            require(integrity == ("ok",), f"SQLite integrity check failed: {integrity}")
            schema = connection.execute("SELECT value FROM index_metadata WHERE key='schema_version'").fetchone()
            require(schema == (INDEX_SCHEMA,), "knowledge index schema is unsupported")
            count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()
            require(count is not None, "knowledge index document count is unavailable")
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise LMMContractError(f"cannot verify knowledge index {path}: {error}") from error
    require_unchanged(path, before, "knowledge index")
    return {
        "schema_version": "vegai-lmm-knowledge-verify-report/v1",
        "index": before.to_mapping(),
        "documents": int(count[0]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="atomically build a Unicode FTS5 index from strict document JSONL")
    build.add_argument("--input", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--max-documents", type=int, default=MAX_DOCUMENTS)
    query = subcommands.add_parser("query", help="create a provenance-bound evidence bundle")
    query.add_argument("--index", required=True, type=Path)
    query.add_argument("--query", required=True)
    query.add_argument("--limit", type=int, default=8)
    query.add_argument("--output", type=Path)
    verify = subcommands.add_parser("verify", help="run SQLite integrity and schema checks")
    verify.add_argument("--index", required=True, type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        if arguments.command == "build":
            require(arguments.max_documents > 0, "--max-documents must be positive")
            payload = build_index(arguments)
        elif arguments.command == "query":
            require(arguments.limit > 0 and arguments.limit <= MAX_EVIDENCE_ITEMS,
                    f"--limit must be in [1, {MAX_EVIDENCE_ITEMS}]")
            payload = query_index(arguments)
        else:
            payload = verify_index(arguments)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except (LMMContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"lmm knowledge: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
