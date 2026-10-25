#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and run one auditable 9 GB causal-LM epoch on COSMOS Turkish text.

This controller is intentionally narrow.  It does not accept an arbitrary URL
or quietly fall back to a smaller/synthetic corpus.  A run succeeds only when
all of these conditions hold:

* the resolved Hugging Face dataset is exactly the declared COSMOS release;
* its resolved immutable revision and dataset card are captured before shards
  are downloaded, and the card declares CC-BY-4.0;
* 9,000,000,000 decimal bytes of unique, transformed-but-source-derived
  Turkish records are constructed without cycling/repeating records;
* a source-document-disjoint exact held-out corpus is constructed separately;
* exactly one native ``vegai_causal_lm train`` child is launched; and
* only the native causal-update epoch timer reported by that child is used for
  training throughput (not compilation, download, preparation, evaluation,
  checkpoint save, or CI orchestration).

The native model is a bounded causal heavy-hitter n-gram model.  It genuinely
predicts/generates next tokens but is not represented as a neural LLM or as a
factual/instruction-following assistant.  The final report preserves that
boundary next to real generated prompt/response evidence.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import unicodedata
from typing import Any, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "vegai-cosmos-turkish-9gb-causal-run/v1"
DATASET_REPOSITORY = "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0"
DATASET_DISPLAY_NAME = "Cosmos-Turkish-Corpus-v1.0"
EXPECTED_LICENSE = "cc-by-4.0"
TRAIN_BYTES = 9_000_000_000
HELDOUT_BYTES = 50_000_000
MAX_RECORD_BYTES = 4096
MIN_DOCUMENT_CHARS = 96
MIN_ALPHA_RATIO = 0.45
# These are intentionally non-source-derived prompts.  They inspect a real
# generated continuation without placing held-out corpus text in reports.
GENERATION_PROMPTS = (
    "Türkiye'nin başkenti",
    "Yapay zeka verilerden",
    "Bilimsel bir açıklama için",
    "Trabzon Karadeniz kıyısında",
    "Türkçe bir cümlede",
)

_EMAIL = re.compile(r"(?iu)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b")
# Conservative phone/IP patterns are rejection criteria, not sanitizers.  A
# document with a direct identifier is never transformed into a synthetic row.
_PHONE = re.compile(r"(?x)(?<!\w)(?:\+?90[ .\-]?)?(?:\(?0?\d{3}\)?[ .\-]?)\d{3}[ .\-]?\d{2}[ .\-]?\d{2}(?!\w)")
_IP = re.compile(r"(?<![\w.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\w.])")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE = re.compile(r"\s+")


class CosmosRunError(RuntimeError):
    """Expected provenance, data, or native-child failure."""


@dataclass(frozen=True)
class SourceShard:
    filename: str
    advertised_bytes: int | None


@dataclass
class PreparationCounters:
    source_rows_seen: int = 0
    rows_kept: int = 0
    rows_rejected_empty_or_short: int = 0
    rows_rejected_quality: int = 0
    rows_rejected_direct_identifier: int = 0
    rows_rejected_duplicate_document: int = 0
    rows_rejected_duplicate_record: int = 0
    records_written: int = 0
    records_final_fragment: int = 0
    source_shards_consumed: list[dict[str, Any]] = field(default_factory=list)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CosmosRunError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as destination:
            json.dump(payload, destination, ensure_ascii=True, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fsync_directory(directory: Path) -> None:
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


def ensure_empty_directory(path: Path) -> None:
    if path.exists():
        require(path.is_dir(), f"run directory is not a directory: {path}")
        require(not any(path.iterdir()), f"run directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


def resolve_executable(path: Path) -> Path:
    resolved = path.resolve()
    require(resolved.is_file(), f"causal binary is not a regular file: {resolved}")
    require(os.access(resolved, os.X_OK), f"causal binary is not executable: {resolved}")
    return resolved


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
    except OSError:
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def load_huggingface_dependencies() -> tuple[Any, Any, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore[import-not-found]
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as error:
        raise CosmosRunError(
            "COSMOS source access requires huggingface_hub and pyarrow; install requirements-cosmos-9gb.txt"
        ) from error
    return HfApi, hf_hub_download, parquet


def parse_declared_license(card: str) -> str:
    """Read the canonical Hugging Face card front matter without PyYAML."""
    if not card.startswith("---"):
        raise CosmosRunError("dataset card has no YAML front matter; cannot verify license")
    closing = card.find("\n---", 3)
    if closing < 0:
        raise CosmosRunError("dataset card front matter is not terminated; cannot verify license")
    front_matter = card[3:closing]
    match = re.search(r"(?im)^license\s*:\s*['\"]?([^\s'\"#]+)", front_matter)
    if not match:
        raise CosmosRunError("dataset card does not declare a license in front matter")
    return match.group(1).strip().lower()


def source_metadata(download_root: Path) -> tuple[str, str, list[SourceShard], dict[str, Any]]:
    """Resolve one immutable source revision and validate the dataset contract."""
    HfApi, hf_hub_download, _ = load_huggingface_dependencies()
    api = HfApi()
    try:
        info = api.dataset_info(DATASET_REPOSITORY, revision="main", files_metadata=True)
    except Exception as error:  # API exception types vary between hub versions.
        raise CosmosRunError(f"cannot resolve COSMOS dataset metadata: {error}") from error
    revision = str(getattr(info, "sha", "")).strip()
    require(re.fullmatch(r"[0-9a-f]{40,64}", revision) is not None,
            "dataset API did not return an immutable hexadecimal revision")
    try:
        card_path = Path(hf_hub_download(
            repo_id=DATASET_REPOSITORY,
            repo_type="dataset",
            filename="README.md",
            revision=revision,
            local_dir=str(download_root),
        ))
    except Exception as error:
        raise CosmosRunError(f"cannot download COSMOS dataset card at {revision}: {error}") from error
    try:
        card = card_path.read_text(encoding="utf-8")
    except OSError as error:
        raise CosmosRunError(f"cannot read downloaded COSMOS dataset card: {error}") from error
    declared_license = parse_declared_license(card)
    require(declared_license == EXPECTED_LICENSE,
            f"dataset card license must be {EXPECTED_LICENSE}, found {declared_license!r}")

    siblings = list(getattr(info, "siblings", []) or [])
    shards: list[SourceShard] = []
    for sibling in siblings:
        filename = str(getattr(sibling, "rfilename", ""))
        if filename.lower().endswith(".parquet"):
            value = getattr(sibling, "size", None)
            advertised = int(value) if isinstance(value, int) and value >= 0 else None
            shards.append(SourceShard(filename=filename, advertised_bytes=advertised))
    shards.sort(key=lambda shard: shard.filename)
    require(shards, "COSMOS resolved revision contains no parquet shards")
    known_bytes = sum(shard.advertised_bytes or 0 for shard in shards)
    # This is a coarse preflight only; the exact final train corpus is verified
    # after parse/quality filtering.  It prevents a suddenly tiny release from
    # being misleadingly called a 9 GB corpus.
    require(known_bytes >= 10_000_000_000,
            f"COSMOS advertised parquet bytes are insufficient for 9 GB construction: {known_bytes}")
    metadata = {
        "repository": DATASET_REPOSITORY,
        "display_name": DATASET_DISPLAY_NAME,
        "resolved_revision": revision,
        "dataset_card_filename": "README.md",
        "dataset_card_sha256": sha256_file(card_path),
        "declared_license": declared_license,
        "advertised_parquet_bytes": known_bytes,
        "advertised_parquet_shards": len(shards),
        "metadata_api": {
            "id": str(getattr(info, "id", "")),
            "sha": revision,
            "last_modified": str(getattr(info, "last_modified", "")),
        },
    }
    return revision, card, shards, metadata


def choose_text_column(schema: Any) -> str:
    """Fail closed rather than guessing a non-text feature of an evolving set."""
    candidates = ("text", "content", "document", "body", "article")
    names = [str(name) for name in schema.names]
    lower = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lower:
            field = schema.field(lower[candidate])
            # PyArrow's string and large_string textual types stringify with
            # stable names across supported versions.
            if str(field.type) in {"string", "large_string"}:
                return lower[candidate]
    rendered = ", ".join(f"{name}:{schema.field(name).type}" for name in names[:32])
    raise CosmosRunError(f"cannot locate an approved UTF-8 text column; schema is {rendered}")


def iter_parquet_text(path: Path, parquet: Any) -> Iterator[str]:
    try:
        source = parquet.ParquetFile(path)
        column = choose_text_column(source.schema_arrow)
        for batch in source.iter_batches(batch_size=8192, columns=[column], use_threads=True):
            for value in batch.column(0).to_pylist():
                if isinstance(value, str):
                    yield value
    except CosmosRunError:
        raise
    except Exception as error:
        raise CosmosRunError(f"cannot parse parquet shard {path.name}: {error}") from error


def canonicalize_document(text: str) -> str | None:
    """Minimal deterministic quality/privacy gate that never invents text."""
    # NFC makes byte identity canonical without transliteration or translation.
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\u00ad", "")
    if _CONTROL.search(normalized):
        return None
    normalized = _SPACE.sub(" ", normalized).strip()
    if len(normalized) < MIN_DOCUMENT_CHARS:
        return None
    if _EMAIL.search(normalized) or _PHONE.search(normalized) or _IP.search(normalized):
        return None
    alphabetic = sum(character.isalpha() for character in normalized)
    visible = sum(not character.isspace() for character in normalized)
    if visible <= 0 or alphabetic / visible < MIN_ALPHA_RATIO:
        return None
    # This corpus is declared Turkish.  We use a deliberately weak structural
    # check to reject mostly non-Latin garbage while avoiding a fabricated
    # language classifier claim.  Source-level language verification remains
    # the documented COSMOS provenance responsibility.
    latin = sum("LATIN" in unicodedata.name(character, "") for character in normalized if character.isalpha())
    if alphabetic > 0 and latin / alphabetic < 0.70:
        return None
    return normalized


def split_utf8_records(text: str, maximum_bytes: int = MAX_RECORD_BYTES) -> Iterator[bytes]:
    """Split source text only at whitespace, preserving every emitted byte."""
    encoded = text.encode("utf-8")
    require(len(encoded) > 0, "empty canonical document cannot be segmented")
    if len(encoded) <= maximum_bytes:
        yield encoded
        return
    words = text.split(" ")
    current: list[str] = []
    current_size = 0
    for word in words:
        word_bytes = word.encode("utf-8")
        if len(word_bytes) > maximum_bytes:
            # A pathological huge lexical item cannot be made valid without
            # cutting a word.  Reject rather than introduce arbitrary pieces.
            if current:
                yield " ".join(current).encode("utf-8")
            return
        addition = len(word_bytes) if not current else len(word_bytes) + 1
        if current and current_size + addition > maximum_bytes:
            yield " ".join(current).encode("utf-8")
            current = [word]
            current_size = len(word_bytes)
        else:
            current.append(word)
            current_size += addition
    if current:
        yield " ".join(current).encode("utf-8")


def valid_utf8_boundary(data: bytes, position: int) -> bool:
    if position < 0 or position > len(data):
        return False
    if position == 0 or position == len(data):
        return True
    return (data[position] & 0xC0) != 0x80


def exact_utf8_substring(data: bytes, target_bytes: int) -> bytes | None:
    """Find a nonempty source-contiguous UTF-8 substring of exactly N bytes."""
    if target_bytes <= 0 or target_bytes > len(data):
        return None
    # At most a 4 KiB prepared record: an exhaustive boundary scan is cheap and
    # avoids fabricated padding when the decimal byte boundary lands mid-row.
    for start in range(0, len(data) - target_bytes + 1):
        end = start + target_bytes
        if valid_utf8_boundary(data, start) and valid_utf8_boundary(data, end):
            candidate = data[start:end]
            try:
                candidate.decode("utf-8")
            except UnicodeDecodeError:
                continue
            return candidate
    return None


class ExactRecordWriter:
    """Exact-size source-derived LF writer with ordered leaf-hash commitment."""

    def __init__(self, path: Path, target_bytes: int, connection: sqlite3.Connection, split: str) -> None:
        require(target_bytes > 1, "exact corpus target must leave room for a source byte and LF")
        self.path = path
        self.target_bytes = target_bytes
        self.connection = connection
        self.split = split
        self.handle = path.open("xb")
        self.bytes_written = 0
        self.records_written = 0
        self.final_fragment_records = 0
        self._ordered_leaves = hashlib.sha256()
        self.completed = False

    @property
    def remaining(self) -> int:
        return self.target_bytes - self.bytes_written

    def _record_seen(self, record: bytes) -> bool:
        digest = hashlib.sha256(record).digest()
        try:
            self.connection.execute("INSERT INTO record_hashes(digest, split) VALUES (?, ?)", (digest, self.split))
        except sqlite3.IntegrityError:
            return True
        return False

    def _commit(self, record: bytes, *, final_fragment: bool = False) -> bool:
        require(record and b"\n" not in record and b"\r" not in record, "invalid prepared record")
        if self._record_seen(record):
            return False
        payload = record + b"\n"
        require(len(payload) <= self.remaining, "internal exact writer overflow")
        self.handle.write(payload)
        self.bytes_written += len(payload)
        self.records_written += 1
        self.final_fragment_records += int(final_fragment)
        leaf = hashlib.sha256(len(record).to_bytes(8, "big") + record).digest()
        self._ordered_leaves.update(leaf)
        if self.bytes_written == self.target_bytes:
            self.completed = True
        return True

    def offer(self, record: bytes) -> str:
        """Offer one source record; return written/duplicate/needs-final/complete."""
        if self.completed:
            return "complete"
        require(record and len(record) <= MAX_RECORD_BYTES, "prepared record violates size contract")
        payload_size = len(record) + 1
        if payload_size <= self.remaining:
            # Never leave a one-byte remainder: it cannot represent a record
            # plus LF.  A subsequent different source row may fit instead.
            if self.remaining - payload_size == 1:
                return "needs-final"
            return "written" if self._commit(record) else "duplicate"
        target_record_bytes = self.remaining - 1
        final = exact_utf8_substring(record, target_record_bytes)
        if final is None:
            return "needs-final"
        return "written" if self._commit(final, final_fragment=(final != record)) else "duplicate"

    def finish(self) -> dict[str, Any]:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        require(self.completed and self.bytes_written == self.target_bytes,
                f"{self.split} writer did not reach exact target: {self.bytes_written}/{self.target_bytes}")
        return {
            "path": self.path.name,
            "bytes": self.bytes_written,
            "sha256": sha256_file(self.path),
            "records": self.records_written,
            "final_fragment_records": self.final_fragment_records,
            "ordered_record_leaf_sha256": self._ordered_leaves.hexdigest(),
        }


def initialize_dedup_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE document_hashes (digest BLOB PRIMARY KEY, split TEXT NOT NULL)")
    connection.execute("CREATE TABLE record_hashes (digest BLOB PRIMARY KEY, split TEXT NOT NULL)")
    return connection


def document_seen(connection: sqlite3.Connection, document: str, split: str) -> bool:
    digest = hashlib.sha256(document.encode("utf-8")).digest()
    try:
        connection.execute("INSERT INTO document_hashes(digest, split) VALUES (?, ?)", (digest, split))
    except sqlite3.IntegrityError:
        return True
    return False


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def check_disk_preflight(run_dir: Path) -> None:
    # 9 GB train + 50 MB held-out + checkpoint/cache/dedupe headroom.  Shards
    # are downloaded one at a time then removed, so full 22.8 GB is not needed.
    minimum = TRAIN_BYTES + HELDOUT_BYTES + 4_000_000_000
    free = disk_free_bytes(run_dir)
    require(free >= minimum,
            f"insufficient free storage for exact corpus/checkpoint headroom: {free} < {minimum}")


def remove_downloaded_file(path: Path, root: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    # Hugging Face local-dir metadata may leave nested empty parent directories.
    for parent in path.parents:
        if parent == root.parent:
            break
        if parent == root:
            break
        try:
            parent.rmdir()
        except OSError:
            break


def build_corpora(run_dir: Path, revision: str, shards: Sequence[SourceShard], metadata: dict[str, Any]) -> dict[str, Any]:
    """Stream source shards into exact train/held-out records, no cycling."""
    _, hf_hub_download, parquet = load_huggingface_dependencies()
    train_path = run_dir / "train-9gb.txt"
    heldout_path = run_dir / "heldout-source-disjoint.txt"
    download_root = run_dir / "download"
    download_root.mkdir()
    dedup_path = run_dir / "dedup.sqlite3"
    connection = initialize_dedup_database(dedup_path)
    train_writer = ExactRecordWriter(train_path, TRAIN_BYTES, connection, "train")
    heldout_writer = ExactRecordWriter(heldout_path, HELDOUT_BYTES, connection, "heldout")
    train_counters = PreparationCounters()
    heldout_counters = PreparationCounters()

    try:
        for shard in shards:
            if train_writer.completed and heldout_writer.completed:
                break
            try:
                downloaded = Path(hf_hub_download(
                    repo_id=DATASET_REPOSITORY,
                    repo_type="dataset",
                    filename=shard.filename,
                    revision=revision,
                    local_dir=str(download_root),
                ))
            except Exception as error:
                raise CosmosRunError(f"cannot download COSMOS shard {shard.filename}: {error}") from error
            local_size = downloaded.stat().st_size
            shard_identity = {
                "filename": shard.filename,
                "advertised_bytes": shard.advertised_bytes,
                "downloaded_bytes": local_size,
                "sha256": sha256_file(downloaded),
            }
            # Both split counter lists intentionally identify source shards but
            # never claim a raw shard is itself entirely train or held-out.
            train_counters.source_shards_consumed.append(shard_identity)
            heldout_counters.source_shards_consumed.append(shard_identity)
            for raw_text in iter_parquet_text(downloaded, parquet):
                # Fill held-out first so it is source-document-disjoint and is
                # never touched by the training writer.
                target_writer: ExactRecordWriter
                target_counters: PreparationCounters
                if not heldout_writer.completed:
                    target_writer, target_counters = heldout_writer, heldout_counters
                elif not train_writer.completed:
                    target_writer, target_counters = train_writer, train_counters
                else:
                    break
                target_counters.source_rows_seen += 1
                canonical = canonicalize_document(raw_text)
                if canonical is None:
                    # Classify the reason without retaining text.
                    raw_normal = unicodedata.normalize("NFC", raw_text)
                    if _EMAIL.search(raw_normal) or _PHONE.search(raw_normal) or _IP.search(raw_normal):
                        target_counters.rows_rejected_direct_identifier += 1
                    elif len(_SPACE.sub(" ", raw_normal).strip()) < MIN_DOCUMENT_CHARS:
                        target_counters.rows_rejected_empty_or_short += 1
                    else:
                        target_counters.rows_rejected_quality += 1
                    continue
                if document_seen(connection, canonical, target_writer.split):
                    target_counters.rows_rejected_duplicate_document += 1
                    continue
                wrote_any = False
                for record in split_utf8_records(canonical):
                    outcome = target_writer.offer(record)
                    if outcome == "duplicate":
                        target_counters.rows_rejected_duplicate_record += 1
                    elif outcome == "written":
                        wrote_any = True
                        target_counters.records_written += 1
                    # ``needs-final`` leaves this source document admitted but
                    # unused; a later record with an exact UTF-8 substring
                    # completes the decimal boundary without padding/repetition.
                    if target_writer.completed:
                        break
                if wrote_any:
                    target_counters.rows_kept += 1
                if train_writer.completed and heldout_writer.completed:
                    break
            connection.commit()
            remove_downloaded_file(downloaded, download_root)
        require(heldout_writer.completed, "source ended before exact held-out corpus target was reached")
        require(train_writer.completed, "source ended before exact 9 GB training corpus target was reached")
        train_manifest = train_writer.finish()
        heldout_manifest = heldout_writer.finish()
    except BaseException:
        train_writer.handle.close()
        heldout_writer.handle.close()
        raise
    finally:
        connection.close()

    require(train_path.stat().st_size == TRAIN_BYTES, "train corpus filesystem size violates exact decimal target")
    require(heldout_path.stat().st_size == HELDOUT_BYTES, "held-out corpus filesystem size violates exact target")
    # Keep only integrity indices rather than source data beyond corpus files.
    metadata["source_shards_consumed"] = [entry["filename"] for entry in train_counters.source_shards_consumed]
    return {
        "schema_version": "vegai-cosmos-turkish-corpus-build/v1",
        "normalization": {
            "unicode": "NFC",
            "whitespace": "all Unicode whitespace collapsed to one ASCII space",
            "text_transliteration": "none",
            "max_record_bytes": MAX_RECORD_BYTES,
            "record_split": "source order, whitespace boundary only",
            "exact_final_boundary": "one source-contiguous UTF-8 substring; no synthetic padding or cycling",
        },
        "privacy_filter": {
            "method": "reject records containing direct email, conservative Turkish phone, or IPv4 patterns",
            "limitation": "this heuristic is not a guarantee that the public source contains no personal data",
        },
        "quality_filter": {
            "minimum_document_characters": MIN_DOCUMENT_CHARS,
            "minimum_alphabetic_visible_ratio": MIN_ALPHA_RATIO,
            "minimum_latin_alphabetic_ratio": 0.70,
            "limitation": "structural quality checks are not an independent Turkish-language classifier",
        },
        "deduplication": {
            "document": "SHA-256 of normalized full source document across both partitions",
            "record": "SHA-256 of emitted normalized record across both partitions",
            "limitation": "exact-content deduplication; it does not claim near-duplicate elimination",
        },
        "train": train_manifest,
        "heldout": heldout_manifest,
        "source_document_disjoint": True,
        "counters": {
            "train": counters_payload(train_counters),
            "heldout": counters_payload(heldout_counters),
        },
        "metadata": metadata,
    }


def counters_payload(counters: PreparationCounters) -> dict[str, Any]:
    return {
        "source_rows_seen": counters.source_rows_seen,
        "rows_kept": counters.rows_kept,
        "rows_rejected_empty_or_short": counters.rows_rejected_empty_or_short,
        "rows_rejected_quality": counters.rows_rejected_quality,
        "rows_rejected_direct_identifier": counters.rows_rejected_direct_identifier,
        "rows_rejected_duplicate_document": counters.rows_rejected_duplicate_document,
        "rows_rejected_duplicate_record": counters.rows_rejected_duplicate_record,
        "records_written": counters.records_written,
        "final_fragment_records": counters.records_final_fragment,
        "source_shards_consumed": [entry["filename"] for entry in counters.source_shards_consumed],
    }


def extract_json_report(output: str, phase: str) -> dict[str, Any]:
    prefix = "CAUSAL_LM_JSON: "
    candidates: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line[len(prefix):])
        except json.JSONDecodeError as error:
            raise CosmosRunError(f"native causal binary emitted invalid JSON report: {error}") from error
        if isinstance(payload, dict):
            candidates.append(payload)
    for payload in reversed(candidates):
        if payload.get("phase") == phase:
            return payload
    raise CosmosRunError(f"native causal binary emitted no {phase!r} JSON report")


def run_child(command: Sequence[str], label: str, timeout_seconds: int) -> tuple[str, float]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command), cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise CosmosRunError(f"{label} exceeded timeout of {timeout_seconds} seconds") from error
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise CosmosRunError(f"{label} failed with status {result.returncode}\n{result.stdout[-16_384:]}")
    return result.stdout, elapsed


def parse_response(output: str) -> str:
    responses = [line[len("CAUSAL_RESPONSE: "):] for line in output.splitlines() if line.startswith("CAUSAL_RESPONSE: ")]
    if len(responses) != 1:
        raise CosmosRunError("native generation child did not emit exactly one response")
    return responses[0]


def run_native_pipeline(arguments: argparse.Namespace, run_dir: Path, corpus: dict[str, Any]) -> dict[str, Any]:
    causal_bin = resolve_executable(arguments.causal_bin)
    train_path = run_dir / corpus["train"]["path"]
    heldout_path = run_dir / corpus["heldout"]["path"]
    checkpoint = run_dir / "cosmos-turkish-causal.vclm"
    command = [
        str(causal_bin), "train", str(train_path), str(checkpoint),
        "--max-vocab", str(arguments.max_vocab),
        "--top-successors", str(arguments.top_successors),
        "--trigram-buckets", str(arguments.trigram_buckets),
        "--trigram-ways", str(arguments.trigram_ways),
        "--min-estimated-frequency", str(arguments.min_estimated_frequency),
    ]
    train_stdout, child_wall_seconds = run_child(command, "native causal training child", arguments.train_timeout_seconds)
    (run_dir / "native-train.stdout.txt").write_text(train_stdout, encoding="utf-8")
    train_report = extract_json_report(train_stdout, "train")
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "native causal train child did not publish checkpoint")
    require(int(train_report.get("corpus_bytes", -1)) == TRAIN_BYTES,
            "native causal child did not consume exact 9 GB training corpus")
    raw_tokens = int(train_report.get("raw_tokens", -1))
    require(raw_tokens > 0, "native causal child reported no raw tokens")

    # Training-only timing: the native executable's timer surrounds exactly the
    # one causal update pass.  The subprocess timer is audit-only and never
    # used as throughput numerator/denominator.
    epoch_seconds = float(train_report.get("training_epoch_wall_seconds", math.nan))
    require(math.isfinite(epoch_seconds) and epoch_seconds > 0.0,
            "native causal child did not publish a positive training epoch timer")
    native_tokens_per_second = raw_tokens / epoch_seconds

    evaluation_stdout, evaluation_child_wall_seconds = run_child(
        [str(causal_bin), "evaluate", str(checkpoint), str(heldout_path)],
        "native held-out causal evaluation", arguments.evaluation_timeout_seconds,
    )
    (run_dir / "native-evaluate.stdout.txt").write_text(evaluation_stdout, encoding="utf-8")
    evaluation = extract_json_report(evaluation_stdout, "evaluate")
    require(int(evaluation.get("records", -1)) > 0, "held-out evaluator read no source-disjoint records")

    generations: list[dict[str, Any]] = []
    for index, prompt in enumerate(GENERATION_PROMPTS):
        generation_stdout, _ = run_child(
            [str(causal_bin), "generate", str(checkpoint), prompt, "--tokens", str(arguments.generation_tokens),
             "--temperature", str(arguments.temperature), "--seed", str(arguments.seed + index)],
            f"native generation {index + 1}", arguments.generation_timeout_seconds,
        )
        generations.append({"prompt": prompt, "response": parse_response(generation_stdout), "seed": arguments.seed + index})

    return {
        "native_binary": {
            "path": causal_bin.name,
            "sha256": sha256_file(causal_bin),
        },
        "training": {
            "command": command[1:],
            "exact_epochs": 1,
            "native_train_report": train_report,
            "native_training_epoch_wall_seconds": epoch_seconds,
            "raw_training_tokens_per_second": native_tokens_per_second,
            "subprocess_wall_seconds_audit_only": child_wall_seconds,
            "timing_scope": "only the in-native causal_lm_train_file update pass; excludes corpus preparation, vocabulary scan, checkpoint save, evaluation, generation, compile, and CI orchestration",
        },
        "checkpoint": {
            "path": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "heldout_evaluation": {
            "source_document_disjoint": True,
            "report": evaluation,
            "subprocess_wall_seconds_not_training": evaluation_child_wall_seconds,
            "metric_scope": "top-1 and bounded-successor coverage for the causal n-gram sketch; neither is factuality, instruction following, nor broad Turkish quality",
        },
        "generated_prompt_responses": generations,
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-bin", type=Path, required=True, help="compiled vegai_causal_lm executable")
    parser.add_argument("--run-dir", type=Path, required=True, help="new empty ignored runtime directory")
    parser.add_argument("--max-vocab", type=int, default=65_538)
    parser.add_argument("--top-successors", type=int, default=12)
    parser.add_argument("--trigram-buckets", type=int, default=131_072)
    parser.add_argument("--trigram-ways", type=int, default=4)
    parser.add_argument("--min-estimated-frequency", type=int, default=2)
    parser.add_argument("--train-timeout-seconds", type=int, default=21_000)
    parser.add_argument("--evaluation-timeout-seconds", type=int, default=3_600)
    parser.add_argument("--generation-timeout-seconds", type=int, default=120)
    parser.add_argument("--generation-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20261025)
    return parser.parse_args(argv)


def validate_arguments(arguments: argparse.Namespace) -> None:
    require(arguments.max_vocab >= 10 and (arguments.max_vocab - 2) % 4 == 0,
            "--max-vocab must be at least 10 and leave a multiple of four candidate slots after BOS/EOS")
    require(2 <= arguments.top_successors <= 64, "--top-successors must be in [2, 64]")
    require(arguments.trigram_buckets > 0 and 1 <= arguments.trigram_ways <= 8,
            "trigram cache arguments are invalid")
    require(arguments.min_estimated_frequency >= 1, "--min-estimated-frequency must be positive")
    require(arguments.train_timeout_seconds > 0 and arguments.evaluation_timeout_seconds > 0 and
            arguments.generation_timeout_seconds > 0, "timeouts must be positive")
    require(1 <= arguments.generation_tokens <= 512 and math.isfinite(arguments.temperature) and arguments.temperature > 0.0,
            "generation controls are invalid")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        validate_arguments(arguments)
        run_dir = arguments.run_dir.resolve()
        ensure_empty_directory(run_dir)
        check_disk_preflight(run_dir)
        started_at = time.time()
        revision, _card, shards, metadata = source_metadata(run_dir / "dataset-card")
        corpus = build_corpora(run_dir, revision, shards, metadata)
        atomic_json(run_dir / "corpus-manifest.json", corpus)
        native = run_native_pipeline(arguments, run_dir, corpus)
        result = {
            "schema_version": SCHEMA,
            "status": "completed",
            "dataset": metadata,
            "corpus": corpus,
            "native": native,
            "source_and_capability_boundary": {
                "source": "real public COSMOS Turkish continual-pretraining corpus; CC-BY-4.0 card verified at resolved immutable revision",
                "model": "native bounded causal word n-gram heavy-hitter sketch, distinct from existing HDC association and TSV retrieval paths",
                "not_claimed": [
                    "neural Transformer equivalence",
                    "general intelligence",
                    "factual reliability",
                    "instruction following",
                    "absence of all personal data",
                    "near-duplicate-free source",
                ],
            },
            "controller": {
                "path": "tools/run_cosmos_turkish_9gb.py",
                "sha256": sha256_file(Path(__file__).resolve()),
                "git_revision": git_revision(),
                "started_unix_seconds": started_at,
                "finished_unix_seconds": time.time(),
            },
        }
        atomic_json(run_dir / "result.json", result)
        print(json.dumps({
            "status": "completed",
            "result": str(run_dir / "result.json"),
            "train_bytes": corpus["train"]["bytes"],
            "checkpoint_sha256": native["checkpoint"]["sha256"],
        }, ensure_ascii=True, sort_keys=True))
        return 0
    except CosmosRunError as error:
        # Keep a bounded failure provenance document when the run directory was
        # successfully created.  It deliberately excludes raw source text.
        try:
            if "run_dir" in locals() and run_dir.is_dir():
                atomic_json(run_dir / "failure.json", {
                    "schema_version": SCHEMA,
                    "status": "failed",
                    "failure": str(error),
                    "controller_git_revision": git_revision(),
                })
        except Exception:
            pass
        print(f"run_cosmos_turkish_9gb: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
