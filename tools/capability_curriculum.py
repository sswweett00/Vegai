#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile provenance-bound, data-driven capability curricula for Vegai.

This is a data-control plane for higher-quality target-domain HDC training, not
a hardcoded answer generator and not a claim that an HDC checkpoint becomes a
universal reasoning, coding, or language model.  An operator supplies the real
licensed records and an immutable policy that names its own capability tags,
difficulty annotation paths, staged curriculum, and held-out coverage gates.

The compiler streams JSONL/JSON/TXT/Parquet/Arrow records through the existing
``vegai-training-record/v1`` reader. Bare TXT deliberately fails this workflow
because it has no metadata annotations. It never inserts labels, capabilities,
solutions, prompts, or templates into ``record.text``. It writes only selected
source text to strict ASCII/LF stage shards and emits a standard
``vegai-continuous-shard-manifest/v1`` ledger consumable by
``tools/continuous_pretrain.py``.

Safety properties:
* source and policy are hash-bound and rechecked before atomic publication;
* exact text duplicates are SQLite-backed and conflicting split/annotation
  declarations fail closed;
* train/validation/test assignment is explicit or deterministic by text hash;
* every declared stage has configurable train and held-out coverage minima;
* metadata never becomes HDC training text; and
* output reports retain hashes and counts, not raw examples or held-out text.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
from typing import Any, Mapping, Sequence

from corpus_manifest import (
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_TOKENS_PER_LINE,
    DEFAULT_TOKEN_LENGTH,
    PreflightStats,
    is_token_byte,
    scan_corpus,
)
from training_data_io import (
    DEFAULT_BATCH_SIZE,
    TrainingDataError,
    TrainingRecord,
    infer_format,
    iter_records,
    split_for_text_digest,
    validate_hdc_record,
)


POLICY_SCHEMA = "vegai-capability-curriculum-policy/v1"
PACKAGE_SCHEMA = "vegai-capability-curriculum/v1"
LEDGER_SCHEMA = "vegai-continuous-shard-manifest/v1"
MAX_STAGES = 64
MAX_CAPABILITIES_PER_RECORD = 128
MAX_PATH_SEGMENTS = 16
MAX_IDENTIFIER_BYTES = 128
CHUNK_BYTES = 1024 * 1024
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class CapabilityCurriculumError(RuntimeError):
    """An operator policy, source record, integrity check, or publication failed."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    sha256: str
    bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class StagePolicy:
    stage_id: str
    required_capabilities: frozenset[str]
    minimum_difficulty: float
    maximum_difficulty: float
    minimum_train_records: int
    minimum_heldout_records: int
    minimum_distinct_train_sources: int
    minimum_distinct_heldout_sources: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "id": self.stage_id,
            "required_capabilities": sorted(self.required_capabilities),
            "minimum_difficulty": self.minimum_difficulty,
            "maximum_difficulty": self.maximum_difficulty,
            "minimum_train_records": self.minimum_train_records,
            "minimum_heldout_records": self.minimum_heldout_records,
            "minimum_distinct_train_sources": self.minimum_distinct_train_sources,
            "minimum_distinct_heldout_sources": self.minimum_distinct_heldout_sources,
        }


@dataclass(frozen=True, slots=True)
class CurriculumPolicy:
    policy_id: str
    provenance: dict[str, str]
    capabilities_path: tuple[str, ...]
    difficulty_path: tuple[str, ...]
    require_record_source: bool
    validation_basis_points: int
    test_basis_points: int
    reject_multiple_matches: bool
    require_every_annotated_record: bool
    stages: tuple[StagePolicy, ...]


@dataclass(frozen=True, slots=True)
class RecordAnnotation:
    capabilities: frozenset[str]
    difficulty: float
    source: str | None

    def digest(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "capabilities": sorted(self.capabilities),
                    "difficulty": self.difficulty,
                }
            )
        ).hexdigest()


@dataclass(slots=True)
class StageWriter:
    stage: StagePolicy
    path: Path
    handle: Any
    records: int = 0
    bytes_written: int = 0

    @classmethod
    def create(cls, stage: StagePolicy, path: Path) -> "StageWriter":
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = path.open("xb")
        except OSError as error:
            raise CapabilityCurriculumError(f"cannot create private stage corpus {path}: {error}") from error
        return cls(stage=stage, path=path, handle=handle)

    def write(self, record: TrainingRecord) -> None:
        try:
            validate_hdc_record(record)
            payload = record.text.encode("ascii") + b"\n"
        except (TrainingDataError, UnicodeEncodeError) as error:
            raise CapabilityCurriculumError(
                f"stage {self.stage.stage_id} cannot export a selected source record under the strict HDC contract: {error}"
            ) from error
        self.handle.write(payload)
        self.records += 1
        self.bytes_written += len(payload)

    def close(self) -> None:
        if self.handle.closed:
            return
        failure: OSError | None = None
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        except OSError as error:
            failure = error
        finally:
            try:
                self.handle.close()
            except OSError as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise CapabilityCurriculumError(f"cannot finalize stage corpus {self.path}: {failure}") from failure


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapabilityCurriculumError(message)


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CapabilityCurriculumError(f"value is not canonical JSON: {error}") from error


def validate_identifier(value: object, field: str) -> str:
    require(isinstance(value, str) and bool(value), f"{field} must be a nonempty identifier")
    require(len(value.encode("utf-8")) <= MAX_IDENTIFIER_BYTES and _IDENTIFIER.fullmatch(value) is not None,
            f"{field} must contain only ASCII letters/digits/._- and start with a letter or digit")
    return value


def validate_text(value: object, field: str, maximum_bytes: int = 16 * 1024) -> str:
    require(isinstance(value, str) and bool(value), f"{field} must be a nonempty string")
    require(len(value.encode("utf-8")) <= maximum_bytes, f"{field} exceeds {maximum_bytes} UTF-8 bytes")
    require("\x00" not in value, f"{field} contains NUL")
    return value


def parse_nonnegative_int(value: object, field: str, maximum: int = 2**63 - 1) -> int:
    require(type(value) is int and 0 <= value <= maximum, f"{field} must be an integer in [0, {maximum}]")
    return value


def parse_probability(value: object, field: str) -> float:
    require(type(value) in {int, float} and not isinstance(value, bool), f"{field} must be a finite number in [0, 1]")
    parsed = float(value)
    require(math.isfinite(parsed) and 0.0 <= parsed <= 1.0, f"{field} must be a finite number in [0, 1]")
    return parsed


def exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    unknown = sorted(str(key) for key in value if key not in expected)
    missing = sorted(expected.difference(value))
    require(not unknown, f"{field} contains unsupported fields: {', '.join(unknown)}")
    require(not missing, f"{field} is missing required fields: {', '.join(missing)}")


def file_identity(path: Path, label: str) -> FileIdentity:
    resolved = path.resolve()
    require(resolved.is_file(), f"{label} is not a regular file: {resolved}")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with resolved.open("rb") as handle:
            while block := handle.read(CHUNK_BYTES):
                digest.update(block)
                byte_count += len(block)
    except OSError as error:
        raise CapabilityCurriculumError(f"cannot hash {label} {resolved}: {error}") from error
    return FileIdentity(path=str(resolved), sha256=digest.hexdigest(), bytes=byte_count)


def require_unchanged(path: Path, before: FileIdentity, label: str) -> None:
    require(file_identity(path, label) == before, f"{label} changed while the curriculum was being compiled: {path}")


def snapshot_input(source: Path, destination: Path) -> tuple[str, int]:
    """Make a private byte snapshot while calculating its identity in one pass."""
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
        raise CapabilityCurriculumError(f"cannot make private curriculum input snapshot from {source}: {error}") from error
    return digest.hexdigest(), byte_count


def load_json_mapping(path: Path, label: str) -> tuple[dict[str, Any], FileIdentity]:
    identity = file_identity(path, label)
    try:
        value = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CapabilityCurriculumError(f"cannot parse {label} {path}: {error}") from error
    require(isinstance(value, Mapping), f"{label} must be one JSON object")
    require_unchanged(path, identity, label)
    return dict(value), identity


def parse_metadata_path(value: object, field: str) -> tuple[str, ...]:
    require(isinstance(value, list) and value and len(value) <= MAX_PATH_SEGMENTS,
            f"{field} must be a nonempty metadata-key array with at most {MAX_PATH_SEGMENTS} segments")
    return tuple(validate_identifier(item, f"{field}[{index}]") for index, item in enumerate(value))


def parse_capabilities(value: object, field: str, *, allow_empty: bool) -> frozenset[str]:
    require(isinstance(value, list) and (bool(value) or allow_empty), f"{field} must be an array")
    require(len(value) <= MAX_CAPABILITIES_PER_RECORD, f"{field} exceeds {MAX_CAPABILITIES_PER_RECORD} values")
    parsed = frozenset(validate_identifier(item, f"{field}[{index}]") for index, item in enumerate(value))
    require(len(parsed) == len(value), f"{field} repeats a capability")
    return parsed


def parse_policy(path: Path) -> tuple[CurriculumPolicy, FileIdentity]:
    value, identity = load_json_mapping(path, "capability curriculum policy")
    expected = {"schema_version", "id", "provenance", "annotations", "splits", "assignment", "stages"}
    exact_fields(value, expected, "capability curriculum policy")
    require(value["schema_version"] == POLICY_SCHEMA,
            f"unsupported capability curriculum policy schema: {value['schema_version']!r}")

    provenance = value["provenance"]
    require(isinstance(provenance, Mapping), "capability curriculum policy.provenance must be an object")
    exact_fields(provenance, {"source", "license", "retrieved_at", "preprocessing"}, "capability curriculum policy.provenance")
    parsed_provenance = {
        field: validate_text(provenance[field], f"capability curriculum policy.provenance.{field}")
        for field in ("source", "license", "retrieved_at", "preprocessing")
    }

    annotations = value["annotations"]
    require(isinstance(annotations, Mapping), "capability curriculum policy.annotations must be an object")
    exact_fields(
        annotations,
        {"capabilities_path", "difficulty_path", "require_record_source"},
        "capability curriculum policy.annotations",
    )
    require(type(annotations["require_record_source"]) is bool,
            "capability curriculum policy.annotations.require_record_source must be boolean")

    splits = value["splits"]
    require(isinstance(splits, Mapping), "capability curriculum policy.splits must be an object")
    exact_fields(splits, {"validation_basis_points", "test_basis_points"}, "capability curriculum policy.splits")
    validation_basis_points = parse_nonnegative_int(
        splits["validation_basis_points"], "capability curriculum policy.splits.validation_basis_points", 9_999
    )
    test_basis_points = parse_nonnegative_int(
        splits["test_basis_points"], "capability curriculum policy.splits.test_basis_points", 9_999
    )
    require(validation_basis_points + test_basis_points < 10_000,
            "capability curriculum policy validation/test basis points must sum to less than 10000")

    assignment = value["assignment"]
    require(isinstance(assignment, Mapping), "capability curriculum policy.assignment must be an object")
    exact_fields(assignment, {"mode", "reject_multiple_matches", "require_every_annotated_record"},
                 "capability curriculum policy.assignment")
    require(assignment["mode"] == "first-match", "capability curriculum policy.assignment.mode must be 'first-match'")
    require(type(assignment["reject_multiple_matches"]) is bool,
            "capability curriculum policy.assignment.reject_multiple_matches must be boolean")
    require(type(assignment["require_every_annotated_record"]) is bool,
            "capability curriculum policy.assignment.require_every_annotated_record must be boolean")

    raw_stages = value["stages"]
    require(isinstance(raw_stages, list) and raw_stages and len(raw_stages) <= MAX_STAGES,
            f"capability curriculum policy.stages must contain 1..{MAX_STAGES} entries")
    stages: list[StagePolicy] = []
    seen_stage_ids: set[str] = set()
    for index, raw in enumerate(raw_stages):
        field = f"capability curriculum policy.stages[{index}]"
        require(isinstance(raw, Mapping), f"{field} must be an object")
        exact_fields(
            raw,
            {
                "id",
                "required_capabilities",
                "minimum_difficulty",
                "maximum_difficulty",
                "minimum_train_records",
                "minimum_heldout_records",
                "minimum_distinct_train_sources",
                "minimum_distinct_heldout_sources",
            },
            field,
        )
        stage_id = validate_identifier(raw["id"], f"{field}.id")
        require(stage_id not in seen_stage_ids, f"capability curriculum policy repeats stage id {stage_id!r}")
        seen_stage_ids.add(stage_id)
        minimum_difficulty = parse_probability(raw["minimum_difficulty"], f"{field}.minimum_difficulty")
        maximum_difficulty = parse_probability(raw["maximum_difficulty"], f"{field}.maximum_difficulty")
        require(minimum_difficulty <= maximum_difficulty, f"{field} difficulty bounds are inverted")
        minimum_train_records = parse_nonnegative_int(raw["minimum_train_records"], f"{field}.minimum_train_records")
        require(minimum_train_records > 0, f"{field}.minimum_train_records must be positive for a trainable stage")
        stages.append(
            StagePolicy(
                stage_id=stage_id,
                required_capabilities=parse_capabilities(raw["required_capabilities"], f"{field}.required_capabilities", allow_empty=True),
                minimum_difficulty=minimum_difficulty,
                maximum_difficulty=maximum_difficulty,
                minimum_train_records=minimum_train_records,
                minimum_heldout_records=parse_nonnegative_int(raw["minimum_heldout_records"], f"{field}.minimum_heldout_records"),
                minimum_distinct_train_sources=parse_nonnegative_int(
                    raw["minimum_distinct_train_sources"], f"{field}.minimum_distinct_train_sources"
                ),
                minimum_distinct_heldout_sources=parse_nonnegative_int(
                    raw["minimum_distinct_heldout_sources"], f"{field}.minimum_distinct_heldout_sources"
                ),
            )
        )

    return CurriculumPolicy(
        policy_id=validate_identifier(value["id"], "capability curriculum policy.id"),
        provenance=parsed_provenance,
        capabilities_path=parse_metadata_path(annotations["capabilities_path"], "capability curriculum policy.annotations.capabilities_path"),
        difficulty_path=parse_metadata_path(annotations["difficulty_path"], "capability curriculum policy.annotations.difficulty_path"),
        require_record_source=annotations["require_record_source"],
        validation_basis_points=validation_basis_points,
        test_basis_points=test_basis_points,
        reject_multiple_matches=assignment["reject_multiple_matches"],
        require_every_annotated_record=assignment["require_every_annotated_record"],
        stages=tuple(stages),
    ), identity


def mapping_at_path(metadata: Mapping[str, Any], path: Sequence[str], field: str) -> object:
    current: object = metadata
    for component in path:
        require(isinstance(current, Mapping), f"{field} cannot traverse metadata key {component!r}")
        require(component in current, f"{field} is missing metadata key {component!r}")
        current = current[component]
    return current


def parse_annotation(record: TrainingRecord, policy: CurriculumPolicy) -> RecordAnnotation:
    require(record.metadata is not None, "curriculum-selected records require metadata annotations")
    capabilities = parse_capabilities(
        mapping_at_path(record.metadata, policy.capabilities_path, "record capability annotation"),
        "record capability annotation",
        allow_empty=False,
    )
    difficulty = parse_probability(
        mapping_at_path(record.metadata, policy.difficulty_path, "record difficulty annotation"),
        "record difficulty annotation",
    )
    if policy.require_record_source:
        require(record.source is not None, "curriculum policy requires every record to declare source provenance")
    return RecordAnnotation(capabilities=capabilities, difficulty=difficulty, source=record.source)


def matching_stages(annotation: RecordAnnotation, policy: CurriculumPolicy) -> list[StagePolicy]:
    return [
        stage
        for stage in policy.stages
        if stage.required_capabilities.issubset(annotation.capabilities)
        and stage.minimum_difficulty <= annotation.difficulty <= stage.maximum_difficulty
    ]


def canonical_tokens(text: str) -> list[bytes]:
    """Mirror the strict ASCII tokenizer only after ``validate_hdc_record`` passes."""
    token = bytearray()
    result: list[bytes] = []
    for value in text.encode("ascii"):
        if is_token_byte(value):
            token.append(value + 32 if 65 <= value <= 90 else value)
        elif token:
            result.append(bytes(token))
            token.clear()
    if token:
        result.append(bytes(token))
    return result


class CurriculumIndex:
    """Bounded-memory exact dedupe, source diversity, and vocabulary audit."""

    def __init__(self, path: Path) -> None:
        try:
            self.connection = sqlite3.connect(path)
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute(
                "CREATE TABLE seen (digest TEXT PRIMARY KEY, split TEXT NOT NULL, annotation_sha256 TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE stage_source (stage_id TEXT NOT NULL, split TEXT NOT NULL, source TEXT NOT NULL, "
                "PRIMARY KEY(stage_id, split, source))"
            )
            self.connection.execute("CREATE TABLE vocabulary (token BLOB PRIMARY KEY)")
            self.connection.commit()
        except sqlite3.Error as error:
            raise CapabilityCurriculumError(f"cannot initialize curriculum SQLite index {path}: {error}") from error
        self.unique_records = 0
        self.duplicate_records = 0

    def accept(self, record: TrainingRecord, split: str, annotation: RecordAnnotation) -> bool:
        digest = record.text_digest()
        annotation_digest = annotation.digest()
        try:
            existing = self.connection.execute(
                "SELECT split, annotation_sha256 FROM seen WHERE digest = ?", (digest,)
            ).fetchone()
            if existing is not None:
                self.duplicate_records += 1
                require(existing[0] == split,
                        "exact duplicate text has conflicting train/validation/test split declarations")
                require(existing[1] == annotation_digest,
                        "exact duplicate text has conflicting capability/difficulty annotations")
                return False
            self.connection.execute(
                "INSERT INTO seen(digest, split, annotation_sha256) VALUES (?, ?, ?)",
                (digest, split, annotation_digest),
            )
            self.unique_records += 1
            if self.unique_records % 4096 == 0:
                self.connection.commit()
            return True
        except sqlite3.Error as error:
            raise CapabilityCurriculumError(f"curriculum exact-dedupe index failed: {error}") from error

    def record_source(self, stage_id: str, split: str, source: str | None) -> None:
        # When source is optional, a stable sentinel permits an honest diversity
        # count of one rather than silently pretending every record is distinct.
        source_key = "<undeclared>" if source is None else source
        try:
            self.connection.execute(
                "INSERT OR IGNORE INTO stage_source(stage_id, split, source) VALUES (?, ?, ?)",
                (stage_id, split, source_key),
            )
        except sqlite3.Error as error:
            raise CapabilityCurriculumError(f"curriculum source-diversity index failed: {error}") from error

    def add_tokens(self, text: str) -> None:
        try:
            self.connection.executemany("INSERT OR IGNORE INTO vocabulary(token) VALUES (?)", ((token,) for token in canonical_tokens(text)))
        except sqlite3.Error as error:
            raise CapabilityCurriculumError(f"curriculum vocabulary index failed: {error}") from error

    def source_count(self, stage_id: str, split: str) -> int:
        try:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM stage_source WHERE stage_id = ? AND split = ?", (stage_id, split)
            ).fetchone()
        except sqlite3.Error as error:
            raise CapabilityCurriculumError(f"cannot aggregate curriculum source diversity: {error}") from error
        assert row is not None
        return int(row[0])

    def heldout_source_count(self, stage_id: str) -> int:
        try:
            row = self.connection.execute(
                "SELECT COUNT(DISTINCT source) FROM stage_source WHERE stage_id = ? AND split IN ('validation', 'test')",
                (stage_id,),
            ).fetchone()
        except sqlite3.Error as error:
            raise CapabilityCurriculumError(f"cannot aggregate held-out curriculum source diversity: {error}") from error
        assert row is not None
        return int(row[0])

    def vocabulary_size(self) -> int:
        try:
            row = self.connection.execute("SELECT COUNT(*) FROM vocabulary").fetchone()
        except sqlite3.Error as error:
            raise CapabilityCurriculumError(f"cannot aggregate curriculum vocabulary: {error}") from error
        assert row is not None
        return int(row[0])

    def close(self) -> None:
        try:
            self.connection.commit()
        finally:
            self.connection.close()


def fsync_directory(directory: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def publish_new_directory(staging: Path, output: Path) -> None:
    """Atomically publish a directory without ever replacing an existing name.

    POSIX ``rename`` may replace an empty destination directory. Linux
    ``renameat2(RENAME_NOREPLACE)``, macOS ``renamex_np(RENAME_EXCL)``, and
    Windows ``os.rename`` provide the required no-replace primitive. Other
    platforms fail closed rather than weaken immutable package publication with
    a check-then-replace race.
    """
    require(not os.path.lexists(output), f"output directory already exists; refusing to overwrite: {output}")
    if os.name == "nt":
        try:
            os.rename(staging, output)
        except FileExistsError as error:
            raise CapabilityCurriculumError(f"output directory already exists; refusing to overwrite: {output}") from error
        except OSError as error:
            raise CapabilityCurriculumError(f"cannot atomically publish curriculum package to {output}: {error}") from error
    elif sys.platform.startswith("linux"):
        try:
            library = ctypes.CDLL(None, use_errno=True)
            renameat2 = library.renameat2
        except (AttributeError, OSError) as error:
            raise CapabilityCurriculumError(
                "atomic no-replace directory publication requires Linux renameat2 on this platform"
            ) from error
        renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,  # AT_FDCWD
            os.fsencode(staging),
            -100,  # AT_FDCWD
            os.fsencode(output),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            failure = ctypes.get_errno()
            if failure in {errno.EEXIST, errno.ENOTEMPTY, errno.EISDIR}:
                raise CapabilityCurriculumError(f"output directory already exists; refusing to overwrite: {output}")
            raise CapabilityCurriculumError(
                f"cannot atomically publish curriculum package to {output}: {os.strerror(failure)}"
            )
    elif sys.platform == "darwin":
        try:
            library = ctypes.CDLL(None, use_errno=True)
            renamex_np = library.renamex_np
        except (AttributeError, OSError) as error:
            raise CapabilityCurriculumError(
                "atomic no-replace directory publication requires macOS renamex_np on this platform"
            ) from error
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        # Darwin's RENAME_EXCL: create destination only when it does not exist.
        result = renamex_np(os.fsencode(staging), os.fsencode(output), 0x00000004)
        if result != 0:
            failure = ctypes.get_errno()
            if failure in {errno.EEXIST, errno.ENOTEMPTY, errno.EISDIR}:
                raise CapabilityCurriculumError(f"output directory already exists; refusing to overwrite: {output}")
            raise CapabilityCurriculumError(
                f"cannot atomically publish curriculum package to {output}: {os.strerror(failure)}"
            )
    else:
        raise CapabilityCurriculumError(
            "atomic no-replace directory publication is unavailable on this platform; refusing unsafe publication"
        )
    fsync_directory(output.parent)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def preflight_mapping(stats: PreflightStats) -> dict[str, int]:
    return {
        "accepted_tokens": stats.accepted_tokens,
        "crlf_records": stats.crlf_records,
        "dropped_tokens": stats.dropped_tokens,
        "maximum_record_bytes": stats.maximum_record_bytes,
        "non_ascii_bytes": stats.non_ascii_bytes,
        "overlong_records": stats.overlong_records,
        "records_with_tokenizer_truncation": stats.records_with_tokenizer_truncation,
    }


def require_lossless_preflight(stats: PreflightStats, label: str) -> None:
    failures: list[str] = []
    if stats.overlong_records:
        failures.append(f"overlong_records={stats.overlong_records}")
    if stats.records_with_tokenizer_truncation:
        failures.append(f"records_with_tokenizer_truncation={stats.records_with_tokenizer_truncation}")
    if stats.dropped_tokens:
        failures.append(f"dropped_tokens={stats.dropped_tokens}")
    if stats.non_ascii_bytes:
        failures.append(f"non_ascii_bytes={stats.non_ascii_bytes}")
    if stats.crlf_records:
        failures.append(f"crlf_records={stats.crlf_records}")
    if stats.accepted_tokens == 0:
        failures.append("accepted_tokens=0")
    require(not failures, f"{label} violates the strict Vegai HDC ingest contract: {', '.join(failures)}")


def stage_filename(index: int, stage: StagePolicy) -> str:
    return f"{index + 1:03d}-{stage.stage_id}.txt"


def resolve_input_compression(path: Path, selected_format: str, requested: str) -> str:
    """Freeze the actual text codec so full replay survives a file relocation."""
    if selected_format in {"jsonl", "json", "txt"}:
        require(requested in {"auto", "none", "gzip"}, "curriculum input compression is unsupported")
        if requested == "auto":
            return "gzip" if path.suffix.lower() == ".gz" else "none"
        return requested
    require(requested in {"auto", "none"},
            "Parquet/Arrow embed their compression; curriculum input compression must be auto or none")
    return "none"


def build_package(
    arguments: argparse.Namespace,
    *,
    report_source_path: str | None = None,
    report_policy_path: str | None = None,
) -> dict[str, Any]:
    source = arguments.input.resolve()
    policy_path = arguments.policy.resolve()
    requested_output = arguments.output_dir
    require(not os.path.lexists(requested_output),
            f"output directory already exists; refusing to overwrite: {requested_output}")
    output = requested_output.resolve()
    require(source.is_file(), f"input dataset does not exist: {source}")
    require(output != source and output != policy_path, "--output-dir must not overwrite an input or policy file")
    require(not os.path.lexists(output), f"output directory already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    policy, observed_policy_identity = parse_policy(policy_path)
    observed_source_identity = file_identity(source, "curriculum input")
    # Keep packages portable and avoid exposing an operator's absolute local
    # filesystem layout.  The digest and byte count, not this basename, are
    # the immutable identities; retain the observed paths only for the
    # in-process stability recheck below.
    reported_policy_path = policy_path.name if report_policy_path is None else report_policy_path
    reported_source_path = source.name if report_source_path is None else report_source_path
    validate_portable_filename(reported_policy_path, "curriculum package policy path")
    validate_portable_filename(reported_source_path, "curriculum package input path")
    policy_identity = FileIdentity(
        path=reported_policy_path,
        sha256=observed_policy_identity.sha256,
        bytes=observed_policy_identity.bytes,
    )
    source_identity = FileIdentity(
        path=reported_source_path,
        sha256=observed_source_identity.sha256,
        bytes=observed_source_identity.bytes,
    )
    selected_format = infer_format(source, arguments.input_format)
    selected_compression = resolve_input_compression(source, selected_format, arguments.input_compression)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    source_snapshot = staging / ".input-snapshot"
    index: CurriculumIndex | None = None
    writers: dict[str, StageWriter] = {}
    try:
        snapshot_digest, snapshot_bytes = snapshot_input(source, source_snapshot)
        require(
            snapshot_digest == observed_source_identity.sha256 and snapshot_bytes == observed_source_identity.bytes,
            "curriculum input changed while its private snapshot was being created",
        )
        index = CurriculumIndex(staging / ".curriculum.sqlite")
        # Retain each handle before opening the next one so exception cleanup
        # can close every private corpus on Windows as well as POSIX.
        for position, stage in enumerate(policy.stages):
            writers[stage.stage_id] = StageWriter.create(
                stage, staging / "shards" / stage_filename(position, stage)
            )
        counts = {
            stage.stage_id: {"train": 0, "validation": 0, "test": 0, "ambiguous": 0}
            for stage in policy.stages
        }
        records_seen = 0
        unassigned_records = 0
        ambiguous_records = 0
        heldout_digest = hashlib.sha256()

        for record in iter_records(
            source_snapshot,
            format=selected_format,
            compression=selected_compression,
            batch_size=arguments.batch_size,
            error_policy="raise",
        ):
            records_seen += 1
            annotation = parse_annotation(record, policy)
            split = record.split or split_for_text_digest(
                record.text_digest(), policy.validation_basis_points, policy.test_basis_points
            )
            require(split in {"train", "validation", "test"}, f"record has unsupported split {split!r}")
            if not index.accept(record, split, annotation):
                continue
            matches = matching_stages(annotation, policy)
            if len(matches) > 1:
                ambiguous_records += 1
                if policy.reject_multiple_matches:
                    names = ", ".join(stage.stage_id for stage in matches)
                    raise CapabilityCurriculumError(
                        f"record matches multiple policy stages ({names}); make annotation ranges disjoint, add a combined stage, "
                        "or explicitly allow first-match assignment"
                    )
            if not matches:
                unassigned_records += 1
                if policy.require_every_annotated_record:
                    raise CapabilityCurriculumError("an annotated record did not match any curriculum stage")
                continue
            stage = matches[0]
            if len(matches) > 1:
                counts[stage.stage_id]["ambiguous"] += 1
            index.record_source(stage.stage_id, split, annotation.source)
            counts[stage.stage_id][split] += 1
            if split == "train":
                writers[stage.stage_id].write(record)
                index.add_tokens(record.text)
            else:
                # Held-out identifiers are deliberately one-way digests. Bind
                # their stage and split too, so the report commits to held-out
                # partitioning without retaining prompts, source text, or
                # expected answers outside the governed evaluation store.
                heldout_digest.update(stage.stage_id.encode("ascii"))
                heldout_digest.update(b"\x00")
                heldout_digest.update(split.encode("ascii"))
                heldout_digest.update(b"\x00")
                heldout_digest.update(bytes.fromhex(record.text_digest()))

        for writer in writers.values():
            writer.close()
        fsync_directory(staging / "shards")
        private_snapshot_identity = file_identity(source_snapshot, "private curriculum input snapshot")
        require(
            private_snapshot_identity.sha256 == observed_source_identity.sha256
            and private_snapshot_identity.bytes == observed_source_identity.bytes,
            "private curriculum input snapshot changed while records were being compiled",
        )
        require_unchanged(source, observed_source_identity, "curriculum input")
        require_unchanged(policy_path, observed_policy_identity, "capability curriculum policy")
        source_snapshot.unlink()
        fsync_directory(staging)

        stages_report: list[dict[str, Any]] = []
        shards: list[dict[str, Any]] = []
        for position, stage in enumerate(policy.stages):
            writer = writers[stage.stage_id]
            train_records = counts[stage.stage_id]["train"]
            heldout_records = counts[stage.stage_id]["validation"] + counts[stage.stage_id]["test"]
            train_sources = index.source_count(stage.stage_id, "train")
            heldout_sources = index.heldout_source_count(stage.stage_id)
            require(train_records >= stage.minimum_train_records,
                    f"stage {stage.stage_id} has {train_records} train records, below policy minimum {stage.minimum_train_records}")
            require(heldout_records >= stage.minimum_heldout_records,
                    f"stage {stage.stage_id} has {heldout_records} held-out records, below policy minimum {stage.minimum_heldout_records}")
            require(train_sources >= stage.minimum_distinct_train_sources,
                    f"stage {stage.stage_id} has {train_sources} train sources, below policy minimum {stage.minimum_distinct_train_sources}")
            require(heldout_sources >= stage.minimum_distinct_heldout_sources,
                    f"stage {stage.stage_id} has {heldout_sources} held-out sources, below policy minimum {stage.minimum_distinct_heldout_sources}")
            try:
                with writer.path.open("rb") as handle:
                    digest, byte_count, preflight = scan_corpus(
                        handle,
                        DEFAULT_MAX_LINE_BYTES,
                        DEFAULT_TOKEN_LENGTH,
                        DEFAULT_MAX_TOKENS_PER_LINE,
                    )
            except OSError as error:
                raise CapabilityCurriculumError(f"cannot scan private stage {stage.stage_id}: {error}") from error
            require(digest == file_identity(writer.path, f"stage {stage.stage_id} corpus").sha256,
                    f"stage {stage.stage_id} hash changed while being finalized")
            require(byte_count == writer.bytes_written and preflight.records == writer.records,
                    f"stage {stage.stage_id} writer accounting changed while being finalized")
            require_lossless_preflight(preflight, f"stage {stage.stage_id}")
            relative_path = writer.path.relative_to(staging).as_posix()
            shard = {
                "id": digest,
                "path": relative_path,
                "sha256": digest,
                "bytes": byte_count,
                "physical_lf_records": preflight.records,
                "preflight": preflight_mapping(preflight),
            }
            shards.append(shard)
            stages_report.append(
                {
                    "id": stage.stage_id,
                    "constraints": stage.to_mapping(),
                    "training": {
                        "path": relative_path,
                        "records": train_records,
                        "distinct_sources": train_sources,
                        "sha256": digest,
                        "bytes": byte_count,
                        "accepted_tokens": preflight.accepted_tokens,
                    },
                    "heldout": {
                        "validation_records": counts[stage.stage_id]["validation"],
                        "test_records": counts[stage.stage_id]["test"],
                        "records": heldout_records,
                        "distinct_sources": heldout_sources,
                    },
                    "ambiguous_first_match_records": counts[stage.stage_id]["ambiguous"],
                }
            )

        ledger = {
            "schema": LEDGER_SCHEMA,
            "declared_provenance": {
                "source": policy.provenance["source"],
                "license": policy.provenance["license"],
                "preprocessing": policy.provenance["preprocessing"],
            },
            "vegai_ascii_ingest_contract": {
                "max_line_bytes": DEFAULT_MAX_LINE_BYTES,
                "token_length_bytes": DEFAULT_TOKEN_LENGTH,
                "max_tokens_per_line": DEFAULT_MAX_TOKENS_PER_LINE,
                "require_ascii_only": True,
                "require_lf_only": True,
                "require_no_dropped_tokens": True,
            },
            "shards": shards,
        }
        ledger_path = staging / "ledger.json"
        atomic_write_json(ledger_path, ledger)
        private_ledger_identity = file_identity(ledger_path, "curriculum training ledger")
        # The package is atomically renamed from staging to ``output`` below.
        # Keep this package-local path portable rather than leaking a temporary
        # or operator-specific absolute pathname.
        ledger_identity = FileIdentity(
            path="ledger.json",
            sha256=private_ledger_identity.sha256,
            bytes=private_ledger_identity.bytes,
        )
        package = {
            "schema_version": PACKAGE_SCHEMA,
            "policy": {"id": policy.policy_id, **policy_identity.to_mapping()},
            "input": {
                "format": selected_format,
                "compression": selected_compression,
                **source_identity.to_mapping(),
            },
            "ledger": ledger_identity.to_mapping(),
            "provenance": dict(policy.provenance),
            "stages": stages_report,
            "records": {
                "seen": records_seen,
                "unique": index.unique_records,
                "duplicates_removed": index.duplicate_records,
                "unassigned": unassigned_records,
                "ambiguous": ambiguous_records,
            },
            "heldout": {
                "record_digest_stream_schema": "stage-nul-split-nul-text-sha256/v1",
                "record_digest_stream_sha256": heldout_digest.hexdigest(),
                "raw_text_persisted": False,
                "contract": (
                    "Held-out source text is never written into the curriculum package. Use governed external target-task "
                    "evaluators for domain quality, safety, reliability, and preference measurement."
                ),
            },
            "capacity_preflight": {
                "unique_training_tokens_min_frequency_1": index.vocabulary_size(),
                "minimum_continual_max_vocabulary": index.vocabulary_size(),
                "contract": (
                    "Set continuous_pretrain --max-vocab at or above this exact selected-training vocabulary count; "
                    "the compiler does not silently evict tokens."
                ),
            },
            "training_boundary": (
                "Stage corpora contain source record.text only. Capability tags, labels, metadata, answers, and policy "
                "instructions are not injected into HDC text. This package improves governed curriculum selection and "
                "measurement readiness; it does not itself prove general intelligence, target-task correctness, or "
                "superiority over another model."
            ),
        }
        atomic_write_json(staging / "curriculum.json", package)
        index.close()
        index = None
        (staging / ".curriculum.sqlite").unlink(missing_ok=True)
        fsync_directory(staging)
        # Recheck immediately before the publication rename as well.  This
        # narrows the mutable-source window after the long stream/validation
        # pass; callers should still keep governed inputs immutable.
        require_unchanged(source, observed_source_identity, "curriculum input")
        require_unchanged(policy_path, observed_policy_identity, "capability curriculum policy")
        publish_new_directory(staging, output)
        return package
    except BaseException:
        for writer in writers.values():
            try:
                writer.close()
            except CapabilityCurriculumError:
                pass
        if index is not None:
            try:
                index.close()
            except sqlite3.Error:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise


def resolve_inside(root: Path, value: object, field: str) -> Path:
    require(isinstance(value, str) and bool(value), f"{field} path is missing")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise CapabilityCurriculumError(f"{field} escapes package root: {value}") from error
    return candidate


def validate_identity_mapping(value: object, field: str) -> tuple[str, int]:
    require(isinstance(value, Mapping), f"{field} must be an identity object")
    display_path = value.get("path")
    digest = value.get("sha256")
    byte_count = value.get("bytes")
    require(isinstance(display_path, str) and bool(display_path) and "\x00" not in display_path,
            f"{field}.path must be a nonempty string without NUL")
    require(isinstance(digest, str) and len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
            f"{field}.sha256 must be lowercase SHA-256")
    require(type(byte_count) is int and byte_count >= 0, f"{field}.bytes must be nonnegative integer")
    return digest, byte_count


def validate_portable_filename(value: object, field: str) -> str:
    require(isinstance(value, str) and bool(value), f"{field} must be a nonempty filename")
    require("/" not in value and "\\" not in value and value not in {".", ".."} and "\x00" not in value,
            f"{field} must be a portable basename")
    return value


def package_layout(root: Path, label: str) -> tuple[set[str], set[str]]:
    """Return strict relative directory/file names without following links."""
    directories: set[str] = set()
    files: set[str] = set()
    try:
        entries = sorted(root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as error:
        raise CapabilityCurriculumError(f"cannot enumerate {label}: {error}") from error
    for entry in entries:
        relative = entry.relative_to(root).as_posix()
        require(not entry.is_symlink(), f"{label} contains a prohibited symbolic link: {relative}")
        if entry.is_dir():
            directories.add(relative)
        elif entry.is_file():
            files.add(relative)
        else:
            raise CapabilityCurriculumError(f"{label} contains an unsupported filesystem object: {relative}")
    return directories, files


def package_tree(root: Path, label: str) -> tuple[set[str], dict[str, tuple[str, int]]]:
    """Return strict relative directory/file identities for an exact comparison."""
    directories, relative_files = package_layout(root, label)
    files = {
        relative: (identity.sha256, identity.bytes)
        for relative in relative_files
        for identity in (file_identity(root / relative, f"{label} file {relative}"),)
    }
    return directories, files


def expected_package_directories(files: set[str]) -> set[str]:
    directories: set[str] = set()
    for filename in files:
        parent = Path(filename).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def validate_package_layout(package_root: Path, shard_paths: Sequence[str]) -> None:
    expected_files = {"curriculum.json", "ledger.json", *shard_paths}
    require(len(expected_files) == len(shard_paths) + 2, "curriculum package has duplicate or reserved shard paths")
    directories, files = package_layout(package_root, "curriculum package")
    require(files == expected_files,
            "curriculum package contains missing, unbound, or unexpected files")
    require(directories == expected_package_directories(expected_files),
            "curriculum package contains unexpected directories")


def require_identical_package_tree(published: Path, replayed: Path) -> int:
    published_directories, published_files = package_tree(published, "published curriculum package")
    replayed_directories, replayed_files = package_tree(replayed, "replayed curriculum package")
    require(published_directories == replayed_directories and set(published_files) == set(replayed_files),
            "source/policy replay produced a different curriculum package layout")
    for relative in sorted(published_files):
        require(published_files[relative] == replayed_files[relative],
                f"source/policy replay differs from the published package at {relative}")
    return len(published_files)


def verify_package(arguments: argparse.Namespace) -> dict[str, Any]:
    package_root = arguments.package.resolve()
    package_path = package_root / "curriculum.json" if package_root.is_dir() else package_root
    package_root = package_path.parent
    package, package_identity = load_json_mapping(package_path, "curriculum package")
    expected = {
        "schema_version", "policy", "input", "ledger", "provenance", "stages", "records", "heldout", "capacity_preflight",
        "training_boundary",
    }
    exact_fields(package, expected, "curriculum package")
    require(package["schema_version"] == PACKAGE_SCHEMA, "curriculum package schema is unsupported")

    policy_identity = package["policy"]
    require(isinstance(policy_identity, Mapping), "curriculum package.policy must be an object")
    exact_fields(policy_identity, {"id", "path", "sha256", "bytes"}, "curriculum package.policy")
    validate_identifier(policy_identity["id"], "curriculum package.policy.id")
    validate_portable_filename(policy_identity["path"], "curriculum package.policy.path")
    validate_identity_mapping(policy_identity, "curriculum package.policy")

    input_identity = package["input"]
    require(isinstance(input_identity, Mapping), "curriculum package.input must be an object")
    exact_fields(input_identity, {"format", "compression", "path", "sha256", "bytes"}, "curriculum package.input")
    require(input_identity["format"] in {"jsonl", "json", "txt", "parquet", "arrow"},
            "curriculum package.input.format is unsupported")
    require(input_identity["compression"] in {"none", "gzip"},
            "curriculum package.input.compression must be a resolved none or gzip codec")
    validate_portable_filename(input_identity["path"], "curriculum package.input.path")
    validate_identity_mapping(input_identity, "curriculum package.input")

    ledger_identity = package["ledger"]
    require(isinstance(ledger_identity, Mapping), "curriculum package.ledger must be an object")
    exact_fields(ledger_identity, {"path", "sha256", "bytes"}, "curriculum package.ledger")
    require(ledger_identity["path"] == "ledger.json", "curriculum package.ledger.path must be ledger.json")
    ledger_digest, ledger_bytes = validate_identity_mapping(ledger_identity, "curriculum package.ledger")
    ledger_path = resolve_inside(package_root, "ledger.json", "curriculum ledger")
    observed_ledger = file_identity(ledger_path, "curriculum ledger")
    require(observed_ledger.sha256 == ledger_digest and observed_ledger.bytes == ledger_bytes,
            "curriculum ledger identity differs from curriculum package")
    ledger, _ = load_json_mapping(ledger_path, "curriculum ledger")
    exact_fields(ledger, {"schema", "declared_provenance", "vegai_ascii_ingest_contract", "shards"}, "curriculum ledger")
    require(ledger.get("schema") == LEDGER_SCHEMA, "curriculum ledger schema is unsupported")
    declared_provenance = ledger.get("declared_provenance")
    require(isinstance(declared_provenance, Mapping), "curriculum ledger declared provenance is malformed")
    exact_fields(declared_provenance, {"source", "license", "preprocessing"}, "curriculum ledger declared provenance")
    contract = ledger.get("vegai_ascii_ingest_contract")
    require(isinstance(contract, Mapping), "curriculum ledger has no HDC ingest contract")
    exact_fields(
        contract,
        {
            "max_line_bytes", "token_length_bytes", "max_tokens_per_line", "require_ascii_only", "require_lf_only",
            "require_no_dropped_tokens",
        },
        "curriculum ledger HDC ingest contract",
    )
    require(
        contract.get("max_line_bytes") == DEFAULT_MAX_LINE_BYTES
        and contract.get("token_length_bytes") == DEFAULT_TOKEN_LENGTH
        and contract.get("max_tokens_per_line") == DEFAULT_MAX_TOKENS_PER_LINE
        and contract.get("require_ascii_only") is True
        and contract.get("require_lf_only") is True
        and contract.get("require_no_dropped_tokens") is True,
        "curriculum ledger HDC ingest contract is malformed",
    )
    package_provenance = package["provenance"]
    require(isinstance(package_provenance, Mapping), "curriculum package provenance is malformed")
    exact_fields(package_provenance, {"source", "license", "retrieved_at", "preprocessing"}, "curriculum package provenance")
    for field in ("source", "license", "retrieved_at", "preprocessing"):
        validate_text(package_provenance[field], f"curriculum package provenance.{field}")
    require(
        declared_provenance == {
            "source": package_provenance["source"],
            "license": package_provenance["license"],
            "preprocessing": package_provenance["preprocessing"],
        },
        "curriculum package/ledger provenance differs",
    )

    raw_stages = package["stages"]
    raw_shards = ledger.get("shards")
    require(isinstance(raw_stages, list) and raw_stages and isinstance(raw_shards, list) and len(raw_stages) == len(raw_shards),
            "curriculum package/ledger stage cardinality differs")
    verified_tokens = 0
    verified_records = 0
    assigned_records = 0
    ambiguous_stage_records = 0
    shard_paths: list[str] = []
    observed_shards: list[tuple[Path, FileIdentity]] = []
    seen_stage_ids: set[str] = set()
    for index, (stage, shard) in enumerate(zip(raw_stages, raw_shards, strict=True)):
        require(isinstance(stage, Mapping) and isinstance(shard, Mapping), f"curriculum stage {index} is malformed")
        exact_fields(stage, {"id", "constraints", "training", "heldout", "ambiguous_first_match_records"},
                     f"curriculum stage {index}")
        stage_id = validate_identifier(stage["id"], f"curriculum stage {index}.id")
        require(stage_id not in seen_stage_ids, f"curriculum package repeats stage id {stage_id!r}")
        seen_stage_ids.add(stage_id)
        constraints = stage["constraints"]
        require(isinstance(constraints, Mapping), f"curriculum stage {index}.constraints is malformed")
        exact_fields(
            constraints,
            {
                "id", "required_capabilities", "minimum_difficulty", "maximum_difficulty", "minimum_train_records",
                "minimum_heldout_records", "minimum_distinct_train_sources", "minimum_distinct_heldout_sources",
            },
            f"curriculum stage {index}.constraints",
        )
        require(constraints["id"] == stage_id, f"curriculum stage {index} constraint id differs")
        parse_capabilities(constraints["required_capabilities"], f"curriculum stage {index}.constraints.required_capabilities", allow_empty=True)
        minimum_difficulty = parse_probability(constraints["minimum_difficulty"], f"curriculum stage {index}.constraints.minimum_difficulty")
        maximum_difficulty = parse_probability(constraints["maximum_difficulty"], f"curriculum stage {index}.constraints.maximum_difficulty")
        require(minimum_difficulty <= maximum_difficulty, f"curriculum stage {index} difficulty bounds are inverted")
        minimum_train_records = parse_nonnegative_int(constraints["minimum_train_records"], f"curriculum stage {index}.constraints.minimum_train_records")
        minimum_heldout_records = parse_nonnegative_int(constraints["minimum_heldout_records"], f"curriculum stage {index}.constraints.minimum_heldout_records")
        minimum_train_sources = parse_nonnegative_int(constraints["minimum_distinct_train_sources"], f"curriculum stage {index}.constraints.minimum_distinct_train_sources")
        minimum_heldout_sources = parse_nonnegative_int(constraints["minimum_distinct_heldout_sources"], f"curriculum stage {index}.constraints.minimum_distinct_heldout_sources")
        require(minimum_train_records > 0, f"curriculum stage {index} has a non-trainable zero record minimum")

        training = stage["training"]
        heldout = stage["heldout"]
        require(isinstance(training, Mapping) and isinstance(heldout, Mapping),
                f"curriculum stage {index} has malformed training or held-out summary")
        exact_fields(training, {"path", "records", "distinct_sources", "sha256", "bytes", "accepted_tokens"},
                     f"curriculum stage {index}.training")
        exact_fields(heldout, {"validation_records", "test_records", "records", "distinct_sources"},
                     f"curriculum stage {index}.heldout")
        train_records = parse_nonnegative_int(training["records"], f"curriculum stage {index}.training.records")
        train_sources = parse_nonnegative_int(training["distinct_sources"], f"curriculum stage {index}.training.distinct_sources")
        train_tokens = parse_nonnegative_int(training["accepted_tokens"], f"curriculum stage {index}.training.accepted_tokens")
        validation_records = parse_nonnegative_int(heldout["validation_records"], f"curriculum stage {index}.heldout.validation_records")
        test_records = parse_nonnegative_int(heldout["test_records"], f"curriculum stage {index}.heldout.test_records")
        heldout_records = parse_nonnegative_int(heldout["records"], f"curriculum stage {index}.heldout.records")
        heldout_sources = parse_nonnegative_int(heldout["distinct_sources"], f"curriculum stage {index}.heldout.distinct_sources")
        require(train_records >= minimum_train_records and train_sources >= minimum_train_sources,
                f"curriculum stage {index} no longer satisfies its train coverage minimum")
        require(heldout_records == validation_records + test_records and heldout_records >= minimum_heldout_records
                and heldout_sources >= minimum_heldout_sources,
                f"curriculum stage {index} no longer satisfies its held-out coverage minimum")
        stage_ambiguous = parse_nonnegative_int(
            stage["ambiguous_first_match_records"], f"curriculum stage {index}.ambiguous_first_match_records"
        )

        exact_fields(shard, {"id", "path", "sha256", "bytes", "physical_lf_records", "preflight"},
                     f"curriculum shard {index}")
        expected_shard_path = f"shards/{index + 1:03d}-{stage_id}.txt"
        require(shard["path"] == expected_shard_path, f"curriculum shard {index} does not follow deterministic stage naming")
        shard_digest, shard_bytes = validate_identity_mapping(shard, f"curriculum shard {index}")
        shard_path = resolve_inside(package_root, shard["path"], f"curriculum shard {index}")
        try:
            with shard_path.open("rb") as handle:
                digest, byte_count, preflight = scan_corpus(
                    handle,
                    DEFAULT_MAX_LINE_BYTES,
                    DEFAULT_TOKEN_LENGTH,
                    DEFAULT_MAX_TOKENS_PER_LINE,
                )
        except OSError as error:
            raise CapabilityCurriculumError(f"cannot scan curriculum shard {index}: {error}") from error
        require(digest == shard_digest == shard.get("id"), f"curriculum shard {index} SHA-256 mismatch")
        require(byte_count == shard_bytes and preflight.records == shard.get("physical_lf_records"),
                f"curriculum shard {index} accounting mismatch")
        require(preflight_mapping(preflight) == shard.get("preflight"), f"curriculum shard {index} preflight mismatch")
        require_lossless_preflight(preflight, f"curriculum shard {index}")
        require(
            training["path"] == shard["path"]
            and training["sha256"] == digest
            and training["bytes"] == byte_count
            and train_records == preflight.records
            and train_tokens == preflight.accepted_tokens,
            f"curriculum stage {index} training summary is not bound to its ledger shard",
        )
        shard_paths.append(shard["path"])
        observed_shards.append(
            (shard_path, FileIdentity(path=str(shard_path.resolve()), sha256=digest, bytes=byte_count))
        )
        verified_tokens += preflight.accepted_tokens
        verified_records += preflight.records
        assigned_records += train_records + heldout_records
        ambiguous_stage_records += stage_ambiguous
    validate_package_layout(package_root, shard_paths)

    records = package["records"]
    require(isinstance(records, Mapping), "curriculum package records summary is malformed")
    exact_fields(records, {"seen", "unique", "duplicates_removed", "unassigned", "ambiguous"}, "curriculum package records")
    seen_records = parse_nonnegative_int(records["seen"], "curriculum package records.seen")
    unique_records = parse_nonnegative_int(records["unique"], "curriculum package records.unique")
    duplicate_records = parse_nonnegative_int(records["duplicates_removed"], "curriculum package records.duplicates_removed")
    require(unique_records + duplicate_records == seen_records, "curriculum package records summary is inconsistent")
    unassigned_records = parse_nonnegative_int(records["unassigned"], "curriculum package records.unassigned")
    ambiguous_records = parse_nonnegative_int(records["ambiguous"], "curriculum package records.ambiguous")
    require(unassigned_records <= unique_records, "curriculum package unassigned count exceeds unique records")
    require(ambiguous_records <= unique_records, "curriculum package ambiguous count exceeds unique records")
    require(assigned_records + unassigned_records == unique_records,
            "curriculum package stage assignment counts are inconsistent")
    require(ambiguous_records == ambiguous_stage_records,
            "curriculum package ambiguous count differs from stage summaries")

    heldout_summary = package["heldout"]
    require(isinstance(heldout_summary, Mapping), "curriculum package held-out summary is malformed")
    exact_fields(heldout_summary, {"record_digest_stream_schema", "record_digest_stream_sha256", "raw_text_persisted", "contract"},
                 "curriculum package held-out summary")
    require(heldout_summary["record_digest_stream_schema"] == "stage-nul-split-nul-text-sha256/v1",
            "curriculum package held-out digest schema is unsupported")
    heldout_digest = heldout_summary["record_digest_stream_sha256"]
    require(isinstance(heldout_digest, str) and len(heldout_digest) == 64
            and all(character in "0123456789abcdef" for character in heldout_digest),
            "curriculum package held-out digest is malformed")
    require(heldout_summary["raw_text_persisted"] is False, "curriculum package must not persist held-out raw text")
    validate_text(heldout_summary["contract"], "curriculum package held-out contract")

    capacity = package["capacity_preflight"]
    require(isinstance(capacity, Mapping), "curriculum package capacity preflight is malformed")
    exact_fields(capacity, {"unique_training_tokens_min_frequency_1", "minimum_continual_max_vocabulary", "contract"},
                 "curriculum package capacity preflight")
    vocabulary_count = parse_nonnegative_int(capacity["unique_training_tokens_min_frequency_1"],
                                              "curriculum package capacity preflight vocabulary count")
    require(capacity["minimum_continual_max_vocabulary"] == vocabulary_count,
            "curriculum package capacity preflight must not silently lower its vocabulary bound")
    validate_text(capacity["contract"], "curriculum package capacity preflight contract")
    validate_text(package["training_boundary"], "curriculum package training boundary")

    if arguments.source is not None:
        expected_input_digest, expected_input_bytes = validate_identity_mapping(package["input"], "curriculum package.input")
        observed_source = file_identity(arguments.source, "provided curriculum source")
        require(observed_source.sha256 == expected_input_digest and observed_source.bytes == expected_input_bytes,
                "provided source differs from the curriculum-bound input")
    if arguments.policy is not None:
        expected_policy_digest, expected_policy_bytes = validate_identity_mapping(package["policy"], "curriculum package.policy")
        observed_policy = file_identity(arguments.policy, "provided curriculum policy")
        require(observed_policy.sha256 == expected_policy_digest and observed_policy.bytes == expected_policy_bytes,
                "provided policy differs from the curriculum-bound policy")

    replayed_files = 0
    if arguments.source is not None and arguments.policy is not None:
        # A pair of matching hashes proves supplied input identity.  Replaying
        # it into an isolated temporary package additionally proves that the
        # package's selection counts, held-out digest, stage constraints,
        # ledger order, and payloads are the deterministic result of exactly
        # that policy/source pair rather than self-reported JSON values.
        with tempfile.TemporaryDirectory(prefix="vegai-curriculum-replay-") as temporary:
            replay_root = Path(temporary) / "package"
            replay_arguments = argparse.Namespace(
                input=arguments.source,
                policy=arguments.policy,
                output_dir=replay_root,
                input_format=input_identity["format"],
                input_compression=input_identity["compression"],
                batch_size=DEFAULT_BATCH_SIZE,
            )
            build_package(
                replay_arguments,
                report_source_path=input_identity["path"],
                report_policy_path=policy_identity["path"],
            )
            replayed_files = require_identical_package_tree(package_root, replay_root)

    # Do not report a mixture of files if a publisher or other process rewrote
    # the package while it was being checked.  The final name/layout check also
    # rejects a newly inserted unbound file.
    for shard_path, observed_shard in observed_shards:
        require(file_identity(shard_path, "curriculum shard") == observed_shard,
                f"curriculum shard changed during verification: {shard_path}")
    require(file_identity(ledger_path, "curriculum ledger") == observed_ledger,
            "curriculum ledger changed during verification")
    require(file_identity(package_path, "curriculum package") == package_identity,
            "curriculum package changed during verification")
    validate_package_layout(package_root, shard_paths)
    return {
        "schema_version": "vegai-capability-curriculum-verification/v1",
        "package": package_identity.to_mapping(),
        "ledger": observed_ledger.to_mapping(),
        "verified_stages": len(raw_shards),
        "verified_training_records": verified_records,
        "verified_accepted_tokens": verified_tokens,
        "source_rechecked": arguments.source is not None,
        "policy_rechecked": arguments.policy is not None,
        "source_policy_replayed": arguments.source is not None and arguments.policy is not None,
        "replayed_files": replayed_files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="compile a data-driven immutable curriculum package")
    build.add_argument("input", type=Path, help="operator-governed training-record stream")
    build.add_argument("--policy", required=True, type=Path, help=f"strict {POLICY_SCHEMA} JSON policy")
    build.add_argument("--output-dir", required=True, type=Path, help="new curriculum package directory; must not exist")
    build.add_argument("--input-format", choices=("auto", "jsonl", "json", "txt", "parquet", "arrow"), default="auto")
    build.add_argument("--input-compression", choices=("auto", "none", "gzip"), default="auto")
    build.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    verify = commands.add_parser(
        "verify",
        help="verify a published package; with both source/policy, replay it privately for full derivation checking",
    )
    verify.add_argument("--package", required=True, type=Path, help="curriculum package directory or curriculum.json")
    verify.add_argument("--source", type=Path, help="optional original source to hash-recheck")
    verify.add_argument("--policy", type=Path, help="optional original policy to hash-recheck")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        if getattr(arguments, "batch_size", DEFAULT_BATCH_SIZE) <= 0:
            raise CapabilityCurriculumError("--batch-size must be positive")
        result = build_package(arguments) if arguments.command == "build" else verify_package(arguments)
    except (CapabilityCurriculumError, TrainingDataError, OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as error:
        print(f"capability curriculum: FAIL: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
