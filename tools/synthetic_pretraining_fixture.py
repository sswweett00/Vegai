#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create and verify an exact-size synthetic corpus for HDC pretraining checks.

The fixture is deliberately *not* a source of factual, linguistic, or domain
knowledge.  It is a deterministic, strict-ASCII clustered-token topology used
to exercise Vegai's streaming vocabulary scan, context learning, checkpointing,
and throughput measurement without redistributing unlicensed text.

``generate`` writes exactly the requested number of bytes (9 MiB by default),
a deterministic JSON manifest, and then streaming-verifies both.  ``verify``
never loads the corpus as one string: it validates byte size, SHA-256, line and
token contracts, aggregate counts, and the manifest binding while scanning.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, BinaryIO, Iterable


SCHEMA = "vegai-synthetic-pretraining-fixture/v1"
ALGORITHM = "clustered-cyclic-ascii-topology/v1"
DEFAULT_TARGET_BYTES = 9 * 1024 * 1024
MAX_LINE_BYTES = 8192
MAX_TOKENS_PER_LINE = 1024
TOKEN_PATTERN = re.compile(r"vgai9m_fixture_cluster_\d{3}_token_\d{3}\Z")
_BYTE_QUANTITY = re.compile(r"\s*([0-9]+)\s*(?:([kmg]?i?b))?\s*\Z", re.IGNORECASE)


class FixtureError(RuntimeError):
    """Raised when a fixture or its manifest violates the checked contract."""


@dataclass(frozen=True, slots=True)
class FixtureConfig:
    """Deterministic topology and physical-size parameters for one corpus."""

    target_bytes: int = DEFAULT_TARGET_BYTES
    clusters: int = 32
    tokens_per_cluster: int = 32
    tokens_per_record: int = 16


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def parse_byte_quantity(value: str) -> int:
    """Parse a positive binary-byte quantity such as ``9MiB`` without floats."""

    match = _BYTE_QUANTITY.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("byte quantity must be an integer optionally followed by B, KiB, MiB, or GiB")
    amount = int(match.group(1))
    suffix = (match.group(2) or "b").lower()
    multiplier = {
        "b": 1,
        "kib": 1024,
        "kb": 1_000,
        "mib": 1024 * 1024,
        "mb": 1_000_000,
        "gib": 1024 * 1024 * 1024,
        "gb": 1_000_000_000,
    }.get(suffix)
    if multiplier is None or amount <= 0:
        raise argparse.ArgumentTypeError("byte quantity must be positive and use B, KiB, MiB, or GiB")
    return amount * multiplier


def token_for(cluster: int, token: int) -> str:
    """Return one tokenizer-valid, fixed-format vocabulary item."""

    return f"vgai9m_fixture_cluster_{cluster:03d}_token_{token:03d}"


def validate_config(config: FixtureConfig) -> None:
    require(config.target_bytes > 0, "target bytes must be positive")
    require(1 <= config.clusters <= 1000, "clusters must be in [1, 1000]")
    require(1 <= config.tokens_per_cluster <= 1000, "tokens per cluster must be in [1, 1000]")
    require(2 <= config.tokens_per_record <= MAX_TOKENS_PER_LINE, "tokens per record must be in [2, 1024]")
    longest = token_for(config.clusters - 1, config.tokens_per_cluster - 1)
    require(len(longest) <= 48, "generated tokens must fit Vegai's 48-byte tokenizer token limit")
    # The greedy writer leaves one padded final record. Bound the sum of a
    # full-record body and that minimum final body so even the largest possible
    # remainder obeys the Fortran formatted-line contract.
    full_record_body_bytes = config.tokens_per_record * len(longest) + (config.tokens_per_record - 1)
    final_record_body_bytes = len(_minimum_final_record(config)) - 1
    require(
        full_record_body_bytes + final_record_body_bytes <= MAX_LINE_BYTES,
        "record topology cannot preserve the 8192-byte line contract",
    )
    # The final, padded record needs at least two valid tokens and LF.
    require(config.target_bytes >= len(_minimum_final_record(config)), "target is too small for a valid final record")


def _minimum_final_record(config: FixtureConfig) -> bytes:
    second = 1 if config.tokens_per_cluster > 1 else 0
    return f"{token_for(0, 0)} {token_for(0, second)}\n".encode("ascii")


def record_tokens(record_index: int, config: FixtureConfig) -> list[str]:
    """Create one clustered record, occasionally retaining a deterministic bridge."""

    cluster = record_index % config.clusters
    round_index = record_index // config.clusters
    phase = (round_index * 5 + cluster * 3) % config.tokens_per_cluster
    stride = 1 if config.tokens_per_cluster == 1 else 1 + (round_index % (config.tokens_per_cluster - 1))
    tokens = [
        token_for(cluster, (phase + position * stride) % config.tokens_per_cluster)
        for position in range(config.tokens_per_record)
    ]
    # A sparse, deterministic adjacent-cluster bridge gives the fixture a
    # nontrivial but still known topology. It is not semantic supervision.
    if config.clusters > 1 and config.tokens_per_record >= 4 and record_index % 7 == 0:
        bridge_cluster = (cluster + 1) % config.clusters
        tokens[-2] = token_for(bridge_cluster, phase)
        tokens[-1] = token_for(bridge_cluster, (phase + 1) % config.tokens_per_cluster)
    return tokens


def full_record(record_index: int, config: FixtureConfig) -> tuple[bytes, list[str]]:
    tokens = record_tokens(record_index, config)
    return (" ".join(tokens) + "\n").encode("ascii"), tokens


def _atomic_replace(path: Path, content: bytes) -> None:
    """Atomically replace a small metadata file in the destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_open_for_corpus(path: Path) -> tuple[Path, BinaryIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return Path(temporary_name), os.fdopen(descriptor, "wb")


def _increment_counts(counts: dict[str, int], tokens: Iterable[str]) -> None:
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1


def _write_corpus(path: Path, config: FixtureConfig) -> dict[str, Any]:
    """Stream deterministic records to an exact physical byte budget."""

    temporary, handle = _atomic_open_for_corpus(path)
    digest = hashlib.sha256()
    written = 0
    record_count = 0
    raw_tokens = 0
    maximum_line_bytes = 0
    token_counts: dict[str, int] = {}
    try:
        minimum_final = _minimum_final_record(config)
        record_index = 0
        with handle:
            while True:
                encoded, tokens = full_record(record_index, config)
                # Preserve room for a final valid record so padding never forms
                # a token that changes the topology or tokenizer contract.
                if written + len(encoded) + len(minimum_final) > config.target_bytes:
                    break
                handle.write(encoded)
                digest.update(encoded)
                written += len(encoded)
                record_count += 1
                raw_tokens += len(tokens)
                maximum_line_bytes = max(maximum_line_bytes, len(encoded) - 1)
                _increment_counts(token_counts, tokens)
                record_index += 1

            remaining = config.target_bytes - written
            require(remaining >= len(minimum_final), "internal final-record budget invariant failed")
            final_tokens = [token_for(0, 0), token_for(0, 1 if config.tokens_per_cluster > 1 else 0)]
            padding = remaining - len(minimum_final)
            encoded_final = minimum_final[:-1] + (b" " * padding) + b"\n"
            require(len(encoded_final) == remaining, "internal exact-size invariant failed")
            handle.write(encoded_final)
            digest.update(encoded_final)
            written += len(encoded_final)
            record_count += 1
            raw_tokens += len(final_tokens)
            maximum_line_bytes = max(maximum_line_bytes, len(encoded_final) - 1)
            _increment_counts(token_counts, final_tokens)
            handle.flush()
            os.fsync(handle.fileno())
        require(written == config.target_bytes, "internal corpus byte-count invariant failed")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    frequencies = list(token_counts.values())
    require(frequencies, "internal corpus emitted no tokens")
    return {
        "bytes": written,
        "sha256": digest.hexdigest(),
        "physical_records": record_count,
        "raw_tokens": raw_tokens,
        "unique_tokens": len(token_counts),
        "minimum_token_frequency": min(frequencies),
        "maximum_token_frequency": max(frequencies),
        "maximum_line_bytes": maximum_line_bytes,
    }


def _manifest_for(corpus: Path, config: FixtureConfig, statistics: dict[str, Any]) -> dict[str, Any]:
    probe_cluster = min(1, config.clusters - 1)
    probe_token = min(1, config.tokens_per_cluster - 1)
    return {
        "schema": SCHEMA,
        "generator": {
            "algorithm": ALGORITHM,
            "deterministic": True,
            "configuration": asdict(config),
        },
        "corpus": {
            "filename": corpus.name,
            "bytes": statistics["bytes"],
            "sha256": statistics["sha256"],
            "encoding": "US-ASCII",
            "line_ending": "LF",
            "streaming_contract": "validated line by line; no corpus-wide text load is required",
        },
        "records": {
            "physical_records": statistics["physical_records"],
            "raw_tokens": statistics["raw_tokens"],
            "unique_tokens": statistics["unique_tokens"],
            "minimum_token_frequency": statistics["minimum_token_frequency"],
            "maximum_token_frequency": statistics["maximum_token_frequency"],
            "maximum_line_bytes": statistics["maximum_line_bytes"],
            "maximum_allowed_line_bytes": MAX_LINE_BYTES,
            "maximum_allowed_tokens_per_line": MAX_TOKENS_PER_LINE,
        },
        "training_contract": {
            "recommended_max_vocabulary": statistics["unique_tokens"],
            "minimum_frequency_upper_bound": statistics["minimum_token_frequency"],
            "probes": [token_for(0, 0), token_for(probe_cluster, probe_token)],
        },
        "provenance": {
            "kind": "synthetic deterministic structural fixture",
            "source_material": "none; algorithmically generated",
            "license": "not applicable; no third-party text is included",
            "permitted_use": "streaming/pretraining plumbing, checkpoint, determinism, and throughput regression checks",
            "prohibited_interpretation": (
                "This is not a factual, natural-language, domain, safety, or production-quality training corpus; "
                "a successful run is not evidence of model knowledge or general capability."
            ),
        },
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FixtureError(f"manifest contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixtureError(f"cannot read fixture manifest {path}: {error}") from error
    require(isinstance(loaded, dict), "fixture manifest root must be an object")
    return loaded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_statistics(path: Path) -> dict[str, Any]:
    """Validate lexical/physical corpus invariants while calculating aggregates."""

    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    raw_tokens = 0
    maximum_line_bytes = 0
    token_counts: dict[str, int] = {}
    try:
        before = path.stat()
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                byte_count += len(line)
                require(line.endswith(b"\n"), f"record {record_count + 1} does not end in LF")
                require(b"\r" not in line, f"record {record_count + 1} contains CR; only LF is allowed")
                body = line[:-1]
                require(0 < len(body) <= MAX_LINE_BYTES, f"record {record_count + 1} exceeds line-size contract")
                maximum_line_bytes = max(maximum_line_bytes, len(body))
                for byte in body:
                    allowed = (ord("a") <= byte <= ord("z")) or (ord("0") <= byte <= ord("9")) or byte in (ord("_"), ord(" "))
                    require(allowed, f"record {record_count + 1} contains a non-strict-ASCII fixture byte")
                pieces = body.split(b" ")
                tokens = [piece for piece in pieces if piece]
                require(tokens, f"record {record_count + 1} has no tokens")
                require(len(tokens) <= MAX_TOKENS_PER_LINE, f"record {record_count + 1} exceeds token-count contract")
                for raw_token in tokens:
                    try:
                        token = raw_token.decode("ascii")
                    except UnicodeDecodeError as error:  # guarded above; retain a clear error if changed
                        raise FixtureError(f"record {record_count + 1} contains non-ASCII token bytes") from error
                    require(TOKEN_PATTERN.fullmatch(token) is not None, f"record {record_count + 1} has malformed token {token!r}")
                    token_counts[token] = token_counts.get(token, 0) + 1
                record_count += 1
                raw_tokens += len(tokens)
        after = path.stat()
    except OSError as error:
        raise FixtureError(f"cannot stream fixture corpus {path}: {error}") from error
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "corpus changed while its integrity scan was running",
    )
    frequencies = list(token_counts.values())
    require(record_count > 0 and frequencies, "fixture corpus must contain records and tokens")
    return {
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "physical_records": record_count,
        "raw_tokens": raw_tokens,
        "unique_tokens": len(token_counts),
        "minimum_token_frequency": min(frequencies),
        "maximum_token_frequency": max(frequencies),
        "maximum_line_bytes": maximum_line_bytes,
        # Kept internal to verification so declared probes can be checked
        # without materializing corpus text. The bounded vocabulary set is not
        # included in the public verification result.
        "_observed_tokens": frozenset(token_counts),
    }


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"manifest field {field!r} must be an object")
    return value


def _require_integer(value: Any, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"manifest field {field!r} must be an integer")
    return value


def verify_fixture(corpus: Path, manifest_path: Path) -> dict[str, Any]:
    """Fail closed unless corpus bytes and every manifest aggregate agree."""

    corpus = corpus.resolve()
    manifest_path = manifest_path.resolve()
    require(corpus.is_file(), f"fixture corpus does not exist or is not a regular file: {corpus}")
    require(manifest_path.is_file(), f"fixture manifest does not exist or is not a regular file: {manifest_path}")
    manifest = _load_json_object(manifest_path)
    require(manifest.get("schema") == SCHEMA, f"unsupported fixture schema: {manifest.get('schema')!r}")
    generator = _require_mapping(manifest.get("generator"), "generator")
    require(generator.get("algorithm") == ALGORITHM, "fixture manifest algorithm is not recognized")
    require(generator.get("deterministic") is True, "fixture manifest must declare deterministic generation")
    configuration = _require_mapping(generator.get("configuration"), "generator.configuration")
    try:
        config = FixtureConfig(
            target_bytes=_require_integer(configuration.get("target_bytes"), "generator.configuration.target_bytes"),
            clusters=_require_integer(configuration.get("clusters"), "generator.configuration.clusters"),
            tokens_per_cluster=_require_integer(
                configuration.get("tokens_per_cluster"), "generator.configuration.tokens_per_cluster"
            ),
            tokens_per_record=_require_integer(
                configuration.get("tokens_per_record"), "generator.configuration.tokens_per_record"
            ),
        )
        validate_config(config)
    except (TypeError, ValueError) as error:
        raise FixtureError(f"invalid fixture generator configuration: {error}") from error

    corpus_metadata = _require_mapping(manifest.get("corpus"), "corpus")
    records = _require_mapping(manifest.get("records"), "records")
    training_contract = _require_mapping(manifest.get("training_contract"), "training_contract")
    provenance = _require_mapping(manifest.get("provenance"), "provenance")
    require(corpus_metadata.get("filename") == corpus.name, "manifest corpus filename does not bind this corpus")
    require(corpus_metadata.get("encoding") == "US-ASCII", "fixture must be declared US-ASCII")
    require(corpus_metadata.get("line_ending") == "LF", "fixture must be declared LF-only")
    require(provenance.get("kind") == "synthetic deterministic structural fixture", "fixture provenance kind is not safe")
    require(provenance.get("source_material") == "none; algorithmically generated", "fixture provenance source is not safe")
    require(isinstance(provenance.get("prohibited_interpretation"), str), "fixture must state its capability limitation")

    statistics = _stream_statistics(corpus)
    for field in (
        "bytes",
        "sha256",
    ):
        require(corpus_metadata.get(field) == statistics[field], f"manifest corpus {field} does not match streamed corpus")
    for field in (
        "physical_records",
        "raw_tokens",
        "unique_tokens",
        "minimum_token_frequency",
        "maximum_token_frequency",
        "maximum_line_bytes",
    ):
        require(records.get(field) == statistics[field], f"manifest records.{field} does not match streamed corpus")
    require(records.get("maximum_allowed_line_bytes") == MAX_LINE_BYTES, "manifest line-size bound differs from runtime contract")
    require(
        records.get("maximum_allowed_tokens_per_line") == MAX_TOKENS_PER_LINE,
        "manifest token-count bound differs from runtime contract",
    )
    require(statistics["bytes"] == config.target_bytes, "corpus byte size does not equal the declared exact target")
    require(
        training_contract.get("recommended_max_vocabulary") == statistics["unique_tokens"],
        "recommended vocabulary capacity does not match corpus vocabulary",
    )
    require(
        training_contract.get("minimum_frequency_upper_bound") == statistics["minimum_token_frequency"],
        "minimum frequency bound does not match corpus",
    )
    probes = training_contract.get("probes")
    require(isinstance(probes, list) and len(probes) >= 1 and all(isinstance(probe, str) for probe in probes), "manifest probes are invalid")
    # The corpus is small enough in vocabulary cardinality to confirm probes
    # with one token pass; no full-text materialization occurred.
    observed_tokens = statistics["_observed_tokens"]
    for probe in probes:
        require(TOKEN_PATTERN.fullmatch(probe) is not None, f"invalid probe token {probe!r}")
        require(probe in observed_tokens, f"manifest probe {probe!r} does not occur in the corpus")
    return {
        "schema": SCHEMA,
        "corpus": corpus_metadata,
        "records": records,
        "training_contract": training_contract,
        "generator_configuration": asdict(config),
        "manifest_sha256": sha256_file(manifest_path),
    }


def generate_fixture(corpus: Path, manifest_path: Path, config: FixtureConfig, *, overwrite: bool) -> dict[str, Any]:
    """Generate a corpus plus manifest atomically per output and verify the result."""

    corpus = corpus.resolve()
    manifest_path = manifest_path.resolve()
    validate_config(config)
    require(corpus != manifest_path, "corpus and manifest paths must differ")
    for destination, label in ((corpus, "corpus"), (manifest_path, "manifest")):
        if destination.exists() and not overwrite:
            raise FixtureError(f"{label} already exists (pass --overwrite only for this deterministic fixture): {destination}")
        if destination.exists() and destination.is_dir():
            raise FixtureError(f"{label} destination is a directory: {destination}")
    statistics = _write_corpus(corpus, config)
    manifest = _manifest_for(corpus, config, statistics)
    _atomic_replace(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return verify_fixture(corpus, manifest_path)


def default_manifest_path(corpus: Path) -> Path:
    return corpus.with_suffix(".manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="operation", required=True)

    generate = subcommands.add_parser("generate", help="write and verify an exact-size deterministic fixture")
    generate.add_argument("--output", required=True, type=Path, help="ignored runtime corpus output path")
    generate.add_argument("--manifest", type=Path, help="manifest path (default: OUTPUT with .manifest.json suffix)")
    generate.add_argument("--target-bytes", type=parse_byte_quantity, default=DEFAULT_TARGET_BYTES, help="exact physical corpus size (default: 9MiB)")
    generate.add_argument("--clusters", type=int, default=32, help="deterministic token clusters (default: 32)")
    generate.add_argument("--tokens-per-cluster", type=int, default=32, help="tokens in each cluster (default: 32)")
    generate.add_argument("--tokens-per-record", type=int, default=16, help="tokens per full record (default: 16)")
    generate.add_argument("--overwrite", action="store_true", help="replace only an existing deterministic fixture output")

    verify = subcommands.add_parser("verify", help="streaming-verify corpus bytes against its manifest")
    verify.add_argument("--corpus", required=True, type=Path, help="fixture corpus path")
    verify.add_argument("--manifest", type=Path, help="manifest path (default: CORPUS with .manifest.json suffix)")
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        if arguments.operation == "generate":
            manifest = arguments.manifest or default_manifest_path(arguments.output)
            result = generate_fixture(
                arguments.output,
                manifest,
                FixtureConfig(
                    target_bytes=arguments.target_bytes,
                    clusters=arguments.clusters,
                    tokens_per_cluster=arguments.tokens_per_cluster,
                    tokens_per_record=arguments.tokens_per_record,
                ),
                overwrite=arguments.overwrite,
            )
        else:
            manifest = arguments.manifest or default_manifest_path(arguments.corpus)
            result = verify_fixture(arguments.corpus, manifest)
    except FixtureError as error:
        print(f"synthetic pretraining fixture: FAIL: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
