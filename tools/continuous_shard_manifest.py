#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create an immutable ordered-shard manifest for lossless continuous training.

Every listed shard is scanned under Vegai's ASCII ingest contract and receives a
SHA-256 identity.  The command refuses data that the strict continuous path
would reject: tokenizer drops, overlong records, non-ASCII bytes, CRLF records,
or zero accepted tokens.  The resulting manifest is atomically replaced so a
follower never observes a partially written shard list.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from corpus_manifest import (
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_TOKENS_PER_LINE,
    DEFAULT_TOKEN_LENGTH,
    PreflightStats,
    scan_corpus,
)


SCHEMA = "vegai-continuous-shard-manifest/v1"


def strict_error(stats: PreflightStats) -> str | None:
    if stats.overlong_records:
        return f"{stats.overlong_records} overlong record(s)"
    if stats.records_with_tokenizer_truncation:
        return f"{stats.records_with_tokenizer_truncation} tokenizer-truncated record(s)"
    if stats.dropped_tokens:
        return f"{stats.dropped_tokens} dropped token(s)"
    if stats.non_ascii_bytes:
        return f"{stats.non_ascii_bytes} non-ASCII byte(s)"
    if stats.crlf_records:
        return f"{stats.crlf_records} CRLF record(s)"
    if stats.accepted_tokens == 0:
        return "zero accepted tokens"
    return None


def fsync_directory(directory: Path) -> None:
    """Best-effort parent-directory durability on POSIX without harming Windows."""

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


def atomic_write(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path, help="ordered immutable local shard files")
    parser.add_argument("--output", required=True, type=Path, help="manifest path to atomically create or replace")
    parser.add_argument("--source", default="", help="human-recorded dataset/source identifier")
    parser.add_argument("--license", default="", help="human-recorded license/usage identifier")
    parser.add_argument("--preprocessing", default="", help="normalization/preprocessing command and version")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output = arguments.output.resolve()
    output_parent = output.parent
    shards: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    try:
        for candidate in arguments.shards:
            path = candidate.resolve()
            if path == output:
                raise ValueError("manifest output must not overwrite one of its input shards")
            with path.open("rb") as handle:
                digest, byte_count, stats = scan_corpus(
                    handle,
                    DEFAULT_MAX_LINE_BYTES,
                    DEFAULT_TOKEN_LENGTH,
                    DEFAULT_MAX_TOKENS_PER_LINE,
                )
            violation = strict_error(stats)
            if violation is not None:
                raise ValueError(f"{path}: cannot enter lossless shard ledger ({violation})")
            if digest in seen_ids:
                raise ValueError(f"{path}: duplicate shard content SHA-256 {digest}")
            seen_ids.add(digest)
            shards.append(
                {
                    "id": digest,
                    "path": os.path.relpath(path, output_parent),
                    "sha256": digest,
                    "bytes": byte_count,
                    "physical_lf_records": stats.records,
                    "preflight": {
                        "accepted_tokens": stats.accepted_tokens,
                        "crlf_records": stats.crlf_records,
                        "dropped_tokens": stats.dropped_tokens,
                        "maximum_record_bytes": stats.maximum_record_bytes,
                        "non_ascii_bytes": stats.non_ascii_bytes,
                        "overlong_records": stats.overlong_records,
                        "records_with_tokenizer_truncation": stats.records_with_tokenizer_truncation,
                    },
                }
            )
        manifest = {
            "schema": SCHEMA,
            "declared_provenance": {
                "source": arguments.source,
                "license": arguments.license,
                "preprocessing": arguments.preprocessing,
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
        atomic_write(output, json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError) as error:
        print(f"continuous shard manifest: FAIL: {error}", file=sys.stderr)
        return 1

    print(f"continuous shard manifest: PASS ({len(shards)} immutable shard(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
