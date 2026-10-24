#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create a deterministic local-corpus manifest for Vegai HDC pretraining.

This tool never downloads data. It fingerprints a caller-provided local file and
runs a bounded preflight that mirrors the repository's ASCII ingest contract.
The Fortran tokenizer remains the normative training implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

DEFAULT_MAX_LINE_BYTES = 8192
DEFAULT_TOKEN_LENGTH = 48
DEFAULT_MAX_TOKENS_PER_LINE = 1024
CHUNK_BYTES = 1024 * 1024


@dataclass
class PreflightStats:
    records: int = 0
    overlong_records: int = 0
    records_with_tokenizer_truncation: int = 0
    accepted_tokens: int = 0
    dropped_tokens: int = 0
    maximum_record_bytes: int = 0
    non_ascii_bytes: int = 0
    crlf_records: int = 0


def is_token_byte(value: int) -> bool:
    return (
        65 <= value <= 90
        or 97 <= value <= 122
        or 48 <= value <= 57
        or value == 95
        or value == 39
    )


def inspect_short_record(
    record: bytes,
    stats: PreflightStats,
    token_length: int,
    max_tokens: int,
) -> None:
    """Mirror the tokenizer's drop-not-truncate behavior for one valid record."""
    token_length_seen = 0
    token_too_long = False
    emitted_tokens = 0
    record_truncated = False

    def emit() -> None:
        nonlocal token_length_seen, token_too_long, emitted_tokens, record_truncated
        if token_length_seen == 0 and not token_too_long:
            return
        if token_too_long or emitted_tokens >= max_tokens:
            stats.dropped_tokens += 1
            record_truncated = True
        else:
            emitted_tokens += 1
            stats.accepted_tokens += 1
        token_length_seen = 0
        token_too_long = False

    for value in record:
        if value > 127:
            stats.non_ascii_bytes += 1
        if is_token_byte(value):
            if token_length_seen < token_length:
                token_length_seen += 1
            else:
                token_too_long = True
        else:
            emit()
    emit()
    if record_truncated:
        stats.records_with_tokenizer_truncation += 1


def finalize_record(
    prefix: bytearray,
    record_bytes: int,
    stats: PreflightStats,
    max_line_bytes: int,
    token_length: int,
    max_tokens: int,
) -> None:
    """Account for one physical LF-delimited record without retaining huge input."""
    stats.records += 1
    if prefix and prefix[-1:] == b"\r":
        # CRLF is surfaced as telemetry rather than normalized silently.  The
        # Fortran Linux ingest treats CR as an in-record delimiter byte, so it
        # contributes to the bounded physical-record length just like any byte.
        stats.crlf_records += 1
    stats.maximum_record_bytes = max(stats.maximum_record_bytes, record_bytes)
    if record_bytes > max_line_bytes:
        stats.overlong_records += 1
        return
    record = bytes(prefix[:record_bytes])
    inspect_short_record(record, stats, token_length, max_tokens)


def scan_corpus(
    handle: BinaryIO,
    max_line_bytes: int,
    token_length: int,
    max_tokens: int,
) -> tuple[str, int, PreflightStats]:
    digest = hashlib.sha256()
    stats = PreflightStats()
    prefix = bytearray()
    record_bytes = 0
    in_record = False
    total_bytes = 0

    while chunk := handle.read(CHUNK_BYTES):
        digest.update(chunk)
        total_bytes += len(chunk)
        for value in chunk:
            if value == 10:  # LF
                finalize_record(prefix, record_bytes, stats, max_line_bytes, token_length, max_tokens)
                prefix.clear()
                record_bytes = 0
                in_record = False
                continue
            in_record = True
            record_bytes += 1
            # Store at most max+one byte: enough to identify an overlong record
            # and enough full content to tokenize every in-contract record.
            if len(prefix) <= max_line_bytes:
                prefix.append(value)

    if in_record:
        finalize_record(prefix, record_bytes, stats, max_line_bytes, token_length, max_tokens)
    return digest.hexdigest(), total_bytes, stats


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic local-corpus metadata for Vegai HDC pretraining."
    )
    parser.add_argument("input", type=Path, help="local corpus file; no network fetch is performed")
    parser.add_argument("--output", type=str, default="-", help="manifest JSON path, or - for stdout")
    parser.add_argument("--source", default="", help="human-recorded source URI or dataset identifier")
    parser.add_argument("--license", default="", help="human-recorded license/usage terms identifier")
    parser.add_argument("--retrieved-at", default="", help="human-recorded source retrieval timestamp")
    parser.add_argument("--preprocessing", default="", help="canonical preprocessing command/version")
    parser.add_argument("--max-line-bytes", type=int, default=DEFAULT_MAX_LINE_BYTES)
    parser.add_argument("--token-length", type=int, default=DEFAULT_TOKEN_LENGTH)
    parser.add_argument("--max-tokens-per-line", type=int, default=DEFAULT_MAX_TOKENS_PER_LINE)
    parser.add_argument("--expect-sha256", default="", help="fail unless the input digest exactly matches")
    parser.add_argument(
        "--strict-contract",
        action="store_true",
        help="fail if the preflight finds overlong records or tokenizer drops",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.max_line_bytes <= 0 or args.token_length <= 0 or args.max_tokens_per_line <= 0:
        print("error: contract limits must be positive", file=sys.stderr)
        return 2
    try:
        with args.input.open("rb") as handle:
            digest, byte_count, stats = scan_corpus(
                handle, args.max_line_bytes, args.token_length, args.max_tokens_per_line
            )
    except OSError as exc:
        print(f"error: cannot read {args.input}: {exc}", file=sys.stderr)
        return 2

    if args.expect_sha256 and digest.lower() != args.expect_sha256.lower():
        print("error: SHA-256 does not match --expect-sha256", file=sys.stderr)
        return 3

    manifest = {
        "schema": "vegai-corpus-manifest/v1",
        "input": {
            "path": str(args.input),
            "sha256": digest,
            "bytes": byte_count,
            "physical_lf_records": stats.records,
        },
        "declared_provenance": {
            "source": args.source,
            "license": args.license,
            "retrieved_at": args.retrieved_at,
            "preprocessing": args.preprocessing,
        },
        "vegai_ascii_ingest_contract": {
            "max_line_bytes": args.max_line_bytes,
            "token_length_bytes": args.token_length,
            "max_tokens_per_line": args.max_tokens_per_line,
            "token_bytes": "A-Z a-z 0-9 underscore apostrophe; ASCII lowercase",
        },
        "preflight": {
            "overlong_records": stats.overlong_records,
            "records_with_tokenizer_truncation": stats.records_with_tokenizer_truncation,
            "accepted_tokens": stats.accepted_tokens,
            "dropped_tokens": stats.dropped_tokens,
            "maximum_record_bytes": stats.maximum_record_bytes,
            "non_ascii_bytes": stats.non_ascii_bytes,
            "crlf_records": stats.crlf_records,
        },
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(rendered)
    else:
        output_path = Path(args.output)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write {output_path}: {exc}", file=sys.stderr)
            return 2

    if args.strict_contract and (
        stats.overlong_records > 0 or stats.records_with_tokenizer_truncation > 0
    ):
        print("error: corpus violates the requested strict ingest contract", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
