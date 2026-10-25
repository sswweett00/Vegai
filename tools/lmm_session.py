#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Crash-safe hash-chained conversation ledger for long LMM dialogue context.

The ledger stores user/assistant/tool/system messages as strict JSONL events.
It does not summarize, rewrite, or invent history.  ``compile`` turns a
verified ledger into a ``vegai-lmm-request/v1`` that ``lmm_gateway.py`` packs
to the selected model's exact token budget.  This separates durable dialogue
state from model-specific long-context eviction policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping

from lmm_contract import (
    MAX_JSON_BYTES,
    MAX_MESSAGES,
    ConversationMessage,
    GenerationRequest,
    LMMContractError,
    canonical_json_bytes,
    file_identity,
    require,
    require_unchanged,
    validate_identifier,
    validate_sha256,
    write_json_atomic,
)


SESSION_EVENT_SCHEMA = "vegai-lmm-session-event/v1"
GENESIS = "GENESIS"
MAX_EVENT_LINE_BYTES = 4 * 1024 * 1024


def event_payload(sequence: int, previous_sha256: str, message: ConversationMessage) -> dict[str, object]:
    return {
        "schema_version": SESSION_EVENT_SCHEMA,
        "sequence": sequence,
        "previous_sha256": previous_sha256,
        "message": message.to_mapping(),
    }


def event_digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def parse_event(value: object, number: int, expected_previous: str) -> tuple[ConversationMessage, str]:
    require(isinstance(value, Mapping), f"session event {number} must be an object")
    fields = {"schema_version", "sequence", "previous_sha256", "message", "event_sha256"}
    unknown = sorted(str(key) for key in value if key not in fields)
    require(not unknown, f"session event {number} has unsupported fields: {', '.join(unknown)}")
    missing = sorted(fields.difference(value))
    require(not missing, f"session event {number} is missing fields: {', '.join(missing)}")
    require(value["schema_version"] == SESSION_EVENT_SCHEMA, f"session event {number} schema is unsupported")
    require(type(value["sequence"]) is int and value["sequence"] == number,
            f"session event {number} sequence is not contiguous")
    previous = value["previous_sha256"]
    require(previous == expected_previous, f"session event {number} previous_sha256 breaks the hash chain")
    if previous != GENESIS:
        validate_sha256(previous, f"session event {number}.previous_sha256")
    message = ConversationMessage.from_mapping(value["message"], f"session event {number}.message")
    payload = event_payload(number, previous, message)
    digest = validate_sha256(value["event_sha256"], f"session event {number}.event_sha256")
    require(digest == event_digest(payload), f"session event {number} event_sha256 does not bind its payload")
    return message, digest


def iter_events(path: Path) -> Iterator[tuple[ConversationMessage, str]]:
    source = path.resolve()
    require(source.is_file(), f"session ledger is not a regular file: {source}")
    identity = file_identity(source, "session ledger", maximum_bytes=MAX_JSON_BYTES)
    expected_previous = GENESIS
    try:
        with source.open("rb") as handle:
            number = 0
            limit = MAX_EVENT_LINE_BYTES + 2
            while True:
                raw = handle.readline(limit)
                if not raw:
                    break
                number += 1
                has_lf = raw.endswith(b"\n")
                payload_bytes = len(raw) - 1 if has_lf else len(raw)
                overlong = payload_bytes > MAX_EVENT_LINE_BYTES or (not has_lf and len(raw) == limit)
                if overlong and not has_lf:
                    while raw and not raw.endswith(b"\n"):
                        raw = handle.readline(limit)
                require(not overlong, f"session event {number} exceeds {MAX_EVENT_LINE_BYTES} bytes")
                require(has_lf, f"session event {number} must terminate with LF")
                raw = raw[:-1]
                require(not raw.endswith(b"\r"), f"session event {number} uses CRLF; strict LF is required")
                require(bool(raw.strip()), f"session event {number} is blank")
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise LMMContractError(f"session event {number} is invalid JSON: {error}") from error
                message, digest = parse_event(parsed, number, expected_previous)
                expected_previous = digest
                yield message, digest
    except OSError as error:
        raise LMMContractError(f"cannot read session ledger {source}: {error}") from error
    require(file_identity(source, "session ledger", maximum_bytes=MAX_JSON_BYTES) == identity,
            "session ledger changed while it was being verified")


def load_events(path: Path) -> tuple[list[ConversationMessage], str]:
    messages: list[ConversationMessage] = []
    head = GENESIS
    for message, digest in iter_events(path):
        messages.append(message)
        head = digest
        require(len(messages) <= MAX_MESSAGES, f"session ledger exceeds {MAX_MESSAGES} messages")
    return messages, head


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


def write_ledger_atomic(path: Path, messages: list[ConversationMessage]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered: list[bytes] = []
    previous = GENESIS
    for sequence, message in enumerate(messages, start=1):
        payload = event_payload(sequence, previous, message)
        digest = event_digest(payload)
        event = dict(payload)
        event["event_sha256"] = digest
        rendered.append(canonical_json_bytes(event) + b"\n")
        previous = digest
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for line in rendered:
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return previous


class LedgerLock:
    """Portable lock-file guard; stale locks are intentionally never deleted silently."""

    def __init__(self, ledger: Path, timeout_seconds: float) -> None:
        self.path = ledger.with_name(f".{ledger.name}.lock")
        self.timeout_seconds = timeout_seconds
        self.descriptor: int | None = None

    def __enter__(self) -> "LedgerLock":
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(self.descriptor, f"pid={os.getpid()}\n".encode("ascii"))
                os.fsync(self.descriptor)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise LMMContractError(f"timed out waiting for session lock {self.path}; inspect/remove only after owner review")
                time.sleep(0.05)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
        self.path.unlink(missing_ok=True)


def read_message(path: Path) -> ConversationMessage:
    identity = file_identity(path, "message JSON", maximum_bytes=MAX_JSON_BYTES)
    try:
        value = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LMMContractError(f"cannot parse message JSON {path}: {error}") from error
    require(file_identity(path, "message JSON", maximum_bytes=MAX_JSON_BYTES) == identity,
            "message JSON changed while it was being read")
    return ConversationMessage.from_mapping(value, "message JSON")


def append_event(arguments: argparse.Namespace) -> dict[str, object]:
    ledger = arguments.ledger.resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    message = read_message(arguments.message)
    with LedgerLock(ledger, arguments.lock_timeout_seconds):
        if ledger.exists():
            messages, head = load_events(ledger)
        else:
            messages, head = [], GENESIS
        require(arguments.expected_head == head,
                f"session head conflict: expected {arguments.expected_head}, actual {head}; reload ledger before append")
        require(message.message_id not in {entry.message_id for entry in messages},
                f"session message id already exists: {message.message_id!r}")
        messages.append(message)
        require(len(messages) <= MAX_MESSAGES, f"session ledger would exceed {MAX_MESSAGES} messages")
        new_head = write_ledger_atomic(ledger, messages)
        published_identity = file_identity(ledger, "published session ledger")
    return {
        "schema_version": "vegai-lmm-session-append-report/v1",
        "ledger": published_identity.to_mapping(),
        "messages": len(messages),
        "previous_head": head,
        "head": new_head,
    }


def compile_request(arguments: argparse.Namespace) -> dict[str, object]:
    ledger = arguments.ledger.resolve()
    # Use the same lock as append so a compiled request has one unambiguous
    # ledger head, instead of needlessly failing when a legitimate append races
    # an otherwise read-only compile.
    with LedgerLock(ledger, arguments.lock_timeout_seconds):
        ledger_identity = file_identity(ledger, "session ledger", maximum_bytes=MAX_JSON_BYTES)
        messages, head = load_events(ledger)
        require_unchanged(ledger, ledger_identity, "session ledger")
        require(messages, "cannot compile an empty session ledger")
        require(arguments.expected_head == head,
                f"session head conflict: expected {arguments.expected_head}, actual {head}")
        request = GenerationRequest(
            request_id=validate_identifier(arguments.request_id, "--request-id"),
            messages=tuple(messages),
            max_new_tokens=arguments.max_new_tokens,
            temperature=arguments.temperature,
            top_p=arguments.top_p,
        )
        # Run the same strict validation used for externally supplied requests.
        validated = GenerationRequest.from_mapping(request.to_mapping())
        require(arguments.output.resolve() != ledger, "--output must not overwrite session ledger")
        require_unchanged(ledger, ledger_identity, "session ledger")
        write_json_atomic(arguments.output, validated.to_mapping())
        require_unchanged(ledger, ledger_identity, "session ledger")
        report = {
            "schema_version": "vegai-lmm-session-compile-report/v1",
            "ledger": ledger_identity.to_mapping(),
            "head": head,
            "request": file_identity(arguments.output, "compiled request").to_mapping(),
            "messages": len(messages),
        }
    return report


def verify_ledger(arguments: argparse.Namespace) -> dict[str, object]:
    ledger = arguments.ledger.resolve()
    ledger_identity = file_identity(ledger, "session ledger", maximum_bytes=MAX_JSON_BYTES)
    messages, head = load_events(ledger)
    require_unchanged(ledger, ledger_identity, "session ledger")
    return {
        "schema_version": "vegai-lmm-session-verify-report/v1",
        "ledger": ledger_identity.to_mapping(),
        "messages": len(messages),
        "head": head,
    }


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number") from error
    if not parsed > 0.0 or not parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    append = subcommands.add_parser("append", help="append one message under an optimistic head check and lock")
    append.add_argument("--ledger", required=True, type=Path)
    append.add_argument("--message", required=True, type=Path, help="strict ConversationMessage JSON object")
    append.add_argument("--expected-head", required=True, help="GENESIS for an empty ledger, otherwise prior append/verify head")
    append.add_argument("--lock-timeout-seconds", type=parse_positive_float, default=30.0)
    compile_parser = subcommands.add_parser("compile", help="compile a verified ledger into a gateway request")
    compile_parser.add_argument("--ledger", required=True, type=Path)
    compile_parser.add_argument("--expected-head", required=True)
    compile_parser.add_argument("--request-id", required=True)
    compile_parser.add_argument("--max-new-tokens", type=int, required=True)
    compile_parser.add_argument("--temperature", type=float, required=True)
    compile_parser.add_argument("--top-p", type=float, required=True)
    compile_parser.add_argument("--output", required=True, type=Path)
    compile_parser.add_argument("--lock-timeout-seconds", type=parse_positive_float, default=30.0)
    verify = subcommands.add_parser("verify", help="verify strict physical rows and every hash-chain edge")
    verify.add_argument("--ledger", required=True, type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        if arguments.command == "append":
            result = append_event(arguments)
        elif arguments.command == "compile":
            result = compile_request(arguments)
        else:
            result = verify_ledger(arguments)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (LMMContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"lmm session: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
