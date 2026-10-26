#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Content-addressed image/audio/video assets for the optional Vegai LMM plane.

The tool never claims to understand media.  It validates a bounded local file,
assigns an immutable SHA-256 identity, records source/license provenance, and
makes the asset available to a backend that *actually* declares the relevant
multimodal capability.  Decoding/embedding/fusion remain the selected model
backend's responsibility.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from lmm_contract import (
    MAX_ASSET_CONTEXT_TOKENS,
    AssetRecord,
    AssetRegistry,
    FileIdentity,
    LMMContractError,
    file_identity,
    parse_positive_int,
    require,
    require_unchanged,
    validate_identifier,
    validate_json_object,
    validate_unicode_text,
)


DEFAULT_MAX_ASSET_BYTES = 4 * 1024 * 1024 * 1024
_EXTENSION_MIME: dict[str, tuple[str, str]] = {
    ".png": ("image", "image/png"),
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".webp": ("image", "image/webp"),
    ".gif": ("image", "image/gif"),
    ".wav": ("audio", "audio/wav"),
    ".flac": ("audio", "audio/flac"),
    ".ogg": ("audio", "audio/ogg"),
    ".mp3": ("audio", "audio/mpeg"),
    ".m4a": ("audio", "audio/mp4"),
    ".mp4": ("video", "video/mp4"),
    ".mkv": ("video", "video/x-matroska"),
    ".webm": ("video", "video/webm"),
    ".mov": ("video", "video/quicktime"),
}


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


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def detect_media(path: Path) -> tuple[str, str, str]:
    """Return modality/MIME/safe extension using extension plus cheap magic checks."""
    suffix = path.suffix.lower()
    require(suffix in _EXTENSION_MIME, f"unsupported media extension {suffix!r}; specify a supported image/audio/video file")
    modality, mime_type = _EXTENSION_MIME[suffix]
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
    except OSError as error:
        raise LMMContractError(f"cannot inspect media file {path}: {error}") from error
    expected_magic: dict[str, bool] = {
        ".png": header.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": header.startswith(b"\xff\xd8\xff"),
        ".jpeg": header.startswith(b"\xff\xd8\xff"),
        ".webp": header.startswith(b"RIFF") and header[8:12] == b"WEBP",
        ".gif": header.startswith((b"GIF87a", b"GIF89a")),
        ".wav": header.startswith(b"RIFF") and header[8:12] == b"WAVE",
        ".flac": header.startswith(b"fLaC"),
        ".ogg": header.startswith(b"OggS"),
        ".mp3": header.startswith(b"ID3") or header.startswith(b"\xff\xfb") or header.startswith(b"\xff\xf3"),
        ".m4a": len(header) >= 12 and header[4:8] == b"ftyp",
        ".mp4": len(header) >= 12 and header[4:8] == b"ftyp",
        ".mov": len(header) >= 12 and header[4:8] == b"ftyp",
        ".mkv": header.startswith(b"\x1aE\xdf\xa3"),
        ".webm": header.startswith(b"\x1aE\xdf\xa3"),
    }
    require(expected_magic[suffix], f"media magic does not match {suffix!r}; refusing mislabeled asset")
    return modality, mime_type, suffix


def parse_metadata(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"metadata must be JSON: {error}") from error
    try:
        return validate_json_object(parsed, "asset metadata", maximum_bytes=64 * 1024)
    except LMMContractError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def copy_content_addressed(source: Path, store: Path, identity: FileIdentity, suffix: str) -> tuple[Path, str]:
    relative = f"sha256/{identity.sha256}{suffix}"
    target = (store / relative).resolve()
    store_root = store.resolve()
    try:
        target.relative_to(store_root)
    except ValueError as error:
        raise LMMContractError("content-addressed target escaped asset store") from error
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = file_identity(target, "existing content-addressed asset")
        require(existing.sha256 == identity.sha256 and existing.bytes == identity.bytes,
                f"content-addressed target collision at {target}")
        return target, relative

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        copied = file_identity(temporary, "copied content-addressed asset")
        require(copied.sha256 == identity.sha256 and copied.bytes == identity.bytes,
                "source asset changed while it was being content-addressed")
        try:
            os.link(temporary, target)
            temporary.unlink(missing_ok=True)
        except FileExistsError:
            existing = file_identity(target, "existing content-addressed asset")
            require(existing.sha256 == identity.sha256 and existing.bytes == identity.bytes,
                    f"content-addressed target collision at {target}")
            temporary.unlink(missing_ok=True)
        except OSError:
            # Cross-filesystem or platform-specific hard-link restrictions are
            # harmless: atomically publish a private copy instead.
            if target.exists():
                existing = file_identity(target, "existing content-addressed asset")
                require(existing.sha256 == identity.sha256 and existing.bytes == identity.bytes,
                        f"content-addressed target collision at {target}")
                temporary.unlink(missing_ok=True)
            else:
                os.replace(temporary, target)
        fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target, relative


class RegistryLock:
    """Serialize registry read-modify-write without silently breaking an ingest."""

    def __init__(self, registry: Path, timeout_seconds: float) -> None:
        self.path = registry.with_name(f".{registry.name}.lock")
        self.timeout_seconds = timeout_seconds
        self.descriptor: int | None = None

    def __enter__(self) -> "RegistryLock":
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(self.descriptor, f"pid={os.getpid()}\n".encode("ascii"))
                os.fsync(self.descriptor)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise LMMContractError(
                        f"timed out waiting for asset registry lock {self.path}; inspect/remove only after owner review"
                    )
                time.sleep(0.05)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
        self.path.unlink(missing_ok=True)


def load_registry_or_empty(path: Path) -> AssetRegistry:
    return AssetRegistry.from_jsonl(path) if path.exists() else AssetRegistry({})


def ingest(arguments: argparse.Namespace) -> dict[str, object]:
    source = arguments.input.resolve()
    require(source.is_file(), f"asset input is not a regular file: {source}")
    identity = file_identity(source, "asset input", maximum_bytes=arguments.max_asset_bytes)
    modality, inferred_mime, suffix = detect_media(source)
    if arguments.mime_type is not None:
        requested_mime = validate_unicode_text(arguments.mime_type, "--mime-type", maximum_bytes=256)
        require(requested_mime == inferred_mime,
                f"--mime-type {requested_mime!r} does not match verified file type {inferred_mime!r}")
    store = arguments.store.resolve()
    registry_path = arguments.registry.resolve()
    require(registry_path != source, "--registry must not overwrite the media input")
    _, relative_path = copy_content_addressed(source, store, identity, suffix)
    asset_id = validate_identifier(arguments.asset_id, "--id")
    metadata = dict(arguments.metadata)
    metadata.update({"ingested_by": "tools/lmm_assets.py", "verified_mime_type": inferred_mime})
    record = AssetRecord(
        asset_id=asset_id,
        modality=modality,
        mime_type=inferred_mime,
        relative_path=relative_path,
        sha256=identity.sha256,
        bytes=identity.bytes,
        context_tokens=parse_positive_int(arguments.context_tokens, "--context-tokens", MAX_ASSET_CONTEXT_TOKENS),
        source=validate_unicode_text(arguments.source, "--source", maximum_bytes=8 * 1024),
        license=validate_unicode_text(arguments.license, "--license", maximum_bytes=8 * 1024),
        retrieved_at=validate_unicode_text(arguments.retrieved_at, "--retrieved-at", maximum_bytes=1_024),
        metadata=validate_json_object(metadata, "asset metadata", maximum_bytes=64 * 1024),
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with RegistryLock(registry_path, arguments.lock_timeout_seconds):
        registry = load_registry_or_empty(registry_path)
        require(asset_id not in registry.assets, f"asset registry already contains id {asset_id!r}")
        records = dict(registry.assets)
        records[record.asset_id] = record
        published = AssetRegistry(records)
        write_bytes_atomic(registry_path, published.to_jsonl_bytes())
        registry_identity = file_identity(registry_path, "published asset registry")
    return {
        "schema_version": "vegai-multimodal-asset-ingest-report/v1",
        "asset": record.to_mapping(),
        "asset_store": str(store),
        "asset_input": identity.to_mapping(),
        "registry": registry_identity.to_mapping(),
    }


def verify(arguments: argparse.Namespace) -> dict[str, object]:
    registry_identity = file_identity(arguments.registry, "asset registry")
    registry = AssetRegistry.from_jsonl(arguments.registry)
    # This command audits the full registry, not only request-referenced assets.
    verified: dict[str, FileIdentity] = {}
    for asset_id, record in registry.assets.items():
        target = record.resolve(arguments.store)
        identity = file_identity(target, f"asset {asset_id}")
        require(identity.sha256 == record.sha256 and identity.bytes == record.bytes,
                f"asset {asset_id} does not match registry identity")
        verified[asset_id] = identity
    require_unchanged(arguments.registry, registry_identity, "asset registry")
    for asset_id, identity in verified.items():
        require_unchanged(Path(identity.path), identity, f"asset {asset_id}")
    return {
        "schema_version": "vegai-multimodal-asset-verify-report/v1",
        "registry": registry_identity.to_mapping(),
        "asset_store": str(arguments.store.resolve()),
        "verified_assets": len(verified),
    }


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number") from error
    if not 0.0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subcommands.add_parser("ingest", help="copy one verified media asset into a content-addressed local store")
    ingest_parser.add_argument("--input", required=True, type=Path)
    ingest_parser.add_argument("--store", required=True, type=Path)
    ingest_parser.add_argument("--registry", required=True, type=Path, help="strict LF JSONL asset registry")
    ingest_parser.add_argument("--id", dest="asset_id", required=True)
    ingest_parser.add_argument("--source", required=True)
    ingest_parser.add_argument("--license", required=True)
    ingest_parser.add_argument("--retrieved-at", required=True)
    ingest_parser.add_argument("--context-tokens", required=True, type=int,
                               help="exact model-specific visual/audio/video token cost for context packing")
    ingest_parser.add_argument("--metadata", default="{}", type=parse_metadata)
    ingest_parser.add_argument("--mime-type")
    ingest_parser.add_argument("--max-asset-bytes", type=int, default=DEFAULT_MAX_ASSET_BYTES)
    ingest_parser.add_argument("--lock-timeout-seconds", type=parse_positive_float, default=30.0)
    verify_parser = subcommands.add_parser("verify", help="rehash every registry asset")
    verify_parser.add_argument("--store", required=True, type=Path)
    verify_parser.add_argument("--registry", required=True, type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        if arguments.command == "ingest":
            require(arguments.max_asset_bytes > 0, "--max-asset-bytes must be positive")
            result = ingest(arguments)
        else:
            result = verify(arguments)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (LMMContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"lmm assets: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
