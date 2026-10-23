#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create and verify immutable model manifests for Vegai's LMM gateway.

A real neural model package is external to this repository.  This tool walks an
operator-owned local package, hashes every regular artifact, and publishes a
``vegai-lmm-model/v1`` manifest consumed by ``lmm_gateway.py``.  It does not
download weights, infer a model's capabilities, or claim that a requested GPU/
NPU device executed the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from lmm_contract import (
    MAX_MODEL_ARTIFACTS,
    LMMContractError,
    ModelManifest,
    file_identity,
    load_tokenizer,
    relative_package_path,
    require,
    require_unchanged,
    resolve_package_path,
    verify_model_artifacts,
    write_json_atomic,
)


def collect_artifacts(root: Path, output: Path) -> list[dict[str, object]]:
    resolved_root = root.resolve()
    require(resolved_root.is_dir(), f"--model-root is not a directory: {resolved_root}")
    artifacts: list[dict[str, object]] = []
    output_resolved = output.resolve()
    for candidate in sorted(resolved_root.rglob("*")):
        # Check this before is_dir(): a symlink to a directory otherwise looks
        # like a harmless directory and would evade the package policy.
        require(not candidate.is_symlink(), f"model package symlink is not allowed: {candidate}")
        if candidate.is_dir():
            continue
        require(candidate.is_file(), f"model package contains a non-regular artifact: {candidate}")
        require(candidate.resolve() != output_resolved, "--output must be outside --model-root or pre-excluded from artifacts")
        relative = candidate.relative_to(resolved_root).as_posix()
        relative_package_path(relative, f"model artifact {relative}")
        identity = file_identity(candidate, f"model artifact {relative}")
        artifacts.append({"path": relative, "sha256": identity.sha256, "bytes": identity.bytes})
        require(len(artifacts) <= MAX_MODEL_ARTIFACTS, f"model package exceeds {MAX_MODEL_ARTIFACTS} artifacts")
    return artifacts


def create_manifest(arguments: argparse.Namespace) -> dict[str, object]:
    require(not arguments.model_root.is_symlink(), "--model-root must not be a symlink")
    root = arguments.model_root.resolve()
    output = arguments.output.resolve()
    require(output != root, "--output must not be --model-root")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise LMMContractError("--output must be outside --model-root so the manifest cannot become an unbound artifact")
    artifacts = collect_artifacts(root, output)
    tokenizer: dict[str, str]
    if arguments.tokenizer_kind == "utf8-byte":
        tokenizer = {"kind": "utf8-byte"}
    else:
        require(arguments.tokenizer_path is not None, "--tokenizer-path is required for tokenizers-json")
        relative = relative_package_path(arguments.tokenizer_path, "--tokenizer-path")
        target = resolve_package_path(root, relative, "--tokenizer-path")
        identity = file_identity(target, "tokenizer")
        tokenizer = {"kind": "tokenizers-json", "path": relative, "sha256": identity.sha256}
    payload: dict[str, Any] = {
        "schema_version": "vegai-lmm-model/v1",
        "model_id": arguments.model_id,
        "engine": arguments.engine,
        "context_window_tokens": arguments.context_window_tokens,
        "max_output_tokens": arguments.max_output_tokens,
        "message_overhead_tokens": arguments.message_overhead_tokens,
        "reply_primer_tokens": arguments.reply_primer_tokens,
        "capabilities": {
            "text": True,
            "image": arguments.image,
            "audio": arguments.audio,
            "video": arguments.video,
        },
        "tokenizer": tokenizer,
        "artifacts": artifacts,
        "prompt_format": arguments.prompt_format,
        "source": arguments.source,
        "license": arguments.license,
        "revision": arguments.revision,
    }
    # One canonical validation before publication means the emitted manifest is
    # exactly what gateway code will accept. Rehash the complete package a
    # second time immediately before publication: a model file changed while
    # the recursive walk was running must not be bound to a stale identity.
    manifest = ModelManifest.from_mapping(payload)
    verified = verify_model_artifacts(manifest, root)
    write_json_atomic(output, manifest.to_mapping())
    for relative, identity in verified.items():
        require_unchanged(Path(identity.path), identity, f"model artifact {relative}")
    return {
        "schema_version": "vegai-lmm-model-manifest-create-report/v1",
        "manifest": file_identity(output, "published model manifest").to_mapping(),
        "model_root": str(root),
        "artifacts": len(artifacts),
        "tokenizer": manifest.tokenizer.to_mapping(),
    }


def verify_manifest(arguments: argparse.Namespace) -> dict[str, object]:
    require(not arguments.model_root.is_symlink(), "--model-root must not be a symlink")
    manifest_path = arguments.manifest.resolve()
    manifest_identity = file_identity(manifest_path, "model manifest")
    manifest = ModelManifest.from_file(manifest_path)
    verified = verify_model_artifacts(manifest, arguments.model_root)
    if manifest.tokenizer.kind == "tokenizers-json":
        # Loading is intentionally part of verification: JSON bytes can hash
        # correctly while still failing the actual Rust tokenizer parser.
        load_tokenizer(manifest, manifest_path)
    for relative, identity in verified.items():
        require_unchanged(Path(identity.path), identity, f"model artifact {relative}")
    require_unchanged(manifest_path, manifest_identity, "model manifest")
    return {
        "schema_version": "vegai-lmm-model-manifest-verify-report/v1",
        "manifest": manifest_identity.to_mapping(),
        "model_root": str(arguments.model_root.resolve()),
        "verified_artifacts": {path: identity.to_mapping() for path, identity in verified.items()},
        "tokenizer_kind": manifest.tokenizer.kind,
    }


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="hash a local model package into vegai-lmm-model/v1")
    create.add_argument("--model-root", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--model-id", required=True)
    create.add_argument("--engine", required=True, choices=("openvino-genai", "openai-compatible", "worker"))
    create.add_argument("--context-window-tokens", required=True, type=parse_positive_int)
    create.add_argument("--max-output-tokens", required=True, type=parse_positive_int)
    create.add_argument("--message-overhead-tokens", type=parse_nonnegative_int, default=4)
    create.add_argument("--reply-primer-tokens", type=parse_nonnegative_int, default=2)
    create.add_argument("--tokenizer-kind", choices=("utf8-byte", "tokenizers-json"), default="tokenizers-json")
    create.add_argument("--tokenizer-path", help="relative tokenizer.json path under --model-root")
    create.add_argument("--prompt-format", choices=("chatml-v1", "json-v1"), default="chatml-v1")
    create.add_argument("--image", action="store_true")
    create.add_argument("--audio", action="store_true")
    create.add_argument("--video", action="store_true")
    create.add_argument("--source", required=True)
    create.add_argument("--license", required=True)
    create.add_argument("--revision", required=True)
    verify = commands.add_parser("verify", help="rehash every artifact declared in a model manifest")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--model-root", required=True, type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        result = create_manifest(arguments) if arguments.command == "create" else verify_manifest(arguments)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (LMMContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"lmm model manifest: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
