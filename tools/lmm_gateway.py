#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate through an actual optional LMM backend with Vegai provenance gates.

This gateway does not synthesize answers itself.  It validates a Unicode,
subword-aware model package; packs bounded long conversation context; injects
explicit grounded evidence; verifies content-addressed image/audio/video
assets; and then invokes one selected backend:

* ``openvino-genai`` for a local OpenVINO GenAI text model;
* ``openai-compatible`` for a configured HTTPS chat-completions service;
* ``worker`` for a local trusted multimodal worker implementing Vegai's JSON
  protocol.

No backend, missing capability, changed artifact, or malformed response is a
fail-closed error.  A successful response means only that the selected model
ran; factuality, safety, and broad LMM quality require separate evaluation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from lmm_contract import (
    GENERATION_REPORT_SCHEMA,
    MAX_JSON_BYTES,
    WORKER_RESPONSE_SCHEMA,
    AssetRegistry,
    EvidenceBundle,
    FileIdentity,
    GenerationRequest,
    GenerationResponse,
    LMMBackendError,
    LMMContractError,
    LMMOptionalDependencyError,
    ModelManifest,
    PackedContext,
    build_worker_request,
    canonical_json_bytes,
    file_identity,
    load_tokenizer,
    pack_context,
    render_prompt,
    require,
    require_unchanged,
    resolve_package_path,
    verify_model_artifacts,
    write_json_atomic,
)


MAX_WORKER_OUTPUT_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_INLINE_IMAGE_BYTES = 10 * 1024 * 1024
# Snapshot delivery avoids a pathname TOCTOU for trusted multimodal workers.
# Keep its default bounded so a hostile/accidental request cannot fill local SSD.
DEFAULT_MAX_WORKER_SNAPSHOT_BYTES = 4 * 1024 * 1024 * 1024


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0 or parsed > 2**63 - 1:
        raise argparse.ArgumentTypeError("must be a positive integer no larger than 2^63-1")
    return parsed


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number") from error
    if not parsed > 0.0 or parsed == float("inf") or parsed != parsed:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def load_command(path: Path) -> tuple[list[str], FileIdentity]:
    identity = file_identity(path, "worker command configuration", maximum_bytes=MAX_JSON_BYTES)
    try:
        value = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LMMContractError(f"cannot parse worker command configuration {path}: {error}") from error
    require(isinstance(value, list) and value, "worker command configuration must be a nonempty JSON array")
    require(all(isinstance(item, str) and item for item in value),
            "worker command configuration entries must be nonempty strings")
    require_unchanged(path, identity, "worker command configuration")
    return list(value), identity


def parse_worker_response(output: bytes) -> GenerationResponse:
    require(len(output) <= MAX_WORKER_OUTPUT_BYTES, f"worker response exceeds {MAX_WORKER_OUTPUT_BYTES} bytes")
    try:
        value = json.loads(output.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LMMBackendError(f"worker did not emit one UTF-8 JSON response: {error}") from error
    response = GenerationResponse.from_mapping(value)
    require(value.get("schema_version") == WORKER_RESPONSE_SCHEMA, "worker response schema mismatch")
    return response


def run_worker(
    command: Sequence[str],
    payload: Mapping[str, object],
    timeout_seconds: float,
) -> GenerationResponse:
    try:
        completed = subprocess.run(
            list(command),
            input=canonical_json_bytes(payload),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
            cwd=Path(__file__).resolve().parents[1],
        )
    except subprocess.TimeoutExpired as error:
        raise LMMBackendError(f"multimodal worker exceeded {timeout_seconds} seconds") from error
    except OSError as error:
        raise LMMBackendError(f"cannot launch multimodal worker: {error}") from error
    if completed.returncode != 0:
        stderr_digest = hashlib.sha256(completed.stderr).hexdigest()
        raise LMMBackendError(
            f"multimodal worker exited {completed.returncode}; stderr-sha256={stderr_digest}, stderr-bytes={len(completed.stderr)}"
        )
    return parse_worker_response(completed.stdout)


def _open_readonly_no_follow(path: Path, label: str):
    """Open a verified file without following a late symlink substitution."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        return os.fdopen(descriptor, "rb")
    except OSError as error:
        raise LMMBackendError(f"cannot open verified {label} {path}: {error}") from error


def read_verified_bytes(identity: FileIdentity, label: str, maximum_bytes: int) -> bytes:
    """Read the exact bytes that an endpoint will receive, then rehash them."""
    require(identity.bytes <= maximum_bytes, f"{label} exceeds {maximum_bytes} byte delivery limit")
    chunks: list[bytes] = []
    total = 0
    digest = hashlib.sha256()
    with _open_readonly_no_follow(Path(identity.path), label) as handle:
        while chunk := handle.read(1024 * 1024):
            total += len(chunk)
            require(total <= identity.bytes, f"{label} grew while it was being read")
            digest.update(chunk)
            chunks.append(chunk)
    require(total == identity.bytes and digest.hexdigest() == identity.sha256,
            f"{label} bytes changed before delivery")
    return b"".join(chunks)


def snapshot_verified_assets(
    verified_assets: Mapping[str, tuple[Any, FileIdentity]],
    destination_root: Path,
    maximum_total_bytes: int,
) -> dict[str, tuple[Any, FileIdentity]]:
    """Copy checked media into private immutable-for-request worker paths.

    A worker otherwise receives a source pathname and could decode a modified
    file after the gateway's initial hash. Streaming into a freshly-created
    private directory binds the bytes the worker can open to the manifest hash.
    """
    planned_bytes = sum(identity.bytes for _, identity in verified_assets.values())
    require(planned_bytes <= maximum_total_bytes,
            f"worker asset snapshot needs {planned_bytes} bytes, above --max-worker-snapshot-bytes={maximum_total_bytes}")
    snapshots: dict[str, tuple[Any, FileIdentity]] = {}
    for asset_id in sorted(verified_assets):
        record, source_identity = verified_assets[asset_id]
        suffix = Path(record.relative_path).suffix
        # IDs permit ':' for protocol compatibility, but that is not a legal
        # filename on every supported OS. A digest gives deterministic,
        # collision-resistant, cross-platform snapshot filenames.
        safe_name = hashlib.sha256(asset_id.encode("ascii")).hexdigest()
        target = destination_root / f"{safe_name}{suffix}"
        digest = hashlib.sha256()
        total = 0
        try:
            with _open_readonly_no_follow(Path(source_identity.path), f"asset {asset_id}") as source, target.open("xb") as copied:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    require(total <= source_identity.bytes, f"asset {asset_id} grew while being snapshotted")
                    digest.update(chunk)
                    copied.write(chunk)
                copied.flush()
                os.fsync(copied.fileno())
        except OSError as error:
            raise LMMBackendError(f"cannot snapshot asset {asset_id}: {error}") from error
        require(total == source_identity.bytes and digest.hexdigest() == source_identity.sha256,
                f"asset {asset_id} bytes changed before worker snapshot")
        snapshots[asset_id] = (
            record,
            FileIdentity(path=str(target.resolve()), sha256=source_identity.sha256, bytes=source_identity.bytes),
        )
    return snapshots


def extract_openvino_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    for name in ("text", "texts"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, (list, tuple)) and len(candidate) == 1 and isinstance(candidate[0], str):
            return candidate[0]
    raise LMMBackendError(f"OpenVINO GenAI returned an unsupported generation result type: {type(value).__name__}")


def run_openvino_genai(
    model_root: Path,
    device: str,
    prompt: str,
    request: GenerationRequest,
) -> GenerationResponse:
    try:
        import openvino_genai
    except ImportError as error:
        raise LMMOptionalDependencyError(
            "openvino-genai backend requires optional openvino-genai; install requirements-lmm-openvino.txt"
        ) from error
    try:
        pipeline = openvino_genai.LLMPipeline(str(model_root.resolve()), device)
        options: dict[str, object] = {"max_new_tokens": request.max_new_tokens}
        if request.temperature > 0.0:
            options.update({"do_sample": True, "temperature": request.temperature, "top_p": request.top_p})
        result = pipeline.generate(prompt, **options)
        text = extract_openvino_text(result)
    except LMMBackendError:
        raise
    except Exception as error:  # Native extension errors differ by OpenVINO release/model package.
        raise LMMBackendError(f"OpenVINO GenAI generation failed: {error}") from error
    return GenerationResponse(
        text=text,
        finish_reason="backend-complete",
        input_tokens=None,
        output_tokens=None,
        backend="openvino-genai",
        # LLMPipeline accepts a requested provider string but this adapter does
        # not independently prove where kernels executed (AUTO/HETERO can
        # dispatch differently), so never relabel the request as observation.
        device=None,
    )


def openai_messages(context: PackedContext, registry: AssetRegistry, verified_assets: Mapping[str, tuple[Any, FileIdentity]]) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for message in context.messages:
        require(message.role != "tool",
                "openai-compatible backend cannot serialize a tool role without a provider-specific tool_call_id; use a worker backend")
        parts: list[dict[str, object]] = []
        has_asset = False
        for part in message.content:
            if part.kind == "text":
                assert part.text is not None
                parts.append({"type": "text", "text": part.text})
                continue
            assert part.asset_id is not None
            record, identity = verified_assets[part.asset_id]
            require(record.modality == "image",
                    "openai-compatible backend only standardizes image asset blocks; use a worker backend for audio/video")
            image_bytes = read_verified_bytes(
                identity,
                f"image asset {record.asset_id}",
                DEFAULT_MAX_INLINE_IMAGE_BYTES,
            )
            encoded = base64.b64encode(image_bytes).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:{record.mime_type};base64,{encoded}", "detail": "auto"}})
            has_asset = True
        if not has_asset and len(parts) == 1 and parts[0]["type"] == "text":
            content: object = parts[0]["text"]
        else:
            content = parts
        messages.append({"role": message.role, "content": content})
    return messages


def response_text_from_openai_payload(payload: object) -> str:
    require(isinstance(payload, Mapping), "OpenAI-compatible endpoint response must be an object")
    choices = payload.get("choices")
    require(isinstance(choices, list) and choices, "OpenAI-compatible endpoint response has no choices")
    first = choices[0]
    require(isinstance(first, Mapping) and isinstance(first.get("message"), Mapping),
            "OpenAI-compatible endpoint choice has no message")
    content = first["message"].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "text" and isinstance(part.get("text"), str):
                fragments.append(part["text"])
        require(fragments, "OpenAI-compatible endpoint content list contains no text parts")
        return "".join(fragments)
    raise LMMBackendError("OpenAI-compatible endpoint returned no text content")


def run_openai_compatible(
    endpoint: str,
    api_key_env: str | None,
    allow_insecure_http: bool,
    model: ModelManifest,
    request: GenerationRequest,
    context: PackedContext,
    registry: AssetRegistry,
    verified_assets: Mapping[str, tuple[Any, FileIdentity]],
    timeout_seconds: float,
) -> GenerationResponse:
    parsed = urlparse(endpoint)
    require(parsed.scheme in {"https", "http"} and bool(parsed.netloc), "--endpoint must be an absolute HTTP(S) URL")
    require(parsed.scheme == "https" or allow_insecure_http,
            "refusing plaintext HTTP endpoint; pass --allow-insecure-http only for a trusted local test deployment")
    if parsed.scheme == "http":
        require(parsed.hostname in {"127.0.0.1", "::1", "localhost"},
                "plaintext HTTP is restricted to a loopback test endpoint; use HTTPS for remote services")
    token: str | None = None
    if api_key_env is not None:
        token = os.environ.get(api_key_env)
        require(bool(token), f"API key environment variable {api_key_env!r} is empty or unset")
    payload = {
        "model": model.model_id,
        "messages": openai_messages(context, registry, verified_assets),
        "max_tokens": request.max_new_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": False,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    http_request = Request(endpoint, data=canonical_json_bytes(payload), headers=headers, method="POST")
    try:
        ssl_context = ssl.create_default_context() if parsed.scheme == "https" else None
        with urlopen(http_request, timeout=timeout_seconds, context=ssl_context) as response:
            raw = response.read(MAX_WORKER_OUTPUT_BYTES + 1)
    except HTTPError as error:
        raw = error.read(MAX_WORKER_OUTPUT_BYTES + 1)
        digest = hashlib.sha256(raw).hexdigest()
        raise LMMBackendError(f"OpenAI-compatible endpoint returned HTTP {error.code}; response-sha256={digest}") from error
    except URLError as error:
        raise LMMBackendError(f"cannot reach OpenAI-compatible endpoint: {error}") from error
    except OSError as error:
        raise LMMBackendError(f"OpenAI-compatible endpoint I/O failed: {error}") from error
    require(len(raw) <= MAX_WORKER_OUTPUT_BYTES, f"endpoint response exceeds {MAX_WORKER_OUTPUT_BYTES} bytes")
    try:
        parsed_payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LMMBackendError(f"endpoint response is not UTF-8 JSON: {error}") from error
    text = response_text_from_openai_payload(parsed_payload)
    usage = parsed_payload.get("usage") if isinstance(parsed_payload, Mapping) else None
    input_tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
    output_tokens = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    require(input_tokens is None or type(input_tokens) is int and input_tokens >= 0,
            "endpoint usage.prompt_tokens is invalid")
    require(output_tokens is None or type(output_tokens) is int and output_tokens >= 0,
            "endpoint usage.completion_tokens is invalid")
    return GenerationResponse(
        text=text,
        finish_reason="endpoint-complete",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        backend="openai-compatible",
        device=None,
    )


def output_collision(output: Path | None, dependencies: Sequence[Path]) -> None:
    if output is None:
        return
    destination = output.resolve()
    for dependency in dependencies:
        resolved_dependency = dependency.resolve()
        same = destination == resolved_dependency
        if resolved_dependency.is_dir():
            try:
                destination.relative_to(resolved_dependency)
                same = True
            except ValueError:
                pass
        if destination.exists():
            try:
                same = same or os.path.samefile(destination, dependency)
            except OSError:
                pass
        require(not same, f"--output must not overwrite or be published inside immutable input {dependency}")


def run_generate(arguments: argparse.Namespace) -> int:
    gateway_path = Path(__file__).resolve()
    model_path = arguments.model.resolve()
    request_path = arguments.request.resolve()
    dependencies = [gateway_path, model_path, request_path]
    gateway_identity = file_identity(gateway_path, "gateway tool", maximum_bytes=MAX_JSON_BYTES)
    model_identity = file_identity(model_path, "model manifest", maximum_bytes=MAX_JSON_BYTES)
    request_identity = file_identity(request_path, "generation request", maximum_bytes=MAX_JSON_BYTES)
    evidence_identity: FileIdentity | None = None
    asset_registry_identity: FileIdentity | None = None
    worker_config_identity: FileIdentity | None = None
    if arguments.evidence is not None:
        dependencies.append(arguments.evidence.resolve())
        evidence_identity = file_identity(arguments.evidence, "evidence bundle", maximum_bytes=MAX_JSON_BYTES)
    if arguments.asset_registry is not None:
        dependencies.append(arguments.asset_registry.resolve())
        asset_registry_identity = file_identity(arguments.asset_registry, "asset registry", maximum_bytes=MAX_JSON_BYTES)
    if arguments.worker_command_json is not None:
        dependencies.append(arguments.worker_command_json.resolve())
        worker_config_identity = file_identity(arguments.worker_command_json, "worker command configuration", maximum_bytes=MAX_JSON_BYTES)
    output_collision(arguments.output, dependencies)

    model = ModelManifest.from_file(model_path)
    request = GenerationRequest.from_file(request_path)
    tokenizer_identity: FileIdentity | None = None
    tokenizer_path: Path | None = None
    if model.tokenizer.kind == "tokenizers-json":
        assert model.tokenizer.relative_path is not None
        tokenizer_path = resolve_package_path(model_path.parent, model.tokenizer.relative_path, "tokenizer.path")
        # Capture this before the optional native parser is called. The
        # post-load comparison below closes the TOCTOU window between parsing
        # a tokenizer and using it for context accounting.
        tokenizer_identity = file_identity(tokenizer_path, "tokenizer")
        dependencies.append(tokenizer_path)
    tokenizer = load_tokenizer(model, model_path)
    if tokenizer_identity is not None and tokenizer_path is not None:
        require_unchanged(tokenizer_path, tokenizer_identity, "tokenizer")
    registry = AssetRegistry.from_jsonl(arguments.asset_registry) if arguments.asset_registry is not None else AssetRegistry({})
    evidence = None if arguments.evidence is None else EvidenceBundle.from_file(arguments.evidence)
    context = pack_context(request, model, tokenizer, registry, evidence)

    artifact_identities: dict[str, FileIdentity] = {}
    if model.artifacts:
        require(arguments.model_root is not None, "--model-root is required to verify declared model artifacts")
        dependencies.append(arguments.model_root.resolve())
        artifact_identities = verify_model_artifacts(model, arguments.model_root)
    elif model.engine == "openvino-genai":
        raise LMMContractError("openvino-genai model manifest must bind artifacts")

    verified_assets: dict[str, tuple[Any, FileIdentity]] = {}
    if any(part.kind == "asset" for message in context.messages for part in message.content):
        require(arguments.asset_registry is not None and arguments.asset_root is not None,
                "--asset-registry and --asset-root are required for multimodal request parts")
        dependencies.append(arguments.asset_root.resolve())
        verified_assets = registry.verify_required(arguments.asset_root, context.messages)
        dependencies.extend(Path(identity.path) for _, identity in verified_assets.values())

    backend = arguments.backend or model.engine
    require(backend == model.engine,
            f"selected backend {backend!r} does not match immutable model manifest engine {model.engine!r}")
    worker_asset_delivery: str | None = None
    worker_snapshot_bytes: int | None = None
    started = time.monotonic()
    if backend == "worker":
        require(arguments.worker_command_json is not None, "worker backend requires --worker-command-json")
        command, reloaded_config_identity = load_command(arguments.worker_command_json)
        require(reloaded_config_identity == worker_config_identity, "worker command configuration changed before launch")
        worker_asset_delivery = arguments.worker_assets_mode if verified_assets else "none"
        if verified_assets and arguments.worker_assets_mode == "snapshot":
            # The directory is private and removed after the trusted worker
            # exits. Copy time is intentionally included in gateway latency.
            with tempfile.TemporaryDirectory(prefix="vegai-lmm-worker-assets-") as directory:
                worker_snapshot_bytes = sum(identity.bytes for _, identity in verified_assets.values())
                worker_assets = snapshot_verified_assets(
                    verified_assets,
                    Path(directory),
                    arguments.max_worker_snapshot_bytes,
                )
                payload = build_worker_request(request, model, context, registry, worker_assets)
                response = run_worker(command, payload, arguments.timeout_seconds)
        else:
            payload = build_worker_request(request, model, context, registry, verified_assets)
            response = run_worker(command, payload, arguments.timeout_seconds)
    elif backend == "openvino-genai":
        require(arguments.model_root is not None, "openvino-genai backend requires --model-root")
        require(not verified_assets, "openvino-genai text backend cannot consume multimodal assets; use a declared VLM worker")
        prompt = render_prompt(context, model, registry)
        response = run_openvino_genai(arguments.model_root, arguments.device, prompt, request)
    else:
        require(arguments.endpoint is not None, "openai-compatible backend requires --endpoint")
        response = run_openai_compatible(
            arguments.endpoint,
            arguments.api_key_env,
            arguments.allow_insecure_http,
            model,
            request,
            context,
            registry,
            verified_assets,
            arguments.timeout_seconds,
        )
    elapsed_seconds = time.monotonic() - started

    # Rehash every byte dependency after the model call. The model backend may
    # be slow; a report is valid only if the request/model/evidence/assets it
    # actually used remained immutable across that transaction.
    require_unchanged(gateway_path, gateway_identity, "gateway tool")
    require_unchanged(model_path, model_identity, "model manifest")
    require_unchanged(request_path, request_identity, "generation request")
    if tokenizer_identity is not None and tokenizer_path is not None:
        require_unchanged(tokenizer_path, tokenizer_identity, "tokenizer")
    if evidence_identity is not None:
        require_unchanged(arguments.evidence, evidence_identity, "evidence bundle")
    if asset_registry_identity is not None:
        require_unchanged(arguments.asset_registry, asset_registry_identity, "asset registry")
    if worker_config_identity is not None:
        require_unchanged(arguments.worker_command_json, worker_config_identity, "worker command configuration")
    for relative, identity in artifact_identities.items():
        require_unchanged(Path(identity.path), identity, f"model artifact {relative}")
    for asset_id, (_, identity) in verified_assets.items():
        require_unchanged(Path(identity.path), identity, f"asset {asset_id}")
    output_collision(
        arguments.output,
        [gateway_path, model_path, request_path]
        + ([] if tokenizer_path is None else [tokenizer_path])
        + ([] if arguments.evidence is None else [arguments.evidence])
        + ([] if arguments.asset_registry is None else [arguments.asset_registry])
        + ([] if arguments.worker_command_json is None else [arguments.worker_command_json])
        + [Path(identity.path) for identity in artifact_identities.values()]
        + [Path(identity.path) for _, identity in verified_assets.values()],
    )

    output_tokens = response.output_tokens if response.output_tokens is not None else tokenizer.count(response.text)
    report: dict[str, Any] = {
        "schema_version": GENERATION_REPORT_SCHEMA,
        "identities": {
            "gateway_tool": gateway_identity.to_mapping(),
            "model_manifest": model_identity.to_mapping(),
            "request": request_identity.to_mapping(),
            "tokenizer": None if tokenizer_identity is None else tokenizer_identity.to_mapping(),
            "evidence": None if evidence_identity is None else evidence_identity.to_mapping(),
            "asset_registry": None if asset_registry_identity is None else asset_registry_identity.to_mapping(),
            "worker_command_configuration": None if worker_config_identity is None else worker_config_identity.to_mapping(),
            "model_artifacts": {relative: identity.to_mapping() for relative, identity in artifact_identities.items()},
            "assets": {asset_id: identity.to_mapping() for asset_id, (_, identity) in verified_assets.items()},
        },
        "model": {
            "model_id": model.model_id,
            "engine": model.engine,
            "revision": model.revision,
            "capabilities": model.capabilities,
            "context_window_tokens": model.context_window_tokens,
            "tokenizer_kind": model.tokenizer.kind,
        },
        "context": context.to_mapping(include_content=False),
        "evidence_document_ids": [] if evidence is None else [item.document_id for item in evidence.items],
        "backend": {
            "name": response.backend,
            "requested_device": arguments.device if backend == "openvino-genai" else None,
            "reported_device": response.device,
            "worker_asset_delivery": worker_asset_delivery,
            "worker_snapshot_bytes": worker_snapshot_bytes,
            "elapsed_seconds": round(elapsed_seconds, 9),
        },
        "response": {
            "sha256": hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
            "bytes": len(response.text.encode("utf-8")),
            "finish_reason": response.finish_reason,
            "input_tokens": response.input_tokens if response.input_tokens is not None else context.input_tokens,
            "output_tokens": output_tokens,
            "text": response.text if arguments.include_response_in_report else None,
        },
        "quality_boundary": (
            "A successful backend call proves only contract-valid generation from the declared model package. "
            "It does not prove factuality, safety, copyright status, broad language quality, or LMM benchmark performance."
        ),
    }
    if arguments.output is not None:
        write_json_atomic(arguments.output, report)
    if arguments.print_json:
        print(json.dumps({"response": response.text, "report": report}, ensure_ascii=False, sort_keys=True))
    else:
        print(response.text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="vegai-lmm-model/v1 immutable manifest")
    parser.add_argument("--request", required=True, type=Path, help="vegai-lmm-request/v1 JSON object")
    parser.add_argument("--evidence", type=Path, help="optional vegai-lmm-evidence/v1 bundle from lmm_knowledge.py")
    parser.add_argument("--asset-registry", type=Path, help="strict JSONL content-addressed asset registry")
    parser.add_argument("--asset-root", type=Path, help="root containing asset registry relative paths")
    parser.add_argument("--model-root", type=Path, help="root containing immutable declared model artifacts")
    parser.add_argument("--backend", choices=("openvino-genai", "openai-compatible", "worker"),
                        help="must match model manifest engine; defaults to immutable manifest value")
    parser.add_argument("--worker-command-json", type=Path,
                        help="trusted local JSON string-array command implementing vegai-lmm-worker-request/v1")
    parser.add_argument("--worker-assets-mode", choices=("snapshot", "direct"), default="snapshot",
                        help="worker media delivery: private hash-checked snapshot (default) or direct immutable-store paths")
    parser.add_argument("--max-worker-snapshot-bytes", type=parse_positive_int,
                        default=DEFAULT_MAX_WORKER_SNAPSHOT_BYTES,
                        help="maximum total bytes copied for one snapshot worker request")
    parser.add_argument("--endpoint", help="HTTPS OpenAI-compatible /chat/completions endpoint")
    parser.add_argument("--api-key-env", help="environment variable containing endpoint bearer token; never stored in report")
    parser.add_argument("--allow-insecure-http", action="store_true", help="allow HTTP only for a trusted local test endpoint")
    parser.add_argument("--device", default="CPU", help="OpenVINO GenAI requested device, e.g. CPU/GPU/NPU")
    parser.add_argument("--timeout-seconds", type=parse_positive_float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output", type=Path, help="optional atomic provenance report path")
    parser.add_argument("--include-response-in-report", action="store_true", help="response text is sensitive; omit it by default")
    parser.add_argument("--print-json", action="store_true", help="emit response plus report JSON to stdout instead of text only")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        return run_generate(arguments)
    except (LMMContractError, LMMOptionalDependencyError, LMMBackendError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"lmm gateway: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
