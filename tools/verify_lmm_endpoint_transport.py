#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Real loopback regressions for endpoint deadlines, privacy and input bounds."""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import lmm_gateway
from lmm_contract import (
    AssetRegistry,
    GenerationRequest,
    LMMContractError,
    ModelManifest,
    UTF8ByteTokenizer,
    file_identity,
    pack_context,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "examples" / "lmm" / "mini_multilingual_model.json"
REQUEST = ROOT / "examples" / "lmm" / "mini_multilingual_request.json"
SECRET = "vegai-endpoint-test-token-not-for-logs"
BODY_SECRET = "provider-error-body-must-not-appear"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, mode: str, redirect_url: str = "") -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.mode = mode
        self.redirect_url = redirect_url
        self.received = threading.Event()
        self.disconnected = threading.Event()
        self.last_body = b""
        self.last_auth = ""
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/v1/chat/completions"

    def close_fixture(self) -> None:
        self.shutdown()
        self.server_close()
        self.thread.join(timeout=5)


class Handler(BaseHTTPRequestHandler):
    server: FixtureServer

    def log_message(self, *_: object) -> None:
        pass

    def do_POST(self) -> None:
        mode = self.server.mode
        if mode == "stall-upload":
            self.server.received.set()
            time.sleep(3.5)
            return
        self.server.last_body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.last_auth = self.headers.get("Authorization", "")
        if mode == "slow-headers":
            self.server.received.set()
            time.sleep(3.5)
            self.reply(200, b'{"choices":[{"message":{"content":"late"}}]}')
            return
        if mode == "success":
            body = json.dumps({"choices": [{"message": {"content": "İstanbul ✓"}}],
                               "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
                              ensure_ascii=False).encode("utf-8")
            self.reply(200, body)
            return
        if mode == "http-error":
            self.reply(403, BODY_SECRET.encode())
            return
        if mode == "duplicate":
            self.reply(200, b'{"choices":[{"message":{"content":"secret", "content":"last"}}]}')
            return
        if mode == "bad-text":
            self.reply(200, b'{"choices":[{"message":{"content":"\\ud800"}}]}')
            return
        if mode == "bad-usage":
            self.reply(200, b'{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":true}}')
            return
        if mode == "oversized":
            self.send_response(200)
            self.send_header("Content-Length", str(lmm_gateway.MAX_WORKER_OUTPUT_BYTES + 1))
            self.end_headers()
            self.server.received.set()
            try:
                remaining = lmm_gateway.MAX_WORKER_OUTPUT_BYTES + 1
                while remaining:
                    chunk = b"x" * min(64 * 1024, remaining)
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.server.disconnected.set()
            return
        if mode in {"slow", "slow-error", "slow-redirect"}:
            self.send_response({"slow": 200, "slow-error": 503, "slow-redirect": 302}[mode])
            if mode == "slow-redirect":
                self.send_header("Location", self.server.redirect_url)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.server.received.set()
            for _ in range(35):
                try:
                    self.wfile.write(b"x")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    self.server.disconnected.set()
                    return
                time.sleep(0.1)
            return
        raise AssertionError(f"unknown fixture mode {mode}")

    def reply(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.server.received.set()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.server.disconnected.set()


def expect_failure(call, contains: str) -> str:
    try:
        call()
    except LMMContractError as error:
        message = str(error)
        require(contains in message, f"unexpected error {message!r}; expected {contains!r}")
        require(SECRET not in message and BODY_SECRET not in message, "sensitive endpoint data appeared in error")
        return message
    raise AssertionError(f"expected endpoint rejection containing {contains!r}")


def main() -> int:
    model = ModelManifest.from_file(MODEL)
    request = GenerationRequest.from_file(REQUEST)
    registry = AssetRegistry({})
    context = pack_context(request, model, UTF8ByteTokenizer(), registry, None)

    def generate(server: FixtureServer, timeout: float = 5.0):
        return lmm_gateway.run_openai_compatible(server.endpoint, "VEGAI_TEST_ENDPOINT_KEY", True, False,
                                                model, request, context, registry, {}, timeout)

    with patch.dict(os.environ, {"VEGAI_TEST_ENDPOINT_KEY": SECRET}):
        server = FixtureServer("success")
        try:
            response = generate(server)
            require(response.text == "İstanbul ✓" and response.input_tokens == 4 and response.output_tokens == 2,
                    "successful endpoint output/usage changed")
            require(server.last_auth == f"Bearer {SECRET}", "endpoint did not receive the declared bearer token")
            sent = json.loads(server.last_body)
            require(sent["stream"] is False and sent["model"] == model.model_id,
                    "endpoint request payload changed")
        finally:
            server.close_fixture()

        server = FixtureServer("http-error")
        try:
            message = expect_failure(lambda: generate(server), "HTTP 403")
            require(hashlib.sha256(BODY_SECRET.encode()).hexdigest() in message,
                    "bounded HTTP error digest is absent")
        finally:
            server.close_fixture()

        for mode, expected in (("duplicate", "strict UTF-8 JSON"), ("bad-text", "surrogate"),
                               ("bad-usage", "usage.prompt_tokens"), ("oversized", "response exceeds")):
            server = FixtureServer(mode)
            try:
                expect_failure(lambda: generate(server), expected)
            finally:
                server.close_fixture()

        # Each byte arrives more frequently than the old socket timeout. The
        # whole operation must nevertheless stop at the total wall deadline.
        for mode in ("slow", "slow-error", "slow-headers", "stall-upload"):
            server = FixtureServer(mode)
            try:
                started = time.monotonic()
                if mode == "stall-upload":
                    call = lambda: lmm_gateway._run_endpoint_transport(
                        server.endpoint, {"messages": [{"content": "x" * (8 * 1024 * 1024)}]}, SECRET, False, 2.0
                    )
                else:
                    call = lambda: generate(server, 1.2)
                expect_failure(call, "exceeded")
                elapsed = time.monotonic() - started
                require(elapsed < (3.5 if mode == "stall-upload" else 3.0),
                        f"{mode} exceeded the total deadline: {elapsed:.2f}s")
                require(server.received.is_set(), f"{mode} never reached the HTTP fixture")
            finally:
                server.close_fixture()

        target = FixtureServer("success")
        redirect = FixtureServer("slow-redirect", target.endpoint)
        try:
            started = time.monotonic()
            expect_failure(lambda: generate(redirect, 3), "redirect")
            require(time.monotonic() - started < 2.5, "redirect waited to consume its slow body")
            require(not target.received.is_set(), "bearer-authenticated request followed the redirect")
        finally:
            redirect.close_fixture()
            target.close_fixture()

        # The key is neither an argv value nor an inherited named environment
        # variable. Only the private, bounded stdin envelope contains it.
        calls: list[tuple[list[str], dict[str, object]]] = []

        def fake_process(command, **kwargs):
            calls.append((list(command), kwargs))
            return SimpleNamespace(returncode=0, stdout=b"OK\n{}")

        with patch.object(lmm_gateway, "run_bounded_process", fake_process):
            require(lmm_gateway._run_endpoint_transport("http://localhost/", {"messages": []}, SECRET,
                                                        False, 2, secret_environment_name="VEGAI_TEST_ENDPOINT_KEY") == b"{}",
                    "transport frame changed")
        command, options = calls[0]
        require(all(SECRET not in item for item in command), "token leaked into child argv")
        require("VEGAI_TEST_ENDPOINT_KEY" not in options["environment"], "key was inherited in child environment")
        require(json.loads(options["stdin_bytes"])["token"] == SECRET, "token was not privately delivered")
        require(options["isolate_process_group"] is False, "trusted leaf escaped the enclosing POSIX group")

        with patch.object(lmm_gateway, "MAX_ENDPOINT_INPUT_BYTES", 128), \
             patch.object(lmm_gateway, "run_bounded_process", side_effect=AssertionError("request must not spawn")):
            expect_failure(lambda: lmm_gateway._run_endpoint_transport("http://localhost/", {"text": "x" * 200},
                                                                      SECRET, False, 1), "request exceeds")

        for endpoint in ("http://127.0.0.1:", "http://127.0.0.1:0", "http://127.0.0.1:99999",
                         "http://example.com/", "http://127.0.0.1/has space"):
            with patch.object(lmm_gateway, "_run_endpoint_transport", side_effect=AssertionError("invalid URL used")):
                expect_failure(lambda: lmm_gateway.run_openai_compatible(endpoint, None, True, False, model,
                                                                         request, context, registry, {}, 1),
                               "endpoint" if "example.com" not in endpoint else "plaintext")

        # The parent's abrupt death must not leave a slow leaf transport alive
        # forever; its own watchdog closes the socket even without its caller.
        with tempfile.TemporaryDirectory(prefix="vegai-endpoint-watchdog-") as temporary:
            root = Path(temporary)
            model_payload = json.loads(MODEL.read_text(encoding="utf-8"))
            model_payload["engine"] = "openai-compatible"
            model_file = root / "model.json"
            model_file.write_text(json.dumps(model_payload), encoding="utf-8")
            report = root / "report.json"
            server = FixtureServer("slow")
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "tools" / "lmm_gateway.py"), "--model", str(model_file),
                 "--request", str(REQUEST), "--endpoint", server.endpoint, "--allow-insecure-http",
                 "--api-key-env", "VEGAI_TEST_ENDPOINT_KEY", "--timeout-seconds", "1.5", "--output", str(report)],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                require(server.received.wait(5), "gateway did not start the watchdog fixture request")
                process.kill()
                process.wait(timeout=5)
                require(server.disconnected.wait(3), "endpoint leaf survived its caller and own deadline")
                require(not report.exists(), "a killed gateway published a report")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                server.close_fixture()

    # Check repeated image expansion before touching any source bytes.
    part = SimpleNamespace(kind="asset", asset_id="image-1")
    message = SimpleNamespace(role="user", content=(part, part, part))
    image = SimpleNamespace(asset_id="image-1", modality="image", mime_type="image/png")
    large_identity = SimpleNamespace(bytes=lmm_gateway.DEFAULT_MAX_INLINE_IMAGE_BYTES)
    with patch.object(lmm_gateway, "read_verified_bytes", side_effect=AssertionError("asset read before preflight")):
        expect_failure(lambda: lmm_gateway.openai_messages(SimpleNamespace(messages=(message,)), registry,
                                                           {"image-1": (image, large_identity)}),
                       "inline image payload exceeds")

    with tempfile.TemporaryDirectory(prefix="vegai-inline-cache-") as temporary:
        path = Path(temporary) / "image.png"
        path.write_bytes(b"abc")
        identity = file_identity(path, "fixture image")
        message.content = (part, part)
        with patch.object(lmm_gateway, "read_verified_bytes", wraps=lmm_gateway.read_verified_bytes) as read:
            serialized = lmm_gateway.openai_messages(SimpleNamespace(messages=(message,)), registry,
                                                      {"image-1": (image, identity)})
            require(read.call_count == 1, "repeated identical asset was read more than once")
            urls = [item["image_url"]["url"] for item in serialized[0]["content"]]
            require(urls == ["data:image/png;base64," + base64.b64encode(b"abc").decode()] * 2,
                    "cached inline image changed the serialized request")

    print("LMM endpoint transport verification: PASS (total deadlines, redirects, privacy, caps, JSON, parent death)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, LMMContractError) as error:
        print(f"LMM endpoint transport verification: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
