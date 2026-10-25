#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Integration verification for the optional Unicode/multimodal LMM plane.

No model weights are fabricated in this test.  A synthetic local worker proves
the strict worker protocol while the test exercises real file provenance,
content-addressed image/audio/video assets, Unicode FTS retrieval, hash-chained
conversation state, deterministic long-context packing, capability fail-closed
behavior, and response/report privacy defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from lmm_process import BoundedProcessError, run_bounded_process
from lmm_contract import (
    EVIDENCE_SAFETY_MESSAGE_ID,
    AssetRegistry,
    EvidenceBundle,
    GenerationRequest,
    LMMContractError,
    ModelManifest,
    UTF8ByteTokenizer,
    file_identity,
    load_tokenizer,
    pack_context,
    resolve_package_path,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tools" / "lmm_assets.py"
KNOWLEDGE = ROOT / "tools" / "lmm_knowledge.py"
SESSION = ROOT / "tools" / "lmm_session.py"
GATEWAY = ROOT / "tools" / "lmm_gateway.py"
BENCHMARK = ROOT / "tools" / "benchmark_lmm_gateway.py"
QUALITY_EVALUATOR = ROOT / "tools" / "lmm_quality_evaluation.py"
MODEL_MANIFEST_TOOL = ROOT / "tools" / "lmm_model_manifest.py"
MODEL_FIXTURE = ROOT / "examples" / "lmm" / "mini_multilingual_model.json"
REQUEST_FIXTURE = ROOT / "examples" / "lmm" / "mini_multilingual_request.json"
KNOWLEDGE_FIXTURE = ROOT / "examples" / "lmm" / "mini_multilingual_knowledge.jsonl"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != expected:
        raise AssertionError(
            f"command returned {completed.returncode}, expected {expected}: {' '.join(command)}\n{completed.stdout}"
        )
    return completed


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def append_message(ledger: Path, message_path: Path, head: str) -> str:
    output = run(
        [
            sys.executable, str(SESSION), "append", "--ledger", str(ledger), "--message", str(message_path),
            "--expected-head", head,
        ]
    )
    return str(json.loads(output.stdout)["head"])


def main() -> int:
    try:
        model = ModelManifest.from_file(MODEL_FIXTURE)
        request = GenerationRequest.from_file(REQUEST_FIXTURE)
        require(model.engine == "worker" and model.capabilities == {"text": True, "image": True, "audio": True, "video": True},
                "checked-in model fixture contract changed")
        require(any("東京" in part.text for message in request.messages for part in message.content if part.kind == "text"),
                "checked-in request fixture lost Unicode coverage")
        byte_tokenizer = UTF8ByteTokenizer()
        unicode_probe = "İstanbul 東京 العربية"
        require(byte_tokenizer.decode(byte_tokenizer.encode(unicode_probe)) == unicode_probe,
                "UTF-8 bootstrap tokenizer is not lossless")
        require(byte_tokenizer.count(unicode_probe) > len(unicode_probe),
                "UTF-8 bootstrap tokenizer is not byte-exact for Unicode")

        with tempfile.TemporaryDirectory(prefix="vegai-lmm-runtime-") as temporary:
            workspace = Path(temporary)
            package_root = workspace / "package-root"
            package_root.mkdir()
            external_file = workspace / "external.bin"
            external_file.write_bytes(b"outside-package")
            package_link = package_root / "unsafe-link"
            try:
                package_link.symlink_to(external_file)
            except (NotImplementedError, OSError):
                # Some locked-down Windows environments cannot create a test
                # symlink. The production check remains active there.
                pass
            else:
                try:
                    resolve_package_path(package_root, "unsafe-link", "test package path")
                except LMMContractError:
                    pass
                else:
                    raise AssertionError("package path resolver accepted a symlinked artifact")
            try:
                from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
            except ImportError:
                # The stdlib UTF-8 bootstrap path remains intentionally usable
                # on a minimal host. CI installs requirements-lmm.txt to test
                # the fast Rust BPE path below.
                pass
            else:
                tokenizer_corpus = workspace / "tokenizer-corpus.txt"
                tokenizer_corpus.write_text("İstanbul 東京 العربية multilingual context\n", encoding="utf-8", newline="\n")
                native_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
                native_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
                native_tokenizer.decoder = decoders.ByteLevel()
                native_tokenizer.train([str(tokenizer_corpus)], trainers.BpeTrainer(vocab_size=128, special_tokens=["[UNK]"]))
                tokenizer_path = workspace / "tokenizer.json"
                native_tokenizer.save(str(tokenizer_path))
                subword_model_payload = json.loads(MODEL_FIXTURE.read_text(encoding="utf-8"))
                subword_model_payload["tokenizer"] = {
                    "kind": "tokenizers-json", "path": "tokenizer.json",
                    "sha256": file_identity(tokenizer_path, "test tokenizer").sha256,
                }
                subword_model_path = workspace / "subword-model.json"
                write_json(subword_model_path, subword_model_payload)
                subword_counter = load_tokenizer(ModelManifest.from_file(subword_model_path), subword_model_path)
                require(subword_counter.count(unicode_probe) > 0,
                        "tokenizers-json Unicode BPE did not return subword ids")

            model_root = workspace / "openvino-model-package"
            model_root.mkdir()
            (model_root / "openvino_model.xml").write_text("<synthetic-openvino-model />\n", encoding="ascii")
            (model_root / "openvino_model.bin").write_bytes(b"synthetic-weight-bytes")
            generated_manifest = workspace / "openvino-model.manifest.json"
            created_manifest = run(
                [
                    sys.executable, str(MODEL_MANIFEST_TOOL), "create", "--model-root", str(model_root),
                    "--output", str(generated_manifest), "--model-id", "synthetic-openvino", "--engine", "openvino-genai",
                    "--context-window-tokens", "512", "--max-output-tokens", "128", "--tokenizer-kind", "utf8-byte",
                    "--source", "synthetic verifier", "--license", "MIT", "--revision", "fixture-v1",
                ]
            )
            require(json.loads(created_manifest.stdout)["artifacts"] == 2,
                    "model manifest builder did not bind every model artifact")
            verified_manifest = run(
                [
                    sys.executable, str(MODEL_MANIFEST_TOOL), "verify", "--manifest", str(generated_manifest),
                    "--model-root", str(model_root),
                ]
            )
            require(len(json.loads(verified_manifest.stdout)["verified_artifacts"]) == 2,
                    "model manifest verifier did not rehash every artifact")
            (model_root / "openvino_model.bin").write_bytes(b"mutated-synthetic-weight-bytes")
            changed_package = run(
                [
                    sys.executable, str(MODEL_MANIFEST_TOOL), "verify", "--manifest", str(generated_manifest),
                    "--model-root", str(model_root),
                ],
                expected=2,
            )
            require("does not match immutable model manifest" in changed_package.stdout,
                    "model manifest verifier accepted a modified model package")
            store = workspace / "assets"
            registry = workspace / "assets.jsonl"
            # Magic is all the asset boundary checks need; no fake media decoder
            # or model encoder is introduced by the fixture.
            image = workspace / "image.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-image")
            audio = workspace / "audio.wav"
            audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEsynthetic-audio")
            video = workspace / "video.mp4"
            video.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00synthetic-video")
            for asset_id, path, tokens in (("image-one", image, 32), ("audio-one", audio, 48), ("video-one", video, 96)):
                run(
                    [
                        sys.executable, str(ASSETS), "ingest", "--input", str(path), "--store", str(store),
                        "--registry", str(registry), "--id", asset_id, "--source", "synthetic verifier", "--license", "MIT",
                        "--retrieved-at", "source-controlled test", "--context-tokens", str(tokens),
                    ]
                )
            verified_assets = run([sys.executable, str(ASSETS), "verify", "--store", str(store), "--registry", str(registry)])
            require(json.loads(verified_assets.stdout)["verified_assets"] == 3, "asset verifier did not inspect all modalities")
            loaded_registry = AssetRegistry.from_jsonl(registry)
            require({entry.modality for entry in loaded_registry.assets.values()} == {"image", "audio", "video"},
                    "asset registry lost a modality")

            index = workspace / "knowledge.sqlite"
            run([sys.executable, str(KNOWLEDGE), "build", "--input", str(KNOWLEDGE_FIXTURE), "--output", str(index)])
            evidence = workspace / "evidence.json"
            for query, expected_document in (("İstanbul", "turkish"), ("東京", "tokyo"), ("العربية", "arabic")):
                selected = evidence if query == "İstanbul" else workspace / f"{expected_document}.json"
                run([sys.executable, str(KNOWLEDGE), "query", "--index", str(index), "--query", query, "--output", str(selected)])
                payload = json.loads(selected.read_text(encoding="utf-8"))
                require(payload["items"] and payload["items"][0]["id"] == expected_document,
                        f"Unicode FTS did not retrieve {expected_document} for {query!r}: {payload}")
            run([sys.executable, str(KNOWLEDGE), "verify", "--index", str(index)])
            evidence_messages = EvidenceBundle.from_file(evidence).to_pinned_messages()
            require(
                evidence_messages
                and evidence_messages[0].message_id == EVIDENCE_SAFETY_MESSAGE_ID
                and evidence_messages[0].role == "system"
                and all(message.role == "user" and message.pinned for message in evidence_messages[1:]),
                "retrieved evidence was not isolated from operator system-message authority",
            )
            require(
                "<vegai-grounded-evidence" not in evidence_messages[0].content[0].text
                and EvidenceBundle.from_file(evidence).items[0].source not in evidence_messages[0].content[0].text,
                "untrusted evidence metadata leaked into the trusted safety instruction",
            )
            duplicate_knowledge = workspace / "duplicate-knowledge.jsonl"
            first_document = KNOWLEDGE_FIXTURE.read_bytes().splitlines()[0]
            duplicate_knowledge.write_bytes(first_document + b"\n" + first_document + b"\n")
            duplicate_result = run(
                [
                    sys.executable, str(KNOWLEDGE), "build", "--input", str(duplicate_knowledge),
                    "--output", str(workspace / "duplicate.sqlite"),
                ],
                expected=2,
            )
            require("repeats id" in duplicate_result.stdout,
                    "streaming knowledge builder accepted a duplicate document id")

            # Durable history must preserve an exact hash chain before becoming
            # a request for long-context packing.
            ledger = workspace / "session.jsonl"
            system_message = workspace / "system-message.json"
            user_message = workspace / "user-message.json"
            write_json(system_message, {"id": "system", "role": "system", "content": [{"type": "text", "text": "Kaynaklari ayir."}], "pinned": True})
            write_json(user_message, {"id": "user", "role": "user", "content": [{"type": "text", "text": "Merhaba 世界"}], "pinned": False})
            head = append_message(ledger, system_message, "GENESIS")
            head = append_message(ledger, user_message, head)
            compiled = workspace / "compiled-session-request.json"
            run(
                [
                    sys.executable, str(SESSION), "compile", "--ledger", str(ledger), "--expected-head", head,
                    "--request-id", "session-request", "--max-new-tokens", "32", "--temperature", "0", "--top-p", "1",
                    "--output", str(compiled),
                ]
            )
            require(GenerationRequest.from_file(compiled).messages[-1].role == "user", "session compile did not yield a valid request")

            multimodal_request = {
                "schema_version": "vegai-lmm-request/v1",
                "id": "multimodal-worker-demo",
                "messages": [
                    {
                        "id": "policy", "role": "system", "pinned": True,
                        "content": [{"type": "text", "text": "Sadece kaynakları ve ilan edilmiş yetenekleri kullan."}],
                    },
                    {
                        "id": "user", "role": "user", "pinned": False,
                        "content": [
                            {"type": "text", "text": "Merhaba İstanbul, 東京 ve العربية. Medyayı değerlendir."},
                            {"type": "asset", "asset_id": "image-one"},
                            {"type": "asset", "asset_id": "audio-one"},
                            {"type": "asset", "asset_id": "video-one"},
                        ],
                    },
                ],
                "max_new_tokens": 64,
                "temperature": 0,
                "top_p": 1,
            }
            multimodal_request_path = workspace / "multimodal-request.json"
            write_json(multimodal_request_path, multimodal_request)
            worker = workspace / "worker.py"
            worker.write_text(
                """import json, sys
payload = json.load(sys.stdin)
assert payload['schema_version'] == 'vegai-lmm-worker-request/v1'
ids = [message['id'] for message in payload['messages']]
assert any(message['id'] == 'vegai-evidence-safety-policy' and message['role'] == 'system' for message in payload['messages'])
assert any(message['id'].startswith('evidence:turkish') and message['role'] == 'user' for message in payload['messages'])
assert ids.index('policy') < ids.index('vegai-evidence-safety-policy') < ids.index('evidence:turkish')
assets = [part for message in payload['messages'] for part in message['content'] if part['type'] == 'asset']
assert {part['modality'] for part in assets} == {'image', 'audio', 'video'}
assert all(part['path'] and part['sha256'] for part in assets)
assert all('vegai-lmm-worker-assets-' in part['path'] for part in assets)
print(json.dumps({
  'schema_version': 'vegai-lmm-worker-response/v1',
  'text': 'Gerçek model worker sözleşmesi Unicode, kaynak ve üç medyayı aldı.',
  'finish_reason': 'stop',
  'input_tokens': 321,
  'output_tokens': 12,
  'backend': 'synthetic-multimodal-worker',
  'device': 'CPU+GPU+NPU-unverified-fixture'
}, ensure_ascii=False))
""",
                encoding="utf-8",
                newline="\n",
            )
            command = workspace / "worker-command.json"
            write_json(command, [sys.executable, str(worker)])
            report = workspace / "generation-report.json"
            generated = run(
                [
                    sys.executable, str(GATEWAY), "--model", str(MODEL_FIXTURE), "--request", str(multimodal_request_path),
                    "--asset-registry", str(registry), "--asset-root", str(store), "--evidence", str(evidence),
                    "--worker-command-json", str(command), "--output", str(report), "--include-response-in-report",
                ]
            )
            require("Unicode, kaynak ve üç medyayı aldı" in generated.stdout,
                    "gateway did not return worker-owned generative text")
            report_payload = json.loads(report.read_text(encoding="utf-8"))
            require(report_payload["response"]["text"].startswith("Gerçek model worker"), "response report did not bind backend output")
            require(set(report_payload["identities"]["assets"]) == {"image-one", "audio-one", "video-one"},
                    "generation report did not bind every media asset")
            require(report_payload["evidence_document_ids"] == ["turkish"], "evidence provenance was not carried into generation")
            require(report_payload["backend"]["worker_asset_delivery"] == "snapshot"
                    and report_payload["backend"]["worker_snapshot_bytes"] > 0,
                    "gateway did not report bounded hash-checked worker media snapshot delivery")

            # Tool/agent consumers need a checked data boundary, not merely a
            # best-effort instruction to emit JSON. The contract is supplied by
            # the operator and applies identically to every gateway backend.
            structured_contract = workspace / "structured-response-contract.json"
            write_json(
                structured_contract,
                {
                    "schema_version": "vegai-lmm-structured-response-contract/v1",
                    "id": "support-decision-v1",
                    "root": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "minimum_bytes": 1, "maximum_bytes": 256},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "needs_human": {"type": "boolean"},
                        },
                        "required": ["answer", "confidence", "needs_human"],
                        "additional_properties": False,
                    },
                },
            )
            structured_worker = workspace / "structured-worker.py"
            structured_worker.write_text(
                """import json, sys
json.load(sys.stdin)
print(json.dumps({
  'schema_version': 'vegai-lmm-worker-response/v1',
  'text': json.dumps({'answer': 'Doğrulanmış yapılandırılmış yanıt.', 'confidence': 0.75, 'needs_human': False}, ensure_ascii=False),
  'finish_reason': 'stop', 'input_tokens': 4, 'output_tokens': 8,
  'backend': 'synthetic-structured-worker', 'device': None
}, ensure_ascii=False))
""",
                encoding="utf-8", newline="\n",
            )
            structured_command = workspace / "structured-worker-command.json"
            write_json(structured_command, [sys.executable, str(structured_worker)])
            structured_report = workspace / "structured-report.json"
            run(
                [
                    sys.executable, str(GATEWAY), "--model", str(MODEL_FIXTURE), "--request", str(compiled),
                    "--worker-command-json", str(structured_command), "--response-contract", str(structured_contract),
                    "--output", str(structured_report),
                ]
            )
            structured_payload = json.loads(structured_report.read_text(encoding="utf-8"))
            require(
                structured_payload["structured_output"]["validated"] is True
                and structured_payload["structured_output"]["contract_id"] == "support-decision-v1"
                and structured_payload["identities"]["structured_response_contract"]["sha256"]
                == file_identity(structured_contract, "structured contract").sha256,
                "gateway did not bind and validate the structured response contract",
            )
            malformed_structured_worker = workspace / "malformed-structured-worker.py"
            malformed_structured_worker.write_text(
                """import json, sys
json.load(sys.stdin)
print(json.dumps({
  'schema_version': 'vegai-lmm-worker-response/v1',
  'text': '{\\"answer\\":\\"wrong extra field\\",\\"confidence\\":0.5,\\"needs_human\\":false,\\"extra\\":true}',
  'finish_reason': 'stop', 'input_tokens': 4, 'output_tokens': 8,
  'backend': 'synthetic-malformed-structured-worker', 'device': None
}))
""",
                encoding="utf-8", newline="\n",
            )
            malformed_structured_command = workspace / "malformed-structured-worker-command.json"
            write_json(malformed_structured_command, [sys.executable, str(malformed_structured_worker)])
            malformed_structured_report = workspace / "must-not-exist-structured-report.json"
            malformed_structured = run(
                [
                    sys.executable, str(GATEWAY), "--model", str(MODEL_FIXTURE), "--request", str(compiled),
                    "--worker-command-json", str(malformed_structured_command), "--response-contract", str(structured_contract),
                    "--output", str(malformed_structured_report),
                ],
                expected=2,
            )
            require("undeclared property" in malformed_structured.stdout and not malformed_structured_report.exists(),
                    "gateway accepted malformed structured model output or published a report")
            nested_structured_worker = workspace / "nested-structured-worker.py"
            nested_structured_worker.write_text(
                """import json, sys
json.load(sys.stdin)
nested = 0
for _ in range(32):
    nested = {'nested': nested}
text = json.dumps({'answer': 'depth test', 'confidence': 0.5, 'needs_human': False, 'extra': nested})
print(json.dumps({
  'schema_version': 'vegai-lmm-worker-response/v1', 'text': text,
  'finish_reason': 'stop', 'input_tokens': 4, 'output_tokens': 8,
  'backend': 'synthetic-nested-structured-worker', 'device': None
}))
""",
                encoding="utf-8", newline="\n",
            )
            nested_structured_command = workspace / "nested-structured-worker-command.json"
            write_json(nested_structured_command, [sys.executable, str(nested_structured_worker)])
            nested_structured_report = workspace / "must-not-exist-nested-structured-report.json"
            nested_structured = run(
                [
                    sys.executable, str(GATEWAY), "--model", str(MODEL_FIXTURE), "--request", str(compiled),
                    "--worker-command-json", str(nested_structured_command), "--response-contract", str(structured_contract),
                    "--output", str(nested_structured_report),
                ],
                expected=2,
            )
            require("maximum nesting depth" in nested_structured.stdout and not nested_structured_report.exists(),
                    "gateway accepted an over-nested structured response or published a report")

            overflow_worker = workspace / "overflow-worker.py"
            overflow_worker.write_text(
                "import sys\njson_bytes = sys.stdin.buffer.read()\nsys.stdout.buffer.write(b'x' * (9 * 1024 * 1024))\nsys.stdout.buffer.flush()\n",
                encoding="utf-8", newline="\n",
            )
            overflow_command = workspace / "overflow-worker-command.json"
            write_json(overflow_command, [sys.executable, str(overflow_worker)])
            overflow_report = workspace / "must-not-exist-overflow-report.json"
            overflow = run(
                [
                    sys.executable, str(GATEWAY), "--model", str(MODEL_FIXTURE), "--request", str(compiled),
                    "--worker-command-json", str(overflow_command), "--output", str(overflow_report),
                ],
                expected=2,
            )
            require("stdout exceeds" in overflow.stdout and not overflow_report.exists(),
                    "gateway did not bound oversized multimodal-worker stdout fail-closed")
            try:
                run_bounded_process(
                    [sys.executable, str(overflow_worker)],
                    cwd=ROOT, timeout_seconds=5.0, stdout_limit_bytes=1024,
                    stderr_limit_bytes=1024, label="outer LMM control-plane fixture", stdin_bytes=b"{}",
                )
            except BoundedProcessError as error:
                require("stdout exceeds" in str(error), "shared LMM child runner returned the wrong overflow category")
            else:
                raise AssertionError("shared LMM child runner accepted oversized stdout")
            stderr_overflow_worker = workspace / "stderr-overflow-worker.py"
            stderr_overflow_worker.write_text(
                "import sys\nsys.stderr.buffer.write(b'e' * 2048)\nsys.stderr.buffer.flush()\n",
                encoding="utf-8", newline="\n",
            )
            try:
                run_bounded_process(
                    [sys.executable, str(stderr_overflow_worker)],
                    cwd=ROOT, timeout_seconds=5.0, stdout_limit_bytes=1024,
                    stderr_limit_bytes=1024, label="outer LMM stderr fixture",
                )
            except BoundedProcessError as error:
                require("stderr exceeds" in str(error), "shared LMM child runner returned the wrong stderr-overflow category")
            else:
                raise AssertionError("shared LMM child runner accepted oversized stderr")
            sleeping_child = workspace / "sleeping-lmm-child.py"
            sleeping_child.write_text("import time\ntime.sleep(2)\n", encoding="utf-8", newline="\n")
            try:
                run_bounded_process(
                    [sys.executable, str(sleeping_child)],
                    cwd=ROOT, timeout_seconds=0.1, stdout_limit_bytes=1024,
                    stderr_limit_bytes=1024, label="outer LMM timeout fixture",
                )
            except BoundedProcessError as error:
                require("exceeded" in str(error), "shared LMM child runner returned the wrong timeout category")
            else:
                raise AssertionError("shared LMM child runner accepted a timed-out child")

            # A held-out quality gate runs a real configured gateway command;
            # it accepts only backend-emitted evidence/abstention markers and
            # retains output hashes rather than raw held-out text in its report.
            quality_worker = workspace / "quality-worker.py"
            quality_worker.write_text(
                """import json, sys
payload = json.load(sys.stdin)
request_id = payload['request']['id']
assert any(message['id'] == 'vegai-evidence-safety-policy' and message['role'] == 'system' for message in payload['messages']) or not any(message['id'].startswith('evidence:') for message in payload['messages'])
evidence = [message for message in payload['messages'] if message['id'].startswith('evidence:')]
assert all(message['role'] == 'user' for message in evidence)
if request_id == 'eval-bad-citation':
    assert any(message['id'] == 'evidence:turkish' for message in evidence)
    text = 'Yanlis kaynak. [[vegai-source:unknown]]'
elif request_id == 'eval-malformed-citation':
    assert any(message['id'] == 'evidence:turkish' for message in evidence)
    text = 'Bozuk kaynak marker. [[vegai-source:turkish'
elif request_id == 'eval-malformed-abstention':
    assert any(message['id'] == 'evidence:turkish' for message in evidence)
    text = 'Bozuk abstention marker. [[vegai-abstain'
elif request_id.startswith('eval-grounded'):
    assert any(message['id'] == 'evidence:turkish' for message in evidence)
    text = 'Kaynakli model cevabi. [[vegai-source:turkish]]'
elif request_id == 'eval-abstain':
    assert not evidence
    text = 'Yeterli kanit yok. [[vegai-abstain]]'
else:
    raise AssertionError(request_id)
print(json.dumps({
  'schema_version': 'vegai-lmm-worker-response/v1',
  'text': text,
  'finish_reason': 'stop',
  'input_tokens': 7,
  'output_tokens': 5,
  'backend': 'synthetic-quality-worker',
  'device': None
}, ensure_ascii=False))
""",
                encoding="utf-8",
                newline="\n",
            )
            quality_worker_command = workspace / "quality-worker-command.json"
            write_json(quality_worker_command, [sys.executable, str(quality_worker)])
            grounded_evidence = json.loads(evidence.read_text(encoding="utf-8"))
            abstain_evidence = dict(grounded_evidence)
            abstain_evidence["items"] = []
            grounded_request = {
                "schema_version": "vegai-lmm-request/v1", "id": "eval-grounded", "max_new_tokens": 32,
                "temperature": 0, "top_p": 1,
                "messages": [
                    {"id": "quality-policy", "role": "system", "pinned": True,
                     "content": [{"type": "text", "text": "Kanita dayali yanitlarda kaynak markerini kendin yaz."}]},
                    {"id": "quality-user", "role": "user", "pinned": False,
                     "content": [{"type": "text", "text": "İstanbul için kaynaklı yanıt ver."}]},
                ],
            }
            abstain_request = dict(grounded_request)
            abstain_request["id"] = "eval-abstain"
            abstain_request["messages"] = [
                grounded_request["messages"][0],
                {"id": "quality-user-abstain", "role": "user", "pinned": False,
                 "content": [{"type": "text", "text": "Kanıt yoksa açıkça abstain et."}]},
            ]
            quality_cases = workspace / "quality-cases.jsonl"
            quality_case_rows = [
                {
                    "schema_version": "vegai-lmm-quality-case/v1", "id": "grounded-citation",
                    "request": grounded_request, "evidence": grounded_evidence,
                    "required_citation_ids": ["turkish"], "allowed_citation_ids": ["turkish"],
                    "require_abstention": False,
                },
                {
                    "schema_version": "vegai-lmm-quality-case/v1", "id": "unsupported-abstention",
                    "request": abstain_request, "evidence": abstain_evidence,
                    "required_citation_ids": [], "allowed_citation_ids": [], "require_abstention": True,
                },
            ]
            quality_cases.write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in quality_case_rows),
                encoding="utf-8", newline="\n",
            )
            quality_command = workspace / "quality-gateway-command.json"
            write_json(
                quality_command,
                [
                    sys.executable, str(GATEWAY), "--model", str(MODEL_FIXTURE), "--request", "{request}",
                    "--evidence", "{evidence}", "--worker-command-json", str(quality_worker_command),
                    "--grounding-mode", "evidence-or-abstain", "--print-json",
                ],
            )
            quality_hmac_key = workspace / "quality-case-hmac.key"
            quality_hmac_key.write_bytes(b"synthetic-quality-case-secret")
            quality_report = workspace / "quality-report.json"
            quality_result = run(
                [
                    sys.executable, str(QUALITY_EVALUATOR), "--cases", str(quality_cases),
                    "--gateway-command-json", str(quality_command), "--case-id-hmac-key", str(quality_hmac_key),
                    "--report", str(quality_report), "--fail-on-gate",
                ]
            )
            require(json.loads(quality_result.stdout)["release_gate_passed"],
                    "held-out LMM quality evaluator did not pass its valid gateway cases")
            quality_payload = json.loads(quality_report.read_text(encoding="utf-8"))
            require(quality_payload["quality_gate"]["passed"] and len(quality_payload["cases"]) == 2,
                    "quality report did not record the configured held-out cases")
            require(quality_payload["configuration"]["case_id_privacy"] == "hmac-sha256"
                    and quality_payload["dependencies"]["case_id_hmac_key"]["path"] == "<redacted>",
                    "quality report did not apply private optional held-out case HMAC handling")
            require(all(len(case["case"]) == 64 and "text" not in case["response"] for case in quality_payload["cases"]),
                    "quality report exposed held-out response text or case identifiers")

            malformed_quality_command = workspace / "malformed-quality-gateway-command.json"
            write_json(
                malformed_quality_command,
                [
                    sys.executable, str(GATEWAY), "--model", "{request}", "--request", str(REQUEST_FIXTURE),
                    "--evidence", "{evidence}", "--print-json",
                ],
            )
            malformed_quality_command_run = run(
                [
                    sys.executable, str(QUALITY_EVALUATOR), "--cases", str(quality_cases),
                    "--gateway-command-json", str(malformed_quality_command),
                    "--report", str(workspace / "must-not-exist-quality-report.json"),
                ],
                expected=2,
            )
            require("direct value of --request" in malformed_quality_command_run.stdout,
                    "quality evaluator accepted an unbound request placeholder command")

            bad_citation_request = dict(grounded_request)
            bad_citation_request["id"] = "eval-bad-citation"
            bad_citation_path = workspace / "bad-citation-request.json"
            write_json(bad_citation_path, bad_citation_request)
            bad_citation = run(
                [
                    sys.executable, str(GATEWAY), "--model", str(MODEL_FIXTURE), "--request", str(bad_citation_path),
                    "--evidence", str(evidence), "--worker-command-json", str(quality_worker_command),
                    "--grounding-mode", "require-citations",
                ],
                expected=2,
            )
            require("not supplied" in bad_citation.stdout,
                    "gateway accepted a model citation absent from the supplied evidence bundle")

            malformed_citation_request = dict(grounded_request)
            malformed_citation_request["id"] = "eval-malformed-citation"
            malformed_citation_path = workspace / "malformed-citation-request.json"
            write_json(malformed_citation_path, malformed_citation_request)
            malformed_citation = run(
                [
                    sys.executable, str(GATEWAY), "--model", str(MODEL_FIXTURE), "--request", str(malformed_citation_path),
                    "--evidence", str(evidence), "--worker-command-json", str(quality_worker_command),
                    "--grounding-mode", "require-citations",
                ],
                expected=2,
            )
            require("malformed" in malformed_citation.stdout,
                    "gateway accepted a malformed model citation marker")

            malformed_abstention_request = dict(grounded_request)
            malformed_abstention_request["id"] = "eval-malformed-abstention"
            malformed_abstention_path = workspace / "malformed-abstention-request.json"
            write_json(malformed_abstention_path, malformed_abstention_request)
            malformed_abstention = run(
                [
                    sys.executable, str(GATEWAY), "--model", str(MODEL_FIXTURE), "--request", str(malformed_abstention_path),
                    "--evidence", str(evidence), "--worker-command-json", str(quality_worker_command),
                    "--grounding-mode", "require-citations",
                ],
                expected=2,
            )
            require("malformed" in malformed_abstention.stdout,
                    "gateway accepted a malformed model abstention marker")

            negative_quality_case = workspace / "negative-quality-cases.jsonl"
            negative_row = dict(quality_case_rows[0])
            negative_row["id"] = "incorrect-abstention-expectation"
            negative_row["request"] = dict(grounded_request)
            negative_row["request"]["id"] = "eval-grounded-negative"
            negative_row["require_abstention"] = True
            negative_row["required_citation_ids"] = []
            negative_row["allowed_citation_ids"] = []
            negative_quality_case.write_text(json.dumps(negative_row, ensure_ascii=False, sort_keys=True) + "\n",
                                             encoding="utf-8", newline="\n")
            negative_quality_report = workspace / "negative-quality-report.json"
            negative_quality = run(
                [
                    sys.executable, str(QUALITY_EVALUATOR), "--cases", str(negative_quality_case),
                    "--gateway-command-json", str(quality_command), "--report", str(negative_quality_report), "--fail-on-gate",
                ],
                expected=3,
            )
            require(not json.loads(negative_quality.stdout)["release_gate_passed"]
                    and negative_quality_report.is_file()
                    and not json.loads(negative_quality_report.read_text(encoding="utf-8"))["quality_gate"]["passed"],
                    "quality release gate did not publish a failing report for an invalid abstention expectation")

            benchmark_child_report = workspace / "benchmark-child-report.json"
            benchmark_command = workspace / "benchmark-command.json"
            write_json(
                benchmark_command,
                [
                    sys.executable, str(GATEWAY), "--model", str(MODEL_FIXTURE), "--request", str(multimodal_request_path),
                    "--asset-registry", str(registry), "--asset-root", str(store), "--evidence", str(evidence),
                    "--worker-command-json", str(command), "--output", str(benchmark_child_report), "--print-json",
                ],
            )
            benchmark_report = workspace / "benchmark.json"
            run(
                [
                    sys.executable, str(BENCHMARK), "--command-json", str(benchmark_command), "--warmups", "0",
                    "--repetitions", "2", "--output", str(benchmark_report),
                ]
            )
            benchmark_payload = json.loads(benchmark_report.read_text(encoding="utf-8"))
            require(benchmark_payload["repetitions"] == 2 and len(benchmark_payload["samples"]) == 2,
                    "benchmark did not record only configured actual gateway samples")
            require(benchmark_payload["methodology"]["timed_unit"].startswith("one complete"),
                    "benchmark methodology does not disclose its timing boundary")

            # An OpenAI-compatible endpoint is a real integration option, not
            # just a manifest claim. A one-request local test server validates
            # the standards-based text/image payload without an API credential.
            endpoint_server = workspace / "endpoint-server.py"
            endpoint_port = workspace / "endpoint-port.txt"
            endpoint_server.write_text(
                """import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
port_file = Path(sys.argv[1])
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        payload = json.loads(raw.decode('utf-8'))
        assert payload['model'] == 'endpoint-fixture'
        parts = payload['messages'][-1]['content']
        assert any(part.get('type') == 'image_url' and part['image_url']['url'].startswith('data:image/png;base64,') for part in parts)
        response = json.dumps({'choices': [{'message': {'content': 'Endpoint model yaniti.'}}], 'usage': {'prompt_tokens': 9, 'completion_tokens': 4}}, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    def log_message(self, *args):
        pass
server = HTTPServer(('127.0.0.1', 0), Handler)
port_file.write_text(str(server.server_port), encoding='ascii')
server.handle_request()
""",
                encoding="utf-8",
                newline="\n",
            )
            endpoint_process = subprocess.Popen([sys.executable, str(endpoint_server), str(endpoint_port)], cwd=ROOT)
            try:
                deadline = time.monotonic() + 10.0
                while not endpoint_port.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                require(endpoint_port.exists(), "OpenAI-compatible fixture server did not publish a port")
                endpoint_model_payload = json.loads(MODEL_FIXTURE.read_text(encoding="utf-8"))
                endpoint_model_payload["model_id"] = "endpoint-fixture"
                endpoint_model_payload["engine"] = "openai-compatible"
                endpoint_model_payload["capabilities"] = {"text": True, "image": True, "audio": False, "video": False}
                endpoint_model = workspace / "endpoint-model.json"
                write_json(endpoint_model, endpoint_model_payload)
                endpoint_request = dict(multimodal_request)
                endpoint_request["id"] = "endpoint-image"
                endpoint_request["messages"] = [
                    endpoint_request["messages"][0],
                    {"id": "endpoint-user", "role": "user", "pinned": False,
                     "content": [{"type": "text", "text": "Resmi değerlendir."}, {"type": "asset", "asset_id": "image-one"}]},
                ]
                endpoint_request_path = workspace / "endpoint-request.json"
                write_json(endpoint_request_path, endpoint_request)
                endpoint_report = workspace / "endpoint-report.json"
                endpoint_generation = run(
                    [
                        sys.executable, str(GATEWAY), "--model", str(endpoint_model), "--request", str(endpoint_request_path),
                        "--asset-registry", str(registry), "--asset-root", str(store), "--endpoint",
                        f"http://127.0.0.1:{endpoint_port.read_text(encoding='ascii')}/v1/chat/completions",
                        "--allow-insecure-http", "--output", str(endpoint_report),
                    ]
                )
                require("Endpoint model yaniti" in endpoint_generation.stdout,
                        "OpenAI-compatible endpoint output was not returned by gateway")
                endpoint_payload = json.loads(endpoint_report.read_text(encoding="utf-8"))
                require(endpoint_payload["response"]["output_tokens"] == 4
                        and endpoint_payload["backend"]["name"] == "openai-compatible",
                        "endpoint response usage/backend provenance was not preserved")
                endpoint_process.wait(timeout=10)
                require(endpoint_process.returncode == 0, "OpenAI-compatible fixture server failed")

                # A live local target is stronger than a closed-port redirect
                # fixture: it proves no follow-up request (and thus no bearer
                # credential) reached the redirect target.
                redirect_target_server = workspace / "redirect-target-server.py"
                redirect_target_port = workspace / "redirect-target-port.txt"
                redirect_target_hit = workspace / "redirect-target-hit.txt"
                redirect_target_server.write_text(
                    """from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        Path(sys.argv[2]).write_text('redirect target contacted', encoding='ascii')
        self.send_response(200)
        self.end_headers()
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', '0')))
        self.do_GET()
    def log_message(self, *args):
        pass
server = HTTPServer(('127.0.0.1', 0), Handler)
Path(sys.argv[1]).write_text(str(server.server_port), encoding='ascii')
server.serve_forever()
""",
                    encoding="utf-8", newline="\n",
                )
                redirect_target_process = subprocess.Popen(
                    [sys.executable, str(redirect_target_server), str(redirect_target_port), str(redirect_target_hit)], cwd=ROOT
                )
                try:
                    deadline = time.monotonic() + 10.0
                    while not redirect_target_port.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    require(redirect_target_port.exists(), "redirect target fixture server did not publish a port")
                    redirect_server = workspace / "redirect-endpoint-server.py"
                    redirect_port = workspace / "redirect-endpoint-port.txt"
                    redirect_server.write_text(
                        """from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(302)
        self.send_header('Location', sys.argv[2])
        self.end_headers()
    def log_message(self, *args):
        pass
server = HTTPServer(('127.0.0.1', 0), Handler)
Path(sys.argv[1]).write_text(str(server.server_port), encoding='ascii')
server.handle_request()
""",
                        encoding="utf-8", newline="\n",
                    )
                    redirect_url = (
                        f"http://127.0.0.1:{redirect_target_port.read_text(encoding='ascii')}"
                        "/credential-bearing-target"
                    )
                    redirect_process = subprocess.Popen(
                        [sys.executable, str(redirect_server), str(redirect_port), redirect_url], cwd=ROOT
                    )
                    try:
                        deadline = time.monotonic() + 10.0
                        while not redirect_port.exists() and time.monotonic() < deadline:
                            time.sleep(0.02)
                        require(redirect_port.exists(), "redirect fixture server did not publish a port")
                        redirect_report = workspace / "must-not-exist-redirect-report.json"
                        run(
                            [
                                sys.executable, str(GATEWAY), "--model", str(endpoint_model), "--request", str(endpoint_request_path),
                                "--asset-registry", str(registry), "--asset-root", str(store), "--endpoint",
                                f"http://127.0.0.1:{redirect_port.read_text(encoding='ascii')}/v1/chat/completions",
                                "--allow-insecure-http", "--output", str(redirect_report),
                            ],
                            expected=2,
                        )
                        require(not redirect_report.exists() and not redirect_target_hit.exists(),
                                "gateway followed an endpoint redirect or published an untrusted response report")
                        redirect_process.wait(timeout=10)
                        require(redirect_process.returncode == 0, "redirect fixture server failed")
                    finally:
                        if redirect_process.poll() is None:
                            redirect_process.terminate()
                            redirect_process.wait(timeout=10)
                finally:
                    if redirect_target_process.poll() is None:
                        redirect_target_process.terminate()
                        redirect_target_process.wait(timeout=10)
            finally:
                if endpoint_process.poll() is None:
                    endpoint_process.terminate()
                    endpoint_process.wait(timeout=10)

            # Context eviction keeps a pinned policy/latest user and evicts old
            # user+assistant turns together under a small, deterministic budget.
            short_model_payload = json.loads(MODEL_FIXTURE.read_text(encoding="utf-8"))
            short_model_payload["context_window_tokens"] = 220
            short_model_payload["max_output_tokens"] = 80
            short_model = ModelManifest.from_mapping(short_model_payload)
            long_request_payload = {
                "schema_version": "vegai-lmm-request/v1", "id": "context-budget", "max_new_tokens": 64,
                "temperature": 0, "top_p": 1,
                "messages": [
                    {"id": "system-policy", "role": "system", "content": [{"type": "text", "text": "policy"}], "pinned": True},
                    {"id": "old-user", "role": "user", "content": [{"type": "text", "text": "x" * 96}], "pinned": False},
                    {"id": "old-assistant", "role": "assistant", "content": [{"type": "text", "text": "y" * 96}], "pinned": False},
                    {"id": "user-turn-2", "role": "user", "content": [{"type": "text", "text": "current"}], "pinned": False},
                ],
            }
            packed = pack_context(GenerationRequest.from_mapping(long_request_payload), short_model, UTF8ByteTokenizer())
            included = {message.message_id for message in packed.messages}
            omitted = set(packed.omitted_message_ids)
            require({"system-policy", "user-turn-2"}.issubset(included), "context packer dropped required policy/latest user")
            require({"old-user", "old-assistant"}.issubset(omitted), "context packer split or retained an over-budget old turn")
            require(packed.input_tokens + 64 <= 220, "context packer exceeded model budget")

            incapable_model = json.loads(MODEL_FIXTURE.read_text(encoding="utf-8"))
            incapable_model["capabilities"] = {"text": True, "image": False, "audio": False, "video": False}
            incapable_path = workspace / "text-only-model.json"
            write_json(incapable_path, incapable_model)
            refusal = run(
                [
                    sys.executable, str(GATEWAY), "--model", str(incapable_path), "--request", str(multimodal_request_path),
                    "--asset-registry", str(registry), "--asset-root", str(store), "--worker-command-json", str(command),
                ],
                expected=2,
            )
            require("does not declare" in refusal.stdout and "capability" in refusal.stdout,
                    f"gateway accepted a modality absent from immutable model capability declaration:\n{refusal.stdout}")
    except (AssertionError, LMMContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"lmm runtime verification: FAIL: {error}", file=sys.stderr)
        return 1

    print("lmm runtime verification: PASS (Unicode, subword-ready contract, model manifest, long context, trust-role evidence isolation, structured output, bounded worker I/O, redirect-safe endpoint, assets, held-out quality gate, capability gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
