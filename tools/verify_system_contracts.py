#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regression tests for strict durable JSON and copy-on-write training commits."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import continuous_pretrain as continuous
import lmm_contract
import lmm_gateway
import lmm_knowledge
import lmm_portfolio
import lmm_quality_evaluation
import lmm_release_governor
import lmm_release_registry
import lmm_session
import lmm_tool_orchestrator
import production_pretrain as production
from verify_continuous_quality_regression import manifest


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def reject(call, error_type: type[Exception], diagnostic: str) -> None:
    try:
        call()
    except error_type as error:
        # Report the raising line: interpreter-version-specific failure paths
        # must be diagnosable from CI output alone.
        require(diagnostic in str(error),
                f"wrong diagnostic {error!s}; expected {diagnostic!r}; "
                f"raised at {traceback.format_exc(limit=-3)}")
    else:
        raise AssertionError(f"accepted invalid input; expected {diagnostic!r}")


def args() -> SimpleNamespace:
    return SimpleNamespace(max_vocab=None, min_frequency=None, window=None, position_shift=None, context_mode=None,
                           epochs=None, initial_plasticity=None, minimum_plasticity=None, decay_events=None)


def verify_json_inputs(root: Path) -> None:
    source = root / "object.json"
    valid = b'{"nested":{"text":"Istanbul"},"number":1.25}'
    source.write_bytes(valid)
    require(lmm_contract.read_json_object(source, "fixture")["number"] == 1.25, "valid LMM JSON failed")
    require(continuous.load_json(source, "fixture")["number"] == 1.25, "valid controller JSON failed")
    require(production.read_json(source, "fixture")["number"] == 1.25, "valid production JSON failed")

    for raw in (b'{"role":"system","role":"user"}', b'{"nested":{"x":1,"x":2}}',
                b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":"\xff"}',
                b'{"x":' + b'[' * 1100 + b'0' + b']' * 1100 + b'}'):
        source.write_bytes(raw)
        reject(lambda: lmm_contract.read_json_object(source, "fixture"), lmm_contract.LMMContractError, "strict")
        reject(lambda: continuous.load_json(source, "fixture"), continuous.ContinuousPretrainError, "cannot load")
        reject(lambda: production.read_json(source, "fixture"), production.ProductionRunError, "cannot load")
        for parse in (lmm_tool_orchestrator.parse_strict_json, lmm_release_governor.parse_strict_json,
                      lmm_release_registry.parse_strict_json):
            reject(lambda: parse(raw, "fixture"), lmm_contract.LMMContractError, "strict")

    # Extra duplicate keys used to be silently ignored while the last value
    # still matched the content-addressed/event-hash contract.
    asset = {
        "schema_version": lmm_contract.ASSET_SCHEMA, "id": "img", "modality": "image", "mime_type": "image/png",
        "path": "img.png", "sha256": "a" * 64, "bytes": 3, "context_tokens": 1,
        "source": "fixture", "license": "MIT", "retrieved_at": "fixture", "metadata": {},
    }
    asset_line = lmm_contract.canonical_json_bytes(asset)
    asset_file = root / "assets.jsonl"
    asset_file.write_bytes(asset_line + b"\n")
    require("img" in lmm_contract.AssetRegistry.from_jsonl(asset_file).assets, "valid asset registry failed")
    asset_file.write_bytes(b'{"id":"decoy",' + asset_line[1:] + b"\n")
    reject(lambda: lmm_contract.AssetRegistry.from_jsonl(asset_file), lmm_contract.LMMContractError, "duplicate")

    knowledge_file = root / "knowledge.jsonl"
    valid_knowledge = (ROOT / "examples" / "lmm" / "mini_multilingual_knowledge.jsonl").read_bytes().splitlines()[0]
    knowledge_file.write_bytes(valid_knowledge + b"\n")
    require(len(list(lmm_knowledge.iter_bounded_jsonl(knowledge_file, 10))) == 1, "valid knowledge failed")
    knowledge_file.write_bytes(b'{"id":"decoy",' + valid_knowledge[1:] + b"\n")
    reject(lambda: list(lmm_knowledge.iter_bounded_jsonl(knowledge_file, 10)),
           lmm_contract.LMMContractError, "duplicate")

    request = lmm_contract.GenerationRequest.from_file(ROOT / "examples" / "lmm" / "mini_multilingual_request.json")
    ledger = root / "session.jsonl"
    lmm_session.write_ledger_atomic(ledger, [request.messages[0]])
    require(len(lmm_session.load_events(ledger)[0]) == 1, "valid hash-chained session failed")
    valid_event = ledger.read_bytes().rstrip(b"\n")
    ledger.write_bytes(b'{"sequence":1,' + valid_event[1:] + b"\n")
    reject(lambda: lmm_session.load_events(ledger), lmm_contract.LMMContractError, "duplicate")

    # Appending must never replace a valid ledger with one its own reader
    # rejects; both row and total limits are pre-publication transaction gates.
    bounded_ledger = root / "bounded-session.jsonl"
    first_head = lmm_session.write_ledger_atomic(bounded_ledger, [request.messages[0]])
    first_bytes = bounded_ledger.read_bytes()
    second_message = root / "second-message.json"
    second_message.write_text(json.dumps(request.messages[1].to_mapping()), encoding="utf-8")
    append_args = SimpleNamespace(ledger=bounded_ledger, message=second_message, expected_head=first_head,
                                  lock_timeout_seconds=2.0)
    with patch.object(lmm_session, "MAX_JSON_BYTES", len(first_bytes) + 40):
        reject(lambda: lmm_session.append_event(append_args), lmm_contract.LMMContractError, "would exceed")
        require(bounded_ledger.read_bytes() == first_bytes and lmm_session.load_events(bounded_ledger)[1] == first_head,
                "rejected oversized append damaged the existing ledger")
    with patch.object(lmm_session, "MAX_EVENT_LINE_BYTES", len(first_bytes) - 2):
        reject(lambda: lmm_session.write_ledger_atomic(bounded_ledger, [request.messages[0]]),
               lmm_contract.LMMContractError, "event 1 exceeds")
        require(bounded_ledger.read_bytes() == first_bytes, "overlong event damaged the existing ledger")
    with patch.object(lmm_session, "MAX_EVENT_LINE_BYTES", len(first_bytes) - 1), \
         patch.object(lmm_session, "MAX_JSON_BYTES", len(first_bytes)):
        require(lmm_session.write_ledger_atomic(bounded_ledger, [request.messages[0]]) == first_head,
                "exact event/ledger byte limits were not accepted")
        require(lmm_session.load_events(bounded_ledger)[1] == first_head, "exact byte-limit ledger was unreadable")
    require(not list(root.glob(".bounded-session.jsonl.*.tmp")) and
            not (root / ".bounded-session.jsonl.lock").exists(),
            "failed ledger append left a private temporary file or lock")

    case_file = root / "cases.jsonl"
    case_file.write_bytes(b'{"id":"first","id":"second"}\n')
    reject(lambda: list(lmm_quality_evaluation.iter_cases(case_file, 10)), lmm_contract.LMMContractError, "duplicate")

    response = {
        "schema_version": lmm_contract.WORKER_RESPONSE_SCHEMA, "text": "ok", "finish_reason": "stop",
        "input_tokens": 1, "output_tokens": 1, "backend": "worker", "device": None,
    }
    response_raw = lmm_contract.canonical_json_bytes(response)
    require(lmm_gateway.parse_worker_response(response_raw).text == "ok", "valid worker response failed")
    reject(lambda: lmm_gateway.parse_worker_response(b'{"text":"decoy",' + response_raw[1:]),
           lmm_contract.LMMBackendError, "strict")
    source.write_bytes(b'{"x":1,"x":2}')
    reject(lambda: lmm_portfolio.load_json_mapping(source, "portfolio"), lmm_contract.LMMContractError, "duplicate")

    # Once a too-long line is detected, strict readers must reject immediately
    # rather than draining a potentially ever-growing producer.
    class NeverDrain:
        calls = 0

        def readline(self, limit: int) -> bytes:
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("overlong line was drained")
            return b"x" * limit

    handle = NeverDrain()
    reject(lambda: next(iter(lmm_contract._iter_bounded_jsonl_lines(handle, 8))),
           lmm_contract.LMMContractError, "exceeds")
    require(handle.calls == 1, "overlong JSONL line was drained before rejection")


def verify_manifest_types(root: Path) -> tuple[Path, list[continuous.Shard]]:
    ledger = manifest(root, fake=True)
    original = json.loads(ledger.read_text(encoding="utf-8"))
    valid_shards = continuous.load_manifest(ledger)
    for category, field in (("shard", "bytes"), ("shard", "physical_lf_records"),
                            ("preflight", "accepted_tokens"), ("preflight", "maximum_record_bytes")):
        altered = copy.deepcopy(original)
        target = altered["shards"][0] if category == "shard" else altered["shards"][0]["preflight"]
        target[field] = True
        ledger.write_text(json.dumps(altered), encoding="utf-8")
        reject(lambda: continuous.load_manifest(ledger), continuous.ContinuousPretrainError, "invalid" if category == "shard" else "malformed")
    ledger.write_text(json.dumps(original), encoding="utf-8")

    state_dir = root / "typed-state"
    state_dir.mkdir()
    state = continuous.fresh_state(continuous.default_policy(args()))
    state["generation"] = False
    (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    reject(lambda: continuous.load_state(state_dir, args()), continuous.ContinuousPretrainError, "generation is invalid")
    state["generation"] = 1
    state["applied_shards"] = [{"id": "a" * 64, "sha256": "a" * 64, "generation": True}]
    (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    reject(lambda: continuous.load_state(state_dir, args()), continuous.ContinuousPretrainError, "inconsistent generation")
    return ledger, valid_shards


def verify_copy_on_write(root: Path, shards: list[continuous.Shard]) -> None:
    class NoDeepcopy:
        def __deepcopy__(self, memo: object) -> object:
            raise AssertionError("immutable historical entry was recursively copied")

    historical = [{"evidence": NoDeepcopy()} for _ in range(10_000)]
    original = {"generation": len(historical), "checkpoint": "old", "checkpoint_sha256": "a" * 64,
                "applied_shards": historical, "policy": {"epochs_per_shard": 1}}
    entry = {"id": "new"}
    updated = continuous.state_with_committed_shard(original, 10_001, "new", "b" * 64, entry)
    require(len(original["applied_shards"]) == 10_000 and original["checkpoint"] == "old",
            "preparing a commit mutated its prior state")
    require(len(updated["applied_shards"]) == 10_001 and updated["applied_shards"][0] is historical[0],
            "commit did not reuse immutable historical entries")
    updated["applied_shards"].append({"id": "later"})
    require(len(original["applied_shards"]) == 10_000, "new journal list aliases the previous list")

    state_dir = root / "transaction"
    state_dir.mkdir()
    state = continuous.fresh_state(continuous.default_policy(args()))
    original_state = copy.deepcopy(state)
    measurement = {
        "scope": "one vegai_pretrain train-or-continue child-process wall-clock interval",
        "start_boundary": "immediately before spawning vegai_pretrain",
        "stop_boundary": "immediately after vegai_pretrain exits",
        "excluded": ["compilation", "quality evaluation", "controller preflight", "CI orchestration"],
        "train_process_wall_seconds": 1.0, "omp_num_threads": 1,
        "stdout_sha256": "a" * 64, "stdout_bytes": 0, "stderr_sha256": "b" * 64, "stderr_bytes": 0,
    }

    def fake_invoke(command, _arguments):
        Path(command[3]).write_bytes(b"fake-verified-checkpoint")
        return "fake output", dict(measurement)

    def fake_metrics(_output, shard, _policy, _generation, _extending):
        return {"vocabulary_observed_tokens": shard.accepted_tokens, "training_raw_tokens": shard.accepted_tokens,
                "training_active_tokens": shard.accepted_tokens, "training_oov_tokens": 0}

    with patch.object(continuous, "invoke", side_effect=fake_invoke), \
         patch.object(continuous, "validate_cli_output", side_effect=fake_metrics), \
         patch.object(continuous, "resource_preflight_for_shard", return_value={"fixture": True}):
        with patch.object(continuous, "atomic_write_json", side_effect=OSError("simulated disk failure")):
            reject(lambda: continuous.commit_one_shard(state_dir, Path("fake-binary"), state, shards[0], args()),
                   OSError, "disk failure")
        require(state == original_state and not (state_dir / "state.json").exists(),
                "failed state publication changed the prior in-memory or on-disk commit marker")
        require(not list((state_dir / "generations").glob(".candidate-*")) and
                not list((state_dir / "generations").glob(".shard-*")),
                "failed state publication left private work files")
        state1 = continuous.commit_one_shard(state_dir, Path("fake-binary"), state, shards[0], args())
        require(state1["generation"] == 1 and state == original_state,
                "first successful commit changed its old state")
        state2 = continuous.commit_one_shard(state_dir, Path("fake-binary"), state1, shards[1], args())
        require(state2["generation"] == 2 and len(state1["applied_shards"]) == 1,
                "second commit modified the prior journal spine")
        reloaded = continuous.load_state(state_dir, args())
        require(reloaded["generation"] == 2 and len(reloaded["applied_shards"]) == 2,
                "copy-on-write journal did not survive a complete validated reload")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vegai-system-contracts-") as temporary:
        root = Path(temporary)
        verify_json_inputs(root)
        _, shards = verify_manifest_types(root)
        verify_copy_on_write(root, shards)
    print("system contracts verification: PASS (strict JSON, numeric types, bounded records, transactional journal)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, lmm_contract.LMMContractError, continuous.ContinuousPretrainError,
            production.ProductionRunError) as error:
        print(f"system contracts verification: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
