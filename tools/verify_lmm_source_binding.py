#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove LMM quality/routing identities bind the gateway's project helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import lmm_gateway
import lmm_portfolio
import lmm_quality_evaluation
from lmm_contract import LMMContractError, file_identity


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("lmm_gateway", "lmm_contract", "lmm_process", "bounded_process")
MODEL = ROOT / "examples" / "lmm" / "mini_multilingual_model.json"
REQUEST = ROOT / "examples" / "lmm" / "mini_multilingual_request.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def reject(call, expected: str) -> None:
    try:
        call()
    except LMMContractError as error:
        require(expected in str(error), f"unexpected failure: {error}")
    else:
        raise AssertionError(f"accepted invalid provenance; expected {expected}")


def main() -> int:
    sources = {name: identity.to_mapping() for name, identity in lmm_gateway.runtime_source_identities().items()}
    identities = {
        "gateway_tool": file_identity(ROOT / "tools" / "lmm_gateway.py", "gateway").to_mapping(),
        "runtime_sources": sources,
        "model_manifest": file_identity(MODEL, "model").to_mapping(),
        "tokenizer": None, "asset_registry": None, "worker_command_configuration": None,
        "structured_response_contract": None, "model_artifacts": {},
    }
    _, quality_digest = lmm_quality_evaluation.stable_runtime_identity({"identities": identities})
    require(quality_digest == lmm_portfolio.gateway_runtime_identity(identities),
            "quality and dispatch runtime identities disagree")
    changed = copy.deepcopy(identities)
    changed["runtime_sources"]["bounded_process"]["sha256"] = "a" * 64
    require(lmm_portfolio.gateway_runtime_identity(changed) != quality_digest,
            "changed bounded-process source did not change the runtime digest")
    missing = copy.deepcopy(identities)
    del missing["runtime_sources"]
    reject(lambda: lmm_quality_evaluation.stable_runtime_identity({"identities": missing}), "runtime_sources")
    reject(lambda: lmm_portfolio.gateway_runtime_identity(missing), "runtime_sources")
    malformed = copy.deepcopy(identities)
    malformed["runtime_sources"]["lmm_contract"]["bytes"] = True
    reject(lambda: lmm_portfolio.gateway_runtime_identity(malformed), "integer")

    with tempfile.TemporaryDirectory(prefix="vegai-lmm-source-binding-") as temporary:
        root = Path(temporary)
        tools = root / "tools"
        tools.mkdir()
        for name in SOURCES:
            shutil.copyfile(ROOT / "tools" / f"{name}.py", tools / f"{name}.py")
        worker = root / "worker.py"
        worker.write_text(
            "import json, pathlib, sys\n"
            "sys.stdin.buffer.read()\n"
            "if sys.argv[1] == 'mutate':\n"
            "    path = pathlib.Path(sys.argv[2])\n"
            "    with path.open('a', encoding='utf-8') as handle: handle.write('\\n# changed during model call\\n')\n"
            "print(json.dumps({'schema_version':'vegai-lmm-worker-response/v1','text':'ok',"
            "'finish_reason':'stop','input_tokens':1,'output_tokens':1,'backend':'fixture','device':None}))\n",
            encoding="utf-8",
        )
        worker_config = root / "worker.json"
        report = root / "report.json"
        copied_contract = tools / "lmm_contract.py"

        def invoke(mode: str, output: Path) -> subprocess.CompletedProcess[str]:
            worker_config.write_text(json.dumps([sys.executable, str(worker), mode, str(copied_contract)]),
                                     encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(tools / "lmm_gateway.py"), "--model", str(MODEL), "--request", str(REQUEST),
                 "--worker-command-json", str(worker_config), "--timeout-seconds", "5", "--output", str(output),
                 "--print-json"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=15, check=False,
            )

        success = invoke("normal", report)
        require(success.returncode == 0, f"copied gateway failed: {success.stderr}")
        result = json.loads(report.read_text(encoding="utf-8"))
        observed = result["identities"]["runtime_sources"]
        require(set(observed) == {"lmm_contract", "lmm_process", "bounded_process"},
                "generation report omitted runtime helper provenance")
        require(observed["lmm_contract"]["sha256"] == file_identity(copied_contract, "copied contract").sha256,
                "gateway reported a different imported contract")

        before = copied_contract.read_bytes()
        collision = invoke("normal", copied_contract)
        require(collision.returncode == 2 and copied_contract.read_bytes() == before,
                "gateway report could overwrite an imported runtime source")

        untrusted_report = root / "must-not-exist.json"
        mutation = invoke("mutate", untrusted_report)
        require(mutation.returncode == 2 and "runtime source lmm_contract" in mutation.stderr,
                f"gateway accepted helper mutation during generation: {mutation.stderr}")
        require(not untrusted_report.exists(), "gateway published a report after helper mutation")

    print("LMM source binding verification: PASS (quality/dispatch digest, legacy rejection, collision, mutation)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, LMMContractError) as error:
        print(f"LMM source binding verification: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
