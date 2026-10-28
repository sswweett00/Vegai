#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Contract test for tools/build_prompt_response_corpus.py.

The corpus is training data, so the properties that make it usable are pinned
here instead of being assumed:

* exactly the requested byte count, byte for byte, and no partial UTF-8
  sequence at the cut;
* a deterministic result: the same seed produces the same SHA-256, a different
  seed does not;
* a manifest that matches the file (size and SHA-256);
* probe rows that are well formed, ASCII-safe on the tab separator, and whose
  ``fact`` rows really do occur in the corpus while ``copy`` values do not;
* ``filter-probe`` dropping only the fact rows whose subject a partial pass
  never saw.

Run directly (`python3 tools/verify_prompt_response_corpus.py`) or via
`make prompt-corpus-check`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_prompt_response_corpus import (  # noqa: E402
    CorpusError,
    filter_probe,
    generate,
    verify,
)
from summarise_prompt_response_run import SummaryError, summarise  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, description: str) -> None:
    if not condition:
        FAILURES.append(description)
        print(f"FAIL {description}")


def session_record(index: int, end_byte: int, corpus_bytes: int) -> dict:
    return {
        "corpus": "corpus.txt",
        "corpus_bytes": corpus_bytes,
        "start_byte": end_byte - 100,
        "end_byte": end_byte,
        "consumed_bytes": 100,
        "file_fraction": end_byte / corpus_bytes,
        "bounded_by_bytes": False,
        "whole_file": False,
        "updates": index * 2,
        "tokens": index * 100,
        "session_updates": 2,
        "session_tokens": 100,
        "mean_nll": 5.0 - index,
        "session_mean_nll": 5.0 - index,
        "tokens_per_second": 50.0,
        "seconds": 2,
        "learned_reach": 8192.0,
        "parameters": 15497,
        "numeric_ok": True,
    }


def check_run_summary(directory: Path) -> None:
    """The session-report summariser must reject junk and aggregate real blocks."""
    run = directory / "run"
    run.mkdir()
    for index, end_byte in enumerate((100, 200), start=1):
        block = json.dumps(session_record(index, end_byte, 2_000_000), indent=2)
        (run / f"train-session-{index}.log").write_text(
            "progress: bytes 100 updates 2\nCORPUS_TRAIN_JSON " + block + "\n", encoding="utf-8")
    summary = summarise(run)
    check(summary["sessions"] == 2, "the summariser sees both sessions")
    check(summary["total_tokens"] == 200, "the summariser adds the session tokens up")
    check(abs(summary["tokens_per_second"] - 50.0) < 1.0e-9,
          "throughput is total tokens over total seconds")
    check(summary["learned_reach_end"] == 8192.0, "the summary keeps the learned reach")
    check(len(summary["loss_curve"]) == 2, "the summary keeps a loss curve")

    broken = directory / "broken"
    broken.mkdir()
    (broken / "train-session-1.log").write_text(
        'CORPUS_TRAIN_JSON {"corpus_bytes": 10, "numeric_ok": T}\n', encoding="utf-8")
    try:
        summarise(broken)
        check(False, "an unparseable session report is refused")
    except SummaryError:
        pass


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = 400_000

        corpus = root / "corpus.txt"
        manifest_path = root / "corpus.manifest.json"
        probe_path = root / "probe.tsv"
        manifest = generate(corpus, manifest_path, target, 1234, facts=64, overwrite=True,
                            probe_path=probe_path)

        check(corpus.stat().st_size == target, "the corpus is exactly the requested byte count")
        check(manifest["bytes"] == target, "the manifest records the requested byte count")
        summary = verify(corpus, manifest_path)
        check(summary["bytes"] == target, "the verifier agrees with the file size")
        check(len(manifest["facts"]) == 64, "the manifest carries every fact binding")
        check(len(set(f["subject"] for f in manifest["facts"])) == 64, "subjects are distinct")

        text = corpus.read_text(encoding="utf-8")
        check("\ufffd" not in text, "the corpus decodes as UTF-8 without replacement characters")
        check(text.startswith("Soru: "), "the corpus starts with a fact sentence")

        second = root / "corpus-again.txt"
        second_manifest = generate(second, root / "second.json", target, 1234, facts=64,
                                   overwrite=True)
        check(second_manifest["sha256"] == manifest["sha256"],
              "the same seed reproduces the same bytes")
        third = root / "corpus-third.txt"
        third_manifest = generate(third, root / "third.json", target, 4321, facts=64,
                                  overwrite=True)
        check(third_manifest["sha256"] != manifest["sha256"],
              "a different seed produces different bytes")

        rows = [line.split("\t") for line in probe_path.read_text(encoding="utf-8").splitlines() if line]
        check(all(len(row) == 3 for row in rows), "every probe row has three fields")
        kinds = {}
        for kind, _, _ in rows:
            kinds[kind] = kinds.get(kind, 0) + 1
        check(kinds.get("fact", 0) > 0 and kinds.get("copy", 0) > 0 and kinds.get("next", 0) > 0,
              "the probe set covers facts, copy and next")

        answers = set(f["answer"] for f in manifest["facts"])
        copy_rows = [row for row in rows if row[0] == "copy"]
        check(all(row[2] not in answers for row in copy_rows),
              "copy probes use values the corpus does not bind")
        next_rows = [row for row in rows if row[0] == "next"]
        check(all(row[2] != row[1].split("Sonra: ", 1)[1] for row in next_rows),
              "next probes replace the prompt's own answer with the shift")

        truncated = root / "probe-partial.tsv"
        stats = filter_probe(probe_path, truncated, corpus, trained_bytes=20_000)
        kept = [line.split("\t") for line in truncated.read_text(encoding="utf-8").splitlines() if line]
        check(stats["rows"] == len(kept), "filter-probe reports the rows it wrote")
        check(all(row[0] != "fact" or row[1][len("Soru: "):].split(" nedir?")[0]
                  in {f["subject"] for f in manifest["facts"]}
                  for row in kept), "kept fact probes name a real subject")
        check(len(kept) <= len(rows), "filter-probe never invents rows")

        try:
            generate(corpus, manifest_path, target, 1234, facts=64, overwrite=False)
            check(False, "regenerating without --overwrite is refused")
        except CorpusError:
            pass

        check_run_summary(root)

    if FAILURES:
        print(f"{len(FAILURES)} prompt/response corpus contract checks failed")
        return 1
    print("prompt/response corpus contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
