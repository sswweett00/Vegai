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
    subject_of_prompt,
    verify,
)
from summarise_prompt_response_run import SummaryError, summarise  # noqa: E402
from report_prompt_response_evidence import collect, examples, render  # noqa: E402

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


ASK_HEADER = "### {kind} | expected: {expected}\n"


def parse_ask_log(text: str) -> list[dict]:
    """Pull the prompt/response objects out of an `ask.log`.

    The log interleaves a human header per prompt with one PROMPT_RESPONSE_JSON
    block, so the blocks are collected the same way the evidence report does it.
    """
    entries: list[dict] = []
    for block in text.split("PROMPT_RESPONSE_JSON {")[1:]:
        body = block.split("}", 1)[0]
        record: dict = {}
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if not line.startswith('"'):
                continue
            key, _, value = line.partition(": ")
            record[key.strip('"')] = value.strip().strip('"')
        entries.append(record)
    return entries


def check_ask_responses(directory: Path) -> None:
    """A prompt that is answered independently of its content is a bug.

    The prompt is passed to `respond` through a fixed-length buffer.  When that
    buffer was measured with `len()` instead of `len_trim()`, every ask was
    preceded by hundreds of padding spaces and all prompts produced one and the
    same response: the signature is that the responses are identical even though
    the prompts and their expected answers are not.  Journal-shaped logs must
    fail that test, so a regression cannot pass unnoticed again.
    """
    run = directory / "ask-check"
    run.mkdir()
    entries = [
        ("fact", "Soru: a nedir? Cevap: ", "b", "zzz"),
        ("fact", "Soru: c nedir? Cevap: ", "d", "qqq"),
        ("copy", "Tekrar et: e. Sonuc: ", "e", "ee"),
    ]
    log = "".join(
        ASK_HEADER.format(kind=kind, expected=expected)
        + "PROMPT_RESPONSE_JSON {\n"
        + f'  "prompt": "{prompt}",\n'
        + f'  "response": "{response}",\n'
        + '  "warmup_bytes": 0\n'
        + "}\n"
        for kind, prompt, expected, response in entries
    )
    (run / "ask.log").write_text(log, encoding="utf-8")
    parsed = parse_ask_log(log)
    check(len(parsed) == 3, "every ask block is parsed")
    check(len({entry["response"] for entry in parsed}) == 3,
          "three different prompts do not collapse into one response")

    degenerate = "".join(
        ASK_HEADER.format(kind=kind, expected=expected)
        + "PROMPT_RESPONSE_JSON {\n"
        + f'  "prompt": "{prompt}",\n'
        + '  "response": "`\u00e2RRR",\n'
        + '  "warmup_bytes": 0\n'
        + "}\n"
        for kind, prompt, expected, _ in entries
    )
    collapsed = {entry["response"] for entry in parse_ask_log(degenerate)}
    check(len(collapsed) == 1,
          "the identical-response signature is detectable, which is what this check guards")


def check_evidence_report(directory: Path) -> None:
    """The evidence report must read the engine's own reports and say what is
    missing instead of inventing a number."""
    run = directory / "evidence"
    run.mkdir()
    (run / "corpus-generate.json").write_text(json.dumps(
        {"bytes": 400_000, "documents": 700, "sha256": "deadbeef"}), encoding="utf-8")
    (run / "corpus-verify.json").write_text(json.dumps(
        {"bytes": 400_000, "documents": 700, "facts": 64}), encoding="utf-8")
    (run / "training-summary.json").write_text(json.dumps({
        "sessions": 2, "corpus_bytes": 400_000, "end_byte": 400_000, "corpus_fraction": 1.0,
        "total_tokens": 1_000_000, "total_seconds": 50, "tokens_per_second": 20_000.0,
        "hours_for_full_corpus": 0.006, "numeric_ok": True, "parameters": 22_897,
        "learned_reach_start": 8192.0, "learned_reach_end": 8000.0,
        "loss_curve": [{"session": 1, "mean_nll": 2.0}, {"session": 2, "mean_nll": 1.5}],
    }), encoding="utf-8")
    (run / "probe.log").write_text(
        'PROBE_EXAMPLES [\n  {"kind": "fact", "prompt": "Soru: a nedir? Cevap: ", '
        '"expected": "b", "response": "b"}\n]\n'
        'PROBE_JSON {\n  "rows": 2,\n  "fact": {"correct": 1, "accuracy": 0.5, "n": 2},\n'
        '  "copy": {"correct": 0, "accuracy": 0.0, "n": 0},\n'
        '  "next": {"correct": 0, "accuracy": 0.0, "n": 0},\n  "accuracy": 0.5\n}\n',
        encoding="utf-8")
    text = render(collect(run))
    joined = "\n".join(text)
    check(any("corpus: 400,000 bytes" in line for line in text), "the report names the corpus size")
    check(any("deadbeef" in line for line in text), "the report keeps the corpus hash")
    check(any("probe fact: exact 1/2" in line for line in text), "the report keeps the probe counts")
    check(any("-> 8000.0 tokens" in line for line in text), "the report keeps the learned reach")
    check("Soru: a nedir?" in joined, "the report keeps a real prompt/response pair")

    empty = directory / "evidence-empty"
    empty.mkdir()
    missing = "\n".join(render(collect(empty)))
    check("training: report missing" in missing,
          "a missing training report is reported as missing, not as zero")
    check(examples("PROBE_EXAMPLES [\n  {\"a\": 1}\n]\n") == [{"a": 1}],
          "the example parser reads a closed array")


def check_ask_log_text(text: str) -> list[str]:
    """Problems with a real `ask.log`, empty when it looks like a working model.

    One response per prompt is the contract; the same response for every prompt
    means the model never saw the prompt, which is exactly what a padding bug or
    a state leak looks like from the outside.
    """
    entries = parse_ask_log(text)
    problems: list[str] = []
    if len(entries) < 2:
        problems.append(f"ask.log has {len(entries)} response block(s); at least two are needed")
        return problems
    if any(not entry.get("response") for entry in entries):
        problems.append("ask.log has a block with an empty response")
    if len({entry.get("response") for entry in entries}) == 1:
        problems.append("every prompt produced the same response, so the prompt was not used")
    return problems


def main() -> int:
    if "--ask-log" in sys.argv:
        index = sys.argv.index("--ask-log")
        if index + 1 >= len(sys.argv):
            print("usage: verify_prompt_response_corpus.py [--ask-log FILE]")
            return 2
        path = Path(sys.argv[index + 1])
        if not path.is_file():
            print(f"missing ask log: {path}")
            return 2
        problems = check_ask_log_text(path.read_text(encoding="utf-8", errors="replace"))
        for problem in problems:
            print(f"FAIL {problem}")
        if problems:
            return 1
        print(f"ask log ok: {len(parse_ask_log(path.read_text(encoding='utf-8', errors='replace')))} responses")
        return 0

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

        # The bindings layout exists because the template layout let a model
        # reach a low loss without learning anything contextual.  Its contract:
        # every line is a binding, the probe asks only about bindings, and the
        # subject is recoverable from both a corpus line and a probe prompt.
        bindings = root / "bindings.txt"
        bindings_probe = root / "bindings.tsv"
        bindings_manifest = generate(bindings, root / "bindings.json", 200_000, 1234, facts=32,
                                     overwrite=True, probe_path=bindings_probe,
                                     layout="bindings")
        check(bindings_manifest["layout"] == "bindings", "the manifest records the layout")
        check(bindings_manifest["algorithm"] == "turkish-bindings/v1",
              "the bindings layout has its own algorithm id")
        lines = [line for line in bindings.read_text(encoding="utf-8").splitlines() if line]
        check(all(line.startswith("S: ") and " -> " in line for line in lines),
              "every line of the bindings corpus is a binding")
        check(verify(bindings, root / "bindings.json")["layout"] == "bindings",
              "the verifier reads the layout back")
        binding_rows = [line.split("\t") for line in
                        bindings_probe.read_text(encoding="utf-8").splitlines() if line]
        check(all(row[0] == "fact" for row in binding_rows),
              "the bindings probe ships facts only, because the other families need prose")
        subject_line = lines[0][len("S: "):].split(" -> ")[0]
        check(any(row[1] == f"S: {subject_line} -> " for row in binding_rows),
              "a binding prompt asks about a subject the corpus binds")
        check(subject_of_prompt(f"S: {subject_line} -> ") == subject_line,
              "the subject is recovered from a bindings prompt")
        filtered = root / "bindings-filtered.tsv"
        stats_bindings = filter_probe(bindings_probe, filtered, bindings, trained_bytes=50_000)
        check(stats_bindings["rows"] == len(binding_rows),
              "filter-probe keeps every binding row the prefix contains")

        check_run_summary(root)
        check_ask_responses(root)
        good = ("### fact | expected: b\nPROMPT_RESPONSE_JSON {\n"
                '  "prompt": "Soru: a nedir? Cevap: ",\n  "response": "b"\n}\n'
                "### fact | expected: d\nPROMPT_RESPONSE_JSON {\n"
                '  "prompt": "Soru: c nedir? Cevap: ",\n  "response": "d"\n}\n')
        check(check_ask_log_text(good) == [], "a healthy ask log passes the live check")
        check(check_ask_log_text(good.replace('"response": "d"', '"response": "b"')) != [],
              "an ask log whose prompts all answer the same byte is rejected")
        check_evidence_report(root)

    if FAILURES:
        print(f"{len(FAILURES)} prompt/response corpus contract checks failed")
        return 1
    print("prompt/response corpus contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
