#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Condense a prompt/response evidence run into one readable block.

Reads what a `prompt-response-600mb` job produced in its run directory --
`corpus-verify.json`, `training-summary.json`, `probe-filter.json`, `probe.log`,
`ask.log` -- and prints a short, honest summary:

* the exact corpus and the exact bytes that were trained on;
* throughput and the loss trajectory, so the run is a curve and not a claim;
* the learned memory reach, before and after training;
* prompt/response accuracies per family, with the number of probes behind each;
* a few prompt -> response lines as they came out.

Nothing here is recomputed from theory: every number is read from a report the
engine wrote.  A missing report is reported as missing, never as zero.

Usage: ``python3 tools/report_prompt_response_evidence.py RUN_DIR [--json]``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROBE_JSON = re.compile(r"^PROBE_JSON \{", re.MULTILINE)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def balanced_object(text: str, marker: str) -> dict | None:
    """The first balanced object that follows ``marker`` in ``text``."""
    start = text.find(marker)
    if start < 0:
        return None
    body = text[start + len(marker):]
    depth = 0
    for index, character in enumerate(body):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(body[:index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def examples(text: str, limit: int = 3) -> list[dict]:
    """The `PROBE_EXAMPLES [ ... ]` array, best effort."""
    start = text.find("PROBE_EXAMPLES [")
    if start < 0:
        return []
    body = text[start + len("PROBE_EXAMPLES ["):]
    # The array's own brackets were consumed by the marker, so the first `]` at
    # depth zero closes it; nested arrays (none today) would raise the depth.
    depth = 0
    for index, character in enumerate(body):
        if character == "[":
            depth += 1
        elif character == "]":
            if depth == 0:
                try:
                    parsed = json.loads("[" + body[:index].rstrip().rstrip(",") + "]")
                except json.JSONDecodeError:
                    return []
                return parsed[:limit]
            depth -= 1
    return []


def collect(run_dir: Path) -> dict:
    corpus = read_json(run_dir / "corpus-verify.json") or {}
    generate = read_json(run_dir / "corpus-generate.json") or {}
    training = read_json(run_dir / "training-summary.json") or {}
    filtering = read_json(run_dir / "probe-filter.json") or {}
    probe_text = (run_dir / "probe.log").read_text(encoding="utf-8", errors="replace") \
        if (run_dir / "probe.log").exists() else ""
    ask_text = (run_dir / "ask.log").read_text(encoding="utf-8", errors="replace") \
        if (run_dir / "ask.log").exists() else ""
    return {
        "corpus": {
            "bytes": corpus.get("bytes", generate.get("bytes")),
            "sha256": generate.get("sha256"),
            "documents": corpus.get("documents", generate.get("documents")),
            "facts": corpus.get("facts"),
        },
        "training": {key: value for key, value in training.items() if key not in ("per_session",)},
        "probe_filter": filtering,
        "probe": balanced_object(probe_text, "PROBE_JSON "),
        "examples": examples(probe_text),
        "ask": [line for line in ask_text.splitlines() if line.startswith("###")][:6],
    }


def render(evidence: dict) -> list[str]:
    corpus = evidence["corpus"]
    training = evidence["training"]
    probe = evidence["probe"] or {}
    lines: list[str] = []

    if corpus.get("bytes"):
        lines.append(f"corpus: {corpus['bytes']:,} bytes, {corpus.get('documents')} documents, "
                     f"{corpus.get('facts')} fact bindings")
        if corpus.get("sha256"):
            lines.append(f"corpus sha256: {corpus['sha256']}")
    else:
        lines.append("corpus: report missing")

    if training:
        fraction = training.get("corpus_fraction")
        lines.append(
            "training: {tokens:,} tokens over {sessions} session(s), {seconds:.0f} s, "
            "{rate:,.0f} tok/s, reached byte {end:,}{of}".format(
                tokens=int(training.get("total_tokens", 0)),
                sessions=int(training.get("sessions", 0)),
                seconds=float(training.get("total_seconds", 0)),
                rate=float(training.get("tokens_per_second", 0)),
                end=int(training.get("end_byte", 0)),
                of=f" of {int(training['corpus_bytes']):,} ({fraction:.1%})"
                   if fraction is not None else "",
            )
        )
        curve = training.get("loss_curve") or []
        if curve:
            first, last = curve[0], curve[-1]
            lines.append(f"loss: {first['mean_nll']:.4f} after session 1 -> {last['mean_nll']:.4f} "
                         f"after session {len(curve)}")
            # The whole curve, because a first-to-last pair hides a rise in the
            # middle and a training run has to be judged on its trend.
            lines.append("loss curve: " + ", ".join(f"{record['mean_nll']:.4f}" for record in curve))
        hours = training.get("hours_for_full_corpus")
        if hours:
            lines.append(f"projection: the whole corpus would take {hours:.2f} h at the measured rate")
        lines.append(f"learned reach: {training.get('learned_reach_start'):.1f} -> "
                     f"{training.get('learned_reach_end'):.1f} tokens; "
                     f"parameters {int(training.get('parameters', 0)):,}")
    else:
        lines.append("training: report missing")

    if probe:
        for family in ("fact", "copy", "next"):
            entry = probe.get(family) or {}
            detail = ""
            if entry.get("correct_bytes") is not None:
                detail = (f"; next-byte accuracy {entry.get('byte_accuracy', 0.0):.3f} over "
                          f"{entry.get('correct_bytes', 0)} bytes")
                if entry.get("mean_nll") is not None:
                    detail += f", teacher-forced nll {entry['mean_nll']:.4f}"
            lines.append(f"probe {family}: exact {entry.get('correct', 0)}/{entry.get('n', 0)} "
                         f"({entry.get('accuracy', 0.0):.3f}){detail}")
        lines.append(f"probe overall: exact {probe.get('accuracy', 0.0):.3f}, next-byte "
                     f"{probe.get('byte_accuracy', 0.0):.3f} over {probe.get('rows', 0)} rows")
    else:
        lines.append("probe: report missing")

    if evidence["probe_filter"].get("dropped_fact_rows") is not None:
        lines.append(f"probe filter: kept {evidence['probe_filter'].get('rows')} rows, dropped "
                     f"{evidence['probe_filter'].get('dropped_fact_rows')} fact rows whose subject the "
                     f"trained bytes never contained")

    for example in evidence["examples"]:
        lines.append(f"{example.get('kind')}: {example.get('prompt')!r} -> "
                     f"{example.get('response')!r} (expected {example.get('expected')!r})")
    return lines


MAX_ANNOTATIONS = 5


def annotation_payloads(evidence: dict) -> list[tuple[str, str]]:
    """Group the evidence into a few annotations.

    GitHub keeps only ten annotations per step, and the prompt -> response lines
    are the part a reader most needs, so they are packed first-few and the
    groups are capped instead of emitting one annotation per line (which
    silently dropped the examples at the end of the list).
    """
    corpus = evidence["corpus"]
    training = evidence["training"]
    probe = evidence["probe"] or {}
    groups: list[tuple[str, list[str]]] = []

    corpus_parts = []
    if corpus.get("bytes"):
        corpus_parts.append(f"{corpus['bytes']:,} bytes, {corpus.get('documents')} documents, "
                            f"{corpus.get('facts')} facts")
    if corpus.get("sha256"):
        corpus_parts.append(f"sha256 {corpus['sha256']}")
    if corpus_parts:
        groups.append(("corpus", corpus_parts))

    training_parts = []
    if training:
        training_parts.append(f"{int(training.get('total_tokens', 0)):,} tokens, "
                              f"{int(training.get('sessions', 0))} sessions, "
                              f"{float(training.get('tokens_per_second', 0)):,.0f} tok/s, "
                              f"byte {int(training.get('end_byte', 0)):,} of "
                              f"{int(training.get('corpus_bytes', 0)):,}")
        curve = training.get("loss_curve") or []
        if curve:
            training_parts.append("loss " + " -> ".join(f"{r['mean_nll']:.4f}" for r in curve))
        training_parts.append(f"learned reach {training.get('learned_reach_start'):.1f} -> "
                              f"{training.get('learned_reach_end'):.1f} tokens, "
                              f"{int(training.get('parameters', 0)):,} parameters")
    else:
        training_parts.append("no training report")
    groups.append(("training", training_parts))

    probe_parts = []
    if probe:
        for family in ("fact", "copy", "next"):
            entry = probe.get(family) or {}
            if int(entry.get("n", 0) or 0) == 0:
                # A family the probe file never contained is absent, not zero.
                continue
            # The teacher-forced NLL says how far the answer bytes are from the
            # model, which is the only number that separates "learned the frame"
            # from "learned the binding" when the exact match is still zero.
            detail = f"{family} exact {entry.get('correct', 0)}/{entry.get('n', 0)}, " \
                     f"next-byte {entry.get('byte_accuracy', 0.0):.3f}"
            if entry.get("mean_nll") is not None:
                detail += f", nll {entry['mean_nll']:.3f}"
            probe_parts.append(detail)
        probe_parts.append(f"overall exact {probe.get('accuracy', 0.0):.3f} / next-byte "
                           f"{probe.get('byte_accuracy', 0.0):.3f}")
    else:
        probe_parts.append("no probe report")
    groups.append(("probe", probe_parts))

    examples = evidence["examples"]
    if examples:
        groups.append(("prompt -> response", [
            f"{e.get('kind')}: {e.get('prompt')!r} -> {e.get('response')!r} "
            f"(expected {e.get('expected')!r}, greedy {e.get('greedy')!r})" for e in examples]))

    ask = evidence["ask"]
    if ask:
        groups.append(("ask headers", ask))

    payloads: list[tuple[str, str]] = []
    for title, parts in groups[:MAX_ANNOTATIONS]:
        text = " | ".join(part for part in parts if part)[:900]
        payloads.append((title, text))
    return payloads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="print the collected evidence as JSON")
    parser.add_argument("--annotations", action="store_true",
                        help="print at most five grouped annotations instead of one line each")
    arguments = parser.parse_args(argv)
    if not arguments.run_dir.is_dir():
        print(f"no run directory: {arguments.run_dir}", file=sys.stderr)
        return 1
    evidence = collect(arguments.run_dir)
    if arguments.annotations:
        for title, text in annotation_payloads(evidence):
            print(f"ANNOTATION::{title}::{text}")
        return 0
    if arguments.json:
        print(json.dumps(evidence, indent=1))
        return 0
    for line in render(evidence):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
