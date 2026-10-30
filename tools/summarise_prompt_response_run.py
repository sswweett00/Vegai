#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarise the per-session reports of a streamed prompt/response run.

`vegai_prompt_engine train` prints one `CORPUS_TRAIN_JSON { ... }` block per
session.  This tool reads those blocks, validates them strictly (a malformed
block is an error, not a warning: the numbers are evidence), and writes the
aggregate a reviewer needs:

* how many bytes and tokens were actually consumed, and out of what;
* the measured throughput of the sessions, and therefore how long the whole
  corpus would take at that rate;
* the loss trajectory across sessions, so "it trained" is a curve and not a
  claim;
* the learned memory reach, which is the point of the learnable dynamics.

Run: ``python3 tools/summarise_prompt_response_run.py RUN_DIR --output FILE``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MARKER = "CORPUS_TRAIN_JSON "
REQUIRED = {
    "corpus_bytes", "start_byte", "end_byte", "consumed_bytes", "updates", "tokens",
    "session_updates", "session_tokens", "mean_nll", "session_mean_nll",
    "tokens_per_second", "seconds", "learned_reach", "parameters", "numeric_ok",
    "whole_file", "bounded_by_bytes",
}


class SummaryError(RuntimeError):
    pass


def extract_blocks(text: str) -> list[dict]:
    """Every balanced ``CORPUS_TRAIN_JSON { ... }`` object in a session log."""
    records: list[dict] = []
    position = text.find(MARKER)
    while position >= 0:
        body = text[position + len(MARKER):]
        depth = 0
        end = None
        for index, character in enumerate(body):
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            raise SummaryError("unterminated CORPUS_TRAIN_JSON block")
        try:
            record = json.loads(body[:end])
        except json.JSONDecodeError as error:
            raise SummaryError(f"invalid JSON in CORPUS_TRAIN_JSON block: {error}") from error
        missing = REQUIRED - set(record)
        if missing:
            raise SummaryError(f"block is missing {sorted(missing)}")
        if not isinstance(record["numeric_ok"], bool):
            raise SummaryError("numeric_ok must be a JSON boolean")
        records.append(record)
        position = text.find(MARKER, position + end + len(MARKER))
    return records


def summarise(run_dir: Path) -> dict:
    sessions = sorted(run_dir.glob("train-session-*.log"))
    records: list[dict] = []
    for path in sessions:
        records.extend(extract_blocks(path.read_text(encoding="utf-8", errors="replace")))
    if not records:
        raise SummaryError(f"no CORPUS_TRAIN_JSON block found in {run_dir}")

    total_tokens = sum(record["session_tokens"] for record in records)
    total_seconds = sum(record["seconds"] for record in records)
    corpus_bytes = records[-1]["corpus_bytes"]
    end_byte = max(record["end_byte"] for record in records)
    throughput = total_tokens / max(1, total_seconds)
    return {
        "sessions": len(records),
        "corpus_bytes": corpus_bytes,
        "end_byte": end_byte,
        "corpus_fraction": end_byte / corpus_bytes if corpus_bytes else None,
        "whole_corpus": end_byte >= corpus_bytes,
        "total_tokens": total_tokens,
        "total_seconds": total_seconds,
        "tokens_per_second": throughput,
        "hours_for_full_corpus": corpus_bytes / throughput / 3600.0 if throughput > 0 else None,
        "numeric_ok": all(record["numeric_ok"] for record in records),
        "parameters": records[-1]["parameters"],
        "learned_reach_start": records[0]["learned_reach"],
        "learned_reach_end": records[-1]["learned_reach"],
        "loss_curve": [{"session": index + 1, "end_byte": record["end_byte"],
                        "mean_nll": record["mean_nll"],
                        "session_mean_nll": record["session_mean_nll"],
                        "tokens_per_second": record["tokens_per_second"]}
                       for index, record in enumerate(records)],
        "per_session": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        summary = summarise(arguments.run_dir)
    except SummaryError as error:
        print(f"summary error: {error}", file=sys.stderr)
        return 1
    text = json.dumps(summary, indent=1) + "\n"
    if arguments.output:
        arguments.output.write_text(text, encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_session"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
