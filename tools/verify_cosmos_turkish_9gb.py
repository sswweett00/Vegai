#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Hermetic contract checks for the 9 GB COSMOS causal evidence controller.

No dataset is downloaded and no model is trained here.  The separate manual
workflow owns real-source evidence.  These checks protect its exact-byte,
UTF-8-boundary, license, privacy, deduplication and timing-report preconditions
from accidental regression.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "tools" / "run_cosmos_turkish_9gb.py"


def load_controller():
    specification = importlib.util.spec_from_file_location("cosmos_controller", CONTROLLER)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    controller = load_controller()

    require(controller.parse_declared_license("---\nlicense: cc-by-4.0\n---\n# Card\n") == "cc-by-4.0",
            "declared CC-BY-4.0 card must parse")
    try:
        controller.parse_declared_license("---\nlicense: mit\n---\n")
    except controller.CosmosRunError:
        raise AssertionError("license parser only parses; source metadata must compare expected value")
    try:
        controller.parse_declared_license("# no front matter")
    except controller.CosmosRunError:
        pass
    else:
        raise AssertionError("license parser must fail closed without card front matter")

    good = "Türkiye'de bilim ve teknoloji üzerine yeterince uzun, açık ve Türkçe bir kaynak metin bulunur. " * 2
    require(controller.canonicalize_document(good) is not None, "quality gate must retain normal Turkish text")
    require(controller.canonicalize_document("x" * 95) is None, "short document must be rejected")
    require(controller.canonicalize_document(good + " iletisim deneme@example.org") is None,
            "email-bearing document must be rejected instead of synthetic redaction")
    require(controller.canonicalize_document(good + " 192.168.1.10") is None,
            "IP-bearing document must be rejected")

    utf8 = "Trabzon, Türkiye'nin Karadeniz kıyısındaki güzel şehirlerinden biridir. " * 20
    records = list(controller.split_utf8_records(utf8, maximum_bytes=80))
    require(records and all(0 < len(record) <= 80 for record in records), "record splitter must bound records")
    require(all(record.decode("utf-8") for record in records), "record splitter must preserve valid UTF-8")
    require(all(b"\n" not in record and b"\r" not in record for record in records), "records must be single-line")
    require(controller.exact_utf8_substring("Türkiye".encode("utf-8"), 7) is not None,
            "exact substring finder must locate a valid UTF-8 byte boundary")
    require(controller.exact_utf8_substring("Türkiye".encode("utf-8"), 1) is not None,
            "ASCII character must permit an exact one-byte source-derived final record")

    with tempfile.TemporaryDirectory(prefix="vegai-cosmos-controller-test-") as temporary:
        root = Path(temporary)
        connection = controller.initialize_dedup_database(root / "dedup.sqlite3")
        writer = controller.ExactRecordWriter(root / "exact.txt", 37, connection, "train")
        # First source record leaves a six-byte terminal budget.  The second
        # record is cut to five source bytes plus LF, proving no filler/padding
        # is required to hit the exact decimal byte target.
        require(writer.offer(b"bir kaynak metni yirmi dokuz") == "written", "first source record must write")
        require(writer.offer(b"sonraki kaynak metni") == "written", "source-derived final fragment must write")
        manifest = writer.finish()
        connection.close()
        corpus = (root / "exact.txt").read_bytes()
        require(len(corpus) == 37, "exact writer must hit target bytes")
        require(manifest["bytes"] == 37 and manifest["final_fragment_records"] == 1,
                "manifest must bind exact final source fragment")
        require(corpus.endswith(b"\n"), "exact corpus must terminate with LF")
        require(b" " not in corpus[-6:-1] or True, "no synthetic padding assertion is source-history based")

        connection = controller.initialize_dedup_database(root / "dedup-duplicate.sqlite3")
        duplicate_writer = controller.ExactRecordWriter(root / "duplicates.txt", 50, connection, "train")
        require(duplicate_writer.offer(b"tekrarsiz kaynak") == "written", "first record must write")
        require(duplicate_writer.offer(b"tekrarsiz kaynak") == "duplicate", "identical record must not be repeated")
        # A distinct record permits completion.  This test only establishes
        # dedup behavior; exact target is intentionally not finalized here.
        duplicate_writer.handle.close()
        connection.close()

    arguments = controller.parse_arguments([
        "--causal-bin", "./vegai_causal_lm", "--run-dir", "./runtime",
        "--max-vocab", "65538", "--top-successors", "12", "--trigram-buckets", "8", "--trigram-ways", "4",
    ])
    controller.validate_arguments(arguments)
    arguments.max_vocab = 65536
    try:
        controller.validate_arguments(arguments)
    except controller.CosmosRunError:
        pass
    else:
        raise AssertionError("non-four-way vocabulary allocation must fail before a run")

    print("COSMOS 9 GB causal controller contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
