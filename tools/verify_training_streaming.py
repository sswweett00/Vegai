#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stdlib-only regression tests for bounded JSON streaming and record validation."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

from training_data_io import (
    DataReadError,
    ReadStats,
    SchemaValidationError,
    TrainingRecord,
    _iter_json_array_values,
    iter_records,
    write_records,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_schema_error(**fields: object) -> None:
    try:
        TrainingRecord(**fields)  # type: ignore[arg-type]
    except SchemaValidationError:
        return
    raise AssertionError(f"invalid record fields accepted: {fields!r}")


def main() -> int:
    expect_schema_error(text="\ud800")
    expect_schema_error(text="valid", id="\ud800")
    expect_schema_error(text="valid", split=[])
    expect_schema_error(text="valid", metadata={"invalid": "\ud800"})
    recursive: dict[str, object] = {}
    recursive["self"] = recursive
    expect_schema_error(text="valid", metadata=recursive)
    deep: object = 0
    for _ in range(1_500):
        deep = [deep]
    expect_schema_error(text="valid", metadata={"nested": deep})

    rng = random.Random(42)

    def random_value(depth: int) -> object:
        if depth and rng.random() < 0.5:
            if rng.random() < 0.5:
                return [random_value(depth - 1) for _ in range(rng.randrange(4))]
            return {f"{i}{rng.choice(['', ']', 'é', chr(34), chr(92)])}": random_value(depth - 1)
                    for i in range(rng.randrange(4))}
        return rng.choice([None, True, False, 0, -123, 1e-20, "", "türkçe", '\\"[]{},', "\n\t"])

    with tempfile.TemporaryDirectory(prefix="vegai-stream-tests-") as temporary:
        root = Path(temporary)
        source = root / "records.json"
        # Vary boundaries down to a single character, including escapes and
        # exponents split across chunks. Compare against Python's JSON decoder.
        for _ in range(150):
            values = [random_value(3) for _ in range(rng.randrange(10))]
            source.write_text(json.dumps(values, ensure_ascii=rng.choice([True, False]),
                                         indent=rng.choice([None, 2])), encoding="utf-8")
            for chunk in (1, 2, 3, 7, 16, 63):
                actual = [value for _, value in _iter_json_array_values(source, "none", chunk, 100_000)]
                require(actual == values, f"JSON parser differed at chunk size {chunk}")
        for raw in ('[{"text":"ok"},]', '[1e,2]', '[1 2]', '[{} {}]', '[{]',
                    '[{"text":"ok"} extra]', '["unterminated]', '[] trailing', '[{}', '[,]'):
            source.write_text(raw, encoding="utf-8")
            for chunk in (1, 2, 7, 64):
                try:
                    list(_iter_json_array_values(source, "none", chunk, 100_000))
                except DataReadError:
                    continue
                raise AssertionError(f"malformed JSON accepted: {raw!r} (chunk {chunk})")

        # A large record must be decoded at most twice: one initial fast-path
        # attempt and one after its boundary. This verifies linear scanning
        # without a flaky wall-clock performance threshold.
        text = ("abc ç \\\" [] {} " * 80_000)
        source.write_text(json.dumps([{"text": text}], ensure_ascii=False), encoding="utf-8")
        decoder_class = json.JSONDecoder
        calls = [0]

        class CountingDecoder(decoder_class):
            def raw_decode(self, *args: object, **kwargs: object):
                calls[0] += 1
                return super().raw_decode(*args, **kwargs)

        with patch("training_data_io.json.JSONDecoder", CountingDecoder):
            values = list(_iter_json_array_values(source, "none", 1024, 8_000_000))
        require(values[0][1]["text"] == text and calls[0] <= 2,
                f"spanning record was rescanned/decoded {calls[0]} times")
        compressed = root / "records.json.gz"
        with gzip.open(compressed, "wt", encoding="utf-8") as handle:
            json.dump([{"text": text}], handle, ensure_ascii=False)
        require(list(iter_records(compressed))[0].text == text, "compressed spanning record changed")

        # Limits count physical UTF-8 bytes, not decoded character count; the
        # boundary is identical whether the record fits in one chunk or not.
        element = '{"text":"çç"}'
        encoded_bytes = len(element.encode("utf-8"))
        source.write_text(f"[{element}]", encoding="utf-8")
        for chunk in (1, 1024):
            require(len(list(_iter_json_array_values(source, "none", chunk, encoded_bytes))) == 1,
                    "exact JSON byte boundary rejected")
            try:
                list(_iter_json_array_values(source, "none", chunk, encoded_bytes - 1))
            except DataReadError:
                pass
            else:
                raise AssertionError("oversized UTF-8 JSON record accepted")
        source.write_text('[{"text":"a"}' + (' ' * 100_000) + ',{"text":"b"}]', encoding="utf-8")
        require([record.text for record in iter_records(source, max_record_bytes=20)] == ["a", "b"],
                "large inter-record whitespace was retained or counted as a record")

        # A frozen TrainingRecord still contains mutable metadata. Corrupting
        # it after construction must not publish a NaN or replace prior data.
        for format_name in ("jsonl", "json"):
            destination = root / f"atomic.{format_name}"
            destination.write_bytes(b"previous contents")
            changed = TrainingRecord(text="ok", metadata={"value": 1})
            assert changed.metadata is not None
            changed.metadata["value"] = float("nan")
            try:
                write_records(destination, [changed], format=format_name)
            except SchemaValidationError:
                pass
            else:
                raise AssertionError(f"{format_name} writer published mutated invalid metadata")
            require(destination.read_bytes() == b"previous contents", "failed write replaced existing data")

        # JSONL corruption can be localized. Deep metadata and invalid Unicode
        # must be diagnosed as bad records, not leak a raw recursion/encoding
        # exception or prevent collect mode from reading the next record.
        jsonl = root / "bad.jsonl"
        jsonl.write_text('{"text":"x","metadata":{"deep":' + '[' * 1500 + '0' + ']' * 1500 + '}}\n'
                         '{"text":"\\ud800"}\n{"text":"ok"}\n', encoding="utf-8")
        issues = []
        stats = ReadStats()
        recovered = list(iter_records(jsonl, error_policy="collect", issues=issues, stats=stats))
        require([record.text for record in recovered] == ["ok"] and stats.issues == 2,
                "collect did not recover after deep/invalid Unicode JSONL")
        require([issue.record_number for issue in issues] == [1, 2], "JSONL issue positions changed")
        source.write_text('[{"text":"ok","metadata":{"deep":' + '[' * 1500 + '0' + ']' * 1500 + '}}]',
                          encoding="utf-8")
        try:
            list(iter_records(source))
        except DataReadError:
            pass
        else:
            raise AssertionError("over-nested JSON did not fail closed")

    print("training streaming verification: PASS (linear cross-chunk parsing, UTF-8 bounds, corruption and schema guards)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, DataReadError) as error:
        print(f"training streaming verification: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
