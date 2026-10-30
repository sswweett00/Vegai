#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Round-trip and failure-path verification for Vegai training-data I/O.

The test intentionally exercises real JSONL/JSON/TXT, pyarrow Parquet/Arrow,
compression, malformed records, rich-to-HDC projection, disk-backed exact
quality accounting, bounded batches, and production-foundry integration from a
JSONL source.  It is standalone rather than tied to a test framework so it can
run in the repository's existing CTest/CI style.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from training_data_io import (
    DataReadError,
    ExactTextDeduplicator,
    ReadStats,
    TrainingDataError,
    TrainingRecord,
    arrow_record_schema,
    infer_format,
    iter_record_batches,
    iter_records,
    iter_records_many,
    split_for_text_digest,
    validate_hdc_record,
    write_hdc_corpus,
    write_records,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DATASET = ROOT / "tools" / "production_dataset.py"
PRODUCTION_PRETRAIN = ROOT / "tools" / "production_pretrain.py"
TRAINING_DATASET = ROOT / "tools" / "training_dataset.py"
EXAMPLE_RICH_DATASET = ROOT / "examples" / "training_data" / "mini_turkish_ascii_records.jsonl"


def fail(message: str) -> None:
    raise AssertionError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def rich_records() -> list[TrainingRecord]:
    return [
        TrainingRecord(
            id="r1", text="alpha beta", label="education", split="train", source="fixture/v1",
            metadata={"rank": 1, "tags": ["alpha", "beta"]},
        ),
        TrainingRecord(
            id="r2", text="gamma delta", label="science", split="validation", source="fixture/v1",
            metadata={"active": True, "score": 0.5},
        ),
        TrainingRecord(
            id="r3", text="epsilon zeta", label="science", split="test", metadata={"nested": {"a": "b"}},
        ),
    ]


def require_round_trip(path: Path, records: list[TrainingRecord], *, format: str, compression: str) -> None:
    report = write_records(path, records, format=format, compression=compression, batch_size=2)
    require(report.records == len(records), f"{format} writer record count is wrong")
    stats = ReadStats()
    restored = list(iter_records(path, format=format, batch_size=2, stats=stats))
    require(restored == records, f"{format} rich round trip changed schema fields")
    require(stats.records_yielded == len(records), f"{format} streaming reader count is wrong")


def main() -> int:
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        print(
            "usage: verify_training_data_io.py\n\n"
            "Round-trip and failure-path verification for Vegai training-data I/O.\n"
            "Parquet/Arrow coverage requires the optional pyarrow dependency;\n"
            "install it with: python3 -m pip install -r requirements-dataio.txt"
        )
        return 0
    try:
        # Confirm this test fails clearly rather than falsely claiming columnar
        # support when a caller omitted the declared optional dependency.
        import pyarrow as pa
        import pyarrow.ipc as ipc

        with tempfile.TemporaryDirectory(prefix="vegai-training-data-io-") as temporary:
            workspace = Path(temporary)
            records = rich_records()

            # Source-controlled rich example is intentionally synthetic,
            # schema-valid, and already compatible with strict native HDC text
            # projection. It provides a small operator-facing pretraining-data
            # fixture without shipping governed real-world corpus material.
            example_records = list(iter_records(EXAMPLE_RICH_DATASET, format="jsonl"))
            require(len(example_records) == 16, "rich example fixture record count changed")
            require({record.label for record in example_records} == {"education", "technology", "health", "environment"},
                    "rich example fixture labels changed")
            for record in example_records:
                validate_hdc_record(record)

            require_round_trip(workspace / "records.jsonl.gz", records, format="jsonl", compression="gzip")
            gzip_again = workspace / "records-again.jsonl.gz"
            write_records(gzip_again, records, format="jsonl", compression="gzip")
            require(
                (workspace / "records.jsonl.gz").read_bytes() == gzip_again.read_bytes(),
                "deterministic gzip writer leaked time or temporary-file metadata",
            )
            require_round_trip(workspace / "records.json", records, format="json", compression="none")
            require_round_trip(workspace / "records.parquet", records, format="parquet", compression="zstd")
            require_round_trip(workspace / "records.arrow", records, format="arrow", compression="zstd")

            # TXT remains the intentionally narrow native-HDC text projection.
            text_records = [TrainingRecord(text="one two"), TrainingRecord(text="three four")]
            text_path = workspace / "records.txt"
            write_records(text_path, text_records, format="txt")
            require(list(iter_records(text_path, format="txt")) == text_records, "TXT text-only round trip failed")
            try:
                write_records(workspace / "lossy.txt", records, format="txt")
            except Exception as error:
                require("text-only" in str(error), "TXT rich-field rejection was not explanatory")
            else:
                fail("TXT writer silently discarded rich fields")

            # Conversion through independent physical encodings must preserve
            # the logical record schema and canonical metadata JSON.
            parquet_path = workspace / "converted.parquet"
            write_records(parquet_path, iter_records(workspace / "records.json", format="json"), format="parquet", compression="snappy")
            arrow_path = workspace / "converted.arrow"
            write_records(arrow_path, iter_records(parquet_path, format="parquet"), format="arrow", compression="zstd")
            require(list(iter_records(arrow_path, format="arrow")) == records, "JSON -> Parquet -> Arrow round trip changed records")
            # Accept both Arrow IPC file and stream containers, while retaining
            # the same explicit schema rather than positional column decoding.
            stream_arrow = workspace / "records-stream.arrow"
            stream_rows = [
                {
                    "schema_version": record.schema_version,
                    "id": record.id,
                    "text": record.text,
                    "label": record.label,
                    "split": record.split,
                    "source": record.source,
                    "metadata_json": (
                        json.dumps(record.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        if record.metadata is not None else None
                    ),
                }
                for record in records
            ]
            with pa.OSFile(str(stream_arrow), "wb") as sink:
                with ipc.new_stream(sink, arrow_record_schema()) as stream_writer:
                    stream_writer.write_batch(pa.RecordBatch.from_pylist(stream_rows, schema=arrow_record_schema()))
            require(list(iter_records(stream_arrow, format="arrow", batch_size=1)) == records,
                    "Arrow IPC stream reader changed rich records")
            for limited_path, limited_format in ((parquet_path, "parquet"), (arrow_path, "arrow")):
                try:
                    list(iter_records(limited_path, format=limited_format, max_record_bytes=4))
                except DataReadError:
                    pass
                else:
                    fail(f"{limited_format} reader ignored configured record byte limit")

            # The incremental JSON reader must handle a deliberately tiny
            # source array and reject trailing syntax instead of json.load-like
            # best effort behavior.
            chunks_path = workspace / "streaming.json"
            chunks_path.write_text(
                "[\n" + ",\n".join(json.dumps(record.to_mapping()) for record in records) + "\n]\n", encoding="utf-8"
            )
            batches = list(iter_record_batches(iter_records(chunks_path, format="json"), batch_size=2))
            require([len(batch) for batch in batches] == [2, 1], "bounded batch iterator did not preserve batch limits")
            try:
                list(iter_records(chunks_path, format="json", batch_size=0))
            except ValueError:
                pass
            else:
                fail("reader accepted nonpositive batch size")
            malformed_json = workspace / "bad.json"
            malformed_json.write_text('[{"text":"alpha"},]', encoding="utf-8")
            try:
                list(iter_records(malformed_json, format="json"))
            except DataReadError:
                pass
            else:
                fail("trailing-comma JSON was accepted")

            # Malformed JSONL is surfaced with a line issue when lossiness is
            # explicit, and remains fail-closed by default.
            malformed_jsonl = workspace / "bad.jsonl"
            malformed_jsonl.write_text(
                '{"text":"valid"}\n{not json}\n{"text":"valid-two","unknown":1}\n{}\n', encoding="utf-8"
            )
            issues = []
            valid = list(iter_records(malformed_jsonl, format="jsonl", error_policy="collect", issues=issues))
            require([record.text for record in valid] == ["valid"], "collect policy yielded malformed JSONL records")
            require(len(issues) == 3 and {issue.record_number for issue in issues} == {2, 3, 4},
                    "JSONL corruption/missing-record issues lost source positions")
            try:
                list(iter_records(malformed_jsonl, format="jsonl"))
            except DataReadError:
                pass
            else:
                fail("default JSONL policy was not fail-closed")

            # Physical JSONL/TXT lines are bounded before decode. An oversized
            # line is drained, reported, and does not prevent explicit collect
            # mode from reaching the following record.
            oversized_jsonl = workspace / "oversized.jsonl"
            oversized_jsonl.write_text('{"text":"' + ("x" * 256) + '"}\n{"text":"ok"}\n', encoding="utf-8")
            oversized_issues = []
            recovered = list(
                iter_records(
                    oversized_jsonl, format="jsonl", max_record_bytes=32,
                    error_policy="collect", issues=oversized_issues,
                )
            )
            require([record.text for record in recovered] == ["ok"], "bounded JSONL reader did not recover after oversized line")
            require(len(oversized_issues) == 1 and oversized_issues[0].record_number == 1, "oversized JSONL line issue was lost")
            crlf_txt = workspace / "crlf.txt"
            crlf_txt.write_bytes(b"one two\r\n")
            try:
                list(iter_records(crlf_txt, format="txt"))
            except DataReadError:
                pass
            else:
                fail("TXT reader silently normalized CRLF")
            corrupt_gzip = workspace / "corrupt.jsonl.gz"
            write_records(corrupt_gzip, [TrainingRecord(text="compressed")], format="jsonl", compression="gzip")
            corrupt_gzip.write_bytes(corrupt_gzip.read_bytes()[:-4])
            try:
                list(iter_records(corrupt_gzip, format="jsonl"))
            except DataReadError:
                pass
            else:
                fail("truncated gzip JSONL was accepted")

            # A logical JSON element may span source chunks, while an explicit
            # configured record limit is still enforced after incremental parse.
            chunked_json = workspace / "chunked.json"
            chunked_json.write_text('[{"text":"' + ("a" * 70_000) + '"}]', encoding="utf-8")
            require(len(list(iter_records(chunked_json, format="json"))) == 1, "incremental JSON reader lost a cross-chunk record")
            try:
                list(iter_records(chunked_json, format="json", max_record_bytes=1024))
            except DataReadError:
                pass
            else:
                fail("configured JSON record byte limit was not enforced")
            many_json = workspace / "many.json"
            write_records(many_json, (TrainingRecord(text=f"row {index}") for index in range(5_000)), format="json")
            require(sum(1 for _ in iter_records(many_json, format="json")) == 5_000,
                    "incremental JSON reader lost records across multiple source chunks")

            # Exact text identity is disk-backed and flags conflicting labels;
            # deterministic split assignment does not depend on input order.
            deduplicator = ExactTextDeduplicator(workspace / "quality.sqlite")
            try:
                first = TrainingRecord(text="same text", label="left")
                conflict = TrainingRecord(text="same text", label="right")
                split = split_for_text_digest(first.text_digest(), 1000, 1000)
                require(deduplicator.accept(first, split), "first exact text was not accepted")
                require(not deduplicator.accept(conflict, split), "exact duplicate was not removed")
                quality = deduplicator.report()
                require(quality.duplicate_text_records == 1 and quality.label_conflicts == 1, "label conflict was not audited")
                require(split == split_for_text_digest(first.text_digest(), 1000, 1000), "split assignment is not stable")
            finally:
                deduplicator.close()

            # The operator-facing prepare command makes cleaning/dedupe/split
            # choices explicit, writes only after an atomic directory build,
            # and retains quality evidence in dataset.json.
            prepare_source = workspace / "prepare.jsonl"
            write_records(
                prepare_source,
                [
                    TrainingRecord(text=" alpha beta ", label="left"),
                    TrainingRecord(text="alpha beta", label="left"),
                    TrainingRecord(text="gamma delta", label="right"),
                    TrainingRecord(text="", label="empty"),
                ],
                format="jsonl",
            )
            prepared_directory = workspace / "prepared"
            prepared = subprocess.run(
                [
                    sys.executable, str(TRAINING_DATASET), "prepare", str(prepare_source),
                    "--output-dir", str(prepared_directory), "--output-format", "parquet",
                    "--output-compression", "zstd", "--strip-outer-whitespace", "--drop-empty",
                    "--validation-basis-points", "0", "--test-basis-points", "0",
                ],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            require(prepared.returncode == 0, f"prepare command failed:\n{prepared.stdout}")
            prepared_manifest = json.loads((prepared_directory / "dataset.json").read_text(encoding="utf-8"))
            prepared_quality = prepared_manifest["quality"]
            require(
                prepared_manifest["output_compression"] == "zstd"
                and prepared_manifest["files"]["train"]["records"] == 2
                and prepared_quality["unique_text_records"] == 2
                and prepared_quality["duplicate_text_records"] == 1
                and prepared_manifest["cleaning"]["empty_records_dropped"] == 1,
                "prepare quality/dedupe/cleaning manifest is inconsistent",
            )
            conflict_source = workspace / "conflict.jsonl"
            write_records(
                conflict_source,
                [TrainingRecord(text="same", label="left"), TrainingRecord(text="same", label="right")],
                format="jsonl",
            )
            conflict_directory = workspace / "conflict-output"
            conflict = subprocess.run(
                [
                    sys.executable, str(TRAINING_DATASET), "prepare", str(conflict_source),
                    "--output-dir", str(conflict_directory), "--reject-label-conflicts",
                    "--validation-basis-points", "0", "--test-basis-points", "0",
                ],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            require(
                conflict.returncode != 0 and not conflict_directory.exists() and "conflicting labels" in conflict.stdout,
                "prepare did not fail atomically on an exact duplicate label conflict",
            )

            # The native corpus adapter keeps labels out of HDC text and rejects
            # Unicode before native Fortran can silently tokenize bytes away.
            hdc_path = workspace / "corpus.txt"
            hdc_report = write_hdc_corpus(hdc_path, records, strict=True)
            require(hdc_report.records == len(records), "HDC text adapter count is wrong")
            require(hdc_path.read_text(encoding="ascii") == "alpha beta\ngamma delta\nepsilon zeta\n", "HDC adapter leaked metadata/labels")
            try:
                write_hdc_corpus(workspace / "bad-corpus.txt", [TrainingRecord(text="istanbul çay")], strict=True)
            except Exception as error:
                require("ASCII" in str(error), "HDC Unicode rejection was not explanatory")
            else:
                fail("HDC export accepted non-ASCII text")

            # Multiple independent files may be streamed in parallel. Arrival
            # order is not part of the contract, but no record may disappear.
            left = workspace / "left.jsonl"
            right = workspace / "right.jsonl"
            write_records(left, [TrainingRecord(text="left a"), TrainingRecord(text="left b")], format="jsonl")
            write_records(right, [TrainingRecord(text="right a"), TrainingRecord(text="right b")], format="jsonl")
            parallel_texts = {record.text for record in iter_records_many([left, right], workers=2)}
            require(parallel_texts == {"left a", "left b", "right a", "right b"}, "bounded parallel reader lost records")

            # A wrong Arrow schema is a data-contract error rather than an
            # accidental positional decode.
            bad_schema_path = workspace / "wrong-schema.arrow"
            wrong = pa.table({"text": ["only field"]})
            with pa.OSFile(str(bad_schema_path), "wb") as sink:
                import pyarrow.ipc as ipc

                with ipc.new_file(sink, wrong.schema) as writer:
                    writer.write_table(wrong)
            try:
                list(iter_records(bad_schema_path, format="arrow"))
            except DataReadError:
                pass
            else:
                fail("Arrow reader accepted a wrong schema")

            # The production foundry now accepts structured JSONL through the
            # adapter, still emitting only strict text shards and binding raw
            # source identity in dataset.json.
            production_source = workspace / "production.jsonl"
            write_records(production_source, [TrainingRecord(text="alpha beta"), TrainingRecord(text="gamma delta")], format="jsonl")
            production_output = workspace / "production-dataset"
            command = [
                sys.executable, str(PRODUCTION_DATASET), "--input", str(production_source), "--input-format", "jsonl",
                "--output-dir", str(production_output), "--source", "data-io verifier", "--license", "MIT",
                "--retrieved-at", "source-controlled verifier", "--preprocessing", "strict ASCII fixture/v1",
                "--validation-basis-points", "0", "--max-shard-bytes", "8192", "--max-shard-tokens", "8",
            ]
            completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            require(completed.returncode == 0, f"structured production adapter failed:\n{completed.stdout}")
            production_manifest = json.loads((production_output / "dataset.json").read_text(encoding="utf-8"))
            adapter = production_manifest.get("input_adapter")
            require(
                isinstance(adapter, dict)
                and adapter.get("source_format") == "jsonl"
                and adapter.get("source_sha256") == hashlib.sha256(production_source.read_bytes()).hexdigest(),
                "production manifest did not bind structured raw-source provenance",
            )
            shard = next((production_output / "train" / "shards").glob("*.txt"))
            require(shard.read_bytes() in {b"alpha beta\ngamma delta\n", b"gamma delta\nalpha beta\n"}, "production adapter emitted non-canonical HDC shard")

            # The production controller accepts the valid raw-to-canonical
            # binding during preflight and rejects a tampered binding before a
            # training child can start. sys.executable is sufficient because
            # --preflight-only never launches the selected binary.
            preflight = subprocess.run(
                [
                    sys.executable, str(PRODUCTION_PRETRAIN), "--pretrain-bin", sys.executable,
                    "--dataset", str(production_output / "dataset.json"), "--run-dir", str(workspace / "preflight-run"),
                    "--max-vocab", "64", "--preflight-only",
                ],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            require(preflight.returncode == 0, f"production adapter preflight rejected valid binding:\n{preflight.stdout}")
            production_manifest["input_adapter"]["canonical_hdc_text_sha256"] = "0" * 64
            (production_output / "dataset.json").write_text(json.dumps(production_manifest), encoding="utf-8")
            tampered = subprocess.run(
                [
                    sys.executable, str(PRODUCTION_PRETRAIN), "--pretrain-bin", sys.executable,
                    "--dataset", str(production_output / "dataset.json"), "--run-dir", str(workspace / "tampered-run"),
                    "--max-vocab", "64", "--preflight-only",
                ],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            require(tampered.returncode != 0 and "input_adapter" in tampered.stdout, "tampered adapter binding was accepted")

            require(infer_format(Path("rows.ndjson")) == "jsonl", "format auto-detection missed ndjson")
            require(infer_format(Path("rows.parquet")) == "parquet", "format auto-detection missed Parquet")
    except ImportError as error:
        # A missing optional columnar dependency must be reported as a clear,
        # actionable contract failure instead of a bare traceback.
        print(
            f"training data I/O verification: FAIL: optional columnar dependency is missing ({error}); "
            "install it with: python3 -m pip install -r requirements-dataio.txt",
            file=sys.stderr,
        )
        return 1
    except (AssertionError, OSError, TrainingDataError, ValueError, json.JSONDecodeError) as error:
        print(f"training data I/O verification: FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "training data I/O verification: PASS "
        "(streaming JSONL/JSON/TXT/Parquet/Arrow round trips, schema/corruption guards, "
        "compression, exact quality audit, bounded parallel reads, strict HDC export, structured production adapter/binding)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
