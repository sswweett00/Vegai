#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a provenance-bound 99M-token Turkish Wikipedia HDC evidence experiment.

The large source is a versioned official Turkish Wikipedia pages-articles dump.
The workflow retrieves and extracts it with a pinned WikiExtractor release; this
controller never retrieves source data itself.  It performs three deliberately
separate operations:

* deterministic article-level train/held-out partitioning before vocabulary
  selection or model updates;
* strict native ASCII corpus construction with exactly 99,000,000 emitted
  tokens and no one-token contextual records; and
* real native Fortran checkpoint inference, both as semantic nearest-neighbour
  inspection and as a held-out-page co-occurrence consistency measurement.

The held-out measurement is not presented as a general Turkish NLP benchmark.
It measures agreement between native checkpoint inference and independently
held-out article-local associations from the same licensed dump.  Grounded FAQ
responses are additionally retrieved verbatim by the native Fortran chat
executable from a separately versioned source; they are not generated prose.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

from run_turkish_wikipedia_9m import (
    MAX_NATIVE_RECORD_BYTES,
    STOP_TOKENS,
    TOKEN_PATTERN,
    TOKENS_PER_RECORD,
    ExperimentError,
    atomic_json,
    atomic_text,
    copy_evidence_files,
    emit_segment,
    git_revision,
    license_identity,
    load_faq_candidates,
    normalize_display_text,
    normalize_tokens,
    parse_active_vocabulary,
    parse_chat_result,
    parse_semantic_results,
    parse_training_metrics,
    require,
    require_success,
    run_child,
    select_semantic_queries,
    sha256_file,
    validate_generated_corpus,
    write_process_transcript,
)


SCHEMA = "vegai-turkish-wikipedia-99m-experiment/v1"
CORPUS_SCHEMA = "vegai-turkish-wikipedia-99m-corpus/v1"
HELDOUT_SCHEMA = "vegai-turkish-wikipedia-heldout-cooccurrence/v1"
TARGET_TOKENS = 99_000_000
VOCABULARY_SIZE_DEFAULT = 32_768
EPOCHS_DEFAULT = 2
CONTEXT_WINDOW_DEFAULT = 2
OMP_THREADS_DEFAULT = 4
HOLDOUT_MODULUS = 100
HOLDOUT_BUCKET = 0
QUALITY_QUERY_COUNT = 128
QUALITY_MIN_QUERY_OCCURRENCES = 12
QUALITY_MIN_ASSOCIATION_COUNT = 3
MAX_EXTRACTED_LINE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class WikiDocument:
    """One WikiExtractor JSON article, retained only while it is streamed."""

    page_id: str
    title: str
    text: str
    source_path: str
    source_line: int

    @property
    def rendered_text(self) -> str:
        # Title is an actual source field, not a controller-invented prompt.
        # It remains a page boundary and therefore never forms context across
        # documents.
        return self.title + "\n" + self.text


def file_sha256_text(path: Path) -> str:
    return str(sha256_file(path)["sha256"])


def require_regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    require(resolved.is_file(), f"{label} is not a regular file: {resolved}")
    return resolved


def json_load_strict(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExperimentError(f"cannot load {label} {path}: {error}") from error
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def extracted_files(root: Path) -> list[Path]:
    """Return WikiExtractor document files, excluding logs and manifests."""
    resolved = root.resolve()
    require(resolved.is_dir(), f"WikiExtractor output root is unavailable: {resolved}")
    files = sorted(path for path in resolved.rglob("wiki_*") if path.is_file())
    require(files, f"WikiExtractor output has no wiki_* document files under {resolved}")
    return files


def iter_wiki_documents(root: Path) -> Iterator[WikiDocument]:
    """Strictly stream the pinned WikiExtractor JSONL output."""
    for document_file in extracted_files(root):
        try:
            with document_file.open("rb") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    require(raw.endswith(b"\n"),
                            f"WikiExtractor document {document_file}:{line_number} is not LF terminated")
                    require(len(raw) <= MAX_EXTRACTED_LINE_BYTES,
                            f"WikiExtractor document {document_file}:{line_number} exceeds the bounded JSONL record size")
                    require(b"\r" not in raw[:-1],
                            f"WikiExtractor document {document_file}:{line_number} contains CR")
                    try:
                        payload = json.loads(raw.decode("utf-8", errors="strict"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ExperimentError(
                            f"WikiExtractor document {document_file}:{line_number} is not valid UTF-8 JSON: {error}"
                        ) from error
                    require(isinstance(payload, dict),
                            f"WikiExtractor document {document_file}:{line_number} is not an object")
                    identifier = payload.get("id")
                    title = payload.get("title")
                    text = payload.get("text")
                    require(isinstance(identifier, (str, int)) and str(identifier),
                            f"WikiExtractor document {document_file}:{line_number} has no page id")
                    require(isinstance(title, str) and title.strip(),
                            f"WikiExtractor document {document_file}:{line_number} has no title")
                    require(isinstance(text, str) and text.strip(),
                            f"WikiExtractor document {document_file}:{line_number} has no text")
                    require("\x00" not in title and "\x00" not in text,
                            f"WikiExtractor document {document_file}:{line_number} contains NUL")
                    yield WikiDocument(
                        page_id=str(identifier),
                        title=title,
                        text=text,
                        source_path=str(document_file.relative_to(root)),
                        source_line=line_number,
                    )
        except OSError as error:
            raise ExperimentError(f"cannot read WikiExtractor document {document_file}: {error}") from error


def is_heldout_document(document: WikiDocument) -> bool:
    """Stable article-level partition independent of document-file ordering."""
    identity = ("vegai-trwiki-page-partition-v1\0" + document.page_id + "\0" + document.title).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % HOLDOUT_MODULUS
    return bucket == HOLDOUT_BUCKET


def normalized_document_tokens(document: WikiDocument) -> list[str]:
    return normalize_tokens(document.rendered_text)


def scan_training_source(root: Path) -> tuple[Counter[str], dict[str, Any]]:
    """Frequency-scan only train-partition articles, never held-out articles."""
    frequencies: Counter[str] = Counter()
    all_articles = 0
    train_articles = 0
    heldout_articles = 0
    train_tokens = 0
    heldout_tokens = 0
    for document in iter_wiki_documents(root):
        all_articles += 1
        tokens = normalized_document_tokens(document)
        if is_heldout_document(document):
            heldout_articles += 1
            heldout_tokens += len(tokens)
            continue
        train_articles += 1
        train_tokens += len(tokens)
        frequencies.update(tokens)
    require(all_articles > 0 and train_articles > 0 and heldout_articles > 0,
            "article partition did not yield nonempty all/train/held-out sets")
    require(train_tokens > 0 and heldout_tokens > 0 and frequencies,
            "WikiExtractor source yielded no normalized train or held-out tokens")
    return frequencies, {
        "partition": {
            "id": "sha256-page-id-title-modulo-v1",
            "holdout_modulus": HOLDOUT_MODULUS,
            "heldout_bucket": HOLDOUT_BUCKET,
            "no_page_id_or_title_from_heldout_bucket_is_used_for_vocabulary_or_model_updates": True,
        },
        "all_articles": all_articles,
        "train_articles": train_articles,
        "heldout_articles": heldout_articles,
        "train_normalized_token_occurrences": train_tokens,
        "heldout_normalized_token_occurrences": heldout_tokens,
        "train_normalized_unique_tokens": len(frequencies),
    }


def selected_vocabulary(frequencies: Counter[str], vocabulary_size: int) -> list[tuple[str, int]]:
    require(vocabulary_size > 128, "99M native evidence requires a vocabulary larger than 128 tokens")
    require(len(frequencies) >= vocabulary_size,
            f"train partition has only {len(frequencies)} normalized unique tokens, below requested vocabulary {vocabulary_size}")
    return sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:vocabulary_size]


def emit_selected_document(
    output: Any,
    tokens: Sequence[str],
    selected: frozenset[str],
    remaining: int,
) -> tuple[int, int, int]:
    """Emit bounded contextual records without crossing filtered/page boundaries."""
    emitted = 0
    records = 0
    selected_seen = 0
    segment: list[str] = []
    for token in tokens:
        if token in selected:
            segment.append(token)
            selected_seen += 1
            continue
        if segment:
            written, made = emit_segment(output, segment, remaining - emitted)
            emitted += written
            records += made
            segment.clear()
            if emitted == remaining:
                return emitted, records, selected_seen
    if segment and emitted < remaining:
        written, made = emit_segment(output, segment, remaining - emitted)
        emitted += written
        records += made
    return emitted, records, selected_seen


def prepare_corpus(
    extracted_root: Path,
    output: Path,
    selected_vocabulary_output: Path,
    *,
    target_tokens: int,
    vocabulary_size: int,
) -> dict[str, Any]:
    require(target_tokens == TARGET_TOKENS,
            f"this evidence controller only permits exactly {TARGET_TOKENS:,} target tokens")
    require(target_tokens % 2 == 0, "target tokens must be even so every emitted token has a context neighbour")
    frequencies, source_stats = scan_training_source(extracted_root)
    ranked = selected_vocabulary(frequencies, vocabulary_size)
    selected = frozenset(token for token, _ in ranked)
    require(len(selected) == vocabulary_size, "frequency selection did not produce a unique bounded vocabulary")

    output.parent.mkdir(parents=True, exist_ok=True)
    selected_vocabulary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    emitted_tokens = 0
    emitted_records = 0
    train_articles_consumed = 0
    train_normalized_tokens_considered = 0
    selected_tokens_seen = 0
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as handle:
            for document in iter_wiki_documents(extracted_root):
                if is_heldout_document(document):
                    continue
                train_articles_consumed += 1
                tokens = normalized_document_tokens(document)
                train_normalized_tokens_considered += len(tokens)
                written, made, seen = emit_selected_document(handle, tokens, selected, target_tokens - emitted_tokens)
                emitted_tokens += written
                emitted_records += made
                selected_tokens_seen += seen
                if emitted_tokens == target_tokens:
                    break
    except OSError as error:
        raise ExperimentError(f"cannot write normalized 99M corpus {output}: {error}") from error
    finally:
        if emitted_tokens != target_tokens and temporary.exists():
            temporary.unlink(missing_ok=True)
    require(emitted_tokens == target_tokens,
            f"selected real-source train tokens ended at {emitted_tokens:,}, below requested {target_tokens:,}; "
            "do not repeat source material to fill the deficit")
    os.replace(temporary, output)
    atomic_text(selected_vocabulary_output, "".join(f"{token}\t{frequency}\n" for token, frequency in ranked))
    corpus_identity = validate_generated_corpus(output, target_tokens)
    require(corpus_identity["physical_lf_records"] == emitted_records,
            "99M corpus record accounting differs from emitted output")
    return {
        "schema": CORPUS_SCHEMA,
        "source_partition": source_stats,
        "normalization": {
            "id": "tr-ascii-v1",
            "description": (
                "Turkish C/c, G/g, dotted/dotless I/i, O/o, S/s, and U/u are explicitly transliterated; "
                "Unicode is NFKD decomposed; remaining non-ASCII code points and combining marks are removed; "
                "ASCII letters are lowercase; the native token grammar retains A-Z a-z 0-9 _ apostrophe."
            ),
            "long_tokens_dropped": True,
        },
        "selection": {
            "method": "train-partition full-frequency descending then ASCII lexical tie-break",
            "selected_vocabulary_size": vocabulary_size,
            "selected_vocabulary_file": selected_vocabulary_output.name,
            "selected_vocabulary_sha256": file_sha256_text(selected_vocabulary_output),
            "filtered_token_is_record_boundary": True,
            "page_is_record_boundary": True,
            "contextless_singleton_segments_dropped": True,
            "minimum_tokens_per_output_record": 2,
            "maximum_tokens_per_output_record": TOKENS_PER_RECORD,
            "train_articles_consumed_until_target": train_articles_consumed,
            "train_normalized_tokens_considered_until_target": train_normalized_tokens_considered,
            "selected_train_tokens_seen_until_target": selected_tokens_seen,
        },
        "output": corpus_identity,
    }


def read_selected_tokens(path: Path) -> frozenset[str]:
    selected: set[str] = set()
    try:
        for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
            fields = line.split("\t")
            require(len(fields) == 2 and TOKEN_PATTERN.fullmatch(fields[0]) is not None,
                    f"selected vocabulary line {line_number} is invalid")
            require(fields[0] not in selected, f"selected vocabulary line {line_number} is duplicated")
            selected.add(fields[0])
    except (OSError, UnicodeError) as error:
        raise ExperimentError(f"cannot read selected vocabulary {path}: {error}") from error
    require(selected, "selected vocabulary is empty")
    return frozenset(selected)


def active_segments(tokens: Sequence[str], active: frozenset[str]) -> Iterable[list[str]]:
    segment: list[str] = []
    for token in tokens:
        if token in active:
            segment.append(token)
        elif segment:
            yield segment
            segment = []
    if segment:
        yield segment


def select_quality_queries(active: Mapping[str, int], count: int = QUALITY_QUERY_COUNT) -> list[str]:
    selected: list[str] = []
    for token in sorted(active, key=lambda value: (-active[value], value)):
        if token in STOP_TOKENS or len(token) < 4 or token.isdigit() or token in selected:
            continue
        selected.append(token)
        if len(selected) == count:
            return selected
    raise ExperimentError(f"fewer than {count} meaningful active tokens are available for held-out inference")


def heldout_cooccurrence_evaluation(
    extracted_root: Path,
    active: Mapping[str, int],
    semantic_results: Sequence[Mapping[str, Any]],
    *,
    context_window: int,
) -> dict[str, Any]:
    """Evaluate actual checkpoint top-1 output against held-out page associations.

    The expected association is the highest held-out conditional association
    score count(query, neighbour) / heldout_frequency(neighbour), subject to a
    minimum observed co-occurrence count.  This deliberately tests a narrow
    distributional property; it is not a semantic gold-standard benchmark.
    """
    require(context_window > 0, "held-out co-occurrence window must be positive")
    queries = [str(item["query"]) for item in semantic_results]
    require(len(queries) == len(set(queries)), "semantic inference query list has duplicates")
    prediction = {str(item["query"]): str(item["nearest_semantic_neighbour"]) for item in semantic_results}
    require(set(queries) == set(prediction), "semantic inference results are incomplete")
    active_set = frozenset(active)
    query_set = frozenset(queries)
    pair_counts: dict[str, Counter[str]] = {query: Counter() for query in queries}
    query_occurrences: Counter[str] = Counter()
    active_occurrences: Counter[str] = Counter()
    heldout_articles = 0
    heldout_normalized_tokens = 0
    heldout_active_tokens = 0
    ordered_pairs = 0

    for document in iter_wiki_documents(extracted_root):
        if not is_heldout_document(document):
            continue
        heldout_articles += 1
        tokens = normalized_document_tokens(document)
        heldout_normalized_tokens += len(tokens)
        for segment in active_segments(tokens, active_set):
            heldout_active_tokens += len(segment)
            active_occurrences.update(segment)
            for center_index, center in enumerate(segment):
                if center not in query_set:
                    continue
                query_occurrences[center] += 1
                first = max(0, center_index - context_window)
                last = min(len(segment), center_index + context_window + 1)
                for neighbour_index in range(first, last):
                    if neighbour_index == center_index:
                        continue
                    neighbour = segment[neighbour_index]
                    if neighbour == center:
                        continue
                    pair_counts[center][neighbour] += 1
                    ordered_pairs += 1

    require(heldout_articles > 0 and heldout_normalized_tokens > 0 and heldout_active_tokens > 0,
            "held-out article partition yielded no normalized active evidence")
    rows: list[dict[str, Any]] = []
    correct = 0
    observed_prediction = 0
    reciprocal_rank_sum = 0.0
    for query in queries:
        counts = pair_counts[query]
        candidates = [
            neighbour for neighbour, count in counts.items()
            if count >= QUALITY_MIN_ASSOCIATION_COUNT and active_occurrences[neighbour] > 0
        ]
        if query_occurrences[query] < QUALITY_MIN_QUERY_OCCURRENCES or not candidates:
            continue
        ranked = sorted(
            candidates,
            key=lambda neighbour: (
                -(counts[neighbour] / active_occurrences[neighbour]),
                -counts[neighbour],
                neighbour,
            ),
        )
        expected = ranked[0]
        predicted = prediction[query]
        predicted_count = counts.get(predicted, 0)
        rank = ranked.index(predicted) + 1 if predicted in ranked else None
        exact = predicted == expected
        if exact:
            correct += 1
        if predicted_count > 0:
            observed_prediction += 1
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank
        rows.append(
            {
                "query": query,
                "query_heldout_occurrences": query_occurrences[query],
                "checkpoint_nearest_semantic_neighbour": predicted,
                "checkpoint_bipolar_similarity": float(
                    next(item["bipolar_similarity"] for item in semantic_results if item["query"] == query)
                ),
                "heldout_expected_association": expected,
                "heldout_expected_pair_count": counts[expected],
                "heldout_expected_neighbour_occurrences": active_occurrences[expected],
                "checkpoint_prediction_pair_count": predicted_count,
                "checkpoint_prediction_rank_by_heldout_association": rank,
                "exact_top1_agreement": exact,
            }
        )
    require(len(rows) >= 32,
            f"held-out evaluation yielded only {len(rows)} eligible queries, below the 32-query minimum")
    evaluated = len(rows)
    return {
        "schema": HELDOUT_SCHEMA,
        "status": "measured",
        "scope": (
            "Article-level held-out Turkish Wikipedia local-co-occurrence consistency for actual native checkpoint "
            "top-1 semantic-neighbour inference. This is not a human semantic-similarity benchmark, held-out "
            "language-model accuracy, factuality evaluation, or general intelligence measurement."
        ),
        "partition": {
            "id": "sha256-page-id-title-modulo-v1",
            "holdout_modulus": HOLDOUT_MODULUS,
            "heldout_bucket": HOLDOUT_BUCKET,
            "train_and_heldout_pages_are_disjoint": True,
        },
        "association_policy": {
            "window": context_window,
            "query_selection": (
                "first deterministic non-stopword active-vocabulary tokens sorted by descending train frequency, "
                f"up to {QUALITY_QUERY_COUNT}"
            ),
            "minimum_query_heldout_occurrences": QUALITY_MIN_QUERY_OCCURRENCES,
            "minimum_pair_count": QUALITY_MIN_ASSOCIATION_COUNT,
            "ranking": "descending heldout pair_count / heldout neighbour occurrence count; then pair count; then token",
        },
        "heldout_source_statistics": {
            "articles": heldout_articles,
            "normalised_token_occurrences": heldout_normalized_tokens,
            "active_token_occurrences": heldout_active_tokens,
            "ordered_query_context_pairs": ordered_pairs,
        },
        "metrics": {
            "requested_queries": len(queries),
            "eligible_queries": evaluated,
            "exact_top1_agreement": correct / evaluated,
            "checkpoint_prediction_observed_in_heldout_window_rate": observed_prediction / evaluated,
            "mean_reciprocal_rank_by_heldout_association": reciprocal_rank_sum / evaluated,
            "exact_top1_agreement_count": correct,
            "prediction_observed_count": observed_prediction,
        },
        "queries": rows,
    }


def source_provenance(arguments: argparse.Namespace, corpus_manifest: Mapping[str, Any]) -> dict[str, Any]:
    archive = require_regular_file(arguments.source_archive, "downloaded official dump archive")
    manifest = require_regular_file(arguments.source_manifest, "downloaded dump status manifest")
    manifest_payload = json_load_strict(manifest, "downloaded dump status manifest")
    for field in ("dump_revision", "source_url", "status_url"):
        require(isinstance(manifest_payload.get(field), str) and manifest_payload[field],
                f"downloaded dump status manifest is missing {field!r}")
    faq_repo = arguments.faq_repo.resolve()
    faq_file = faq_repo / "dataset_v1.txt"
    require(faq_repo.is_dir() and faq_file.is_file(), "independent FAQ repository/data file is unavailable")
    return {
        "Turkish_Wikipedia_training_source": {
            "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_project_url": arguments.dataset_url,
            "dump_revision": manifest_payload["dump_revision"],
            "source_url": manifest_payload["source_url"],
            "status_url": manifest_payload["status_url"],
            "declared_license": arguments.dataset_license,
            "attribution": arguments.dataset_attribution,
            "dump_status_manifest": sha256_file(manifest),
            "archive": sha256_file(archive),
            "extractor": {
                "name": "WikiExtractor",
                "version": arguments.wikiextractor_version,
                "mode": "--json --no-templates --discard_empty; source pages are split before vocabulary selection",
            },
            "source_partition": corpus_manifest["source_partition"],
        },
        "separate_grounded_dialogue_source": {
            "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_project_url": arguments.faq_url,
            "dataset_project_revision": git_revision(faq_repo),
            "declared_license": arguments.faq_license,
            "license_file": license_identity(faq_repo),
            "dataset_file": sha256_file(faq_file),
            "format": "source-webpage-url|question|answer; selected answers are ASCII-normalized literal TSV fields",
        },
    }


def create_grounded_dialogue(
    arguments: argparse.Namespace,
    run_dir: Path,
    checkpoint: Path,
    active: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[Path], Path, Path]:
    faq_file = arguments.faq_repo.resolve() / "dataset_v1.txt"
    entries = load_faq_candidates(faq_file, active, required=11)
    knowledge = run_dir / "grounded-dialogue-knowledge.tsv"
    atomic_text(knowledge, "".join(f"{entry['prompt']}\t{entry['response']}\n" for entry in entries))
    provenance = run_dir / "grounded-dialogue-knowledge-provenance.json"
    atomic_json(
        provenance,
        {
            "source": "Aliumutcan/Turkish_NLP_FAQ_DATASET dataset_v1.txt",
            "selection": entries,
            "retrieval_contract": "Responses are loaded from this TSV and returned verbatim only; vegai_chat is not a free-form text generator.",
        },
    )
    results: list[dict[str, Any]] = []
    transcripts: list[Path] = []
    for entry in entries:
        command = (
            str(arguments.chat_bin.resolve()), "ask", str(checkpoint), str(knowledge),
            "--prompt", str(entry["prompt"]), "--max-entries", str(len(entries)),
            "--history-turns", "0", "--min-known-tokens", "1", "--min-similarity", "0.95",
            "--min-margin", "0.01", "--require-all-known",
        )
        result = run_child(command, timeout_seconds=arguments.inspect_timeout_seconds, cwd=Path.cwd())
        stem = f"grounded-dialogue-{entry['ordinal']:02d}"
        write_process_transcript(run_dir, stem, result)
        require_success(result, f"grounded dialogue {entry['ordinal']}")
        parsed = parse_chat_result(result.stdout, str(entry["prompt"]), str(entry["response"]))
        results.append({**entry, **parsed})
        transcripts.extend([run_dir / f"{stem}.stdout.txt", run_dir / f"{stem}.stderr.txt", run_dir / f"{stem}.command.txt"])
    result_tsv = run_dir / "grounded-dialogue-results.tsv"
    rows = [
        "ordinal\tprompt\tmatched_prompt\tbest_bipolar_similarity\trunner_up_bipolar_similarity\tgrounding_margin\tgrounded\tanswer\tsource_url\tsource_row\n"
    ]
    for row in results:
        rows.append(
            f"{row['ordinal']}\t{row['prompt']}\t{row['matched_prompt']}\t{row['best_bipolar_similarity']:.5f}\t"
            f"{row['runner_up_bipolar_similarity']:.5f}\t{row['grounding_margin']:.5f}\ttrue\t{row['answer']}\t"
            f"{row['source_url']}\t{row['source_row']}\n"
        )
    atomic_text(result_tsv, "".join(rows))
    return results, transcripts, knowledge, provenance


def run_experiment(arguments: argparse.Namespace) -> dict[str, Any]:
    run_dir = arguments.run_dir.resolve()
    require(not run_dir.exists() or not any(run_dir.iterdir()), f"run directory must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    require(arguments.epochs >= 2, "99M RESONANT evidence requires at least two epochs")
    require(arguments.context_window >= 1 and arguments.context_window < TOKENS_PER_RECORD,
            "context window must be positive and smaller than the emitted-record bound")
    require(arguments.omp_threads > 0, "OMP thread count must be positive")
    require(arguments.vocabulary_size > 128, "vocabulary size is too small for the 99M evidence contract")
    pretrain = require_regular_file(arguments.pretrain_bin, "pretraining executable")
    chat = require_regular_file(arguments.chat_bin, "chat executable")
    require(os.access(pretrain, os.X_OK) and os.access(chat, os.X_OK), "native executable is not executable")

    corpus = run_dir / "turkish_wikipedia_99m_native_ascii.txt"
    selected_vocabulary = run_dir / "selected-source-vocabulary.tsv"
    corpus_manifest = prepare_corpus(
        arguments.extracted_root.resolve(), corpus, selected_vocabulary,
        target_tokens=arguments.target_tokens, vocabulary_size=arguments.vocabulary_size,
    )
    provenance = source_provenance(arguments, corpus_manifest)
    provenance_path = run_dir / "provenance.json"
    corpus_manifest_path = run_dir / "corpus-manifest.json"
    atomic_json(provenance_path, provenance)
    atomic_json(corpus_manifest_path, corpus_manifest)

    checkpoint = run_dir / "turkish_wikipedia_99m_resonant.vgai"
    training_command = (
        str(pretrain), "train", str(corpus), str(checkpoint),
        "--max-vocab", str(arguments.vocabulary_size), "--min-frequency", "1",
        "--window", str(arguments.context_window), "--position-shift", "41",
        "--context-mode", "resonant", "--epochs", str(arguments.epochs), "--strict",
        "--parallel-context-build", "--parallel-min-contexts", "32",
    )
    old_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(arguments.omp_threads)
    try:
        training = run_child(training_command, timeout_seconds=arguments.train_timeout_seconds, cwd=Path.cwd())
    finally:
        if old_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old_omp
    write_process_transcript(run_dir, "native-training", training)
    require_success(training, "native Fortran 99M training")
    require(checkpoint.is_file(), "native training reported success without a checkpoint")
    metrics = parse_training_metrics(training.stdout)
    expected_updates = arguments.target_tokens * arguments.epochs
    expected_records = int(corpus_manifest["output"]["physical_lf_records"]) * arguments.epochs
    require(metrics["completed epochs"] == arguments.epochs, "native train completed an unexpected epoch count")
    require(metrics["corpus lines"] == expected_records, "native train record count differs from manifest")
    for field in ("raw tokens", "active tokens", "prototype updates"):
        require(metrics[field] == expected_updates, f"native {field} differs from exact expected model updates")
    require(metrics["out-of-vocabulary tokens"] == 0, "native corpus emitted out-of-vocabulary tokens")
    require(metrics["tokenizer truncations"] == 0 and metrics["non-ASCII input bytes"] == 0 and
            metrics["dropped overlong records"] == 0, "native strict input telemetry is nonzero")
    require(metrics["resonant frozen-snapshot contexts"] > 0,
            "RESONANT training did not construct frozen-snapshot contexts")

    active_vocabulary = run_dir / "active-vocabulary.tsv"
    active_export = run_child(
        (str(pretrain), "inspect", str(checkpoint), "--active-vocabulary-file", str(active_vocabulary)),
        timeout_seconds=arguments.inspect_timeout_seconds, cwd=Path.cwd(),
    )
    write_process_transcript(run_dir, "active-vocabulary-export", active_export)
    require_success(active_export, "native active-vocabulary export")
    active = parse_active_vocabulary(active_vocabulary)
    selected = read_selected_tokens(selected_vocabulary)
    # Frequency selection is performed over the complete train partition while
    # the emitted stream stops exactly at 99M. A rare tail-selected type can
    # therefore legitimately occur only after the exact cutoff. What matters
    # for this run is the stronger native gate above: every emitted token was
    # active and updated, with no OOV telemetry. Every active type must still
    # be provenance-selected and capacity-bounded.
    require(len(active) <= arguments.vocabulary_size and frozenset(active).issubset(selected),
            "native active vocabulary is not a capacity-bounded subset of the provenance-selected vocabulary")

    semantic_queries = select_semantic_queries(active, count=11)
    semantic_query_file = run_dir / "semantic-queries.txt"
    atomic_text(semantic_query_file, "\n".join(semantic_queries) + "\n")
    semantic = run_child(
        (str(pretrain), "inspect", str(checkpoint), "--query-file", str(semantic_query_file)),
        timeout_seconds=arguments.inspect_timeout_seconds, cwd=Path.cwd(),
    )
    write_process_transcript(run_dir, "semantic-retrieval", semantic)
    require_success(semantic, "native semantic retrieval")
    semantic_results = parse_semantic_results(semantic.stdout, semantic_queries)
    semantic_results_path = run_dir / "semantic-retrieval-results.json"
    atomic_json(semantic_results_path, {"queries": semantic_results})

    quality_queries = select_quality_queries(active)
    quality_query_file = run_dir / "heldout-quality-queries.txt"
    atomic_text(quality_query_file, "\n".join(quality_queries) + "\n")
    quality_inference = run_child(
        (str(pretrain), "inspect", str(checkpoint), "--query-file", str(quality_query_file)),
        timeout_seconds=arguments.inspect_timeout_seconds, cwd=Path.cwd(),
    )
    write_process_transcript(run_dir, "heldout-semantic-inference", quality_inference)
    require_success(quality_inference, "native held-out semantic inference")
    quality_semantic_results = parse_semantic_results(quality_inference.stdout, quality_queries)
    heldout_evaluation = heldout_cooccurrence_evaluation(
        arguments.extracted_root.resolve(), active, quality_semantic_results, context_window=arguments.context_window,
    )
    heldout_path = run_dir / "heldout-cooccurrence-evaluation.json"
    atomic_json(heldout_path, heldout_evaluation)

    dialogue_results, dialogue_transcripts, knowledge, knowledge_provenance = create_grounded_dialogue(
        arguments, run_dir, checkpoint, active
    )
    dialogue_results_path = run_dir / "grounded-dialogue-results.tsv"
    checkpoint_identity = sha256_file(checkpoint)
    report = {
        "schema": SCHEMA,
        "status": "succeeded",
        "scope": {
            "native_model": "Fortran HDC/VSA distributional checkpoint with two-epoch RESONANT frozen semantic feedback",
            "not_claimed": [
                "free-form LLM generation",
                "general Turkish language understanding",
                "factual correctness beyond explicitly cited grounded TSV sources",
                "human semantic benchmark performance",
                "general intelligence",
            ],
        },
        "provenance": provenance,
        "corpus": corpus_manifest,
        "training": {
            "command": list(training.command),
            "omp_num_threads": arguments.omp_threads,
            "context_window": arguments.context_window,
            "train_process_wall_seconds": training.elapsed_seconds,
            "raw_training_tokens_per_second": expected_updates / training.elapsed_seconds,
            "metrics": metrics,
            "timing_scope": "Only the native vegai_pretrain train child; retrieval, corpus preparation, source extraction, downloads, compilation, hashing, and CI overhead are excluded.",
            "resonant_epoch_feedback": "Epoch two uses only the committed epoch-one frozen semantic snapshot as topical feedback.",
        },
        "checkpoint": checkpoint_identity,
        "active_vocabulary": {"file": sha256_file(active_vocabulary), "active_entries": len(active)},
        "semantic_retrieval": {
            "selection": "Named Turkish probes active in this exact checkpoint, then deterministic high-frequency non-stopword fallback.",
            "results": semantic_results,
            "interpretation": "Self-excluded checkpoint nearest-neighbour inspection, not generated responses or held-out semantic accuracy.",
        },
        "heldout_inference_quality": heldout_evaluation,
        "grounded_dialogue": {
            "knowledge_tsv": sha256_file(knowledge),
            "selection_provenance": sha256_file(knowledge_provenance),
            "policy": {"min_similarity": 0.95, "min_margin": 0.01, "require_all_prompt_tokens_known": True, "history_turns": 0},
            "results": dialogue_results,
            "interpretation": "Each answer is the exact ASCII-normalized TSV response field retrieved by native Fortran vegai_chat; it is not generated prose.",
        },
    }
    report_path = run_dir / "experiment-report.json"
    atomic_json(report_path, report)
    evidence_files = [
        arguments.source_manifest.resolve(), provenance_path, corpus_manifest_path, selected_vocabulary, active_vocabulary,
        semantic_query_file, semantic_results_path, quality_query_file, heldout_path,
        knowledge, knowledge_provenance, dialogue_results_path,
        run_dir / "native-training.command.txt", run_dir / "native-training.stdout.txt", run_dir / "native-training.stderr.txt",
        run_dir / "active-vocabulary-export.command.txt", run_dir / "active-vocabulary-export.stdout.txt", run_dir / "active-vocabulary-export.stderr.txt",
        run_dir / "semantic-retrieval.command.txt", run_dir / "semantic-retrieval.stdout.txt", run_dir / "semantic-retrieval.stderr.txt",
        run_dir / "heldout-semantic-inference.command.txt", run_dir / "heldout-semantic-inference.stdout.txt", run_dir / "heldout-semantic-inference.stderr.txt",
        *dialogue_transcripts,
    ]
    copy_evidence_files(run_dir, evidence_files, report_path, checkpoint)
    return report


def controller_smoke_test() -> None:
    """Test partitioning and strict records on temporary WikiExtractor-shaped JSON.

    The generated documents here are exclusively a controller test fixture.  It
    is never a training/evidence source and cannot be reported as real data.
    """
    with tempfile.TemporaryDirectory(prefix="vegai-trwiki-99m-smoke-") as temporary_name:
        temporary = Path(temporary_name)
        root = temporary / "extracted" / "AA"
        root.mkdir(parents=True)
        words = ("aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj", "kk", "ll", "mm", "nn", "oo", "pp", "qq", "rr", "ss", "tt", "uu", "vv", "ww", "xx", "yy", "zz", "aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh", "iii", "jjj", "kkk", "lll", "mmm", "nnn", "ooo", "ppp", "qqq", "rrr", "sss", "ttt", "uuu", "vvv", "www", "xxx", "yyy", "zzz", "aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg", "hhhh", "iiii", "jjjj", "kkkk", "llll", "mmmm", "nnnn", "oooo", "pppp", "qqqq", "rrrr", "ssss", "tttt", "uuuu", "vvvv", "wwww", "xxxx", "yyyy", "zzzz", "aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "fffff", "ggggg", "hhhhh", "iiiii", "jjjjj", "kkkkk", "lllll", "mmmmm", "nnnnn", "ooooo", "ppppp", "qqqqq", "rrrrr", "sssss", "ttttt", "uuuuu", "vvvvv", "wwwww", "xxxxx", "yyyyy", "zzzzz", "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu")
        train_id = next(str(index) for index in range(1, 10_000)
                        if not is_heldout_document(WikiDocument(str(index), "train", "x", "", 0)))
        heldout_id = next(str(index) for index in range(1, 10_000)
                          if is_heldout_document(WikiDocument(str(index), "heldout", "x", "", 0)))
        body = " ".join(words[:128])
        rows = [
            {"id": train_id, "title": "train", "text": "aa filtered bb\n" + "\n".join([body] * 8)},
            {"id": heldout_id, "title": "heldout", "text": "\n".join([body] * 3)},
        ]
        document_file = root / "wiki_00"
        document_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
        # The production target is intentionally not used by a unit smoke. The
        # same emission/validation code is exercised on an exact smaller even
        # target; production hard-locks TARGET_TOKENS above.
        frequencies, stats = scan_training_source(temporary / "extracted")
        require(stats["train_articles"] == 1 and stats["heldout_articles"] == 1, "partition smoke is not disjoint")
        ranked = selected_vocabulary(frequencies, 129)
        selected = frozenset(token for token, _ in ranked)
        output = temporary / "smoke.txt"
        emitted = 0
        records = 0
        with output.open("w", encoding="ascii", newline="\n") as handle:
            for document in iter_wiki_documents(temporary / "extracted"):
                if is_heldout_document(document):
                    continue
                written, made, _seen = emit_selected_document(handle, normalized_document_tokens(document), selected, 512 - emitted)
                emitted += written
                records += made
        require(emitted == 512 and records > 0, "smoke did not emit an exact bounded corpus")
        identity = validate_generated_corpus(output, 512)
        require(identity["native_token_count"] == 512 and identity["physical_lf_records"] == records,
                "smoke corpus violates strict token/record accounting")
        require(normalize_tokens("Çığ ÖŞÜ öğrenci " + "q" * 49) == ["cig", "osu", "ogrenci"],
                "smoke tokenizer projection changed unexpectedly")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-bin", type=Path, required=True)
    parser.add_argument("--chat-bin", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True, help="WikiExtractor --json output root")
    parser.add_argument("--source-archive", type=Path, required=True, help="downloaded versioned Wikimedia XML archive")
    parser.add_argument("--source-manifest", type=Path, required=True, help="workflow-produced exact dump status manifest")
    parser.add_argument("--faq-repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-url", default="https://dumps.wikimedia.org/trwiki/")
    parser.add_argument("--dataset-license", default="CC-BY-SA-4.0")
    parser.add_argument("--dataset-attribution", default="Turkish Wikipedia contributors / Wikimedia Foundation")
    parser.add_argument("--wikiextractor-version", default="3.0.6")
    parser.add_argument("--faq-url", default="https://github.com/Aliumutcan/Turkish_NLP_FAQ_DATASET")
    parser.add_argument("--faq-license", default="MIT")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--vocabulary-size", type=int, default=VOCABULARY_SIZE_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--context-window", type=int, default=CONTEXT_WINDOW_DEFAULT)
    parser.add_argument("--omp-threads", type=int, default=OMP_THREADS_DEFAULT)
    parser.add_argument("--train-timeout-seconds", type=float, default=18_000.0)
    parser.add_argument("--inspect-timeout-seconds", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    if sys.argv[1:] == ["--controller-smoke-test"]:
        try:
            controller_smoke_test()
        except ExperimentError as error:
            print(f"Turkish Wikipedia 99M controller smoke: FAIL: {error}", file=sys.stderr)
            return 1
        print("Turkish Wikipedia 99M controller smoke: PASS")
        return 0
    arguments = parse_arguments()
    try:
        require(arguments.train_timeout_seconds > 0 and arguments.inspect_timeout_seconds > 0,
                "all child timeouts must be positive")
        report = run_experiment(arguments)
    except ExperimentError as error:
        run_dir = arguments.run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(run_dir / "experiment-failure.json", {"schema": SCHEMA, "status": "failed", "error": str(error)})
        print(f"Turkish Wikipedia 99M experiment: FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "report": str(arguments.run_dir / "experiment-report.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
