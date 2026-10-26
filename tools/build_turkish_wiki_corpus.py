#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a provenance-bound, quality-filtered Turkish corpus for Vegai.

This is the data plane of the 99-billion-token pretraining program.  It is
deliberately honest about scale: the manifest records exactly how many real
source documents/records/bytes survived every filter stage, and the 99B
target is reported as a target, never as an achieved count.  The builder

* streams an official Turkish Wikipedia ``pages-articles`` XML dump
  (``.xml.bz2``) or a directory of UTF-8 text files (test/operator mode);
* cleans wikitext deterministically (templates, refs, tables, file/category
  links, markup) without translation or transliteration;
* applies fail-closed Turkish quality filters with a full rejection funnel;
* removes exact duplicate documents and exact duplicate paragraphs
  (>=120 normalized chars) across the whole stream;
* splits train/validation/held-out test at the *document* level by stable SHA-256 buckets
  before any tokenizer or model sees the text;
* emits LF-framed records bounded for the native byte-BPE trainer
  (``BPE_MAX_INPUT_BYTES=8192``), a tokenizer sample, and an optional
  byte-budgeted first-epoch slice -- all SHA-256 bound in a manifest.

Near-duplicate (fuzzy) detection and a complete PII audit are intentionally
not claimed; direct email/telephone/IPv4 patterns reject a document, as in
the earlier COSMOS controller.
"""
from __future__ import annotations

import argparse
import bz2
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from typing import Any, Iterator

SCHEMA = "vegai-turkish-wiki-corpus/v4"
TARGET_TOKENS_99B = 99_000_000_000
SPLIT_MODULUS = 200
SPLIT_TEST_BUCKET = 0
SPLIT_VALIDATION_BUCKET = 1
PARAGRAPH_DEDUPE_MIN_CHARS = 120

TURKISH_LETTERS = set(
    "abcçdefgğhıijklmnoöprsştuüvyzqwx"
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWX"
)
TURKISH_FUNCTION_WORDS = {
    "bir", "ve", "de", "da", "bu", "ile", "için", "çok", "olarak", "olan",
    "gibi", "daha", "en", "ne", "kadar", "sonra", "önce", "ama", "ancak",
    "çünkü", "ya", "hem", "ise", "üzere", "rağmen", "göre", "karşı",
    "tarafından", "zaman", "şey", "kendi", "her", "hiç", "ki", "mi",
    "nasıl", "hangi", "neden", "yer", "insanlar", "arasında", "üzerinde",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?90[\s.-]?)?(?:\(?0?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


class CorpusError(Exception):
    """Fail-closed corpus construction error."""


@dataclass
class Document:
    page_id: str
    title: str
    text: str


@dataclass
class Funnel:
    source_documents: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    duplicate_documents: int = 0
    duplicate_paragraphs: int = 0
    train_documents: int = 0
    validation_documents: int = 0
    heldout_documents: int = 0

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_documents": self.source_documents,
            "rejected": dict(sorted(self.rejected.items())),
            "rejected_total": sum(self.rejected.values()),
            "duplicate_documents": self.duplicate_documents,
            "duplicate_paragraphs": self.duplicate_paragraphs,
            "train_documents": self.train_documents,
            "validation_documents": self.validation_documents,
            "heldout_documents": self.heldout_documents,
            "accepted_documents": self.train_documents + self.validation_documents + self.heldout_documents,
        }


def localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def iter_wiki_documents(path: Path) -> Iterator[Document]:
    """Stream namespace-tolerant MediaWiki export pages from a bz2 dump."""
    with bz2.open(path, "rb") as handle:
        context = ET.iterparse(handle, events=("start", "end"))
        root: ET.Element | None = None
        for event, element in context:
            if root is None:
                if event == "start":
                    root = element
                continue
            if event != "end" or localname(element.tag) != "page":
                continue
            title = ""
            page_id = ""
            namespace = ""
            text = ""
            is_redirect = False
            for child in element:
                name = localname(child.tag)
                if name == "title":
                    title = child.text or ""
                elif name == "id" and not page_id:
                    page_id = child.text or ""
                elif name == "ns":
                    namespace = child.text or ""
                elif name == "redirect":
                    is_redirect = True
                elif name == "revision":
                    for grandchild in child:
                        if localname(grandchild.tag) == "text":
                            text = grandchild.text or ""
            # Clearing the page alone leaves an empty child object attached to
            # the root for every page in a multi-million-row dump. Clear the
            # root's completed children as well to keep XML-tree memory bounded.
            element.clear()
            root.clear()
            if is_redirect or namespace != "0":
                continue
            if title and text:
                yield Document(page_id=page_id or title, title=title, text=text)


def iter_text_dir_documents(root: Path) -> Iterator[Document]:
    for path in sorted(root.rglob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        title = lines[0].strip() if lines else path.stem
        body = "\n".join(lines[1:]) if len(lines) > 1 else raw
        yield Document(page_id=path.name, title=title or path.stem, text=body)


def strip_templates(text: str) -> str:
    """Remove balanced {{...}} templates, including nested ones."""
    output: list[str] = []
    depth = 0
    index = 0
    while index < len(text):
        if text.startswith("{{", index):
            depth += 1
            index += 2
            continue
        if text.startswith("}}", index) and depth > 0:
            depth -= 1
            index += 2
            continue
        if depth == 0:
            output.append(text[index])
        index += 1
    return "".join(output)


def clean_wikitext(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = strip_templates(text)
    # Tables.
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.DOTALL)
    # File/category links carry no prose.
    text = re.sub(r"\[\[(?:Dosya|File|Resim|Image|Kategori|Category):[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    # Internal links: [[target|label]] -> label, [[target]] -> target.
    text = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    # External links: [url label] -> label, bare [url] -> nothing.
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    # Headings, emphasis, HTML tags, magic words.
    text = re.sub(r"^=+[^=\n]*=+\s*$", " ", text, flags=re.MULTILINE)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"<[^>]{1,200}>", " ", text)
    text = re.sub(r"__[A-Z]+__", " ", text)
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith(("|", "!", "{|", "|}")):
            continue
        stripped = re.sub(r"^[*#;:]+", "", stripped).strip()
        lines.append(stripped)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def quality_rejection_reason(text: str, *, min_chars: int, max_chars: int) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if EMAIL_RE.search(text) or IPV4_RE.search(text) or PHONE_RE.search(text):
        return "direct_pii_pattern"
    nons = [char for char in text if not char.isspace()]
    if not nons:
        return "empty"
    letters = [char for char in nons if char.isalpha()]
    digits = sum(1 for char in nons if char.isdigit())
    if len(letters) / len(nons) < 0.62:
        return "low_alpha_ratio"
    if digits / len(nons) > 0.25:
        return "digit_heavy"
    turkish_letters = sum(1 for char in letters if char in TURKISH_LETTERS)
    if letters and turkish_letters / len(letters) < 0.88:
        return "non_turkish_script"
    words = [match.group(0).lower() for match in WORD_RE.finditer(text)]
    if len(words) < 80:
        return "too_few_words"
    mean_word = sum(len(word) for word in words) / len(words)
    if mean_word < 3.0 or mean_word > 12.0:
        return "word_length_anomaly"
    function_ratio = sum(1 for word in words if word in TURKISH_FUNCTION_WORDS) / len(words)
    if function_ratio < 0.015 or function_ratio > 0.40:
        return "function_word_anomaly"
    if len(words) >= 300 and len(set(words)) / len(words) < 0.18:
        return "repetitive"
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 8:
        short_ratio = sum(1 for line in lines if len(line) < 50) / len(lines)
        if short_ratio > 0.5:
            return "list_like"
    return None


def split_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        paragraph = " ".join(block.split())
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs


def chunk_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    chunks: list[str] = []
    current = ""
    for sentence in SENTENCE_SPLIT_RE.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(sentence), max_chars):
                chunks.append(sentence[start:start + max_chars])
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def split_bucket(document: Document) -> int:
    """Stable source-document bucket; same title/page identity never crosses splits."""
    identity = f"vegai-trwiki-split-v2\0{document.page_id}\0{document.title}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % SPLIT_MODULUS


class HashedWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("wb")
        self.hasher = hashlib.sha256()
        self.bytes_written = 0
        self.records = 0
        self.approx_words = 0

    def write_record(self, record: str) -> None:
        payload = (record + "\n").encode("utf-8")
        self.handle.write(payload)
        self.hasher.update(payload)
        self.bytes_written += len(payload)
        self.records += 1
        self.approx_words += record.count(" ") + 1

    def close(self) -> dict[str, Any]:
        self.handle.close()
        return {
            "path": self.path.name,
            "bytes": self.bytes_written,
            "records": self.records,
            "approx_words": self.approx_words,
            "sha256": self.hasher.hexdigest(),
        }


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_byte_budget_slices(
    source: Path, destinations: list[Path], budgets: list[int]
) -> list[dict[str, Any]]:
    """Write deterministic, evenly-spaced whole-record samples under byte caps.

    Limited samples use two sequential passes to count records and select
    evenly-spaced record midpoints across the entire training split. Multiple
    outputs share both passes, avoiding a separate full-file scan per artifact.
    Output order is retained; records are never split; memory is O(max record
    size), independent of corpus size. A zero or corpus-sized budget copies the
    complete file.
    """
    if not destinations or len(destinations) != len(budgets) or any(budget < 0 for budget in budgets):
        raise CorpusError("one or more byte-budget destinations and non-negative budgets must correspond")
    writers = [HashedWriter(path) for path in destinations]
    source_bytes = source.stat().st_size
    full_outputs = [budget <= 0 or budget >= source_bytes for budget in budgets]
    needs_sampling = any(not full for full in full_outputs)
    record_count = 0
    if needs_sampling:
        with source.open("rb") as handle:
            record_count = sum(1 for _ in handle)

    target_counts = [0] * len(budgets)
    selected_indices = [0] * len(budgets)
    next_targets = [0] * len(budgets)
    used_bytes = [0] * len(budgets)
    for index, budget in enumerate(budgets):
        if not full_outputs[index] and record_count > 0:
            target_counts[index] = min(
                record_count,
                max(1, (budget * record_count + source_bytes // 2) // source_bytes),
            )
            next_targets[index] = record_count // (2 * target_counts[index])

    def write_raw(writer: HashedWriter, raw: bytes) -> None:
        writer.handle.write(raw)
        writer.hasher.update(raw)
        writer.bytes_written += len(raw)
        writer.records += 1
        writer.approx_words += raw.count(b" ") + 1

    try:
        with source.open("rb") as handle:
            for record_index, raw in enumerate(handle):
                for index, writer in enumerate(writers):
                    if full_outputs[index]:
                        write_raw(writer, raw)
                    elif record_index == next_targets[index]:
                        if used_bytes[index] + len(raw) <= budgets[index]:
                            write_raw(writer, raw)
                            used_bytes[index] += len(raw)
                        selected_indices[index] += 1
                        if selected_indices[index] < target_counts[index]:
                            next_targets[index] = (
                                (2 * selected_indices[index] + 1) * record_count
                            ) // (2 * target_counts[index])
                if not any(full_outputs) and all(
                    selected_indices[index] >= target_counts[index]
                    for index in range(len(budgets))
                    if target_counts[index] > 0
                ):
                    break
    finally:
        metadata = [writer.close() for writer in writers]
    return metadata


def write_byte_budget_slice(source: Path, destination: Path, budget: int) -> dict[str, Any]:
    """Single-output compatibility wrapper for deterministic whole-record sampling."""
    return write_byte_budget_slices(source, [destination], [budget])[0]


def build_corpus(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.wiki_xml_bz2) == bool(args.text_dir):
        raise CorpusError("provide exactly one of --wiki-xml-bz2 or --text-dir")
    if args.wiki_xml_bz2:
        source_path = Path(args.wiki_xml_bz2)
        if not source_path.is_file():
            raise CorpusError(f"dump not found: {source_path}")
        source_sha256 = sha256_file(source_path)
        documents = iter_wiki_documents(source_path)
        source_meta = {
            "kind": "trwiki-pages-articles-xml-bz2",
            "file": source_path.name,
            "sha256": source_sha256,
            "url": args.source_url,
            "license": args.license,
        }
    else:
        source_root = Path(args.text_dir)
        if not source_root.is_dir():
            raise CorpusError(f"text dir not found: {source_root}")
        documents = iter_text_dir_documents(source_root)
        source_meta = {
            "kind": "operator-text-dir",
            "file": source_root.name,
            "sha256": None,
            "url": args.source_url,
            "license": args.license,
        }

    funnel = Funnel()
    seen_documents: set[bytes] = set()
    seen_paragraphs: set[bytes] = set()
    train_writer = HashedWriter(output_dir / "train.txt")
    validation_writer = HashedWriter(output_dir / "validation.txt")
    heldout_writer = HashedWriter(output_dir / "heldout.txt")
    try:
        for document in documents:
            if args.max_docs and funnel.source_documents >= args.max_docs:
                break
            funnel.source_documents += 1
            cleaned = clean_wikitext(document.text) if args.wiki_xml_bz2 else unicodedata.normalize("NFC", document.text)
            reason = quality_rejection_reason(cleaned, min_chars=args.min_chars, max_chars=args.max_chars)
            if reason is not None:
                funnel.reject(reason)
                continue
            document_key = hashlib.sha256(" ".join(cleaned.split()).encode("utf-8")).digest()
            if document_key in seen_documents:
                funnel.duplicate_documents += 1
                continue
            seen_documents.add(document_key)

            paragraphs: list[str] = []
            for paragraph in split_paragraphs(cleaned):
                normalized = " ".join(paragraph.lower().split())
                if len(normalized) >= PARAGRAPH_DEDUPE_MIN_CHARS:
                    key = hashlib.sha256(normalized.encode("utf-8")).digest()
                    if key in seen_paragraphs:
                        funnel.duplicate_paragraphs += 1
                        continue
                    seen_paragraphs.add(key)
                paragraphs.append(paragraph)
            if not paragraphs:
                funnel.reject("all_paragraphs_duplicate")
                continue

            title = " ".join(document.title.split())
            records: list[str] = []
            for index, paragraph in enumerate(paragraphs):
                chunks = chunk_paragraph(paragraph, args.max_record_chars)
                if index == 0 and title and not paragraph.startswith(title):
                    chunks[0] = f"{title}. {chunks[0]}"
                    if len(chunks[0].encode('utf-8')) >= 8192:
                        chunks = chunk_paragraph(chunks[0], args.max_record_chars)
                records.extend(chunks)
            records = [record for record in records if len(record.encode("utf-8")) < 8192 and len(record) >= 40]
            if not records:
                funnel.reject("no_usable_records")
                continue

            bucket = split_bucket(document)
            if bucket == SPLIT_TEST_BUCKET:
                funnel.heldout_documents += 1
                writer = heldout_writer
            elif bucket == SPLIT_VALIDATION_BUCKET:
                funnel.validation_documents += 1
                writer = validation_writer
            else:
                funnel.train_documents += 1
                writer = train_writer
            for record in records:
                writer.write_record(record)
    finally:
        train_meta = train_writer.close()
        validation_meta = validation_writer.close()
        heldout_meta = heldout_writer.close()

    if train_meta["records"] == 0:
        raise CorpusError("quality pipeline accepted zero training records; refusing to publish an empty corpus")
    if validation_meta["records"] == 0 or heldout_meta["records"] == 0:
        raise CorpusError("document split produced empty validation or held-out test data; refusing an unevaluable corpus")

    sample_meta, epoch_meta = write_byte_budget_slices(
        output_dir / "train.txt",
        [output_dir / "tokenizer_sample.txt", output_dir / "train_epoch.txt"],
        [args.tokenizer_sample_bytes, args.train_byte_budget],
    )

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "retrieved_at": args.retrieved_at,
        "source": source_meta,
        "target": {
            "program_token_target": TARGET_TOKENS_99B,
            "target_label": "99B-token Turkish pretraining program",
            "honesty_note": (
                "program_token_target is the multi-source program goal, not the size of this corpus. "
                "This manifest only claims the exact accepted counts below."
            ),
        },
        "split": {
            "rule": "sha256(vegai-trwiki-split-v2|page_id|title) mod 200: 0 -> heldout_test, 1 -> validation, 2..199 -> train",
            "level": "source-document",
            "validation_fraction_nominal": 1 / SPLIT_MODULUS,
            "heldout_test_fraction_nominal": 1 / SPLIT_MODULUS,
            "train_fraction_nominal": (SPLIT_MODULUS - 2) / SPLIT_MODULUS,
            "selection_uses": "validation only",
            "final_reporting_uses": "heldout test only",
        },
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "max_record_chars": args.max_record_chars,
            "max_docs": args.max_docs,
            "paragraph_dedupe_min_chars": PARAGRAPH_DEDUPE_MIN_CHARS,
            "near_duplicate_detection": "not-claimed",
            "pii_audit": "direct email/telephone/IPv4 regex rejection only; not a complete PII audit",
        },
        "funnel": funnel.as_dict(),
        "outputs": {
            "train": train_meta,
            "validation": validation_meta,
            "heldout": heldout_meta,
            "tokenizer_sample": sample_meta,
            "train_epoch": epoch_meta,
        },
        "train_byte_budget": args.train_byte_budget,
        "tokenizer_sample_bytes": args.tokenizer_sample_bytes,
        "byte_budget_selection": (
            "deterministic evenly-spaced whole-record samples across the complete train split; "
            "tokenizer and epoch artifacts share two streaming passes; source order retained; "
            "strict byte caps; zero budget means full split"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-xml-bz2")
    parser.add_argument("--text-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license", default="CC BY-SA 4.0")
    parser.add_argument("--retrieved-at", default="unspecified")
    parser.add_argument("--min-chars", type=int, default=700)
    parser.add_argument("--max-chars", type=int, default=400_000)
    parser.add_argument("--max-record-chars", type=int, default=3_600)
    parser.add_argument("--max-docs", type=int, default=0, help="0 = no limit (probe sizing only; recorded in manifest)")
    parser.add_argument("--tokenizer-sample-bytes", type=int, default=3_000_000)
    parser.add_argument("--train-byte-budget", type=int, default=0, help="0 = full train split as the epoch corpus")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.min_chars < 1 or args.max_chars < args.min_chars or args.max_record_chars < 1:
            raise CorpusError("character bounds must be positive and max-chars >= min-chars")
        if args.max_docs < 0 or args.tokenizer_sample_bytes < 1 or args.train_byte_budget < 0:
            raise CorpusError("max-docs/train-byte-budget must be non-negative and tokenizer sample positive")
        manifest = build_corpus(args)
    except CorpusError as error:
        print(f"build_turkish_wiki_corpus: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "funnel": manifest["funnel"], "outputs": manifest["outputs"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
