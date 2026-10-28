#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a byte-exact, deterministic Turkish prompt/response corpus.

The repository does **not** ship 600 MB of licensed text, and this sandbox has
no route to a Wikipedia dump, so this tool builds a structured synthetic corpus
instead of pretending to download one.  Everything it emits is reproducible
from the seed recorded in the manifest, and the manifest binds the exact byte
count and SHA-256, so a training run can name the data it consumed.

Why synthetic is still worth training on
----------------------------------------
The point of this corpus is not to claim language knowledge.  It is to give the
engine a *measurable* task that exercises exactly what the architecture claims:

* **associative recall** -- 512 subjects, each bound to one answer word.  The
  question sentence appears in the corpus with the answer a few bytes later, and
  the evaluation prompts ask the same subject *without* the answer following,
  so a correct response proves the byte stream carried the binding.
* **copy** and **successor** patterns -- two cheap tasks that a model must get
  right before any recall number is believable.
* **distractors** -- every fact paragraph is surrounded by prose about unrelated
  subjects, so remembering subject *i* requires ignoring the rest.

Layout
------
The corpus is a sequence of documents.  Each document contains a few fact
sentences, one copy task, one successor task, and filler prose.  Facts are
shuffled per document with a deterministic stream so the same subject appears at
irregular offsets; that is what makes recall a real retrieval problem rather
than a fixed-position lookup.

Output: ``<corpus>.txt`` (exactly ``--bytes`` bytes), ``<corpus>.manifest.json``
(facts, sizes, hashes, generation parameters) and a verification pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

SCHEMA = "vegai-prompt-response-corpus/v1"
ALGORITHM = "turkish-patterned-facts/v1"
DEFAULT_BYTES = 600_000_000
DEFAULT_SEED = 20261028

# Open Turkish syllables.  Combined deterministically this yields pronounceable
# pseudo-words, which keeps the token distribution byte-realistic (multi-byte
# UTF-8 letters, word-length distribution, spaces and punctuation) without
# copying anyone's text.
ONSETS = ["", "b", "c", "ç", "d", "f", "g", "ğ", "h", "k", "l", "m", "n", "p", "r", "s", "ş",
          "t", "v", "y", "z", "tr", "st", "kl", "pr"]
NUCLEI = ["a", "e", "ı", "i", "o", "ö", "u", "ü"]
CODAS = ["", "n", "r", "l", "k", "m", "s", "t", "y", "z", "ş"]

FILLERS = [
    "Bu bölümde konuya genel bir giriş yapılıyor ve ayrıntılar sıralanıyor.",
    "Kayıtlar düzenli tutulduğunda sonraki adımlar daha kolay izlenir.",
    "Kısa cümleler okumayı hızlandırır, uzun cümleler ayrıntı taşır.",
    "Notlar tekrar edildikçe kalıcı hale gelir ve geri çağırma kolaylaşır.",
    "Her belge kendi içinde tutarlı olacak şekilde düzenlenmiştir.",
    "Sayılar ve adlar karıştırılmadan yazılmalıdır.",
    "Bir sonraki paragraf önceki cümlenin devamı niteliğindedir.",
    "Anlatım sade tutulduğunda anlaşılırlık artar.",
]


class CorpusError(RuntimeError):
    pass


def pseudo_word(rng: random.Random, syllables: int) -> str:
    parts = []
    for _ in range(syllables):
        parts.append(rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS))
    return "".join(parts)


def build_vocabulary(rng: random.Random, count: int, syllables: int, forbidden: set[str]) -> list[str]:
    """Distinct words: duplicates would make a recall error unmeasurable."""
    words: list[str] = []
    seen = set(forbidden)
    while len(words) < count:
        candidate = pseudo_word(rng, syllables)
        if len(candidate) < 4 or candidate in seen:
            continue
        seen.add(candidate)
        words.append(candidate)
    return words


def fact_sentence(subject: str, answer: str) -> str:
    return f"Soru: {subject} nedir? Cevap: {answer}."


def copy_sentence(value: str) -> str:
    return f"Tekrar et: {value}. Sonuç: {value}."


def successor_sentence(value: str) -> str:
    """`Say: abcdef. Sonra: bcdefg.` -- a local shift the model can learn.

    `value` is restricted to a-z so the shift is unambiguous; Turkish letters
    would need a case table to define "next", which is not the point here.
    """
    following = "".join(chr((ord(ch) - 97 + 1) % 26 + 97) for ch in value)
    return f"Say: {value}. Sonra: {following}."


def document(rng: random.Random, subjects: list[str], answers: list[str], copies: list[str],
             successors: list[str]) -> str:
    lines = []
    picks = rng.sample(range(len(subjects)), 6)
    for index, pick in enumerate(picks):
        lines.append(fact_sentence(subjects[pick], answers[pick]))
        if index % 2 == 0:
            lines.append(rng.choice(FILLERS))
    lines.append(copy_sentence(rng.choice(copies)))
    lines.append(successor_sentence(rng.choice(successors)))
    lines.append(rng.choice(FILLERS))
    return "\n".join(lines) + "\n\n"


PROBE_FACTS = 200
PROBE_COPY = 200
PROBE_NEXT = 100


def write_probe(path: Path, subjects: list[str], answers: list[str], seed: int) -> dict:
    """Emit the prompt/response probe set used by `vegai_prompt_engine probe`.

    Three families, each measurable by exact match:

    * ``fact`` -- subjects that *do* occur in the corpus, with the answer that
      the corpus binds to them.  This measures whether the recurrent memory
      actually carried a binding across the stream.
    * ``copy`` -- values that never occur in the corpus, so a correct answer
      cannot come from memorisation; it has to come from the in-context copy
      pattern.
    * ``next`` -- unseen ascii strings that must be shifted by one letter, the
      same local rule the corpus repeats.

    Format: ``kind<TAB>prompt<TAB>expected``, newline separated, UTF-8, and no
    tab inside a field.  `filter-probe` later drops the fact rows that a
    partial training pass never saw.
    """
    rng = random.Random(seed ^ 0x5EED)
    rows: list[tuple[str, str, str]] = []
    picks = rng.sample(range(len(subjects)), min(PROBE_FACTS, len(subjects)))
    for pick in picks:
        rows.append(("fact", f"Soru: {subjects[pick]} nedir? Cevap: ", answers[pick]))
    unseen_copy = build_vocabulary(rng, PROBE_COPY, syllables=3,
                                   forbidden=set(subjects) | set(answers))
    for value in unseen_copy:
        rows.append(("copy", f"Tekrar et: {value}. Sonuç: ", value))
    for _ in range(PROBE_NEXT):
        value = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        following = "".join(chr((ord(ch) - 97 + 1) % 26 + 97) for ch in value)
        rows.append(("next", f"Say: {value}. Sonra: ", following))
    path.write_text("".join(f"{kind}\t{prompt}\t{expected}\n" for kind, prompt, expected in rows),
                    encoding="utf-8")
    counts: dict[str, int] = {}
    for kind, _, _ in rows:
        counts[kind] = counts.get(kind, 0) + 1
    return {"probe": str(path), "rows": len(rows), "families": counts}


def filter_probe(probe_path: Path, output_path: Path, corpus: Path, trained_bytes: int) -> dict:
    """Drop ``fact`` rows whose subject a partial pass never saw.

    The trained prefix is scanned once; the subjects that appear in it are
    exactly the bindings the model could have learned.  ``copy`` and ``next``
    rows are kept: they test a rule, not a fact.
    """
    seen: set[str] = set()
    remaining = trained_bytes
    with open(corpus, "rb") as handle:
        while remaining > 0:
            chunk = handle.readline()
            if not chunk:
                break
            remaining -= len(chunk)
            line = chunk.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("Soru: ") and " nedir? Cevap: " in line:
                subject = line[len("Soru: "):].split(" nedir? Cevap: ", 1)[0]
                seen.add(subject)
    kept: list[str] = []
    dropped = 0
    with open(probe_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            kind, prompt, expected = line.rstrip("\n").split("\t", 2)
            if kind == "fact":
                subject = prompt[len("Soru: "):].split(" nedir? Cevap: ", 1)[0]
                if subject not in seen:
                    dropped += 1
                    continue
            kept.append(f"{kind}\t{prompt}\t{expected}\n")
    output_path.write_text("".join(kept), encoding="utf-8")
    return {"probe": str(output_path), "rows": len(kept), "dropped_fact_rows": dropped,
            "subjects_seen_in_prefix": len(seen), "trained_bytes": trained_bytes}


def generate(path: Path, manifest_path: Path, target_bytes: int, seed: int, *,
             facts: int = 2048, overwrite: bool = False, probe_path: Path | None = None) -> dict:
    if target_bytes <= 0:
        raise CorpusError("target byte count must be positive")
    if path.exists() and not overwrite:
        raise CorpusError(f"{path} exists; pass --overwrite to replace it")

    rng = random.Random(seed)
    subjects = build_vocabulary(rng, facts, syllables=2, forbidden=set())
    answers = build_vocabulary(rng, facts, syllables=3, forbidden=set(subjects))
    copies = build_vocabulary(rng, 128, syllables=2, forbidden=set(subjects) | set(answers))
    # ASCII-only values for the successor task, so the shift needs no case table.
    successors = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6)) for _ in range(64)]

    digest = hashlib.sha256()
    written = 0
    documents = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), prefix=".corpus-")
    try:
        with handle as stream:
            while written < target_bytes:
                block = document(rng, subjects, answers, copies, successors)
                payload = block.encode("utf-8")
                remaining = target_bytes - written
                if len(payload) > remaining:
                    # Truncate on a UTF-8 boundary: never emit a partial sequence.
                    payload = payload[:remaining]
                    while payload and (payload[-1] & 0xC0) == 0x80:
                        payload = payload[:-1]
                    if payload and payload[-1] >= 0xC0:
                        payload = payload[:-1]
                    remaining = len(payload)
                    if remaining <= 0:
                        break
                stream.write(payload)
                digest.update(payload)
                written += len(payload)
                documents += 1
        # Pad to the exact byte count with newlines (valid UTF-8, keeps the
        # contract "exactly N bytes").
        if written < target_bytes:
            padding = b"\n" * (target_bytes - written)
            with open(handle.name, "ab") as append:
                append.write(padding)
            digest.update(padding)
            written += len(padding)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise

    manifest = {
        "schema": SCHEMA,
        "algorithm": ALGORITHM,
        "seed": seed,
        "target_bytes": target_bytes,
        "bytes": written,
        "sha256": digest.hexdigest(),
        "documents": documents,
        "facts": [{"subject": s, "answer": a} for s, a in zip(subjects, answers)],
        "copy_values": copies,
        "successor_values": successors,
        "paths": {"corpus": str(path)},
        "honesty": {
            "synthetic": True,
            "language": "tr",
            "claim": "structured synthetic Turkish for engine validation; not a source of world knowledge",
        },
    }
    if probe_path is not None:
        manifest["probe"] = write_probe(probe_path, subjects, answers, seed)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def verify(path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise CorpusError("unexpected manifest schema")
    size = path.stat().st_size
    if size != manifest["bytes"] or size != manifest["target_bytes"]:
        raise CorpusError(f"size {size} does not match the manifest")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    if digest.hexdigest() != manifest["sha256"]:
        raise CorpusError("SHA-256 does not match the manifest")
    text = path.read_text(encoding="utf-8")
    return {"bytes": size, "documents": text.count("\n\n"), "facts": len(manifest["facts"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "verify", "filter-probe"])
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--probe-out", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trained-bytes", type=int, default=0)
    parser.add_argument("--bytes", type=int, default=DEFAULT_BYTES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--facts", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    manifest_path = arguments.manifest or arguments.corpus.with_suffix(".manifest.json")

    try:
        if arguments.action == "filter-probe":
            if arguments.probe is None or arguments.out is None:
                raise CorpusError("filter-probe needs --probe and --out")
            if arguments.trained_bytes <= 0:
                raise CorpusError("filter-probe needs a positive --trained-bytes")
            print(json.dumps({"event": "filtered",
                              **filter_probe(arguments.probe, arguments.out, arguments.corpus,
                                             arguments.trained_bytes)}, ensure_ascii=False))
            return 0
        if arguments.action == "generate":
            manifest = generate(arguments.corpus, manifest_path, arguments.bytes, arguments.seed,
                                facts=arguments.facts, overwrite=arguments.overwrite,
                                probe_path=arguments.probe_out)
            summary = verify(arguments.corpus, manifest_path)
            print(json.dumps({"event": "generated", **{k: manifest[k] for k in
                                                        ("bytes", "sha256", "documents", "seed")},
                              "verified_bytes": summary["bytes"],
                              "verified_documents": summary["documents"],
                              **({"probe": manifest["probe"]} if "probe" in manifest else {})},
                             ensure_ascii=False))
        else:
            print(json.dumps({"event": "verified", **verify(arguments.corpus, manifest_path)},
                             ensure_ascii=False))
    except (CorpusError, OSError) as error:
        print(f"corpus error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
