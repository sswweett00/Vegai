# Policy-driven capability curriculum packaging

> **Scope and honesty boundary:** This component is a governed data-selection,
> isolation, lineage, and continual-training control plane. It does **not** add
> model weights, generate answers, insert chain-of-thought, provide a neural
> objective, or establish general intelligence, reasoning correctness, coding
> correctness, or superiority to an LMM. The native learner remains the bounded
> HDC/VSA learner described in [PRETRAINING.md](PRETRAINING.md). Measurable
> improvement requires real licensed data, a suitable model/objective, target
> task evaluation, independent review, and reproducible comparisons.

`tools/capability_curriculum.py` compiles externally supplied rich records and
an operator-owned policy into ordered strict-HDC training shards. Its policy
is intentionally metadata-driven: an operator can define any approved tags and
stages (including, for example, reasoning, problem solving, or coding) without
those labels, examples, prompts, answers, or task facts being embedded in the
program.

The result has a standard `vegai-continuous-shard-manifest/v1` `ledger.json`,
so [CONTINUOUS_PRETRAINING.md](CONTINUOUS_PRETRAINING.md)'s exactly-once
controller can consume it in stage order.

## 1. What problem it solves

A broad data lake alone is not evidence that a training run has the required
coverage for a target domain. The compiler adds an auditable boundary between:

1. **governed rich records** — source text plus operator-defined provenance,
   split, capability, and difficulty annotations;
2. **an immutable policy** — stage order, metadata paths, coverage minima,
   split rule, and ambiguity rule; and
3. **strict native training payloads** — only selected `record.text`, as ASCII
   LF records, with no labels or metadata injected into the native HDC corpus.

It rejects rather than silently repairs missing metadata, conflicting duplicate
annotations, cross-split text leakage, unrepresented stages, source-diversity
shortfalls, tokenizer loss, changing inputs, malformed package contents, or an
attempt to overwrite a previously published package.

## 2. Inputs and dependencies

### Rich records are required

The input is read through `tools/training_data_io.py` and must satisfy the
existing `vegai-training-record/v1` contract. The policy's `capabilities_path`
and `difficulty_path` resolve inside every input record's `metadata` object;
when `require_record_source` is true, every input record also needs its own
`source` field.

Consequently JSONL, JSON, Parquet, and Arrow can carry the required
annotations. TXT is accepted by the common reader but is intentionally
rejected by this compiler: the repository's text-only format has no metadata,
so it cannot prove curriculum assignment or held-out isolation. This is a
fail-closed design rather than an implicit sidecar convention.

| Input path | Runtime dependency |
|---|---|
| JSONL / JSON | Python standard library |
| Parquet / Arrow | optional `pyarrow`, installed with `requirements-dataio.txt` |
| TXT | no extra dependency, but unsuitable for this annotation-dependent compiler |

The compiler itself uses only the standard library plus existing repository
tools. It reads records in a stream and keeps exact text identity, source
coverage, and selected-training vocabulary identity in a private SQLite index;
it does not load the whole data lake into Python memory. For identity-safe
parsing it also makes one private byte-for-byte input snapshot in the staging
directory, so reserve temporary SSD space roughly equal to the compressed/raw
source file. That snapshot is hash-checked and deleted before publication; it
is never part of the package.

### Strict HDC projection

Only selected **train** records are exported. Every exported text must already
meet the native strict contract:

- ASCII only, LF output records;
- at most 8,192 bytes per physical record;
- at most 1,024 tokens per record;
- at most 48 bytes per token; and
- no tokenizer truncation or dropped token.

Unicode normalization, translation/transliteration, cleaning, license review,
PII handling, safety review, source-level overlap work, and semantic
near-duplicate detection remain upstream governed responsibilities. Record any
intentional reversible preprocessing in policy provenance; this tool does not
silently perform it.

## 3. Policy schema

The strict JSON schema identifier is
`vegai-capability-curriculum-policy/v1`. Unknown and missing fields are errors.
Start from
[`examples/capability_curriculum/policy.template.json`](../examples/capability_curriculum/policy.template.json),
then replace every placeholder with reviewed values. That file is only a
configuration skeleton; it contains no training records or answers.

| Top-level field | Required content |
|---|---|
| `schema_version` | exact policy schema identifier |
| `id` | stable ASCII identifier for the reviewed policy |
| `provenance` | nonempty `source`, `license`, `retrieved_at`, and `preprocessing` strings |
| `annotations` | metadata key paths plus source-provenance requirement |
| `splits` | deterministic validation/test basis-point allocation when `record.split` is absent |
| `assignment` | ordered first-match behavior and strictness flags |
| `stages` | one to 64 ordered coverage-gated stages |

### Annotation paths

`annotations.capabilities_path` and `annotations.difficulty_path` are nonempty
arrays of ASCII metadata-key identifiers. At those locations each input
record must expose:

- an array of one to 128 distinct capability identifiers; and
- a finite difficulty number in `[0, 1]`.

The names have no hardcoded meaning to the implementation. A stage matches if
its `required_capabilities` are a subset of the record annotation and its
inclusive difficulty range contains the record difficulty. An empty required
capability list is permitted only when an operator deliberately wants a
metadata-annotated difficulty-only stage.

### Split rule and exact deduplication

An explicit `record.split` (`train`, `validation`, or `test`) wins. Otherwise
the repository's deterministic text-SHA-256 split is used:

```text
bucket = big_endian_u64(SHA256(UTF-8 record.text)[:8]) mod 10000
bucket < test_basis_points                                  -> test
bucket < test_basis_points + validation_basis_points        -> validation
otherwise                                                   -> train
```

Exact identity is `SHA-256(UTF-8 record.text)` and is stored in a private
SQLite index. The first declaration establishes the split and the canonical
capability/difficulty annotation. A later identical text with a different
split or annotation fails the build. Identical text with the same declaration
is emitted once only. This enforces exact train/held-out isolation; it does not
claim to detect semantic paraphrases or source-family leakage.

### Ordered stages and coverage gates

`assignment.mode` is currently the explicit deterministic value `first-match`.
Stages are processed in policy order. If multiple stages match, the policy can
fail closed (`reject_multiple_matches: true`) or deliberately send a record to
the first stage while reporting the ambiguity. If no stage matches,
`require_every_annotated_record: true` makes the build fail.

Each stage specifies all of the following:

| Field | Gate |
|---|---|
| `id` | unique stable stage identifier and deterministic shard filename component |
| `required_capabilities` | operator-defined subset match |
| `minimum_difficulty`, `maximum_difficulty` | inclusive bounds in `[0, 1]` |
| `minimum_train_records` | positive count; an empty training stage is never valid |
| `minimum_heldout_records` | required validation + test coverage count |
| `minimum_distinct_train_sources` | source diversity after exact deduplication |
| `minimum_distinct_heldout_sources` | distinct source count across validation and test, not a double-counted sum |

Counts are gates, not estimates of task quality. Set them from a reviewed data
and evaluation plan rather than treating a passing count as a capability claim.
If `require_record_source` is deliberately false, all missing sources map to
one stable undeclared sentinel; they cannot falsely satisfy a diversity minimum
greater than one.

## 4. Build and verify

Use a new output directory for every immutable release:

```bash
python3 tools/capability_curriculum.py build DATASET \
  --policy POLICY.json \
  --output-dir releases/curriculum-release
```

`DATASET` may be JSONL/JSON/Parquet/Arrow. Use `--input-format` or
`--input-compression` only when automatic inference is unsuitable. The source
and policy are byte-hashed before processing and re-hashed before publication.
The record stream is read from a private byte-identical snapshot whose digest
must match the source identity; a change causes failure and staging cleanup.
Publication is private staging
plus file/directory synchronization where supported and an atomic **no-replace**
directory publication. An existing output directory (including a dangling
symlink) is never replaced. On a platform without a safe no-replace directory
primitive, publication fails rather than falling back to a check-then-replace
race.

A successful package contains exactly:

```text
curriculum-release/
  curriculum.json
  ledger.json
  shards/
    001-<policy-stage-id>.txt
    002-<policy-stage-id>.txt
    ...
```

`ledger.json` is consumed directly by `continuous_pretrain.py`. `curriculum.json`
contains policy/source byte identities, the resolved `none`/`gzip` text codec,
selected-stage counts, strict preflight facts, a held-out digest, and the exact
`min_frequency=1` selected vocabulary count. Paths stored in the portable
package are basenames/package-relative, not an operator workstation's absolute
path.

The held-out digest follows
`stage-nul-split-nul-text-sha256/v1`: it commits to selected stage and split
using one-way text digests but does not write held-out prompt, answer, or raw
text into the package. Keep actual target tasks and expected results in the
separately governed evaluation store.

Basic package/payload verification is:

```bash
python3 tools/capability_curriculum.py verify \
  --package releases/curriculum-release
```

For the strongest local derivation check, pass **both** original inputs:

```bash
python3 tools/capability_curriculum.py verify \
  --package releases/curriculum-release \
  --source DATASET --policy POLICY.json
```

With both arguments, verification first checks their immutable identities,
then recompiles them into a private temporary directory and byte-compares the
complete package tree. This detects an altered self-reported coverage count,
held-out digest, constraints, shard order, ledger, or payload. It deliberately
performs an additional source read and temporary build, so budget the I/O for
large releases. It never alters the published package, source, or policy;
temporary replay material is deleted on exit. A single `--source` or `--policy`
only performs the corresponding identity check, not full derivation replay.

## 5. Hand-off to continual training

Inspect the returned `capacity_preflight` before a run. For a newly initialized
controller, `--max-vocab` must be at least
`minimum_continual_max_vocabulary`; this exact lower bound assumes
`--min-frequency 1`, which the continual path requires. For an existing
controller, plan capacity for its pre-existing active vocabulary **plus** new
unique tokens—do not treat the per-package bound as a complete future-capacity
calculation.

After source/policy replay verification and independent target-task release
gates, hand off the package ledger in its existing policy order:

```bash
python3 tools/continuous_pretrain.py \
  --pretrain-bin ./build/vegai_pretrain \
  --manifest releases/curriculum-release/ledger.json \
  --state-dir state/continual \
  --max-vocab REVIEWED_CAPACITY \
  --min-frequency 1 \
  --context-mode coherent
```

The controller's immutable manifest, checkpoint, token/OOV, candidate, and
exactly-once rules still apply unchanged. A curriculum package is a shard
publisher, not a bypass around those safeguards. See
[CONTINUOUS_PRETRAINING.md](CONTINUOUS_PRETRAINING.md) for append-only manifest
and state-journal rules, and [PRODUCTION_PRETRAINING.md](PRODUCTION_PRETRAINING.md)
for full production data/run release controls.

## 6. Test coverage and release checklist

The stdlib-only integration verifier is registered in both build surfaces:

```bash
python3 tools/verify_capability_curriculum.py
make capability-check
# CMake/CTest name: vegai_capability_curriculum
```

It creates temporary protocol fixtures only; it does not create a quality or
capability training dataset. It covers successful build/replay, text-only
projection, manifest-controller compatibility, portable paths, immutable
no-overwrite publication (including a dangling output link), exact dedupe,
annotation-conflict rejection,
split-conflict rejection, bare-TXT metadata rejection, held-out coverage
failure with no output publication, package-report
replay rejection, unbound package-file rejection, and shard tamper rejection.
Where optional PyArrow is
installed, it additionally replays Parquet and Arrow input; otherwise that
optional adapter branch is explicitly reported as skipped.

Before promotion, operators should at minimum:

1. review licenses, consent, safety, privacy, source independence, and
   preprocessing provenance upstream;
2. freeze and hash the policy and source release;
3. set meaningful train/held-out/source coverage gates for each target domain;
4. run full `verify --source --policy` and independently preserve its report;
5. run target-domain, held-out, safety, reliability, latency, and cost
   evaluation appropriate to the real model plane; and
6. retain the policy, source identifiers, package hashes, controller state, and
   independent evaluation evidence together with the release decision.

For external LMM model selection and measured target-domain routing, use the
separate evidence contracts in [LMM_PORTFOLIO_ROUTING.md](LMM_PORTFOLIO_ROUTING.md).
This curriculum layer complements that work; it cannot substitute for it.
