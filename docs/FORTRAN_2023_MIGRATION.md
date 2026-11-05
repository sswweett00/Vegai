# Fortran-first migration ledger (2018 baseline, 2023 opt-in)

## Status and truthfulness boundary

Vegai's packed-HDC runtime, tokenizer, vocabulary, pretraining, checkpoint
reader/writer, dialogue retrieval, resonator network, and field executable are
Fortran sources.  The repository also still contains migration-era Python
control-plane tools.  Therefore it would be false to describe the entire
repository as "already entirely Fortran".

This document records the enforced direction instead: the **default native
build, runtime smoke, portable profile, bounded resource preflight, and
same-host checkpoint smoke are executable without Python**.  Python tooling is
currently an explicit optional legacy/integration layer, not a hidden
requirement of `vegai_control`, the core Fortran CMake tests, or the native
Make verification target.

The native baseline added in this migration increment is:

| Native capability | Fortran entry point | Evidence |
|---|---|---|
| compiler/runtime profile | `vegai_control profile` | CTest `vegai_fortran_control_profile` |
| bounded RAM/checkpoint preflight arithmetic | `vegai_control resource-plan` | property assertions in `test/test_all.f90` |
| vocab → train → stream checkpoint → load smoke | `vegai_control smoke` | CTest `vegai_fortran_control_smoke` |
| SHA-256 corpus/checkpoint identity attestation | `vegai_control attest` | FIPS `abc` vector property test; native smoke→attest CTest |
| native bounded corpus fingerprint and ASCII preflight | `vegai_control corpus-manifest` | single-pass SHA-256/counter fixtures and CLI regression |
| bounded semantic-neighbour quality gate | `vegai_control quality-gate` | native accept/reject property fixture in `test/test_all.f90` |
| offline UD text extraction | `vegai_ud_text_extract` | `vegai_ud_text_extract_contract`, byte and failure fixtures; [limits](UD_TEXT_EXTRACTION.md) |
| no-Python local verification | `make check` | native property, stream, sparse, context and control tests |

`vegai_control` deliberately does **not** run a shell command to guess RAM,
SSD space, CPU model, GPU, or NPU.  Standard Fortran has no portable OS
inventory API.  Its resource preflight instead accepts audited
`--available-memory-bytes` and `--available-storage-bytes`; `ready` is false
when either value is unknown, and `--require-ready` also exits nonzero when
values are omitted or insufficient.  This avoids a
misleading claim of a pure-Fortran machine probe that actually depends on
platform shell utilities. The profile exposes real64 radix/digits, file-storage
unit bits and whether the SHA-256 octet representation check passed; these are
compiler/runtime facts, not proof that an untested machine is supported.

## Native usage

```bash
# CMake's presets leave the Python legacy test layer off by default.
cmake --preset portable-serial
cmake --build --preset portable-serial --parallel
ctest --test-dir build/portable-serial --output-on-failure

# Portable profile: compiler/environment facts only, no OS shell probe.
./build/portable-serial/vegai_control profile --format json

# Auditable capacity preflight.  Values below are operator-supplied facts.
./build/portable-serial/vegai_control resource-plan \
  examples/corpus/toy_hdc_corpus.txt \
  --checkpoint /safe/runtime/toy.vgai --max-vocab 128 --context-mode resonant \
  --available-memory-bytes 8589934592 \
  --available-storage-bytes 17179869184 --require-ready --format json

# Single-pass fingerprint plus strict HDC tokenizer contract. Output defaults
# to stdout; --output requires a new path and will not replace existing data.
./build/portable-serial/vegai_control corpus-manifest \
  examples/corpus/toy_hdc_corpus.txt --strict-contract \
  --source local-example --license MIT --output /safe/runtime/corpus-manifest.json

# Entire train/checkpoint/load smoke is performed by the compiled Fortran code.
# Its checkpoint destination must be new, not an existing checkpoint or corpus.
./build/portable-serial/vegai_control smoke \
  examples/corpus/toy_hdc_corpus.txt /safe/runtime/toy.vgai

# Stream-I/O SHA-256 identities plus a native checkpoint-load validation.
./build/portable-serial/vegai_control attest \
  examples/corpus/toy_hdc_corpus.txt /safe/runtime/toy.vgai --format json

# A CASES.tsv line is: query_token<TAB>expected_self_excluded_neighbour.
# All thresholds are mandatory, so an invocation cannot silently inherit a
# permissive default policy. A rejection still emits its aggregate report.
./build/portable-serial/vegai_control quality-gate /safe/runtime/toy.vgai CASES.tsv \
  --minimum-cases 20 --minimum-top1-accuracy 0.80 \
  --minimum-top1-wilson-lower-bound-95 0.55 --minimum-mean-similarity 0.05 \
  --minimum-case-similarity -0.10 --maximum-incorrect-cases 4 --format json

# GNU Make native verification (no Python executable is called).
make check
# Until migrated, independently maintained legacy verifiers remain opt-in.
make legacy-python-check
```

CMake's default `VEGAI_ENABLE_OPENMP=ON` permits a warning and serial fallback
when OpenMP is absent. To make missing OpenMP a configure error, add
`-DVEGAI_REQUIRE_OPENMP=ON`; setting it together with
`-DVEGAI_ENABLE_OPENMP=OFF` is rejected. This proves compiler/runtime detection
at configure time, not that a particular number of threads or a speedup will be
achieved on every host.

The resource plan is a deterministic reservation estimate, not a prediction of
an OS allocator, page cache, NUMA policy, memory pressure, thermal throttle, or
GPU/NPU residency. `--context-mode resonant` adds exactly one 2,048-byte
semantic snapshot per vocabulary-capacity slot before reserve fraction; other
modes report zero snapshot bytes. For binary64, reserve bytes are the exact
ceiling of the represented fraction times the resident-byte count, even if a
floating multiply rounds down to an integer. For a nonzero reserve on other
real formats, the fallback conservatively reserves the full permitted maximum
(4× resident bytes) and can substantially overestimate; target-specific
validation is still required. On binary64 the result can exceed the legacy
truncating formula by a byte;
this is a preflight, not additional training work. It never turns an unknown host
value into a successful capacity assertion.

### Native corpus preflight and file safety

`corpus-manifest` scans a caller-provided, immutable regular file once in
64 KiB chunks, simultaneously computing SHA-256 and LF-delimited record/token
statistics with bounded memory. Its defaults match HDC (8,192 physical bytes
per record, 48 bytes per token, 1,024 accepted tokens per record), and custom
positive limits can be supplied. With custom limits it calculates configured
and native-HDC violations during the same read, so a looser custom limit cannot
mask a native violation. Long records are counted but excluded from token and
non-ASCII counters, as in the trainer. CR before LF counts toward the physical
bound, conservatively: some
platform formatted readers normalize CRLF, so a near-limit CRLF input may be
rejected here even if a particular trainer accepts it. A final CR without LF is
not a CRLF record in the native telemetry. The legacy Python scanner counts a
terminal CR as CRLF and, for overlong records, looks at the last retained prefix
byte rather than necessarily the last physical byte. The two CRLF telemetry
fields therefore can differ on these edges; existing legacy manifests are not
rewritten. SHA-256 and the scanner require eight-bit file storage units and the
checked signed-octet representation; they reject unsupported representations
rather than silently generating a wrong digest. Compilers lacking the required
integer kinds cannot build these modules. Control-plane finite checks recognize
binary64 only after checking representative encodings; other real formats use
an IEEE-intrinsic fallback that needs separate signaling-NaN/FPE validation.
Unsafe fast-math assumptions are outside the checked configuration. Empty lexical input is not a promise
that vocabulary construction or training will succeed.

Native `--strict-contract` rejects overlong records, dropped tokens **and
non-ASCII bytes**, matching the HDC trainer's strict ASCII policy. The legacy
`tools/corpus_manifest.py --strict-contract` does not reject non-ASCII alone,
so these two strict switches are deliberately **not equivalent**. If custom
limits are selected, `--strict-contract` requires both the configured and the
native HDC limits to pass; their results are reported separately. Unlike the
legacy tool, the native command never creates parent directories or replaces
outputs. Staging and journals are not ported. No preflight scan is inserted into a training invocation or hot
loop; running this separate command incurs the single requested read.

Report destinations and smoke checkpoints are created without replacing an
existing file. This prevents normal same-path/hard-link alias overwrites, but
standard Fortran cannot portably identify every special file, pathname alias,
symlink race, or concurrent same-size mutation. The preflight checks file size
around its read, not inode identity, mtime, locking, or an atomic snapshot;
operators must provide immutable regular files in controlled directories. A
failed write may leave an incomplete new artifact; no atomic publish or fsync
is claimed. User-supplied CLI values must fit 512 bytes and be valid UTF-8;
path/provenance values ending with a blank are rejected to avoid silently
trimming a different name. Critical checkpoints still require independent
backups: adding another checkpoint copy would violate the chosen cost boundary.

`attest` computes streaming SHA-256 identities in Fortran and refuses a
checkpoint that the native reader cannot load. `--quality-report PATH` can bind
the bytes of an already-produced report as a third identity. It intentionally
does not claim a signature, remote provenance, secret management, file lock,
or atomic publication; those transaction properties remain a separate migration
row.

The native quality gate loads the same-host checkpoint through the Fortran
reader, obtains each query's self-excluded nearest semantic neighbour, and
applies all supplied aggregate thresholds (including the 95% Wilson lower
bound). The case loader validates at most 4,096 rows in one bounded pass; each
row must be two distinct lowercase ASCII HDC tokens of at most 48 bytes separated by one
tab, with no surrounding whitespace. LF, CRLF and an unterminated last row are
accepted; a bare final CR and overlong rows are rejected. The input must remain
immutable throughout the read. Held-out status and train/case disjointness are
operator responsibilities, not verified by this gate. Its TSV case format has
no JSON schema, SHA-256/HMAC identity binding, secret handling, atomic release publication, or multi-neighbour alternatives
yet. It is a real fail-closed execution gate, but **not** a replacement claim
for the broader legacy provenance/release evaluator until those native pieces
exist and are tested.

## Build and test policy

`VEGAI_ENABLE_LEGACY_PYTHON_TESTS=OFF` is the CMake default and is made
explicit in all public CMake presets.  It leaves the native Fortran property,
profile, and train/checkpoint/load tests enabled.  Migration-era Python
integration tests can be deliberately selected with:

```bash
cmake -S . -B build-legacy -G Ninja \
  -DVEGAI_BUILD_TESTS=ON -DVEGAI_ENABLE_LEGACY_PYTHON_TESTS=ON
```

The main CI currently enables that optional switch on purpose so existing
independent reference and governed-release regression evidence remains covered
while equivalent native components are implemented.  A separate native-only
workflow targets GNU Fortran 2018 on Linux, macOS and Windows and a checked
Linux OpenMP build; defining jobs is not evidence that they passed. Its action
revisions are pinned, but hosted images and toolchain downloads are not a
hermetic/reproducible supply-chain guarantee. The currently observed test platform and
compiler are recorded in the operations audit; other combinations require an
actual successful run. Neither CI choice makes the default runtime depend on
Python.

## Source migration inventory

The following is an implementation order, not a claim that the rows below have
already been converted.

| Priority | Existing migration-era area | Native target and completion evidence |
|---|---|---|
| Complete baseline | hardware profile and capacity preflight | `src/fortran_control.f90`; native CTest/property coverage |
| Complete baseline | minimum training/checkpoint control path | `vegai_control smoke`; actual train/save/load CTest |
| Complete baseline | bounded top-1 semantic-neighbour quality policy | `src/fortran_quality_gate.f90`; strict TSV cases, explicit threshold policy, Wilson bound, native accept/reject fixture |
| Complete baseline | SHA-256 corpus/checkpoint content identity | `src/fortran_integrity.f90`; streaming FIPS vector plus native checkpoint validation |
| Partial baseline | HDC ASCII corpus manifest | `src/corpus_preflight.f90`; configurable limits, one-pass SHA-256 and stricter native/non-ASCII checks; no staging or publication transaction |
| Next | strict text/JSONL staging and deterministic shard ledger | Fortran parser plus malformed/duplicate/restart tests; not yet ported |
| Next | attestation/report identity and immutable output publication | Fortran transaction/report module; byte-level fixture, failure-injection, and cross-process locking evidence; current SHA-256 identity alone is not sufficient |
| Next | continuous-pretraining scheduler and frozen-anchor gate | Fortran derived-type journal/state machine; deliberate interruption/resume and quality-regression fixtures |
| Next | full held-out quality evaluator/release decision | versioned JSON case/policy provenance, cryptographic identity, atomic publication and replay fixtures; the current TSV aggregate gate is deliberately narrower |
| Boundary | Parquet/Arrow and OpenVINO/oneAPI | narrow `ISO_C_BINDING` adapters with SDK-present CI evidence; no pretend pure-Fortran binding |
| Boundary | live LMM/LLM provider orchestration | explicit ABI/service adapter and provider integration tests; never represented as native model training merely because the control wrapper is Fortran |

Until a row has its native module, executable test, and documented operational
boundary, its Python implementation remains legacy rather than being silently
relabelled as Fortran.

## Fortran facilities used and required

The native path uses `iso_fortran_env` kinds and standard output/error units,
modern default-initialized derived types, allocatable JSON rendering, stream
checkpoint I/O in the training core, and the established OpenMP training
implementation.  The next modules must preserve deterministic source-order
commit and checkpoint continuation semantics; OpenMP must remain an optional
acceleration, not a change in serialized model state.

External libraries are a different portability layer.  A future OpenVINO,
oneAPI, Arrow, Parquet, filesystem-capacity, cryptographic-hash, or OS-inventory
integration must enter through a small documented C ABI using `ISO_C_BINDING`
and must be tested with that dependency installed.  Standard Fortran alone
cannot honestly provide those APIs across Linux, macOS, and Windows.

## Non-negotiable invariants during migration

1. Checkpoint compatibility stays scoped to the documented same
   compiler/ABI/endianness family; stream-unformatted files are not promised as
   cross-compiler artifacts.
2. A native continuation must preserve byte-identical state where the current
   checkpoint contract requires it; a scheduler rewrite gets dedicated
   interruption/resume evidence before replacing a legacy path.
3. Quality gates stay fail closed.  Synthetic topology smoke proves execution
   and lineage, not language-model, LMM, or general-intelligence quality.
4. Child-only train timing continues to exclude corpus construction, compile,
   CI setup, and report assembly.
5. CPU is the portable baseline.  GPU/NPU utilization is reported only after a
   real device-dispatch adapter and evidence exist; discovery or an SDK import
   alone is not a compute claim.
