# Fortran 2023-first migration ledger

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
| SHA-256 corpus/checkpoint identity attestation | `vegai_control attest` | FIPS `abc` vector property test; native smoke→attest CTest dependency |
| bounded semantic-neighbour quality gate | `vegai_control quality-gate` | native accept/reject property fixture in `test/test_all.f90` |
| no-Python local verification | `make fortran-check` | native property test plus profile/smoke commands |

`vegai_control` deliberately does **not** run a shell command to guess RAM,
SSD space, CPU model, GPU, or NPU.  Standard Fortran has no portable OS
inventory API.  Its resource preflight instead accepts audited
`--available-memory-bytes` and `--available-storage-bytes`; with
`--require-ready`, omitted or insufficient values fail closed.  This avoids a
misleading claim of a pure-Fortran machine probe that actually depends on
platform shell utilities.

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
  --checkpoint /safe/runtime/toy.vgai --max-vocab 128 \
  --available-memory-bytes 8589934592 \
  --available-storage-bytes 17179869184 --require-ready --format json

# Entire train/checkpoint/load smoke is performed by the compiled Fortran code.
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

# GNU Make equivalent (no Python executable is called by this target).
make fortran-check
```

The resource plan is a deterministic reservation estimate, not a prediction of
an OS allocator, page cache, NUMA policy, memory pressure, thermal throttle, or
GPU/NPU residency.  It reports its formula and never turns an unknown host
value into a successful capacity assertion.

`attest` computes streaming SHA-256 identities in Fortran and refuses a
checkpoint that the native reader cannot load. `--quality-report PATH` can bind
the bytes of an already-produced report as a third identity. It intentionally
does not claim a signature, remote provenance, secret management, file lock,
or atomic publication; those transaction properties remain a separate migration
row.

The native quality gate loads the same-host checkpoint through the Fortran
reader, obtains each query's self-excluded nearest semantic neighbour, and
applies all supplied aggregate thresholds (including the 95% Wilson lower
bound). Its TSV case format has no JSON schema, SHA-256/HMAC identity binding,
secret handling, atomic release publication, or multi-neighbour alternatives
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
while equivalent native components are implemented.  This CI choice does not
make the default runtime or native smoke dependent on Python.

## Source migration inventory

The following is an implementation order, not a claim that the rows below have
already been converted.

| Priority | Existing migration-era area | Native target and completion evidence |
|---|---|---|
| Complete baseline | hardware profile and capacity preflight | `src/fortran_control.f90`; native CTest/property coverage |
| Complete baseline | minimum training/checkpoint control path | `vegai_control smoke`; actual train/save/load CTest |
| Complete baseline | bounded top-1 semantic-neighbour quality policy | `src/fortran_quality_gate.f90`; strict TSV cases, explicit threshold policy, Wilson bound, native accept/reject fixture |
| Complete baseline | SHA-256 corpus/checkpoint content identity | `src/fortran_integrity.f90`; streaming FIPS vector plus native checkpoint validation |
| Next | corpus manifest, strict text/JSONL staging, deterministic shard ledger | Fortran stream-I/O parser plus native SHA-256 integration; malformed/duplicate/restart tests |
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
