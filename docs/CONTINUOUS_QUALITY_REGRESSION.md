# Continuous pretraining quality-regression gate

`tools/continuous_pretrain.py` normally protects **data and transaction
integrity**: immutable shards, strict ASCII/LF ingest, exactly-once generation
journals and checkpoint SHA-256 identities. Those controls do not prove that a
continued model retained a target-domain behaviour learned by its earlier
checkpoint.

This optional gate closes that operational gap. When configured from the first
shard, a continual state may publish a checkpoint only after:

1. `tools/pretraining_quality_evaluation.py` measures the candidate against the
   same operator-owned, frozen held-out semantic-neighbour case manifest and
   absolute quality policy; and
2. `tools/continuous_quality_gate.py` independently validates the two redacted
   evaluator reports and applies an immutable anchor-to-candidate regression
   policy.

The anchor is generation 1. Therefore a later candidate is compared directly
with the first qualified checkpoint, not only with the immediately preceding
shard. This makes cumulative forgetting visible.

> **Scope boundary:** This is a narrow semantic-neighbour retrieval signal for
> exactly the supplied held-out annotations. Passing does **not** prove general
> intelligence, language fluency, factuality, safety, coding skill, multimodal
> reasoning, LLM/LMM equivalence, or production quality. The split declaration
> and annotation governance remain an upstream operator responsibility.

## Inputs and immutable binding

Supply all four options on the initial controller invocation:

```bash
python3 tools/continuous_pretrain.py \
  --pretrain-bin ./build/vegai_pretrain \
  --manifest data/continual-ledger.json \
  --state-dir state/continual \
  --max-vocab 50000 --min-frequency 1 --window 4 \
  --context-mode hierarchical --epochs 1 \
  --quality-cases /secure/evaluations/domain-cases.json \
  --quality-policy /secure/evaluations/domain-absolute-policy.json \
  --quality-redaction-key-file /secure/secrets/domain-hmac.key \
  --quality-regression-policy /secure/evaluations/domain-regression-policy.json \
  --quality-scratch-dir /fast-local-ssd/vegai-quality \
  --quality-timeout-seconds 120
```

The controller binds the following into `state.json` before native training:

- evaluator source SHA-256;
- case-manifest and absolute-policy SHA-256/byte identities, case count,
  `held-out` split declaration and explicit absolute thresholds;
- regression-policy SHA-256/byte identity and all maximum-degradation limits;
- comparator source SHA-256, case safety bound and inspect timeout; and
- the exact native training executable SHA-256/byte identity.

The redaction-key file is validated but neither its path, bytes nor a durable
hash is written to the state. It creates HMAC identifiers in the two reports.
Rotating it mid-continuation makes those identifiers differ, so comparison
fails closed rather than treating a changed redaction namespace as comparable.

On any resume of a quality-guarded state, supply the same four quality inputs.
A changed case set, absolute policy, regression policy, evaluator/comparator
source or native executable is rejected **before** a pending shard can run.
An ungated historical state cannot silently gain a gate mid-run; create a new
state/anchor instead.

## Regression policy schema

The policy is one strict JSON object with no defaults or unknown fields:

```json
{
  "schema_version": "vegai-continuous-quality-regression-policy/v1",
  "policy_id": "support-continuation-r2026-10",
  "maximum_top1_accuracy_drop": 0.02,
  "maximum_top1_wilson_lower_bound_95_drop": 0.03,
  "maximum_mean_bipolar_similarity_drop": 0.04,
  "maximum_minimum_bipolar_similarity_drop": 0.10,
  "maximum_incorrect_top1_cases_increase": 3,
  "maximum_top1_hit_regressions": 3,
  "maximum_case_similarity_drop": 0.12,
  "maximum_case_similarity_regressions": 5
}
```

| Field | Rule |
|---|---|
| `maximum_top1_accuracy_drop` | Anchor Top-1 accuracy minus candidate Top-1 accuracy; allowed range `[0,1]`. |
| `maximum_top1_wilson_lower_bound_95_drop` | Same comparison for fixed two-sided 95% Wilson lower bound; `[0,1]`. |
| `maximum_mean_bipolar_similarity_drop` | Anchor minus candidate mean Top-1 bipolar similarity; `[0,2]`. |
| `maximum_minimum_bipolar_similarity_drop` | Anchor minus candidate worst-case similarity; `[0,2]`. |
| `maximum_incorrect_top1_cases_increase` | Candidate incorrect-case count minus anchor count. |
| `maximum_top1_hit_regressions` | Cases that were Top-1 acceptable in the anchor but not in the candidate. |
| `maximum_case_similarity_drop` | A per-case similarity drop threshold; `[0,2]`. |
| `maximum_case_similarity_regressions` | Maximum count of cases whose drop exceeds the preceding threshold. |

The comparator writes actual deltas even when they are negative (an
improvement) or within tolerance. Do not set thresholds from the candidate
being selected; calibrate them on independent target-domain baseline/canary
experiments and document the risk decision.

## Publication and failures

For every candidate the controller first creates a private checkpoint. It then
runs the quality evaluator, followed by the comparator. Only a `passed`
comparison permits the candidate checkpoint to be renamed into
`generations/` and committed by atomic `state.json` replacement.

- For generation 1, the candidate is self-compared. This validates the
  absolute held-out policy and establishes the immutable anchor.
- For every later generation, the initial anchor report/checkpoint is compared
  to the private candidate.
- An evaluator absolute-policy failure or regression-policy failure leaves the
  durable generation/state unchanged. The private candidate checkpoint is
  deleted.
- Redacted quality and regression reports are retained in
  `quality-rejections/` with a `rejected-before-checkpoint-publication` record.
  They contain no raw held-out strings or HMAC key material.
- A malformed/missing/tampered report, changed source/input identity, capacity
  failure, child timeout or output-limit violation is an execution failure;
  no candidate is committed.

Accepted entries carry report paths, sizes and SHA-256 values under
`applied_shards[].quality`. `continuous_pretrain_status.py` hashes those files
on its read-only audit. Tampering an accepted report invalidates the state.

## Resource, timing, and throughput contract

Before each native child, the controller runs the portable RAM/checkpoint-space
preflight for that exact private shard/candidate. A configured quality gate also
checks scratch capacity for private evaluator copies of the binary and
checkpoint. CPU/GPU/NPU inventory is not mistaken for accelerator dispatch.

Every newly committed generation contains `training_measurement`:

- timer starts immediately before `vegai_pretrain train` or `continue` is
  spawned and stops immediately after that child exits;
- bounded stdout/stderr digests and byte counts are recorded, while raw child
  output is not put in the state;
- `raw_training_tokens_per_second` and `active_training_tokens_per_second`
  are calculated only from that child interval; and
- compilation, controller preflight, source scans, checkpoint publication,
  quality inspection/comparison, report work and CI orchestration are explicit
  exclusions.

`--omp-threads` pins `OMP_NUM_THREADS`; `OMP_PROC_BIND=close` and
`OMP_PLACES=cores` are supplied to the native child. The setting is a measured
operational choice, not a portable promise of speedup on every CPU/dual-RAM/SSD
configuration.

## Repository verification

```bash
make continuous-quality-check
# or after a CMake build:
ctest --test-dir build --output-on-failure -R continuous_quality_regression
```

The CTest verifier runs a real compiled HDC checkpoint through the complete
anchor/candidate evaluator path. A separate hermetic native-ABI fixture
intentionally forgets held-out cases and proves that the candidate is rejected,
rejection evidence remains, state stays at generation 1, and later policy or
binary mutation cannot bypass the immutable gate.

## Exact 900 MB then 90 MB controlled experiment

The manual GitHub workflow **Exact 900 MB and 90 MB continuous pretraining
quality gate** calls:

```bash
python3 tools/run_900mb_continuous_pretraining.py \
  --pretrain-bin ./build/vegai_pretrain \
  --run-dir /fast-local-ssd/vegai-900mb-then-90mb \
  --omp-threads 2 --train-timeout-seconds 2400
```

It generates exactly `900,000,000` bytes (`900MB`, decimal), then exactly
`90,000,000` bytes (`90MB`, decimal), records the two native child-only timings,
and uploads the corpus/manifests/checkpoint ledger/redacted reports as an
artifact. The generated corpus is a deterministic clustered ASCII topology
fixture. Its held-out topology annotations test this gate's mechanics and
continuation stability only; they must not be described as a general-language
or real-domain quality measurement.
