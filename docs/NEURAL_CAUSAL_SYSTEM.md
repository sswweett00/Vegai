# Native Neural Causal System: Design, Bounds, and Assurance Scope

## Status and scope

This document specifies the native Fortran path added alongside—not in place of—the existing HDC, retrieval, and bounded n-gram systems:

| Surface | What it is | What it is **not** |
|---|---|---|
| `vegai_neural_lm` | A deterministic causal inference/reference kernel for byte-BPE tokens, local attention, decayed retention, and calibrated sampling | A trained Turkish assistant or an evaluation claim |
| `vegai_neural_pretrain tokenizer-train` | A deterministic, bounded-memory merge learner for the byte-BPE codec | A claim that tokenizer training is fast at arbitrary corpus scale |
| `vegai_neural_pretrain train` | Teacher-forced, next-token adaptation of the tied output head, with checkpointing | Full BPTT through attention/retention, a foundation-model pretraining result, or a Turkish-quality benchmark |
| Existing `vegai_causal_lm` | The prior bounded approximate n-gram baseline | This neural architecture |

The implementation is native Fortran and is compiled as part of the project’s Fortran 2023 configuration. It is intentionally truthful about the optimization boundary: the current trainer updates the **tied output embedding/bias head** by cross-entropy; local attention, retention, RMS scales, gates, and FFN parameters remain frozen. That is a useful real training and resume path, but it is not presented as full-model back-propagation.

No language-quality, factuality, instruction-following, reasoning, safety, hardware-speed, GPU/NPU-dispatch, or general-intelligence result follows from these invariants.

---

## 1. Lossless byte-BPE tokenizer

### Codec model

The initial alphabet is every octet:

\[
T_0(b) = b+1,\qquad b\in\{0,\ldots,255\}.
\]

A learned merge at rank \(r\) is an ordered pair of already-defined symbols,

\[
M_r=(a_r,b_r), \qquad T_r=256+r.
\]

`src/subword_bpe.f90` stores this immutable merge DAG and a collision-checked pair index. Encoding starts from raw bytes, repeatedly applies the lowest-rank applicable pair, and uses a linked-list/min-heap implementation to avoid repeated array compaction and global rescans. The heap accepts stale candidates but validates a pair at pop time, preserving lowest-rank and leftmost-tie semantics.

Decoding recursively expands each merged symbol back to its two operands. Therefore, by induction on token construction:

- each base token decodes to exactly one original byte;
- each merge token decodes to the concatenation of two tokens that already decode exactly;
- every encoded sequence decodes to exactly the supplied byte sequence.

This is a **byte preservation** property. It does not imply that arbitrary bytes are valid UTF-8, but it does preserve all UTF-8 bytes used by Turkish text—including unseen suffixes, apostrophes, diacritics, and spelling variants—without an unknown-token fallback.

### Deterministic merge learning

`src/subword_bpe_training.f90` adds `bpe_train_utf8_file` and the CLI command:

```text
vegai_neural_pretrain tokenizer-train CORPUS TOKENIZER TARGET_VOCAB [MIN_PAIR_FREQUENCY]
```

It is deliberately an **exact-rescan** learner:

1. tokenize every retained line under the current merge table;
2. count adjacent token pairs in a collision-checked bounded hash table;
3. choose maximum frequency, breaking ties by `(left_token, right_token)`;
4. add one merge and repeat until the target vocabulary or frequency exhaustion.

Its resource model is:

\[
T = O(mB),\qquad M = O(C),
\]

where \(m\) is the number of added merges, \(B\) retained corpus bytes scanned per pass, and \(C\) configured pair-table capacity. If the table cannot hold all distinct observed pairs, training fails with `BPET_ERR_PAIR_CAPACITY`; it never silently drops statistics. This favors auditability over pretending to be a universal large-corpus tokenizer trainer.

The line adapter reads an LF-framed unformatted byte stream: it preserves every non-LF byte, including trailing spaces and CR bytes, and rejects/records overlong records instead of trimming or partially training them. The bounded codec limit is `BPE_MAX_INPUT_BYTES`. Corpus manifests, licensing, document boundaries, and exact source hashes must be recorded by the caller/workflow; tokenizer code does not infer provenance from a path.

### Tokenizer persistence

`bpe_save`/`bpe_load` serialize the ordered merge list, then rebuild the collision-safe index on load. This protects encoding semantics from internal hash-table layout changes. The current stream-unformatted artifact is a same-compiler-family ABI artifact, not a cross-architecture checkpoint portability promise.

---

## 2. Causal hybrid local-attention / retention architecture

For a token \(z_t\), let \(E\) be the tied input/output embedding matrix and \(x_t=E_{:,z_t}\). The implementation uses RMS normalization

\[
\operatorname{RMS}(x;s,\epsilon)=s\odot\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}.
\]

For each head, projections from the normalized input produce \(q_t,k_t,v_t\); rotary position transforms are applied using the bounded configured period. The model combines two strictly causal context channels.

### Exact bounded local channel

A ring cache keeps at most \(W\) earlier keys and values. The local context attends to the current key/value and cached earlier positions:

\[
\ell_t=\operatorname{softmax}\left(\frac{q_t^\top k_j}{\sqrt{d_h}}\right)_{j\in\{t-W,\ldots,t\}}V_j.
\]

No cache item is written until the current local attention and retained context have been computed.

### Decayed positive-feature retention channel

With \(\phi(u)=\operatorname{softplus}(u)>0\), memory decay \(0<\lambda<1\), and head width \(d_h\):

\[
A_t=\lambda A_{t-1}+\phi(k_t)v_t^\top,
\]
\[
b_t=\lambda b_{t-1}+\phi(k_t),
\]
\[
g_t=\frac{\phi(q_t)^\top A_t}{\phi(q_t)^\top b_t+\epsilon}.
\]

A learned per-head sigmoid gate mixes global and local context:

\[
c_t=\sigma(\gamma)g_t+[1-\sigma(\gamma)]\ell_t.
\]

The residual/FFN/output path is:

\[
r_t=x_t+W_o c_t,
\]
\[
h_t=r_t+W_2\left[\sigma(W_g\operatorname{RMS}(r_t))\odot\tanh(W_v\operatorname{RMS}(r_t))\right],
\]
\[
\text{logits}_t=E^\top h_t+b.
\]

Tying \(E\) reduces parameter duplication and makes the trainable output-head derivative transparent.

### Causality and chunk equivalence

The causal invariant is provable by induction over `neural_decoder_step`:

- Base case: the initial cache and retention tensors contain no tokens.
- Inductive step: the state before step \(t\) contains only \(z_{\le t-1}\); the step accepts only \(z_t\), computes output, then appends its key/value.
- Thus \(h_t\) and \(\text{logits}_t\) are functions only of \(z_{\le t}\), never \(z_{>t}\).

The state recurrence and ring-cache updates are deterministic functions of the previous state and current token. Processing a prefix in one call sequence or in chunks therefore produces the same state and logits, subject to the exact same real32 operation order.

`test/test_all.f90` executes both checks: identical prefixes on streams with different future suffixes have identical saved prefix logits, and a four-token prefix matches a 2+2 streaming split.

### Finite-state boundary

For bounded \(\|v_t\|\) and \(\|\phi(k_t)\|\), the decayed recurrence has a geometric envelope. Elementwise,

\[
|A_t|\le \lambda^t|A_0|+\frac{\sup|\phi(k)v|}{1-\lambda}.
\]

Additionally, \(\phi(q)^\top b_t+\epsilon\ge\epsilon>0\), so the retention division cannot have a zero denominator under finite operands.

This is a mathematical stability condition, **not** a blanket floating-point proof. An arbitrary externally corrupted/unbounded parameter tensor can still overflow finite precision. The kernel checks finite inputs/outputs and CI executes a 512-token deterministic long-stream finite-state test. It does not claim universal behavior under arbitrary IEEE exceptions or adversarial memory corruption.

---

## 3. Resource model and physical-performance boundary

Let \(V\) be vocabulary size, \(d\) width, \(H\) heads, \(d_h=d/H\), \(f\) FFN width, and \(W\) local window. `neural_decoder_plan` reports the following real32 model:

\[
P=dV+V+2d+4d^2+3df+H,
\]

covering tied embeddings, bias, two RMS scales, Q/K/V, attention output, three FFN projections, and mix gates.

Per stream, state storage is

\[
S_{\text{state}}=4\,[H d_h^2+H d_h+2dW]\ \text{bytes}.
\]

The reported dense-matmul proxy is

\[
F_{\text{dense}}\approx2(4d^2+3df+dV)
\]

FLOPs/token, with local score work reported separately as

\[
F_{\text{local-score}}\approx2H d_h(W+1).
\]

These are countable engineering models, not a throughput promise. Actual latency depends on compiler, BLAS/vectorization choices, memory bandwidth, cache behavior, CPU ISA, power limits, OS scheduling, and accelerator backend. The tied-output update has an optional OpenMP parallel loop across independent vocabulary columns (enabled only at vocabulary size \(\ge512\)); it preserves online token order and has no reduction. The native neural path otherwise runs on the portable Fortran CPU path; it does **not** silently claim OpenVINO, GPU, NPU, or joint CPU/GPU/NPU neural execution.

For a specific machine, use `vegai_neural_lm plan` to obtain the configuration-dependent architectural counts. For a trained checkpoint, `vegai_neural_pretrain plan CHECKPOINT TOKENIZER` reports a conservative sequential-training resident-memory floor: parameters, tokenizer, stream state, record token buffer, and step temporaries. It intentionally excludes compiler/runtime, allocator, OS page cache, OpenMP-team, and accelerator overhead, so it is not a RAM-sufficiency certificate. Then benchmark the exact built binary and record compiler/version, threads, thermal/power state, corpus/tokenizer, and measured wall time.

---

## 4. Repetition-aware calibrated decoding

`src/calibrated_decoder.f90` takes model logits and a causal history. For candidate \(i\), count \(f_i\), presence indicator \(I[f_i>0]\), and immediate-repeat indicator \(I[i=z_{t-1}]\), it forms

\[
\tilde z_i=\frac{z_i}{T}-\alpha f_i-\beta I[f_i>0]-\gamma I[i=z_{t-1}].
\]

A candidate that would complete an already observed no-repeat \(n\)-gram is masked before normalization. The remaining distribution is stabilized with max-subtracted exponentiation:

\[
p_i=\frac{\exp(\tilde z_i-\max_j\tilde z_j)}{\sum_j\exp(\tilde z_j-\max_j\tilde z_j)}.
\]

Top-\(k\)/top-\(p\) selection is bounded (`DECODER_MAX_TOP_K=256`) and uses a seeded deterministic RNG. Same model distribution, policy, history, and seed produce the same selected token.

Temperature calibration optimizes held-out negative log likelihood in inverse temperature \(a=1/T\):

\[
L(a)=\frac1N\sum_n\left[\log\sum_i e^{a z_{n,i}}-a z_{n,y_n}\right].
\]

\[
L''(a)=\frac1N\sum_n\operatorname{Var}_{p_a}(z_n)\ge0.
\]

The implementation uses bounded bisection and retains \(T=1\) if finite precision would make the chosen candidate worse. Thus the non-worsening result applies to the same finite held-out logit/target matrix supplied to calibration—not to future text quality, calibration drift, or arbitrary decoding policies.

---

## 5. Current teacher-forced training and continuous checkpoint contract

The implemented update is the trainable tied output head only. For softmax \(p\) and target \(y\), the derivative used is

\[
\frac{\partial L}{\partial E_{:,i}}=(p_i-\mathbb{1}[i=y])h,
\qquad
\frac{\partial L}{\partial b_i}=p_i-\mathbb{1}[i=y].
\]

`neural_pretraining` applies this update after each teacher-forced causal step, with explicit hidden-norm clipping and an explicit per-value embedding/bias envelope. Because `E` is tied, these output-head updates also affect later input embeddings; gradients through that later effect and through the causal core are intentionally not claimed.

The training update now uses the single-state Apex phase-coherent rule on each contiguous tied-output embedding column and bias value. The update uses a rational limiter, gradient clipping, decoupled embedding decay, and the configured learning-rate schedule; the causal core remains frozen. No performance or quality superiority claim is made without a fresh native Fortran run on a pinned host and corpus.

Checkpoint format **v4** stores the Apex configuration and one first-momentum array for embeddings and bias; unsupported older formats are rejected as a deliberate breaking change. Full and chunked training continue to use the persisted update counter and checkpointed state. Metrics include perplexity and top-1/top-5 ranks of the true token under the pre-update distribution; the CLI also provides a no-update `evaluate` command, `init-checkpoint`, explicit model/optimizer hyperparameters, and `tokenizer-train-fast` (an in-memory apply-merge learner verified against exact-rescan by tests). The real-Turkish orchestration now uses a document-level SHA split into train, validation, and held-out test partitions. Tokenizer and updates consume only training data; the initial and one-epoch checkpoints are compared on validation NLL, then only the selected checkpoint is evaluated on held-out test. The runner verifies manifest hashes and split identities before invoking the Fortran CLI. A governed epoch contract is documented in [`TURKISH_99B_NEURAL_PRETRAINING.md`](TURKISH_99B_NEURAL_PRETRAINING.md).

A checkpoint can be sampled through the integrated causal surface:

```text
vegai_neural_pretrain generate CHECKPOINT TOKENIZER PROMPT [TOKENS] [SEED]
```

It runs prompt tokens strictly left-to-right through the bounded state, applies the calibrated repetition-aware distribution before sampling each token, and decodes generated BPE symbols byte-for-byte. This is an inference plumbing/invariant surface, not a claim that an unbenchmarked checkpoint produces useful Turkish text.

Checkpoint payloads include every neural parameter, decoder configuration, tokenizer cardinality/merge count binding, trainer configuration, update counters, source-record cursor, and cumulative NLL. A tokenizer is stored separately with `bpe_save`. Exact continuation requires all of the following:

1. same checkpoint ABI/compiler family;
2. same loaded tokenizer merge table;
3. same attested source bytes and record order;
4. same preprocessing and bounded-record policy;
5. same training configuration;
6. resume from the recorded source position when checkpointing mid-corpus.

The library exposes explicit `skip_source_records` and `maximum_new_source_records` arguments for the last item. The corresponding checkpointed CLI is:

```text
vegai_neural_pretrain train-chunk CORPUS CHECKPOINT TOKENIZER NEW_RECORDS [START_RECORD]
```

Without an explicit `START_RECORD`, an existing checkpoint’s recorded source cursor is used. This permits deterministic bounded chunks but does **not** authenticate that a replacement file at the same path has identical bytes/order; the caller must attest that boundary. The default `resume` command remains intentionally an epoch-boundary operation rather than silently assuming a path identifies byte-identical provenance.

The native smoke test verifies only that a tiny synthetic alternating-token task lowers its evaluation NLL and that save/load reproduces next-step logits/counters exactly. The repository also contains an MIT repository-authored, eight-record UTF-8 fixture at `examples/neural_pretraining/`; its byte count and SHA-256 are CI-checked before a byte-BPE → train → deterministic-generate plumbing smoke. Neither fixture is Turkish language-quality evidence.

---

## 6. Executable assurance matrix

| Property | Evidence in repository |
|---|---|
| Turkish UTF-8 byte round trip | `test/test_all.f90`, `vegai_neural_lm self-test` |
| Deterministic bounded exact-rescan merge learning | native test on disposable corpus; `vegai_neural_pretrain tokenizer-train` CI smoke |
| Prefix/chunk state equivalence | native property test and `NEURAL_SELF_TEST_JSON` |
| Causal no-future-prefix behavior | twin-prefix/different-suffix property test |
| Finite retained state under deterministic long stream | 512-step native property test |
| Normalized masked decoder probabilities | native property test and JSON smoke output |
| No-repeat masking and same-seed sampling | native property test |
| Convex calibration non-worse supplied-set NLL | native property test |
| Tied-output toy-loss update/checkpoint round trip | `vegai_neural_pretrain smoke` in Fortran CI |
| Apex single-momentum checkpoint bit-identity (v4) + schedule anchors | native property tests in `test/test_all.f90` |
| Streamed logits ≡ transpose-matmul reference (real32 tolerance, same argmax) | native property test in `test/test_all.f90` |
| Chunked epoch ≡ single-call prediction-token consumption + full source coverage | `tools/verify_neural_turkish_epoch1.py` native micro runs |
| In-memory apply-merge ≡ exact-rescan merge sequence | native property test (same token ids on probe record) |
| Byte-BPE → two checkpointed continuous chunks → deterministic generation wiring | native Fortran CI on the immutable eight-record fixture |
| Actual Turkish quality/throughput benchmark | probe-scale evidence runs on real trwiki text (baseline + three A/Bs, `TURKISH_99B_NEURAL_PRETRAINING.md` §3.4); full-slice epoch evidence governed by the same workflow |

Run the locally available target when a Fortran compiler is present:

```bash
cmake -S . -B build -DVEGAI_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/vegai_neural_lm self-test
./build/vegai_neural_pretrain smoke
```

The Arena sandbox used for this work does not contain a local Fortran compiler, so native compilation is independently exercised by the repository’s GitHub Actions Fortran verification workflow.
