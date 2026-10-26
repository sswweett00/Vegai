# Gap analysis — comparative optimizer study

## Evidence status

No comparative observations exist in `bench/results/`. The environment has no Fortran compiler, so the fixed-configuration pilot is unexecuted. Every Δ, confidence interval and p-value below is **not measured**, not zero and not an equality claim. No winner or Pareto status is inferred from formulas or source review.

## Implemented reference coverage

Benchmark-only real64 vector/diagonal variants in `bench/ref/ref_vector_optimizers.f90` now include the requested SGD, momentum, Nesterov, AdaGrad, RMSprop, Adam, AdamW, NAdam, RAdam, Lion, Adafactor-vector, LAMB, LARS, Sophia-G, Prodigy and Schedule-Free AdamW families. Factored matrix Adafactor, rank-2 Shampoo and Muon prototypes are in `bench/ref/ref_matrix_optimizers.f90`. Formula notes, citations, hand-step tests and schedule-free train/eval conversion are present. No compiler tests have run; the references remain unvalidated source prototypes.

## Dimension × rival status

Every cell for every rival is `N/M` (not measured). Rival columns are: SGD; SGD+momentum; Nesterov; AdaGrad; RMSprop; Adam; AdamW; NAdam; RAdam; Lion; Adafactor; LAMB; LARS; Sophia; Shampoo; Muon; Prodigy; Schedule-Free AdamW.

| Dimension | Δ Apex−rival | 95% CI | p | Pareto status | Gap/evidence required |
|---|---:|---|---:|---|---|
| Convergence steps/time | N/M | N/M | N/M | N/M | Matched objective and stopping rule |
| Final loss / accuracy | N/M | N/M | N/M | N/M | Full task-specific held-out metrics |
| Optimizer state bytes/parameter | N/M | N/M | N/M | N/M | Exact method-specific and measured state layouts |
| FLOPs / step and wall time | N/M | N/M | N/M | N/M | Instrumented kernels, equal hardware/compiler |
| Generalization gap | N/M | N/M | N/M | N/M | Train/validation splits and repeated seeds |
| Numerical stability | N/M | N/M | N/M | N/M | Per-step finite checks and failure counts |
| Hyperparameter robustness | N/M | N/M | N/M | N/M | Equal 20-trial search budget per rival/task |
| Scale 10³→10⁷ | N/M | N/M | N/M | N/M | Multiple fixed parameter-count runs |
| Noise/sparsity tolerance | N/M | N/M | N/M | N/M | Matched noisy and sparse gradient generation |
| Warm-start/transfer | N/M | N/M | N/M | N/M | Saved state and equal continuation protocol |
| real32 vs real64 | N/M | N/M | N/M | N/M | Separate deterministic runs and finite/error checks |

## Outstanding gaps

1. Compiler/runtime unavailable: no binary or measurements.
2. Requested optimizer families now have benchmark-only source prototypes, but they have not been compiled and are not validated against independent golden implementations. The fixed quadratic pilot dispatches only vector methods; Phase 2 source dispatches matrix Shampoo/Muon/Adafactor on low-rank factorization.
3. Objective/gradient definitions and test sources cover conditioned quadratics, Rosenbrock, Rastrigin, dense/sparse logistic, synthetic two-hidden-layer MLP, low-rank matrix factorization, indexed noisy gradients and a compact fixed-attention/trained-FFN LM task. Only the well-conditioned quadratic is connected to the pilot; the LM task is not a full trainable transformer.
4. Phase 2 now adds disjoint test partitions and evaluates only the validation-selected model on test data; Phase 3 uses test loss as the paired outcome. This pipeline is source-only and has not compiled or run.
5. A Phase 3 Fortran analyzer consumes independent test-loss outcomes and computes paired summaries, bootstrap intervals, Wilcoxon, Holm and Pareto outputs, with grid/selection integrity checks. It has not compiled or run because the compiler is unavailable; no observations or empirical Pareto conclusions exist.
6. Apex was not changed in this phase; redesign waits for a valid benchmark baseline.
