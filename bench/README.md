# Optimizer benchmark — Phase 0 scaffold and Phase 1 baseline

Benchmark-only sources are isolated from `src/apex_optimizer*`:

- `bench/bench_contracts.f90`: trial/grid records, deterministic xorshift64/Box–Muller RNG, procedure contracts, CSV output, and result validation.
- `bench/bench_tasks.f90`: seeded configurable-condition quadratic, Rosenbrock, Rastrigin, dense/sparse logistic, analytic two-hidden-layer MLP, synthetic tiny attention/FFN language-model classification, low-rank matrix factorization and step-indexed reproducible noisy-quadratic objectives. Sample-based tasks expose full/train/validation/test evaluation; logistic/MLP/LM use deterministic contiguous 70/15/15 splits, while matrix factorization uses disjoint seeded rating masks with the same proportions. Partition tests verify the weighted train/validation/test losses and gradients reconstruct the full-data result.
- `bench/bench_tuning.f90`: deterministic exactly-20 candidate proposals per recognized optimizer label, including Apex and relevant beta/LR/epsilon/decay/rho fields, plus finite validation-loss selection.
- `bench/bench_phase2_search.f90`: train-only candidate updates followed by validation-only scoring, selection across the deterministic 20-candidate grid, five fixed seeds, and auditable candidate-level CSV rows. It searches Apex and all compatible vector references; matrix-shaped references are searched on matrix factorization, and exact-diagonal Sophia-G on quadratic tasks.
- `bench/bench_phase3_analysis.f90`: Fortran-only CSV reader that audits complete 20-candidate/five-seed grids and selected minima, then produces paired mean/SD, seeded bootstrap 95% intervals, exact Wilcoxon p-values, per-task Holm correction and loss/search-time/state-byte Pareto fronts.
- `bench/bench_statistics.f90`: paired mean/sample deviation, seeded percentile bootstrap CI, exact signed-rank through n=20 with normal approximation beyond, Holm correction and Pareto-front utility.
- `bench/ref/ref_vector_optimizers.f90`: 16 real64 vector/diagonal update variants with per-method state and adjacent citations/variant notes, including Prodigy and Schedule-Free AdamW. It also includes the remaining vector rivals through Sophia-G with exact task-supplied diagonal curvature. The unfactored Adafactor vector path is not a substitute for matrix-factored Adafactor.
- `bench/ref/ref_matrix_optimizers.f90`: factored matrix Adafactor, rank-2 Shampoo (cumulative Kronecker factors with Jacobi inverse fourth roots), and Muon (Nesterov matrix momentum plus five Newton–Schulz iterations), with independent state-byte accounting.
- `bench/bench_vector_baseline.f90`: the small Apex + 16 vector-method quadratic pilot, five seeds, 1,000 steps.
- `bench/bench_phase1_baseline.f90`: fixed-config multi-task baseline scaffold, ten task settings × five seeds; Apex plus all compatible vector rivals, and matrix references on low-rank factorization. Each trial has equal 1,000-step and 60-second caps; Sophia-G is run only on quadratic tasks with exact curvature. Intended output: `bench/results/baseline/phase1_fixed_multitask.csv`.
- `bench/test_bench_contracts.f90`, `bench/test_bench_tasks.f90`, `bench/test_bench_tuning.f90`, `bench/test_bench_statistics.f90`, and both reference test files: deterministic RNG/CSV, hand-checked objectives, finite-difference gradients, fixed tuning budgets, paired inference, Pareto utility and optimizer state checks.

Run from the repository root:

```sh
make bench-test
make bench-contract-test
make bench-task-test
make bench-tuning-test
make bench-statistics-test
make bench-phase3-test
make bench-ref-test
make bench-matrix-ref-test
make bench-vector-baseline
make bench-phase1-baseline
make bench-phase2-search
make bench-phase3-analysis
```

`make bench-test` runs the full benchmark-only test suite sequentially, avoiding concurrent writes to the shared Fortran module directory. The test commands exercise benchmark contracts, task definitions, tuning candidates, statistics and vector/matrix reference tests. `bench-phase3-test` runs an end-to-end analyzer contract using a generated temporary fixture (not benchmark evidence) to check sample-task and analytic-task holdout flags, selection auditing, paired outcomes, confidence intervals, Wilcoxon/Holm values, and Pareto classification; it also confirms the analyzer rejects an incorrect validation winner and any test score attached to a non-selected candidate. `bench-vector-baseline` is the small one-task pilot; `bench-phase1-baseline` runs the fixed-config task/seed matrix. `bench-phase2-search` performs a 20-candidate train/validation search per compatible method/task/seed, with the same step and wall-clock caps for every candidate; by default it writes `bench/results/phase2/validation_search.csv`. The executable accepts optional output path, max steps, and wall-clock seconds arguments (for example, `build/vegai_bench_phase2_search /tmp/smoke.csv 2 1`). CSV rows include every candidate and mark the validation-selected row. Only after selection does the runner evaluate that candidate once on a disjoint test partition; non-selected rows contain no test score. `validation_is_holdout` and `test_is_holdout` identify sample-based tasks with disjoint data; analytic objectives have both flags false and use the deterministic full objective. The test score is not used for tuning. After a search CSV exists, `make bench-phase3-analysis` reads it and writes `bench/results/phase2/paired_analysis.csv`; the binary also accepts explicit input and output paths. It validates full five-seed grids and selected minima before computing paired results. `make bench-apex-smoke` remains an Apex-only runner for harness troubleshooting.

## Not yet integrated / not measured

All requested method families and task families now have benchmark-only source prototypes and unit-test sources. Sophia-G requires task-supplied diagonal curvature. The synthetic LM task builds fixed seeded token embeddings and a causal single-head attention context, then evaluates a trainable FFN/softmax head; it is a compact task prototype, not a full trainable transformer stack. The fixed 128-parameter quadratic pilot still does not dispatch matrix-shaped Shampoo/Muon/Adafactor, and no full task-by-method baseline exists. Compiler-backed validation, executing the search and analysis on a Fortran compiler, real32 comparison, FLOP counting, and measured results remain outstanding. Test partitions are now evaluated only after tuning selection, but no empirical outcome has been generated. The analysis source consumes actual search output when available, but no result has been generated here. No full comparative study is claimed. The compiler is absent in this environment, so these Fortran sources have not been compiled or run; no CSV observation has been generated.
