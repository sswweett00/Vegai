# Benchmark protocol and status

## Current commands

```sh
make bench-test
make bench-contract-test
make bench-task-test
make bench-tuning-test
make bench-statistics-test
make bench-phase3-test
make bench-ref-test
make bench-matrix-ref-test
make bench-phase1-baseline
make bench-phase2-search
make bench-phase3-analysis
```

`bench-vector-baseline` remains the small Apex + 16 vector-method pilot on one well-conditioned quadratic, five seeds and 1,000 steps.

`bench-phase1-baseline` adds a fixed-configuration Fortran runner for ten task settings (including two quadratic condition numbers) and five seeds. It schedules Apex plus compatible vector methods, and runs factored Adafactor/Shampoo/Muon on low-rank factorization. Trials share 1,000-update and 60-second caps and write `bench/results/baseline/phase1_fixed_multitask.csv`. `bench-phase2-search` runs Apex and compatible references over the deterministic 20-candidate grids, trains on the training split and selects by validation loss; each candidate receives equal 1,000-update/60-second caps by default and five seeds are used. Matrix references run on matrix factorization; Sophia-G runs on quadratic tasks with exact task-supplied diagonal curvature. It writes candidate rows and marks the selected configuration in `bench/results/phase2/validation_search.csv`. The CSV marks whether validation/test are held out: sample-based logistic/MLP/LM/matrix-factorization tasks use disjoint training/validation/test observations (70/15/15); analytic objectives are marked non-held-out and scored on their full deterministic objective. Only the validation-selected candidate is evaluated on test data, after selection. `bench-phase3-analysis` verifies the five-seed/20-candidate grids and selected validation minima, then uses independent test loss for paired summaries, seeded bootstrap intervals, exact Wilcoxon tests, per-task Holm correction, and Pareto classification over test loss, full search runtime and state bytes. The runner is source-only here: no CSV or measurement exists until a compiler-backed run succeeds. `make bench-apex-smoke` remains an Apex-only harness. No benchmark example is wired through fpm.

Contract, task, tuning, statistics, vector-reference and matrix-reference tests now have source files and Make/CMake targets. `make bench-test` runs the Make test suite sequentially to avoid concurrent writes to the shared module directory. `make bench-phase3-test` adds an end-to-end integration contract: it generates a temporary five-seed/twenty-candidate CSV, runs the Phase 3 analyzer, and checks both held-out sample-task and no-holdout analytic-task flags, selection, paired inference and Pareto outputs, then verifies rejection of a wrong selected candidate and leaked non-selected test score; this generated fixture is test-only and is not benchmark evidence. None of these tests was compiled or executed in this environment.

## Implemented reference variants

The independent module `bench/ref/ref_vector_optimizers.f90` implements real64 vector variants of SGD, classical momentum, Nesterov, AdaGrad, uncentered RMSprop, Adam, AdamW, scheduled NAdam, RAdam, Lion, Adafactor's unfactored-vector branch, LAMB, layer-level LARS, Sophia-G with task-supplied exact diagonal curvature, original-reference Prodigy, and Schedule-Free AdamW with its train/eval iterate conversion. The source documents paper and reference-code variants. First-step checks and state accounting are in `bench/ref/test_ref_vector_optimizers.f90`.

The requested rival families now have source prototypes: 16 vector/diagonal variants plus factored matrix Adafactor, Shampoo and Muon. Adafactor's vector and matrix forms are shape-specific implementations of the same named competitor. Matrix methods/tests are compiler-unvalidated and are not included in the vector pilot. Benchmark dispatch is confined to `bench/`; no rival branch is added to production Apex.

## Task coverage

`bench/bench_tasks.f90` defines configurable-condition quadratics (including `κ≈10²` and `κ≈10⁶`), Rosenbrock, Rastrigin, seeded dense/sparse logistic regression, an analytic two-hidden-layer MLP, low-rank matrix factorization, step-indexed noisy quadratic gradients, and a compact synthetic next-token task using fixed seeded token embeddings/causal attention features with a trainable FFN/softmax head. Unit tests include hand-checked classical objectives, seed replay, noise replay and finite-difference checks of the analytic MLP/LM/matrix-factorization gradients. Only one well-conditioned quadratic is connected to the pilot; the task tests have not compiled or run. The LM task is a scaffold, not a full trainable transformer.

A complete experiment requires each task × optimizer × precision × seed cell to use at least five preregistered seeds, equal max-step and wall-time caps, equal 20-trial tuning budgets, IEEE finite checks after every step, and logged final/train/validation metrics, steps, elapsed time, FLOPs, optimizer state bytes, compiler/hardware/configuration hashes. Report mean±std and confidence intervals, then apply the procedure in `SIGNIFICANCE.md`.

## Evidence and machine

No benchmark command could compile. Benchmark build attempts stop with `gfortran: No such file or directory`; this blocks the tests, Phase 1/2 runners and Phase 3 analyzer. No executable was measured. `make test` is likewise blocked. Host reports x86_64 and `Intel(R) Xeon(R) Processor @ 2.60GHz`; no compiler version or `-O` flags are available. No empirical timings, task results or optimizer rankings are reported.

## Memory accounting

Apex's declared optimizer state is one real64 momentum value per parameter: **8 bytes/parameter**, excluding allocator/metadata overhead. The benchmark counts reference optimizer arrays by active state; it does not count model parameters as optimizer state. These are declared-payload calculations, not RSS measurements. FLOPs and milliseconds per step are unmeasured.

## Limits

No BLAS/LAPACK implementation or fallback pair has been tested. No result exists for scale, generalization, sparse/noisy gradients, warm-start transfer, precision sensitivity, hyperparameter robustness, ablation or significance. Pareto status is unknown. Do not use the pilot harness as evidence that Apex or a rival is superior.
