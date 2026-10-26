# Discovery — Apex and comparative benchmark

## Repository/baseline state

The active branch already contains the Apex v2 implementation from the preceding task. At the start of this benchmark request, the worktree had uncommitted changes and untracked Apex/test/documentation files; they are intentionally retained. No change to `src/apex_optimizer.f90` or `src/apex_optimizer_impl.f90` has been made for the current benchmark request.

Build surfaces: GNU Make (`Makefile`), CMake (`CMakeLists.txt`, Fortran standard 2023), and fpm (`fpm.toml`, automatic executable/test discovery). The Apex module and submodule are already ordered before `neural_pretraining` in Make/CMake. Apex tests are in `test/test_apex.f90`; the earlier project-level test attempt was blocked by the missing `gfortran` executable.

## Apex API map

- `apex_config_t`: `real64` learning rate, beta1/beta2, epsilon, gamma, weight decay, clipping norm, warmup/total update counts, scheduler enum and minimum LR ratio.
- `apex_state_t`: private `real64` parameter and single momentum arrays, persisted update counter and config.
- `apex_factors_t`: precomputed effective rate, decoupled decay factor and update scalars used by the neural trainer's column/scalar kernels.
- `apex_error_t`: enum-backed error code and fixed diagnostic string.
- Public procedures/generic: config validation, initialization, state update, `apex_apply` generic, callback `apex_optimize`, factors construction, mixed-precision column/scalar update kernels, parameter/count accessors, stream save/load, and namelist config read.
- Implementation uses IEEE finite checks, `do concurrent`, module/submodule separation, and real64 state. Neural training stores one real64 momentum value for each tied-output parameter; the model weights remain real32.
- Existing test coverage includes quadratic and logistic examples, invalid NaN/Inf inputs, namelist reading and bit-exact Apex state round-trip. None of those tests has yet been executed with a Fortran compiler in this environment.

## Benchmark scaffold added in this phase

`bench/bench_contracts.f90` defines trial/grid records, deterministic xorshift64/Box–Muller seed utilities, procedure contracts, CSV serialization and result validation. `bench/bench_tasks.f90` defines conditioned quadratics, Rosenbrock, Rastrigin, dense/sparse logistic, analytic MLP, compact seeded attention/FFN LM, low-rank matrix factorization, and repeatable indexed noisy gradients, with finite-difference and hand-value tests. `bench/ref/ref_vector_optimizers.f90` now contains 16 source-noted vector/diagonal variants, including Prodigy and Schedule-Free AdamW; `bench/ref/ref_matrix_optimizers.f90` adds factored Adafactor, Shampoo and Muon with matrix tests. None has compiler-backed verification; Prodigy and Schedule-Free AdamW source prototypes/tests now exist but remain unvalidated. `bench/bench_vector_baseline.f90` remains the five-seed, 1,000-step quadratic pilot. New `bench/bench_phase1_baseline.f90` loops ten task settings and five seeds, dispatches Apex plus compatible vector methods and matrix methods on factorization, with matched 1,000-step/60-second caps. Phase 2 now includes `bench/bench_phase2_search.f90`: Apex and compatible references train on a deterministic 70/15/15 train/validation/test split where observations exist, choose only by validation loss, and test-score only the selected candidate. The grid is equal per candidate and uses five fixed seeds. Phase 3 `bench/bench_phase3_analysis.f90` audits the search CSV and computes paired/bootstrap/Wilcoxon/Holm/Pareto summaries using test-loss outcomes. The output CSVs do not exist yet; no benchmark compiler is installed, so the runner and analyzer have not been compiled or run. Make targets include `bench-phase2-search` and `bench-phase3-analysis`; optional CMake targets are enabled with `VEGAI_BUILD_BENCHMARKS=ON`.

This remains **not** the required comparative benchmark: all requested families now have source-level prototypes, but compiler validation and independent numerical verification are unavailable. Phase 2 source performs train/validation search and evaluates only the selected candidate on an independent test partition for sample-based tasks; Phase 3 source analyzes test outcomes. Neither has been executed. Multiple-precision runs, FLOP counting and statistics on actual observations remain outstanding. Result directories contain explanatory README files, not observations.

## Environment and baseline attempt

The available host reports `Intel(R) Xeon(R) Processor @ 2.60GHz` on x86_64. Compiler/build probes find no `gfortran`, `ifx`, `ifort`, `flang`, `flang-new`, `lfortran`, `fpm` or `cmake` in PATH. A supported compiler version, optimization flags and run-time measurements therefore cannot be reported. `make bench-contract-test`, `make bench-task-test`, `make bench-tuning-test`, `make bench-statistics-test`, `make bench-ref-test`, `make bench-matrix-ref-test`, `make bench-apex-smoke`, `make bench-vector-baseline` and `make bench-phase1-baseline` reached GNU compile recipes and stopped with `gfortran: No such file or directory`; no test/benchmark executable ran.

## Scope choice

The request requires all 18 rivals, the requested task families, 20-trial tuning, at least five seeds per cell, multiple precision/statistical dimensions and reproducible measured results. Phase 0 sources, a Phase 1 fixed-config runner, and a Phase 2 validation-selected search runner now exist in source. Without a Fortran compiler, no prototype/test can be compiler-verified and no empirical baseline exists; multiple precision and statistical analysis of actual results remain open, so the research sequence remains **in progress**, not complete. Apex core work remains held.
