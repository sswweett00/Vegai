# Phase 2 tuning, test evaluation and analysis status

`bench/bench_phase2_search.f90` consumes the deterministic 20-candidate grid. For each compatible task/method/seed, it trains on the training partition, chooses the lowest validation loss, and then evaluates only that selected model on the independent test partition. Non-selected candidate rows carry no test score. Per-candidate step/time caps are equal (defaults: 1,000 steps and 60 seconds); five fixed seeds are used. Matrix optimizers run on matrix factorization and exact-diagonal Sophia-G on quadratic tasks.

Sample-based tasks use deterministic disjoint 70/15/15 train/validation/test partitions. Matrix-factorization rating masks follow the same proportions. The CSV flags `validation_is_holdout` and `test_is_holdout`; both are false for analytic objectives, where the full deterministic objective is used. Test loss is not used for candidate selection.

Run `make bench-phase2-search` from the repository root, or invoke the built executable with optional `output.csv max_steps wall_clock_seconds` arguments. Default candidate output is `bench/results/phase2/validation_search.csv`.

`bench/bench_phase3_analysis.f90` audits five-seed/20-candidate completeness and validation-based selection, then computes paired test-loss summaries, seeded bootstrap intervals, exact signed-rank tests, per-task Holm-adjusted p-values, and Pareto classifications over test loss, total search seconds, and state bytes. Run `make bench-phase3-analysis` after the Phase 2 CSV exists; default analysis output is `bench/results/phase2/paired_analysis.csv`.

Fortran compilation and execution remain unverified because no Fortran compiler is installed. No candidate has been measured and no optimizer-performance conclusion is supported.
