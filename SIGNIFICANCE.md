# Statistical significance

## Current status

**No hypothesis tests were run on optimizer observations.** `bench/bench_statistics.f90` now provides source-level paired summaries, seeded percentile bootstrap intervals, exact two-sided signed-rank inference through 20 nonzero pairs (normal approximation above that), Holm correction, and Pareto-front classification. Their unit tests have not compiled or run. There are no measured observations in `bench/results/`, so no empirical interval, p-value or equivalence claim can be reported. `p` is N/A, not greater than 0.05. Consequently, no superiority statement is made.

## Preregistered analysis contract

For each task × optimizer × precision × hyperparameter-selection outcome, retain per-seed observations and compare optimizers on matched task/seed pairs. Report mean and sample standard deviation; bootstrap the paired mean difference with a fixed, recorded resampling seed and at least 10,000 resamples. Use a two-sided paired Wilcoxon signed-rank test for the confirmatory comparison when its assumptions and nonzero differences permit it. Apply Holm correction across confirmatory rival comparisons within each metric/task family. Treat adjusted `p < 0.05` as evidence of a difference; otherwise label the result statistically unresolved/equivalent only if a preregistered practical-equivalence margin is also satisfied.

This protocol is design guidance only; it has not been executed and does not certify any result.
