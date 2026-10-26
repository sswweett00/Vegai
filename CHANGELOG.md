# Changelog

## Unreleased — Apex v2

- **Removed:** multi-optimizer selection, the legacy gradient-only update path, and the two-moment state/checkpoint representation.
- **Added/Changed:** Apex is now the sole native neural-training update system, with one momentum array, phase-coherent modulation, rational limiting, gradient clipping, weight decay, and configurable scheduling.
- **Added:** Fortran 2023 `apex_optimizer` public module plus implementation submodule, structured error type and enum codes, abstract gradient callback, generic update API, runtime namelist loading, and stream-format state persistence.
- **Changed:** neural checkpoints are version 4 and store the Apex configuration plus single-momentum state. Earlier checkpoint versions are rejected; no compatibility shim is provided.
- **Added:** Apex unit, quadratic convergence, logistic-regression, finite-input, and exact save/load round-trip tests; an executable example and standalone benchmark.
- **Added (benchmark-only):** Phase 0 deterministic task prototypes/tests (conditioned quadratics, Rosenbrock, Rastrigin, dense/sparse logistic, MLP, compact fixed-attention LM, low-rank matrix factorization, noisy gradients) and Prodigy/Schedule-Free reference prototypes. Added a Phase 1 fixed-config multi-task runner with five seeds, equal step/wall limits, vector/matrix dispatch and CSV output. These sources have not compiled or run because this environment has no Fortran compiler; no comparative result is claimed.
- **Added (neural training quality control):** the Turkish corpus manifest now assigns source documents to train/validation/held-out-test partitions; tokenizer/model updates consume train only, the initial versus one-epoch checkpoint is selected by validation NLL, and held-out test evaluation runs only after selection. Selected-checkpoint publication is atomic, and evidence binds the three partition hashes.
- **Breaking:** training CLI no longer accepts an optimizer-selection flag or previous optimizer-specific parameter names. Use `--beta1`, `--beta2`, `--epsilon`, `--gamma`, `--weight-decay`, `--grad-clip`, and `--schedule`.
