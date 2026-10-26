# Phase 1 fixed multi-task baseline

`make bench-phase1-baseline` is intended to generate `bench/results/baseline/phase1_fixed_multitask.csv`. It schedules the five preregistered seeds across ten task settings, runs Apex plus the 16 vector/diagonal reference variants (Sophia-G only where the objective provides exact diagonal curvature), and runs factored Adafactor, Shampoo and Muon on the low-rank factorization task. Each trial is capped at 1,000 updates and 60 seconds, real64 only, at a fixed shared configuration.

This is an initial baseline harness, not the full comparative study: no hyperparameter search, real32, held-out split, FLOP instrumentation, bootstrap/Wilcoxon tests or Pareto analysis are performed. No CSV observations exist until the Fortran target compiles and runs.
