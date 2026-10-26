# Reference optimizer implementations

`ref_vector_optimizers.f90` contains isolated real64 vector implementations for SGD, SGD+momentum, Nesterov, AdaGrad, RMSprop, Adam, AdamW, scheduled NAdam, RAdam, Lion, unfactored-vector Adafactor, LAMB, LARS, Sophia-G with task-supplied exact diagonal curvature, Prodigy (original single-group D-adaptation reference, `d0=1e-6`, `beta3=sqrt(beta2)`, no bias correction), and Schedule-Free AdamW (official AdamW base, `r=0`, `weight_lr_power=2`, no inner first-moment EMA). Schedule-Free evaluation-iterate conversion is exposed separately. Variants and citations are documented beside implementations.

`ref_matrix_optimizers.f90` contains:

- Factored matrix Adafactor: row/column second-moment estimates, paper's `1-t**(-0.8)` decay, fixed external learning rate and update-RMS clipping.
- Rank-2 Shampoo: cumulative left/right Kronecker factors and symmetric inverse-fourth-root Jacobi eigensolvers.
- Muon: Nesterov matrix momentum and five Newton–Schulz polar iterations.

The sources cite variants beside implementations. Prodigy follows the original research/reference implementation ([paper](https://arxiv.org/abs/2306.06101), [official code](https://github.com/konstmish/prodigy)); Schedule-Free AdamW follows the published AdamW reference semantics ([paper](https://arxiv.org/abs/2405.15682), [official code](https://github.com/facebookresearch/schedule_free)). `make bench-ref-test` and `make bench-matrix-ref-test` are the isolated unit-test targets. Tests include first-step hand checks, matrix shape/state accounting and non-finite input rejection. No target has compiled or run in this environment; matrix numerical references remain prototypes pending compiler-backed validation.

All named optimizer families now have benchmark-only source implementations, with Adafactor's unfactored vector and factored matrix forms documented as separate shape variants. The present references are still source-level prototypes: no compiler-backed test has passed. No rival dispatch is connected to production Apex; rival code and selection remain under `bench/ref/` and benchmark drivers.
