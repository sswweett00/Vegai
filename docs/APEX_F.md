# Apex v2 — Fortran 2023

Apex is the single native optimizer used by Vegai's tied-output training path. Its implementation is split between the public API in `src/apex_optimizer.f90` and its implementation submodule `src/apex_optimizer_impl.f90`. The algorithm is experimental; no universal convergence, quality, or performance claim is made.

## State and update

Apex stores one `real64` momentum value per trainable parameter:

```text
s       = g * m_previous
m       = beta1*m_previous + (1-beta1)*g
u       = beta1*m + (1-beta1)*g
phase   = 1 + 0.1*beta2 when s > 0
          beta2         when s < 0
          1             when s = 0
Psi(u)  = u / (1 + epsilon + gamma*abs(u))
eta_eff = scheduled_lr / (1-beta1**step)
w       = (1-scheduled_lr*weight_decay)*w - eta_eff*phase*Psi(u)
```

`beta2` is a dimensionless phase-response control, not a second-moment state. Global gradient clipping is applied before the update. Scheduler choices are constant, warmup/inverse-square-root, and warmup/cosine. The schedule and update count are persisted for exact continuation.

## Public API

- `apex_config_t`, `apex_state_t`, `apex_factors_t`, `apex_error_t` are Apex-owned types.
- `apex_initialize` creates a state from finite initial parameters and validated configuration.
- `apex_update` applies one finite gradient update; `apex_optimize` accepts an abstract gradient callback through a `procedure` dummy argument.
- `apex_get_parameters` and `apex_update_count` read the state.
- `apex_save` and `apex_load` persist and restore a stream-format state; parameter and momentum arrays are serialized without numerical conversion.
- `apex_read_config` reads an `&apex ... /` Fortran namelist.
- `apex_make_factors`, `apex_update_column`, and `apex_update_scalar` are the same Apex update kernels used by the native neural trainer.

Public operations return a structured `apex_error_t`; callers can inspect its enum-backed `code` and diagnostic `message`. Invalid/non-finite inputs are rejected before update work. Kernels use Fortran `do concurrent` and make no explicit ISA assumption. BLAS is not needed for this elementwise update.

## Build and run

The project build manifest includes the module and submodule in the native Fortran library. With a Fortran 2023 compiler and OpenMP available, build the Apex benchmark using:

```sh
make apex-benchmark
build/vegai_apex_benchmark
```

The benchmark is a 10-million-element real64 synthetic objective and reports elapsed time, a modeled 5N state-stream traffic lower bound, and its final objective. Full finite-input validation and clipping add traffic beyond that lower bound. It does not compare another update system or establish general performance.

## Compatibility

Native neural checkpoint format v4 stores the Apex configuration and its single momentum state. Older checkpoint versions are deliberately rejected; they are not translated or silently resumed under changed update semantics.
