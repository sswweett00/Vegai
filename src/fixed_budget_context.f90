! SPDX-License-Identifier: MIT
!==============================================================================
!  fixed_budget_context.f90
!
!  A causal context engine whose *physical* resources and per-token work are
!  independent of the logical context horizon.
!
!  The problem with the previous decoder
!  -------------------------------------
!  In src/neural_retention_decoder.f90 `--window` becomes `config%local_window`
!  and then: (a) the K/V cache is allocated (head_width, heads, local_window),
!  (b) `local_attention` loops over up to `local_window` cached entries for
!  every head of every token, (c) `scores(local_window + 1)` is an automatic
!  array built on every step, and (d) both resource-plan counters carry a
!  `window` factor.  So the flag the user reads as "context horizon" is in fact
!  a physical buffer length: raising it raises state bytes and per-token time.
!
!  The separation used here
!  ------------------------
!    * `logical_horizon` (int64) only sets the *decay constants* of the
!      multi-scale recurrent memory.  It appears in no array bound, no
!      allocation, no loop count after `fbc_model_init` has run.
!    * `local_window` (int32, fixed physical ring buffer) bounds the exact
!      causal attention.
!    * `memory_scales` and `memory_slots` are fixed counts decided at
!      construction.
!
!  Three context channels, mixed by a learnable per-channel softmax:
!
!    A. Local, exact, causal.  Ring buffer of the last `local_window` keys and
!       values; scores carry a learnable relative-distance bias.  Cost is
!       O(local_window * width) per token, independent of the horizon.
!    B. Multi-scale recurrent memory.  J fixed scales with
!           R_j <- lambda_j R_j + (1 - lambda_j) phi(k) v^T
!           z_j <- lambda_j z_j + (1 - lambda_j) phi(k)
!           read_j = phi(q)^T R_j / (phi(q)^T z_j + eps)
!       lambda_j is derived from the horizon *once*, at construction, from a log
!       spacing between `local_window` and `logical_horizon`.  Nothing in the
!       per-token path recomputes a transcendental; phi is softplus and is
!       evaluated on the projected key/query only.
!    C. Fixed-capacity slots.  M slots with a key, a value and a scalar
!       salience.  Write-then-read: the current token writes into the slots
!       (gated by a sigmoid of the key/slot similarity) and the same token then
!       reads from the updated slots.  That ordering is deliberate: it puts the
!       write gate on the differentiable path of the current prediction, so the
!       write side really is trained instead of being frozen.
!
!  Memory honesty
!  --------------
!  The recurrent memory is a lossy, fixed-size summary.  A horizon of 1e12 does
!  not mean a token from 1e12 steps ago is recoverable: it means the slowest
!  scale has a decay time of 1e12 steps *in real64*, and that the state still
!  carries that token's contribution weighted by exp(-delta/1e12).  Two hard
!  limits are computed and reported rather than hidden:
!    * real64 cannot represent 1 - 1/tau below eps/2, which caps any reachable
!      decay time at ~4.5e15 steps (FBC_MAX_HORIZON);
!    * the slowest scale degenerates to an unweighted running sum once
!      1/tau drops under eps, so horizons beyond the cap are clamped and the
!      clamp is reported (`horizon_clamped`, `achievable_horizon`).
!
!  Everything is plain Fortran: no C binding, no external library, no OpenMP
!  directive.  Kernels are column-major (an expert's matrix column is the unit
!  stride) and use the intrinsic MATMUL/DOT_PRODUCT forms, which is what the
!  project measured to be fastest on contiguous operands.
!==============================================================================
module fixed_budget_context
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   implicit none (type, external)
   private

   integer(int32), parameter, public :: FBC_OK = 0_int32
   integer(int32), parameter, public :: FBC_ERR_INVALID_ARGUMENT = -1401_int32
   integer(int32), parameter, public :: FBC_ERR_ALLOCATION = -1402_int32
   integer(int32), parameter, public :: FBC_ERR_NOT_READY = -1403_int32
   integer(int32), parameter, public :: FBC_ERR_NUMERIC = -1404_int32
   integer(int32), parameter, public :: FBC_ERR_ORDER = -1405_int32

   ! Context channels, in the order their logits are stored.
   integer(int32), parameter, public :: FBC_CHANNEL_LOCAL = 1_int32
   integer(int32), parameter, public :: FBC_CHANNEL_RETENTION = 2_int32
   integer(int32), parameter, public :: FBC_CHANNEL_SLOTS = 3_int32
   integer(int32), parameter, public :: FBC_CHANNEL_COUNT = 3_int32

   ! Largest decay time real64 can still distinguish from "no decay" once a
   ! repeatable scale j has lambda = exp(-1/tau): 1/tau must stay above eps/2.
   real(real64), parameter, public :: FBC_MAX_HORIZON = 4.0e15_real64
   real(real64), parameter, public :: FBC_MIN_HORIZON = 1.0_real64

   integer(int32), parameter, public :: FBC_MAX_LOCAL_WINDOW = 4096_int32
   integer(int32), parameter, public :: FBC_MAX_MEMORY_SCALES = 16_int32
   integer(int32), parameter, public :: FBC_MAX_MEMORY_SLOTS = 1024_int32
   integer(int32), parameter, public :: FBC_MAX_CHUNK_TOKENS = 8192_int32

   !> Static description of the engine.  Note which fields are *semantic*
   !> (horizon) and which are *physical* (window, scales, slots).
   type, public :: fbc_config_t
      integer(int32) :: vocabulary_size = 256_int32
      integer(int32) :: width = 64_int32
      integer(int32) :: heads = 4_int32
      integer(int32) :: ffn_width = 128_int32
      ! Physical ring buffer for the exact causal attention.
      integer(int32) :: local_window = 32_int32
      ! Fixed number of decay scales and of information slots.
      integer(int32) :: memory_scales = 4_int32
      integer(int32) :: memory_slots = 64_int32
      ! Semantic reach, in tokens.  int64 because 1e12 does not fit an int32.
      integer(int64) :: logical_horizon = 32_int64
      real(real64) :: rms_epsilon = 1.0e-6_real64
      ! Decay of the slot salience accumulator (a fixed hyper-parameter, not a
      ! horizon function: it describes how fast "importance" is forgotten).
      real(real64) :: slot_decay = 0.99_real64
      real(real64) :: slot_temperature = 1.0_real64
   end type fbc_config_t

   !> One stream's recurrent state.  Every extent here is either a physical
   !> constant of the configuration or derived from it -- never from the horizon.
   type, public :: fbc_state_t
      logical :: initialized = .false.
      integer(int32) :: width = 0_int32
      integer(int32) :: heads = 0_int32
      integer(int32) :: head_width = 0_int32
      integer(int32) :: local_window = 0_int32
      integer(int32) :: memory_scales = 0_int32
      integer(int32) :: memory_slots = 0_int32
      integer(int32) :: cache_head = 0_int32
      integer(int32) :: cache_count = 0_int32
      integer(int64) :: steps = 0_int64
      real(real64), allocatable :: retention_numerator(:, :, :, :)   ! (dh, dh, head, scale)
      real(real64), allocatable :: retention_normalizer(:, :, :)     ! (dh, head, scale)
      real(real64), allocatable :: local_key(:, :, :)                ! (dh, head, window)
      real(real64), allocatable :: local_value(:, :, :)              ! (dh, head, window)
      real(real64), allocatable :: slot_key(:, :)                    ! (width, slots)
      real(real64), allocatable :: slot_value(:, :)                  ! (width, slots)
      real(real64), allocatable :: slot_salience(:)                  ! (slots)
   end type fbc_state_t

   !> All learnable arrays.  The layout mirrors fbc_gradient_t one-to-one so the
   !> optimizer can be a single generic loop over both.
   type, public :: fbc_model_t
      type(fbc_config_t) :: config
      logical :: initialized = .false.
      integer(int32) :: head_width = 0_int32
      real(real64) :: effective_horizon = 0.0_real64
      logical :: horizon_clamped = .false.
      real(real64), allocatable :: scale_decay(:)                    ! (scale) lambda_j
      real(real64), allocatable :: token_embedding(:, :)             ! (width, vocab)
      real(real64), allocatable :: output_bias(:)                    ! (vocab)
      real(real64), allocatable :: query(:, :, :)                    ! (dh, width, head)
      real(real64), allocatable :: key(:, :, :)
      real(real64), allocatable :: value(:, :, :)
      real(real64), allocatable :: attention_output(:, :)            ! (width, width)
      real(real64), allocatable :: attention_rms(:)                  ! (width)
      real(real64), allocatable :: ffn_rms(:)                        ! (width)
      real(real64), allocatable :: ffn_gate(:, :)                    ! (ffn, width)
      real(real64), allocatable :: ffn_value(:, :)                   ! (ffn, width)
      real(real64), allocatable :: ffn_output(:, :)                  ! (width, ffn)
      real(real64), allocatable :: distance_bias(:)                  ! (local_window)
      real(real64), allocatable :: scale_logit(:)                    ! (scale)
      real(real64), allocatable :: channel_logit(:)                  ! (3)
      real(real64), allocatable :: slot_query_proj(:, :)             ! (width, width)
      real(real64), allocatable :: slot_key_proj(:, :)               ! (width, width)
      real(real64), allocatable :: slot_value_proj(:, :)             ! (width, width)
      real(real64), allocatable :: slot_read_proj(:, :)              ! (width, width)
      real(real64) :: slot_salience_weight = 0.0_real64
   end type fbc_model_t

   !> Per-token activations.  Sized by the configuration only; the trainer reuses
   !> one of these for the whole run, so a token allocates nothing.
   type, public :: fbc_workspace_t
      logical :: initialized = .false.
      integer(int32) :: local_count = 0_int32
      integer(int32) :: token = 0_int32
      real(real64), allocatable :: x(:)                 ! (width)
      real(real64), allocatable :: normalized(:)        ! (width)
      real(real64), allocatable :: q(:, :), k(:, :), v(:, :)   ! (dh, head)
      real(real64), allocatable :: phi_q(:, :), phi_k(:, :)    ! (dh, head)
      real(real64), allocatable :: read_denominator(:, :)      ! (head, scale)
      real(real64), allocatable :: read_scale(:, :, :)         ! (dh, head, scale)
      real(real64), allocatable :: scale_alpha(:)              ! (scale)
      real(real64), allocatable :: local_weight(:, :)          ! (window+1, head), 1 = self
      integer(int32), allocatable :: local_position(:)         ! (window)
      real(real64), allocatable :: local_context(:, :)         ! (dh, head)
      real(real64), allocatable :: memory_context(:, :)        ! (dh, head)
      real(real64), allocatable :: slot_query(:)               ! (width)
      real(real64), allocatable :: slot_write_key(:)           ! (width)
      real(real64), allocatable :: slot_write_value(:)         ! (width)
      real(real64), allocatable :: slot_write_gate(:)          ! (slots)
      real(real64), allocatable :: slot_key_delta(:, :)        ! (width, slots)
      real(real64), allocatable :: slot_value_delta(:, :)      ! (width, slots)
      real(real64), allocatable :: slot_read_score(:)          ! (slots)
      real(real64), allocatable :: slot_read_weight(:)         ! (slots)
      real(real64), allocatable :: slot_read(:)                ! (width)
      real(real64), allocatable :: slot_context(:)             ! (width)
      real(real64), allocatable :: channel_alpha(:)            ! (3)
      real(real64), allocatable :: channel_context(:, :)       ! (width, 3)
      real(real64), allocatable :: combined(:)                 ! (width)
      real(real64), allocatable :: attention_residual(:)       ! (width)
      real(real64), allocatable :: rhat_attention(:)           ! (width)
      real(real64), allocatable :: rhat_ffn(:)                 ! (width)
      real(real64) :: inverse_rms_attention = 0.0_real64
      real(real64) :: inverse_rms_ffn = 0.0_real64
      real(real64), allocatable :: ffn_pre_gate(:), ffn_pre_value(:), ffn_gated(:)
      real(real64), allocatable :: hidden(:)                   ! (width)
   end type fbc_workspace_t

   !> Structural resource accounting.  Every number here is a function of the
   !> configuration *except* the ones explicitly labelled with the horizon.
   type, public :: fbc_plan_t
      integer(int64) :: parameter_count = 0_int64
      integer(int64) :: parameter_bytes_real64 = 0_int64
      integer(int64) :: state_bytes_real64 = 0_int64
      integer(int64) :: local_cache_elements = 0_int64
      integer(int64) :: retention_state_elements = 0_int64
      integer(int64) :: slot_state_elements = 0_int64
      integer(int64) :: forward_flops_per_token = 0_int64
      integer(int64) :: backward_flops_per_token = 0_int64
      integer(int64) :: local_score_operations_per_token = 0_int64
      integer(int32) :: memory_scales = 0_int32
      integer(int32) :: memory_slots = 0_int32
      integer(int32) :: local_window = 0_int32
      integer(int64) :: logical_horizon = 0_int64
      real(real64) :: effective_horizon = 0.0_real64
      logical :: horizon_clamped = .false.
   end type fbc_plan_t

   interface fbc_fill
      module procedure fbc_fill_1d, fbc_fill_2d, fbc_fill_3d
   end interface

   public :: fbc_config_is_valid
   public :: fbc_model_init, fbc_model_destroy, fbc_model_is_ready
   public :: fbc_state_init, fbc_state_destroy, fbc_state_reset, fbc_state_is_ready
   public :: fbc_workspace_init, fbc_workspace_destroy
   public :: fbc_plan, fbc_step, fbc_sigmoid, fbc_softplus, fbc_softplus_slope, fbc_parameter_count
   public :: fbc_configure_scales, fbc_state_bytes, fbc_parse_horizon, fbc_parse_count

contains

   !----------------------------------------------------------------------------
   !  Small numerically safe kernels
   !----------------------------------------------------------------------------

   pure real(real64) function fbc_sigmoid(value) result(transformed)
      real(real64), intent(in) :: value

      if (value >= 0.0_real64) then
         transformed = 1.0_real64 / (1.0_real64 + exp(-value))
      else
         transformed = exp(value) / (1.0_real64 + exp(value))
      end if
   end function fbc_sigmoid

   pure real(real64) function fbc_softplus(value) result(transformed)
      real(real64), intent(in) :: value

      if (value > 30.0_real64) then
         transformed = value
      else if (value < -30.0_real64) then
         transformed = exp(value)
      else
         transformed = log(1.0_real64 + exp(value))
      end if
   end function fbc_softplus

   pure real(real64) function fbc_softplus_slope(value) result(slope)
      real(real64), intent(in) :: value

      slope = fbc_sigmoid(value)
   end function fbc_softplus_slope

   !> Stable softmax of a small vector; the caller guarantees size >= 1.
   pure subroutine fbc_softmax(values, probabilities)
      real(real64), intent(in) :: values(:)
      real(real64), intent(out) :: probabilities(:)
      real(real64) :: maximum, normalizer
      integer(int32) :: index

      maximum = maxval(values)
      normalizer = 0.0_real64
      do index = 1_int32, size(values, kind=int32)
         probabilities(index) = exp(values(index) - maximum)
         normalizer = normalizer + probabilities(index)
      end do
      if (normalizer > 0.0_real64 .and. ieee_is_finite(normalizer)) then
         probabilities = probabilities / normalizer
      else
         probabilities = 1.0_real64 / real(size(values, kind=int32), real64)
      end if
   end subroutine fbc_softmax

   !----------------------------------------------------------------------------
   !  Validation and configuration
   !----------------------------------------------------------------------------

   pure logical function fbc_config_is_valid(config) result(valid)
      type(fbc_config_t), intent(in) :: config
      integer(int32) :: head_width

      valid = .false.
      if (config%heads < 1_int32 .or. config%width < 2_int32) return
      head_width = config%width / config%heads
      if (modulo(config%width, config%heads) /= 0_int32) return
      if (head_width < 1_int32) return
      if (config%vocabulary_size < 2_int32) return
      if (config%local_window < 1_int32 .or. config%local_window > FBC_MAX_LOCAL_WINDOW) return
      if (config%memory_scales < 1_int32 .or. config%memory_scales > FBC_MAX_MEMORY_SCALES) return
      if (config%memory_slots < 1_int32 .or. config%memory_slots > FBC_MAX_MEMORY_SLOTS) return
      if (config%logical_horizon < 1_int64) return
      if (config%ffn_width < config%width) return
      if (config%rms_epsilon <= 0.0_real64) return
      if (config%slot_decay < 0.0_real64 .or. config%slot_decay >= 1.0_real64) return
      if (config%slot_temperature <= 0.0_real64) return
      valid = .true.
   end function fbc_config_is_valid

   !> Derives the per-scale decay constants from the logical horizon.
   !>
   !> tau_j is log-spaced between the physical window (the shortest useful
   !> reach) and the requested horizon.  lambda_j = exp(-1/tau_j) is the
   !> per-token retention, so one scale is memory of "the last few dozen
   !> tokens" and the last one carries the requested reach.
   !>
   !> This is the only place the horizon is read.  After it runs, the horizon is
   !> a number in `model%effective_horizon` and nothing else.
   pure subroutine fbc_configure_scales(config, decay, effective_horizon, clamped)
      type(fbc_config_t), intent(in) :: config
      real(real64), intent(out) :: decay(config%memory_scales)
      real(real64), intent(out) :: effective_horizon
      logical, intent(out) :: clamped
      real(real64) :: shortest, longest, ratio, log_ratio, tau
      integer(int32) :: scale

      shortest = real(max(1_int32, config%local_window), real64)
      longest = real(config%logical_horizon, real64)
      clamped = .false.
      if (longest > FBC_MAX_HORIZON) then
         longest = FBC_MAX_HORIZON
         clamped = .true.
      end if
      if (longest < shortest) longest = shortest
      effective_horizon = longest
      if (config%memory_scales == 1_int32) then
         decay(1_int32) = exp(-1.0_real64 / longest)
         return
      end if
      log_ratio = log(longest / shortest)
      ratio = real(config%memory_scales - 1_int32, real64)
      do scale = 1_int32, config%memory_scales
         tau = exp(log(max(shortest, tiny(1.0_real64))) + log_ratio * real(scale - 1_int32, real64) / ratio)
         ! exp(-1/tau) can round to exactly 1.0 for tau beyond the reachable
         ! limit; keep a representable decrement instead of silently losing the
         ! decay.
         decay(scale) = min(1.0_real64 - epsilon(1.0_real64), exp(-1.0_real64 / tau))
      end do
   end subroutine fbc_configure_scales

   !> Strict positive int64 parser for the logical horizon.  Rejects empty,
   !> signed, non-numeric, zero and overflowing inputs instead of wrapping.
   pure subroutine fbc_parse_horizon(text, value, status)
      character(len=*), intent(in) :: text
      integer(int64), intent(out) :: value
      integer(int32), intent(out) :: status
      integer(int32) :: index, digit
      integer(int64), parameter :: limit = huge(0_int64)
      integer(int64) :: accumulator

      value = 0_int64
      status = FBC_OK
      if (len_trim(text) == 0) then
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      accumulator = 0_int64
      do index = 1_int32, len_trim(text)
         digit = iachar(text(index:index)) - iachar('0')
         if (digit < 0_int32 .or. digit > 9_int32) then
            status = FBC_ERR_INVALID_ARGUMENT
            value = 0_int64
            return
         end if
         ! Overflow is checked before it happens: accumulator*10 + digit must fit.
         if (accumulator > (limit - int(digit, int64)) / 10_int64) then
            status = FBC_ERR_INVALID_ARGUMENT
            value = 0_int64
            return
         end if
         accumulator = accumulator * 10_int64 + int(digit, int64)
      end do
      if (accumulator < 1_int64) then
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      value = accumulator
   end subroutine fbc_parse_horizon

   !> Same strictness for the int32 extents, with an explicit upper bound.
   pure subroutine fbc_parse_count(text, minimum, maximum, value, status)
      character(len=*), intent(in) :: text
      integer(int32), intent(in) :: minimum, maximum
      integer(int32), intent(out) :: value
      integer(int32), intent(out) :: status
      integer(int64) :: wide

      value = 0_int32
      call fbc_parse_horizon(text, wide, status)
      if (status /= FBC_OK) return
      if (wide > int(maximum, int64)) then
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      if (wide < int(minimum, int64)) then
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      value = int(wide, int32)
   end subroutine fbc_parse_count

   !----------------------------------------------------------------------------
   !  Model construction
   !----------------------------------------------------------------------------

   pure logical function fbc_model_is_ready(model) result(ready)
      type(fbc_model_t), intent(in) :: model

      ready = model%initialized .and. allocated(model%token_embedding) .and. &
              allocated(model%output_bias) .and. allocated(model%query) .and. allocated(model%key) .and. &
              allocated(model%value) .and. allocated(model%attention_output) .and. &
              allocated(model%attention_rms) .and. allocated(model%ffn_rms) .and. &
              allocated(model%ffn_gate) .and. allocated(model%ffn_value) .and. &
              allocated(model%ffn_output) .and. allocated(model%distance_bias) .and. &
              allocated(model%scale_logit) .and. allocated(model%channel_logit) .and. &
              allocated(model%slot_query_proj) .and. allocated(model%slot_key_proj) .and. &
              allocated(model%slot_value_proj) .and. allocated(model%slot_read_proj) .and. &
              allocated(model%scale_decay)
   end function fbc_model_is_ready

   !> Deterministic xorshift stream, identical to the one the rest of the project
   !> uses, so a seed reproduces the same model on any host.
   pure subroutine fbc_next_unit(state, value)
      integer(int64), intent(inout) :: state
      real(real64), intent(out) :: value
      integer(int64) :: nonnegative

      state = ieor(state, ishftc(state, 13))
      state = ieor(state, shiftr(state, 7))
      state = ieor(state, ishftc(state, 17))
      nonnegative = iand(state, huge(0_int64))
      value = real(nonnegative, real64) / real(huge(0_int64), real64)
   end subroutine fbc_next_unit

   !> Uniform init over a contiguous array.  Three explicit ranks instead of one
   !> assumed-rank dummy: assumed-rank entities cannot be indexed without a
   !> SELECT RANK, and a copy through RESHAPE would defeat the point.
   subroutine fbc_fill_1d(matrix, fan_in, state)
      real(real64), intent(out) :: matrix(:)
      integer(int32), intent(in) :: fan_in
      integer(int64), intent(inout) :: state
      real(real64) :: scale, uniform
      integer(int32) :: slot

      scale = sqrt(3.0_real64 / real(max(1_int32, fan_in), real64))
      do slot = 1_int32, size(matrix, kind=int32)
         call fbc_next_unit(state, uniform)
         matrix(slot) = (2.0_real64 * uniform - 1.0_real64) * scale
      end do
   end subroutine fbc_fill_1d

   subroutine fbc_fill_2d(matrix, fan_in, state)
      real(real64), intent(out) :: matrix(:, :)
      integer(int32), intent(in) :: fan_in
      integer(int64), intent(inout) :: state
      real(real64) :: scale, uniform
      integer(int32) :: row, column

      scale = sqrt(3.0_real64 / real(max(1_int32, fan_in), real64))
      do column = 1_int32, size(matrix, 2, kind=int32)
         do row = 1_int32, size(matrix, 1, kind=int32)
            call fbc_next_unit(state, uniform)
            matrix(row, column) = (2.0_real64 * uniform - 1.0_real64) * scale
         end do
      end do
   end subroutine fbc_fill_2d

   subroutine fbc_fill_3d(matrix, fan_in, state)
      real(real64), intent(out) :: matrix(:, :, :)
      integer(int32), intent(in) :: fan_in
      integer(int64), intent(inout) :: state
      real(real64) :: scale, uniform
      integer(int32) :: row, column, index

      scale = sqrt(3.0_real64 / real(max(1_int32, fan_in), real64))
      do index = 1_int32, size(matrix, 3, kind=int32)
         do column = 1_int32, size(matrix, 2, kind=int32)
            do row = 1_int32, size(matrix, 1, kind=int32)
               call fbc_next_unit(state, uniform)
               matrix(row, column, index) = (2.0_real64 * uniform - 1.0_real64) * scale
            end do
         end do
      end do
   end subroutine fbc_fill_3d

   subroutine fbc_model_init(model, config, seed, status)
      type(fbc_model_t), intent(inout) :: model
      type(fbc_config_t), intent(in) :: config
      integer(int64), intent(in) :: seed
      integer(int32), intent(out) :: status
      integer(int32) :: allocation_status, head_width, head
      integer(int64) :: random_state

      call fbc_model_destroy(model)
      status = FBC_OK
      if (.not. fbc_config_is_valid(config)) then
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      head_width = config%width / config%heads
      model%config = config
      model%head_width = head_width
      allocate(model%scale_decay(config%memory_scales), &
               model%token_embedding(config%width, config%vocabulary_size), &
               model%output_bias(config%vocabulary_size), &
               model%query(head_width, config%width, config%heads), &
               model%key(head_width, config%width, config%heads), &
               model%value(head_width, config%width, config%heads), &
               model%attention_output(config%width, config%width), &
               model%attention_rms(config%width), model%ffn_rms(config%width), &
               model%ffn_gate(config%ffn_width, config%width), &
               model%ffn_value(config%ffn_width, config%width), &
               model%ffn_output(config%width, config%ffn_width), &
               model%distance_bias(config%local_window + 1_int32), &
               model%scale_logit(config%memory_scales), &
               model%channel_logit(FBC_CHANNEL_COUNT), &
               model%slot_query_proj(config%width, config%width), &
               model%slot_key_proj(config%width, config%width), &
               model%slot_value_proj(config%width, config%width), &
               model%slot_read_proj(config%width, config%width), stat=allocation_status)
      if (allocation_status /= 0) then
         status = FBC_ERR_ALLOCATION
         call fbc_model_destroy(model)
         return
      end if
      random_state = seed
      if (random_state == 0_int64) random_state = int(z'6A09E667F3BCC909', int64)
      call fbc_fill(model%token_embedding, config%width, random_state)
      call fbc_fill(model%attention_output, config%width, random_state)
      call fbc_fill(model%ffn_gate, config%width, random_state)
      call fbc_fill(model%ffn_value, config%width, random_state)
      call fbc_fill(model%ffn_output, config%ffn_width, random_state)
      call fbc_fill(model%slot_query_proj, config%width, random_state)
      call fbc_fill(model%slot_key_proj, config%width, random_state)
      call fbc_fill(model%slot_value_proj, config%width, random_state)
      call fbc_fill(model%slot_read_proj, config%width, random_state)
      do head = 1_int32, config%heads
         call fbc_fill(model%query(:, :, head), config%width, random_state)
         call fbc_fill(model%key(:, :, head), config%width, random_state)
         call fbc_fill(model%value(:, :, head), config%width, random_state)
      end do
      model%output_bias = 0.0_real64
      model%attention_rms = 1.0_real64
      model%ffn_rms = 1.0_real64
      model%distance_bias = 0.0_real64
      model%scale_logit = 0.0_real64
      model%channel_logit = 0.0_real64
      model%slot_salience_weight = 0.0_real64
      call fbc_configure_scales(config, model%scale_decay, model%effective_horizon, model%horizon_clamped)
      model%initialized = .true.
   end subroutine fbc_model_init

   subroutine fbc_model_destroy(model)
      type(fbc_model_t), intent(inout) :: model

      if (allocated(model%scale_decay)) deallocate(model%scale_decay)
      if (allocated(model%token_embedding)) deallocate(model%token_embedding)
      if (allocated(model%output_bias)) deallocate(model%output_bias)
      if (allocated(model%query)) deallocate(model%query)
      if (allocated(model%key)) deallocate(model%key)
      if (allocated(model%value)) deallocate(model%value)
      if (allocated(model%attention_output)) deallocate(model%attention_output)
      if (allocated(model%attention_rms)) deallocate(model%attention_rms)
      if (allocated(model%ffn_rms)) deallocate(model%ffn_rms)
      if (allocated(model%ffn_gate)) deallocate(model%ffn_gate)
      if (allocated(model%ffn_value)) deallocate(model%ffn_value)
      if (allocated(model%ffn_output)) deallocate(model%ffn_output)
      if (allocated(model%distance_bias)) deallocate(model%distance_bias)
      if (allocated(model%scale_logit)) deallocate(model%scale_logit)
      if (allocated(model%channel_logit)) deallocate(model%channel_logit)
      if (allocated(model%slot_query_proj)) deallocate(model%slot_query_proj)
      if (allocated(model%slot_key_proj)) deallocate(model%slot_key_proj)
      if (allocated(model%slot_value_proj)) deallocate(model%slot_value_proj)
      if (allocated(model%slot_read_proj)) deallocate(model%slot_read_proj)
      model = fbc_model_t()
   end subroutine fbc_model_destroy

   !----------------------------------------------------------------------------
   !  Per-stream state and workspace
   !----------------------------------------------------------------------------

   pure logical function fbc_state_is_ready(state, model) result(ready)
      type(fbc_state_t), intent(in) :: state
      type(fbc_model_t), intent(in) :: model

      ready = state%initialized .and. fbc_model_is_ready(model) .and. &
              state%width == model%config%width .and. state%heads == model%config%heads .and. &
              state%local_window == model%config%local_window .and. &
              state%memory_scales == model%config%memory_scales .and. &
              state%memory_slots == model%config%memory_slots .and. &
              allocated(state%retention_numerator) .and. allocated(state%retention_normalizer) .and. &
              allocated(state%local_key) .and. allocated(state%local_value) .and. &
              allocated(state%slot_key) .and. allocated(state%slot_value) .and. &
              allocated(state%slot_salience)
   end function fbc_state_is_ready

   subroutine fbc_state_init(state, model, status)
      type(fbc_state_t), intent(inout) :: state
      type(fbc_model_t), intent(in) :: model
      integer(int32), intent(out) :: status
      integer(int32) :: allocation_status, head_width

      call fbc_state_destroy(state)
      status = FBC_OK
      if (.not. fbc_model_is_ready(model)) then
         status = FBC_ERR_NOT_READY
         return
      end if
      head_width = model%head_width
      state%width = model%config%width
      state%heads = model%config%heads
      state%head_width = head_width
      state%local_window = model%config%local_window
      state%memory_scales = model%config%memory_scales
      state%memory_slots = model%config%memory_slots
      allocate(state%retention_numerator(head_width, head_width, model%config%heads, model%config%memory_scales), &
               state%retention_normalizer(head_width, model%config%heads, model%config%memory_scales), &
               state%local_key(head_width, model%config%heads, model%config%local_window), &
               state%local_value(head_width, model%config%heads, model%config%local_window), &
               state%slot_key(model%config%width, model%config%memory_slots), &
               state%slot_value(model%config%width, model%config%memory_slots), &
               state%slot_salience(model%config%memory_slots), stat=allocation_status)
      if (allocation_status /= 0) then
         status = FBC_ERR_ALLOCATION
         call fbc_state_destroy(state)
         return
      end if
      state%initialized = .true.
      call fbc_state_reset(state, model, status)
   end subroutine fbc_state_init

   subroutine fbc_state_reset(state, model, status)
      type(fbc_state_t), intent(inout) :: state
      type(fbc_model_t), intent(in) :: model
      integer(int32), intent(out) :: status

      status = FBC_OK
      if (.not. fbc_state_is_ready(state, model)) then
         status = FBC_ERR_NOT_READY
         return
      end if
      state%retention_numerator = 0.0_real64
      state%retention_normalizer = 0.0_real64
      state%local_key = 0.0_real64
      state%local_value = 0.0_real64
      state%slot_key = 0.0_real64
      state%slot_value = 0.0_real64
      state%slot_salience = 0.0_real64
      state%cache_head = 0_int32
      state%cache_count = 0_int32
      state%steps = 0_int64
   end subroutine fbc_state_reset

   subroutine fbc_state_destroy(state)
      type(fbc_state_t), intent(inout) :: state

      if (allocated(state%retention_numerator)) deallocate(state%retention_numerator)
      if (allocated(state%retention_normalizer)) deallocate(state%retention_normalizer)
      if (allocated(state%local_key)) deallocate(state%local_key)
      if (allocated(state%local_value)) deallocate(state%local_value)
      if (allocated(state%slot_key)) deallocate(state%slot_key)
      if (allocated(state%slot_value)) deallocate(state%slot_value)
      if (allocated(state%slot_salience)) deallocate(state%slot_salience)
      state = fbc_state_t()
   end subroutine fbc_state_destroy

   subroutine fbc_workspace_init(workspace, model, status)
      type(fbc_workspace_t), intent(inout) :: workspace
      type(fbc_model_t), intent(in) :: model
      integer(int32), intent(out) :: status
      integer(int32) :: allocation_status, width, heads, head_width, scales, slots, window, ffn

      call fbc_workspace_destroy(workspace)
      status = FBC_OK
      if (.not. fbc_model_is_ready(model)) then
         status = FBC_ERR_NOT_READY
         return
      end if
      width = model%config%width
      heads = model%config%heads
      head_width = model%head_width
      scales = model%config%memory_scales
      slots = model%config%memory_slots
      window = model%config%local_window
      ffn = model%config%ffn_width
      allocate(workspace%x(width), workspace%normalized(width), &
               workspace%q(head_width, heads), workspace%k(head_width, heads), &
               workspace%v(head_width, heads), workspace%phi_q(head_width, heads), &
               workspace%phi_k(head_width, heads), workspace%read_denominator(heads, scales), &
               workspace%read_scale(head_width, heads, scales), workspace%scale_alpha(scales), &
               workspace%local_weight(window + 1_int32, heads), workspace%local_position(window), &
               workspace%local_context(head_width, heads), workspace%memory_context(head_width, heads), &
               workspace%slot_query(width), workspace%slot_write_key(width), &
               workspace%slot_write_value(width), workspace%slot_write_gate(slots), &
               workspace%slot_key_delta(width, slots), workspace%slot_value_delta(width, slots), &
               workspace%slot_read_score(slots), workspace%slot_read_weight(slots), &
               workspace%slot_read(width), workspace%slot_context(width), &
               workspace%channel_alpha(FBC_CHANNEL_COUNT), &
               workspace%channel_context(width, FBC_CHANNEL_COUNT), &
               workspace%combined(width), workspace%attention_residual(width), &
               workspace%rhat_attention(width), workspace%rhat_ffn(width), &
               workspace%ffn_pre_gate(ffn), workspace%ffn_pre_value(ffn), &
               workspace%ffn_gated(ffn), workspace%hidden(width), stat=allocation_status)
      if (allocation_status /= 0) then
         status = FBC_ERR_ALLOCATION
         call fbc_workspace_destroy(workspace)
         return
      end if
      workspace%initialized = .true.
   end subroutine fbc_workspace_init

   subroutine fbc_workspace_destroy(workspace)
      type(fbc_workspace_t), intent(inout) :: workspace

      if (allocated(workspace%x)) deallocate(workspace%x)
      if (allocated(workspace%normalized)) deallocate(workspace%normalized)
      if (allocated(workspace%q)) deallocate(workspace%q)
      if (allocated(workspace%k)) deallocate(workspace%k)
      if (allocated(workspace%v)) deallocate(workspace%v)
      if (allocated(workspace%phi_q)) deallocate(workspace%phi_q)
      if (allocated(workspace%phi_k)) deallocate(workspace%phi_k)
      if (allocated(workspace%read_denominator)) deallocate(workspace%read_denominator)
      if (allocated(workspace%read_scale)) deallocate(workspace%read_scale)
      if (allocated(workspace%scale_alpha)) deallocate(workspace%scale_alpha)
      if (allocated(workspace%local_weight)) deallocate(workspace%local_weight)
      if (allocated(workspace%local_position)) deallocate(workspace%local_position)
      if (allocated(workspace%local_context)) deallocate(workspace%local_context)
      if (allocated(workspace%memory_context)) deallocate(workspace%memory_context)
      if (allocated(workspace%slot_query)) deallocate(workspace%slot_query)
      if (allocated(workspace%slot_write_key)) deallocate(workspace%slot_write_key)
      if (allocated(workspace%slot_write_value)) deallocate(workspace%slot_write_value)
      if (allocated(workspace%slot_write_gate)) deallocate(workspace%slot_write_gate)
      if (allocated(workspace%slot_key_delta)) deallocate(workspace%slot_key_delta)
      if (allocated(workspace%slot_value_delta)) deallocate(workspace%slot_value_delta)
      if (allocated(workspace%slot_read_score)) deallocate(workspace%slot_read_score)
      if (allocated(workspace%slot_read_weight)) deallocate(workspace%slot_read_weight)
      if (allocated(workspace%slot_read)) deallocate(workspace%slot_read)
      if (allocated(workspace%slot_context)) deallocate(workspace%slot_context)
      if (allocated(workspace%channel_alpha)) deallocate(workspace%channel_alpha)
      if (allocated(workspace%channel_context)) deallocate(workspace%channel_context)
      if (allocated(workspace%combined)) deallocate(workspace%combined)
      if (allocated(workspace%attention_residual)) deallocate(workspace%attention_residual)
      if (allocated(workspace%rhat_attention)) deallocate(workspace%rhat_attention)
      if (allocated(workspace%rhat_ffn)) deallocate(workspace%rhat_ffn)
      if (allocated(workspace%ffn_pre_gate)) deallocate(workspace%ffn_pre_gate)
      if (allocated(workspace%ffn_pre_value)) deallocate(workspace%ffn_pre_value)
      if (allocated(workspace%ffn_gated)) deallocate(workspace%ffn_gated)
      if (allocated(workspace%hidden)) deallocate(workspace%hidden)
      workspace = fbc_workspace_t()
   end subroutine fbc_workspace_destroy

   !----------------------------------------------------------------------------
   !  Structural plan
   !----------------------------------------------------------------------------

   pure integer(int64) function fbc_parameter_count(config) result(count)
      type(fbc_config_t), intent(in) :: config
      integer(int64) :: width, vocab, heads, head_width, ffn, window, scales

      width = int(config%width, int64)
      vocab = int(config%vocabulary_size, int64)
      heads = int(config%heads, int64)
      head_width = int(config%width / config%heads, int64)
      ffn = int(config%ffn_width, int64)
      window = int(config%local_window, int64)
      scales = int(config%memory_scales, int64)
      ! embedding + bias + 3 head projections + output projection + 2 norms
      count = width * vocab + vocab + 3_int64 * heads * head_width * width + width * width + 2_int64 * width
      ! gated FFN (gate, value, output) + distance bias + scale/channel logits
      count = count + 3_int64 * width * ffn + window + 1_int64 + scales + FBC_CHANNEL_COUNT
      ! slot projections (query, write key, write value, read output) + salience weight
      count = count + 4_int64 * width * width + 1_int64
   end function fbc_parameter_count

   pure integer(int64) function fbc_state_bytes(config) result(bytes)
      type(fbc_config_t), intent(in) :: config
      integer(int64) :: width, heads, head_width, window, scales, slots

      width = int(config%width, int64)
      heads = int(config%heads, int64)
      head_width = int(config%width / config%heads, int64)
      window = int(config%local_window, int64)
      scales = int(config%memory_scales, int64)
      slots = int(config%memory_slots, int64)
      bytes = 8_int64 * (heads * scales * head_width * head_width + heads * scales * head_width + &
                         2_int64 * heads * window * head_width + 2_int64 * width * slots + slots)
   end function fbc_state_bytes

   pure subroutine fbc_plan(config, plan)
      type(fbc_config_t), intent(in) :: config
      type(fbc_plan_t), intent(out) :: plan
      integer(int64) :: width, vocab, heads, head_width, ffn, window, scales, slots
      real(real64) :: unused_decay(config%memory_scales)
      logical :: unused_clamped

      plan = fbc_plan_t()
      if (.not. fbc_config_is_valid(config)) return
      width = int(config%width, int64)
      vocab = int(config%vocabulary_size, int64)
      heads = int(config%heads, int64)
      head_width = int(config%width / config%heads, int64)
      ffn = int(config%ffn_width, int64)
      window = int(config%local_window, int64)
      scales = int(config%memory_scales, int64)
      slots = int(config%memory_slots, int64)
      plan%local_window = config%local_window
      plan%memory_scales = config%memory_scales
      plan%memory_slots = config%memory_slots
      plan%logical_horizon = config%logical_horizon
      call fbc_configure_scales(config, unused_decay, plan%effective_horizon, plan%horizon_clamped)
      plan%parameter_count = fbc_parameter_count(config)
      plan%parameter_bytes_real64 = 8_int64 * plan%parameter_count
      plan%state_bytes_real64 = fbc_state_bytes(config)
      plan%local_cache_elements = 2_int64 * heads * window * head_width
      plan%retention_state_elements = heads * scales * (head_width * head_width + head_width)
      plan%slot_state_elements = 2_int64 * width * slots + slots
      ! Q/K/V + output projection + FFN + tied logits, forward.
      plan%forward_flops_per_token = 2_int64 * (3_int64 * heads * head_width * width + width * width + &
                                                3_int64 * width * ffn + width * vocab) + &
                                     2_int64 * (heads * scales * head_width * head_width) + &
                                     4_int64 * width * width * 2_int64 + &
                                     2_int64 * (2_int64 * width * slots)
      ! The backward pass is the same dense products plus the elementwise chain;
      ! counted as 2x the forward matmul work plus the gate arithmetic.
      plan%backward_flops_per_token = 2_int64 * plan%forward_flops_per_token + &
                                      12_int64 * (width * slots + width + ffn + heads * scales)
      plan%local_score_operations_per_token = 2_int64 * heads * (window + 1_int64) * (head_width + 1_int64)
   end subroutine fbc_plan

   !----------------------------------------------------------------------------
   !  One causal step
   !----------------------------------------------------------------------------
   !  Everything the step touches is bounded by the configuration.  The horizon
   !  appears only through model%scale_decay, which was computed at init.
   subroutine fbc_step(model, state, workspace, token, hidden, logits, status)
      type(fbc_model_t), intent(in) :: model
      type(fbc_state_t), intent(inout) :: state
      type(fbc_workspace_t), intent(inout) :: workspace
      integer(int32), intent(in) :: token
      real(real64), intent(out) :: hidden(:)
      real(real64), intent(out) :: logits(:)
      integer(int32), intent(out) :: status
      integer(int32) :: width, heads, head_width, scales, slots, window, ffn
      integer(int32) :: head, scale, component, slot, ordinal, position, channel
      real(real64) :: score, maximum, normalizer, weight, denominator, inverse_rms, mean_square
      real(real64) :: scale_factor, slot_scale, gate, salience, context_scale

      status = FBC_OK
      if (.not. fbc_model_is_ready(model) .or. .not. fbc_state_is_ready(state, model)) then
         status = FBC_ERR_NOT_READY
         return
      end if
      if (.not. workspace%initialized) then
         status = FBC_ERR_NOT_READY
         return
      end if
      if (token < 1_int32 .or. token > model%config%vocabulary_size) then
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      if (size(logits) /= model%config%vocabulary_size .or. size(hidden) /= model%config%width) then
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      width = model%config%width
      heads = model%config%heads
      head_width = model%head_width
      scales = model%config%memory_scales
      slots = model%config%memory_slots
      window = model%config%local_window
      ffn = model%config%ffn_width
      workspace%local_count = state%cache_count
      workspace%token = token

      ! ---- 1. embedding and the attention RMSNorm -----------------------------
      workspace%x = model%token_embedding(:, token)
      mean_square = dot_product(workspace%x, workspace%x) / real(width, real64)
      inverse_rms = 1.0_real64 / sqrt(mean_square + model%config%rms_epsilon)
      workspace%inverse_rms_attention = inverse_rms
      workspace%rhat_attention = workspace%x * inverse_rms
      workspace%normalized = workspace%rhat_attention * model%attention_rms

      ! ---- 2. head projections ------------------------------------------------
      do head = 1_int32, heads
         workspace%q(:, head) = matmul(model%query(:, :, head), workspace%normalized)
         workspace%k(:, head) = matmul(model%key(:, :, head), workspace%normalized)
         workspace%v(:, head) = matmul(model%value(:, :, head), workspace%normalized)
      end do

      ! ---- 3. exact causal attention over the fixed ring buffer --------------
      context_scale = 1.0_real64 / sqrt(real(head_width, real64))
      do head = 1_int32, heads
         ! index 1 is the current token (distance 0); 2..count+1 are the cached
         ! entries from most recent to oldest.
         workspace%local_weight(1_int32, head) = dot_product(workspace%q(:, head), workspace%k(:, head)) * &
                                                 context_scale + model%distance_bias(1_int32)
         maximum = workspace%local_weight(1_int32, head)
         do ordinal = 1_int32, state%cache_count
            position = modulo(state%cache_head - ordinal, window) + 1_int32
            workspace%local_position(ordinal) = position
            score = dot_product(workspace%q(:, head), state%local_key(:, head, position)) * context_scale + &
                    model%distance_bias(ordinal + 1_int32)
            workspace%local_weight(ordinal + 1_int32, head) = score
            maximum = max(maximum, score)
         end do
         normalizer = 0.0_real64
         do ordinal = 1_int32, state%cache_count + 1_int32
            weight = exp(workspace%local_weight(ordinal, head) - maximum)
            workspace%local_weight(ordinal, head) = weight
            normalizer = normalizer + weight
         end do
         if (normalizer <= 0.0_real64 .or. .not. ieee_is_finite(normalizer)) then
            status = FBC_ERR_NUMERIC
            return
         end if
         workspace%local_weight(1_int32:state%cache_count + 1_int32, head) = &
            workspace%local_weight(1_int32:state%cache_count + 1_int32, head) / normalizer
         workspace%local_context(:, head) = workspace%local_weight(1_int32, head) * workspace%v(:, head)
         do ordinal = 1_int32, state%cache_count
            position = workspace%local_position(ordinal)
            workspace%local_context(:, head) = workspace%local_context(:, head) + &
               workspace%local_weight(ordinal + 1_int32, head) * state%local_value(:, head, position)
         end do
      end do

      ! ---- 4. multi-scale recurrent memory (write, then read) ----------------
      do head = 1_int32, heads
         do component = 1_int32, head_width
            workspace%phi_k(component, head) = fbc_softplus(workspace%k(component, head))
            workspace%phi_q(component, head) = fbc_softplus(workspace%q(component, head))
         end do
      end do
      call fbc_softmax(model%scale_logit, workspace%scale_alpha)
      do head = 1_int32, heads
         do scale = 1_int32, scales
            scale_factor = model%scale_decay(scale)
            ! R <- lambda R + (1 - lambda) phi(k) v^T, column-major over the
            ! output feature so the inner loop is contiguous.
            do component = 1_int32, head_width
               state%retention_numerator(:, component, head, scale) = &
                  scale_factor * state%retention_numerator(:, component, head, scale) + &
                  (1.0_real64 - scale_factor) * workspace%phi_k(:, head) * workspace%v(component, head)
            end do
            state%retention_normalizer(:, head, scale) = scale_factor * state%retention_normalizer(:, head, scale) + &
                                                         (1.0_real64 - scale_factor) * workspace%phi_k(:, head)
            denominator = dot_product(workspace%phi_q(:, head), state%retention_normalizer(:, head, scale)) + &
                          model%config%rms_epsilon
            if (.not. ieee_is_finite(denominator) .or. denominator <= 0.0_real64) then
               status = FBC_ERR_NUMERIC
               return
            end if
            workspace%read_denominator(head, scale) = denominator
            ! read(c) = sum_i phi_q(i) R(i, c): a vector-matrix product, so the
            ! unit-stride inner loop walks R's columns with no transposed copy.
            workspace%read_scale(:, head, scale) = matmul(workspace%phi_q(:, head), &
                                                          state%retention_numerator(:, :, head, scale)) / denominator
         end do
      end do
      workspace%memory_context = 0.0_real64
      do head = 1_int32, heads
         do scale = 1_int32, scales
            workspace%memory_context(:, head) = workspace%memory_context(:, head) + &
               workspace%scale_alpha(scale) * workspace%read_scale(:, head, scale)
         end do
      end do

      ! ---- 5. slots: write the current token, then read ----------------------
      workspace%slot_query = matmul(model%slot_query_proj, workspace%normalized)
      workspace%slot_write_key = matmul(model%slot_key_proj, workspace%normalized)
      workspace%slot_write_value = matmul(model%slot_value_proj, workspace%normalized)
      slot_scale = 1.0_real64 / sqrt(real(width, real64))
      do slot = 1_int32, slots
         score = dot_product(workspace%slot_write_key, state%slot_key(:, slot)) * slot_scale
         gate = fbc_sigmoid(score / model%config%slot_temperature)
         workspace%slot_write_gate(slot) = gate
         ! The delta is kept so the backward pass can differentiate the write
         ! gate without storing the pre-write slot matrix.
         workspace%slot_key_delta(:, slot) = workspace%slot_write_key - state%slot_key(:, slot)
         workspace%slot_value_delta(:, slot) = workspace%slot_write_value - state%slot_value(:, slot)
         salience = state%slot_salience(slot)
         state%slot_key(:, slot) = state%slot_key(:, slot) + gate * workspace%slot_key_delta(:, slot)
         state%slot_value(:, slot) = state%slot_value(:, slot) + gate * workspace%slot_value_delta(:, slot)
         state%slot_salience(slot) = model%config%slot_decay * salience + &
                                     (1.0_real64 - model%config%slot_decay) * gate
         workspace%slot_read_score(slot) = dot_product(workspace%slot_query, state%slot_key(:, slot)) * slot_scale + &
                                           model%slot_salience_weight * state%slot_salience(slot)
      end do
      call fbc_softmax(workspace%slot_read_score, workspace%slot_read_weight)
      workspace%slot_read = 0.0_real64
      do slot = 1_int32, slots
         workspace%slot_read = workspace%slot_read + workspace%slot_read_weight(slot) * state%slot_value(:, slot)
      end do
      workspace%slot_context = matmul(model%slot_read_proj, workspace%slot_read)

      ! ---- 6. learnable channel mixing --------------------------------------
      do head = 1_int32, heads
         workspace%channel_context((head - 1_int32) * head_width + 1_int32:head * head_width, FBC_CHANNEL_LOCAL) = &
            workspace%local_context(:, head)
         workspace%channel_context((head - 1_int32) * head_width + 1_int32:head * head_width, &
                                   FBC_CHANNEL_RETENTION) = workspace%memory_context(:, head)
      end do
      workspace%channel_context(:, FBC_CHANNEL_SLOTS) = workspace%slot_context
      call fbc_softmax(model%channel_logit, workspace%channel_alpha)
      workspace%combined = 0.0_real64
      do channel = 1_int32, FBC_CHANNEL_COUNT
         workspace%combined = workspace%combined + workspace%channel_alpha(channel) * &
                              workspace%channel_context(:, channel)
      end do

      ! ---- 7. residual, FFN, tied logits ------------------------------------
      workspace%attention_residual = workspace%x + matmul(model%attention_output, workspace%combined)
      mean_square = dot_product(workspace%attention_residual, workspace%attention_residual) / real(width, real64)
      inverse_rms = 1.0_real64 / sqrt(mean_square + model%config%rms_epsilon)
      workspace%inverse_rms_ffn = inverse_rms
      workspace%rhat_ffn = workspace%attention_residual * inverse_rms
      workspace%normalized = workspace%rhat_ffn * model%ffn_rms
      workspace%ffn_pre_gate = matmul(model%ffn_gate, workspace%normalized)
      workspace%ffn_pre_value = matmul(model%ffn_value, workspace%normalized)
      do component = 1_int32, ffn
         workspace%ffn_gated(component) = fbc_sigmoid(workspace%ffn_pre_gate(component)) * &
                                         tanh(workspace%ffn_pre_value(component))
      end do
      workspace%hidden = workspace%attention_residual + matmul(model%ffn_output, workspace%ffn_gated)
      hidden = workspace%hidden
      logits = matmul(workspace%hidden, model%token_embedding) + model%output_bias

      ! ---- 8. commit the ring buffer (the current token becomes history) ----
      position = modulo(state%cache_head, window) + 1_int32
      state%local_key(:, :, position) = workspace%k
      state%local_value(:, :, position) = workspace%v
      state%cache_head = position
      state%cache_count = min(state%cache_count + 1_int32, window)
      state%steps = state%steps + 1_int64
   end subroutine fbc_step

end module fixed_budget_context
