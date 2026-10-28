! SPDX-License-Identifier: MIT
!==============================================================================
!  fixed_budget_context_tests.f90
!
!  Executable contract tests for the fixed-budget context engine.  Every check
!  is an assertion; the program exits non-zero if any fails.
!
!    1. strict option parsing (empty, sign, zero, letters, exponent, overflow)
!    2. horizon invariance of every physical size and structural counter for
!       horizons 32 / 8192 / 1e6 / 1e12, plus arithmetic clamping at 1e16
!    3. no future leakage (changing a later token cannot change earlier logits)
!    4. chunk/stream equivalence (one pass vs four chunks: identical state)
!    5. determinism (same seed twice, different seed differs)
!    6. long stream stays finite and the recurrent state stays bounded
!    7. finite-difference gradient check of the analytic backward pass
!    8. training reduces the loss on a learnable synthetic task
!    9. every memory component receives a gradient and changes under training
!   10. remote influence: an answer 93 tokens behind the query reaches the
!       query logits through the fixed-size state and provably cannot reach
!       them through the local ring buffer alone
!   11. checkpoint round-trip equality and explicit legacy rejection
!
!  Nothing here is timing based; the invariance checks read structural
!  counters, so they cannot pass or fail because a runner was busy.
!==============================================================================
program fixed_budget_context_tests
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64, output_unit
   use fixed_budget_context
   use fixed_budget_backward
   use fixed_budget_optimizer
   use fixed_budget_training
   implicit none (type, external)

   integer(int32) :: checks = 0_int32, failures = 0_int32

   call test_defaults
   call test_option_parsing
   call test_horizon_invariance
   call test_no_future_leakage
   call test_chunk_equivalence
   call test_determinism
   call test_long_stream_finite
   call test_gradient_check
   call test_training_reduces_loss
   call test_memory_components_train
   call test_remote_influence
   call test_checkpoint_roundtrip

   write(output_unit, '(A,I0,A,I0,A)') 'fixed-budget context checks: ', checks, ', failures: ', &
      failures, '.'
   if (failures > 0_int32) then
      write(output_unit, '(A,I0,A)') 'FAIL: ', failures, ' contract(s) violated'
      error stop 1_int32
   end if
   write(output_unit, '(A)') 'PASS: all fixed-budget context contracts hold'
contains

   subroutine check(condition, description)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: description

      checks = checks + 1_int32
      if (.not. condition) then
         failures = failures + 1_int32
         write(output_unit, '(A,A)') 'FAIL: ', description
      end if
   end subroutine check

   !> Small shape used by the gradient and training checks: width 8, two heads
   !> of four, two scales, four slots.
   pure function tiny_config(horizon) result(config)
      integer(int64), intent(in) :: horizon
      type(fbc_config_t) :: config

      config%vocabulary_size = 16_int32
      config%width = 8_int32
      config%heads = 2_int32
      config%ffn_width = 16_int32
      config%local_window = 4_int32
      config%memory_scales = 2_int32
      config%memory_slots = 4_int32
      config%logical_horizon = horizon
   end function tiny_config

   !> The project default shape, one size down.
   pure function small_config(horizon) result(config)
      integer(int64), intent(in) :: horizon
      type(fbc_config_t) :: config

      config%vocabulary_size = 32_int32
      config%width = 32_int32
      config%heads = 2_int32
      config%ffn_width = 64_int32
      config%local_window = 8_int32
      config%memory_scales = 3_int32
      config%memory_slots = 16_int32
      config%logical_horizon = horizon
   end function small_config

   !----------------------------------------------------------------------------
   !  0. Defaults
   !----------------------------------------------------------------------------
   !> The default horizon is pinned here because it is a design decision, not an
   !> accident: at 8192 with a 32-token physical window the four default scales
   !> are 32 / 203 / 1290 / 8192.  At horizon = local_window all four collapse to
   !> the same lambda and the multi-scale channel is dead weight.
   subroutine test_defaults
      type(fbc_config_t) :: config
      real(real64) :: decay(8)
      real(real64) :: effective
      logical :: clamped
      integer(int32) :: index
      logical :: distinct

      config = fbc_config_t()
      call check(config%logical_horizon == 8192_int64, 'the default logical horizon is 8192')
      call check(config%local_window == 32_int32, 'the default physical window is 32')
      call check(config%memory_scales >= 2_int32, 'the default is multi-scale')
      call check(fbc_config_is_valid(config), 'the default configuration is valid')

      ! The default ladder must actually be a ladder.
      call fbc_configure_scales(config, decay, effective, clamped)
      distinct = .true.
      do index = 2_int32, config%memory_scales
         if (abs(decay(index) - decay(index - 1_int32)) < 1.0e-9_real64) distinct = .false.
      end do
      call check(distinct, 'the default scales are genuinely distinct decay constants')
      call check(.not. clamped, 'the default horizon needs no arithmetic clamping')
      call check(effective == 8192.0_real64, 'the default effective horizon is 8192 tokens')

      ! The learnable dynamics must start at the ladder they replaced, otherwise
      ! the improvement would be a silent behaviour change.
      block
         type(fbc_model_t) :: model
         integer(int32) :: status, scale_index
         logical :: matches

         call fbc_model_init(model, config, 4242_int64, status)
         matches = .true.
         do scale_index = 1_int32, config%memory_scales
            if (abs(model%scale_decay(scale_index) - decay(scale_index)) > 1.0e-12_real64) matches = .false.
            if (abs(model%write_strength(scale_index) - (1.0_real64 - decay(scale_index))) > 1.0e-12_real64) &
               matches = .false.
         end do
         call check(matches, 'learned dynamics start exactly at lambda_j and 1 - lambda_j')
         call check(abs(fbc_learned_horizon(model) - effective) / effective < 0.01_real64, &
                    'the learned reach starts at the configured horizon')
         call fbc_model_destroy(model)
      end block

      ! And the degenerate case must stay visible: horizon = window collapses.
      ! Only the first `memory_scales` entries are written, so the comparison
      ! uses that count rather than the extent of the scratch array.
      block
         type(fbc_config_t) :: degenerate
         integer(int32) :: scales

         degenerate = small_config(8_int64)
         scales = degenerate%memory_scales
         call fbc_configure_scales(degenerate, decay, effective, clamped)
         call check(abs(decay(1_int32) - decay(scales)) < 1.0e-12_real64, &
                    'horizon equal to the local window does collapse every scale onto one lambda')
         call check(abs(effective - 8.0_real64) < 1.0e-12_real64, &
                    'a horizon below the local window reports the window as its effective reach')
      end block
   end subroutine test_defaults

   !----------------------------------------------------------------------------
   !  1. Strict option parsing
   !----------------------------------------------------------------------------
   subroutine test_option_parsing
      integer(int64) :: horizon
      integer(int32) :: status, count, task

      call fbc_parse_horizon('32', horizon, status)
      call check(status == FBC_OK .and. horizon == 32_int64, 'accepts a plain horizon')
      call fbc_parse_horizon('1000000000000', horizon, status)
      call check(status == FBC_OK .and. horizon == 1000000000000_int64, &
                 'accepts a 1e12 horizon in int64')
      call fbc_parse_horizon('', horizon, status)
      call check(status /= FBC_OK, 'rejects an empty horizon')
      call fbc_parse_horizon('0', horizon, status)
      call check(status /= FBC_OK, 'rejects a zero horizon')
      call fbc_parse_horizon('-5', horizon, status)
      call check(status /= FBC_OK, 'rejects a negative horizon instead of wrapping')
      call fbc_parse_horizon('+5', horizon, status)
      call check(status /= FBC_OK, 'rejects a signed horizon')
      call fbc_parse_horizon('abc', horizon, status)
      call check(status /= FBC_OK, 'rejects a non-numeric horizon')
      call fbc_parse_horizon('1e12', horizon, status)
      call check(status /= FBC_OK, 'rejects scientific notation instead of guessing')
      call fbc_parse_horizon('12x', horizon, status)
      call check(status /= FBC_OK, 'rejects a trailing junk character')
      call fbc_parse_horizon('99999999999999999999', horizon, status)
      call check(status /= FBC_OK, 'rejects an overflowing horizon')
      call fbc_parse_count('4096', 1_int32, 4096_int32, count, status)
      call check(status == FBC_OK .and. count == 4096_int32, 'accepts a count at its bound')
      call fbc_parse_count('4097', 1_int32, 4096_int32, count, status)
      call check(status /= FBC_OK, 'rejects a count above its bound')
      call fbc_parse_count('0', 1_int32, 4096_int32, count, status)
      call check(status /= FBC_OK, 'rejects a zero count')
      call fbc_parse_task('needle', task, status)
      call check(status == FBC_OK .and. task == FBC_TASK_NEEDLE, 'parses the needle task')
      call fbc_parse_task('other', task, status)
      call check(status /= FBC_OK, 'rejects an unknown task')
   end subroutine test_option_parsing

   !----------------------------------------------------------------------------
   !  2. Horizon invariance
   !----------------------------------------------------------------------------
   subroutine test_horizon_invariance
      integer(int64), parameter :: horizons(4) = [32_int64, 8192_int64, 1000000_int64, &
                                                  1000000000000_int64]
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_plan_t) :: plan, reference
      integer(int32) :: index, status
      integer(int64) :: reference_state_elements, reference_parameters
      real(real64) :: reference_decay, reference_effective
      logical :: sizes_match, counters_match, decay_moved, horizon_moved

      sizes_match = .true.
      counters_match = .true.
      decay_moved = .false.
      horizon_moved = .false.
      reference_state_elements = -1_int64
      reference_parameters = -1_int64
      reference_decay = 0.0_real64
      reference_effective = 0.0_real64
      reference = fbc_plan_t()

      do index = 1_int32, size(horizons, kind=int32)
         config = small_config(horizons(index))
         call check(fbc_config_is_valid(config), 'configuration is valid at every horizon')
         call fbc_plan(config, plan)
         call fbc_model_init(model, config, 7_int64, status)
         call fbc_state_init(state, model, status)
         if (index == 1_int32) then
            reference = plan
            reference_parameters = plan%parameter_count
            reference_state_elements = int(size(state%retention_numerator), int64)
            reference_decay = model%scale_decay(size(model%scale_decay, kind=int32))
            reference_effective = model%effective_horizon
         else
            sizes_match = sizes_match .and. &
                          plan%parameter_count == reference_parameters .and. &
                          plan%state_bytes_real64 == reference%state_bytes_real64 .and. &
                          plan%local_cache_elements == reference%local_cache_elements .and. &
                          plan%slot_state_elements == reference%slot_state_elements .and. &
                          int(size(state%retention_numerator), int64) == reference_state_elements .and. &
                          plan%parameter_bytes_real64 == 8_int64 * reference_parameters
            counters_match = counters_match .and. &
                             plan%forward_flops_per_token == reference%forward_flops_per_token .and. &
                             plan%backward_flops_per_token == reference%backward_flops_per_token .and. &
                             plan%local_score_operations_per_token == &
                             reference%local_score_operations_per_token
            decay_moved = decay_moved .or. &
                          abs(model%scale_decay(size(model%scale_decay, kind=int32)) - &
                              reference_decay) > 1.0e-12_real64
            horizon_moved = horizon_moved .or. abs(model%effective_horizon - reference_effective) > 1.0_real64
         end if
         call fbc_state_destroy(state)
         call fbc_model_destroy(model)
      end do
      call check(sizes_match, 'parameters and stream-state elements are horizon-invariant')
      call check(counters_match, 'per-token operation counters are horizon-invariant')
      call check(decay_moved, 'the slowest decay constant does follow the horizon')
      call check(horizon_moved, 'the reported effective horizon does follow the horizon')
      call check(reference%forward_flops_per_token > 0_int64, 'the plan reports a forward cost')
      call check(reference%state_bytes_real64 > 0_int64, 'the plan reports stream-state bytes')

      ! Arithmetic clamping must be reported, never hidden behind a huge number.
      call fbc_plan(small_config(10000000000000000_int64), plan)
      call check(plan%horizon_clamped, 'flags a horizon beyond real64 resolution')
      call check(plan%effective_horizon <= FBC_MAX_HORIZON, &
                 'clamps the effective horizon to what real64 can represent')
      call fbc_plan(small_config(10000000000000000_int64), plan)
      call check(plan%parameter_count == reference_parameters, &
                 'clamping does not change the physical shape either')
   end subroutine test_horizon_invariance

   !----------------------------------------------------------------------------
   !  3. No future leakage
   !----------------------------------------------------------------------------
   subroutine test_no_future_leakage
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      integer(int32), parameter :: length = 16_int32, vocabulary = 32_int32
      integer(int32) :: status, index, position, token
      integer(int32) :: tokens(length)
      real(real64) :: hidden(vocabulary), logits(vocabulary)
      real(real64) :: baseline(vocabulary, length)
      logical :: identical

      config = small_config(4096_int64)
      call fbc_model_init(model, config, 11_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      do index = 1_int32, length
         tokens(index) = 5_int32 + modulo(index * 7_int32, 20_int32)
      end do
      do index = 1_int32, length
         call fbc_step(model, state, workspace, tokens(index), hidden, logits, status)
         baseline(:, index) = logits
      end do

      ! Re-run the same prefix with the final token replaced.  Every earlier
      ! position must reproduce its logits bit for bit.
      call fbc_state_reset(state, model, status)
      identical = .true.
      do index = 1_int32, length
         token = tokens(index)
         if (index == length) token = 1_int32
         call fbc_step(model, state, workspace, token, hidden, logits, status)
         if (index < length) then
            do position = 1_int32, vocabulary
               if (logits(position) /= baseline(position, index)) identical = .false.
            end do
         end if
      end do
      call check(identical, 'changing a later token cannot change earlier logits')

      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_no_future_leakage

   !----------------------------------------------------------------------------
   !  4. Chunk / stream equivalence
   !----------------------------------------------------------------------------
   !> The engine has no notion of a call boundary: a driver that stops every
   !> `chunk_size` tokens must produce exactly the state and the logits of a
   !> driver that never stops.  This is the property that lets a bounded
   !> activation budget train an unbounded stream.
   subroutine test_chunk_equivalence
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: whole, chunked
      type(fbc_workspace_t) :: workspace
      integer(int32), parameter :: length = 48_int32, chunk_size = 12_int32
      integer(int32) :: status, index, chunk, first, last
      integer(int32) :: tokens(length)
      real(real64) :: hidden(32), logits(32), reference(32, length)
      logical :: logits_match

      config = small_config(65536_int64)
      call fbc_model_init(model, config, 23_int64, status)
      call fbc_state_init(whole, model, status)
      call fbc_state_init(chunked, model, status)
      call fbc_workspace_init(workspace, model, status)
      do index = 1_int32, length
         tokens(index) = 3_int32 + modulo(index * 5_int32, 25_int32)
      end do
      do index = 1_int32, length
         call fbc_step(model, whole, workspace, tokens(index), hidden, logits, status)
         reference(:, index) = logits
      end do

      logits_match = .true.
      do chunk = 1_int32, length / chunk_size
         first = (chunk - 1_int32) * chunk_size + 1_int32
         last = first + chunk_size - 1_int32
         do index = first, last
            call fbc_step(model, chunked, workspace, tokens(index), hidden, logits, status)
            if (any(logits /= reference(:, index))) logits_match = .false.
         end do
      end do
      call check(logits_match, 'chunked processing reproduces the single-pass logits')
      call check(all(chunked%retention_numerator == whole%retention_numerator) .and. &
                 all(chunked%retention_normalizer == whole%retention_normalizer), &
                 'the multi-scale recurrent state is identical after chunked processing')
      call check(all(chunked%slot_key == whole%slot_key) .and. &
                 all(chunked%slot_value == whole%slot_value) .and. &
                 all(chunked%slot_salience == whole%slot_salience), &
                 'the slot state is identical after chunked processing')
      call check(all(chunked%local_key == whole%local_key) .and. all(chunked%local_value == whole%local_value), &
                 'the ring buffer is identical after chunked processing')
      call check(chunked%cache_head == whole%cache_head .and. chunked%cache_count == whole%cache_count .and. &
                 chunked%steps == whole%steps, 'the stream position is identical after chunked processing')

      call fbc_state_destroy(whole)
      call fbc_state_destroy(chunked)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_chunk_equivalence

   !----------------------------------------------------------------------------
   !  5. Determinism
   !----------------------------------------------------------------------------
   subroutine test_determinism
      type(fbc_config_t) :: config
      type(fbc_model_t) :: first, second, third
      type(fbc_state_t) :: state_a, state_b, state_c
      type(fbc_workspace_t) :: workspace
      integer(int32) :: status
      real(real64) :: hidden(32), logits_a(32), logits_b(32), logits_c(32)

      config = small_config(8192_int64)
      call fbc_model_init(first, config, 31_int64, status)
      call fbc_model_init(second, config, 31_int64, status)
      call fbc_model_init(third, config, 32_int64, status)
      ! One state per model: a state carries a stream, so sharing one would feed
      ! the second model a state that the first had already advanced.
      call fbc_state_init(state_a, first, status)
      call fbc_state_init(state_b, second, status)
      call fbc_state_init(state_c, third, status)
      call fbc_workspace_init(workspace, first, status)
      call fbc_step(first, state_a, workspace, 7_int32, hidden, logits_a, status)
      call fbc_step(second, state_b, workspace, 7_int32, hidden, logits_b, status)
      call fbc_step(third, state_c, workspace, 7_int32, hidden, logits_c, status)
      call check(all(logits_a == logits_b), 'identical seeds give bit-identical logits')
      call check(any(logits_a /= logits_c), 'a different seed gives different logits')
      call fbc_state_destroy(state_a)
      call fbc_state_destroy(state_b)
      call fbc_state_destroy(state_c)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(first)
      call fbc_model_destroy(second)
      call fbc_model_destroy(third)
   end subroutine test_determinism

   !----------------------------------------------------------------------------
   !  6. Long stream
   !----------------------------------------------------------------------------
   subroutine test_long_stream_finite
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      integer(int32), parameter :: steps = 1500_int32
      integer(int32) :: status, index
      real(real64) :: hidden(32), logits(32), bound
      logical :: finite

      config = small_config(1000000000000_int64)
      call fbc_model_init(model, config, 41_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      finite = .true.
      do index = 1_int32, steps
         call fbc_step(model, state, workspace, 1_int32 + modulo(index * 13_int32, 31_int32), &
                       hidden, logits, status)
         if (status /= FBC_OK .or. any(logits /= logits) .or. maxval(abs(logits)) > 1.0e8_real64) then
            finite = .false.
            exit
         end if
      end do
      bound = max(maxval(abs(state%retention_numerator)), maxval(abs(state%retention_normalizer)))
      call check(finite, 'a 1500-token stream at a 1e12 horizon stays finite')
      call check(bound < 1.0e7_real64, 'the recurrent state magnitude stays bounded')
      call check(state%steps == int(steps, int64), 'the step counter matches the tokens consumed')
      call check(state%cache_count == config%local_window, 'the ring buffer fills and then stays fixed')
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_long_stream_finite

   !----------------------------------------------------------------------------
   !  7. Finite-difference gradient check of the analytic backward
   !----------------------------------------------------------------------------
   !> The backward pass is the exact derivative of one implemented step taken
   !> from a frozen state (depth-1 truncated BPTT).  So a central difference of
   !> exactly that one-step loss, taken from the same frozen state, must match
   !> the analytic gradient.  Eight parameters are probed, one per block of the
   !> network: readout weight, channel logit, the two slot write projections,
   !> the slot query projection, the distance bias, an FFN projection and the
   !> tied embedding.
   subroutine test_gradient_check
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: base, probe
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: gradient
      integer(int32), parameter :: vocabulary = 16_int32, width = 8_int32
      integer(int32) :: status, index
      integer(int32) :: warm(3)
      real(real64) :: hidden(width), logits(vocabulary), dlogits(vocabulary), loss, delta, original
      real(real64) :: plus_loss, minus_loss, numeric

      config = tiny_config(1024_int64)
      call fbc_model_init(model, config, 5_int64, status)
      call fbc_state_init(base, model, status)
      call fbc_workspace_init(workspace, model, status)
      call fbc_gradient_init(gradient, model, status)
      warm = [3_int32, 7_int32, 11_int32]
      do index = 1_int32, 3
         call fbc_step(model, base, workspace, warm(index), hidden, logits, status)
      end do

      ! Analytic gradient of the next single step, from the frozen state.
      call fbc_gradient_zero(gradient)
      call fbc_state_init(probe, model, status)
      probe = base
      call fbc_step(model, probe, workspace, 9_int32, hidden, logits, status)
      call fbc_loss_and_gradient(logits, 5_int32, dlogits, loss, status)
      call fbc_step_backward(model, probe, workspace, dlogits, gradient, status)
      delta = 1.0e-6_real64

      block
         original = model%scale_logit(1_int32)
         model%scale_logit(1_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%scale_logit(1_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%scale_logit(1_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%scale_logit(1_int32), &
                    'finite difference matches the multi-scale readout gradient')
      end block
      block
         original = model%channel_logit(3_int32)
         model%channel_logit(3_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%channel_logit(3_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%channel_logit(3_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%channel_logit(3_int32), &
                    'finite difference matches the slot-channel mixing gradient')
      end block
      block
         original = model%slot_key_proj(2_int32, 3_int32)
         model%slot_key_proj(2_int32, 3_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%slot_key_proj(2_int32, 3_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%slot_key_proj(2_int32, 3_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%slot_key_proj(2_int32, 3_int32), &
                    'finite difference matches the slot write-key gradient (write-then-read)')
      end block
      block
         original = model%slot_value_proj(4_int32, 1_int32)
         model%slot_value_proj(4_int32, 1_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%slot_value_proj(4_int32, 1_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%slot_value_proj(4_int32, 1_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%slot_value_proj(4_int32, 1_int32), &
                    'finite difference matches the slot write-value gradient')
      end block
      block
         original = model%slot_query_proj(3_int32, 5_int32)
         model%slot_query_proj(3_int32, 5_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%slot_query_proj(3_int32, 5_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%slot_query_proj(3_int32, 5_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%slot_query_proj(3_int32, 5_int32), &
                    'finite difference matches the slot recall-query gradient')
      end block
      block
         original = model%decay_logit(1_int32)
         model%decay_logit(1_int32) = original + delta
         call fbc_refresh_dynamics(model)
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%decay_logit(1_int32) = original - delta
         call fbc_refresh_dynamics(model)
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%decay_logit(1_int32) = original
         call fbc_refresh_dynamics(model)
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%decay_logit(1_int32), &
                    'finite difference matches the learned decay gradient')
      end block
      block
         original = model%write_strength_logit(2_int32)
         model%write_strength_logit(2_int32) = original + delta
         call fbc_refresh_dynamics(model)
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%write_strength_logit(2_int32) = original - delta
         call fbc_refresh_dynamics(model)
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%write_strength_logit(2_int32) = original
         call fbc_refresh_dynamics(model)
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%write_strength_logit(2_int32), &
                    'finite difference matches the learned write-strength gradient')
      end block
      block
         original = model%distance_bias(2_int32)
         model%distance_bias(2_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%distance_bias(2_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%distance_bias(2_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%distance_bias(2_int32), &
                    'finite difference matches the relative-distance bias gradient')
      end block
      block
         original = model%ffn_gate(3_int32, 4_int32)
         model%ffn_gate(3_int32, 4_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%ffn_gate(3_int32, 4_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%ffn_gate(3_int32, 4_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%ffn_gate(3_int32, 4_int32), &
                    'finite difference matches the FFN gate gradient')
      end block
      block
         original = model%query(2_int32, 5_int32, 1_int32)
         model%query(2_int32, 5_int32, 1_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%query(2_int32, 5_int32, 1_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%query(2_int32, 5_int32, 1_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%query(2_int32, 5_int32, 1_int32), &
                    'finite difference matches a query projection gradient')
      end block
      block
         original = model%value(2_int32, 3_int32, 1_int32)
         model%value(2_int32, 3_int32, 1_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%value(2_int32, 3_int32, 1_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%value(2_int32, 3_int32, 1_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%value(2_int32, 3_int32, 1_int32), &
                    'finite difference matches a value projection gradient (intra-step retention write)')
      end block
      block
         original = model%key(1_int32, 4_int32, 2_int32)
         model%key(1_int32, 4_int32, 2_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%key(1_int32, 4_int32, 2_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%key(1_int32, 4_int32, 2_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%key(1_int32, 4_int32, 2_int32), &
                    'finite difference matches a key projection gradient (intra-step retention write)')
      end block
      block
         original = model%token_embedding(3_int32, 9_int32)
         model%token_embedding(3_int32, 9_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%token_embedding(3_int32, 9_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%token_embedding(3_int32, 9_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%token_embedding(3_int32, 9_int32), &
                    'finite difference matches the tied-embedding gradient')
      end block

      call check(numeric == numeric .and. abs(numeric) > 0.0_real64, &
                 'the last finite-difference probe actually moves the loss')

      call fbc_gradient_destroy(gradient)
      call fbc_state_destroy(base)
      call fbc_state_destroy(probe)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_gradient_check

   !> Loss of one step replayed from a frozen state: exactly the objective the
   !> depth-1 truncated backward pass differentiates.
   subroutine replay_loss(model, base, workspace, token, target, loss, status)
      type(fbc_model_t), intent(in) :: model
      type(fbc_state_t), intent(in) :: base
      type(fbc_workspace_t), intent(inout) :: workspace
      integer(int32), intent(in) :: token, target
      real(real64), intent(out) :: loss
      integer(int32), intent(out) :: status
      type(fbc_state_t) :: probe
      ! Sized from the dummy model: a specification expression, so this is an
      ! automatic array and costs no heap traffic.
      real(real64) :: hidden(model%config%width)
      real(real64) :: logits(model%config%vocabulary_size)
      real(real64) :: dlogits(model%config%vocabulary_size)

      call fbc_state_init(probe, model, status)
      probe = base
      call fbc_step(model, probe, workspace, token, hidden, logits, status)
      call fbc_loss_and_gradient(logits, target, dlogits, loss, status)
      call fbc_state_destroy(probe)
   end subroutine replay_loss

   !> Compares one finite-difference probe with its analytic counterpart and
   !> echoes both numbers, so a failure names the value instead of only the
   !> parameter.
   subroutine report_gradient(numeric, analytic, description)
      real(real64), intent(in) :: numeric, analytic
      character(len=*), intent(in) :: description
      real(real64) :: scale, error

      scale = max(1.0e-7_real64, abs(numeric) + abs(analytic))
      error = abs(numeric - analytic) / scale
      if (error >= 1.0e-4_real64) then
         write(output_unit, '(A,2ES14.6,A,ES10.3)') '  gradient probe numeric/analytic ', numeric, &
            analytic, ' relerr ', error
      end if
      call check(error < 1.0e-4_real64, description)
   end subroutine report_gradient

   !----------------------------------------------------------------------------
   !  8. Training reduces the loss
   !----------------------------------------------------------------------------
   subroutine test_training_reduces_loss
      type(fbc_config_t) :: config
      type(fbc_update_config_t) :: update_config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: gradient
      type(fbc_optimizer_t) :: optimizer
      type(fbc_metrics_t) :: metrics
      integer(int32), parameter :: token_count = 6144_int32, chunk = 16_int32
      integer(int32) :: status
      integer(int32), allocatable :: tokens(:)
      logical, allocatable :: mask(:)
      real(real64) :: before, after, accuracy
      integer(int64) :: answer_count

      config = small_config(4096_int64)
      config%vocabulary_size = 8_int32      ! a five-way deterministic successor rule
      config%width = 16_int32               ! width 32 would waste CI time here
      config%ffn_width = 32_int32
      update_config%learning_rate = 2.0e-3_real64

      allocate(tokens(token_count + 1_int32), mask(token_count))
      call fbc_generate_tokens(FBC_TASK_MARKOV, token_count, config%vocabulary_size, 16_int32, 3_int64, &
                               tokens(1_int32:token_count), mask, status)
      call check(status == FBC_OK, 'the markov corpus generates')
      tokens(token_count + 1_int32) = tokens(1_int32)

      call fbc_model_init(model, config, 3_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      call fbc_gradient_init(gradient, model, status)
      call fbc_optimizer_init(optimizer, config, status)

      call fbc_evaluate_stream(model, state, tokens, token_count, mask, workspace, before, accuracy, &
                               answer_count, status)
      call check(status == FBT_OK, 'pre-training evaluation succeeds')
      call fbc_state_reset(state, model, status)
      call fbc_train_stream(model, optimizer, state, tokens, token_count, chunk, update_config, &
                            workspace, gradient, metrics, status)
      call check(status == FBT_OK, 'chunked training completes')
      call check(metrics%tokens == int(token_count, int64), 'training consumes the whole token budget')
      call check(metrics%updates >= 1_int64, 'training performs at least one optimiser update')
      call fbc_state_reset(state, model, status)
      call fbc_evaluate_stream(model, state, tokens, token_count, mask, workspace, after, accuracy, &
                               answer_count, status)
      call check(after < before, 'training reduces the loss on a learnable task')
      write(output_unit, '(A,F0.4,A,F0.4)') '  markov nll before/after: ', before, ' / ', after

      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_gradient_destroy(gradient)
      call fbc_optimizer_destroy(optimizer)
      call fbc_model_destroy(model)
      deallocate(tokens, mask)
   end subroutine test_training_reduces_loss

   !----------------------------------------------------------------------------
   !  9. Every memory component is trained
   !----------------------------------------------------------------------------
   subroutine test_memory_components_train
      type(fbc_config_t) :: config
      type(fbc_update_config_t) :: update_config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: gradient
      type(fbc_optimizer_t) :: optimizer
      type(fbc_metrics_t) :: metrics
      integer(int32), parameter :: token_count = 128_int32, chunk = 32_int32
      integer(int32) :: status, index
      integer(int32), allocatable :: tokens(:)
      logical, allocatable :: mask(:)
      real(real64) :: slots_before(32, 16), scales_before(3), channels_before(3), salience_before
      real(real64) :: decays_before(3), writes_before(3)
      real(real64), allocatable :: hidden(:), logits(:), dlogits(:)
      real(real64) :: loss, norm

      config = small_config(2048_int64)
      update_config%learning_rate = 3.0e-3_real64
      allocate(tokens(token_count + 1_int32), mask(token_count))
      call fbc_generate_tokens(FBC_TASK_NEEDLE, token_count, config%vocabulary_size, 32_int32, 9_int64, &
                               tokens(1_int32:token_count), mask, status)
      call check(status == FBC_OK, 'the needle corpus generates')
      tokens(token_count + 1_int32) = 1_int32

      call fbc_model_init(model, config, 9_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      call fbc_gradient_init(gradient, model, status)
      call fbc_optimizer_init(optimizer, config, status)
      allocate(hidden(config%width), logits(config%vocabulary_size), dlogits(config%vocabulary_size))

      ! Warm the stream first: a fresh store has every slot identical, so the
      ! read is then independent of the keys and their gradient is exactly zero
      ! for a legitimate reason. Four tokens make the slots distinct.
      do index = 1_int32, 4
         call fbc_step(model, state, workspace, tokens(index), hidden, logits, status)
      end do
      ! One single-token backward proves that every new component sits on a
      ! differentiable path, including the slot write (write-then-read).
      call fbc_gradient_zero(gradient)
      call fbc_step(model, state, workspace, tokens(5_int32), hidden, logits, status)
      call fbc_loss_and_gradient(logits, tokens(6_int32), dlogits, loss, status)
      call fbc_step_backward(model, state, workspace, dlogits, gradient, status)
      call check(sum(abs(gradient%slot_key_proj)) > 0.0_real64, &
                 'the slot write-key projection receives a gradient')
      call check(sum(abs(gradient%slot_value_proj)) > 0.0_real64, &
                 'the slot write-value projection receives a gradient')
      call check(sum(abs(gradient%slot_query_proj)) > 0.0_real64, &
                 'the slot recall-query projection receives a gradient')
      call check(sum(abs(gradient%slot_read_proj)) > 0.0_real64, &
                 'the slot read projection receives a gradient')
      call check(sum(abs(gradient%scale_logit)) > 0.0_real64, &
                 'the multi-scale readout receives a gradient')
      call check(sum(abs(gradient%channel_logit)) > 0.0_real64, &
                 'the channel mixer receives a gradient')
      call check(sum(abs(gradient%attention_rms)) > 0.0_real64 .and. &
                 sum(abs(gradient%ffn_rms)) > 0.0_real64, &
                 'both normalization scales receive a gradient')
      call check(abs(gradient%slot_salience_weight) > 0.0_real64, &
                 'the slot salience weight receives a gradient')
      call check(sum(abs(gradient%decay_logit)) > 0.0_real64, &
                 'the memory decay logits receive a gradient')
      call check(sum(abs(gradient%write_strength_logit)) > 0.0_real64, &
                 'the memory write-strength logits receive a gradient')
      norm = fbc_gradient_norm(gradient)
      call check(norm > 0.0_real64 .and. norm == norm, 'the gradient norm is finite and positive')

      slots_before = state%slot_key(1_int32:32, 1_int32:16)
      scales_before = model%scale_logit
      decays_before = model%decay_logit
      writes_before = model%write_strength_logit
      channels_before = model%channel_logit
      salience_before = model%slot_salience_weight
      call fbc_train_stream(model, optimizer, state, tokens, token_count, chunk, update_config, &
                            workspace, gradient, metrics, status)
      call check(status == FBT_OK, 'needle training completes')
      call check(maxval(abs(state%slot_key(1_int32:32, 1_int32:16) - slots_before)) > 0.0_real64, &
                 'the slot state really changes while training')
      call check(any(model%scale_logit /= scales_before), &
                 'the multi-scale readout weights are optimised')
      call check(any(model%channel_logit /= channels_before), 'the channel mixer is optimised')
      call check(model%slot_salience_weight /= salience_before, 'the salience weight is optimised')
      call check(any(model%decay_logit /= decays_before), 'the memory decay is optimised')
      call check(any(model%write_strength_logit /= writes_before), 'the memory write strength is optimised')
      call check(all(model%scale_logit == model%scale_logit), 'the readout weights stay finite')

      deallocate(hidden, logits, dlogits)
      deallocate(tokens, mask)
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_gradient_destroy(gradient)
      call fbc_optimizer_destroy(optimizer)
      call fbc_model_destroy(model)
   end subroutine test_memory_components_train

   !----------------------------------------------------------------------------
   !  10. Remote influence through the fixed-size state
   !----------------------------------------------------------------------------
   !> A hand-built needle stream: KEY, ANSWER at position 2, 93 fillers, QUERY at
   !> position 96, so the answer sits 93 tokens behind the query while the
   !> physical local window is 8.  Changing only the answer must move the logits
   !> at the query when the memory channels are live, and must not move them at
   !> all when those channels are switched off.
   subroutine test_remote_influence
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      integer(int32), parameter :: length = 96_int32, query_position = 96_int32
      integer(int32) :: status, index
      integer(int32) :: tokens(length)
      real(real64) :: hidden(32), logits(32), with_memory, without_memory

      config = small_config(1000000_int64)
      config%local_window = 8_int32
      tokens = 17_int32
      tokens(1_int32) = 1_int32                                  ! KEY
      tokens(2_int32) = 5_int32                                  ! ANSWER, 94 steps back
      tokens(query_position) = 2_int32                           ! QUERY
      do index = 3_int32, query_position - 1_int32
         tokens(index) = 24_int32                                 ! fillers
      end do
      call check(query_position - 2_int32 > config%local_window, &
                 'the answer lies far outside the physical local window')

      call fbc_model_init(model, config, 17_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      call probe_influence(model, state, workspace, tokens, query_position, 5_int32, 9_int32, &
                           with_memory, status)
      write(output_unit, '(A,ES12.4)') '  remote influence with memory       : ', with_memory
      call check(with_memory > 1.0e-6_real64, &
                 'the fixed-size state carries a 93-token-distant answer into the query logits')

      ! Same stream, memory channels disabled: only the ring buffer remains, and
      ! it cannot reach 93 tokens back.
      model%channel_logit = [0.0_real64, -700.0_real64, -700.0_real64]
      call probe_influence(model, state, workspace, tokens, query_position, 5_int32, 9_int32, &
                           without_memory, status)
      write(output_unit, '(A,ES12.4)') '  remote influence without memory    : ', without_memory
      call check(without_memory < 1.0e-10_real64, &
                 'with memory disabled the local window cannot see that far (no leakage)')

      ! A short-range change must still be visible, otherwise the previous check
      ! would be satisfied by a model that ignores its input entirely.
      call probe_influence(model, state, workspace, tokens, query_position, query_position - 3_int32, &
                           25_int32, without_memory, status)
      call check(without_memory > 1.0e-6_real64, &
                 'the local ring buffer does carry a token inside its window')

      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_remote_influence

   !> Runs the stream twice, changing only the token at `changed_position`, and
   !> returns the maximum absolute change in the logits at `query_position`.
   subroutine probe_influence(model, state, workspace, tokens, query_position, changed_position, &
                              replacement, influence, status)
      type(fbc_model_t), intent(in) :: model
      type(fbc_state_t), intent(inout) :: state
      type(fbc_workspace_t), intent(inout) :: workspace
      integer(int32), intent(in) :: tokens(:), query_position, changed_position, replacement
      real(real64), intent(out) :: influence
      integer(int32), intent(out) :: status
      real(real64) :: hidden(32), logits(32), first(32), second(32)
      integer(int32) :: index, token
      integer(int32) :: reference(size(tokens))

      reference = tokens
      influence = 0.0_real64
      first = 0.0_real64
      second = 0.0_real64
      do index = 1_int32, 2
         call fbc_state_reset(state, model, status)
         do token = 1_int32, query_position
            if (index == 2_int32 .and. token == changed_position) then
               call fbc_step(model, state, workspace, replacement, hidden, logits, status)
            else
               call fbc_step(model, state, workspace, reference(token), hidden, logits, status)
            end if
         end do
         if (index == 1_int32) then
            first = logits
         else
            second = logits
         end if
      end do
      influence = maxval(abs(first - second))
   end subroutine probe_influence

   !----------------------------------------------------------------------------
   !  11. Checkpoint round trip and legacy rejection
   !----------------------------------------------------------------------------
   subroutine test_checkpoint_roundtrip
      character(len=*), parameter :: path = '/tmp/fbc-roundtrip.vgfb'
      character(len=*), parameter :: legacy = '/tmp/fbc-legacy.vgfb'
      type(fbc_config_t) :: config
      type(fbc_update_config_t) :: update_config, loaded_config
      type(fbc_model_t) :: model, restored
      type(fbc_optimizer_t) :: optimizer, restored_optimizer
      type(fbc_metrics_t) :: counters, restored_counters
      type(fbc_checkpoint_meta_t) :: meta
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      integer(int32) :: status, index
      real(real64) :: hidden(32), logits(32)
      integer :: unit, io_status

      config = small_config(1000000000000_int64)
      call fbc_model_init(model, config, 13_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      call fbc_optimizer_init(optimizer, config, status)
      do index = 1_int32, 5
         call fbc_step(model, state, workspace, 3_int32 + index, hidden, logits, status)
      end do
      counters = fbc_metrics_t()
      counters%updates = 4_int64
      counters%tokens = 256_int64
      counters%loss_sum = 12.5_real64
      update_config%learning_rate = 1.5e-3_real64

      call fbc_checkpoint_save(path, model, optimizer, update_config, counters, status)
      call check(status == FBT_OK, 'checkpoint save succeeds')

      call fbc_checkpoint_load(path, restored, restored_optimizer, loaded_config, restored_counters, status)
      call check(status == FBT_OK, 'checkpoint load succeeds')
      call check(restored_counters%updates == counters%updates .and. &
                 restored_counters%tokens == counters%tokens, 'counters survive the round trip')
      call check(abs(loaded_config%learning_rate - update_config%learning_rate) == 0.0_real64, &
                 'optimizer hyper-parameters survive the round trip')
      call check(abs(restored%slot_salience_weight - model%slot_salience_weight) == 0.0_real64, &
                 'the salience weight survives the round trip')
      call check(all(restored%decay_logit == model%decay_logit) .and. &
                 all(restored%write_strength_logit == model%write_strength_logit) .and. &
                 all(restored%write_strength == model%write_strength), &
                 'the learnable memory dynamics survive the round trip')
      call check(all(restored%scale_logit == model%scale_logit) .and. &
                 all(restored%channel_logit == model%channel_logit) .and. &
                 all(restored%slot_key_proj == model%slot_key_proj), &
                 'every memory weight survives the round trip bit for bit')
      call check(all(restored%token_embedding == model%token_embedding) .and. &
                 all(restored%ffn_gate == model%ffn_gate), 'the rest of the parameters too')
      call check((restored%horizon_clamped .eqv. model%horizon_clamped) .and. &
                 (abs(restored%effective_horizon - model%effective_horizon) == 0.0_real64), &
                 'the effective horizon and clamp flag survive the round trip')
      call check(restored%config%logical_horizon == config%logical_horizon .and. &
                 restored%config%local_window == config%local_window .and. &
                 restored%config%memory_scales == config%memory_scales .and. &
                 restored%config%memory_slots == config%memory_slots, &
                 'the configuration, including the logical horizon, survives the round trip')

      ! Logit equality from the freshly loaded model.
      block
         type(fbc_state_t) :: saved_state, restored_state
         real(real64) :: reference_logits(32), loaded_logits(32)

         ! Both models start from an empty stream, so the comparison isolates the
         ! parameters instead of an accumulated state.
         call fbc_state_init(saved_state, model, status)
         call fbc_state_init(restored_state, restored, status)
         call fbc_step(model, saved_state, workspace, 11_int32, hidden, reference_logits, status)
         call fbc_step(restored, restored_state, workspace, 11_int32, hidden, loaded_logits, status)
         call check(all(reference_logits == loaded_logits), &
                    'the reloaded model reproduces the logits of the saved one exactly')
         call fbc_state_destroy(saved_state)
         call fbc_state_destroy(restored_state)
      end block

      call fbc_checkpoint_meta_read(path, meta, status)
      call check(status == FBT_OK .and. meta%logical_horizon == config%logical_horizon, &
                 'the header reports the stored logical horizon')
      call check(meta%local_window == config%local_window .and. &
                 meta%memory_scales == config%memory_scales .and. &
                 meta%memory_slots == config%memory_slots .and. &
                 meta%vocabulary_size == config%vocabulary_size, &
                 'the header reports the physical configuration')

      ! A legacy header must be rejected with its own status, never converted.
      open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      call check(io_status == 0, 'the legacy fixture can be written')
      write(unit) FBC_LEGACY_MAGIC
      write(unit) 5_int32
      close(unit)
      call fbc_checkpoint_meta_read(legacy, meta, status)
      call check(status == FBT_ERR_LEGACY, 'a legacy VGNR header is rejected with a distinct status')
      call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, &
                               restored_counters, status)
      call check(status == FBT_ERR_LEGACY, 'loading a legacy checkpoint never converts it silently')

      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
      call fbc_model_destroy(restored)
      call fbc_optimizer_destroy(optimizer)
      call fbc_optimizer_destroy(restored_optimizer)
   end subroutine test_checkpoint_roundtrip

end program fixed_budget_context_tests
