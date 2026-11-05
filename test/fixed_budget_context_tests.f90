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
!    9. legacy memory training paths and the symmetric-slot limitation
!   10. remote influence: an answer 93 tokens behind the query reaches the
!       query logits through the fixed-size state and provably cannot reach
!       them through the local ring buffer alone
!   11. checkpoint round-trip equality and explicit legacy rejection
!   12. the context router: zero rows reproduce the input-only gate bit for
!       bit, and a non-zero router mixes the channels differently depending on
!       what the three channels returned for *that* token
!   13. the induction (shifted) read: the local channel can read the value that
!       followed the position its attention matched, i.e. "after this byte came
!       that byte", which a single in-place read cannot express
!   14. the plan counts what exists: parameter count against the real arrays,
!       state-buffer and byte counters against the real allocations, and the
!       optimiser's update counter against the same number (every parameter
!       exactly once, none twice, none missed)
!   15. a reset state is indistinguishable from a freshly initialised one
!   16. a *full* ring buffer: every slot holds the key/value of the token at its
!       distance and the shifted read equals the hand-computed sum -- the
!       finite-difference package warms up three steps with a window of four,
!       so it never sees the regime training actually runs in
!
!  Nothing here is timing based; the invariance checks read structural
!  counters, so they cannot pass or fail because a runner was busy.
!==============================================================================
program fixed_budget_context_tests
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64, output_unit
   use portable_cli, only: cli_valid_utf8
   use utf8_generation
   use fixed_budget_context
   use fixed_budget_backward
   use fixed_budget_optimizer
   use fixed_budget_training
   use fixed_budget_natural_eval
   implicit none (type, external)

   integer(int32) :: checks = 0_int32, failures = 0_int32

   call test_defaults
   call test_utf8_generation
   call test_natural_record_eval
   call test_option_parsing
   call test_horizon_invariance
   call test_no_future_leakage
   call test_chunk_equivalence
   call test_determinism
   call test_long_stream_finite
   call test_legacy_slot_symmetry
   call test_anchored_slot_diversity
   call test_slot_rule_state_binding
   call test_stable_loss
   call test_numeric_boundaries
   call test_stream_error_boundaries
   call test_workspace_gradient_binding
   call test_optimizer_rule
   call test_gradient_check(FBC_TRAINING_RULE_V6)
   call test_gradient_check(FBC_TRAINING_RULE_V7)
   call test_outer_accumulation_parity
   call test_full_ring_backward(FBC_TRAINING_RULE_V6)
   call test_full_ring_backward(FBC_TRAINING_RULE_V7)
   call test_clamped_decay_backward
   call test_training_reduces_loss
   call test_partial_training_chunk
   call test_legacy_memory_training
   call test_remote_influence
   call test_checkpoint_roundtrip
   call test_full_ring_checkpoint_continuation
   call test_context_router_conditions_mixture
   call test_induction_read_is_shifted
   call test_plan_counts_match_reality
   call test_state_reset_is_fresh
   call test_full_window_ring

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

   !> UTF-8 generation masks, including boundary scalars and exhausted byte budgets.
   subroutine test_utf8_generation
      type(utf8_generation_state_t) :: state, copy
      integer(int32) :: token, budget, count_allowed, lead, second, third, fourth, count_second, i, j
      integer(int32) :: leads(4), minima(4), maxima(4), lengths(4), expected_counts(4), scalar_count, scalar_length
      type(utf8_generation_state_t) :: after_lead, after_second, after_third
      logical :: accepted, correct, independent_ok
      character(len=4) :: codepoint

      expected_counts = [96_int32, 126_int32, 142_int32, 147_int32]
      do budget = 1_int32, 4_int32
         call utf8_generation_reset(state)
         count_allowed = 0_int32
         do token = 1_int32, 256_int32
            if (utf8_generation_allows(state, token, budget)) count_allowed = count_allowed + 1_int32
         end do
         call check(count_allowed == expected_counts(budget), 'UTF-8 lead mask respects remaining byte budget')
      end do

      ! A complete legal two-byte codepoint is independently checked using the
      ! CLI validator, not the generation state itself.
      independent_ok = .true.
      do lead = 194_int32, 223_int32
         do second = 128_int32, 191_int32
            call utf8_generation_reset(state)
            call utf8_generation_advance(state, lead, 2_int32, accepted)
            independent_ok = independent_ok .and. accepted .and. .not. utf8_generation_complete(state)
            call utf8_generation_advance(state, second, 1_int32, accepted)
            codepoint = achar(lead) // achar(second) // '  '
            independent_ok = independent_ok .and. accepted .and. utf8_generation_complete(state) .and. &
                             cli_valid_utf8(codepoint(1:2))
         end do
      end do
      call check(independent_ok, 'all 1,920 legal two-byte scalars agree with the independent validator')

      ! Four constrained second-byte ranges prohibit overlong encodings,
      ! surrogates and codepoints above U+10FFFF.
      leads = [224_int32, 237_int32, 240_int32, 244_int32]
      minima = [160_int32, 128_int32, 144_int32, 128_int32]
      maxima = [191_int32, 159_int32, 191_int32, 143_int32]
      lengths = [3_int32, 3_int32, 4_int32, 4_int32]
      correct = .true.
      do i = 1_int32, size(leads)
         call utf8_generation_reset(state)
         call utf8_generation_advance(state, leads(i), lengths(i), accepted)
         correct = correct .and. accepted
         count_second = 0_int32
         do token = 1_int32, 256_int32
            if (utf8_generation_allows(state, token, lengths(i) - 1_int32)) then
               count_second = count_second + 1_int32
               if (token < minima(i) .or. token > maxima(i)) correct = .false.
            end if
         end do
         correct = correct .and. count_second == maxima(i) - minima(i) + 1_int32
         correct = correct .and. .not. utf8_generation_allows(state, 10_int32, lengths(i) - 1_int32)
         correct = correct .and. .not. utf8_generation_allows(state, 46_int32, lengths(i) - 1_int32)
         copy = state
         call utf8_generation_advance(copy, minima(i) - 1_int32, lengths(i) - 1_int32, accepted)
         correct = correct .and. .not. accepted .and. &
                   utf8_generation_allows(copy, minima(i), lengths(i) - 1_int32)
         call utf8_generation_advance(state, minima(i), lengths(i) - 1_int32, accepted)
         correct = correct .and. accepted
         do j = 3_int32, lengths(i)
            call utf8_generation_advance(state, 128_int32, lengths(i) - j + 1_int32, accepted)
            correct = correct .and. accepted
         end do
         codepoint = achar(leads(i)) // achar(minima(i)) // achar(128) // achar(128)
         correct = correct .and. utf8_generation_complete(state) .and. cli_valid_utf8(codepoint(1:lengths(i)))
      end do
      call check(correct, 'UTF-8 constrained continuation ranges and rejection preserve state')

      ! Exhaust all legal three- and four-byte Unicode scalars. The independent
      ! whole-string validator checks every completed state, including boundary
      ! continuations after the specially restricted first byte.
      scalar_count = 0_int32
      independent_ok = .true.
      do lead = 224_int32, 244_int32
         if (lead <= 239_int32) then
            scalar_length = 3_int32
         else
            scalar_length = 4_int32
         end if
         call utf8_generation_reset(after_lead)
         call utf8_generation_advance(after_lead, lead, scalar_length, accepted)
         independent_ok = independent_ok .and. accepted
         do second = 128_int32, 191_int32
            if (.not. utf8_generation_allows(after_lead, second, scalar_length - 1_int32)) cycle
            after_second = after_lead
            call utf8_generation_advance(after_second, second, scalar_length - 1_int32, accepted)
            independent_ok = independent_ok .and. accepted
            do third = 128_int32, 191_int32
               after_third = after_second
               call utf8_generation_advance(after_third, third, scalar_length - 2_int32, accepted)
               independent_ok = independent_ok .and. accepted
               if (scalar_length == 3_int32) then
                  codepoint = achar(lead) // achar(second) // achar(third) // ' '
                  independent_ok = independent_ok .and. utf8_generation_complete(after_third) .and. &
                                   cli_valid_utf8(codepoint(1:3))
                  scalar_count = scalar_count + 1_int32
               else
                  do fourth = 128_int32, 191_int32
                     state = after_third
                     call utf8_generation_advance(state, fourth, 1_int32, accepted)
                     codepoint = achar(lead) // achar(second) // achar(third) // achar(fourth)
                     independent_ok = independent_ok .and. accepted .and. utf8_generation_complete(state) .and. &
                                      cli_valid_utf8(codepoint)
                     scalar_count = scalar_count + 1_int32
                  end do
               end if
            end do
         end do
      end do
      call check(independent_ok .and. scalar_count == 1110016_int32, &
                 'all 1,110,016 valid three- and four-byte scalars agree with the independent validator')

      call utf8_generation_reset(state)
      call check(.not. utf8_generation_allows(state, 194_int32, 1_int32) .and. &
                 .not. utf8_generation_allows(state, 240_int32, 3_int32) .and. &
                 .not. utf8_generation_allows(state, 255_int32, 4_int32) .and. &
                 .not. utf8_generation_allows(state, 256_int32, 4_int32) .and. &
                 .not. utf8_generation_allows(state, 0_int32, 4_int32), &
                 'UTF-8 generation forbids incomplete, invalid and NUL byte tokens')
      call utf8_generation_advance(state, 240_int32, 4_int32, accepted)
      call check(accepted .and. .not. utf8_generation_allows(state, 144_int32, 2_int32), &
                 'UTF-8 pending continuation cannot exceed the remaining byte budget')
      call utf8_generation_reset(state)
      call check(utf8_generation_complete(state) .and. utf8_generation_allows(state, 65_int32, 1_int32), &
                 'UTF-8 decoder reset starts at a codepoint boundary')
   end subroutine test_utf8_generation

   !> Compare cold-record NLL with independent per-record stream evaluations.
   subroutine test_natural_record_eval
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      type(fbc_record_metrics_t) :: metrics
      integer(int32) :: status
      integer(int64) :: answer_count, old_steps
      real(real64) :: first_nll, first_prefix, second_nll, accuracy
      integer(int32), parameter :: records(8) = [1_int32, 2_int32, 3_int32, 10_int32, &
                                               4_int32, 5_int32, 10_int32, 6_int32]

      config = fbc_config_t(vocabulary_size=32_int32, width=8_int32, heads=2_int32, ffn_width=16_int32, &
                            local_window=4_int32, memory_scales=2_int32, memory_slots=4_int32, &
                            logical_horizon=128_int64)
      call fbc_model_init(model, config, 42_int64, status)
      call check(status == FBC_OK, 'record evaluation model initialises')
      call fbc_state_init(state, model, status)
      call check(status == FBC_OK, 'record evaluation state initialises')
      call fbc_workspace_init(workspace, model, status)
      call check(status == FBC_OK, 'record evaluation workspace initialises')
      call fbc_evaluate_lf_records(model, state, workspace, records, 1_int32, metrics, status)
      call check(status == FBT_OK .and. metrics%records == 3_int64 .and. metrics%scored_records == 2_int64 .and. &
                 metrics%predictions == 3_int64 .and. metrics%prefix_predictions == 2_int64, &
                 'record evaluation excludes first bytes, separators and one-byte records')
      call fbc_state_reset(state, model, status)
      call fbc_evaluate_stream(model, state, records(1:3), 2_int32, [.false., .false.], workspace, &
                               first_nll, accuracy, answer_count, status)
      call check(status == FBT_OK, 'first record independently scores')
      call fbc_state_reset(state, model, status)
      call fbc_evaluate_stream(model, state, records(1:2), 1_int32, [.false.], workspace, &
                               first_prefix, accuracy, answer_count, status)
      call check(status == FBT_OK, 'first record prefix independently scores')
      call fbc_state_reset(state, model, status)
      call fbc_evaluate_stream(model, state, records(5:6), 1_int32, [.false.], workspace, &
                               second_nll, accuracy, answer_count, status)
      call check(status == FBT_OK, 'second record independently scores')
      call check(abs(metrics%mean_nll - (2.0_real64 * first_nll + second_nll) / 3.0_real64) < 1.0e-12_real64, &
                 'cold record NLL agrees with separate reset stream scores')
      call check(abs(metrics%prefix_mean_nll - (first_prefix + second_nll) / 2.0_real64) < 1.0e-12_real64, &
                 'short-prefix NLL agrees with separate reset stream scores')
      old_steps = state%steps
      call fbc_evaluate_lf_records(model, state, workspace, records, 0_int32, metrics, status)
      call check(status == FBT_ERR_ARGUMENT .and. metrics%predictions == 0_int64 .and. state%steps == old_steps, &
                 'record evaluator rejects a zero prefix before advancing state')
      call fbc_evaluate_lf_records(model, state, workspace, [1_int32, 10_int32, 2_int32], 1_int32, metrics, status)
      call check(status == FBT_ERR_ARGUMENT .and. metrics%predictions == 0_int64, &
                 'record evaluator rejects inputs without within-record transitions')
      call fbc_evaluate_lf_records(model, state, workspace, [1_int32, 33_int32], 1_int32, metrics, status)
      call check(status == FBC_ERR_INVALID_ARGUMENT .and. metrics%predictions == 0_int64, &
                 'record evaluator rejects an invalid target without publishing partial metrics')
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_natural_record_eval

   !> NLL must remain meaningful when the target softmax probability underflows.
   subroutine test_stable_loss
      real(real64) :: logits(3), dlogits(3), loss, reference
      integer(int32) :: status

      logits = [1000.0_real64, 0.0_real64, -1000.0_real64]
      call fbc_loss_and_gradient(logits, 3_int32, dlogits, loss, status)
      call check(status == FBC_OK .and. abs(loss - 2000.0_real64) < 1.0e-12_real64, &
                 'cross entropy does not cap an underflowing target probability')
      call check(abs(dlogits(1) - 1.0_real64) < 1.0e-12_real64 .and. &
                 abs(dlogits(3) + 1.0_real64) < 1.0e-12_real64, &
                 'extreme-logit gradient remains the exact softmax residual')
      logits = logits - 1200.0_real64
      call fbc_loss_and_gradient(logits, 3_int32, dlogits, loss, status)
      call check(status == FBC_OK .and. abs(loss - 2000.0_real64) < 1.0e-12_real64, &
                 'cross entropy is invariant to a common logit shift')
      logits = [3.0_real64, 1.0_real64, -1.0_real64]
      reference = 2.0_real64 + log(1.0_real64 + exp(-2.0_real64) + exp(-4.0_real64))
      call fbc_loss_and_gradient(logits, 2_int32, dlogits, loss, status)
      call check(status == FBC_OK .and. abs(loss - reference) < 1.0e-12_real64 .and. &
                 abs(sum(dlogits)) < 1.0e-12_real64, 'normal cross entropy and gradient agree with reference')
      block
         real(real64) :: strided_logits(6), strided_gradient(6), strided_loss
         strided_logits = -42.0_real64
         strided_logits(1:5:2) = logits
         strided_gradient = 42.0_real64
         call fbc_loss_and_gradient(strided_logits(1:5:2), 2_int32, strided_gradient(1:5:2), strided_loss, status)
         call check(status == FBC_OK .and. abs(strided_loss - loss) < 1.0e-14_real64 .and. &
                    all(abs(strided_gradient(1:5:2) - dlogits) < 1.0e-14_real64) .and. &
                    all(strided_gradient(2:6:2) == 42.0_real64), &
                    'public cross entropy accepts strided views without changing intervening elements')
      end block
      call fbc_loss_and_gradient(logits, 2_int32, dlogits(:2), loss, status)
      call check(status == FBC_ERR_INVALID_ARGUMENT, 'cross entropy rejects mismatched gradient shape')

      logits = [huge(0.0_real64), -huge(0.0_real64), 0.0_real64]
      call fbc_loss_and_gradient(logits, 1_int32, dlogits, loss, status)
      call check(status == FBC_OK .and. loss == 0.0_real64 .and. all(dlogits == 0.0_real64), &
                 'opposite finite extremes yield zero loss for the maximum without overflow')
      call fbc_loss_and_gradient(logits, 3_int32, dlogits, loss, status)
      call check(status == FBC_OK .and. loss == huge(0.0_real64) .and. &
                 all(dlogits == [1.0_real64, 0.0_real64, -1.0_real64]), &
                 'maximum representable target gap and its gradient remain finite')
      call fbc_loss_and_gradient(logits, 2_int32, dlogits, loss, status)
      call check(status == FBC_ERR_NUMERIC .and. all(dlogits == 0.0_real64), &
                 'unrepresentable target NLL is rejected before overflowing')
      logits = [0.5_real64 * huge(0.0_real64), -0.5_real64 * huge(0.0_real64), 0.0_real64]
      call fbc_loss_and_gradient(logits, 2_int32, dlogits, loss, status)
      call check(status == FBC_OK .and. loss == huge(0.0_real64), &
                 'a representable extreme gap at the real64 boundary is accepted')
      logits = -huge(0.0_real64)
      call fbc_loss_and_gradient(logits, 2_int32, dlogits, loss, status)
      call check(status == FBC_OK .and. abs(loss - log(3.0_real64)) < 1.0e-12_real64 .and. &
                 abs(sum(dlogits)) < 1.0e-12_real64, 'equal negative extremes produce the correct uniform loss')
      logits = [1.0_real64, transfer(int(z'7FF8000000000000', int64), 0.0_real64), 0.0_real64]
      call fbc_loss_and_gradient(logits, 1_int32, dlogits, loss, status)
      call check(status == FBC_ERR_NUMERIC, 'quiet NaN logit returns a numeric error')
      logits(2) = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
      call fbc_loss_and_gradient(logits, 1_int32, dlogits, loss, status)
      call check(status == FBC_ERR_NUMERIC, 'signaling NaN logit returns a numeric error without an FPE')
      logits(2) = transfer(int(z'7FF0000000000000', int64), 0.0_real64)
      call fbc_loss_and_gradient(logits, 1_int32, dlogits, loss, status)
      call check(status == FBC_ERR_NUMERIC, 'infinite logit returns a numeric error')
   end subroutine test_stable_loss

   subroutine test_numeric_boundaries
      type(fbc_config_t) :: config
      type(fbc_plan_t) :: plan
      type(fbc_model_t) :: model
      type(fbc_gradient_t) :: gradient, uninitialized_gradient
      type(fbc_optimizer_t) :: optimizer
      integer(int32) :: status
      integer(int32) :: tokens(20)
      logical :: mask(20)
      real(real64) :: norm

      config = tiny_config(8192_int64)
      config%width = FBC_MAX_WIDTH
      config%heads = 1_int32
      config%ffn_width = huge(0_int32)
      config%vocabulary_size = huge(0_int32)
      config%local_window = FBC_MAX_LOCAL_WINDOW
      config%memory_scales = FBC_MAX_MEMORY_SCALES
      config%memory_slots = FBC_MAX_MEMORY_SLOTS
      call fbc_plan(config, plan)
      call check(plan%parameter_count == 146366990473236533_int64 .and. &
                 plan%parameter_bytes_real64 == 1170935923785892264_int64 .and. &
                 plan%state_bytes_real64 == 36030173555990528_int64, &
                 'context resource counts remain exact at the largest supported width')
      call check(plan%forward_flops_per_token > 0_int64 .and. plan%backward_flops_per_token > 0_int64, &
                 'maximum-width forward and backward cost estimates stay positive')
      config%width = FBC_MAX_WIDTH + 1_int32
      call fbc_plan(config, plan)
      call check(plan%parameter_count == 0_int64 .and. fbc_parameter_count(config) == -1_int64, &
                 'context plan rejects oversized dimensions rather than wrapping counts')
      config = tiny_config(8192_int64)
      config%heads = 0_int32
      call fbc_plan(config, plan)
      call fbc_optimizer_init(optimizer, config, status)
      call check(status == FBC_ERR_INVALID_ARGUMENT .and. plan%parameter_count == 0_int64 .and. &
                 fbc_parameter_count(config) == -1_int64 .and. fbc_state_bytes(config) == -1_int64, &
                 'zero heads is rejected before optimizer/plan division')
      config = tiny_config(8192_int64)
      config%memory_scales = huge(0_int32)
      call fbc_plan(config, plan)
      call check(plan%parameter_count == 0_int64, 'plan validates scale count before allocating temporary storage')
      config = tiny_config(8192_int64)
      config%rms_epsilon = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
      call check(.not. fbc_config_is_valid(config), 'signaling NaN configuration is rejected without an FPE')
      config = tiny_config(8192_int64)
      config%slot_temperature = transfer(int(z'7FF0000000000000', int64), 0.0_real64)
      call check(.not. fbc_config_is_valid(config), 'infinite slot temperature is rejected')
      config = tiny_config(8192_int64)
      config%slot_decay = transfer(int(z'7FF8000000000000', int64), 0.0_real64)
      call check(.not. fbc_config_is_valid(config), 'NaN slot decay is rejected')

      config = tiny_config(8192_int64)
      call fbc_model_init(model, config, 29_int64, status)
      call fbc_gradient_init(gradient, model, status)
      call check(status == FBC_OK, 'gradient norm boundary fixture initializes')
      call fbc_gradient_zero(gradient)
      gradient%token_embedding(1, 1) = 3.0_real64
      gradient%token_embedding(2, 1) = 4.0_real64
      norm = fbc_gradient_norm(gradient)
      call check(norm == 5.0_real64, 'ordinary gradient norm is the Euclidean norm')
      call fbc_gradient_zero(gradient)
      gradient%token_embedding(1:2, 1) = 1.0e200_real64
      norm = fbc_gradient_norm(gradient)
      call check(abs(norm / 1.0e200_real64 - sqrt(2.0_real64)) < 1.0e-14_real64, &
                 'large representable gradient norm avoids squaring overflow')
      call fbc_gradient_zero(gradient)
      gradient%token_embedding(1:2, 1) = 1.0e-200_real64
      norm = fbc_gradient_norm(gradient)
      call check(abs(norm / 1.0e-200_real64 - sqrt(2.0_real64)) < 1.0e-14_real64, &
                 'small representable gradient norm avoids squaring underflow')
      call fbc_gradient_zero(gradient)
      gradient%token_embedding(1:3, 1) = [3.0_real64, 4.0_real64, 1.0e-200_real64]
      norm = fbc_gradient_norm(gradient)
      call check(norm == 5.0_real64, 'ordinary and tiny values in one block are not counted twice')
      call fbc_gradient_zero(gradient)
      gradient%token_embedding(1, 1) = -3.0e200_real64
      gradient%output_bias(1) = 4.0e200_real64
      norm = fbc_gradient_norm(gradient)
      call check(abs(norm / 1.0e200_real64 - 5.0_real64) < 1.0e-14_real64, &
                 'scaled norm correctly combines large values across different arrays')
      call fbc_gradient_zero(gradient)
      gradient%token_embedding(1:2, 1) = huge(0.0_real64)
      norm = fbc_gradient_norm(gradient)
      call check(norm < 0.0_real64, 'unrepresentable gradient norm returns an error sentinel without overflow')
      call fbc_gradient_zero(gradient)
      gradient%token_embedding(1, 1) = huge(0.0_real64)
      norm = fbc_gradient_norm(gradient)
      call check(norm == huge(0.0_real64), 'one maximum finite component has a representable norm')
      call fbc_gradient_zero(gradient)
      gradient%token_embedding(1, 1) = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
      norm = fbc_gradient_norm(gradient)
      call check(norm < 0.0_real64, 'signaling NaN gradient returns an error sentinel without an FPE')
      norm = fbc_gradient_norm(uninitialized_gradient)
      call check(norm < 0.0_real64, 'uninitialized gradient cannot silently report zero norm')
      call fbc_gradient_destroy(gradient)
      call fbc_model_destroy(model)

      call fbc_generate_tokens(FBC_TASK_NEEDLE, 20_int32, 16_int32, 3_int32, 9_int64, tokens, mask, status)
      call check(status == FBC_OK .and. all(tokens == [1, 4, 10, 15, 10, 2, 4, 1, 6, 14, &
                                                      13, 9, 2, 6, 1, 4, 11, 14, 10, 2]), &
                 'bounded needle generation preserves the established stream')
      call check(count(mask) == 2 .and. mask(6) .and. mask(13), 'needle answer positions are unchanged')
      call fbc_generate_tokens(FBC_TASK_NEEDLE, 20_int32, 16_int32, huge(0_int32), 9_int64, tokens, mask, status)
      call check(status == FBC_OK .and. tokens(1) == 1_int32 .and. .not. any(mask) .and. &
                 all(tokens >= 1_int32) .and. all(tokens <= 16_int32), &
                 'maximum distance terminates after the available tokens without integer overflow')
   end subroutine test_numeric_boundaries

   ! Independent numerical oracle for both agreement signs, across the matrix,
   ! vector and scalar update paths. This locks the historical restart rule.
   subroutine test_optimizer_rule
      type(fbc_config_t) :: config
      type(fbc_update_config_t) :: update_config
      type(fbc_model_t) :: model
      type(fbc_gradient_t) :: gradient
      type(fbc_optimizer_t) :: optimizer
      integer(int32) :: status, sign_value
      integer(int64) :: updates
      real(real64) :: gamma, expected_energy, expected_velocity, expected_parameter, saved_parameter

      config = tiny_config(8192_int64)
      call fbc_model_init(model, config, 33_int64, status)
      call fbc_gradient_init(gradient, model, status)
      call fbc_optimizer_init(optimizer, config, status)
      call check(status == FBC_OK, 'optimizer oracle fixture initializes')
      update_config%learning_rate = 0.02_real64
      update_config%momentum = 0.8_real64
      update_config%energy_decay = 0.6_real64
      update_config%restart_damp = 0.25_real64
      update_config%weight_decay = 0.01_real64
      update_config%epsilon = 0.1_real64
      do sign_value = -1_int32, 1_int32, 2_int32
         call fbc_gradient_zero(gradient)
         call fbc_optimizer_reset(optimizer)
         gradient%token_embedding(:, 1) = 2.0_real64
         gradient%output_bias = 2.0_real64
         gradient%slot_salience_weight = 2.0_real64
         optimizer%velocity_token_embedding(:, 1) = real(sign_value, real64)
         optimizer%velocity_output_bias = real(sign_value, real64)
         optimizer%slot_salience_weight_velocity = real(sign_value, real64)
         optimizer%energy_token_embedding(1) = 0.25_real64
         optimizer%energy_output_bias = 0.25_real64
         optimizer%slot_salience_weight_energy = 0.25_real64
         model%token_embedding(:, 1) = 0.5_real64
         model%output_bias = 0.5_real64
         model%slot_salience_weight = 0.5_real64
         gamma = 0.8_real64
         if (sign_value < 0_int32) gamma = 0.8_real64 * 0.25_real64
         expected_energy = 0.6_real64 * 0.25_real64 + 0.4_real64 * 4.0_real64
         expected_velocity = gamma * real(sign_value, real64) + 0.2_real64 * 2.0_real64
         expected_parameter = (1.0_real64 - 0.02_real64 * 0.01_real64) * 0.5_real64 - &
                              0.02_real64 * expected_velocity / (sqrt(expected_energy) + 0.1_real64)
         call fbc_optimizer_apply(optimizer, model, gradient, update_config, status, updates)
         call check(status == FBC_OK .and. updates == fbc_parameter_count(config), &
                    'optimizer oracle updates every parameter exactly once')
         call check(maxval(abs(model%token_embedding(:, 1) - expected_parameter)) < 1.0e-14_real64 .and. &
                    abs(optimizer%energy_token_embedding(1) - expected_energy) < 1.0e-14_real64 .and. &
                    maxval(abs(optimizer%velocity_token_embedding(:, 1) - expected_velocity)) < 1.0e-14_real64, &
                    'matrix optimizer agrees with independent positive/negative agreement oracle')
         call check(maxval(abs(model%output_bias - expected_parameter)) < 1.0e-14_real64 .and. &
                    abs(optimizer%energy_output_bias - expected_energy) < 1.0e-14_real64 .and. &
                    maxval(abs(optimizer%velocity_output_bias - expected_velocity)) < 1.0e-14_real64, &
                    'vector optimizer agrees with independent positive/negative agreement oracle')
         call check(abs(model%slot_salience_weight - expected_parameter) < 1.0e-14_real64 .and. &
                    abs(optimizer%slot_salience_weight_energy - expected_energy) < 1.0e-14_real64 .and. &
                    abs(optimizer%slot_salience_weight_velocity - expected_velocity) < 1.0e-14_real64, &
                    'scalar optimizer agrees with independent positive/negative agreement oracle')
      end do
      saved_parameter = model%token_embedding(1, 1)
      update_config%momentum = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
      call fbc_optimizer_apply(optimizer, model, gradient, update_config, status, updates)
      call check(status == FBC_ERR_INVALID_ARGUMENT .and. updates == 0_int64 .and. &
                 model%token_embedding(1, 1) == saved_parameter, &
                 'nonfinite optimizer settings fail before changing a parameter')
      update_config = fbc_update_config_t()
      call fbc_gradient_zero(gradient)
      gradient%token_embedding(1, 1) = 1.0e200_real64
      call fbc_optimizer_apply(optimizer, model, gradient, update_config, status, updates)
      call check(status == FBC_ERR_NUMERIC .and. updates == 0_int64 .and. &
                 model%token_embedding(1, 1) == saved_parameter, &
                 'direct optimizer rejects a finite gradient whose square would overflow')
      gradient%token_embedding(1, 1) = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
      call fbc_optimizer_apply(optimizer, model, gradient, update_config, status, updates)
      call check(status == FBC_ERR_NUMERIC .and. updates == 0_int64 .and. &
                 model%token_embedding(1, 1) == saved_parameter, &
                 'direct optimizer rejects a signaling NaN gradient before any update')
      update_config = fbc_update_config_t()
      update_config%weight_decay = 1000.0_real64
      call check(.not. fbc_update_config_is_valid(update_config), 'lr times weight decay must stay below one')
      update_config = fbc_update_config_t()
      update_config%learning_rate = huge(0.0_real64)
      update_config%epsilon = tiny(0.0_real64)
      call check(.not. fbc_update_config_is_valid(update_config), 'overflowing lr divided by epsilon is rejected')
      config%heads = 0_int32
      call fbc_optimizer_init(optimizer, config, status)
      call check(status == FBC_ERR_INVALID_ARGUMENT .and. optimizer%initialized .and. &
                 abs(optimizer%velocity_token_embedding(1, 1) - expected_velocity) < 1.0e-14_real64, &
                 'invalid reinitialization preserves the existing optimizer state')
      call fbc_optimizer_destroy(optimizer)
      call fbc_gradient_destroy(gradient)
      call fbc_model_destroy(model)
   end subroutine test_optimizer_rule

   !> A non-divisible budget must not silently add work; explicit opt-in must
   !> match an explicit full + short update.
   subroutine test_partial_training_chunk
      type(fbc_config_t) :: config
      type(fbc_update_config_t) :: update_config
      type(fbc_model_t) :: combined_model, split_model
      type(fbc_state_t) :: combined_state, split_state
      type(fbc_workspace_t) :: combined_workspace, split_workspace
      type(fbc_gradient_t) :: combined_gradient, split_gradient
      type(fbc_optimizer_t) :: combined_optimizer, split_optimizer
      type(fbc_metrics_t) :: combined_metrics, split_metrics
      integer(int32), parameter :: tokens(6) = [4_int32, 5_int32, 6_int32, 7_int32, 8_int32, 9_int32]
      integer(int32) :: status

      config = tiny_config(512_int64)
      call fbc_model_init(combined_model, config, 919_int64, status)
      call check(status == FBC_OK, 'partial-chunk model initializes')
      call fbc_model_init(split_model, config, 919_int64, status)
      call fbc_state_init(combined_state, combined_model, status)
      call fbc_state_init(split_state, split_model, status)
      call fbc_workspace_init(combined_workspace, combined_model, status)
      call fbc_workspace_init(split_workspace, split_model, status)
      call fbc_gradient_init(combined_gradient, combined_model, status)
      call fbc_gradient_init(split_gradient, split_model, status)
      call fbc_optimizer_init(combined_optimizer, config, status)
      call fbc_optimizer_init(split_optimizer, config, status)
      call check(status == FBC_OK, 'partial-chunk optimizer and workspaces initialize')
      update_config%momentum = 1.0_real64
      call fbc_train_stream(combined_model, combined_optimizer, combined_state, tokens, 4_int32, 4_int32, &
                            update_config, combined_workspace, combined_gradient, combined_metrics, status)
      call check(status == FBT_ERR_ARGUMENT .and. combined_state%steps == 0_int64 .and. &
                 combined_metrics%updates == 0_int64, 'invalid optimizer settings are rejected before training')
      update_config = fbc_update_config_t()
      call fbc_train_stream(combined_model, combined_optimizer, combined_state, tokens, 5_int32, 4_int32, &
                            update_config, combined_workspace, combined_gradient, combined_metrics, status)
      call check(status == FBT_ERR_ARGUMENT .and. combined_metrics%tokens == 0_int64 .and. &
                 combined_metrics%updates == 0_int64 .and. combined_state%steps == 0_int64, &
                 'non-divisible budget is rejected before any training work by default')
      call check(all(combined_model%token_embedding == split_model%token_embedding) .and. &
                 all(combined_optimizer%velocity_token_embedding == split_optimizer%velocity_token_embedding), &
                 'rejected partial chunk leaves model and optimizer unchanged')
      call fbc_train_stream(combined_model, combined_optimizer, combined_state, tokens, 5_int32, 4_int32, &
                            update_config, combined_workspace, combined_gradient, combined_metrics, status, &
                            allow_partial_chunk=.true.)
      call check(status == FBT_OK .and. combined_metrics%tokens == 5_int64 .and. &
                 combined_metrics%updates == 2_int64 .and. combined_state%steps == 5_int64, &
                 'partial final chunk consumes all five predictions in two updates')
      call fbc_train_stream(split_model, split_optimizer, split_state, tokens(:5), 4_int32, 4_int32, &
                            update_config, split_workspace, split_gradient, split_metrics, status)
      call fbc_train_stream(split_model, split_optimizer, split_state, tokens(5:), 1_int32, 1_int32, &
                            update_config, split_workspace, split_gradient, split_metrics, status)
      call check(status == FBT_OK .and. split_metrics%tokens == 5_int64 .and. split_metrics%updates == 2_int64, &
                 'explicit short-chunk reference consumes the same budget')
      call check(all(combined_model%token_embedding == split_model%token_embedding) .and. &
                 all(combined_model%output_bias == split_model%output_bias) .and. &
                 all(combined_model%query == split_model%query), &
                 'partial chunk uses the same mean gradient as an explicit short update')
      call check(all(combined_optimizer%velocity_token_embedding == split_optimizer%velocity_token_embedding) .and. &
                 all(combined_optimizer%energy_token_embedding == split_optimizer%energy_token_embedding), &
                 'partial chunk preserves optimizer state equivalence')
      call fbc_train_stream(combined_model, combined_optimizer, combined_state, tokens, 0_int32, 4_int32, &
                            update_config, combined_workspace, combined_gradient, combined_metrics, status)
      call check(status == FBT_OK .and. combined_metrics%tokens == 5_int64, 'zero-token budget is a no-op')
      call fbc_train_stream(combined_model, combined_optimizer, combined_state, tokens, -1_int32, 4_int32, &
                            update_config, combined_workspace, combined_gradient, combined_metrics, status)
      call check(status == FBT_ERR_ARGUMENT, 'negative token budget is rejected')
      call fbc_gradient_destroy(combined_gradient)
      call fbc_gradient_destroy(split_gradient)
      call fbc_optimizer_destroy(combined_optimizer)
      call fbc_optimizer_destroy(split_optimizer)
      call fbc_workspace_destroy(combined_workspace)
      call fbc_workspace_destroy(split_workspace)
      call fbc_state_destroy(combined_state)
      call fbc_state_destroy(split_state)
      call fbc_model_destroy(combined_model)
      call fbc_model_destroy(split_model)
   end subroutine test_partial_training_chunk

   ! Invalid lengths and unrepresentable counters/loss sums must not become
   ! successful but wrong reports or undefined signed/float arithmetic.
   subroutine test_stream_error_boundaries
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: gradient
      type(fbc_optimizer_t) :: optimizer
      type(fbc_update_config_t) :: update_config
      type(fbc_metrics_t) :: metrics
      integer(int32) :: status, tokens(3)
      integer(int64) :: answer_count
      real(real64) :: mean_nll, answer_accuracy, old_bias(16)
      logical :: mask(2)

      config = tiny_config(8192_int64)
      call fbc_model_init(model, config, 47_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      call fbc_gradient_init(gradient, model, status)
      call fbc_optimizer_init(optimizer, config, status)
      tokens = [2_int32, 2_int32, 2_int32]
      mask = .false.
      call fbc_evaluate_stream(model, state, tokens, -1_int32, mask, workspace, &
                               mean_nll, answer_accuracy, answer_count, status)
      call check(status == FBT_ERR_ARGUMENT .and. state%steps == 0_int64, &
                 'evaluation rejects a negative budget before state changes')
      call fbc_evaluate_stream(model, state, tokens, huge(0_int32), mask, workspace, &
                               mean_nll, answer_accuracy, answer_count, status)
      call check(status == FBT_ERR_ARGUMENT .and. state%steps == 0_int64, &
                 'evaluation rejects an overflowing budget before state changes')
      call fbc_evaluate_stream(model, state, tokens, 2_int32, mask(:1), workspace, &
                               mean_nll, answer_accuracy, answer_count, status)
      call check(status == FBT_ERR_ARGUMENT .and. state%steps == 0_int64, &
                 'evaluation rejects an undersized answer mask before state changes')
      call fbc_evaluate_stream(model, state, tokens, 0_int32, mask(:0), workspace, &
                               mean_nll, answer_accuracy, answer_count, status)
      call check(status == FBT_OK .and. mean_nll == 0.0_real64 .and. answer_count == 0_int64 .and. &
                 answer_accuracy == 0.0_real64 .and. state%steps == 0_int64, 'zero-token evaluation is a no-op')

      metrics = fbc_metrics_t()
      metrics%tokens = huge(0_int64)
      call fbc_train_stream(model, optimizer, state, tokens, 2_int32, 2_int32, update_config, &
                            workspace, gradient, metrics, status)
      call check(status == FBT_ERR_ARGUMENT .and. state%steps == 0_int64, 'token counter overflow is rejected early')
      metrics = fbc_metrics_t()
      metrics%updates = huge(0_int64)
      call fbc_train_stream(model, optimizer, state, tokens, 2_int32, 2_int32, update_config, &
                            workspace, gradient, metrics, status)
      call check(status == FBT_ERR_ARGUMENT .and. state%steps == 0_int64, 'update counter overflow is rejected early')
      metrics = fbc_metrics_t()
      metrics%parameter_updates = huge(0_int64)
      call fbc_train_stream(model, optimizer, state, tokens, 2_int32, 2_int32, update_config, &
                            workspace, gradient, metrics, status)
      call check(status == FBT_ERR_ARGUMENT .and. state%steps == 0_int64, &
                 'parameter-update counter overflow is rejected early')
      metrics = fbc_metrics_t()
      metrics%loss_sum = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
      call fbc_train_stream(model, optimizer, state, tokens, 2_int32, 2_int32, update_config, &
                            workspace, gradient, metrics, status)
      call check(status == FBT_ERR_NUMERIC .and. .not. metrics%numeric_ok .and. state%steps == 0_int64, &
                 'a signaling NaN in the incoming loss sum fails before work')

      model%output_bias = 0.0_real64
      model%output_bias(1) = 0.75_real64 * huge(0.0_real64)
      old_bias = model%output_bias
      call fbc_evaluate_stream(model, state, tokens, 2_int32, mask, workspace, &
                               mean_nll, answer_accuracy, answer_count, status)
      call check(status == FBT_ERR_NUMERIC .and. state%steps == 2_int64, &
                 'evaluation rejects an unrepresentable loss sum without floating overflow')
      call fbc_state_reset(state, model, status)
      metrics = fbc_metrics_t()
      call fbc_train_stream(model, optimizer, state, tokens, 2_int32, 2_int32, update_config, &
                            workspace, gradient, metrics, status)
      call check(status == FBT_ERR_NUMERIC .and. .not. metrics%numeric_ok .and. metrics%tokens == 1_int64 .and. &
                 metrics%updates == 0_int64 .and. state%steps == 2_int64 .and. all(model%output_bias == old_bias), &
                 'overflowing training loss fails before an optimizer update and reports the advanced state')
      call fbc_gradient_destroy(gradient)
      call fbc_optimizer_destroy(optimizer)
      call fbc_workspace_destroy(workspace)
      call fbc_state_destroy(state)
      call fbc_model_destroy(model)
   end subroutine test_stream_error_boundaries

   subroutine test_workspace_gradient_binding
      type(fbc_config_t) :: config, other_config
      type(fbc_model_t) :: model, other_model
      type(fbc_state_t) :: state, missing_state
      type(fbc_workspace_t) :: workspace, other_workspace, stale_workspace
      type(fbc_gradient_t) :: gradient, other_gradient
      type(fbc_optimizer_t) :: optimizer, other_optimizer
      type(fbc_update_config_t) :: update_config
      type(fbc_metrics_t) :: metrics
      integer(int32) :: status, tokens(2), dimension
      integer(int64) :: parameter_updates, old_steps
      integer :: unit, io_status
      real(real64) :: hidden(8), logits(16), dlogits(16), old_embedding(8, 16)
      character(len=11) :: sentinel
      character(len=*), parameter :: sentinel_path = 'fbc_layout_guard_sentinel.vgfb'

      config = tiny_config(8192_int64)
      other_config = config
      other_config%ffn_width = 24_int32
      call fbc_model_init(model, config, 97_int64, status)
      call fbc_model_init(other_model, other_config, 97_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      call fbc_workspace_init(other_workspace, other_model, status)
      call fbc_gradient_init(gradient, model, status)
      call fbc_gradient_init(other_gradient, other_model, status)
      call fbc_optimizer_init(optimizer, config, status)
      call fbc_optimizer_init(other_optimizer, other_config, status)
      tokens = [2_int32, 3_int32]
      dlogits = 0.0_real64
      dlogits(2) = -0.5_real64
      dlogits(3) = 0.5_real64

      do dimension = 1_int32, 7_int32
         workspace%layout(dimension) = -1_int32
         call fbc_step(model, state, workspace, 2_int32, hidden, logits, status)
         call check(status == FBC_ERR_NOT_READY .and. state%steps == 0_int64, &
                    'every workspace shape component is checked before the forward step')
         workspace%layout = fbc_config_layout(config)
      end do
      call fbc_step(model, state, other_workspace, 2_int32, hidden, logits, status)
      call check(status == FBC_ERR_NOT_READY .and. state%steps == 0_int64, &
                 'a workspace initialized for a different model is rejected before state changes')
      call fbc_step_backward(model, missing_state, workspace, dlogits, gradient, status)
      call check(status == FBC_ERR_NOT_READY, 'backward rejects an uninitialized state')
      call fbc_step_backward(model, state, workspace, dlogits, gradient, status)
      call check(status == FBC_ERR_ORDER .and. all(gradient%token_embedding == 0.0_real64), &
                 'backward rejects a workspace that has no completed forward')

      call fbc_step(model, state, workspace, 2_int32, hidden, logits, status, defer_local_commit=.true.)
      call check(status == FBC_OK .and. workspace%forward_step == 1_int64, 'a deferred forward publishes its step stamp')
      call fbc_step_backward(model, state, workspace, dlogits, other_gradient, status)
      call check(status == FBC_ERR_NOT_READY .and. all(other_gradient%token_embedding == 0.0_real64), &
                 'backward rejects a gradient initialized for another shape before writing')
      call fbc_finish_local_step(state, other_workspace, status)
      call check(status == FBC_ERR_ORDER .and. state%steps == 0_int64, &
                 'a workspace without the current forward cannot finish a deferred state')
      call fbc_step_backward(model, state, workspace, dlogits, gradient, status)
      call check(status == FBC_OK, 'matched deferred forward/backward succeeds')
      call fbc_finish_local_step(state, workspace, status)
      call check(status == FBC_OK .and. state%steps == 1_int64, 'matched deferred state commits')
      stale_workspace = workspace
      call fbc_step(model, state, workspace, 3_int32, hidden, logits, status)
      call check(status == FBC_OK .and. state%steps == 2_int64, 'next forward advances the stamp')
      call fbc_step_backward(model, state, stale_workspace, dlogits, gradient, status)
      call check(status == FBC_ERR_ORDER, 'backward rejects activations from an older step')

      old_steps = state%steps
      old_embedding = model%token_embedding
      metrics = fbc_metrics_t()
      call fbc_train_stream(model, other_optimizer, state, tokens, 1_int32, 1_int32, update_config, &
                            workspace, gradient, metrics, status)
      call check(status == FBC_ERR_NOT_READY .and. state%steps == old_steps .and. metrics%tokens == 0_int64 .and. &
                 all(model%token_embedding == old_embedding), 'mismatched optimizer is rejected before training')
      call fbc_train_stream(model, optimizer, state, tokens, 1_int32, 1_int32, update_config, &
                            workspace, other_gradient, metrics, status)
      call check(status == FBC_ERR_NOT_READY .and. state%steps == old_steps .and. metrics%tokens == 0_int64, &
                 'mismatched gradient is rejected before training')
      call fbc_optimizer_apply(other_optimizer, model, gradient, update_config, status, parameter_updates)
      call check(status == FBC_ERR_NOT_READY .and. parameter_updates == 0_int64 .and. &
                 all(model%token_embedding == old_embedding), 'optimizer rejects a different model shape before writing')

      open(newunit=unit, file=sentinel_path, status='replace', access='stream', form='unformatted', iostat=io_status)
      if (io_status == 0) then
         write(unit, iostat=io_status) 'layout-safe'
         close(unit)
      end if
      call check(io_status == 0, 'layout checkpoint sentinel created')
      call fbc_checkpoint_save(sentinel_path, model, other_optimizer, update_config, metrics, status)
      call check(status == FBT_ERR_ARGUMENT, 'checkpoint rejects a mismatched optimizer shape')
      sentinel = ''
      open(newunit=unit, file=sentinel_path, status='old', access='stream', form='unformatted', iostat=io_status)
      if (io_status == 0) then
         read(unit, iostat=io_status) sentinel
         close(unit, status='delete')
      end if
      call check(io_status == 0 .and. sentinel == 'layout-safe', &
                 'mismatched optimizer cannot replace an existing checkpoint')

      call fbc_state_reset(state, model, status)
      state%steps = huge(0_int64)
      call fbc_step(model, state, workspace, 2_int32, hidden, logits, status)
      call check(status == FBC_ERR_NUMERIC .and. state%steps == huge(0_int64), &
                 'state step counter cannot overflow on a forward pass')
      call fbc_workspace_destroy(stale_workspace)
      call fbc_optimizer_destroy(other_optimizer)
      call fbc_optimizer_destroy(optimizer)
      call fbc_gradient_destroy(other_gradient)
      call fbc_gradient_destroy(gradient)
      call fbc_workspace_destroy(other_workspace)
      call fbc_workspace_destroy(workspace)
      call fbc_state_destroy(state)
      call fbc_model_destroy(other_model)
      call fbc_model_destroy(model)
   end subroutine test_workspace_gradient_binding

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

      ! The ladder's requested/effective horizon and the learned decay clamp
      ! are distinct: the former can exceed what FBC_MAX_DECAY permits.
      block
         type(fbc_config_t) :: long_config
         type(fbc_model_t) :: long_model
         integer(int32) :: status
         long_config = tiny_config(2000000000000_int64)
         call fbc_model_init(long_model, long_config, 1_int64, status)
         call check(status == FBC_OK .and. .not. long_model%horizon_clamped .and. &
                    long_model%effective_horizon == 2.0e12_real64 .and. &
                    long_model%scale_decay(long_config%memory_scales) == FBC_MAX_DECAY .and. &
                    fbc_learned_horizon(long_model) < long_model%effective_horizon, &
                    'decay-clamped learned reach is distinguished from the requested ladder horizon')
         call fbc_model_destroy(long_model)
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

   ! V5/V6 checkpoints start every slot at zero and write them with the same
   ! rule. Keep their trajectory unchanged, but do not mistake repeated slots
   ! or floating-point roundoff for independently addressable memory.
   subroutine test_legacy_slot_symmetry
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      integer(int32) :: rule, status, index, slot
      real(real64) :: hidden(8), logits(16)
      logical :: identical, succeeded

      config = tiny_config(1024_int64)
      config%memory_slots = 8_int32
      do rule = FBC_TRAINING_RULE_V5, FBC_TRAINING_RULE_V6
         call fbc_model_init(model, config, 5_int64, status)
         model%training_rule = rule
         call fbc_state_init(state, model, status)
         call fbc_workspace_init(workspace, model, status)
         identical = .true.
         succeeded = .true.
         do index = 1_int32, 512_int32
            call fbc_step(model, state, workspace, 1_int32 + modulo(index * 7_int32, 16_int32), &
                          hidden, logits, status)
            if (status /= FBC_OK) then
               succeeded = .false.
               exit
            end if
            do slot = 2_int32, config%memory_slots
               identical = identical .and. all(state%slot_key(:, slot) == state%slot_key(:, 1_int32)) .and. &
                  all(state%slot_value(:, slot) == state%slot_value(:, 1_int32)) .and. &
                  state%slot_salience(slot) == state%slot_salience(1_int32)
            end do
         end do
         call check(succeeded, 'the legacy slot-symmetry stream completes')
         call check(identical, 'legacy slots remain identical throughout a varying 512-token stream')
         call fbc_state_destroy(state)
         call fbc_workspace_destroy(workspace)
         call fbc_model_destroy(model)
      end do
   end subroutine test_legacy_slot_symmetry

   ! Multiple V7 slots retain their identities over a varied stream; values and
   ! salience must also diverge, not just the fixed key coordinates. With one
   ! slot the rule reduces exactly to V6, including the full key update.
   subroutine test_anchored_slot_diversity
      type(fbc_config_t) :: config
      type(fbc_model_t) :: anchored, single_legacy
      type(fbc_state_t) :: state, single_state
      type(fbc_workspace_t) :: workspace, single_workspace
      integer(int32), parameter :: counts(5) = [1_int32, 2_int32, 3_int32, 8_int32, 64_int32]
      integer(int32) :: count_index, slots, status, index
      real(real64) :: hidden(8), logits(16), other_hidden(8), other_logits(16)
      real(real64), allocatable :: anchors(:)
      logical :: succeeded, stable, single_identical

      do count_index = 1_int32, size(counts, kind=int32)
         config = tiny_config(1024_int64)
         slots = counts(count_index)
         config%memory_slots = slots
         call fbc_model_init(anchored, config, 5_int64, status)
         anchored%training_rule = FBC_TRAINING_RULE_V7
         call fbc_state_init(state, anchored, status)
         call fbc_workspace_init(workspace, anchored, status)
         allocate(anchors(slots))
         anchors = state%slot_key(1_int32, :)
         if (slots > 1_int32) then
            call check(all(anchors(2:) > anchors(:slots - 1_int32)), &
                       'V7 assigns deterministically ordered, distinct slot anchors')
         else
            single_legacy = anchored
            single_legacy%training_rule = FBC_TRAINING_RULE_V6
            call fbc_state_init(single_state, single_legacy, status)
            call fbc_workspace_init(single_workspace, single_legacy, status)
         end if
         succeeded = .true.
         stable = .true.
         single_identical = .true.
         do index = 1_int32, 1500_int32
            call fbc_step(anchored, state, workspace, 1_int32 + modulo(index * 7_int32, 16_int32), &
                          hidden, logits, status)
            if (status /= FBC_OK) then
               succeeded = .false.
               exit
            end if
            if (slots > 1_int32) stable = stable .and. all(state%slot_key(1_int32, :) == anchors)
            if (slots == 1_int32) then
               call fbc_step(single_legacy, single_state, single_workspace, &
                             1_int32 + modulo(index * 7_int32, 16_int32), other_hidden, other_logits, status)
               if (status /= FBC_OK) then
                  succeeded = .false.
                  exit
               end if
               single_identical = single_identical .and. all(hidden == other_hidden) .and. all(logits == other_logits) &
                  .and. all(state%slot_key == single_state%slot_key) .and. all(state%slot_value == single_state%slot_value) &
                  .and. all(state%slot_salience == single_state%slot_salience)
            end if
         end do
         call check(succeeded, 'the anchored-slot stream completes')
         if (slots > 1_int32) then
            call check(stable, 'V7 anchors are bit-stable throughout a varying 1500-token stream')
            call check(maxval(abs(state%slot_value(:, 1_int32) - state%slot_value(:, slots))) > 1.0e-6_real64 .and. &
                       abs(state%slot_salience(1_int32) - state%slot_salience(slots)) > 1.0e-6_real64, &
                       'different anchors produce meaningfully different values and salience')
         else
            call check(single_identical, 'one-slot V7 and V6 have identical stream outputs and slot state')
            block
               type(fbc_gradient_t) :: a_gradient, b_gradient
               real(real64) :: a_dlogits(16), b_dlogits(16), a_loss, b_loss
               call fbc_state_reset(state, anchored, status)
               call fbc_state_reset(single_state, single_legacy, status)
               call fbc_gradient_init(a_gradient, anchored, status)
               call fbc_gradient_init(b_gradient, single_legacy, status)
               do index = 1_int32, 4_int32
                  call fbc_step(anchored, state, workspace, index + 1_int32, hidden, logits, status)
                  call fbc_step(single_legacy, single_state, single_workspace, index + 1_int32, &
                                other_hidden, other_logits, status)
               end do
               call fbc_loss_and_gradient(logits, 5_int32, a_dlogits, a_loss, status)
               call fbc_loss_and_gradient(other_logits, 5_int32, b_dlogits, b_loss, status)
               call fbc_step_backward(anchored, state, workspace, a_dlogits, a_gradient, status)
               call fbc_step_backward(single_legacy, single_state, single_workspace, b_dlogits, b_gradient, status)
               call check(status == FBC_OK .and. a_loss == b_loss .and. all(a_dlogits == b_dlogits) .and. &
                          all(a_gradient%slot_key_proj == b_gradient%slot_key_proj) .and. &
                          all(a_gradient%slot_query_proj == b_gradient%slot_query_proj) .and. &
                          all(a_gradient%slot_value_proj == b_gradient%slot_value_proj) .and. &
                          all(a_gradient%token_embedding == b_gradient%token_embedding), &
                          'one-slot V7 and V6 produce identical loss and tested backward tensors')
               call fbc_gradient_destroy(a_gradient)
               call fbc_gradient_destroy(b_gradient)
            end block
            call fbc_state_destroy(single_state)
            call fbc_workspace_destroy(single_workspace)
            call fbc_model_destroy(single_legacy)
         end if
         call fbc_state_reset(state, anchored, status)
         call check(status == FBC_OK .and. all(state%slot_key(1_int32, :) == anchors) .and. &
                    all(state%slot_key(2_int32:, :) == 0.0_real64) .and. &
                    all(state%slot_value == 0.0_real64) .and. all(state%slot_salience == 0.0_real64), &
                    'reset recreates V7 slot identities and clears all dynamic components')
         deallocate(anchors)
         call fbc_state_destroy(state)
         call fbc_workspace_destroy(workspace)
         call fbc_model_destroy(anchored)
      end do
   end subroutine test_anchored_slot_diversity

   ! A live V6 slot state cannot be silently reused with V7 semantics, or the
   ! other way around. Reset is the explicit, allocation-free transition.
   subroutine test_slot_rule_state_binding
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state, before, rejected
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: gradient
      integer(int32) :: status
      real(real64) :: hidden(8), logits(16), dlogits(16), loss

      config = tiny_config(1024_int64)
      call fbc_model_init(model, config, 31_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      call fbc_gradient_init(gradient, model, status)
      call fbc_step(model, state, workspace, 3_int32, hidden, logits, status)
      call fbc_loss_and_gradient(logits, 7_int32, dlogits, loss, status)
      call check(status == FBC_OK .and. .not. state%anchored_slots, 'state binding fixture starts with V6')
      before = state
      model%training_rule = FBC_TRAINING_RULE_V7
      call fbc_step(model, state, workspace, 4_int32, hidden, logits, status)
      call check(status == FBC_ERR_ORDER .and. state%steps == before%steps .and. &
                 all(state%slot_key == before%slot_key) .and. all(state%slot_value == before%slot_value), &
                 'V7 forward rejects a live V6 state without changing it')
      call fbc_step_backward(model, state, workspace, dlogits, gradient, status)
      call check(status == FBC_ERR_ORDER .and. all(gradient%token_embedding == 0.0_real64), &
                 'V7 backward rejects a live V6 state without updating the gradient')
      call fbc_state_reset(state, model, status)
      call check(status == FBC_OK .and. state%anchored_slots .and. state%steps == 0_int64 .and. &
                 state%slot_key(1_int32, 1_int32) < state%slot_key(1_int32, config%memory_slots), &
                 'reset explicitly creates the V7 slot identities')
      call fbc_step(model, state, workspace, 4_int32, hidden, logits, status)
      call fbc_loss_and_gradient(logits, 7_int32, dlogits, loss, status)
      before = state
      model%training_rule = FBC_TRAINING_RULE_V6
      call fbc_step(model, state, workspace, 5_int32, hidden, logits, status)
      call check(status == FBC_ERR_ORDER .and. state%steps == before%steps .and. &
                 all(state%slot_key == before%slot_key), 'V6 forward rejects a live V7 state unchanged')
      call fbc_step_backward(model, state, workspace, dlogits, gradient, status)
      call check(status == FBC_ERR_ORDER .and. all(gradient%token_embedding == 0.0_real64), &
                 'V6 backward rejects a live V7 state unchanged')
      call fbc_state_reset(state, model, status)
      call check(status == FBC_OK .and. .not. state%anchored_slots .and. all(state%slot_key == 0.0_real64), &
                 'reset explicitly returns to V6 slot semantics')
      before = state
      model%training_rule = 99_int32
      call fbc_state_reset(state, model, status)
      call check(status == FBC_ERR_INVALID_ARGUMENT .and. state%steps == before%steps .and. &
                 all(state%slot_key == before%slot_key), 'reset rejects unknown rules without mutating state')
      call fbc_state_init(rejected, model, status)
      call check(status == FBC_ERR_INVALID_ARGUMENT .and. .not. rejected%initialized .and. &
                 .not. allocated(rejected%slot_key), 'initializing an unknown rule leaves no live state')
      call fbc_step(model, state, workspace, 3_int32, hidden, logits, status)
      call check(status == FBC_ERR_INVALID_ARGUMENT .and. state%steps == before%steps, &
                 'forward rejects an unknown rule before mutating state')
      call fbc_state_destroy(rejected)
      call fbc_state_destroy(before)
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_gradient_destroy(gradient)
      call fbc_model_destroy(model)
      block
         type(fbc_model_t) :: legacy
         type(fbc_state_t) :: legacy_state
         type(fbc_workspace_t) :: legacy_workspace
         real(real64) :: legacy_hidden(8), legacy_logits(16)
         config%memory_slots = 1_int32
         call fbc_model_init(model, config, 31_int64, status)
         call fbc_state_init(state, model, status)
         call fbc_workspace_init(workspace, model, status)
         call fbc_step(model, state, workspace, 3_int32, hidden, logits, status)
         legacy = model
         legacy_state = state
         call fbc_workspace_init(legacy_workspace, legacy, status)
         model%training_rule = FBC_TRAINING_RULE_V7
         call fbc_step(model, state, workspace, 4_int32, hidden, logits, status)
         call check(status == FBC_OK .and. .not. state%anchored_slots, &
                    'one-slot V7 may use the equivalent V6 state without a reset')
         call fbc_step(legacy, legacy_state, legacy_workspace, 4_int32, legacy_hidden, legacy_logits, status)
         call check(status == FBC_OK .and. all(hidden == legacy_hidden) .and. all(logits == legacy_logits) .and. &
                    all(state%slot_key == legacy_state%slot_key), 'one-slot rule transition remains V6-equivalent')
         call fbc_workspace_destroy(legacy_workspace)
         call fbc_state_destroy(legacy_state)
         call fbc_model_destroy(legacy)
         call fbc_workspace_destroy(workspace)
         call fbc_state_destroy(state)
         call fbc_model_destroy(model)
      end block
   end subroutine test_slot_rule_state_binding

   !----------------------------------------------------------------------------
   !  7. Finite-difference gradient check of the analytic backward
   !----------------------------------------------------------------------------
   !> The backward pass is the exact derivative of one implemented step taken
   !> from a frozen state (depth-1 truncated BPTT).  So a central difference of
   !> exactly that one-step loss, taken from the same frozen state, must match
   !> the analytic gradient.  Eight parameters are probed, one per block of the
   !> network: readout weight, channel logit, the input-dependent channel and
   !> scale gates, the two slot write projections, the slot query projection,
   !> the distance bias, an FFN projection and the tied embedding.
   subroutine test_gradient_check(rule)
      integer(int32), intent(in) :: rule
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: base, probe
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: gradient
      integer(int32), parameter :: vocabulary = 16_int32, width = 8_int32
      integer(int32) :: status, index
      integer(int32) :: warm(3)
      real(real64) :: hidden(width), logits(vocabulary), dlogits(vocabulary), loss, delta, original
      real(real64) :: plus_loss, minus_loss, numeric, worst_error, error
      real(real64) :: direction_delta, dot, sweep_numeric
      type(fbc_model_t) :: saved
      integer(int32) :: component, column

      config = tiny_config(1024_int64)
      call fbc_model_init(model, config, 5_int64, status)
      model%training_rule = rule
      ! The router starts at exactly zero, which makes the one-step loss linear
      ! in it: a central difference would then agree with any gradient built
      ! from the right channel derivatives alone, and would not test the term
      ! that the router contributes to the *context* derivatives.  Give three
      ! entries a value first so every probe below also exercises that path.
      model%channel_router(FBC_CHANNEL_RETENTION, 2_int32) = 0.30_real64
      model%channel_router(FBC_CHANNEL_SLOTS, &
                           (3_int32 - 1_int32) * FBC_CHANNEL_COUNT + FBC_CHANNEL_RETENTION) = &
         -0.20_real64
      model%channel_router(FBC_CHANNEL_LOCAL, &
                           (5_int32 - 1_int32) * FBC_CHANNEL_COUNT + FBC_CHANNEL_SLOTS) = 0.25_real64
      ! The induction read shares the local attention weights, so it changes the
      ! weight derivative that the q/k/v and distance-bias probes below check.
      ! With a zero weight those probes would pass even if the shifted term were
      ! missing entirely, so the weight is non-zero before any probe runs.
      model%induction_mix(1_int32) = 0.40_real64
      model%induction_mix(2_int32) = -0.25_real64
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
      if (rule == FBC_TRAINING_RULE_V7) then
         call check(sum(abs(gradient%slot_query_proj)) > 1.0e-10_real64 .and. &
                    abs(gradient%slot_salience_weight) > 1.0e-10_real64, &
                    'distinct V7 slots provide nontrivial query and salience gradients')
      end if
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
         ! The input-dependent channel gate: logit(c) = channel_logit(c) +
         ! channel_gate_proj(c, :) . normalized, so a row of the projection and
         ! the shared bias must both verify against the same step.
         original = model%channel_gate_proj(FBC_CHANNEL_RETENTION, 2_int32)
         model%channel_gate_proj(FBC_CHANNEL_RETENTION, 2_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%channel_gate_proj(FBC_CHANNEL_RETENTION, 2_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%channel_gate_proj(FBC_CHANNEL_RETENTION, 2_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%channel_gate_proj(FBC_CHANNEL_RETENTION, 2_int32), &
                    'finite difference matches the input-dependent channel gate gradient')
      end block
      block
         original = model%scale_gate_proj(1_int32, 3_int32)
         model%scale_gate_proj(1_int32, 3_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%scale_gate_proj(1_int32, 3_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%scale_gate_proj(1_int32, 3_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%scale_gate_proj(1_int32, 3_int32), &
                    'finite difference matches the input-dependent scale gate gradient')
      end block
      block
         ! The context router: logit(c) += sum_{s,k} router(c, (k-1)*3+s) context(k, s).
         ! Column 2 is component 1 of the *retention* context feeding the
         ! retention weight, column 19 is component 7 of the *local* context
         ! feeding the slot weight -- one probe per direction that matters, the
         ! "read more/less of this channel" pair.
         original = model%channel_router(FBC_CHANNEL_RETENTION, 2_int32)
         model%channel_router(FBC_CHANNEL_RETENTION, 2_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%channel_router(FBC_CHANNEL_RETENTION, 2_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%channel_router(FBC_CHANNEL_RETENTION, 2_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%channel_router(FBC_CHANNEL_RETENTION, 2_int32), &
                    'finite difference matches the context-router gradient (retention context)')
         call check(abs(gradient%channel_router(FBC_CHANNEL_RETENTION, 2_int32)) > 1.0e-12_real64, &
                    'the probed router entry really carries a gradient (not 0 against 0)')
      end block
      block
         original = model%channel_router(FBC_CHANNEL_SLOTS, 19_int32)
         model%channel_router(FBC_CHANNEL_SLOTS, 19_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%channel_router(FBC_CHANNEL_SLOTS, 19_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%channel_router(FBC_CHANNEL_SLOTS, 19_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%channel_router(FBC_CHANNEL_SLOTS, 19_int32), &
                    'finite difference matches the context-router gradient (local context -> slots)')
         call check(abs(gradient%channel_router(FBC_CHANNEL_SLOTS, 19_int32)) > 1.0e-12_real64, &
                    'the second probed router entry carries a gradient too')
      end block
      block
         ! The induction weight scales the shifted read, so its derivative is the
         ! overlap between that read and everything the local channel sends on.
         original = model%induction_mix(1_int32)
         model%induction_mix(1_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%induction_mix(1_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%induction_mix(1_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%induction_mix(1_int32), &
                    'finite difference matches the induction (shifted read) weight')
         call check(abs(gradient%induction_mix(1_int32)) > 1.0e-12_real64, &
                    'the induction weight carries a gradient, so the probes above are meaningful')
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
      ! V7 freezes the first key component in the state, but its projection
      ! still affects the write gate. Check that exact derivative independently.
      if (rule == FBC_TRAINING_RULE_V7) then
         original = model%slot_key_proj(1_int32, 3_int32)
         model%slot_key_proj(1_int32, 3_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%slot_key_proj(1_int32, 3_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%slot_key_proj(1_int32, 3_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%slot_key_proj(1_int32, 3_int32), &
                              'anchored key component has the correct write-gate gradient')
         call check(abs(gradient%slot_key_proj(1_int32, 3_int32)) > 1.0e-12_real64, &
                    'the anchored gate gradient is not merely numerical zero')
      end if
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
      if (rule == FBC_TRAINING_RULE_V7) then
         original = model%slot_salience_weight
         model%slot_salience_weight = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%slot_salience_weight = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%slot_salience_weight = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%slot_salience_weight, &
                              'anchored salience-selection gradient matches finite difference')
      end if
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
         ! A tied-embedding column that is NOT the token being read.  Only the
         ! output path can reach it, so this is the probe that pins down the
         ! full outer product: an earlier revision wrote only the current
         ! token's column and this derivative was silently zero.
         original = model%token_embedding(3_int32, 12_int32)
         model%token_embedding(3_int32, 12_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%token_embedding(3_int32, 12_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%token_embedding(3_int32, 12_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%token_embedding(3_int32, 12_int32), &
                    'finite difference matches an embedding column only the output path can reach')
         call check(abs(gradient%token_embedding(3_int32, 12_int32)) > 1.0e-14_real64, &
                    'that output-only column carries a real gradient')
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

      ! The whole tied embedding against finite differences, element by
      ! element.  Probing one column is what let the original defect survive
      ! 138 checks: if the column probed happens to be the token being read, the
      ! output path contributes to it through logits(token) as well, so the
      ! truncated version and the correct version agree there and only there.
      ! Every column has to be probed, because the wrong version differs from
      ! the right one on all of the others.
      worst_error = 0.0_real64
      do column = 1_int32, vocabulary
         do component = 1_int32, width
            original = model%token_embedding(component, column)
            model%token_embedding(component, column) = original + delta
            call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
            model%token_embedding(component, column) = original - delta
            call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
            model%token_embedding(component, column) = original
            numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
            error = abs(numeric - gradient%token_embedding(component, column))
            error = error / max(1.0_real64, abs(numeric))
            worst_error = max(worst_error, error)
         end do
      end do
      if (worst_error >= 1.0e-5_real64) then
         write(output_unit, '(A,ES12.4)') '  worst tied-embedding relative error ', worst_error
      end if
      call check(worst_error < 1.0e-5_real64, &
                 'every tied-embedding entry, all vocabulary columns, matches finite differences')

      call check(numeric == numeric .and. abs(numeric) > 0.0_real64, &
                 'the last finite-difference probe actually moves the loss')

      ! One direction over every learnable element at once.  The probes above
      ! are hand-picked which is exactly how a whole tensor can stay untested:
      ! the tied-embedding defect hid on the ten thousand columns nobody probed.
      ! Here a checkerboard direction touches all twenty-five tensors, and the
      ! analytic side is the same traversal applied to the gradient, so a tensor
      ! the backward pass never writes makes the two numbers disagree.
      direction_delta = 1.0e-7_real64
      saved = model
      call walk_direction(model, gradient, direction_delta, dot, .true.)
      ! The decay and write-strength caches are derived from their logits and are
      ! what the step actually reads, exactly as training refreshes them after
      ! every update; without this the two logits in the direction would move a
      ! cache nobody recomputes and the finite difference would see nothing.
      call fbc_refresh_dynamics(model)
      call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
      model = saved
      call walk_direction(model, gradient, -direction_delta, dot, .true.)
      call fbc_refresh_dynamics(model)
      call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
      model = saved
      call walk_direction(model, gradient, 0.0_real64, dot, .false.)
      sweep_numeric = (plus_loss - minus_loss) / (2.0_real64 * direction_delta)
      if (abs(sweep_numeric - dot) >= 1.0e-6_real64 + 1.0e-4_real64 * abs(dot)) then
         write(output_unit, '(A,2ES14.6)') '  directional sweep numeric/analytic ', sweep_numeric, dot
      end if
      call check(abs(sweep_numeric - dot) < 1.0e-6_real64 + 1.0e-4_real64 * abs(dot), &
                 'one checkerboard direction over all twenty-five tensors matches finite differences')

      call fbc_gradient_destroy(gradient)
      call fbc_state_destroy(base)
      call fbc_state_destroy(probe)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_gradient_check

   ! Before the ring is full and away from decay clamps, both training rules
   ! must compute the same derivatives. This also checks v6's in-place outer
   ! accumulation against the legacy expression, including a second addition.
   subroutine test_outer_accumulation_parity
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: old_gradient, new_gradient
      real(real64) :: hidden(8), logits(16), dlogits(16), loss
      integer(int32) :: status, repetition
      logical :: same

      config = tiny_config(8192_int64)
      call fbc_model_init(model, config, 42_int64, status)
      call check(status == FBC_OK, 'outer parity model initializes')
      model%training_rule = FBC_TRAINING_RULE_V6
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      call fbc_gradient_init(old_gradient, model, status)
      call fbc_gradient_init(new_gradient, model, status)
      call fbc_step(model, state, workspace, 3_int32, hidden, logits, status)
      call fbc_loss_and_gradient(logits, 7_int32, dlogits, loss, status)
      call check(status == FBC_OK, 'outer parity forward and loss succeed')
      do repetition = 1_int32, 2_int32
         model%training_rule = FBC_TRAINING_RULE_V5
         call fbc_step_backward(model, state, workspace, dlogits, old_gradient, status)
         call check(status == FBC_OK, 'legacy outer backward succeeds')
         model%training_rule = FBC_TRAINING_RULE_V6
         call fbc_step_backward(model, state, workspace, dlogits, new_gradient, status)
         call check(status == FBC_OK, 'in-place outer backward succeeds')
      end do
      same = maxval(abs(old_gradient%ffn_output - new_gradient%ffn_output)) < 1.0e-12_real64 .and. &
             maxval(abs(old_gradient%ffn_gate - new_gradient%ffn_gate)) < 1.0e-12_real64 .and. &
             maxval(abs(old_gradient%ffn_value - new_gradient%ffn_value)) < 1.0e-12_real64 .and. &
             maxval(abs(old_gradient%attention_output - new_gradient%attention_output)) < 1.0e-12_real64 .and. &
             maxval(abs(old_gradient%slot_read_proj - new_gradient%slot_read_proj)) < 1.0e-12_real64
      call check(same, 'v6 in-place outer products match repeated legacy accumulation')
      call fbc_gradient_destroy(old_gradient)
      call fbc_gradient_destroy(new_gradient)
      call fbc_workspace_destroy(workspace)
      call fbc_state_destroy(state)
      call fbc_model_destroy(model)
   end subroutine test_outer_accumulation_parity

   !> The full ring used to overwrite the oldest key/value before backward read
   !> them. Probe both a one-slot ring and a repeatedly wrapped four-slot ring;
   !> all finite differences replay from the exact same frozen pre-step state.
   subroutine test_full_ring_backward(rule)
      integer(int32), intent(in) :: rule
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model, saved
      type(fbc_state_t) :: base, pending, immediate, error_state
      type(fbc_workspace_t) :: workspace, immediate_workspace
      type(fbc_gradient_t) :: gradient, legacy_gradient
      type(fbc_optimizer_t) :: optimizer
      type(fbc_update_config_t) :: update_config
      type(fbc_metrics_t) :: metrics
      integer(int32) :: status, window, index
      integer(int32) :: invalid_tokens(2)
      real(real64) :: hidden(8), logits(16), dlogits(16), immediate_hidden(8), immediate_logits(16)
      real(real64) :: ignored_hidden(8), ignored_logits(16), loss, original, plus_loss, minus_loss, numeric, dot
      real(real64), parameter :: delta = 1.0e-6_real64, direction_delta = 1.0e-7_real64

      do window = 1_int32, 4_int32, 3_int32
         config = tiny_config(1024_int64)
         config%local_window = window
         call fbc_model_init(model, config, 157_int64, status)
         model%training_rule = rule
         model%induction_mix = [0.4_real64, -0.25_real64]
         model%channel_router(FBC_CHANNEL_RETENTION, 2_int32) = 0.30_real64
         model%channel_router(FBC_CHANNEL_SLOTS, 4_int32) = -0.20_real64
         call fbc_state_init(base, model, status)
         call fbc_workspace_init(workspace, model, status)
         call fbc_workspace_init(immediate_workspace, model, status)
         call fbc_gradient_init(gradient, model, status)
         call fbc_gradient_init(legacy_gradient, model, status)
         do index = 1_int32, 2_int32 * window + 3_int32
            call fbc_step(model, base, workspace, 1_int32 + modulo(3_int32 * index, 15_int32), hidden, logits, status)
         end do
         call check(status == FBC_OK .and. base%cache_count == window, 'full-ring gradient fixture wraps')

         pending = base
         call fbc_step(model, pending, workspace, 9_int32, hidden, logits, status, defer_local_commit=.true.)
         call check(status == FBC_OK .and. pending%cache_count < 0_int32, 'full-ring local write is deferred')
         call fbc_loss_and_gradient(logits, 5_int32, dlogits, loss, status)
         call fbc_step(model, pending, workspace, 3_int32, ignored_hidden, ignored_logits, status)
         call check(status == FBC_ERR_ORDER, 'another forward step is rejected while the write is pending')
         call fbc_gradient_zero(gradient)
         call fbc_step_backward(model, pending, workspace, dlogits, gradient, status)
         call check(status == FBC_OK, 'v6 backward reads the pre-write full ring')
         call fbc_finish_local_step(pending, workspace, status)
         call check(status == FBC_OK, 'deferred local write commits')
         call fbc_finish_local_step(pending, workspace, status)
         call check(status == FBC_ERR_ORDER, 'a deferred write cannot be committed twice')

         immediate = base
         call fbc_step(model, immediate, immediate_workspace, 9_int32, immediate_hidden, immediate_logits, status)
         call check(all(hidden == immediate_hidden) .and. all(logits == immediate_logits), &
                    'deferred and immediate forward outputs are identical')
         call check(all(pending%local_key == immediate%local_key) .and. &
                    all(pending%local_value == immediate%local_value) .and. &
                    all(pending%retention_numerator == immediate%retention_numerator) .and. &
                    all(pending%retention_normalizer == immediate%retention_normalizer) .and. &
                    all(pending%slot_key == immediate%slot_key) .and. all(pending%slot_value == immediate%slot_value) .and. &
                    all(pending%slot_salience == immediate%slot_salience) .and. &
                    pending%cache_head == immediate%cache_head .and. pending%cache_count == immediate%cache_count .and. &
                    pending%steps == immediate%steps, 'deferred commit leaves the same complete stream state')
         call fbc_gradient_zero(legacy_gradient)
         call fbc_step_backward(model, immediate, immediate_workspace, dlogits, legacy_gradient, status)
         call check(status == FBC_ERR_ORDER .and. all(legacy_gradient%query == 0.0_real64), &
                    'v6 refuses a full-ring backward after the old entry has been overwritten')
         if (rule == FBC_TRAINING_RULE_V6) then
            ! V5 and V6 have the same forward state; compare their historical
            ! and corrected full-ring derivatives without mixing V7 state.
            model%training_rule = FBC_TRAINING_RULE_V5
            call fbc_step_backward(model, immediate, immediate_workspace, dlogits, legacy_gradient, status)
            call check(status == FBC_OK .and. sum(abs(legacy_gradient%query - gradient%query)) > 1.0e-9_real64, &
                       'legacy v5 retains the historical full-ring gradient, distinct from v6')
            model%training_rule = rule
         end if

         original = model%query(1_int32, 2_int32, 1_int32)
         model%query(1_int32, 2_int32, 1_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%query(1_int32, 2_int32, 1_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%query(1_int32, 2_int32, 1_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%query(1_int32, 2_int32, 1_int32), &
                              'full-ring query projection matches frozen-state finite difference')

         original = model%distance_bias(window + 1_int32)
         model%distance_bias(window + 1_int32) = original + delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%distance_bias(window + 1_int32) = original - delta
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model%distance_bias(window + 1_int32) = original
         numeric = (plus_loss - minus_loss) / (2.0_real64 * delta)
         call report_gradient(numeric, gradient%distance_bias(window + 1_int32), &
                              'full-ring oldest distance bias matches finite difference')

         saved = model
         call walk_direction(model, gradient, direction_delta, dot, .true.)
         call fbc_refresh_dynamics(model)
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model = saved
         call walk_direction(model, gradient, -direction_delta, dot, .true.)
         call fbc_refresh_dynamics(model)
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         model = saved
         call walk_direction(model, gradient, 0.0_real64, dot, .false.)
         numeric = (plus_loss - minus_loss) / (2.0_real64 * direction_delta)
         call check(abs(numeric - dot) < 1.0e-6_real64 + 1.0e-4_real64 * abs(dot), &
                    'all-parameter directional gradient matches with a full ring')

         ! Even a rejected target must leave a complete forward state, just as
         ! the historical immediate-write path did, and perform no update.
         error_state = base
         call fbc_optimizer_init(optimizer, config, status)
         metrics = fbc_metrics_t()
         invalid_tokens = [9_int32, 17_int32]
         call fbc_train_stream(model, optimizer, error_state, invalid_tokens, 1_int32, 1_int32, &
                               update_config, workspace, gradient, metrics, status)
         call check(status == FBC_ERR_INVALID_ARGUMENT .and. error_state%cache_count == window .and. &
                    error_state%steps == immediate%steps .and. all(error_state%local_key == immediate%local_key) .and. &
                    all(error_state%local_value == immediate%local_value) .and. metrics%updates == 0_int64, &
                    'failed training target finishes the pending local write without an update')

         call fbc_optimizer_destroy(optimizer)
         call fbc_gradient_destroy(gradient)
         call fbc_gradient_destroy(legacy_gradient)
         call fbc_state_destroy(base)
         call fbc_state_destroy(pending)
         call fbc_state_destroy(immediate)
         call fbc_state_destroy(error_state)
         call fbc_workspace_destroy(workspace)
         call fbc_workspace_destroy(immediate_workspace)
         call fbc_model_destroy(model)
         call fbc_model_destroy(saved)
      end do
   end subroutine test_full_ring_backward

   !> The cached decay is clipped. A logit beyond either bound cannot change
   !> the forward computation; its true frozen-state gradient must be zero.
   subroutine test_clamped_decay_backward
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: base, probe
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: gradient
      integer(int32) :: status, index, side
      real(real64) :: hidden(8), logits(16), dlogits(16), loss, original, plus_loss, minus_loss
      real(real64), parameter :: delta = 1.0e-4_real64

      config = tiny_config(1024_int64)
      call fbc_model_init(model, config, 81_int64, status)
      call fbc_state_init(base, model, status)
      call fbc_workspace_init(workspace, model, status)
      call fbc_gradient_init(gradient, model, status)
      do index = 1_int32, 3_int32
         call fbc_step(model, base, workspace, 1_int32 + index, hidden, logits, status)
      end do
      do side = 1_int32, 2_int32
         if (side == 1_int32) then
            original = -20.0_real64
         else
            original = 40.0_real64
         end if
         model%decay_logit(1_int32) = original
         call fbc_refresh_dynamics(model)
         probe = base
         call fbc_step(model, probe, workspace, 9_int32, hidden, logits, status, defer_local_commit=.true.)
         call fbc_loss_and_gradient(logits, 5_int32, dlogits, loss, status)
         call fbc_gradient_zero(gradient)
         call fbc_step_backward(model, probe, workspace, dlogits, gradient, status)
         call check(status == FBC_OK .and. gradient%decay_logit(1_int32) == 0.0_real64, &
                    'v6 gives zero gradient outside a decay clamp')
         call fbc_finish_local_step(probe, workspace, status)
         model%decay_logit(1_int32) = original + delta
         call fbc_refresh_dynamics(model)
         call replay_loss(model, base, workspace, 9_int32, 5_int32, plus_loss, status)
         model%decay_logit(1_int32) = original - delta
         call fbc_refresh_dynamics(model)
         call replay_loss(model, base, workspace, 9_int32, 5_int32, minus_loss, status)
         call check(plus_loss == minus_loss, 'clamped-decay loss is unchanged by a local logit perturbation')
      end do
      call fbc_gradient_destroy(gradient)
      call fbc_state_destroy(base)
      call fbc_state_destroy(probe)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_clamped_decay_backward

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

   !> A single checkerboard direction over every learnable element.
   !>
   !> `apply = .true.` displaces `model` by `delta * s` for every element, where
   !> `s` alternates +1/-1 along one fixed traversal.  `apply = .false.` walks the
   !> same order and accumulates `sum(s * gradient_element)` into `dot`.  The
   !> traversal lives in exactly one place, so the two sides cannot drift apart,
   !> and a tensor the backward pass never writes contributes nothing to `dot`
   !> while the finite difference still sees it -- one number, all 25 tensors.
   subroutine walk_direction(model, gradient, delta, dot, apply)
      type(fbc_model_t), intent(inout) :: model
      type(fbc_gradient_t), intent(in) :: gradient
      real(real64), intent(in) :: delta
      real(real64), intent(out) :: dot
      logical, intent(in) :: apply
      integer(int32) :: tensor, i, j, k, counter
      real(real64) :: sign_value

      dot = 0.0_real64
      counter = 0_int32
      do tensor = 1_int32, 25_int32
         select case (tensor)
         case (1_int32)
            do j = 1_int32, size(model%token_embedding, 2, kind=int32)
               do i = 1_int32, size(model%token_embedding, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%token_embedding(i, j) = model%token_embedding(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%token_embedding(i, j)
                  end if
               end do
            end do
         case (2_int32)
            do i = 1_int32, size(model%output_bias, kind=int32)
               counter = counter + 1_int32
               sign_value = 1.0_real64
               if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
               if (apply) then
                  model%output_bias(i) = model%output_bias(i) + delta * sign_value
               else
                  dot = dot + sign_value * gradient%output_bias(i)
               end if
            end do
         case (3_int32)
            do k = 1_int32, size(model%query, 3, kind=int32)
               do j = 1_int32, size(model%query, 2, kind=int32)
                  do i = 1_int32, size(model%query, 1, kind=int32)
                     counter = counter + 1_int32
                     sign_value = 1.0_real64
                     if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                     if (apply) then
                        model%query(i, j, k) = model%query(i, j, k) + delta * sign_value
                     else
                        dot = dot + sign_value * gradient%query(i, j, k)
                     end if
                  end do
               end do
            end do
         case (4_int32)
            do k = 1_int32, size(model%key, 3, kind=int32)
               do j = 1_int32, size(model%key, 2, kind=int32)
                  do i = 1_int32, size(model%key, 1, kind=int32)
                     counter = counter + 1_int32
                     sign_value = 1.0_real64
                     if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                     if (apply) then
                        model%key(i, j, k) = model%key(i, j, k) + delta * sign_value
                     else
                        dot = dot + sign_value * gradient%key(i, j, k)
                     end if
                  end do
               end do
            end do
         case (5_int32)
            do k = 1_int32, size(model%value, 3, kind=int32)
               do j = 1_int32, size(model%value, 2, kind=int32)
                  do i = 1_int32, size(model%value, 1, kind=int32)
                     counter = counter + 1_int32
                     sign_value = 1.0_real64
                     if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                     if (apply) then
                        model%value(i, j, k) = model%value(i, j, k) + delta * sign_value
                     else
                        dot = dot + sign_value * gradient%value(i, j, k)
                     end if
                  end do
               end do
            end do
         case (6_int32)
            do j = 1_int32, size(model%attention_output, 2, kind=int32)
               do i = 1_int32, size(model%attention_output, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%attention_output(i, j) = model%attention_output(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%attention_output(i, j)
                  end if
               end do
            end do
         case (7_int32)
            do i = 1_int32, size(model%attention_rms, kind=int32)
               counter = counter + 1_int32
               sign_value = 1.0_real64
               if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
               if (apply) then
                  model%attention_rms(i) = model%attention_rms(i) + delta * sign_value
               else
                  dot = dot + sign_value * gradient%attention_rms(i)
               end if
            end do
         case (8_int32)
            do i = 1_int32, size(model%ffn_rms, kind=int32)
               counter = counter + 1_int32
               sign_value = 1.0_real64
               if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
               if (apply) then
                  model%ffn_rms(i) = model%ffn_rms(i) + delta * sign_value
               else
                  dot = dot + sign_value * gradient%ffn_rms(i)
               end if
            end do
         case (9_int32)
            do j = 1_int32, size(model%ffn_gate, 2, kind=int32)
               do i = 1_int32, size(model%ffn_gate, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%ffn_gate(i, j) = model%ffn_gate(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%ffn_gate(i, j)
                  end if
               end do
            end do
         case (10_int32)
            do j = 1_int32, size(model%ffn_value, 2, kind=int32)
               do i = 1_int32, size(model%ffn_value, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%ffn_value(i, j) = model%ffn_value(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%ffn_value(i, j)
                  end if
               end do
            end do
         case (11_int32)
            do j = 1_int32, size(model%ffn_output, 2, kind=int32)
               do i = 1_int32, size(model%ffn_output, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%ffn_output(i, j) = model%ffn_output(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%ffn_output(i, j)
                  end if
               end do
            end do
         case (12_int32)
            do i = 1_int32, size(model%distance_bias, kind=int32)
               counter = counter + 1_int32
               sign_value = 1.0_real64
               if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
               if (apply) then
                  model%distance_bias(i) = model%distance_bias(i) + delta * sign_value
               else
                  dot = dot + sign_value * gradient%distance_bias(i)
               end if
            end do
         case (13_int32)
            do i = 1_int32, size(model%scale_logit, kind=int32)
               counter = counter + 1_int32
               sign_value = 1.0_real64
               if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
               if (apply) then
                  model%scale_logit(i) = model%scale_logit(i) + delta * sign_value
               else
                  dot = dot + sign_value * gradient%scale_logit(i)
               end if
            end do
         case (14_int32)
            do i = 1_int32, size(model%decay_logit, kind=int32)
               counter = counter + 1_int32
               sign_value = 1.0_real64
               if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
               if (apply) then
                  model%decay_logit(i) = model%decay_logit(i) + delta * sign_value
               else
                  dot = dot + sign_value * gradient%decay_logit(i)
               end if
            end do
         case (15_int32)
            do i = 1_int32, size(model%write_strength_logit, kind=int32)
               counter = counter + 1_int32
               sign_value = 1.0_real64
               if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
               if (apply) then
                  model%write_strength_logit(i) = model%write_strength_logit(i) + delta * sign_value
               else
                  dot = dot + sign_value * gradient%write_strength_logit(i)
               end if
            end do
         case (16_int32)
            do i = 1_int32, size(model%channel_logit, kind=int32)
               counter = counter + 1_int32
               sign_value = 1.0_real64
               if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
               if (apply) then
                  model%channel_logit(i) = model%channel_logit(i) + delta * sign_value
               else
                  dot = dot + sign_value * gradient%channel_logit(i)
               end if
            end do
         case (17_int32)
            do j = 1_int32, size(model%channel_gate_proj, 2, kind=int32)
               do i = 1_int32, size(model%channel_gate_proj, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%channel_gate_proj(i, j) = model%channel_gate_proj(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%channel_gate_proj(i, j)
                  end if
               end do
            end do
         case (18_int32)
            do j = 1_int32, size(model%channel_router, 2, kind=int32)
               do i = 1_int32, size(model%channel_router, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%channel_router(i, j) = model%channel_router(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%channel_router(i, j)
                  end if
               end do
            end do
         case (19_int32)
            do i = 1_int32, size(model%induction_mix, kind=int32)
               counter = counter + 1_int32
               sign_value = 1.0_real64
               if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
               if (apply) then
                  model%induction_mix(i) = model%induction_mix(i) + delta * sign_value
               else
                  dot = dot + sign_value * gradient%induction_mix(i)
               end if
            end do
         case (20_int32)
            do j = 1_int32, size(model%scale_gate_proj, 2, kind=int32)
               do i = 1_int32, size(model%scale_gate_proj, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%scale_gate_proj(i, j) = model%scale_gate_proj(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%scale_gate_proj(i, j)
                  end if
               end do
            end do
         case (21_int32)
            do j = 1_int32, size(model%slot_query_proj, 2, kind=int32)
               do i = 1_int32, size(model%slot_query_proj, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%slot_query_proj(i, j) = model%slot_query_proj(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%slot_query_proj(i, j)
                  end if
               end do
            end do
         case (22_int32)
            do j = 1_int32, size(model%slot_key_proj, 2, kind=int32)
               do i = 1_int32, size(model%slot_key_proj, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%slot_key_proj(i, j) = model%slot_key_proj(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%slot_key_proj(i, j)
                  end if
               end do
            end do
         case (23_int32)
            do j = 1_int32, size(model%slot_value_proj, 2, kind=int32)
               do i = 1_int32, size(model%slot_value_proj, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%slot_value_proj(i, j) = model%slot_value_proj(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%slot_value_proj(i, j)
                  end if
               end do
            end do
         case (24_int32)
            do j = 1_int32, size(model%slot_read_proj, 2, kind=int32)
               do i = 1_int32, size(model%slot_read_proj, 1, kind=int32)
                  counter = counter + 1_int32
                  sign_value = 1.0_real64
                  if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
                  if (apply) then
                     model%slot_read_proj(i, j) = model%slot_read_proj(i, j) + delta * sign_value
                  else
                     dot = dot + sign_value * gradient%slot_read_proj(i, j)
                  end if
               end do
            end do
         case (25_int32)
            counter = counter + 1_int32
            sign_value = 1.0_real64
            if (mod(counter, 2_int32) == 0_int32) sign_value = -1.0_real64
            if (apply) then
               model%slot_salience_weight = model%slot_salience_weight + delta * sign_value
            else
               dot = dot + sign_value * gradient%slot_salience_weight
            end if
         case default
            dot = 0.0_real64
         end select
      end do
   end subroutine walk_direction

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
   !  9. Legacy memory training and the limits of symmetric slots
   !----------------------------------------------------------------------------
   subroutine test_legacy_memory_training
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
      real(real64) :: slots_before(32, 16), scales_before(3), channels_before(3)
      real(real64) :: decays_before(3), writes_before(3)
      real(real64), allocatable :: hidden(:), logits(:), dlogits(:)
      real(real64), allocatable :: gate_before(:, :), scale_gate_before(:, :)
      real(real64), allocatable :: router_before(:, :)
      real(real64), allocatable :: induction_before(:)
      real(real64) :: loss, norm

      config = small_config(2048_int64)
      update_config%learning_rate = 3.0e-3_real64
      allocate(tokens(token_count + 1_int32), mask(token_count))
      call fbc_generate_tokens(FBC_TASK_NEEDLE, token_count, config%vocabulary_size, 32_int32, 9_int64, &
                               tokens(1_int32:token_count), mask, status)
      call check(status == FBC_OK, 'the needle corpus generates')
      tokens(token_count + 1_int32) = 1_int32

      call fbc_model_init(model, config, 9_int64, status)
      model%training_rule = FBC_TRAINING_RULE_V6
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      call fbc_gradient_init(gradient, model, status)
      call fbc_optimizer_init(optimizer, config, status)
      allocate(hidden(config%width), logits(config%vocabulary_size), dlogits(config%vocabulary_size))

      ! Warm the shared slot state before the write-key probe. In V6 the slots
      ! remain identical, so neither query nor salience can choose among distinct
      ! values; tiny roundoff is not evidence that those paths learned a choice.
      do index = 1_int32, 4
         call fbc_step(model, state, workspace, tokens(index), hidden, logits, status)
      end do
      ! Probe the single-token write-then-read path. The value and shared write
      ! gate can have real gradients even when recall-query selection cannot.
      call fbc_gradient_zero(gradient)
      call fbc_step(model, state, workspace, tokens(5_int32), hidden, logits, status)
      call fbc_loss_and_gradient(logits, tokens(6_int32), dlogits, loss, status)
      call fbc_step_backward(model, state, workspace, dlogits, gradient, status)
      call check(sum(abs(gradient%slot_key_proj)) > 0.0_real64, &
                 'the slot write-key projection receives a gradient')
      call check(sum(abs(gradient%slot_value_proj)) > 0.0_real64, &
                 'the slot write-value projection receives a gradient')
      call check(sum(abs(gradient%slot_query_proj)) < 1.0e-12_real64, &
                 'identical legacy slot values carry no meaningful recall-query gradient')
      call check(sum(abs(gradient%slot_read_proj)) > 0.0_real64, &
                 'the slot read projection receives a gradient')
      call check(sum(abs(gradient%scale_logit)) > 0.0_real64, &
                 'the multi-scale readout receives a gradient')
      call check(sum(abs(gradient%channel_logit)) > 0.0_real64, &
                 'the channel mixer receives a gradient')
      call check(sum(abs(gradient%channel_gate_proj)) > 0.0_real64, &
                 'the input-dependent channel gate receives a gradient')
      call check(sum(abs(gradient%channel_router)) > 0.0_real64, &
                 'the context router receives a gradient')
      call check(sum(abs(gradient%induction_mix)) > 0.0_real64, &
                 'the induction weights receive a gradient')
      call check(sum(abs(gradient%scale_gate_proj)) > 0.0_real64, &
                 'the input-dependent scale gate receives a gradient')
      call check(sum(abs(gradient%attention_rms)) > 0.0_real64 .and. &
                 sum(abs(gradient%ffn_rms)) > 0.0_real64, &
                 'both normalization scales receive a gradient')
      call check(abs(gradient%slot_salience_weight) < 1.0e-12_real64, &
                 'identical legacy slot values carry no meaningful salience-selection gradient')
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
      gate_before = model%channel_gate_proj
      scale_gate_before = model%scale_gate_proj
      router_before = model%channel_router
      induction_before = model%induction_mix
      call fbc_train_stream(model, optimizer, state, tokens, token_count, chunk, update_config, &
                            workspace, gradient, metrics, status)
      call check(status == FBT_OK, 'needle training completes')
      call check(maxval(abs(state%slot_key(1_int32:32, 1_int32:16) - slots_before)) > 0.0_real64, &
                 'the slot state really changes while training')
      call check(any(model%scale_logit /= scales_before), &
                 'the multi-scale readout weights are optimised')
      call check(any(model%channel_logit /= channels_before), 'the channel mixer is optimised')
      call check(any(model%channel_gate_proj /= gate_before), &
                 'the input-dependent channel gate is optimised, not just present')
      call check(any(model%scale_gate_proj /= scale_gate_before), &
                 'the input-dependent scale gate is optimised, not just present')
      call check(any(model%channel_router /= router_before), &
                 'the context router is optimised, not just present')
      call check(any(model%induction_mix /= induction_before), &
                 'the induction weights are optimised, not just present')
      call check(fbc_real64_is_finite(model%slot_salience_weight), 'the legacy salience weight stays finite')
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
   end subroutine test_legacy_memory_training

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
      character(len=128) :: path, legacy
      type(fbc_config_t) :: config
      type(fbc_update_config_t) :: update_config, loaded_config, bad_update_config
      type(fbc_model_t) :: model, restored
      type(fbc_optimizer_t) :: optimizer, restored_optimizer
      type(fbc_metrics_t) :: counters, restored_counters
      type(fbc_checkpoint_meta_t) :: meta
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      integer(int32) :: status, index
      real(real64) :: hidden(32), logits(32)
      integer :: unit, io_status

      ! Reserve names in the writable working directory with exclusive CREATE.
      ! This avoids Unix-only /tmp and collisions between concurrent tests.
      call reserve_test_path('fbc-roundtrip-', path, io_status)
      call check(io_status == 0, 'a unique checkpoint fixture can be reserved')
      if (io_status /= 0) return
      call reserve_test_path('fbc-legacy-', legacy, io_status)
      call check(io_status == 0, 'a unique legacy fixture can be reserved')
      if (io_status /= 0) then
         call delete_test_path(path)
         return
      end if

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
      bad_update_config = update_config
      bad_update_config%epsilon = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
      call fbc_checkpoint_save(path, model, optimizer, bad_update_config, counters, status)
      call check(status == FBT_ERR_ARGUMENT, 'invalid optimizer state cannot overwrite a valid checkpoint')
      block
         type(fbc_metrics_t) :: bad_counters
         real(real64) :: good_horizon, good_temperature
         bad_counters = counters
         bad_counters%loss_sum = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
         call fbc_checkpoint_save(path, model, optimizer, update_config, bad_counters, status)
         call check(status == FBT_ERR_NUMERIC, 'nonfinite loss cannot replace a valid checkpoint')
         bad_counters = counters
         bad_counters%tokens = -1_int64
         call fbc_checkpoint_save(path, model, optimizer, update_config, bad_counters, status)
         call check(status == FBT_ERR_ARGUMENT, 'negative counters cannot replace a valid checkpoint')
         bad_counters = counters
         bad_counters%numeric_ok = .false.
         call fbc_checkpoint_save(path, model, optimizer, update_config, bad_counters, status)
         call check(status == FBT_ERR_NUMERIC, 'failed numerical training cannot replace a valid checkpoint')
         good_horizon = model%effective_horizon
         model%effective_horizon = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
         call fbc_checkpoint_save(path, model, optimizer, update_config, counters, status)
         call check(status == FBT_ERR_NUMERIC, 'nonfinite horizon cannot replace a valid checkpoint')
         model%effective_horizon = good_horizon
         good_temperature = model%config%slot_temperature
         model%config%slot_temperature = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
         call fbc_checkpoint_save(path, model, optimizer, update_config, counters, status)
         call check(status == FBT_ERR_ARGUMENT, 'nonfinite model configuration cannot replace a valid checkpoint')
         model%config%slot_temperature = good_temperature
      end block

      call fbc_checkpoint_load(path, restored, restored_optimizer, loaded_config, restored_counters, status)
      call check(status == FBT_OK, 'checkpoint load succeeds')
      call check(restored_counters%updates == counters%updates .and. &
                 restored_counters%tokens == counters%tokens, 'counters survive the round trip')
      call check(abs(loaded_config%learning_rate - update_config%learning_rate) == 0.0_real64, &
                 'optimizer hyper-parameters survive the round trip')
      call check(all(restored%channel_gate_proj == model%channel_gate_proj) .and. &
                 all(restored%scale_gate_proj == model%scale_gate_proj), &
                 'the current checkpoint restores both input-dependent gates exactly')
      call check(all(restored%channel_router == model%channel_router), &
                 'the current checkpoint restores the context router exactly')
      call check(all(restored%induction_mix == model%induction_mix), &
                 'the v5 checkpoint restores the induction weights exactly')
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
      call check(meta%format_version == FBC_TRAINING_RULE_V6 .and. &
                 restored%training_rule == FBC_TRAINING_RULE_V6, 'default checkpoints identify the v6 rule')

      ! V7 changes forward state semantics despite sharing the wire payload;
      ! it must round-trip with its own version, never masquerade as V5/V6.
      model%training_rule = FBC_TRAINING_RULE_V7
      call fbc_checkpoint_save(legacy, model, optimizer, update_config, counters, status)
      call check(status == FBT_OK, 'anchored V7 checkpoint saves')
      call fbc_checkpoint_meta_read(legacy, meta, status)
      call check(status == FBT_OK .and. meta%format_version == FBC_TRAINING_RULE_V7 .and. meta%convertible, &
                 'anchored V7 metadata reports its exact semantics')
      call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, restored_counters, status)
      call check(status == FBT_OK .and. restored%training_rule == FBC_TRAINING_RULE_V7 .and. &
                 all(restored%token_embedding == model%token_embedding), 'anchored V7 model round-trips')
      block
         type(fbc_state_t) :: saved_state, restored_state
         real(real64) :: reference_logits(32), loaded_logits(32)
         call fbc_state_init(saved_state, model, status)
         call fbc_state_init(restored_state, restored, status)
         call fbc_step(model, saved_state, workspace, 11_int32, hidden, reference_logits, status)
         call fbc_step(restored, restored_state, workspace, 11_int32, hidden, loaded_logits, status)
         call check(status == FBC_OK .and. all(reference_logits == loaded_logits) .and. &
                    all(saved_state%slot_key == restored_state%slot_key), &
                    'V7 reload preserves anchored state and forward logits')
         call fbc_state_destroy(saved_state)
         call fbc_state_destroy(restored_state)
      end block

      ! The v5 and v6 payloads have identical layouts. Preserve v5's historical
      ! rule across both load and re-save rather than silently changing resume.
      model%training_rule = FBC_TRAINING_RULE_V5
      call fbc_checkpoint_save(legacy, model, optimizer, update_config, counters, status)
      call check(status == FBT_OK, 'historical v5 checkpoint can be written')
      call fbc_checkpoint_meta_read(legacy, meta, status)
      call check(status == FBT_OK .and. meta%format_version == FBC_TRAINING_RULE_V5 .and. meta%convertible, &
                 'v5 metadata retains its version and is readable')
      call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, restored_counters, status)
      call check(status == FBT_OK .and. restored%training_rule == FBC_TRAINING_RULE_V5 .and. &
                 all(restored%token_embedding == model%token_embedding), 'v5 load retains weights and training rule')
      call fbc_checkpoint_save(legacy, restored, restored_optimizer, loaded_config, restored_counters, status)
      call fbc_checkpoint_meta_read(legacy, meta, status)
      call check(status == FBT_OK .and. meta%format_version == FBC_TRAINING_RULE_V5, &
                 're-saving a v5 model does not silently upgrade its training rule')
      model%training_rule = FBC_TRAINING_RULE_V6

      ! A legacy header must be rejected with its own status, never converted.
      open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      call check(io_status == 0, 'the legacy fixture can be written')
      write(unit) FBC_LEGACY_MAGIC
      write(unit) 5_int32
      close(unit)
      call fbc_checkpoint_meta_read(legacy, meta, status)
      call check(status == FBT_ERR_LEGACY .and. .not. meta%convertible, &
                 'a legacy VGNR header is rejected without claiming convertibility')
      call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, &
                               restored_counters, status)
      call check(status == FBT_ERR_LEGACY, 'loading a legacy checkpoint never converts it silently')

      ! A v2 header (the version before the input-dependent gates) must be
      ! refused with its own status: the payload has no value for the two gate
      ! projections, and a v3 binary that silently assumed zeros would be
      ! reporting a different model than the file describes.
      open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      call check(io_status == 0, 'the v2 fixture can be written')
      write(unit) FBC_CHECKPOINT_MAGIC
      write(unit) 2_int32
      close(unit)
      call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, &
                               restored_counters, status)
      call check(status == FBT_ERR_VERSION, 'a v2 checkpoint is refused, never converted to v3')
      ! Version 4 is the format just before this one: it has the router but no
      ! induction weights.  It is the file a reader is most likely to meet, and
      ! it must be refused rather than completed with zero weights, which would
      ! silently disable a learned capability.
      open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      call check(io_status == 0, 'the v4 fixture can be written')
      write(unit) FBC_CHECKPOINT_MAGIC
      write(unit) 4_int32
      close(unit)
      call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, &
                               restored_counters, status)
      call check(status == FBT_ERR_VERSION, 'a v4 checkpoint is refused, never converted to v5')
      open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      write(unit) FBC_CHECKPOINT_MAGIC
      write(unit) 4_int32
      write(unit) config%vocabulary_size, config%width, config%heads, config%ffn_width, &
                  config%local_window, config%memory_scales, config%memory_slots, &
                  config%logical_horizon, config%rms_epsilon, config%slot_decay, config%slot_temperature
      write(unit) 0.0_real64
      write(unit) 0_int32
      write(unit) 0_int64, 0_int64, 0.0_real64
      close(unit)
      call fbc_checkpoint_meta_read(legacy, meta, status)
      call check(status == FBT_ERR_VERSION .and. meta%format_version == 4_int32 .and. &
                 .not. meta%convertible, 'a v4 header is reported as the older version it is')
      ! Version 3 is the format that had the two input-dependent gates but no
      ! router, so it is the version a reader is most likely to meet in a real
      ! directory: it must be refused, not completed with a zero router that
      ! would silently change the model.
      open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      call check(io_status == 0, 'the v3 fixture can be written')
      write(unit) FBC_CHECKPOINT_MAGIC
      write(unit) 3_int32
      close(unit)
      call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, &
                               restored_counters, status)
      call check(status == FBT_ERR_VERSION, 'a v3 checkpoint is refused, never converted to v4')
      open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      write(unit) FBC_CHECKPOINT_MAGIC
      write(unit) 3_int32
      write(unit) config%vocabulary_size, config%width, config%heads, config%ffn_width, &
                  config%local_window, config%memory_scales, config%memory_slots, &
                  config%logical_horizon, config%rms_epsilon, config%slot_decay, config%slot_temperature
      write(unit) 0.0_real64
      write(unit) 0_int32
      write(unit) 0_int64, 0_int64, 0.0_real64
      close(unit)
      call fbc_checkpoint_meta_read(legacy, meta, status)
      call check(status == FBT_ERR_VERSION .and. meta%format_version == 3_int32 .and. &
                 .not. meta%convertible, 'a v3 header is reported as the older version it is')
      open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      write(unit) FBC_CHECKPOINT_MAGIC
      write(unit) 2_int32
      write(unit) config%vocabulary_size, config%width, config%heads, config%ffn_width, &
                  config%local_window, config%memory_scales, config%memory_slots, &
                  config%logical_horizon, config%rms_epsilon, config%slot_decay, config%slot_temperature
      write(unit) 0.0_real64
      write(unit) 0_int32
      write(unit) 0_int64, 0_int64, 0.0_real64
      close(unit)
      call fbc_checkpoint_meta_read(legacy, meta, status)
      call check(status == FBT_ERR_VERSION, 'the header reader reports the older version instead of guessing')
      call check(meta%format_version == 2_int32 .and. .not. meta%convertible, &
                 'the header keeps the real version number and marks the file non-convertible')

      ! Complete v5 header with a signaling NaN optimizer epsilon: the loader
      ! must reject it before model allocation or arithmetic on that value.
      open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      write(unit) FBC_CHECKPOINT_MAGIC, 5_int32
      write(unit) config%vocabulary_size, config%width, config%heads, config%ffn_width, &
                  config%local_window, config%memory_scales, config%memory_slots, &
                  config%logical_horizon, config%rms_epsilon, config%slot_decay, config%slot_temperature
      write(unit) model%effective_horizon, 0_int32, counters%updates, counters%tokens, counters%loss_sum
      write(unit) bad_update_config%learning_rate, bad_update_config%momentum, bad_update_config%energy_decay, &
                  bad_update_config%restart_damp, bad_update_config%weight_decay, bad_update_config%epsilon, &
                  bad_update_config%gradient_clip
      close(unit)
      call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, restored_counters, status)
      call check(status == FBT_ERR_FORMAT .and. .not. restored%initialized .and. .not. restored_optimizer%initialized, &
                 'checkpoint with signaling NaN hyperparameter is rejected before allocation')
      block
         type(fbc_metrics_t) :: bad_counters
         real(real64) :: bad_horizon
         integer(int32) :: bad_flag, fixture
         do fixture = 1_int32, 4_int32
            bad_counters = counters
            bad_horizon = model%effective_horizon
            bad_flag = 0_int32
            select case (fixture)
            case (1_int32)
               bad_counters%tokens = -1_int64
            case (2_int32)
               bad_counters%loss_sum = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
            case (3_int32)
               bad_horizon = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
            case (4_int32)
               bad_flag = 2_int32
            end select
            open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
                 form='unformatted', iostat=io_status)
            write(unit) FBC_CHECKPOINT_MAGIC, FBC_TRAINING_RULE_V6
            write(unit) config%vocabulary_size, config%width, config%heads, config%ffn_width, &
                        config%local_window, config%memory_scales, config%memory_slots, &
                        config%logical_horizon, config%rms_epsilon, config%slot_decay, config%slot_temperature
            write(unit) bad_horizon, bad_flag, bad_counters%updates, bad_counters%tokens, bad_counters%loss_sum
            write(unit) update_config%learning_rate, update_config%momentum, update_config%energy_decay, &
                        update_config%restart_damp, update_config%weight_decay, update_config%epsilon, &
                        update_config%gradient_clip
            close(unit)
            call fbc_checkpoint_meta_read(legacy, meta, status)
            call check(status == FBT_ERR_FORMAT .and. .not. meta%convertible, &
                       'checkpoint inspection rejects invalid scalar header metadata')
            call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, restored_counters, status)
            call check(status == FBT_ERR_FORMAT .and. .not. restored%initialized .and. &
                       .not. restored_optimizer%initialized, 'invalid checkpoint metadata is rejected before allocation')
         end do
      end block
      block
         type(fbc_config_t) :: bad_config
         integer(int32) :: fixture
         ! Header inspection is intentionally not payload validation, but a
         ! malformed physical model configuration must never be called usable.
         do fixture = 1_int32, 9_int32
            bad_config = config
            select case (fixture)
            case (1_int32)
               bad_config%width = 0_int32
            case (2_int32)
               bad_config%heads = 0_int32
            case (3_int32)
               bad_config%heads = 3_int32
            case (4_int32)
               bad_config%ffn_width = bad_config%width - 1_int32
            case (5_int32)
               bad_config%slot_decay = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
            case (6_int32)
               bad_config%rms_epsilon = -1.0_real64
            case (7_int32)
               bad_config%slot_temperature = transfer(int(z'7FF0000000000001', int64), 0.0_real64)
            case (8_int32)
               bad_config%logical_horizon = 0_int64
            case (9_int32)
               bad_config%local_window = FBC_MAX_LOCAL_WINDOW + 1_int32
            end select
            open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
                 form='unformatted', iostat=io_status)
            write(unit) FBC_CHECKPOINT_MAGIC, FBC_TRAINING_RULE_V6
            write(unit) bad_config%vocabulary_size, bad_config%width, bad_config%heads, bad_config%ffn_width, &
                        bad_config%local_window, bad_config%memory_scales, bad_config%memory_slots, &
                        bad_config%logical_horizon, bad_config%rms_epsilon, &
                        bad_config%slot_decay, bad_config%slot_temperature
            write(unit) model%effective_horizon, 0_int32, counters%updates, counters%tokens, counters%loss_sum
            write(unit) update_config%learning_rate, update_config%momentum, update_config%energy_decay, &
                        update_config%restart_damp, update_config%weight_decay, update_config%epsilon, &
                        update_config%gradient_clip
            close(unit)
            call fbc_checkpoint_meta_read(legacy, meta, status)
            call check(status == FBT_ERR_FORMAT .and. .not. meta%convertible, &
                       'inspection refuses a malformed or nonfinite physical model configuration')
            call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, restored_counters, status)
            call check(status == FBT_ERR_FORMAT .and. .not. restored%initialized .and. &
                       .not. restored_optimizer%initialized, &
                       'malformed checkpoint model configuration is rejected before allocation')
         end do
      end block
      block
         type(fbc_config_t) :: header_only_config
         integer(int32) :: fixture
         do fixture = 1_int32, 2_int32
            header_only_config = config
            if (fixture == 2_int32) then
               ! Valid extents near the supported maximum, but no payload. This
               ! must be rejected before an attempted gigantic model allocation.
               header_only_config%width = FBC_MAX_WIDTH
               header_only_config%heads = 1_int32
               header_only_config%ffn_width = huge(0_int32)
               header_only_config%vocabulary_size = huge(0_int32)
               header_only_config%local_window = FBC_MAX_LOCAL_WINDOW
               header_only_config%memory_scales = FBC_MAX_MEMORY_SCALES
               header_only_config%memory_slots = FBC_MAX_MEMORY_SLOTS
            end if
            open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
                 form='unformatted', iostat=io_status)
            write(unit) FBC_CHECKPOINT_MAGIC, FBC_TRAINING_RULE_V6
            write(unit) header_only_config%vocabulary_size, header_only_config%width, &
                        header_only_config%heads, header_only_config%ffn_width, header_only_config%local_window, &
                        header_only_config%memory_scales, header_only_config%memory_slots, &
                        header_only_config%logical_horizon, header_only_config%rms_epsilon, &
                        header_only_config%slot_decay, header_only_config%slot_temperature
            write(unit) model%effective_horizon, 0_int32, counters%updates, counters%tokens, counters%loss_sum
            write(unit) update_config%learning_rate, update_config%momentum, update_config%energy_decay, &
                        update_config%restart_damp, update_config%weight_decay, update_config%epsilon, &
                        update_config%gradient_clip
            close(unit)
            call fbc_checkpoint_meta_read(legacy, meta, status)
            call check(status == FBT_OK .and. meta%convertible, &
                       'header-only inspection does not claim to check missing payload')
            call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, restored_counters, status)
            call check(status == FBT_ERR_FORMAT .and. .not. restored%initialized .and. &
                       .not. restored_optimizer%initialized, &
                       'short checkpoint is rejected before its model or optimizer is allocated')
         end do
      end block

      block
         integer(int64) :: file_size
         integer :: character_units
         character(len=:), allocatable :: partial
         file_size = -1_int64
         inquire(iolength=character_units) 'x'
         inquire(file=path, size=file_size, iostat=io_status)
         call check(io_status == 0 .and. file_size > 1_int64 .and. file_size <= int(huge(0), int64), &
                    'saved checkpoint size is available for a truncation fixture')
         if (io_status == 0 .and. file_size > 1_int64 .and. file_size <= int(huge(0), int64) .and. &
             character_units == 1) then
            allocate(character(len=int(file_size - 1_int64)) :: partial)
            open(newunit=unit, file=path, status='old', action='read', access='stream', &
                 form='unformatted', iostat=io_status)
            if (io_status == 0) then
               read(unit, iostat=io_status) partial
               close(unit)
            end if
            call check(io_status == 0, 'truncation fixture reads the checkpoint prefix')
            if (io_status == 0) then
               open(newunit=unit, file=legacy, status='replace', action='write', access='stream', &
                    form='unformatted', iostat=io_status)
               if (io_status == 0) then
                  write(unit, iostat=io_status) partial
                  close(unit)
               end if
               call check(io_status == 0, 'truncation fixture writes all but the final storage unit')
               if (io_status == 0) then
                  call fbc_checkpoint_load(legacy, restored, restored_optimizer, loaded_config, restored_counters, status)
                  call check(status == FBT_ERR_FORMAT .and. .not. restored%initialized .and. &
                             .not. restored_optimizer%initialized, &
                             'a checkpoint truncated by one storage unit is rejected before allocation')
               end if
            end if
            deallocate(partial)
         end if
      end block

      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
      call fbc_model_destroy(restored)
      call fbc_optimizer_destroy(optimizer)
      call fbc_optimizer_destroy(restored_optimizer)
      call delete_test_path(path)
      call delete_test_path(legacy)
   end subroutine test_checkpoint_roundtrip

   subroutine reserve_test_path(prefix, path, io_status)
      character(len=*), intent(in) :: prefix
      character(len=*), intent(out) :: path
      integer, intent(out) :: io_status
      integer(int64) :: clock_value
      integer :: attempt, unit

      call system_clock(count=clock_value)
      do attempt = 1, 1000
         write(path, '(A,I0,A,I0,A)') prefix, clock_value, '-', attempt, '.vgfb'
         open(newunit=unit, file=trim(path), status='new', action='write', &
              access='stream', form='unformatted', iostat=io_status)
         if (io_status == 0) then
            close(unit, iostat=io_status)
            return
         end if
      end do
   end subroutine reserve_test_path

   subroutine delete_test_path(path)
      character(len=*), intent(in) :: path
      integer :: unit, io_status
      open(newunit=unit, file=trim(path), status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete', iostat=io_status)
      call check(io_status == 0, 'the reserved checkpoint fixture is removed')
   end subroutine delete_test_path

   !> Checkpointed continuation is exact when the caller retains the recurrent
   !> stream state. The file itself does not serialize that state; both rules
   !> are checked with a ring that wraps many times before and after the save.
   subroutine test_full_ring_checkpoint_continuation
      type(fbc_config_t) :: config
      type(fbc_update_config_t) :: update_config, loaded_config
      type(fbc_model_t) :: whole_model, split_model, restored_model
      type(fbc_optimizer_t) :: whole_optimizer, split_optimizer, restored_optimizer
      type(fbc_state_t) :: whole_state, split_state
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: gradient
      type(fbc_metrics_t) :: whole_metrics, split_metrics, restored_metrics
      integer(int32), parameter :: n = 128_int32, chunk = 16_int32
      integer(int32) :: tokens(n + 1_int32), status, rule
      logical :: mask(n)
      character(len=128) :: first_path, second_path
      integer :: io_status

      call reserve_test_path('fbc-continuation-a-', first_path, io_status)
      call check(io_status == 0, 'reserve the first continuation fixture')
      if (io_status /= 0) return
      call reserve_test_path('fbc-continuation-b-', second_path, io_status)
      call check(io_status == 0, 'reserve the second continuation fixture')
      if (io_status /= 0) then
         call delete_test_path(first_path)
         return
      end if
      config = tiny_config(1024_int64)
      update_config = fbc_update_config_t()
      call fbc_generate_tokens(FBC_TASK_NEEDLE, n, config%vocabulary_size, 7_int32, 131_int64, tokens(:n), mask, status)
      tokens(n + 1_int32) = 1_int32
      do rule = FBC_TRAINING_RULE_V5, FBC_TRAINING_RULE_V7
         call fbc_model_init(whole_model, config, 91_int64, status)
         whole_model%training_rule = rule
         split_model = whole_model
         call fbc_state_init(whole_state, whole_model, status)
         call fbc_state_init(split_state, split_model, status)
         call fbc_workspace_init(workspace, whole_model, status)
         call fbc_gradient_init(gradient, whole_model, status)
         call fbc_optimizer_init(whole_optimizer, config, status)
         call fbc_optimizer_init(split_optimizer, config, status)
         whole_metrics = fbc_metrics_t()
         split_metrics = fbc_metrics_t()
         call fbc_train_stream(whole_model, whole_optimizer, whole_state, tokens, n, chunk, update_config, &
                               workspace, gradient, whole_metrics, status)
         call check(status == FBT_OK, 'full-ring one-pass training succeeds')
         call fbc_train_stream(split_model, split_optimizer, split_state, tokens(:n / 2 + 1), n / 2, chunk, &
                               update_config, workspace, gradient, split_metrics, status)
         call check(status == FBT_OK, 'full-ring first training segment succeeds')
         call fbc_checkpoint_save(first_path, split_model, split_optimizer, update_config, split_metrics, status)
         call check(status == FBT_OK, 'full-ring partial checkpoint saves')
         call fbc_checkpoint_load(first_path, restored_model, restored_optimizer, loaded_config, restored_metrics, status)
         call check(status == FBT_OK .and. restored_model%training_rule == rule, &
                    'full-ring checkpoint retains its training rule')
         call fbc_train_stream(restored_model, restored_optimizer, split_state, tokens(n / 2 + 1:), n / 2, chunk, &
                               loaded_config, workspace, gradient, restored_metrics, status)
         call check(status == FBT_OK, 'full-ring continuation succeeds with retained stream state')
         call check(whole_state%steps == split_state%steps .and. whole_state%cache_head == split_state%cache_head .and. &
                    whole_state%cache_count == split_state%cache_count .and. &
                    all(whole_state%local_key == split_state%local_key) .and. &
                    all(whole_state%local_value == split_state%local_value) .and. &
                    all(whole_state%retention_numerator == split_state%retention_numerator) .and. &
                    all(whole_state%retention_normalizer == split_state%retention_normalizer) .and. &
                    all(whole_state%slot_key == split_state%slot_key) .and. &
                    all(whole_state%slot_value == split_state%slot_value) .and. &
                    all(whole_state%slot_salience == split_state%slot_salience), &
                    'checkpoint continuation retains the same recurrent stream state')
         call fbc_checkpoint_save(first_path, whole_model, whole_optimizer, update_config, whole_metrics, status)
         call check(status == FBT_OK, 'one-pass final checkpoint saves')
         call fbc_checkpoint_save(second_path, restored_model, restored_optimizer, loaded_config, restored_metrics, status)
         call check(status == FBT_OK .and. test_files_identical(first_path, second_path), &
                    'one-pass and resumed model/optimizer checkpoints are byte-identical')
         call fbc_model_destroy(whole_model)
         call fbc_model_destroy(split_model)
         call fbc_model_destroy(restored_model)
         call fbc_optimizer_destroy(whole_optimizer)
         call fbc_optimizer_destroy(split_optimizer)
         call fbc_optimizer_destroy(restored_optimizer)
         call fbc_state_destroy(whole_state)
         call fbc_state_destroy(split_state)
         call fbc_workspace_destroy(workspace)
         call fbc_gradient_destroy(gradient)
      end do
      call delete_test_path(first_path)
      call delete_test_path(second_path)
   end subroutine test_full_ring_checkpoint_continuation

   logical function test_files_identical(first_path, second_path) result(equal)
      character(len=*), intent(in) :: first_path, second_path
      character(len=4096) :: first_chunk, second_chunk
      integer(int64) :: first_size, second_size, remaining
      integer :: first_unit, second_unit, io_status, count

      equal = .false.
      open(newunit=first_unit, file=trim(first_path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) return
      open(newunit=second_unit, file=trim(second_path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         close(first_unit)
         return
      end if
      inquire(unit=first_unit, size=first_size, iostat=io_status)
      if (io_status == 0) inquire(unit=second_unit, size=second_size, iostat=io_status)
      if (io_status == 0) then
         if (first_size == second_size .and. first_size >= 0_int64) then
            equal = .true.
            remaining = first_size
            do while (remaining > 0_int64)
               count = int(min(remaining, int(len(first_chunk), int64)))
               read(first_unit, iostat=io_status) first_chunk(:count)
               if (io_status == 0) read(second_unit, iostat=io_status) second_chunk(:count)
               if (io_status /= 0) then
                  equal = .false.
                  exit
               end if
               if (first_chunk(:count) /= second_chunk(:count)) then
                  equal = .false.
                  exit
               end if
               remaining = remaining - int(count, int64)
            end do
         end if
      end if
      close(first_unit)
      close(second_unit)
   end function test_files_identical

   !----------------------------------------------------------------------------
   !  12. The context router conditions the mixture on what the channels returned
   !----------------------------------------------------------------------------
   !> Three properties, in the order they matter:
   !>   (a) with a zero router the engine is exactly the input-only gate, so the
   !>       feature is additive and an older model's behaviour is a special case;
   !>   (b) a non-zero router changes the mixture and the output logits, so the
   !>       new parameters really are read by the forward pass;
   !>   (c) the term depends on the three channel vectors of the token being
   !>       processed, not just on the token: two different contexts from the
   !>       same frozen state produce different router terms.  That is the whole
   !>       point -- "the window just found a repeat, read from it" is a
   !>       statement about context, which a token-only gate cannot make.
   subroutine test_context_router_conditions_mixture
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state, probe
      type(fbc_workspace_t) :: workspace
      integer(int32) :: status, index, channel
      real(real64) :: hidden(32), logits(32), baseline(32), expected(FBC_CHANNEL_COUNT)
      real(real64) :: alpha_zero(FBC_CHANNEL_COUNT), alpha_routed(FBC_CHANNEL_COUNT)
      real(real64), allocatable :: context_a(:, :), context_b(:, :)
      real(real64) :: term_a(FBC_CHANNEL_COUNT), term_b(FBC_CHANNEL_COUNT)

      config = small_config(1000000000000_int64)
      call fbc_model_init(model, config, 29_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      allocate(context_a(config%width, FBC_CHANNEL_COUNT))
      allocate(context_b(config%width, FBC_CHANNEL_COUNT))
      call check(all(model%channel_router == 0.0_real64), &
                 'the context router is initialised to zero, so the gate starts input-only')

      ! Warm the state so all three channels return something measurable.
      do index = 1_int32, 6
         call fbc_step(model, state, workspace, 1_int32 + modulo(index * 5_int32, 23_int32), &
                       hidden, logits, status)
      end do
      call check(sum(abs(workspace%channel_context)) > 0.0_real64, &
                 'the three channel contexts are non-zero, so the router has something to read')

      probe = state
      call fbc_step(model, probe, workspace, 7_int32, hidden, baseline, status)
      alpha_zero = workspace%channel_alpha
      context_a = workspace%channel_context
      ! (a) The effective logits are exactly channel_logit + gate_proj . x.
      do channel = 1_int32, FBC_CHANNEL_COUNT
         expected(channel) = model%channel_logit(channel) + &
            dot_product(model%channel_gate_proj(channel, :), workspace%normalized)
      end do
      call check(maxval(abs(workspace%channel_logit_eff - expected)) < 1.0e-12_real64, &
                 'with a zero router the effective logits are exactly the input-only gate')
      call check(all(alpha_zero > 0.0_real64) .and. &
                 abs(sum(alpha_zero) - 1.0_real64) < 1.0e-12_real64, &
                 'the zero-router mixture is still a probability vector')

      ! (b) The retention row gets a small weight on *every* context component
      ! plus two strong entries, so the check cannot pass because a few
      ! components happened to be equal: any difference between two contexts
      ! must show up in the term.  The local and slot rows stay exactly zero,
      ! which is what the "untouched channel" check below relies on.
      model%channel_router(FBC_CHANNEL_RETENTION, :) = 0.01_real64
      model%channel_router(FBC_CHANNEL_RETENTION, 2_int32) = 0.35_real64
      model%channel_router(FBC_CHANNEL_RETENTION, &
                           (7_int32 - 1_int32) * FBC_CHANNEL_COUNT + FBC_CHANNEL_LOCAL) = -0.30_real64
      probe = state
      call fbc_step(model, probe, workspace, 7_int32, hidden, logits, status)
      alpha_routed = workspace%channel_alpha
      call check(maxval(abs(alpha_routed - alpha_zero)) > 1.0e-6_real64, &
                 'a non-zero router changes the channel mixture')
      call check(maxval(abs(logits - baseline)) > 1.0e-8_real64, &
                 'and the changed mixture reaches the output logits')

      ! (c) The term is a function of the context, not of the token alone: a
      ! different token from the same frozen state returns a different set of
      ! channel vectors, and the router must score them differently.
      probe = state
      call fbc_step(model, probe, workspace, 13_int32, hidden, logits, status)
      context_b = workspace%channel_context
      call check(maxval(abs(context_a - context_b)) > 0.0_real64, &
                 'two tokens produce different channel contexts from the same state')
      term_a = router_term(model, context_a)
      term_b = router_term(model, context_b)
      call check(maxval(abs(term_a - term_b)) > 0.0_real64, &
                 'the router scores two different contexts differently (it is context dependent)')
      call check(maxval(abs(term_a)) > 0.0_real64, 'the router term is non-zero for a non-zero router')
      call check(term_a(FBC_CHANNEL_LOCAL) == 0.0_real64 .and. term_a(FBC_CHANNEL_SLOTS) == 0.0_real64, &
                 'a channel whose router row is zero is left untouched')

      ! (a) again, now that the router has been written: zeroing it must restore
      ! the input-only behaviour bit for bit, which is what makes this change
      ! safe for a checkpoint that has not learned it yet.
      model%channel_router = 0.0_real64
      probe = state
      call fbc_step(model, probe, workspace, 7_int32, hidden, logits, status)
      call check(all(logits == baseline) .and. all(workspace%channel_alpha == alpha_zero), &
                 'zeroing the router restores the input-only mixture exactly')

      deallocate(context_a, context_b)
      call fbc_state_destroy(state)
      call fbc_state_destroy(probe)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_context_router_conditions_mixture

   !----------------------------------------------------------------------------
   !  14. The plan counts what exists, and the optimiser touches it all
   !----------------------------------------------------------------------------
   !> Four counters have to agree with the objects they describe, because every
   !> number this project reports is built from them:
   !>   * `fbc_parameter_count` versus the sum of the parameter array sizes,
   !>   * the plan's state element counters versus the allocated state buffers,
   !>   * `state_bytes_real64` versus eight bytes per element,
   !>   * the optimiser's `parameter_updates` versus the same parameter count.
   !> The last one is the interesting one: it is a running count of the elements
   !> the update rule actually wrote, so a parameter that is updated twice, or
   !> never, or a model array the optimiser was never given, breaks the equality.
   subroutine test_plan_counts_match_reality
      type(fbc_config_t) :: config
      type(fbc_plan_t) :: plan
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_optimizer_t) :: optimizer
      type(fbc_gradient_t) :: gradient
      type(fbc_update_config_t) :: update_config
      integer(int32) :: status
      integer(int64) :: parameter_updates, state_elements, energy_elements, energy_formula

      config = small_config(4096_int64)
      call fbc_plan(config, plan)
      call fbc_model_init(model, config, 101_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_optimizer_init(optimizer, config, status)
      call fbc_gradient_init(gradient, model, status)

      call check(model_parameter_elements(model) == plan%parameter_count, &
                 'the plan counts exactly the parameter elements the model allocates')
      call check(plan%parameter_bytes_real64 == 8_int64 * plan%parameter_count, &
                 'the parameter byte counter is eight bytes per element')
      ! Independently sum the serialized reduced-energy array shapes. The ten
      ! scalar energies cover vector groups and salience; every momentum has
      ! the same shape as its model parameter. This protects the checkpoint
      ! pre-allocation length formula against future optimizer layout drift.
      energy_elements = size(optimizer%energy_token_embedding, kind=int64) + &
         size(optimizer%energy_query, kind=int64) + size(optimizer%energy_key, kind=int64) + &
         size(optimizer%energy_value, kind=int64) + size(optimizer%energy_attention_output, kind=int64) + &
         size(optimizer%energy_ffn_gate, kind=int64) + size(optimizer%energy_ffn_value, kind=int64) + &
         size(optimizer%energy_ffn_output, kind=int64) + size(optimizer%energy_channel_gate_proj, kind=int64) + &
         size(optimizer%energy_channel_router, kind=int64) + size(optimizer%energy_scale_gate_proj, kind=int64) + &
         size(optimizer%energy_slot_query_proj, kind=int64) + size(optimizer%energy_slot_key_proj, kind=int64) + &
         size(optimizer%energy_slot_value_proj, kind=int64) + size(optimizer%energy_slot_read_proj, kind=int64) + &
         10_int64
      energy_formula = int(config%vocabulary_size, int64) + &
                       3_int64 * int(config%width, int64) * int(config%heads, int64) + &
                       12_int64 * int(config%width, int64) + int(config%ffn_width, int64) + 10_int64
      call check(energy_elements == energy_formula, &
                 'checkpoint reduced-energy length matches the optimizer arrays allocated by this layout')

      call check(plan%local_cache_elements == size(state%local_key) + size(state%local_value), &
                 'the local cache counter equals the ring buffer it describes')
      call check(plan%retention_state_elements == &
                 size(state%retention_numerator) + size(state%retention_normalizer), &
                 'the retention counter equals the recurrent memory it describes')
      call check(plan%slot_state_elements == &
                 size(state%slot_key) + size(state%slot_value) + size(state%slot_salience), &
                 'the slot counter equals the slot state it describes')
      state_elements = plan%local_cache_elements + plan%retention_state_elements + &
                       plan%slot_state_elements
      call check(plan%state_bytes_real64 == 8_int64 * state_elements, &
                 'the state byte counter equals eight bytes per state element')

      ! A zero gradient: the update rule still walks every parameter, so the
      ! counter it returns must equal the parameter count exactly.
      call fbc_gradient_zero(gradient)
      update_config = fbc_update_config_t()
      call fbc_optimizer_apply(optimizer, model, gradient, update_config, status, parameter_updates)
      call check(status == 0, 'the optimiser accepts a zero gradient')
      call check(parameter_updates == fbc_parameter_count(config), &
                 'the optimiser writes every parameter exactly once, and no other element')
      call check(maxval(abs(model%token_embedding)) < 1.0e3_real64 .and. &
                 maxval(abs(model%output_bias)) < 1.0e3_real64, &
                 'a zero gradient with no weight decay leaves the weights finite and unmoved')

      call fbc_gradient_destroy(gradient)
      call fbc_optimizer_destroy(optimizer)
      call fbc_state_destroy(state)
      call fbc_model_destroy(model)
   end subroutine test_plan_counts_match_reality

   !> Every trainable array, summed, as an independent check of the count.
   pure function model_parameter_elements(model) result(elements)
      type(fbc_model_t), intent(in) :: model
      integer(int64) :: elements

      elements = size(model%token_embedding, kind=int64) + size(model%output_bias, kind=int64) + &
                 size(model%query, kind=int64) + size(model%key, kind=int64) + &
                 size(model%value, kind=int64) + size(model%attention_output, kind=int64) + &
                 size(model%attention_rms, kind=int64) + size(model%ffn_rms, kind=int64) + &
                 size(model%ffn_gate, kind=int64) + size(model%ffn_value, kind=int64) + &
                 size(model%ffn_output, kind=int64) + size(model%distance_bias, kind=int64) + &
                 size(model%scale_logit, kind=int64) + size(model%channel_logit, kind=int64) + &
                 size(model%decay_logit, kind=int64) + size(model%write_strength_logit, kind=int64) + &
                 size(model%channel_gate_proj, kind=int64) + size(model%scale_gate_proj, kind=int64) + &
                 size(model%channel_router, kind=int64) + size(model%induction_mix, kind=int64) + &
                 size(model%slot_query_proj, kind=int64) + size(model%slot_key_proj, kind=int64) + &
                 size(model%slot_value_proj, kind=int64) + size(model%slot_read_proj, kind=int64) + &
                 1_int64
   end function model_parameter_elements

   !----------------------------------------------------------------------------
   !  15. A reset state is a fresh state
   !----------------------------------------------------------------------------
   !> `reset_every` and every prompt restart rely on this: after `fbc_state_reset`
   !> the model must answer exactly as it would from a state that never carried
   !> anything.  A single array left uncleared here would leak history into every
   !> prompt, which is invisible in a loss curve and fatal for the probes.
   subroutine test_state_reset_is_fresh
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: fresh, dirty
      type(fbc_workspace_t) :: workspace
      integer(int32) :: status, index
      real(real64) :: hidden(32), logits(32), reference(32)

      config = small_config(1000000_int64)
      call fbc_model_init(model, config, 151_int64, status)
      call fbc_state_init(fresh, model, status)
      call fbc_state_init(dirty, model, status)
      call fbc_workspace_init(workspace, model, status)

      call fbc_step(model, fresh, workspace, 7_int32, hidden, reference, status)
      call check(status == FBC_OK, 'the fresh-state reference step runs')
      do index = 1_int32, 40
         call fbc_step(model, dirty, workspace, 1_int32 + modulo(index * 7_int32, 31_int32), &
                       hidden, logits, status)
      end do
      call check(dirty%cache_count == config%local_window .and. dirty%steps == 40_int64, &
                 'the dirty state really filled up before the reset')
      call fbc_state_reset(dirty, model, status)
      call check(status == FBC_OK, 'the state reset succeeds')
      call fbc_step(model, dirty, workspace, 7_int32, hidden, logits, status)
      call check(all(logits == reference), &
                 'after a reset the logits are bit-for-bit those of a fresh state')

      call fbc_state_destroy(fresh)
      call fbc_state_destroy(dirty)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_state_reset_is_fresh

   !----------------------------------------------------------------------------
   !  16. A full ring buffer
   !----------------------------------------------------------------------------
   !> Two properties, both recomputed from the token stream instead of from the
   !> engine's own indices, so an off-by-one in the ring or in the shifted read
   !> fails here:
   !>   (a) every slot holds exactly the key/value of the token at its distance;
   !>   (b) the shifted read is `sum_d w_{d+1} * value(token at distance d - 1)`,
   !>       with distance 0 being the current token.
   subroutine test_full_window_ring
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      integer(int32) :: status, step, head, ordinal, position, distance
      integer(int32), parameter :: length = 11_int32
      integer(int32) :: stream(length)
      real(real64) :: hidden(8), logits(16)
      real(real64), allocatable :: expected_v(:), expected_k(:), expected_read(:)

      config = tiny_config(1024_int64)
      call fbc_model_init(model, config, 157_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      allocate(expected_v(config%width / config%heads), expected_k(config%width / config%heads), &
               expected_read(config%width / config%heads))
      ! 11 steps with a window of 4 wraps the ring more than twice, and the
      ! values repeat so a slot that was never overwritten would still look
      ! plausible if only one distance were checked.
      stream = [3_int32, 5_int32, 7_int32, 9_int32, 11_int32, 13_int32, 3_int32, 5_int32, &
                7_int32, 9_int32, 11_int32]
      do step = 1_int32, length
         call fbc_step(model, state, workspace, stream(step), hidden, logits, status)
      end do
      call check(status == FBC_OK, 'the full-ring stream runs')
      call check(state%cache_count == config%local_window, 'the ring buffer is full')
      call check(state%steps == int(length, int64), 'the step counter matches the tokens consumed')

      do head = 1_int32, config%heads
         ! Distances 1..window-1 were not touched by the final write, so the
         ! positions the last step used must still hold the token at that
         ! distance from the last position.
         do ordinal = 1_int32, config%local_window - 1_int32
            position = workspace%local_position(ordinal)
            expected_v = matmul(model%value(:, :, head), normalized_input(model, stream(length - ordinal)))
            expected_k = matmul(model%key(:, :, head), normalized_input(model, stream(length - ordinal)))
            call check(maxval(abs(state%local_value(:, head, position) - expected_v)) < 1.0e-12_real64, &
                       'each ring slot holds the value of the token at its distance')
            call check(maxval(abs(state%local_key(:, head, position) - expected_k)) < 1.0e-12_real64, &
                       'and the key of that token')
         end do
         ! The slot the final write replaced now holds the current token.
         position = workspace%local_position(config%local_window)
         expected_v = matmul(model%value(:, :, head), normalized_input(model, stream(length)))
         call check(maxval(abs(state%local_value(:, head, position) - expected_v)) < 1.0e-12_real64, &
                    'the wrapped slot was overwritten by the current token')

         ! (b) the shifted read of that same step, recomputed by hand
         expected_read = workspace%local_weight(2_int32, head) * &
                         matmul(model%value(:, :, head), normalized_input(model, stream(length)))
         do distance = 1_int32, config%local_window - 1_int32
            expected_read = expected_read + workspace%local_weight(distance + 2_int32, head) * &
               matmul(model%value(:, :, head), normalized_input(model, stream(length - distance)))
         end do
         call check(maxval(abs(workspace%induction_context(:, head) - expected_read)) < 1.0e-12_real64, &
                    'the shifted read on a full buffer is the successor of every matched position')
      end do

      deallocate(expected_v, expected_k, expected_read)
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_full_window_ring

   !> The model's normalised input for one token, recomputed outside the step.
   pure function normalized_input(model, token) result(vector)
      type(fbc_model_t), intent(in) :: model
      integer(int32), intent(in) :: token
      real(real64) :: vector(model%config%width)
      real(real64) :: x(model%config%width), mean_square, inverse_rms

      x = model%token_embedding(:, token)
      mean_square = dot_product(x, x) / real(model%config%width, real64)
      inverse_rms = 1.0_real64 / sqrt(mean_square + model%config%rms_epsilon)
      vector = (x * inverse_rms) * model%attention_rms
   end function normalized_input

   !----------------------------------------------------------------------------
   !  13. The induction (shifted) read
   !----------------------------------------------------------------------------
   !> A single in-place read can only reproduce a byte that *equals* the matched
   !> one: it reads the value stored at the position it attended to.  Induction
   !> needs the value stored one position *later* -- "after this byte came that
   !> byte" -- and that is what the shifted read adds.
   !>
   !> The stream is `5 7 5 7 5`, so at the last step the token being read is 5,
   !> its earlier occurrences are the token-5 entries, and the byte that followed
   !> each of them is 7.  The projections are arranged so the two reads cannot be
   !> confused: the key sees component 1 (token 5 -> large, token 7 -> 0) and the
   !> value sees component 2 (token 5 -> 0, token 7 -> large).  The in-place read
   !> of the matched positions must therefore be (numerically) zero while the
   !> shifted read is the value of token 7, weighted by exactly the sum of the
   !> weights of the matched positions.
   subroutine test_induction_read_is_shifted
      type(fbc_config_t) :: config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      integer(int32) :: status, head, step
      integer(int32), parameter :: pattern(5) = [5_int32, 7_int32, 5_int32, 7_int32, 5_int32]
      real(real64) :: hidden(8), logits(16)
      real(real64) :: x(8), normalized(8), mean_square, inverse_rms
      real(real64), allocatable :: expected(:), shifted(:), plain(:)
      real(real64) :: matched_weight

      config = tiny_config(1024_int64)
      call fbc_model_init(model, config, 71_int64, status)
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      allocate(expected(config%width / config%heads), shifted(config%width / config%heads), &
               plain(config%width / config%heads))

      call check(all(model%induction_mix == 0.0_real64), &
                 'the induction weights start at zero, so the channel is unchanged at init')

      ! A deliberately readable model: two orthogonal one-hot embeddings, a key
      ! that sees the first component (times four, so the match is sharp), a
      ! value that sees the second, and no distance bias.
      model%token_embedding = 0.0_real64
      model%token_embedding(1_int32, 5_int32) = 1.0_real64
      model%token_embedding(2_int32, 7_int32) = 1.0_real64
      model%output_bias = 0.0_real64
      model%attention_rms = 1.0_real64
      model%distance_bias = 0.0_real64
      model%query = 0.0_real64
      model%key = 0.0_real64
      model%value = 0.0_real64
      do head = 1_int32, config%heads
         model%query(1_int32, 1_int32, head) = 4.0_real64
         model%key(1_int32, 1_int32, head) = 4.0_real64
         model%value(1_int32, 2_int32, head) = 1.0_real64
      end do

      ! At zero weight the local channel is exactly the in-place read, which only
      ! ever sees token-5 values here, i.e. nothing at all.
      do step = 1_int32, size(pattern)
         call fbc_step(model, state, workspace, pattern(step), hidden, logits, status)
      end do
      call check(status == FBC_OK, 'the induction probe stream runs')
      do head = 1_int32, config%heads
         call check(maxval(abs(workspace%local_context(:, head))) < 1.0e-12_real64, &
                    'with a zero weight the local channel is exactly the in-place read')
      end do

      ! Now switch the shifted read on and replay the same stream.
      model%induction_mix = 1.0_real64
      call fbc_state_reset(state, model, status)
      do step = 1_int32, size(pattern)
         call fbc_step(model, state, workspace, pattern(step), hidden, logits, status)
      end do

      ! The value of token 7, computed by hand from the same normalisation.
      x = 0.0_real64
      x(2) = 1.0_real64
      mean_square = dot_product(x, x) / real(config%width, real64)
      inverse_rms = 1.0_real64 / sqrt(mean_square + config%rms_epsilon)
      normalized = x * inverse_rms * model%attention_rms
      do head = 1_int32, config%heads
         expected = matmul(model%value(:, :, head), normalized)
         shifted = workspace%induction_context(:, head)
         plain = workspace%local_context(:, head) - shifted
         call check(maxval(abs(expected)) > 0.0_real64, 'the hand-computed successor value is not degenerate')
         call check(maxval(abs(plain)) < 1.0e-12_real64, &
                    'the in-place read sees only the matched token-5 values (zero here)')
         ! Distances 2 and 4 are the two token-5 occurrences; the byte that
         ! followed each of them is the token 7 at distance 1 and 3.
         matched_weight = workspace%local_weight(3_int32, head) + workspace%local_weight(5_int32, head)
         ! Three positions match maximally: this step's own key and the two
         ! token-5 entries, so the two shifted weights are two thirds.
         call check(abs(matched_weight - 2.0_real64 / 3.0_real64) < 1.0e-9_real64, &
                    'the two token-5 positions carry exactly their share of the attention')
         call check(maxval(abs(shifted - matched_weight * expected)) < 1.0e-9_real64, &
                    'the shifted read is the value that FOLLOWED each matched position')
      end do

      deallocate(expected, shifted, plain)
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_model_destroy(model)
   end subroutine test_induction_read_is_shifted

   !> The router term, computed from the documented column layout by hand, so the
   !> test checks the engine's own mapping rather than calling the engine's
   !> helper (which would agree with itself by construction).
   pure function router_term(model, contexts) result(term)
      type(fbc_model_t), intent(in) :: model
      real(real64), intent(in) :: contexts(:, :)
      real(real64) :: term(FBC_CHANNEL_COUNT)
      integer(int32) :: component, source, column

      term = 0.0_real64
      do source = 1_int32, FBC_CHANNEL_COUNT
         do component = 1_int32, size(contexts, 1, kind=int32)
            column = (component - 1_int32) * FBC_CHANNEL_COUNT + source
            term = term + model%channel_router(:, column) * contexts(component, source)
         end do
      end do
   end function router_term

end program fixed_budget_context_tests
