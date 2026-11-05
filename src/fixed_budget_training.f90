! SPDX-License-Identifier: MIT
!==============================================================================
!  fixed_budget_training.f90
!
!  Bounded-cost training for the fixed-budget context engine: token streams,
!  fixed token chunks, real gradients for every learnable array, and a
!  versioned checkpoint that carries the whole optimiser state.
!
!  Cost contract
!  -------------
!  The activation memory of a chunk is one fbc_workspace_t, not one per token:
!  the backward pass runs immediately after each step, so nothing is retained
!  for the whole chunk.  Chunk length bounds the number of steps between
!  optimiser updates; it never appears in a tensor shape, and it is independent
!  of the logical horizon.
!
!  Gradient contract
!  -----------------
!  Gradients are accumulated over a chunk and the optimiser is applied once per
!  chunk, so `--train-chunk-tokens` plays the role of a batch size.  The
!  recurrent state is carried across chunks (training and inference use the same
!  fbc_step), and no gradient crosses a chunk boundary: the truncation is the
!  depth-1 truncated BPTT documented in fixed_budget_backward.f90.
!
!  Checkpoint contract
!  -------------------
!  Format `VGFB0001` versions 5-7 store the training rule, physical dimensions,
!  logical/effective horizon, clamp flag, parameters, optimiser state and
!  cumulative counters. Versions 2-4 are inspectable but cannot be resumed;
!  legacy `VGNR0001` checkpoints are rejected with a distinct status. The
!  recurrent stream state is not stored and must be reset on resume.
!==============================================================================
module fixed_budget_training
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use fixed_budget_context
   use fixed_budget_backward
   use fixed_budget_optimizer
   implicit none (type, external)
   private

   integer(int32), parameter, public :: FBT_OK = 0_int32
   integer(int32), parameter, public :: FBT_ERR_IO = -1501_int32
   integer(int32), parameter, public :: FBT_ERR_FORMAT = -1502_int32
   integer(int32), parameter, public :: FBT_ERR_LEGACY = -1503_int32
   integer(int32), parameter, public :: FBT_ERR_ARGUMENT = -1504_int32
   integer(int32), parameter, public :: FBT_ERR_NUMERIC = -1505_int32
   !> A checkpoint of a known but older format: readable, not convertible.
   integer(int32), parameter, public :: FBT_ERR_VERSION = -1506_int32

   character(len=8), parameter, public :: FBC_CHECKPOINT_MAGIC = 'VGFB0001'
   character(len=8), parameter, public :: FBC_LEGACY_MAGIC = 'VGNR0001'
   !> Versions 2..5 added parameters; files before v5 are refused rather than
   !> padded. V6 fixes the full-ring/clamped-decay backward rule. V7 retains the
   !> same payload but changes slot-state initialization and writes. V5/V6 stay
   !> readable and are re-saved with their own original semantics.
   integer(int32), parameter, public :: FBC_CHECKPOINT_VERSION = FBC_TRAINING_RULE_V7
   integer(int32), parameter, public :: FBC_CHECKPOINT_MIN_VERSION = 1_int32
   integer(int32), parameter :: FBC_CHECKPOINT_MIN_LOAD_VERSION = FBC_TRAINING_RULE_V5

   ! Synthetic tasks.  They exist because the contract under test is structural
   ! (does the state carry information across a distance the local window cannot
   ! reach?) and needs a known answer position, not a natural-language corpus.
   integer(int32), parameter, public :: FBC_TASK_MARKOV = 1_int32
   integer(int32), parameter, public :: FBC_TASK_NEEDLE = 2_int32

   type, public :: fbc_metrics_t
      integer(int64) :: tokens = 0_int64
      integer(int64) :: updates = 0_int64
      integer(int64) :: parameter_updates = 0_int64
      real(real64) :: loss_sum = 0.0_real64
      real(real64) :: last_gradient_norm = 0.0_real64
      logical :: numeric_ok = .true.
   end type fbc_metrics_t

   type, public :: fbc_checkpoint_meta_t
      integer(int64) :: logical_horizon = 0_int64
      integer(int32) :: local_window = 0_int32
      integer(int32) :: memory_scales = 0_int32
      integer(int32) :: memory_slots = 0_int32
      integer(int32) :: width = 0_int32
      integer(int32) :: heads = 0_int32
      integer(int32) :: ffn_width = 0_int32
      integer(int32) :: vocabulary_size = 0_int32
      integer(int64) :: updates = 0_int64
      integer(int64) :: tokens = 0_int64
      real(real64) :: effective_horizon = 0.0_real64
      logical :: horizon_clamped = .false.
      integer(int32) :: format_version = FBC_CHECKPOINT_VERSION
      ! Header compatibility only, never a payload-integrity guarantee. An
      ! error or unread header is fail-closed rather than convertible by default.
      logical :: convertible = .false.
   end type fbc_checkpoint_meta_t

   public :: fbc_loss_and_gradient, fbc_train_stream, fbc_evaluate_stream
   public :: fbc_generate_tokens, fbc_checkpoint_save, fbc_checkpoint_load, fbc_checkpoint_meta_read
   public :: fbc_parse_task

contains

   !----------------------------------------------------------------------------
   !  Loss
   !----------------------------------------------------------------------------

   ! Local bit check stays inline in the per-token logit reduction; the public
   ! context helper provides the same validation for configuration scalars.
   pure elemental logical function loss_logit_is_finite(value) result(finite)
      real(real64), intent(in) :: value
      integer(int64) :: bits

      if (radix(value) == 2 .and. digits(value) == 53 .and. maxexponent(value) == 1024 .and. &
          minexponent(value) == -1021 .and. storage_size(value) == 64) then
         bits = transfer(value, 0_int64)
         finite = iand(bits, int(z'7FF0000000000000', int64)) /= int(z'7FF0000000000000', int64)
      else
         finite = ieee_is_finite(value)
      end if
   end function loss_logit_is_finite

   !> Cross-entropy of one prediction and dLoss/dlogits = softmax - onehot.
   pure subroutine fbc_loss_and_gradient(logits, target, dlogits, loss, status)
      real(real64), intent(in) :: logits(:)
      integer(int32), intent(in) :: target
      real(real64), intent(out) :: dlogits(:)
      real(real64), intent(out) :: loss
      integer(int32), intent(out) :: status
      real(real64) :: maximum, minimum, normalizer, half_gap, safe_logit
      integer(int64) :: index, bits
      logical :: wide_range, finite_logit, invalid_logit

      status = FBC_OK
      loss = 0.0_real64
      if (size(dlogits, kind=int32) /= size(logits, kind=int32) .or. &
          target < 1_int32 .or. target > size(logits, kind=int32)) then
         dlogits = 0.0_real64
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      ! Combine finiteness, max and min in the existing logit reduction. In
      ! particular, do not perform arithmetic or comparison on signaling NaNs.
      maximum = -huge(0.0_real64)
      minimum = huge(0.0_real64)
      invalid_logit = .false.
      if (radix(maximum) == 2 .and. digits(maximum) == 53 .and. maxexponent(maximum) == 1024 .and. &
          minexponent(maximum) == -1021 .and. storage_size(maximum) == 64) then
         ! Integer masking keeps nonfinite bit patterns out of comparisons while
         ! combining max/min/validation in one pass. Vectorization is compiler-dependent.
         do index = 1_int64, size(logits, kind=int64)
            bits = transfer(logits(index), 0_int64)
            finite_logit = iand(bits, int(z'7FF0000000000000', int64)) /= int(z'7FF0000000000000', int64)
            safe_logit = transfer(merge(bits, 0_int64, finite_logit), 0.0_real64)
            maximum = max(maximum, safe_logit)
            minimum = min(minimum, safe_logit)
            invalid_logit = invalid_logit .or. .not. finite_logit
         end do
      else
         do index = 1_int64, size(logits, kind=int64)
            if (.not. loss_logit_is_finite(logits(index))) then
               invalid_logit = .true.
               exit
            end if
            maximum = max(maximum, logits(index))
            minimum = min(minimum, logits(index))
         end do
      end if
      if (invalid_logit) then
         dlogits = 0.0_real64
         status = FBC_ERR_NUMERIC
         return
      end if

      ! Opposite finite extremes can overflow on logit - maximum even when the
      ! target is the maximum and the mathematically correct loss is zero.
      wide_range = .false.
      if (maximum > 0.0_real64 .and. minimum < 0.0_real64) then
         wide_range = maximum >= huge(0.0_real64) + minimum
      end if
      normalizer = 0.0_real64
      if (wide_range) then
         do index = 1_int64, size(logits, kind=int64)
            if (logits(index) < 0.0_real64) then
               half_gap = 0.5_real64 * maximum - 0.5_real64 * logits(index)
               if (half_gap > 0.5_real64 * huge(0.0_real64)) then
                  dlogits(index) = 0.0_real64
               else
                  dlogits(index) = exp(-2.0_real64 * half_gap)
               end if
            else
               dlogits(index) = exp(logits(index) - maximum)
            end if
            normalizer = normalizer + dlogits(index)
         end do
      else
         do index = 1_int64, size(logits, kind=int64)
            dlogits(index) = exp(logits(index) - maximum)
            normalizer = normalizer + dlogits(index)
         end do
      end if
      if (.not. loss_logit_is_finite(normalizer)) then
         dlogits = 0.0_real64
         status = FBC_ERR_NUMERIC
         return
      end if
      if (normalizer <= 0.0_real64) then
         dlogits = 0.0_real64
         status = FBC_ERR_NUMERIC
         return
      end if
      ! Compute NLL in logit space: -log(softmax(target)) would lose all
      ! information once the target probability underflows to zero. In the
      ! wide-range case, halve before subtracting and reject an unrepresentable
      ! target gap before doubling it (rather than overflowing first).
      if (wide_range .and. logits(target) < 0.0_real64) then
         half_gap = 0.5_real64 * maximum - 0.5_real64 * logits(target)
         if (half_gap > 0.5_real64 * huge(0.0_real64)) then
            dlogits = 0.0_real64
            status = FBC_ERR_NUMERIC
            return
         end if
         loss = 2.0_real64 * half_gap + log(normalizer)
      else
         loss = (maximum - logits(target)) + log(normalizer)
      end if
      dlogits = dlogits / normalizer
      dlogits(target) = dlogits(target) - 1.0_real64
   end subroutine fbc_loss_and_gradient

   ! The training work arrays are allocated contiguously. Give the compiler an
   ! explicit-shape view of those same arrays, without copying them or imposing
   ! CONTIGUOUS on the public loss routine (which also accepts strided callers).
   ! This preserves the contiguous loss kernel even with checked allocations.
   pure subroutine fbc_training_loss(vocabulary_size, logits, target, dlogits, loss, status)
      integer(int32), intent(in) :: vocabulary_size, target
      real(real64), intent(in) :: logits(vocabulary_size)
      real(real64), intent(out) :: dlogits(vocabulary_size), loss
      integer(int32), intent(out) :: status

      call fbc_loss_and_gradient(logits, target, dlogits, loss, status)
   end subroutine fbc_training_loss

   !----------------------------------------------------------------------------
   !  Synthetic token streams
   !----------------------------------------------------------------------------

   pure subroutine fbc_next_random(state, value)
      integer(int64), intent(inout) :: state
      integer(int64), intent(out) :: value

      state = ieor(state, ishftc(state, 13))
      state = ieor(state, ishftc(state, 7))
      state = ieor(state, ishftc(state, 17))
      value = iand(state, huge(0_int64))
   end subroutine fbc_next_random

   !> Builds a deterministic token stream plus a mask marking the positions whose
   !> target is a *recalled answer* (the only positions where the needle task can
   !> show remote recall).
   !>
   !> `needle` blocks look like
   !>     KEY ANSWER filler x distance QUERY ANSWER(next block start)
   !> so the answer the model must produce sits `distance + 1` tokens in the past,
   !> far beyond any physical local window used by the tests.
   subroutine fbc_generate_tokens(task, token_count, vocabulary_size, distance, seed, tokens, answer_mask, status)
      integer(int32), intent(in) :: task, token_count, vocabulary_size, distance
      integer(int64), intent(in) :: seed
      integer(int32), intent(out) :: tokens(:)
      logical, intent(out) :: answer_mask(:)
      integer(int32), intent(out) :: status
      integer(int32) :: answer_count, answer_first, filler_first, answer
      integer(int64) :: random_state, draw, position, offset, block_stride, filler_limit

      status = FBC_OK
      tokens = 1_int32
      answer_mask = .false.
      if (size(tokens, kind=int32) /= token_count .or. size(answer_mask, kind=int32) /= token_count) then
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      random_state = seed
      if (random_state == 0_int64) random_state = int(z'243F6A8885A308D3', int64)
      select case (task)
      case (FBC_TASK_MARKOV)
         if (vocabulary_size < 8_int32) then
            status = FBC_ERR_INVALID_ARGUMENT
            return
         end if
         do position = 1_int64, int(token_count, int64)
            call fbc_next_random(random_state, draw)
            tokens(position) = 4_int32 + int(mod(draw, int(vocabulary_size - 3_int32, int64)), int32)
         end do
         ! The learnable rule is "next = previous + 1 (wrapping)", which is a
         ! short-range task: it measures optimisation, not memory.
         do position = 2_int64, int(token_count, int64)
            tokens(position) = 4_int32 + modulo(tokens(position - 1_int32) - 4_int32 + 1_int32, &
                                                vocabulary_size - 3_int32)
         end do
      case (FBC_TASK_NEEDLE)
         if (vocabulary_size < 16_int32 .or. distance < 1_int32) then
            status = FBC_ERR_INVALID_ARGUMENT
            return
         end if
         answer_first = 3_int32
         answer_count = (vocabulary_size - 4_int32) / 2_int32
         filler_first = answer_first + answer_count
         ! Use int64 for index arithmetic even when distance is near int32's
         ! limit, and never iterate over filler positions beyond this stream.
         block_stride = int(distance, int64) + 4_int64
         position = 1_int64
         do while (position <= int(token_count, int64))
            call fbc_next_random(random_state, draw)
            answer = answer_first + int(mod(draw, int(answer_count, int64)), int32)
            tokens(position) = 1_int32                                   ! KEY
            if (position + 1_int64 <= int(token_count, int64)) tokens(position + 1_int64) = answer
            filler_limit = min(int(distance, int64) + 1_int64, int(token_count, int64) - position)
            do offset = 2_int64, filler_limit
               call fbc_next_random(random_state, draw)
               tokens(position + offset) = filler_first + int(mod(draw, &
                  int(max(1_int32, vocabulary_size - filler_first + 1_int32), int64)), int32)
            end do
            if (position + int(distance, int64) + 2_int64 <= int(token_count, int64)) then
               tokens(position + int(distance, int64) + 2_int64) = 2_int32 ! QUERY
               ! The target of the QUERY position is the answer that follows it,
               ! so the mask marks the position *after* the query.
               if (position + int(distance, int64) + 3_int64 <= int(token_count, int64)) then
                  tokens(position + int(distance, int64) + 3_int64) = answer
                  answer_mask(position + int(distance, int64) + 2_int64) = .true.
               end if
            end if
            position = position + block_stride
         end do
      case default
         status = FBC_ERR_INVALID_ARGUMENT
      end select
   end subroutine fbc_generate_tokens

   !> Task name to id, for the CLI and its tests.
   pure subroutine fbc_parse_task(text, task, status)
      character(len=*), intent(in) :: text
      integer(int32), intent(out) :: task
      integer(int32), intent(out) :: status

      task = 0_int32
      status = FBC_OK
      select case (trim(text))
      case ('markov')
         task = FBC_TASK_MARKOV
      case ('needle')
         task = FBC_TASK_NEEDLE
      case default
         status = FBT_ERR_ARGUMENT
      end select
   end subroutine fbc_parse_task

   !----------------------------------------------------------------------------
   !  Training
   !----------------------------------------------------------------------------

   !> Trains over `token_count` tokens in fixed chunks.
   !>
   !> One optimiser update per chunk of at most `chunk_tokens` predictions.
   !> Non-divisible budgets are rejected before work by default: automatically
   !> training a short tail would add an update relative to the old path, which
   !> silently discarded it. Pass `allow_partial_chunk=.true.` explicitly to
   !> train the tail and accept its extra work. Chunk size bounds work between
   !> updates and is not a tensor dimension or derived from the horizon.
   subroutine fbc_train_stream(model, optimizer, state, tokens, token_count, chunk_tokens, &
                               update_config, workspace, gradient, metrics, status, allow_partial_chunk)
      type(fbc_model_t), intent(inout) :: model
      type(fbc_optimizer_t), intent(inout) :: optimizer
      type(fbc_state_t), intent(inout) :: state
      integer(int32), intent(in) :: tokens(:), token_count, chunk_tokens
      type(fbc_update_config_t), intent(in) :: update_config
      type(fbc_workspace_t), intent(inout) :: workspace
      type(fbc_gradient_t), intent(inout) :: gradient
      type(fbc_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      logical, intent(in), optional :: allow_partial_chunk
      integer(int32) :: position, offset, target, step_status, finish_status, current_chunk
      integer :: allocation_status
      logical :: permit_partial, defer_local
      integer(int64) :: parameter_updates, planned_updates, planned_parameters
      real(real64) :: loss, norm, scale, effective_norm
      real(real64), allocatable :: logits(:), hidden(:), dlogits(:)

      status = FBT_OK
      if (.not. fbc_training_inputs_are_ready(model, state, workspace, gradient, optimizer)) then
         status = FBC_ERR_NOT_READY
         return
      end if
      if (.not. fbc_update_config_is_valid(update_config)) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      if (model%training_rule /= FBC_TRAINING_RULE_V5 .and. model%training_rule /= FBC_TRAINING_RULE_V6 .and. &
          model%training_rule /= FBC_TRAINING_RULE_V7) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      if (chunk_tokens < 1_int32 .or. chunk_tokens > FBC_MAX_CHUNK_TOKENS) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      ! Use a wide size query and reject the single int32 count that would
      ! overflow the last position + 1 and the post-loop position increment.
      if (token_count < 0_int32 .or. token_count == huge(token_count) .or. &
          int(token_count, int64) >= size(tokens, kind=int64)) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      if (token_count == 0_int32) return
      permit_partial = .false.
      if (present(allow_partial_chunk)) permit_partial = allow_partial_chunk
      if (modulo(token_count, chunk_tokens) /= 0_int32 .and. .not. permit_partial) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      ! Check accumulated counters once per call, not by scanning parameters or
      ! by adding per-token checks. No update may wrap a reporting counter.
      planned_updates = int(token_count / chunk_tokens, int64)
      if (modulo(token_count, chunk_tokens) /= 0_int32) planned_updates = planned_updates + 1_int64
      planned_parameters = fbc_parameter_count(model%config)
      if (planned_parameters <= 0_int64 .or. metrics%tokens < 0_int64 .or. metrics%updates < 0_int64 .or. &
          metrics%parameter_updates < 0_int64) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      if (metrics%tokens > huge(0_int64) - int(token_count, int64) .or. &
          metrics%updates > huge(0_int64) - planned_updates) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      if (planned_updates > (huge(0_int64) - metrics%parameter_updates) / planned_parameters) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      if (.not. fbc_real64_is_finite(metrics%loss_sum)) then
         metrics%numeric_ok = .false.
         status = FBT_ERR_NUMERIC
         return
      end if
      if (metrics%loss_sum < 0.0_real64 .or. .not. metrics%numeric_ok) then
         status = FBT_ERR_NUMERIC
         metrics%numeric_ok = .false.
         return
      end if
      allocate(logits(model%config%vocabulary_size), hidden(model%config%width), &
               dlogits(model%config%vocabulary_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = FBC_ERR_ALLOCATION
         return
      end if
      position = 1_int32
      do while (position <= token_count)
         current_chunk = min(chunk_tokens, token_count - position + 1_int32)
         call fbc_gradient_zero(gradient)
         do offset = 0_int32, current_chunk - 1_int32
            target = tokens(position + offset + 1_int32)
            ! Only a full ring loses an activation on commit. Delay that write
            ! for V6/V7 until after backward. V5 retains its historical path.
            defer_local = model%training_rule /= FBC_TRAINING_RULE_V5 .and. &
                          state%cache_count == model%config%local_window
            call fbc_step(model, state, workspace, tokens(position + offset), hidden, logits, step_status, &
                          defer_local_commit=defer_local)
            if (step_status /= FBC_OK) then
               if (step_status == FBC_ERR_NUMERIC) metrics%numeric_ok = .false.
               status = step_status
               deallocate(logits, hidden, dlogits)
               return
            end if
            call fbc_training_loss(model%config%vocabulary_size, logits, target, dlogits, loss, step_status)
            if (step_status == FBC_OK) then
               ! Both terms are nonnegative. Reject an unrepresentable running
               ! sum before it can overflow, and before backward or an update.
               if (loss > huge(0.0_real64) - metrics%loss_sum) then
                  step_status = FBT_ERR_NUMERIC
               else
                  call fbc_step_backward(model, state, workspace, dlogits, gradient, step_status)
               end if
            end if
            if (defer_local) then
               ! Complete the state even if loss/backward rejected the token;
               ! the old path had already committed the forward step on error.
               call fbc_finish_local_step(state, workspace, finish_status)
               if (step_status == FBC_OK) step_status = finish_status
            end if
            if (step_status /= FBC_OK) then
               if (step_status == FBC_ERR_NUMERIC .or. step_status == FBT_ERR_NUMERIC) metrics%numeric_ok = .false.
               status = step_status
               deallocate(logits, hidden, dlogits)
               return
            end if
            metrics%loss_sum = metrics%loss_sum + loss
            metrics%tokens = metrics%tokens + 1_int64
         end do
         ! Mean gradient of the chunk, then the optional global clip.  The clip
         ! is a single scalar rescale, so it cannot reorder the reductions.
         scale = 1.0_real64 / real(current_chunk, real64)
         call fbc_scale_gradient(gradient, scale)
         norm = fbc_gradient_norm(gradient)
         if (norm < 0.0_real64) then
            metrics%numeric_ok = .false.
            status = FBT_ERR_NUMERIC
            deallocate(logits, hidden, dlogits)
            return
         end if
         metrics%last_gradient_norm = norm
         effective_norm = norm
         if (update_config%gradient_clip > 0.0_real64 .and. norm > update_config%gradient_clip) then
            call fbc_scale_gradient(gradient, update_config%gradient_clip / norm)
            effective_norm = update_config%gradient_clip
         end if
         call fbc_optimizer_apply(optimizer, model, gradient, update_config, step_status, parameter_updates, &
                                  validated_gradient_norm=effective_norm)
         if (step_status /= 0_int32) then
            if (step_status == FBC_ERR_NUMERIC) then
               metrics%numeric_ok = .false.
               status = FBT_ERR_NUMERIC
            else
               status = FBT_ERR_ARGUMENT
            end if
            deallocate(logits, hidden, dlogits)
            return
         end if
         ! The cached dynamics are a pure function of the logits, so refresh them
         ! once per optimiser update: forward and backward keep reading arrays
         ! and the token loop evaluates no transcendental for them.
         call fbc_refresh_dynamics(model)
         metrics%updates = metrics%updates + 1_int64
         metrics%parameter_updates = metrics%parameter_updates + parameter_updates
         position = position + current_chunk
      end do
      deallocate(logits, hidden, dlogits)
   end subroutine fbc_train_stream

   !> In-place rescale of every gradient array.  One pass, no temporary.
   subroutine fbc_scale_gradient(gradient, scale)
      type(fbc_gradient_t), intent(inout) :: gradient
      real(real64), intent(in) :: scale

      gradient%token_embedding = gradient%token_embedding * scale
      gradient%output_bias = gradient%output_bias * scale
      gradient%query = gradient%query * scale
      gradient%key = gradient%key * scale
      gradient%value = gradient%value * scale
      gradient%attention_output = gradient%attention_output * scale
      gradient%attention_rms = gradient%attention_rms * scale
      gradient%ffn_rms = gradient%ffn_rms * scale
      gradient%ffn_gate = gradient%ffn_gate * scale
      gradient%ffn_value = gradient%ffn_value * scale
      gradient%ffn_output = gradient%ffn_output * scale
      gradient%distance_bias = gradient%distance_bias * scale
      gradient%scale_logit = gradient%scale_logit * scale
      gradient%decay_logit = gradient%decay_logit * scale
      gradient%write_strength_logit = gradient%write_strength_logit * scale
      gradient%channel_logit = gradient%channel_logit * scale
      gradient%channel_gate_proj = gradient%channel_gate_proj * scale
      gradient%channel_router = gradient%channel_router * scale
      gradient%induction_mix = gradient%induction_mix * scale
      gradient%scale_gate_proj = gradient%scale_gate_proj * scale
      gradient%slot_query_proj = gradient%slot_query_proj * scale
      gradient%slot_key_proj = gradient%slot_key_proj * scale
      gradient%slot_value_proj = gradient%slot_value_proj * scale
      gradient%slot_read_proj = gradient%slot_read_proj * scale
      gradient%slot_salience_weight = gradient%slot_salience_weight * scale
   end subroutine fbc_scale_gradient

   !----------------------------------------------------------------------------
   !  Evaluation
   !----------------------------------------------------------------------------

   !> Forward-only pass.  Returns mean NLL over the scored tokens and the
   !> accuracy on the masked (recall) positions.  The state is advanced exactly
   !> as in training: the same fbc_step, so streaming evaluation and training
   !> cannot drift apart.
   subroutine fbc_evaluate_stream(model, state, tokens, token_count, answer_mask, workspace, &
                                  mean_nll, answer_accuracy, answer_count, status)
      type(fbc_model_t), intent(in) :: model
      type(fbc_state_t), intent(inout) :: state
      integer(int32), intent(in) :: tokens(:), token_count
      logical, intent(in) :: answer_mask(:)
      type(fbc_workspace_t), intent(inout) :: workspace
      real(real64), intent(out) :: mean_nll, answer_accuracy
      integer(int64), intent(out) :: answer_count
      integer(int32), intent(out) :: status
      integer(int32) :: position, target, step_status, best, index
      integer :: allocation_status
      real(real64) :: loss, total, best_score
      integer(int64) :: correct
      real(real64), allocatable :: logits(:), hidden(:), dlogits(:)

      status = FBT_OK
      mean_nll = 0.0_real64
      answer_accuracy = 0.0_real64
      answer_count = 0_int64
      correct = 0_int64
      total = 0.0_real64
      ! Reject bad stream and mask lengths before advancing the state. Use a
      ! wide size query and avoid the final int32 position increment overflowing.
      if (token_count < 0_int32 .or. token_count == huge(token_count) .or. &
          int(token_count, int64) >= size(tokens, kind=int64) .or. &
          int(token_count, int64) > size(answer_mask, kind=int64)) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      allocate(logits(model%config%vocabulary_size), hidden(model%config%width), &
               dlogits(model%config%vocabulary_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = FBC_ERR_ALLOCATION
         return
      end if
      do position = 1_int32, token_count
         target = tokens(position + 1_int32)
         call fbc_step(model, state, workspace, tokens(position), hidden, logits, step_status)
         if (step_status /= FBC_OK) then
            status = step_status
            deallocate(logits, hidden, dlogits)
            return
         end if
         call fbc_loss_and_gradient(logits, target, dlogits, loss, step_status)
         if (step_status /= FBC_OK) then
            status = step_status
            deallocate(logits, hidden, dlogits)
            return
         end if
         if (loss > huge(0.0_real64) - total) then
            status = FBT_ERR_NUMERIC
            deallocate(logits, hidden, dlogits)
            return
         end if
         total = total + loss
         if (answer_mask(position)) then
            best = 1_int32
            best_score = logits(1_int32)
            do index = 2_int32, model%config%vocabulary_size
               if (logits(index) > best_score) then
                  best_score = logits(index)
                  best = index
               end if
            end do
            answer_count = answer_count + 1_int64
            if (best == target) correct = correct + 1_int64
         end if
      end do
      mean_nll = total / real(max(1_int32, token_count), real64)
      if (answer_count > 0_int64) answer_accuracy = real(correct, real64) / real(answer_count, real64)
      deallocate(logits, hidden, dlogits)
   end subroutine fbc_evaluate_stream

   !----------------------------------------------------------------------------
   !  Checkpoint
   !----------------------------------------------------------------------------

   subroutine fbc_checkpoint_save(path, model, optimizer, update_config, state_counters, status)
      character(len=*), intent(in) :: path
      type(fbc_model_t), intent(in) :: model
      type(fbc_optimizer_t), intent(in) :: optimizer
      type(fbc_update_config_t), intent(in) :: update_config
      type(fbc_metrics_t), intent(in) :: state_counters
      integer(int32), intent(out) :: status
      integer(int32) :: unit, io_status, write_status
      logical :: ok

      status = FBT_OK
      if (.not. fbc_model_is_ready(model) .or. .not. fbc_config_is_valid(model%config) .or. &
          .not. optimizer%initialized .or. len_trim(path) == 0 .or. .not. fbc_update_config_is_valid(update_config)) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      if (.not. fbc_layout_matches(optimizer%layout, model%config)) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      if (.not. fbc_real64_is_finite(model%effective_horizon)) then
         status = FBT_ERR_NUMERIC
         return
      end if
      if (model%effective_horizon <= 0.0_real64) then
         status = FBT_ERR_NUMERIC
         return
      end if
      if (model%training_rule /= FBC_TRAINING_RULE_V5 .and. model%training_rule /= FBC_TRAINING_RULE_V6 .and. &
          model%training_rule /= FBC_TRAINING_RULE_V7) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      ! Validate scalar metadata before opening with REPLACE. This protects an
      ! existing checkpoint from a known-invalid save without any extra copy.
      if (state_counters%updates < 0_int64 .or. state_counters%tokens < 0_int64 .or. &
          state_counters%parameter_updates < 0_int64) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      if (.not. fbc_real64_is_finite(state_counters%loss_sum) .or. .not. state_counters%numeric_ok) then
         status = FBT_ERR_NUMERIC
         return
      end if
      if (state_counters%loss_sum < 0.0_real64) then
         status = FBT_ERR_NUMERIC
         return
      end if
      open(newunit=unit, file=trim(path), status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = FBT_ERR_IO
         return
      end if
      write(unit, iostat=io_status) FBC_CHECKPOINT_MAGIC
      if (io_status == 0) write(unit, iostat=io_status) model%training_rule
      if (io_status == 0) write(unit, iostat=io_status) model%config%vocabulary_size, model%config%width, &
                                                        model%config%heads, model%config%ffn_width, &
                                                        model%config%local_window, model%config%memory_scales, &
                                                        model%config%memory_slots, model%config%logical_horizon, &
                                                        model%config%rms_epsilon, model%config%slot_decay, &
                                                        model%config%slot_temperature
      if (io_status == 0) write(unit, iostat=io_status) model%effective_horizon
      if (io_status == 0) then
         if (model%horizon_clamped) then
            write(unit, iostat=io_status) 1_int32
         else
            write(unit, iostat=io_status) 0_int32
         end if
      end if
      if (io_status == 0) write(unit, iostat=io_status) state_counters%updates, state_counters%tokens, &
                                                        state_counters%loss_sum
      if (io_status == 0) write(unit, iostat=io_status) update_config%learning_rate, update_config%momentum, &
                                                        update_config%energy_decay, update_config%restart_damp, &
                                                        update_config%weight_decay, update_config%epsilon, &
                                                        update_config%gradient_clip
      call write_payload(unit, model, io_status)
      call write_optimizer(unit, optimizer, io_status)
      write_status = io_status
      close(unit, iostat=io_status)
      ok = write_status == 0 .and. io_status == 0
      if (.not. ok) status = FBT_ERR_IO
   end subroutine fbc_checkpoint_save

   subroutine write_payload(unit, model, io_status)
      integer, intent(in) :: unit
      type(fbc_model_t), intent(in) :: model
      integer, intent(inout) :: io_status

      if (io_status == 0) write(unit, iostat=io_status) model%token_embedding
      if (io_status == 0) write(unit, iostat=io_status) model%output_bias
      if (io_status == 0) write(unit, iostat=io_status) model%query
      if (io_status == 0) write(unit, iostat=io_status) model%key
      if (io_status == 0) write(unit, iostat=io_status) model%value
      if (io_status == 0) write(unit, iostat=io_status) model%attention_output
      if (io_status == 0) write(unit, iostat=io_status) model%attention_rms
      if (io_status == 0) write(unit, iostat=io_status) model%ffn_rms
      if (io_status == 0) write(unit, iostat=io_status) model%ffn_gate
      if (io_status == 0) write(unit, iostat=io_status) model%ffn_value
      if (io_status == 0) write(unit, iostat=io_status) model%ffn_output
      if (io_status == 0) write(unit, iostat=io_status) model%distance_bias
      if (io_status == 0) write(unit, iostat=io_status) model%scale_logit
      if (io_status == 0) write(unit, iostat=io_status) model%decay_logit
      if (io_status == 0) write(unit, iostat=io_status) model%write_strength_logit
      if (io_status == 0) write(unit, iostat=io_status) model%channel_logit
      if (io_status == 0) write(unit, iostat=io_status) model%slot_query_proj
      if (io_status == 0) write(unit, iostat=io_status) model%slot_key_proj
      if (io_status == 0) write(unit, iostat=io_status) model%slot_value_proj
      if (io_status == 0) write(unit, iostat=io_status) model%slot_read_proj
      if (io_status == 0) write(unit, iostat=io_status) model%slot_salience_weight
      if (io_status == 0) write(unit, iostat=io_status) model%channel_gate_proj
      if (io_status == 0) write(unit, iostat=io_status) model%channel_router
      if (io_status == 0) write(unit, iostat=io_status) model%induction_mix
      if (io_status == 0) write(unit, iostat=io_status) model%scale_gate_proj
   end subroutine write_payload

   subroutine read_payload(unit, model, io_status)
      integer, intent(in) :: unit
      type(fbc_model_t), intent(inout) :: model
      integer, intent(inout) :: io_status

      if (io_status == 0) read(unit, iostat=io_status) model%token_embedding
      if (io_status == 0) read(unit, iostat=io_status) model%output_bias
      if (io_status == 0) read(unit, iostat=io_status) model%query
      if (io_status == 0) read(unit, iostat=io_status) model%key
      if (io_status == 0) read(unit, iostat=io_status) model%value
      if (io_status == 0) read(unit, iostat=io_status) model%attention_output
      if (io_status == 0) read(unit, iostat=io_status) model%attention_rms
      if (io_status == 0) read(unit, iostat=io_status) model%ffn_rms
      if (io_status == 0) read(unit, iostat=io_status) model%ffn_gate
      if (io_status == 0) read(unit, iostat=io_status) model%ffn_value
      if (io_status == 0) read(unit, iostat=io_status) model%ffn_output
      if (io_status == 0) read(unit, iostat=io_status) model%distance_bias
      if (io_status == 0) read(unit, iostat=io_status) model%scale_logit
      if (io_status == 0) read(unit, iostat=io_status) model%decay_logit
      if (io_status == 0) read(unit, iostat=io_status) model%write_strength_logit
      if (io_status == 0) read(unit, iostat=io_status) model%channel_logit
      if (io_status == 0) read(unit, iostat=io_status) model%slot_query_proj
      if (io_status == 0) read(unit, iostat=io_status) model%slot_key_proj
      if (io_status == 0) read(unit, iostat=io_status) model%slot_value_proj
      if (io_status == 0) read(unit, iostat=io_status) model%slot_read_proj
      if (io_status == 0) read(unit, iostat=io_status) model%slot_salience_weight
      if (io_status == 0) read(unit, iostat=io_status) model%channel_gate_proj
      if (io_status == 0) read(unit, iostat=io_status) model%channel_router
      if (io_status == 0) read(unit, iostat=io_status) model%induction_mix
      if (io_status == 0) read(unit, iostat=io_status) model%scale_gate_proj
   end subroutine read_payload

   subroutine write_optimizer(unit, optimizer, io_status)
      integer, intent(in) :: unit
      type(fbc_optimizer_t), intent(in) :: optimizer
      integer, intent(inout) :: io_status

      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_token_embedding, &
                                                        optimizer%energy_token_embedding
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_output_bias, optimizer%energy_output_bias
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_query, optimizer%energy_query
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_key, optimizer%energy_key
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_value, optimizer%energy_value
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_attention_output, &
                                                        optimizer%energy_attention_output
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_attention_rms, &
                                                        optimizer%energy_attention_rms
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_ffn_rms, optimizer%energy_ffn_rms
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_ffn_gate, optimizer%energy_ffn_gate
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_ffn_value, optimizer%energy_ffn_value
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_ffn_output, optimizer%energy_ffn_output
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_distance_bias, &
                                                        optimizer%energy_distance_bias
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_scale_logit, optimizer%energy_scale_logit
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_decay_logit, &
                                                        optimizer%energy_decay_logit
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_write_strength_logit, &
                                                        optimizer%energy_write_strength_logit
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_channel_logit, &
                                                        optimizer%energy_channel_logit
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_slot_query_proj, &
                                                        optimizer%energy_slot_query_proj
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_slot_key_proj, &
                                                        optimizer%energy_slot_key_proj
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_slot_value_proj, &
                                                        optimizer%energy_slot_value_proj
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_slot_read_proj, &
                                                        optimizer%energy_slot_read_proj
      if (io_status == 0) write(unit, iostat=io_status) optimizer%slot_salience_weight_velocity, &
                                                        optimizer%slot_salience_weight_energy
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_channel_gate_proj, &
                                                        optimizer%energy_channel_gate_proj
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_channel_router, &
                                                        optimizer%energy_channel_router
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_induction_mix, &
                                                        optimizer%energy_induction_mix
      if (io_status == 0) write(unit, iostat=io_status) optimizer%velocity_scale_gate_proj, &
                                                        optimizer%energy_scale_gate_proj
   end subroutine write_optimizer

   subroutine read_optimizer(unit, optimizer, io_status)
      integer, intent(in) :: unit
      type(fbc_optimizer_t), intent(inout) :: optimizer
      integer, intent(inout) :: io_status

      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_token_embedding, &
                                                       optimizer%energy_token_embedding
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_output_bias, optimizer%energy_output_bias
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_query, optimizer%energy_query
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_key, optimizer%energy_key
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_value, optimizer%energy_value
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_attention_output, &
                                                       optimizer%energy_attention_output
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_attention_rms, &
                                                       optimizer%energy_attention_rms
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_ffn_rms, optimizer%energy_ffn_rms
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_ffn_gate, optimizer%energy_ffn_gate
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_ffn_value, optimizer%energy_ffn_value
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_ffn_output, optimizer%energy_ffn_output
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_distance_bias, &
                                                       optimizer%energy_distance_bias
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_scale_logit, optimizer%energy_scale_logit
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_decay_logit, &
                                                       optimizer%energy_decay_logit
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_write_strength_logit, &
                                                       optimizer%energy_write_strength_logit
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_channel_logit, &
                                                       optimizer%energy_channel_logit
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_slot_query_proj, &
                                                       optimizer%energy_slot_query_proj
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_slot_key_proj, &
                                                       optimizer%energy_slot_key_proj
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_slot_value_proj, &
                                                       optimizer%energy_slot_value_proj
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_slot_read_proj, &
                                                       optimizer%energy_slot_read_proj
      if (io_status == 0) read(unit, iostat=io_status) optimizer%slot_salience_weight_velocity, &
                                                      optimizer%slot_salience_weight_energy
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_channel_gate_proj, &
                                                      optimizer%energy_channel_gate_proj
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_channel_router, &
                                                      optimizer%energy_channel_router
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_induction_mix, &
                                                      optimizer%energy_induction_mix
      if (io_status == 0) read(unit, iostat=io_status) optimizer%velocity_scale_gate_proj, &
                                                      optimizer%energy_scale_gate_proj
   end subroutine read_optimizer

   !> Loads a checkpoint into a fresh model/optimizer pair.  Returns
   !> FBT_ERR_LEGACY for a `VGNR0001` file so the caller can say exactly what to
   !> do instead of failing with a generic format error.
   subroutine fbc_checkpoint_load(path, model, optimizer, update_config, counters, status)
      character(len=*), intent(in) :: path
      type(fbc_model_t), intent(inout) :: model
      type(fbc_optimizer_t), intent(inout) :: optimizer
      type(fbc_update_config_t), intent(out) :: update_config
      type(fbc_metrics_t), intent(out) :: counters
      integer(int32), intent(out) :: status
      integer(int32) :: unit, io_status, version, clamped_flag, allocation_status
      integer :: real_storage_units
      integer(int64) :: file_units, path_units, payload_position, parameter_count, energy_count, stored_reals
      integer(int64) :: payload_units, header_units
      character(len=8) :: magic
      type(fbc_config_t) :: config
      real(real64) :: effective_horizon

      version = 0_int32
      clamped_flag = 0_int32
      io_status = 0
      effective_horizon = 0.0_real64

      call fbc_model_destroy(model)
      call fbc_optimizer_destroy(optimizer)
      counters = fbc_metrics_t()
      update_config = fbc_update_config_t()
      status = FBT_OK
      if (len_trim(path) == 0) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      ! Reject unknown-length and empty sources before OPEN, which avoids
      ! blocking on a zero-sized FIFO on processors that report its size as 0.
      ! Standard Fortran cannot identify every special file or eliminate a
      ! concurrent path-replacement race.
      path_units = -1_int64
      inquire(file=trim(path), size=path_units, iostat=io_status)
      if (io_status /= 0 .or. path_units < 0_int64) then
         status = FBT_ERR_IO
         return
      end if
      if (path_units == 0_int64) then
         status = FBT_ERR_FORMAT
         return
      end if
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = FBT_ERR_IO
         return
      end if
      read(unit, iostat=io_status) magic
      if (io_status /= 0) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      if (magic == FBC_LEGACY_MAGIC) then
         close(unit)
         status = FBT_ERR_LEGACY
         return
      end if
      if (magic /= FBC_CHECKPOINT_MAGIC) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      read(unit, iostat=io_status) version
      if (io_status /= 0) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      if (version < FBC_CHECKPOINT_MIN_VERSION .or. version > FBC_CHECKPOINT_VERSION) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      if (version < FBC_CHECKPOINT_MIN_LOAD_VERSION) then
         ! Pre-v5 files lack learned arrays and cannot describe this model.
         close(unit)
         status = FBT_ERR_VERSION
         return
      end if
      read(unit, iostat=io_status) config%vocabulary_size, config%width, config%heads, config%ffn_width, &
                                   config%local_window, config%memory_scales, config%memory_slots, &
                                   config%logical_horizon, config%rms_epsilon, config%slot_decay, &
                                   config%slot_temperature
      if (io_status == 0) read(unit, iostat=io_status) effective_horizon
      if (io_status == 0) read(unit, iostat=io_status) clamped_flag
      if (io_status == 0) read(unit, iostat=io_status) counters%updates, counters%tokens, counters%loss_sum
      if (io_status == 0) read(unit, iostat=io_status) update_config%learning_rate, update_config%momentum, &
                                                       update_config%energy_decay, update_config%restart_damp, &
                                                       update_config%weight_decay, update_config%epsilon, &
                                                       update_config%gradient_clip
      if (io_status /= 0) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      if (.not. fbc_config_is_valid(config) .or. .not. fbc_update_config_is_valid(update_config) .or. &
          counters%updates < 0_int64 .or. counters%tokens < 0_int64 .or. &
          (clamped_flag /= 0_int32 .and. clamped_flag /= 1_int32)) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      if (.not. fbc_real64_is_finite(counters%loss_sum) .or. .not. fbc_real64_is_finite(effective_horizon)) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      if (counters%loss_sum < 0.0_real64 .or. effective_horizon <= 0.0_real64) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      ! V5-V7 store each parameter, a same-shaped momentum, and reduced
      ! per-column/per-vector optimizer energies. Reject a truncated file
      ! before allocating according to an untrusted header. IOLENGTH and SIZE
      ! use file-storage units, not an assumed number of bytes. This length
      ! check is not a payload hash or a memory guarantee for sparse files.
      inquire(iolength=real_storage_units) 0.0_real64
      file_units = -1_int64
      payload_position = 0_int64
      inquire(unit=unit, size=file_units, pos=payload_position, iostat=io_status)
      if (io_status /= 0 .or. file_units < 0_int64 .or. payload_position < 1_int64 .or. real_storage_units <= 0) then
         close(unit)
         status = FBT_ERR_IO
         return
      end if
      parameter_count = fbc_parameter_count(config)
      if (parameter_count < 0_int64) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      ! Energy shapes: embedding(vocab), Q/K/V(width,heads), output(width),
      ! FFN gates(2*width), FFN output(ffn), channel gate(width), router
      ! (3*width), scale gate(width), four slot projections(4*width), plus ten
      ! scalar energies. The parameter count already includes scalar momentum.
      energy_count = int(config%vocabulary_size, int64) + &
                     3_int64 * int(config%width, int64) * int(config%heads, int64) + &
                     12_int64 * int(config%width, int64) + int(config%ffn_width, int64) + 10_int64
      if (parameter_count > (huge(0_int64) - energy_count) / 2_int64) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      stored_reals = 2_int64 * parameter_count + energy_count
      if (stored_reals > huge(0_int64) / int(real_storage_units, int64)) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      payload_units = stored_reals * int(real_storage_units, int64)
      header_units = payload_position - 1_int64
      if (header_units > huge(0_int64) - payload_units) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      if (file_units < header_units + payload_units) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      call fbc_model_init(model, config, 1_int64, allocation_status)
      if (allocation_status /= FBC_OK) then
         close(unit)
         status = allocation_status
         return
      end if
      model%effective_horizon = effective_horizon
      model%horizon_clamped = clamped_flag /= 0_int32
      model%training_rule = version
      call fbc_optimizer_init(optimizer, config, allocation_status)
      if (allocation_status /= 0_int32) then
         close(unit)
         call fbc_model_destroy(model)
         call fbc_optimizer_destroy(optimizer)
         status = FBT_ERR_IO
         return
      end if
      call read_payload(unit, model, io_status)
      call read_optimizer(unit, optimizer, io_status)
      close(unit)
      if (io_status /= 0) then
         call fbc_model_destroy(model)
         call fbc_optimizer_destroy(optimizer)
         status = FBT_ERR_FORMAT
         return
      end if
      ! The cached dynamics are derived from the logits that were just read, so
      ! the cache has to be rebuilt before the model is used again.
      call fbc_refresh_dynamics(model)
   end subroutine fbc_checkpoint_load

   !> Reads only the header, for `inspect`.
   subroutine fbc_checkpoint_meta_read(path, meta, status)
      character(len=*), intent(in) :: path
      type(fbc_checkpoint_meta_t), intent(out) :: meta
      integer(int32), intent(out) :: status
      integer(int32) :: unit, io_status, version, clamped_flag
      integer(int64) :: path_units
      character(len=8) :: magic
      type(fbc_config_t) :: header_config
      real(real64) :: rms_epsilon, slot_decay, slot_temperature, loss_sum

      meta = fbc_checkpoint_meta_t()
      status = FBT_OK
      version = 0_int32
      clamped_flag = 0_int32
      io_status = 0
      ! Inspect also avoids opening a source whose reported length is unknown
      ! or zero. On Linux this rejects a FIFO without waiting for a writer.
      path_units = -1_int64
      inquire(file=trim(path), size=path_units, iostat=io_status)
      if (io_status /= 0 .or. path_units < 0_int64) then
         status = FBT_ERR_IO
         return
      end if
      if (path_units == 0_int64) then
         status = FBT_ERR_FORMAT
         return
      end if
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = FBT_ERR_IO
         return
      end if
      read(unit, iostat=io_status) magic
      if (io_status /= 0) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      if (magic == FBC_LEGACY_MAGIC) then
         close(unit)
         status = FBT_ERR_LEGACY
         return
      end if
      if (magic /= FBC_CHECKPOINT_MAGIC) then
         close(unit)
         status = FBT_ERR_FORMAT
         return
      end if
      read(unit, iostat=io_status) version
      if (io_status == 0 .and. version >= FBC_CHECKPOINT_MIN_VERSION .and. &
          version <= FBC_CHECKPOINT_VERSION) then
         read(unit, iostat=io_status) meta%vocabulary_size, meta%width, meta%heads, meta%ffn_width, &
                                      meta%local_window, meta%memory_scales, meta%memory_slots, &
                                      meta%logical_horizon, rms_epsilon, slot_decay, slot_temperature
      end if
      if (io_status == 0) read(unit, iostat=io_status) meta%effective_horizon
      if (io_status == 0) read(unit, iostat=io_status) clamped_flag
      if (io_status == 0) read(unit, iostat=io_status) meta%updates, meta%tokens
      loss_sum = 0.0_real64
      if (io_status == 0 .and. version >= FBC_CHECKPOINT_MIN_LOAD_VERSION) read(unit, iostat=io_status) loss_sum
      close(unit)
      if (io_status /= 0 .or. version < FBC_CHECKPOINT_MIN_VERSION .or. &
          version > FBC_CHECKPOINT_VERSION) then
         status = FBT_ERR_FORMAT
         return
      end if
      meta%horizon_clamped = clamped_flag /= 0_int32
      meta%format_version = version
      ! V5/V6/V7 have the same payload but retain their own model semantics.
      if (version < FBC_CHECKPOINT_MIN_LOAD_VERSION) then
         meta%convertible = .false.
         status = FBT_ERR_VERSION
         return
      end if
      ! Header-only inspection cannot validate the payload, but it must not
      ! call a structurally or numerically invalid configuration convertible.
      header_config%vocabulary_size = meta%vocabulary_size
      header_config%width = meta%width
      header_config%heads = meta%heads
      header_config%ffn_width = meta%ffn_width
      header_config%local_window = meta%local_window
      header_config%memory_scales = meta%memory_scales
      header_config%memory_slots = meta%memory_slots
      header_config%logical_horizon = meta%logical_horizon
      header_config%rms_epsilon = rms_epsilon
      header_config%slot_decay = slot_decay
      header_config%slot_temperature = slot_temperature
      if (.not. fbc_config_is_valid(header_config)) then
         status = FBT_ERR_FORMAT
         return
      end if
      if (meta%updates < 0_int64 .or. meta%tokens < 0_int64 .or. &
          (clamped_flag /= 0_int32 .and. clamped_flag /= 1_int32)) then
         meta%convertible = .false.
         status = FBT_ERR_FORMAT
         return
      end if
      if (.not. fbc_real64_is_finite(loss_sum) .or. .not. fbc_real64_is_finite(meta%effective_horizon)) then
         meta%convertible = .false.
         status = FBT_ERR_FORMAT
         return
      end if
      if (loss_sum < 0.0_real64 .or. meta%effective_horizon <= 0.0_real64) then
         status = FBT_ERR_FORMAT
         return
      end if
      meta%convertible = .true.
   end subroutine fbc_checkpoint_meta_read

end module fixed_budget_training
