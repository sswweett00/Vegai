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
!  Format `VGFB0001` version 3 stores: architecture version, logical horizon,
!  physical local window, memory scales/slots, the effective (achievable)
!  horizon and its clamp flag, every learnable array, full optimiser state,
!  the update counter and the token counter.  Legacy `VGNR0001` checkpoints are
!  *rejected with a distinct status*, never silently converted: they belong to
!  the older decoder and must be used with the older binary.
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
   !> Version 2 adds the learnable memory dynamics (a decay logit and a write
   !> logit per scale).  Version 3 adds the two input-dependent gate
   !> projections.  An older file has no value for the arrays a newer version
   !> added, and inventing one (a zero gate, say) would silently change the
   !> model, so every older version is refused with FBT_ERR_VERSION instead of
   !> being converted.
   integer(int32), parameter, public :: FBC_CHECKPOINT_VERSION = 3_int32
   integer(int32), parameter, public :: FBC_CHECKPOINT_MIN_VERSION = 1_int32

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
      logical :: convertible = .true.
   end type fbc_checkpoint_meta_t

   public :: fbc_loss_and_gradient, fbc_train_stream, fbc_evaluate_stream
   public :: fbc_generate_tokens, fbc_checkpoint_save, fbc_checkpoint_load, fbc_checkpoint_meta_read
   public :: fbc_parse_task

contains

   !----------------------------------------------------------------------------
   !  Loss
   !----------------------------------------------------------------------------

   !> Cross-entropy of one prediction and dLoss/dlogits = softmax - onehot.
   pure subroutine fbc_loss_and_gradient(logits, target, dlogits, loss, status)
      real(real64), intent(in) :: logits(:)
      integer(int32), intent(in) :: target
      real(real64), intent(out) :: dlogits(:)
      real(real64), intent(out) :: loss
      integer(int32), intent(out) :: status
      real(real64) :: maximum, normalizer, probability

      status = FBC_OK
      loss = 0.0_real64
      dlogits = 0.0_real64
      if (target < 1_int32 .or. target > size(logits, kind=int32)) then
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      maximum = maxval(logits)
      normalizer = 0.0_real64
      dlogits = exp(logits - maximum)
      normalizer = sum(dlogits)
      if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_real64) then
         status = FBC_ERR_NUMERIC
         return
      end if
      dlogits = dlogits / normalizer
      probability = dlogits(target)
      if (probability < 1.0e-30_real64) probability = 1.0e-30_real64
      loss = -log(probability)
      dlogits(target) = dlogits(target) - 1.0_real64
      if (.not. ieee_is_finite(loss)) status = FBC_ERR_NUMERIC
   end subroutine fbc_loss_and_gradient

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
      integer(int32) :: position, offset, answer_count, answer_first, filler_first, answer, block_length
      integer(int64) :: random_state, draw

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
         do position = 1_int32, token_count
            call fbc_next_random(random_state, draw)
            tokens(position) = 4_int32 + int(mod(draw, int(vocabulary_size - 3_int32, int64)), int32)
         end do
         ! The learnable rule is "next = previous + 1 (wrapping)", which is a
         ! short-range task: it measures optimisation, not memory.
         do position = 2_int32, token_count
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
         block_length = 3_int32 + distance
         position = 1_int32
         do while (position <= token_count)
            call fbc_next_random(random_state, draw)
            answer = answer_first + int(mod(draw, int(answer_count, int64)), int32)
            tokens(position) = 1_int32                                   ! KEY
            if (position + 1_int32 <= token_count) tokens(position + 1_int32) = answer
            do offset = 2_int32, distance + 1_int32
               if (position + offset <= token_count) then
                  call fbc_next_random(random_state, draw)
                  tokens(position + offset) = filler_first + int(mod(draw, &
                     int(max(1_int32, vocabulary_size - filler_first + 1_int32), int64)), int32)
               end if
            end do
            if (position + distance + 2_int32 <= token_count) then
               tokens(position + distance + 2_int32) = 2_int32          ! QUERY
               ! The target of the QUERY position is the answer that follows it,
               ! so the mask marks the position *after* the query.
               if (position + distance + 3_int32 <= token_count) then
                  tokens(position + distance + 3_int32) = answer
                  answer_mask(position + distance + 2_int32) = .true.
               end if
            end if
            position = position + block_length + 1_int32
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
   !> One optimiser update per chunk of `chunk_tokens` predictions.  The chunk
   !> size bounds the work between updates and nothing else: it is not a tensor
   !> dimension and it is not derived from the horizon.
   subroutine fbc_train_stream(model, optimizer, state, tokens, token_count, chunk_tokens, &
                               update_config, workspace, gradient, metrics, status)
      type(fbc_model_t), intent(inout) :: model
      type(fbc_optimizer_t), intent(inout) :: optimizer
      type(fbc_state_t), intent(inout) :: state
      integer(int32), intent(in) :: tokens(:), token_count, chunk_tokens
      type(fbc_update_config_t), intent(in) :: update_config
      type(fbc_workspace_t), intent(inout) :: workspace
      type(fbc_gradient_t), intent(inout) :: gradient
      type(fbc_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      integer(int32) :: position, offset, target, step_status
      integer(int64) :: parameter_updates
      real(real64) :: loss, norm, scale
      real(real64), allocatable :: logits(:), hidden(:), dlogits(:)

      status = FBT_OK
      if (chunk_tokens < 1_int32 .or. chunk_tokens > FBC_MAX_CHUNK_TOKENS) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      if (token_count + 1_int32 > size(tokens, kind=int32)) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      allocate(logits(model%config%vocabulary_size), hidden(model%config%width), &
               dlogits(model%config%vocabulary_size))
      position = 1_int32
      do while (position + chunk_tokens <= token_count + 1_int32)
         call fbc_gradient_zero(gradient)
         do offset = 0_int32, chunk_tokens - 1_int32
            target = tokens(position + offset + 1_int32)
            call fbc_step(model, state, workspace, tokens(position + offset), hidden, logits, step_status)
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
            call fbc_step_backward(model, state, workspace, dlogits, gradient, step_status)
            if (step_status /= FBC_OK) then
               status = step_status
               deallocate(logits, hidden, dlogits)
               return
            end if
            metrics%loss_sum = metrics%loss_sum + loss
            metrics%tokens = metrics%tokens + 1_int64
         end do
         ! Mean gradient of the chunk, then the optional global clip.  The clip
         ! is a single scalar rescale, so it cannot reorder the reductions.
         scale = 1.0_real64 / real(chunk_tokens, real64)
         call fbc_scale_gradient(gradient, scale)
         norm = fbc_gradient_norm(gradient)
         metrics%last_gradient_norm = norm
         if (update_config%gradient_clip > 0.0_real64 .and. norm > update_config%gradient_clip) then
            call fbc_scale_gradient(gradient, update_config%gradient_clip / norm)
         end if
         if (.not. ieee_is_finite(norm)) then
            metrics%numeric_ok = .false.
            status = FBT_ERR_NUMERIC
            deallocate(logits, hidden, dlogits)
            return
         end if
         call fbc_optimizer_apply(optimizer, model, gradient, update_config, step_status, parameter_updates)
         if (step_status /= 0_int32) then
            status = FBT_ERR_ARGUMENT
            deallocate(logits, hidden, dlogits)
            return
         end if
         ! The cached dynamics are a pure function of the logits, so refresh them
         ! once per optimiser update: forward and backward keep reading arrays
         ! and the token loop evaluates no transcendental for them.
         call fbc_refresh_dynamics(model)
         metrics%updates = metrics%updates + 1_int64
         metrics%parameter_updates = metrics%parameter_updates + parameter_updates
         position = position + chunk_tokens
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
      real(real64) :: loss, total, best_score
      integer(int64) :: correct
      real(real64), allocatable :: logits(:), hidden(:), dlogits(:)

      status = FBT_OK
      mean_nll = 0.0_real64
      answer_accuracy = 0.0_real64
      answer_count = 0_int64
      correct = 0_int64
      total = 0.0_real64
      ! Both outputs are intent(out), but `answer_accuracy` used to be assigned
      ! only inside the masked branch: a stream with no masked answer position
      ! then returned whatever the caller's variable happened to hold.
      answer_accuracy = 0.0_real64
      answer_count = 0_int64
      mean_nll = 0.0_real64
      if (token_count + 1_int32 > size(tokens, kind=int32)) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      allocate(logits(model%config%vocabulary_size), hidden(model%config%width), &
               dlogits(model%config%vocabulary_size))
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
      if (.not. fbc_model_is_ready(model) .or. .not. optimizer%initialized .or. len_trim(path) == 0) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      open(newunit=unit, file=trim(path), status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = FBT_ERR_IO
         return
      end if
      write(unit, iostat=io_status) FBC_CHECKPOINT_MAGIC
      if (io_status == 0) write(unit, iostat=io_status) FBC_CHECKPOINT_VERSION
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
      if (version < FBC_CHECKPOINT_VERSION) then
         ! A readable older format that cannot describe this model: refused, not
         ! converted, so an old file can never be silently reinterpreted.
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
      call fbc_model_init(model, config, 1_int64, allocation_status)
      if (allocation_status /= FBC_OK) then
         close(unit)
         status = allocation_status
         return
      end if
      model%effective_horizon = effective_horizon
      model%horizon_clamped = clamped_flag /= 0_int32
      call fbc_optimizer_init(optimizer, config, allocation_status)
      if (allocation_status /= 0_int32) then
         close(unit)
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
      character(len=8) :: magic
      real(real64) :: rms_epsilon, slot_decay, slot_temperature

      meta = fbc_checkpoint_meta_t()
      status = FBT_OK
      version = 0_int32
      clamped_flag = 0_int32
      io_status = 0
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
      close(unit)
      if (io_status /= 0 .or. version < FBC_CHECKPOINT_MIN_VERSION .or. &
          version > FBC_CHECKPOINT_VERSION) then
         status = FBT_ERR_FORMAT
         return
      end if
      meta%horizon_clamped = clamped_flag /= 0_int32
      ! An older format is reported as such: readable metadata, not convertible.
      if (version < FBC_CHECKPOINT_VERSION) then
         meta%format_version = version
         meta%convertible = .false.
         status = FBT_ERR_VERSION
         return
      end if
   end subroutine fbc_checkpoint_meta_read

end module fixed_budget_training
