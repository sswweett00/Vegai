! SPDX-License-Identifier: MIT
!==============================================================================
!  neural_pretraining.f90
!  Deterministic, bounded-state teacher-forced adaptation for the native
!  neural retention decoder.
!
!  This module intentionally names its currently implemented optimization
!  scope: it differentiates the tied output embedding matrix exactly for the
!  next-token cross-entropy while the causal core remains a fixed feature map.
!  It is therefore a real, resumable neural training path, but it must not be
!  represented as full back-propagation through local attention or retention.
!  The narrow scope makes its gradient, memory bound, and resume semantics
!  auditable and creates a stable native checkpoint boundary for later full
!  BPTT/optimizer backends.
!
!  Training-system v2 adds, inside that same auditable scope: AdamW with
!  bit-exact checkpointed moments, warmup/inverse-square-root and
!  warmup/cosine learning-rate schedules, label smoothing, top-1/top-5
!  next-token metrics, a no-update file evaluation path, and a reusable
!  per-file workspace (the previous implementation reallocated the decoder
!  state per record and scan-checked the full embedding matrix for
!  finiteness on every token).  None of this changes the frozen-core
!  boundary above.
!==============================================================================
module neural_pretraining
   use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use subword_bpe
   use utf8_record_stream
   use neural_retention_decoder
   implicit none
   private

   integer(int32), parameter, public :: NPT_OK = 0_int32
   integer(int32), parameter, public :: NPT_ERR_INVALID_ARGUMENT = -1000_int32
   integer(int32), parameter, public :: NPT_ERR_MODEL = -1001_int32
   integer(int32), parameter, public :: NPT_ERR_ALLOCATION = -1002_int32
   integer(int32), parameter, public :: NPT_ERR_IO = -1003_int32
   integer(int32), parameter, public :: NPT_ERR_FORMAT = -1004_int32
   integer(int32), parameter, public :: NPT_ERR_TOKENIZER = -1005_int32
   integer(int32), parameter, public :: NPT_ERR_NUMERICAL = -1006_int32
   integer(int32), parameter, public :: NPT_ERR_INCOMPATIBLE = -1007_int32

   ! This mode is deliberately explicit.  No caller can mistake a frozen-core
   ! output-adapter update for full neural back-propagation.
   integer(int32), parameter, public :: NPT_TRAINABLE_TIED_OUTPUT = 1_int32

   ! Optimizer identifiers for the tied-output update.  SGD is the legacy
   ! v1 update (kept bit-reproducible for pre-v3 checkpoints); AdamW is the
   ! training-system v2 default for newly constructed configurations.
   integer(int32), parameter, public :: NPT_OPTIMIZER_SGD = 1_int32
   integer(int32), parameter, public :: NPT_OPTIMIZER_ADAMW = 2_int32

   ! Upper bound on gradient_accumulation.  It bounds two stack allocations
   ! (the staging buffers are O(vocabulary * accumulation) and the flush
   ! kernel's clipped-hidden block is O(width * accumulation)), and it costs
   ! nothing in throughput: past a few hundred tokens per step the optimizer
   ! state is amortized to noise and only the batch memory grows.
   integer(int32), parameter, public :: NPT_MAX_GRADIENT_ACCUMULATION = 512_int32

   ! Learning-rate schedule identifiers.  The schedule is a pure function of
   ! the persisted update counter, so a checkpoint/resume boundary cannot
   ! change the effective learning rate of any update.
   integer(int32), parameter, public :: NPT_SCHEDULE_CONSTANT = 0_int32
   integer(int32), parameter, public :: NPT_SCHEDULE_WARMUP_INVSQRT = 1_int32
   integer(int32), parameter, public :: NPT_SCHEDULE_WARMUP_COSINE = 2_int32

   character(len=8), parameter :: NEURAL_CHECKPOINT_MAGIC = 'VGNR0001'
   ! v2 binds all decoder configuration fields (including retention decay,
   ! numerical epsilon, and rotary period) to the serialized parameter state.
   ! v3 additionally binds the optimizer/schedule configuration and the exact
   ! AdamW first/second moments, so a resumed stream is bit-identical to the
   ! one-pass stream.  Loading a v2 checkpoint restores the legacy SGD
   ! configuration with no moments, preserving its original update semantics.
   ! v4 adds the mini-batch accumulation factor to the serialized training
   ! configuration, so a resume can never silently regroup the updates of a
   ! run that was started with a different gradient_accumulation.  v2 and v3
   ! files still load (v3 binds accumulation = 1, the path it was written by).
   integer(int32), parameter :: NEURAL_CHECKPOINT_VERSION = 4_int32
   integer(int32), parameter :: NEURAL_CHECKPOINT_VERSION_V3 = 3_int32
   integer(int32), parameter :: NEURAL_CHECKPOINT_VERSION_V2 = 2_int32
   integer(int32), parameter :: FILE_LINE_BYTES = 32768_int32

   type, public :: neural_train_config_t
      integer(int32) :: trainable_mode = NPT_TRAINABLE_TIED_OUTPUT
      ! Peak step size for the exact tied-output cross-entropy gradient.
      real(real64) :: learning_rate = 0.025_real64
      ! A conservative norm bound on the hidden vector used by the update.
      real(real64) :: gradient_clip_norm = 4.0_real64
      ! Per-value safety envelope for the mutable tied embedding matrix.
      real(real64) :: embedding_value_limit = 8.0_real64
      integer(int32) :: maximum_tokens_per_record = 8192_int32
      ! Training-system v2 fields.  Defaults keep every pre-v2 named-arg
      ! construction valid; the CLI sets schedule/horizon explicitly.
      integer(int32) :: optimizer = NPT_OPTIMIZER_ADAMW
      real(real64) :: adam_beta1 = 0.9_real64
      real(real64) :: adam_beta2 = 0.999_real64
      real(real64) :: adam_epsilon = 1.0e-8_real64
      ! Decoupled AdamW weight decay, applied to embeddings only (not bias).
      real(real64) :: weight_decay = 0.01_real64
      ! Label-smoothing mass in [0, 0.5).  Reported NLL stays unsmoothed.
      real(real64) :: label_smoothing = 0.0_real64
      integer(int32) :: lr_schedule = NPT_SCHEDULE_CONSTANT
      integer(int64) :: warmup_updates = 0_int64
      ! Cosine horizon in updates; required (> warmup) for the cosine mode.
      integer(int64) :: total_updates = 0_int64
      ! Cosine floor as a fraction of the peak learning rate, in [0, 1].
      real(real64) :: minimum_lr_ratio = 0.1_real64
      ! Prediction tokens accumulated into one optimizer step.  1 keeps the
      ! exact one-step-per-token path.  N > 1 evaluates N per-token gradients
      ! at the pre-step parameters, averages them, and applies a single step,
      ! which is ordinary mini-batch SGD/AdamW.  The batch is always flushed
      ! at the end of a record, and every checkpoint boundary is a record
      ! boundary, so "chunk + checkpoint + chunk" stays bit-identical to the
      ! one-pass stream for any fixed accumulation.  The effective batch is
      ! min(gradient_accumulation, maximum_tokens_per_record) and is capped
      ! at NPT_MAX_GRADIENT_ACCUMULATION.
      integer(int32) :: gradient_accumulation = 1_int32
   end type neural_train_config_t

   type, public :: neural_train_state_t
      logical :: ready = .false.
      integer(int64) :: records_completed = 0_int64
      ! Count raw source records independently from eligible training records.
      ! A resume cursor is meaningful only with the same attested source order.
      integer(int64) :: source_records_completed = 0_int64
      integer(int64) :: updates_completed = 0_int64
      integer(int64) :: tokens_seen = 0_int64
      real(real64) :: cumulative_nll = 0.0_real64
      ! AdamW moments, lazily allocated at the shape of the first model that
      ! updates through this state.  They are part of the v3 checkpoint so a
      ! resumed run applies bit-identical updates to a one-pass run.  The
      ! Adam step index is derived from updates_completed, never stored
      ! separately, which makes counter/moment divergence impossible.
      ! AdamW first/second moments live in the parameter precision (real32).
      ! The element update is sqrt + divide bound, and real32 lets the compiler
      ! use 8-wide vector sqrt/divide instead of 4-wide: that is what makes the
      ! per-token update loop fast, since its cost is arithmetic, not memory.
      ! Storing moments at the same precision as the parameters they update is
      ! also the mainstream configuration (PyTorch keeps exp_avg/exp_avg_sq in
      ! the parameter dtype); the gradient itself is still accumulated real64.
      real(real32), allocatable :: adam_m_embedding(:, :)
      real(real32), allocatable :: adam_v_embedding(:, :)
      real(real32), allocatable :: adam_m_bias(:)
      real(real32), allocatable :: adam_v_bias(:)
   end type neural_train_state_t

   type, public :: neural_train_metrics_t
      integer(int64) :: records = 0_int64
      integer(int64) :: prediction_tokens = 0_int64
      integer(int64) :: skipped_records = 0_int64
      integer(int64) :: tokenizer_truncations = 0_int64
      real(real64) :: cumulative_nll = 0.0_real64
      real(real64) :: mean_nll = 0.0_real64
      ! Rank of the true next token under the pre-update distribution,
      ! counted with lowest-index tie-breaking (matching greedy argmax).
      integer(int64) :: top1_correct = 0_int64
      integer(int64) :: top5_correct = 0_int64
      ! Cumulative wall time per phase.  A throughput claim has to be
      ! attributable: without this, "the update is the bottleneck" is a guess.
      ! The cost is a few clock reads per prediction.
      real(real64) :: decode_seconds = 0.0_real64
      real(real64) :: project_seconds = 0.0_real64
      real(real64) :: score_seconds = 0.0_real64
      real(real64) :: update_seconds = 0.0_real64
   end type neural_train_metrics_t

   ! Conservative resident-memory accounting for one sequential training
   ! stream. It deliberately excludes compiler/runtime/OpenMP overhead and is
   ! therefore a resource floor, not a host-RAM sufficiency certification.
   type, public :: neural_training_resource_plan_t
      integer(int64) :: decoder_parameter_bytes_real32 = 0_int64
      integer(int64) :: tokenizer_bytes = 0_int64
      integer(int64) :: decoder_state_bytes_real32 = 0_int64
      integer(int64) :: record_token_buffer_bytes = 0_int64
      integer(int64) :: per_step_working_bytes = 0_int64
      ! real64 AdamW moments (zero for SGD): 2 x (width*vocabulary + vocab).
      integer(int64) :: optimizer_state_bytes_real64 = 0_int64
      integer(int64) :: minimum_resident_bytes = 0_int64
   end type neural_training_resource_plan_t

   ! Reusable per-stream buffers.  The file trainer allocates these once and
   ! resets only the decoder state per record; the previous implementation
   ! reallocated the retention state and probability vectors for every
   ! record, which dominated wall time on short-record corpora.
   type :: neural_train_workspace_t
      logical :: ready = .false.
      type(neural_decoder_state_t) :: decoder_state
      real(real32), allocatable :: hidden(:)
      real(real32), allocatable :: logits(:)
      real(real64), allocatable :: probabilities(:)
      ! Mini-batch staging buffers, allocated only when
      ! gradient_accumulation > 1.  RESIDUALS is (vocabulary, capacity) so a
      ! column holds the whole smoothed residual vector of one prediction
      ! token, which keeps the flush kernel's inner accumulation contiguous.
      ! The residual is stored in real32 (the parameter precision) because it
      ! is one of ~B terms in an averaged gradient; the accumulate itself is
      ! real64.
      real(real32), allocatable :: batch_residuals(:, :)
      ! (capacity + projection lanes, width): one contiguous hidden row per
      ! staged token, padded so the block projection's last lane group reads
      ! defined values.
      real(real32), allocatable :: batch_hidden(:, :)
      ! (capacity, vocabulary): one row per staged token, so a row is
      ! contiguous and can be handed to stable_softmax without a copy.  Filled
      ! by a single pass over the tied embedding when the block is resolved,
      ! which is the whole point of deferring the projection.
      real(real32), allocatable :: batch_logits(:, :)
      real(real64), allocatable :: batch_clip(:)
      integer(int32), allocatable :: batch_targets(:)
      integer(int32) :: batch_count = 0_int32
      integer(int32) :: batch_capacity = 0_int32
   end type neural_train_workspace_t

   public :: neural_train_config_is_valid, neural_train_state_init, neural_train_metrics_reset
   public :: neural_train_token_sequence, neural_evaluate_token_sequence, neural_train_utf8_file
   public :: neural_evaluate_utf8_file, neural_scheduled_learning_rate
   public :: neural_training_resource_plan, neural_checkpoint_save, neural_checkpoint_load

contains

   ! Two optimizer invariants are enforced here rather than trusted from the
   ! caller.  AdamW applies decoupled decay as theta <- (1 - lr*wd)*theta, so
   ! lr*wd >= 1 would annihilate (or invert the sign of) every embedding while
   ! the run still reported success; checking the peak rate is sufficient
   ! because every implemented schedule returns rate <= peak rate.  The legacy
   ! SGD path has no decay term at all, so a non-zero weight_decay on an SGD
   ! configuration would be a silently ignored claim that is nevertheless
   ! serialized into the checkpoint and the evidence report.
   pure logical function neural_train_config_is_valid(config)
      type(neural_train_config_t), intent(in) :: config

      neural_train_config_is_valid = config%trainable_mode == NPT_TRAINABLE_TIED_OUTPUT .and. &
                                     config%gradient_accumulation >= 1_int32 .and. &
                                     config%gradient_accumulation <= NPT_MAX_GRADIENT_ACCUMULATION .and. &
                                     config%gradient_accumulation <= config%maximum_tokens_per_record .and. &
                                     ieee_is_finite(config%learning_rate) .and. config%learning_rate > 0.0_real64 .and. &
                                     ieee_is_finite(config%gradient_clip_norm) .and. config%gradient_clip_norm > 0.0_real64 .and. &
                                     ieee_is_finite(config%embedding_value_limit) .and. &
                                     config%embedding_value_limit > 0.0_real64 .and. &
                                     config%maximum_tokens_per_record >= 2_int32 .and. &
                                     (config%optimizer == NPT_OPTIMIZER_SGD .or. &
                                      config%optimizer == NPT_OPTIMIZER_ADAMW) .and. &
                                     ieee_is_finite(config%adam_beta1) .and. config%adam_beta1 >= 0.0_real64 .and. &
                                     config%adam_beta1 < 1.0_real64 .and. &
                                     ieee_is_finite(config%adam_beta2) .and. config%adam_beta2 >= 0.0_real64 .and. &
                                     config%adam_beta2 < 1.0_real64 .and. &
                                     ieee_is_finite(config%adam_epsilon) .and. config%adam_epsilon > 0.0_real64 .and. &
                                     ieee_is_finite(config%weight_decay) .and. config%weight_decay >= 0.0_real64 .and. &
                                     config%learning_rate * config%weight_decay < 1.0_real64 .and. &
                                     (config%optimizer == NPT_OPTIMIZER_ADAMW .or. &
                                      config%weight_decay == 0.0_real64) .and. &
                                     ieee_is_finite(config%label_smoothing) .and. &
                                     config%label_smoothing >= 0.0_real64 .and. config%label_smoothing < 0.5_real64 .and. &
                                     (config%lr_schedule == NPT_SCHEDULE_CONSTANT .or. &
                                      config%lr_schedule == NPT_SCHEDULE_WARMUP_INVSQRT .or. &
                                      config%lr_schedule == NPT_SCHEDULE_WARMUP_COSINE) .and. &
                                     config%warmup_updates >= 0_int64 .and. &
                                     ieee_is_finite(config%minimum_lr_ratio) .and. &
                                     config%minimum_lr_ratio >= 0.0_real64 .and. config%minimum_lr_ratio <= 1.0_real64 .and. &
                                     (config%lr_schedule /= NPT_SCHEDULE_WARMUP_COSINE .or. &
                                      config%total_updates > config%warmup_updates)
   end function neural_train_config_is_valid

   ! Effective learning rate for the (1-based) update that is about to run.
   ! This is a pure function of the persisted update counter so checkpoint
   ! boundaries can never perturb the schedule.
   pure real(real64) function neural_scheduled_learning_rate(config, update_index) result(rate)
      type(neural_train_config_t), intent(in) :: config
      integer(int64), intent(in) :: update_index
      real(real64) :: step, warmup, horizon, progress, cosine_part
      real(real64), parameter :: PI = 3.14159265358979323846_real64

      rate = config%learning_rate
      step = real(max(1_int64, update_index), real64)
      select case (config%lr_schedule)
      case (NPT_SCHEDULE_WARMUP_INVSQRT)
         warmup = real(max(1_int64, config%warmup_updates), real64)
         if (step <= warmup) then
            rate = config%learning_rate * step / warmup
         else
            rate = config%learning_rate * sqrt(warmup / step)
         end if
      case (NPT_SCHEDULE_WARMUP_COSINE)
         warmup = real(max(0_int64, config%warmup_updates), real64)
         horizon = real(max(config%warmup_updates + 1_int64, config%total_updates), real64)
         if (step <= warmup .and. warmup > 0.0_real64) then
            rate = config%learning_rate * step / warmup
         else
            progress = min(1.0_real64, max(0.0_real64, (step - warmup) / max(1.0_real64, horizon - warmup)))
            cosine_part = 0.5_real64 * (1.0_real64 + cos(PI * progress))
            rate = config%learning_rate * (config%minimum_lr_ratio + &
                                           (1.0_real64 - config%minimum_lr_ratio) * cosine_part)
         end if
      case default
         rate = config%learning_rate
      end select
   end function neural_scheduled_learning_rate

   subroutine neural_train_state_init(state)
      type(neural_train_state_t), intent(out) :: state

      state = neural_train_state_t()
      state%ready = .true.
   end subroutine neural_train_state_init

   subroutine neural_train_metrics_reset(metrics)
      type(neural_train_metrics_t), intent(out) :: metrics

      metrics = neural_train_metrics_t()
   end subroutine neural_train_metrics_reset

   subroutine neural_training_resource_plan(decoder_config, bpe, train_config, plan, status)
      type(neural_decoder_config_t), intent(in) :: decoder_config
      type(bpe_model_t), intent(in) :: bpe
      type(neural_train_config_t), intent(in) :: train_config
      type(neural_training_resource_plan_t), intent(out) :: plan
      integer(int32), intent(out) :: status
      type(neural_resource_plan_t) :: decoder_plan
      integer(int32) :: decoder_status
      integer(int64) :: width, vocabulary, ffn_width, accumulation

      plan = neural_training_resource_plan_t()
      status = NPT_OK
      if (.not. neural_decoder_config_is_valid(decoder_config) .or. .not. bpe_is_initialized(bpe) .or. &
          .not. neural_train_config_is_valid(train_config) .or. decoder_config%vocabulary_size < bpe%token_count) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      call neural_decoder_plan(decoder_config, decoder_plan, decoder_status)
      if (decoder_status /= NRD_OK) then
         status = NPT_ERR_MODEL
         return
      end if
      width = int(decoder_config%width, int64)
      vocabulary = int(decoder_config%vocabulary_size, int64)
      ffn_width = int(decoder_config%ffn_width, int64)
      plan%decoder_parameter_bytes_real32 = decoder_plan%parameter_bytes_real32
      plan%tokenizer_bytes = bpe_model_footprint_bytes(bpe)
      plan%decoder_state_bytes_real32 = decoder_plan%per_stream_state_bytes_real32
      plan%record_token_buffer_bytes = 4_int64 * int(train_config%maximum_tokens_per_record, int64)
      ! Decoder temporaries: 10 width-sized real32 vectors plus three FFN
      ! vectors; trainer allocations: hidden, logits, and real64 softmax.
      plan%per_step_working_bytes = 4_int64 * (11_int64 * width + 3_int64 * ffn_width + vocabulary) + &
                                    8_int64 * vocabulary
      ! Mini-batch staging per accumulated token: the residual column, the
      ! deferred logits column (both real32 per vocabulary entry), the hidden
      ! column and the clip/target scalars.
      if (train_config%gradient_accumulation > 1_int32) then
         accumulation = int(min(train_config%gradient_accumulation, train_config%maximum_tokens_per_record), int64)
         plan%per_step_working_bytes = plan%per_step_working_bytes + &
                                        accumulation * (8_int64 * vocabulary + 4_int64 * width + 8_int64 + 4_int64)
      end if
      plan%optimizer_state_bytes_real64 = 0_int64
      if (train_config%optimizer == NPT_OPTIMIZER_ADAMW) then
         ! Two moment arrays per element, each in the parameter precision.
         plan%optimizer_state_bytes_real64 = 8_int64 * (width * vocabulary + vocabulary)
      end if
      plan%minimum_resident_bytes = plan%decoder_parameter_bytes_real32 + plan%tokenizer_bytes + &
                                    plan%decoder_state_bytes_real32 + plan%record_token_buffer_bytes + &
                                    plan%per_step_working_bytes + plan%optimizer_state_bytes_real64
   end subroutine neural_training_resource_plan

   subroutine stable_softmax(logits, probabilities, status)
      real(real32), contiguous, intent(in) :: logits(:)
      real(real64), contiguous, intent(out) :: probabilities(:)
      integer(int32), intent(out) :: status
      real(real64) :: maximum, normalizer
      integer(int32) :: index

      status = NPT_OK
      if (size(logits, kind=int32) <= 0_int32 .or. size(probabilities, kind=int32) /= size(logits, kind=int32) .or. &
          .not. all(ieee_is_finite(logits))) then
         status = NPT_ERR_NUMERICAL
         return
      end if
      maximum = maxval(real(logits, real64))
      ! Exponentials are independent per token, so the optional parallel
      ! fill stores bit-identical values; the normalizer then accumulates
      ! in the same ascending order as the sequential form, keeping the
      ! resulting distribution bit-identical at any thread count.
      !$omp parallel do schedule(static) if(size(logits) >= 1024)
      do index = 1_int32, size(logits, kind=int32)
         probabilities(index) = exp(real(logits(index), real64) - maximum)
      end do
      !$omp end parallel do
      ! The reduction is order-fixed (the intrinsic sums in array order), so
      ! the distribution is identical at any thread count; writing it as a
      ! reduction rather than a scalar loop lets the compiler vectorize it.
      normalizer = sum(probabilities)
      if (.not. ieee_is_finite(normalizer) .or. normalizer <= tiny(1.0_real64)) then
         status = NPT_ERR_NUMERICAL
         return
      end if
      probabilities = probabilities / normalizer
      if (.not. all(ieee_is_finite(probabilities))) status = NPT_ERR_NUMERICAL
   end subroutine stable_softmax

   subroutine ensure_optimizer_state(model, config, state, status)
      type(neural_decoder_t), intent(in) :: model
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(inout) :: state
      integer(int32), intent(out) :: status
      integer :: allocation_status

      status = NPT_OK
      if (config%optimizer /= NPT_OPTIMIZER_ADAMW) return
      if (allocated(state%adam_m_embedding)) then
         ! Sequential checks: Fortran does not guarantee short-circuit
         ! evaluation, so never call size() on a maybe-unallocated array in
         ! the same expression as its allocated() guard.
         if (.not. allocated(state%adam_v_embedding)) status = NPT_ERR_INCOMPATIBLE
         if (.not. allocated(state%adam_m_bias)) status = NPT_ERR_INCOMPATIBLE
         if (.not. allocated(state%adam_v_bias)) status = NPT_ERR_INCOMPATIBLE
         if (status == NPT_OK) then
            if (size(state%adam_m_embedding, 1, kind=int32) /= model%config%width .or. &
                size(state%adam_m_embedding, 2, kind=int32) /= model%config%vocabulary_size .or. &
                size(state%adam_v_embedding, 1, kind=int32) /= model%config%width .or. &
                size(state%adam_v_embedding, 2, kind=int32) /= model%config%vocabulary_size .or. &
                size(state%adam_m_bias, kind=int32) /= model%config%vocabulary_size .or. &
                size(state%adam_v_bias, kind=int32) /= model%config%vocabulary_size) then
               status = NPT_ERR_INCOMPATIBLE
            end if
         end if
         return
      end if
      allocate(state%adam_m_embedding(model%config%width, model%config%vocabulary_size), &
               state%adam_v_embedding(model%config%width, model%config%vocabulary_size), &
               state%adam_m_bias(model%config%vocabulary_size), &
               state%adam_v_bias(model%config%vocabulary_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = NPT_ERR_ALLOCATION
         return
      end if
      state%adam_m_embedding = 0.0_real32
      state%adam_v_embedding = 0.0_real32
      state%adam_m_bias = 0.0_real32
      state%adam_v_bias = 0.0_real32
   end subroutine ensure_optimizer_state

   ! Monotonic wall clock for the phase counters.
   real(real64) function clock_seconds()
      integer(int64) :: count, rate

      call system_clock(count, rate)
      if (rate <= 0_int64) then
         clock_seconds = 0.0_real64
      else
         clock_seconds = real(count, real64) / real(rate, real64)
      end if
   end function clock_seconds

   subroutine workspace_ensure(workspace, model, batch_capacity, status)
      type(neural_train_workspace_t), intent(inout) :: workspace
      type(neural_decoder_t), intent(in) :: model
      integer(int32), intent(in) :: batch_capacity
      integer(int32), intent(out) :: status
      integer :: allocation_status
      integer(int32) :: decoder_status

      status = NPT_OK
      if (workspace%ready) then
         if (size(workspace%hidden, kind=int32) == model%config%width .and. &
             size(workspace%logits, kind=int32) == model%config%vocabulary_size .and. &
             workspace%batch_capacity == batch_capacity) return
         call workspace_destroy(workspace)
      end if
      allocate(workspace%hidden(model%config%width), workspace%logits(model%config%vocabulary_size), &
               workspace%probabilities(model%config%vocabulary_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = NPT_ERR_ALLOCATION
         return
      end if
      workspace%batch_capacity = batch_capacity
      workspace%batch_count = 0_int32
      if (batch_capacity > 1_int32) then
         allocate(workspace%batch_residuals(model%config%vocabulary_size, batch_capacity), &
                  workspace%batch_hidden(batch_capacity + NRD_LOGIT_BLOCK_LANES, model%config%width), &
                  workspace%batch_logits(batch_capacity, model%config%vocabulary_size), &
                  workspace%batch_clip(batch_capacity), workspace%batch_targets(batch_capacity), &
                  stat=allocation_status)
         if (allocation_status /= 0) then
            call workspace_destroy(workspace)
            status = NPT_ERR_ALLOCATION
            return
         end if
         ! The logits projection consumes a whole lane group at a time, so the
         ! columns past the last staged token must read defined values.
         workspace%batch_hidden = 0.0_real32
      end if
      call neural_decoder_state_init(workspace%decoder_state, model, decoder_status)
      if (decoder_status /= NRD_OK) then
         call workspace_destroy(workspace)
         status = NPT_ERR_MODEL
         return
      end if
      workspace%ready = .true.
   end subroutine workspace_ensure

   subroutine workspace_destroy(workspace)
      type(neural_train_workspace_t), intent(inout) :: workspace

      call neural_decoder_state_destroy(workspace%decoder_state)
      if (allocated(workspace%hidden)) deallocate(workspace%hidden)
      if (allocated(workspace%logits)) deallocate(workspace%logits)
      if (allocated(workspace%probabilities)) deallocate(workspace%probabilities)
      if (allocated(workspace%batch_residuals)) deallocate(workspace%batch_residuals)
      if (allocated(workspace%batch_hidden)) deallocate(workspace%batch_hidden)
      if (allocated(workspace%batch_logits)) deallocate(workspace%batch_logits)
      if (allocated(workspace%batch_clip)) deallocate(workspace%batch_clip)
      if (allocated(workspace%batch_targets)) deallocate(workspace%batch_targets)
      workspace%batch_count = 0_int32
      workspace%batch_capacity = 0_int32
      workspace%ready = .false.
   end subroutine workspace_destroy

   ! Gradient-clip scale for one hidden vector.  Factored out so the per-token
   ! path and the mini-batch staging path cannot drift apart: both must apply
   ! exactly the same bound, otherwise a batched run would not be the average
   ! of the per-token gradients it replaces.
   pure subroutine hidden_clip(hidden, config, clip_scale, finite_norm)
      real(real32), intent(in) :: hidden(:)
      type(neural_train_config_t), intent(in) :: config
      real(real64), intent(out) :: clip_scale
      logical, intent(out) :: finite_norm
      real(real64) :: hidden_norm

      hidden_norm = sqrt(sum(real(hidden, real64) * real(hidden, real64)))
      finite_norm = ieee_is_finite(hidden_norm)
      clip_scale = min(1.0_real64, config%gradient_clip_norm / max(hidden_norm, tiny(1.0_real64)))
   end subroutine hidden_clip

   ! One exact tied-output cross-entropy update.  neural_decoder_step uses
   ! logits = E^T h + b, so dL/dE(:,v) = residual(v) * h and dL/db(v) =
   ! residual(v), where residual = softmax - smoothed_target.  The hidden
   ! vector is norm-clipped before forming the gradient (legacy semantics).
   !
   ! Finiteness is now proven rather than re-scanned: every input is checked
   ! finite, |clip_scale * h_i| <= gradient_clip_norm, |residual| <= 1, the
   ! Adam denominator is >= epsilon > 0, moments are convex combinations of
   ! bounded gradients, and every proposal is clamped to the embedding value
   ! limit.  The previous implementation re-scanned the full V-by-d matrix
   ! with ieee_is_finite after every single token, which cost as much as the
   ! update itself; checkpoint save still verifies the full payload.
   subroutine tied_output_update(model, hidden, probabilities, target, config, state, status)
      type(neural_decoder_t), intent(inout) :: model
      real(real32), intent(in) :: hidden(:)
      real(real64), intent(in) :: probabilities(:)
      integer(int32), intent(in) :: target
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(inout) :: state
      integer(int32), intent(out) :: status
      integer(int32) :: vocabulary_size, width, token, feature
      integer(int64) :: update_index
      real(real64) :: clip_scale, residual, proposal, scale, rate
      logical :: norm_is_finite
      real(real64) :: smooth_mass, beta1_pow, beta2_pow, adam_step_scale
      real(real64) :: m_hat_factor, decay_factor
      real(real64) :: beta1, beta2, one_minus_beta1, one_minus_beta2, adam_epsilon, value_limit
      real(real64) :: scaled_rate, inverse_step_scale, scaled_residual
      real(real32) :: proposal32, m32, v32, gradient32, scaled_residual32
      real(real32) :: beta1_32, one_minus_beta1_32, beta2_32, one_minus_beta2_32, adam_epsilon_32
      real(real32) :: scaled_rate_32, inverse_step_scale_32, decay_factor_32, value_limit_32

      status = NPT_OK
      vocabulary_size = model%config%vocabulary_size
      width = model%config%width
      ! Probabilities come only from stable_softmax, whose success contract
      ! already guarantees a fully finite distribution; the former full
      ! probability scan here cost O(vocabulary) per token for a condition
      ! that cannot occur on any reachable path.  The hidden scan (O(width))
      ! and the post-update target-column sentinel stay.
      if (target < 1_int32 .or. target > vocabulary_size .or. size(hidden, kind=int32) /= width .or. &
          size(probabilities, kind=int32) /= vocabulary_size .or. .not. all(ieee_is_finite(hidden))) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      call hidden_clip(hidden, config, clip_scale, norm_is_finite)
      if (.not. norm_is_finite) then
         status = NPT_ERR_NUMERICAL
         return
      end if
      update_index = state%updates_completed + 1_int64
      rate = neural_scheduled_learning_rate(config, update_index)
      smooth_mass = config%label_smoothing / real(vocabulary_size, real64)
      ! Defence in depth for the decoupled-decay invariant.  Validation
      ! rejects peak_lr*weight_decay >= 1 and no schedule exceeds the peak, so
      ! this can only fire for a configuration mutated after validation; the
      ! alternative is a decay_factor <= 0 that silently zeroes or inverts the
      ! embedding it is supposed to regularise.
      if (config%optimizer == NPT_OPTIMIZER_ADAMW .and. rate * config%weight_decay >= 1.0_real64) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if

      if (config%optimizer == NPT_OPTIMIZER_ADAMW) then
         call ensure_optimizer_state(model, config, state, status)
         if (status /= NPT_OK) return
         beta1_pow = config%adam_beta1 ** real(update_index, real64)
         beta2_pow = config%adam_beta2 ** real(update_index, real64)
         ! Fold both bias corrections into the step: mhat/(sqrt(vhat)+eps)
         ! equals sqrt(1-b2^t)/(1-b1^t) * m/(sqrt(v)+eps') only with a
         ! rescaled epsilon, so keep the textbook form per element instead
         ! and precompute just the two correction denominators here.
         m_hat_factor = 1.0_real64 - beta1_pow
         adam_step_scale = sqrt(max(0.0_real64, 1.0_real64 - beta2_pow))
         if (.not. ieee_is_finite(m_hat_factor) .or. .not. ieee_is_finite(adam_step_scale) .or. &
             m_hat_factor <= 0.0_real64) then
            status = NPT_ERR_NUMERICAL
            return
         end if
         decay_factor = 1.0_real64 - rate * config%weight_decay
         ! Loop-invariant scalars.  The moment recurrences keep their exact
         ! expressions (bit-identical states, so checkpoint continuation is
         ! unaffected); only the proposal folds the two per-element
         ! divisions into loop-invariant scaled_rate/inverse_step_scale
         ! multipliers, a reassociation-level change bounded by real64 ulp
         ! rounding and covered by the quality A/B in the evidence suite.
         beta1 = config%adam_beta1
         beta2 = config%adam_beta2
         one_minus_beta1 = 1.0_real64 - beta1
         one_minus_beta2 = 1.0_real64 - beta2
         adam_epsilon = config%adam_epsilon
         value_limit = config%embedding_value_limit
         scaled_rate = rate / m_hat_factor
         inverse_step_scale = 1.0_real64 / adam_step_scale
         ! The per-element chain runs in the parameter precision (real32).  Its
         ! cost is one sqrt and one divide per element over vocabulary x width
         ! elements, i.e. pure arithmetic, and 8-wide real32 vector sqrt/divide
         ! is roughly twice as fast as the 4-wide real64 form.  The scalars are
         ! converted once instead of per element.
         beta1_32 = real(beta1, real32)
         one_minus_beta1_32 = real(one_minus_beta1, real32)
         beta2_32 = real(beta2, real32)
         one_minus_beta2_32 = real(one_minus_beta2, real32)
         adam_epsilon_32 = real(adam_epsilon, real32)
         value_limit_32 = real(value_limit, real32)
         scaled_rate_32 = real(scaled_rate, real32)
         inverse_step_scale_32 = real(inverse_step_scale, real32)
         decay_factor_32 = real(decay_factor, real32)
         ! Columns (and their moment columns) are disjoint, so the OpenMP
         ! loop below needs no reduction and preserves online token order.
         !$omp parallel do private(residual, proposal, scaled_residual, feature, &
         !$omp proposal32, m32, v32, gradient32, scaled_residual32) &
         !$omp schedule(static) if(vocabulary_size >= 512_int32)
         do token = 1_int32, vocabulary_size
            residual = probabilities(token) - smooth_mass
            if (token == target) residual = residual - (1.0_real64 - config%label_smoothing)
            scaled_residual = residual * clip_scale
            scaled_residual32 = real(scaled_residual, real32)
            !$omp simd private(proposal32)
            do feature = 1_int32, width
               gradient32 = scaled_residual32 * hidden(feature)
               m32 = beta1_32 * state%adam_m_embedding(feature, token) + one_minus_beta1_32 * gradient32
               v32 = beta2_32 * state%adam_v_embedding(feature, token) + one_minus_beta2_32 * gradient32 * gradient32
               state%adam_m_embedding(feature, token) = m32
               state%adam_v_embedding(feature, token) = v32
               proposal32 = decay_factor_32 * model%token_embedding(feature, token) - &
                            scaled_rate_32 * m32 / (sqrt(v32) * inverse_step_scale_32 + adam_epsilon_32)
               model%token_embedding(feature, token) = max(-value_limit_32, min(value_limit_32, proposal32))
            end do
            m32 = beta1_32 * state%adam_m_bias(token) + one_minus_beta1_32 * real(residual, real32)
            v32 = beta2_32 * state%adam_v_bias(token) + &
                  one_minus_beta2_32 * real(residual, real32) * real(residual, real32)
            state%adam_m_bias(token) = m32
            state%adam_v_bias(token) = v32
            proposal32 = model%output_bias(token) - &
                         scaled_rate_32 * m32 / (sqrt(v32) * inverse_step_scale_32 + adam_epsilon_32)
            model%output_bias(token) = max(-value_limit_32, min(value_limit_32, proposal32))
         end do
         !$omp end parallel do
      else
         ! Legacy SGD update, preserved with its original operation order for
         ! pre-v3 checkpoint reproducibility (scheduled rate only replaces
         ! the constant peak; the constant schedule returns it unchanged).
         scale = rate * clip_scale
         !$omp parallel do private(residual, proposal, feature) schedule(static) if(vocabulary_size >= 512_int32)
         do token = 1_int32, vocabulary_size
            residual = probabilities(token) - smooth_mass
            if (token == target) residual = residual - (1.0_real64 - config%label_smoothing)
            !$omp simd private(proposal)
            do feature = 1_int32, width
               proposal = real(model%token_embedding(feature, token), real64) - &
                          scale * residual * real(hidden(feature), real64)
               proposal = max(-config%embedding_value_limit, min(config%embedding_value_limit, proposal))
               model%token_embedding(feature, token) = real(proposal, real32)
            end do
            proposal = real(model%output_bias(token), real64) - scale * residual
            proposal = max(-config%embedding_value_limit, min(config%embedding_value_limit, proposal))
            model%output_bias(token) = real(proposal, real32)
         end do
         !$omp end parallel do
      end if
      ! Cheap sentinel: the clamped target column must be finite; see the
      ! boundedness argument in the header comment for the full matrix.
      if (.not. all(ieee_is_finite(model%token_embedding(:, target))) .or. &
          .not. ieee_is_finite(model%output_bias(target))) then
         status = NPT_ERR_NUMERICAL
      end if
   end subroutine tied_output_update

   ! One mini-batch tied-output update.  GRADIENT_ACCUMULATION per-token
   ! gradients, all evaluated at the pre-step parameters, are averaged and
   ! applied as a single optimizer step -- ordinary mini-batch SGD/AdamW.  The
   ! average (not the sum) is what keeps the effective step size independent
   ! of how many tokens a record happened to contribute.
   !
   ! The motivation is measured, not theoretical: at the production shape
   ! (vocabulary 4096, width 96) the per-token path streams the tied
   ! embedding, both AdamW moments and the two bias moments once per
   ! prediction token, about 17 MB of L3-resident traffic per token.  That
   ! dominated the epoch wall clock (~80% of it).  Batching streams the same
   ! state once per B tokens, so the update cost per token falls by roughly a
   ! factor of B while the arithmetic done is identical.
   !
   ! Only the batch column's own state is touched, so the OpenMP loop over
   ! vocabulary columns needs no reduction and no critical section.
   subroutine batched_tied_output_update(model, residuals, hidden_batch, clip_batch, targets, batch_count, &
                                         config, state, status)
      type(neural_decoder_t), intent(inout) :: model
      ! (vocabulary, capacity): column b holds the unsmoothed-mean residual of
      ! staged token b, i.e. softmax - label_smoothing/V, minus
      ! (1 - label_smoothing) at that token's target.
      ! CONTIGUOUS keeps the column axpy vectorizable; without it gfortran
      ! assumes an arbitrary stride and emits a scalar loop.
      real(real32), contiguous, intent(in) :: residuals(:, :)
      ! (capacity, width): one contiguous hidden row per staged token.  The
      ! transposed layout is what makes the batch reduction below a vector axpy
      ! over features; with (width, capacity) every read of a batch member's
      ! hidden vector was a stride-width gather, which measured ~20x slower.
      real(real32), contiguous, intent(in) :: hidden_batch(:, :)
      real(real64), contiguous, intent(in) :: clip_batch(:)
      integer(int32), contiguous, intent(in) :: targets(:)
      integer(int32), intent(in) :: batch_count
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(inout) :: state
      integer(int32), intent(out) :: status
      integer(int32) :: vocabulary_size, width, token, feature, batch_index
      integer(int64) :: update_index
      real(real64) :: rate, inverse_batch, gradient, proposal, scale, value_limit
      real(real64) :: beta1, beta2, one_minus_beta1, one_minus_beta2, adam_epsilon
      real(real64) :: beta1_pow, beta2_pow, m_hat_factor, adam_step_scale, decay_factor
      real(real64) :: scaled_rate, inverse_step_scale
      real(real32) :: proposal32, m32, v32, gradient32, inverse_batch_32
      real(real32) :: beta1_32, one_minus_beta1_32, beta2_32, one_minus_beta2_32, adam_epsilon_32
      real(real32) :: scaled_rate_32, inverse_step_scale_32, decay_factor_32, value_limit_32, scale_32
      ! One width-sized accumulator per vocabulary column, private per thread.
      real(real32) :: column_gradient(size(hidden_batch, 2)), column_bias_gradient
      ! Bounded by construction: the staging capacity is capped at
      ! NPT_MAX_GRADIENT_ACCUMULATION, so this block is O(width*512) reals.
      real(real32) :: clipped_hidden(size(hidden_batch, 1), size(hidden_batch, 2))

      status = NPT_OK
      vocabulary_size = model%config%vocabulary_size
      width = model%config%width
      if (batch_count < 1_int32 .or. batch_count > size(residuals, 2, kind=int32) .or. &
          batch_count > size(hidden_batch, 1, kind=int32) .or. batch_count > size(clip_batch, kind=int32) .or. &
          batch_count > size(targets, kind=int32) .or. size(residuals, 1, kind=int32) /= vocabulary_size .or. &
          size(hidden_batch, 2, kind=int32) /= width) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      update_index = state%updates_completed + 1_int64
      rate = neural_scheduled_learning_rate(config, update_index)
      inverse_batch = 1.0_real64 / real(batch_count, real64)
      value_limit = config%embedding_value_limit
      ! The per-token clip is folded into the staged hidden vectors: the
      ! embedding gradient is clip * residual * hidden and the output-bias
      ! gradient is the unclipped residual (legacy semantics), so the residual
      ! itself must stay untouched.
      do batch_index = 1_int32, batch_count
         !$omp simd
         do feature = 1_int32, width
            clipped_hidden(batch_index, feature) = real(clip_batch(batch_index), real32) * &
                                                   hidden_batch(batch_index, feature)
         end do
      end do

      if (config%optimizer == NPT_OPTIMIZER_ADAMW) then
         call ensure_optimizer_state(model, config, state, status)
         if (status /= NPT_OK) return
         if (.not. ieee_is_finite(rate) .or. rate <= 0.0_real64) then
            status = NPT_ERR_NUMERICAL
            return
         end if
         if (rate * config%weight_decay >= 1.0_real64) then
            status = NPT_ERR_INVALID_ARGUMENT
            return
         end if
         beta1_pow = config%adam_beta1 ** real(update_index, real64)
         beta2_pow = config%adam_beta2 ** real(update_index, real64)
         m_hat_factor = 1.0_real64 - beta1_pow
         adam_step_scale = sqrt(max(0.0_real64, 1.0_real64 - beta2_pow))
         if (.not. ieee_is_finite(m_hat_factor) .or. .not. ieee_is_finite(adam_step_scale) .or. &
             m_hat_factor <= 0.0_real64) then
            status = NPT_ERR_NUMERICAL
            return
         end if
         decay_factor = 1.0_real64 - rate * config%weight_decay
         beta1 = config%adam_beta1
         beta2 = config%adam_beta2
         one_minus_beta1 = 1.0_real64 - beta1
         one_minus_beta2 = 1.0_real64 - beta2
         adam_epsilon = config%adam_epsilon
         scaled_rate = rate / m_hat_factor
         inverse_step_scale = 1.0_real64 / adam_step_scale
         ! Same real32 element chain as the per-token path: the batch only
         ! amortizes memory traffic, the sqrt/divide arithmetic per element is
         ! unchanged, so the element precision is what decides the cost.
         beta1_32 = real(beta1, real32)
         one_minus_beta1_32 = real(one_minus_beta1, real32)
         beta2_32 = real(beta2, real32)
         one_minus_beta2_32 = real(one_minus_beta2, real32)
         adam_epsilon_32 = real(adam_epsilon, real32)
         value_limit_32 = real(value_limit, real32)
         scaled_rate_32 = real(scaled_rate, real32)
         inverse_step_scale_32 = real(inverse_step_scale, real32)
         decay_factor_32 = real(decay_factor, real32)
         inverse_batch_32 = real(inverse_batch, real32)
         ! Loop order matters: for one vocabulary column the batch reduction is
         ! a sequence of axpy updates over the width vector, so the inner loop
         ! runs over features (contiguous, 8-wide vectorizable) and each batch
         ! member contributes one broadcast scalar.  The previous order -- batch
         ! innermost per element -- was a chain of B dependent scalar FMAs per
         ! element, which is why the batched arm measured *slower* than the
         ! per-token arm once the element math became cheap.
         !$omp parallel do private(token, feature, batch_index, gradient, gradient32, m32, v32, proposal32, &
         !$omp column_gradient, column_bias_gradient) &
         !$omp schedule(static) if(vocabulary_size >= 512_int32)
         do token = 1_int32, vocabulary_size
            column_gradient = 0.0_real32
            do batch_index = 1_int32, batch_count
               gradient32 = real(residuals(token, batch_index), real32)
               !$omp simd
               do feature = 1_int32, width
                  column_gradient(feature) = column_gradient(feature) + &
                                             gradient32 * clipped_hidden(batch_index, feature)
               end do
            end do
            column_bias_gradient = 0.0_real32
            do batch_index = 1_int32, batch_count
               column_bias_gradient = column_bias_gradient + real(residuals(token, batch_index), real32)
            end do
            !$omp simd private(m32, v32, proposal32)
            do feature = 1_int32, width
               gradient32 = column_gradient(feature) * inverse_batch_32
               m32 = beta1_32 * state%adam_m_embedding(feature, token) + one_minus_beta1_32 * gradient32
               v32 = beta2_32 * state%adam_v_embedding(feature, token) + one_minus_beta2_32 * gradient32 * gradient32
               state%adam_m_embedding(feature, token) = m32
               state%adam_v_embedding(feature, token) = v32
               proposal32 = decay_factor_32 * model%token_embedding(feature, token) - &
                            scaled_rate_32 * m32 / (sqrt(v32) * inverse_step_scale_32 + adam_epsilon_32)
               model%token_embedding(feature, token) = max(-value_limit_32, min(value_limit_32, proposal32))
            end do
            gradient32 = column_bias_gradient * inverse_batch_32
            m32 = beta1_32 * state%adam_m_bias(token) + one_minus_beta1_32 * gradient32
            v32 = beta2_32 * state%adam_v_bias(token) + one_minus_beta2_32 * gradient32 * gradient32
            state%adam_m_bias(token) = m32
            state%adam_v_bias(token) = v32
            proposal32 = model%output_bias(token) - &
                         scaled_rate_32 * m32 / (sqrt(v32) * inverse_step_scale_32 + adam_epsilon_32)
            model%output_bias(token) = max(-value_limit_32, min(value_limit_32, proposal32))
         end do
         !$omp end parallel do
      else
         ! Legacy SGD: the batched form is the arithmetic mean of the per-token
         ! updates, accumulated in one pass over the tied embedding.
         scale = rate * inverse_batch
         scale_32 = real(scale, real32)
         value_limit_32 = real(value_limit, real32)
         ! Same column-accumulator order as the AdamW branch above: axpy over the
         ! width vector per batch member, then one pass over the column.
         !$omp parallel do private(token, feature, batch_index, gradient32, proposal32, &
         !$omp column_gradient, column_bias_gradient) &
         !$omp schedule(static) if(vocabulary_size >= 512_int32)
         do token = 1_int32, vocabulary_size
            column_gradient = 0.0_real32
            do batch_index = 1_int32, batch_count
               gradient32 = real(residuals(token, batch_index), real32)
               !$omp simd
               do feature = 1_int32, width
                  column_gradient(feature) = column_gradient(feature) + &
                                             gradient32 * clipped_hidden(batch_index, feature)
               end do
            end do
            column_bias_gradient = 0.0_real32
            do batch_index = 1_int32, batch_count
               column_bias_gradient = column_bias_gradient + real(residuals(token, batch_index), real32)
            end do
            !$omp simd private(proposal32)
            do feature = 1_int32, width
               proposal32 = model%token_embedding(feature, token) - scale_32 * column_gradient(feature)
               model%token_embedding(feature, token) = max(-value_limit_32, min(value_limit_32, proposal32))
            end do
            proposal32 = model%output_bias(token) - scale_32 * column_bias_gradient
            model%output_bias(token) = max(-value_limit_32, min(value_limit_32, proposal32))
         end do
         !$omp end parallel do
      end if
      ! Canary identical in spirit to the per-token sentinel: every touched
      ! column is clamped to +/-value_limit, so only the staged target columns
      ! need a final finiteness probe.
      do batch_index = 1_int32, batch_count
         token = targets(batch_index)
         if (token < 1_int32 .or. token > vocabulary_size) then
            status = NPT_ERR_INVALID_ARGUMENT
            return
         end if
         if (.not. all(ieee_is_finite(model%token_embedding(:, token))) .or. &
             .not. ieee_is_finite(model%output_bias(token))) then
            status = NPT_ERR_NUMERICAL
            return
         end if
      end do
   end subroutine batched_tied_output_update

   ! Records one prediction: the teacher-forced loss, the running NLL sums and
   ! the top-1/top-5 bookkeeping.  Both the per-token path and the resolving
   ! block path call it in token order, so a block of deferred predictions sums
   ! in exactly the order the per-token path would have used.
   subroutine observe_prediction(metrics, probabilities, target, loss, status)
      type(neural_train_metrics_t), intent(inout) :: metrics
      real(real64), intent(in) :: probabilities(:)
      integer(int32), intent(in) :: target
      real(real64), intent(out) :: loss
      integer(int32), intent(out) :: status
      integer(int32) :: target_rank
      real(real64) :: target_probability

      status = NPT_OK
      loss = -log(max(probabilities(target), tiny(1.0_real64)))
      if (.not. ieee_is_finite(loss)) then
         status = NPT_ERR_NUMERICAL
         return
      end if
      metrics%prediction_tokens = metrics%prediction_tokens + 1_int64
      metrics%cumulative_nll = metrics%cumulative_nll + loss
      ! Rank of the true token under the pre-update distribution.  Ties
      ! resolve to the lowest index, matching greedy argmax decoding.
      target_probability = probabilities(target)
      ! rank = #{p_c > p_t} + #{p_c == p_t, c < t}; the two counts are array
      ! comparisons over contiguous slices, which vectorize, instead of a
      ! scalar scan over the whole vocabulary for every prediction.
      target_rank = count(probabilities(target + 1_int32:) > target_probability) + &
                    count(probabilities(:target - 1_int32) >= target_probability)
      if (target_rank == 0_int32) metrics%top1_correct = metrics%top1_correct + 1_int64
      if (target_rank < 5_int32) metrics%top5_correct = metrics%top5_correct + 1_int64
   end subroutine observe_prediction

   ! Stages one prediction token's hidden state and target.  The residual vector
   ! is filled in later (stage_batch_residual) because its softmax needs the
   ! logits of the whole block, and those are computed in one pass over the tied
   ! embedding when the block is resolved.
   subroutine stage_pending_hidden(workspace, config, hidden, target, status)
      type(neural_train_workspace_t), intent(inout) :: workspace
      type(neural_train_config_t), intent(in) :: config
      real(real32), intent(in) :: hidden(:)
      integer(int32), intent(in) :: target
      integer(int32), intent(out) :: status
      integer(int32) :: slot, vocabulary_size
      real(real64) :: clip_scale
      logical :: norm_is_finite

      status = NPT_OK
      if (.not. allocated(workspace%batch_residuals) .or. .not. allocated(workspace%batch_hidden) .or. &
          .not. allocated(workspace%batch_targets) .or. &
          workspace%batch_capacity <= 1_int32 .or. workspace%batch_count >= workspace%batch_capacity .or. &
          size(hidden, kind=int32) /= size(workspace%batch_hidden, 2, kind=int32)) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      vocabulary_size = size(workspace%batch_residuals, 1, kind=int32)
      if (target < 1_int32 .or. target > vocabulary_size) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      call hidden_clip(hidden, config, clip_scale, norm_is_finite)
      if (.not. norm_is_finite) then
         status = NPT_ERR_NUMERICAL
         return
      end if
      slot = workspace%batch_count + 1_int32
      workspace%batch_hidden(slot, :) = hidden
      workspace%batch_clip(slot) = clip_scale
      workspace%batch_targets(slot) = target
      workspace%batch_count = slot
   end subroutine stage_pending_hidden

   ! Fills the residual column of one staged token.  The residual is stored
   ! unclipped because the output-bias gradient never sees the clip.
   subroutine stage_batch_residual(workspace, config, probabilities, target, slot, status)
      type(neural_train_workspace_t), intent(inout) :: workspace
      type(neural_train_config_t), intent(in) :: config
      real(real64), intent(in) :: probabilities(:)
      integer(int32), intent(in) :: target
      integer(int32), intent(in) :: slot
      integer(int32), intent(out) :: status
      integer(int32) :: vocabulary_size, token
      real(real64) :: smooth_mass, residual

      status = NPT_OK
      vocabulary_size = size(probabilities, kind=int32)
      if (.not. allocated(workspace%batch_residuals) .or. slot < 1_int32 .or. slot > workspace%batch_count .or. &
          vocabulary_size /= size(workspace%batch_residuals, 1, kind=int32) .or. &
          target < 1_int32 .or. target > vocabulary_size) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      smooth_mass = config%label_smoothing / real(vocabulary_size, real64)
      ! Columns of the staging buffer are disjoint, so this parallel fill stores
      ! independent elements and needs no reduction.
      !$omp parallel do private(token, residual) schedule(static) if(vocabulary_size >= 1024_int32)
      do token = 1_int32, vocabulary_size
         residual = probabilities(token) - smooth_mass
         if (token == target) residual = residual - (1.0_real64 - config%label_smoothing)
         workspace%batch_residuals(token, slot) = real(residual, real32)
      end do
      !$omp end parallel do
   end subroutine stage_batch_residual

   ! Turns a staged block into one optimizer step: project the block's logits in
   ! a single pass over the tied embedding, score every prediction in block
   ! order, fill the residual columns, then apply the averaged update.  Every
   ! quantity is computed from the pre-step parameters, exactly as the per-token
   ! path does; only the logits projection is shared across the block.
   subroutine resolve_train_batch(model, workspace, config, state, metrics, sequence_nll, status)
      type(neural_decoder_t), intent(inout) :: model
      type(neural_train_workspace_t), intent(inout) :: workspace
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(inout) :: state
      type(neural_train_metrics_t), intent(inout) :: metrics
      real(real64), intent(inout) :: sequence_nll
      integer(int32), intent(out) :: status
      integer(int32) :: count, slot, decoder_status
      real(real64) :: loss, phase_start

      status = NPT_OK
      count = workspace%batch_count
      if (count <= 0_int32) return
      if (.not. allocated(workspace%batch_logits)) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      phase_start = clock_seconds()
      call neural_decoder_project_logits(model, workspace%batch_hidden, count, workspace%batch_logits, decoder_status)
      metrics%project_seconds = metrics%project_seconds + (clock_seconds() - phase_start)
      if (decoder_status /= NRD_OK) then
         status = NPT_ERR_MODEL
         return
      end if
      phase_start = clock_seconds()
      do slot = 1_int32, count
         ! Row of the block: contiguous, so no staging copy is needed.
         call stable_softmax(workspace%batch_logits(slot, :), workspace%probabilities, status)
         if (status /= NPT_OK) return
         call observe_prediction(metrics, workspace%probabilities, workspace%batch_targets(slot), loss, status)
         if (status /= NPT_OK) return
         sequence_nll = sequence_nll + loss
         call stage_batch_residual(workspace, config, workspace%probabilities, workspace%batch_targets(slot), slot, status)
         if (status /= NPT_OK) return
      end do
      metrics%score_seconds = metrics%score_seconds + (clock_seconds() - phase_start)
      phase_start = clock_seconds()
      call batched_tied_output_update(model, workspace%batch_residuals, workspace%batch_hidden, &
                                      workspace%batch_clip, workspace%batch_targets, count, config, state, status)
      metrics%update_seconds = metrics%update_seconds + (clock_seconds() - phase_start)
      if (status /= NPT_OK) return
      state%updates_completed = state%updates_completed + 1_int64
      workspace%batch_count = 0_int32
   end subroutine resolve_train_batch

   subroutine process_token_sequence(model, tokens, token_count, config, state, metrics, update_model, workspace, status)
      type(neural_decoder_t), intent(inout) :: model
      integer(int32), intent(in) :: tokens(:)
      integer(int32), intent(in) :: token_count
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(inout) :: state
      type(neural_train_metrics_t), intent(inout) :: metrics
      logical, intent(in) :: update_model
      type(neural_train_workspace_t), intent(inout) :: workspace
      integer(int32), intent(out) :: status
      integer(int32) :: position, decoder_status, target
      integer(int32) :: batch_capacity
      real(real64) :: loss, sequence_nll, phase_start
      logical :: accumulating

      status = NPT_OK
      if (.not. neural_decoder_is_ready(model) .or. .not. neural_train_config_is_valid(config) .or. &
          token_count < 2_int32 .or. token_count > size(tokens, kind=int32) .or. &
          token_count > config%maximum_tokens_per_record) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      if (any(tokens(:token_count) < 1_int32) .or. any(tokens(:token_count) > model%config%vocabulary_size)) then
         status = NPT_ERR_INCOMPATIBLE
         return
      end if
      ! gradient_accumulation = 1 keeps the exact one-step-per-token path, so
      ! every existing contract (bit-identical continuation, per-token update
      ! counting) is untouched unless a caller opts into batching.
      batch_capacity = 1_int32
      if (update_model .and. config%gradient_accumulation > 1_int32) then
         ! Evaluation never stages a gradient, so it must not pay for the
         ! staging buffers (they are O(vocabulary * accumulation) bytes).
         batch_capacity = min(config%gradient_accumulation, config%maximum_tokens_per_record)
      end if
      accumulating = batch_capacity > 1_int32
      call workspace_ensure(workspace, model, batch_capacity, status)
      if (status /= NPT_OK) return
      sequence_nll = 0.0_real64
      ! Reset is allocation-free and zero-equivalent to a fresh state init,
      ! so chunked training remains bit-identical to the one-pass order.
      call neural_decoder_state_reset(workspace%decoder_state, model, decoder_status)
      if (decoder_status /= NRD_OK) then
         status = NPT_ERR_MODEL
         return
      end if
      do position = 1_int32, token_count - 1_int32
         target = tokens(position + 1_int32)
         if (accumulating) then
            ! Forward without the tied projection.  The projection of the whole
            ! block happens once, when the block is resolved, so the
            ! (vocabulary x width) embedding is streamed once per block instead
            ! of once per prediction token -- that read is the single largest
            ! memory stream of a step.
            phase_start = clock_seconds()
            call neural_decoder_step_hidden(model, workspace%decoder_state, tokens(position), &
                                            workspace%hidden, decoder_status)
            metrics%decode_seconds = metrics%decode_seconds + (clock_seconds() - phase_start)
            if (decoder_status /= NRD_OK) then
               status = NPT_ERR_MODEL
               exit
            end if
            call stage_pending_hidden(workspace, config, workspace%hidden, target, status)
            if (status /= NPT_OK) exit
            if (workspace%batch_count >= batch_capacity) then
               call resolve_train_batch(model, workspace, config, state, metrics, sequence_nll, status)
               if (status /= NPT_OK) exit
            end if
         else
            phase_start = clock_seconds()
            call neural_decoder_step(model, workspace%decoder_state, tokens(position), workspace%hidden, &
                                     workspace%logits, decoder_status)
            metrics%decode_seconds = metrics%decode_seconds + (clock_seconds() - phase_start)
            if (decoder_status /= NRD_OK) then
               status = NPT_ERR_MODEL
               exit
            end if
            phase_start = clock_seconds()
            call stable_softmax(workspace%logits, workspace%probabilities, status)
            if (status /= NPT_OK) exit
            call observe_prediction(metrics, workspace%probabilities, target, loss, status)
            metrics%score_seconds = metrics%score_seconds + (clock_seconds() - phase_start)
            if (status /= NPT_OK) exit
            sequence_nll = sequence_nll + loss
            if (update_model) then
               phase_start = clock_seconds()
               call tied_output_update(model, workspace%hidden, workspace%probabilities, target, config, state, status)
               metrics%update_seconds = metrics%update_seconds + (clock_seconds() - phase_start)
               if (status /= NPT_OK) exit
               state%updates_completed = state%updates_completed + 1_int64
            end if
         end if
      end do
      ! Flush the trailing partial batch of this record.  Doing it here (rather
      ! than across records) is what preserves chunk/resume bit-identity: every
      ! checkpoint boundary in process_utf8_file is a record boundary.
      if (status == NPT_OK .and. update_model .and. accumulating .and. workspace%batch_count > 0_int32) then
         call resolve_train_batch(model, workspace, config, state, metrics, sequence_nll, status)
      end if
      if (status /= NPT_OK) return
      metrics%records = metrics%records + 1_int64
      if (metrics%prediction_tokens > 0_int64) then
         metrics%mean_nll = metrics%cumulative_nll / real(metrics%prediction_tokens, real64)
      end if
      if (update_model) then
         state%records_completed = state%records_completed + 1_int64
         state%tokens_seen = state%tokens_seen + int(token_count, int64)
         ! Keep the state reduction per record, independent of the caller's
         ! cumulative reporting buffer, so checkpointed chunks preserve the
         ! same floating-point addition order as one-pass training.
         state%cumulative_nll = state%cumulative_nll + sequence_nll
         state%ready = .true.
      end if
   end subroutine process_token_sequence

   subroutine neural_train_token_sequence(model, tokens, token_count, config, state, metrics, status)
      type(neural_decoder_t), intent(inout) :: model
      integer(int32), intent(in) :: tokens(:)
      integer(int32), intent(in) :: token_count
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(inout) :: state
      type(neural_train_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      type(neural_train_workspace_t) :: workspace

      if (.not. state%ready) call neural_train_state_init(state)
      call process_token_sequence(model, tokens, token_count, config, state, metrics, .true., workspace, status)
      call workspace_destroy(workspace)
   end subroutine neural_train_token_sequence

   subroutine neural_evaluate_token_sequence(model, tokens, token_count, config, metrics, status)
      type(neural_decoder_t), intent(inout) :: model
      integer(int32), intent(in) :: tokens(:)
      integer(int32), intent(in) :: token_count
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      type(neural_train_state_t) :: ignored_state
      type(neural_train_workspace_t) :: workspace

      call neural_train_state_init(ignored_state)
      call process_token_sequence(model, tokens, token_count, config, ignored_state, metrics, .false., workspace, status)
      call workspace_destroy(workspace)
   end subroutine neural_evaluate_token_sequence

   ! Line-oriented UTF-8 adapter. It delegates every byte in each retained line
   ! to the byte-BPE codec. The fixed record cap is an explicit bounded-I/O
   ! contract: records too large for the caller-provided token buffer are
   ! counted as truncations rather than silently partially trained.
   subroutine process_utf8_file(model, bpe, path, config, state, metrics, status, update_model, skip_source_records, &
                                maximum_new_source_records)
      type(neural_decoder_t), intent(inout) :: model
      type(bpe_model_t), intent(in) :: bpe
      character(len=*), intent(in) :: path
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(inout) :: state
      type(neural_train_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      logical, intent(in) :: update_model
      ! Resume only after validating identical source provenance/order outside
      ! this low-level byte-stream routine.
      integer(int64), intent(in), optional :: skip_source_records
      ! Stop after this many newly consumed source records. Together with the
      ! explicit skip cursor this permits checkpointed continuous chunks.
      integer(int64), intent(in), optional :: maximum_new_source_records
      character(len=FILE_LINE_BYTES) :: line
      type(neural_train_workspace_t) :: workspace
      integer(int32), allocatable :: tokens(:)
      integer(int32) :: token_count, bpe_status, line_bytes, stream_status
      integer(int64) :: skipped_source_records, source_record, source_records_this_call, source_record_limit
      integer :: unit, io_status, allocation_status
      logical :: end_of_file, oversized

      status = NPT_OK
      skipped_source_records = 0_int64
      source_record_limit = huge(0_int64)
      if (present(skip_source_records)) skipped_source_records = skip_source_records
      if (present(maximum_new_source_records)) source_record_limit = maximum_new_source_records
      if (.not. state%ready) call neural_train_state_init(state)
      if (.not. neural_decoder_is_ready(model) .or. .not. bpe_is_initialized(bpe) .or. &
          .not. neural_train_config_is_valid(config) .or. len_trim(path) == 0 .or. &
          skipped_source_records < 0_int64 .or. source_record_limit < 1_int64 .or. &
          model%config%vocabulary_size < bpe%token_count) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      allocate(tokens(config%maximum_tokens_per_record), stat=allocation_status)
      if (allocation_status /= 0) then
         status = NPT_ERR_ALLOCATION
         return
      end if
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         deallocate(tokens)
         status = NPT_ERR_IO
         return
      end if
      source_record = 0_int64
      source_records_this_call = 0_int64
      do
         if (source_records_this_call >= source_record_limit) exit
         call utf8_stream_read_record(unit, line, line_bytes, end_of_file, oversized, stream_status)
         if (stream_status /= UTF8_STREAM_OK) then
            close(unit)
            deallocate(tokens)
            status = NPT_ERR_IO
            return
         end if
         if (end_of_file) exit
         source_record = source_record + 1_int64
         if (source_record <= skipped_source_records) cycle
         source_records_this_call = source_records_this_call + 1_int64
         state%source_records_completed = state%source_records_completed + 1_int64
         if (oversized .or. line_bytes > BPE_MAX_INPUT_BYTES) then
            metrics%tokenizer_truncations = metrics%tokenizer_truncations + 1_int64
            metrics%skipped_records = metrics%skipped_records + 1_int64
            cycle
         end if
         if (line_bytes < 2_int32) then
            metrics%skipped_records = metrics%skipped_records + 1_int64
            cycle
         end if
         call bpe_encode(bpe, line, line_bytes, tokens, token_count, bpe_status)
         if (bpe_status == BPE_ERR_OUTPUT) then
            metrics%tokenizer_truncations = metrics%tokenizer_truncations + 1_int64
            metrics%skipped_records = metrics%skipped_records + 1_int64
            cycle
         end if
         if (bpe_status /= BPE_OK) then
            close(unit)
            deallocate(tokens)
            call workspace_destroy(workspace)
            status = NPT_ERR_TOKENIZER
            return
         end if
         if (token_count < 2_int32) then
            metrics%skipped_records = metrics%skipped_records + 1_int64
            cycle
         end if
         call process_token_sequence(model, tokens, token_count, config, state, metrics, update_model, workspace, status)
         if (status /= NPT_OK) then
            close(unit)
            deallocate(tokens)
            call workspace_destroy(workspace)
            return
         end if
      end do
      if (source_record < skipped_source_records) then
         close(unit)
         deallocate(tokens)
         call workspace_destroy(workspace)
         status = NPT_ERR_INCOMPATIBLE
         return
      end if
      close(unit, iostat=io_status)
      deallocate(tokens)
      call workspace_destroy(workspace)
      if (io_status /= 0) status = NPT_ERR_IO
   end subroutine process_utf8_file

   subroutine neural_train_utf8_file(model, bpe, path, config, state, metrics, status, skip_source_records, &
                                     maximum_new_source_records)
      type(neural_decoder_t), intent(inout) :: model
      type(bpe_model_t), intent(in) :: bpe
      character(len=*), intent(in) :: path
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(inout) :: state
      type(neural_train_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      integer(int64), intent(in), optional :: skip_source_records
      integer(int64), intent(in), optional :: maximum_new_source_records

      call process_utf8_file(model, bpe, path, config, state, metrics, status, .true., &
                             skip_source_records, maximum_new_source_records)
   end subroutine neural_train_utf8_file

   ! No-update held-out evaluation over the same line/BPE adapter.  The
   ! training state and source cursor are never advanced; only metrics move.
   subroutine neural_evaluate_utf8_file(model, bpe, path, config, metrics, status, maximum_source_records)
      type(neural_decoder_t), intent(inout) :: model
      type(bpe_model_t), intent(in) :: bpe
      character(len=*), intent(in) :: path
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      integer(int64), intent(in), optional :: maximum_source_records
      type(neural_train_state_t) :: ignored_state

      call neural_train_state_init(ignored_state)
      call process_utf8_file(model, bpe, path, config, ignored_state, metrics, status, .false., &
                             maximum_new_source_records=maximum_source_records)
   end subroutine neural_evaluate_utf8_file

   subroutine write_model_payload(unit, model, io_status)
      integer, intent(in) :: unit
      type(neural_decoder_t), intent(in) :: model
      integer, intent(inout) :: io_status

      if (io_status == 0) write(unit, iostat=io_status) model%token_embedding
      if (io_status == 0) write(unit, iostat=io_status) model%output_bias
      if (io_status == 0) write(unit, iostat=io_status) model%attention_rms
      if (io_status == 0) write(unit, iostat=io_status) model%ffn_rms
      if (io_status == 0) write(unit, iostat=io_status) model%query
      if (io_status == 0) write(unit, iostat=io_status) model%key
      if (io_status == 0) write(unit, iostat=io_status) model%value
      if (io_status == 0) write(unit, iostat=io_status) model%attention_output
      if (io_status == 0) write(unit, iostat=io_status) model%ffn_gate
      if (io_status == 0) write(unit, iostat=io_status) model%ffn_value
      if (io_status == 0) write(unit, iostat=io_status) model%ffn_output
      if (io_status == 0) write(unit, iostat=io_status) model%global_mix_logit
   end subroutine write_model_payload

   subroutine read_model_payload(unit, model, io_status)
      integer, intent(in) :: unit
      type(neural_decoder_t), intent(inout) :: model
      integer, intent(inout) :: io_status

      if (io_status == 0) read(unit, iostat=io_status) model%token_embedding
      if (io_status == 0) read(unit, iostat=io_status) model%output_bias
      if (io_status == 0) read(unit, iostat=io_status) model%attention_rms
      if (io_status == 0) read(unit, iostat=io_status) model%ffn_rms
      if (io_status == 0) read(unit, iostat=io_status) model%query
      if (io_status == 0) read(unit, iostat=io_status) model%key
      if (io_status == 0) read(unit, iostat=io_status) model%value
      if (io_status == 0) read(unit, iostat=io_status) model%attention_output
      if (io_status == 0) read(unit, iostat=io_status) model%ffn_gate
      if (io_status == 0) read(unit, iostat=io_status) model%ffn_value
      if (io_status == 0) read(unit, iostat=io_status) model%ffn_output
      if (io_status == 0) read(unit, iostat=io_status) model%global_mix_logit
   end subroutine read_model_payload

   pure logical function model_payload_is_finite(model)
      type(neural_decoder_t), intent(in) :: model

      model_payload_is_finite = all(ieee_is_finite(model%token_embedding)) .and. &
                                all(ieee_is_finite(model%output_bias)) .and. all(ieee_is_finite(model%attention_rms)) .and. &
                                all(ieee_is_finite(model%ffn_rms)) .and. all(ieee_is_finite(model%query)) .and. &
                                all(ieee_is_finite(model%key)) .and. all(ieee_is_finite(model%value)) .and. &
                                all(ieee_is_finite(model%attention_output)) .and. all(ieee_is_finite(model%ffn_gate)) .and. &
                                all(ieee_is_finite(model%ffn_value)) .and. all(ieee_is_finite(model%ffn_output)) .and. &
                                all(ieee_is_finite(model%global_mix_logit))
   end function model_payload_is_finite

   pure logical function optimizer_state_is_present(state)
      type(neural_train_state_t), intent(in) :: state

      optimizer_state_is_present = allocated(state%adam_m_embedding) .and. allocated(state%adam_v_embedding) .and. &
                                   allocated(state%adam_m_bias) .and. allocated(state%adam_v_bias)
   end function optimizer_state_is_present

   pure logical function optimizer_state_is_finite(state)
      type(neural_train_state_t), intent(in) :: state

      optimizer_state_is_finite = .true.
      if (allocated(state%adam_m_embedding)) optimizer_state_is_finite = &
         optimizer_state_is_finite .and. all(ieee_is_finite(state%adam_m_embedding))
      if (allocated(state%adam_v_embedding)) optimizer_state_is_finite = &
         optimizer_state_is_finite .and. all(ieee_is_finite(state%adam_v_embedding)) .and. &
         all(state%adam_v_embedding >= 0.0_real32)
      if (allocated(state%adam_m_bias)) optimizer_state_is_finite = &
         optimizer_state_is_finite .and. all(ieee_is_finite(state%adam_m_bias))
      if (allocated(state%adam_v_bias)) optimizer_state_is_finite = &
         optimizer_state_is_finite .and. all(ieee_is_finite(state%adam_v_bias)) .and. &
         all(state%adam_v_bias >= 0.0_real32)
   end function optimizer_state_is_finite

   pure logical function optimizer_state_matches(state, model)
      type(neural_train_state_t), intent(in) :: state
      type(neural_decoder_t), intent(in) :: model

      optimizer_state_matches = .true.
      if (allocated(state%adam_m_embedding)) then
         if (.not. optimizer_state_is_present(state)) then
            optimizer_state_matches = .false.
         else if (size(state%adam_m_embedding, 1, kind=int32) /= model%config%width .or. &
                  size(state%adam_m_embedding, 2, kind=int32) /= model%config%vocabulary_size .or. &
                  size(state%adam_v_embedding, 1, kind=int32) /= model%config%width .or. &
                  size(state%adam_v_embedding, 2, kind=int32) /= model%config%vocabulary_size .or. &
                  size(state%adam_m_bias, kind=int32) /= model%config%vocabulary_size .or. &
                  size(state%adam_v_bias, kind=int32) /= model%config%vocabulary_size) then
            optimizer_state_matches = .false.
         end if
      end if
   end function optimizer_state_matches

   ! Checkpoints include every mutable neural parameter and the update cursor.
   ! Exact continuation additionally requires replaying the same source records
   ! in the same order after state%source_records_completed; that data-cursor contract
   ! is intentionally explicit rather than inferred from a filename.
   subroutine neural_checkpoint_save(path, model, bpe, config, state, status)
      character(len=*), intent(in) :: path
      type(neural_decoder_t), intent(in) :: model
      type(bpe_model_t), intent(in) :: bpe
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(in) :: state
      integer(int32), intent(out) :: status
      integer :: unit, io_status, write_status

      status = NPT_OK
      if (len_trim(path) == 0 .or. .not. neural_decoder_is_ready(model) .or. .not. bpe_is_initialized(bpe) .or. &
          .not. neural_train_config_is_valid(config) .or. .not. state%ready .or. &
          model%config%vocabulary_size < bpe%token_count .or. .not. model_payload_is_finite(model) .or. &
          .not. optimizer_state_is_finite(state) .or. .not. optimizer_state_matches(state, model)) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      open(newunit=unit, file=trim(path), status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = NPT_ERR_IO
         return
      end if
      write(unit, iostat=io_status) NEURAL_CHECKPOINT_MAGIC
      if (io_status == 0) write(unit, iostat=io_status) NEURAL_CHECKPOINT_VERSION
      if (io_status == 0) write(unit, iostat=io_status) model%config%vocabulary_size, model%config%width, model%config%heads, &
                                                         model%config%local_window, model%config%ffn_width, &
                                                         model%config%memory_decay, model%config%rms_epsilon, &
                                                         model%config%rotary_period
      if (io_status == 0) write(unit, iostat=io_status) bpe%token_count, bpe%merge_count
      if (io_status == 0) write(unit, iostat=io_status) config%trainable_mode, config%learning_rate, config%gradient_clip_norm, &
                                                         config%embedding_value_limit, config%maximum_tokens_per_record
      if (io_status == 0) write(unit, iostat=io_status) config%optimizer, config%adam_beta1, config%adam_beta2, &
                                                         config%adam_epsilon, config%weight_decay, config%label_smoothing, &
                                                         config%lr_schedule, config%warmup_updates, config%total_updates, &
                                                         config%minimum_lr_ratio
      if (io_status == 0) write(unit, iostat=io_status) config%gradient_accumulation
      if (io_status == 0) write(unit, iostat=io_status) state%records_completed, state%source_records_completed, &
                                                         state%updates_completed, state%tokens_seen, state%cumulative_nll
      if (io_status == 0) then
         if (optimizer_state_is_present(state)) then
            write(unit, iostat=io_status) 1_int32
         else
            write(unit, iostat=io_status) 0_int32
         end if
      end if
      if (io_status == 0 .and. optimizer_state_is_present(state)) then
         write(unit, iostat=io_status) state%adam_m_embedding
         if (io_status == 0) write(unit, iostat=io_status) state%adam_v_embedding
         if (io_status == 0) write(unit, iostat=io_status) state%adam_m_bias
         if (io_status == 0) write(unit, iostat=io_status) state%adam_v_bias
      end if
      call write_model_payload(unit, model, io_status)
      write_status = io_status
      close(unit, iostat=io_status)
      if (write_status /= 0 .or. io_status /= 0) status = NPT_ERR_IO
   end subroutine neural_checkpoint_save

   subroutine neural_checkpoint_load(path, bpe, model, config, state, status)
      character(len=*), intent(in) :: path
      type(bpe_model_t), intent(in) :: bpe
      type(neural_decoder_t), intent(inout) :: model
      type(neural_train_config_t), intent(out) :: config
      type(neural_train_state_t), intent(out) :: state
      integer(int32), intent(out) :: status
      character(len=8) :: magic
      integer :: unit, io_status, read_status
      integer :: allocation_status
      integer(int32) :: version, stored_bpe_tokens, stored_bpe_merges, stored_optimizer_flag
      type(neural_decoder_config_t) :: decoder_config
      ! Only used to read a v3 (real64 moment) checkpoint.
      real(real64), allocatable :: m_wide(:, :), v_wide(:, :), m_bias_wide(:), v_bias_wide(:)

      call neural_decoder_destroy(model)
      config = neural_train_config_t()
      state = neural_train_state_t()
      status = NPT_OK
      if (len_trim(path) == 0 .or. .not. bpe_is_initialized(bpe)) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = NPT_ERR_IO
         return
      end if
      read(unit, iostat=io_status) magic
      if (io_status == 0) read(unit, iostat=io_status) version
      if (io_status == 0) read(unit, iostat=io_status) decoder_config%vocabulary_size, decoder_config%width, decoder_config%heads, &
                                                       decoder_config%local_window, decoder_config%ffn_width, &
                                                       decoder_config%memory_decay, decoder_config%rms_epsilon, &
                                                       decoder_config%rotary_period
      if (io_status == 0) read(unit, iostat=io_status) stored_bpe_tokens, stored_bpe_merges
      if (io_status == 0) read(unit, iostat=io_status) config%trainable_mode, config%learning_rate, config%gradient_clip_norm, &
                                                       config%embedding_value_limit, config%maximum_tokens_per_record
      stored_optimizer_flag = 0_int32
      if (io_status == 0) then
         if (version == NEURAL_CHECKPOINT_VERSION .or. version == NEURAL_CHECKPOINT_VERSION_V3) then
            read(unit, iostat=io_status) config%optimizer, config%adam_beta1, config%adam_beta2, &
                                         config%adam_epsilon, config%weight_decay, config%label_smoothing, &
                                         config%lr_schedule, config%warmup_updates, config%total_updates, &
                                         config%minimum_lr_ratio
            if (io_status == 0 .and. version == NEURAL_CHECKPOINT_VERSION) then
               read(unit, iostat=io_status) config%gradient_accumulation
            else if (io_status == 0) then
               ! v3 files were written by the one-step-per-token path.
               config%gradient_accumulation = 1_int32
            end if
         else if (version == NEURAL_CHECKPOINT_VERSION_V2) then
            ! v2 checkpoints were written by the SGD-only trainer; restore
            ! exactly those semantics instead of the v3 type defaults.
            config%optimizer = NPT_OPTIMIZER_SGD
            config%weight_decay = 0.0_real64
            config%label_smoothing = 0.0_real64
            config%lr_schedule = NPT_SCHEDULE_CONSTANT
            config%warmup_updates = 0_int64
            config%total_updates = 0_int64
         end if
      end if
      if (io_status == 0) read(unit, iostat=io_status) state%records_completed, state%source_records_completed, &
                                                       state%updates_completed, state%tokens_seen, state%cumulative_nll
      if (io_status == 0 .and. (version == NEURAL_CHECKPOINT_VERSION .or. &
                                 version == NEURAL_CHECKPOINT_VERSION_V3)) then
         read(unit, iostat=io_status) stored_optimizer_flag
      end if
      if (io_status /= 0) then
         close(unit)
         status = NPT_ERR_IO
         return
      end if
      ! Legacy SGD checkpoints may carry a non-zero, then ignored, weight decay.
      ! Normalize that field before validation so the restored configuration
      ! describes the update that will actually run: SGD applies no decoupled
      ! decay at all, and the trainer now rejects such a contradictory config.
      if (config%optimizer /= NPT_OPTIMIZER_ADAMW) config%weight_decay = 0.0_real64
      if (magic /= NEURAL_CHECKPOINT_MAGIC .or. &
          (version /= NEURAL_CHECKPOINT_VERSION .and. version /= NEURAL_CHECKPOINT_VERSION_V3 .and. &
           version /= NEURAL_CHECKPOINT_VERSION_V2) .or. &
          stored_bpe_tokens /= bpe%token_count .or. stored_bpe_merges /= bpe%merge_count .or. &
          .not. neural_decoder_config_is_valid(decoder_config) .or. .not. neural_train_config_is_valid(config) .or. &
          (stored_optimizer_flag /= 0_int32 .and. stored_optimizer_flag /= 1_int32) .or. &
          state%records_completed < 0_int64 .or. state%source_records_completed < 0_int64 .or. &
          state%updates_completed < 0_int64 .or. state%tokens_seen < 0_int64 .or. &
          .not. ieee_is_finite(state%cumulative_nll)) then
         close(unit)
         status = NPT_ERR_FORMAT
         return
      end if
      call neural_decoder_init(model, decoder_config, 1_int64, status)
      if (status /= NRD_OK) then
         close(unit)
         call neural_decoder_destroy(model)
         status = NPT_ERR_FORMAT
         return
      end if
      if (stored_optimizer_flag == 1_int32) then
         allocate(state%adam_m_embedding(decoder_config%width, decoder_config%vocabulary_size), &
                  state%adam_v_embedding(decoder_config%width, decoder_config%vocabulary_size), &
                  state%adam_m_bias(decoder_config%vocabulary_size), &
                  state%adam_v_bias(decoder_config%vocabulary_size), stat=allocation_status)
         if (allocation_status /= 0) then
            close(unit)
            call neural_decoder_destroy(model)
            state = neural_train_state_t()
            status = NPT_ERR_ALLOCATION
            return
         end if
         if (version == NEURAL_CHECKPOINT_VERSION) then
            ! v4 stores the moments in the parameter precision.
            read(unit, iostat=io_status) state%adam_m_embedding
            if (io_status == 0) read(unit, iostat=io_status) state%adam_v_embedding
            if (io_status == 0) read(unit, iostat=io_status) state%adam_m_bias
            if (io_status == 0) read(unit, iostat=io_status) state%adam_v_bias
         else
            ! v3 stored real64 moments; convert on load so an existing run stays
            ! resumable.  The conversion is the only precision change and it is
            ! monotone (real64 -> real32 rounds towards the nearest value).
            allocate(m_wide(decoder_config%width, decoder_config%vocabulary_size), &
                     v_wide(decoder_config%width, decoder_config%vocabulary_size), &
                     m_bias_wide(decoder_config%vocabulary_size), &
                     v_bias_wide(decoder_config%vocabulary_size), stat=allocation_status)
            if (allocation_status /= 0) then
               close(unit)
               call neural_decoder_destroy(model)
               state = neural_train_state_t()
               status = NPT_ERR_ALLOCATION
               return
            end if
            read(unit, iostat=io_status) m_wide
            if (io_status == 0) read(unit, iostat=io_status) v_wide
            if (io_status == 0) read(unit, iostat=io_status) m_bias_wide
            if (io_status == 0) read(unit, iostat=io_status) v_bias_wide
            if (io_status == 0) then
               state%adam_m_embedding = real(m_wide, real32)
               state%adam_v_embedding = real(v_wide, real32)
               state%adam_m_bias = real(m_bias_wide, real32)
               state%adam_v_bias = real(v_bias_wide, real32)
            end if
            deallocate(m_wide, v_wide, m_bias_wide, v_bias_wide)
         end if
         if (io_status /= 0 .or. .not. optimizer_state_is_finite(state)) then
            close(unit)
            call neural_decoder_destroy(model)
            state = neural_train_state_t()
            status = NPT_ERR_IO
            return
         end if
      end if
      call read_model_payload(unit, model, io_status)
      read_status = io_status
      close(unit, iostat=io_status)
      if (read_status /= 0 .or. io_status /= 0 .or. .not. model_payload_is_finite(model)) then
         call neural_decoder_destroy(model)
         state = neural_train_state_t()
         status = NPT_ERR_IO
         return
      end if
      state%ready = .true.
      status = NPT_OK
   end subroutine neural_checkpoint_load

end module neural_pretraining
