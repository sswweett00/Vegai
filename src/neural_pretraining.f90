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
!  auditable and creates a stable native checkpoint boundary for future
!  gradient-model work outside this change.
!
!  The current training path uses the single-state Apex update with
!  checkpointed momentum, warmup/inverse-square-root and warmup/cosine
!  learning-rate schedules, label smoothing, top-1/top-5
!  next-token metrics, a no-update file evaluation path, and a reusable
!  per-file workspace (the previous implementation reallocated the decoder
!  state per record and scan-checked the full embedding matrix for
!  finiteness on every token).  None of this changes the frozen-core
!  boundary above.
!==============================================================================
module neural_pretraining
   use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use apex_optimizer, only: apex_config_t, apex_factors_t, apex_error_t, apex_make_factors, &
                             apex_update_column, apex_update_scalar, APEX_OK
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

   ! Learning-rate schedule identifiers.  The schedule is a pure function of
   ! the persisted update counter, so a checkpoint/resume boundary cannot
   ! change the effective learning rate of any update.
   integer(int32), parameter, public :: NPT_SCHEDULE_CONSTANT = 0_int32
   integer(int32), parameter, public :: NPT_SCHEDULE_WARMUP_INVSQRT = 1_int32
   integer(int32), parameter, public :: NPT_SCHEDULE_WARMUP_COSINE = 2_int32

   character(len=8), parameter :: NEURAL_CHECKPOINT_MAGIC = 'VGNR0001'
   ! Version 4 stores the Apex configuration and its single momentum state.
   ! Earlier checkpoint formats are intentionally rejected after this breaking
   ! change; a resumed stream retains exact state and update-counter semantics.
   integer(int32), parameter :: NEURAL_CHECKPOINT_VERSION = 4_int32
   integer(int32), parameter :: FILE_LINE_BYTES = 32768_int32

   type, public :: neural_train_config_t
      integer(int32) :: trainable_mode = NPT_TRAINABLE_TIED_OUTPUT
      ! Peak step size for the exact tied-output cross-entropy gradient.
      real(real64) :: learning_rate = 0.01_real64
      ! A conservative norm bound on the hidden vector used by the update.
      real(real64) :: gradient_clip_norm = 4.0_real64
      ! Per-value safety envelope for the mutable tied embedding matrix.
      real(real64) :: embedding_value_limit = 8.0_real64
      integer(int32) :: maximum_tokens_per_record = 8192_int32
      ! Apex phase-coherent update controls; beta2 sets gain/damping response.
      real(real64) :: apex_beta1 = 0.9_real64
      real(real64) :: apex_beta2 = 0.5_real64
      real(real64) :: apex_epsilon = 1.0e-8_real64
      real(real64) :: apex_gamma = 0.5_real64
      real(real64) :: weight_decay = 0.01_real64
      ! Label-smoothing mass in [0, 0.5).  Reported NLL stays unsmoothed.
      real(real64) :: label_smoothing = 0.0_real64
      integer(int32) :: lr_schedule = NPT_SCHEDULE_CONSTANT
      integer(int64) :: warmup_updates = 0_int64
      ! Cosine horizon in updates; required (> warmup) for the cosine mode.
      integer(int64) :: total_updates = 0_int64
      ! Cosine floor as a fraction of the peak learning rate, in [0, 1].
      real(real64) :: minimum_lr_ratio = 0.1_real64
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
      ! Single Apex momentum state, lazily allocated and serialized in v4.
      real(real64), allocatable :: apex_m_embedding(:, :)
      real(real64), allocatable :: apex_m_bias(:)
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
      ! One real64 Apex momentum array for embeddings and bias.
      integer(int64) :: apex_state_bytes_real64 = 0_int64
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
   end type neural_train_workspace_t

   public :: neural_train_config_is_valid, neural_train_state_init, neural_train_metrics_reset
   public :: neural_train_token_sequence, neural_evaluate_token_sequence, neural_train_utf8_file
   public :: neural_evaluate_utf8_file, neural_scheduled_learning_rate
   public :: neural_training_resource_plan, neural_checkpoint_save, neural_checkpoint_load

contains

   pure logical function neural_train_config_is_valid(config)
      type(neural_train_config_t), intent(in) :: config

      neural_train_config_is_valid = config%trainable_mode == NPT_TRAINABLE_TIED_OUTPUT .and. &
                                     ieee_is_finite(config%learning_rate) .and. config%learning_rate > 0.0_real64 .and. &
                                     ieee_is_finite(config%gradient_clip_norm) .and. config%gradient_clip_norm > 0.0_real64 .and. &
                                     ieee_is_finite(config%embedding_value_limit) .and. &
                                     config%embedding_value_limit > 0.0_real64 .and. &
                                     config%maximum_tokens_per_record >= 2_int32 .and. &
                                     ieee_is_finite(config%apex_beta1) .and. config%apex_beta1 >= 0.0_real64 .and. &
                                     config%apex_beta1 < 1.0_real64 .and. &
                                     ieee_is_finite(config%apex_beta2) .and. config%apex_beta2 >= 0.0_real64 .and. &
                                     config%apex_beta2 < 1.0_real64 .and. &
                                     ieee_is_finite(config%apex_epsilon) .and. config%apex_epsilon > 0.0_real64 .and. &
                                     ieee_is_finite(config%apex_gamma) .and. config%apex_gamma >= 0.0_real64 .and. &
                                     ieee_is_finite(config%weight_decay) .and. config%weight_decay >= 0.0_real64 .and. &
                                     ieee_is_finite(config%label_smoothing) .and. &
                                     config%label_smoothing >= 0.0_real64 .and. config%label_smoothing < 0.5_real64 .and. &
                                     (config%lr_schedule == NPT_SCHEDULE_CONSTANT .or. &
                                      config%lr_schedule == NPT_SCHEDULE_WARMUP_INVSQRT .or. &
                                      config%lr_schedule == NPT_SCHEDULE_WARMUP_COSINE) .and. &
                                     config%warmup_updates >= 0_int64 .and. config%total_updates >= 0_int64 .and. &
                                     ieee_is_finite(config%learning_rate * config%gradient_clip_norm) .and. &
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
      integer(int64) :: width, vocabulary, ffn_width

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
      plan%apex_state_bytes_real64 = 8_int64 * (width * vocabulary + vocabulary)
      plan%minimum_resident_bytes = plan%decoder_parameter_bytes_real32 + plan%tokenizer_bytes + &
                                    plan%decoder_state_bytes_real32 + plan%record_token_buffer_bytes + &
                                    plan%per_step_working_bytes + plan%apex_state_bytes_real64
   end subroutine neural_training_resource_plan

   subroutine stable_softmax(logits, probabilities, status)
      real(real32), intent(in) :: logits(:)
      real(real64), intent(out) :: probabilities(:)
      integer(int32), intent(out) :: status
      real(real64) :: maximum, normalizer, logit_value
      integer(int32) :: index

      status = NPT_OK
      if (size(logits, kind=int32) <= 0_int32 .or. size(probabilities, kind=int32) /= size(logits, kind=int32)) then
         status = NPT_ERR_NUMERICAL
         return
      end if
      ! Fuse the finite-value validation with the maximum search. The prior
      ! ALL(IEEE_IS_FINITE(logits)) plus MAXVAL(REAL(logits, real64)) did two
      ! complete passes and materialized a converted temporary over the whole
      ! vocabulary before exponentiation.
      maximum = -huge(1.0_real64)
      do index = 1_int32, size(logits, kind=int32)
         if (.not. ieee_is_finite(logits(index))) then
            status = NPT_ERR_NUMERICAL
            return
         end if
         logit_value = real(logits(index), real64)
         maximum = max(maximum, logit_value)
      end do
      ! Exponentials are independent per token, so the optional parallel
      ! fill stores bit-identical values; the normalizer then accumulates
      ! in the same ascending order as the sequential form, keeping the
      ! resulting distribution bit-identical at any thread count.
      !$omp parallel do schedule(static) if(size(logits) >= 1024)
      do index = 1_int32, size(logits, kind=int32)
         probabilities(index) = exp(real(logits(index), real64) - maximum)
      end do
      !$omp end parallel do
      normalizer = 0.0_real64
      do index = 1_int32, size(logits, kind=int32)
         normalizer = normalizer + probabilities(index)
      end do
      if (.not. ieee_is_finite(normalizer) .or. normalizer <= tiny(1.0_real64)) then
         status = NPT_ERR_NUMERICAL
         return
      end if
      probabilities = probabilities / normalizer
      ! Every numerator is exp(logit - maximum) with finite logits and hence
      ! lies in [0, 1]; normalizer is finite and at least one, so this divide
      ! cannot produce a non-finite probability. Avoid another vocabulary scan.
   end subroutine stable_softmax

   subroutine ensure_apex_state(model, state, status)
      type(neural_decoder_t), intent(in) :: model
      type(neural_train_state_t), intent(inout) :: state
      integer(int32), intent(out) :: status
      integer :: allocation_status

      status = NPT_OK
      if (allocated(state%apex_m_embedding)) then
         if (.not. allocated(state%apex_m_bias)) then
            status = NPT_ERR_INCOMPATIBLE
            return
         end if
         if (size(state%apex_m_embedding, 1, kind=int32) /= model%config%width .or. &
             size(state%apex_m_embedding, 2, kind=int32) /= model%config%vocabulary_size .or. &
             size(state%apex_m_bias, kind=int32) /= model%config%vocabulary_size) then
            status = NPT_ERR_INCOMPATIBLE
         end if
         return
      end if
      allocate(state%apex_m_embedding(model%config%width, model%config%vocabulary_size), &
               state%apex_m_bias(model%config%vocabulary_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = NPT_ERR_ALLOCATION
         return
      end if
      state%apex_m_embedding = 0.0_real64
      state%apex_m_bias = 0.0_real64
   end subroutine ensure_apex_state

   subroutine workspace_ensure(workspace, model, status)
      type(neural_train_workspace_t), intent(inout) :: workspace
      type(neural_decoder_t), intent(in) :: model
      integer(int32), intent(out) :: status
      integer :: allocation_status
      integer(int32) :: decoder_status

      status = NPT_OK
      if (workspace%ready) then
         if (size(workspace%hidden, kind=int32) == model%config%width .and. &
             size(workspace%logits, kind=int32) == model%config%vocabulary_size) return
         call workspace_destroy(workspace)
      end if
      allocate(workspace%hidden(model%config%width), workspace%logits(model%config%vocabulary_size), &
               workspace%probabilities(model%config%vocabulary_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = NPT_ERR_ALLOCATION
         return
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
      workspace%ready = .false.
   end subroutine workspace_destroy

   ! One exact tied-output cross-entropy update.  neural_decoder_step uses
   ! logits = E^T h + b, so dL/dE(:,v) = residual(v) * h and dL/db(v) =
   ! residual(v), where residual = softmax - smoothed_target.  The hidden
   ! vector is norm-clipped before forming the gradient (legacy semantics).
   !
   ! The Apex update uses one finite momentum state per trainable parameter.
   ! Hidden-vector clipping bounds the embedding gradient; every weight remains
   ! inside the configured value envelope and checkpoint payloads are verified.
   subroutine tied_output_update(model, hidden, probabilities, target, config, state, status)
      type(neural_decoder_t), intent(inout) :: model
      real(real32), intent(in) :: hidden(:)
      real(real64), intent(in) :: probabilities(:)
      integer(int32), intent(in) :: target
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(inout) :: state
      integer(int32), intent(out) :: status
      integer(int32) :: vocabulary_size, width, token
      integer(int64) :: update_index
      real(real64) :: hidden_norm, clip_scale, residual, smooth_mass, scaled_residual
      real(real64) :: value_limit
      type(apex_config_t) :: apex_config
      type(apex_factors_t) :: factors, bias_factors
      type(apex_error_t) :: apex_error

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
      hidden_norm = sqrt(sum(real(hidden, real64) * real(hidden, real64)))
      if (.not. ieee_is_finite(hidden_norm)) then
         status = NPT_ERR_NUMERICAL
         return
      end if
      clip_scale = min(1.0_real64, config%gradient_clip_norm / max(hidden_norm, tiny(1.0_real64)))
      update_index = state%updates_completed + 1_int64
      call ensure_apex_state(model, state, status)
      if (status /= NPT_OK) return
      apex_config%learning_rate = config%learning_rate
      apex_config%beta1 = config%apex_beta1
      apex_config%beta2 = config%apex_beta2
      apex_config%epsilon = config%apex_epsilon
      apex_config%gamma = config%apex_gamma
      apex_config%weight_decay = config%weight_decay
      apex_config%gradient_clip = config%gradient_clip_norm
      apex_config%warmup_updates = config%warmup_updates
      apex_config%total_updates = config%total_updates
      apex_config%minimum_lr_ratio = config%minimum_lr_ratio
      apex_config%scheduler = int(config%lr_schedule, kind(apex_config%scheduler))
      call apex_make_factors(apex_config, update_index, factors, apex_error)
      if (apex_error%code /= APEX_OK) then
         status = NPT_ERR_NUMERICAL
         return
      end if
      bias_factors = factors
      bias_factors%decay_factor = 1.0_real64
      value_limit = config%embedding_value_limit
      smooth_mass = config%label_smoothing / real(vocabulary_size, real64)

      ! The embedding columns are disjoint across tokens. The single Apex
      ! momentum recurrence is applied directly to each contiguous column.
      !$omp parallel do private(residual, scaled_residual, apex_error) reduction(min:status) &
      !$omp& schedule(static) if(vocabulary_size >= 512_int32)
      do token = 1_int32, vocabulary_size
         residual = probabilities(token) - smooth_mass
         if (token == target) residual = residual - (1.0_real64 - config%label_smoothing)
         scaled_residual = residual * clip_scale
         call apex_update_column(model%token_embedding(:, token), hidden, state%apex_m_embedding(:, token), &
                                 scaled_residual, factors, apex_error)
         if (apex_error%code /= APEX_OK) then
            status = NPT_ERR_NUMERICAL
            cycle
         end if
         model%token_embedding(:, token) = max(-value_limit, min(value_limit, model%token_embedding(:, token)))
         call apex_update_scalar(model%output_bias(token), residual, state%apex_m_bias(token), bias_factors, apex_error)
         if (apex_error%code /= APEX_OK) status = NPT_ERR_NUMERICAL
      end do
      !$omp end parallel do
      if (status /= NPT_OK) return
      ! Cheap sentinel: the clamped target column must be finite; see the
      ! boundedness argument in the header comment for the full matrix.
      if (.not. all(ieee_is_finite(model%token_embedding(:, target))) .or. &
          .not. ieee_is_finite(model%output_bias(target))) then
         status = NPT_ERR_NUMERICAL
      end if
   end subroutine tied_output_update

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
      integer(int32) :: position, decoder_status, target, candidate, target_rank
      real(real64) :: loss, sequence_nll, target_probability

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
      call workspace_ensure(workspace, model, status)
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
         call neural_decoder_step(model, workspace%decoder_state, tokens(position), workspace%hidden, &
                                  workspace%logits, decoder_status)
         if (decoder_status /= NRD_OK) then
            status = NPT_ERR_MODEL
            exit
         end if
         call stable_softmax(workspace%logits, workspace%probabilities, status)
         if (status /= NPT_OK) exit
         target = tokens(position + 1_int32)
         loss = -log(max(workspace%probabilities(target), tiny(1.0_real64)))
         if (.not. ieee_is_finite(loss)) then
            status = NPT_ERR_NUMERICAL
            exit
         end if
         metrics%prediction_tokens = metrics%prediction_tokens + 1_int64
         metrics%cumulative_nll = metrics%cumulative_nll + loss
         sequence_nll = sequence_nll + loss
         ! Rank of the true token under the pre-update distribution.  Ties
         ! resolve to the lowest index, matching greedy argmax decoding.
         target_probability = workspace%probabilities(target)
         target_rank = 0_int32
         do candidate = 1_int32, model%config%vocabulary_size
            if (workspace%probabilities(candidate) > target_probability .or. &
                (workspace%probabilities(candidate) == target_probability .and. candidate < target)) then
               target_rank = target_rank + 1_int32
            end if
         end do
         if (target_rank == 0_int32) metrics%top1_correct = metrics%top1_correct + 1_int64
         if (target_rank < 5_int32) metrics%top5_correct = metrics%top5_correct + 1_int64
         if (update_model) then
            call tied_output_update(model, workspace%hidden, workspace%probabilities, target, config, state, status)
            if (status /= NPT_OK) exit
            state%updates_completed = state%updates_completed + 1_int64
         end if
      end do
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

   pure logical function apex_state_is_present(state)
      type(neural_train_state_t), intent(in) :: state
      apex_state_is_present = allocated(state%apex_m_embedding) .and. allocated(state%apex_m_bias)
   end function apex_state_is_present

   pure logical function apex_state_is_finite(state)
      type(neural_train_state_t), intent(in) :: state
      apex_state_is_finite = .true.
      if (allocated(state%apex_m_embedding)) apex_state_is_finite = &
         apex_state_is_finite .and. all(ieee_is_finite(state%apex_m_embedding))
      if (allocated(state%apex_m_bias)) apex_state_is_finite = &
         apex_state_is_finite .and. all(ieee_is_finite(state%apex_m_bias))
   end function apex_state_is_finite

   pure logical function apex_state_matches(state, model)
      type(neural_train_state_t), intent(in) :: state
      type(neural_decoder_t), intent(in) :: model
      apex_state_matches = .true.
      if (allocated(state%apex_m_embedding) .neqv. allocated(state%apex_m_bias)) then
         apex_state_matches = .false.
         return
      end if
      if (allocated(state%apex_m_embedding)) then
         if (size(state%apex_m_embedding, 1, kind=int32) /= model%config%width .or. &
             size(state%apex_m_embedding, 2, kind=int32) /= model%config%vocabulary_size .or. &
             size(state%apex_m_bias, kind=int32) /= model%config%vocabulary_size) then
            apex_state_matches = .false.
         end if
      end if
   end function apex_state_matches

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
          .not. apex_state_is_finite(state) .or. .not. apex_state_matches(state, model)) then
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
      if (io_status == 0) write(unit, iostat=io_status) config%apex_beta1, config%apex_beta2, &
                                                         config%apex_epsilon, config%apex_gamma, &
                                                         config%weight_decay, config%label_smoothing, config%lr_schedule, &
                                                         config%warmup_updates, config%total_updates, config%minimum_lr_ratio
      if (io_status == 0) write(unit, iostat=io_status) state%records_completed, state%source_records_completed, &
                                                         state%updates_completed, state%tokens_seen, state%cumulative_nll
      if (io_status == 0) then
         if (apex_state_is_present(state)) then
            write(unit, iostat=io_status) 1_int32
         else
            write(unit, iostat=io_status) 0_int32
         end if
      end if
      if (io_status == 0 .and. apex_state_is_present(state)) then
         write(unit, iostat=io_status) state%apex_m_embedding
         if (io_status == 0) write(unit, iostat=io_status) state%apex_m_bias
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
      integer :: unit, io_status, read_status, allocation_status
      integer(int32) :: version, stored_bpe_tokens, stored_bpe_merges, stored_apex_state
      type(neural_decoder_config_t) :: decoder_config

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
      if (io_status == 0) read(unit, iostat=io_status) decoder_config%vocabulary_size, decoder_config%width, &
         decoder_config%heads, decoder_config%local_window, decoder_config%ffn_width, &
         decoder_config%memory_decay, decoder_config%rms_epsilon, decoder_config%rotary_period
      if (io_status == 0) read(unit, iostat=io_status) stored_bpe_tokens, stored_bpe_merges
      if (io_status == 0) read(unit, iostat=io_status) config%trainable_mode, config%learning_rate, &
         config%gradient_clip_norm, config%embedding_value_limit, config%maximum_tokens_per_record
      if (io_status == 0) read(unit, iostat=io_status) config%apex_beta1, config%apex_beta2, &
         config%apex_epsilon, config%apex_gamma, config%weight_decay, config%label_smoothing, &
         config%lr_schedule, config%warmup_updates, config%total_updates, config%minimum_lr_ratio
      if (io_status == 0) read(unit, iostat=io_status) state%records_completed, state%source_records_completed, &
         state%updates_completed, state%tokens_seen, state%cumulative_nll
      if (io_status == 0) read(unit, iostat=io_status) stored_apex_state
      if (io_status /= 0) then
         close(unit)
         status = NPT_ERR_IO
         return
      end if
      if (magic /= NEURAL_CHECKPOINT_MAGIC .or. version /= NEURAL_CHECKPOINT_VERSION .or. &
          stored_bpe_tokens /= bpe%token_count .or. stored_bpe_merges /= bpe%merge_count .or. &
          .not. neural_decoder_config_is_valid(decoder_config) .or. .not. neural_train_config_is_valid(config) .or. &
          (stored_apex_state /= 0_int32 .and. stored_apex_state /= 1_int32) .or. &
          (stored_apex_state == 0_int32 .and. state%updates_completed > 0_int64) .or. &
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
      if (stored_apex_state == 1_int32) then
         allocate(state%apex_m_embedding(decoder_config%width, decoder_config%vocabulary_size), &
                  state%apex_m_bias(decoder_config%vocabulary_size), stat=allocation_status)
         if (allocation_status /= 0) then
            close(unit)
            call neural_decoder_destroy(model)
            state = neural_train_state_t()
            status = NPT_ERR_ALLOCATION
            return
         end if
         read(unit, iostat=io_status) state%apex_m_embedding
         if (io_status == 0) read(unit, iostat=io_status) state%apex_m_bias
         if (io_status /= 0 .or. .not. apex_state_is_finite(state)) then
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
