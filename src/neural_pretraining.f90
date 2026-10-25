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

   character(len=8), parameter :: NEURAL_CHECKPOINT_MAGIC = 'VGNR0001'
   ! v2 binds all decoder configuration fields (including retention decay,
   ! numerical epsilon, and rotary period) to the serialized parameter state.
   integer(int32), parameter :: NEURAL_CHECKPOINT_VERSION = 2_int32
   integer(int32), parameter :: FILE_LINE_BYTES = 32768_int32

   type, public :: neural_train_config_t
      integer(int32) :: trainable_mode = NPT_TRAINABLE_TIED_OUTPUT
      ! Step size for the exact tied-output cross-entropy gradient.
      real(real64) :: learning_rate = 0.025_real64
      ! A conservative norm bound on the hidden vector used by the update.
      real(real64) :: gradient_clip_norm = 4.0_real64
      ! Per-value safety envelope for the mutable tied embedding matrix.
      real(real64) :: embedding_value_limit = 8.0_real64
      integer(int32) :: maximum_tokens_per_record = 8192_int32
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
   end type neural_train_state_t

   type, public :: neural_train_metrics_t
      integer(int64) :: records = 0_int64
      integer(int64) :: prediction_tokens = 0_int64
      integer(int64) :: skipped_records = 0_int64
      integer(int64) :: tokenizer_truncations = 0_int64
      real(real64) :: cumulative_nll = 0.0_real64
      real(real64) :: mean_nll = 0.0_real64
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
      integer(int64) :: minimum_resident_bytes = 0_int64
   end type neural_training_resource_plan_t

   public :: neural_train_config_is_valid, neural_train_state_init, neural_train_metrics_reset
   public :: neural_train_token_sequence, neural_evaluate_token_sequence, neural_train_utf8_file
   public :: neural_training_resource_plan, neural_checkpoint_save, neural_checkpoint_load

contains

   pure logical function neural_train_config_is_valid(config)
      type(neural_train_config_t), intent(in) :: config

      neural_train_config_is_valid = config%trainable_mode == NPT_TRAINABLE_TIED_OUTPUT .and. &
                                     ieee_is_finite(config%learning_rate) .and. config%learning_rate > 0.0_real64 .and. &
                                     ieee_is_finite(config%gradient_clip_norm) .and. config%gradient_clip_norm > 0.0_real64 .and. &
                                     ieee_is_finite(config%embedding_value_limit) .and. &
                                     config%embedding_value_limit > 0.0_real64 .and. &
                                     config%maximum_tokens_per_record >= 2_int32
   end function neural_train_config_is_valid

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
      plan%minimum_resident_bytes = plan%decoder_parameter_bytes_real32 + plan%tokenizer_bytes + &
                                    plan%decoder_state_bytes_real32 + plan%record_token_buffer_bytes + &
                                    plan%per_step_working_bytes
   end subroutine neural_training_resource_plan

   subroutine stable_softmax(logits, probabilities, status)
      real(real32), intent(in) :: logits(:)
      real(real64), intent(out) :: probabilities(:)
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
      normalizer = 0.0_real64
      do index = 1_int32, size(logits, kind=int32)
         probabilities(index) = exp(real(logits(index), real64) - maximum)
         normalizer = normalizer + probabilities(index)
      end do
      if (.not. ieee_is_finite(normalizer) .or. normalizer <= tiny(1.0_real64)) then
         status = NPT_ERR_NUMERICAL
         return
      end if
      probabilities = probabilities / normalizer
      if (.not. all(ieee_is_finite(probabilities))) status = NPT_ERR_NUMERICAL
   end subroutine stable_softmax

   subroutine tied_output_update(model, hidden, probabilities, target, config, status)
      type(neural_decoder_t), intent(inout) :: model
      real(real32), intent(in) :: hidden(:)
      real(real64), intent(in) :: probabilities(:)
      integer(int32), intent(in) :: target
      type(neural_train_config_t), intent(in) :: config
      integer(int32), intent(out) :: status
      integer(int32) :: vocabulary_size, width, token, feature
      real(real64) :: hidden_norm, clip_scale, residual, proposal, scale

      status = NPT_OK
      vocabulary_size = model%config%vocabulary_size
      width = model%config%width
      if (target < 1_int32 .or. target > vocabulary_size .or. size(hidden, kind=int32) /= width .or. &
          size(probabilities, kind=int32) /= vocabulary_size .or. .not. all(ieee_is_finite(hidden)) .or. &
          .not. all(ieee_is_finite(probabilities))) then
         status = NPT_ERR_INVALID_ARGUMENT
         return
      end if
      hidden_norm = sqrt(sum(real(hidden, real64) * real(hidden, real64)))
      if (.not. ieee_is_finite(hidden_norm)) then
         status = NPT_ERR_NUMERICAL
         return
      end if
      clip_scale = min(1.0_real64, config%gradient_clip_norm / max(hidden_norm, tiny(1.0_real64)))
      ! neural_decoder_step uses logits = E^T h + b, so this is the
      ! cross-entropy gradient for its tied output head (with norm clipping).
      scale = config%learning_rate * clip_scale
      ! Columns are disjoint. An OpenMP-enabled build can therefore update a
      ! sufficiently large vocabulary in parallel without changing the online
      ! token order or requiring a reduction; a compiler without OpenMP simply
      ! treats these directives as comments.
      !$omp parallel do private(residual, proposal, feature) schedule(static) if(vocabulary_size >= 512_int32)
      do token = 1_int32, vocabulary_size
         residual = probabilities(token)
         if (token == target) residual = residual - 1.0_real64
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
      if (.not. all(ieee_is_finite(model%token_embedding)) .or. .not. all(ieee_is_finite(model%output_bias))) then
         status = NPT_ERR_NUMERICAL
      end if
   end subroutine tied_output_update

   subroutine process_token_sequence(model, tokens, token_count, config, state, metrics, update_model, status)
      type(neural_decoder_t), intent(inout) :: model
      integer(int32), intent(in) :: tokens(:)
      integer(int32), intent(in) :: token_count
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(inout) :: state
      type(neural_train_metrics_t), intent(inout) :: metrics
      logical, intent(in) :: update_model
      integer(int32), intent(out) :: status
      type(neural_decoder_state_t) :: decoder_state
      real(real32), allocatable :: hidden(:), logits(:)
      real(real64), allocatable :: probabilities(:)
      integer :: allocation_status
      integer(int32) :: position, decoder_status, target
      real(real64) :: loss, sequence_nll

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
      allocate(hidden(model%config%width), logits(model%config%vocabulary_size), &
               probabilities(model%config%vocabulary_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = NPT_ERR_ALLOCATION
         return
      end if
      sequence_nll = 0.0_real64
      call neural_decoder_state_init(decoder_state, model, decoder_status)
      if (decoder_status /= NRD_OK) then
         deallocate(hidden, logits, probabilities)
         status = NPT_ERR_MODEL
         return
      end if
      do position = 1_int32, token_count - 1_int32
         call neural_decoder_step(model, decoder_state, tokens(position), hidden, logits, decoder_status)
         if (decoder_status /= NRD_OK) then
            status = NPT_ERR_MODEL
            exit
         end if
         call stable_softmax(logits, probabilities, status)
         if (status /= NPT_OK) exit
         target = tokens(position + 1_int32)
         loss = -log(max(probabilities(target), tiny(1.0_real64)))
         if (.not. ieee_is_finite(loss)) then
            status = NPT_ERR_NUMERICAL
            exit
         end if
         metrics%prediction_tokens = metrics%prediction_tokens + 1_int64
         metrics%cumulative_nll = metrics%cumulative_nll + loss
         sequence_nll = sequence_nll + loss
         if (update_model) then
            call tied_output_update(model, hidden, probabilities, target, config, status)
            if (status /= NPT_OK) exit
            state%updates_completed = state%updates_completed + 1_int64
         end if
      end do
      call neural_decoder_state_destroy(decoder_state)
      deallocate(hidden, logits, probabilities)
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

      if (.not. state%ready) call neural_train_state_init(state)
      call process_token_sequence(model, tokens, token_count, config, state, metrics, .true., status)
   end subroutine neural_train_token_sequence

   subroutine neural_evaluate_token_sequence(model, tokens, token_count, config, metrics, status)
      type(neural_decoder_t), intent(inout) :: model
      integer(int32), intent(in) :: tokens(:)
      integer(int32), intent(in) :: token_count
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      type(neural_train_state_t) :: ignored_state

      call neural_train_state_init(ignored_state)
      call process_token_sequence(model, tokens, token_count, config, ignored_state, metrics, .false., status)
   end subroutine neural_evaluate_token_sequence

   ! Line-oriented UTF-8 adapter. It delegates every byte in each retained line
   ! to the byte-BPE codec. The fixed record cap is an explicit bounded-I/O
   ! contract: records too large for the caller-provided token buffer are
   ! counted as truncations rather than silently partially trained.
   subroutine neural_train_utf8_file(model, bpe, path, config, state, metrics, status, skip_source_records, &
                                      maximum_new_source_records)
      type(neural_decoder_t), intent(inout) :: model
      type(bpe_model_t), intent(in) :: bpe
      character(len=*), intent(in) :: path
      type(neural_train_config_t), intent(in) :: config
      type(neural_train_state_t), intent(inout) :: state
      type(neural_train_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      ! Resume only after validating identical source provenance/order outside
      ! this low-level byte-stream routine.
      integer(int64), intent(in), optional :: skip_source_records
      ! Stop after this many newly consumed source records. Together with the
      ! explicit skip cursor this permits checkpointed continuous chunks.
      integer(int64), intent(in), optional :: maximum_new_source_records
      character(len=FILE_LINE_BYTES) :: line
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
            status = NPT_ERR_TOKENIZER
            return
         end if
         if (token_count < 2_int32) then
            metrics%skipped_records = metrics%skipped_records + 1_int64
            cycle
         end if
         call neural_train_token_sequence(model, tokens, token_count, config, state, metrics, status)
         if (status /= NPT_OK) then
            close(unit)
            deallocate(tokens)
            return
         end if
      end do
      if (source_record < skipped_source_records) then
         close(unit)
         deallocate(tokens)
         status = NPT_ERR_INCOMPATIBLE
         return
      end if
      close(unit, iostat=io_status)
      deallocate(tokens)
      if (io_status /= 0) status = NPT_ERR_IO
   end subroutine neural_train_utf8_file

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
          model%config%vocabulary_size < bpe%token_count .or. .not. model_payload_is_finite(model)) then
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
      if (io_status == 0) write(unit, iostat=io_status) state%records_completed, state%source_records_completed, &
                                                         state%updates_completed, state%tokens_seen, state%cumulative_nll
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
      integer(int32) :: version, stored_bpe_tokens, stored_bpe_merges
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
      if (io_status == 0) read(unit, iostat=io_status) decoder_config%vocabulary_size, decoder_config%width, decoder_config%heads, &
                                                       decoder_config%local_window, decoder_config%ffn_width, &
                                                       decoder_config%memory_decay, decoder_config%rms_epsilon, &
                                                       decoder_config%rotary_period
      if (io_status == 0) read(unit, iostat=io_status) stored_bpe_tokens, stored_bpe_merges
      if (io_status == 0) read(unit, iostat=io_status) config%trainable_mode, config%learning_rate, config%gradient_clip_norm, &
                                                       config%embedding_value_limit, config%maximum_tokens_per_record
      if (io_status == 0) read(unit, iostat=io_status) state%records_completed, state%source_records_completed, &
                                                       state%updates_completed, state%tokens_seen, state%cumulative_nll
      if (io_status /= 0) then
         close(unit)
         status = NPT_ERR_IO
         return
      end if
      if (magic /= NEURAL_CHECKPOINT_MAGIC .or. version /= NEURAL_CHECKPOINT_VERSION .or. &
          stored_bpe_tokens /= bpe%token_count .or. stored_bpe_merges /= bpe%merge_count .or. &
          .not. neural_decoder_config_is_valid(decoder_config) .or. .not. neural_train_config_is_valid(config) .or. &
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
      call read_model_payload(unit, model, io_status)
      read_status = io_status
      close(unit, iostat=io_status)
      if (read_status /= 0 .or. io_status /= 0 .or. .not. model_payload_is_finite(model)) then
         call neural_decoder_destroy(model)
         status = NPT_ERR_IO
         return
      end if
      state%ready = .true.
      status = NPT_OK
   end subroutine neural_checkpoint_load

end module neural_pretraining
