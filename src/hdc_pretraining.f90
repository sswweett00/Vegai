! SPDX-License-Identifier: MIT
!==============================================================================
!  hdc_pretraining.f90
!  Two-pass, streaming, single-pass-Hebbian pretraining pipeline for Vegai.
!
!  This trains a distributional HDC context model, not a Transformer and not a
!  gradient-based language model.  Each active token owns a deterministic base
!  hypervector and a bounded-plasticity context prototype learned from ordered
!  windowed contexts.  The implementation is designed for reproducible corpus
!  passes, bounded memory, checkpoint/resume, and explicit data-loss metrics.
!==============================================================================
module hdc_pretraining
   use, intrinsic :: iso_fortran_env, only: int32, int64, real32, iostat_end, iostat_eor
   use hyper_types
   use text_tokenizer
   use vocabulary
   implicit none
   private

   integer(int32), parameter, public :: PT_OK = 0_int32
   integer(int32), parameter, public :: PT_ERR_INVALID_ARGUMENT = -500_int32
   integer(int32), parameter, public :: PT_ERR_IO = -501_int32
   integer(int32), parameter, public :: PT_ERR_VOCABULARY = -502_int32
   integer(int32), parameter, public :: PT_ERR_ALLOCATION = -503_int32
   integer(int32), parameter, public :: PT_ERR_MODEL_NOT_READY = -504_int32
   integer(int32), parameter, public :: PT_ERR_TOKENIZATION = -505_int32
   integer(int32), parameter, public :: PT_ERR_INCOMPATIBLE_CONFIG = -506_int32
   integer(int32), parameter, public :: PT_ERR_CHECKPOINT = -507_int32
   integer(int32), parameter, public :: PT_ERR_COUNTER_OVERFLOW = -508_int32
   integer(int32), parameter, public :: PT_ERR_NOT_FOUND = -509_int32
   integer(int32), parameter, public :: PT_ERR_EMPTY_VOCABULARY = -510_int32
   integer(int32), parameter, public :: PT_ERR_HYPERVECTOR = -511_int32
   integer(int32), parameter, public :: PT_ERR_RECORD_TOO_LONG = -512_int32

   character(len=8), parameter :: CHECKPOINT_MAGIC = 'VGHDC001'
   integer(int32), parameter :: CHECKPOINT_VERSION = 3_int32

   type, public :: hdc_pretrain_config_t
      integer(int32) :: max_vocabulary = 20000_int32
      integer(int64) :: min_frequency = 2_int64
      integer(int32) :: context_window = 4_int32
      integer(int32) :: position_shift_bits = 73_int32
      integer(int32) :: epochs = 1_int32
      integer(int32) :: initial_plasticity_percent = 25_int32
      integer(int32) :: minimum_plasticity_percent = 1_int32
      integer(int32) :: plasticity_decay_events = 32_int32
      integer(int64) :: tie_seed = int(z'6A09E667F3BCC909', int64)
      logical :: strict_tokenization = .false.
   end type hdc_pretrain_config_t

   type, public :: pretrain_metrics_t
      integer(int64) :: corpus_lines = 0_int64
      integer(int64) :: raw_tokens = 0_int64
      integer(int64) :: active_tokens = 0_int64
      integer(int64) :: oov_tokens = 0_int64
      integer(int64) :: context_pairs = 0_int64
      integer(int64) :: center_updates = 0_int64
      integer(int64) :: tokenizer_truncations = 0_int64
      integer(int64) :: non_ascii_bytes = 0_int64
      integer(int64) :: overlong_records = 0_int64
   end type pretrain_metrics_t

   type, public :: hdc_pretrain_model_t
      type(vocabulary_t) :: vocabulary
      logical :: ready = .false.
      integer(int32) :: context_window = 0_int32
      integer(int32) :: position_shift_bits = 0_int32
      integer(int32) :: initial_plasticity_percent = 0_int32
      integer(int32) :: minimum_plasticity_percent = 0_int32
      integer(int32) :: plasticity_decay_events = 0_int32
      integer(int64) :: tie_seed = 0_int64
      integer(int64) :: training_epochs_completed = 0_int64
      type(pretrain_metrics_t) :: totals
      ! Word-major packed SoA data planes: each column is one hypervector.
      integer(int64), allocatable :: base_bits(:, :)
      integer(int64), allocatable :: semantic_bits(:, :)
      integer(int64), allocatable :: experience_count(:)
   end type hdc_pretrain_model_t

   public :: pretrain_build_vocabulary, pretrain_train_file
   public :: pretrain_model_destroy, pretrain_model_is_ready
   public :: pretrain_metrics_reset, pretrain_model_footprint_bytes
   public :: pretrain_checkpoint_save, pretrain_checkpoint_load
   public :: pretrain_lookup_token, pretrain_get_semantic_hv, pretrain_nearest_semantic
   public :: pretrain_config_from_model

contains

   pure function pretrain_model_is_ready(model) result(ready)
      type(hdc_pretrain_model_t), intent(in) :: model
      logical :: ready

      ready = model%ready .and. vocab_is_initialized(model%vocabulary) .and. &
              allocated(model%base_bits) .and. allocated(model%semantic_bits) .and. &
              allocated(model%experience_count)
   end function pretrain_model_is_ready

   subroutine pretrain_metrics_reset(metrics)
      type(pretrain_metrics_t), intent(inout) :: metrics

      metrics%corpus_lines = 0_int64
      metrics%raw_tokens = 0_int64
      metrics%active_tokens = 0_int64
      metrics%oov_tokens = 0_int64
      metrics%context_pairs = 0_int64
      metrics%center_updates = 0_int64
      metrics%tokenizer_truncations = 0_int64
      metrics%non_ascii_bytes = 0_int64
      metrics%overlong_records = 0_int64
   end subroutine pretrain_metrics_reset

   subroutine pretrain_model_destroy(model)
      type(hdc_pretrain_model_t), intent(inout) :: model

      call vocab_destroy(model%vocabulary)
      if (allocated(model%base_bits)) deallocate(model%base_bits)
      if (allocated(model%semantic_bits)) deallocate(model%semantic_bits)
      if (allocated(model%experience_count)) deallocate(model%experience_count)
      model%ready = .false.
      model%context_window = 0_int32
      model%position_shift_bits = 0_int32
      model%initial_plasticity_percent = 0_int32
      model%minimum_plasticity_percent = 0_int32
      model%plasticity_decay_events = 0_int32
      model%tie_seed = 0_int64
      model%training_epochs_completed = 0_int64
      call pretrain_metrics_reset(model%totals)
   end subroutine pretrain_model_destroy

   subroutine validate_pretrain_config(config, status)
      type(hdc_pretrain_config_t), intent(in) :: config
      integer(int32), intent(out) :: status
      integer(int64) :: maximum_shift

      status = PT_OK
      if (config%max_vocabulary <= 0_int32 .or. config%min_frequency <= 0_int64) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%context_window <= 0_int32 .or. &
          config%context_window >= MAX_TOKENS_PER_LINE) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%position_shift_bits == 0_int32 .or. config%epochs <= 0_int32) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%initial_plasticity_percent <= 0_int32 .or. &
          config%initial_plasticity_percent > 100_int32 .or. &
          config%minimum_plasticity_percent <= 0_int32 .or. &
          config%minimum_plasticity_percent > config%initial_plasticity_percent .or. &
          config%plasticity_decay_events <= 0_int32) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if

      maximum_shift = int(config%context_window, int64) * &
                      abs(int(config%position_shift_bits, int64))
      if (maximum_shift > int(huge(0_int32), int64)) status = PT_ERR_INVALID_ARGUMENT
   end subroutine validate_pretrain_config

   subroutine add_nonnegative(counter, delta, status)
      integer(int64), intent(inout) :: counter
      integer(int64), intent(in) :: delta
      integer(int32), intent(out) :: status

      status = PT_OK
      if (delta < 0_int64 .or. counter > huge(0_int64) - delta) then
         status = PT_ERR_COUNTER_OVERFLOW
         return
      end if
      counter = counter + delta
   end subroutine add_nonnegative

   subroutine add_metrics(destination, source, status)
      type(pretrain_metrics_t), intent(inout) :: destination
      type(pretrain_metrics_t), intent(in) :: source
      integer(int32), intent(out) :: status

      status = PT_OK
      if (source%corpus_lines < 0_int64 .or. source%raw_tokens < 0_int64 .or. &
          source%active_tokens < 0_int64 .or. source%oov_tokens < 0_int64 .or. &
          source%context_pairs < 0_int64 .or. source%center_updates < 0_int64 .or. &
          source%tokenizer_truncations < 0_int64 .or. source%non_ascii_bytes < 0_int64 .or. &
          source%overlong_records < 0_int64) then
         status = PT_ERR_COUNTER_OVERFLOW
         return
      end if
      if (destination%corpus_lines > huge(0_int64) - source%corpus_lines .or. &
          destination%raw_tokens > huge(0_int64) - source%raw_tokens .or. &
          destination%active_tokens > huge(0_int64) - source%active_tokens .or. &
          destination%oov_tokens > huge(0_int64) - source%oov_tokens .or. &
          destination%context_pairs > huge(0_int64) - source%context_pairs .or. &
          destination%center_updates > huge(0_int64) - source%center_updates .or. &
          destination%tokenizer_truncations > huge(0_int64) - source%tokenizer_truncations .or. &
          destination%non_ascii_bytes > huge(0_int64) - source%non_ascii_bytes .or. &
          destination%overlong_records > huge(0_int64) - source%overlong_records) then
         status = PT_ERR_COUNTER_OVERFLOW
         return
      end if
      destination%corpus_lines = destination%corpus_lines + source%corpus_lines
      destination%raw_tokens = destination%raw_tokens + source%raw_tokens
      destination%active_tokens = destination%active_tokens + source%active_tokens
      destination%oov_tokens = destination%oov_tokens + source%oov_tokens
      destination%context_pairs = destination%context_pairs + source%context_pairs
      destination%center_updates = destination%center_updates + source%center_updates
      destination%tokenizer_truncations = destination%tokenizer_truncations + source%tokenizer_truncations
      destination%non_ascii_bytes = destination%non_ascii_bytes + source%non_ascii_bytes
      destination%overlong_records = destination%overlong_records + source%overlong_records
   end subroutine add_metrics

   ! Reads one physical formatted record without silently accepting a record
   ! longer than MAX_LINE_CHARS.  A too-long record is fully drained and exposed
   ! to the caller as telemetry, so a non-strict run can drop it deterministically
   ! rather than learn from a prefix.  EOF is distinct from malformed I/O.
   subroutine read_corpus_record(unit, line, eof, overlong, status)
      integer, intent(in) :: unit
      character(len=*), intent(out) :: line
      logical, intent(out) :: eof, overlong
      integer(int32), intent(out) :: status
      character(len=1) :: probe
      character(len=MAX_LINE_CHARS) :: discard_buffer
      integer :: io_status, transferred

      line = ''
      eof = .false.
      overlong = .false.
      status = PT_OK
      transferred = 0
      read(unit, '(A)', advance='no', size=transferred, iostat=io_status) line
      if (io_status == iostat_end) then
         eof = .true.
         return
      end if
      if (io_status /= 0 .and. io_status /= iostat_eor) then
         status = PT_ERR_IO
         return
      end if
      if (io_status == iostat_eor) return

      ! The first transfer filled the fixed record buffer.  Probe once to
      ! distinguish an exactly-sized record from an overlong record.
      probe = ''
      transferred = 0
      read(unit, '(A)', advance='no', size=transferred, iostat=io_status) probe
      if (transferred > 0) overlong = .true.
      if (io_status /= 0 .and. io_status /= iostat_eor .and. io_status /= iostat_end) then
         status = PT_ERR_IO
         return
      end if
      if (.not. overlong) then
         if (io_status == iostat_eor .or. io_status == iostat_end) return
         status = PT_ERR_IO
         return
      end if

      ! If the probe did not finish this record, consume its remaining bytes.
      do while (io_status == 0)
         discard_buffer = ''
         transferred = 0
         read(unit, '(A)', advance='no', size=transferred, iostat=io_status) discard_buffer
         if (io_status /= 0 .and. io_status /= iostat_eor .and. io_status /= iostat_end) then
            status = PT_ERR_IO
            return
         end if
      end do
   end subroutine read_corpus_record

   subroutine allocate_model_storage(model, status)
      type(hdc_pretrain_model_t), intent(inout) :: model
      integer(int32), intent(out) :: status
      integer(int32) :: allocation_status

      status = PT_OK
      if (.not. vocab_is_initialized(model%vocabulary)) then
         status = PT_ERR_VOCABULARY
         return
      end if
      allocate(model%base_bits(HV_WORDS, model%vocabulary%capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         status = PT_ERR_ALLOCATION
         return
      end if
      allocate(model%semantic_bits(HV_WORDS, model%vocabulary%capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         deallocate(model%base_bits)
         status = PT_ERR_ALLOCATION
         return
      end if
      allocate(model%experience_count(model%vocabulary%capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         deallocate(model%base_bits)
         deallocate(model%semantic_bits)
         status = PT_ERR_ALLOCATION
         return
      end if
      model%base_bits = 0_int64
      model%semantic_bits = 0_int64
      model%experience_count = 0_int64
   end subroutine allocate_model_storage

   subroutine initialize_active_hypervectors(model)
      type(hdc_pretrain_model_t), intent(inout) :: model
      type(hv_t) :: vector
      integer(int32) :: slot

      do slot = 1, model%vocabulary%count
         if (.not. model%vocabulary%active(slot)) cycle
         vector = hv_from_seed(vocab_seed_for_slot(model%vocabulary, slot))
         model%base_bits(:, slot) = vector%bits
         model%semantic_bits(:, slot) = vector%bits
      end do
   end subroutine initialize_active_hypervectors

   ! First corpus pass: observe all normalized tokens, finalize min-frequency
   ! activity, then allocate deterministic token identities and semantic traces.
   subroutine pretrain_build_vocabulary(model, corpus_path, config, metrics, status)
      type(hdc_pretrain_model_t), intent(inout) :: model
      character(len=*), intent(in) :: corpus_path
      type(hdc_pretrain_config_t), intent(in) :: config
      type(pretrain_metrics_t), intent(out) :: metrics
      integer(int32), intent(out) :: status
      character(len=MAX_LINE_CHARS) :: line
      character(len=TOKEN_LEN) :: tokens(MAX_TOKENS_PER_LINE)
      integer :: unit, io_status
      integer(int32) :: token_index, n_tokens, vocab_status
      integer(int64) :: non_ascii_count
      logical :: eof, overlong, truncated

      call pretrain_model_destroy(model)
      call pretrain_metrics_reset(metrics)
      call validate_pretrain_config(config, status)
      if (status /= PT_OK) return
      if (len_trim(corpus_path) == 0) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if

      call vocab_init(model%vocabulary, config%max_vocabulary, config%min_frequency, vocab_status)
      if (vocab_status /= VOCAB_OK) then
         status = PT_ERR_VOCABULARY
         call pretrain_model_destroy(model)
         return
      end if

      open(newunit=unit, file=trim(corpus_path), status='old', action='read', &
           form='formatted', iostat=io_status)
      if (io_status /= 0) then
         status = PT_ERR_IO
         call pretrain_model_destroy(model)
         return
      end if

      do
         call read_corpus_record(unit, line, eof, overlong, status)
         if (status /= PT_OK) then
            close(unit)
            call pretrain_model_destroy(model)
            return
         end if
         if (eof) exit
         call add_nonnegative(metrics%corpus_lines, 1_int64, status)
         if (status /= PT_OK) exit
         if (overlong) then
            call add_nonnegative(metrics%overlong_records, 1_int64, status)
            if (status /= PT_OK) exit
            if (config%strict_tokenization) then
               status = PT_ERR_RECORD_TOO_LONG
               exit
            end if
            cycle
         end if
         call tokenize_ascii_line(line, tokens, n_tokens, truncated, non_ascii_count)
         call add_nonnegative(metrics%non_ascii_bytes, non_ascii_count, status)
         if (status /= PT_OK) exit
         if (truncated) then
            call add_nonnegative(metrics%tokenizer_truncations, 1_int64, status)
            if (status /= PT_OK) exit
            if (config%strict_tokenization) then
               status = PT_ERR_TOKENIZATION
               exit
            end if
         end if
         do token_index = 1, n_tokens
            call add_nonnegative(metrics%raw_tokens, 1_int64, status)
            if (status /= PT_OK) exit
            call vocab_observe(model%vocabulary, tokens(token_index), vocab_status)
            if (vocab_status /= VOCAB_OK) then
               status = PT_ERR_VOCABULARY
               exit
            end if
         end do
         if (status /= PT_OK) exit
      end do
      close(unit)
      if (status /= PT_OK) then
         call pretrain_model_destroy(model)
         return
      end if

      call vocab_finalize(model%vocabulary, vocab_status)
      if (vocab_status /= VOCAB_OK) then
         status = PT_ERR_VOCABULARY
         call pretrain_model_destroy(model)
         return
      end if
      if (model%vocabulary%active_count == 0_int32) then
         status = PT_ERR_EMPTY_VOCABULARY
         call pretrain_model_destroy(model)
         return
      end if

      call allocate_model_storage(model, status)
      if (status /= PT_OK) then
         call pretrain_model_destroy(model)
         return
      end if
      call initialize_active_hypervectors(model)
      model%context_window = config%context_window
      model%position_shift_bits = config%position_shift_bits
      model%initial_plasticity_percent = config%initial_plasticity_percent
      model%minimum_plasticity_percent = config%minimum_plasticity_percent
      model%plasticity_decay_events = config%plasticity_decay_events
      model%tie_seed = config%tie_seed
      model%ready = .true.
   end subroutine pretrain_build_vocabulary

   pure function scheduled_plasticity_percent(model, experiences) result(percent)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int64), intent(in) :: experiences
      integer(int32) :: percent
      integer(int64) :: decay

      decay = int(model%plasticity_decay_events, int64)
      if (experiences < decay) then
         percent = model%initial_plasticity_percent
      else if (experiences < 4_int64 * decay) then
         percent = max(model%minimum_plasticity_percent, &
                       (model%initial_plasticity_percent + 1_int32) / 2_int32)
      else if (experiences < 16_int64 * decay) then
         percent = max(model%minimum_plasticity_percent, &
                       (model%initial_plasticity_percent + 3_int32) / 4_int32)
      else
         percent = model%minimum_plasticity_percent
      end if
   end function scheduled_plasticity_percent

   subroutine pretrain_update_center(model, config, token_slots, n_tokens, center_index, metrics, status)
      type(hdc_pretrain_model_t), intent(inout) :: model
      type(hdc_pretrain_config_t), intent(in) :: config
      integer(int32), intent(in) :: token_slots(:)
      integer(int32), intent(in) :: n_tokens, center_index
      type(pretrain_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      type(hv_acc_t) :: accumulator
      type(hv_t) :: neighbor, positioned, context, prototype, updated
      integer(int32) :: center_slot, offset, neighbor_index, neighbor_slot
      integer(int32) :: add_status, update_status, shift_bits, neighbors
      integer(int32) :: plasticity_percent
      integer(int64) :: tie_seed

      status = PT_OK
      center_slot = token_slots(center_index)
      if (center_slot == 0_int32) return

      call hv_acc_reset(accumulator)
      neighbors = 0_int32
      do offset = -config%context_window, config%context_window
         if (offset == 0_int32) cycle
         neighbor_index = center_index + offset
         if (neighbor_index < 1_int32 .or. neighbor_index > n_tokens) cycle
         neighbor_slot = token_slots(neighbor_index)
         if (neighbor_slot == 0_int32) cycle

         neighbor%bits = model%base_bits(:, neighbor_slot)
         shift_bits = int(int(offset, int64) * int(config%position_shift_bits, int64), int32)
         positioned = hv_permute(neighbor, shift_bits)
         call hv_acc_add(accumulator, positioned, add_status)
         if (add_status /= HV_OK) then
            status = PT_ERR_HYPERVECTOR
            return
         end if
         neighbors = neighbors + 1_int32
         call add_nonnegative(metrics%context_pairs, 1_int64, status)
         if (status /= PT_OK) return
      end do
      if (neighbors == 0_int32) return

      tie_seed = ieor(vocab_seed_for_slot(model%vocabulary, center_slot), &
                      ishftc(model%experience_count(center_slot), 19))
      context = hv_acc_finalize(accumulator, ieor(config%tie_seed, tie_seed))
      prototype%bits = model%semantic_bits(:, center_slot)
      plasticity_percent = scheduled_plasticity_percent(model, model%experience_count(center_slot))
      call hv_hebbian_update_percent(prototype, context, plasticity_percent, tie_seed, updated, update_status)
      if (update_status /= HV_OK) then
         status = PT_ERR_HYPERVECTOR
         return
      end if
      if (model%experience_count(center_slot) == huge(0_int64)) then
         status = PT_ERR_COUNTER_OVERFLOW
         return
      end if
      model%semantic_bits(:, center_slot) = updated%bits
      model%experience_count(center_slot) = model%experience_count(center_slot) + 1_int64
      call add_nonnegative(metrics%center_updates, 1_int64, status)
   end subroutine pretrain_update_center

   subroutine train_one_epoch(model, corpus_path, config, metrics, status)
      type(hdc_pretrain_model_t), intent(inout) :: model
      character(len=*), intent(in) :: corpus_path
      type(hdc_pretrain_config_t), intent(in) :: config
      type(pretrain_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      character(len=MAX_LINE_CHARS) :: line
      character(len=TOKEN_LEN) :: tokens(MAX_TOKENS_PER_LINE)
      integer(int32) :: token_slots(MAX_TOKENS_PER_LINE)
      integer :: unit, io_status
      integer(int32) :: token_index, n_tokens, slot, vocab_status
      integer(int64) :: non_ascii_count
      logical :: eof, found, overlong, truncated

      status = PT_OK
      open(newunit=unit, file=trim(corpus_path), status='old', action='read', &
           form='formatted', iostat=io_status)
      if (io_status /= 0) then
         status = PT_ERR_IO
         return
      end if

      do
         call read_corpus_record(unit, line, eof, overlong, status)
         if (status /= PT_OK) then
            close(unit)
            return
         end if
         if (eof) exit
         call add_nonnegative(metrics%corpus_lines, 1_int64, status)
         if (status /= PT_OK) exit
         if (overlong) then
            call add_nonnegative(metrics%overlong_records, 1_int64, status)
            if (status /= PT_OK) exit
            if (config%strict_tokenization) then
               status = PT_ERR_RECORD_TOO_LONG
               exit
            end if
            cycle
         end if
         call tokenize_ascii_line(line, tokens, n_tokens, truncated, non_ascii_count)
         call add_nonnegative(metrics%non_ascii_bytes, non_ascii_count, status)
         if (status /= PT_OK) exit
         if (truncated) then
            call add_nonnegative(metrics%tokenizer_truncations, 1_int64, status)
            if (status /= PT_OK) exit
            if (config%strict_tokenization) then
               status = PT_ERR_TOKENIZATION
               exit
            end if
         end if

         token_slots = 0_int32
         do token_index = 1, n_tokens
            call add_nonnegative(metrics%raw_tokens, 1_int64, status)
            if (status /= PT_OK) exit
            call vocab_lookup(model%vocabulary, tokens(token_index), slot, found, vocab_status, .true.)
            if (found) then
               token_slots(token_index) = slot
               call add_nonnegative(metrics%active_tokens, 1_int64, status)
            else if (vocab_status == VOCAB_ERR_NOT_FOUND) then
               call add_nonnegative(metrics%oov_tokens, 1_int64, status)
            else
               status = PT_ERR_VOCABULARY
            end if
            if (status /= PT_OK) exit
         end do
         if (status /= PT_OK) exit

         do token_index = 1, n_tokens
            call pretrain_update_center(model, config, token_slots, n_tokens, token_index, metrics, status)
            if (status /= PT_OK) exit
         end do
         if (status /= PT_OK) exit
      end do
      close(unit)
   end subroutine train_one_epoch

   subroutine pretrain_train_file(model, corpus_path, config, metrics, status)
      type(hdc_pretrain_model_t), intent(inout) :: model
      character(len=*), intent(in) :: corpus_path
      type(hdc_pretrain_config_t), intent(in) :: config
      type(pretrain_metrics_t), intent(out) :: metrics
      integer(int32), intent(out) :: status
      integer(int32) :: epoch, tally_status, original_status

      call pretrain_metrics_reset(metrics)
      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if
      call validate_pretrain_config(config, status)
      if (status /= PT_OK) return
      if (len_trim(corpus_path) == 0) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%max_vocabulary /= model%vocabulary%capacity .or. &
          config%min_frequency /= model%vocabulary%min_frequency .or. &
          config%context_window /= model%context_window .or. &
          config%position_shift_bits /= model%position_shift_bits .or. &
          config%initial_plasticity_percent /= model%initial_plasticity_percent .or. &
          config%minimum_plasticity_percent /= model%minimum_plasticity_percent .or. &
          config%plasticity_decay_events /= model%plasticity_decay_events .or. &
          config%tie_seed /= model%tie_seed) then
         status = PT_ERR_INCOMPATIBLE_CONFIG
         return
      end if

      do epoch = 1, config%epochs
         call train_one_epoch(model, corpus_path, config, metrics, status)
         if (status /= PT_OK) then
            original_status = status
            call add_metrics(model%totals, metrics, tally_status)
            if (tally_status /= PT_OK) then
               status = tally_status
            else
               status = original_status
            end if
            return
         end if
         if (model%training_epochs_completed == huge(0_int64)) then
            call add_metrics(model%totals, metrics, tally_status)
            if (tally_status /= PT_OK) then
               status = tally_status
            else
               status = PT_ERR_COUNTER_OVERFLOW
            end if
            return
         end if
         model%training_epochs_completed = model%training_epochs_completed + 1_int64
      end do
      call add_metrics(model%totals, metrics, status)
   end subroutine pretrain_train_file

   subroutine pretrain_lookup_token(model, token, slot, found, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      character(len=*), intent(in) :: token
      integer(int32), intent(out) :: slot
      logical, intent(out) :: found
      integer(int32), intent(out) :: status
      integer(int32) :: vocab_status

      slot = 0_int32
      found = .false.
      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if
      call vocab_lookup(model%vocabulary, token, slot, found, vocab_status, .true.)
      if (vocab_status == VOCAB_ERR_NOT_FOUND .or. .not. found) then
         status = PT_ERR_NOT_FOUND
      else if (vocab_status /= VOCAB_OK) then
         status = PT_ERR_VOCABULARY
      end if
   end subroutine pretrain_lookup_token

   subroutine pretrain_get_semantic_hv(model, slot, semantic_hv, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int32), intent(in) :: slot
      type(hv_t), intent(out) :: semantic_hv
      integer(int32), intent(out) :: status

      semantic_hv = hv_zero()
      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if
      if (slot < 1_int32 .or. slot > model%vocabulary%count .or. &
          .not. model%vocabulary%active(slot)) then
         status = PT_ERR_NOT_FOUND
         return
      end if
      semantic_hv%bits = model%semantic_bits(:, slot)
   end subroutine pretrain_get_semantic_hv

   subroutine pretrain_nearest_semantic(model, query_hv, best_slot, best_token, best_similarity, &
                                        status, exclude_slot)
      type(hdc_pretrain_model_t), intent(in) :: model
      type(hv_t), intent(in) :: query_hv
      integer(int32), intent(out) :: best_slot
      character(len=TOKEN_LEN), intent(out) :: best_token
      real(real32), intent(out) :: best_similarity
      integer(int32), intent(out) :: status
      integer(int32), intent(in), optional :: exclude_slot
      type(hv_t) :: candidate
      integer(int32) :: slot, distance, best_distance, excluded

      best_slot = 0_int32
      best_token = ''
      best_similarity = -1.0_real32
      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if

      excluded = 0_int32
      if (present(exclude_slot)) excluded = exclude_slot
      best_distance = HV_BITS + 1_int32
      do slot = 1, model%vocabulary%count
         if (.not. model%vocabulary%active(slot) .or. slot == excluded) cycle
         candidate%bits = model%semantic_bits(:, slot)
         distance = hv_hamming(query_hv, candidate)
         if (distance < best_distance) then
            best_distance = distance
            best_slot = slot
         end if
      end do
      if (best_slot == 0_int32) then
         status = PT_ERR_NOT_FOUND
         return
      end if
      best_token = model%vocabulary%tokens(best_slot)
      best_similarity = 1.0_real32 - 2.0_real32 * real(best_distance, real32) &
                        / real(HV_BITS, real32)
   end subroutine pretrain_nearest_semantic

   subroutine pretrain_config_from_model(model, config, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      type(hdc_pretrain_config_t), intent(out) :: config
      integer(int32), intent(out) :: status

      config = hdc_pretrain_config_t()
      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if
      config%max_vocabulary = model%vocabulary%capacity
      config%min_frequency = model%vocabulary%min_frequency
      config%context_window = model%context_window
      config%position_shift_bits = model%position_shift_bits
      config%initial_plasticity_percent = model%initial_plasticity_percent
      config%minimum_plasticity_percent = model%minimum_plasticity_percent
      config%plasticity_decay_events = model%plasticity_decay_events
      config%tie_seed = model%tie_seed
   end subroutine pretrain_config_from_model

   pure function pretrain_model_footprint_bytes(model) result(total_bytes)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int64) :: total_bytes

      total_bytes = vocab_footprint_bytes(model%vocabulary)
      if (allocated(model%base_bits)) then
         total_bytes = total_bytes + size(model%base_bits, kind=int64) * &
                       int(storage_size(model%base_bits(1, 1)) / 8, int64)
      end if
      if (allocated(model%semantic_bits)) then
         total_bytes = total_bytes + size(model%semantic_bits, kind=int64) * &
                       int(storage_size(model%semantic_bits(1, 1)) / 8, int64)
      end if
      if (allocated(model%experience_count)) then
         total_bytes = total_bytes + size(model%experience_count, kind=int64) * &
                       int(storage_size(model%experience_count(1)) / 8, int64)
      end if
   end function pretrain_model_footprint_bytes

   ! A deterministic accidental-corruption detector for a checkpoint payload.
   ! It is not a cryptographic signature and must not be used as an authenticity
   ! mechanism.  Stream binary layout remains compiler/ABI-local by contract.
   pure function checkpoint_mix(state, value) result(mixed)
      integer(int64), intent(in) :: state, value
      integer(int64) :: mixed

      mixed = ieor(ishftc(state, 9), value)
      mixed = ieor(mixed, ishftc(value, -13))
      mixed = ieor(mixed, ishftc(state, -27))
   end function checkpoint_mix

   pure function model_checkpoint_checksum(model) result(checksum)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int64) :: checksum
      integer(int32) :: slot, word, character_index

      checksum = int(z'BB67AE8584CAA73B', int64)
      checksum = checkpoint_mix(checksum, int(CHECKPOINT_VERSION, int64))
      checksum = checkpoint_mix(checksum, int(model%vocabulary%capacity, int64))
      checksum = checkpoint_mix(checksum, int(model%vocabulary%count, int64))
      checksum = checkpoint_mix(checksum, int(model%vocabulary%active_count, int64))
      checksum = checkpoint_mix(checksum, model%vocabulary%min_frequency)
      checksum = checkpoint_mix(checksum, int(model%context_window, int64))
      checksum = checkpoint_mix(checksum, int(model%position_shift_bits, int64))
      checksum = checkpoint_mix(checksum, int(model%initial_plasticity_percent, int64))
      checksum = checkpoint_mix(checksum, int(model%minimum_plasticity_percent, int64))
      checksum = checkpoint_mix(checksum, int(model%plasticity_decay_events, int64))
      checksum = checkpoint_mix(checksum, model%tie_seed)
      checksum = checkpoint_mix(checksum, model%training_epochs_completed)
      checksum = checkpoint_mix(checksum, model%totals%corpus_lines)
      checksum = checkpoint_mix(checksum, model%totals%raw_tokens)
      checksum = checkpoint_mix(checksum, model%totals%active_tokens)
      checksum = checkpoint_mix(checksum, model%totals%oov_tokens)
      checksum = checkpoint_mix(checksum, model%totals%context_pairs)
      checksum = checkpoint_mix(checksum, model%totals%center_updates)
      checksum = checkpoint_mix(checksum, model%totals%tokenizer_truncations)
      checksum = checkpoint_mix(checksum, model%totals%non_ascii_bytes)
      checksum = checkpoint_mix(checksum, model%totals%overlong_records)
      do slot = 1, model%vocabulary%capacity
         do character_index = 1, TOKEN_LEN
            checksum = checkpoint_mix(checksum, &
                                      int(iachar(model%vocabulary%tokens(slot)(character_index:character_index)), int64))
         end do
         checksum = checkpoint_mix(checksum, model%vocabulary%frequencies(slot))
         if (model%vocabulary%active(slot)) then
            checksum = checkpoint_mix(checksum, 1_int64)
         else
            checksum = checkpoint_mix(checksum, 0_int64)
         end if
         checksum = checkpoint_mix(checksum, model%experience_count(slot))
         do word = 1, HV_WORDS
            checksum = checkpoint_mix(checksum, model%base_bits(word, slot))
            checksum = checkpoint_mix(checksum, model%semantic_bits(word, slot))
         end do
      end do
   end function model_checkpoint_checksum

   subroutine pretrain_checkpoint_save(model, checkpoint_path, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      character(len=*), intent(in) :: checkpoint_path
      integer(int32), intent(out) :: status
      integer :: unit, io_status
      integer(int64) :: checksum

      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if
      if (len_trim(checkpoint_path) == 0) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if

      open(newunit=unit, file=trim(checkpoint_path), status='replace', action='write', &
           access='stream', form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = PT_ERR_IO
         return
      end if

      write(unit, iostat=io_status) CHECKPOINT_MAGIC, CHECKPOINT_VERSION
      if (io_status == 0) write(unit, iostat=io_status) model%vocabulary%capacity, &
         model%vocabulary%count, model%vocabulary%active_count, model%vocabulary%min_frequency
      if (io_status == 0) write(unit, iostat=io_status) model%context_window, model%position_shift_bits, &
         model%initial_plasticity_percent, model%minimum_plasticity_percent, &
         model%plasticity_decay_events, model%tie_seed
      if (io_status == 0) write(unit, iostat=io_status) model%training_epochs_completed
      if (io_status == 0) write(unit, iostat=io_status) model%totals%corpus_lines, model%totals%raw_tokens, &
         model%totals%active_tokens, model%totals%oov_tokens, model%totals%context_pairs, &
         model%totals%center_updates, model%totals%tokenizer_truncations, &
         model%totals%non_ascii_bytes, model%totals%overlong_records
      if (io_status == 0) write(unit, iostat=io_status) model%vocabulary%tokens
      if (io_status == 0) write(unit, iostat=io_status) model%vocabulary%frequencies
      if (io_status == 0) write(unit, iostat=io_status) model%vocabulary%active
      if (io_status == 0) write(unit, iostat=io_status) model%base_bits
      if (io_status == 0) write(unit, iostat=io_status) model%semantic_bits
      if (io_status == 0) write(unit, iostat=io_status) model%experience_count
      if (io_status == 0) then
         checksum = model_checkpoint_checksum(model)
         write(unit, iostat=io_status) checksum
      end if
      close(unit)
      if (io_status /= 0) status = PT_ERR_CHECKPOINT
   end subroutine pretrain_checkpoint_save

   subroutine pretrain_checkpoint_load(checkpoint_path, model, status)
      character(len=*), intent(in) :: checkpoint_path
      type(hdc_pretrain_model_t), intent(inout) :: model
      integer(int32), intent(out) :: status
      character(len=8) :: magic
      type(hdc_pretrain_config_t) :: checkpoint_config
      integer :: unit, io_status
      integer(int32) :: version, capacity, count, stored_active_count, vocab_status, slot
      integer(int64) :: min_frequency, stored_checksum

      call pretrain_model_destroy(model)
      status = PT_OK
      if (len_trim(checkpoint_path) == 0) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      open(newunit=unit, file=trim(checkpoint_path), status='old', action='read', &
           access='stream', form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = PT_ERR_IO
         return
      end if

      read(unit, iostat=io_status) magic, version
      if (io_status /= 0) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         return
      end if
      if (magic /= CHECKPOINT_MAGIC .or. version /= CHECKPOINT_VERSION) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         return
      end if
      read(unit, iostat=io_status) capacity, count, stored_active_count, min_frequency
      if (io_status /= 0) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         return
      end if
      if (capacity <= 0_int32 .or. count <= 0_int32 .or. count > capacity .or. &
          stored_active_count <= 0_int32 .or. stored_active_count > count .or. &
          min_frequency <= 0_int64) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         return
      end if
      call vocab_init(model%vocabulary, capacity, min_frequency, vocab_status)
      if (vocab_status /= VOCAB_OK) then
         close(unit)
         status = PT_ERR_VOCABULARY
         call pretrain_model_destroy(model)
         return
      end if
      model%vocabulary%count = count

      read(unit, iostat=io_status) model%context_window, model%position_shift_bits, &
         model%initial_plasticity_percent, model%minimum_plasticity_percent, &
         model%plasticity_decay_events, model%tie_seed
      if (io_status == 0) read(unit, iostat=io_status) model%training_epochs_completed
      if (io_status == 0) read(unit, iostat=io_status) model%totals%corpus_lines, model%totals%raw_tokens, &
         model%totals%active_tokens, model%totals%oov_tokens, model%totals%context_pairs, &
         model%totals%center_updates, model%totals%tokenizer_truncations, &
         model%totals%non_ascii_bytes, model%totals%overlong_records
      if (io_status == 0) read(unit, iostat=io_status) model%vocabulary%tokens
      if (io_status == 0) read(unit, iostat=io_status) model%vocabulary%frequencies
      if (io_status == 0) read(unit, iostat=io_status) model%vocabulary%active
      if (io_status /= 0) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         call pretrain_model_destroy(model)
         return
      end if

      checkpoint_config = hdc_pretrain_config_t()
      checkpoint_config%max_vocabulary = capacity
      checkpoint_config%min_frequency = min_frequency
      checkpoint_config%context_window = model%context_window
      checkpoint_config%position_shift_bits = model%position_shift_bits
      checkpoint_config%initial_plasticity_percent = model%initial_plasticity_percent
      checkpoint_config%minimum_plasticity_percent = model%minimum_plasticity_percent
      checkpoint_config%plasticity_decay_events = model%plasticity_decay_events
      checkpoint_config%tie_seed = model%tie_seed
      call validate_pretrain_config(checkpoint_config, status)
      if (status /= PT_OK .or. model%training_epochs_completed < 0_int64 .or. &
          model%totals%corpus_lines < 0_int64 .or. model%totals%raw_tokens < 0_int64 .or. &
          model%totals%active_tokens < 0_int64 .or. model%totals%oov_tokens < 0_int64 .or. &
          model%totals%context_pairs < 0_int64 .or. model%totals%center_updates < 0_int64 .or. &
          model%totals%tokenizer_truncations < 0_int64 .or. model%totals%non_ascii_bytes < 0_int64 .or. &
          model%totals%overlong_records < 0_int64 .or. &
          model%totals%active_tokens > model%totals%raw_tokens .or. &
          model%totals%oov_tokens > model%totals%raw_tokens .or. &
          model%totals%center_updates > model%totals%active_tokens) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         call pretrain_model_destroy(model)
         return
      end if

      do slot = 1, count
         if (model%vocabulary%frequencies(slot) <= 0_int64 .or. &
             (model%vocabulary%active(slot) .neqv. &
              (model%vocabulary%frequencies(slot) >= min_frequency))) then
            close(unit)
            status = PT_ERR_CHECKPOINT
            call pretrain_model_destroy(model)
            return
         end if
      end do
      call vocab_rebuild_index(model%vocabulary, vocab_status)
      if (vocab_status /= VOCAB_OK .or. model%vocabulary%active_count /= stored_active_count) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         call pretrain_model_destroy(model)
         return
      end if
      call allocate_model_storage(model, status)
      if (status /= PT_OK) then
         close(unit)
         call pretrain_model_destroy(model)
         return
      end if
      read(unit, iostat=io_status) model%base_bits
      if (io_status == 0) read(unit, iostat=io_status) model%semantic_bits
      if (io_status == 0) read(unit, iostat=io_status) model%experience_count
      if (io_status == 0) read(unit, iostat=io_status) stored_checksum
      close(unit)
      if (io_status /= 0) then
         status = PT_ERR_CHECKPOINT
         call pretrain_model_destroy(model)
         return
      end if
      if (any(model%experience_count < 0_int64) .or. &
          stored_checksum /= model_checkpoint_checksum(model)) then
         status = PT_ERR_CHECKPOINT
         call pretrain_model_destroy(model)
         return
      end if
      model%ready = .true.
   end subroutine pretrain_checkpoint_load

end module hdc_pretraining
