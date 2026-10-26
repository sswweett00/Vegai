! SPDX-License-Identifier: MIT
!==============================================================================
!  subword_bpe_training.f90
!  Deterministic bounded-memory merge learner for the byte-BPE codec.
!
!  This is an exact-rescan trainer: every accepted merge causes one complete
!  corpus rescan under the current tokenizer. Pair statistics are collision
!  checked in an explicitly bounded table; capacity exhaustion is a hard error,
!  never silent approximate counting. The cost is deliberately transparent:
!  O(number_of_added_merges * corpus_bytes) time and O(pair_table_capacity)
!  working memory. It is an auditable tokenizer-construction path, not a claim
!  that this reference trainer is the fastest way to build a huge tokenizer.
!==============================================================================
module subword_bpe_training
   use, intrinsic :: iso_fortran_env, only: int32, int64
   use subword_bpe
   use utf8_record_stream
   implicit none
   private

   integer(int32), parameter, public :: BPET_OK = 0_int32
   integer(int32), parameter, public :: BPET_ERR_INVALID_ARGUMENT = -850_int32
   integer(int32), parameter, public :: BPET_ERR_ALLOCATION = -851_int32
   integer(int32), parameter, public :: BPET_ERR_IO = -852_int32
   integer(int32), parameter, public :: BPET_ERR_PAIR_CAPACITY = -853_int32
   integer(int32), parameter, public :: BPET_ERR_COUNTER_OVERFLOW = -854_int32
   integer(int32), parameter, public :: BPET_ERR_CODEC = -855_int32
   integer(int32), parameter, public :: BPET_ERR_TOKEN_CAPACITY = -856_int32

   type, public :: bpe_training_config_t
      integer(int32) :: target_vocabulary_size = 1024_int32
      integer(int64) :: minimum_pair_frequency = 2_int64
      ! Exact collision-checked statistics table. Raise this rather than
      ! accepting silently dropped pairs on a diverse corpus.
      integer(int32) :: pair_table_capacity = 65537_int32
      integer(int32) :: maximum_line_bytes = BPE_MAX_INPUT_BYTES
      ! Bound for the in-memory apply-merge trainer.  Exceeding it is a hard
      ! BPET_ERR_TOKEN_CAPACITY error (train the tokenizer on a governed
      ! sample instead); statistics are never silently truncated.
      integer(int64) :: maximum_in_memory_tokens = 16777216_int64
   end type bpe_training_config_t

   type, public :: bpe_training_metrics_t
      integer(int64) :: corpus_passes = 0_int64
      integer(int64) :: records_scanned = 0_int64
      integer(int64) :: retained_input_bytes = 0_int64
      integer(int64) :: oversized_records = 0_int64
      integer(int64) :: merges_added = 0_int64
      integer(int64) :: final_best_pair_frequency = 0_int64
      logical :: exhausted_before_target = .false.
   end type bpe_training_metrics_t

   public :: bpe_training_config_is_valid, bpe_training_metrics_reset, bpe_train_utf8_file
   public :: bpe_train_utf8_file_in_memory

contains

   pure logical function bpe_training_config_is_valid(config)
      type(bpe_training_config_t), intent(in) :: config

      bpe_training_config_is_valid = config%target_vocabulary_size >= BPE_BASE_VOCABULARY .and. &
                                     config%minimum_pair_frequency >= 1_int64 .and. &
                                     config%pair_table_capacity >= 3_int32 .and. &
                                     config%maximum_line_bytes >= 1_int32 .and. &
                                     config%maximum_line_bytes <= BPE_MAX_INPUT_BYTES .and. &
                                     config%maximum_in_memory_tokens >= 2_int64 .and. &
                                     config%maximum_in_memory_tokens <= int(huge(0_int32), int64)
   end function bpe_training_config_is_valid

   subroutine bpe_training_metrics_reset(metrics)
      type(bpe_training_metrics_t), intent(out) :: metrics

      metrics = bpe_training_metrics_t()
   end subroutine bpe_training_metrics_reset

   pure integer(int64) function training_pair_key(left_token, right_token) result(key)
      integer(int32), intent(in) :: left_token, right_token
      integer(int64), parameter :: low32_mask = int(z'00000000FFFFFFFF', int64)

      key = ior(shiftl(int(left_token, int64), 32), iand(int(right_token, int64), low32_mask))
   end function training_pair_key

   pure integer(int32) function training_hash_slot(key, capacity) result(slot)
      integer(int64), intent(in) :: key
      integer(int32), intent(in) :: capacity
      integer(int64) :: mixed

      mixed = ieor(key, shiftr(key, 29))
      mixed = ieor(mixed, shiftl(mixed, 17))
      mixed = ieor(mixed, shiftr(mixed, 23))
      slot = int(modulo(iand(mixed, huge(0_int64)), int(capacity, int64)), int32) + 1_int32
   end function training_hash_slot

   subroutine pair_table_increment(keys, left_tokens, right_tokens, counts, used, left_token, right_token, status)
      integer(int64), intent(inout) :: keys(:), counts(:)
      integer(int32), intent(inout) :: left_tokens(:), right_tokens(:)
      logical, intent(inout) :: used(:)
      integer(int32), intent(in) :: left_token, right_token
      integer(int32), intent(out) :: status
      integer(int32) :: slot, probe, capacity
      integer(int64) :: key

      status = BPET_OK
      capacity = size(keys, kind=int32)
      key = training_pair_key(left_token, right_token)
      slot = training_hash_slot(key, capacity)
      do probe = 1_int32, capacity
         if (.not. used(slot)) then
            used(slot) = .true.
            keys(slot) = key
            left_tokens(slot) = left_token
            right_tokens(slot) = right_token
            counts(slot) = 1_int64
            return
         end if
         if (keys(slot) == key) then
            if (counts(slot) == huge(0_int64)) then
               status = BPET_ERR_COUNTER_OVERFLOW
               return
            end if
            counts(slot) = counts(slot) + 1_int64
            return
         end if
         slot = modulo(slot, capacity) + 1_int32
      end do
      status = BPET_ERR_PAIR_CAPACITY
   end subroutine pair_table_increment

   subroutine count_current_pairs(model, path, config, keys, left_tokens, right_tokens, counts, used, metrics, status)
      type(bpe_model_t), intent(in) :: model
      character(len=*), intent(in) :: path
      type(bpe_training_config_t), intent(in) :: config
      integer(int64), intent(inout) :: keys(:), counts(:)
      integer(int32), intent(inout) :: left_tokens(:), right_tokens(:)
      logical, intent(inout) :: used(:)
      type(bpe_training_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      character(len=BPE_MAX_INPUT_BYTES) :: line
      integer(int32) :: tokens(BPE_MAX_INPUT_BYTES), token_count, line_bytes, pair_index, bpe_status, stream_status
      integer :: unit, io_status
      logical :: end_of_file, oversized

      status = BPET_OK
      keys = 0_int64
      left_tokens = 0_int32
      right_tokens = 0_int32
      counts = 0_int64
      used = .false.
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = BPET_ERR_IO
         return
      end if
      do
         call utf8_stream_read_record(unit, line, line_bytes, end_of_file, oversized, stream_status)
         if (stream_status /= UTF8_STREAM_OK) then
            close(unit)
            status = BPET_ERR_IO
            return
         end if
         if (end_of_file) exit
         metrics%records_scanned = metrics%records_scanned + 1_int64
         if (oversized .or. line_bytes > config%maximum_line_bytes) then
            metrics%oversized_records = metrics%oversized_records + 1_int64
            cycle
         end if
         metrics%retained_input_bytes = metrics%retained_input_bytes + int(line_bytes, int64)
         if (line_bytes < 2_int32) cycle
         call bpe_encode(model, line, line_bytes, tokens, token_count, bpe_status)
         if (bpe_status /= BPE_OK) then
            close(unit)
            status = BPET_ERR_CODEC
            return
         end if
         do pair_index = 1_int32, token_count - 1_int32
            call pair_table_increment(keys, left_tokens, right_tokens, counts, used, tokens(pair_index), &
                                      tokens(pair_index + 1_int32), status)
            if (status /= BPET_OK) then
               close(unit)
               return
            end if
         end do
      end do
      close(unit, iostat=io_status)
      if (io_status /= 0) status = BPET_ERR_IO
   end subroutine count_current_pairs

   subroutine choose_best_pair(left_tokens, right_tokens, counts, used, minimum_frequency, left_token, right_token, frequency)
      integer(int32), intent(in) :: left_tokens(:), right_tokens(:)
      integer(int64), intent(in) :: counts(:)
      logical, intent(in) :: used(:)
      integer(int64), intent(in) :: minimum_frequency
      integer(int32), intent(out) :: left_token, right_token
      integer(int64), intent(out) :: frequency
      integer(int32) :: slot

      left_token = 0_int32
      right_token = 0_int32
      frequency = 0_int64
      do slot = 1_int32, size(counts, kind=int32)
         if (.not. used(slot) .or. counts(slot) < minimum_frequency) cycle
         if (counts(slot) > frequency .or. &
             (counts(slot) == frequency .and. (left_token == 0_int32 .or. left_tokens(slot) < left_token .or. &
              (left_tokens(slot) == left_token .and. right_tokens(slot) < right_token)))) then
            left_token = left_tokens(slot)
            right_token = right_tokens(slot)
            frequency = counts(slot)
         end if
      end do
   end subroutine choose_best_pair

   subroutine bpe_train_utf8_file(model, path, config, metrics, status)
      type(bpe_model_t), intent(inout) :: model
      character(len=*), intent(in) :: path
      type(bpe_training_config_t), intent(in) :: config
      type(bpe_training_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      integer(int64), allocatable :: keys(:), counts(:)
      integer(int32), allocatable :: left_tokens(:), right_tokens(:)
      logical, allocatable :: used(:)
      integer :: allocation_status
      integer(int32) :: left_token, right_token, merged_token, bpe_status
      integer(int64) :: frequency

      status = BPET_OK
      if (.not. bpe_is_initialized(model) .or. .not. bpe_training_config_is_valid(config) .or. len_trim(path) == 0 .or. &
          config%target_vocabulary_size < model%token_count .or. &
          config%target_vocabulary_size > BPE_BASE_VOCABULARY + model%maximum_merges) then
         status = BPET_ERR_INVALID_ARGUMENT
         return
      end if
      allocate(keys(config%pair_table_capacity), counts(config%pair_table_capacity), &
               left_tokens(config%pair_table_capacity), right_tokens(config%pair_table_capacity), &
               used(config%pair_table_capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         status = BPET_ERR_ALLOCATION
         return
      end if
      do while (model%token_count < config%target_vocabulary_size)
         call count_current_pairs(model, path, config, keys, left_tokens, right_tokens, counts, used, metrics, status)
         if (status /= BPET_OK) exit
         metrics%corpus_passes = metrics%corpus_passes + 1_int64
         call choose_best_pair(left_tokens, right_tokens, counts, used, config%minimum_pair_frequency, &
                               left_token, right_token, frequency)
         metrics%final_best_pair_frequency = frequency
         if (frequency < config%minimum_pair_frequency) then
            metrics%exhausted_before_target = .true.
            exit
         end if
         call bpe_add_merge(model, left_token, right_token, merged_token, bpe_status)
         if (bpe_status /= BPE_OK) then
            status = BPET_ERR_CODEC
            exit
         end if
         metrics%merges_added = metrics%merges_added + 1_int64
      end do
      deallocate(keys, counts, left_tokens, right_tokens, used)
   end subroutine bpe_train_utf8_file

   ! In-memory apply-merge trainer.  It tokenizes the retained corpus once
   ! under the current model, then alternates exact pair counting with a
   ! left-to-right non-overlapping application of the single newly chosen
   ! merge.  A new merge always has the largest rank, so re-encoding under
   ! the extended table (the exact-rescan algorithm) first reproduces the
   ! previous sequences and then applies this one pair greedily leftmost --
   ! which is exactly one left-to-right scan.  Both trainers therefore
   ! learn bit-identical merge sequences; this one avoids O(merges) file
   ! rescans and full heap re-encodes, which made multi-thousand-merge
   ! tokenizer training on megabyte samples impractical.  Memory is
   ! O(retained_tokens + pair_table_capacity) and strictly fail-closed.
   subroutine bpe_train_utf8_file_in_memory(model, path, config, metrics, status)
      type(bpe_model_t), intent(inout) :: model
      character(len=*), intent(in) :: path
      type(bpe_training_config_t), intent(in) :: config
      type(bpe_training_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      character(len=BPE_MAX_INPUT_BYTES) :: line
      integer(int32) :: line_tokens(BPE_MAX_INPUT_BYTES)
      integer(int32), allocatable :: all_tokens(:), left_tokens(:), right_tokens(:)
      integer(int64), allocatable :: record_start(:), record_length(:), keys(:), counts(:)
      logical, allocatable :: used(:)
      integer :: unit, io_status, allocation_status
      integer(int32) :: token_count, line_bytes, bpe_status, stream_status
      integer(int32) :: left_token, right_token, merged_token
      integer(int64) :: file_bytes, capacity, total_tokens, record_count, frequency
      integer(int64) :: record_index, offset, read_position, write_position, new_length
      logical :: end_of_file, oversized

      status = BPET_OK
      if (.not. bpe_is_initialized(model) .or. .not. bpe_training_config_is_valid(config) .or. len_trim(path) == 0 .or. &
          config%target_vocabulary_size < model%token_count .or. &
          config%target_vocabulary_size > BPE_BASE_VOCABULARY + model%maximum_merges) then
         status = BPET_ERR_INVALID_ARGUMENT
         return
      end if
      ! Encoded tokens never exceed source bytes, so the file size is an
      ! exact allocation bound and no growth/reallocation path is needed.
      inquire(file=trim(path), size=file_bytes)
      if (file_bytes < 0_int64) then
         status = BPET_ERR_IO
         return
      end if
      capacity = min(config%maximum_in_memory_tokens, max(2_int64, file_bytes))
      allocate(all_tokens(capacity), record_start(capacity), record_length(capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         status = BPET_ERR_ALLOCATION
         return
      end if

      total_tokens = 0_int64
      record_count = 0_int64
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         deallocate(all_tokens, record_start, record_length)
         status = BPET_ERR_IO
         return
      end if
      do
         call utf8_stream_read_record(unit, line, line_bytes, end_of_file, oversized, stream_status)
         if (stream_status /= UTF8_STREAM_OK) then
            close(unit)
            deallocate(all_tokens, record_start, record_length)
            status = BPET_ERR_IO
            return
         end if
         if (end_of_file) exit
         metrics%records_scanned = metrics%records_scanned + 1_int64
         if (oversized .or. line_bytes > config%maximum_line_bytes) then
            metrics%oversized_records = metrics%oversized_records + 1_int64
            cycle
         end if
         metrics%retained_input_bytes = metrics%retained_input_bytes + int(line_bytes, int64)
         if (line_bytes < 2_int32) cycle
         call bpe_encode(model, line, line_bytes, line_tokens, token_count, bpe_status)
         if (bpe_status /= BPE_OK) then
            close(unit)
            deallocate(all_tokens, record_start, record_length)
            status = BPET_ERR_CODEC
            return
         end if
         if (token_count < 2_int32) cycle
         if (total_tokens + int(token_count, int64) > capacity) then
            close(unit)
            deallocate(all_tokens, record_start, record_length)
            status = BPET_ERR_TOKEN_CAPACITY
            return
         end if
         record_count = record_count + 1_int64
         record_start(record_count) = total_tokens + 1_int64
         record_length(record_count) = int(token_count, int64)
         all_tokens(total_tokens + 1_int64:total_tokens + int(token_count, int64)) = line_tokens(:token_count)
         total_tokens = total_tokens + int(token_count, int64)
      end do
      close(unit, iostat=io_status)
      if (io_status /= 0) then
         deallocate(all_tokens, record_start, record_length)
         status = BPET_ERR_IO
         return
      end if

      allocate(keys(config%pair_table_capacity), counts(config%pair_table_capacity), &
               left_tokens(config%pair_table_capacity), right_tokens(config%pair_table_capacity), &
               used(config%pair_table_capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         deallocate(all_tokens, record_start, record_length)
         status = BPET_ERR_ALLOCATION
         return
      end if

      do while (model%token_count < config%target_vocabulary_size)
         keys = 0_int64
         left_tokens = 0_int32
         right_tokens = 0_int32
         counts = 0_int64
         used = .false.
         do record_index = 1_int64, record_count
            do offset = 0_int64, record_length(record_index) - 2_int64
               call pair_table_increment(keys, left_tokens, right_tokens, counts, used, &
                                         all_tokens(record_start(record_index) + offset), &
                                         all_tokens(record_start(record_index) + offset + 1_int64), status)
               if (status /= BPET_OK) exit
            end do
            if (status /= BPET_OK) exit
         end do
         if (status /= BPET_OK) exit
         metrics%corpus_passes = metrics%corpus_passes + 1_int64
         call choose_best_pair(left_tokens, right_tokens, counts, used, config%minimum_pair_frequency, &
                               left_token, right_token, frequency)
         metrics%final_best_pair_frequency = frequency
         if (frequency < config%minimum_pair_frequency) then
            metrics%exhausted_before_target = .true.
            exit
         end if
         call bpe_add_merge(model, left_token, right_token, merged_token, bpe_status)
         if (bpe_status /= BPE_OK) then
            status = BPET_ERR_CODEC
            exit
         end if
         metrics%merges_added = metrics%merges_added + 1_int64
         ! In-place compaction: writes always trail reads because each step
         ! consumes at least one token and emits exactly one.
         write_position = 1_int64
         do record_index = 1_int64, record_count
            read_position = record_start(record_index)
            record_start(record_index) = write_position
            new_length = 0_int64
            offset = 0_int64
            do while (offset < record_length(record_index))
               if (offset + 1_int64 < record_length(record_index) .and. &
                   all_tokens(read_position + offset) == left_token .and. &
                   all_tokens(read_position + offset + 1_int64) == right_token) then
                  all_tokens(write_position + new_length) = merged_token
                  new_length = new_length + 1_int64
                  offset = offset + 2_int64
               else
                  all_tokens(write_position + new_length) = all_tokens(read_position + offset)
                  new_length = new_length + 1_int64
                  offset = offset + 1_int64
               end if
            end do
            record_length(record_index) = new_length
            write_position = write_position + new_length
         end do
         total_tokens = write_position - 1_int64
      end do
      deallocate(keys, counts, left_tokens, right_tokens, used)
      deallocate(all_tokens, record_start, record_length)
   end subroutine bpe_train_utf8_file_in_memory

end module subword_bpe_training
