! SPDX-License-Identifier: MIT
!==============================================================================
!  causal_ngram_lm.f90
!  Bounded, UTF-8-preserving causal next-token language model.
!
!  This module is deliberately separate from Vegai's HDC association model.
!  It trains an actual causal objective: for each observed token it retains a
!  bounded heavy-hitter sketch of P(next_token | previous token) and
!  P(next_token | previous two tokens).  It can therefore score and generate
!  continuations, unlike semantic-neighbour lookup or TSV retrieval.
!
!  It is still a compact count-based n-gram model, not a neural Transformer:
!  finite successor sketches lose tail information, contexts collide under a
!  bounded set-associative cache, and its continuations are local statistical
!  completions rather than instruction following or factual reasoning.
!==============================================================================
module causal_ngram_lm
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64, iostat_end
   implicit none
   private

   integer(int32), parameter, public :: CLM_OK = 0_int32
   integer(int32), parameter, public :: CLM_ERR_INVALID_ARGUMENT = -700_int32
   integer(int32), parameter, public :: CLM_ERR_ALLOCATION = -701_int32
   integer(int32), parameter, public :: CLM_ERR_IO = -702_int32
   integer(int32), parameter, public :: CLM_ERR_FORMAT = -703_int32
   integer(int32), parameter, public :: CLM_ERR_NOT_READY = -704_int32
   integer(int32), parameter, public :: CLM_ERR_OVERFLOW = -705_int32

   integer(int32), parameter, public :: CLM_TOKEN_BYTES = 64_int32
   integer(int32), parameter, public :: CLM_MAX_LINE_BYTES = 8192_int32
   integer(int32), parameter, public :: CLM_MAX_TOKENS_PER_LINE = 2048_int32
   integer(int32), parameter :: CLM_BOS = 1_int32
   integer(int32), parameter :: CLM_EOS = 2_int32
   character(len=12), parameter :: CLM_MAGIC = 'VEGAI_CLM_V1'
   integer(int32), parameter :: CLM_VERSION = 1_int32

   type, public :: causal_lm_config_t
      integer(int32) :: max_vocabulary = 65538_int32
      integer(int32) :: successors_per_context = 12_int32
      integer(int32) :: trigram_buckets = 131072_int32
      integer(int32) :: trigram_ways = 4_int32
      integer(int64) :: min_estimated_frequency = 2_int64
   end type causal_lm_config_t

   type, public :: causal_lm_metrics_t
      integer(int64) :: input_bytes = 0_int64
      integer(int64) :: records = 0_int64
      integer(int64) :: raw_tokens = 0_int64
      integer(int64) :: vocabulary_candidate_tokens = 0_int64
      integer(int64) :: active_vocabulary = 0_int64
      integer(int64) :: known_training_tokens = 0_int64
      integer(int64) :: unknown_training_tokens = 0_int64
      integer(int64) :: bigram_events = 0_int64
      integer(int64) :: trigram_events = 0_int64
      integer(int64) :: malformed_or_overlong_records = 0_int64
   end type causal_lm_metrics_t

   type, public :: causal_lm_evaluation_t
      integer(int64) :: records = 0_int64
      integer(int64) :: eligible_transitions = 0_int64
      integer(int64) :: top1_hits = 0_int64
      integer(int64) :: successor_coverage_hits = 0_int64
      integer(int64) :: trigram_context_hits = 0_int64
      integer(int64) :: bigram_context_hits = 0_int64
      real(real64) :: top1_accuracy = 0.0_real64
      real(real64) :: successor_coverage = 0.0_real64
   end type causal_lm_evaluation_t

   type, public :: causal_lm_t
      type(causal_lm_config_t) :: config
      logical :: initialized = .false.
      logical :: vocabulary_finalized = .false.
      integer(int32) :: candidate_buckets = 0_int32
      integer(int32) :: candidate_ways = 4_int32
      integer(int32) :: vocabulary_slots = 0_int32
      integer(int32) :: active_vocabulary = 0_int32
      ! Direct-mapped four-way space-saving vocabulary candidate table.  A
      ! token's fingerprint chooses one bucket, so lookup remains O(1) in the
      ! training hot path and memory remains bounded for multi-gigabyte input.
      character(len=CLM_TOKEN_BYTES), allocatable :: token_text(:)
      integer(int64), allocatable :: token_estimates(:)
      logical, allocatable :: token_active(:)
      integer(int64), allocatable :: unigram_counts(:)
      integer(int64), allocatable :: bigram_totals(:)
      integer(int32), allocatable :: bigram_successors(:, :)
      integer(int64), allocatable :: bigram_counts(:, :)
      ! Fixed set-associative (previous-previous, previous) context cache.
      integer(int32), allocatable :: trigram_first(:, :)
      integer(int32), allocatable :: trigram_second(:, :)
      integer(int64), allocatable :: trigram_seen(:, :)
      integer(int64), allocatable :: trigram_totals(:, :)
      integer(int32), allocatable :: trigram_successors(:, :, :)
      integer(int64), allocatable :: trigram_counts(:, :, :)
   end type causal_lm_t

   public :: causal_lm_init, causal_lm_destroy, causal_lm_is_ready
   public :: causal_lm_build_vocabulary, causal_lm_train_file
   public :: causal_lm_save, causal_lm_load, causal_lm_generate
   public :: causal_lm_evaluate_file, causal_lm_model_bytes
   public :: causal_lm_tokenize_utf8

contains

   pure logical function valid_config(config)
      type(causal_lm_config_t), intent(in) :: config
      integer(int64) :: slots

      slots = int(config%trigram_buckets, int64) * int(config%trigram_ways, int64)
      valid_config = config%max_vocabulary >= 10_int32 .and. &
                     config%successors_per_context >= 2_int32 .and. &
                     config%successors_per_context <= 64_int32 .and. &
                     config%trigram_buckets >= 1_int32 .and. &
                     config%trigram_ways >= 1_int32 .and. config%trigram_ways <= 8_int32 .and. &
                     config%min_estimated_frequency >= 1_int64 .and. &
                     slots > 0_int64 .and. slots <= int(huge(0_int32), int64)
   end function valid_config

   pure logical function causal_lm_is_ready(model)
      type(causal_lm_t), intent(in) :: model

      causal_lm_is_ready = model%initialized .and. model%vocabulary_finalized .and. &
                           model%active_vocabulary >= 2_int32 .and. allocated(model%token_text) .and. &
                           allocated(model%bigram_successors) .and. allocated(model%trigram_successors)
   end function causal_lm_is_ready

   subroutine causal_lm_destroy(model)
      type(causal_lm_t), intent(inout) :: model

      if (allocated(model%token_text)) deallocate(model%token_text)
      if (allocated(model%token_estimates)) deallocate(model%token_estimates)
      if (allocated(model%token_active)) deallocate(model%token_active)
      if (allocated(model%unigram_counts)) deallocate(model%unigram_counts)
      if (allocated(model%bigram_totals)) deallocate(model%bigram_totals)
      if (allocated(model%bigram_successors)) deallocate(model%bigram_successors)
      if (allocated(model%bigram_counts)) deallocate(model%bigram_counts)
      if (allocated(model%trigram_first)) deallocate(model%trigram_first)
      if (allocated(model%trigram_second)) deallocate(model%trigram_second)
      if (allocated(model%trigram_seen)) deallocate(model%trigram_seen)
      if (allocated(model%trigram_totals)) deallocate(model%trigram_totals)
      if (allocated(model%trigram_successors)) deallocate(model%trigram_successors)
      if (allocated(model%trigram_counts)) deallocate(model%trigram_counts)
      model = causal_lm_t()
   end subroutine causal_lm_destroy

   subroutine causal_lm_init(model, config, status)
      type(causal_lm_t), intent(inout) :: model
      type(causal_lm_config_t), intent(in) :: config
      integer(int32), intent(out) :: status
      integer :: allocation_status
      integer(int32) :: candidate_slots

      call causal_lm_destroy(model)
      status = CLM_OK
      if (.not. valid_config(config)) then
         status = CLM_ERR_INVALID_ARGUMENT
         return
      end if
      candidate_slots = config%max_vocabulary - 2_int32
      if (modulo(candidate_slots, model%candidate_ways) /= 0_int32) then
         status = CLM_ERR_INVALID_ARGUMENT
         return
      end if
      model%config = config
      model%candidate_buckets = candidate_slots / model%candidate_ways
      model%vocabulary_slots = config%max_vocabulary

      allocate(model%token_text(config%max_vocabulary), stat=allocation_status)
      if (allocation_status /= 0) then; status = CLM_ERR_ALLOCATION; call causal_lm_destroy(model); return; end if
      allocate(model%token_estimates(config%max_vocabulary), stat=allocation_status)
      if (allocation_status /= 0) then; status = CLM_ERR_ALLOCATION; call causal_lm_destroy(model); return; end if
      allocate(model%token_active(config%max_vocabulary), stat=allocation_status)
      if (allocation_status /= 0) then; status = CLM_ERR_ALLOCATION; call causal_lm_destroy(model); return; end if
      allocate(model%unigram_counts(config%max_vocabulary), model%bigram_totals(config%max_vocabulary), &
               stat=allocation_status)
      if (allocation_status /= 0) then; status = CLM_ERR_ALLOCATION; call causal_lm_destroy(model); return; end if
      allocate(model%bigram_successors(config%successors_per_context, config%max_vocabulary), &
               model%bigram_counts(config%successors_per_context, config%max_vocabulary), stat=allocation_status)
      if (allocation_status /= 0) then; status = CLM_ERR_ALLOCATION; call causal_lm_destroy(model); return; end if
      allocate(model%trigram_first(config%trigram_ways, config%trigram_buckets), &
               model%trigram_second(config%trigram_ways, config%trigram_buckets), &
               model%trigram_seen(config%trigram_ways, config%trigram_buckets), &
               model%trigram_totals(config%trigram_ways, config%trigram_buckets), stat=allocation_status)
      if (allocation_status /= 0) then; status = CLM_ERR_ALLOCATION; call causal_lm_destroy(model); return; end if
      allocate(model%trigram_successors(config%successors_per_context, config%trigram_ways, config%trigram_buckets), &
               model%trigram_counts(config%successors_per_context, config%trigram_ways, config%trigram_buckets), &
               stat=allocation_status)
      if (allocation_status /= 0) then; status = CLM_ERR_ALLOCATION; call causal_lm_destroy(model); return; end if

      model%token_text = ''
      model%token_estimates = 0_int64
      model%token_active = .false.
      model%unigram_counts = 0_int64
      model%bigram_totals = 0_int64
      model%bigram_successors = 0_int32
      model%bigram_counts = 0_int64
      model%trigram_first = 0_int32
      model%trigram_second = 0_int32
      model%trigram_seen = 0_int64
      model%trigram_totals = 0_int64
      model%trigram_successors = 0_int32
      model%trigram_counts = 0_int64
      model%token_text(CLM_BOS) = '<bos>'
      model%token_text(CLM_EOS) = '<eos>'
      model%token_estimates(CLM_BOS:CLM_EOS) = huge(0_int64) / 4_int64
      model%token_active(CLM_BOS:CLM_EOS) = .true.
      model%active_vocabulary = 2_int32
      model%initialized = .true.
   end subroutine causal_lm_init

   pure logical function byte_is_word(character)
      character(len=1), intent(in) :: character
      integer :: code

      code = iachar(character)
      byte_is_word = (code >= iachar('A') .and. code <= iachar('Z')) .or. &
                     (code >= iachar('a') .and. code <= iachar('z')) .or. &
                     (code >= iachar('0') .and. code <= iachar('9')) .or. &
                     character == "'" .or. character == '-' .or. code >= 128
   end function byte_is_word

   pure logical function byte_is_punctuation(character)
      character(len=1), intent(in) :: character
      byte_is_punctuation = index('.,;:!?()[]{}"', character) > 0
   end function byte_is_punctuation

   pure subroutine append_token_byte(token, token_length, character, too_long)
      character(len=CLM_TOKEN_BYTES), intent(inout) :: token
      integer(int32), intent(inout) :: token_length
      character(len=1), intent(in) :: character
      logical, intent(inout) :: too_long

      if (too_long) return
      if (token_length >= CLM_TOKEN_BYTES) then
         too_long = .true.
         return
      end if
      token_length = token_length + 1_int32
      token(token_length:token_length) = character
   end subroutine append_token_byte

   pure subroutine emit_causal_token(current, current_length, too_long, tokens, n_tokens, truncated)
      character(len=CLM_TOKEN_BYTES), intent(in) :: current
      integer(int32), intent(in) :: current_length
      logical, intent(in) :: too_long
      character(len=CLM_TOKEN_BYTES), intent(inout) :: tokens(:)
      integer(int32), intent(inout) :: n_tokens
      logical, intent(inout) :: truncated
      character(len=CLM_TOKEN_BYTES) :: normalized

      if (current_length <= 0_int32) return
      if (too_long) then
         truncated = .true.
         return
      end if
      if (n_tokens >= size(tokens, kind=int32)) then
         truncated = .true.
         return
      end if
      call lowercase_turkish_token(current(:current_length), normalized)
      if (len_trim(normalized) <= 0) then
         truncated = .true.
         return
      end if
      n_tokens = n_tokens + 1_int32
      tokens(n_tokens) = normalized
   end subroutine emit_causal_token

   pure subroutine lowercase_turkish_token(input, output)
      ! UTF-8 byte-aware casing for the Turkish alphabet.  It deliberately
      ! leaves unrelated scripts unchanged instead of corrupting a codepoint.
      character(len=*), intent(in) :: input
      character(len=CLM_TOKEN_BYTES), intent(out) :: output
      integer :: source, destination, code, next_code

      output = ''
      source = 1
      destination = 1
      do while (source <= len_trim(input) .and. destination <= CLM_TOKEN_BYTES)
         code = iachar(input(source:source))
         if (code >= iachar('A') .and. code <= iachar('Z')) then
            if (input(source:source) == 'I') then
               if (destination + 1 > CLM_TOKEN_BYTES) return
               output(destination:destination + 1) = achar(196)//achar(177) ! ı
               destination = destination + 2
            else
               output(destination:destination) = achar(code + iachar('a') - iachar('A'))
               destination = destination + 1
            end if
            source = source + 1
         else if (code == 195 .and. source < len_trim(input)) then
            next_code = iachar(input(source + 1:source + 1))
            if (next_code == 135 .or. next_code == 150 .or. next_code == 156 .or. next_code == 156 .or. &
                next_code == 130 .or. next_code == 142) then
               if (destination + 1 > CLM_TOKEN_BYTES) return
               output(destination:destination) = achar(195)
               select case (next_code)
               case (135); output(destination + 1:destination + 1) = achar(167) ! Ç -> ç
               case (150); output(destination + 1:destination + 1) = achar(182) ! Ö -> ö
               case (156); output(destination + 1:destination + 1) = achar(188) ! Ü -> ü
               case (130); output(destination + 1:destination + 1) = achar(162) ! Â -> â
               case (142); output(destination + 1:destination + 1) = achar(174) ! Î -> î
               end select
               destination = destination + 2
               source = source + 2
            else
               output(destination:destination) = input(source:source)
               destination = destination + 1
               source = source + 1
            end if
         else if (code == 196 .and. source < len_trim(input)) then
            next_code = iachar(input(source + 1:source + 1))
            if (next_code == 158) then
               if (destination + 1 > CLM_TOKEN_BYTES) return
               output(destination:destination + 1) = achar(196)//achar(159) ! Ğ -> ğ
               destination = destination + 2
               source = source + 2
            else if (next_code == 176) then
               output(destination:destination) = 'i' ! İ -> i
               destination = destination + 1
               source = source + 2
            else
               output(destination:destination) = input(source:source)
               destination = destination + 1
               source = source + 1
            end if
         else if (code == 197 .and. source < len_trim(input)) then
            next_code = iachar(input(source + 1:source + 1))
            if (next_code == 158) then
               if (destination + 1 > CLM_TOKEN_BYTES) return
               output(destination:destination + 1) = achar(197)//achar(159) ! Ş -> ş
               destination = destination + 2
               source = source + 2
            else
               output(destination:destination) = input(source:source)
               destination = destination + 1
               source = source + 1
            end if
         else
            output(destination:destination) = input(source:source)
            destination = destination + 1
            source = source + 1
         end if
      end do
   end subroutine lowercase_turkish_token

   pure subroutine causal_lm_tokenize_utf8(line, tokens, n_tokens, truncated)
      character(len=*), intent(in) :: line
      character(len=CLM_TOKEN_BYTES), intent(out) :: tokens(:)
      integer(int32), intent(out) :: n_tokens
      logical, intent(out) :: truncated
      character(len=CLM_TOKEN_BYTES) :: current
      integer(int32) :: position, current_length
      logical :: too_long

      tokens = ''
      n_tokens = 0_int32
      truncated = .false.
      current = ''
      current_length = 0_int32
      too_long = .false.
      do position = 1_int32, int(len_trim(line), int32)
         if (byte_is_word(line(position:position))) then
            call append_token_byte(current, current_length, line(position:position), too_long)
         else
            call emit_causal_token(current, current_length, too_long, tokens, n_tokens, truncated)
            current = ''
            current_length = 0_int32
            too_long = .false.
            if (byte_is_punctuation(line(position:position))) then
               if (n_tokens >= size(tokens, kind=int32)) then
                  truncated = .true.
               else
                  n_tokens = n_tokens + 1_int32
                  tokens(n_tokens) = line(position:position)
               end if
            end if
         end if
      end do
      call emit_causal_token(current, current_length, too_long, tokens, n_tokens, truncated)
   end subroutine causal_lm_tokenize_utf8

   pure function token_fingerprint(token) result(fingerprint)
      character(len=*), intent(in) :: token
      integer(int64) :: fingerprint
      integer :: position

      fingerprint = int(z'9E3779B97F4A7C15', int64)
      do position = 1, len_trim(token)
         fingerprint = ieor(fingerprint, int(iachar(token(position:position)), int64))
         fingerprint = ieor(ishftc(fingerprint, 17), ishftc(fingerprint, -13))
         fingerprint = ieor(fingerprint, int(position, int64) * int(z'BF58476D1CE4E5B9', int64))
      end do
      fingerprint = ieor(fingerprint, shiftr(fingerprint, 29))
   end function token_fingerprint

   pure function bucket_for_fingerprint(fingerprint, buckets) result(bucket)
      integer(int64), intent(in) :: fingerprint
      integer(int32), intent(in) :: buckets
      integer(int32) :: bucket
      integer(int64) :: positive

      positive = iand(fingerprint, huge(0_int64))
      bucket = int(modulo(positive, int(buckets, int64)), int32) + 1_int32
   end function bucket_for_fingerprint

   pure function candidate_slot(model, bucket, way) result(slot)
      type(causal_lm_t), intent(in) :: model
      integer(int32), intent(in) :: bucket, way
      integer(int32) :: slot
      slot = 2_int32 + (bucket - 1_int32) * model%candidate_ways + way
   end function candidate_slot

   subroutine observe_vocabulary_token(model, token, status)
      type(causal_lm_t), intent(inout) :: model
      character(len=*), intent(in) :: token
      integer(int32), intent(out) :: status
      integer(int32) :: bucket, way, slot, replacement_way, replacement_slot
      integer(int64) :: fingerprint, minimum

      status = CLM_OK
      if (.not. model%initialized .or. model%vocabulary_finalized .or. len_trim(token) <= 0) then
         status = CLM_ERR_INVALID_ARGUMENT
         return
      end if
      fingerprint = token_fingerprint(token)
      bucket = bucket_for_fingerprint(fingerprint, model%candidate_buckets)
      replacement_way = 1_int32
      minimum = huge(0_int64)
      do way = 1_int32, model%candidate_ways
         slot = candidate_slot(model, bucket, way)
         if (len_trim(model%token_text(slot)) == 0) then
            model%token_text(slot) = token
            model%token_estimates(slot) = 1_int64
            return
         end if
         if (trim(model%token_text(slot)) == trim(token)) then
            if (model%token_estimates(slot) == huge(0_int64)) then
               status = CLM_ERR_OVERFLOW
               return
            end if
            model%token_estimates(slot) = model%token_estimates(slot) + 1_int64
            return
         end if
         if (model%token_estimates(slot) < minimum) then
            minimum = model%token_estimates(slot)
            replacement_way = way
         end if
      end do
      replacement_slot = candidate_slot(model, bucket, replacement_way)
      if (minimum == huge(0_int64)) then
         status = CLM_ERR_OVERFLOW
         return
      end if
      model%token_text(replacement_slot) = token
      model%token_estimates(replacement_slot) = minimum + 1_int64
   end subroutine observe_vocabulary_token

   subroutine finalize_vocabulary(model, status)
      type(causal_lm_t), intent(inout) :: model
      integer(int32), intent(out) :: status
      integer(int32) :: slot

      status = CLM_OK
      if (.not. model%initialized .or. model%vocabulary_finalized) then
         status = CLM_ERR_INVALID_ARGUMENT
         return
      end if
      model%token_active = .false.
      model%token_active(CLM_BOS:CLM_EOS) = .true.
      model%active_vocabulary = 2_int32
      do slot = 3_int32, model%vocabulary_slots
         if (len_trim(model%token_text(slot)) > 0 .and. &
             model%token_estimates(slot) >= model%config%min_estimated_frequency) then
            model%token_active(slot) = .true.
            model%active_vocabulary = model%active_vocabulary + 1_int32
         else
            model%token_text(slot) = ''
            model%token_estimates(slot) = 0_int64
         end if
      end do
      model%vocabulary_finalized = .true.
   end subroutine finalize_vocabulary

   subroutine lookup_token(model, token, slot, found)
      type(causal_lm_t), intent(in) :: model
      character(len=*), intent(in) :: token
      integer(int32), intent(out) :: slot
      logical, intent(out) :: found
      integer(int32) :: bucket, way, candidate

      slot = 0_int32
      found = .false.
      if (.not. model%initialized .or. len_trim(token) <= 0) return
      bucket = bucket_for_fingerprint(token_fingerprint(token), model%candidate_buckets)
      do way = 1_int32, model%candidate_ways
         candidate = candidate_slot(model, bucket, way)
         if (model%token_active(candidate) .and. trim(model%token_text(candidate)) == trim(token)) then
            slot = candidate
            found = .true.
            return
         end if
      end do
   end subroutine lookup_token

   subroutine read_causal_record(unit, line, eof, overlong, status)
      integer, intent(in) :: unit
      character(len=CLM_MAX_LINE_BYTES), intent(out) :: line
      logical, intent(out) :: eof, overlong
      integer(int32), intent(out) :: status
      integer :: io_status

      line = ''
      eof = .false.
      overlong = .false.
      status = CLM_OK
      read(unit, '(A)', iostat=io_status) line
      if (io_status == iostat_end) then
         eof = .true.
         return
      end if
      if (io_status /= 0) then
         status = CLM_ERR_IO
         return
      end if
      ! The corpus constructor guarantees bounded records.  A nonblank final
      ! byte means it may have been truncated by formatted record input.
      if (len_trim(line) == CLM_MAX_LINE_BYTES) overlong = .true.
   end subroutine read_causal_record

   subroutine causal_lm_build_vocabulary(model, corpus_path, metrics, status)
      type(causal_lm_t), intent(inout) :: model
      character(len=*), intent(in) :: corpus_path
      type(causal_lm_metrics_t), intent(out) :: metrics
      integer(int32), intent(out) :: status
      character(len=CLM_MAX_LINE_BYTES) :: line
      character(len=CLM_TOKEN_BYTES) :: tokens(CLM_MAX_TOKENS_PER_LINE)
      integer :: unit, io_status
      integer(int32) :: n_tokens, token_index, local_status
      logical :: eof, overlong, truncated

      metrics = causal_lm_metrics_t()
      status = CLM_OK
      if (.not. model%initialized .or. model%vocabulary_finalized .or. len_trim(corpus_path) == 0) then
         status = CLM_ERR_INVALID_ARGUMENT
         return
      end if
      open(newunit=unit, file=trim(corpus_path), status='old', action='read', form='formatted', iostat=io_status)
      if (io_status /= 0) then
         status = CLM_ERR_IO
         return
      end if
      do
         call read_causal_record(unit, line, eof, overlong, local_status)
         if (local_status /= CLM_OK) then; close(unit); status = local_status; return; end if
         if (eof) exit
         metrics%records = metrics%records + 1_int64
         metrics%input_bytes = metrics%input_bytes + int(len_trim(line), int64) + 1_int64
         if (overlong) then
            metrics%malformed_or_overlong_records = metrics%malformed_or_overlong_records + 1_int64
            cycle
         end if
         call causal_lm_tokenize_utf8(line, tokens, n_tokens, truncated)
         if (truncated) metrics%malformed_or_overlong_records = metrics%malformed_or_overlong_records + 1_int64
         do token_index = 1_int32, n_tokens
            call observe_vocabulary_token(model, tokens(token_index), local_status)
            if (local_status /= CLM_OK) then; close(unit); status = local_status; return; end if
            metrics%raw_tokens = metrics%raw_tokens + 1_int64
         end do
      end do
      close(unit)
      call finalize_vocabulary(model, status)
      if (status /= CLM_OK) return
      metrics%active_vocabulary = int(model%active_vocabulary, int64)
      metrics%vocabulary_candidate_tokens = int(count(model%token_active(3:)), int64)
   end subroutine causal_lm_build_vocabulary

   subroutine safe_increment(counter, status)
      integer(int64), intent(inout) :: counter
      integer(int32), intent(out) :: status
      status = CLM_OK
      if (counter == huge(0_int64)) then
         status = CLM_ERR_OVERFLOW
      else
         counter = counter + 1_int64
      end if
   end subroutine safe_increment

   subroutine update_successors(successors, counts, total, target, status)
      integer(int32), intent(inout) :: successors(:)
      integer(int64), intent(inout) :: counts(:)
      integer(int64), intent(inout) :: total
      integer(int32), intent(in) :: target
      integer(int32), intent(out) :: status
      integer :: index, replacement
      integer(int64) :: minimum

      call safe_increment(total, status)
      if (status /= CLM_OK) return
      replacement = 1
      minimum = huge(0_int64)
      do index = 1, size(successors)
         if (successors(index) == target) then
            call safe_increment(counts(index), status)
            return
         end if
         if (successors(index) == 0_int32) then
            successors(index) = target
            counts(index) = 1_int64
            return
         end if
         if (counts(index) < minimum) then
            minimum = counts(index)
            replacement = index
         end if
      end do
      if (minimum == huge(0_int64)) then
         status = CLM_ERR_OVERFLOW
         return
      end if
      successors(replacement) = target
      counts(replacement) = minimum + 1_int64
   end subroutine update_successors

   pure function pair_fingerprint(first, second) result(fingerprint)
      integer(int32), intent(in) :: first, second
      integer(int64) :: fingerprint

      fingerprint = ieor(int(first, int64) * int(z'9E3779B97F4A7C15', int64), &
                         int(second, int64) * int(z'C2B2AE3D27D4EB4F', int64))
      fingerprint = ieor(fingerprint, shiftr(fingerprint, 31))
      fingerprint = ieor(fingerprint, ishftc(fingerprint, 23))
   end function pair_fingerprint

   subroutine find_or_allocate_trigram_context(model, first, second, bucket, way, status)
      type(causal_lm_t), intent(inout) :: model
      integer(int32), intent(in) :: first, second
      integer(int32), intent(out) :: bucket, way
      integer(int32), intent(out) :: status
      integer(int32) :: candidate, replacement
      integer(int64) :: minimum

      status = CLM_OK
      bucket = bucket_for_fingerprint(pair_fingerprint(first, second), model%config%trigram_buckets)
      replacement = 1_int32
      minimum = huge(0_int64)
      do candidate = 1_int32, model%config%trigram_ways
         if (model%trigram_seen(candidate, bucket) > 0_int64 .and. &
             model%trigram_first(candidate, bucket) == first .and. &
             model%trigram_second(candidate, bucket) == second) then
            way = candidate
            return
         end if
         if (model%trigram_seen(candidate, bucket) == 0_int64) then
            model%trigram_first(candidate, bucket) = first
            model%trigram_second(candidate, bucket) = second
            model%trigram_seen(candidate, bucket) = 0_int64
            model%trigram_totals(candidate, bucket) = 0_int64
            model%trigram_successors(:, candidate, bucket) = 0_int32
            model%trigram_counts(:, candidate, bucket) = 0_int64
            way = candidate
            return
         end if
         if (model%trigram_seen(candidate, bucket) < minimum) then
            minimum = model%trigram_seen(candidate, bucket)
            replacement = candidate
         end if
      end do
      ! Bounded space-saving context replacement.  The incoming context obtains
      ! the evicted context's count as an error bound, then receives its event.
      model%trigram_first(replacement, bucket) = first
      model%trigram_second(replacement, bucket) = second
      model%trigram_seen(replacement, bucket) = minimum
      model%trigram_totals(replacement, bucket) = 0_int64
      model%trigram_successors(:, replacement, bucket) = 0_int32
      model%trigram_counts(:, replacement, bucket) = 0_int64
      way = replacement
   end subroutine find_or_allocate_trigram_context

   subroutine update_transition(model, previous_two, previous_one, target, metrics, status)
      type(causal_lm_t), intent(inout) :: model
      integer(int32), intent(in) :: previous_two, previous_one, target
      type(causal_lm_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      integer(int32) :: bucket, way

      status = CLM_OK
      call safe_increment(model%unigram_counts(target), status)
      if (status /= CLM_OK) return
      call update_successors(model%bigram_successors(:, previous_one), model%bigram_counts(:, previous_one), &
                             model%bigram_totals(previous_one), target, status)
      if (status /= CLM_OK) return
      call safe_increment(metrics%bigram_events, status)
      if (status /= CLM_OK) return
      call find_or_allocate_trigram_context(model, previous_two, previous_one, bucket, way, status)
      if (status /= CLM_OK) return
      call safe_increment(model%trigram_seen(way, bucket), status)
      if (status /= CLM_OK) return
      call update_successors(model%trigram_successors(:, way, bucket), model%trigram_counts(:, way, bucket), &
                             model%trigram_totals(way, bucket), target, status)
      if (status /= CLM_OK) return
      call safe_increment(metrics%trigram_events, status)
   end subroutine update_transition

   subroutine causal_lm_train_file(model, corpus_path, metrics, status)
      type(causal_lm_t), intent(inout) :: model
      character(len=*), intent(in) :: corpus_path
      type(causal_lm_metrics_t), intent(out) :: metrics
      integer(int32), intent(out) :: status
      character(len=CLM_MAX_LINE_BYTES) :: line
      character(len=CLM_TOKEN_BYTES) :: tokens(CLM_MAX_TOKENS_PER_LINE)
      integer(int32) :: slots(CLM_MAX_TOKENS_PER_LINE)
      integer :: unit, io_status
      integer(int32) :: n_tokens, token_index, local_status, previous_one, previous_two
      logical :: eof, overlong, truncated, found

      metrics = causal_lm_metrics_t()
      status = CLM_OK
      if (.not. causal_lm_is_ready(model) .or. len_trim(corpus_path) == 0) then
         status = CLM_ERR_NOT_READY
         return
      end if
      open(newunit=unit, file=trim(corpus_path), status='old', action='read', form='formatted', iostat=io_status)
      if (io_status /= 0) then; status = CLM_ERR_IO; return; end if
      do
         call read_causal_record(unit, line, eof, overlong, local_status)
         if (local_status /= CLM_OK) then; close(unit); status = local_status; return; end if
         if (eof) exit
         metrics%records = metrics%records + 1_int64
         metrics%input_bytes = metrics%input_bytes + int(len_trim(line), int64) + 1_int64
         if (overlong) then
            metrics%malformed_or_overlong_records = metrics%malformed_or_overlong_records + 1_int64
            cycle
         end if
         call causal_lm_tokenize_utf8(line, tokens, n_tokens, truncated)
         if (truncated) metrics%malformed_or_overlong_records = metrics%malformed_or_overlong_records + 1_int64
         previous_two = CLM_BOS
         previous_one = CLM_BOS
         do token_index = 1_int32, n_tokens
            metrics%raw_tokens = metrics%raw_tokens + 1_int64
            call lookup_token(model, tokens(token_index), slots(token_index), found)
            if (.not. found) then
               metrics%unknown_training_tokens = metrics%unknown_training_tokens + 1_int64
               cycle
            end if
            metrics%known_training_tokens = metrics%known_training_tokens + 1_int64
            call update_transition(model, previous_two, previous_one, slots(token_index), metrics, local_status)
            if (local_status /= CLM_OK) then; close(unit); status = local_status; return; end if
            previous_two = previous_one
            previous_one = slots(token_index)
         end do
         call update_transition(model, previous_two, previous_one, CLM_EOS, metrics, local_status)
         if (local_status /= CLM_OK) then; close(unit); status = local_status; return; end if
      end do
      close(unit)
      metrics%active_vocabulary = int(model%active_vocabulary, int64)
   end subroutine causal_lm_train_file

   subroutine find_trigram_context(model, first, second, bucket, way, found)
      type(causal_lm_t), intent(in) :: model
      integer(int32), intent(in) :: first, second
      integer(int32), intent(out) :: bucket, way
      logical, intent(out) :: found
      integer(int32) :: candidate

      bucket = bucket_for_fingerprint(pair_fingerprint(first, second), model%config%trigram_buckets)
      way = 0_int32
      found = .false.
      do candidate = 1_int32, model%config%trigram_ways
         if (model%trigram_seen(candidate, bucket) > 0_int64 .and. &
             model%trigram_first(candidate, bucket) == first .and. &
             model%trigram_second(candidate, bucket) == second) then
            way = candidate
            found = .true.
            return
         end if
      end do
   end subroutine find_trigram_context

   subroutine select_best_successor(successors, counts, selected, coverage_target, target_in_coverage)
      integer(int32), intent(in) :: successors(:)
      integer(int64), intent(in) :: counts(:)
      integer(int32), intent(out) :: selected
      integer(int32), intent(in), optional :: coverage_target
      logical, intent(out), optional :: target_in_coverage
      integer :: index
      integer(int64) :: best_count

      selected = 0_int32
      best_count = -1_int64
      if (present(target_in_coverage)) target_in_coverage = .false.
      do index = 1, size(successors)
         if (successors(index) <= 0_int32) cycle
         if (counts(index) > best_count .or. (counts(index) == best_count .and. successors(index) < selected)) then
            selected = successors(index)
            best_count = counts(index)
         end if
         if (present(coverage_target) .and. present(target_in_coverage)) then
            if (successors(index) == coverage_target) target_in_coverage = .true.
         end if
      end do
   end subroutine select_best_successor

   subroutine evaluate_transition(model, previous_two, previous_one, target, evaluation)
      type(causal_lm_t), intent(in) :: model
      integer(int32), intent(in) :: previous_two, previous_one, target
      type(causal_lm_evaluation_t), intent(inout) :: evaluation
      integer(int32) :: bucket, way, predicted
      logical :: found, covered

      evaluation%eligible_transitions = evaluation%eligible_transitions + 1_int64
      call find_trigram_context(model, previous_two, previous_one, bucket, way, found)
      if (found .and. model%trigram_totals(way, bucket) > 0_int64) then
         evaluation%trigram_context_hits = evaluation%trigram_context_hits + 1_int64
         call select_best_successor(model%trigram_successors(:, way, bucket), model%trigram_counts(:, way, bucket), &
                                    predicted, target, covered)
      else if (model%bigram_totals(previous_one) > 0_int64) then
         evaluation%bigram_context_hits = evaluation%bigram_context_hits + 1_int64
         call select_best_successor(model%bigram_successors(:, previous_one), model%bigram_counts(:, previous_one), &
                                    predicted, target, covered)
      else
         predicted = 0_int32
         covered = .false.
      end if
      if (predicted == target) evaluation%top1_hits = evaluation%top1_hits + 1_int64
      if (covered) evaluation%successor_coverage_hits = evaluation%successor_coverage_hits + 1_int64
   end subroutine evaluate_transition

   subroutine causal_lm_evaluate_file(model, corpus_path, evaluation, status)
      type(causal_lm_t), intent(in) :: model
      character(len=*), intent(in) :: corpus_path
      type(causal_lm_evaluation_t), intent(out) :: evaluation
      integer(int32), intent(out) :: status
      character(len=CLM_MAX_LINE_BYTES) :: line
      character(len=CLM_TOKEN_BYTES) :: tokens(CLM_MAX_TOKENS_PER_LINE)
      integer :: unit, io_status
      integer(int32) :: n_tokens, token_index, previous_one, previous_two, target
      logical :: eof, overlong, truncated, found

      evaluation = causal_lm_evaluation_t()
      status = CLM_OK
      if (.not. causal_lm_is_ready(model) .or. len_trim(corpus_path) == 0) then; status = CLM_ERR_NOT_READY; return; end if
      open(newunit=unit, file=trim(corpus_path), status='old', action='read', form='formatted', iostat=io_status)
      if (io_status /= 0) then; status = CLM_ERR_IO; return; end if
      do
         call read_causal_record(unit, line, eof, overlong, status)
         if (status /= CLM_OK) then; close(unit); return; end if
         if (eof) exit
         evaluation%records = evaluation%records + 1_int64
         if (overlong) cycle
         call causal_lm_tokenize_utf8(line, tokens, n_tokens, truncated)
         previous_two = CLM_BOS
         previous_one = CLM_BOS
         do token_index = 1_int32, n_tokens
            call lookup_token(model, tokens(token_index), target, found)
            if (.not. found) cycle
            call evaluate_transition(model, previous_two, previous_one, target, evaluation)
            previous_two = previous_one
            previous_one = target
         end do
         call evaluate_transition(model, previous_two, previous_one, CLM_EOS, evaluation)
      end do
      close(unit)
      if (evaluation%eligible_transitions > 0_int64) then
         evaluation%top1_accuracy = real(evaluation%top1_hits, real64) / real(evaluation%eligible_transitions, real64)
         evaluation%successor_coverage = real(evaluation%successor_coverage_hits, real64) / &
                                         real(evaluation%eligible_transitions, real64)
      end if
   end subroutine causal_lm_evaluate_file

   subroutine next_random(state, value)
      integer(int64), intent(inout) :: state
      integer(int64), intent(out) :: value

      state = ieor(state, ishftc(state, 13))
      state = ieor(state, shiftr(state, 7))
      state = ieor(state, ishftc(state, 17))
      value = iand(state, huge(0_int64))
   end subroutine next_random

   subroutine sample_successor(successors, counts, temperature, state, selected)
      integer(int32), intent(in) :: successors(:)
      integer(int64), intent(in) :: counts(:)
      real(real64), intent(in) :: temperature
      integer(int64), intent(inout) :: state
      integer(int32), intent(out) :: selected
      real(real64) :: weights(size(successors)), total, draw, running
      integer(int64) :: random_value
      integer :: index

      selected = 0_int32
      weights = 0.0_real64
      total = 0.0_real64
      do index = 1, size(successors)
         if (successors(index) > 0_int32 .and. counts(index) > 0_int64) then
            weights(index) = exp(log(real(counts(index), real64)) / temperature)
            total = total + weights(index)
         end if
      end do
      if (total <= 0.0_real64) return
      call next_random(state, random_value)
      draw = real(random_value, real64) / real(huge(0_int64), real64) * total
      running = 0.0_real64
      do index = 1, size(successors)
         running = running + weights(index)
         if (draw <= running .and. successors(index) > 0_int32) then
            selected = successors(index)
            return
         end if
      end do
      call select_best_successor(successors, counts, selected)
   end subroutine sample_successor

   pure logical function is_punctuation_token(token)
      character(len=*), intent(in) :: token
      is_punctuation_token = len_trim(token) == 1 .and. index('.,;:!?)]}', token(1:1)) > 0
   end function is_punctuation_token

   subroutine append_generated_token(output, output_length, token, status)
      character(len=*), intent(inout) :: output
      integer(int32), intent(inout) :: output_length
      character(len=*), intent(in) :: token
      integer(int32), intent(inout) :: status
      integer(int32) :: token_length

      if (status /= CLM_OK) return
      token_length = int(len_trim(token), int32)
      if (token_length <= 0_int32) return
      if (output_length > 0_int32 .and. .not. is_punctuation_token(token)) then
         if (output_length >= len(output)) then; status = CLM_ERR_OVERFLOW; return; end if
         output_length = output_length + 1_int32
         output(output_length:output_length) = ' '
      end if
      if (output_length + token_length > len(output)) then; status = CLM_ERR_OVERFLOW; return; end if
      output(output_length + 1_int32:output_length + token_length) = token(:token_length)
      output_length = output_length + token_length
   end subroutine append_generated_token

   subroutine causal_lm_generate(model, prompt, max_new_tokens, temperature, seed, completion, status)
      type(causal_lm_t), intent(in) :: model
      character(len=*), intent(in) :: prompt
      integer(int32), intent(in) :: max_new_tokens
      real(real64), intent(in) :: temperature
      integer(int64), intent(in) :: seed
      character(len=*), intent(out) :: completion
      integer(int32), intent(out) :: status
      character(len=CLM_TOKEN_BYTES) :: tokens(CLM_MAX_TOKENS_PER_LINE)
      integer(int32) :: n_tokens, token_index, previous_two, previous_one, selected, bucket, way, output_length
      integer(int64) :: random_state
      logical :: truncated, found

      completion = ''
      status = CLM_OK
      if (.not. causal_lm_is_ready(model) .or. max_new_tokens <= 0_int32 .or. &
          temperature <= 0.0_real64 .or. len_trim(prompt) <= 0) then
         status = CLM_ERR_INVALID_ARGUMENT
         return
      end if
      call causal_lm_tokenize_utf8(prompt, tokens, n_tokens, truncated)
      previous_two = CLM_BOS
      previous_one = CLM_BOS
      do token_index = 1_int32, n_tokens
         call lookup_token(model, tokens(token_index), selected, found)
         if (.not. found) cycle
         previous_two = previous_one
         previous_one = selected
      end do
      random_state = seed
      if (random_state == 0_int64) random_state = int(z'6A09E667F3BCC909', int64)
      output_length = 0_int32
      do token_index = 1_int32, max_new_tokens
         call find_trigram_context(model, previous_two, previous_one, bucket, way, found)
         if (found .and. model%trigram_totals(way, bucket) > 0_int64) then
            call sample_successor(model%trigram_successors(:, way, bucket), model%trigram_counts(:, way, bucket), &
                                  temperature, random_state, selected)
         else if (model%bigram_totals(previous_one) > 0_int64) then
            call sample_successor(model%bigram_successors(:, previous_one), model%bigram_counts(:, previous_one), &
                                  temperature, random_state, selected)
         else
            exit
         end if
         if (selected <= 0_int32 .or. selected == CLM_EOS) exit
         call append_generated_token(completion, output_length, model%token_text(selected), status)
         if (status /= CLM_OK) return
         previous_two = previous_one
         previous_one = selected
      end do
      if (output_length == 0_int32) completion = '(devam üretilemedi)'
   end subroutine causal_lm_generate

   function causal_lm_model_bytes(model) result(bytes)
      type(causal_lm_t), intent(in) :: model
      integer(int64) :: bytes

      bytes = 0_int64
      if (.not. model%initialized) return
      bytes = bytes + int(size(model%token_text), int64) * int(CLM_TOKEN_BYTES, int64)
      bytes = bytes + int(size(model%token_estimates), int64) * 8_int64
      bytes = bytes + int(size(model%token_active), int64)
      bytes = bytes + int(size(model%unigram_counts), int64) * 8_int64
      bytes = bytes + int(size(model%bigram_totals), int64) * 8_int64
      bytes = bytes + int(size(model%bigram_successors), int64) * 4_int64
      bytes = bytes + int(size(model%bigram_counts), int64) * 8_int64
      bytes = bytes + int(size(model%trigram_first), int64) * 4_int64
      bytes = bytes + int(size(model%trigram_second), int64) * 4_int64
      bytes = bytes + int(size(model%trigram_seen), int64) * 8_int64
      bytes = bytes + int(size(model%trigram_totals), int64) * 8_int64
      bytes = bytes + int(size(model%trigram_successors), int64) * 4_int64
      bytes = bytes + int(size(model%trigram_counts), int64) * 8_int64
   end function causal_lm_model_bytes

   subroutine causal_lm_save(model, checkpoint_path, status)
      type(causal_lm_t), intent(in) :: model
      character(len=*), intent(in) :: checkpoint_path
      integer(int32), intent(out) :: status
      integer :: unit, io_status

      status = CLM_OK
      if (.not. causal_lm_is_ready(model) .or. len_trim(checkpoint_path) == 0) then; status = CLM_ERR_NOT_READY; return; end if
      open(newunit=unit, file=trim(checkpoint_path), status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then; status = CLM_ERR_IO; return; end if
      write(unit, iostat=io_status) CLM_MAGIC, CLM_VERSION, model%config, model%candidate_buckets, model%candidate_ways, &
                                  model%vocabulary_slots, model%active_vocabulary, model%vocabulary_finalized
      if (io_status == 0) write(unit, iostat=io_status) model%token_text, model%token_estimates, model%token_active, &
                                                   model%unigram_counts, model%bigram_totals, model%bigram_successors, &
                                                   model%bigram_counts, model%trigram_first, model%trigram_second, &
                                                   model%trigram_seen, model%trigram_totals, model%trigram_successors, &
                                                   model%trigram_counts
      close(unit)
      if (io_status /= 0) status = CLM_ERR_IO
   end subroutine causal_lm_save

   subroutine causal_lm_load(checkpoint_path, model, status)
      character(len=*), intent(in) :: checkpoint_path
      type(causal_lm_t), intent(inout) :: model
      integer(int32), intent(out) :: status
      character(len=12) :: magic
      integer(int32) :: version, candidate_buckets, candidate_ways, vocabulary_slots, active_vocabulary
      logical :: vocabulary_finalized
      type(causal_lm_config_t) :: config
      integer :: unit, io_status

      call causal_lm_destroy(model)
      status = CLM_OK
      if (len_trim(checkpoint_path) == 0) then; status = CLM_ERR_INVALID_ARGUMENT; return; end if
      open(newunit=unit, file=trim(checkpoint_path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then; status = CLM_ERR_IO; return; end if
      read(unit, iostat=io_status) magic, version, config, candidate_buckets, candidate_ways, vocabulary_slots, &
                                  active_vocabulary, vocabulary_finalized
      if (io_status /= 0 .or. magic /= CLM_MAGIC .or. version /= CLM_VERSION .or. .not. valid_config(config) .or. &
          vocabulary_slots /= config%max_vocabulary .or. candidate_ways /= 4_int32 .or. &
          candidate_buckets /= (config%max_vocabulary - 2_int32) / 4_int32) then
         close(unit); status = CLM_ERR_FORMAT; return
      end if
      call causal_lm_init(model, config, status)
      if (status /= CLM_OK) then; close(unit); return; end if
      read(unit, iostat=io_status) model%token_text, model%token_estimates, model%token_active, model%unigram_counts, &
                                  model%bigram_totals, model%bigram_successors, model%bigram_counts, model%trigram_first, &
                                  model%trigram_second, model%trigram_seen, model%trigram_totals, model%trigram_successors, &
                                  model%trigram_counts
      close(unit)
      if (io_status /= 0) then; call causal_lm_destroy(model); status = CLM_ERR_IO; return; end if
      model%active_vocabulary = active_vocabulary
      model%vocabulary_finalized = vocabulary_finalized
      if (.not. causal_lm_is_ready(model)) then; call causal_lm_destroy(model); status = CLM_ERR_FORMAT; end if
   end subroutine causal_lm_load

end module causal_ngram_lm
