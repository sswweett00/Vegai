! SPDX-License-Identifier: MIT
!==============================================================================
!  dialogue_retrieval.f90
!  Bounded, data-driven conversational retrieval over a trained Vegai HDC model.
!
!  This module deliberately does not synthesize prose or hide uncertainty.  It
!  encodes prompts from checkpoint-owned semantic hypervectors, retrieves only
!  a response explicitly supplied by an operator-owned TSV knowledge base, and
!  abstains when configurable grounding thresholds are not met.  The knowledge
!  base, not source-code branches, supplies every user-visible answer.
!==============================================================================
module dialogue_retrieval
   use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real32, iostat_end
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use hyper_types, only: HV_BITS, HV_WORDS, HV_OK, hv_t, hv_acc_t, hv_zero, hv_acc_reset, &
                          hv_acc_add, hv_acc_finalize, hv_equal, hv_hamming, hv_similarity_from_distance
   use text_tokenizer, only: TOKEN_LEN, MAX_LINE_CHARS, MAX_TOKENS_PER_LINE, tokenize_ascii_line
   use turkish_text, only: TURKISH_TEXT_OK, turkish_fold_prompt_utf8, turkish_validate_display_utf8
   use hdc_pretraining, only: PT_OK, PT_ERR_NOT_FOUND, hdc_pretrain_model_t, pretrain_model_is_ready, &
                               pretrain_lookup_token, pretrain_get_semantic_hv
   implicit none
   private

   integer(int32), parameter, public :: DIALOGUE_OK = 0_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_INVALID_ARGUMENT = -600_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_ALLOCATION = -601_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_IO = -602_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_FORMAT = -603_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_CAPACITY = -604_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_TOKENIZATION = -605_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_MODEL = -606_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_NO_KNOWN_TOKENS = -607_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_UNKNOWN_TOKEN = -608_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_DUPLICATE_PROMPT = -609_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_RECORD_TOO_LONG = -610_int32
   integer(int32), parameter, public :: DIALOGUE_ERR_EMPTY_INDEX = -611_int32

   ! These are explicit storage bounds rather than hidden allocation behavior.
   ! Input records remain subject to the repository-wide 8,192-byte bounded
   ! UTF-8 ingest ceiling; prompts/responses have narrower conversational
   ! limits so a huge answer cannot silently become process memory pressure.
   integer(int32), parameter, public :: DIALOGUE_MAX_PROMPT_CHARS = 1024_int32
   integer(int32), parameter, public :: DIALOGUE_MAX_RESPONSE_CHARS = 2048_int32

   integer(int64), parameter :: DIALOGUE_TIE_SEED = int(z'243F6A8885A308D3', int64)

   type, public :: dialogue_config_t
      integer(int32) :: max_entries = 4096_int32
      integer(int32) :: max_history_turns = 3_int32
      integer(int32) :: min_known_tokens = 1_int32
      ! The current prompt remains weighted more strongly than each historical
      ! prompt.  Both are policy fields, not response-specific rules.
      integer(int16) :: current_prompt_weight = 2_int16
      integer(int16) :: history_turn_weight = 1_int16
      ! Legacy semantic-only grounding policy. It remains the default so an
      ! existing deployment retains its checkpoint-only decision boundary.
      real(real32) :: minimum_similarity = 0.05_real32
      real(real32) :: minimum_margin = 0.01_real32
      ! A hybrid policy can combine the checkpoint's semantic association with
      ! deterministic lexical/morphological evidence from operator-owned data.
      ! It is opt-in (lexical_weight = 0 by default); scores are normalized to
      ! [0, 1] before the hybrid thresholds are applied.
      real(real32) :: semantic_weight = 1.0_real32
      real(real32) :: lexical_weight = 0.0_real32
      real(real32) :: minimum_hybrid_score = 0.50_real32
      real(real32) :: minimum_hybrid_margin = 0.01_real32
      logical :: require_all_prompt_tokens_known = .false.
      integer(int64) :: tie_seed = DIALOGUE_TIE_SEED
   end type dialogue_config_t

   type, public :: dialogue_index_t
      type(dialogue_config_t) :: config = dialogue_config_t()
      integer(int32) :: capacity = 0_int32
      integer(int32) :: count = 0_int32
      ! Packed SoA prompt embeddings; each column is one 16,384-bit prompt.
      integer(int64), allocatable :: prompt_bits(:, :)
      integer(int32), allocatable :: prompt_token_counts(:)
      integer(int32), allocatable :: prompt_ignored_token_counts(:)
      ! prompts preserves the source UTF-8 text for provenance/display; the
      ! parallel canonical form is the deterministic ASCII Turkish projection
      ! used by tokenization and lexical retrieval.
      character(len=:), allocatable :: prompts(:)
      character(len=:), allocatable :: canonical_prompts(:)
      character(len=:), allocatable :: responses(:)
   end type dialogue_index_t

   type, public :: dialogue_session_t
      ! -1 means uninitialized.  A capacity of zero is a valid stateless
      ! session, useful for deterministic one-shot retrieval and services that
      ! hold conversation state outside this process.
      integer(int32) :: capacity = -1_int32
      integer(int32) :: count = 0_int32
      integer(int32) :: next_slot = 1_int32
      integer(int64), allocatable :: prompt_bits(:, :)
   end type dialogue_session_t

   type, public :: dialogue_answer_t
      logical :: found = .false.
      integer(int32) :: entry = 0_int32
      integer(int32) :: candidates_examined = 0_int32
      integer(int32) :: query_known_tokens = 0_int32
      integer(int32) :: query_ignored_tokens = 0_int32
      real(real32) :: similarity = -1.0_real32
      real(real32) :: runner_up_similarity = -1.0_real32
      real(real32) :: margin = -1.0_real32
      ! These fields disclose the active hybrid evidence. In legacy mode the
      ! hybrid score is simply the normalized bipolar semantic similarity.
      real(real32) :: lexical_similarity = 0.0_real32
      real(real32) :: runner_up_lexical_similarity = 0.0_real32
      real(real32) :: hybrid_score = 0.0_real32
      real(real32) :: runner_up_hybrid_score = 0.0_real32
      real(real32) :: hybrid_margin = 0.0_real32
      character(len=DIALOGUE_MAX_PROMPT_CHARS) :: matched_prompt = ''
      character(len=DIALOGUE_MAX_RESPONSE_CHARS) :: response = ''
   end type dialogue_answer_t

   public :: dialogue_index_init, dialogue_index_destroy, dialogue_index_is_initialized
   public :: dialogue_index_add, dialogue_index_load_tsv, dialogue_index_footprint_bytes
   public :: dialogue_session_init, dialogue_session_destroy, dialogue_session_reset
   public :: dialogue_session_is_initialized, dialogue_ask

contains

   pure logical function dialogue_index_is_initialized(index) result(initialized)
      type(dialogue_index_t), intent(in) :: index

      ! Components are public so callers can inspect telemetry.  Validate both
      ! allocation and shape before a later data-plane access; do not rely on
      ! short-circuit evaluation of .and. in standard Fortran.
      initialized = .false.
      if (index%capacity <= 0_int32 .or. index%count < 0_int32 .or. &
          index%count > index%capacity) return
      if (.not. allocated(index%prompt_bits) .or. .not. allocated(index%prompt_token_counts) .or. &
          .not. allocated(index%prompt_ignored_token_counts) .or. .not. allocated(index%prompts) .or. &
          .not. allocated(index%canonical_prompts) .or. .not. allocated(index%responses)) return
      if (size(index%prompt_bits, 1) /= HV_WORDS .or. size(index%prompt_bits, 2) /= index%capacity) return
      if (size(index%prompt_token_counts) /= index%capacity .or. &
          size(index%prompt_ignored_token_counts) /= index%capacity .or. &
          size(index%prompts) /= index%capacity .or. &
          size(index%canonical_prompts) /= index%capacity .or. size(index%responses) /= index%capacity) return
      initialized = .true.
   end function dialogue_index_is_initialized

   pure logical function dialogue_session_is_initialized(session) result(initialized)
      type(dialogue_session_t), intent(in) :: session

      initialized = .false.
      if (session%capacity < 0_int32 .or. session%count < 0_int32 .or. &
          session%count > session%capacity .or. session%next_slot < 1_int32) return
      if (session%capacity == 0_int32) then
         initialized = .not. allocated(session%prompt_bits)
         return
      end if
      if (.not. allocated(session%prompt_bits)) return
      if (size(session%prompt_bits, 1) /= HV_WORDS .or. size(session%prompt_bits, 2) /= session%capacity .or. &
          session%next_slot > session%capacity) return
      initialized = .true.
   end function dialogue_session_is_initialized

   subroutine dialogue_index_destroy(index)
      type(dialogue_index_t), intent(inout) :: index

      if (allocated(index%prompt_bits)) deallocate(index%prompt_bits)
      if (allocated(index%prompt_token_counts)) deallocate(index%prompt_token_counts)
      if (allocated(index%prompt_ignored_token_counts)) deallocate(index%prompt_ignored_token_counts)
      if (allocated(index%prompts)) deallocate(index%prompts)
      if (allocated(index%canonical_prompts)) deallocate(index%canonical_prompts)
      if (allocated(index%responses)) deallocate(index%responses)
      index%config = dialogue_config_t()
      index%capacity = 0_int32
      index%count = 0_int32
   end subroutine dialogue_index_destroy

   subroutine dialogue_session_destroy(session)
      type(dialogue_session_t), intent(inout) :: session

      if (allocated(session%prompt_bits)) deallocate(session%prompt_bits)
      session%capacity = -1_int32
      session%count = 0_int32
      session%next_slot = 1_int32
   end subroutine dialogue_session_destroy

   subroutine validate_dialogue_config(config, status)
      type(dialogue_config_t), intent(in) :: config
      integer(int32), intent(out) :: status
      integer(int64) :: maximum_weighted_items

      status = DIALOGUE_OK
      if (config%max_entries <= 0_int32 .or. config%max_history_turns < 0_int32 .or. &
          config%min_known_tokens <= 0_int32 .or. config%current_prompt_weight <= 0_int16 .or. &
          config%history_turn_weight <= 0_int16) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      ! Ordered comparisons alone accept IEEE NaN because every NaN comparison
      ! is false.  Reject non-finite thresholds before they can silently turn a
      ! similarity or ambiguity gate into an always-pass policy.
      if (.not. ieee_is_finite(config%minimum_similarity) .or. &
          .not. ieee_is_finite(config%minimum_margin) .or. &
          .not. ieee_is_finite(config%semantic_weight) .or. &
          .not. ieee_is_finite(config%lexical_weight) .or. &
          .not. ieee_is_finite(config%minimum_hybrid_score) .or. &
          .not. ieee_is_finite(config%minimum_hybrid_margin)) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      ! Do this in a second IF. Fortran does not guarantee short-circuit
      ! evaluation of .or.; comparing a NaN here could otherwise defeat the
      ! fail-closed check under an invalid-operation floating-point trap.
      if (config%minimum_similarity < -1.0_real32 .or. config%minimum_similarity > 1.0_real32 .or. &
          config%minimum_margin < 0.0_real32 .or. config%minimum_margin > 2.0_real32 .or. &
          config%semantic_weight < 0.0_real32 .or. config%semantic_weight > 1.0_real32 .or. &
          config%lexical_weight < 0.0_real32 .or. config%lexical_weight > 1.0_real32 .or. &
          (config%semantic_weight + config%lexical_weight) <= 0.0_real32 .or. &
          config%minimum_hybrid_score < 0.0_real32 .or. config%minimum_hybrid_score > 1.0_real32 .or. &
          config%minimum_hybrid_margin < 0.0_real32 .or. config%minimum_hybrid_margin > 1.0_real32) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      ! A signed int16 accumulator is used for deterministic HDC bundling.  A
      ! policy that could overflow it is rejected before any session/index is
      ! allocated, even when ordinary prompts happen to be shorter.
      maximum_weighted_items = int(MAX_TOKENS_PER_LINE, int64) * &
                               int(config%current_prompt_weight, int64) + &
                               int(config%max_history_turns, int64) * &
                               int(config%history_turn_weight, int64)
      if (maximum_weighted_items > int(huge(0_int16), int64)) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
      end if
   end subroutine validate_dialogue_config

   subroutine dialogue_index_init(index, config, status)
      type(dialogue_index_t), intent(out) :: index
      type(dialogue_config_t), intent(in) :: config
      integer(int32), intent(out) :: status
      integer :: allocation_status

      index = dialogue_index_t()
      call validate_dialogue_config(config, status)
      if (status /= DIALOGUE_OK) return

      index%config = config
      index%capacity = config%max_entries
      allocate(index%prompt_bits(HV_WORDS, index%capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         status = DIALOGUE_ERR_ALLOCATION
         call dialogue_index_destroy(index)
         return
      end if
      allocate(index%prompt_token_counts(index%capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         status = DIALOGUE_ERR_ALLOCATION
         call dialogue_index_destroy(index)
         return
      end if
      allocate(index%prompt_ignored_token_counts(index%capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         status = DIALOGUE_ERR_ALLOCATION
         call dialogue_index_destroy(index)
         return
      end if
      allocate(character(len=DIALOGUE_MAX_PROMPT_CHARS) :: index%prompts(index%capacity), &
               stat=allocation_status)
      if (allocation_status /= 0) then
         status = DIALOGUE_ERR_ALLOCATION
         call dialogue_index_destroy(index)
         return
      end if
      allocate(character(len=DIALOGUE_MAX_PROMPT_CHARS) :: index%canonical_prompts(index%capacity), &
               stat=allocation_status)
      if (allocation_status /= 0) then
         status = DIALOGUE_ERR_ALLOCATION
         call dialogue_index_destroy(index)
         return
      end if
      allocate(character(len=DIALOGUE_MAX_RESPONSE_CHARS) :: index%responses(index%capacity), &
               stat=allocation_status)
      if (allocation_status /= 0) then
         status = DIALOGUE_ERR_ALLOCATION
         call dialogue_index_destroy(index)
         return
      end if
      index%prompt_bits = 0_int64
      index%prompt_token_counts = 0_int32
      index%prompt_ignored_token_counts = 0_int32
      index%prompts = ''
      index%canonical_prompts = ''
      index%responses = ''
      status = DIALOGUE_OK
   end subroutine dialogue_index_init

   subroutine dialogue_session_init(session, max_history_turns, status)
      type(dialogue_session_t), intent(out) :: session
      integer(int32), intent(in) :: max_history_turns
      integer(int32), intent(out) :: status
      integer :: allocation_status

      session = dialogue_session_t()
      if (max_history_turns < 0_int32) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      session%capacity = max_history_turns
      session%count = 0_int32
      session%next_slot = 1_int32
      if (session%capacity == 0_int32) then
         status = DIALOGUE_OK
         return
      end if
      allocate(session%prompt_bits(HV_WORDS, session%capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         status = DIALOGUE_ERR_ALLOCATION
         call dialogue_session_destroy(session)
         return
      end if
      session%prompt_bits = 0_int64
      status = DIALOGUE_OK
   end subroutine dialogue_session_init

   subroutine dialogue_session_reset(session, status)
      type(dialogue_session_t), intent(inout) :: session
      integer(int32), intent(out) :: status

      status = DIALOGUE_OK
      if (.not. dialogue_session_is_initialized(session)) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      session%count = 0_int32
      session%next_slot = 1_int32
      if (allocated(session%prompt_bits)) session%prompt_bits = 0_int64
   end subroutine dialogue_session_reset

   pure function dialogue_index_footprint_bytes(index) result(total_bytes)
      type(dialogue_index_t), intent(in) :: index
      integer(int64) :: total_bytes

      total_bytes = 0_int64
      if (.not. dialogue_index_is_initialized(index)) return
      total_bytes = total_bytes + size(index%prompt_bits, kind=int64) * 8_int64
      total_bytes = total_bytes + size(index%prompt_token_counts, kind=int64) * 4_int64
      total_bytes = total_bytes + size(index%prompt_ignored_token_counts, kind=int64) * 4_int64
      total_bytes = total_bytes + size(index%prompts, kind=int64) * int(DIALOGUE_MAX_PROMPT_CHARS, int64)
      total_bytes = total_bytes + size(index%canonical_prompts, kind=int64) * int(DIALOGUE_MAX_PROMPT_CHARS, int64)
      total_bytes = total_bytes + size(index%responses, kind=int64) * int(DIALOGUE_MAX_RESPONSE_CHARS, int64)
   end function dialogue_index_footprint_bytes

   subroutine validate_dialogue_prompt(text, canonical_text, status)
      character(len=*), intent(in) :: text
      character(len=*), intent(out) :: canonical_text
      integer(int32), intent(out) :: status
      integer(int32) :: text_status

      canonical_text = ''
      status = DIALOGUE_OK
      call turkish_validate_display_utf8(text, DIALOGUE_MAX_PROMPT_CHARS, text_status)
      if (text_status /= TURKISH_TEXT_OK) then
         status = DIALOGUE_ERR_TOKENIZATION
         return
      end if
      call turkish_fold_prompt_utf8(trim(text), canonical_text, text_status)
      if (text_status /= TURKISH_TEXT_OK) then
         status = DIALOGUE_ERR_TOKENIZATION
      end if
   end subroutine validate_dialogue_prompt

   subroutine validate_dialogue_response(text, status)
      character(len=*), intent(in) :: text
      integer(int32), intent(out) :: status
      integer(int32) :: text_status

      status = DIALOGUE_OK
      call turkish_validate_display_utf8(text, DIALOGUE_MAX_RESPONSE_CHARS, text_status)
      if (text_status /= TURKISH_TEXT_OK) status = DIALOGUE_ERR_TOKENIZATION
   end subroutine validate_dialogue_response

   subroutine dialogue_encode_prompt(index, model, prompt, prompt_hv, known_tokens, ignored_tokens, status, &
                                     canonical_prompt)
      type(dialogue_index_t), intent(in) :: index
      type(hdc_pretrain_model_t), intent(in) :: model
      character(len=*), intent(in) :: prompt
      type(hv_t), intent(out) :: prompt_hv
      integer(int32), intent(out) :: known_tokens, ignored_tokens
      integer(int32), intent(out) :: status
      character(len=*), intent(out), optional :: canonical_prompt
      character(len=DIALOGUE_MAX_PROMPT_CHARS) :: canonical_text
      character(len=TOKEN_LEN) :: tokens(MAX_TOKENS_PER_LINE)
      type(hv_acc_t) :: accumulator
      type(hv_t) :: semantic_hv
      integer(int32) :: token, n_tokens, slot, lookup_status, vector_status, accumulator_status
      integer(int64) :: non_ascii_bytes
      logical :: truncated, found

      prompt_hv = hv_zero()
      known_tokens = 0_int32
      ignored_tokens = 0_int32
      canonical_text = ''
      if (present(canonical_prompt)) canonical_prompt = ''
      status = DIALOGUE_OK
      if (.not. dialogue_index_is_initialized(index) .or. .not. pretrain_model_is_ready(model)) then
         status = DIALOGUE_ERR_MODEL
         return
      end if
      call validate_dialogue_prompt(prompt, canonical_text, status)
      if (status /= DIALOGUE_OK) return
      call tokenize_ascii_line(trim(canonical_text), tokens, n_tokens, truncated, non_ascii_bytes)
      if (truncated .or. non_ascii_bytes /= 0_int64) then
         status = DIALOGUE_ERR_TOKENIZATION
         return
      end if
      if (n_tokens == 0_int32) then
         status = DIALOGUE_ERR_NO_KNOWN_TOKENS
         return
      end if

      call hv_acc_reset(accumulator)
      do token = 1, n_tokens
         call pretrain_lookup_token(model, tokens(token), slot, found, lookup_status)
         if (lookup_status == PT_OK .and. found) then
            call pretrain_get_semantic_hv(model, slot, semantic_hv, vector_status)
            if (vector_status /= PT_OK) then
               status = DIALOGUE_ERR_MODEL
               return
            end if
            call hv_acc_add(accumulator, semantic_hv, accumulator_status, index%config%current_prompt_weight)
            if (accumulator_status /= HV_OK) then
               status = DIALOGUE_ERR_INVALID_ARGUMENT
               return
            end if
            known_tokens = known_tokens + 1_int32
         else if (lookup_status == PT_ERR_NOT_FOUND) then
            ignored_tokens = ignored_tokens + 1_int32
         else
            status = DIALOGUE_ERR_MODEL
            return
         end if
      end do
      if (known_tokens < index%config%min_known_tokens) then
         status = DIALOGUE_ERR_NO_KNOWN_TOKENS
         return
      end if
      if (index%config%require_all_prompt_tokens_known .and. ignored_tokens /= 0_int32) then
         status = DIALOGUE_ERR_UNKNOWN_TOKEN
         return
      end if
      prompt_hv = hv_acc_finalize(accumulator, index%config%tie_seed)
      if (present(canonical_prompt)) canonical_prompt = canonical_text
   end subroutine dialogue_encode_prompt

   subroutine dialogue_index_add(index, model, prompt, response, status)
      type(dialogue_index_t), intent(inout) :: index
      type(hdc_pretrain_model_t), intent(in) :: model
      character(len=*), intent(in) :: prompt, response
      integer(int32), intent(out) :: status
      type(hv_t) :: prompt_hv, existing_hv
      character(len=DIALOGUE_MAX_PROMPT_CHARS) :: canonical_prompt
      integer(int32) :: known_tokens, ignored_tokens, entry

      status = DIALOGUE_OK
      if (.not. dialogue_index_is_initialized(index)) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      if (index%count >= index%capacity) then
         status = DIALOGUE_ERR_CAPACITY
         return
      end if
      call validate_dialogue_response(response, status)
      if (status /= DIALOGUE_OK) return
      call dialogue_encode_prompt(index, model, prompt, prompt_hv, known_tokens, ignored_tokens, status, canonical_prompt)
      if (status /= DIALOGUE_OK) return
      do entry = 1, index%count
         existing_hv%bits = index%prompt_bits(:, entry)
         if (hv_equal(prompt_hv, existing_hv)) then
            status = DIALOGUE_ERR_DUPLICATE_PROMPT
            return
         end if
      end do
      entry = index%count + 1_int32
      index%prompt_bits(:, entry) = prompt_hv%bits
      index%prompt_token_counts(entry) = known_tokens
      index%prompt_ignored_token_counts(entry) = ignored_tokens
      index%prompts(entry) = trim(prompt)
      index%canonical_prompts(entry) = trim(canonical_prompt)
      index%responses(entry) = trim(response)
      index%count = entry
   end subroutine dialogue_index_add

   subroutine read_dialogue_record(unit, line, eof, overlong, status)
      integer, intent(in) :: unit
      character(len=*), intent(out) :: line
      logical, intent(out) :: eof, overlong
      integer(int32), intent(out) :: status
      character(len=1) :: byte
      integer :: io_status, line_length

      ! Use an unformatted stream rather than formatted records: this keeps the
      ! strict knowledge-base contract byte-visible on every host, including a
      ! literal CR in a CRLF file.  Drain an overlong physical record before
      ! reporting it, so no prefix can be interpreted as the next TSV row.
      line = ''
      eof = .false.
      overlong = .false.
      status = DIALOGUE_OK
      line_length = 0
      do
         read(unit, iostat=io_status) byte
         if (io_status == iostat_end) then
            if (line_length == 0 .and. .not. overlong) eof = .true.
            return
         end if
         if (io_status /= 0) then
            status = DIALOGUE_ERR_IO
            return
         end if
         if (byte == achar(10)) return
         if (overlong) cycle
         if (line_length >= len(line)) then
            overlong = .true.
            cycle
         end if
         line_length = line_length + 1
         line(line_length:line_length) = byte
      end do
   end subroutine read_dialogue_record

   subroutine parse_dialogue_tsv_row(line, prompt, response, status)
      character(len=*), intent(in) :: line
      character(len=DIALOGUE_MAX_PROMPT_CHARS), intent(out) :: prompt
      character(len=DIALOGUE_MAX_RESPONSE_CHARS), intent(out) :: response
      integer(int32), intent(out) :: status
      character(len=DIALOGUE_MAX_PROMPT_CHARS) :: canonical_prompt
      integer :: position, line_length, separator

      prompt = ''
      response = ''
      status = DIALOGUE_OK
      line_length = len_trim(line)
      if (line_length <= 0) then
         status = DIALOGUE_ERR_FORMAT
         return
      end if
      do position = 1, line_length
         ! Only the TAB separator is permitted as an ASCII control byte. UTF-8
         ! bytes are validated field-by-field after the separator is known.
         if (iachar(line(position:position)) < 32 .and. line(position:position) /= achar(9)) then
            status = DIALOGUE_ERR_TOKENIZATION
            return
         end if
      end do
      separator = index(line(:line_length), achar(9))
      if (separator <= 1 .or. separator >= line_length) then
         status = DIALOGUE_ERR_FORMAT
         return
      end if
      if (index(line(separator + 1:line_length), achar(9)) /= 0) then
         status = DIALOGUE_ERR_FORMAT
         return
      end if
      if (separator - 1 > DIALOGUE_MAX_PROMPT_CHARS .or. &
          line_length - separator > DIALOGUE_MAX_RESPONSE_CHARS) then
         status = DIALOGUE_ERR_RECORD_TOO_LONG
         return
      end if
      prompt = adjustl(line(:separator - 1))
      response = adjustl(line(separator + 1:line_length))
      call validate_dialogue_prompt(prompt, canonical_prompt, status)
      if (status /= DIALOGUE_OK) return
      call validate_dialogue_response(response, status)
   end subroutine parse_dialogue_tsv_row

   subroutine dialogue_index_load_tsv(index, model, path, entries_loaded, status)
      type(dialogue_index_t), intent(inout) :: index
      type(hdc_pretrain_model_t), intent(in) :: model
      character(len=*), intent(in) :: path
      integer(int32), intent(out) :: entries_loaded
      integer(int32), intent(out) :: status
      character(len=MAX_LINE_CHARS) :: line
      character(len=DIALOGUE_MAX_PROMPT_CHARS) :: prompt
      character(len=DIALOGUE_MAX_RESPONSE_CHARS) :: response
      logical :: eof, overlong
      integer :: unit, io_status

      entries_loaded = 0_int32
      status = DIALOGUE_OK
      if (.not. dialogue_index_is_initialized(index) .or. .not. pretrain_model_is_ready(model) .or. &
          index%count /= 0_int32 .or. len_trim(path) == 0) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', form='unformatted', &
           iostat=io_status)
      if (io_status /= 0) then
         status = DIALOGUE_ERR_IO
         call dialogue_index_destroy(index)
         return
      end if
      do
         call read_dialogue_record(unit, line, eof, overlong, status)
         if (status /= DIALOGUE_OK .or. eof) exit
         if (overlong) then
            status = DIALOGUE_ERR_RECORD_TOO_LONG
            exit
         end if
         if (len_trim(line) == 0) cycle
         call parse_dialogue_tsv_row(line, prompt, response, status)
         if (status /= DIALOGUE_OK) exit
         call dialogue_index_add(index, model, prompt, response, status)
         if (status /= DIALOGUE_OK) exit
         ! Knowledge-base prompts are publication-time data, unlike a live user
         ! utterance.  Refuse a row that would silently lose a checkpoint-OOV
         ! token so every indexed answer has a fully auditable retrieval key.
         if (index%prompt_ignored_token_counts(index%count) /= 0_int32) then
            status = DIALOGUE_ERR_UNKNOWN_TOKEN
            exit
         end if
         entries_loaded = entries_loaded + 1_int32
      end do
      close(unit)
      if (status == DIALOGUE_OK .and. entries_loaded == 0_int32) status = DIALOGUE_ERR_EMPTY_INDEX
      ! Never leave a partially trusted knowledge base usable after a strict
      ! publication/load failure.  The caller must explicitly reinitialize and
      ! load a corrected immutable TSV revision.
      if (status /= DIALOGUE_OK) then
         entries_loaded = 0_int32
         call dialogue_index_destroy(index)
      end if
   end subroutine dialogue_index_load_tsv

   subroutine compose_dialogue_query(index, session, prompt_hv, query_hv, status)
      type(dialogue_index_t), intent(in) :: index
      type(dialogue_session_t), intent(in) :: session
      type(hv_t), intent(in) :: prompt_hv
      type(hv_t), intent(out) :: query_hv
      integer(int32), intent(out) :: status
      type(hv_acc_t) :: accumulator
      type(hv_t) :: historical_hv
      integer(int32) :: slot, accumulator_status

      query_hv = hv_zero()
      status = DIALOGUE_OK
      if (.not. dialogue_session_is_initialized(session) .or. &
          session%capacity /= index%config%max_history_turns) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      call hv_acc_reset(accumulator)
      call hv_acc_add(accumulator, prompt_hv, accumulator_status, index%config%current_prompt_weight)
      if (accumulator_status /= HV_OK) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      if (session%capacity > 0_int32) then
         do slot = 1, session%count
            historical_hv%bits = session%prompt_bits(:, slot)
            call hv_acc_add(accumulator, historical_hv, accumulator_status, index%config%history_turn_weight)
            if (accumulator_status /= HV_OK) then
               status = DIALOGUE_ERR_INVALID_ARGUMENT
               return
            end if
         end do
      end if
      query_hv = hv_acc_finalize(accumulator, index%config%tie_seed)
   end subroutine compose_dialogue_query

   pure real(real32) function dialogue_bounded_edit_similarity(left, right) result(similarity)
      ! A bounded Levenshtein pass is used only after a shared Turkish root
      ! prefix has been established. It recognizes transparent inflectional
      ! forms such as kayit/kaydi without turning arbitrary short words into a
      ! lexical match. TOKEN_LEN bounds both stack rows and work per comparison.
      character(len=*), intent(in) :: left, right
      integer :: previous(0:TOKEN_LEN), current(0:TOKEN_LEN)
      integer :: left_length, right_length, left_position, right_position, substitution_cost

      similarity = 0.0_real32
      left_length = len_trim(left)
      right_length = len_trim(right)
      if (left_length <= 0 .or. right_length <= 0 .or. left_length > TOKEN_LEN .or. right_length > TOKEN_LEN) return
      do right_position = 0, right_length
         previous(right_position) = right_position
      end do
      do left_position = 1, left_length
         current(0) = left_position
         do right_position = 1, right_length
            substitution_cost = 1
            if (left(left_position:left_position) == right(right_position:right_position)) substitution_cost = 0
            current(right_position) = min(previous(right_position) + 1, current(right_position - 1) + 1, &
                                          previous(right_position - 1) + substitution_cost)
         end do
         previous(0:right_length) = current(0:right_length)
      end do
      similarity = 1.0_real32 - real(previous(right_length), real32) / real(max(left_length, right_length), real32)
   end function dialogue_bounded_edit_similarity

   pure real(real32) function dialogue_token_similarity(left, right) result(similarity)
      ! Turkish is agglutinative. Exact words remain the strongest lexical
      ! evidence. A conservative shared-root prefix plus bounded edit score
      ! supplies a transparent morphology-aware fallback without claiming
      ! linguistic understanding.
      character(len=*), intent(in) :: left, right
      integer :: left_length, right_length, common_length, position
      real(real32) :: edit_similarity

      similarity = 0.0_real32
      left_length = len_trim(left)
      right_length = len_trim(right)
      if (left_length <= 0 .or. right_length <= 0) return
      if (left(:left_length) == right(:right_length)) then
         similarity = 1.0_real32
         return
      end if
      if (min(left_length, right_length) < 5) return
      common_length = 0
      do position = 1, min(left_length, right_length)
         if (left(position:position) /= right(position:position)) exit
         common_length = common_length + 1
      end do
      ! At least three shared bytes are required before edit distance can
      ! contribute. This covers Turkish consonant softening plus suffixes while
      ! keeping generic spelling-neighbour matches out of the lexical score.
      if (common_length < 3) return
      if (common_length >= 4) then
         similarity = 0.60_real32 + 0.30_real32 * &
                      real(common_length, real32) / real(max(left_length, right_length), real32)
      end if
      edit_similarity = dialogue_bounded_edit_similarity(left, right)
      if (edit_similarity >= 0.55_real32) similarity = max(similarity, edit_similarity)
   end function dialogue_token_similarity

   subroutine dialogue_lexical_similarity(query, candidate, similarity)
      character(len=*), intent(in) :: query, candidate
      real(real32), intent(out) :: similarity
      character(len=TOKEN_LEN) :: query_tokens(MAX_TOKENS_PER_LINE)
      character(len=TOKEN_LEN) :: candidate_tokens(MAX_TOKENS_PER_LINE)
      logical :: query_truncated, candidate_truncated, used(MAX_TOKENS_PER_LINE)
      integer(int64) :: query_non_ascii, candidate_non_ascii
      integer(int32) :: query_count, candidate_count, query_position, candidate_position
      integer(int32) :: best_candidate, bigram_hits
      real(real32) :: token_score, best_token_score, matched_weight, unigram_score, bigram_score

      similarity = 0.0_real32
      call tokenize_ascii_line(trim(query), query_tokens, query_count, query_truncated, query_non_ascii)
      call tokenize_ascii_line(trim(candidate), candidate_tokens, candidate_count, candidate_truncated, &
                               candidate_non_ascii)
      if (query_truncated .or. candidate_truncated .or. query_non_ascii /= 0_int64 .or. &
          candidate_non_ascii /= 0_int64 .or. query_count <= 0_int32 .or. candidate_count <= 0_int32) return

      used = .false.
      matched_weight = 0.0_real32
      do query_position = 1, query_count
         best_candidate = 0_int32
         best_token_score = 0.0_real32
         do candidate_position = 1, candidate_count
            if (used(candidate_position)) cycle
            token_score = dialogue_token_similarity(query_tokens(query_position), candidate_tokens(candidate_position))
            if (token_score > best_token_score) then
               best_token_score = token_score
               best_candidate = candidate_position
            end if
         end do
         if (best_candidate > 0_int32) then
            used(best_candidate) = .true.
            matched_weight = matched_weight + best_token_score
         end if
      end do
      unigram_score = 2.0_real32 * matched_weight / real(query_count + candidate_count, real32)

      bigram_hits = 0_int32
      if (query_count >= 2_int32 .and. candidate_count >= 2_int32) then
         do query_position = 1, query_count - 1_int32
            do candidate_position = 1, candidate_count - 1_int32
               if (trim(query_tokens(query_position)) == trim(candidate_tokens(candidate_position)) .and. &
                   trim(query_tokens(query_position + 1_int32)) == trim(candidate_tokens(candidate_position + 1_int32))) then
                  bigram_hits = bigram_hits + 1_int32
                  exit
               end if
            end do
         end do
         bigram_score = 2.0_real32 * real(bigram_hits, real32) / &
                        real((query_count - 1_int32) + (candidate_count - 1_int32), real32)
      else
         bigram_score = 0.0_real32
      end if
      similarity = min(1.0_real32, max(0.0_real32, 0.85_real32 * unigram_score + 0.15_real32 * bigram_score))
   end subroutine dialogue_lexical_similarity

   pure real(real32) function dialogue_hybrid_score(config, semantic_similarity, lexical_similarity) result(score)
      type(dialogue_config_t), intent(in) :: config
      real(real32), intent(in) :: semantic_similarity, lexical_similarity
      real(real32) :: semantic_component, total_weight

      semantic_component = 0.5_real32 * (semantic_similarity + 1.0_real32)
      total_weight = config%semantic_weight + config%lexical_weight
      score = (config%semantic_weight * semantic_component + config%lexical_weight * lexical_similarity) / total_weight
   end function dialogue_hybrid_score

   subroutine select_dialogue_answer(index, query_hv, canonical_prompt, answer)
      type(dialogue_index_t), intent(in) :: index
      type(hv_t), intent(in) :: query_hv
      character(len=*), intent(in) :: canonical_prompt
      type(dialogue_answer_t), intent(out) :: answer
      type(hv_t) :: candidate_hv
      integer(int32) :: entry, distance, best_entry, runner_up_entry
      real(real32) :: candidate_similarity, candidate_lexical, candidate_hybrid, candidate_rank
      real(real32) :: best_similarity, best_lexical, best_hybrid, best_rank
      real(real32) :: runner_up_similarity, runner_up_lexical, runner_up_hybrid, runner_up_rank
      logical :: hybrid_enabled

      answer = dialogue_answer_t()
      answer%candidates_examined = index%count
      hybrid_enabled = index%config%lexical_weight > 0.0_real32
      best_entry = 0_int32
      runner_up_entry = 0_int32
      best_rank = -huge(0.0_real32)
      runner_up_rank = -huge(0.0_real32)
      best_similarity = -1.0_real32
      best_lexical = 0.0_real32
      best_hybrid = 0.0_real32
      runner_up_similarity = -1.0_real32
      runner_up_lexical = 0.0_real32
      runner_up_hybrid = 0.0_real32

      do entry = 1, index%count
         candidate_hv%bits = index%prompt_bits(:, entry)
         distance = hv_hamming(query_hv, candidate_hv)
         candidate_similarity = hv_similarity_from_distance(distance)
         call dialogue_lexical_similarity(canonical_prompt, index%canonical_prompts(entry), candidate_lexical)
         candidate_hybrid = dialogue_hybrid_score(index%config, candidate_similarity, candidate_lexical)
         if (hybrid_enabled) then
            candidate_rank = candidate_hybrid
         else
            candidate_rank = candidate_similarity
         end if
         ! Strict comparison intentionally retains input order as the stable
         ! deterministic tie-breaker, matching the legacy exact scan.
         if (candidate_rank > best_rank) then
            runner_up_entry = best_entry
            runner_up_rank = best_rank
            runner_up_similarity = best_similarity
            runner_up_lexical = best_lexical
            runner_up_hybrid = best_hybrid
            best_entry = entry
            best_rank = candidate_rank
            best_similarity = candidate_similarity
            best_lexical = candidate_lexical
            best_hybrid = candidate_hybrid
         else if (candidate_rank > runner_up_rank) then
            runner_up_entry = entry
            runner_up_rank = candidate_rank
            runner_up_similarity = candidate_similarity
            runner_up_lexical = candidate_lexical
            runner_up_hybrid = candidate_hybrid
         end if
      end do
      if (best_entry == 0_int32) return
      answer%entry = best_entry
      answer%similarity = best_similarity
      answer%lexical_similarity = best_lexical
      answer%hybrid_score = best_hybrid
      if (runner_up_entry > 0_int32) then
         answer%runner_up_similarity = runner_up_similarity
         answer%runner_up_lexical_similarity = runner_up_lexical
         answer%runner_up_hybrid_score = runner_up_hybrid
         answer%margin = best_similarity - runner_up_similarity
         answer%hybrid_margin = best_hybrid - runner_up_hybrid
      else
         answer%runner_up_similarity = -1.0_real32
         answer%runner_up_lexical_similarity = 0.0_real32
         answer%runner_up_hybrid_score = 0.0_real32
         answer%margin = 2.0_real32
         answer%hybrid_margin = 1.0_real32
      end if
      if (hybrid_enabled) then
         ! In hybrid mode the disclosed lexical evidence is intentionally part
         ! of grounding. The legacy semantic thresholds are not silently mixed
         ! into that different policy; callers set the explicit hybrid gates.
         ! ``margin`` is the stable public active-ranking margin; a hybrid
         ! winner can legitimately have lower raw semantic similarity than its
         ! hybrid runner-up, so exposing that semantic difference as a grounding
         ! margin would be negative and misleading.
         answer%margin = answer%hybrid_margin
         if (answer%hybrid_score < index%config%minimum_hybrid_score .or. &
             answer%hybrid_margin < index%config%minimum_hybrid_margin) return
      else
         if (answer%similarity < index%config%minimum_similarity .or. &
             answer%margin < index%config%minimum_margin) return
      end if
      answer%found = .true.
      answer%matched_prompt = index%prompts(best_entry)
      answer%response = index%responses(best_entry)
   end subroutine select_dialogue_answer

   subroutine dialogue_session_remember(session, prompt_hv, status)
      type(dialogue_session_t), intent(inout) :: session
      type(hv_t), intent(in) :: prompt_hv
      integer(int32), intent(out) :: status
      integer(int32) :: slot

      status = DIALOGUE_OK
      if (.not. dialogue_session_is_initialized(session)) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      if (session%capacity == 0_int32) return
      slot = session%next_slot
      session%prompt_bits(:, slot) = prompt_hv%bits
      if (session%count < session%capacity) session%count = session%count + 1_int32
      session%next_slot = modulo(slot, session%capacity) + 1_int32
   end subroutine dialogue_session_remember

   subroutine dialogue_ask(index, model, session, prompt, answer, status)
      type(dialogue_index_t), intent(in) :: index
      type(hdc_pretrain_model_t), intent(in) :: model
      type(dialogue_session_t), intent(inout) :: session
      character(len=*), intent(in) :: prompt
      type(dialogue_answer_t), intent(out) :: answer
      integer(int32), intent(out) :: status
      type(hv_t) :: prompt_hv, query_hv
      character(len=DIALOGUE_MAX_PROMPT_CHARS) :: canonical_prompt
      integer(int32) :: known_tokens, ignored_tokens

      answer = dialogue_answer_t()
      status = DIALOGUE_OK
      if (.not. dialogue_index_is_initialized(index) .or. .not. dialogue_session_is_initialized(session)) then
         status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      call dialogue_encode_prompt(index, model, prompt, prompt_hv, known_tokens, ignored_tokens, status, canonical_prompt)
      if (status /= DIALOGUE_OK) return
      call compose_dialogue_query(index, session, prompt_hv, query_hv, status)
      if (status /= DIALOGUE_OK) return
      call select_dialogue_answer(index, query_hv, canonical_prompt, answer)
      answer%query_known_tokens = known_tokens
      answer%query_ignored_tokens = ignored_tokens
      call dialogue_session_remember(session, prompt_hv, status)
   end subroutine dialogue_ask

end module dialogue_retrieval
