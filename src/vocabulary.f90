! SPDX-License-Identifier: MIT
!==============================================================================
!  vocabulary.f90
!  Fixed-capacity deterministic vocabulary for two-pass streaming pretraining.
!==============================================================================
module vocabulary
   use, intrinsic :: iso_fortran_env, only: int32, int64
   use text_tokenizer, only: TOKEN_LEN
   implicit none
   private

   integer(int32), parameter, public :: VOCAB_OK = 0_int32
   integer(int32), parameter, public :: VOCAB_ERR_INVALID_ARGUMENT = -400_int32
   integer(int32), parameter, public :: VOCAB_ERR_ALLOCATION = -401_int32
   integer(int32), parameter, public :: VOCAB_ERR_FULL = -402_int32
   integer(int32), parameter, public :: VOCAB_ERR_NOT_FOUND = -403_int32
   integer(int32), parameter, public :: VOCAB_ERR_FREQUENCY_OVERFLOW = -404_int32
   integer(int32), parameter, public :: VOCAB_ERR_DUPLICATE = -405_int32

   integer(int64), parameter :: VOCAB_SEED_TAG = int(z'A0761D6478BD642F', int64)

   type, public :: vocabulary_t
      integer(int32) :: capacity = 0_int32
      integer(int32) :: count = 0_int32
      integer(int32) :: active_count = 0_int32
      integer(int32) :: hash_size = 0_int32
      integer(int64) :: min_frequency = 1_int64
      character(len=TOKEN_LEN), allocatable :: tokens(:)
      integer(int64), allocatable :: frequencies(:)
      logical, allocatable :: active(:)
      logical, allocatable :: hash_used(:)
      integer(int64), allocatable :: hash_keys(:)
      integer(int32), allocatable :: hash_slots(:)
   end type vocabulary_t

   public :: vocab_init, vocab_destroy, vocab_is_initialized
   public :: vocab_observe, vocab_finalize, vocab_lookup, vocab_rebuild_index
   public :: vocab_token_fingerprint, vocab_seed_for_slot, vocab_footprint_bytes

contains

   pure function vocab_is_initialized(vocab) result(initialized)
      type(vocabulary_t), intent(in) :: vocab
      logical :: initialized

      initialized = vocab%capacity > 0_int32 .and. allocated(vocab%tokens) .and. &
                    allocated(vocab%frequencies) .and. allocated(vocab%active) .and. &
                    allocated(vocab%hash_used) .and. allocated(vocab%hash_keys) .and. &
                    allocated(vocab%hash_slots)
   end function vocab_is_initialized

   subroutine vocab_destroy(vocab)
      type(vocabulary_t), intent(inout) :: vocab

      if (allocated(vocab%tokens)) deallocate(vocab%tokens)
      if (allocated(vocab%frequencies)) deallocate(vocab%frequencies)
      if (allocated(vocab%active)) deallocate(vocab%active)
      if (allocated(vocab%hash_used)) deallocate(vocab%hash_used)
      if (allocated(vocab%hash_keys)) deallocate(vocab%hash_keys)
      if (allocated(vocab%hash_slots)) deallocate(vocab%hash_slots)
      vocab%capacity = 0_int32
      vocab%count = 0_int32
      vocab%active_count = 0_int32
      vocab%hash_size = 0_int32
      vocab%min_frequency = 1_int64
   end subroutine vocab_destroy

   subroutine vocab_init(vocab, max_terms, min_frequency, status)
      type(vocabulary_t), intent(inout) :: vocab
      integer(int32), intent(in) :: max_terms
      integer(int64), intent(in) :: min_frequency
      integer(int32), intent(out) :: status
      integer(int32) :: allocation_status
      integer(int64) :: requested_hash_size

      call vocab_destroy(vocab)
      status = VOCAB_OK
      if (max_terms <= 0_int32 .or. min_frequency <= 0_int64) then
         status = VOCAB_ERR_INVALID_ARGUMENT
         return
      end if
      requested_hash_size = 2_int64 * int(max_terms, int64) + 1_int64
      if (requested_hash_size > int(huge(0_int32), int64)) then
         status = VOCAB_ERR_INVALID_ARGUMENT
         return
      end if

      vocab%capacity = max_terms
      vocab%hash_size = int(requested_hash_size, int32)
      vocab%min_frequency = min_frequency

      allocate(vocab%tokens(max_terms), stat=allocation_status)
      if (allocation_status /= 0) then
         status = VOCAB_ERR_ALLOCATION
         call vocab_destroy(vocab)
         return
      end if
      allocate(vocab%frequencies(max_terms), stat=allocation_status)
      if (allocation_status /= 0) then
         status = VOCAB_ERR_ALLOCATION
         call vocab_destroy(vocab)
         return
      end if
      allocate(vocab%active(max_terms), stat=allocation_status)
      if (allocation_status /= 0) then
         status = VOCAB_ERR_ALLOCATION
         call vocab_destroy(vocab)
         return
      end if
      allocate(vocab%hash_used(vocab%hash_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = VOCAB_ERR_ALLOCATION
         call vocab_destroy(vocab)
         return
      end if
      allocate(vocab%hash_keys(vocab%hash_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = VOCAB_ERR_ALLOCATION
         call vocab_destroy(vocab)
         return
      end if
      allocate(vocab%hash_slots(vocab%hash_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = VOCAB_ERR_ALLOCATION
         call vocab_destroy(vocab)
         return
      end if

      vocab%tokens = ''
      vocab%frequencies = 0_int64
      vocab%active = .false.
      vocab%hash_used = .false.
      vocab%hash_keys = 0_int64
      vocab%hash_slots = 0_int32
   end subroutine vocab_init

   ! A bit-only string mixer.  It is deterministic, avoids signed overflow, and
   ! never treats a hash equality as token equality: lookup compares full text.
   pure function vocab_token_fingerprint(token) result(fingerprint)
      character(len=*), intent(in) :: token
      integer(int64) :: fingerprint
      integer(int32) :: position, code

      fingerprint = int(z'CBF29CE484222325', int64)
      do position = 1, len_trim(token)
         code = int(iachar(token(position:position)), int32)
         fingerprint = ieor(fingerprint, int(code, int64))
         fingerprint = ieor(ishftc(fingerprint, 11), ishftc(fingerprint, -7))
         fingerprint = ieor(fingerprint, ishftc(int(position, int64), modulo(position, 63_int32)))
      end do
   end function vocab_token_fingerprint

   pure function vocab_hash_start(fingerprint, hash_size) result(index)
      integer(int64), intent(in) :: fingerprint
      integer(int32), intent(in) :: hash_size
      integer(int32) :: index
      integer(int64) :: nonnegative, mixed

      mixed = ieor(fingerprint, ishft(fingerprint, -29))
      mixed = ieor(mixed, ishftc(mixed, 17))
      nonnegative = iand(mixed, huge(0_int64))
      index = int(modulo(nonnegative, int(hash_size, int64)), int32) + 1_int32
   end function vocab_hash_start

   subroutine vocab_hash_lookup(vocab, token, fingerprint, slot, found, table_index)
      type(vocabulary_t), intent(in) :: vocab
      character(len=*), intent(in) :: token
      integer(int64), intent(in) :: fingerprint
      integer(int32), intent(out) :: slot, table_index
      logical, intent(out) :: found
      integer(int32) :: probe, step, candidate_slot

      slot = 0_int32
      table_index = 0_int32
      found = .false.
      if (.not. vocab_is_initialized(vocab)) return

      probe = vocab_hash_start(fingerprint, vocab%hash_size)
      do step = 1, vocab%hash_size
         if (.not. vocab%hash_used(probe)) then
            table_index = probe
            return
         end if
         if (vocab%hash_keys(probe) == fingerprint) then
            candidate_slot = vocab%hash_slots(probe)
            if (trim(vocab%tokens(candidate_slot)) == trim(token)) then
               slot = candidate_slot
               table_index = probe
               found = .true.
               return
            end if
         end if
         probe = modulo(probe, vocab%hash_size) + 1_int32
      end do
   end subroutine vocab_hash_lookup

   subroutine vocab_observe(vocab, token, status)
      type(vocabulary_t), intent(inout) :: vocab
      character(len=*), intent(in) :: token
      integer(int32), intent(out) :: status
      integer(int64) :: fingerprint
      integer(int32) :: slot, table_index
      logical :: found

      status = VOCAB_OK
      if (.not. vocab_is_initialized(vocab)) then
         status = VOCAB_ERR_INVALID_ARGUMENT
         return
      end if
      if (len_trim(token) == 0 .or. len_trim(token) > TOKEN_LEN) then
         status = VOCAB_ERR_INVALID_ARGUMENT
         return
      end if

      fingerprint = vocab_token_fingerprint(token)
      call vocab_hash_lookup(vocab, token, fingerprint, slot, found, table_index)
      if (found) then
         if (vocab%frequencies(slot) == huge(0_int64)) then
            status = VOCAB_ERR_FREQUENCY_OVERFLOW
            return
         end if
         vocab%frequencies(slot) = vocab%frequencies(slot) + 1_int64
         return
      end if
      if (vocab%count >= vocab%capacity .or. table_index == 0_int32) then
         status = VOCAB_ERR_FULL
         return
      end if

      slot = vocab%count + 1_int32
      vocab%tokens(slot) = adjustl(token)
      vocab%frequencies(slot) = 1_int64
      vocab%active(slot) = .false.
      vocab%hash_used(table_index) = .true.
      vocab%hash_keys(table_index) = fingerprint
      vocab%hash_slots(table_index) = slot
      vocab%count = slot
   end subroutine vocab_observe

   subroutine vocab_finalize(vocab, status)
      type(vocabulary_t), intent(inout) :: vocab
      integer(int32), intent(out) :: status
      integer(int32) :: slot

      status = VOCAB_OK
      if (.not. vocab_is_initialized(vocab)) then
         status = VOCAB_ERR_INVALID_ARGUMENT
         return
      end if
      vocab%active = .false.
      vocab%active_count = 0_int32
      do slot = 1, vocab%count
         if (vocab%frequencies(slot) >= vocab%min_frequency) then
            vocab%active(slot) = .true.
            vocab%active_count = vocab%active_count + 1_int32
         end if
      end do
   end subroutine vocab_finalize

   subroutine vocab_lookup(vocab, token, slot, found, status, active_only)
      type(vocabulary_t), intent(in) :: vocab
      character(len=*), intent(in) :: token
      integer(int32), intent(out) :: slot
      logical, intent(out) :: found
      integer(int32), intent(out) :: status
      logical, intent(in), optional :: active_only
      integer(int64) :: fingerprint
      integer(int32) :: table_index
      logical :: require_active

      slot = 0_int32
      found = .false.
      status = VOCAB_OK
      if (.not. vocab_is_initialized(vocab)) then
         status = VOCAB_ERR_INVALID_ARGUMENT
         return
      end if
      if (len_trim(token) == 0 .or. len_trim(token) > TOKEN_LEN) then
         status = VOCAB_ERR_NOT_FOUND
         return
      end if

      require_active = .false.
      if (present(active_only)) require_active = active_only
      fingerprint = vocab_token_fingerprint(token)
      call vocab_hash_lookup(vocab, token, fingerprint, slot, found, table_index)
      if (.not. found) then
         status = VOCAB_ERR_NOT_FOUND
         return
      end if
      if (require_active .and. .not. vocab%active(slot)) then
         found = .false.
         status = VOCAB_ERR_NOT_FOUND
      end if
   end subroutine vocab_lookup

   ! Reconstructs the transient hash index after a checkpoint restore.  Stored
   ! token strings and counts are preserved exactly; active_count is recomputed.
   subroutine vocab_rebuild_index(vocab, status)
      type(vocabulary_t), intent(inout) :: vocab
      integer(int32), intent(out) :: status
      integer(int64) :: fingerprint
      integer(int32) :: slot, found_slot, table_index
      logical :: found

      status = VOCAB_OK
      if (.not. vocab_is_initialized(vocab)) then
         status = VOCAB_ERR_INVALID_ARGUMENT
         return
      end if
      if (vocab%count < 0_int32 .or. vocab%count > vocab%capacity) then
         status = VOCAB_ERR_INVALID_ARGUMENT
         return
      end if

      vocab%hash_used = .false.
      vocab%hash_keys = 0_int64
      vocab%hash_slots = 0_int32
      vocab%active_count = 0_int32
      do slot = 1, vocab%count
         if (len_trim(vocab%tokens(slot)) == 0) then
            status = VOCAB_ERR_INVALID_ARGUMENT
            return
         end if
         fingerprint = vocab_token_fingerprint(vocab%tokens(slot))
         call vocab_hash_lookup(vocab, vocab%tokens(slot), fingerprint, found_slot, found, table_index)
         if (found) then
            status = VOCAB_ERR_DUPLICATE
            return
         end if
         if (table_index == 0_int32) then
            status = VOCAB_ERR_FULL
            return
         end if
         vocab%hash_used(table_index) = .true.
         vocab%hash_keys(table_index) = fingerprint
         vocab%hash_slots(table_index) = slot
         if (vocab%active(slot)) vocab%active_count = vocab%active_count + 1_int32
      end do
   end subroutine vocab_rebuild_index

   pure function vocab_seed_for_slot(vocab, slot) result(seed)
      type(vocabulary_t), intent(in) :: vocab
      integer(int32), intent(in) :: slot
      integer(int64) :: seed

      seed = 0_int64
      if (slot < 1_int32 .or. slot > vocab%count) return
      seed = ieor(vocab_token_fingerprint(vocab%tokens(slot)), VOCAB_SEED_TAG)
      seed = ieor(seed, ishftc(int(slot, int64), 23))
   end function vocab_seed_for_slot

   pure function vocab_footprint_bytes(vocab) result(total_bytes)
      type(vocabulary_t), intent(in) :: vocab
      integer(int64) :: total_bytes

      total_bytes = 0_int64
      if (allocated(vocab%tokens)) then
         total_bytes = total_bytes + size(vocab%tokens, kind=int64) * &
                       int(storage_size(vocab%tokens(1)) / 8, int64)
      end if
      if (allocated(vocab%frequencies)) then
         total_bytes = total_bytes + size(vocab%frequencies, kind=int64) * &
                       int(storage_size(vocab%frequencies(1)) / 8, int64)
      end if
      if (allocated(vocab%active)) then
         total_bytes = total_bytes + size(vocab%active, kind=int64) * &
                       int(storage_size(vocab%active(1)) / 8, int64)
      end if
      if (allocated(vocab%hash_used)) then
         total_bytes = total_bytes + size(vocab%hash_used, kind=int64) * &
                       int(storage_size(vocab%hash_used(1)) / 8, int64)
      end if
      if (allocated(vocab%hash_keys)) then
         total_bytes = total_bytes + size(vocab%hash_keys, kind=int64) * &
                       int(storage_size(vocab%hash_keys(1)) / 8, int64)
      end if
      if (allocated(vocab%hash_slots)) then
         total_bytes = total_bytes + size(vocab%hash_slots, kind=int64) * &
                       int(storage_size(vocab%hash_slots(1)) / 8, int64)
      end if
   end function vocab_footprint_bytes

end module vocabulary
