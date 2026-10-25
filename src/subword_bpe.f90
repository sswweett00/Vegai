! SPDX-License-Identifier: MIT
!==============================================================================
!  subword_bpe.f90
!  Native byte-level BPE codec with lossless UTF-8 byte fallback.
!
!  The base vocabulary is the 256 possible octets.  A merge token is a pair of
!  already-defined token IDs, so any byte sequence remains representable even
!  if no merge table is available.  This is the crucial safety property for
!  Turkish UTF-8: an unseen suffix, apostrophe form, or spelling never becomes
!  out-of-vocabulary merely because a word vocabulary omitted it.
!
!  The module is a deterministic codec.  Corpus-scale merge learning is kept
!  outside the runtime kernel because its corpus/provenance policy must be
!  explicit; an immutable merge table can be loaded/attested by a future
!  trainer without changing this encoding contract.
!==============================================================================
module subword_bpe
   use, intrinsic :: iso_fortran_env, only: int32, int64
   implicit none
   private

   integer(int32), parameter, public :: BPE_OK = 0_int32
   integer(int32), parameter, public :: BPE_ERR_INVALID_ARGUMENT = -800_int32
   integer(int32), parameter, public :: BPE_ERR_ALLOCATION = -801_int32
   integer(int32), parameter, public :: BPE_ERR_CAPACITY = -802_int32
   integer(int32), parameter, public :: BPE_ERR_UNKNOWN_TOKEN = -803_int32
   integer(int32), parameter, public :: BPE_ERR_OUTPUT = -804_int32
   integer(int32), parameter, public :: BPE_ERR_DUPLICATE = -805_int32
   integer(int32), parameter, public :: BPE_ERR_IO = -806_int32
   integer(int32), parameter, public :: BPE_ERR_FORMAT = -807_int32

   character(len=8), parameter :: BPE_FILE_MAGIC = 'VGBPE001'
   integer(int32), parameter :: BPE_FILE_VERSION = 1_int32

   integer(int32), parameter, public :: BPE_BASE_VOCABULARY = 256_int32
   integer(int32), parameter, public :: BPE_MAX_INPUT_BYTES = 8192_int32

   type, public :: bpe_model_t
      integer(int32) :: maximum_merges = 0_int32
      integer(int32) :: merge_count = 0_int32
      integer(int32) :: token_count = BPE_BASE_VOCABULARY
      integer(int32) :: hash_size = 0_int32
      integer(int32), allocatable :: left(:)
      integer(int32), allocatable :: right(:)
      integer(int32), allocatable :: result(:)
      integer(int32), allocatable :: rank(:)
      logical, allocatable :: hash_used(:)
      integer(int64), allocatable :: hash_keys(:)
      integer(int32), allocatable :: hash_slots(:)
   end type bpe_model_t

   public :: bpe_init, bpe_destroy, bpe_is_initialized, bpe_add_merge
   public :: bpe_encode, bpe_decode, bpe_model_footprint_bytes
   public :: bpe_base_token_for_byte, bpe_save, bpe_load

contains

   pure logical function bpe_is_initialized(model)
      type(bpe_model_t), intent(in) :: model

      bpe_is_initialized = model%maximum_merges > 0_int32 .and. model%hash_size > 0_int32 .and. &
                           allocated(model%left) .and. allocated(model%right) .and. &
                           allocated(model%result) .and. allocated(model%rank) .and. &
                           allocated(model%hash_used) .and. allocated(model%hash_keys) .and. &
                           allocated(model%hash_slots)
   end function bpe_is_initialized

   subroutine bpe_destroy(model)
      type(bpe_model_t), intent(inout) :: model

      if (allocated(model%left)) deallocate(model%left)
      if (allocated(model%right)) deallocate(model%right)
      if (allocated(model%result)) deallocate(model%result)
      if (allocated(model%rank)) deallocate(model%rank)
      if (allocated(model%hash_used)) deallocate(model%hash_used)
      if (allocated(model%hash_keys)) deallocate(model%hash_keys)
      if (allocated(model%hash_slots)) deallocate(model%hash_slots)
      model = bpe_model_t()
   end subroutine bpe_destroy

   subroutine bpe_init(model, maximum_merges, status)
      type(bpe_model_t), intent(inout) :: model
      integer(int32), intent(in) :: maximum_merges
      integer(int32), intent(out) :: status
      integer(int64) :: requested_hash_size
      integer :: allocation_status

      call bpe_destroy(model)
      status = BPE_OK
      if (maximum_merges <= 0_int32) then
         status = BPE_ERR_INVALID_ARGUMENT
         return
      end if
      requested_hash_size = 2_int64 * int(maximum_merges, int64) + 1_int64
      if (requested_hash_size > int(huge(0_int32), int64)) then
         status = BPE_ERR_INVALID_ARGUMENT
         return
      end if
      model%maximum_merges = maximum_merges
      model%hash_size = int(requested_hash_size, int32)
      allocate(model%left(maximum_merges), model%right(maximum_merges), model%result(maximum_merges), &
               model%rank(maximum_merges), stat=allocation_status)
      if (allocation_status /= 0) then
         status = BPE_ERR_ALLOCATION
         call bpe_destroy(model)
         return
      end if
      allocate(model%hash_used(model%hash_size), model%hash_keys(model%hash_size), &
               model%hash_slots(model%hash_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = BPE_ERR_ALLOCATION
         call bpe_destroy(model)
         return
      end if
      model%left = 0_int32
      model%right = 0_int32
      model%result = 0_int32
      model%rank = 0_int32
      model%hash_used = .false.
      model%hash_keys = 0_int64
      model%hash_slots = 0_int32
   end subroutine bpe_init

   pure integer(int32) function bpe_base_token_for_byte(byte_value) result(token)
      integer(int32), intent(in) :: byte_value

      if (byte_value < 0_int32 .or. byte_value > 255_int32) then
         token = 0_int32
      else
         token = byte_value + 1_int32
      end if
   end function bpe_base_token_for_byte

   pure integer(int64) function pair_key(left, right) result(key)
      integer(int32), intent(in) :: left, right

      key = ior(shiftl(int(left, int64), 32), iand(int(right, int64), int(z'00000000FFFFFFFF', int64)))
   end function pair_key

   pure integer(int32) function hash_start(key, hash_size) result(index)
      integer(int64), intent(in) :: key
      integer(int32), intent(in) :: hash_size
      integer(int64) :: mixed, nonnegative

      mixed = ieor(key, shiftr(key, 29))
      mixed = ieor(mixed, ishftc(mixed, 17))
      mixed = ieor(mixed, shiftr(mixed, 31))
      nonnegative = iand(mixed, huge(0_int64))
      index = int(modulo(nonnegative, int(hash_size, int64)), int32) + 1_int32
   end function hash_start

   subroutine find_merge_slot(model, left, right, merge_slot, hash_index, found)
      type(bpe_model_t), intent(in) :: model
      integer(int32), intent(in) :: left, right
      integer(int32), intent(out) :: merge_slot, hash_index
      logical, intent(out) :: found
      integer(int32) :: probe, step, candidate
      integer(int64) :: key

      merge_slot = 0_int32
      hash_index = 0_int32
      found = .false.
      if (.not. bpe_is_initialized(model)) return
      key = pair_key(left, right)
      probe = hash_start(key, model%hash_size)
      do step = 1_int32, model%hash_size
         if (.not. model%hash_used(probe)) then
            hash_index = probe
            return
         end if
         if (model%hash_keys(probe) == key) then
            candidate = model%hash_slots(probe)
            if (candidate >= 1_int32 .and. candidate <= model%merge_count) then
               if (model%left(candidate) == left .and. model%right(candidate) == right) then
                  merge_slot = candidate
                  hash_index = probe
                  found = .true.
                  return
               end if
            end if
         end if
         probe = modulo(probe, model%hash_size) + 1_int32
      end do
   end subroutine find_merge_slot

   subroutine bpe_add_merge(model, left, right, merged_token, status)
      type(bpe_model_t), intent(inout) :: model
      integer(int32), intent(in) :: left, right
      integer(int32), intent(out) :: merged_token
      integer(int32), intent(out) :: status
      integer(int32) :: slot, hash_index
      logical :: found

      merged_token = 0_int32
      status = BPE_OK
      if (.not. bpe_is_initialized(model) .or. left <= 0_int32 .or. right <= 0_int32 .or. &
          left > model%token_count .or. right > model%token_count) then
         status = BPE_ERR_INVALID_ARGUMENT
         return
      end if
      call find_merge_slot(model, left, right, slot, hash_index, found)
      if (found) then
         status = BPE_ERR_DUPLICATE
         return
      end if
      if (model%merge_count >= model%maximum_merges .or. hash_index == 0_int32) then
         status = BPE_ERR_CAPACITY
         return
      end if
      slot = model%merge_count + 1_int32
      model%left(slot) = left
      model%right(slot) = right
      model%token_count = model%token_count + 1_int32
      model%result(slot) = model%token_count
      model%rank(slot) = slot
      model%hash_used(hash_index) = .true.
      model%hash_keys(hash_index) = pair_key(left, right)
      model%hash_slots(hash_index) = slot
      model%merge_count = slot
      merged_token = model%result(slot)
   end subroutine bpe_add_merge

   subroutine lookup_merge(model, left, right, merged_token, merge_rank, found)
      type(bpe_model_t), intent(in) :: model
      integer(int32), intent(in) :: left, right
      integer(int32), intent(out) :: merged_token, merge_rank
      logical, intent(out) :: found
      integer(int32) :: slot, hash_index

      merged_token = 0_int32
      merge_rank = huge(0_int32)
      call find_merge_slot(model, left, right, slot, hash_index, found)
      if (found) then
         merged_token = model%result(slot)
         merge_rank = model%rank(slot)
      end if
   end subroutine lookup_merge

   ! Min-heap entries are intentionally allowed to become stale after a
   ! neighboring merge. Validation at pop time rechecks the current linked
   ! pair. That removes O(n) compaction and repeated global pair scans while
   ! retaining the standard lowest-rank, leftmost-tie BPE merge semantics.
   subroutine heap_push(positions, ranks, heap_count, position, rank)
      integer(int32), intent(inout) :: positions(:), ranks(:)
      integer(int32), intent(inout) :: heap_count
      integer(int32), intent(in) :: position, rank
      integer(int32) :: child, parent, saved_position, saved_rank

      heap_count = heap_count + 1_int32
      child = heap_count
      positions(child) = position
      ranks(child) = rank
      do while (child > 1_int32)
         parent = child / 2_int32
         if (ranks(parent) < ranks(child) .or. &
             (ranks(parent) == ranks(child) .and. positions(parent) <= positions(child))) exit
         saved_position = positions(parent)
         saved_rank = ranks(parent)
         positions(parent) = positions(child)
         ranks(parent) = ranks(child)
         positions(child) = saved_position
         ranks(child) = saved_rank
         child = parent
      end do
   end subroutine heap_push

   subroutine heap_pop(positions, ranks, heap_count, position, rank)
      integer(int32), intent(inout) :: positions(:), ranks(:)
      integer(int32), intent(inout) :: heap_count
      integer(int32), intent(out) :: position, rank
      integer(int32) :: parent, child, selected, saved_position, saved_rank

      position = 0_int32
      rank = huge(0_int32)
      if (heap_count <= 0_int32) return
      position = positions(1)
      rank = ranks(1)
      positions(1) = positions(heap_count)
      ranks(1) = ranks(heap_count)
      heap_count = heap_count - 1_int32
      parent = 1_int32
      do while (2_int32 * parent <= heap_count)
         child = 2_int32 * parent
         selected = child
         if (child + 1_int32 <= heap_count) then
            if (ranks(child + 1_int32) < ranks(child) .or. &
                (ranks(child + 1_int32) == ranks(child) .and. positions(child + 1_int32) < positions(child))) then
               selected = child + 1_int32
            end if
         end if
         if (ranks(parent) < ranks(selected) .or. &
             (ranks(parent) == ranks(selected) .and. positions(parent) <= positions(selected))) exit
         saved_position = positions(parent)
         saved_rank = ranks(parent)
         positions(parent) = positions(selected)
         ranks(parent) = ranks(selected)
         positions(selected) = saved_position
         ranks(selected) = saved_rank
         parent = selected
      end do
   end subroutine heap_pop

   subroutine push_pair_if_merge(model, symbols, left_position, right_position, heap_positions, heap_ranks, heap_count)
      type(bpe_model_t), intent(in) :: model
      integer(int32), intent(in) :: symbols(:)
      integer(int32), intent(in) :: left_position, right_position
      integer(int32), intent(inout) :: heap_positions(:), heap_ranks(:)
      integer(int32), intent(inout) :: heap_count
      integer(int32) :: merged_token, merge_rank
      logical :: found

      if (left_position <= 0_int32 .or. right_position <= 0_int32) return
      call lookup_merge(model, symbols(left_position), symbols(right_position), merged_token, merge_rank, found)
      if (found) call heap_push(heap_positions, heap_ranks, heap_count, left_position, merge_rank)
   end subroutine push_pair_if_merge

   subroutine bpe_encode(model, input, input_bytes, tokens, token_count, status)
      type(bpe_model_t), intent(in) :: model
      character(len=*), intent(in) :: input
      integer(int32), intent(in) :: input_bytes
      integer(int32), intent(out) :: tokens(:)
      integer(int32), intent(out) :: token_count
      integer(int32), intent(out) :: status
      integer(int32) :: symbols(BPE_MAX_INPUT_BYTES), previous(BPE_MAX_INPUT_BYTES), following(BPE_MAX_INPUT_BYTES)
      logical :: active(BPE_MAX_INPUT_BYTES)
      integer(int32) :: heap_positions(3 * BPE_MAX_INPUT_BYTES), heap_ranks(3 * BPE_MAX_INPUT_BYTES)
      integer(int32) :: heap_count, count, position, right_position, next_position
      integer(int32) :: popped_position, popped_rank, candidate_rank, candidate_token, output_position
      logical :: found

      tokens = 0_int32
      token_count = 0_int32
      status = BPE_OK
      if (.not. bpe_is_initialized(model) .or. input_bytes < 0_int32 .or. input_bytes > len(input) .or. &
          input_bytes > BPE_MAX_INPUT_BYTES) then
         status = BPE_ERR_INVALID_ARGUMENT
         return
      end if
      if (input_bytes == 0_int32) return
      count = input_bytes
      active = .false.
      previous = 0_int32
      following = 0_int32
      heap_count = 0_int32
      do position = 1_int32, count
         symbols(position) = bpe_base_token_for_byte(int(iachar(input(position:position)), int32))
         active(position) = .true.
         if (position > 1_int32) previous(position) = position - 1_int32
         if (position < count) following(position) = position + 1_int32
      end do
      do position = 1_int32, count - 1_int32
         call push_pair_if_merge(model, symbols, position, position + 1_int32, heap_positions, heap_ranks, heap_count)
      end do
      do while (heap_count > 0_int32)
         call heap_pop(heap_positions, heap_ranks, heap_count, popped_position, popped_rank)
         if (popped_position <= 0_int32 .or. .not. active(popped_position)) cycle
         right_position = following(popped_position)
         if (right_position <= 0_int32 .or. .not. active(right_position)) cycle
         call lookup_merge(model, symbols(popped_position), symbols(right_position), candidate_token, candidate_rank, found)
         if (.not. found .or. candidate_rank /= popped_rank) cycle
         symbols(popped_position) = candidate_token
         active(right_position) = .false.
         next_position = following(right_position)
         following(popped_position) = next_position
         if (next_position > 0_int32) previous(next_position) = popped_position
         previous(right_position) = 0_int32
         following(right_position) = 0_int32
         count = count - 1_int32
         if (previous(popped_position) > 0_int32) then
            call push_pair_if_merge(model, symbols, previous(popped_position), popped_position, &
                                    heap_positions, heap_ranks, heap_count)
         end if
         if (next_position > 0_int32) then
            call push_pair_if_merge(model, symbols, popped_position, next_position, heap_positions, heap_ranks, heap_count)
         end if
      end do
      if (count > size(tokens, kind=int32)) then
         status = BPE_ERR_OUTPUT
         return
      end if
      output_position = 1_int32
      position = 1_int32
      do while (position > 0_int32)
         tokens(output_position) = symbols(position)
         output_position = output_position + 1_int32
         position = following(position)
      end do
      token_count = count
   end subroutine bpe_encode

   recursive subroutine append_decoded_token(model, token, output, output_bytes, status, depth)
      type(bpe_model_t), intent(in) :: model
      integer(int32), intent(in) :: token
      character(len=*), intent(inout) :: output
      integer(int32), intent(inout) :: output_bytes
      integer(int32), intent(inout) :: status
      integer(int32), intent(in) :: depth
      integer(int32) :: merge_slot

      if (status /= BPE_OK) return
      if (token >= 1_int32 .and. token <= BPE_BASE_VOCABULARY) then
         if (output_bytes >= len(output)) then
            status = BPE_ERR_OUTPUT
            return
         end if
         output_bytes = output_bytes + 1_int32
         output(output_bytes:output_bytes) = achar(token - 1_int32)
         return
      end if
      merge_slot = token - BPE_BASE_VOCABULARY
      if (merge_slot < 1_int32 .or. merge_slot > model%merge_count .or. depth > model%merge_count) then
         status = BPE_ERR_UNKNOWN_TOKEN
         return
      end if
      call append_decoded_token(model, model%left(merge_slot), output, output_bytes, status, depth + 1_int32)
      call append_decoded_token(model, model%right(merge_slot), output, output_bytes, status, depth + 1_int32)
   end subroutine append_decoded_token

   subroutine bpe_decode(model, tokens, token_count, output, output_bytes, status)
      type(bpe_model_t), intent(in) :: model
      integer(int32), intent(in) :: tokens(:)
      integer(int32), intent(in) :: token_count
      character(len=*), intent(out) :: output
      integer(int32), intent(out) :: output_bytes
      integer(int32), intent(out) :: status
      integer(int32) :: position

      output = ''
      output_bytes = 0_int32
      status = BPE_OK
      if (.not. bpe_is_initialized(model) .or. token_count < 0_int32 .or. token_count > size(tokens, kind=int32)) then
         status = BPE_ERR_INVALID_ARGUMENT
         return
      end if
      do position = 1_int32, token_count
         call append_decoded_token(model, tokens(position), output, output_bytes, status, 0_int32)
         if (status /= BPE_OK) return
      end do
   end subroutine bpe_decode

   function bpe_model_footprint_bytes(model) result(bytes)
      type(bpe_model_t), intent(in) :: model
      integer(int64) :: bytes

      bytes = 0_int64
      if (.not. bpe_is_initialized(model)) return
      bytes = bytes + 4_int64 * int(size(model%left), int64)
      bytes = bytes + 4_int64 * int(size(model%right), int64)
      bytes = bytes + 4_int64 * int(size(model%result), int64)
      bytes = bytes + 4_int64 * int(size(model%rank), int64)
      bytes = bytes + int(size(model%hash_used), int64)
      bytes = bytes + 8_int64 * int(size(model%hash_keys), int64)
      bytes = bytes + 4_int64 * int(size(model%hash_slots), int64)
   end function bpe_model_footprint_bytes


   ! The merge list is serialized rather than raw hash-table storage. On load
   ! the collision-safe index is rebuilt through bpe_add_merge, so hash-table
   ! capacity/layout changes cannot alter codec semantics. Stream-unformatted
   ! files are a same-compiler-family artifact format, not a cross-ABI claim.
   subroutine bpe_save(model, path, status)
      type(bpe_model_t), intent(in) :: model
      character(len=*), intent(in) :: path
      integer(int32), intent(out) :: status
      integer :: unit, io_status, write_status
      integer(int32) :: merge_slot

      status = BPE_OK
      if (.not. bpe_is_initialized(model) .or. len_trim(path) == 0) then
         status = BPE_ERR_INVALID_ARGUMENT
         return
      end if
      open(newunit=unit, file=trim(path), status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = BPE_ERR_IO
         return
      end if
      write(unit, iostat=io_status) BPE_FILE_MAGIC
      if (io_status == 0) write(unit, iostat=io_status) BPE_FILE_VERSION
      if (io_status == 0) write(unit, iostat=io_status) model%maximum_merges
      if (io_status == 0) write(unit, iostat=io_status) model%merge_count
      do merge_slot = 1_int32, model%merge_count
         if (io_status /= 0) exit
         write(unit, iostat=io_status) model%left(merge_slot), model%right(merge_slot)
      end do
      write_status = io_status
      close(unit, iostat=io_status)
      if (write_status /= 0 .or. io_status /= 0) status = BPE_ERR_IO
   end subroutine bpe_save

   subroutine bpe_load(model, path, status)
      type(bpe_model_t), intent(inout) :: model
      character(len=*), intent(in) :: path
      integer(int32), intent(out) :: status
      character(len=8) :: magic
      integer :: unit, io_status
      integer(int32) :: version, maximum_merges, merge_count, merge_slot
      integer(int32) :: left_token, right_token, result_token

      call bpe_destroy(model)
      status = BPE_OK
      if (len_trim(path) == 0) then
         status = BPE_ERR_INVALID_ARGUMENT
         return
      end if
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = BPE_ERR_IO
         return
      end if
      read(unit, iostat=io_status) magic
      if (io_status == 0) read(unit, iostat=io_status) version
      if (io_status == 0) read(unit, iostat=io_status) maximum_merges
      if (io_status == 0) read(unit, iostat=io_status) merge_count
      if (io_status /= 0) then
         close(unit)
         status = BPE_ERR_IO
         return
      end if
      if (magic /= BPE_FILE_MAGIC .or. version /= BPE_FILE_VERSION .or. maximum_merges <= 0_int32 .or. &
          merge_count < 0_int32 .or. merge_count > maximum_merges) then
         close(unit)
         status = BPE_ERR_FORMAT
         return
      end if
      call bpe_init(model, maximum_merges, status)
      if (status /= BPE_OK) then
         close(unit)
         return
      end if
      do merge_slot = 1_int32, merge_count
         read(unit, iostat=io_status) left_token, right_token
         if (io_status /= 0) then
            close(unit)
            call bpe_destroy(model)
            status = BPE_ERR_IO
            return
         end if
         call bpe_add_merge(model, left_token, right_token, result_token, status)
         if (status /= BPE_OK .or. result_token /= BPE_BASE_VOCABULARY + merge_slot) then
            close(unit)
            call bpe_destroy(model)
            status = BPE_ERR_FORMAT
            return
         end if
      end do
      close(unit, iostat=io_status)
      if (io_status /= 0) then
         call bpe_destroy(model)
         status = BPE_ERR_IO
      end if
   end subroutine bpe_load

end module subword_bpe
