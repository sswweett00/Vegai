! SPDX-License-Identifier: MIT
!==============================================================================
!  symbolic_memory.f90
!  Fixed-capacity associative cleanup memory for packed hypervectors.
!
!  The hypervectors are stored as vector_bits(HV_WORDS, capacity), rather than
!  an array of derived types.  A symbol's 256 words are contiguous in a column,
!  while each word plane is also contiguous across symbols: a genuine packed
!  Structure-of-Arrays-friendly layout with no per-symbol heap allocation.
!==============================================================================
module symbolic_memory
   use, intrinsic :: iso_fortran_env, only: int32, int64, real32
   use hyper_types
   implicit none
   private

   integer(int32), parameter, public :: LABEL_LEN = 48_int32
   integer(int32), parameter, public :: LSH_BUCKETS = 4096_int32
   integer(int32), parameter, public :: BUCKET_DEPTH = 8_int32
   integer(int32), parameter, public :: MEM_UNKNOWN_ID = -huge(0_int32)

   integer(int32), parameter, public :: MEM_OK = 0_int32
   integer(int32), parameter, public :: MEM_ERR_INVALID_ARGUMENT = -100_int32
   integer(int32), parameter, public :: MEM_ERR_ALLOCATION = -101_int32
   integer(int32), parameter, public :: MEM_ERR_FULL = -102_int32
   integer(int32), parameter, public :: MEM_ERR_DUPLICATE_ID = -103_int32
   integer(int32), parameter, public :: MEM_ERR_NOT_FOUND = -104_int32
   integer(int32), parameter, public :: MEM_ERR_EMPTY = -105_int32
   integer(int32), parameter, public :: MEM_ERR_NO_LSH_CANDIDATE = -106_int32
   integer(int32), parameter, public :: MEM_ERR_COUNTER_OVERFLOW = -107_int32

   type, public :: hds_memory_t
      integer(int32) :: capacity = 0_int32
      integer(int32) :: count = 0_int32
      integer(int32) :: hash_size = 0_int32

      ! Packed SoA matrix: each column is one 2 KiB hypervector.
      integer(int64), allocatable :: vector_bits(:, :)
      integer(int32), allocatable :: ids(:)
      character(len=LABEL_LEN), allocatable :: labels(:)
      integer(int32), allocatable :: experience_count(:)

      ! Open-addressed ID index.  Occupancy is separate from keys, so every
      ! int32 identifier, including -1, is supported without a sentinel clash.
      logical, allocatable :: hash_used(:)
      integer(int32), allocatable :: id_hash_keys(:)
      integer(int32), allocatable :: id_hash_slots(:)

      ! Fixed-depth LSH candidate table.  It accelerates exact hits and offers
      ! an explicitly approximate candidate-only query API.
      integer(int32), allocatable :: lsh_table(:, :)
      integer(int32), allocatable :: lsh_counts(:)
   end type hds_memory_t

   type, public :: cleanup_result_t
      logical :: found = .false.
      logical :: exact_search = .false.
      logical :: lsh_seeded = .false.
      integer(int32) :: id = MEM_UNKNOWN_ID
      integer(int32) :: slot = 0_int32
      integer(int32) :: distance = HV_BITS + 1_int32
      integer(int32) :: candidates_examined = 0_int32
      character(len=LABEL_LEN) :: label = 'UNKNOWN'
      real(real32) :: similarity = -1.0_real32
   end type cleanup_result_t

   public :: mem_init, mem_destroy, mem_is_initialized
   public :: mem_register_symbol, mem_get_by_id, mem_cleanup, mem_cleanup_lsh
   public :: mem_associate_hebbian, mem_rebuild_lsh, mem_footprint_bytes
   public :: mem_lsh_bucket

contains

   pure function mem_is_initialized(mem) result(initialized)
      type(hds_memory_t), intent(in) :: mem
      logical :: initialized

      initialized = mem%capacity > 0_int32 .and. allocated(mem%vector_bits) .and. &
                    allocated(mem%ids) .and. allocated(mem%hash_used) .and. &
                    allocated(mem%lsh_table)
   end function mem_is_initialized

   subroutine mem_destroy(mem)
      type(hds_memory_t), intent(inout) :: mem

      if (allocated(mem%vector_bits)) deallocate(mem%vector_bits)
      if (allocated(mem%ids)) deallocate(mem%ids)
      if (allocated(mem%labels)) deallocate(mem%labels)
      if (allocated(mem%experience_count)) deallocate(mem%experience_count)
      if (allocated(mem%hash_used)) deallocate(mem%hash_used)
      if (allocated(mem%id_hash_keys)) deallocate(mem%id_hash_keys)
      if (allocated(mem%id_hash_slots)) deallocate(mem%id_hash_slots)
      if (allocated(mem%lsh_table)) deallocate(mem%lsh_table)
      if (allocated(mem%lsh_counts)) deallocate(mem%lsh_counts)
      mem%capacity = 0_int32
      mem%count = 0_int32
      mem%hash_size = 0_int32
   end subroutine mem_destroy

   ! Initializes every storage area once.  All normal registration, lookup,
   ! cleanup, and Hebbian update operations are allocation-free afterwards.
   subroutine mem_init(mem, max_symbols, status)
      type(hds_memory_t), intent(out) :: mem
      integer(int32), intent(in) :: max_symbols
      integer(int32), intent(out) :: status
      integer(int32) :: allocation_status
      integer(int64) :: wanted_hash_size

      status = MEM_OK
      mem%capacity = 0_int32
      mem%count = 0_int32
      mem%hash_size = 0_int32

      if (max_symbols <= 0_int32) then
         status = MEM_ERR_INVALID_ARGUMENT
         return
      end if

      wanted_hash_size = 2_int64 * int(max_symbols, int64) + 1_int64
      if (wanted_hash_size > int(huge(0_int32), int64)) then
         status = MEM_ERR_INVALID_ARGUMENT
         return
      end if

      mem%capacity = max_symbols
      mem%hash_size = int(wanted_hash_size, int32)

      allocate(mem%vector_bits(HV_WORDS, max_symbols), stat=allocation_status)
      if (allocation_status /= 0) then
         status = MEM_ERR_ALLOCATION
         call mem_destroy(mem)
         return
      end if
      allocate(mem%ids(max_symbols), stat=allocation_status)
      if (allocation_status /= 0) then
         status = MEM_ERR_ALLOCATION
         call mem_destroy(mem)
         return
      end if
      allocate(mem%labels(max_symbols), stat=allocation_status)
      if (allocation_status /= 0) then
         status = MEM_ERR_ALLOCATION
         call mem_destroy(mem)
         return
      end if
      allocate(mem%experience_count(max_symbols), stat=allocation_status)
      if (allocation_status /= 0) then
         status = MEM_ERR_ALLOCATION
         call mem_destroy(mem)
         return
      end if
      allocate(mem%hash_used(mem%hash_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = MEM_ERR_ALLOCATION
         call mem_destroy(mem)
         return
      end if
      allocate(mem%id_hash_keys(mem%hash_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = MEM_ERR_ALLOCATION
         call mem_destroy(mem)
         return
      end if
      allocate(mem%id_hash_slots(mem%hash_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = MEM_ERR_ALLOCATION
         call mem_destroy(mem)
         return
      end if
      allocate(mem%lsh_table(BUCKET_DEPTH, LSH_BUCKETS), stat=allocation_status)
      if (allocation_status /= 0) then
         status = MEM_ERR_ALLOCATION
         call mem_destroy(mem)
         return
      end if
      allocate(mem%lsh_counts(LSH_BUCKETS), stat=allocation_status)
      if (allocation_status /= 0) then
         status = MEM_ERR_ALLOCATION
         call mem_destroy(mem)
         return
      end if

      mem%vector_bits = 0_int64
      mem%ids = 0_int32
      mem%labels = ''
      mem%experience_count = 0_int32
      mem%hash_used = .false.
      mem%id_hash_keys = 0_int32
      mem%id_hash_slots = 0_int32
      mem%lsh_table = 0_int32
      mem%lsh_counts = 0_int32
   end subroutine mem_init

   ! The 12 selected bits deliberately come from separated word positions.
   ! This is only a candidate index; it is never treated as an exact metric.
   pure function mem_lsh_bucket(v) result(bucket)
      type(hv_t), intent(in) :: v
      integer(int32) :: bucket
      integer(int32) :: projection_bit, word_index

      bucket = 0_int32
      do projection_bit = 0, 11
         word_index = projection_bit * 21_int32 + 1_int32
         if (btest(v%bits(word_index), projection_bit)) then
            bucket = ibset(bucket, projection_bit)
         end if
      end do
      bucket = bucket + 1_int32
   end function mem_lsh_bucket

   pure function hash_start_index(symbol_id, hash_size) result(index)
      integer(int32), intent(in) :: symbol_id, hash_size
      integer(int32) :: index, mixed, nonnegative

      mixed = symbol_id
      mixed = ieor(mixed, ishft(mixed, -16))
      mixed = ieor(mixed, ishft(mixed, 7))
      mixed = ieor(mixed, ishft(mixed, -9))
      nonnegative = iand(mixed, huge(0_int32))
      index = modulo(nonnegative, hash_size) + 1_int32
   end function hash_start_index

   ! Returns the entry slot when found, and otherwise the free hash-table cell
   ! where the caller may insert.  The table has no deletion/tombstone state.
   subroutine hash_lookup(mem, symbol_id, slot, found, table_index)
      type(hds_memory_t), intent(in) :: mem
      integer(int32), intent(in) :: symbol_id
      integer(int32), intent(out) :: slot, table_index
      logical, intent(out) :: found
      integer(int32) :: probe, step

      slot = 0_int32
      table_index = 0_int32
      found = .false.
      if (.not. mem_is_initialized(mem)) return

      probe = hash_start_index(symbol_id, mem%hash_size)
      do step = 1, mem%hash_size
         if (.not. mem%hash_used(probe)) then
            table_index = probe
            return
         end if
         if (mem%id_hash_keys(probe) == symbol_id) then
            slot = mem%id_hash_slots(probe)
            table_index = probe
            found = .true.
            return
         end if
         probe = modulo(probe, mem%hash_size) + 1_int32
      end do
   end subroutine hash_lookup

   subroutine lsh_insert_slot(mem, slot)
      type(hds_memory_t), intent(inout) :: mem
      integer(int32), intent(in) :: slot
      type(hv_t) :: vector
      integer(int32) :: bucket, bucket_count

      vector%bits = mem%vector_bits(:, slot)
      bucket = mem_lsh_bucket(vector)
      bucket_count = mem%lsh_counts(bucket)
      if (bucket_count < BUCKET_DEPTH) then
         mem%lsh_counts(bucket) = bucket_count + 1_int32
         mem%lsh_table(bucket_count + 1_int32, bucket) = slot
      end if
   end subroutine lsh_insert_slot

   ! Registers one immutable base symbol.  Duplicate IDs are rejected rather
   ! than silently creating a stale LSH entry or a duplicate cleanup candidate.
   subroutine mem_register_symbol(mem, symbol_id, label, out_hv, status, custom_hv)
      type(hds_memory_t), intent(inout) :: mem
      integer(int32), intent(in) :: symbol_id
      character(len=*), intent(in) :: label
      type(hv_t), intent(out) :: out_hv
      integer(int32), intent(out) :: status
      type(hv_t), intent(in), optional :: custom_hv
      integer(int32) :: slot, table_index
      logical :: found

      out_hv = hv_zero()
      status = MEM_OK
      if (.not. mem_is_initialized(mem)) then
         status = MEM_ERR_INVALID_ARGUMENT
         return
      end if

      call hash_lookup(mem, symbol_id, slot, found, table_index)
      if (found) then
         status = MEM_ERR_DUPLICATE_ID
         return
      end if
      if (mem%count >= mem%capacity .or. table_index == 0_int32) then
         status = MEM_ERR_FULL
         return
      end if

      slot = mem%count + 1_int32
      if (present(custom_hv)) then
         out_hv = custom_hv
      else
         out_hv = hv_from_seed(hv_seed_from_id(symbol_id))
      end if

      mem%vector_bits(:, slot) = out_hv%bits
      mem%ids(slot) = symbol_id
      mem%labels(slot) = label
      mem%experience_count(slot) = 1_int32
      mem%hash_used(table_index) = .true.
      mem%id_hash_keys(table_index) = symbol_id
      mem%id_hash_slots(table_index) = slot
      mem%count = slot
      call lsh_insert_slot(mem, slot)
   end subroutine mem_register_symbol

   subroutine mem_get_by_id(mem, symbol_id, out_hv, found, status)
      type(hds_memory_t), intent(in) :: mem
      integer(int32), intent(in) :: symbol_id
      type(hv_t), intent(out) :: out_hv
      logical, intent(out) :: found
      integer(int32), intent(out) :: status
      integer(int32) :: slot, table_index

      out_hv = hv_zero()
      found = .false.
      status = MEM_OK
      if (.not. mem_is_initialized(mem)) then
         status = MEM_ERR_INVALID_ARGUMENT
         return
      end if

      call hash_lookup(mem, symbol_id, slot, found, table_index)
      if (.not. found) then
         status = MEM_ERR_NOT_FOUND
         return
      end if
      out_hv%bits = mem%vector_bits(:, slot)
   end subroutine mem_get_by_id

   subroutine reset_cleanup_result(result)
      type(cleanup_result_t), intent(out) :: result

      result%found = .false.
      result%exact_search = .false.
      result%lsh_seeded = .false.
      result%id = MEM_UNKNOWN_ID
      result%slot = 0_int32
      result%distance = HV_BITS + 1_int32
      result%candidates_examined = 0_int32
      result%label = 'UNKNOWN'
      result%similarity = -1.0_real32
   end subroutine reset_cleanup_result

   subroutine assign_cleanup_result(mem, slot, distance, exact_search, lsh_seeded, &
                                    candidates_examined, result)
      type(hds_memory_t), intent(in) :: mem
      integer(int32), intent(in) :: slot, distance, candidates_examined
      logical, intent(in) :: exact_search, lsh_seeded
      type(cleanup_result_t), intent(out) :: result

      call reset_cleanup_result(result)
      result%found = .true.
      result%exact_search = exact_search
      result%lsh_seeded = lsh_seeded
      result%id = mem%ids(slot)
      result%slot = slot
      result%distance = distance
      result%candidates_examined = candidates_examined
      result%label = mem%labels(slot)
      result%similarity = 1.0_real32 - 2.0_real32 * real(distance, real32) &
                          / real(HV_BITS, real32)
   end subroutine assign_cleanup_result

   ! Exact nearest-neighbour cleanup.  A matching LSH entry (distance zero) is
   ! a mathematically safe early exit; every non-exact query scans all symbols
   ! and resolves ties deterministically in favour of the lowest slot.
   subroutine mem_cleanup(mem, query_hv, result, status)
      type(hds_memory_t), intent(in) :: mem
      type(hv_t), intent(in) :: query_hv
      type(cleanup_result_t), intent(out) :: result
      integer(int32), intent(out) :: status
      type(hv_t) :: candidate
      integer(int32) :: bucket, k, slot, i, distance, min_distance
      logical :: lsh_seeded

      call reset_cleanup_result(result)
      status = MEM_OK
      if (.not. mem_is_initialized(mem)) then
         status = MEM_ERR_INVALID_ARGUMENT
         return
      end if
      if (mem%count == 0_int32) then
         status = MEM_ERR_EMPTY
         return
      end if

      lsh_seeded = .false.
      bucket = mem_lsh_bucket(query_hv)
      do k = 1, mem%lsh_counts(bucket)
         slot = mem%lsh_table(k, bucket)
         candidate%bits = mem%vector_bits(:, slot)
         distance = hv_hamming(query_hv, candidate)
         lsh_seeded = .true.
         if (distance == 0_int32) then
            call assign_cleanup_result(mem, slot, distance, .true., .true., k, result)
            return
         end if
      end do

      min_distance = HV_BITS + 1_int32
      !$omp parallel do default(shared) private(i, candidate, distance) reduction(min:min_distance)
      do i = 1, mem%count
         candidate%bits = mem%vector_bits(:, i)
         distance = hv_hamming(query_hv, candidate)
         min_distance = min(min_distance, distance)
      end do
      !$omp end parallel do

      slot = 1_int32
      do i = 1, mem%count
         candidate%bits = mem%vector_bits(:, i)
         if (hv_hamming(query_hv, candidate) == min_distance) then
            slot = i
            exit
         end if
      end do
      call assign_cleanup_result(mem, slot, min_distance, .true., lsh_seeded, &
                                 mem%count + mem%lsh_counts(bucket), result)
   end subroutine mem_cleanup

   ! Candidate-only LSH lookup.  This is intentionally a separate API because
   ! fixed-depth LSH can omit the true nearest neighbour under collisions.
   subroutine mem_cleanup_lsh(mem, query_hv, result, status)
      type(hds_memory_t), intent(in) :: mem
      type(hv_t), intent(in) :: query_hv
      type(cleanup_result_t), intent(out) :: result
      integer(int32), intent(out) :: status
      type(hv_t) :: candidate
      integer(int32) :: bucket, k, slot, best_slot, distance, min_distance

      call reset_cleanup_result(result)
      status = MEM_OK
      if (.not. mem_is_initialized(mem)) then
         status = MEM_ERR_INVALID_ARGUMENT
         return
      end if
      if (mem%count == 0_int32) then
         status = MEM_ERR_EMPTY
         return
      end if

      bucket = mem_lsh_bucket(query_hv)
      if (mem%lsh_counts(bucket) == 0_int32) then
         status = MEM_ERR_NO_LSH_CANDIDATE
         return
      end if

      min_distance = HV_BITS + 1_int32
      best_slot = 0_int32
      do k = 1, mem%lsh_counts(bucket)
         slot = mem%lsh_table(k, bucket)
         candidate%bits = mem%vector_bits(:, slot)
         distance = hv_hamming(query_hv, candidate)
         if (distance < min_distance) then
            min_distance = distance
            best_slot = slot
         end if
      end do
      call assign_cleanup_result(mem, best_slot, min_distance, .false., .true., &
                                 mem%lsh_counts(bucket), result)
   end subroutine mem_cleanup_lsh

   ! Rebuilds only the small fixed LSH index after a mutable prototype changes.
   ! The base packed vector store and ID hash table remain allocation-free.
   subroutine mem_rebuild_lsh(mem, status)
      type(hds_memory_t), intent(inout) :: mem
      integer(int32), intent(out) :: status
      integer(int32) :: slot

      status = MEM_OK
      if (.not. mem_is_initialized(mem)) then
         status = MEM_ERR_INVALID_ARGUMENT
         return
      end if
      mem%lsh_table = 0_int32
      mem%lsh_counts = 0_int32
      do slot = 1, mem%count
         call lsh_insert_slot(mem, slot)
      end do
   end subroutine mem_rebuild_lsh

   ! Applies a deterministic sparse plasticity update to a registered symbol.
   ! LSH is rebuilt so the approximate index can never retain stale signatures.
   subroutine mem_associate_hebbian(mem, target_id, context_hv, status)
      type(hds_memory_t), intent(inout) :: mem
      integer(int32), intent(in) :: target_id
      type(hv_t), intent(in) :: context_hv
      integer(int32), intent(out) :: status
      type(hv_t) :: previous, updated
      integer(int32) :: slot, table_index
      integer(int64) :: seed
      logical :: found

      status = MEM_OK
      if (.not. mem_is_initialized(mem)) then
         status = MEM_ERR_INVALID_ARGUMENT
         return
      end if
      call hash_lookup(mem, target_id, slot, found, table_index)
      if (.not. found) then
         status = MEM_ERR_NOT_FOUND
         return
      end if
      if (mem%experience_count(slot) == huge(mem%experience_count(slot))) then
         status = MEM_ERR_COUNTER_OVERFLOW
         return
      end if

      previous%bits = mem%vector_bits(:, slot)
      mem%experience_count(slot) = mem%experience_count(slot) + 1_int32
      seed = ieor(hv_seed_from_id(target_id), &
                  ishftc(int(mem%experience_count(slot), int64), 23))
      updated = hv_hebbian_update(previous, context_hv, seed)
      mem%vector_bits(:, slot) = updated%bits
      call mem_rebuild_lsh(mem, status)
   end subroutine mem_associate_hebbian

   ! Exact allocated-storage estimate, based on the storage unit sizes reported
   ! by this compiler.  It excludes allocator metadata and OpenMP runtime state.
   pure function mem_footprint_bytes(mem) result(total_bytes)
      type(hds_memory_t), intent(in) :: mem
      integer(int64) :: total_bytes

      total_bytes = 0_int64
      if (allocated(mem%vector_bits)) then
         total_bytes = total_bytes + size(mem%vector_bits, kind=int64) * &
                       int(storage_size(mem%vector_bits(1, 1)) / 8, int64)
      end if
      if (allocated(mem%ids)) then
         total_bytes = total_bytes + size(mem%ids, kind=int64) * &
                       int(storage_size(mem%ids(1)) / 8, int64)
      end if
      if (allocated(mem%labels)) then
         total_bytes = total_bytes + size(mem%labels, kind=int64) * &
                       int(storage_size(mem%labels(1)) / 8, int64)
      end if
      if (allocated(mem%experience_count)) then
         total_bytes = total_bytes + size(mem%experience_count, kind=int64) * &
                       int(storage_size(mem%experience_count(1)) / 8, int64)
      end if
      if (allocated(mem%hash_used)) then
         total_bytes = total_bytes + size(mem%hash_used, kind=int64) * &
                       int(storage_size(mem%hash_used(1)) / 8, int64)
      end if
      if (allocated(mem%id_hash_keys)) then
         total_bytes = total_bytes + size(mem%id_hash_keys, kind=int64) * &
                       int(storage_size(mem%id_hash_keys(1)) / 8, int64)
      end if
      if (allocated(mem%id_hash_slots)) then
         total_bytes = total_bytes + size(mem%id_hash_slots, kind=int64) * &
                       int(storage_size(mem%id_hash_slots(1)) / 8, int64)
      end if
      if (allocated(mem%lsh_table)) then
         total_bytes = total_bytes + size(mem%lsh_table, kind=int64) * &
                       int(storage_size(mem%lsh_table(1, 1)) / 8, int64)
      end if
      if (allocated(mem%lsh_counts)) then
         total_bytes = total_bytes + size(mem%lsh_counts, kind=int64) * &
                       int(storage_size(mem%lsh_counts(1)) / 8, int64)
      end if
   end function mem_footprint_bytes

end module symbolic_memory
