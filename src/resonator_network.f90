! SPDX-License-Identifier: MIT
!==============================================================================
!  resonator_network.f90
!  Multi-factor factorization for binary/bipolar VSA/HDC representations.
!
!  Given a target T = F_1 XOR F_2 XOR ... XOR F_N and one cleanup codebook per
!  factor slot, the discrete Resonator Network repeatedly estimates:
!
!    F_i <- Cleanup_i(T XOR product_{j /= i}(F_j))
!
!  The network is a bounded, deterministic multi-restart heuristic.  An
!  optional meet-in-the-middle exact verifier is included for bounded codebook
!  products.  It certifies a result and detects at least one alternative exact
!  factorization; it does not make an unbounded factorization problem easy.
!==============================================================================
module resonator_network
   use, intrinsic :: iso_fortran_env, only: int32, int64, real32
   use hyper_types
   use symbolic_memory, only: hds_memory_t, cleanup_result_t, LABEL_LEN, MEM_OK, &
                              mem_cleanup, mem_is_initialized
   implicit none
   private

   integer(int32), parameter, public :: RN_OK = 0_int32
   integer(int32), parameter, public :: RN_WARN_AMBIGUOUS = 1_int32
   integer(int32), parameter, public :: RN_ERR_INVALID_ARGUMENT = -300_int32
   integer(int32), parameter, public :: RN_ERR_ALLOCATION = -301_int32
   integer(int32), parameter, public :: RN_ERR_CODEBOOK = -302_int32
   integer(int32), parameter, public :: RN_ERR_UNRESOLVED = -303_int32
   integer(int32), parameter, public :: RN_ERR_SEARCH_LIMIT = -304_int32
   integer(int32), parameter, public :: RN_ERR_MEMORY = -305_int32

   ! The exact verifier bounds the number of combinations in EACH half of the
   ! split, not the total Cartesian product of every factor codebook.
   type, public :: resonator_config_t
      integer(int32) :: max_iterations = 32_int32
      integer(int32) :: restarts = 8_int32
      integer(int32) :: max_exact_combinations = 65536_int32
      integer(int64) :: initialization_seed = int(z'4D595DF4D0F33173', int64)
      logical :: synchronous_updates = .true.
      logical :: enable_exact_verifier = .true.
   end type resonator_config_t

   type, public :: resonator_result_t
      ! network_converged reports the iterative Resonator phase alone; converged
      ! reports the final solver result after an optional exact verification pass.
      logical :: converged = .false.
      logical :: network_converged = .false.
      logical :: exact = .false.
      logical :: used_exact_verifier = .false.
      logical :: exact_verifier_attempted = .false.
      logical :: search_limit_reached = .false.
      logical :: ambiguous = .false.
      integer(int32) :: n_factors = 0_int32
      integer(int32) :: iterations = 0_int32
      integer(int32) :: restarts_attempted = 0_int32
      integer(int32) :: network_best_distance = HV_BITS + 1_int32
      integer(int32) :: hamming_distance = HV_BITS + 1_int32
      ! 0 = none, 1 = one, 2 = two or more (deliberately capped at two).
      integer(int32) :: exact_solution_count = 0_int32
      real(real32) :: similarity = -1.0_real32
      type(hv_t), allocatable :: factors(:)
      integer(int32), allocatable :: factor_slots(:)
      integer(int32), allocatable :: factor_ids(:)
      character(len=LABEL_LEN), allocatable :: factor_labels(:)
   end type resonator_result_t

   public :: resonator_factorize
   public :: resonator_result_init, resonator_result_destroy

contains

   subroutine resonator_result_destroy(result)
      type(resonator_result_t), intent(inout) :: result

      if (allocated(result%factors)) deallocate(result%factors)
      if (allocated(result%factor_slots)) deallocate(result%factor_slots)
      if (allocated(result%factor_ids)) deallocate(result%factor_ids)
      if (allocated(result%factor_labels)) deallocate(result%factor_labels)
      result%converged = .false.
      result%network_converged = .false.
      result%exact = .false.
      result%used_exact_verifier = .false.
      result%exact_verifier_attempted = .false.
      result%search_limit_reached = .false.
      result%ambiguous = .false.
      result%n_factors = 0_int32
      result%iterations = 0_int32
      result%restarts_attempted = 0_int32
      result%network_best_distance = HV_BITS + 1_int32
      result%hamming_distance = HV_BITS + 1_int32
      result%exact_solution_count = 0_int32
      result%similarity = -1.0_real32
   end subroutine resonator_result_destroy

   subroutine resonator_result_init(result, n_factors, status)
      type(resonator_result_t), intent(inout) :: result
      integer(int32), intent(in) :: n_factors
      integer(int32), intent(out) :: status
      integer(int32) :: allocation_status, factor

      call resonator_result_destroy(result)
      status = RN_OK
      if (n_factors <= 0_int32) then
         status = RN_ERR_INVALID_ARGUMENT
         return
      end if

      allocate(result%factors(n_factors), stat=allocation_status)
      if (allocation_status /= 0) then
         status = RN_ERR_ALLOCATION
         call resonator_result_destroy(result)
         return
      end if
      allocate(result%factor_slots(n_factors), stat=allocation_status)
      if (allocation_status /= 0) then
         status = RN_ERR_ALLOCATION
         call resonator_result_destroy(result)
         return
      end if
      allocate(result%factor_ids(n_factors), stat=allocation_status)
      if (allocation_status /= 0) then
         status = RN_ERR_ALLOCATION
         call resonator_result_destroy(result)
         return
      end if
      allocate(result%factor_labels(n_factors), stat=allocation_status)
      if (allocation_status /= 0) then
         status = RN_ERR_ALLOCATION
         call resonator_result_destroy(result)
         return
      end if

      result%n_factors = n_factors
      do factor = 1, n_factors
         result%factors(factor) = hv_zero()
      end do
      result%factor_slots = 0_int32
      result%factor_ids = 0_int32
      result%factor_labels = 'UNKNOWN'
   end subroutine resonator_result_init

   subroutine resonator_factorize(target_hv, codebooks, config, result, status)
      type(hv_t), intent(in) :: target_hv
      type(hds_memory_t), intent(in) :: codebooks(:)
      type(resonator_config_t), intent(in) :: config
      type(resonator_result_t), intent(inout) :: result
      integer(int32), intent(out) :: status
      integer(int32), allocatable :: current_slots(:), next_slots(:)
      integer(int32), allocatable :: restart_best_slots(:), global_best_slots(:)
      integer(int32) :: n_factors, factor, restart, allocation_status
      integer(int32) :: restart_distance, global_best_distance, iteration_count
      integer(int32) :: total_iterations, network_status, verifier_status
      logical :: restart_exact, network_exact

      status = RN_OK
      call resonator_result_destroy(result)
      n_factors = size(codebooks, kind=int32)
      if (n_factors < 2_int32) then
         status = RN_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%max_iterations <= 0_int32 .or. config%restarts <= 0_int32) then
         status = RN_ERR_INVALID_ARGUMENT
         return
      end if
      if (int(config%max_iterations, int64) * int(config%restarts, int64) > &
          int(huge(0_int32), int64)) then
         status = RN_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%enable_exact_verifier .and. config%max_exact_combinations <= 0_int32) then
         status = RN_ERR_INVALID_ARGUMENT
         return
      end if

      do factor = 1, n_factors
         if (.not. mem_is_initialized(codebooks(factor))) then
            status = RN_ERR_CODEBOOK
            return
         end if
         if (codebooks(factor)%count <= 0_int32 .or. &
             codebooks(factor)%count > codebooks(factor)%capacity) then
            status = RN_ERR_CODEBOOK
            return
         end if
      end do

      call resonator_result_init(result, n_factors, status)
      if (status /= RN_OK) return

      allocate(current_slots(n_factors), next_slots(n_factors), &
               restart_best_slots(n_factors), global_best_slots(n_factors), &
               stat=allocation_status)
      if (allocation_status /= 0) then
         status = RN_ERR_ALLOCATION
         call resonator_result_destroy(result)
         return
      end if

      global_best_distance = HV_BITS + 1_int32
      global_best_slots = 1_int32
      total_iterations = 0_int32
      network_exact = .false.

      do restart = 1, config%restarts
         call initialize_slots(codebooks, restart, config%initialization_seed, current_slots)
         call run_discrete_resonator(target_hv, codebooks, config, current_slots, next_slots, &
                                     restart_best_slots, restart_distance, iteration_count, &
                                     restart_exact, network_status)
         if (network_status /= RN_OK) then
            status = network_status
            call resonator_result_destroy(result)
            return
         end if

         total_iterations = total_iterations + iteration_count
         if (restart_distance < global_best_distance) then
            global_best_distance = restart_distance
            global_best_slots = restart_best_slots
         end if
         if (restart_exact) then
            network_exact = .true.
            exit
         end if
      end do

      result%iterations = total_iterations
      result%restarts_attempted = min(restart, config%restarts)
      result%network_best_distance = global_best_distance
      call populate_result(codebooks, global_best_slots, global_best_distance, result)
      result%converged = network_exact
      result%network_converged = network_exact
      result%exact = network_exact
      if (network_exact) result%exact_solution_count = 1_int32

      ! The exact verifier runs after the Resonator Network, not instead of it.
      ! It can replace a heuristic solution with a deterministic certificate and
      ! can expose ambiguity caused by duplicate/equivalent codebook entries.
      if (config%enable_exact_verifier) then
         call exact_verify_factorization(target_hv, codebooks, config, result, verifier_status)
         select case (verifier_status)
         case (RN_OK, RN_WARN_AMBIGUOUS)
            status = verifier_status
         case (RN_ERR_SEARCH_LIMIT)
            result%search_limit_reached = .true.
            if (network_exact) then
               status = RN_OK
            else
               status = RN_ERR_SEARCH_LIMIT
            end if
         case (RN_ERR_UNRESOLVED)
            if (network_exact) then
               status = RN_OK
            else
               status = RN_ERR_UNRESOLVED
            end if
         case default
            status = verifier_status
         end select
      else if (network_exact) then
         status = RN_OK
      else
         status = RN_ERR_UNRESOLVED
      end if
   end subroutine resonator_factorize

   ! Seed each restart without signed arithmetic overflow.  The hypervector
   ! generator supplies a deterministic, well-diffused 64-bit selection word.
   subroutine initialize_slots(codebooks, restart, seed, slots)
      type(hds_memory_t), intent(in) :: codebooks(:)
      integer(int32), intent(in) :: restart
      integer(int64), intent(in) :: seed
      integer(int32), intent(out) :: slots(:)
      type(hv_t) :: mixer
      integer(int64) :: state, nonnegative
      integer(int32) :: factor

      do factor = 1, size(codebooks, kind=int32)
         state = ieor(seed, ishftc(int(restart, int64), 17))
         state = ieor(state, ishftc(int(factor, int64), -23))
         mixer = hv_from_seed(state)
         nonnegative = iand(mixer%bits(1), huge(0_int64))
         slots(factor) = int(modulo(nonnegative, int(codebooks(factor)%count, int64)), int32) &
                         + 1_int32
      end do
   end subroutine initialize_slots

   subroutine run_discrete_resonator(target_hv, codebooks, config, current_slots, next_slots, &
                                     best_slots, best_distance, iterations, exact, status)
      type(hv_t), intent(in) :: target_hv
      type(hds_memory_t), intent(in) :: codebooks(:)
      type(resonator_config_t), intent(in) :: config
      integer(int32), intent(inout) :: current_slots(:)
      integer(int32), intent(out) :: next_slots(:), best_slots(:)
      integer(int32), intent(out) :: best_distance, iterations, status
      logical, intent(out) :: exact
      type(hv_t) :: residual, composite, candidate
      type(cleanup_result_t) :: cleanup
      integer(int32) :: factor, other, cleanup_status, distance, iteration
      logical :: changed

      status = RN_OK
      exact = .false.
      iterations = 0_int32
      call compose_slots(codebooks, current_slots, composite)
      best_distance = hv_hamming(target_hv, composite)
      best_slots = current_slots
      if (best_distance == 0_int32) then
         exact = .true.
         return
      end if

      do iteration = 1, config%max_iterations
         iterations = iteration
         changed = .false.
         if (config%synchronous_updates) then
            do factor = 1, size(codebooks, kind=int32)
               residual = target_hv
               do other = 1, size(codebooks, kind=int32)
                  if (other /= factor) then
                     candidate%bits = codebooks(other)%vector_bits(:, current_slots(other))
                     residual = hv_bind(residual, candidate)
                  end if
               end do
               call mem_cleanup(codebooks(factor), residual, cleanup, cleanup_status)
               if (cleanup_status /= MEM_OK) then
                  status = RN_ERR_MEMORY
                  return
               end if
               next_slots(factor) = cleanup%slot
               if (next_slots(factor) /= current_slots(factor)) changed = .true.
            end do
            current_slots = next_slots
         else
            do factor = 1, size(codebooks, kind=int32)
               residual = target_hv
               do other = 1, size(codebooks, kind=int32)
                  if (other /= factor) then
                     candidate%bits = codebooks(other)%vector_bits(:, current_slots(other))
                     residual = hv_bind(residual, candidate)
                  end if
               end do
               call mem_cleanup(codebooks(factor), residual, cleanup, cleanup_status)
               if (cleanup_status /= MEM_OK) then
                  status = RN_ERR_MEMORY
                  return
               end if
               if (cleanup%slot /= current_slots(factor)) changed = .true.
               current_slots(factor) = cleanup%slot
            end do
            next_slots = current_slots
         end if

         call compose_slots(codebooks, current_slots, composite)
         distance = hv_hamming(target_hv, composite)
         if (distance < best_distance) then
            best_distance = distance
            best_slots = current_slots
         end if
         if (distance == 0_int32) then
            exact = .true.
            return
         end if
         if (.not. changed) return
      end do
   end subroutine run_discrete_resonator

   subroutine compose_slots(codebooks, slots, composite)
      type(hds_memory_t), intent(in) :: codebooks(:)
      integer(int32), intent(in) :: slots(:)
      type(hv_t), intent(out) :: composite
      type(hv_t) :: candidate
      integer(int32) :: factor

      composite = hv_zero()
      do factor = 1, size(codebooks, kind=int32)
         candidate%bits = codebooks(factor)%vector_bits(:, slots(factor))
         composite = hv_bind(composite, candidate)
      end do
   end subroutine compose_slots

   subroutine populate_result(codebooks, slots, distance, result)
      type(hds_memory_t), intent(in) :: codebooks(:)
      integer(int32), intent(in) :: slots(:)
      integer(int32), intent(in) :: distance
      type(resonator_result_t), intent(inout) :: result
      integer(int32) :: factor

      do factor = 1, size(codebooks, kind=int32)
         result%factor_slots(factor) = slots(factor)
         result%factor_ids(factor) = codebooks(factor)%ids(slots(factor))
         result%factor_labels(factor) = codebooks(factor)%labels(slots(factor))
         result%factors(factor)%bits = codebooks(factor)%vector_bits(:, slots(factor))
      end do
      result%hamming_distance = distance
      result%similarity = 1.0_real32 - 2.0_real32 * real(distance, real32) &
                          / real(HV_BITS, real32)
   end subroutine populate_result

   ! Computes a non-cryptographic 64-bit fingerprint for the bounded exact
   ! verifier.  A full hv_equal check is always made after a hash hit, so a
   ! fingerprint collision cannot produce a false factorization.
   pure function hv_fingerprint(v) result(fingerprint)
      type(hv_t), intent(in) :: v
      integer(int64) :: fingerprint

      fingerprint = ieor(v%bits(1), ishftc(v%bits(67), 13))
      fingerprint = ieor(fingerprint, ishftc(v%bits(131), -21))
      fingerprint = ieor(fingerprint, ishftc(v%bits(193), 37))
      fingerprint = ieor(fingerprint, v%bits(HV_WORDS))
   end function hv_fingerprint

   pure function fingerprint_bucket(fingerprint, hash_size) result(bucket)
      integer(int64), intent(in) :: fingerprint
      integer(int32), intent(in) :: hash_size
      integer(int32) :: bucket
      integer(int64) :: nonnegative

      nonnegative = iand(fingerprint, huge(0_int64))
      bucket = int(modulo(nonnegative, int(hash_size, int64)), int32) + 1_int32
   end function fingerprint_bucket

   subroutine bounded_combination_count(codebooks, first_factor, factor_count, max_count, &
                                        combination_count, within_limit)
      type(hds_memory_t), intent(in) :: codebooks(:)
      integer(int32), intent(in) :: first_factor, factor_count, max_count
      integer(int32), intent(out) :: combination_count
      logical, intent(out) :: within_limit
      integer(int32) :: offset, candidates

      combination_count = 1_int32
      within_limit = .true.
      do offset = 0, factor_count - 1_int32
         candidates = codebooks(first_factor + offset)%count
         if (candidates <= 0_int32) then
            combination_count = 0_int32
            within_limit = .false.
            return
         end if
         if (combination_count > max_count / candidates) then
            combination_count = 0_int32
            within_limit = .false.
            return
         end if
         combination_count = combination_count * candidates
      end do
   end subroutine bounded_combination_count

   subroutine increment_slot_tuple(codebooks, first_factor, factor_count, slots, exhausted)
      type(hds_memory_t), intent(in) :: codebooks(:)
      integer(int32), intent(in) :: first_factor, factor_count
      integer(int32), intent(inout) :: slots(:)
      logical, intent(out) :: exhausted
      integer(int32) :: position

      exhausted = .false.
      do position = factor_count, 1_int32, -1_int32
         if (slots(position) < codebooks(first_factor + position - 1_int32)%count) then
            slots(position) = slots(position) + 1_int32
            return
         end if
         slots(position) = 1_int32
      end do
      exhausted = .true.
   end subroutine increment_slot_tuple

   subroutine compose_range(codebooks, first_factor, factor_count, slots, composite)
      type(hds_memory_t), intent(in) :: codebooks(:)
      integer(int32), intent(in) :: first_factor, factor_count
      integer(int32), intent(in) :: slots(:)
      type(hv_t), intent(out) :: composite
      type(hv_t) :: candidate
      integer(int32) :: offset

      composite = hv_zero()
      do offset = 1, factor_count
         candidate%bits = codebooks(first_factor + offset - 1_int32)%vector_bits(:, slots(offset))
         composite = hv_bind(composite, candidate)
      end do
   end subroutine compose_range

   ! Bounded meet-in-the-middle certificate.  Right-side products are indexed
   ! by a fingerprint chain; each potential hit is recomposed and fully checked.
   subroutine exact_verify_factorization(target_hv, codebooks, config, result, status)
      type(hv_t), intent(in) :: target_hv
      type(hds_memory_t), intent(in) :: codebooks(:)
      type(resonator_config_t), intent(in) :: config
      type(resonator_result_t), intent(inout) :: result
      integer(int32), intent(out) :: status
      integer(int64), allocatable :: right_keys(:)
      integer(int32), allocatable :: bucket_heads(:), next_entry(:), right_slots(:, :)
      integer(int32), allocatable :: left_tuple(:), right_tuple(:), solution_slots(:)
      type(hv_t) :: right_product, left_product, wanted_right, recomposed_right
      integer(int32) :: n_factors, left_factor_count, right_factor_count
      integer(int32) :: left_count, right_count, hash_size, entry, bucket, chain_entry
      integer(int32) :: allocation_status, exact_solutions
      logical :: within_left, within_right, exhausted

      status = RN_OK
      result%exact_verifier_attempted = .true.
      n_factors = size(codebooks, kind=int32)
      left_factor_count = n_factors / 2_int32
      right_factor_count = n_factors - left_factor_count

      call bounded_combination_count(codebooks, 1_int32, left_factor_count, &
                                     config%max_exact_combinations, left_count, within_left)
      call bounded_combination_count(codebooks, left_factor_count + 1_int32, right_factor_count, &
                                     config%max_exact_combinations, right_count, within_right)
      if (.not. within_left .or. .not. within_right) then
         status = RN_ERR_SEARCH_LIMIT
         return
      end if
      if (int(right_count, int64) * 2_int64 + 1_int64 > int(huge(0_int32), int64)) then
         status = RN_ERR_SEARCH_LIMIT
         return
      end if
      hash_size = 2_int32 * right_count + 1_int32

      allocate(right_keys(right_count), next_entry(right_count), &
               right_slots(right_factor_count, right_count), bucket_heads(hash_size), &
               left_tuple(left_factor_count), right_tuple(right_factor_count), &
               solution_slots(n_factors), stat=allocation_status)
      if (allocation_status /= 0) then
         status = RN_ERR_ALLOCATION
         return
      end if

      right_keys = 0_int64
      next_entry = 0_int32
      right_slots = 0_int32
      bucket_heads = 0_int32
      right_tuple = 1_int32

      do entry = 1, right_count
         call compose_range(codebooks, left_factor_count + 1_int32, right_factor_count, &
                            right_tuple, right_product)
         right_keys(entry) = hv_fingerprint(right_product)
         right_slots(:, entry) = right_tuple
         bucket = fingerprint_bucket(right_keys(entry), hash_size)
         next_entry(entry) = bucket_heads(bucket)
         bucket_heads(bucket) = entry
         call increment_slot_tuple(codebooks, left_factor_count + 1_int32, right_factor_count, &
                                   right_tuple, exhausted)
      end do

      exact_solutions = 0_int32
      left_tuple = 1_int32
      left_enumeration: do entry = 1, left_count
         call compose_range(codebooks, 1_int32, left_factor_count, left_tuple, left_product)
         wanted_right = hv_bind(target_hv, left_product)
         bucket = fingerprint_bucket(hv_fingerprint(wanted_right), hash_size)
         chain_entry = bucket_heads(bucket)

         do while (chain_entry /= 0_int32)
            if (right_keys(chain_entry) == hv_fingerprint(wanted_right)) then
               call compose_range(codebooks, left_factor_count + 1_int32, right_factor_count, &
                                  right_slots(:, chain_entry), recomposed_right)
               if (hv_equal(wanted_right, recomposed_right)) then
                  exact_solutions = min(2_int32, exact_solutions + 1_int32)
                  if (exact_solutions == 1_int32) then
                     solution_slots(1:left_factor_count) = left_tuple
                     solution_slots(left_factor_count + 1_int32:n_factors) = &
                        right_slots(:, chain_entry)
                  end if
                  if (exact_solutions == 2_int32) exit left_enumeration
               end if
            end if
            chain_entry = next_entry(chain_entry)
         end do

         call increment_slot_tuple(codebooks, 1_int32, left_factor_count, left_tuple, exhausted)
      end do left_enumeration

      if (exact_solutions == 0_int32) then
         status = RN_ERR_UNRESOLVED
         return
      end if

      call populate_result(codebooks, solution_slots, 0_int32, result)
      result%converged = .true.
      result%exact = .true.
      result%used_exact_verifier = .true.
      result%exact_solution_count = exact_solutions
      result%ambiguous = exact_solutions > 1_int32
      if (result%ambiguous) then
         status = RN_WARN_AMBIGUOUS
      else
         status = RN_OK
      end if
   end subroutine exact_verify_factorization

end module resonator_network
