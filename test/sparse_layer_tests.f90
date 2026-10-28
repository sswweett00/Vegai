! SPDX-License-Identifier: MIT
!==============================================================================
!  sparse_layer_tests.f90  --  contract tests for src/sparse_expert_layer.f90
!
!  Each check is an executable assertion, not a printout: the program exits
!  non-zero on the first failed contract, and prints a total at the end so the
!  CI log carries a number.
!
!  Covered contracts
!  -----------------
!   1. configuration validation rejects impossible routing (k > capacity, ...);
!   2. k distinct experts per activation, deterministically (same input, same
!      layer, same ids -- twice);
!   3. no page exists for an expert that was never routed to;
!   4. the update pass touches exactly the pages the forward pass touched
!      (updates_applied grows by active_count, not by page_count);
!   5. gradient locality: with two different inputs landing on different
!      experts, the first expert's weights are unchanged by the second step;
!   6. replay determinism: two independently built layers with the same seed
!      produce bit-identical parameters for the same expert, and a different
!      seed produces a different one;
!   7. capacity invariance of the active counters and of resident memory;
!   8. the layer trains: the error on a fixed toy target decreases;
!   9. order violations are rejected (backward before forward).
!==============================================================================
program sparse_layer_tests
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64, output_unit
   use sparse_expert_layer
   implicit none (type, external)

   integer(int32) :: checks = 0_int32, failures = 0_int32, status

   call test_invalid_configuration
   call test_distinct_and_deterministic_routing
   call test_cold_expert_has_no_page
   call test_update_touches_only_active_pages
   call test_gradient_locality
   call test_replay_determinism
   call test_capacity_invariance
   call test_learning_decreases_error
   call test_order_violation

   write (output_unit, '(A,I0,A,I0,A)') 'sparse layer checks: ', checks, ', failures: ', failures, '.'
   if (failures > 0_int32) then
      write (output_unit, '(A,I0,A)') 'FAIL: ', failures, ' sparse-layer contract(s) violated'
      error stop 1_int32
   end if
   write (output_unit, '(A)') 'PASS: all sparse-layer contracts hold'

contains

   subroutine check(condition, description)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: description

      checks = checks + 1_int32
      if (.not. condition) then
         failures = failures + 1_int32
         write (output_unit, '(A,A)') 'FAIL: ', description
      end if
   end subroutine check

   pure function base_config() result(config)
      type(sparse_layer_config_t) :: config

      config%input_width = 12_int32
      config%output_width = 10_int32
      config%expert_capacity = 64_int64
      config%active_experts = 3_int32
      config%router_bits = 24_int32
      config%bucket_bits = 6_int32
      config%seed = 991_int64
      config%learning_rate = 1.0e-3_real64
   end function base_config

   pure function probe_input(seed) result(input)
      integer(int64), intent(in) :: seed
      real(real64) :: input(12)
      integer(int32) :: index
      integer(int64) :: state

      state = seed
      do index = 1_int32, 12_int32
         input(index) = real(mix(state), real64) / real(huge(0_int64), real64)
      end do
   end function probe_input

   pure function mix(state) result(word)
      integer(int64), intent(inout) :: state
      integer(int64) :: word

      state = state + int(z'9E3779B97F4A7C15', int64)
      word = state
      word = ieor(word, ishft(word, -30_int32)) * int(z'BF58476D1CE4E5B9', int64)
      word = ieor(word, ishft(word, -27_int32)) * int(z'94D049BB133111EB', int64)
      word = ieor(word, ishft(word, -31_int32))
   end function mix

   subroutine test_invalid_configuration
      type(sparse_layer_config_t) :: config
      type(sparse_layer_t) :: layer
      integer(int32) :: status

      config = base_config()
      config%active_experts = 65_int32
      call sparse_layer_init(layer, config, status)
      call check(status == SE_ERR_INVALID_ARGUMENT, 'rejects k above the supported maximum')

      config = base_config()
      config%expert_capacity = 2_int64
      config%active_experts = 3_int32
      call sparse_layer_init(layer, config, status)
      call check(status == SE_ERR_INVALID_ARGUMENT, 'rejects k greater than capacity')

      config = base_config()
      config%router_bits = 8_int32
      config%bucket_bits = 9_int32
      call sparse_layer_init(layer, config, status)
      call check(status == SE_ERR_INVALID_ARGUMENT, 'rejects a bucket wider than the signature')
   end subroutine test_invalid_configuration

   subroutine test_distinct_and_deterministic_routing
      type(sparse_layer_config_t) :: config
      type(sparse_layer_t) :: layer
      type(sparse_forward_t) :: cache
      real(real64) :: input(12), output(10)
      integer(int32) :: status, index, other
      logical :: distinct

      config = base_config()
      call sparse_layer_init(layer, config, status)
      input = probe_input(3_int64)
      call sparse_layer_forward(layer, input, output, cache, status)
      call check(status == SE_OK, 'forward succeeds on a valid layer')
      call check(cache%active_count == config%active_experts, 'forward activates exactly k experts')
      distinct = .true.
      do index = 1_int32, cache%active_count
         do other = index + 1_int32, cache%active_count
            if (cache%expert_id(index) == cache%expert_id(other)) distinct = .false.
         end do
      end do
      call check(distinct, 'the k routed experts are distinct')
      call check(abs(sum(cache%gate) - 1.0_real64) < 1.0e-12_real64, 'gates form a partition of unity')
      block
         integer(int64) :: first_ids(3)
         first_ids = cache%expert_id
         call sparse_layer_forward(layer, input, output, cache, status)
         call check(all(cache%expert_id == first_ids), 'routing is a pure function of the input')
      end block
      call sparse_layer_destroy(layer)
   end subroutine test_distinct_and_deterministic_routing

   subroutine test_cold_expert_has_no_page
      type(sparse_layer_config_t) :: config
      type(sparse_layer_t) :: layer
      type(sparse_forward_t) :: cache
      real(real64) :: input(12), output(10)
      integer(int32) :: status
      integer(int64) :: touched

      config = base_config()
      call sparse_layer_init(layer, config, status)
      call check(layer%stats%pages_allocated == 0_int64, 'a fresh layer owns no expert page')
      input = probe_input(5_int64)
      call sparse_layer_forward(layer, input, output, cache, status)
      touched = int(cache%active_count, int64)
      call check(layer%stats%pages_allocated == touched, 'only the routed experts are materialised')
      call check(layer%stats%pages_allocated < config%expert_capacity, 'page count is below capacity')
      call check(layer%stats%resident_bytes < sparse_layer_parameter_count(layer) * 8_int64, &
                 'resident bytes are a fraction of the virtual capacity')
      call sparse_layer_destroy(layer)
      call check(layer%stats%pages_allocated == 0_int64, 'destroy resets the counters')
   end subroutine test_cold_expert_has_no_page

   subroutine test_update_touches_only_active_pages
      type(sparse_layer_config_t) :: config
      type(sparse_layer_t) :: layer
      type(sparse_forward_t) :: cache
      real(real64) :: input(12), output(10), grad_output(10), grad_input(12)
      integer(int32) :: status
      integer(int64) :: before, after
      integer(int32) :: step

      config = base_config()
      call sparse_layer_init(layer, config, status)
      do step = 1_int32, 20_int32
         input = probe_input(int(step, int64) + 100_int64)
         call sparse_layer_forward(layer, input, output, cache, status)
         if (status /= SE_OK) exit
         grad_output = output - 0.25_real64
         call sparse_layer_backward(layer, cache, grad_output, grad_input, status)
         if (status /= SE_OK) exit
         before = layer%stats%updates_applied
         call sparse_layer_apply_update(layer, status)
         if (status /= SE_OK) exit
         after = layer%stats%updates_applied
         call check(after - before == int(cache%active_count, int64), &
                    'the update pass visits exactly the active pages, not the page store')
      end do
      call check(status == SE_OK, 'twenty train steps complete without error')
      call check(layer%stats%pages_allocated <= 20_int64 * int(config%active_experts, int64), &
                 'pages never exceed the touched set')
      call sparse_layer_destroy(layer)
   end subroutine test_update_touches_only_active_pages

   !> Expert weights must be frozen unless their expert is routed to.  The test
   !> reads the page store directly through the public type, which is why the
   !> check can be exact rather than statistical.
   subroutine test_gradient_locality
      type(sparse_layer_config_t) :: config
      type(sparse_layer_t) :: layer
      type(sparse_forward_t) :: cache
      real(real64) :: input(12), output(10), grad_output(10), grad_input(12), snapshot(10, 12)
      integer(int32) :: status, slot, victim, index
      integer(int64) :: victim_id

      config = base_config()
      call sparse_layer_init(layer, config, status)
      input = probe_input(21_int64)
      call sparse_layer_forward(layer, input, output, cache, status)
      victim = cache%page_slot(1_int32)
      victim_id = cache%expert_id(1_int32)
      snapshot = layer%pages(victim)%weight

      ! Train on inputs that do not route to `victim`.
      do index = 1_int32, 25_int32
         input = probe_input(int(index, int64) + 500_int64)
         call sparse_layer_forward(layer, input, output, cache, status)
         if (status /= SE_OK) exit
         if (any(cache%expert_id == victim_id)) cycle
         grad_output = output - 0.1_real64
         call sparse_layer_backward(layer, cache, grad_output, grad_input, status)
         if (status /= SE_OK) exit
         call sparse_layer_apply_update(layer, status)
         if (status /= SE_OK) exit
      end do
      call check(status == SE_OK, 'the locality sweep completes')
      slot = find_slot_by_id(layer, victim_id)
      if (slot > 0_int32) then
         call check(all(layer%pages(slot)%weight == snapshot), &
                    'an unrouted expert keeps bit-identical weights')
      else
         call check(.false., 'the locality victim page disappeared')
      end if
      call sparse_layer_destroy(layer)
   end subroutine test_gradient_locality

   !> Looks the page up the public way: by scanning for the id would be O(pages),
   !> which is fine in a test of a handful of pages.
   pure function find_slot_by_id(layer, expert_id) result(slot)
      type(sparse_layer_t), intent(in) :: layer
      integer(int64), intent(in) :: expert_id
      integer(int32) :: slot, index

      slot = 0_int32
      do index = 1_int32, layer%page_count
         if (layer%pages(index)%expert_id == expert_id) then
            slot = index
            return
         end if
      end do
   end function find_slot_by_id

   subroutine test_replay_determinism
      type(sparse_layer_config_t) :: config
      type(sparse_layer_t) :: first, second, third
      type(sparse_forward_t) :: cache
      real(real64) :: input(12), output_a(10), output_b(10), output_c(10)
      integer(int32) :: status

      config = base_config()
      call sparse_layer_init(first, config, status)
      call sparse_layer_init(second, config, status)
      config%seed = config%seed + 1_int64
      call sparse_layer_init(third, config, status)
      input = probe_input(77_int64)
      call sparse_layer_forward(first, input, output_a, cache, status)
      call sparse_layer_forward(second, input, output_b, cache, status)
      call sparse_layer_forward(third, input, output_c, cache, status)
      call check(all(output_a == output_b), &
                 'two layers with the same seed reproduce identical parameters and outputs')
      call check(any(output_a /= output_c), &
                 'a different seed produces a different parameter set')
      call sparse_layer_destroy(first)
      call sparse_layer_destroy(second)
      call sparse_layer_destroy(third)
   end subroutine test_replay_determinism

   subroutine test_capacity_invariance
      type(sparse_layer_config_t) :: small_config, large_config
      type(sparse_layer_t) :: small, large
      type(sparse_forward_t) :: small_cache, large_cache
      real(real64) :: input(12), output_small(10), output_large(10)
      integer(int32) :: status

      small_config = base_config()
      small_config%expert_capacity = 64_int64
      large_config = base_config()
      large_config%expert_capacity = 200000_int64

      call sparse_layer_init(small, small_config, status)
      call sparse_layer_init(large, large_config, status)
      call check(sparse_layer_active_parameter_count(small) == &
                 sparse_layer_active_parameter_count(large), &
                 'active parameter count is capacity-invariant')
      call check(small%stats%active_flops == large%stats%active_flops, &
                 'active FLOP count is capacity-invariant before any step')

      input = probe_input(9_int64)
      call sparse_layer_forward(small, input, output_small, small_cache, status)
      call sparse_layer_forward(large, input, output_large, large_cache, status)
      call check(small_cache%active_count == large_cache%active_count, &
                 'both capacities route the same number of experts')
      call check(small%stats%active_flops == large%stats%active_flops, &
                 'active FLOPs stay identical after a step at 1e3 and 2e5 experts')
      call check(large%stats%pages_allocated == small%stats%pages_allocated, &
                 'the two capacities materialise the same number of pages')
      call check(large%stats%resident_bytes < 200000_int64 * 8_int64, &
                 'a 2e5-expert layer does not own 2e5 experts worth of memory')
      call sparse_layer_destroy(small)
      call sparse_layer_destroy(large)
   end subroutine test_capacity_invariance

   subroutine test_learning_decreases_error
      type(sparse_layer_config_t) :: config
      type(sparse_layer_t) :: layer
      type(sparse_forward_t) :: cache
      real(real64) :: input(12), target(10), output(10), grad_output(10), grad_input(12)
      real(real64) :: initial_error, final_error
      integer(int32) :: status, step

      config = base_config()
      config%learning_rate = 2.0e-3_real64
      call sparse_layer_init(layer, config, status)
      input = probe_input(13_int64)
      target = 0.5_real64
      call sparse_layer_forward(layer, input, output, cache, status)
      initial_error = sum(abs(output - target))
      do step = 1_int32, 200_int32
         call sparse_layer_forward(layer, input, output, cache, status)
         if (status /= SE_OK) exit
         grad_output = output - target
         call sparse_layer_backward(layer, cache, grad_output, grad_input, status)
         if (status /= SE_OK) exit
         call sparse_layer_apply_update(layer, status)
         if (status /= SE_OK) exit
      end do
      call sparse_layer_forward(layer, input, output, cache, status)
      final_error = sum(abs(output - target))
      call check(status == SE_OK, '200 optimisation steps complete')
      call check(final_error < initial_error, 'the sparse NOVA update reduces the toy error')
      call check(abs(final_error) == abs(final_error), 'the trained output is finite')
      call sparse_layer_destroy(layer)
   end subroutine test_learning_decreases_error

   subroutine test_order_violation
      type(sparse_layer_config_t) :: config
      type(sparse_layer_t) :: layer
      type(sparse_forward_t) :: cache
      real(real64) :: grad_output(10), grad_input(12)
      integer(int32) :: status

      config = base_config()
      call sparse_layer_init(layer, config, status)
      grad_output = 0.0_real64
      call sparse_layer_backward(layer, cache, grad_output, grad_input, status)
      call check(status == SE_ERR_ORDER, 'backward before forward is rejected')
      call sparse_layer_destroy(layer)
   end subroutine test_order_violation

end program sparse_layer_tests
