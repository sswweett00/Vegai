! SPDX-License-Identifier: MIT
program test_all
   use, intrinsic :: iso_fortran_env, only: int16, int32, int64
   use hyper_types
   use symbolic_memory
   use reasoning_core
   use test_support
   implicit none

   call test_hypervector_algebra()
   call test_symbolic_memory()
   call test_reasoning_pipeline()
   call finish_tests()

contains

   subroutine test_hypervector_algebra()
      type(hv_t) :: a, b, c, d, zero, noisy, noisy_again, bundled, expected
      type(hv_t) :: ones, before, updated_a, updated_b, splitmix_zero
      type(hv_acc_t) :: accumulator, accumulator_before
      integer(int32) :: status, distance
      real :: similarity

      splitmix_zero = hv_from_seed(0_int64)
      call assert_true(splitmix_zero%bits(1) == int(z'E220A8397B1DCDAF', int64), &
                       'portable SplitMix64 first word matches standard oracle')
      call assert_true(hv_seed_from_id(-1_int32) /= hv_seed_from_id(0_int32), &
                       'ID seed mixer preserves distinct signed IDs')

      a = hv_from_seed(12345_int64)
      b = hv_from_seed(67890_int64)
      c = hv_from_seed(-7654321_int64)
      d = hv_from_seed(12345_int64)
      zero = hv_zero()

      call assert_hv_equal(a, d, 'same seed produces identical vector')
      call assert_true(.not. hv_equal(a, b), 'different seeds produce different vectors')
      distance = hv_hamming(a, b)
      call assert_true(distance > 7000_int32 .and. distance < 9400_int32, &
                       'random vectors are near Hamming-orthogonal')
      similarity = hv_similarity(a, a)
      call assert_true(abs(similarity - 1.0) < 1.0e-6, 'identity similarity is one')
      call assert_hv_equal(hv_bind(a, zero), a, 'zero vector is XOR binding identity')
      call assert_hv_equal(hv_bind(a, hv_bind(a, b)), b, 'binding is self-inverse')
      call assert_hv_equal(hv_permute(hv_permute(a, 129_int32), -129_int32), a, &
                           'positive then negative permutation is identity')
      call assert_hv_equal(hv_permute(a, HV_BITS + 19_int32), hv_permute(a, 19_int32), &
                           'permutation wraps modulo dimension')

      call hv_acc_reset(accumulator)
      call hv_acc_add(accumulator, a, status)
      call assert_i32_equal(status, HV_OK, 'first accumulator add succeeds')
      call hv_acc_add(accumulator, b, status)
      call assert_i32_equal(status, HV_OK, 'second accumulator add succeeds')
      call hv_acc_add(accumulator, c, status)
      call assert_i32_equal(status, HV_OK, 'third accumulator add succeeds')
      bundled = hv_acc_finalize(accumulator, 77_int64)
      expected = hv_bundle3(a, b, c)
      call assert_hv_equal(bundled, expected, 'three-way accumulator equals majority bundle')

      ones%bits = not(0_int64)
      call hv_acc_reset(accumulator)
      accumulator%counts = 32767_int16
      accumulator%sample_count = 1_int32
      accumulator_before = accumulator
      call hv_acc_add(accumulator, ones, status)
      call assert_i32_equal(status, HV_ERR_ACCUMULATOR_OVERFLOW, &
                            'counter overflow is reported')
      call assert_true(all(accumulator%counts == accumulator_before%counts), &
                       'failed accumulator add is atomic')
      call assert_i32_equal(accumulator%sample_count, accumulator_before%sample_count, &
                            'failed accumulator add does not increment sample count')

      call hv_inject_noise_exact(a, 4096_int32, 44_int64, noisy, status)
      call assert_i32_equal(status, HV_OK, 'exact noise injection succeeds')
      call assert_i32_equal(hv_hamming(a, noisy), 4096_int32, 'exact noise flips requested bits')
      call hv_inject_noise_exact(a, 4096_int32, 44_int64, noisy_again, status)
      call assert_hv_equal(noisy, noisy_again, 'noise injection is deterministic')
      call hv_inject_noise_percent(a, 25_int32, 99_int64, noisy_again, status)
      call assert_i32_equal(hv_hamming(a, noisy_again), 4096_int32, &
                            '25 percent maps to 4096 exact flips')
      call hv_inject_noise_exact(a, -1_int32, 1_int64, noisy_again, status)
      call assert_i32_equal(status, HV_ERR_INVALID_ARGUMENT, 'invalid noise count is rejected')
      call assert_hv_equal(noisy_again, a, 'invalid noise leaves output unchanged')

      updated_a = hv_hebbian_update(a, b, 9_int64)
      updated_b = hv_hebbian_update(a, b, 9_int64)
      call assert_hv_equal(updated_a, updated_b, 'Hebbian plasticity mask is deterministic')
      call assert_true(hv_hamming(updated_a, a) > 1000_int32, &
                       'plastic update changes a substantial sparse subset')
   end subroutine test_hypervector_algebra

   subroutine test_symbolic_memory()
      type(hds_memory_t) :: memory, empty_memory, small_memory
      type(cleanup_result_t) :: result
      type(hv_t) :: a, b, c, out_hv, noisy, before, after
      integer(int32) :: status
      logical :: found

      a = hv_from_seed(101_int64)
      b = hv_from_seed(202_int64)
      c = hv_from_seed(303_int64)

      call mem_init(memory, 16_int32, status)
      call assert_i32_equal(status, MEM_OK, 'memory initialization succeeds')
      call assert_true(mem_is_initialized(memory), 'memory reports initialized')
      call assert_true(mem_footprint_bytes(memory) > 0_int64, 'memory footprint is nonzero')

      call mem_register_symbol(memory, 7_int32, 'ALPHA', out_hv, status, a)
      call assert_i32_equal(status, MEM_OK, 'register custom symbol')
      call assert_hv_equal(out_hv, a, 'registration returns stored vector')
      call mem_register_symbol(memory, -1_int32, 'BETA_NEGATIVE_ID', out_hv, status, b)
      call assert_i32_equal(status, MEM_OK, 'negative symbol ID is supported')
      call mem_register_symbol(memory, 19_int32, 'GAMMA', out_hv, status, c)
      call assert_i32_equal(status, MEM_OK, 'third symbol registration succeeds')
      call mem_register_symbol(memory, 7_int32, 'DUPLICATE', out_hv, status)
      call assert_i32_equal(status, MEM_ERR_DUPLICATE_ID, 'duplicate ID is rejected')

      call mem_get_by_id(memory, -1_int32, out_hv, found, status)
      call assert_i32_equal(status, MEM_OK, 'negative ID lookup succeeds')
      call assert_true(found, 'negative ID lookup returns found')
      call assert_hv_equal(out_hv, b, 'negative ID retrieves correct vector')
      call mem_get_by_id(memory, 999_int32, out_hv, found, status)
      call assert_i32_equal(status, MEM_ERR_NOT_FOUND, 'missing ID reports not found')
      call assert_true(.not. found, 'missing ID has found false')

      call mem_cleanup(memory, a, result, status)
      call assert_i32_equal(status, MEM_OK, 'exact cleanup succeeds')
      call assert_i32_equal(result%id, 7_int32, 'exact cleanup returns correct ID')
      call assert_i32_equal(result%distance, 0_int32, 'exact cleanup has zero distance')
      call assert_true(result%exact_search, 'full cleanup marks exact mode')
      call assert_true(trim(result%label) == 'ALPHA', 'cleanup returns stored label')

      call mem_cleanup_lsh(memory, a, result, status)
      call assert_i32_equal(status, MEM_OK, 'LSH cleanup finds exact bucket candidate')
      call assert_i32_equal(result%id, 7_int32, 'LSH cleanup returns exact vector')
      call assert_true(.not. result%exact_search, 'LSH-only result is marked approximate')

      call hv_inject_noise_exact(b, 3000_int32, 888_int64, noisy, status)
      call mem_cleanup(memory, noisy, result, status)
      call assert_i32_equal(status, MEM_OK, 'noisy full cleanup succeeds')
      call assert_i32_equal(result%id, -1_int32, 'noisy cleanup restores target')

      call mem_get_by_id(memory, -1_int32, before, found, status)
      call mem_associate_hebbian(memory, -1_int32, c, status)
      call assert_i32_equal(status, MEM_OK, 'Hebbian memory association succeeds')
      call assert_i32_equal(memory%experience_count(2), 2_int32, 'experience count increments')
      call mem_get_by_id(memory, -1_int32, after, found, status)
      call assert_true(.not. hv_equal(before, after), 'Hebbian association changes prototype')
      call mem_cleanup(memory, after, result, status)
      call assert_i32_equal(result%id, -1_int32, 'rebuilt LSH keeps updated exact retrieval')

      call mem_init(empty_memory, 2_int32, status)
      call mem_cleanup(empty_memory, a, result, status)
      call assert_i32_equal(status, MEM_ERR_EMPTY, 'empty memory cleanup is rejected')
      call mem_destroy(empty_memory)

      call mem_init(small_memory, 1_int32, status)
      call mem_register_symbol(small_memory, 1_int32, 'ONLY', out_hv, status)
      call mem_register_symbol(small_memory, 2_int32, 'OVERFLOW', out_hv, status)
      call assert_i32_equal(status, MEM_ERR_FULL, 'fixed-capacity memory rejects overflow')
      call mem_destroy(small_memory)

      call mem_destroy(memory)
      call assert_true(.not. mem_is_initialized(memory), 'destroy resets initialized state')
   end subroutine test_symbolic_memory

   subroutine test_reasoning_pipeline()
      type(hds_memory_t) :: memory
      type(cleanup_result_t) :: result
      type(hv_t) :: roles(3), source_values(3), target_values(3), unused(2)
      type(hv_t) :: frame, sequence_forward, sequence_reverse, mapping
      type(hv_t) :: causal_values(5), causes(4), effects(4), inferred, next_inferred
      type(hv_t) :: dummy
      integer(int32) :: status, i
      integer(int32), parameter :: EXPECTED_CHAIN(4) = [32_int32, 33_int32, 34_int32, 35_int32]

      call mem_init(memory, 128_int32, status)
      call assert_i32_equal(status, MEM_OK, 'reasoning memory initializes')

      do i = 1, 3
         call mem_register_symbol(memory, i, 'ROLE', roles(i), status)
         call assert_i32_equal(status, MEM_OK, 'role registration succeeds')
         call mem_register_symbol(memory, 10_int32 + i, 'SOURCE', source_values(i), status)
         call assert_i32_equal(status, MEM_OK, 'source registration succeeds')
         call mem_register_symbol(memory, 20_int32 + i, 'TARGET', target_values(i), status)
         call assert_i32_equal(status, MEM_OK, 'target registration succeeds')
      end do
      do i = 1, 5
         call mem_register_symbol(memory, 30_int32 + i, 'CAUSAL', causal_values(i), status)
         call assert_i32_equal(status, MEM_OK, 'causal vocabulary registration succeeds')
      end do
      do i = 100, 149
         call mem_register_symbol(memory, i, 'DISTRACTOR', dummy, status)
         call assert_i32_equal(status, MEM_OK, 'distractor registration succeeds')
      end do

      call encode_role_filler_frame(roles, source_values, frame, status)
      call assert_i32_equal(status, REASON_OK, 'frame encoding succeeds')
      call query_frame_role(memory, frame, roles(2), result, status)
      call assert_i32_equal(status, MEM_OK, 'frame query succeeds')
      call assert_i32_equal(result%id, 12_int32, 'frame unbinding recovers aligned filler')
      call encode_role_filler_frame(roles(1:2), source_values, frame, status)
      call assert_i32_equal(status, REASON_ERR_INVALID_SHAPE, 'mismatched frame arrays are rejected')

      call encode_sequence_ngram(source_values(1:2), sequence_forward, status)
      call assert_i32_equal(status, REASON_OK, 'forward sequence encoding succeeds')
      call encode_sequence_ngram([source_values(2), source_values(1)], sequence_reverse, status)
      call assert_i32_equal(status, REASON_OK, 'reverse sequence encoding succeeds')
      call assert_true(.not. hv_equal(sequence_forward, sequence_reverse), &
                       'permuted n-grams preserve order')
      call encode_sequence_ngram(unused(1:0), sequence_forward, status)
      call assert_i32_equal(status, REASON_ERR_INVALID_SHAPE, 'empty sequence is rejected')

      call encode_analogy_mapping(source_values, target_values, mapping, status)
      call assert_i32_equal(status, REASON_OK, 'aligned analogy mapping succeeds')
      call solve_analogy_zero_shot(memory, mapping, source_values(1), result, status)
      call assert_i32_equal(status, MEM_OK, 'analogy cleanup succeeds')
      call assert_i32_equal(result%id, 21_int32, 'analogy maps source to aligned target')

      causes = causal_values(1:4)
      effects = causal_values(2:5)
      call encode_causal_rule_bank(causes, effects, mapping, status)
      call assert_i32_equal(status, REASON_OK, 'causal bank encoding succeeds')
      inferred = causal_values(1)
      do i = 1, 4
         call infer_causal_step(memory, mapping, inferred, next_inferred, result, status)
         call assert_i32_equal(status, MEM_OK, 'causal inference step succeeds')
         call assert_i32_equal(result%id, EXPECTED_CHAIN(i), 'causal chain selects next state')
         inferred = next_inferred
      end do

      call mem_destroy(memory)
   end subroutine test_reasoning_pipeline

end program test_all
