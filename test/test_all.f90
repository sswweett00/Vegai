! SPDX-License-Identifier: MIT
program test_all
   use, intrinsic :: iso_fortran_env, only: int8, int16, int32, int64, real32, real64
   use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_positive_inf, ieee_value, ieee_is_finite
   use hyper_types
   use symbolic_memory
   use reasoning_core
   use resonator_network
   use hdc_pretraining
   use turkish_text, only: TURKISH_TEXT_OK, TURKISH_TEXT_ERR_CONTROL, turkish_fold_prompt_utf8, &
                           turkish_validate_display_utf8
   use dialogue_retrieval
   use tokenless_field
   use fortran_control, only: FC_OK, FC_ERR_INVALID_ARGUMENT, fortran_resource_plan_t, &
                              fortran_resource_plan_estimate
   use fortran_integrity, only: FI_OK, sha256_file
   use fortran_quality_gate, only: QG_OK, QG_ERR_INVALID_ARGUMENT, QG_ERR_CASE, QG_ERR_REJECTED, &
                                   quality_case_t, quality_policy_t, &
                                   quality_report_t, quality_case_file_load, quality_gate_evaluate
   use text_tokenizer, only: TOKEN_LEN, MAX_LINE_CHARS, MAX_TOKENS_PER_LINE, tokenize_ascii_line
   use subword_bpe
   use neural_retention_decoder
   use calibrated_decoder
   use vocabulary
   use test_support
   implicit none

   call test_hypervector_algebra()
   call test_symbolic_memory()
   call test_reasoning_pipeline()
   call test_resonator_network()
   call test_tokenless_signal_field()
   call test_neural_causal_design()
   call test_text_ingest()
   call test_fortran_control()
   call test_fortran_integrity()
   call test_fortran_quality_gate()
   call test_hdc_pretraining()
   call test_turkish_text_boundary()
   call test_dialogue_retrieval()
   call finish_tests()

contains

   subroutine test_hypervector_algebra()
      type(hv_t) :: a, b, c, d, zero, noisy, noisy_again, bundled, expected
      type(hv_t) :: ones, before, updated_a, updated_b, splitmix_zero
      type(hv_acc_t) :: accumulator, accumulator_before
      type(hv_bitacc_t) :: bit_accumulator, fused_accumulator, fused_before
      type(hv_t) :: bundle_inputs(8)
      integer(int32) :: status, distance, i
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
      call assert_true(abs(hv_similarity_from_distance(HV_BITS / 2_int32)) < 1.0e-6, &
                       'central Hamming-to-bipolar mapping has exact orthogonal midpoint')
      call assert_true(abs(hv_similarity_from_distance(HV_BITS) + 1.0_real32) < 1.0e-6, &
                       'central Hamming-to-bipolar mapping has exact antipodal endpoint')
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

      ! The high-throughput bit-sliced bundle must remain bit-identical to the
      ! checked signed accumulator for both odd majorities and even seeded ties.
      do i = 1, size(bundle_inputs)
         bundle_inputs(i) = hv_from_seed(int(1000 + i, int64))
      end do
      call hv_acc_reset(accumulator)
      call hv_bitacc_reset(bit_accumulator)
      do i = 1, size(bundle_inputs)
         call hv_acc_add(accumulator, bundle_inputs(i), status)
         call assert_i32_equal(status, HV_OK, 'reference accumulator accepts unit bundle input')
         call hv_bitacc_add(bit_accumulator, bundle_inputs(i), status)
         call assert_i32_equal(status, HV_OK, 'bit-sliced accumulator accepts unit bundle input')
         call assert_hv_equal(hv_bitacc_finalize(bit_accumulator, 991_int64), &
                              hv_acc_finalize(accumulator, 991_int64), &
                              'bit-sliced bundle matches signed accumulator at every cardinality')
      end do

      ! A bounded reset intentionally leaves irrelevant high carry planes
      ! untouched.  Growing through several powers of two must lazily clear
      ! each newly needed plane and remain equivalent to the signed oracle.
      ! Reuse bit_accumulator after its prior eight-input state to exercise
      ! stale high-plane initialization rather than a freshly zeroed object.
      call hv_acc_reset(accumulator)
      call hv_bitacc_reset(bit_accumulator, 3_int32)
      do i = 1, size(bundle_inputs)
         call hv_acc_add(accumulator, bundle_inputs(i), status)
         call assert_i32_equal(status, HV_OK, 'bounded-reset reference accumulator accepts input')
         call hv_bitacc_add(bit_accumulator, bundle_inputs(i), status)
         call assert_i32_equal(status, HV_OK, 'bounded-reset bit accumulator accepts input')
         call assert_hv_equal(hv_bitacc_finalize(bit_accumulator, 991_int64), &
                              hv_acc_finalize(accumulator, 991_int64), &
                              'bounded-reset bit accumulator lazily initializes carry planes')
      end do

      ! The fused raw-SoA kernels must preserve every counter plane, not merely
      ! the final bundled vector.  This is the optimization used by fast HDC
      ! pretraining context updates.
      call hv_bitacc_reset(bit_accumulator)
      call hv_bitacc_reset(fused_accumulator)
      call hv_bitacc_add(bit_accumulator, hv_permute(bundle_inputs(1), -97_int32), status)
      call assert_i32_equal(status, HV_OK, 'reference permuted bit-sliced add succeeds')
      call hv_bitacc_add_permuted_words(fused_accumulator, bundle_inputs(1)%bits, -97_int32, status)
      call assert_i32_equal(status, HV_OK, 'fused permuted bit-sliced add succeeds')
      call assert_true(all(fused_accumulator%planes == bit_accumulator%planes), &
                       'fused permuted add preserves every population-counter plane')
      call assert_i32_equal(fused_accumulator%sample_count, bit_accumulator%sample_count, &
                            'fused permuted add preserves sample count')

      call hv_bitacc_reset(bit_accumulator)
      call hv_bitacc_reset(fused_accumulator)
      call hv_bitacc_add(bit_accumulator, hv_permute(bundle_inputs(2), 129_int32), status)
      call assert_i32_equal(status, HV_OK, 'reference first pair add succeeds')
      call hv_bitacc_add(bit_accumulator, bundle_inputs(2), status)
      call assert_i32_equal(status, HV_OK, 'reference second pair add succeeds')
      call hv_bitacc_add_permuted_pair_words(fused_accumulator, bundle_inputs(2)%bits, 129_int32, status)
      call assert_i32_equal(status, HV_OK, 'fused positional-topic pair add succeeds')
      call assert_true(all(fused_accumulator%planes == bit_accumulator%planes), &
                       'fused pair add preserves every population-counter plane')
      call assert_i32_equal(fused_accumulator%sample_count, bit_accumulator%sample_count, &
                            'fused pair add preserves sample count')

      ! Learned semantic feedback uses a different topical source than the
      ! immutable position identity.  Its mixed fused kernel must be identical
      ! to the two explicit adds before it can be used in an epoch snapshot.
      call hv_bitacc_reset(bit_accumulator)
      call hv_bitacc_reset(fused_accumulator)
      call hv_bitacc_add(bit_accumulator, hv_permute(bundle_inputs(3), -191_int32), status)
      call assert_i32_equal(status, HV_OK, 'reference mixed pair position add succeeds')
      call hv_bitacc_add(bit_accumulator, bundle_inputs(4), status)
      call assert_i32_equal(status, HV_OK, 'reference mixed pair semantic add succeeds')
      call hv_bitacc_add_permuted_mixed_pair_words(fused_accumulator, bundle_inputs(3)%bits, &
                                                   bundle_inputs(4)%bits, -191_int32, status)
      call assert_i32_equal(status, HV_OK, 'fused mixed positional-semantic pair add succeeds')
      call assert_true(all(fused_accumulator%planes == bit_accumulator%planes), &
                       'fused mixed pair add preserves every population-counter plane')
      call assert_i32_equal(fused_accumulator%sample_count, bit_accumulator%sample_count, &
                            'fused mixed pair add preserves sample count')
      fused_accumulator%sample_count = HV_BITACC_MAX_ITEMS - 1_int32
      fused_before = fused_accumulator
      call hv_bitacc_add_permuted_pair_words(fused_accumulator, bundle_inputs(2)%bits, 1_int32, status)
      call assert_i32_equal(status, HV_ERR_ACCUMULATOR_OVERFLOW, 'fused pair preflight reports capacity overflow')
      call assert_true(all(fused_accumulator%planes == fused_before%planes) .and. &
                       fused_accumulator%sample_count == fused_before%sample_count, &
                       'failed fused pair add is atomic')

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
      ! High rates exercise the complement sampler: it must retain exact
      ! cardinality without the old deterministic low-index completion bias.
      call hv_inject_noise_percent(a, 99_int32, 123_int64, noisy_again, status)
      call assert_i32_equal(status, HV_OK, '99 percent complement noise selection succeeds')
      call assert_i32_equal(hv_hamming(a, noisy_again), 16220_int32, &
                            '99 percent complement sampler preserves exact rounded cardinality')
      call hv_inject_noise_percent(a, 99_int32, 123_int64, noisy, status)
      call assert_i32_equal(status, HV_OK, 'repeated 99 percent complement selection succeeds')
      call assert_hv_equal(noisy, noisy_again, 'high-density complement sampler is seed-deterministic')
      call hv_inject_noise_percent(a, 100_int32, 123_int64, noisy, status)
      call assert_hv_equal(noisy, hv_bind(a, ones), '100 percent exact noise complements every dimension')
      call hv_inject_noise_exact(a, -1_int32, 1_int64, noisy_again, status)
      call assert_i32_equal(status, HV_ERR_INVALID_ARGUMENT, 'invalid noise count is rejected')
      call assert_hv_equal(noisy_again, a, 'invalid noise leaves output unchanged')

      updated_a = hv_hebbian_update(a, b, 9_int64)
      updated_b = hv_hebbian_update(a, b, 9_int64)
      call assert_hv_equal(updated_a, updated_b, 'Hebbian plasticity mask is deterministic')
      call assert_true(hv_hamming(updated_a, a) > 1000_int32, &
                       'plastic update changes a substantial sparse subset')

      ones%bits = not(a%bits)
      call hv_hebbian_update_percent(a, ones, 25_int32, 31_int64, updated_a, status)
      call assert_i32_equal(status, HV_OK, 'exact-rate Hebbian update succeeds')
      call assert_i32_equal(hv_hamming(a, updated_a), 4096_int32, &
                            'exact-rate update selects exactly 25 percent when inputs differ')
      call hv_hebbian_update_percent(a, ones, 101_int32, 31_int64, updated_b, status)
      call assert_i32_equal(status, HV_ERR_INVALID_ARGUMENT, 'invalid exact-rate update is rejected')
      call assert_hv_equal(updated_b, a, 'invalid exact-rate update preserves prototype output')
   end subroutine test_hypervector_algebra

   subroutine test_symbolic_memory()
      type(hds_memory_t) :: memory, empty_memory, small_memory, tie_memory
      type(cleanup_result_t) :: result
      type(hv_t) :: a, b, c, out_hv, noisy, before, after, tie_a, tie_b, tie_query
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

      ! Two candidates at the same nonzero Hamming distance must select the
      ! lowest registration slot. This exercises the one-pass lexicographic
      ! OpenMP reduction rather than relying on a serial second distance scan.
      tie_a = hv_zero()
      tie_b = hv_zero()
      tie_b%bits(1) = ibset(tie_b%bits(1), 0)
      tie_b%bits(1) = ibset(tie_b%bits(1), 1)
      tie_query = hv_zero()
      tie_query%bits(1) = ibset(tie_query%bits(1), 0)
      call mem_init(tie_memory, 2_int32, status)
      call assert_i32_equal(status, MEM_OK, 'tie-break memory initializes')
      call mem_register_symbol(tie_memory, 501_int32, 'TIE:LOW', out_hv, status, tie_a)
      call assert_i32_equal(status, MEM_OK, 'first tied candidate registers')
      call mem_register_symbol(tie_memory, 502_int32, 'TIE:HIGH', out_hv, status, tie_b)
      call assert_i32_equal(status, MEM_OK, 'second tied candidate registers')
      call mem_cleanup(tie_memory, tie_query, result, status)
      call assert_i32_equal(status, MEM_OK, 'nonzero tied cleanup succeeds')
      call assert_i32_equal(result%distance, 1_int32, 'tied cleanup records the exact Hamming distance')
      call assert_i32_equal(result%id, 501_int32, 'tied cleanup deterministically selects the lower slot')
      call mem_destroy(tie_memory)

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

   subroutine test_resonator_network()
      type(hds_memory_t) :: codebooks(3), ambiguous_books(3), invalid_books(2)
      type(resonator_config_t) :: config, invalid_config
      type(resonator_result_t) :: result, ambiguous_result
      type(hv_t) :: a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4
      type(hv_t) :: duplicate_a, duplicate_b, duplicate_c, target, reconstructed, dummy
      integer(int32) :: status, i

      do i = 1, 3
         call mem_init(codebooks(i), 4_int32, status)
         call assert_i32_equal(status, MEM_OK, 'resonator codebook initializes')
      end do

      call mem_register_symbol(codebooks(1), 1001_int32, 'A:ONE', a1, status)
      call mem_register_symbol(codebooks(1), 1002_int32, 'A:TWO', a2, status)
      call mem_register_symbol(codebooks(1), 1003_int32, 'A:THREE', a3, status)
      call mem_register_symbol(codebooks(1), 1004_int32, 'A:FOUR', a4, status)
      call mem_register_symbol(codebooks(2), 2001_int32, 'B:ONE', b1, status)
      call mem_register_symbol(codebooks(2), 2002_int32, 'B:TWO', b2, status)
      call mem_register_symbol(codebooks(2), 2003_int32, 'B:THREE', b3, status)
      call mem_register_symbol(codebooks(2), 2004_int32, 'B:FOUR', b4, status)
      call mem_register_symbol(codebooks(3), 3001_int32, 'C:ONE', c1, status)
      call mem_register_symbol(codebooks(3), 3002_int32, 'C:TWO', c2, status)
      call mem_register_symbol(codebooks(3), 3003_int32, 'C:THREE', c3, status)
      call mem_register_symbol(codebooks(3), 3004_int32, 'C:FOUR', c4, status)
      call assert_i32_equal(status, MEM_OK, 'resonator candidate registration succeeds')

      target = hv_bind(hv_bind(a2, b3), c4)
      config%max_iterations = 12_int32
      config%restarts = 4_int32
      config%max_exact_combinations = 64_int32
      config%synchronous_updates = .true.
      config%enable_exact_verifier = .true.
      call resonator_factorize(target, codebooks, config, result, status)
      call assert_i32_equal(status, RN_OK, 'resonator exact verification succeeds')
      call assert_true(result%exact, 'resonator result is exact')
      call assert_true(result%used_exact_verifier, 'bounded verifier certifies factorization')
      call assert_i32_equal(result%hamming_distance, 0_int32, 'certified result has zero residual')
      call assert_i32_equal(result%factor_ids(1), 1002_int32, 'resonator restores first factor')
      call assert_i32_equal(result%factor_ids(2), 2003_int32, 'resonator restores second factor')
      call assert_i32_equal(result%factor_ids(3), 3004_int32, 'resonator restores third factor')
      reconstructed = hv_bind(hv_bind(result%factors(1), result%factors(2)), result%factors(3))
      call assert_hv_equal(reconstructed, target, 'reported factors rebind to target')
      call resonator_result_destroy(result)
      call assert_true(.not. allocated(result%factors), 'resonator result destroy releases workspace')

      ! Exercise the maintained-composite Gauss-Seidel path separately from the
      ! synchronous path above. The exact verifier remains an independent
      ! oracle for the returned tuple, not a substitute for residual algebra.
      config%synchronous_updates = .false.
      call resonator_factorize(target, codebooks, config, result, status)
      call assert_i32_equal(status, RN_OK, 'asynchronous resonator factorization succeeds')
      call assert_true(result%exact, 'asynchronous resonator result is exact')
      reconstructed = hv_bind(hv_bind(result%factors(1), result%factors(2)), result%factors(3))
      call assert_hv_equal(reconstructed, target, 'asynchronous reported factors rebind to target')
      call resonator_result_destroy(result)
      config%synchronous_updates = .true.

      invalid_config = config
      invalid_config%max_exact_combinations = 0_int32
      call resonator_factorize(target, codebooks, invalid_config, result, status)
      call assert_i32_equal(status, RN_ERR_INVALID_ARGUMENT, 'invalid verifier bound is rejected')
      call resonator_factorize(target, invalid_books, config, result, status)
      call assert_i32_equal(status, RN_ERR_CODEBOOK, 'uninitialized factor codebook is rejected')

      do i = 1, 3
         call mem_init(ambiguous_books(i), 2_int32, status)
         call assert_i32_equal(status, MEM_OK, 'ambiguous codebook initializes')
      end do
      duplicate_a = hv_from_seed(777_int64)
      duplicate_b = hv_from_seed(888_int64)
      duplicate_c = hv_from_seed(999_int64)
      call mem_register_symbol(ambiguous_books(1), 9001_int32, 'DUPLICATE:A1', dummy, status, duplicate_a)
      call mem_register_symbol(ambiguous_books(1), 9002_int32, 'DUPLICATE:A2', dummy, status, duplicate_a)
      call mem_register_symbol(ambiguous_books(2), 9101_int32, 'UNIQUE:B', dummy, status, duplicate_b)
      call mem_register_symbol(ambiguous_books(3), 9201_int32, 'UNIQUE:C', dummy, status, duplicate_c)
      call assert_i32_equal(status, MEM_OK, 'ambiguous candidates register successfully')
      target = hv_bind(hv_bind(duplicate_a, duplicate_b), duplicate_c)
      config%max_exact_combinations = 8_int32
      call resonator_factorize(target, ambiguous_books, config, ambiguous_result, status)
      call assert_i32_equal(status, RN_WARN_AMBIGUOUS, 'verifier reports duplicate exact factorization')
      call assert_true(ambiguous_result%ambiguous, 'ambiguous result metadata is set')
      call assert_i32_equal(ambiguous_result%exact_solution_count, 2_int32, &
                            'ambiguity count is capped at two or more')
      call resonator_result_destroy(ambiguous_result)

      do i = 1, 3
         call mem_destroy(codebooks(i))
         call mem_destroy(ambiguous_books(i))
      end do
   end subroutine test_resonator_network

   subroutine test_tokenless_signal_field()
      integer(int8), parameter :: RAW_BYTES(8) = [ &
         0_int8, -1_int8, 1_int8, int(z'80', int8), 64_int8, -64_int8, 10_int8, 0_int8 ]
      integer(int8), parameter :: REORDERED_BYTES(8) = [ &
         0_int8, 10_int8, -64_int8, 64_int8, int(z'80', int8), 1_int8, -1_int8, 0_int8 ]
      integer(int8) :: empty_bytes(0)
      integer(int8), allocatable :: too_large_bytes(:)
      integer(int64) :: spectrum(HV_BITS), invalid_spectrum(HV_BITS)
      integer(int64) :: spectral_energy, residual
      integer(int32) :: status
      type(field_encoding_t) :: encoding, same_encoding, reordered_encoding
      type(hv_t) :: reconstructed, negative_field, positive_field

      ! This path handles raw octets directly.  Values below cover signed
      ! Fortran int8 representations of 0, 255, 128, and other byte values;
      ! it is intentionally independent of the ASCII text tokenizer.
      call field_encode_bytes(RAW_BYTES, encoding, status)
      call assert_i32_equal(status, FIELD_OK, 'raw byte-event field accepts arbitrary octets')
      call assert_i32_equal(encoding%byte_count, 8_int32, 'byte-event field reports exact bounded input size')
      call field_encode_bytes(RAW_BYTES, same_encoding, status)
      call assert_i32_equal(status, FIELD_OK, 'same raw byte stream re-encodes')
      call assert_hv_equal(encoding%order_field, same_encoding%order_field, &
                           'byte-event order field is deterministic')
      call assert_hv_equal(encoding%distribution_field, same_encoding%distribution_field, &
                           'byte-event distribution field is deterministic')

      call field_encode_bytes(REORDERED_BYTES, reordered_encoding, status)
      call assert_i32_equal(status, FIELD_OK, 'reordered raw byte stream encodes')
      call assert_hv_equal(encoding%distribution_field, reordered_encoding%distribution_field, &
                           'distribution field is explicitly order-insensitive')
      call assert_true(.not. hv_equal(encoding%order_field, reordered_encoding%order_field), &
                       'position-sensitive raw-byte field changes for reordered events')
      call assert_true(field_fingerprint(encoding%order_field) /= field_fingerprint(reordered_encoding%order_field), &
                       'rolling diagnostic fingerprint distinguishes the tested ordering')

      call field_walsh_forward(encoding%order_field, spectrum, status)
      call assert_i32_equal(status, FIELD_OK, 'Walsh forward transform accepts packed bipolar field')
      call field_spectral_energy(spectrum, spectral_energy, status)
      call assert_i32_equal(status, FIELD_OK, 'bounded Walsh spectrum has safe exact energy')
      call assert_true(spectral_energy == FIELD_WALSH_EXPECTED_ENERGY, &
                       'Walsh Parseval energy is exactly dimension squared')
      call field_parseval_residual(spectrum, residual, status)
      call assert_i32_equal(status, FIELD_OK, 'Parseval residual evaluates valid spectrum')
      call assert_true(residual == 0_int64, 'Walsh Parseval residual is exactly zero')
      call assert_true(field_bipolar_energy(encoding%order_field) == int(HV_BITS, int64), &
                       'packed field has exact bipolar energy equal to dimension')
      call field_walsh_inverse(spectrum, reconstructed, status)
      call assert_i32_equal(status, FIELD_OK, 'valid bounded Walsh spectrum has exact inverse')
      call assert_hv_equal(reconstructed, encoding%order_field, 'Walsh inverse reconstructs every field bit')

      ! The two constant bipolar fields exercise the exact DC Walsh column:
      ! H times +1 is [D, 0, ...], and H times -1 is [-D, 0, ...].
      positive_field%bits = not(0_int64)
      call field_walsh_forward(positive_field, spectrum, status)
      call assert_i32_equal(status, FIELD_OK, 'positive constant field transforms')
      call assert_true(spectrum(1) == int(HV_BITS, int64) .and. all(spectrum(2:) == 0_int64), &
                       'positive constant field has exact Walsh DC spectrum')
      call field_walsh_inverse(spectrum, reconstructed, status)
      call assert_i32_equal(status, FIELD_OK, 'positive constant spectrum inverts')
      call assert_hv_equal(reconstructed, positive_field, 'positive constant Walsh inverse is exact')
      negative_field = hv_zero()
      call field_walsh_forward(negative_field, spectrum, status)
      call assert_i32_equal(status, FIELD_OK, 'negative constant field transforms')
      call assert_true(spectrum(1) == -int(HV_BITS, int64) .and. all(spectrum(2:) == 0_int64), &
                       'negative constant field has exact Walsh DC spectrum')
      call field_walsh_inverse(spectrum, reconstructed, status)
      call assert_i32_equal(status, FIELD_OK, 'negative constant spectrum inverts')
      call assert_hv_equal(reconstructed, negative_field, 'negative constant Walsh inverse is exact')

      invalid_spectrum = spectrum
      invalid_spectrum(1) = int(HV_BITS, int64) + 1_int64
      call field_spectral_energy(invalid_spectrum, spectral_energy, status)
      call assert_i32_equal(status, FIELD_ERR_INVALID_SPECTRUM, 'out-of-range spectrum is fail-closed')
      call field_walsh_inverse(invalid_spectrum, reconstructed, status)
      call assert_i32_equal(status, FIELD_ERR_INVALID_SPECTRUM, 'inverse rejects out-of-range spectrum')
      call assert_hv_equal(reconstructed, hv_zero(), 'rejected spectrum publishes no partial field')
      invalid_spectrum = 0_int64
      call field_walsh_inverse(invalid_spectrum, reconstructed, status)
      call assert_i32_equal(status, FIELD_ERR_INVALID_SPECTRUM, &
                            'inverse rejects bounded coefficients without bipolar reconstruction')

      call field_encode_bytes(empty_bytes, encoding, status)
      call assert_i32_equal(status, FIELD_ERR_EMPTY_INPUT, 'empty byte stream is rejected explicitly')
      allocate(too_large_bytes(FIELD_MAX_BYTES + 1_int32))
      too_large_bytes = 0_int8
      call field_encode_bytes(too_large_bytes, encoding, status)
      call assert_i32_equal(status, FIELD_ERR_CAPACITY, 'oversized byte stream is rejected before accumulation')
      deallocate(too_large_bytes)
   end subroutine test_tokenless_signal_field

   subroutine test_neural_causal_design()
      type(bpe_model_t) :: bpe
      type(neural_decoder_t) :: model
      type(neural_decoder_state_t) :: full_state, chunked_state
      type(neural_decoder_config_t) :: config
      type(neural_resource_plan_t) :: plan
      type(decoding_policy_t) :: policy
      type(temperature_calibration_t) :: calibration
      character(len=64) :: source, decoded
      integer(int32) :: encoded(64), encoded_count, decoded_bytes, status, merged, index
      integer(int32) :: history(8), selected, targets(4)
      integer(int64) :: random_state
      real(real32) :: hidden_full(16), hidden_chunked(16), logits_full(64), logits_chunked(64)
      real(real64) :: raw_logits(4), probabilities(4), heldout_logits(3, 4)

      source = ''
      source = 'Türkçe'
      call bpe_init(bpe, 8_int32, status)
      call assert_i32_equal(status, BPE_OK, 'byte-BPE model initializes')
      call bpe_add_merge(bpe, bpe_base_token_for_byte(int(iachar(source(1:1)), int32)), &
                         bpe_base_token_for_byte(int(iachar(source(2:2)), int32)), merged, status)
      call assert_i32_equal(status, BPE_OK, 'byte-BPE accepts first UTF-8 merge')
      call bpe_add_merge(bpe, merged, bpe_base_token_for_byte(int(iachar(source(3:3)), int32)), merged, status)
      call assert_i32_equal(status, BPE_OK, 'byte-BPE accepts chained UTF-8 merge')
      call bpe_encode(bpe, source, int(len_trim(source), int32), encoded, encoded_count, status)
      call assert_i32_equal(status, BPE_OK, 'byte-BPE encodes Turkish UTF-8 bytes')
      call bpe_decode(bpe, encoded, encoded_count, decoded, decoded_bytes, status)
      call assert_i32_equal(status, BPE_OK, 'byte-BPE decodes Turkish UTF-8 bytes')
      if (decoded_bytes > 0_int32) then
         call assert_true(decoded_bytes == len_trim(source) .and. decoded(:decoded_bytes) == source(:decoded_bytes), &
                          'byte-BPE round trip is byte-lossless')
      else
         call assert_true(.false., 'byte-BPE output unexpectedly empty')
      end if

      config = neural_decoder_config_t(vocabulary_size=64_int32, width=16_int32, heads=2_int32, &
                                      local_window=4_int32, ffn_width=32_int32)
      call neural_decoder_init(model, config, 20261025_int64, status)
      call assert_i32_equal(status, NRD_OK, 'hybrid neural retention decoder initializes')
      call neural_decoder_state_init(full_state, model, status)
      call assert_i32_equal(status, NRD_OK, 'full-prefix neural state initializes')
      call neural_decoder_state_init(chunked_state, model, status)
      call assert_i32_equal(status, NRD_OK, 'chunked neural state initializes')
      do index = 1_int32, 4_int32
         call neural_decoder_step(model, full_state, index, hidden_full, logits_full, status)
         call assert_i32_equal(status, NRD_OK, 'full-prefix causal neural step succeeds')
      end do
      do index = 1_int32, 2_int32
         call neural_decoder_step(model, chunked_state, index, hidden_chunked, logits_chunked, status)
         call assert_i32_equal(status, NRD_OK, 'first causal stream chunk succeeds')
      end do
      do index = 3_int32, 4_int32
         call neural_decoder_step(model, chunked_state, index, hidden_chunked, logits_chunked, status)
         call assert_i32_equal(status, NRD_OK, 'second causal stream chunk succeeds')
      end do
      call assert_true(maxval(abs(full_state%retention_numerator - chunked_state%retention_numerator)) < 1.0e-6_real32, &
                       'retention state is exactly prefix/chunk invariant')
      call assert_true(maxval(abs(logits_full - logits_chunked)) < 1.0e-6_real32, &
                       'causal logits are invariant to chunk boundaries')
      call assert_true(all(ieee_is_finite(full_state%retention_numerator)) .and. all(ieee_is_finite(logits_full)), &
                       'positive-feature retention state and logits remain finite')
      call neural_decoder_plan(config, plan, status)
      call assert_i32_equal(status, NRD_OK, 'neural resource planner succeeds')
      call assert_true(plan%parameter_count > 0_int64 .and. plan%per_stream_state_bytes_real32 > 0_int64 .and. &
                       plan%approximate_forward_flops_per_token > 0_int64, &
                       'neural resource plan exposes finite positive costs')

      raw_logits = [7.0_real64, 2.0_real64, 1.0_real64, 0.0_real64]
      history = [1_int32, 2_int32, 1_int32, 2_int32, 0_int32, 0_int32, 0_int32, 0_int32]
      policy = decoding_policy_t(temperature=1.0_real64, frequency_penalty=0.0_real64, presence_penalty=0.0_real64, &
                                 immediate_repeat_penalty=0.0_real64, no_repeat_ngram=3_int32, top_k=4_int32, top_p=1.0_real64)
      call decoder_distribution(raw_logits, history, 4_int32, policy, probabilities, status)
      call assert_i32_equal(status, DECODER_OK, 'calibrated decoder distribution succeeds')
      call assert_true(abs(sum(probabilities) - 1.0_real64) < 1.0e-12_real64 .and. probabilities(1) == 0.0_real64, &
                       'decoder renormalizes after no-repeat mask')
      random_state = 91_int64
      call decoder_select(probabilities, policy, random_state, selected, status)
      call assert_i32_equal(status, DECODER_OK, 'calibrated decoder selection succeeds')
      call assert_true(selected /= 1_int32, 'repetition-aware decoder cannot select masked repeated ngram')

      heldout_logits(:, 1) = [2.0_real64, 0.0_real64, -1.0_real64]
      heldout_logits(:, 2) = [1.5_real64, 0.0_real64, -1.0_real64]
      heldout_logits(:, 3) = [0.5_real64, 0.0_real64, -1.0_real64]
      heldout_logits(:, 4) = [1.0_real64, 0.0_real64, -1.0_real64]
      targets = [1_int32, 1_int32, 2_int32, 1_int32]
      call decoder_temperature_calibrate(heldout_logits, targets, 4_int32, calibration, status)
      call assert_i32_equal(status, DECODER_OK, 'convex inverse-temperature calibration succeeds')
      call assert_true(calibration%mean_nll_after <= calibration%mean_nll_before + 1.0e-12_real64, &
                       'temperature calibration retains non-worse held-out NLL')

      call bpe_destroy(bpe)
      call neural_decoder_state_destroy(full_state)
      call neural_decoder_state_destroy(chunked_state)
      call neural_decoder_destroy(model)
   end subroutine test_neural_causal_design

   subroutine test_text_ingest()
      type(vocabulary_t) :: vocab
      character(len=TOKEN_LEN) :: tokens(MAX_TOKENS_PER_LINE)
      integer(int32) :: n_tokens, status, slot
      integer(int64) :: fingerprint
      logical :: found, truncated

      call tokenize_ascii_line("HELLO, can't; _id42!", tokens, n_tokens, truncated)
      call assert_i32_equal(n_tokens, 3_int32, 'ASCII tokenizer emits expected token count')
      call assert_true(.not. truncated, 'ordinary ASCII tokenization has no truncation')
      call assert_true(trim(tokens(1)) == 'hello', 'ASCII tokenizer lowercases letters')
      call assert_true(trim(tokens(2)) == "can't", 'ASCII tokenizer keeps internal apostrophe')
      call assert_true(trim(tokens(3)) == '_id42', 'ASCII tokenizer keeps underscore and digits')
      call tokenize_ascii_line(repeat('z', TOKEN_LEN + 1_int32)//' valid', tokens, n_tokens, truncated)
      call assert_i32_equal(n_tokens, 1_int32, 'overlong token is dropped rather than truncated')
      call assert_true(truncated .and. trim(tokens(1)) == 'valid', &
                       'overlong token reports telemetry while preserving later token')

      call vocab_init(vocab, 4_int32, 2_int64, status)
      call assert_i32_equal(status, VOCAB_OK, 'fixed vocabulary initializes')
      call vocab_observe(vocab, 'alpha', status)
      call vocab_observe(vocab, 'beta', status)
      call vocab_observe(vocab, 'alpha', status)
      call vocab_observe(vocab, 'gamma', status)
      call assert_i32_equal(status, VOCAB_OK, 'vocabulary observes tokens and frequencies')
      call vocab_finalize(vocab, status)
      call assert_i32_equal(status, VOCAB_OK, 'vocabulary frequency finalization succeeds')
      call assert_i32_equal(vocab%active_count, 1_int32, 'only thresholded token is active')
      call vocab_lookup(vocab, 'alpha', slot, found, status, .true.)
      call assert_i32_equal(status, VOCAB_OK, 'active vocabulary lookup succeeds')
      call assert_true(found .and. slot == 1_int32, 'active lookup returns first-seen slot')
      call vocab_lookup(vocab, 'beta', slot, found, status, .true.)
      call assert_i32_equal(status, VOCAB_ERR_NOT_FOUND, 'inactive token is excluded from active lookup')
      call vocab_rebuild_index(vocab, status)
      call assert_i32_equal(status, VOCAB_OK, 'vocabulary hash index rebuild succeeds')
      call vocab_lookup(vocab, 'alpha', slot, found, status, .true.)
      call assert_i32_equal(status, VOCAB_OK, 'rebuilt vocabulary index preserves lookup')
      fingerprint = vocab_token_fingerprint('alpha')
      call vocab_lookup_fingerprint(vocab, 'not-present', fingerprint, slot, found, status, .true.)
      call assert_i32_equal(status, VOCAB_ERR_NOT_FOUND, &
                            'precomputed fingerprint still requires exact padded-token equality')
      call assert_true(.not. found, 'fingerprint lookup cannot merge an unrelated token')
      call vocab_destroy(vocab)
   end subroutine test_text_ingest

   subroutine test_fortran_control()
      type(fortran_resource_plan_t) :: plan
      integer(int32) :: status

      call fortran_resource_plan_estimate(128_int64, 0.25_real64, 2_int32, plan, status)
      call assert_i32_equal(status, FC_OK, 'native Fortran resource planner accepts bounded request')
      call assert_true(plan%model_payload_bytes == 128_int64 * 4104_int64, &
                       'native Fortran resource planner keeps exact packed model payload')
      call assert_true(plan%vocabulary_allowance_bytes == 128_int64 * 256_int64, &
                       'native Fortran resource planner keeps vocabulary allowance')
      call assert_true(plan%line_update_scratch_bytes == 128_int64 * 4_int64, &
                       'native Fortran resource planner keeps source-order scratch estimate')
      call assert_true(plan%lookup_cache_workspace_bytes == 278528_int64, &
                       'native Fortran resource planner keeps bounded cache estimate')
      call assert_true(plan%required_available_bytes > plan%model_payload_bytes, &
                       'native Fortran resource planner includes staging and reserve')
      call assert_i32_equal(plan%recommended_omp_threads, 2_int32, &
                            'native Fortran resource planner preserves explicit thread cap')

      call fortran_resource_plan_estimate(128_int64, 0.25_real64, 2_int32, plan, status, &
                                          context_mode=PT_CONTEXT_RESONANT)
      call assert_i32_equal(status, FC_OK, 'resource planner accepts the resonant feedback mode')
      call assert_true(plan%resonant_snapshot_bytes == 128_int64 * 2048_int64, &
                       'resource planner reserves one frozen semantic snapshot per resonant slot')
      call assert_true(plan%required_available_bytes > 128_int64 * 4104_int64, &
                       'resonant preflight includes snapshot, staging, and reserve rather than model state alone')

      call fortran_resource_plan_estimate(0_int64, 0.25_real64, 1_int32, plan, status)
      call assert_i32_equal(status, FC_ERR_INVALID_ARGUMENT, &
                            'native Fortran resource planner rejects zero vocabulary capacity')
      call fortran_resource_plan_estimate(128_int64, -0.01_real64, 1_int32, plan, status)
      call assert_i32_equal(status, FC_ERR_INVALID_ARGUMENT, &
                            'native Fortran resource planner rejects negative reserve fraction')
      call fortran_resource_plan_estimate(128_int64, ieee_value(0.0_real64, ieee_quiet_nan), 1_int32, plan, status)
      call assert_i32_equal(status, FC_ERR_INVALID_ARGUMENT, &
                            'native Fortran resource planner rejects nonfinite reserve fraction')
      call fortran_resource_plan_estimate(100000001_int64, 0.25_real64, 1_int32, plan, status)
      call assert_i32_equal(status, FC_ERR_INVALID_ARGUMENT, &
                            'native Fortran resource planner rejects overflowing capacity request')
   end subroutine test_fortran_control

   subroutine test_fortran_integrity()
      character(len=*), parameter :: PAYLOAD_PATH = 'vegai_fortran_integrity_test.bin'
      character(len=64) :: digest
      integer(int64) :: byte_count
      integer(int32) :: status
      integer :: unit, io_status

      open(newunit=unit, file=PAYLOAD_PATH, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      call assert_true(io_status == 0, 'native Fortran SHA-256 fixture opens')
      if (io_status /= 0) return
      write(unit) 'abc'
      close(unit)
      call sha256_file(PAYLOAD_PATH, digest, byte_count, status)
      call assert_i32_equal(status, FI_OK, 'native Fortran SHA-256 streams fixture bytes')
      call assert_true(byte_count == 3_int64, 'native Fortran SHA-256 reports exact stream byte count')
      call assert_true(trim(digest) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad', &
                       'native Fortran SHA-256 matches the FIPS abc vector')
      open(newunit=unit, file=PAYLOAD_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
   end subroutine test_fortran_integrity

   subroutine test_fortran_quality_gate()
      character(len=*), parameter :: CORPUS_PATH = 'vegai_quality_gate_test_corpus.txt'
      character(len=*), parameter :: CASES_PATH = 'vegai_quality_gate_test_cases.tsv'
      type(hdc_pretrain_model_t) :: model
      type(hdc_pretrain_config_t) :: config
      type(pretrain_metrics_t) :: vocabulary_metrics, training_metrics
      type(quality_case_t), allocatable :: cases(:)
      type(quality_policy_t) :: policy
      type(quality_report_t) :: report
      type(hv_t) :: query_hv
      character(len=TOKEN_LEN) :: neighbour_token
      integer(int32) :: status, query_slot, neighbour_slot
      integer :: unit, io_status
      logical :: found
      real(real32) :: similarity

      open(newunit=unit, file=CORPUS_PATH, status='replace', action='write', form='formatted', iostat=io_status)
      call assert_true(io_status == 0, 'native quality-gate corpus fixture opens')
      if (io_status /= 0) return
      write(unit, '(A)') 'alpha beta gamma'
      write(unit, '(A)') 'alpha beta gamma'
      write(unit, '(A)') 'alpha beta gamma'
      write(unit, '(A)') 'delta epsilon zeta'
      write(unit, '(A)') 'delta epsilon zeta'
      close(unit)

      found = .false.
      neighbour_token = ''
      config = hdc_pretrain_config_t()
      config%max_vocabulary = 16_int32
      config%min_frequency = 2_int64
      config%context_window = 2_int32
      config%position_shift_bits = 17_int32
      config%epochs = 1_int32
      config%strict_tokenization = .true.
      call pretrain_build_vocabulary(model, CORPUS_PATH, config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'native quality-gate vocabulary fixture builds')
      if (status == PT_OK) then
         call pretrain_train_file(model, CORPUS_PATH, config, training_metrics, status)
         call assert_i32_equal(status, PT_OK, 'native quality-gate training fixture completes')
      end if
      if (status == PT_OK) then
         call pretrain_lookup_token(model, 'alpha', query_slot, found, status)
         call assert_i32_equal(status, PT_OK, 'native quality-gate query token resolves')
         call assert_true(found, 'native quality-gate query token is active')
      end if
      if (status == PT_OK .and. found) then
         call pretrain_get_semantic_hv(model, query_slot, query_hv, status)
         call assert_i32_equal(status, PT_OK, 'native quality-gate retrieves query semantic vector')
      end if
      if (status == PT_OK .and. found) then
         call pretrain_nearest_semantic(model, query_hv, neighbour_slot, neighbour_token, similarity, status, &
                                        exclude_slot=query_slot)
         call assert_i32_equal(status, PT_OK, 'native quality-gate derives self-excluded expected neighbour')
      end if
      if (status == PT_OK .and. len_trim(neighbour_token) > 0) then
         open(newunit=unit, file=CASES_PATH, status='replace', action='write', form='formatted', iostat=io_status)
         call assert_true(io_status == 0, 'native quality-gate TSV case fixture opens')
         if (io_status == 0) then
            write(unit, '(A)') 'alpha' // achar(9) // trim(neighbour_token)
            close(unit)
            call quality_case_file_load(CASES_PATH, cases, status)
            call assert_i32_equal(status, QG_OK, 'native quality-gate loads strict TSV case')
         end if
      end if
      if (allocated(cases)) then
         policy = quality_policy_t()
         policy%minimum_cases = 1_int32
         policy%minimum_top1_accuracy = 1.0_real64
         policy%minimum_top1_wilson_lower_bound_95 = 0.0_real64
         policy%minimum_mean_similarity = -1.0_real64
         policy%minimum_case_similarity = -1.0_real64
         policy%maximum_incorrect_cases = 0_int32
         call quality_gate_evaluate(model, cases, policy, report, status)
         call assert_i32_equal(status, QG_OK, 'native quality-gate accepts matching self-excluded case')
         call assert_true(report%passed .and. report%correct_cases == 1_int32, &
                          'native quality-gate publishes a passed aggregate only for matching case')

         policy%minimum_top1_accuracy = ieee_value(0.0_real64, ieee_quiet_nan)
         call quality_gate_evaluate(model, cases, policy, report, status)
         call assert_i32_equal(status, QG_ERR_INVALID_ARGUMENT, &
                               'native quality-gate rejects nonfinite policy thresholds')
         policy%minimum_top1_accuracy = 1.0_real64
         cases(1)%expected_neighbour = 'omega'
         call quality_gate_evaluate(model, cases, policy, report, status)
         call assert_i32_equal(status, QG_ERR_REJECTED, 'native quality-gate rejects incorrect expected neighbour')
         call assert_true(.not. report%passed .and. report%incorrect_cases == 1_int32, &
                          'native quality-gate fail-closes incorrect neighbour expectation')
         deallocate(cases)
         open(newunit=unit, file=CASES_PATH, status='replace', action='write', form='formatted', iostat=io_status)
         call assert_true(io_status == 0, 'native quality-gate malformed TSV fixture opens')
         if (io_status == 0) then
            write(unit, '(A)') 'malformed case without tab'
            close(unit)
            call quality_case_file_load(CASES_PATH, cases, status)
            call assert_i32_equal(status, QG_ERR_CASE, 'native quality-gate rejects malformed TSV rows')
            if (allocated(cases)) deallocate(cases)
         end if
      end if
      call pretrain_model_destroy(model)
      open(newunit=unit, file=CORPUS_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=CASES_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
   end subroutine test_fortran_quality_gate

   subroutine test_hdc_pretraining()
      character(len=*), parameter :: CORPUS_PATH = 'vegai_pretraining_test_corpus.txt'
      character(len=*), parameter :: CHECKPOINT_PATH = 'vegai_pretraining_test_checkpoint.vgai'
      character(len=*), parameter :: CONTINUATION_A_PATH = 'vegai_continuation_a.txt'
      character(len=*), parameter :: CONTINUATION_B_PATH = 'vegai_continuation_b.txt'
      character(len=*), parameter :: HIERARCHICAL_PATH = 'vegai_hierarchical_context.txt'
      character(len=*), parameter :: DEBIASED_REPEAT_PATH = 'vegai_debiased_repetition.txt'
      character(len=*), parameter :: PARALLEL_CONTEXT_PATH = 'vegai_parallel_context.txt'
      type(hdc_pretrain_model_t) :: model_a, model_b, restored, legacy_model, positional_model, continuation_model
      type(hdc_pretrain_model_t) :: sampler_legacy_model, sampler_legacy_restored
      type(hdc_pretrain_model_t) :: hierarchical_model, local_only_model, hierarchical_replay, short_hierarchical_model
      type(hdc_pretrain_model_t) :: coherent_model, coherent_replay, short_coherent_model
      type(hdc_pretrain_model_t) :: debiased_unique_model, debiased_repeat_model, debiased_repeat_hybrid_model
      type(hdc_pretrain_model_t) :: resonant_model, resonant_replay, resonant_hybrid_model, resonant_restored
      type(hdc_pretrain_model_t) :: cache_disabled_model
      type(hdc_pretrain_model_t) :: context_serial_model, context_parallel_model
      type(hdc_pretrain_config_t) :: config, resumed_config, strict_config, legacy_config, positional_config, continuation_config
      type(hdc_pretrain_config_t) :: reference_config, hierarchical_config, local_only_config, coherent_config, debiased_config
      type(hdc_pretrain_config_t) :: resonant_config
      type(hdc_pretrain_config_t) :: openvino_rejected_config, context_serial_config, context_parallel_config, cache_disabled_config
      type(pretrain_metrics_t) :: vocabulary_metrics, train_metrics, extension_metrics
      type(pretrain_execution_report_t) :: default_execution, serial_execution, parallel_execution, cache_disabled_execution
      type(pretrain_execution_report_t) :: debiased_unique_execution, debiased_repeat_execution, resonant_execution
      type(hv_t) :: semantic_hv, semantic_before_extension, semantic_after_extension
      character(len=TOKEN_LEN) :: nearest_token
      integer :: unit, io_status
      integer(int32) :: status, slot, nearest_slot
      logical :: found
      real :: similarity

      open(newunit=unit, file=CORPUS_PATH, status='replace', action='write', &
           form='formatted', iostat=io_status)
      call assert_true(io_status == 0, 'pretraining fixture corpus opens')
      if (io_status /= 0) return
      write(unit, '(A)') 'king queen royal court'
      write(unit, '(A)') 'queen king royal court'
      write(unit, '(A)') 'king queen royal court'
      write(unit, '(A)') 'dog cat animal pet'
      write(unit, '(A)') 'cat dog animal pet'
      write(unit, '(A)') 'dog cat animal pet'
      close(unit)

      config = hdc_pretrain_config_t()
      config%max_vocabulary = 32_int32
      config%min_frequency = 2_int64
      config%context_window = 2_int32
      config%position_shift_bits = 41_int32
      config%epochs = 1_int32
      config%initial_plasticity_percent = 25_int32
      config%minimum_plasticity_percent = 1_int32
      config%plasticity_decay_events = 4_int32
      call assert_i32_equal(config%context_mode, PT_CONTEXT_HIERARCHICAL, &
                            'hierarchical context encoder is the long-record quality default')
      call assert_true(trim(pretrain_context_mode_name(config%context_mode)) == 'hierarchical', &
                       'context encoder exposes a stable human-readable name')

      call pretrain_build_vocabulary(model_a, CORPUS_PATH, config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'pretraining vocabulary pass succeeds')
      call assert_i32_equal(model_a%vocabulary%count, 8_int32, 'vocabulary preserves first-seen unique tokens')
      call assert_i32_equal(model_a%vocabulary%active_count, 8_int32, 'frequency threshold activates repeated tokens')
      call assert_true(vocabulary_metrics%raw_tokens == 24_int64, 'vocabulary pass counts raw tokens')
      call assert_true(vocabulary_metrics%tokenizer_truncations == 0_int64, &
                       'fixture corpus has no tokenizer truncation')
      call assert_true(vocabulary_metrics%non_ascii_bytes == 0_int64, &
                       'fixture corpus has no non-ASCII input bytes')
      call assert_true(pretrain_model_is_ready(model_a), 'vocabulary pass initializes trainable model')
      call assert_true(trim(pretrain_plasticity_sampler_name(model_a%plasticity_sampler)) == 'floyd-complement', &
                       'fresh model exposes the v7 Floyd/complement sampler lineage')
      call assert_true(pretrain_model_footprint_bytes(model_a) > 0_int64, &
                       'pretraining model reports resident footprint')

      call pretrain_train_file(model_a, CORPUS_PATH, config, train_metrics, status, default_execution)
      call assert_i32_equal(status, PT_OK, 'streaming HDC training pass succeeds')
      call assert_true(train_metrics%raw_tokens == 24_int64, 'training pass counts raw tokens')
      call assert_true(train_metrics%active_tokens == 24_int64, 'all fixture tokens are active')
      call assert_true(train_metrics%oov_tokens == 0_int64, 'fixture corpus has no OOV tokens')
      call assert_true(train_metrics%context_pairs == 60_int64, 'windowed ordered context count is exact')
      call assert_true(train_metrics%center_updates == 24_int64, 'one prototype update occurs per active center')
      call assert_true(default_execution%cpu_serial_batches == 6_int64 .and. &
                       default_execution%cpu_serial_contexts == 24_int64, &
                       'adaptive scheduler avoids OpenMP teams for short records by default')
      call assert_true(default_execution%cpu_parallel_batches == 0_int64, &
                       'short-record fixture does not select the parallel scheduler')
      call assert_true(default_execution%vocabulary_lookup_cache_hits > 0_int64 .and. &
                       default_execution%vocabulary_lookup_cache_misses > 0_int64 .and. &
                       default_execution%vocabulary_lookup_cache_hits + &
                       default_execution%vocabulary_lookup_cache_misses == train_metrics%raw_tokens, &
                       'exact-token lookup cache reports a complete hit/miss execution partition')
      call assert_true(model_a%training_epochs_completed == 1_int64, 'one completed epoch is recorded')

      ! The vocabulary cache is strictly execution-only. Disable it from the
      ! first lookup and require every checkpoint-relevant plane to match the
      ! cached trajectory byte-for-byte, including repeated input tokens.
      cache_disabled_config = config
      cache_disabled_config%vocabulary_lookup_cache = .false.
      call pretrain_build_vocabulary(cache_disabled_model, CORPUS_PATH, cache_disabled_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'cache-disabled vocabulary pass succeeds')
      call pretrain_train_file(cache_disabled_model, CORPUS_PATH, cache_disabled_config, train_metrics, status, &
                               cache_disabled_execution)
      call assert_i32_equal(status, PT_OK, 'cache-disabled training pass succeeds')
      call assert_true(cache_disabled_execution%vocabulary_lookup_cache_hits == 0_int64 .and. &
                       cache_disabled_execution%vocabulary_lookup_cache_misses == 0_int64, &
                       'disabled lookup cache has no cache telemetry')
      call assert_true(all(cache_disabled_model%base_bits == model_a%base_bits) .and. &
                       all(cache_disabled_model%semantic_bits == model_a%semantic_bits) .and. &
                       all(cache_disabled_model%experience_count == model_a%experience_count), &
                       'lookup-cache toggle preserves every checkpoint-relevant plane')

      call pretrain_lookup_token(model_a, 'king', slot, found, status)
      call assert_i32_equal(status, PT_OK, 'active token lookup succeeds')
      call assert_true(found, 'active token lookup reports found')
      call pretrain_get_semantic_hv(model_a, slot, semantic_hv, status)
      call assert_i32_equal(status, PT_OK, 'semantic prototype retrieval succeeds')
      call pretrain_nearest_semantic(model_a, semantic_hv, nearest_slot, nearest_token, similarity, status, slot)
      call assert_i32_equal(status, PT_OK, 'nearest semantic search succeeds')
      call assert_true(nearest_slot /= slot .and. len_trim(nearest_token) > 0, &
                       'nearest semantic search excludes the requested token')

      ! The fused fast hybrid kernel must be bit-identical to the retained
      ! rotate-then-add reference path.  The implementation selector is not
      ! model semantics and therefore does not enter checkpoint compatibility.
      reference_config = config
      reference_config%reference_context_kernel = .true.
      call pretrain_build_vocabulary(model_b, CORPUS_PATH, reference_config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'reference-kernel vocabulary pass succeeds')
      call pretrain_train_file(model_b, CORPUS_PATH, reference_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'reference-kernel training pass succeeds')
      call assert_true(all(model_a%base_bits == model_b%base_bits), 'base hypervectors are reproducible')
      call assert_true(all(model_a%semantic_bits == model_b%semantic_bits), &
                       'fused hybrid kernel remains bit-identical to the reference implementation')
      call assert_true(all(model_a%experience_count == model_b%experience_count), &
                       'fused hybrid kernel preserves deterministic learning schedule counters')

      ! OpenMP context construction is an execution optimization, not a second
      ! training algorithm.  Repeated centers in one record exercise the
      ! planned experience-counter path, so a serial and a parallel build must
      ! leave exactly the same checkpoint-relevant state.
      open(newunit=unit, file=PARALLEL_CONTEXT_PATH, status='replace', action='write', &
           form='formatted', iostat=io_status)
      call assert_true(io_status == 0, 'parallel context fixture opens')
      if (io_status == 0) then
         write(unit, '(A)') 'alpha beta alpha gamma alpha beta gamma alpha'
         write(unit, '(A)') 'gamma alpha beta alpha gamma beta alpha gamma'
         write(unit, '(A)') 'alpha beta alpha gamma alpha beta gamma alpha'
         close(unit)
      end if
      context_serial_config = config
      context_serial_config%max_vocabulary = 16_int32
      context_serial_config%min_frequency = 1_int64
      context_serial_config%strict_tokenization = .true.
      context_serial_config%parallel_context_build = .false.
      context_parallel_config = context_serial_config
      context_parallel_config%parallel_context_build = .true.
      ! Force the adaptive scheduler's OpenMP path even for this deliberately
      ! short repeated-center fixture.
      context_parallel_config%parallel_min_contexts = 1_int32
      call pretrain_build_vocabulary(context_serial_model, PARALLEL_CONTEXT_PATH, context_serial_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'serial context fixture vocabulary succeeds')
      call pretrain_train_file(context_serial_model, PARALLEL_CONTEXT_PATH, context_serial_config, &
                               train_metrics, status, serial_execution)
      call assert_i32_equal(status, PT_OK, 'serial context fixture training succeeds')
      call pretrain_build_vocabulary(context_parallel_model, PARALLEL_CONTEXT_PATH, context_parallel_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'parallel context fixture vocabulary succeeds')
      call pretrain_train_file(context_parallel_model, PARALLEL_CONTEXT_PATH, context_parallel_config, &
                               train_metrics, status, parallel_execution)
      call assert_i32_equal(status, PT_OK, 'parallel context fixture training succeeds')
      call assert_true(serial_execution%cpu_serial_batches == 3_int64 .and. &
                       serial_execution%cpu_parallel_batches == 0_int64, &
                       'explicit serial configuration reports only serial scheduler batches')
      call assert_true(parallel_execution%cpu_parallel_batches == 3_int64 .and. &
                       parallel_execution%cpu_serial_batches == 0_int64, &
                       'minimum-context override forces parallel scheduler batches')
      call assert_true(all(context_serial_model%base_bits == context_parallel_model%base_bits), &
                       'parallel context construction preserves base identities')
      call assert_true(all(context_serial_model%semantic_bits == context_parallel_model%semantic_bits), &
                       'parallel context construction is bit-identical with repeated centers')
      call assert_true(all(context_serial_model%experience_count == context_parallel_model%experience_count), &
                       'parallel context construction preserves source-ordered experiences')

      ! Feedback contexts are also immutable for an entire epoch. Exercise two
      ! epochs so the second pass actually reads learned snapshot prototypes,
      ! then require serial/OpenMP checkpoint-relevant state to remain exact.
      context_serial_config%context_mode = PT_CONTEXT_RESONANT
      context_serial_config%epochs = 2_int32
      context_parallel_config = context_serial_config
      context_parallel_config%parallel_context_build = .true.
      context_parallel_config%parallel_min_contexts = 1_int32
      call pretrain_build_vocabulary(context_serial_model, PARALLEL_CONTEXT_PATH, context_serial_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'serial resonant context vocabulary succeeds')
      call pretrain_train_file(context_serial_model, PARALLEL_CONTEXT_PATH, context_serial_config, &
                               train_metrics, status, serial_execution)
      call assert_i32_equal(status, PT_OK, 'serial resonant context training succeeds')
      call pretrain_build_vocabulary(context_parallel_model, PARALLEL_CONTEXT_PATH, context_parallel_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'parallel resonant context vocabulary succeeds')
      call pretrain_train_file(context_parallel_model, PARALLEL_CONTEXT_PATH, context_parallel_config, &
                               train_metrics, status, parallel_execution)
      call assert_i32_equal(status, PT_OK, 'parallel resonant context training succeeds')
      call assert_true(serial_execution%resonant_snapshot_contexts > 0_int64 .and. &
                       parallel_execution%resonant_snapshot_contexts > 0_int64, &
                       'resonant serial and OpenMP paths both consume frozen snapshots after epoch one')
      call assert_true(all(context_serial_model%semantic_bits == context_parallel_model%semantic_bits) .and. &
                       all(context_serial_model%experience_count == context_parallel_model%experience_count), &
                       'resonant serial/OpenMP construction remains byte-identical across learned feedback')

      ! The ordinary eight-token fixture is intentionally below the hierarchy
      ! threshold, so hierarchical mode must reduce exactly to local hybrid.
      hierarchical_config = config
      hierarchical_config%context_mode = PT_CONTEXT_HIERARCHICAL
      call assert_true(trim(pretrain_context_mode_name(hierarchical_config%context_mode)) == 'hierarchical', &
                       'hierarchical context encoder exposes a stable human-readable name')
      call pretrain_build_vocabulary(short_hierarchical_model, CORPUS_PATH, hierarchical_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'short-record hierarchical vocabulary pass succeeds')
      call pretrain_train_file(short_hierarchical_model, CORPUS_PATH, hierarchical_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'short-record hierarchical training succeeds')
      call assert_true(all(short_hierarchical_model%semantic_bits == model_a%semantic_bits), &
                       'hierarchical threshold leaves short-record hybrid semantics unchanged')
      coherent_config = hierarchical_config
      coherent_config%context_mode = PT_CONTEXT_COHERENT
      call assert_true(trim(pretrain_context_mode_name(coherent_config%context_mode)) == 'coherent', &
                       'coherent context encoder exposes a stable human-readable name')
      call pretrain_build_vocabulary(short_coherent_model, CORPUS_PATH, coherent_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'short-record coherent vocabulary succeeds')
      call pretrain_train_file(short_coherent_model, CORPUS_PATH, coherent_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'short-record coherent training succeeds')
      call assert_true(all(short_coherent_model%semantic_bits == model_a%semantic_bits), &
                       'coherent threshold leaves short-record hybrid semantics unchanged')
      debiased_config = hierarchical_config
      debiased_config%context_mode = PT_CONTEXT_DEBIASED
      call assert_true(trim(pretrain_context_mode_name(debiased_config%context_mode)) == 'debiased', &
                       'de-biased context encoder exposes a stable human-readable name')
      call pretrain_build_vocabulary(debiased_repeat_model, CORPUS_PATH, debiased_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'short-record de-biased vocabulary succeeds')
      call pretrain_train_file(debiased_repeat_model, CORPUS_PATH, debiased_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'short-record de-biased training succeeds')
      call assert_true(all(debiased_repeat_model%semantic_bits == model_a%semantic_bits), &
                       'de-biased threshold leaves short-record hybrid semantics unchanged')
      call pretrain_model_destroy(debiased_repeat_model)

      ! RESONANT feeds an epoch-frozen learned topical plane back into local
      ! contexts while retaining immutable base identities for positional
      ! evidence. A fresh model starts with semantic=base, so epoch one must be
      ! exactly HYBRID; epoch two must use distinct learned input. The frozen
      ! source keeps fused/reference, OpenMP, and checkpoint replay contracts
      ! deterministic instead of introducing within-epoch update-order effects.
      resonant_config = config
      resonant_config%context_mode = PT_CONTEXT_RESONANT
      call assert_true(trim(pretrain_context_mode_name(resonant_config%context_mode)) == 'resonant', &
                       'resonant semantic-feedback encoder exposes a stable human-readable name')
      call pretrain_build_vocabulary(resonant_model, CORPUS_PATH, resonant_config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'resonant vocabulary pass succeeds')
      call assert_true(pretrain_epoch_snapshot_bytes(resonant_model, resonant_config) == 32_int64 * 2048_int64, &
                       'resonant model exposes its one-per-epoch semantic snapshot reservation')
      call pretrain_train_file(resonant_model, CORPUS_PATH, resonant_config, train_metrics, status, resonant_execution)
      call assert_i32_equal(status, PT_OK, 'resonant first epoch succeeds')
      call assert_true(resonant_execution%resonant_snapshot_contexts == 24_int64, &
                       'resonant execution reports every frozen-snapshot context')
      call assert_true(all(resonant_model%base_bits == model_a%base_bits) .and. &
                       all(resonant_model%semantic_bits == model_a%semantic_bits) .and. &
                       all(resonant_model%experience_count == model_a%experience_count), &
                       'resonant first epoch is byte-identical to local hybrid initialization')

      reference_config = resonant_config
      reference_config%reference_context_kernel = .true.
      call pretrain_build_vocabulary(resonant_replay, CORPUS_PATH, reference_config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'resonant reference vocabulary pass succeeds')
      call pretrain_train_file(resonant_replay, CORPUS_PATH, reference_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'resonant reference first epoch succeeds')
      call assert_true(all(resonant_replay%semantic_bits == resonant_model%semantic_bits), &
                       'resonant mixed fused kernel is bit-identical to its reference path')

      local_only_config = config
      local_only_config%context_mode = PT_CONTEXT_HYBRID
      local_only_config%epochs = 2_int32
      call pretrain_build_vocabulary(resonant_hybrid_model, CORPUS_PATH, local_only_config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'two-epoch hybrid comparison vocabulary rebuild succeeds')
      call pretrain_train_file(resonant_hybrid_model, CORPUS_PATH, local_only_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'two-epoch hybrid comparison training succeeds')

      call pretrain_train_file(resonant_model, CORPUS_PATH, resonant_config, train_metrics, status, resonant_execution)
      call assert_i32_equal(status, PT_OK, 'resonant second epoch succeeds')
      call pretrain_train_file(resonant_replay, CORPUS_PATH, reference_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'resonant reference second epoch succeeds')
      call assert_true(all(resonant_replay%semantic_bits == resonant_model%semantic_bits) .and. &
                       all(resonant_replay%experience_count == resonant_model%experience_count), &
                       'resonant fused/reference path remains bit-identical after feedback begins')
      call assert_true(all(resonant_model%base_bits == resonant_hybrid_model%base_bits) .and. &
                       all(resonant_model%experience_count == resonant_hybrid_model%experience_count), &
                       'resonant feedback changes context evidence without changing identities or update schedule')
      call assert_true(any(resonant_model%semantic_bits /= resonant_hybrid_model%semantic_bits), &
                       'second resonant epoch consumes learned semantic neighbours rather than static identities')

      call pretrain_checkpoint_save(resonant_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'resonant checkpoint save succeeds')
      call assert_checkpoint_header_version(CHECKPOINT_PATH, 8_int32, 'resonant')
      call pretrain_checkpoint_load(CHECKPOINT_PATH, resonant_restored, status)
      call assert_i32_equal(status, PT_OK, 'resonant checkpoint load succeeds')
      call assert_i32_equal(resonant_restored%context_mode, PT_CONTEXT_RESONANT, &
                            'resonant checkpoint preserves semantic-feedback identity')
      call assert_true(all(resonant_restored%semantic_bits == resonant_model%semantic_bits), &
                       'resonant checkpoint restores learned prototypes before its next frozen snapshot')
      call pretrain_config_from_model(resonant_restored, resumed_config, status)
      call assert_i32_equal(status, PT_OK, 'resonant checkpoint exposes a compatible resume configuration')
      resumed_config%epochs = 1_int32
      call pretrain_train_file(resonant_model, CORPUS_PATH, resonant_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'uninterrupted resonant third epoch succeeds')
      call pretrain_train_file(resonant_restored, CORPUS_PATH, resumed_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'resonant checkpoint-resumed third epoch succeeds')
      call assert_true(all(resonant_restored%semantic_bits == resonant_model%semantic_bits) .and. &
                       all(resonant_restored%experience_count == resonant_model%experience_count), &
                       'resonant continuation remains byte-identical after a checkpoint boundary')

      ! Coherent cross-fold inputs have no representation in the immutable
      ! OpenVINO lane ABI. Reject the configuration before any worker/process
      ! can be started rather than silently changing encoder semantics.
      openvino_rejected_config = coherent_config
      openvino_rejected_config%context_provider = PT_CONTEXT_PROVIDER_OPENVINO
      openvino_rejected_config%openvino_worker = 'tools/openvino_context_worker.py'
      openvino_rejected_config%openvino_python = 'python3'
      openvino_rejected_config%openvino_work_dir = '.'
      openvino_rejected_config%openvino_device = 'GPU'
      openvino_rejected_config%openvino_min_contexts = 1_int32
      call pretrain_build_vocabulary(model_b, CORPUS_PATH, openvino_rejected_config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_ERR_INVALID_ARGUMENT, &
                            'coherent mode rejects the unsupported OpenVINO worker ABI')
      openvino_rejected_config%context_mode = PT_CONTEXT_DEBIASED
      call pretrain_build_vocabulary(model_b, CORPUS_PATH, openvino_rejected_config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_ERR_INVALID_ARGUMENT, &
                            'de-biased mode rejects the unsupported OpenVINO worker ABI')
      openvino_rejected_config%context_mode = PT_CONTEXT_RESONANT
      call pretrain_build_vocabulary(model_b, CORPUS_PATH, openvino_rejected_config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_ERR_INVALID_ARGUMENT, &
                            'resonant frozen-snapshot mode rejects the unsupported OpenVINO worker ABI')

      ! The fast positional encoder is an implementation-only optimization: it
      ! must preserve the legacy semantic plane exactly.  Hybrid adds a second,
      ! order-invariant context view, so it should intentionally differ while
      ! retaining all stream counters and deterministic replay guarantees.
      legacy_config = config
      legacy_config%context_mode = PT_CONTEXT_LEGACY_POSITIONAL
      positional_config = config
      positional_config%context_mode = PT_CONTEXT_POSITIONAL
      call pretrain_build_vocabulary(legacy_model, CORPUS_PATH, legacy_config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'legacy comparison vocabulary pass succeeds')
      call pretrain_train_file(legacy_model, CORPUS_PATH, legacy_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'legacy comparison training pass succeeds')
      call pretrain_build_vocabulary(positional_model, CORPUS_PATH, positional_config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'fast positional vocabulary pass succeeds')
      call pretrain_train_file(positional_model, CORPUS_PATH, positional_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'fast positional training pass succeeds')
      call assert_true(all(legacy_model%semantic_bits == positional_model%semantic_bits), &
                       'bit-sliced positional encoder is bit-identical to legacy learning')
      call assert_true(all(legacy_model%experience_count == positional_model%experience_count), &
                       'fast positional encoder preserves learning schedule counters')
      call assert_true(any(model_a%semantic_bits /= positional_model%semantic_bits), &
                       'hybrid encoder contributes distinct order-invariant semantic evidence')

      ! Hierarchical mode adds a whole-record topical summary only for long
      ! records.  The controlled fixture has twelve active tokens, beyond the
      ! threshold, so it must differ from local-only hybrid while remaining
      ! deterministic across replay and checkpoint round-trip.
      open(newunit=unit, file=HIERARCHICAL_PATH, status='replace', action='write', &
           form='formatted', iostat=io_status)
      call assert_true(io_status == 0, 'hierarchical context fixture opens')
      if (io_status == 0) then
         write(unit, '(A)') 'amber birch cedar dahlia ember fern granite harbor indigo juniper kestrel lilac'
         write(unit, '(A)') 'amber birch cedar dahlia ember fern granite harbor indigo juniper kestrel lilac'
         close(unit)
      end if
      hierarchical_config = config
      hierarchical_config%max_vocabulary = 16_int32
      hierarchical_config%min_frequency = 1_int64
      hierarchical_config%epochs = 1_int32
      hierarchical_config%strict_tokenization = .true.
      hierarchical_config%context_mode = PT_CONTEXT_HIERARCHICAL
      local_only_config = hierarchical_config
      local_only_config%context_mode = PT_CONTEXT_HYBRID
      call pretrain_build_vocabulary(hierarchical_model, HIERARCHICAL_PATH, hierarchical_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'hierarchical vocabulary pass succeeds')
      call pretrain_train_file(hierarchical_model, HIERARCHICAL_PATH, hierarchical_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'hierarchical long-record training succeeds')
      call pretrain_build_vocabulary(local_only_model, HIERARCHICAL_PATH, local_only_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'local-only comparison vocabulary pass succeeds')
      call pretrain_train_file(local_only_model, HIERARCHICAL_PATH, local_only_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'local-only comparison training succeeds')
      call assert_true(any(hierarchical_model%semantic_bits /= local_only_model%semantic_bits), &
                       'hierarchical long-record summary changes learned semantic prototypes')
      coherent_config = hierarchical_config
      coherent_config%context_mode = PT_CONTEXT_COHERENT
      call pretrain_build_vocabulary(coherent_model, HIERARCHICAL_PATH, coherent_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'coherent vocabulary pass succeeds')
      call pretrain_train_file(coherent_model, HIERARCHICAL_PATH, coherent_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'coherent long-record training succeeds')
      call assert_true(any(coherent_model%semantic_bits /= hierarchical_model%semantic_bits), &
                       'coherent cross-fold topical view differs from self-inclusive hierarchy')
      reference_config = hierarchical_config
      reference_config%reference_context_kernel = .true.
      call pretrain_build_vocabulary(hierarchical_replay, HIERARCHICAL_PATH, reference_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'hierarchical reference vocabulary pass succeeds')
      call pretrain_train_file(hierarchical_replay, HIERARCHICAL_PATH, reference_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'hierarchical reference training succeeds')
      call assert_true(all(hierarchical_model%semantic_bits == hierarchical_replay%semantic_bits), &
                       'hierarchical fused local context remains bit-identical to reference learning')
      reference_config = coherent_config
      reference_config%reference_context_kernel = .true.
      call pretrain_build_vocabulary(coherent_replay, HIERARCHICAL_PATH, reference_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'coherent reference vocabulary pass succeeds')
      call pretrain_train_file(coherent_replay, HIERARCHICAL_PATH, reference_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'coherent reference training succeeds')
      call assert_true(all(coherent_model%semantic_bits == coherent_replay%semantic_bits), &
                       'coherent fused local context remains bit-identical to reference learning')

      ! With no duplicate active slots, DEBIASED has the same whole-record
      ! majority input and must therefore be byte-identical to HIERARCHICAL.
      debiased_config = hierarchical_config
      debiased_config%context_mode = PT_CONTEXT_DEBIASED
      call pretrain_build_vocabulary(debiased_unique_model, HIERARCHICAL_PATH, debiased_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'unique-record de-biased vocabulary pass succeeds')
      call pretrain_train_file(debiased_unique_model, HIERARCHICAL_PATH, debiased_config, train_metrics, status, &
                               debiased_unique_execution)
      call assert_i32_equal(status, PT_OK, 'unique-record de-biased training succeeds')
      call assert_true(debiased_unique_execution%debiased_topic_records == 2_int64 .and. &
                       debiased_unique_execution%debiased_topic_suppressed_records == 0_int64 .and. &
                       debiased_unique_execution%debiased_topic_distinct_tokens == 24_int64, &
                       'de-biased telemetry records distinct long-record topical evidence')
      call assert_true(all(debiased_unique_model%base_bits == hierarchical_model%base_bits) .and. &
                       all(debiased_unique_model%semantic_bits == hierarchical_model%semantic_bits) .and. &
                       all(debiased_unique_model%experience_count == hierarchical_model%experience_count), &
                       'de-biased mode is bit-identical to hierarchy on duplicate-free long records')

      ! A record can be long in occurrences yet narrow in distinct vocabulary.
      ! In that case DEBIASED explicitly withholds its global channel and
      ! becomes local HYBRID without changing the ordered update schedule.
      open(newunit=unit, file=DEBIASED_REPEAT_PATH, status='replace', action='write', &
           form='formatted', iostat=io_status)
      call assert_true(io_status == 0, 'de-biased repetition fixture opens')
      if (io_status == 0) then
         write(unit, '(A)') 'amber birch cedar dahlia amber birch cedar dahlia amber birch cedar dahlia'
         write(unit, '(A)') 'dahlia cedar birch amber dahlia cedar birch amber dahlia cedar birch amber'
         close(unit)
      end if
      debiased_config = hierarchical_config
      debiased_config%context_mode = PT_CONTEXT_DEBIASED
      local_only_config = debiased_config
      local_only_config%context_mode = PT_CONTEXT_HYBRID
      call pretrain_build_vocabulary(debiased_repeat_model, DEBIASED_REPEAT_PATH, debiased_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'repetitive de-biased vocabulary pass succeeds')
      call pretrain_train_file(debiased_repeat_model, DEBIASED_REPEAT_PATH, debiased_config, train_metrics, status, &
                               debiased_repeat_execution)
      call assert_i32_equal(status, PT_OK, 'repetitive de-biased training succeeds')
      call assert_true(debiased_repeat_execution%debiased_topic_records == 0_int64 .and. &
                       debiased_repeat_execution%debiased_topic_suppressed_records == 2_int64 .and. &
                       debiased_repeat_execution%debiased_topic_distinct_tokens == 8_int64, &
                       'de-biased telemetry suppresses repetition-dominated global evidence')
      call pretrain_build_vocabulary(debiased_repeat_hybrid_model, DEBIASED_REPEAT_PATH, local_only_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'repetitive hybrid vocabulary pass succeeds')
      call pretrain_train_file(debiased_repeat_hybrid_model, DEBIASED_REPEAT_PATH, local_only_config, &
                               train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'repetitive hybrid training succeeds')
      call assert_true(all(debiased_repeat_model%base_bits == debiased_repeat_hybrid_model%base_bits) .and. &
                       all(debiased_repeat_model%semantic_bits == debiased_repeat_hybrid_model%semantic_bits) .and. &
                       all(debiased_repeat_model%experience_count == debiased_repeat_hybrid_model%experience_count), &
                       'de-biased repetition suppression is bit-identical to local hybrid')
      ! v7 binds the new unbiased exact-mask sampler to every newly-trained
      ! encoder lineage. Reading the raw header prevents a future broad writer
      ! version change from silently weakening this continuation contract.
      call pretrain_checkpoint_save(legacy_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'legacy checkpoint save succeeds')
      call assert_checkpoint_header_version(CHECKPOINT_PATH, 7_int32, 'legacy')
      call pretrain_checkpoint_save(positional_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'positional checkpoint save succeeds')
      call assert_checkpoint_header_version(CHECKPOINT_PATH, 7_int32, 'positional')
      call pretrain_checkpoint_save(local_only_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'hybrid checkpoint save succeeds')
      call assert_checkpoint_header_version(CHECKPOINT_PATH, 7_int32, 'hybrid')
      call pretrain_checkpoint_save(hierarchical_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'hierarchical checkpoint save succeeds')
      call assert_checkpoint_header_version(CHECKPOINT_PATH, 7_int32, 'hierarchical')
      call pretrain_checkpoint_load(CHECKPOINT_PATH, restored, status)
      call assert_i32_equal(status, PT_OK, 'hierarchical checkpoint load succeeds')
      call assert_i32_equal(restored%context_mode, PT_CONTEXT_HIERARCHICAL, &
                            'checkpoint preserves hierarchical encoder semantics')
      call assert_true(all(restored%semantic_bits == hierarchical_model%semantic_bits), &
                       'hierarchical checkpoint restores semantic prototypes')
      call assert_i32_equal(coherent_model%context_mode, PT_CONTEXT_COHERENT, &
                            'coherent model retains its context-mode identity before checkpointing')
      call pretrain_checkpoint_save(coherent_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'coherent checkpoint save succeeds')
      call assert_checkpoint_header_version(CHECKPOINT_PATH, 7_int32, 'coherent')
      call pretrain_checkpoint_load(CHECKPOINT_PATH, restored, status)
      call assert_i32_equal(status, PT_OK, 'coherent checkpoint load succeeds')
      call assert_i32_equal(restored%context_mode, PT_CONTEXT_COHERENT, &
                            'checkpoint preserves coherent encoder semantics')
      call assert_true(all(restored%semantic_bits == coherent_model%semantic_bits), &
                       'coherent checkpoint restores semantic prototypes')
      call assert_i32_equal(debiased_unique_model%context_mode, PT_CONTEXT_DEBIASED, &
                            'de-biased model retains its context-mode identity before checkpointing')
      call pretrain_checkpoint_save(debiased_unique_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'de-biased checkpoint save succeeds')
      call assert_checkpoint_header_version(CHECKPOINT_PATH, 7_int32, 'de-biased')
      call pretrain_checkpoint_load(CHECKPOINT_PATH, restored, status)
      call assert_i32_equal(status, PT_OK, 'de-biased checkpoint load succeeds')
      call assert_i32_equal(restored%context_mode, PT_CONTEXT_DEBIASED, &
                            'checkpoint preserves de-biased encoder semantics')
      call assert_true(all(restored%semantic_bits == debiased_unique_model%semantic_bits), &
                       'de-biased checkpoint restores semantic prototypes')
      debiased_unique_model%plasticity_sampler = PT_PLASTICITY_SAMPLER_LEGACY
      call pretrain_checkpoint_save(debiased_unique_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_ERR_CHECKPOINT, &
                            'de-biased semantic mode refuses a legacy sampler checkpoint lineage')
      debiased_unique_model%plasticity_sampler = PT_PLASTICITY_SAMPLER_FLOYD_COMPLEMENT

      ! Exercise a synthetic historical coherent v6 lineage rather than only a
      ! parser branch. Its continuation must keep the old mask stream while
      ! retaining the coherent context contract.
      coherent_model%plasticity_sampler = PT_PLASTICITY_SAMPLER_LEGACY
      call pretrain_checkpoint_save(coherent_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'historical coherent checkpoint save succeeds')
      call assert_checkpoint_header_version(CHECKPOINT_PATH, 6_int32, 'historical coherent sampler')
      call pretrain_checkpoint_load(CHECKPOINT_PATH, restored, status)
      call assert_i32_equal(status, PT_OK, 'historical coherent checkpoint load succeeds')
      call assert_i32_equal(restored%context_mode, PT_CONTEXT_COHERENT, &
                            'v6 checkpoint retains coherent context semantics')
      call assert_i32_equal(restored%plasticity_sampler, PT_PLASTICITY_SAMPLER_LEGACY, &
                            'v6 checkpoint restores historical sampler identity')
      call pretrain_config_from_model(restored, resumed_config, status)
      call assert_i32_equal(status, PT_OK, 'historical coherent resume configuration is available')
      resumed_config%epochs = 1_int32
      call pretrain_train_file(coherent_model, HIERARCHICAL_PATH, coherent_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'historical coherent uninterrupted epoch succeeds')
      call pretrain_train_file(restored, HIERARCHICAL_PATH, resumed_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'historical coherent resumed epoch succeeds')
      call assert_true(all(restored%semantic_bits == coherent_model%semantic_bits), &
                       'v6 coherent resume retains historical selection stream')

      call pretrain_checkpoint_save(model_a, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'pretraining checkpoint save succeeds')
      call pretrain_checkpoint_load(CHECKPOINT_PATH, restored, status)
      call assert_i32_equal(status, PT_OK, 'pretraining checkpoint load succeeds')
      call assert_i32_equal(restored%vocabulary%active_count, model_a%vocabulary%active_count, &
                            'checkpoint rebuild restores active vocabulary count')
      call assert_i32_equal(restored%context_mode, PT_CONTEXT_HIERARCHICAL, &
                            'checkpoint preserves default hierarchical encoder semantics')
      call assert_true(all(restored%base_bits == model_a%base_bits), 'checkpoint restores base hypervectors')
      call assert_true(all(restored%semantic_bits == model_a%semantic_bits), &
                       'checkpoint restores semantic hypervectors')
      call assert_true(all(restored%experience_count == model_a%experience_count), &
                       'checkpoint restores experience counters')

      call pretrain_config_from_model(restored, resumed_config, status)
      call assert_i32_equal(status, PT_OK, 'checkpoint exposes compatible resume configuration')
      resumed_config%epochs = 1_int32
      call pretrain_train_file(model_a, CORPUS_PATH, config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'reference second epoch succeeds')
      call pretrain_train_file(restored, CORPUS_PATH, resumed_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'checkpoint resume epoch succeeds')
      call assert_true(all(restored%semantic_bits == model_a%semantic_bits), &
                       'checkpoint resume remains bit-identical to uninterrupted training')
      call assert_true(all(restored%experience_count == model_a%experience_count), &
                       'checkpoint resume preserves deterministic learning schedule')

      ! Cache state itself is never serialized. A checkpoint produced with the
      ! cache disabled must resume identically when the default cache is on.
      call pretrain_checkpoint_save(cache_disabled_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'cache-disabled checkpoint save succeeds')
      call pretrain_checkpoint_load(CHECKPOINT_PATH, restored, status)
      call assert_i32_equal(status, PT_OK, 'cache-disabled checkpoint load succeeds')
      call pretrain_config_from_model(restored, resumed_config, status)
      call assert_i32_equal(status, PT_OK, 'cache-disabled checkpoint exposes resume configuration')
      call assert_true(resumed_config%vocabulary_lookup_cache, &
                       'resume defaults to the transient lookup cache without changing semantic state')
      resumed_config%epochs = 1_int32
      call pretrain_train_file(cache_disabled_model, CORPUS_PATH, cache_disabled_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'cache-disabled uninterrupted second epoch succeeds')
      call pretrain_train_file(restored, CORPUS_PATH, resumed_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'cache-enabled resume from cache-disabled checkpoint succeeds')
      call assert_true(all(restored%semantic_bits == cache_disabled_model%semantic_bits) .and. &
                       all(restored%experience_count == cache_disabled_model%experience_count), &
                       'lookup-cache choice preserves deterministic checkpoint continuation')

      resumed_config%context_window = resumed_config%context_window + 1_int32
      call pretrain_train_file(restored, CORPUS_PATH, resumed_config, train_metrics, status)
      call assert_i32_equal(status, PT_ERR_INCOMPATIBLE_CONFIG, &
                            'context-window mismatch is rejected on resume')
      resumed_config%context_window = model_a%context_window
      resumed_config%context_mode = PT_CONTEXT_POSITIONAL
      call pretrain_train_file(restored, CORPUS_PATH, resumed_config, train_metrics, status)
      call assert_i32_equal(status, PT_ERR_INCOMPATIBLE_CONFIG, &
                            'context encoder mismatch is rejected on resume')

      ! A pre-v7 checkpoint must continue with its historical selection stream,
      ! not silently switch to the new unbiased sampler and diverge after the
      ! first resumed update. The version-selected compatibility metadata keeps
      ! v5 as the on-disk identity of this synthetic legacy lineage.
      call pretrain_build_vocabulary(sampler_legacy_model, CORPUS_PATH, config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'legacy-sampler vocabulary pass succeeds')
      sampler_legacy_model%plasticity_sampler = PT_PLASTICITY_SAMPLER_LEGACY
      call pretrain_train_file(sampler_legacy_model, CORPUS_PATH, config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'legacy-sampler initial epoch succeeds')
      call pretrain_checkpoint_save(sampler_legacy_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'legacy-sampler checkpoint save succeeds')
      call assert_checkpoint_header_version(CHECKPOINT_PATH, 5_int32, 'legacy sampler compatibility')
      call pretrain_checkpoint_load(CHECKPOINT_PATH, sampler_legacy_restored, status)
      call assert_i32_equal(status, PT_OK, 'legacy-sampler checkpoint load succeeds')
      call assert_i32_equal(sampler_legacy_restored%plasticity_sampler, PT_PLASTICITY_SAMPLER_LEGACY, &
                            'v5 checkpoint restores its historical sampler identity')
      call assert_true(trim(pretrain_plasticity_sampler_name(sampler_legacy_restored%plasticity_sampler)) == &
                       'legacy-bounded-retry', 'v5 sampler identity has an inspect-safe name')
      call pretrain_config_from_model(sampler_legacy_restored, resumed_config, status)
      call assert_i32_equal(status, PT_OK, 'legacy-sampler resume configuration is available')
      resumed_config%epochs = 1_int32
      call pretrain_train_file(sampler_legacy_model, CORPUS_PATH, config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'legacy-sampler uninterrupted second epoch succeeds')
      call pretrain_train_file(sampler_legacy_restored, CORPUS_PATH, resumed_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'legacy-sampler resumed second epoch succeeds')
      call assert_true(all(sampler_legacy_restored%semantic_bits == sampler_legacy_model%semantic_bits), &
                       'v5 resume retains the historical deterministic selection stream')
      call assert_true(all(sampler_legacy_restored%experience_count == sampler_legacy_model%experience_count), &
                       'v5 resume retains the historical experience schedule')

      ! Continuous pretraining grows a fixed-capacity vocabulary transactionally
      ! at shard boundaries.  With min_frequency=1 every accepted token in a
      ! strict shard becomes active before that same shard is learned, so no
      ! token becomes an avoidable OOV loss.
      open(newunit=unit, file=CONTINUATION_A_PATH, status='replace', action='write', &
           form='formatted', iostat=io_status)
      call assert_true(io_status == 0, 'first continuation shard opens')
      if (io_status == 0) then
         write(unit, '(A)') 'alpha beta'
         write(unit, '(A)') 'beta alpha'
         close(unit)
      end if
      open(newunit=unit, file=CONTINUATION_B_PATH, status='replace', action='write', &
           form='formatted', iostat=io_status)
      call assert_true(io_status == 0, 'second continuation shard opens')
      if (io_status == 0) then
         write(unit, '(A)') 'beta gamma'
         write(unit, '(A)') 'gamma beta'
         close(unit)
      end if
      continuation_config = config
      continuation_config%max_vocabulary = 16_int32
      continuation_config%min_frequency = 1_int64
      continuation_config%epochs = 1_int32
      continuation_config%strict_tokenization = .true.
      call pretrain_build_vocabulary(continuation_model, CONTINUATION_A_PATH, continuation_config, &
                                     vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'first continuation shard builds vocabulary')
      call assert_i32_equal(continuation_model%vocabulary%count, 2_int32, &
                            'first shard supplies two vocabulary terms')
      call pretrain_train_file(continuation_model, CONTINUATION_A_PATH, continuation_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'first continuation shard trains')
      call pretrain_lookup_token(continuation_model, 'beta', slot, found, status)
      call assert_true(found .and. status == PT_OK, 'existing continuation token is queryable')
      call pretrain_get_semantic_hv(continuation_model, slot, semantic_before_extension, status)
      call assert_i32_equal(status, PT_OK, 'existing continuation prototype is readable')
      call pretrain_extend_vocabulary(continuation_model, CONTINUATION_B_PATH, continuation_config, &
                                      extension_metrics, status)
      call assert_i32_equal(status, PT_OK, 'second continuation shard extends vocabulary')
      call assert_true(extension_metrics%raw_tokens == 4_int64, &
                       'vocabulary extension observes all second-shard tokens')
      call assert_i32_equal(continuation_model%vocabulary%count, 3_int32, &
                            'extension appends unseen gamma token')
      call assert_i32_equal(continuation_model%vocabulary%active_count, 3_int32, &
                            'min-frequency-one extension activates every observed token')
      call pretrain_lookup_token(continuation_model, 'beta', slot, found, status)
      call pretrain_get_semantic_hv(continuation_model, slot, semantic_after_extension, status)
      call assert_hv_equal(semantic_after_extension, semantic_before_extension, &
                           'vocabulary extension never resets existing semantic prototypes')
      call pretrain_train_file(continuation_model, CONTINUATION_B_PATH, continuation_config, train_metrics, status)
      call assert_i32_equal(status, PT_OK, 'second continuation shard trains after extension')
      call assert_true(train_metrics%raw_tokens == 4_int64 .and. train_metrics%active_tokens == 4_int64 .and. &
                       train_metrics%oov_tokens == 0_int64, 'strict continuous shard has no token loss or OOV')
      call pretrain_lookup_token(continuation_model, 'gamma', slot, found, status)
      call assert_true(found .and. status == PT_OK, 'new continuation token becomes immediately queryable')

      open(newunit=unit, file=CORPUS_PATH, status='replace', action='write', &
           form='formatted', iostat=io_status)
      call assert_true(io_status == 0, 'strict tokenizer fixture opens')
      if (io_status == 0) then
         write(unit, '(A)') repeat('x', 49)
         close(unit)
         strict_config = config
         strict_config%strict_tokenization = .true.
         call pretrain_build_vocabulary(model_b, CORPUS_PATH, strict_config, vocabulary_metrics, status)
         call assert_i32_equal(status, PT_ERR_TOKENIZATION, 'strict mode rejects dropped long token')
      end if

      open(newunit=unit, file=CORPUS_PATH, status='replace', action='write', &
           form='formatted', iostat=io_status)
      call assert_true(io_status == 0, 'strict non-ASCII fixture opens')
      if (io_status == 0) then
         write(unit, '(A)') 'plain' // achar(195) // achar(167)
         close(unit)
         strict_config = config
         strict_config%strict_tokenization = .true.
         call pretrain_build_vocabulary(model_b, CORPUS_PATH, strict_config, vocabulary_metrics, status)
         call assert_i32_equal(status, PT_ERR_TOKENIZATION, 'strict mode rejects non-ASCII bytes')
      end if

      open(newunit=unit, file=CORPUS_PATH, status='replace', action='write', &
           form='formatted', iostat=io_status)
      call assert_true(io_status == 0, 'overlong record fixture opens')
      if (io_status == 0) then
         write(unit, '(A)') repeat('y', MAX_LINE_CHARS + 1_int32)
         write(unit, '(A)') 'safe safe'
         close(unit)
         call pretrain_build_vocabulary(model_b, CORPUS_PATH, config, vocabulary_metrics, status)
         call assert_i32_equal(status, PT_OK, 'non-strict mode drops overlong record and continues')
         call assert_true(vocabulary_metrics%overlong_records == 1_int64, &
                          'overlong record is explicitly counted')
         call assert_true(vocabulary_metrics%raw_tokens == 2_int64, &
                          'overlong record prefix never enters vocabulary')
         strict_config = config
         strict_config%strict_tokenization = .true.
         call pretrain_build_vocabulary(model_b, CORPUS_PATH, strict_config, vocabulary_metrics, status)
         call assert_i32_equal(status, PT_ERR_RECORD_TOO_LONG, &
                               'strict mode rejects overlong physical record')
      end if

      call pretrain_model_destroy(model_a)
      call pretrain_model_destroy(model_b)
      call pretrain_model_destroy(restored)
      call pretrain_model_destroy(legacy_model)
      call pretrain_model_destroy(positional_model)
      call pretrain_model_destroy(continuation_model)
      call pretrain_model_destroy(sampler_legacy_model)
      call pretrain_model_destroy(sampler_legacy_restored)
      call pretrain_model_destroy(hierarchical_model)
      call pretrain_model_destroy(local_only_model)
      call pretrain_model_destroy(hierarchical_replay)
      call pretrain_model_destroy(short_hierarchical_model)
      call pretrain_model_destroy(debiased_unique_model)
      call pretrain_model_destroy(debiased_repeat_model)
      call pretrain_model_destroy(debiased_repeat_hybrid_model)
      call pretrain_model_destroy(resonant_model)
      call pretrain_model_destroy(resonant_replay)
      call pretrain_model_destroy(resonant_hybrid_model)
      call pretrain_model_destroy(resonant_restored)
      call pretrain_model_destroy(cache_disabled_model)
      call pretrain_model_destroy(context_serial_model)
      call pretrain_model_destroy(context_parallel_model)
      open(newunit=unit, file=PARALLEL_CONTEXT_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=HIERARCHICAL_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=DEBIASED_REPEAT_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=CONTINUATION_A_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=CONTINUATION_B_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=CORPUS_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=CHECKPOINT_PATH, status='old', access='stream', form='unformatted', &
           iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
   end subroutine test_hdc_pretraining

   subroutine assert_checkpoint_header_version(checkpoint_path, expected_version, encoder_name)
      character(len=*), intent(in) :: checkpoint_path, encoder_name
      integer(int32), intent(in) :: expected_version
      character(len=8) :: magic
      integer :: checkpoint_unit, checkpoint_io_status
      integer(int32) :: checkpoint_version

      magic = ''
      checkpoint_version = 0_int32
      open(newunit=checkpoint_unit, file=checkpoint_path, status='old', access='stream', &
           form='unformatted', action='read', iostat=checkpoint_io_status)
      call assert_true(checkpoint_io_status == 0, trim(encoder_name) // ' checkpoint header opens')
      if (checkpoint_io_status /= 0) return
      read(checkpoint_unit, iostat=checkpoint_io_status) magic, checkpoint_version
      close(checkpoint_unit)
      call assert_true(checkpoint_io_status == 0 .and. magic == 'VGHDC001', &
                       trim(encoder_name) // ' checkpoint header is readable')
      if (checkpoint_io_status == 0 .and. magic == 'VGHDC001') then
         call assert_i32_equal(checkpoint_version, expected_version, &
                               trim(encoder_name) // ' checkpoint has the expected format version')
      end if
   end subroutine assert_checkpoint_header_version

   subroutine test_turkish_text_boundary()
      character(len=64) :: input, folded
      integer(int32) :: status

      ! Construct UTF-8 bytes directly so this test does not depend on the
      ! source-file locale. It covers standard Turkish diacritics, Turkish
      ! circumflex spellings, and a public-source Romanian comma-below-s typo.
      input = 'h' // achar(195) // achar(162) // 'l' // achar(195) // achar(162) // ' ' // &
              achar(195) // achar(174) // ' ' // achar(195) // achar(187) // ' ' // achar(200) // achar(153)
      call turkish_fold_prompt_utf8(input, folded, status)
      call assert_i32_equal(status, TURKISH_TEXT_OK, 'Turkish orthographic UTF-8 variants fold successfully')
      call assert_true(trim(folded) == 'hala i u s', 'Turkish orthographic variants retain deterministic ASCII projection')
      input = achar(194) // achar(128)
      call turkish_validate_display_utf8(input, 64_int32, status)
      call assert_i32_equal(status, TURKISH_TEXT_ERR_CONTROL, 'UTF-8 encoded C1 display control is rejected')
   end subroutine test_turkish_text_boundary

   subroutine test_dialogue_retrieval()
      character(len=*), parameter :: CORPUS_PATH = 'vegai_dialogue_test_corpus.txt'
      character(len=*), parameter :: KNOWLEDGE_PATH = 'vegai_dialogue_test_knowledge.tsv'
      character(len=*), parameter :: REJECTED_KNOWLEDGE_PATH = 'vegai_dialogue_rejected_knowledge.tsv'
      type(hdc_pretrain_model_t) :: model
      type(hdc_pretrain_config_t) :: training_config
      type(pretrain_metrics_t) :: vocabulary_metrics, training_metrics
      type(dialogue_config_t) :: dialogue_config, nonfinite_dialogue_config, hybrid_config
      type(dialogue_index_t) :: index, loaded_index, rejected_index, hybrid_index
      type(dialogue_session_t) :: session, loaded_session, hybrid_session
      type(dialogue_answer_t) :: answer
      character(len=DIALOGUE_MAX_PROMPT_CHARS) :: utf8_prompt, utf8_query
      character(len=DIALOGUE_MAX_RESPONSE_CHARS) :: utf8_response
      integer :: unit, io_status
      integer(int32) :: status, entries_loaded

      open(newunit=unit, file=CORPUS_PATH, status='replace', action='write', &
           form='formatted', iostat=io_status)
      call assert_true(io_status == 0, 'dialogue retrieval corpus fixture opens')
      if (io_status /= 0) return
      write(unit, '(A)') 'alpha beta gamma'
      write(unit, '(A)') 'alpha beta delta'
      write(unit, '(A)') 'omega psi phi'
      write(unit, '(A)') 'omega psi tau'
      write(unit, '(A)') 'gamma delta theta'
      write(unit, '(A)') 'ogrenci kayit nasil yapilir'
      write(unit, '(A)') 'ogrenci ders program nerede'
      close(unit)

      training_config = hdc_pretrain_config_t()
      training_config%max_vocabulary = 16_int32
      training_config%min_frequency = 1_int64
      training_config%context_window = 2_int32
      training_config%position_shift_bits = 41_int32
      training_config%epochs = 1_int32
      training_config%initial_plasticity_percent = 25_int32
      training_config%minimum_plasticity_percent = 1_int32
      training_config%plasticity_decay_events = 8_int32
      training_config%strict_tokenization = .true.
      call pretrain_build_vocabulary(model, CORPUS_PATH, training_config, vocabulary_metrics, status)
      call assert_i32_equal(status, PT_OK, 'dialogue fixture vocabulary builds')
      call pretrain_train_file(model, CORPUS_PATH, training_config, training_metrics, status)
      call assert_i32_equal(status, PT_OK, 'dialogue fixture HDC training succeeds')

      nonfinite_dialogue_config = dialogue_config_t()
      nonfinite_dialogue_config%minimum_similarity = ieee_value(0.0_real32, ieee_quiet_nan)
      call dialogue_index_init(rejected_index, nonfinite_dialogue_config, status)
      call assert_i32_equal(status, DIALOGUE_ERR_INVALID_ARGUMENT, &
                            'NaN dialogue similarity threshold is rejected fail-closed')
      call assert_true(.not. dialogue_index_is_initialized(rejected_index), &
                       'NaN threshold does not allocate a dialogue index')
      nonfinite_dialogue_config = dialogue_config_t()
      nonfinite_dialogue_config%minimum_margin = ieee_value(0.0_real32, ieee_positive_inf)
      call dialogue_index_init(rejected_index, nonfinite_dialogue_config, status)
      call assert_i32_equal(status, DIALOGUE_ERR_INVALID_ARGUMENT, &
                            'infinite dialogue margin threshold is rejected fail-closed')
      nonfinite_dialogue_config = dialogue_config_t()
      nonfinite_dialogue_config%lexical_weight = ieee_value(0.0_real32, ieee_quiet_nan)
      call dialogue_index_init(rejected_index, nonfinite_dialogue_config, status)
      call assert_i32_equal(status, DIALOGUE_ERR_INVALID_ARGUMENT, &
                            'NaN hybrid lexical weight is rejected fail-closed')

      dialogue_config = dialogue_config_t()
      dialogue_config%max_entries = 3_int32
      dialogue_config%max_history_turns = 2_int32
      dialogue_config%minimum_similarity = -1.0
      dialogue_config%minimum_margin = 0.0
      call dialogue_index_init(index, dialogue_config, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'bounded dialogue index initializes')
      call assert_true(dialogue_index_is_initialized(index), 'dialogue index reports initialized')
      call assert_true(dialogue_index_footprint_bytes(index) > 0_int64, 'dialogue index reports bounded footprint')
      call dialogue_index_add(index, model, 'alpha beta', 'first response comes from data', status)
      call assert_i32_equal(status, DIALOGUE_OK, 'first data-driven dialogue row registers')
      call dialogue_index_add(index, model, 'omega psi', 'second response comes from data', status)
      call assert_i32_equal(status, DIALOGUE_OK, 'second data-driven dialogue row registers')
      call dialogue_index_add(index, model, 'alpha beta', 'ambiguous duplicate response', status)
      call assert_i32_equal(status, DIALOGUE_ERR_DUPLICATE_PROMPT, 'duplicate prompt embedding is rejected')
      call dialogue_index_add(index, model, 'gamma delta', 'third response comes from data', status)
      call assert_i32_equal(status, DIALOGUE_OK, 'third data-driven dialogue row registers')
      call dialogue_index_add(index, model, 'alpha gamma', 'over-capacity response', status)
      call assert_i32_equal(status, DIALOGUE_ERR_CAPACITY, 'bounded dialogue index rejects excess rows')

      call dialogue_session_init(session, dialogue_config%max_history_turns, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'bounded dialogue session initializes')
      call dialogue_ask(index, model, session, 'alpha beta', answer, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'exact dialogue prompt retrieves')
      call assert_true(answer%found .and. trim(answer%response) == 'first response comes from data', &
                       'dialogue answer is the registered data payload rather than code text')
      call assert_true(answer%query_known_tokens == 2_int32 .and. answer%query_ignored_tokens == 0_int32, &
                       'dialogue prompt reports active and ignored token accounting')
      call assert_true(session%count == 1_int32, 'first dialogue prompt enters bounded session memory')
      call dialogue_ask(index, model, session, 'omega psi', answer, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'second dialogue prompt accepts bounded history')
      call dialogue_ask(index, model, session, 'gamma delta', answer, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'third dialogue prompt wraps bounded history')
      call assert_true(session%count == 2_int32, 'dialogue history remains bounded at configured turn count')
      call dialogue_session_reset(session, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'dialogue session reset succeeds')
      call dialogue_ask(index, model, session, 'alpha beta', answer, status)
      call assert_true(answer%found .and. trim(answer%response) == 'first response comes from data', &
                       'session reset restores stateless exact retrieval')
      call dialogue_ask(index, model, session, 'unseen', answer, status)
      call assert_i32_equal(status, DIALOGUE_ERR_NO_KNOWN_TOKENS, 'prompt with no active checkpoint token is rejected')

      open(newunit=unit, file=KNOWLEDGE_PATH, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      call assert_true(io_status == 0, 'dialogue TSV fixture opens')
      if (io_status == 0) then
         write(unit) 'alpha beta' // achar(9) // 'loaded response from TSV' // achar(10)
         write(unit) 'omega psi' // achar(9) // 'second loaded response from TSV' // achar(10)
         close(unit)
      end if
      dialogue_config%max_entries = 3_int32
      dialogue_config%max_history_turns = 0_int32
      call dialogue_index_init(loaded_index, dialogue_config, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'TSV dialogue index initializes')
      call dialogue_index_load_tsv(loaded_index, model, KNOWLEDGE_PATH, entries_loaded, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'strict TSV knowledge base loads')
      call assert_i32_equal(entries_loaded, 2_int32, 'TSV loader publishes each data row')
      call dialogue_session_init(loaded_session, 0_int32, status)
      call dialogue_ask(loaded_index, model, loaded_session, 'alpha beta', answer, status)
      call assert_true(answer%found .and. trim(answer%response) == 'loaded response from TSV', &
                       'TSV content controls the user-visible dialogue answer')

      ! Build Turkish UTF-8 strings with raw byte constructors so the native
      ! test remains source-encoding-independent. The trained checkpoint keeps
      ! an ASCII projection, while the dialogue boundary accepts/display
      ! preserves Turkish characters.
      utf8_prompt = achar(195) // achar(150) // achar(196) // achar(159) // 'renci kay' // achar(196) // &
                    achar(177) // 't nas' // achar(196) // achar(177) // 'l yap' // achar(196) // achar(177) // &
                    'l' // achar(196) // achar(177) // 'r'
      utf8_query = achar(195) // achar(150) // achar(196) // achar(159) // 'rencinin kayd' // achar(196) // &
                   achar(177) // ' nas' // achar(196) // achar(177) // 'l yap' // achar(196) // achar(177) // 'labilir'
      utf8_response = achar(195) // achar(150) // achar(196) // achar(159) // 'renci kayd' // achar(196) // &
                      achar(177) // ' i' // achar(195) // achar(167) // 'in veri taban' // achar(196) // achar(177) // &
                      'ndaki kayna' // achar(196) // achar(159) // 'a bakiniz.'
      hybrid_config = dialogue_config_t()
      hybrid_config%max_entries = 2_int32
      hybrid_config%max_history_turns = 0_int32
      hybrid_config%minimum_similarity = -1.0_real32
      hybrid_config%minimum_margin = 0.0_real32
      hybrid_config%semantic_weight = 0.25_real32
      hybrid_config%lexical_weight = 0.75_real32
      hybrid_config%minimum_hybrid_score = 0.50_real32
      hybrid_config%minimum_hybrid_margin = 0.00_real32
      call dialogue_index_init(hybrid_index, hybrid_config, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'hybrid Turkish dialogue index initializes')
      call dialogue_index_add(hybrid_index, model, utf8_prompt, utf8_response, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'UTF-8 Turkish prompt and response register natively')
      call dialogue_index_add(hybrid_index, model, 'ogrenci ders program nerede', 'second Turkish retrieval target', status)
      call assert_i32_equal(status, DIALOGUE_OK, 'hybrid comparison prompt registers')
      call dialogue_session_init(hybrid_session, 0_int32, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'stateless hybrid dialogue session initializes')
      call dialogue_ask(hybrid_index, model, hybrid_session, utf8_prompt, answer, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'exact UTF-8 Turkish prompt is accepted')
      call assert_true(answer%found .and. trim(answer%matched_prompt) == trim(utf8_prompt) .and. &
                       trim(answer%response) == trim(utf8_response), &
                       'UTF-8 Turkish source text is retained verbatim in grounded output')
      call dialogue_ask(hybrid_index, model, hybrid_session, utf8_query, answer, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'morphological Turkish hybrid query is accepted')
      call assert_true(answer%found .and. trim(answer%response) == trim(utf8_response) .and. &
                       answer%lexical_similarity > 0.65_real32 .and. answer%hybrid_score > 0.50_real32 .and. &
                       answer%margin >= 0.0_real32 .and. answer%margin == answer%hybrid_margin, &
                       'hybrid retrieval uses transparent Turkish lexical/morphology evidence and active-policy margin')

      open(newunit=unit, file=REJECTED_KNOWLEDGE_PATH, status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      call assert_true(io_status == 0, 'rejected dialogue TSV fixture opens')
      if (io_status == 0) then
         write(unit) 'alpha unseen' // achar(9) // 'this row must not be partially indexed' // achar(10)
         close(unit)
      end if
      call dialogue_index_init(rejected_index, dialogue_config, status)
      call assert_i32_equal(status, DIALOGUE_OK, 'rejected TSV index initializes')
      call dialogue_index_load_tsv(rejected_index, model, REJECTED_KNOWLEDGE_PATH, entries_loaded, status)
      call assert_i32_equal(status, DIALOGUE_ERR_UNKNOWN_TOKEN, 'TSV prompt with checkpoint OOV token is rejected')
      call assert_i32_equal(entries_loaded, 0_int32, 'failed strict TSV load publishes no partial entry count')
      call assert_true(.not. dialogue_index_is_initialized(rejected_index), &
                       'failed strict TSV load destroys partial knowledge index')

      call dialogue_session_destroy(session)
      call dialogue_session_destroy(loaded_session)
      call dialogue_session_destroy(hybrid_session)
      call dialogue_index_destroy(index)
      call dialogue_index_destroy(loaded_index)
      call dialogue_index_destroy(rejected_index)
      call dialogue_index_destroy(hybrid_index)
      call pretrain_model_destroy(model)
      open(newunit=unit, file=CORPUS_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=KNOWLEDGE_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=REJECTED_KNOWLEDGE_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
   end subroutine test_dialogue_retrieval

end program test_all
