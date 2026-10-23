! SPDX-License-Identifier: MIT
program test_all
   use, intrinsic :: iso_fortran_env, only: error_unit, int16, int32, int64
   use hyper_types
   use symbolic_memory
   use reasoning_core
   use resonator_network
   use hdc_pretraining
   use dialogue_retrieval
   use text_tokenizer, only: TOKEN_LEN, MAX_LINE_CHARS, MAX_TOKENS_PER_LINE, tokenize_ascii_line
   use vocabulary
   use test_support
   implicit none

   call test_hypervector_algebra()
   call test_symbolic_memory()
   call test_reasoning_pipeline()
   call test_resonator_network()
   call test_text_ingest()
   call test_hdc_pretraining()
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

   subroutine test_text_ingest()
      type(vocabulary_t) :: vocab
      character(len=TOKEN_LEN) :: tokens(MAX_TOKENS_PER_LINE)
      integer(int32) :: n_tokens, status, slot
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
      call vocab_destroy(vocab)
   end subroutine test_text_ingest

   subroutine test_hdc_pretraining()
      character(len=*), parameter :: CORPUS_PATH = 'vegai_pretraining_test_corpus.txt'
      character(len=*), parameter :: CHECKPOINT_PATH = 'vegai_pretraining_test_checkpoint.vgai'
      character(len=*), parameter :: CONTINUATION_A_PATH = 'vegai_continuation_a.txt'
      character(len=*), parameter :: CONTINUATION_B_PATH = 'vegai_continuation_b.txt'
      character(len=*), parameter :: HIERARCHICAL_PATH = 'vegai_hierarchical_context.txt'
      character(len=*), parameter :: PARALLEL_CONTEXT_PATH = 'vegai_parallel_context.txt'
      type(hdc_pretrain_model_t) :: model_a, model_b, restored, legacy_model, positional_model, continuation_model
      type(hdc_pretrain_model_t) :: hierarchical_model, local_only_model, hierarchical_replay, short_hierarchical_model
      type(hdc_pretrain_model_t) :: coherent_model, coherent_replay, short_coherent_model
      type(hdc_pretrain_model_t) :: context_serial_model, context_parallel_model
      type(hdc_pretrain_config_t) :: config, resumed_config, strict_config, legacy_config, positional_config, continuation_config
      type(hdc_pretrain_config_t) :: reference_config, hierarchical_config, local_only_config, coherent_config
      type(hdc_pretrain_config_t) :: context_serial_config, context_parallel_config
      type(pretrain_metrics_t) :: vocabulary_metrics, train_metrics, extension_metrics
      type(pretrain_execution_report_t) :: default_execution, serial_execution, parallel_execution
      type(hv_t) :: semantic_hv, semantic_before_extension, semantic_after_extension
      character(len=TOKEN_LEN) :: nearest_token
      character(len=8) :: checkpoint_magic
      integer :: unit, io_status
      integer(int32) :: status, slot, nearest_slot, checkpoint_version
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
      call assert_true(model_a%training_epochs_completed == 1_int64, 'one completed epoch is recorded')

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
      call pretrain_checkpoint_save(hierarchical_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'hierarchical checkpoint save succeeds')
      call pretrain_checkpoint_load(CHECKPOINT_PATH, restored, status)
      call assert_i32_equal(status, PT_OK, 'hierarchical checkpoint load succeeds')
      call assert_i32_equal(restored%context_mode, PT_CONTEXT_HIERARCHICAL, &
                            'checkpoint preserves hierarchical encoder semantics')
      call assert_true(all(restored%semantic_bits == hierarchical_model%semantic_bits), &
                       'hierarchical checkpoint restores semantic prototypes')
      call pretrain_checkpoint_save(coherent_model, CHECKPOINT_PATH, status)
      call assert_i32_equal(status, PT_OK, 'coherent checkpoint save succeeds')
      open(newunit=unit, file=CHECKPOINT_PATH, status='old', access='stream', form='unformatted', &
           action='read', iostat=io_status)
      call assert_true(io_status == 0, 'coherent checkpoint header opens')
      if (io_status == 0) then
         read(unit, iostat=io_status) checkpoint_magic, checkpoint_version
         close(unit)
         call assert_true(io_status == 0 .and. checkpoint_magic == 'VGHDC001', &
                          'coherent checkpoint header is readable')
         if (checkpoint_version /= 6_int32) then
            write(error_unit, '(A,I0)') 'ERROR: coherent checkpoint header version=', checkpoint_version
         end if
         call assert_i32_equal(checkpoint_version, 6_int32, 'coherent checkpoint uses format version six')
      end if
      call pretrain_checkpoint_load(CHECKPOINT_PATH, restored, status)
      if (status /= PT_OK) write(error_unit, '(A,I0)') 'ERROR: coherent checkpoint load status=', status
      call assert_i32_equal(status, PT_OK, 'coherent checkpoint load succeeds')
      call assert_i32_equal(restored%context_mode, PT_CONTEXT_COHERENT, &
                            'checkpoint preserves coherent encoder semantics')
      call assert_true(all(restored%semantic_bits == coherent_model%semantic_bits), &
                       'coherent checkpoint restores semantic prototypes')

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

      resumed_config%context_window = resumed_config%context_window + 1_int32
      call pretrain_train_file(restored, CORPUS_PATH, resumed_config, train_metrics, status)
      call assert_i32_equal(status, PT_ERR_INCOMPATIBLE_CONFIG, &
                            'context-window mismatch is rejected on resume')
      resumed_config%context_window = model_a%context_window
      resumed_config%context_mode = PT_CONTEXT_POSITIONAL
      call pretrain_train_file(restored, CORPUS_PATH, resumed_config, train_metrics, status)
      call assert_i32_equal(status, PT_ERR_INCOMPATIBLE_CONFIG, &
                            'context encoder mismatch is rejected on resume')

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
      call pretrain_model_destroy(hierarchical_model)
      call pretrain_model_destroy(local_only_model)
      call pretrain_model_destroy(hierarchical_replay)
      call pretrain_model_destroy(short_hierarchical_model)
      call pretrain_model_destroy(context_serial_model)
      call pretrain_model_destroy(context_parallel_model)
      open(newunit=unit, file=PARALLEL_CONTEXT_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=HIERARCHICAL_PATH, status='old', iostat=io_status)
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

   subroutine test_dialogue_retrieval()
      character(len=*), parameter :: CORPUS_PATH = 'vegai_dialogue_test_corpus.txt'
      character(len=*), parameter :: KNOWLEDGE_PATH = 'vegai_dialogue_test_knowledge.tsv'
      character(len=*), parameter :: REJECTED_KNOWLEDGE_PATH = 'vegai_dialogue_rejected_knowledge.tsv'
      type(hdc_pretrain_model_t) :: model
      type(hdc_pretrain_config_t) :: training_config
      type(pretrain_metrics_t) :: vocabulary_metrics, training_metrics
      type(dialogue_config_t) :: dialogue_config
      type(dialogue_index_t) :: index, loaded_index, rejected_index
      type(dialogue_session_t) :: session, loaded_session
      type(dialogue_answer_t) :: answer
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
      call dialogue_index_destroy(index)
      call dialogue_index_destroy(loaded_index)
      call dialogue_index_destroy(rejected_index)
      call pretrain_model_destroy(model)
      open(newunit=unit, file=CORPUS_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=KNOWLEDGE_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=REJECTED_KNOWLEDGE_PATH, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
   end subroutine test_dialogue_retrieval

end program test_all
