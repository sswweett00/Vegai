! SPDX-License-Identifier: MIT
!==============================================================================
!  Fortran 2023 executable verification demo for Vegai HDS-Fortran.
!==============================================================================
program vegai_hds_demo
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64, output_unit
   use hyper_types
   use symbolic_memory
   use reasoning_core
   use resonator_network
   implicit none

   integer(int32), parameter :: MAX_CONCEPTS = 1024_int32
   integer(int32), parameter :: CHAIN_LENGTH = 4_int32
   integer(int32), parameter :: EXPECTED_CHAIN(CHAIN_LENGTH) = [ &
      402_int32, 403_int32, 404_int32, 405_int32 ]
   integer(int32), parameter :: RESONATOR_FACTORS = 3_int32
   integer(int32), parameter :: EXPECTED_RESONATOR(RESONATOR_FACTORS) = [ &
      301_int32, 402_int32, 203_int32 ]

   type(hds_memory_t) :: memory
   type(hds_memory_t) :: resonator_books(RESONATOR_FACTORS)
   type(cleanup_result_t) :: result
   type(resonator_config_t) :: resonator_config
   type(resonator_result_t) :: resonator_result
   type(hv_t) :: r_unit, r_function, r_output
   type(hv_t) :: s_alu, s_arithmetic, s_flags
   type(hv_t) :: s_hippocampus, s_consolidation, s_long_memory
   type(hv_t) :: s_sensor_anomaly, s_heat_rise, s_pressure_spike
   type(hv_t) :: s_open_valve, s_shutdown, dummy
   type(hv_t) :: roles(3), cpu_fillers(3), brain_fillers(3)
   type(hv_t) :: cpu_frame, brain_frame, noisy_cpu_frame, analogy_mapping
   type(hv_t) :: causes(CHAIN_LENGTH), effects(CHAIN_LENGTH)
   type(hv_t) :: causal_bank, current_state, next_state
   type(hv_t) :: resonator_target

   integer(int32) :: status, index, flipped_bits
   integer(int64) :: start_count, end_count, clock_rate
   real(real64) :: elapsed_us
   logical :: all_passed

   all_passed = .true.
   call mem_init(memory, MAX_CONCEPTS, status)
   call require_mem_ok(status, 'mem_init')

   write(output_unit, '(A)') '========================================================================'
   write(output_unit, '(A)') ' VEGAI HDS-FORTRAN 2023 | VERIFIED 16,384-BIT SYMBOLIC ENGINE'
   write(output_unit, '(A)') '========================================================================'
   write(output_unit, '(A,I0,A)') ' Hypervector dimension       : ', HV_BITS, ' bits'
   write(output_unit, '(A,I0,A)') ' Packed vector payload       : ', HV_BYTES, ' bytes (32 x 64-byte lines)'
   write(output_unit, '(A,I0,A)') ' Reserved symbol capacity    : ', MAX_CONCEPTS, ' symbols'
   write(output_unit, '(A,F10.2,A)') ' Allocated memory footprint  : ', &
      real(mem_footprint_bytes(memory), real64) / 1024.0_real64, ' KiB'
   write(output_unit, '(A)') '------------------------------------------------------------------------'

   call system_clock(count_rate=clock_rate)
   call system_clock(count=start_count)

   ! Roles.
   call add_symbol(101_int32, 'ROLE:PROCESSING_UNIT', r_unit)
   call add_symbol(102_int32, 'ROLE:PRIMARY_FUNCTION', r_function)
   call add_symbol(103_int32, 'ROLE:GENERATED_OUTPUT', r_output)

   ! CPU domain.
   call add_symbol(201_int32, 'HARDWARE:ALU_CORE', s_alu)
   call add_symbol(202_int32, 'FUNCTION:LOGIC_ARITHMETIC', s_arithmetic)
   call add_symbol(203_int32, 'OUTPUT:STATUS_FLAGS', s_flags)

   ! Cognitive-neuroscience domain.
   call add_symbol(301_int32, 'NEURO:HIPPOCAMPUS', s_hippocampus)
   call add_symbol(302_int32, 'FUNCTION:MEMORY_CONSOLIDATION', s_consolidation)
   call add_symbol(303_int32, 'OUTPUT:EPISODIC_TRACE', s_long_memory)

   ! Causal-chain vocabulary.
   call add_symbol(401_int32, 'EVENT:SENSOR_ANOMALY', s_sensor_anomaly)
   call add_symbol(402_int32, 'EVENT:CRITICAL_HEAT_RISE', s_heat_rise)
   call add_symbol(403_int32, 'EVENT:PRESSURE_SPIKE', s_pressure_spike)
   call add_symbol(404_int32, 'ACTION:OPEN_RELIEF_VALVE', s_open_valve)
   call add_symbol(405_int32, 'ACTION:REACTOR_SHUTDOWN', s_shutdown)

   ! Fill the cleanup memory with realistic unrelated vocabulary.  This makes
   ! the three assertions actual associative-retrieval tests, not tiny demos.
   do index = 500_int32, 999_int32
      call mem_register_symbol(memory, index, 'BACKGROUND_CONCEPT', dummy, status)
      call require_mem_ok(status, 'background symbol registration')
   end do

   !==========================================================================
   ! TEST 1: Multi-role frame retrieval after exactly 25 percent bit noise.
   !==========================================================================
   roles = [r_unit, r_function, r_output]
   cpu_fillers = [s_alu, s_arithmetic, s_flags]
   brain_fillers = [s_hippocampus, s_consolidation, s_long_memory]

   call encode_role_filler_frame(roles, cpu_fillers, cpu_frame, status)
   call require_reason_ok(status, 'encode CPU frame')
   call encode_role_filler_frame(roles, brain_fillers, brain_frame, status)
   call require_reason_ok(status, 'encode brain frame')

   call hv_inject_noise_percent(cpu_frame, 25_int32, 135792468_int64, &
                                noisy_cpu_frame, status)
   call require_hv_ok(status, 'inject exact 25 percent noise')
   flipped_bits = hv_hamming(cpu_frame, noisy_cpu_frame)

   call query_frame_role(memory, noisy_cpu_frame, r_function, result, status)
   call require_mem_ok(status, 'frame role cleanup')

   write(output_unit, '(A)') '[TEST 1] NOISY ROLE-FILLER FRAME UNBINDING'
   write(output_unit, '(A,I0,A)') ' Exact injected corruption : ', flipped_bits, ' bits'
   call report_result(' Frame_CPU XOR ROLE:PRIMARY_FUNCTION', result, 202_int32, all_passed)
   write(output_unit, '(A)') '------------------------------------------------------------------------'

   !==========================================================================
   ! TEST 2: Aligned structural analogy.  The mapping is a bundle of aligned
   ! source-target relations; it is deliberately not a XOR of holistic frames.
   !==========================================================================
   call encode_analogy_mapping(cpu_fillers, brain_fillers, analogy_mapping, status)
   call require_reason_ok(status, 'encode aligned analogy mapping')
   call solve_analogy_zero_shot(memory, analogy_mapping, s_alu, result, status)
   call require_mem_ok(status, 'analogy cleanup')

   write(output_unit, '(A)') '[TEST 2] ZERO-SHOT ALIGNED ANALOGY'
   call report_result(' ALU_CORE maps to', result, 301_int32, all_passed)
   write(output_unit, '(A)') '------------------------------------------------------------------------'

   !==========================================================================
   ! TEST 3: Four-rule bank and cleanup after every forward-chaining step.
   !==========================================================================
   causes = [s_sensor_anomaly, s_heat_rise, s_pressure_spike, s_open_valve]
   effects = [s_heat_rise, s_pressure_spike, s_open_valve, s_shutdown]
   call encode_causal_rule_bank(causes, effects, causal_bank, status)
   call require_reason_ok(status, 'encode causal rule bank')

   write(output_unit, '(A)') '[TEST 3] FOUR-STEP CAUSAL FORWARD CHAINING'
   write(output_unit, '(A)') ' Start: EVENT:SENSOR_ANOMALY'
   current_state = s_sensor_anomaly
   do index = 1_int32, CHAIN_LENGTH
      call infer_causal_step(memory, causal_bank, current_state, next_state, result, status)
      call require_mem_ok(status, 'causal cleanup')
      call report_chain_step(index, result, EXPECTED_CHAIN(index), all_passed)
      current_state = next_state
   end do
   write(output_unit, '(A)') '------------------------------------------------------------------------'

   !==========================================================================
   ! TEST 4: Three-factor Resonator Network decomposition.  Each factor slot
   ! owns an independent candidate codebook; the verifier certifies the tuple.
   !==========================================================================
   do index = 1_int32, RESONATOR_FACTORS
      call mem_init(resonator_books(index), 3_int32, status)
      call require_mem_ok(status, 'resonator codebook initialization')
   end do

   call mem_register_symbol(resonator_books(1), 201_int32, 'F1:ALU', dummy, status, s_alu)
   call require_mem_ok(status, 'resonator factor-1 registration')
   call mem_register_symbol(resonator_books(1), 301_int32, 'F1:HIPPOCAMPUS', dummy, status, s_hippocampus)
   call require_mem_ok(status, 'resonator factor-1 registration')
   call mem_register_symbol(resonator_books(1), 401_int32, 'F1:ANOMALY', dummy, status, s_sensor_anomaly)
   call require_mem_ok(status, 'resonator factor-1 registration')

   call mem_register_symbol(resonator_books(2), 202_int32, 'F2:ARITHMETIC', dummy, status, s_arithmetic)
   call require_mem_ok(status, 'resonator factor-2 registration')
   call mem_register_symbol(resonator_books(2), 302_int32, 'F2:CONSOLIDATION', dummy, status, s_consolidation)
   call require_mem_ok(status, 'resonator factor-2 registration')
   call mem_register_symbol(resonator_books(2), 402_int32, 'F2:HEAT_RISE', dummy, status, s_heat_rise)
   call require_mem_ok(status, 'resonator factor-2 registration')

   call mem_register_symbol(resonator_books(3), 203_int32, 'F3:FLAGS', dummy, status, s_flags)
   call require_mem_ok(status, 'resonator factor-3 registration')
   call mem_register_symbol(resonator_books(3), 303_int32, 'F3:EPISODIC_TRACE', dummy, status, s_long_memory)
   call require_mem_ok(status, 'resonator factor-3 registration')
   call mem_register_symbol(resonator_books(3), 403_int32, 'F3:PRESSURE', dummy, status, s_pressure_spike)
   call require_mem_ok(status, 'resonator factor-3 registration')

   resonator_target = hv_bind(hv_bind(s_hippocampus, s_heat_rise), s_flags)
   resonator_config%max_iterations = 16_int32
   resonator_config%restarts = 4_int32
   resonator_config%max_exact_combinations = 64_int32
   resonator_config%synchronous_updates = .true.
   resonator_config%enable_exact_verifier = .true.
   call resonator_factorize(resonator_target, resonator_books, resonator_config, &
                            resonator_result, status)
   call require_resonator_ok(status, 'three-factor decomposition')

   write(output_unit, '(A)') '[TEST 4] RESONATOR NETWORK MULTI-FACTOR DECOMPOSITION'
   call report_resonator(resonator_result, EXPECTED_RESONATOR, all_passed)

   call system_clock(count=end_count)
   if (clock_rate > 0_int64) then
      elapsed_us = real(end_count - start_count, real64) * 1.0e6_real64 / real(clock_rate, real64)
   else
      elapsed_us = -1.0_real64
   end if

   write(output_unit, '(A)') '------------------------------------------------------------------------'
   if (elapsed_us >= 0.0_real64) then
      write(output_unit, '(A,F12.2,A)') ' Registration + all cognitive tests: ', elapsed_us, ' microseconds'
   else
      write(output_unit, '(A)') ' Registration + all cognitive tests: clock unavailable'
   end if
   if (all_passed) then
      write(output_unit, '(A)') ' VERIFICATION RESULT: PASS'
   else
      write(output_unit, '(A)') ' VERIFICATION RESULT: FAIL'
   end if
   write(output_unit, '(A)') '========================================================================'

   call resonator_result_destroy(resonator_result)
   do index = 1_int32, RESONATOR_FACTORS
      call mem_destroy(resonator_books(index))
   end do
   call mem_destroy(memory)
   if (.not. all_passed) error stop 1

contains

   subroutine add_symbol(symbol_id, label, vector)
      integer(int32), intent(in) :: symbol_id
      character(len=*), intent(in) :: label
      type(hv_t), intent(out) :: vector

      call mem_register_symbol(memory, symbol_id, label, vector, status)
      call require_mem_ok(status, 'symbol registration')
   end subroutine add_symbol

   subroutine require_mem_ok(code, operation)
      integer(int32), intent(in) :: code
      character(len=*), intent(in) :: operation

      if (code /= MEM_OK) then
         write(output_unit, '(A,A,A,I0)') 'FATAL memory error in ', trim(operation), ': ', code
         error stop 2
      end if
   end subroutine require_mem_ok

   subroutine require_reason_ok(code, operation)
      integer(int32), intent(in) :: code
      character(len=*), intent(in) :: operation

      if (code /= REASON_OK) then
         write(output_unit, '(A,A,A,I0)') 'FATAL reasoning error in ', trim(operation), ': ', code
         error stop 3
      end if
   end subroutine require_reason_ok

   subroutine require_hv_ok(code, operation)
      integer(int32), intent(in) :: code
      character(len=*), intent(in) :: operation

      if (code /= HV_OK) then
         write(output_unit, '(A,A,A,I0)') 'FATAL hypervector error in ', trim(operation), ': ', code
         error stop 4
      end if
   end subroutine require_hv_ok

   subroutine require_resonator_ok(code, operation)
      integer(int32), intent(in) :: code
      character(len=*), intent(in) :: operation

      if (code < RN_OK) then
         write(output_unit, '(A,A,A,I0)') 'FATAL resonator error in ', trim(operation), ': ', code
         error stop 5
      end if
   end subroutine require_resonator_ok

   subroutine report_resonator(network_result, expected_ids, passed)
      type(resonator_result_t), intent(in) :: network_result
      integer(int32), intent(in) :: expected_ids(:)
      logical, intent(inout) :: passed
      integer(int32) :: factor

      write(output_unit, '(A,L1,A,L1,A,I0,A,I0,A,F7.3)') ' Exact=', network_result%exact, &
         ' | Network converged=', network_result%network_converged, ' | network Hamming=', &
         network_result%network_best_distance, ' | final Hamming=', &
         network_result%hamming_distance, ' | sim=', network_result%similarity
      write(output_unit, '(A,L1,A,I0,A,I0)') ' Exact verifier=', &
         network_result%used_exact_verifier, ' | iterations=', network_result%iterations, &
         ' | restarts=', network_result%restarts_attempted
      do factor = 1, network_result%n_factors
         write(output_unit, '(A,I0,A,A,A,I0,A,I0)') ' Factor ', factor, ' -> ', &
            trim(network_result%factor_labels(factor)), ' [ID=', &
            network_result%factor_ids(factor), '] slot=', network_result%factor_slots(factor)
         if (factor > size(expected_ids, kind=int32)) then
            write(output_unit, '(A)') '   Assertion: FAIL (missing expected factor ID)'
            passed = .false.
         else if (network_result%factor_ids(factor) /= expected_ids(factor)) then
            write(output_unit, '(A)') '   Assertion: FAIL'
            passed = .false.
         end if
      end do
      if (network_result%exact .and. network_result%hamming_distance == 0_int32) then
         write(output_unit, '(A)') ' Assertion: PASS'
      else
         write(output_unit, '(A)') ' Assertion: FAIL (non-zero resonator residual)'
         passed = .false.
      end if
   end subroutine report_resonator

   subroutine report_result(prefix, cleanup, expected_id, passed)
      character(len=*), intent(in) :: prefix
      type(cleanup_result_t), intent(in) :: cleanup
      integer(int32), intent(in) :: expected_id
      logical, intent(inout) :: passed

      write(output_unit, '(A,A,A,A,I0,A,F7.3,A,I0)') trim(prefix), ' -> ', &
         trim(cleanup%label), ' [ID=', cleanup%id, '] sim=', cleanup%similarity, &
         ' distance=', cleanup%distance
      if (cleanup%id == expected_id) then
         write(output_unit, '(A)') ' Assertion: PASS'
      else
         write(output_unit, '(A,I0,A,I0)') ' Assertion: FAIL (expected ID ', expected_id, &
                                            ', got ', cleanup%id
         passed = .false.
      end if
   end subroutine report_result

   subroutine report_chain_step(step, cleanup, expected_id, passed)
      integer(int32), intent(in) :: step, expected_id
      type(cleanup_result_t), intent(in) :: cleanup
      logical, intent(inout) :: passed

      write(output_unit, '(A,I0,A,A,A,I0,A,F7.3)') ' Step ', step, ' -> ', &
         trim(cleanup%label), ' [ID=', cleanup%id, '] sim=', cleanup%similarity
      if (cleanup%id /= expected_id) then
         write(output_unit, '(A,I0,A,I0)') '   FAIL: expected ID ', expected_id, ', got ', cleanup%id
         passed = .false.
      end if
   end subroutine report_chain_step

end program vegai_hds_demo
