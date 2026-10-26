program apex_benchmark
   use, intrinsic :: iso_fortran_env, only : real64, int64, output_unit, error_unit
   use apex_optimizer, only : apex_state_t, apex_config_t, apex_error_t, apex_initialize, apex_update, &
                              apex_get_parameters, apex_update_count, APEX_OK
   implicit none

   integer(int64), parameter :: element_count = 10000000_int64
   integer(int64), parameter :: repetitions = 5_int64
   real(real64), allocatable :: initial(:), gradient(:), final_parameters(:)
   type(apex_state_t) :: state
   type(apex_config_t) :: config
   type(apex_error_t) :: error
   integer(int64) :: index, iteration, clock_start, clock_end, clock_rate
   real(real64) :: elapsed_seconds, objective
   integer :: allocation_status

   allocate(initial(element_count), gradient(element_count), final_parameters(element_count), stat=allocation_status)
   if (allocation_status /= 0) then
      write(error_unit, '(a)') 'Apex benchmark: allocation failed'
      stop 1
   end if
   do index = 1_int64, element_count
      initial(index) = 0.25_real64 * sin(real(index, real64) * 1.0e-5_real64)
      gradient(index) = 0.25_real64 * cos(real(index, real64) * 7.0e-6_real64)
   end do
   config%learning_rate = 1.0e-3_real64
   config%beta1 = 0.9_real64
   config%beta2 = 0.5_real64
   config%gamma = 0.5_real64
   config%gradient_clip = 1.0_real64
   call apex_initialize(state, initial, config, error)
   if (error%code /= APEX_OK) then
      write(error_unit, '(a,a)') 'Apex benchmark initialization failed: ', trim(error%message)
      stop 1
   end if

   call system_clock(count_rate=clock_rate)
   call system_clock(clock_start)
   do iteration = 1_int64, repetitions
      call apex_update(state, gradient, error)
      if (error%code /= APEX_OK) then
         write(error_unit, '(a,a)') 'Apex benchmark update failed: ', trim(error%message)
         stop 1
      end if
   end do
   call system_clock(clock_end)
   elapsed_seconds = real(clock_end - clock_start, real64) / real(clock_rate, real64)
   call apex_get_parameters(state, final_parameters, error)
   if (error%code /= APEX_OK) then
      write(error_unit, '(a,a)') 'Apex benchmark readback failed: ', trim(error%message)
      stop 1
   end if
   objective = 0.5_real64 * sum(final_parameters * final_parameters) / real(element_count, real64)

   write(output_unit, '(a,i0)') 'elements: ', element_count
   write(output_unit, '(a,i0)') 'iterations: ', apex_update_count(state)
   write(output_unit, '(a,f12.3)') 'Apex elapsed (ms): ', elapsed_seconds * 1000.0_real64
   write(output_unit, '(a,f12.3)') 'Apex modeled state-traffic lower bound (GB/s, 5N transfers): ', &
      real(5_int64 * element_count * 8_int64 * repetitions, real64) / elapsed_seconds / 1.0e9_real64
   write(output_unit, '(a,es16.8)') 'final mean quadratic objective: ', objective
   write(output_unit, '(a)') 'Synthetic single-host measurement; not a comparative performance claim.'
end program apex_benchmark
