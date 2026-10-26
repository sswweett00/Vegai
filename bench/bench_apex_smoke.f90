program bench_apex_smoke
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64, error_unit
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   use apex_optimizer, only : apex_state_t, apex_config_t, apex_error_t, apex_initialize, apex_optimize, &
                              apex_get_parameters, APEX_OK
   use bench_contracts, only : bench_trial_t, bench_seed_state, bench_normal, &
                               bench_write_csv_header, bench_write_csv_row, bench_trial_is_valid, BENCH_OK
   implicit none

   integer(int64), parameter :: seeds(5) = [1_int64, 7_int64, 19_int64, 101_int64, 2026_int64]
   integer(int64), parameter :: parameter_count = 128_int64, max_steps = 1000_int64
   character(len=*), parameter :: result_path = 'bench/results/baseline/apex_smoke.csv'
   type(apex_state_t) :: state
   type(apex_config_t) :: config
   type(apex_error_t) :: apex_error
   type(bench_trial_t) :: trial
   real(real64) :: initial(parameter_count), parameters(parameter_count), gradient(parameter_count), loss
   integer(int64) :: rng, seed, clock_start, clock_end, clock_rate
   integer :: unit, io_status, i
   integer(int32) :: status

   config%learning_rate = 0.02_real64
   config%beta1 = 0.9_real64
   config%beta2 = 0.5_real64
   config%epsilon = 1.0e-8_real64
   config%gamma = 0.5_real64
   config%weight_decay = 0.0_real64
   config%gradient_clip = 10.0_real64
   config%scheduler = 0

   open(newunit=unit, file=result_path, status='replace', action='write', iostat=io_status)
   if (io_status /= 0) then
      write(error_unit, '(a,a)') 'Could not open benchmark result file: ', result_path
      stop 1
   end if
   call bench_write_csv_header(unit, status)
   if (status /= BENCH_OK) then
      close(unit)
      write(error_unit, '(a)') 'Could not write benchmark CSV header'
      stop 1
   end if

   call system_clock(count_rate=clock_rate)
   if (clock_rate <= 0_int64) call fail('timer setup', 'system_clock returned a non-positive rate')
   do i = 1, size(seeds)
      seed = seeds(i)
      rng = bench_seed_state(seed)
      call initialize_quadratic()
      call apex_initialize(state, initial, config, apex_error)
      if (apex_error%code /= APEX_OK) call fail('Apex initialization', apex_error%message)
      call quadratic_gradient(initial, gradient, trial%initial_loss, apex_error)
      if (apex_error%code /= APEX_OK) call fail('Initial objective', apex_error%message)
      call system_clock(clock_start)
      call apex_optimize(state, quadratic_gradient, max_steps, apex_error)
      call system_clock(clock_end)
      if (apex_error%code /= APEX_OK) call fail('Apex optimization', apex_error%message)
      call apex_get_parameters(state, parameters, apex_error)
      if (apex_error%code /= APEX_OK) call fail('Apex readback', apex_error%message)
      call quadratic_gradient(parameters, gradient, loss, apex_error)
      if (apex_error%code /= APEX_OK) call fail('Final objective', apex_error%message)

      trial%task = 'quadratic-well-conditioned'
      trial%optimizer = 'Apex'
      trial%seed = seed
      trial%max_steps = max_steps
      trial%steps = max_steps
      trial%state_bytes = 2_int64 * parameter_count * 8_int64
      trial%learning_rate = config%learning_rate
      trial%beta1 = config%beta1
      trial%beta2 = config%beta2
      trial%epsilon = config%epsilon
      trial%final_loss = loss
      trial%elapsed_seconds = real(clock_end - clock_start, real64) / real(clock_rate, real64)
      trial%finite = all(ieee_is_finite(parameters)) .and. ieee_is_finite(loss)
      trial%status = BENCH_OK
      if (.not. trial%finite .or. .not. bench_trial_is_valid(trial)) &
         call fail('trial validation', 'non-finite or inconsistent result record')
      call bench_write_csv_row(unit, trial, status)
      if (status /= BENCH_OK) call fail('CSV row', 'write failed')
   end do
   close(unit, iostat=io_status)
   if (io_status /= 0) then
      write(error_unit, '(a)') 'Could not close benchmark result file cleanly'
      stop 1
   end if

contains

   subroutine initialize_quadratic()
      integer(int64) :: index
      do index = 1_int64, parameter_count
         initial(index) = bench_normal(rng)
      end do
   end subroutine initialize_quadratic

   subroutine quadratic_gradient(x, grad, objective, callback_status)
      real(real64), contiguous, intent(in) :: x(:)
      real(real64), contiguous, intent(out) :: grad(:)
      real(real64), intent(out) :: objective
      type(apex_error_t), intent(out) :: callback_status
      grad = x
      objective = 0.5_real64 * sum(x * x)
      callback_status%code = APEX_OK
      callback_status%message = ''
   end subroutine quadratic_gradient

   subroutine fail(operation, message)
      character(len=*), intent(in) :: operation, message
      close(unit)
      write(error_unit, '(a,a,a,a)') 'Benchmark ', trim(operation), ' failed: ', trim(message)
      stop 1
   end subroutine fail

end program bench_apex_smoke
