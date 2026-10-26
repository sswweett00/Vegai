program bench_vector_baseline
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64, error_unit
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   use apex_optimizer, only : apex_state_t, apex_config_t, apex_error_t, apex_initialize, apex_optimize, &
                              apex_get_parameters, APEX_OK
   use bench_contracts, only : bench_trial_t, bench_seed_state, bench_normal, &
                               bench_write_csv_header, bench_write_csv_row, bench_trial_is_valid, BENCH_OK
   use ref_vector_optimizers, only : ref_vector_state_t, REF_OK, ref_vector_initialize, ref_vector_step, &
                                     ref_vector_state_bytes, ref_vector_name, REF_SGD, REF_MOMENTUM, &
                                     REF_NESTEROV, REF_ADAGRAD, REF_RMSPROP, REF_ADAM, REF_ADAMW, &
                                     REF_NADAM, REF_RADAM, REF_LION, REF_ADAFACTOR_VECTOR, REF_LAMB, REF_LARS, &
                                     REF_SOPHIA_G, REF_PRODIGY, REF_SCHEDULE_FREE_ADAMW, ref_vector_eval_parameters
   implicit none

   integer(int64), parameter :: seeds(5) = [1_int64, 7_int64, 19_int64, 101_int64, 2026_int64]
   integer(int32), parameter :: reference_kinds(16) = [REF_SGD, REF_MOMENTUM, REF_NESTEROV, REF_ADAGRAD, &
      REF_RMSPROP, REF_ADAM, REF_ADAMW, REF_NADAM, REF_RADAM, REF_LION, REF_ADAFACTOR_VECTOR, &
      REF_LAMB, REF_LARS, REF_SOPHIA_G, REF_PRODIGY, REF_SCHEDULE_FREE_ADAMW]
   integer(int64), parameter :: parameter_count = 128_int64, max_steps = 1000_int64
   character(len=*), parameter :: result_path = 'bench/results/baseline/vector_quadratic.csv'
   type(apex_config_t) :: apex_config
   type(bench_trial_t) :: trial
   real(real64) :: initial(parameter_count), curvature_diagonal(parameter_count)
   integer(int64) :: seed, rng
   integer :: unit, io_status, seed_index, method_index
   integer(int32) :: status

   apex_config%learning_rate = 0.02_real64
   apex_config%beta1 = 0.9_real64
   apex_config%beta2 = 0.99_real64
   apex_config%epsilon = 1.0e-8_real64
   apex_config%gamma = 0.5_real64
   apex_config%weight_decay = 0.0_real64
   apex_config%gradient_clip = 1000.0_real64
   apex_config%scheduler = 0
   curvature_diagonal = 1.0_real64

   open(newunit=unit, file=result_path, status='replace', action='write', iostat=io_status)
   if (io_status /= 0) call fail('result open', result_path)
   call bench_write_csv_header(unit, status)
   if (status /= BENCH_OK) call fail('CSV header', 'write failed')
   do seed_index = 1, size(seeds)
      seed = seeds(seed_index)
      rng = bench_seed_state(seed)
      call fill_initial()
      call run_apex(seed)
      do method_index = 1, size(reference_kinds)
         call run_reference(seed, reference_kinds(method_index))
      end do
   end do
   close(unit, iostat=io_status)
   if (io_status /= 0) call fail('result close', 'I/O failure')

contains

   subroutine fill_initial()
      integer(int64) :: index
      do index = 1_int64, parameter_count
         initial(index) = bench_normal(rng)
      end do
   end subroutine fill_initial

   subroutine run_apex(current_seed)
      integer(int64), intent(in) :: current_seed
      type(apex_state_t) :: apex_state
      type(apex_error_t) :: apex_error
      real(real64) :: parameters(parameter_count), gradient(parameter_count), initial_loss, final_loss
      integer(int64) :: start_tick, end_tick, rate

      call apex_initialize(apex_state, initial, apex_config, apex_error)
      if (apex_error%code /= APEX_OK) call fail('Apex initialize', apex_error%message)
      call quadratic_gradient(initial, gradient, initial_loss, apex_error)
      if (apex_error%code /= APEX_OK) call fail('Apex initial objective', apex_error%message)
      call system_clock(count_rate=rate)
      if (rate <= 0_int64) call fail('clock setup', 'non-positive system_clock rate')
      call system_clock(start_tick)
      call apex_optimize(apex_state, quadratic_gradient, max_steps, apex_error)
      call system_clock(end_tick)
      if (apex_error%code /= APEX_OK) call fail('Apex update', apex_error%message)
      call apex_get_parameters(apex_state, parameters, apex_error)
      if (apex_error%code /= APEX_OK) call fail('Apex readback', apex_error%message)
      call quadratic_gradient(parameters, gradient, final_loss, apex_error)
      if (apex_error%code /= APEX_OK) call fail('Apex final objective', apex_error%message)

      trial = bench_trial_t()
      trial%task = 'quadratic-well-conditioned'
      trial%optimizer = 'Apex'
      trial%seed = current_seed
      trial%max_steps = max_steps
      trial%steps = max_steps
      trial%state_bytes = parameter_count * int(storage_size(0.0_real64) / 8, int64)
      trial%learning_rate = apex_config%learning_rate
      trial%beta1 = apex_config%beta1
      trial%beta2 = apex_config%beta2
      trial%epsilon = apex_config%epsilon
      trial%initial_loss = initial_loss
      trial%final_loss = final_loss
      trial%elapsed_seconds = real(end_tick - start_tick, real64) / real(rate, real64)
      trial%finite = all(ieee_is_finite(parameters)) .and. ieee_is_finite(final_loss)
      trial%status = BENCH_OK
      call emit_trial()
   end subroutine run_apex

   subroutine run_reference(current_seed, kind)
      integer(int64), intent(in) :: current_seed
      integer(int32), intent(in) :: kind
      type(ref_vector_state_t) :: state
      real(real64) :: parameters(parameter_count), gradient(parameter_count), loss, initial_loss
      integer(int64) :: step, start_tick, end_tick, rate
      integer(int32) :: ref_status

      parameters = initial
      call ref_vector_initialize(kind, parameter_count, state, ref_status)
      if (ref_status /= REF_OK) call fail('reference initialize', trim(ref_vector_name(kind)))
      initial_loss = quadratic_loss(parameters)
      call system_clock(count_rate=rate)
      if (rate <= 0_int64) call fail('clock setup', 'non-positive system_clock rate')
      call system_clock(start_tick)
      do step = 1_int64, max_steps
         gradient = parameters
         loss = quadratic_loss(parameters)
         if (.not. ieee_is_finite(loss)) call fail('reference objective', 'non-finite loss')
         call ref_vector_step(parameters, gradient, state, apex_config%learning_rate, apex_config%beta1, &
                              0.99_real64, apex_config%epsilon, 0.0_real64, ref_status, &
                              curvature_diagonal, 0.04_real64)
         if (ref_status /= REF_OK) call fail('reference update', trim(ref_vector_name(kind)))
      end do
      call system_clock(end_tick)
      if (kind == REF_SCHEDULE_FREE_ADAMW) then
         call ref_vector_eval_parameters(parameters, state, 0.9_real64, ref_status)
         if (ref_status /= REF_OK) call fail('Schedule-Free eval conversion', 'invalid state')
      end if
      loss = quadratic_loss(parameters)

      trial = bench_trial_t()
      trial%task = 'quadratic-well-conditioned'
      trial%optimizer = trim(ref_vector_name(kind))
      trial%seed = current_seed
      trial%max_steps = max_steps
      trial%steps = max_steps
      trial%state_bytes = ref_vector_state_bytes(state)
      trial%learning_rate = apex_config%learning_rate
      trial%beta1 = apex_config%beta1
      trial%beta2 = 0.99_real64
      trial%epsilon = apex_config%epsilon
      if (kind == REF_SOPHIA_G) trial%rho = 0.04_real64
      trial%initial_loss = initial_loss
      trial%final_loss = loss
      trial%elapsed_seconds = real(end_tick - start_tick, real64) / real(rate, real64)
      trial%finite = all(ieee_is_finite(parameters)) .and. ieee_is_finite(loss)
      trial%status = BENCH_OK
      call emit_trial()
   end subroutine run_reference

   subroutine quadratic_gradient(parameters, gradient, loss, callback_error)
      real(real64), contiguous, intent(in) :: parameters(:)
      real(real64), contiguous, intent(out) :: gradient(:)
      real(real64), intent(out) :: loss
      type(apex_error_t), intent(out) :: callback_error
      gradient = parameters
      loss = quadratic_loss(parameters)
      callback_error%code = APEX_OK
      callback_error%message = ''
   end subroutine quadratic_gradient

   pure real(real64) function quadratic_loss(parameters)
      real(real64), intent(in) :: parameters(:)
      quadratic_loss = 0.5_real64 * sum(parameters * parameters)
   end function quadratic_loss

   subroutine emit_trial()
      call require_finite(trial)
      call bench_write_csv_row(unit, trial, status)
      if (status /= BENCH_OK) call fail('CSV row', 'write failed')
   end subroutine emit_trial

   subroutine require_finite(record)
      type(bench_trial_t), intent(in) :: record
      if (.not. record%finite .or. .not. bench_trial_is_valid(record)) &
         call fail('trial validation', 'non-finite or inconsistent trial record')
   end subroutine require_finite

   subroutine fail(operation, message)
      character(len=*), intent(in) :: operation, message
      close(unit)
      write(error_unit, '(a,a,a,a)') 'Benchmark ', trim(operation), ' failed: ', trim(message)
      stop 1
   end subroutine fail

end program bench_vector_baseline
