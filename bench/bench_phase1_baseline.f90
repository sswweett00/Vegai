! Fixed-configuration Phase 1 baseline harness. It is a research scaffold:
! it does not perform tuning or establish optimizer rankings.
program bench_phase1_baseline
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64, error_unit
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   use apex_optimizer, only : apex_config_t, apex_error_t, apex_state_t, apex_initialize, apex_update, &
                              apex_get_parameters, APEX_OK
   use bench_contracts, only : bench_trial_t, bench_seed_state, bench_normal, bench_uniform, &
                               bench_write_csv_header, bench_write_csv_row, BENCH_OK, BENCH_NUMERICAL
   use bench_tasks, only : bench_task_t, bench_task_initialize, bench_task_evaluate, &
                           bench_task_curvature_diagonal, TASK_QUADRATIC, TASK_ROSENBROCK, TASK_RASTRIGIN, &
                           TASK_LOGISTIC, TASK_SPARSE_LOGISTIC, TASK_MLP, TASK_TRANSFORMER_LM, &
                           TASK_MATRIX_FACTORIZATION, TASK_NOISY_QUADRATIC
   use ref_vector_optimizers, only : ref_vector_state_t, ref_vector_initialize, ref_vector_step, &
      ref_vector_state_bytes, ref_vector_name, ref_vector_eval_parameters, REF_OK, REF_SOPHIA_G, &
      REF_PRODIGY, REF_SCHEDULE_FREE_ADAMW, REF_SGD, REF_MOMENTUM, REF_NESTEROV, REF_ADAGRAD, &
      REF_RMSPROP, REF_ADAM, REF_ADAMW, REF_NADAM, REF_RADAM, REF_LION, REF_ADAFACTOR_VECTOR, REF_LAMB, REF_LARS
   use ref_matrix_optimizers, only : ref_matrix_state_t, ref_matrix_initialize, ref_matrix_step, &
      ref_matrix_state_bytes, ref_matrix_name, REF_MATRIX_OK, REF_ADAFACTOR_MATRIX, REF_SHAMPOO, REF_MUON
   implicit none

   integer(int64), parameter :: seeds(5) = [1_int64, 7_int64, 19_int64, 101_int64, 2026_int64]
   integer(int64), parameter :: max_steps = 1000_int64
   real(real64), parameter :: wall_clock_limit = 60.0_real64
   real(real64), parameter :: learning_rate = 0.02_real64, beta1 = 0.9_real64, beta2 = 0.99_real64
   real(real64), parameter :: epsilon = 1.0e-8_real64, weight_decay = 0.0_real64, sophia_rho = 0.04_real64
   integer(int32), parameter :: task_kinds(10) = [TASK_QUADRATIC, TASK_QUADRATIC, TASK_ROSENBROCK, &
      TASK_RASTRIGIN, TASK_LOGISTIC, TASK_SPARSE_LOGISTIC, TASK_MLP, TASK_TRANSFORMER_LM, &
      TASK_MATRIX_FACTORIZATION, TASK_NOISY_QUADRATIC]
   integer(int64), parameter :: parameter_counts(10) = [128_int64, 128_int64, 32_int64, 32_int64, &
      32_int64, 32_int64, 121_int64, 112_int64, 54_int64, 32_int64]
   real(real64), parameter :: condition_numbers(10) = [100.0_real64, 1.0e6_real64, 1.0_real64, 1.0_real64, &
      1.0_real64, 1.0_real64, 1.0_real64, 1.0_real64, 1.0_real64, 100.0_real64]
   character(len=48), parameter :: task_labels(10) = [character(len=48) :: &
      'quadratic-kappa-1e2', 'quadratic-kappa-1e6', 'Rosenbrock', 'Rastrigin', 'logistic-dense', &
      'logistic-sparse', 'synthetic-MLP', 'synthetic-transformer-LM', 'low-rank-matrix-factorization', &
      'noisy-quadratic']
   integer(int32), parameter :: vector_kinds(16) = [REF_SGD, REF_MOMENTUM, REF_NESTEROV, REF_ADAGRAD, &
      REF_RMSPROP, REF_ADAM, REF_ADAMW, REF_NADAM, REF_RADAM, REF_LION, REF_ADAFACTOR_VECTOR, &
      REF_LAMB, REF_LARS, REF_SOPHIA_G, REF_PRODIGY, REF_SCHEDULE_FREE_ADAMW]
   integer(int32), parameter :: matrix_kinds(3) = [REF_ADAFACTOR_MATRIX, REF_SHAMPOO, REF_MUON]
   character(len=*), parameter :: result_path = 'bench/results/baseline/phase1_fixed_multitask.csv'
   type(apex_config_t) :: apex_config
   type(bench_task_t) :: task
   type(bench_trial_t) :: trial
   real(real64), allocatable :: initial(:), gradient(:), curvature(:)
   integer(int32) :: status
   integer(int64) :: seed
   integer :: unit, io_status, task_index, seed_index, method_index

   apex_config%learning_rate = learning_rate
   apex_config%beta1 = beta1
   apex_config%beta2 = beta2
   apex_config%epsilon = epsilon
   apex_config%gamma = 0.5_real64
   apex_config%weight_decay = weight_decay
   apex_config%gradient_clip = 1000.0_real64
   apex_config%scheduler = 0

   open(newunit=unit, file=result_path, status='replace', action='write', iostat=io_status)
   if (io_status /= 0) call fatal('result open failed')
   call bench_write_csv_header(unit, status)
   if (status /= BENCH_OK) call fatal('CSV header write failed')

   do task_index = 1, size(task_kinds)
      do seed_index = 1, size(seeds)
         seed = seeds(seed_index)
         call bench_task_initialize(task, task_kinds(task_index), parameter_counts(task_index), seed, status, &
                                    condition_numbers(task_index), 128_int32, 0.25_real64)
         if (status /= BENCH_OK) call fatal('task initialization failed: '//trim(task_labels(task_index)))
         allocate(initial(parameter_counts(task_index)), gradient(parameter_counts(task_index)), stat=io_status)
         if (io_status /= 0) call fatal('trial vector allocation failed')
         call make_initial_parameters(task_kinds(task_index), seed, initial)
         call run_apex(task, initial, task_labels(task_index), seed)

         do method_index = 1, size(vector_kinds)
            if (vector_kinds(method_index) == REF_SOPHIA_G) then
               call bench_task_curvature_diagonal(task, curvature, status)
               if (status /= BENCH_OK) cycle
            end if
            call run_vector_reference(task, initial, task_labels(task_index), seed, vector_kinds(method_index))
            if (allocated(curvature)) deallocate(curvature)
         end do

         if (task_kinds(task_index) == TASK_MATRIX_FACTORIZATION) then
            do method_index = 1, size(matrix_kinds)
               call run_matrix_reference(task, initial, task_labels(task_index), seed, matrix_kinds(method_index))
            end do
         end if
         deallocate(initial, gradient)
      end do
   end do
   close(unit, iostat=io_status)
   if (io_status /= 0) call fatal('result close failed')

contains

   subroutine make_initial_parameters(kind, current_seed, parameters)
      integer(int32), intent(in) :: kind
      integer(int64), intent(in) :: current_seed
      real(real64), intent(out) :: parameters(:)
      integer(int64) :: rng
      integer :: i
      rng = bench_seed_state(ieor(current_seed, int(z'5A17D3E4B9216C0F', int64)))
      select case (kind)
      case (TASK_ROSENBROCK)
         parameters = 1.0_real64
         do i = 1, size(parameters), 2
            parameters(i) = -1.2_real64
         end do
      case (TASK_RASTRIGIN)
         do i = 1, size(parameters)
            parameters(i) = 8.0_real64 * (bench_uniform(rng) - 0.5_real64)
         end do
      case (TASK_LOGISTIC, TASK_SPARSE_LOGISTIC)
         parameters = 0.0_real64
      case (TASK_MLP, TASK_TRANSFORMER_LM, TASK_MATRIX_FACTORIZATION)
         do i = 1, size(parameters)
            parameters(i) = 0.05_real64 * bench_normal(rng)
         end do
      case default
         do i = 1, size(parameters)
            parameters(i) = 0.1_real64 * bench_normal(rng)
         end do
      end select
   end subroutine make_initial_parameters

   subroutine run_apex(active_task, start_parameters, task_name, current_seed)
      type(bench_task_t), intent(in) :: active_task
      real(real64), intent(in) :: start_parameters(:)
      character(len=*), intent(in) :: task_name
      integer(int64), intent(in) :: current_seed
      type(apex_state_t) :: state
      type(apex_error_t) :: error
      real(real64) :: parameters(size(start_parameters)), objective, initial_loss, final_loss, elapsed
      integer(int64) :: tick_start, tick_end, clock_rate, current_step, completed_steps
      integer(int32) :: local_status

      initial_loss = huge(1.0_real64)
      final_loss = huge(1.0_real64)
      elapsed = 0.0_real64
      completed_steps = 0_int64
      parameters = start_parameters
      call apex_initialize(state, parameters, apex_config, error)
      if (error%code /= APEX_OK) then
         call failed_trial(task_name, 'Apex', current_seed, initial_loss, completed_steps, 0_int64, elapsed)
         return
      end if
      call bench_task_evaluate(active_task, parameters, gradient, initial_loss, local_status, 0_int64)
      if (local_status /= BENCH_OK) then
         call failed_trial(task_name, 'Apex', current_seed, initial_loss, completed_steps, 0_int64, elapsed)
         return
      end if
      call system_clock(count_rate=clock_rate)
      if (clock_rate <= 0_int64) call fatal('system_clock returned a non-positive rate')
      call system_clock(tick_start)
      current_step = 0_int64
      do current_step = 1_int64, max_steps
         call apex_get_parameters(state, parameters, error)
         if (error%code /= APEX_OK) exit
         call bench_task_evaluate(active_task, parameters, gradient, objective, local_status, current_step)
         if (local_status /= BENCH_OK) exit
         call apex_update(state, gradient, error)
         if (error%code /= APEX_OK) exit
         completed_steps = current_step
         call system_clock(tick_end)
         elapsed = ticks_to_seconds(tick_start, tick_end, clock_rate)
         if (elapsed >= wall_clock_limit) exit
      end do
      if (error%code == APEX_OK .and. local_status == BENCH_OK) then
         call apex_get_parameters(state, parameters, error)
      end if
      call system_clock(tick_end)
      elapsed = ticks_to_seconds(tick_start, tick_end, clock_rate)
      if (error%code /= APEX_OK .or. local_status /= BENCH_OK) then
         call failed_trial(task_name, 'Apex', current_seed, initial_loss, completed_steps, &
                           parameter_bytes(size(parameters)), elapsed)
         return
      end if
      call bench_task_evaluate(active_task, parameters, gradient, final_loss, local_status, completed_steps)
      if (local_status /= BENCH_OK) then
         call failed_trial(task_name, 'Apex', current_seed, initial_loss, completed_steps, &
                           parameter_bytes(size(parameters)), elapsed)
         return
      end if
      trial = bench_trial_t()
      trial%task = task_name
      trial%optimizer = 'Apex'
      trial%seed = current_seed
      trial%max_steps = max_steps
      trial%steps = completed_steps
      trial%learning_rate = learning_rate
      trial%beta1 = beta1
      trial%beta2 = beta2
      trial%epsilon = epsilon
      trial%state_bytes = parameter_bytes(size(parameters))
      trial%initial_loss = initial_loss
      trial%final_loss = final_loss
      trial%elapsed_seconds = elapsed
      trial%finite = all(ieee_is_finite(parameters)) .and. ieee_is_finite(final_loss)
      trial%status = BENCH_OK
      call emit_trial()
   end subroutine run_apex

   subroutine run_vector_reference(active_task, start_parameters, task_name, current_seed, kind)
      type(bench_task_t), intent(in) :: active_task
      real(real64), intent(in) :: start_parameters(:)
      character(len=*), intent(in) :: task_name
      integer(int64), intent(in) :: current_seed
      integer(int32), intent(in) :: kind
      type(ref_vector_state_t) :: state
      real(real64) :: parameters(size(start_parameters)), objective, initial_loss, final_loss, elapsed
      integer(int64) :: tick_start, tick_end, clock_rate, current_step, completed_steps
      integer(int32) :: local_status

      initial_loss = huge(1.0_real64)
      final_loss = huge(1.0_real64)
      elapsed = 0.0_real64
      completed_steps = 0_int64
      parameters = start_parameters
      call ref_vector_initialize(kind, int(size(parameters), int64), state, local_status)
      if (local_status /= REF_OK) then
         call failed_trial(task_name, trim(ref_vector_name(kind)), current_seed, 0.0_real64, 0_int64, &
                           0_int64, 0.0_real64)
         return
      end if
      call bench_task_evaluate(active_task, parameters, gradient, initial_loss, local_status, 0_int64)
      if (local_status /= BENCH_OK) then
         call failed_trial(task_name, trim(ref_vector_name(kind)), current_seed, initial_loss, 0_int64, &
                           ref_vector_state_bytes(state), 0.0_real64)
         return
      end if
      call system_clock(count_rate=clock_rate)
      if (clock_rate <= 0_int64) call fatal('system_clock returned a non-positive rate')
      call system_clock(tick_start)
      current_step = 0_int64
      do current_step = 1_int64, max_steps
         call bench_task_evaluate(active_task, parameters, gradient, objective, local_status, current_step)
         if (local_status /= BENCH_OK) exit
         if (kind == REF_SOPHIA_G) then
            call ref_vector_step(parameters, gradient, state, learning_rate, beta1, beta2, epsilon, &
                                 weight_decay, local_status, curvature, sophia_rho)
         else
            call ref_vector_step(parameters, gradient, state, learning_rate, beta1, beta2, epsilon, &
                                 weight_decay, local_status)
         end if
         if (local_status /= REF_OK) exit
         completed_steps = current_step
         call system_clock(tick_end)
         elapsed = ticks_to_seconds(tick_start, tick_end, clock_rate)
         if (elapsed >= wall_clock_limit) exit
      end do
      call system_clock(tick_end)
      elapsed = ticks_to_seconds(tick_start, tick_end, clock_rate)
      if (local_status /= REF_OK) then
         call failed_trial(task_name, trim(ref_vector_name(kind)), current_seed, initial_loss, &
                           completed_steps, ref_vector_state_bytes(state), elapsed)
         return
      end if
      if (kind == REF_SCHEDULE_FREE_ADAMW) then
         call ref_vector_eval_parameters(parameters, state, beta1, local_status)
         if (local_status /= REF_OK) then
            call failed_trial(task_name, trim(ref_vector_name(kind)), current_seed, initial_loss, &
                              completed_steps, ref_vector_state_bytes(state), elapsed)
            return
         end if
      end if
      call bench_task_evaluate(active_task, parameters, gradient, final_loss, local_status, completed_steps)
      if (local_status /= BENCH_OK) then
         call failed_trial(task_name, trim(ref_vector_name(kind)), current_seed, initial_loss, &
                           completed_steps, ref_vector_state_bytes(state), elapsed)
         return
      end if
      trial = bench_trial_t()
      trial%task = task_name
      trial%optimizer = trim(ref_vector_name(kind))
      trial%seed = current_seed
      trial%max_steps = max_steps
      trial%steps = completed_steps
      trial%learning_rate = learning_rate
      trial%beta1 = beta1
      trial%beta2 = beta2
      trial%epsilon = epsilon
      trial%rho = 0.0_real64
      if (kind == REF_SOPHIA_G) trial%rho = sophia_rho
      trial%state_bytes = ref_vector_state_bytes(state)
      trial%initial_loss = initial_loss
      trial%final_loss = final_loss
      trial%elapsed_seconds = elapsed
      trial%finite = all(ieee_is_finite(parameters)) .and. ieee_is_finite(final_loss)
      trial%status = BENCH_OK
      call emit_trial()
   end subroutine run_vector_reference

   subroutine run_matrix_reference(active_task, start_parameters, task_name, current_seed, kind)
      type(bench_task_t), intent(in) :: active_task
      real(real64), intent(in) :: start_parameters(:)
      character(len=*), intent(in) :: task_name
      integer(int64), intent(in) :: current_seed
      integer(int32), intent(in) :: kind
      type(ref_matrix_state_t) :: user_state, item_state
      real(real64) :: parameters(size(start_parameters)), objective, initial_loss, final_loss, elapsed
      real(real64) :: user_factors(8, 3), item_factors(3, 10), user_gradient(8, 3), item_gradient(3, 10)
      integer(int64) :: tick_start, tick_end, clock_rate, current_step, completed_steps
      integer(int32) :: local_status

      initial_loss = huge(1.0_real64)
      final_loss = huge(1.0_real64)
      elapsed = 0.0_real64
      completed_steps = 0_int64
      parameters = start_parameters
      user_factors = reshape(parameters(1:24), shape(user_factors))
      item_factors = reshape(parameters(25:54), shape(item_factors))
      call ref_matrix_initialize(kind, 8_int64, 3_int64, user_state, local_status)
      if (local_status /= REF_MATRIX_OK) then
         call failed_trial(task_name, trim(ref_matrix_name(kind)), current_seed, 0.0_real64, 0_int64, &
                           0_int64, 0.0_real64)
         return
      end if
      call ref_matrix_initialize(kind, 3_int64, 10_int64, item_state, local_status)
      if (local_status /= REF_MATRIX_OK) then
         call failed_trial(task_name, trim(ref_matrix_name(kind)), current_seed, 0.0_real64, 0_int64, &
                           0_int64, 0.0_real64)
         return
      end if
      call bench_task_evaluate(active_task, parameters, gradient, initial_loss, local_status, 0_int64)
      if (local_status /= BENCH_OK) then
         call failed_trial(task_name, trim(ref_matrix_name(kind)), current_seed, initial_loss, 0_int64, &
                           ref_matrix_state_bytes(user_state) + ref_matrix_state_bytes(item_state), 0.0_real64)
         return
      end if
      call system_clock(count_rate=clock_rate)
      if (clock_rate <= 0_int64) call fatal('system_clock returned a non-positive rate')
      call system_clock(tick_start)
      current_step = 0_int64
      do current_step = 1_int64, max_steps
         call bench_task_evaluate(active_task, parameters, gradient, objective, local_status, current_step)
         if (local_status /= BENCH_OK) exit
         user_gradient = reshape(gradient(1:24), shape(user_gradient))
         item_gradient = reshape(gradient(25:54), shape(item_gradient))
         call ref_matrix_step(user_factors, user_gradient, user_state, learning_rate, beta1, epsilon, local_status)
         if (local_status /= REF_MATRIX_OK) exit
         call ref_matrix_step(item_factors, item_gradient, item_state, learning_rate, beta1, epsilon, local_status)
         if (local_status /= REF_MATRIX_OK) exit
         parameters(1:24) = reshape(user_factors, [24])
         parameters(25:54) = reshape(item_factors, [30])
         completed_steps = current_step
         call system_clock(tick_end)
         elapsed = ticks_to_seconds(tick_start, tick_end, clock_rate)
         if (elapsed >= wall_clock_limit) exit
      end do
      call system_clock(tick_end)
      elapsed = ticks_to_seconds(tick_start, tick_end, clock_rate)
      if (local_status /= REF_MATRIX_OK) then
         call failed_trial(task_name, trim(ref_matrix_name(kind)), current_seed, initial_loss, &
                           completed_steps, &
                           ref_matrix_state_bytes(user_state) + ref_matrix_state_bytes(item_state), elapsed)
         return
      end if
      call bench_task_evaluate(active_task, parameters, gradient, final_loss, local_status, completed_steps)
      if (local_status /= BENCH_OK) then
         call failed_trial(task_name, trim(ref_matrix_name(kind)), current_seed, initial_loss, &
                           completed_steps, &
                           ref_matrix_state_bytes(user_state) + ref_matrix_state_bytes(item_state), elapsed)
         return
      end if
      trial = bench_trial_t()
      trial%task = task_name
      trial%optimizer = trim(ref_matrix_name(kind))
      trial%seed = current_seed
      trial%max_steps = max_steps
      trial%steps = completed_steps
      trial%learning_rate = learning_rate
      trial%beta1 = beta1
      trial%beta2 = beta2
      trial%epsilon = epsilon
      trial%state_bytes = ref_matrix_state_bytes(user_state) + ref_matrix_state_bytes(item_state)
      trial%initial_loss = initial_loss
      trial%final_loss = final_loss
      trial%elapsed_seconds = elapsed
      trial%finite = all(ieee_is_finite(parameters)) .and. ieee_is_finite(final_loss)
      trial%status = BENCH_OK
      call emit_trial()
   end subroutine run_matrix_reference

   subroutine failed_trial(task_name, optimizer_name, current_seed, initial_value, completed_steps, state_bytes, elapsed)
      character(len=*), intent(in) :: task_name, optimizer_name
      integer(int64), intent(in) :: current_seed, completed_steps, state_bytes
      real(real64), intent(in) :: initial_value, elapsed
      trial = bench_trial_t()
      trial%task = task_name
      trial%optimizer = optimizer_name
      trial%seed = current_seed
      trial%max_steps = max_steps
      trial%steps = max(0_int64, min(max_steps, completed_steps))
      trial%learning_rate = learning_rate
      trial%beta1 = beta1
      trial%beta2 = beta2
      trial%epsilon = epsilon
      trial%state_bytes = max(0_int64, state_bytes)
      trial%initial_loss = initial_value
      if (.not. ieee_is_finite(trial%initial_loss)) trial%initial_loss = huge(1.0_real64)
      trial%final_loss = huge(1.0_real64)
      trial%elapsed_seconds = max(0.0_real64, elapsed)
      trial%finite = .false.
      trial%status = BENCH_NUMERICAL
      call emit_trial()
   end subroutine failed_trial

   subroutine emit_trial()
      call bench_write_csv_row(unit, trial, status)
      if (status /= BENCH_OK) call fatal('CSV row write failed')
   end subroutine emit_trial

   pure real(real64) function ticks_to_seconds(start_tick, end_tick, rate) result(seconds)
      integer(int64), intent(in) :: start_tick, end_tick, rate
      seconds = 0.0_real64
      if (rate > 0_int64 .and. end_tick >= start_tick) &
         seconds = real(end_tick - start_tick, real64) / real(rate, real64)
   end function ticks_to_seconds

   pure integer(int64) function parameter_bytes(count) result(bytes)
      integer, intent(in) :: count
      bytes = int(count, int64) * int(storage_size(0.0_real64) / 8, int64)
   end function parameter_bytes

   subroutine fatal(message)
      character(len=*), intent(in) :: message
      close(unit)
      write(error_unit, '(a)') trim(message)
      stop 1
   end subroutine fatal

end program bench_phase1_baseline
