! Equal-budget hyperparameter search with train/validation selection and a
! one-time independent test evaluation of the selected candidate.
program bench_phase2_search
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64, error_unit
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite, ieee_value, ieee_quiet_nan
   use apex_optimizer, only : apex_config_t, apex_error_t, apex_state_t, apex_initialize, apex_update, &
                              apex_get_parameters, APEX_OK
   use bench_contracts, only : bench_seed_state, bench_normal, bench_uniform, BENCH_OK, BENCH_INVALID, BENCH_NUMERICAL
   use bench_tasks, only : bench_task_t, bench_task_initialize, bench_task_evaluate, &
      bench_task_curvature_diagonal, bench_task_has_validation_split, bench_task_has_test_split, &
      TASK_QUADRATIC, TASK_ROSENBROCK, TASK_RASTRIGIN, TASK_LOGISTIC, TASK_SPARSE_LOGISTIC, TASK_MLP, &
      TASK_TRANSFORMER_LM, TASK_MATRIX_FACTORIZATION, TASK_NOISY_QUADRATIC, TASK_PARTITION_TRAIN, &
      TASK_PARTITION_VALIDATION, TASK_PARTITION_TEST
   use bench_tuning, only : bench_hyperparameters_t, BENCH_TUNING_TRIALS, bench_make_tuning_grid, &
                            bench_choose_best_validation
   use ref_vector_optimizers, only : ref_vector_state_t, ref_vector_initialize, ref_vector_step, &
      ref_vector_state_bytes, ref_vector_name, ref_vector_eval_parameters, REF_OK, REF_SOPHIA_G, &
      REF_PRODIGY, REF_SCHEDULE_FREE_ADAMW, REF_SGD, REF_MOMENTUM, REF_NESTEROV, REF_ADAGRAD, &
      REF_RMSPROP, REF_ADAM, REF_ADAMW, REF_NADAM, REF_RADAM, REF_LION, REF_ADAFACTOR_VECTOR, REF_LAMB, REF_LARS
   use ref_matrix_optimizers, only : ref_matrix_state_t, ref_matrix_initialize, ref_matrix_step, &
      ref_matrix_state_bytes, ref_matrix_name, REF_MATRIX_OK, REF_ADAFACTOR_MATRIX, REF_SHAMPOO, REF_MUON
   implicit none

   integer(int64), parameter :: seeds(5) = [1_int64, 7_int64, 19_int64, 101_int64, 2026_int64]
   integer(int64), parameter :: default_steps = 1000_int64
   real(real64), parameter :: default_time = 60.0_real64
   integer(int32), parameter :: task_kinds(10) = [TASK_QUADRATIC, TASK_QUADRATIC, TASK_ROSENBROCK, &
      TASK_RASTRIGIN, TASK_LOGISTIC, TASK_SPARSE_LOGISTIC, TASK_MLP, TASK_TRANSFORMER_LM, &
      TASK_MATRIX_FACTORIZATION, TASK_NOISY_QUADRATIC]
   integer(int64), parameter :: parameter_counts(10) = [128_int64, 128_int64, 32_int64, 32_int64, &
      32_int64, 32_int64, 121_int64, 112_int64, 54_int64, 32_int64]
   real(real64), parameter :: condition_numbers(10) = [100.0_real64, 1.0e6_real64, 1.0_real64, 1.0_real64, &
      1.0_real64, 1.0_real64, 1.0_real64, 1.0_real64, 1.0_real64, 100.0_real64]
   character(len=48), parameter :: task_labels(10) = [character(len=48) :: 'quadratic-kappa-1e2', &
      'quadratic-kappa-1e6', 'Rosenbrock', 'Rastrigin', 'logistic-dense', 'logistic-sparse', &
      'synthetic-MLP', 'synthetic-transformer-LM', 'low-rank-matrix-factorization', 'noisy-quadratic']
   integer(int32), parameter :: vector_kinds(16) = [REF_SGD, REF_MOMENTUM, REF_NESTEROV, REF_ADAGRAD, &
      REF_RMSPROP, REF_ADAM, REF_ADAMW, REF_NADAM, REF_RADAM, REF_LION, REF_ADAFACTOR_VECTOR, &
      REF_LAMB, REF_LARS, REF_SOPHIA_G, REF_PRODIGY, REF_SCHEDULE_FREE_ADAMW]
   integer(int32), parameter :: matrix_kinds(3) = [REF_ADAFACTOR_MATRIX, REF_SHAMPOO, REF_MUON]

   type :: candidate_t
      type(bench_hyperparameters_t) :: hp
      real(real64) :: train_initial = 0.0_real64
      real(real64) :: validation_loss = 0.0_real64
      real(real64) :: test_loss = 0.0_real64
      real(real64) :: elapsed = 0.0_real64
      real(real64) :: test_elapsed = 0.0_real64
      integer(int64) :: steps = 0_int64
      integer(int64) :: state_bytes = 0_int64
      integer(int32) :: status = BENCH_NUMERICAL
      integer(int32) :: test_status = BENCH_INVALID
      logical :: validation_holdout = .false.
      logical :: test_holdout = .false.
      real(real64), allocatable :: parameters(:)
   end type candidate_t

   type(bench_task_t) :: task
   type(candidate_t) :: candidates(BENCH_TUNING_TRIALS)
   real(real64), allocatable :: initial(:)
   integer(int64) :: max_steps, seed
   real(real64) :: wall_limit
   integer(int32) :: status
   integer :: unit, io_status, task_i, seed_i, method_i, arg_status
   character(len=512) :: result_path, arg

   unit = 0
   result_path = 'bench/results/phase2/validation_search.csv'
   max_steps = default_steps
   wall_limit = default_time
   call get_command_argument(1, result_path, status=arg_status)
   if (arg_status /= 0) call fatal('result-path argument too long')
   if (len_trim(result_path) == 0) result_path = 'bench/results/phase2/validation_search.csv'
   call get_command_argument(2, arg, status=arg_status)
   if (arg_status == 0 .and. len_trim(arg) > 0) then
      read(arg, *, iostat=io_status) max_steps
      if (io_status /= 0 .or. max_steps <= 0_int64) call fatal('max_steps must be positive')
   end if
   call get_command_argument(3, arg, status=arg_status)
   if (arg_status == 0 .and. len_trim(arg) > 0) then
      read(arg, *, iostat=io_status) wall_limit
      if (io_status /= 0 .or. .not. ieee_is_finite(wall_limit) .or. wall_limit <= 0.0_real64) &
         call fatal('wall-clock limit must be positive and finite')
   end if

   open(newunit=unit, file=trim(result_path), status='replace', action='write', iostat=io_status)
   if (io_status /= 0) call fatal('cannot open candidate CSV')
   write(unit, '(a)', iostat=io_status) &
      'task,optimizer,seed,candidate_id,selected,validation_is_holdout,test_is_holdout,'// &
      'learning_rate,beta1,beta2,epsilon,weight_decay,rho,max_steps,steps,training_initial_loss,'// &
      'validation_loss,test_loss,test_status,elapsed_seconds,test_elapsed_seconds,state_bytes,status'
   if (io_status /= 0) call fatal('cannot write candidate CSV header')

   do task_i = 1, size(task_kinds)
      do seed_i = 1, size(seeds)
         seed = seeds(seed_i)
         call bench_task_initialize(task, task_kinds(task_i), parameter_counts(task_i), seed, status, &
                                    condition_numbers(task_i), 128_int32, 0.25_real64)
         if (status /= BENCH_OK) call fatal('task initialization failed: '//trim(task_labels(task_i)))
         allocate(initial(parameter_counts(task_i)), stat=io_status)
         if (io_status /= 0) call fatal('initial parameter allocation failed')
         call make_initial_parameters(task_kinds(task_i), seed, initial)
         call tune_one('Apex', 0_int32, task, initial, task_labels(task_i), seed)
         do method_i = 1, size(vector_kinds)
            if (vector_kinds(method_i) == REF_SOPHIA_G .and. task_kinds(task_i) /= TASK_QUADRATIC .and. &
                task_kinds(task_i) /= TASK_NOISY_QUADRATIC) cycle
            call tune_one(trim(ref_vector_name(vector_kinds(method_i))), vector_kinds(method_i), &
                          task, initial, task_labels(task_i), seed)
         end do
         if (task_kinds(task_i) == TASK_MATRIX_FACTORIZATION) then
            do method_i = 1, size(matrix_kinds)
               call tune_one(trim(ref_matrix_name(matrix_kinds(method_i))), -matrix_kinds(method_i), &
                             task, initial, task_labels(task_i), seed)
            end do
         end if
         deallocate(initial)
      end do
   end do
   close(unit, iostat=io_status)
   if (io_status /= 0) call fatal('candidate CSV close failed')

contains

   subroutine tune_one(name, kind, active_task, start_parameters, task_name, current_seed)
      character(len=*), intent(in) :: name, task_name
      integer(int32), intent(in) :: kind
      type(bench_task_t), intent(in) :: active_task
      real(real64), intent(in) :: start_parameters(:)
      integer(int64), intent(in) :: current_seed
      type(bench_hyperparameters_t) :: grid(BENCH_TUNING_TRIALS)
      real(real64) :: scores(BENCH_TUNING_TRIALS), test_gradient(size(start_parameters))
      integer(int32) :: grid_status, winner, test_eval_status
      integer(int64) :: test_tick_start, test_tick_end, tick_rate
      integer :: c
      call bench_make_tuning_grid(name, current_seed, grid, grid_status)
      if (grid_status /= BENCH_OK) call fatal('unsupported tuning optimizer '//trim(name))
      scores = ieee_value(0.0_real64, ieee_quiet_nan)
      do c = 1, BENCH_TUNING_TRIALS
         call initialize_candidate(candidates(c))
         candidates(c)%hp = grid(c)
         candidates(c)%validation_holdout = bench_task_has_validation_split(active_task)
         candidates(c)%test_holdout = bench_task_has_test_split(active_task)
         call train_candidate(kind, active_task, start_parameters, candidates(c))
         scores(c) = candidates(c)%validation_loss
      end do
      call bench_choose_best_validation(grid, scores, winner, grid_status)
      if (grid_status == BENCH_OK) then
         call system_clock(count_rate=tick_rate)
         if (tick_rate <= 0_int64) call fatal('system_clock has invalid rate')
         call system_clock(test_tick_start)
         call bench_task_evaluate(active_task, candidates(winner)%parameters, test_gradient, &
                                  candidates(winner)%test_loss, test_eval_status, candidates(winner)%steps, &
                                  TASK_PARTITION_TEST)
         call system_clock(test_tick_end)
         candidates(winner)%test_elapsed = ticks_to_seconds(test_tick_start, test_tick_end, tick_rate)
         candidates(winner)%test_status = test_eval_status
         if (test_eval_status /= BENCH_OK .or. .not. ieee_is_finite(candidates(winner)%test_loss)) then
            candidates(winner)%test_loss = ieee_value(0.0_real64, ieee_quiet_nan)
            candidates(winner)%test_status = BENCH_NUMERICAL
         end if
      end if
      do c = 1, BENCH_TUNING_TRIALS
         call write_candidate(task_name, name, current_seed, candidates(c), int(c, int32) == winner)
      end do
   end subroutine tune_one

   subroutine train_candidate(kind, active_task, start_parameters, result)
      integer(int32), intent(in) :: kind
      type(bench_task_t), intent(in) :: active_task
      real(real64), intent(in) :: start_parameters(:)
      type(candidate_t), intent(inout) :: result
      if (kind == 0_int32) then
         call train_apex(active_task, start_parameters, result)
      else if (kind > 0_int32) then
         call train_vector(kind, active_task, start_parameters, result)
      else
         call train_matrix(-kind, active_task, start_parameters, result)
      end if
   end subroutine train_candidate

   subroutine train_apex(active_task, start, result)
      type(bench_task_t), intent(in) :: active_task
      real(real64), intent(in) :: start(:)
      type(candidate_t), intent(inout) :: result
      type(apex_config_t) :: config
      type(apex_error_t) :: error
      type(apex_state_t) :: state
      real(real64) :: p(size(start)), g(size(start)), objective
      integer(int64) :: tick0, tick1, rate, step
      integer(int32) :: local_status
      call initialize_candidate(result)
      call system_clock(count_rate=rate)
      if (rate <= 0_int64) return
      call system_clock(tick0)
      config%learning_rate = result%hp%learning_rate
      config%beta1 = result%hp%beta1
      config%beta2 = result%hp%beta2
      config%epsilon = result%hp%epsilon
      config%gamma = 0.5_real64
      config%weight_decay = result%hp%weight_decay
      config%gradient_clip = 1000.0_real64
      config%scheduler = 0
      p = start
      call apex_initialize(state, p, config, error)
      if (error%code /= APEX_OK) return
      call bench_task_evaluate(active_task, p, g, result%train_initial, local_status, 0_int64, TASK_PARTITION_TRAIN)
      if (local_status /= BENCH_OK) return
      call system_clock(tick1)
      result%elapsed = ticks_to_seconds(tick0, tick1, rate)
      if (result%elapsed >= wall_limit) return
      result%status = BENCH_OK
      do step = 1_int64, max_steps
         call apex_get_parameters(state, p, error)
         if (error%code /= APEX_OK) exit
         call bench_task_evaluate(active_task, p, g, objective, local_status, step, TASK_PARTITION_TRAIN)
         if (local_status /= BENCH_OK) exit
         call apex_update(state, g, error)
         if (error%code /= APEX_OK) exit
         result%steps = step
         call system_clock(tick1)
         result%elapsed = ticks_to_seconds(tick0, tick1, rate)
         if (result%elapsed >= wall_limit) exit
      end do
      if (error%code /= APEX_OK .or. local_status /= BENCH_OK) then
         result%status = BENCH_NUMERICAL
         return
      end if
      call apex_get_parameters(state, p, error)
      if (error%code /= APEX_OK) then
         result%status = BENCH_NUMERICAL
         return
      end if
      call bench_task_evaluate(active_task, p, g, result%validation_loss, local_status, &
                               result%steps, TASK_PARTITION_VALIDATION)
      if (local_status /= BENCH_OK) then
         result%status = BENCH_NUMERICAL
         return
      end if
      call system_clock(tick1)
      result%elapsed = ticks_to_seconds(tick0, tick1, rate)
      result%state_bytes = int(size(p), int64) * int(storage_size(0.0_real64) / 8, int64)
      result%parameters = p
      result%status = BENCH_OK
   end subroutine train_apex

   subroutine train_vector(kind, active_task, start, result)
      integer(int32), intent(in) :: kind
      type(bench_task_t), intent(in) :: active_task
      real(real64), intent(in) :: start(:)
      type(candidate_t), intent(inout) :: result
      type(ref_vector_state_t) :: state
      real(real64) :: p(size(start)), g(size(start)), objective
      real(real64), allocatable :: curvature(:)
      integer(int64) :: tick0, tick1, rate, step
      integer(int32) :: local_status
      call initialize_candidate(result)
      call system_clock(count_rate=rate)
      if (rate <= 0_int64) return
      call system_clock(tick0)
      p = start
      call ref_vector_initialize(kind, int(size(p), int64), state, local_status)
      if (local_status /= REF_OK) return
      if (kind == REF_SOPHIA_G) then
         call bench_task_curvature_diagonal(active_task, curvature, local_status)
         if (local_status /= BENCH_OK) return
      end if
      call bench_task_evaluate(active_task, p, g, result%train_initial, local_status, 0_int64, TASK_PARTITION_TRAIN)
      if (local_status /= BENCH_OK) return
      call system_clock(tick1)
      result%elapsed = ticks_to_seconds(tick0, tick1, rate)
      if (result%elapsed >= wall_limit) return
      result%status = BENCH_OK
      do step = 1_int64, max_steps
         call bench_task_evaluate(active_task, p, g, objective, local_status, step, TASK_PARTITION_TRAIN)
         if (local_status /= BENCH_OK) exit
         if (kind == REF_SOPHIA_G) then
            call ref_vector_step(p, g, state, result%hp%learning_rate, result%hp%beta1, result%hp%beta2, &
                                 result%hp%epsilon, result%hp%weight_decay, local_status, curvature, result%hp%rho)
         else
            call ref_vector_step(p, g, state, result%hp%learning_rate, result%hp%beta1, result%hp%beta2, &
                                 result%hp%epsilon, result%hp%weight_decay, local_status)
         end if
         if (local_status /= REF_OK) exit
         result%steps = step
         call system_clock(tick1)
         result%elapsed = ticks_to_seconds(tick0, tick1, rate)
         if (result%elapsed >= wall_limit) exit
      end do
      if (local_status /= REF_OK) then
         result%status = BENCH_NUMERICAL
         return
      end if
      if (kind == REF_SCHEDULE_FREE_ADAMW) then
         call ref_vector_eval_parameters(p, state, result%hp%beta1, local_status)
         if (local_status /= REF_OK) then
            result%status = BENCH_NUMERICAL
            return
         end if
      end if
      call bench_task_evaluate(active_task, p, g, result%validation_loss, local_status, &
                               result%steps, TASK_PARTITION_VALIDATION)
      if (local_status /= BENCH_OK) then
         result%status = BENCH_NUMERICAL
         return
      end if
      call system_clock(tick1)
      result%elapsed = ticks_to_seconds(tick0, tick1, rate)
      result%state_bytes = ref_vector_state_bytes(state)
      result%parameters = p
      result%status = BENCH_OK
   end subroutine train_vector

   subroutine train_matrix(kind, active_task, start, result)
      integer(int32), intent(in) :: kind
      type(bench_task_t), intent(in) :: active_task
      real(real64), intent(in) :: start(:)
      type(candidate_t), intent(inout) :: result
      type(ref_matrix_state_t) :: state_u, state_v
      real(real64) :: p(size(start)), g(size(start)), objective
      real(real64) :: u(8, 3), v(3, 10), gu(8, 3), gv(3, 10)
      integer(int64) :: tick0, tick1, rate, step
      integer(int32) :: local_status
      call initialize_candidate(result)
      call system_clock(count_rate=rate)
      if (rate <= 0_int64) return
      call system_clock(tick0)
      p = start
      u = reshape(p(1:24), shape(u))
      v = reshape(p(25:54), shape(v))
      call ref_matrix_initialize(kind, 8_int64, 3_int64, state_u, local_status)
      if (local_status /= REF_MATRIX_OK) return
      call ref_matrix_initialize(kind, 3_int64, 10_int64, state_v, local_status)
      if (local_status /= REF_MATRIX_OK) return
      call bench_task_evaluate(active_task, p, g, result%train_initial, local_status, 0_int64, TASK_PARTITION_TRAIN)
      if (local_status /= BENCH_OK) return
      call system_clock(tick1)
      result%elapsed = ticks_to_seconds(tick0, tick1, rate)
      if (result%elapsed >= wall_limit) return
      result%status = BENCH_OK
      do step = 1_int64, max_steps
         call bench_task_evaluate(active_task, p, g, objective, local_status, step, TASK_PARTITION_TRAIN)
         if (local_status /= BENCH_OK) exit
         gu = reshape(g(1:24), shape(gu))
         gv = reshape(g(25:54), shape(gv))
         call ref_matrix_step(u, gu, state_u, result%hp%learning_rate, result%hp%beta1, result%hp%epsilon, local_status)
         if (local_status /= REF_MATRIX_OK) exit
         call ref_matrix_step(v, gv, state_v, result%hp%learning_rate, result%hp%beta1, result%hp%epsilon, local_status)
         if (local_status /= REF_MATRIX_OK) exit
         p(1:24) = reshape(u, [24])
         p(25:54) = reshape(v, [30])
         result%steps = step
         call system_clock(tick1)
         result%elapsed = ticks_to_seconds(tick0, tick1, rate)
         if (result%elapsed >= wall_limit) exit
      end do
      if (local_status /= REF_MATRIX_OK) then
         result%status = BENCH_NUMERICAL
         return
      end if
      call bench_task_evaluate(active_task, p, g, result%validation_loss, local_status, &
                               result%steps, TASK_PARTITION_VALIDATION)
      if (local_status /= BENCH_OK) then
         result%status = BENCH_NUMERICAL
         return
      end if
      call system_clock(tick1)
      result%elapsed = ticks_to_seconds(tick0, tick1, rate)
      result%state_bytes = ref_matrix_state_bytes(state_u) + ref_matrix_state_bytes(state_v)
      result%parameters = p
      result%status = BENCH_OK
   end subroutine train_matrix

   subroutine initialize_candidate(result)
      type(candidate_t), intent(inout) :: result
      result%train_initial = huge(1.0_real64)
      result%validation_loss = ieee_value(0.0_real64, ieee_quiet_nan)
      result%test_loss = ieee_value(0.0_real64, ieee_quiet_nan)
      result%elapsed = 0.0_real64
      result%test_elapsed = 0.0_real64
      result%steps = 0_int64
      result%state_bytes = 0_int64
      result%status = BENCH_NUMERICAL
      result%test_status = BENCH_INVALID
      if (allocated(result%parameters)) deallocate(result%parameters)
   end subroutine initialize_candidate

   subroutine write_candidate(task_name, optimizer, current_seed, result, selected)
      character(len=*), intent(in) :: task_name, optimizer
      integer(int64), intent(in) :: current_seed
      type(candidate_t), intent(in) :: result
      logical, intent(in) :: selected
      integer :: write_status
      real(real64) :: validation_output, test_output
      validation_output = ieee_value(0.0_real64, ieee_quiet_nan)
      test_output = ieee_value(0.0_real64, ieee_quiet_nan)
      if (result%status == BENCH_OK) validation_output = result%validation_loss
      if (result%test_status == BENCH_OK) test_output = result%test_loss
      write(unit, '(a,",",a,",",i0,",",i0,",",l1,",",l1,",",l1,",",6(es24.16,","),i0,",",i0,",", &
                   3(es24.16,","),i0,",",es24.16,",",es24.16,",",i0,",",i0)', iostat=write_status) &
         trim(task_name), trim(optimizer), current_seed, result%hp%trial_id, selected, &
         result%validation_holdout, result%test_holdout, result%hp%learning_rate, result%hp%beta1, &
         result%hp%beta2, result%hp%epsilon, result%hp%weight_decay, result%hp%rho, max_steps, result%steps, &
         result%train_initial, validation_output, test_output, result%test_status, result%elapsed, &
         result%test_elapsed, result%state_bytes, result%status
      if (write_status /= 0) call fatal('candidate CSV row write failed')
   end subroutine write_candidate

   subroutine make_initial_parameters(kind, current_seed, p)
      integer(int32), intent(in) :: kind
      integer(int64), intent(in) :: current_seed
      real(real64), intent(out) :: p(:)
      integer(int64) :: rng
      integer :: i
      rng = bench_seed_state(ieor(current_seed, int(z'5A17D3E4B9216C0F', int64)))
      select case (kind)
      case (TASK_ROSENBROCK)
         p = 1.0_real64
         do i = 1, size(p), 2
            p(i) = -1.2_real64
         end do
      case (TASK_RASTRIGIN)
         do i = 1, size(p)
            p(i) = 8.0_real64 * (bench_uniform(rng) - 0.5_real64)
         end do
      case (TASK_LOGISTIC, TASK_SPARSE_LOGISTIC)
         p = 0.0_real64
      case (TASK_MLP, TASK_TRANSFORMER_LM, TASK_MATRIX_FACTORIZATION)
         do i = 1, size(p)
            p(i) = 0.05_real64 * bench_normal(rng)
         end do
      case default
         do i = 1, size(p)
            p(i) = 0.1_real64 * bench_normal(rng)
         end do
      end select
   end subroutine make_initial_parameters

   pure real(real64) function ticks_to_seconds(first_tick, last_tick, rate) result(seconds)
      integer(int64), intent(in) :: first_tick, last_tick, rate
      seconds = 0.0_real64
      if (rate > 0_int64 .and. last_tick >= first_tick) &
         seconds = real(last_tick - first_tick, real64) / real(rate, real64)
   end function ticks_to_seconds

   subroutine fatal(message)
      character(len=*), intent(in) :: message
      if (unit /= 0) close(unit)
      write(error_unit, '(a)') trim(message)
      stop 1
   end subroutine fatal

end program bench_phase2_search
