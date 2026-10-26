program test_bench_tasks
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan, ieee_is_finite
   use bench_tasks
   use bench_contracts, only : BENCH_OK, BENCH_NUMERICAL
   implicit none

   type(bench_task_t) :: quadratic, rosenbrock, rastrigin, logistic_a, logistic_b
   type(bench_task_t) :: sparse_a, sparse_b, mlp, transformer, factorization, noisy
   real(real64) :: parameters(3), gradient(3), loss, repeat_gradient(3), repeat_loss, nan_parameters(3)
   real(real64), allocatable :: diagonal(:)
   real(real64) :: rosen_x(2), rosen_gradient(2), rosen_loss, quad_x(2), quad_gradient(2), quad_loss
   real(real64) :: mlp_parameters(121), transformer_parameters(112), factor_parameters(54)
   real(real64) :: noise_grad_a(3), noise_grad_b(3), noise_grad_c(3), noise_loss_a, noise_loss_b, noise_loss_c
   integer(int32) :: status
   integer :: index

   call bench_task_initialize(quadratic, TASK_QUADRATIC, 2_int64, 1_int64, status, 100.0_real64)
   call require(status == BENCH_OK, 'ill-conditioned quadratic task initializes')
   call require(.not. bench_task_has_validation_split(quadratic) .and. .not. bench_task_has_test_split(quadratic), &
                'analytic quadratic reports that it has no held-out data partition')
   call bench_task_curvature_diagonal(quadratic, diagonal, status)
   call require(status == BENCH_OK .and. all(diagonal == [1.0_real64, 100.0_real64]), &
                'quadratic exposes exact diagonal curvature for Sophia-G')
   quad_x = [1.0_real64, 2.0_real64]
   call bench_task_evaluate(quadratic, quad_x, quad_gradient, quad_loss, status)
   call require(status == BENCH_OK .and. abs(quad_loss - 200.5_real64) < 1.0e-12_real64, &
                'ill-conditioned quadratic loss matches closed form')
   call require(all(abs(quad_gradient - [1.0_real64, 200.0_real64]) < 1.0e-12_real64), &
                'ill-conditioned quadratic gradient matches closed form')
   call bench_task_initialize(quadratic, TASK_QUADRATIC, 3_int64, 1_int64, status, 1.0e6_real64)
   call require(status == BENCH_OK, 'highly ill-conditioned quadratic (kappa 1e6) initializes')
   call bench_task_curvature_diagonal(quadratic, diagonal, status)
   call require(status == BENCH_OK .and. all(diagonal == [1.0_real64, 1.0e3_real64, 1.0e6_real64]), &
                'high-condition quadratic curvature spectrum is exact')

   call bench_task_initialize(rosenbrock, TASK_ROSENBROCK, 2_int64, 7_int64, status)
   rosen_x = [-1.2_real64, 1.0_real64]
   call bench_task_evaluate(rosenbrock, rosen_x, rosen_gradient, rosen_loss, status)
   call require(status == BENCH_OK .and. abs(rosen_loss - 24.2_real64) < 1.0e-12_real64, &
                'Rosenbrock loss matches known point')
   call require(all(abs(rosen_gradient - [-215.6_real64, -88.0_real64]) < 1.0e-10_real64), &
                'Rosenbrock gradient matches known point')

   call bench_task_initialize(rastrigin, TASK_RASTRIGIN, 3_int64, 11_int64, status)
   parameters = 0.0_real64
   call bench_task_evaluate(rastrigin, parameters, gradient, loss, status)
   call require(status == BENCH_OK .and. loss == 0.0_real64 .and. all(gradient == 0.0_real64), &
                'Rastrigin global minimum is exact')

   call bench_task_initialize(logistic_a, TASK_LOGISTIC, 3_int64, 123_int64, status, sample_count=64_int32)
   call bench_task_initialize(logistic_b, TASK_LOGISTIC, 3_int64, 123_int64, status, sample_count=64_int32)
   call require(status == BENCH_OK, 'dense logistic task initializes')
   call require(bench_task_has_validation_split(logistic_a) .and. bench_task_has_test_split(logistic_a), &
                'sample-based logistic task has held-out validation and test partitions')
   parameters = 0.0_real64
   call bench_task_evaluate(logistic_a, parameters, gradient, loss, status)
   call bench_task_evaluate(logistic_b, parameters, repeat_gradient, repeat_loss, status)
   call require(status == BENCH_OK .and. loss == repeat_loss .and. all(gradient == repeat_gradient), &
                'dense logistic synthetic data is reproducible by seed')
   call check_partitions(logistic_a, parameters, 64, 'dense logistic train/validation partitions evaluate')
   call bench_task_initialize(sparse_a, TASK_SPARSE_LOGISTIC, 3_int64, 123_int64, status, sample_count=64_int32)
   call bench_task_initialize(sparse_b, TASK_SPARSE_LOGISTIC, 3_int64, 123_int64, status, sample_count=64_int32)
   call bench_task_evaluate(sparse_a, parameters, gradient, loss, status)
   call bench_task_evaluate(sparse_b, parameters, repeat_gradient, repeat_loss, status)
   call require(status == BENCH_OK .and. loss == repeat_loss .and. all(gradient == repeat_gradient), &
                'sparse logistic task is deterministic by seed')
   call check_partitions(sparse_a, parameters, 64, 'sparse logistic train/validation partitions evaluate')

   call bench_task_initialize(mlp, TASK_MLP, 121_int64, 41_int64, status, sample_count=32_int32)
   call require(status == BENCH_OK, 'synthetic two-hidden-layer MLP task initializes')
   do index = 1, size(mlp_parameters)
      mlp_parameters(index) = 0.04_real64 * sin(real(index, real64))
   end do
   call check_gradient(mlp, mlp_parameters, 17, 'MLP analytic gradient matches finite difference')
   call check_partitions(mlp, mlp_parameters, 32, 'MLP train/validation partitions evaluate')

   call bench_task_initialize(transformer, TASK_TRANSFORMER_LM, 112_int64, 51_int64, status, sample_count=32_int32)
   call require(status == BENCH_OK, 'synthetic attention-plus-FFN LM task initializes')
   do index = 1, size(transformer_parameters)
      transformer_parameters(index) = 0.03_real64 * cos(real(index, real64))
   end do
   call check_gradient(transformer, transformer_parameters, 47, &
                       'transformer LM analytic gradient matches finite difference')
   call check_partitions(transformer, transformer_parameters, 32, 'transformer train/validation partitions evaluate')

   call bench_task_initialize(factorization, TASK_MATRIX_FACTORIZATION, 54_int64, 61_int64, status)
   call require(status == BENCH_OK, 'synthetic low-rank matrix factorization initializes')
   call require(bench_task_has_validation_split(factorization), &
                'matrix factorization declares a held-out rating partition')
   do index = 1, size(factor_parameters)
      factor_parameters(index) = 0.02_real64 * sin(0.3_real64 * real(index, real64))
   end do
   call check_gradient(factorization, factor_parameters, 20, &
                       'matrix factorization analytic gradient matches finite difference')
   call check_partitions(factorization, factor_parameters, 80, &
                         'matrix factorization train/validation masks evaluate independently')

   call bench_task_initialize(noisy, TASK_NOISY_QUADRATIC, 3_int64, 71_int64, status, &
                              condition_number=10.0_real64, noise_scale=0.25_real64)
   parameters = [0.2_real64, -0.4_real64, 0.7_real64]
   call bench_task_evaluate(noisy, parameters, noise_grad_a, noise_loss_a, status, 3_int64)
   call bench_task_evaluate(noisy, parameters, noise_grad_b, noise_loss_b, status, 3_int64)
   call bench_task_evaluate(noisy, parameters, noise_grad_c, noise_loss_c, status, 4_int64)
   call require(status == BENCH_OK .and. all(noise_grad_a == noise_grad_b) .and. noise_loss_a == noise_loss_b, &
                'noisy gradient is exactly replayable for a seed and step')
   call require(any(noise_grad_a /= noise_grad_c) .and. noise_loss_a == noise_loss_c, &
                'different update indices produce different noise without changing objective loss')

   nan_parameters = 0.0_real64
   nan_parameters(1) = ieee_value(0.0_real64, ieee_quiet_nan)
   call bench_task_evaluate(logistic_a, nan_parameters, gradient, loss, status)
   call require(status == BENCH_NUMERICAL, 'non-finite task parameters are rejected')

contains

   subroutine check_partitions(task, values, total_items, message)
      type(bench_task_t), intent(in) :: task
      real(real64), intent(in) :: values(:)
      integer, intent(in) :: total_items
      character(len=*), intent(in) :: message
      real(real64) :: train_gradient(size(values)), validation_gradient(size(values)), test_gradient(size(values))
      real(real64) :: full_gradient(size(values))
      real(real64) :: train_loss, validation_loss, test_loss, full_loss, combined_loss
      integer(int32) :: local_status
      integer :: train_items, validation_items, test_items
      train_items = max(1, (7 * total_items) / 10)
      validation_items = max(1, (15 * total_items) / 100)
      test_items = total_items - train_items - validation_items
      call bench_task_evaluate(task, values, train_gradient, train_loss, local_status, partition=TASK_PARTITION_TRAIN)
      call require(local_status == BENCH_OK .and. ieee_is_finite(train_loss) .and. &
                   all(ieee_is_finite(train_gradient)), 'training partition evaluation succeeds')
      call bench_task_evaluate(task, values, validation_gradient, validation_loss, local_status, &
                               partition=TASK_PARTITION_VALIDATION)
      call require(local_status == BENCH_OK .and. ieee_is_finite(validation_loss) .and. &
                   all(ieee_is_finite(validation_gradient)), message)
      call bench_task_evaluate(task, values, test_gradient, test_loss, local_status, partition=TASK_PARTITION_TEST)
      call require(local_status == BENCH_OK .and. ieee_is_finite(test_loss) .and. &
                   all(ieee_is_finite(test_gradient)), 'independent test partition evaluation succeeds')
      call bench_task_evaluate(task, values, full_gradient, full_loss, local_status)
      call require(local_status == BENCH_OK, 'full data partition evaluation succeeds')
      combined_loss = (real(train_items, real64) * train_loss + real(validation_items, real64) * validation_loss + &
                       real(test_items, real64) * test_loss) / real(total_items, real64)
      call require(abs(combined_loss - full_loss) < 1.0e-12_real64 * max(1.0_real64, abs(full_loss)), &
                   'train, validation and test partitions are disjoint and exhaustive')
      call require(all(abs((real(train_items, real64) * train_gradient + &
                            real(validation_items, real64) * validation_gradient + &
                            real(test_items, real64) * test_gradient) / real(total_items, real64) - full_gradient) < &
                       1.0e-12_real64 * max(1.0_real64, maxval(abs(full_gradient)))), &
                   'three partition gradients combine to full-data gradient')
      call bench_task_evaluate(task, values, full_gradient, full_loss, local_status, partition=99_int32)
      call require(local_status /= BENCH_OK, 'unknown data partition is rejected')
   end subroutine check_partitions

   subroutine check_gradient(task, values, coordinate, message)
      type(bench_task_t), intent(in) :: task
      real(real64), intent(in) :: values(:)
      integer, intent(in) :: coordinate
      character(len=*), intent(in) :: message
      real(real64) :: plus(size(values)), minus(size(values)), grad(size(values))
      real(real64) :: scratch(size(values)), loss_plus, loss_minus, loss_base, finite_difference, h
      integer(int32) :: local_status
      plus = values
      minus = values
      h = 1.0e-6_real64
      plus(coordinate) = plus(coordinate) + h
      minus(coordinate) = minus(coordinate) - h
      call bench_task_evaluate(task, values, grad, loss_base, local_status)
      call require(local_status == BENCH_OK .and. ieee_is_finite(loss_base), &
                   'gradient base-point evaluation succeeds')
      call bench_task_evaluate(task, plus, scratch, loss_plus, local_status)
      call require(local_status == BENCH_OK, 'gradient positive perturbation succeeds')
      call bench_task_evaluate(task, minus, scratch, loss_minus, local_status)
      call require(local_status == BENCH_OK, 'gradient negative perturbation succeeds')
      finite_difference = (loss_plus - loss_minus) / (2.0_real64 * h)
      call require(abs(grad(coordinate) - finite_difference) < 2.0e-5_real64 * &
                   max(1.0_real64, abs(finite_difference)), message)
   end subroutine check_gradient

   subroutine require(condition, message)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: message
      if (.not. condition) then
         write(*, '(a,a)') 'FAIL: ', trim(message)
         stop 1
      end if
   end subroutine require

end program test_bench_tasks
