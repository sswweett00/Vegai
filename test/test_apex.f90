program test_apex
   use, intrinsic :: iso_fortran_env, only : real32, real64, int64, output_unit
   use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan, ieee_positive_inf, ieee_is_finite
   use apex_optimizer
   implicit none

   type(apex_config_t) :: config, parsed_config
   type(apex_state_t) :: quadratic, logistic, restored, nan_state
   type(apex_error_t) :: error
   type(apex_factors_t) :: factors
   real(real64) :: initial(64), result(64), gradient(64), before(64)
   real(real64) :: logistic_initial(3), logistic_result(3), moment
   real(real32) :: weight
   real(real32) :: hidden(3), weights(3)
   real(real64) :: vector_momentum(3), nan_value, inf_value
   real(real64) :: initial_loss, final_loss
   integer(int64) :: index
   integer :: failures, unit

   failures = 0
   config%learning_rate = 0.03_real64
   config%beta1 = 0.9_real64
   config%beta2 = 0.5_real64
   config%epsilon = 1.0e-8_real64
   config%gamma = 0.5_real64
   config%weight_decay = 0.0_real64
   config%gradient_clip = 100.0_real64
   config%scheduler = APEX_SCHEDULE_CONSTANT
   call check(apex_config_valid(config), 'valid Apex configuration')
   config%learning_rate = -1.0_real64
   call check(.not. apex_config_valid(config), 'invalid configuration rejected')
   config%learning_rate = 0.03_real64

   do index = 1_int64, size(initial, kind=int64)
      initial(index) = 0.25_real64 * sin(real(index, real64) * 0.1_real64)
   end do
   initial_loss = 0.5_real64 * sum(initial * initial) / real(size(initial), real64)
   call apex_initialize(quadratic, initial, config, error)
   call check(error%code == APEX_OK, 'initialize quadratic state')
   call apex_optimize(quadratic, quadratic_gradient, 300_int64, error)
   call check(error%code == APEX_OK, 'quadratic callback optimization')
   call apex_get_parameters(quadratic, result, error)
   call check(error%code == APEX_OK .and. all(ieee_is_finite(result)), 'finite quadratic parameters')
   final_loss = 0.5_real64 * sum(result * result) / real(size(result), real64)
   call check(final_loss < initial_loss * 1.0e-8_real64, 'quadratic loss converges')
   call check(apex_update_count(quadratic) == 300_int64, 'update counter advances exactly')

   logistic_initial = 0.0_real64
   call apex_initialize(logistic, logistic_initial, config, error)
   call check(error%code == APEX_OK, 'initialize logistic state')
   call logistic_gradient(logistic_initial, gradient(1:3), initial_loss, error)
   call check(error%code == APEX_OK, 'logistic callback initial evaluation')
   call apex_optimize(logistic, logistic_gradient, 1000_int64, error)
   call check(error%code == APEX_OK, 'logistic regression callback optimization')
   call apex_get_parameters(logistic, logistic_result, error)
   call logistic_gradient(logistic_result, gradient(1:3), final_loss, error)
   call check(error%code == APEX_OK .and. final_loss < initial_loss * 0.25_real64, &
              'logistic cross-entropy decreases')

   call apex_get_parameters(quadratic, before, error)
   inf_value = ieee_value(0.0_real64, ieee_positive_inf)
   gradient = 0.0_real64
   gradient(1) = inf_value
   call apex_update(quadratic, gradient, error)
   call check(error%code == APEX_NUMERICAL_ERROR, 'infinite gradient rejected')
   call apex_get_parameters(quadratic, result, error)
   call check(all(result == before) .and. apex_update_count(quadratic) == 300_int64, &
              'invalid gradient leaves optimizer parameters and counter unchanged')

   nan_value = ieee_value(0.0_real64, ieee_quiet_nan)
   initial(1) = nan_value
   call apex_initialize(nan_state, initial, config, error)
   call check(error%code == APEX_NUMERICAL_ERROR, 'NaN initial parameters rejected')
   initial(1) = 0.0_real64

   call apex_make_factors(config, 1_int64, factors, error)
   call check(error%code == APEX_OK .and. factors%effective_rate > 0.0_real64, 'loop-invariant factors generated')
   weights = [0.1_real64, -0.2_real64, 0.3_real64]
   hidden = [0.2_real64, -0.4_real64, 0.6_real64]
   vector_momentum = 0.0_real64
   call apex_apply(weights, hidden, vector_momentum, 0.25_real64, factors, error)
   call check(error%code == APEX_OK .and. all(ieee_is_finite(weights)), 'generic vector-column API')
   weight = 0.5_real64
   moment = 0.0_real64
   call apex_apply(weight, -0.3_real64, moment, factors, error)
   call check(error%code == APEX_OK .and. ieee_is_finite(weight) .and. ieee_is_finite(moment), &
              'generic scalar API')

   open(newunit=unit, file='test-apex-config.nml', status='replace', action='write')
   write(unit, '(a)') '&apex learning_rate=0.02, beta1=0.8, beta2=0.4, epsilon=1.0e-8, gamma=0.7,'
   write(unit, '(a)') ' weight_decay=0.01, gradient_clip=2.0, warmup_updates=5, total_updates=20,'
   write(unit, '(a)') ' scheduler_name="cosine", minimum_lr_ratio=0.1 /'
   close(unit)
   call apex_read_config('test-apex-config.nml', parsed_config, error)
   call check(error%code == APEX_OK .and. parsed_config%scheduler == APEX_SCHEDULE_WARMUP_COSINE .and. &
              parsed_config%beta1 == 0.8_real64, 'runtime namelist configuration')
   open(newunit=unit, file='test-apex-config.nml', status='old')
   close(unit, status='delete')

   call apex_save(quadratic, 'test-apex-state.bin', error)
   call check(error%code == APEX_OK, 'save Apex state')
   call apex_load(restored, 'test-apex-state.bin', error)
   call check(error%code == APEX_OK, 'load Apex state')
   call apex_get_parameters(restored, result, error)
   call check(error%code == APEX_OK .and. all(result == before) .and. &
              apex_update_count(restored) == apex_update_count(quadratic), 'save/load exact round trip')
   gradient = 0.001_real64
   call apex_update(quadratic, gradient, error)
   call check(error%code == APEX_OK, 'advance original state after round trip')
   call apex_update(restored, gradient, error)
   call check(error%code == APEX_OK, 'advance restored state after round trip')
   call apex_get_parameters(quadratic, before, error)
   call apex_get_parameters(restored, result, error)
   call check(all(before == result), 'resumed state produces bit-identical next update')
   open(newunit=unit, file='test-apex-state.bin', status='old')
   close(unit, status='delete')

   if (failures == 0) then
      write(output_unit, '(a)') 'Apex tests: PASS'
   else
      write(output_unit, '(a,i0)') 'Apex tests: FAIL count=', failures
      stop 1
   end if

contains

   subroutine check(condition, label)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: label
      if (.not. condition) then
         failures = failures + 1
         write(output_unit, '(a,a)') 'FAIL: ', trim(label)
      end if
   end subroutine check

   subroutine quadratic_gradient(parameters, grad, objective, callback_error)
      real(real64), contiguous, intent(in) :: parameters(:)
      real(real64), contiguous, intent(out) :: grad(:)
      real(real64), intent(out) :: objective
      type(apex_error_t), intent(out) :: callback_error
      grad = parameters
      objective = 0.5_real64 * sum(parameters * parameters)
      callback_error%code = APEX_OK
      callback_error%message = ''
   end subroutine quadratic_gradient

   subroutine logistic_gradient(parameters, grad, objective, callback_error)
      real(real64), contiguous, intent(in) :: parameters(:)
      real(real64), contiguous, intent(out) :: grad(:)
      real(real64), intent(out) :: objective
      type(apex_error_t), intent(out) :: callback_error
      real(real64), parameter :: samples(2,4) = reshape([2.0_real64, 1.0_real64, &
         1.0_real64, 2.0_real64, -2.0_real64, -1.0_real64, -1.0_real64, -2.0_real64], [2,4])
      real(real64), parameter :: labels(4) = [1.0_real64, 1.0_real64, 0.0_real64, 0.0_real64]
      real(real64) :: z, probability, residual
      integer :: sample
      grad = 0.0_real64
      objective = 0.0_real64
      do sample = 1, size(labels)
         z = parameters(1) + parameters(2) * samples(1,sample) + parameters(3) * samples(2,sample)
         if (z >= 0.0_real64) then
            probability = 1.0_real64 / (1.0_real64 + exp(-z))
            objective = objective + (1.0_real64-labels(sample))*z + log(1.0_real64 + exp(-z))
         else
            probability = exp(z) / (1.0_real64 + exp(z))
            objective = objective - labels(sample)*z + log(1.0_real64 + exp(z))
         end if
         residual = probability - labels(sample)
         grad(1) = grad(1) + residual
         grad(2) = grad(2) + residual * samples(1,sample)
         grad(3) = grad(3) + residual * samples(2,sample)
      end do
      objective = objective / real(size(labels), real64)
      grad = grad / real(size(labels), real64)
      callback_error%code = APEX_OK
      callback_error%message = ''
   end subroutine logistic_gradient

end program test_apex
