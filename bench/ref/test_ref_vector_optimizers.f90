program test_ref_vector_optimizers
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan, ieee_is_finite
   use ref_vector_optimizers
   implicit none

   integer(int32), parameter :: kinds(16) = [REF_SGD, REF_MOMENTUM, REF_NESTEROV, REF_ADAGRAD, &
      REF_RMSPROP, REF_ADAM, REF_ADAMW, REF_NADAM, REF_RADAM, REF_LION, REF_ADAFACTOR_VECTOR, &
      REF_LAMB, REF_LARS, REF_SOPHIA_G, REF_PRODIGY, REF_SCHEDULE_FREE_ADAMW]
   type(ref_vector_state_t) :: state
   real(real64) :: parameter(1), gradient(1), nan_gradient(1), curvature(1), expected, schedulefree_train(1)
   integer(int32) :: status
   integer :: index

   curvature = 1.0_real64
   do index = 1, size(kinds)
      parameter = 1.0_real64
      gradient = 1.0_real64
      call ref_vector_initialize(kinds(index), 1_int64, state, status)
      call require(status == REF_OK, 'reference state initializes')
      if (kinds(index) == REF_SOPHIA_G) then
         call ref_vector_step(parameter, gradient, state, 0.1_real64, 0.9_real64, 0.99_real64, &
                              1.0e-8_real64, 0.0_real64, status, curvature, 0.04_real64)
      else
         call ref_vector_step(parameter, gradient, state, 0.1_real64, 0.9_real64, 0.99_real64, &
                              1.0e-8_real64, 0.0_real64, status)
      end if
      call require(status == REF_OK .and. all(ieee_is_finite(parameter)), 'one-step reference result is finite')
      if (kinds(index) == REF_SCHEDULE_FREE_ADAMW) then
         call ref_vector_eval_parameters(parameter, state, 0.9_real64, status)
         call require(status == REF_OK .and. all(ieee_is_finite(parameter)), &
                      'Schedule-Free AdamW train-to-eval conversion is finite')
      end if
      call require(ref_vector_state_bytes(state) >= 0_int64, 'state-byte accounting is non-negative')
      if (kinds(index) == REF_SGD) then
         call require(ref_vector_state_bytes(state) == 0_int64, 'SGD has no per-parameter optimizer state')
      end if
      call require(trim(ref_vector_name(kinds(index))) /= 'unknown', 'reference has a registered name')
      select case (kinds(index))
      case (REF_SGD, REF_MOMENTUM, REF_ADAGRAD, REF_ADAM, REF_ADAMW, REF_RADAM, &
            REF_LION, REF_ADAFACTOR_VECTOR, REF_LAMB, REF_SOPHIA_G, REF_SCHEDULE_FREE_ADAMW)
         expected = 0.9_real64
         call require(abs(parameter(1) - expected) < 2.0e-6_real64, 'hand-checked first-step update')
      case (REF_PRODIGY)
         expected = 0.9999999_real64
         call require(abs(parameter(1) - expected) < 2.0e-12_real64, 'hand-checked Prodigy first step')
      case (REF_RMSPROP)
         call require(abs(parameter(1)) < 2.0e-6_real64, 'hand-checked RMSprop first step')
      case (REF_NESTEROV)
         expected = 0.81_real64
         call require(abs(parameter(1) - expected) < 2.0e-6_real64, 'hand-checked Nesterov first step')
      case (REF_NADAM)
         call require(parameter(1) < 1.0_real64 .and. parameter(1) > 0.8_real64, 'scheduled NAdam moves downhill')
      case (REF_LARS)
         expected = 0.9999_real64
         call require(abs(parameter(1) - expected) < 2.0e-6_real64, 'hand-checked LARS first step')
      end select
      if (kinds(index) == REF_SCHEDULE_FREE_ADAMW) then
         gradient = parameter
         call ref_vector_step(parameter, gradient, state, 0.1_real64, 0.9_real64, 0.99_real64, &
                              1.0e-8_real64, 0.0_real64, status)
         call require(status == REF_OK, 'Schedule-Free AdamW second train step succeeds')
         schedulefree_train = parameter
         call ref_vector_eval_parameters(parameter, state, 0.9_real64, status)
         call require(status == REF_OK .and. abs(parameter(1) - schedulefree_train(1)) > 1.0e-12_real64, &
                      'Schedule-Free AdamW evaluation iterate differs after averaging')
      end if
   end do

   parameter = 1.0_real64
   gradient = 1.0_real64
   call ref_vector_initialize(REF_SGD, 1_int64, state, status)
   nan_gradient(1) = ieee_value(0.0_real64, ieee_quiet_nan)
   call ref_vector_step(parameter, nan_gradient, state, 0.1_real64, 0.9_real64, 0.99_real64, &
                        1.0e-8_real64, 0.0_real64, status)
   call require(status == REF_NUMERICAL .and. parameter(1) == 1.0_real64, &
                'NaN gradient is rejected before parameter mutation')

contains

   subroutine require(condition, message)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: message
      if (.not. condition) then
         write(*, '(a,a)') 'FAIL: ', trim(message)
         stop 1
      end if
   end subroutine require

end program test_ref_vector_optimizers
