program test_ref_matrix_optimizers
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan, ieee_is_finite
   use ref_matrix_optimizers
   implicit none

   type(ref_matrix_state_t) :: shampoo_state, muon_state, adafactor_state
   real(real64) :: parameters(2,2), gradient(2,2)
   real(real64) :: wide_parameters(2,3), wide_gradient(2,3), nan_gradient(2,3), wide_before(2,3)
   integer(int32) :: status

   parameters = 1.0_real64
   gradient = reshape([1.0_real64, 0.0_real64, 0.0_real64, -1.0_real64], [2,2])
   call ref_matrix_initialize(REF_SHAMPOO, 2_int64, 2_int64, shampoo_state, status)
   call require(status == REF_MATRIX_OK, 'Shampoo matrix state initializes')
   call require(ref_matrix_state_bytes(shampoo_state) == 96_int64, 'Shampoo state byte count includes momentum and factors')
   call ref_matrix_step(parameters, gradient, shampoo_state, 0.01_real64, 0.9_real64, 1.0e-6_real64, status)
   call require(status == REF_MATRIX_OK .and. all(ieee_is_finite(parameters)), 'Shampoo update remains finite')

   wide_parameters = 1.0_real64
   wide_gradient = 1.0_real64
   call ref_matrix_initialize(REF_ADAFACTOR_MATRIX, 2_int64, 3_int64, adafactor_state, status)
   call require(status == REF_MATRIX_OK, 'factored Adafactor state initializes')
   call require(ref_matrix_state_bytes(adafactor_state) == 40_int64, 'factored Adafactor stores row and column statistics')
   call ref_matrix_step(wide_parameters, wide_gradient, adafactor_state, &
                        0.01_real64, 0.9_real64, 1.0e-6_real64, status)
   call require(status == REF_MATRIX_OK .and. all(abs(wide_parameters - 0.99_real64) < 2.0e-6_real64), &
                'factored Adafactor first step matches unit-gradient hand calculation')

   wide_parameters = 1.0_real64
   wide_gradient = reshape([1.0_real64, -0.5_real64, 0.25_real64, 0.75_real64, -0.3_real64, 0.2_real64], [2,3])
   call ref_matrix_initialize(REF_MUON, 2_int64, 3_int64, muon_state, status)
   call require(status == REF_MATRIX_OK, 'Muon matrix state initializes')
   call require(ref_matrix_state_bytes(muon_state) == 48_int64, 'Muon state stores one real64 momentum matrix')
   call ref_matrix_step(parameters(1:2,1:2), gradient(1:2,1:2), muon_state, &
                        0.01_real64, 0.9_real64, 1.0e-6_real64, status)
   call require(status == REF_MATRIX_INVALID, 'matrix shape mismatch is rejected')

   wide_parameters = 1.0_real64
   call ref_matrix_initialize(REF_MUON, 2_int64, 3_int64, muon_state, status)
   call require(status == REF_MATRIX_OK, 'Muon state reinitializes for valid shape')
   call ref_matrix_step(wide_parameters, wide_gradient, muon_state, 0.01_real64, 0.9_real64, 1.0e-6_real64, status)
   call require(status == REF_MATRIX_OK .and. all(ieee_is_finite(wide_parameters)), &
                'Muon orthogonalized update remains finite')

   wide_before = wide_parameters
   nan_gradient = wide_gradient
   nan_gradient(1,1) = ieee_value(0.0_real64, ieee_quiet_nan)
   call ref_matrix_step(wide_parameters, nan_gradient, muon_state, 0.01_real64, 0.9_real64, 1.0e-6_real64, status)
   call require(status == REF_MATRIX_INVALID .or. status == REF_MATRIX_NUMERICAL, 'non-finite matrix gradient is rejected')
   call require(all(wide_parameters == wide_before), 'invalid matrix input leaves parameters unchanged')

contains

   subroutine require(condition, message)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: message
      if (.not. condition) then
         write(*, '(a,a)') 'FAIL: ', trim(message)
         stop 1
      end if
   end subroutine require

end program test_ref_matrix_optimizers
