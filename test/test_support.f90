! SPDX-License-Identifier: MIT
module test_support
   use, intrinsic :: iso_fortran_env, only: int32, output_unit
   use hyper_types, only: hv_t, hv_equal
   implicit none
   private

   integer(int32) :: checks = 0_int32
   integer(int32) :: failures = 0_int32

   public :: assert_true, assert_i32_equal, assert_hv_equal, finish_tests

contains

   subroutine assert_true(condition, message)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: message

      checks = checks + 1_int32
      if (.not. condition) then
         failures = failures + 1_int32
         write(output_unit, '(A,A)') 'FAIL: ', trim(message)
      end if
   end subroutine assert_true

   subroutine assert_i32_equal(actual, expected, message)
      integer(int32), intent(in) :: actual, expected
      character(len=*), intent(in) :: message

      call assert_true(actual == expected, message)
      if (actual /= expected) then
         write(output_unit, '(A,I0,A,I0)') '      expected=', expected, ', actual=', actual
      end if
   end subroutine assert_i32_equal

   subroutine assert_hv_equal(actual, expected, message)
      type(hv_t), intent(in) :: actual, expected
      character(len=*), intent(in) :: message

      call assert_true(hv_equal(actual, expected), message)
   end subroutine assert_hv_equal

   subroutine finish_tests()
      write(output_unit, '(A,I0,A,I0)') 'Checks: ', checks, ', failures: ', failures
      if (failures /= 0_int32) error stop 1
   end subroutine finish_tests

end module test_support
