! SPDX-License-Identifier: MIT
! Native Fortran control-plane entry point.  Keep policy-critical portable
! profile, bounded resource arithmetic, and same-host native smoke execution
! inside the Fortran build; optional legacy adapters are not invoked here.
program vegai_control
   use, intrinsic :: iso_fortran_env, only: int32
   use fortran_control, only: fortran_control_main, FC_OK
   implicit none

   integer(int32) :: status

   call fortran_control_main(status)
   if (status /= FC_OK) error stop int(status)
end program vegai_control
