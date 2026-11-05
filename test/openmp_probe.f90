! SPDX-License-Identifier: MIT
! CMake registers this runtime probe when the operator requires OpenMP. fpm may
! discover it in either profile, in which case it skips if OpenMP is absent.
! This checks the runtime, not a speedup benchmark.
program openmp_probe
   use, intrinsic :: iso_fortran_env, only: output_unit
   !$ use omp_lib, only: omp_get_num_threads
   implicit none (type, external)
   integer :: threads
   logical :: openmp_enabled

   ! fpm also discovers this program in serial builds. OpenMP conditional
   ! sentinels keep it compilable without the module/runtime in that profile.
   openmp_enabled = .false.
   !$ openmp_enabled = .true.
   if (.not. openmp_enabled) then
      write(output_unit, '(A)') 'SKIP: OpenMP was not enabled for this test executable'
      stop
   end if
   threads = 0
   !$omp parallel num_threads(2) default(none) reduction(max:threads)
   !$ threads = omp_get_num_threads()
   !$omp end parallel
   if (threads /= 2) error stop 'OpenMP runtime did not create the required two-thread team'
   write(output_unit, '(A,I0)') 'OpenMP runtime team size: ', threads
end program openmp_probe
