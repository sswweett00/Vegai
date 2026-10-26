program test_bench_contracts
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   use bench_contracts
   implicit none

   type(bench_trial_t) :: trial
   integer(int64) :: first_state, second_state
   real(real64) :: first_value, second_value, gaussian
   integer :: unit, io_status
   integer(int32) :: status
   character(len=256) :: header

   first_state = bench_seed_state(73_int64)
   second_state = bench_seed_state(73_int64)
   first_value = bench_uniform(first_state)
   second_value = bench_uniform(second_state)
   call require(first_value == second_value, 'uniform stream is reproducible')
   call require(first_value >= 0.0_real64 .and. first_value < 1.0_real64, 'uniform is in [0,1)')
   gaussian = bench_normal(first_state)
   call require(ieee_is_finite(gaussian), 'normal sample is finite')

   trial%task = 'contract-test'
   trial%optimizer = 'test'
   trial%seed = 73_int64
   trial%max_steps = 10_int64
   trial%steps = 10_int64
   trial%state_bytes = 8_int64
   trial%learning_rate = 1.0e-2_real64
   trial%beta1 = 0.9_real64
   trial%beta2 = 0.5_real64
   trial%epsilon = 1.0e-8_real64
   trial%initial_loss = 1.0_real64
   trial%final_loss = 0.1_real64
   trial%elapsed_seconds = 0.25_real64
   trial%finite = .true.
   trial%status = BENCH_OK
   call require(bench_trial_is_valid(trial), 'valid trial record is accepted')

   open(newunit=unit, status='scratch', action='readwrite', form='formatted', iostat=io_status)
   call require(io_status == 0, 'scratch CSV stream opens')
   call bench_write_csv_header(unit, status)
   call require(status == BENCH_OK, 'CSV header writes')
   call bench_write_csv_row(unit, trial, status)
   call require(status == BENCH_OK, 'CSV row writes')
   rewind(unit, iostat=io_status)
   call require(io_status == 0, 'CSV stream rewinds')
   read(unit, '(a)', iostat=io_status) header
   call require(io_status == 0 .and. index(header, 'optimizer,seed') > 0, 'CSV header can be read back')
   close(unit)

contains

   subroutine require(condition, message)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: message
      if (.not. condition) then
         write(*, '(a,a)') 'FAIL: ', trim(message)
         stop 1
      end if
   end subroutine require

end program test_bench_contracts
