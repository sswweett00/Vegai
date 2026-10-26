program test_bench_tuning
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan
   use bench_tuning
   use bench_contracts, only : BENCH_OK, BENCH_INVALID
   implicit none

   type(bench_hyperparameters_t) :: adamw_grid(BENCH_TUNING_TRIALS), adamw_repeat(BENCH_TUNING_TRIALS)
   type(bench_hyperparameters_t) :: prodigy_grid(BENCH_TUNING_TRIALS), sophia_grid(BENCH_TUNING_TRIALS)
   type(bench_hyperparameters_t) :: apex_grid(BENCH_TUNING_TRIALS)
   real(real64) :: validation_losses(BENCH_TUNING_TRIALS)
   integer(int32) :: status, best_index
   integer :: i

   call bench_make_tuning_grid('AdamW', 37_int64, adamw_grid, status)
   call require(status == BENCH_OK, 'AdamW receives exactly twenty candidate configurations')
   call bench_make_tuning_grid('AdamW', 37_int64, adamw_repeat, status)
   call require(status == BENCH_OK, 'same seed regenerates candidate configurations')
   do i = 1, BENCH_TUNING_TRIALS
      call require(adamw_grid(i)%trial_id == int(i, int32), 'candidate IDs are ordered and unique')
      call require(adamw_grid(i)%learning_rate == adamw_repeat(i)%learning_rate .and. &
                   adamw_grid(i)%beta1 == adamw_repeat(i)%beta1 .and. &
                   adamw_grid(i)%beta2 == adamw_repeat(i)%beta2, 'grid is deterministic')
      call require(adamw_grid(i)%learning_rate >= 1.0e-4_real64 .and. &
                   adamw_grid(i)%learning_rate <= 0.1_real64, 'AdamW learning rate remains in declared range')
      call require(adamw_grid(i)%beta1 >= 0.8_real64 .and. adamw_grid(i)%beta1 < 1.0_real64 .and. &
                   adamw_grid(i)%beta2 >= 0.95_real64 .and. adamw_grid(i)%beta2 < 1.0_real64, &
                   'AdamW beta values are admissible')
   end do

   call bench_make_tuning_grid('Apex', 37_int64, apex_grid, status)
   call require(status == BENCH_OK .and. all(apex_grid%beta1 >= 0.8_real64) .and. &
                all(apex_grid%beta2 >= 0.95_real64), 'Apex receives LR and both beta search dimensions')
   call bench_make_tuning_grid('not-an-optimizer', 37_int64, adamw_grid, status)
   call require(status == BENCH_INVALID, 'unsupported optimizer labels are rejected')
   call bench_make_tuning_grid('Prodigy', 37_int64, prodigy_grid, status)
   call require(status == BENCH_OK .and. all(prodigy_grid%learning_rate >= 0.1_real64) .and. &
                all(prodigy_grid%learning_rate <= 10.0_real64), 'Prodigy has a dedicated LR scale range')
   call bench_make_tuning_grid('Sophia-G exact-diagonal', 37_int64, sophia_grid, status)
   call require(status == BENCH_OK .and. all(sophia_grid%rho > 0.0_real64), &
                'Sophia-G candidates include positive rho values')

   do i = 1, BENCH_TUNING_TRIALS
      validation_losses(i) = real(BENCH_TUNING_TRIALS - i + 1, real64)
   end do
   validation_losses(3) = ieee_value(0.0_real64, ieee_quiet_nan)
   validation_losses(11) = -1.0_real64
   call bench_choose_best_validation(adamw_grid, validation_losses, best_index, status)
   call require(status == BENCH_OK .and. best_index == 11_int32, &
                'validation selector ignores non-finite trials and selects minimum finite loss')
   validation_losses = ieee_value(0.0_real64, ieee_quiet_nan)
   call bench_choose_best_validation(adamw_grid, validation_losses, best_index, status)
   call require(status == BENCH_INVALID .and. best_index == 0_int32, 'all failed tuning trials have no winner')

contains

   subroutine require(condition, message)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: message
      if (.not. condition) then
         write(*, '(a,a)') 'FAIL: ', trim(message)
         stop 1
      end if
   end subroutine require

end program test_bench_tuning
