! Equal-budget deterministic hyperparameter proposals for benchmark studies.
! Candidate generation is not a measurement or a ranking procedure.
module bench_tuning
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   use bench_contracts, only : bench_seed_state, bench_uniform, BENCH_OK, BENCH_INVALID
   implicit none
   private

   integer(int32), parameter, public :: BENCH_TUNING_TRIALS = 20_int32

   type, public :: bench_hyperparameters_t
      integer(int32) :: trial_id = 0_int32
      real(real64) :: learning_rate = 0.0_real64
      real(real64) :: beta1 = 0.0_real64
      real(real64) :: beta2 = 0.0_real64
      real(real64) :: epsilon = 0.0_real64
      real(real64) :: weight_decay = 0.0_real64
      real(real64) :: rho = 0.0_real64
   end type bench_hyperparameters_t

   public :: bench_make_tuning_grid, bench_choose_best_validation

contains

   subroutine bench_make_tuning_grid(optimizer, seed, grid, status)
      character(len=*), intent(in) :: optimizer
      integer(int64), intent(in) :: seed
      type(bench_hyperparameters_t), intent(out) :: grid(BENCH_TUNING_TRIALS)
      integer(int32), intent(out) :: status
      real(real64), parameter :: beta1_choices(4) = [0.8_real64, 0.9_real64, 0.95_real64, 0.99_real64]
      real(real64), parameter :: beta2_choices(4) = [0.95_real64, 0.99_real64, 0.999_real64, 0.9999_real64]
      real(real64), parameter :: epsilon_choices(4) = [1.0e-8_real64, 1.0e-6_real64, 1.0e-5_real64, 1.0e-4_real64]
      real(real64), parameter :: decay_choices(4) = [0.0_real64, 1.0e-5_real64, 1.0e-4_real64, 1.0e-3_real64]
      real(real64), parameter :: rho_choices(4) = [0.01_real64, 0.04_real64, 0.1_real64, 1.0_real64]
      real(real64) :: lower_log_lr, upper_log_lr, u
      integer(int64) :: rng
      integer :: i, index1, index2, index3, index4
      logical :: uses_beta1, uses_beta2

      status = BENCH_OK
      if (seed < 0_int64 .or. .not. optimizer_is_supported(optimizer)) then
         status = BENCH_INVALID
         return
      end if
      lower_log_lr = log10(1.0e-4_real64)
      upper_log_lr = log10(1.0e-1_real64)
      if (trim(optimizer) == 'Prodigy') then
         lower_log_lr = log10(0.1_real64)
         upper_log_lr = log10(10.0_real64)
      end if
      uses_beta1 = optimizer_uses_beta1(optimizer)
      uses_beta2 = optimizer_uses_beta2(optimizer)
      rng = bench_seed_state(ieor(seed, int(z'BB67AE8584CAA73B', int64)))
      do i = 1, BENCH_TUNING_TRIALS
         u = (real(i - 1, real64) + bench_uniform(rng)) / real(BENCH_TUNING_TRIALS, real64)
         grid(i)%trial_id = int(i, int32)
         grid(i)%learning_rate = 10.0_real64**(lower_log_lr + u * (upper_log_lr - lower_log_lr))
         index1 = 1 + int(bench_uniform(rng) * 4.0_real64)
         index2 = 1 + int(bench_uniform(rng) * 4.0_real64)
         index3 = 1 + int(bench_uniform(rng) * 4.0_real64)
         index4 = 1 + int(bench_uniform(rng) * 4.0_real64)
         grid(i)%beta1 = 0.9_real64
         grid(i)%beta2 = 0.99_real64
         if (uses_beta1) grid(i)%beta1 = beta1_choices(index1)
         if (uses_beta2) grid(i)%beta2 = beta2_choices(index2)
         grid(i)%epsilon = epsilon_choices(index3)
         grid(i)%weight_decay = decay_choices(index4)
         grid(i)%rho = rho_choices(1 + modulo(i - 1, size(rho_choices)))
         if (trim(optimizer) /= 'Sophia-G' .and. trim(optimizer) /= 'Sophia-G exact-diagonal') &
            grid(i)%rho = 0.0_real64
      end do
   end subroutine bench_make_tuning_grid

   subroutine bench_choose_best_validation(grid, validation_losses, best_index, status)
      type(bench_hyperparameters_t), intent(in) :: grid(:)
      real(real64), intent(in) :: validation_losses(:)
      integer(int32), intent(out) :: best_index, status
      integer :: i

      status = BENCH_INVALID
      best_index = 0_int32
      if (size(grid) == 0 .or. size(validation_losses) /= size(grid)) return
      do i = 1, size(validation_losses)
         if (.not. ieee_is_finite(validation_losses(i))) cycle
         if (best_index == 0_int32) then
            best_index = int(i, int32)
         else if (validation_losses(i) < validation_losses(best_index)) then
            best_index = int(i, int32)
         end if
      end do
      if (best_index > 0_int32) status = BENCH_OK
   end subroutine bench_choose_best_validation

   pure logical function optimizer_is_supported(optimizer)
      character(len=*), intent(in) :: optimizer
      select case (trim(optimizer))
      case ('Apex', 'SGD', 'SGD+momentum', 'Nesterov', 'AdaGrad', 'RMSprop', 'Adam', 'AdamW', 'NAdam', &
            'RAdam', 'Lion', 'Adafactor-vector', 'Adafactor-matrix', 'LAMB', 'LARS', &
            'Sophia-G exact-diagonal', 'Prodigy', 'Schedule-Free AdamW', 'Shampoo', 'Muon')
         optimizer_is_supported = .true.
      case default
         optimizer_is_supported = .false.
      end select
   end function optimizer_is_supported

   pure logical function optimizer_uses_beta1(optimizer)
      character(len=*), intent(in) :: optimizer
      select case (trim(optimizer))
      case ('Apex', 'SGD+momentum', 'Nesterov', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'Lion', &
            'LAMB', 'LARS', 'Sophia-G', 'Sophia-G exact-diagonal', 'Prodigy', 'Schedule-Free AdamW', 'Shampoo', 'Muon')
         optimizer_uses_beta1 = .true.
      case default
         optimizer_uses_beta1 = .false.
      end select
   end function optimizer_uses_beta1

   pure logical function optimizer_uses_beta2(optimizer)
      character(len=*), intent(in) :: optimizer
      select case (trim(optimizer))
      case ('Apex', 'RMSprop', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'Lion', 'LAMB', 'Sophia-G', &
            'Sophia-G exact-diagonal', &
            'Prodigy', 'Schedule-Free AdamW')
         optimizer_uses_beta2 = .true.
      case default
         optimizer_uses_beta2 = .false.
      end select
   end function optimizer_uses_beta2

end module bench_tuning
