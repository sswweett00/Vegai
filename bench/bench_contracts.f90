! Benchmark-only contracts. This module does not participate in Apex builds.
module bench_contracts
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   implicit none
   private

   integer(int32), parameter, public :: BENCH_OK = 0_int32
   integer(int32), parameter, public :: BENCH_INVALID = 1_int32
   integer(int32), parameter, public :: BENCH_NUMERICAL = 2_int32

   type, public :: bench_trial_t
      character(len=48) :: task = ''
      character(len=48) :: optimizer = ''
      integer(int64) :: seed = 1_int64
      integer(int64) :: max_steps = 0_int64
      integer(int64) :: steps = 0_int64
      integer(int64) :: state_bytes = 0_int64
      real(real64) :: learning_rate = 0.0_real64
      real(real64) :: beta1 = 0.0_real64
      real(real64) :: beta2 = 0.0_real64
      real(real64) :: epsilon = 0.0_real64
      real(real64) :: rho = 0.0_real64
      real(real64) :: initial_loss = 0.0_real64
      real(real64) :: final_loss = 0.0_real64
      real(real64) :: elapsed_seconds = 0.0_real64
      logical :: finite = .false.
      integer(int32) :: status = BENCH_INVALID
   end type bench_trial_t

   type, public :: bench_grid_t
      real(real64), allocatable :: learning_rates(:)
      real(real64), allocatable :: beta1_values(:)
      real(real64), allocatable :: beta2_values(:)
      real(real64), allocatable :: epsilon_values(:)
      integer(int32) :: max_trials = 20_int32
   end type bench_grid_t

   abstract interface
      subroutine bench_objective(parameters, gradient, loss, status)
         import :: real64, int32
         real(real64), contiguous, intent(in) :: parameters(:)
         real(real64), contiguous, intent(out) :: gradient(:)
         real(real64), intent(out) :: loss
         integer(int32), intent(out) :: status
      end subroutine bench_objective

      subroutine bench_update(parameters, gradient, optimizer_state, step, learning_rate, beta1, beta2, &
                              epsilon, status)
         import :: real64, int64, int32
         real(real64), contiguous, intent(inout) :: parameters(:)
         real(real64), contiguous, intent(in) :: gradient(:)
         real(real64), contiguous, intent(inout) :: optimizer_state(:, :)
         integer(int64), intent(in) :: step
         real(real64), intent(in) :: learning_rate, beta1, beta2, epsilon
         integer(int32), intent(out) :: status
      end subroutine bench_update
   end interface

   public :: bench_objective, bench_update, bench_seed_state, bench_uniform, bench_normal
   public :: bench_write_csv_header, bench_write_csv_row, bench_trial_is_valid

contains

   pure integer(int64) function bench_seed_state(seed) result(state)
      integer(int64), intent(in) :: seed
      state = seed
      if (state == 0_int64) state = int(z'6A09E667F3BCC909', int64)
   end function bench_seed_state

   real(real64) function bench_uniform(state) result(value)
      integer(int64), intent(inout) :: state
      integer(int64) :: mantissa
      ! xorshift64 state transition. Using only bitwise operations avoids
      ! signed-integer multiplication overflow and is fully deterministic.
      state = ieor(state, ishft(state, 13))
      state = ieor(state, shiftr(state, 7))
      state = ieor(state, ishft(state, 17))
      if (state == 0_int64) state = int(z'6A09E667F3BCC909', int64)
      mantissa = iand(shiftr(state, 11), int(z'001FFFFFFFFFFFFF', int64))
      value = real(mantissa, real64) / 9007199254740992.0_real64
   end function bench_uniform

   real(real64) function bench_normal(state) result(value)
      integer(int64), intent(inout) :: state
      real(real64) :: u1, u2
      u1 = max(bench_uniform(state), tiny(1.0_real64))
      u2 = bench_uniform(state)
      value = sqrt(-2.0_real64 * log(u1)) * cos(2.0_real64 * acos(-1.0_real64) * u2)
   end function bench_normal

   subroutine bench_write_csv_header(unit, status)
      integer, intent(in) :: unit
      integer(int32), intent(out) :: status
      integer :: io_status
      write(unit, '(a)', iostat=io_status) &
         'task,optimizer,seed,max_steps,steps,learning_rate,beta1,beta2,epsilon,state_bytes,'// &
         'rho,initial_loss,final_loss,elapsed_seconds,finite,status'
      status = BENCH_OK
      if (io_status /= 0) status = BENCH_INVALID
   end subroutine bench_write_csv_header

   subroutine bench_write_csv_row(unit, trial, status)
      integer, intent(in) :: unit
      type(bench_trial_t), intent(in) :: trial
      integer(int32), intent(out) :: status
      integer :: io_status
      write(unit, '(a,",",a,",",i0,",",i0,",",i0,",",4(es24.16,","),i0,",",es24.16,",", &
                   3(es24.16,","),l1,",",i0)', iostat=io_status) &
         trim(trial%task), trim(trial%optimizer), trial%seed, trial%max_steps, trial%steps, &
         trial%learning_rate, trial%beta1, trial%beta2, trial%epsilon, trial%state_bytes, trial%rho, &
         trial%initial_loss, trial%final_loss, trial%elapsed_seconds, trial%finite, trial%status
      status = BENCH_OK
      if (io_status /= 0) status = BENCH_INVALID
   end subroutine bench_write_csv_row

   pure logical function bench_trial_is_valid(trial)
      type(bench_trial_t), intent(in) :: trial
      bench_trial_is_valid = len_trim(trial%task) > 0 .and. len_trim(trial%optimizer) > 0 .and. &
         trial%max_steps > 0_int64 .and. &
         trial%steps >= 0_int64 .and. trial%steps <= trial%max_steps .and. trial%state_bytes >= 0_int64 .and. &
         ieee_is_finite(trial%learning_rate) .and. ieee_is_finite(trial%beta1) .and. &
         ieee_is_finite(trial%beta2) .and. ieee_is_finite(trial%epsilon) .and. ieee_is_finite(trial%rho) .and. &
         ieee_is_finite(trial%initial_loss) .and. ieee_is_finite(trial%final_loss) .and. &
         ieee_is_finite(trial%elapsed_seconds) .and. trial%elapsed_seconds >= 0.0_real64
   end function bench_trial_is_valid

end module bench_contracts
