! Fortran 2023 public contract for the single-state Apex optimizer.
module apex_optimizer
   use, intrinsic :: iso_fortran_env, only : real32, real64, int32, int64
   use, intrinsic :: iso_c_binding, only : c_int
   implicit none
   private

   enum, bind(c)
      enumerator :: APEX_OK = 0, APEX_INVALID_ARGUMENT = 1, APEX_NUMERICAL_ERROR = 2
      enumerator :: APEX_IO_ERROR = 3, APEX_SHAPE_ERROR = 4, APEX_ALLOCATION_ERROR = 5
   end enum
   enum, bind(c)
      enumerator :: APEX_SCHEDULE_CONSTANT = 0, APEX_SCHEDULE_WARMUP_INVSQRT = 1
      enumerator :: APEX_SCHEDULE_WARMUP_COSINE = 2
   end enum

   type, public :: apex_error_t
      integer(c_int) :: code = APEX_OK
      character(len=192) :: message = ''
   end type apex_error_t

   type, public :: apex_config_t
      real(real64) :: learning_rate = 1.0e-3_real64
      real(real64) :: beta1 = 0.9_real64
      ! Phase response: aligned gain is 1+0.1*beta2; reversal damping is beta2.
      real(real64) :: beta2 = 0.5_real64
      real(real64) :: epsilon = 1.0e-8_real64
      real(real64) :: gamma = 0.5_real64
      real(real64) :: weight_decay = 0.0_real64
      real(real64) :: gradient_clip = 1.0_real64
      integer(int64) :: warmup_updates = 0_int64
      integer(int64) :: total_updates = 0_int64
      integer(c_int) :: scheduler = APEX_SCHEDULE_CONSTANT
      real(real64) :: minimum_lr_ratio = 0.1_real64
   end type apex_config_t

   type, public :: apex_state_t
      private
      real(real64), allocatable :: parameters(:)
      real(real64), allocatable :: momentum(:)
      integer(int64) :: updates = 0_int64
      type(apex_config_t) :: config
   end type apex_state_t

   type, public :: apex_factors_t
      real(real64) :: effective_rate = 0.0_real64
      real(real64) :: decay_factor = 1.0_real64
      real(real64) :: beta1 = 0.9_real64
      real(real64) :: beta2 = 0.5_real64
      real(real64) :: epsilon = 1.0e-8_real64
      real(real64) :: gamma = 0.5_real64
   end type apex_factors_t

   abstract interface
      subroutine apex_gradient_callback(parameters, gradient, objective, error)
         import :: real64, apex_error_t
         real(real64), contiguous, intent(in) :: parameters(:)
         real(real64), contiguous, intent(out) :: gradient(:)
         real(real64), intent(out) :: objective
         type(apex_error_t), intent(out) :: error
      end subroutine apex_gradient_callback
   end interface

   public :: APEX_OK, APEX_INVALID_ARGUMENT, APEX_NUMERICAL_ERROR, APEX_IO_ERROR
   public :: APEX_SHAPE_ERROR, APEX_ALLOCATION_ERROR
   public :: APEX_SCHEDULE_CONSTANT, APEX_SCHEDULE_WARMUP_INVSQRT, APEX_SCHEDULE_WARMUP_COSINE
   public :: apex_config_t, apex_error_t, apex_state_t, apex_factors_t, apex_gradient_callback
   public :: apex_config_valid, apex_initialize, apex_update, apex_apply, apex_optimize
   public :: apex_get_parameters, apex_update_count, apex_save, apex_load, apex_read_config
   public :: apex_make_factors, apex_update_column, apex_update_scalar

   interface
      module logical function apex_config_valid(config)
         import :: apex_config_t
         type(apex_config_t), intent(in) :: config
      end function apex_config_valid

      module subroutine apex_make_factors(config, update_index, factors, error)
         import :: apex_config_t, apex_factors_t, apex_error_t, int64
         type(apex_config_t), intent(in) :: config
         integer(int64), intent(in) :: update_index
         type(apex_factors_t), intent(out) :: factors
         type(apex_error_t), intent(out) :: error
      end subroutine apex_make_factors

      module subroutine apex_update_column(weights, hidden, momentum, gradient_scale, factors, error)
         import :: real32, real64, apex_factors_t, apex_error_t
         real(real32), contiguous, intent(inout) :: weights(:)
         real(real32), contiguous, intent(in) :: hidden(:)
         real(real64), contiguous, intent(inout) :: momentum(:)
         real(real64), intent(in) :: gradient_scale
         type(apex_factors_t), intent(in) :: factors
         type(apex_error_t), intent(out) :: error
      end subroutine apex_update_column

      module subroutine apex_update_scalar(weight, gradient, momentum, factors, error)
         import :: real32, real64, apex_factors_t, apex_error_t
         real(real32), intent(inout) :: weight
         real(real64), intent(in) :: gradient
         real(real64), intent(inout) :: momentum
         type(apex_factors_t), intent(in) :: factors
         type(apex_error_t), intent(out) :: error
      end subroutine apex_update_scalar

      module subroutine apex_initialize(state, initial_parameters, config, error)
         import :: apex_state_t, apex_config_t, apex_error_t, real64
         type(apex_state_t), intent(inout) :: state
         real(real64), contiguous, intent(in) :: initial_parameters(:)
         type(apex_config_t), intent(in) :: config
         type(apex_error_t), intent(out) :: error
      end subroutine apex_initialize

      module subroutine apex_update(state, gradient, error)
         import :: apex_state_t, apex_error_t, real64
         type(apex_state_t), intent(inout) :: state
         real(real64), contiguous, intent(in) :: gradient(:)
         type(apex_error_t), intent(out) :: error
      end subroutine apex_update

      module subroutine apex_optimize(state, gradient_callback, update_count, error)
         import :: apex_state_t, apex_gradient_callback, apex_error_t, int64
         type(apex_state_t), intent(inout) :: state
         procedure(apex_gradient_callback) :: gradient_callback
         integer(int64), intent(in) :: update_count
         type(apex_error_t), intent(out) :: error
      end subroutine apex_optimize

      module subroutine apex_get_parameters(state, parameters, error)
         import :: apex_state_t, apex_error_t, real64
         type(apex_state_t), intent(in) :: state
         real(real64), contiguous, intent(out) :: parameters(:)
         type(apex_error_t), intent(out) :: error
      end subroutine apex_get_parameters

      module integer(int64) function apex_update_count(state)
         import :: apex_state_t, int64
         type(apex_state_t), intent(in) :: state
      end function apex_update_count

      module subroutine apex_save(state, path, error)
         import :: apex_state_t, apex_error_t
         type(apex_state_t), intent(in) :: state
         character(len=*), intent(in) :: path
         type(apex_error_t), intent(out) :: error
      end subroutine apex_save

      module subroutine apex_load(state, path, error)
         import :: apex_state_t, apex_error_t
         type(apex_state_t), intent(inout) :: state
         character(len=*), intent(in) :: path
         type(apex_error_t), intent(out) :: error
      end subroutine apex_load

      module subroutine apex_read_config(path, config, error)
         import :: apex_config_t, apex_error_t
         character(len=*), intent(in) :: path
         type(apex_config_t), intent(out) :: config
         type(apex_error_t), intent(out) :: error
      end subroutine apex_read_config
   end interface

   interface apex_apply
      module procedure apex_update, apex_update_column, apex_update_scalar
   end interface apex_apply

end module apex_optimizer
