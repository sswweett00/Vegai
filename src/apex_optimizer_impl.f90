submodule (apex_optimizer) apex_optimizer_impl
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   implicit none
contains

   module logical function apex_config_valid(config)
      type(apex_config_t), intent(in) :: config
      apex_config_valid = ieee_is_finite(config%learning_rate) .and. config%learning_rate > 0.0_real64 .and. &
         ieee_is_finite(config%beta1) .and. config%beta1 >= 0.0_real64 .and. config%beta1 < 1.0_real64 .and. &
         ieee_is_finite(config%beta2) .and. config%beta2 >= 0.0_real64 .and. config%beta2 < 1.0_real64 .and. &
         ieee_is_finite(config%epsilon) .and. config%epsilon > 0.0_real64 .and. &
         ieee_is_finite(config%gamma) .and. config%gamma >= 0.0_real64 .and. &
         ieee_is_finite(config%weight_decay) .and. config%weight_decay >= 0.0_real64 .and. &
         ieee_is_finite(config%gradient_clip) .and. config%gradient_clip > 0.0_real64 .and. &
         config%warmup_updates >= 0_int64 .and. config%total_updates >= 0_int64 .and. &
         ieee_is_finite(config%minimum_lr_ratio) .and. config%minimum_lr_ratio >= 0.0_real64 .and. &
         config%minimum_lr_ratio <= 1.0_real64 .and. &
         (config%scheduler == APEX_SCHEDULE_CONSTANT .or. &
          config%scheduler == APEX_SCHEDULE_WARMUP_INVSQRT .or. &
          config%scheduler == APEX_SCHEDULE_WARMUP_COSINE) .and. &
         (config%scheduler /= APEX_SCHEDULE_WARMUP_COSINE .or. &
          config%total_updates > config%warmup_updates) .and. &
         ieee_is_finite(config%learning_rate * config%gradient_clip)
   end function apex_config_valid

   module subroutine apex_make_factors(config, update_index, factors, error)
      type(apex_config_t), intent(in) :: config
      integer(int64), intent(in) :: update_index
      type(apex_factors_t), intent(out) :: factors
      type(apex_error_t), intent(out) :: error
      real(real64) :: rate, bias_correction
      call clear_error(error)
      factors = apex_factors_t()
      if (.not. apex_config_valid(config) .or. update_index <= 0_int64) then
         call set_error(error, APEX_INVALID_ARGUMENT, 'invalid Apex configuration or update index')
         return
      end if
      rate = scheduled_rate(config, update_index)
      bias_correction = 1.0_real64 - config%beta1**real(update_index, real64)
      if (.not. ieee_is_finite(rate) .or. .not. ieee_is_finite(bias_correction) .or. &
          rate <= 0.0_real64 .or. bias_correction <= 0.0_real64) then
         call set_error(error, APEX_NUMERICAL_ERROR, 'Apex schedule produced an invalid step scale')
         return
      end if
      factors%effective_rate = rate / bias_correction
      factors%decay_factor = 1.0_real64 - rate * config%weight_decay
      factors%beta1 = config%beta1
      factors%beta2 = config%beta2
      factors%epsilon = config%epsilon
      factors%gamma = config%gamma
      if (.not. ieee_is_finite(factors%effective_rate) .or. &
          .not. ieee_is_finite(factors%decay_factor) .or. factors%decay_factor < 0.0_real64) then
         call set_error(error, APEX_NUMERICAL_ERROR, 'Apex step scale is not finite or stable')
      end if
   end subroutine apex_make_factors

   module subroutine apex_update_column(weights, hidden, momentum, gradient_scale, factors, error)
      real(real32), contiguous, intent(inout) :: weights(:)
      real(real32), contiguous, intent(in) :: hidden(:)
      real(real64), contiguous, intent(inout) :: momentum(:)
      real(real64), intent(in) :: gradient_scale
      type(apex_factors_t), intent(in) :: factors
      type(apex_error_t), intent(out) :: error
      integer :: i
      real(real64) :: one_minus_beta1, old_m, g, phase_product, phase_factor, next_m, lookahead, psi
      call clear_error(error)
      if (.not. factors_valid(factors)) then
         call set_error(error, APEX_INVALID_ARGUMENT, 'Apex update factors are invalid')
         return
      end if
      if (size(weights) /= size(hidden) .or. size(weights) /= size(momentum) .or. size(weights) == 0) then
         call set_error(error, APEX_SHAPE_ERROR, 'Apex column arrays have incompatible shapes')
         return
      end if
      if (.not. ieee_is_finite(gradient_scale) .or. .not. all(ieee_is_finite(weights)) .or. &
          .not. all(ieee_is_finite(hidden)) .or. .not. all(ieee_is_finite(momentum))) then
         call set_error(error, APEX_NUMERICAL_ERROR, 'Apex column input contains NaN or infinity')
         return
      end if
      one_minus_beta1 = 1.0_real64 - factors%beta1
      do concurrent (i = 1:size(weights)) local(old_m, g, phase_product, phase_factor, next_m, lookahead, psi)
         old_m = momentum(i)
         g = gradient_scale * real(hidden(i), real64)
         phase_product = g * old_m
         phase_factor = merge(1.0_real64 + 0.1_real64 * factors%beta2, &
                              merge(factors%beta2, 1.0_real64, phase_product < 0.0_real64), &
                              phase_product > 0.0_real64)
         next_m = factors%beta1 * old_m + one_minus_beta1 * g
         lookahead = factors%beta1 * next_m + one_minus_beta1 * g
         psi = lookahead / (1.0_real64 + factors%epsilon + factors%gamma * abs(lookahead))
         momentum(i) = next_m
         weights(i) = real(factors%decay_factor * real(weights(i), real64) - &
                           factors%effective_rate * phase_factor * psi, real32)
      end do
      if (.not. all(ieee_is_finite(weights)) .or. .not. all(ieee_is_finite(momentum))) &
         call set_error(error, APEX_NUMERICAL_ERROR, 'Apex column update generated a non-finite value')
   end subroutine apex_update_column

   module subroutine apex_update_scalar(weight, gradient, momentum, factors, error)
      real(real32), intent(inout) :: weight
      real(real64), intent(in) :: gradient
      real(real64), intent(inout) :: momentum
      type(apex_factors_t), intent(in) :: factors
      type(apex_error_t), intent(out) :: error
      real(real64) :: phase_product, phase_factor, next_m, lookahead, psi
      call clear_error(error)
      if (.not. factors_valid(factors)) then
         call set_error(error, APEX_INVALID_ARGUMENT, 'Apex update factors are invalid')
         return
      end if
      if (.not. ieee_is_finite(weight) .or. .not. ieee_is_finite(gradient) .or. &
          .not. ieee_is_finite(momentum)) then
         call set_error(error, APEX_NUMERICAL_ERROR, 'Apex scalar input contains NaN or infinity')
         return
      end if
      phase_product = gradient * momentum
      phase_factor = merge(1.0_real64 + 0.1_real64 * factors%beta2, &
                           merge(factors%beta2, 1.0_real64, phase_product < 0.0_real64), &
                           phase_product > 0.0_real64)
      next_m = factors%beta1 * momentum + (1.0_real64 - factors%beta1) * gradient
      lookahead = factors%beta1 * next_m + (1.0_real64 - factors%beta1) * gradient
      psi = lookahead / (1.0_real64 + factors%epsilon + factors%gamma * abs(lookahead))
      momentum = next_m
      weight = real(factors%decay_factor * real(weight, real64) - &
                    factors%effective_rate * phase_factor * psi, real32)
      if (.not. ieee_is_finite(weight) .or. .not. ieee_is_finite(momentum)) &
         call set_error(error, APEX_NUMERICAL_ERROR, 'Apex scalar update generated a non-finite value')
   end subroutine apex_update_scalar

   module subroutine apex_initialize(state, initial_parameters, config, error)
      type(apex_state_t), intent(inout) :: state
      real(real64), contiguous, intent(in) :: initial_parameters(:)
      type(apex_config_t), intent(in) :: config
      type(apex_error_t), intent(out) :: error
      type(apex_state_t) :: candidate
      integer :: allocation_status
      call clear_error(error)
      if (size(initial_parameters) == 0 .or. .not. apex_config_valid(config)) then
         call set_error(error, APEX_INVALID_ARGUMENT, 'invalid shape or Apex configuration')
         return
      end if
      if (.not. all(ieee_is_finite(initial_parameters))) then
         call set_error(error, APEX_NUMERICAL_ERROR, 'initial parameters contain NaN or infinity')
         return
      end if
      allocate(candidate%parameters(size(initial_parameters)), candidate%momentum(size(initial_parameters)), &
               stat=allocation_status)
      if (allocation_status /= 0) then
         call set_error(error, APEX_ALLOCATION_ERROR, 'could not allocate Apex state')
         return
      end if
      candidate%parameters = initial_parameters
      candidate%momentum = 0.0_real64
      candidate%updates = 0_int64
      candidate%config = config
      state = candidate
   end subroutine apex_initialize

   module subroutine apex_update(state, gradient, error)
      type(apex_state_t), intent(inout) :: state
      real(real64), contiguous, intent(in) :: gradient(:)
      type(apex_error_t), intent(out) :: error
      integer :: i
      real(real64) :: max_gradient, scaled_norm, clip_scale
      real(real64) :: old_m, g, phase_product, phase_factor, next_m, lookahead, psi
      type(apex_factors_t) :: factors

      call clear_error(error)
      if (.not. allocated(state%parameters) .or. .not. allocated(state%momentum)) then
         call set_error(error, APEX_INVALID_ARGUMENT, 'Apex state has not been initialized')
         return
      end if
      if (size(gradient) /= size(state%parameters) .or. size(state%momentum) /= size(state%parameters)) then
         call set_error(error, APEX_SHAPE_ERROR, 'gradient shape does not match Apex state')
         return
      end if
      if (.not. apex_config_valid(state%config)) then
         call set_error(error, APEX_INVALID_ARGUMENT, 'Apex configuration is invalid')
         return
      end if
      if (state%updates == huge(state%updates)) then
         call set_error(error, APEX_INVALID_ARGUMENT, 'Apex update counter is exhausted')
         return
      end if
      if (.not. all(ieee_is_finite(gradient)) .or. .not. all(ieee_is_finite(state%parameters)) .or. &
          .not. all(ieee_is_finite(state%momentum))) then
         call set_error(error, APEX_NUMERICAL_ERROR, 'Apex input state contains NaN or infinity')
         return
      end if
      call apex_make_factors(state%config, state%updates + 1_int64, factors, error)
      if (error%code /= APEX_OK) return
      max_gradient = maxval(abs(gradient))
      clip_scale = 1.0_real64
      if (max_gradient > 0.0_real64) then
         scaled_norm = sqrt(sum((gradient / max_gradient)**2))
         if (max_gradient > state%config%gradient_clip / scaled_norm) &
            clip_scale = (state%config%gradient_clip / max_gradient) / scaled_norm
      end if
      associate (weights => state%parameters, first_state => state%momentum)
         do concurrent (i = 1:size(gradient)) local(old_m, g, phase_product, phase_factor, next_m, lookahead, psi)
            old_m = first_state(i)
            g = gradient(i) * clip_scale
            phase_product = g * old_m
            phase_factor = merge(1.0_real64 + 0.1_real64 * factors%beta2, &
                                 merge(factors%beta2, 1.0_real64, phase_product < 0.0_real64), &
                                 phase_product > 0.0_real64)
            next_m = factors%beta1 * old_m + (1.0_real64 - factors%beta1) * g
            lookahead = factors%beta1 * next_m + (1.0_real64 - factors%beta1) * g
            psi = lookahead / (1.0_real64 + factors%epsilon + factors%gamma * abs(lookahead))
            first_state(i) = next_m
            weights(i) = factors%decay_factor * weights(i) - factors%effective_rate * phase_factor * psi
         end do
      end associate
      state%updates = state%updates + 1_int64
      if (.not. all(ieee_is_finite(state%parameters)) .or. .not. all(ieee_is_finite(state%momentum))) &
         call set_error(error, APEX_NUMERICAL_ERROR, 'Apex update generated a non-finite state')
   end subroutine apex_update

   module subroutine apex_optimize(state, gradient_callback, update_count, error)
      type(apex_state_t), intent(inout) :: state
      procedure(apex_gradient_callback) :: gradient_callback
      integer(int64), intent(in) :: update_count
      type(apex_error_t), intent(out) :: error
      real(real64), allocatable :: gradient(:)
      integer(int64) :: iteration
      integer :: allocation_status
      type(apex_error_t) :: callback_error
      call clear_error(error)
      if (update_count < 0_int64 .or. .not. allocated(state%parameters)) then
         call set_error(error, APEX_INVALID_ARGUMENT, 'invalid update count or uninitialized Apex state')
         return
      end if
      allocate(gradient(size(state%parameters)), stat=allocation_status)
      if (allocation_status /= 0) then
         call set_error(error, APEX_ALLOCATION_ERROR, 'could not allocate gradient workspace')
         return
      end if
      do iteration = 1_int64, update_count
         block
            real(real64) :: objective
            call gradient_callback(state%parameters, gradient, objective, callback_error)
            if (callback_error%code /= APEX_OK) then
               error = callback_error
               return
            end if
            if (.not. ieee_is_finite(objective)) then
               call set_error(error, APEX_NUMERICAL_ERROR, 'objective callback returned NaN or infinity')
               return
            end if
            call apex_update(state, gradient, error)
            if (error%code /= APEX_OK) return
         end block
      end do
   end subroutine apex_optimize

   module subroutine apex_get_parameters(state, parameters, error)
      type(apex_state_t), intent(in) :: state
      real(real64), contiguous, intent(out) :: parameters(:)
      type(apex_error_t), intent(out) :: error
      call clear_error(error)
      if (.not. allocated(state%parameters)) then
         call set_error(error, APEX_INVALID_ARGUMENT, 'Apex state has not been initialized')
      else if (size(parameters) /= size(state%parameters)) then
         call set_error(error, APEX_SHAPE_ERROR, 'output parameter shape does not match Apex state')
      else
         parameters = state%parameters
      end if
   end subroutine apex_get_parameters

   module integer(int64) function apex_update_count(state)
      type(apex_state_t), intent(in) :: state
      apex_update_count = state%updates
   end function apex_update_count

   module subroutine apex_save(state, path, error)
      type(apex_state_t), intent(in) :: state
      character(len=*), intent(in) :: path
      type(apex_error_t), intent(out) :: error
      character(len=8), parameter :: magic = 'APEXF002'
      integer(int32), parameter :: format_version = 2_int32
      integer :: unit, io_status, close_status
      integer(int64) :: parameter_count
      call clear_error(error)
      if (.not. allocated(state%parameters) .or. .not. allocated(state%momentum) .or. &
          .not. apex_config_valid(state%config)) then
         call set_error(error, APEX_INVALID_ARGUMENT, 'cannot save an uninitialized or invalid Apex state')
         return
      end if
      parameter_count = size(state%parameters, kind=int64)
      open(newunit=unit, file=path, status='replace', access='stream', form='unformatted', &
           action='write', iostat=io_status)
      if (io_status /= 0) then
         call set_error(error, APEX_IO_ERROR, 'could not open Apex state for writing')
         return
      end if
      write(unit, iostat=io_status) magic, format_version, parameter_count, state%updates
      if (io_status == 0) write(unit, iostat=io_status) state%config%learning_rate, state%config%beta1, &
         state%config%beta2, state%config%epsilon, state%config%gamma, state%config%weight_decay, &
         state%config%gradient_clip, state%config%warmup_updates, state%config%total_updates, &
         state%config%scheduler, state%config%minimum_lr_ratio
      if (io_status == 0) write(unit, iostat=io_status) state%parameters
      if (io_status == 0) write(unit, iostat=io_status) state%momentum
      close(unit, iostat=close_status)
      if (io_status /= 0 .or. close_status /= 0) &
         call set_error(error, APEX_IO_ERROR, 'failed writing Apex state stream')
   end subroutine apex_save

   module subroutine apex_load(state, path, error)
      type(apex_state_t), intent(inout) :: state
      character(len=*), intent(in) :: path
      type(apex_error_t), intent(out) :: error
      type(apex_state_t) :: candidate
      character(len=8), parameter :: expected_magic = 'APEXF002'
      character(len=8) :: magic
      integer(int32) :: format_version
      integer(int64) :: parameter_count
      integer :: unit, io_status, close_status, allocation_status
      call clear_error(error)
      open(newunit=unit, file=path, status='old', access='stream', form='unformatted', &
           action='read', iostat=io_status)
      if (io_status /= 0) then
         call set_error(error, APEX_IO_ERROR, 'could not open Apex state stream')
         return
      end if
      read(unit, iostat=io_status) magic, format_version, parameter_count, candidate%updates
      if (io_status == 0 .and. (magic /= expected_magic .or. format_version /= 2_int32 .or. &
                                parameter_count <= 0_int64 .or. candidate%updates < 0_int64)) io_status = 1
      if (io_status == 0) read(unit, iostat=io_status) candidate%config%learning_rate, candidate%config%beta1, &
         candidate%config%beta2, candidate%config%epsilon, candidate%config%gamma, candidate%config%weight_decay, &
         candidate%config%gradient_clip, candidate%config%warmup_updates, candidate%config%total_updates, &
         candidate%config%scheduler, candidate%config%minimum_lr_ratio
      if (io_status == 0 .and. .not. apex_config_valid(candidate%config)) io_status = 1
      if (io_status == 0) then
         allocate(candidate%parameters(parameter_count), candidate%momentum(parameter_count), stat=allocation_status)
         if (allocation_status /= 0) io_status = 1
      end if
      if (io_status == 0) read(unit, iostat=io_status) candidate%parameters
      if (io_status == 0) read(unit, iostat=io_status) candidate%momentum
      close(unit, iostat=close_status)
      if (io_status /= 0 .or. close_status /= 0) then
         call set_error(error, APEX_IO_ERROR, 'Apex state stream is truncated or invalid')
         return
      end if
      if (.not. all(ieee_is_finite(candidate%parameters)) .or. &
          .not. all(ieee_is_finite(candidate%momentum))) then
         call set_error(error, APEX_NUMERICAL_ERROR, 'saved Apex state contains NaN or infinity')
         return
      end if
      state = candidate
   end subroutine apex_load

   module subroutine apex_read_config(path, config, error)
      character(len=*), intent(in) :: path
      type(apex_config_t), intent(out) :: config
      type(apex_error_t), intent(out) :: error
      real(real64) :: learning_rate, beta1, beta2, epsilon, gamma, weight_decay
      real(real64) :: gradient_clip, minimum_lr_ratio
      integer(int64) :: warmup_updates, total_updates
      character(len=32) :: scheduler_name
      integer :: unit, io_status
      namelist /apex/ learning_rate, beta1, beta2, epsilon, gamma, weight_decay, gradient_clip, &
                      warmup_updates, total_updates, scheduler_name, minimum_lr_ratio
      call clear_error(error)
      learning_rate = 1.0e-3_real64
      beta1 = 0.9_real64
      beta2 = 0.5_real64
      epsilon = 1.0e-8_real64
      gamma = 0.5_real64
      weight_decay = 0.0_real64
      gradient_clip = 1.0_real64
      warmup_updates = 0_int64
      total_updates = 0_int64
      minimum_lr_ratio = 0.1_real64
      scheduler_name = 'constant'
      open(newunit=unit, file=path, status='old', action='read', iostat=io_status)
      if (io_status /= 0) then
         call set_error(error, APEX_IO_ERROR, 'could not open Apex namelist')
         return
      end if
      read(unit, nml=apex, iostat=io_status)
      close(unit)
      if (io_status /= 0) then
         call set_error(error, APEX_IO_ERROR, 'could not parse Apex namelist')
         return
      end if
      config%learning_rate = learning_rate
      config%beta1 = beta1
      config%beta2 = beta2
      config%epsilon = epsilon
      config%gamma = gamma
      config%weight_decay = weight_decay
      config%gradient_clip = gradient_clip
      config%warmup_updates = warmup_updates
      config%total_updates = total_updates
      config%minimum_lr_ratio = minimum_lr_ratio
      select case (trim(scheduler_name))
      case ('constant')
         config%scheduler = APEX_SCHEDULE_CONSTANT
      case ('invsqrt')
         config%scheduler = APEX_SCHEDULE_WARMUP_INVSQRT
      case ('cosine')
         config%scheduler = APEX_SCHEDULE_WARMUP_COSINE
      case default
         call set_error(error, APEX_INVALID_ARGUMENT, 'unsupported Apex scheduler')
         return
      end select
      if (.not. apex_config_valid(config)) call set_error(error, APEX_INVALID_ARGUMENT, 'invalid Apex config values')
   end subroutine apex_read_config

   pure real(real64) function scheduled_rate(config, update_index)
      type(apex_config_t), intent(in) :: config
      integer(int64), intent(in) :: update_index
      real(real64) :: progress, warmup_fraction
      integer(int64) :: warmup
      warmup = config%warmup_updates
      select case (config%scheduler)
      case (APEX_SCHEDULE_WARMUP_INVSQRT)
         if (warmup > 0_int64 .and. update_index <= warmup) then
            warmup_fraction = real(update_index, real64) / real(warmup, real64)
            scheduled_rate = config%learning_rate * warmup_fraction
         else if (warmup > 0_int64) then
            scheduled_rate = config%learning_rate * sqrt(real(warmup, real64) / real(update_index, real64))
         else
            scheduled_rate = config%learning_rate / sqrt(real(update_index, real64))
         end if
      case (APEX_SCHEDULE_WARMUP_COSINE)
         if (warmup > 0_int64 .and. update_index <= warmup) then
            scheduled_rate = config%learning_rate * real(update_index, real64) / real(warmup, real64)
         else
            progress = min(1.0_real64, real(update_index - warmup, real64) / &
                                      real(config%total_updates - warmup, real64))
            scheduled_rate = config%learning_rate * &
               (config%minimum_lr_ratio + (1.0_real64 - config%minimum_lr_ratio) * &
                0.5_real64 * (1.0_real64 + cos(acos(-1.0_real64) * progress)))
         end if
      case default
         scheduled_rate = config%learning_rate
      end select
   end function scheduled_rate

   pure logical function factors_valid(factors)
      type(apex_factors_t), intent(in) :: factors
      factors_valid = ieee_is_finite(factors%effective_rate) .and. factors%effective_rate > 0.0_real64 .and. &
         ieee_is_finite(factors%decay_factor) .and. factors%decay_factor >= 0.0_real64 .and. &
         ieee_is_finite(factors%beta1) .and. factors%beta1 >= 0.0_real64 .and. factors%beta1 < 1.0_real64 .and. &
         ieee_is_finite(factors%beta2) .and. factors%beta2 >= 0.0_real64 .and. factors%beta2 < 1.0_real64 .and. &
         ieee_is_finite(factors%epsilon) .and. factors%epsilon > 0.0_real64 .and. &
         ieee_is_finite(factors%gamma) .and. factors%gamma >= 0.0_real64
   end function factors_valid

   subroutine clear_error(error)
      type(apex_error_t), intent(out) :: error
      error%code = APEX_OK
      error%message = ''
   end subroutine clear_error

   subroutine set_error(error, code, message)
      type(apex_error_t), intent(out) :: error
      integer(c_int), intent(in) :: code
      character(len=*), intent(in) :: message
      error%code = code
      error%message = message
   end subroutine set_error

end submodule apex_optimizer_impl
