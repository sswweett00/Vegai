! SPDX-License-Identifier: MIT
! V6/V7 comparison: one equal-budget train per arm, two held-out streams.
! Validation and scoring happen outside the timed training region.
program benchmark_slot_quality
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real64, output_unit
   use portable_cli, only: CLI_OK, cli_get_argument
   use fixed_budget_context
   use fixed_budget_backward
   use fixed_budget_optimizer
   use fixed_budget_training
   implicit none (type, external)
   integer(int32), parameter :: run_count = 5_int32
   integer(int32), parameter :: seeds(run_count) = [21_int32, 42_int32, 73_int32, 101_int32, 137_int32]
   integer(int64), parameter :: max_bytes = 16777216_int64
   type(fbc_config_t) :: config
   type(fbc_plan_t) :: plan
   type(fbc_update_config_t) :: update_config
   type(fbc_model_t) :: initial, model
   type(fbc_state_t) :: state
   type(fbc_workspace_t) :: workspace
   type(fbc_gradient_t) :: gradient
   type(fbc_optimizer_t) :: optimizer
   type(fbc_metrics_t) :: metrics
   integer(int32), allocatable :: train(:), a(:), b(:)
   logical, allocatable :: mask_a(:), mask_b(:)
   character(len=1024) :: train_path, a_path, b_path, argument
   integer(int32) :: chunk, lr_micros, train_count, a_count, b_count, status, run, pass, arm
   integer(int32) :: order(2), rule(2)
   integer(int64) :: start_clock, end_clock, clock_rate, answer_count
   integer :: arg_count, allocation_status
   real(real64) :: a_nll(2), b_nll(2), seconds(2), recall
   real(real64) :: delta_a(run_count), delta_b(run_count), ratios(run_count)
   logical :: prompt_shape

   arg_count = command_argument_count()
   if (arg_count < 3 .or. arg_count > 6) error stop 'expected train heldout-a heldout-b [chunk] [lr-micros] [small|prompt]'
   call get_arg(1, train_path, .true.)
   call get_arg(2, a_path, .true.)
   call get_arg(3, b_path, .true.)
   if (trim(train_path) == trim(a_path) .or. trim(train_path) == trim(b_path) .or. &
       trim(a_path) == trim(b_path)) error stop 'train and held-out paths must differ'
   chunk = 512_int32
   lr_micros = 6000_int32
   prompt_shape = .true.
   if (arg_count >= 4) then
      call get_arg(4, argument, .false.)
      call fbc_parse_count(trim(argument), 1_int32, FBC_MAX_CHUNK_TOKENS, chunk, status)
      if (status /= FBC_OK) error stop 'invalid chunk'
   end if
   if (arg_count >= 5) then
      call get_arg(5, argument, .false.)
      call fbc_parse_count(trim(argument), 1_int32, 999999_int32, lr_micros, status)
      if (status /= FBC_OK) error stop 'invalid learning rate'
   end if
   if (arg_count >= 6) then
      call get_arg(6, argument, .false.)
      select case (trim(argument))
      case ('small')
         prompt_shape = .false.
      case ('prompt')
         prompt_shape = .true.
      case default
         error stop 'shape must be small or prompt'
      end select
   end if
   call read_tokens(trim(train_path), train)
   call read_tokens(trim(a_path), a)
   call read_tokens(trim(b_path), b)
   if (same_bytes(train, a) .or. same_bytes(train, b) .or. same_bytes(a, b)) &
      error stop 'train and held-out streams must differ byte-for-byte'
   train_count = ((size(train, kind=int32) - 1_int32) / chunk) * chunk
   a_count = size(a, kind=int32) - 1_int32
   b_count = size(b, kind=int32) - 1_int32
   if (train_count < chunk .or. min(a_count,b_count) < 1_int32) error stop 'corpus too short'
   allocate(mask_a(a_count), mask_b(b_count), stat=allocation_status)
   if (allocation_status /= 0) error stop 'held-out mask allocation failed'
   mask_a = .false.
   mask_b = .false.

   config = fbc_config_t(vocabulary_size=256_int32, width=16_int32, heads=2_int32, ffn_width=32_int32, &
      local_window=8_int32, memory_scales=2_int32, memory_slots=8_int32, logical_horizon=8192_int64)
   if (prompt_shape) then
      config%width = 32_int32
      config%ffn_width = 64_int32
      config%local_window = 32_int32
      config%memory_scales = 4_int32
      config%memory_slots = 64_int32
   end if
   update_config = fbc_update_config_t()
   update_config%learning_rate = real(lr_micros, real64) / 1000000.0_real64
   call fbc_plan(config, plan)
   rule = [FBC_TRAINING_RULE_V6, FBC_TRAINING_RULE_V7]
   write(output_unit, '(A)') 'schema=vegai-context-slot-quality/v1'
   write(output_unit, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0)') 'train_predictions=', train_count, &
      ' heldout_a_predictions=', a_count, ' heldout_b_predictions=', b_count, ' chunk=', chunk, &
      ' parameters=', plan%parameter_count, ' state_array_bytes=', plan%state_bytes_real64
   write(output_unit, '(A,I0)') 'both_learning_rate_micros=', lr_micros
   write(output_unit, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0)') 'width=', config%width, ' heads=', config%heads, &
      ' ffn=', config%ffn_width, ' window=', config%local_window, ' scales=', config%memory_scales, &
      ' slots=', config%memory_slots
   write(output_unit, '(A)') 'seed,order,updates,v6_a_nll,v7_a_nll,delta_a,v6_b_nll,v7_b_nll,delta_b,' // &
      'v6_seconds,v7_seconds,time_ratio'
   do run = 1_int32, run_count
      call fbc_model_init(initial, config, int(seeds(run),int64), status)
      if (status /= FBC_OK) error stop 'initial model'
      if (modulo(run, 2_int32) == 1_int32) then
         order = [1_int32, 2_int32]
      else
         order = [2_int32, 1_int32]
      end if
      do pass = 1_int32, 2_int32
         arm = order(pass)
         model = initial
         model%training_rule = rule(arm)
         call fbc_state_init(state, model, status)
         if (status /= FBC_OK) error stop 'state'
         call fbc_workspace_init(workspace, model, status)
         if (status /= FBC_OK) error stop 'workspace'
         call fbc_gradient_init(gradient, model, status)
         if (status /= FBC_OK) error stop 'gradient'
         call fbc_optimizer_init(optimizer, config, status)
         if (status /= 0_int32) error stop 'optimizer'
         metrics = fbc_metrics_t()
         call system_clock(start_clock, clock_rate)
         call fbc_train_stream(model, optimizer, state, train, train_count, chunk, update_config, &
                               workspace, gradient, metrics, status)
         call system_clock(end_clock)
         if (status /= FBT_OK .or. .not. metrics%numeric_ok) error stop 'training'
         if (clock_rate <= 0_int64 .or. end_clock <= start_clock) error stop 'wall clock resolution'
         seconds(arm) = real(end_clock - start_clock, real64) / real(clock_rate, real64)
         if (metrics%tokens /= int(train_count, int64) .or. metrics%updates /= int(train_count / chunk, int64) .or. &
             metrics%parameter_updates /= plan%parameter_count * metrics%updates) error stop 'unequal training budget'
         call fbc_state_reset(state, model, status)
         if (status /= FBC_OK) error stop 'reset A'
         call fbc_evaluate_stream(model, state, a, a_count, mask_a, workspace, a_nll(arm), &
                                  recall, answer_count, status)
         if (status /= FBT_OK .or. .not. fbc_real64_is_finite(a_nll(arm))) error stop 'heldout A'
         call fbc_state_reset(state, model, status)
         if (status /= FBC_OK) error stop 'reset B'
         call fbc_evaluate_stream(model, state, b, b_count, mask_b, workspace, b_nll(arm), &
                                  recall, answer_count, status)
         if (status /= FBT_OK .or. .not. fbc_real64_is_finite(b_nll(arm))) error stop 'heldout B'
         call fbc_state_destroy(state)
         call fbc_workspace_destroy(workspace)
         call fbc_gradient_destroy(gradient)
         call fbc_optimizer_destroy(optimizer)
         call fbc_model_destroy(model)
      end do
      delta_a(run) = a_nll(2) - a_nll(1)
      delta_b(run) = b_nll(2) - b_nll(1)
      ratios(run) = seconds(2) / seconds(1)
      write(output_unit, '(I0,",",I0,I0,",",I0,9(",",ES24.16E3))') seeds(run), order, train_count / chunk, &
         a_nll(1), a_nll(2), delta_a(run), b_nll(1), b_nll(2), delta_b(run), seconds(1), seconds(2), ratios(run)
      call fbc_model_destroy(initial)
   end do
   call sort_small(delta_a)
   call sort_small(delta_b)
   call sort_small(ratios)
   write(output_unit, '(A,ES24.16E3,A,ES24.16E3,A,ES24.16E3)') 'median_delta_a=', delta_a(3), &
      ' median_delta_b=', delta_b(3), ' median_time_ratio=', ratios(3)
   write(output_unit, '(A)') 'NLL and timing apply only to these files, seeds and this host; not a production guarantee.'
contains
   subroutine get_arg(index, value, is_path)
      integer, intent(in) :: index
      character(len=*), intent(out) :: value
      logical, intent(in) :: is_path
      integer(int32) :: result
      call cli_get_argument(index, value, result, path=is_path)
      if (result /= CLI_OK) error stop 'invalid or overlong argument'
   end subroutine get_arg

   logical function same_bytes(first, second) result(same)
      integer(int32), intent(in) :: first(:), second(:)
      same = .false.
      if (size(first) == size(second)) same = all(first == second)
   end function same_bytes

   subroutine read_tokens(path, tokens)
      character(len=*), intent(in) :: path
      integer(int32), allocatable, intent(out) :: tokens(:)
      integer(int8), allocatable :: bytes(:)
      integer(int64) :: byte_count
      integer :: unit, result, index
      open(newunit=unit, file=path, status='old', action='read', access='stream', form='unformatted', iostat=result)
      if (result /= 0) error stop 'cannot open benchmark corpus'
      inquire(unit=unit, size=byte_count, iostat=result)
      if (result /= 0) error stop 'cannot inspect benchmark corpus size'
      if (byte_count < 2_int64 .or. byte_count > max_bytes) error stop 'benchmark corpus size invalid'
      allocate(bytes(int(byte_count)), tokens(int(byte_count)), stat=result)
      if (result /= 0) error stop 'benchmark allocation failed'
      read(unit, iostat=result) bytes
      if (result /= 0) error stop 'cannot read complete benchmark corpus'
      close(unit, iostat=result)
      if (result /= 0) error stop 'cannot close benchmark corpus'
      do index = 1, size(bytes)
         tokens(index) = iand(int(bytes(index), int32), 255_int32)
         if (tokens(index) == 0_int32) tokens(index) = 256_int32
      end do
   end subroutine read_tokens

   subroutine sort_small(values)
      real(real64), intent(inout) :: values(:)
      real(real64) :: key
      integer :: index, previous
      do index = 2, size(values)
         key = values(index)
         previous = index - 1
         do while (previous >= 1)
            if (values(previous) <= key) exit
            values(previous + 1) = values(previous)
            previous = previous - 1
         end do
         values(previous + 1) = key
      end do
   end subroutine sort_small
end program benchmark_slot_quality
