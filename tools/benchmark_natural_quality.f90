! SPDX-License-Identifier: MIT
! Same-rule learning-rate comparison: equal-budget training and two held-outs.
! Validation and scoring happen outside the timed training region. In addition
! to continuous stream NLL, score independent LF records from a cold state and
! report the first 32 within-record byte predictions separately. This is a
! byte-level metric, not a test of language understanding or dialogue quality.
program benchmark_natural_quality
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real64, output_unit
   use portable_cli, only: CLI_OK, cli_get_argument
   use fixed_budget_context
   use fixed_budget_backward
   use fixed_budget_optimizer
   use fixed_budget_training
   use fixed_budget_natural_eval
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
   integer(int32) :: order(2), candidate_lr_micros
   integer(int64) :: start_clock, end_clock, clock_rate, answer_count
   integer :: arg_count, allocation_status
   real(real64) :: a_nll(2), b_nll(2), seconds(2), recall
   real(real64) :: a_cold(2), b_cold(2), a_prefix(2), b_prefix(2)
   type(fbc_record_metrics_t) :: a_records, b_records
   real(real64) :: delta_a(run_count), delta_b(run_count), ratios(run_count)
   real(real64) :: delta_a_cold(run_count), delta_b_cold(run_count), delta_a_prefix(run_count), delta_b_prefix(run_count)
   integer(int32) :: better_a, better_b, better_a_cold, better_b_cold, better_a_prefix, better_b_prefix
   logical :: prompt_shape

   arg_count = command_argument_count()
   if (arg_count < 3 .or. arg_count > 7) error stop  &
      'expected train heldout-a heldout-b [chunk] [base-lr-micros] [small|prompt] [candidate-lr-micros]'
   call get_arg(1, train_path, .true.)
   call get_arg(2, a_path, .true.)
   call get_arg(3, b_path, .true.)
   if (trim(train_path) == trim(a_path) .or. trim(train_path) == trim(b_path) .or. &
       trim(a_path) == trim(b_path)) error stop 'train and held-out paths must differ'
   chunk = 512_int32
   lr_micros = 6000_int32
   prompt_shape = .true.
   candidate_lr_micros = 10000_int32
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
   if (arg_count >= 7) then
      call get_arg(7, argument, .false.)
      call fbc_parse_count(trim(argument), 1_int32, 999999_int32, candidate_lr_micros, status)
      if (status /= FBC_OK) error stop 'invalid candidate learning rate'
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

   write(output_unit, '(A)') 'schema=vegai-natural-lr-quality/v1'
   write(output_unit, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0)') 'train_predictions=', train_count, &
      ' heldout_a_predictions=', a_count, ' heldout_b_predictions=', b_count, ' chunk=', chunk, &
      ' parameters=', plan%parameter_count, ' state_array_bytes=', plan%state_bytes_real64
   write(output_unit, '(A,I0,A,I0)') 'base_learning_rate_micros=', lr_micros, &
      ' candidate_learning_rate_micros=', candidate_lr_micros
   write(output_unit, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0)') 'width=', config%width, ' heads=', config%heads, &
      ' ffn=', config%ffn_width, ' window=', config%local_window, ' scales=', config%memory_scales, &
      ' slots=', config%memory_slots
   write(output_unit, '(A)') 'seed,order,updates,base_a,cand_a,delta_a,base_b,cand_b,delta_b,' // &
      'base_a_cold,cand_a_cold,base_b_cold,cand_b_cold,base_a_prefix,cand_a_prefix,' // &
      'base_b_prefix,cand_b_prefix,base_seconds,cand_seconds,time_ratio'
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
         model%training_rule = FBC_TRAINING_RULE_V6
         call fbc_state_init(state, model, status)
         if (status /= FBC_OK) error stop 'state'
         call fbc_workspace_init(workspace, model, status)
         if (status /= FBC_OK) error stop 'workspace'
         call fbc_gradient_init(gradient, model, status)
         if (status /= FBC_OK) error stop 'gradient'
         call fbc_optimizer_init(optimizer, config, status)
         if (status /= 0_int32) error stop 'optimizer'
         metrics = fbc_metrics_t()
         if (arm == 1_int32) then
            update_config%learning_rate = real(lr_micros, real64) / 1000000.0_real64
         else
            update_config%learning_rate = real(candidate_lr_micros, real64) / 1000000.0_real64
         end if
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
         call fbc_evaluate_lf_records(model, state, workspace, a, 32_int32, a_records, status)
         if (status /= FBT_OK) error stop 'empty sentence evaluation or invalid heldout A'
         a_cold(arm) = a_records%mean_nll
         a_prefix(arm) = a_records%prefix_mean_nll
         call fbc_evaluate_lf_records(model, state, workspace, b, 32_int32, b_records, status)
         if (status /= FBT_OK) error stop 'empty sentence evaluation or invalid heldout B'
         b_cold(arm) = b_records%mean_nll
         b_prefix(arm) = b_records%prefix_mean_nll
         call fbc_state_destroy(state)
         call fbc_workspace_destroy(workspace)
         call fbc_gradient_destroy(gradient)
         call fbc_optimizer_destroy(optimizer)
         call fbc_model_destroy(model)
      end do
      delta_a(run) = a_nll(2) - a_nll(1)
      delta_b(run) = b_nll(2) - b_nll(1)
      delta_a_cold(run) = a_cold(2) - a_cold(1)
      delta_b_cold(run) = b_cold(2) - b_cold(1)
      delta_a_prefix(run) = a_prefix(2) - a_prefix(1)
      delta_b_prefix(run) = b_prefix(2) - b_prefix(1)
      ratios(run) = seconds(2) / seconds(1)
      write(output_unit, '(I0,",",I0,I0,",",I0,17(",",ES24.16E3))') seeds(run), order, train_count / chunk, &
         a_nll(1), a_nll(2), delta_a(run), b_nll(1), b_nll(2), delta_b(run), &
         a_cold(1), a_cold(2), b_cold(1), b_cold(2), a_prefix(1), a_prefix(2), b_prefix(1), b_prefix(2), &
         seconds(1), seconds(2), ratios(run)
      call fbc_model_destroy(initial)
   end do
   better_a = count(delta_a < 0.0_real64)
   better_b = count(delta_b < 0.0_real64)
   better_a_cold = count(delta_a_cold < 0.0_real64)
   better_b_cold = count(delta_b_cold < 0.0_real64)
   better_a_prefix = count(delta_a_prefix < 0.0_real64)
   better_b_prefix = count(delta_b_prefix < 0.0_real64)
   call sort_small(delta_a)
   call sort_small(delta_b)
   call sort_small(delta_a_cold)
   call sort_small(delta_b_cold)
   call sort_small(delta_a_prefix)
   call sort_small(delta_b_prefix)
   call sort_small(ratios)
   write(output_unit, '(A,ES24.16E3,A,ES24.16E3,A,ES24.16E3)') 'median_delta_a=', delta_a(3), &
      ' median_delta_b=', delta_b(3), ' median_time_ratio=', ratios(3)
   write(output_unit, '(A,ES24.16E3,A,ES24.16E3)') 'median_cold_a=', delta_a_cold(3), &
      ' median_cold_b=', delta_b_cold(3)
   write(output_unit, '(A,ES24.16E3,A,ES24.16E3)') 'median_prefix_a=', delta_a_prefix(3), &
      ' median_prefix_b=', delta_b_prefix(3)
   write(output_unit, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0)') 'improved_a=', better_a, ' improved_b=', better_b, &
      ' improved_cold_a=', better_a_cold, ' improved_cold_b=', better_b_cold, &
      ' improved_prefix_a=', better_a_prefix, ' improved_prefix_b=', better_b_prefix
   write(output_unit, '(A,I0,A,I0,A,I0,A,I0)') 'a_cold_count=', a_records%predictions, &
      ' b_cold_count=', b_records%predictions, ' a_prefix_count=', a_records%prefix_predictions, &
      ' b_prefix_count=', b_records%prefix_predictions
   write(output_unit, '(A,I0,A,I0,A,I0,A,I0)') 'a_records=', a_records%records, &
      ' a_scored_records=', a_records%scored_records, ' b_records=', b_records%records, &
      ' b_scored_records=', b_records%scored_records
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
      integer(int64) :: byte_count, path_count
      integer :: unit, result, index
      logical :: exists
      ! Reject common zero/unknown-size devices before OPEN. This does not
      ! detect same-size replacement or prevent a hostile path race.
      inquire(file=path, exist=exists, size=path_count, iostat=result)
      if (result /= 0) error stop 'cannot inspect benchmark corpus size'
      if (.not. exists) error stop 'cannot open benchmark corpus'
      if (path_count < 2_int64 .or. path_count > max_bytes) error stop 'benchmark corpus size invalid'
      open(newunit=unit, file=path, status='old', action='read', access='stream', form='unformatted', iostat=result)
      if (result /= 0) error stop 'cannot open benchmark corpus'
      inquire(unit=unit, size=byte_count, iostat=result)
      if (result /= 0) error stop 'cannot inspect benchmark corpus size'
      if (byte_count /= path_count) error stop 'benchmark corpus size changed before read'
      allocate(bytes(int(byte_count)), tokens(int(byte_count)), stat=result)
      if (result /= 0) error stop 'benchmark allocation failed'
      read(unit, iostat=result) bytes
      if (result /= 0) error stop 'cannot read complete benchmark corpus'
      inquire(unit=unit, size=path_count, iostat=result)
      if (result /= 0) error stop 'cannot inspect benchmark corpus after read'
      if (path_count /= byte_count) error stop 'benchmark corpus size changed during read'
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
end program benchmark_natural_quality
