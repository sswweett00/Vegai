! SPDX-License-Identifier: MIT
! Equal-budget, no-Python comparison of the historical and corrected context
! backward rules on caller-supplied, independently split byte files.
program benchmark_context_quality
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real64, error_unit, output_unit
   use portable_cli, only: CLI_OK, cli_get_argument
   use fixed_budget_context
   use fixed_budget_backward
   use fixed_budget_optimizer
   use fixed_budget_training
   implicit none (type, external)

   integer(int32), parameter :: run_count = 5_int32
   integer(int64), parameter :: seeds(run_count) = [21_int64, 42_int64, 73_int64, 101_int64, 137_int64]
   integer(int64), parameter :: max_bytes = 16777216_int64
   type(fbc_config_t) :: config
   type(fbc_update_config_t) :: update_config
   type(fbc_model_t) :: initial, model
   type(fbc_state_t) :: state
   type(fbc_workspace_t) :: workspace
   type(fbc_gradient_t) :: gradient
   type(fbc_optimizer_t) :: optimizer
   type(fbc_metrics_t) :: metrics
   type(fbc_plan_t) :: plan
   integer(int32), allocatable :: train(:), heldout(:)
   logical, allocatable :: answers(:)
   character(len=1024) :: train_path, heldout_path, argument
   integer(int32) :: chunk, candidate_lr_micros, train_count, eval_count, status, run, pass, arm, ios
   integer(int32) :: order(2), rule(2)
   integer(int64) :: start_clock, end_clock, clock_rate, answer_count
   real(real64) :: nll(2), recall, seconds(2), deltas(run_count), ratios(run_count), initial_nll
   logical :: prompt_shape

   if (command_argument_count() < 2 .or. command_argument_count() > 5) then
      write(error_unit, '(A)') 'usage: vegai_context_quality_ab TRAIN HELDOUT [CHUNK [V6_LR_MICROS [small|prompt]]]'
      error stop 2
   end if
   call cli_get_argument(1, train_path, status, path=.true.)
   if (status /= CLI_OK) error stop 'invalid train path'
   if (len_trim(train_path) == 0) error stop 'invalid train path'
   call cli_get_argument(2, heldout_path, status, path=.true.)
   if (status /= CLI_OK) error stop 'invalid held-out path'
   if (len_trim(heldout_path) == 0) error stop 'invalid held-out path'
   if (trim(train_path) == trim(heldout_path)) error stop 'train and held-out paths must differ'
   chunk = 32_int32
   candidate_lr_micros = 3000_int32
   if (command_argument_count() >= 3) then
      call get_command_argument(3, argument, status=ios)
      if (ios /= 0) error stop 'invalid chunk'
      call fbc_parse_count(trim(argument), 1_int32, FBC_MAX_CHUNK_TOKENS, chunk, status)
      if (status /= FBC_OK) error stop 'chunk must be an integer in 1..8192'
   end if
   if (command_argument_count() >= 4) then
      call get_command_argument(4, argument, status=ios)
      if (ios /= 0) error stop 'invalid v6 learning rate'
      call fbc_parse_count(trim(argument), 1_int32, 999999_int32, candidate_lr_micros, status)
      if (status /= FBC_OK) error stop 'v6 LR in millionths must be an integer in 1..999999'
   end if
   prompt_shape = .false.
   if (command_argument_count() == 5) then
      call get_command_argument(5, argument, status=ios)
      if (ios /= 0) error stop 'invalid shape'
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
   call read_tokens(trim(heldout_path), heldout)
   if (size(train) == size(heldout)) then
      if (all(train == heldout)) error stop 'train and held-out files are byte-identical'
   end if
   train_count = ((size(train, kind=int32) - 1_int32) / chunk) * chunk
   eval_count = size(heldout, kind=int32) - 1_int32
   if (train_count < chunk .or. eval_count < 1_int32) error stop 'corpus too short for the selected chunk'
   allocate(answers(eval_count))
   answers = .false.

   ! This is the same initial model in both arms. By default only the versioned
   ! backward rule differs; the optional v6 learning rate is reported explicitly.
   ! The held-out bytes and initial model are never used for updates.
   config = fbc_config_t(vocabulary_size=256_int32, width=16_int32, heads=2_int32, ffn_width=32_int32, &
                         local_window=8_int32, memory_scales=2_int32, memory_slots=8_int32, &
                         logical_horizon=8192_int64)
   if (prompt_shape) then
      config%width = 32_int32
      config%ffn_width = 64_int32
      config%local_window = 32_int32
      config%memory_scales = 4_int32
      config%memory_slots = 64_int32
   end if
   update_config = fbc_update_config_t()
   update_config%learning_rate = 3.0e-3_real64
   call fbc_plan(config, plan)
   rule = [FBC_TRAINING_RULE_V5, FBC_TRAINING_RULE_V6]
   write(output_unit, '(A)') 'schema=vegai-context-byte-quality-ab/v1'
   write(output_unit, '(A,I0,A,I0,A,I0,A,I0,A,I0)') 'train_predictions=', train_count, &
      ' heldout_predictions=', eval_count, ' chunk=', chunk, ' parameters=', plan%parameter_count, &
      ' state_array_bytes=', plan%state_bytes_real64
   write(output_unit, '(A,I0,A,I0)') 'v5_learning_rate_micros=', 3000_int32, &
      ' v6_learning_rate_micros=', candidate_lr_micros
   write(output_unit, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0)') 'width=', config%width, ' heads=', config%heads, &
      ' ffn=', config%ffn_width, ' window=', config%local_window, ' scales=', config%memory_scales, &
      ' slots=', config%memory_slots
   write(output_unit, '(A)') 'seed,order,updates,initial_nll,v5_nll,v6_nll,v6_minus_v5,v5_seconds,v6_seconds,time_ratio'

   do run = 1_int32, run_count
      call fbc_model_init(initial, config, seeds(run), status)
      if (status /= FBC_OK) error stop 'initial model'
      call fbc_state_init(state, initial, status)
      if (status /= FBC_OK) error stop 'initial state'
      call fbc_workspace_init(workspace, initial, status)
      if (status /= FBC_OK) error stop 'initial workspace'
      call fbc_evaluate_stream(initial, state, heldout, eval_count, answers, workspace, initial_nll, &
                               recall, answer_count, status)
      if (status /= FBT_OK) error stop 'initial held-out evaluation'
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      if (modulo(run, 2_int32) == 1_int32) then
         order = [1_int32, 2_int32]
      else
         order = [2_int32, 1_int32]
      end if
      do pass = 1_int32, 2_int32
         arm = order(pass)
         model = initial
         model%training_rule = rule(arm)
         if (arm == 1_int32) then
            update_config%learning_rate = 3.0e-3_real64
         else
            update_config%learning_rate = real(candidate_lr_micros, real64) / 1000000.0_real64
         end if
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
         if (status /= FBT_OK) error stop 'training'
         if (clock_rate <= 0_int64 .or. end_clock <= start_clock) error stop 'insufficient wall clock resolution'
         seconds(arm) = real(end_clock - start_clock, real64) / real(clock_rate, real64)
         if (metrics%tokens /= int(train_count, int64) .or. metrics%updates /= int(train_count / chunk, int64) .or. &
             metrics%parameter_updates /= plan%parameter_count * metrics%updates) error stop 'unequal training budget'
         call fbc_state_reset(state, model, status)
         if (status /= FBC_OK) error stop 'reset before evaluation'
         call fbc_evaluate_stream(model, state, heldout, eval_count, answers, workspace, nll(arm), &
                                  recall, answer_count, status)
         if (status /= FBT_OK) error stop 'held-out evaluation'
         call fbc_state_destroy(state)
         call fbc_workspace_destroy(workspace)
         call fbc_gradient_destroy(gradient)
         call fbc_optimizer_destroy(optimizer)
         call fbc_model_destroy(model)
      end do
      deltas(run) = nll(2) - nll(1)
      ratios(run) = seconds(2) / seconds(1)
      write(output_unit, '(I0,",",I0,I0,",",I0,7(",",ES24.16E3))') seeds(run), order, train_count / chunk, &
         initial_nll, nll(1), nll(2), deltas(run), seconds(1), seconds(2), ratios(run)
      call fbc_model_destroy(initial)
   end do
   call sort_small(deltas)
   call sort_small(ratios)
   write(output_unit, '(A,ES24.16E3,A,ES24.16E3)') 'median_nll_delta=', deltas(3), ' median_time_ratio=', ratios(3)
   write(output_unit, '(A)') 'NLL and timing apply only to these files, seeds and this host; not a production guarantee.'

contains

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
         ! Legacy prompt checkpoints map bytes 1..255 directly; byte 0 was
         ! not trainable. Reject it here so both arms receive the same tokens.
         tokens(index) = iand(int(bytes(index), int32), 255_int32)
         if (tokens(index) == 0_int32) error stop 'NUL is unsupported in the v5 comparison arm'
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
end program benchmark_context_quality
