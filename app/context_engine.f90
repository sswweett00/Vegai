! SPDX-License-Identifier: MIT
!==============================================================================
!  context_engine.f90  --  CLI for the fixed-budget context engine
!
!  Commands
!    plan      --window H ...                structural resource plan (JSON)
!    inspect   CHECKPOINT                    header metadata only (JSON)
!    train     ... --checkpoint FILE         train on a synthetic stream
!    evaluate  CHECKPOINT ...                held-out NLL / ppl / recall
!    bench     ...                           per-token latency + throughput
!    sweep     ...                           the invariance table across horizons
!
!  Option semantics (deliberately separated)
!    --window              logical context horizon, int64, tokens.  It only
!                          selects the recurrent decay constants.  It is used
!                          in no allocation and no loop bound.
!    --local-window        physical ring-buffer length for exact causal
!                          attention.  This is the only size knob for the cache.
!    --memory-scales       fixed number of decay scales.
!    --memory-slots        fixed number of information slots.
!    --train-chunk-tokens  predictions per optimiser update; bounds the work
!                          between updates and nothing else.
!
!  Every numeric option is parsed strictly: an empty, signed, non-numeric,
!  zero, negative or overflowing value is rejected with a message naming the
!  option, and the process exits non-zero.
!==============================================================================
program context_engine
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64, output_unit, error_unit
   use fixed_budget_context
   use fixed_budget_backward
   use fixed_budget_optimizer
   use fixed_budget_training
   implicit none (type, external)

   integer(int32), parameter :: EXIT_USAGE = 2_int32

   character(len=32) :: command
   type(fbc_config_t) :: config
   type(fbc_update_config_t) :: update_config
   integer(int64) :: seed, token_budget
   integer(int32) :: task, distance, chunk_tokens, status
   character(len=512) :: checkpoint_path
   logical :: have_checkpoint

   call read_command(command)
   if (len_trim(command) == 0) call usage_and_stop('missing command')
   if (trim(command) == '--help' .or. trim(command) == '-h') call print_usage_and_stop()

   config%vocabulary_size = 32_int32
   config%width = 32_int32
   config%heads = 2_int32
   config%ffn_width = 64_int32
   config%local_window = 32_int32
   config%memory_scales = 4_int32
   config%memory_slots = 64_int32
   config%logical_horizon = 32_int64
   update_config%learning_rate = 3.0e-3_real64
   seed = 20261028_int64
   token_budget = 4096_int64
   task = FBC_TASK_NEEDLE
   distance = 64_int32
   chunk_tokens = 256_int32
   checkpoint_path = ''
   have_checkpoint = .false.

   call parse_options(config, update_config, seed, token_budget, task, distance, chunk_tokens, &
                      checkpoint_path, have_checkpoint)

   select case (trim(command))
   case ('plan')
      call command_plan(config)
   case ('inspect')
      if (.not. have_checkpoint) call usage_and_stop('inspect needs a checkpoint path')
      call command_inspect(checkpoint_path)
   case ('train')
      if (.not. have_checkpoint) call usage_and_stop('train needs --checkpoint FILE')
      call command_train(config, update_config, seed, token_budget, task, distance, chunk_tokens, &
                         checkpoint_path)
   case ('evaluate')
      if (.not. have_checkpoint) call usage_and_stop('evaluate needs a checkpoint path')
      call command_evaluate(token_budget, task, distance, checkpoint_path)
   case ('bench')
      call command_bench(config, update_config, seed, token_budget, task, distance, chunk_tokens)
   case ('sweep')
      call command_sweep(update_config, seed, token_budget, task, distance, chunk_tokens)
   case default
      call usage_and_stop('unknown command '//trim(command))
   end select

contains

   subroutine read_command(name)
      character(len=*), intent(out) :: name

      name = ''
      if (command_argument_count() >= 1) call get_command_argument(1, name)
   end subroutine read_command

   subroutine print_usage_and_stop()
      write(output_unit, '(A)') 'usage: context_engine <plan|inspect|train|evaluate|bench|sweep> [options]'
      write(output_unit, '(A)') '  --window N              logical horizon in tokens (int64, >= 1)'
      write(output_unit, '(A)') '  --local-window N        physical ring buffer, 1..4096 (default 32)'
      write(output_unit, '(A)') '  --memory-scales N       fixed decay scales, 1..16 (default 4)'
      write(output_unit, '(A)') '  --memory-slots N        fixed information slots, 1..1024 (default 64)'
      write(output_unit, '(A)') '  --train-chunk-tokens N  predictions per update, 1..8192 (default 256)'
      write(output_unit, '(A)') '  --width N --heads N --ffn N --vocab N'
      write(output_unit, '(A)') '  --seed N --lr X --tokens N --task markov|needle --distance N'
      write(output_unit, '(A)') '  --checkpoint FILE'
      stop 0
   end subroutine print_usage_and_stop

   subroutine usage_and_stop(message)
      character(len=*), intent(in) :: message

      write(error_unit, '(A,A)') 'context-engine usage error: ', trim(message)
      stop EXIT_USAGE
   end subroutine usage_and_stop

   !> Parses trailing `--key value` pairs.  Unknown keys and malformed values
   !> are usage errors; nothing is silently ignored.
   subroutine parse_options(config, update_config, seed, token_budget, task, distance, chunk_tokens, &
                            checkpoint_path, have_checkpoint)
      type(fbc_config_t), intent(inout) :: config
      type(fbc_update_config_t), intent(inout) :: update_config
      integer(int64), intent(inout) :: seed, token_budget
      integer(int32), intent(inout) :: task, distance, chunk_tokens
      character(len=*), intent(inout) :: checkpoint_path
      logical, intent(inout) :: have_checkpoint
      integer(int32) :: index, parse_status

      index = 2_int32
      do while (index <= command_argument_count())
         call parse_pair(index, config, update_config, seed, token_budget, task, distance, chunk_tokens, &
                         checkpoint_path, have_checkpoint, parse_status)
         if (parse_status /= 0_int32) call usage_and_stop('malformed or unknown option')
         index = index + 2_int32
      end do
      if (config%logical_horizon < 1_int64) call usage_and_stop('--window must be >= 1')
      if (chunk_tokens < 1_int32 .or. chunk_tokens > FBC_MAX_CHUNK_TOKENS) &
         call usage_and_stop('--train-chunk-tokens out of range')
   end subroutine parse_options

   subroutine parse_pair(index, config, update_config, seed, token_budget, task, distance, chunk_tokens, &
                         checkpoint_path, have_checkpoint, parse_status)
      integer(int32), intent(in) :: index
      type(fbc_config_t), intent(inout) :: config
      type(fbc_update_config_t), intent(inout) :: update_config
      integer(int64), intent(inout) :: seed, token_budget
      integer(int32), intent(inout) :: task, distance, chunk_tokens
      character(len=*), intent(inout) :: checkpoint_path
      logical, intent(inout) :: have_checkpoint
      integer(int32), intent(out) :: parse_status
      character(len=256) :: key, value

      parse_status = 0_int32
      call get_command_argument(index, key)
      if (index + 1_int32 > command_argument_count()) then
         write(error_unit, '(A,A)') 'option without a value: ', trim(key)
         parse_status = 1_int32
         return
      end if
      call get_command_argument(index + 1_int32, value)
      select case (trim(key))
      case ('--window')
         call fbc_parse_horizon(value, config%logical_horizon, parse_status)
      case ('--local-window')
         call fbc_parse_count(value, 1_int32, FBC_MAX_LOCAL_WINDOW, config%local_window, parse_status)
      case ('--memory-scales')
         call fbc_parse_count(value, 1_int32, FBC_MAX_MEMORY_SCALES, config%memory_scales, parse_status)
      case ('--memory-slots')
         call fbc_parse_count(value, 1_int32, FBC_MAX_MEMORY_SLOTS, config%memory_slots, parse_status)
      case ('--train-chunk-tokens')
         call fbc_parse_count(value, 1_int32, FBC_MAX_CHUNK_TOKENS, chunk_tokens, parse_status)
      case ('--width')
         call fbc_parse_count(value, 2_int32, 1024_int32, config%width, parse_status)
      case ('--heads')
         call fbc_parse_count(value, 1_int32, 64_int32, config%heads, parse_status)
      case ('--ffn')
         call fbc_parse_count(value, 2_int32, 8192_int32, config%ffn_width, parse_status)
      case ('--vocab')
         call fbc_parse_count(value, 16_int32, 65536_int32, config%vocabulary_size, parse_status)
      case ('--distance')
         call fbc_parse_count(value, 1_int32, 1000000_int32, distance, parse_status)
      case ('--tokens')
         call fbc_parse_horizon(value, token_budget, parse_status)
      case ('--seed')
         read(value, *, iostat=parse_status) seed
      case ('--lr')
         read(value, *, iostat=parse_status) update_config%learning_rate
         if (parse_status == 0 .and. (update_config%learning_rate <= 0.0_real64 .or. &
                                      update_config%learning_rate >= 1.0_real64)) parse_status = 1_int32
      case ('--task')
         call fbc_parse_task(value, task, parse_status)
      case ('--checkpoint')
         checkpoint_path = trim(value)
         have_checkpoint = len_trim(checkpoint_path) > 0
         if (.not. have_checkpoint) parse_status = 1_int32
      case default
         write(error_unit, '(A,A)') 'unknown option: ', trim(key)
         parse_status = 1_int32
      end select
      if (parse_status /= 0_int32) then
         write(error_unit, '(A,A,A,A)') 'invalid value for ', trim(key), ': ', trim(value)
      end if
   end subroutine parse_pair

   !----------------------------------------------------------------------------
   !  plan / inspect
   !----------------------------------------------------------------------------

   subroutine command_plan(config)
      type(fbc_config_t), intent(in) :: config
      type(fbc_plan_t) :: plan

      if (.not. fbc_config_is_valid(config)) call usage_and_stop('configuration is not valid')
      call fbc_plan(config, plan)
      call print_plan(plan)
   end subroutine command_plan

   subroutine print_plan(plan)
      type(fbc_plan_t), intent(in) :: plan
      character(len=:), allocatable :: line

      write(output_unit, '(A)') '== fixed-budget context plan =='
      write(output_unit, '(A,I0)') 'logical horizon (tokens)        : ', plan%logical_horizon
      write(output_unit, '(A,F0.1)') 'effective (achievable) horizon : ', plan%effective_horizon
      write(output_unit, '(A,L1)') 'horizon clamped to arithmetic  : ', plan%horizon_clamped
      write(output_unit, '(A,I0)') 'local window (physical)        : ', plan%local_window
      write(output_unit, '(A,I0)') 'memory scales                  : ', plan%memory_scales
      write(output_unit, '(A,I0)') 'memory slots                   : ', plan%memory_slots
      write(output_unit, '(A,I0)') 'parameters                     : ', plan%parameter_count
      write(output_unit, '(A,I0)') 'parameter bytes (real64)       : ', plan%parameter_bytes_real64
      write(output_unit, '(A,I0)') 'stream state bytes (real64)    : ', plan%state_bytes_real64
      write(output_unit, '(A,I0)') 'local cache elements           : ', plan%local_cache_elements
      write(output_unit, '(A,I0)') 'retention state elements       : ', plan%retention_state_elements
      write(output_unit, '(A,I0)') 'slot state elements            : ', plan%slot_state_elements
      write(output_unit, '(A,I0)') 'forward ops / token            : ', plan%forward_flops_per_token
      write(output_unit, '(A,I0)') 'backward ops / token           : ', plan%backward_flops_per_token
      write(output_unit, '(A,I0)') 'local score ops / token        : ', plan%local_score_operations_per_token
      line = '{"schema":"vegai-fixed-budget-plan/v1","logical_horizon":'//i64_text(plan%logical_horizon)// &
             ',"effective_horizon":'//r64_text(plan%effective_horizon)// &
             ',"horizon_clamped":'//json_bool(plan%horizon_clamped)// &
             ',"local_window":'//i64_text(int(plan%local_window, int64))// &
             ',"memory_scales":'//i64_text(int(plan%memory_scales, int64))// &
             ',"memory_slots":'//i64_text(int(plan%memory_slots, int64))// &
             ',"parameters":'//i64_text(plan%parameter_count)// &
             ',"state_bytes":'//i64_text(plan%state_bytes_real64)// &
             ',"local_cache_elements":'//i64_text(plan%local_cache_elements)// &
             ',"retention_state_elements":'//i64_text(plan%retention_state_elements)// &
             ',"slot_state_elements":'//i64_text(plan%slot_state_elements)// &
             ',"forward_ops_per_token":'//i64_text(plan%forward_flops_per_token)// &
             ',"backward_ops_per_token":'//i64_text(plan%backward_flops_per_token)//'}'
      write(output_unit, '(A)') 'CONTEXT_PLAN_JSON: '//trim(line)
   end subroutine print_plan

   subroutine command_inspect(path)
      character(len=*), intent(in) :: path
      type(fbc_checkpoint_meta_t) :: meta
      integer(int32) :: status

      call fbc_checkpoint_meta_read(path, meta, status)
      if (status == FBT_ERR_LEGACY) then
         write(error_unit, '(A)') 'this is a legacy VGNR checkpoint; use vegai_neural_pretrain for it'
         stop 1_int32
      end if
      if (status /= FBT_OK) call usage_and_stop('cannot read checkpoint header')
      write(output_unit, '(A)') '== checkpoint =='
      write(output_unit, '(A,I0)') 'logical horizon        : ', meta%logical_horizon
      write(output_unit, '(A,F0.1)') 'effective horizon      : ', meta%effective_horizon
      write(output_unit, '(A,L1)') 'horizon clamped        : ', meta%horizon_clamped
      write(output_unit, '(A,I0)') 'local window           : ', meta%local_window
      write(output_unit, '(A,I0)') 'memory scales          : ', meta%memory_scales
      write(output_unit, '(A,I0)') 'memory slots           : ', meta%memory_slots
      write(output_unit, '(A,I0)') 'width/heads/ffn/vocab  : ', meta%width, '/', meta%heads, '/', &
         meta%ffn_width, '/', meta%vocabulary_size
      write(output_unit, '(A,I0,A,I0)') 'updates/tokens         : ', meta%updates, '/', meta%tokens
   end subroutine command_inspect

   !----------------------------------------------------------------------------
   !  train / evaluate
   !----------------------------------------------------------------------------

   subroutine command_train(config, update_config, seed, token_budget, task, distance, chunk_tokens, path)
      type(fbc_config_t), intent(in) :: config
      type(fbc_update_config_t), intent(in) :: update_config
      integer(int64), intent(in) :: seed, token_budget
      integer(int32), intent(in) :: task, distance, chunk_tokens
      character(len=*), intent(in) :: path
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: gradient
      type(fbc_optimizer_t) :: optimizer
      type(fbc_metrics_t) :: metrics
      integer(int32) :: status, token_count
      integer(int32), allocatable :: tokens(:)
      logical, allocatable :: answer_mask(:)
      real(real64) :: started, elapsed

      if (.not. fbc_config_is_valid(config)) call usage_and_stop('configuration is not valid')
      token_count = int(min(token_budget, 20000000_int64), int32)
      allocate(tokens(token_count + 1_int32), answer_mask(token_count))
      call fbc_generate_tokens(task, token_count, config%vocabulary_size, distance, seed, tokens(1_int32:token_count), &
                               answer_mask, status)
      if (status /= FBC_OK) call usage_and_stop('token generation failed for this vocabulary/distance')
      tokens(token_count + 1_int32) = 1_int32
      call fbc_model_init(model, config, seed, status)
      if (status /= FBC_OK) call usage_and_stop('model initialisation failed')
      call fbc_state_init(state, model, status)
      if (status /= FBC_OK) call usage_and_stop('stream state allocation failed')
      call fbc_workspace_init(workspace, model, status)
      if (status /= FBC_OK) call usage_and_stop('workspace allocation failed')
      call fbc_gradient_init(gradient, model, status)
      if (status /= FBC_OK) call usage_and_stop('gradient allocation failed')
      call fbc_optimizer_init(optimizer, config, status)
      if (status /= 0_int32) call usage_and_stop('optimizer allocation failed')

      call wall_clock(started)
      call fbc_train_stream(model, optimizer, state, tokens, token_count, chunk_tokens, update_config, &
                            workspace, gradient, metrics, status)
      call wall_clock(elapsed)
      elapsed = elapsed - started
      if (status /= FBT_OK) then
         write(error_unit, '(A,I0)') 'training failed, status ', status
         stop 1_int32
      end if
      call fbc_checkpoint_save(path, model, optimizer, update_config, metrics, status)
      if (status /= FBT_OK) then
         write(error_unit, '(A,I0)') 'checkpoint write failed, status ', status
         stop 1_int32
      end if
      write(output_unit, '(A,I0,A,I0,A,F0.2,A)') 'trained ', metrics%tokens, ' tokens in ', metrics%updates, &
         ' updates, ', elapsed, ' s'
      write(output_unit, '(A,F0.4)') 'mean training nll   : ', metrics%loss_sum / real(max(1_int64, metrics%tokens), real64)
      write(output_unit, '(A,F0.1)') 'tokens/s            : ', &
         real(metrics%tokens, real64) / max(elapsed, 1.0e-9_real64)
      write(output_unit, '(A,F0.6)') 'last gradient norm  : ', metrics%last_gradient_norm
      write(output_unit, '(A,A)') 'checkpoint          : ', trim(path)

      call fbc_model_destroy(model)
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_gradient_destroy(gradient)
      call fbc_optimizer_destroy(optimizer)
      deallocate(tokens, answer_mask)
   end subroutine command_train

   subroutine command_evaluate(token_budget, task, distance, path)
      integer(int64), intent(in) :: token_budget
      integer(int32), intent(in) :: task, distance
      character(len=*), intent(in) :: path
      type(fbc_model_t) :: model
      type(fbc_optimizer_t) :: optimizer
      type(fbc_update_config_t) :: loaded_config
      type(fbc_metrics_t) :: counters
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      integer(int32) :: status, token_count
      integer(int32), allocatable :: tokens(:)
      logical, allocatable :: answer_mask(:)
      real(real64) :: mean_nll, accuracy
      integer(int64) :: answer_count

      call fbc_checkpoint_load(path, model, optimizer, loaded_config, counters, status)
      if (status == FBT_ERR_LEGACY) then
         write(error_unit, '(A)') 'legacy VGNR checkpoint: load it with vegai_neural_pretrain'
         stop 1_int32
      end if
      if (status /= FBT_OK) call usage_and_stop('cannot load checkpoint')
      token_count = int(min(token_budget, 20000000_int64), int32)
      allocate(tokens(token_count + 1_int32), answer_mask(token_count))
      call fbc_generate_tokens(task, token_count, model%config%vocabulary_size, distance, 7_int64, &
                               tokens(1_int32:token_count), answer_mask, status)
      if (status /= FBC_OK) call usage_and_stop('token generation failed')
      tokens(token_count + 1_int32) = 1_int32
      call fbc_state_init(state, model, status)
      call fbc_workspace_init(workspace, model, status)
      call fbc_evaluate_stream(model, state, tokens, token_count, answer_mask, workspace, mean_nll, &
                               accuracy, answer_count, status)
      if (status /= FBC_OK) then
         write(error_unit, '(A,I0)') 'evaluation failed, status ', status
         stop 1_int32
      end if
      write(output_unit, '(A,F0.6)') 'mean nll            : ', mean_nll
      write(output_unit, '(A,F0.2)') 'perplexity          : ', exp(min(mean_nll, 50.0_real64))
      write(output_unit, '(A,F0.4,A,I0,A)') 'remote recall       : ', accuracy, ' over ', answer_count, ' queries'
      call fbc_model_destroy(model)
      call fbc_optimizer_destroy(optimizer)
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      deallocate(tokens, answer_mask)
   end subroutine command_evaluate

   !----------------------------------------------------------------------------
   !  bench / sweep
   !----------------------------------------------------------------------------

   subroutine command_bench(config, update_config, seed, token_budget, task, distance, chunk_tokens)
      type(fbc_config_t), intent(in) :: config
      type(fbc_update_config_t), intent(in) :: update_config
      integer(int64), intent(in) :: seed, token_budget
      integer(int32), intent(in) :: task, distance, chunk_tokens
      call run_benchmark(config, update_config, seed, token_budget, task, distance, chunk_tokens, .true.)
   end subroutine command_bench

   !> One measurement of one configuration.  `verbose` prints the human table;
   !> the machine line is always printed so a sweep can be parsed.
   subroutine run_benchmark(config, update_config, seed, token_budget, task, distance, chunk_tokens, verbose)
      type(fbc_config_t), intent(in) :: config
      type(fbc_update_config_t), intent(in) :: update_config
      integer(int64), intent(in) :: seed, token_budget
      integer(int32), intent(in) :: task, distance, chunk_tokens
      logical, intent(in) :: verbose
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: gradient
      type(fbc_optimizer_t) :: optimizer
      type(fbc_metrics_t) :: metrics
      type(fbc_plan_t) :: plan
      integer(int32) :: status, token_count, index
      integer(int32), allocatable :: tokens(:)
      logical, allocatable :: answer_mask(:)
      real(real64), allocatable :: hidden(:), logits(:), latencies(:)
      integer(int64) :: start_clock, end_clock, rate, latency_start
      real(real64) :: latency_total, inference_throughput, training_seconds, p50, p95
      integer(int64) :: training_tokens

      if (.not. fbc_config_is_valid(config)) call usage_and_stop('configuration is not valid')
      call fbc_plan(config, plan)
      token_count = int(min(max(token_budget, 64_int64), 1000000_int64), int32)
      allocate(tokens(token_count + 1_int32), answer_mask(token_count))
      call fbc_generate_tokens(task, token_count, config%vocabulary_size, distance, seed, &
                               tokens(1_int32:token_count), answer_mask, status)
      if (status /= FBC_OK) call usage_and_stop('token generation failed')
      tokens(token_count + 1_int32) = 1_int32
      allocate(hidden(config%width), logits(config%vocabulary_size), latencies(token_count))
      call fbc_model_init(model, config, seed, status)
      if (status /= FBC_OK) call usage_and_stop('model initialisation failed')
      call fbc_state_init(state, model, status)
      if (status /= FBC_OK) call usage_and_stop('stream state allocation failed')
      call fbc_workspace_init(workspace, model, status)
      if (status /= FBC_OK) call usage_and_stop('workspace allocation failed')
      call fbc_gradient_init(gradient, model, status)
      if (status /= FBC_OK) call usage_and_stop('gradient allocation failed')
      call fbc_optimizer_init(optimizer, config, status)
      if (status /= 0_int32) call usage_and_stop('optimizer allocation failed')

      ! Inference: one forward step per token, timed individually for p50/p95.
      call system_clock(start_clock, rate)
      latency_total = 0.0_real64
      do index = 1_int32, token_count
         call system_clock(latency_start, rate)
         call fbc_step(model, state, workspace, tokens(index), hidden, logits, status)
         call system_clock(end_clock, rate)
         latencies(index) = real(end_clock - latency_start, real64) * 1.0e6_real64 / real(rate, real64)
         latency_total = latency_total + latencies(index)
         if (status /= FBC_OK) exit
      end do
      call system_clock(end_clock, rate)
      inference_throughput = real(token_count, real64) * real(rate, real64) / &
                             max(1.0_real64, real(end_clock - start_clock, real64))
      ! Percentiles come from the first 1000 token timings: enough for a stable
      ! p50/p95, and the selection sort below stays cheap.
      call percentiles(latencies(1_int32:max(1_int32, min(token_count, 1000_int32))), p50, p95)

      ! Training: same token budget, same chunk size, so the two numbers are
      ! comparable and the only difference is the backward pass.
      call fbc_state_reset(state, model, status)
      call system_clock(start_clock, rate)
      call fbc_train_stream(model, optimizer, state, tokens, token_count, chunk_tokens, update_config, &
                            workspace, gradient, metrics, status)
      call system_clock(end_clock, rate)
      training_seconds = real(end_clock - start_clock, real64) / real(rate, real64)
      training_tokens = max(1_int64, metrics%tokens)

      if (verbose) then
         write(output_unit, '(A)') '== fixed-budget context benchmark =='
         write(output_unit, '(A,I0)') 'logical horizon        : ', plan%logical_horizon
         write(output_unit, '(A,F0.1)') 'effective horizon      : ', plan%effective_horizon
         write(output_unit, '(A,I0)') 'local window           : ', plan%local_window
         write(output_unit, '(A,I0)') 'parameters             : ', plan%parameter_count
         write(output_unit, '(A,I0)') 'stream state bytes     : ', plan%state_bytes_real64
         write(output_unit, '(A,I0)') 'forward ops / token    : ', plan%forward_flops_per_token
         write(output_unit, '(A,I0)') 'local score ops / token: ', plan%local_score_operations_per_token
         write(output_unit, '(A,I0)') 'tokens measured        : ', int(token_count, int64)
         write(output_unit, '(A,F0.2)') 'inference tokens/s     : ', inference_throughput
         write(output_unit, '(A,F0.2)') 'inference p50 us       : ', p50
         write(output_unit, '(A,F0.2)') 'inference p95 us       : ', p95
         write(output_unit, '(A,F0.2)') 'training tokens/s      : ', real(training_tokens, real64) / &
            max(training_seconds, 1.0e-9_real64)
         write(output_unit, '(A,F0.4)') 'training mean nll      : ', metrics%loss_sum / real(training_tokens, real64)
      end if
      write(output_unit, '(A)') 'CONTEXT_BENCH_JSON: {"horizon":'//i64_text(plan%logical_horizon)// &
         ',"effective_horizon":'//r64_text(plan%effective_horizon)// &
         ',"parameters":'//i64_text(plan%parameter_count)// &
         ',"state_bytes":'//i64_text(plan%state_bytes_real64)// &
         ',"forward_ops_per_token":'//i64_text(plan%forward_flops_per_token)// &
         ',"p50_us":'//r64_text(p50)//',"p95_us":'//r64_text(p95)// &
         ',"inference_tokens_per_s":'//r64_text(inference_throughput)// &
         ',"training_tokens_per_s":'//r64_text(real(training_tokens, real64) / max(training_seconds, 1.0e-9_real64))// &
         ',"training_nll":'//r64_text(metrics%loss_sum / real(training_tokens, real64))//'}'

      call fbc_model_destroy(model)
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_gradient_destroy(gradient)
      call fbc_optimizer_destroy(optimizer)
      deallocate(tokens, answer_mask, hidden, logits, latencies)
      if (status /= FBT_OK) then
         write(error_unit, '(A,I0)') 'benchmark step failed, status ', status
         stop 1_int32
      end if
   end subroutine run_benchmark

   !> The evidence table: identical physical shapes, four horizons.
   subroutine command_sweep(update_config, seed, token_budget, task, distance, chunk_tokens)
      type(fbc_update_config_t), intent(in) :: update_config
      integer(int64), intent(in) :: seed, token_budget
      integer(int32), intent(in) :: task, distance, chunk_tokens
      integer(int64), parameter :: horizons(4) = [32_int64, 8192_int64, 1000000_int64, 1000000000000_int64]
      type(fbc_config_t) :: config
      integer(int32) :: index

      do index = 1_int32, size(horizons, kind=int32)
         config%vocabulary_size = 32_int32
         config%width = 32_int32
         config%heads = 2_int32
         config%ffn_width = 64_int32
         config%local_window = 32_int32
         config%memory_scales = 4_int32
         config%memory_slots = 64_int32
         config%logical_horizon = horizons(index)
         call run_benchmark(config, update_config, seed, token_budget, task, distance, chunk_tokens, .true.)
      end do
   end subroutine command_sweep

   !> p50 and p95 of a latency sample, by sorting a copy.
   subroutine percentiles(samples, p50, p95)
      real(real64), intent(in) :: samples(:)
      real(real64), intent(out) :: p50, p95
      real(real64), allocatable :: sorted(:)
      integer(int32) :: count, index, jndex
      real(real64) :: swap

      count = size(samples, kind=int32)
      allocate(sorted(count))
      sorted = samples
      do index = 2_int32, count
         swap = sorted(index)
         jndex = index - 1_int32
         do while (jndex >= 1_int32)
            if (sorted(jndex) <= swap) exit
            sorted(jndex + 1_int32) = sorted(jndex)
            jndex = jndex - 1_int32
         end do
         sorted(jndex + 1_int32) = swap
      end do
      p50 = sorted(max(1_int32, (count + 1_int32) / 2_int32))
      p95 = sorted(max(1_int32, min(count, (count * 95_int32 + 99_int32) / 100_int32)))
      deallocate(sorted)
   end subroutine percentiles

   subroutine wall_clock(seconds)
      real(real64), intent(out) :: seconds
      integer(int64) :: ticks, rate

      call system_clock(ticks, rate)
      seconds = real(ticks, real64) / real(rate, real64)
   end subroutine wall_clock

   !> A JSON boolean with exactly the characters it names; MERGE would require
   !> equal-length literals and pad the shorter one.
   function json_bool(flag) result(text)
      logical, intent(in) :: flag
      character(len=:), allocatable :: text

      if (flag) then
         text = 'true'
      else
         text = 'false'
      end if
   end function json_bool

   !> Decimal text without padding.  A fixed-length result would concatenate its
   !> trailing blanks into the JSON line, so both helpers return exactly the
   !> characters they wrote.
   function i64_text(value) result(text)
      integer(int64), intent(in) :: value
      character(len=:), allocatable :: text
      character(len=32) :: buffer

      write(buffer, '(I0)') value
      text = trim(buffer)
   end function i64_text

   function r64_text(value) result(text)
      real(real64), intent(in) :: value
      character(len=:), allocatable :: text
      character(len=48) :: buffer

      write(buffer, '(F0.4)') value
      text = trim(buffer)
   end function r64_text

end program context_engine
