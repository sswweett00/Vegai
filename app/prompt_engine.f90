! SPDX-License-Identifier: MIT
!==============================================================================
!  prompt_engine.f90  --  byte-level training on a corpus, prompt -> response
!
!  Commands
!    train  --corpus FILE --checkpoint FILE   train the context engine on the
!                                             raw bytes of a corpus, streaming
!                                             it block by block
!    ask    --checkpoint FILE --prompt TEXT   feed the prompt, continue it
!    probe  --checkpoint FILE --probe FILE    score prompt/response pairs
!
!  Why bytes
!  ---------
!  The corpus is UTF-8 text and the engine vocabulary default is 256, so
!  token id = byte value.  There is no tokenizer and no vocabulary file, hence
!  no vocabulary drift between training and evaluation: the exact bytes the
!  corpus manifest declares are the exact tokens the model consumed.
!
!  Streaming contract
!  ------------------
!  The corpus is never loaded into memory.  A block of `chunk + 1` bytes is
!  read at an explicit file position, one optimiser update is applied, and the
!  position advances by `chunk` bytes, so the block's last byte is re-read as
!  the first byte of the next block and the prediction shift stays exact
!  across block boundaries.  The recurrent state is *not* reset between
!  blocks: memory carries across the whole file, which is the point of a
!  recurrent channel.
!
!  Every bound that can stop the run is explicit and reported: `--bytes` (how
!  much of the file to consume), `--max-updates`, `--max-seconds`.  A stopped
!  run prints the byte offset it reached and the fraction of the file that
!  is, so a partial pass is never mistaken for a full one.
!==============================================================================
program prompt_engine
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real64, output_unit, error_unit
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use fixed_budget_context
   use fixed_budget_backward
   use fixed_budget_optimizer
   use fixed_budget_training
   implicit none (type, external)

   integer(int32), parameter :: EXIT_USAGE = 2_int32
   integer(int32), parameter :: MAX_PROMPT = 8192_int32
   integer(int32), parameter :: MAX_RESPONSE = 256_int32
   integer(int32), parameter :: MAX_PROBE_ROWS = 4096_int32
   integer(int64), parameter :: MAX_WARMUP = 65536_int64
   integer(int32), parameter :: DOT = 46_int32
   integer(int32), parameter :: NEWLINE = 10_int32

   character(len=32) :: command
   type(fbc_config_t) :: config
   type(fbc_update_config_t) :: update_config
   character(len=1024) :: corpus_path, checkpoint_path, probe_path, prompt_text, resume_path
   logical :: learning_rate_given
   integer(int64) :: seed, byte_limit, start_byte, max_updates, max_seconds, progress_every, reset_every
   integer(int32) :: chunk_tokens, max_new
   real(real64) :: temperature
   ! Bytes a probe or an ask streams into the recurrent state before the prompt.
   ! Training never sees a cold state after its first token, so a with-context
   ! evaluation is the only one that measures the same regime; see docs.
   character(len=1024) :: warmup_path
   integer(int64) :: warmup_bytes
   integer(int32), allocatable :: warmup_tokens(:)
   integer(int32) :: warmup_count
   ! The number of bytes a prompt really has.  A `character(len=*)` dummy cannot
   ! tell a meaningful trailing space from padding, and every frame in this
   ! corpus ends with one ("... Cevap: ", "S: subject -> "), so the length has
   ! to travel beside the string.
   integer(int32) :: prompt_bytes
   integer(int32) :: argument_bytes

   call read_command(command)
   if (len_trim(command) == 0) call usage_and_stop('missing command')
   if (trim(command) == '--help' .or. trim(command) == '-h') call print_usage_and_stop()

   config%vocabulary_size = 256_int32
   config%width = 32_int32
   config%heads = 2_int32
   config%ffn_width = 64_int32
   config%local_window = 32_int32
   config%memory_scales = 4_int32
   config%memory_slots = 64_int32
   config%logical_horizon = 8192_int64
   update_config%learning_rate = 3.0e-3_real64
   seed = 20261028_int64
   chunk_tokens = 512_int32
   byte_limit = 0_int64
   start_byte = 0_int64
   max_updates = 0_int64
   max_seconds = 0_int64
   progress_every = 100_int64
   max_new = 48_int32
   temperature = 0.0_real64
   corpus_path = ''
   checkpoint_path = ''
   probe_path = 'probe.tsv'
   prompt_text = ''
   resume_path = ''
   warmup_path = ''
   warmup_bytes = 0_int64
   warmup_count = 0_int32
   prompt_bytes = 0_int32
   argument_bytes = 0_int32
   reset_every = 0_int64
   learning_rate_given = .false.

   call parse_options()

   select case (trim(command))
   case ('train')
      call command_train(config, update_config, seed, trim(corpus_path), trim(checkpoint_path), &
                         trim(resume_path), chunk_tokens, start_byte, byte_limit, max_updates, &
                         max_seconds, progress_every, learning_rate_given, reset_every)
   case ('ask')
      call command_ask(trim(checkpoint_path), prompt_text, max_new, temperature, seed)
   case ('probe')
      call command_probe(trim(checkpoint_path), trim(probe_path), max_new, temperature, seed)
   case default
      call usage_and_stop('unknown command: '//trim(command))
   end select

contains

   subroutine usage_and_stop(message)
      character(len=*), intent(in) :: message
      write(error_unit, '(A,A)') 'prompt_engine: ', trim(message)
      write(error_unit, '(A)') 'run with --help for the accepted options'
      stop EXIT_USAGE
   end subroutine usage_and_stop

   subroutine print_usage_and_stop()
      write(output_unit, '(A)') 'usage: vegai_prompt_engine COMMAND [options]'
      write(output_unit, '(A)') ''
      write(output_unit, '(A)') '  train  --corpus FILE --checkpoint FILE  [--bytes N] [--max-updates N]'
      write(output_unit, '(A)') '         [--max-seconds S] [--train-chunk-tokens N] [--width N]'
      write(output_unit, '(A)') '         [--heads N] [--ffn N] [--window H] [--local-window N]'
      write(output_unit, '(A)') '         [--memory-scales N] [--memory-slots N] [--lr R] [--seed N]'
      write(output_unit, '(A)') '         [--reset-every N]'
      write(output_unit, '(A)') '  ask    --checkpoint FILE --prompt TEXT  [--max-new N] [--temperature R]'
      write(output_unit, '(A)') '  probe  --checkpoint FILE --probe FILE   [--max-new N] [--temperature R]'
      write(output_unit, '(A)') '         [--warmup-bytes N] [--warmup-from FILE]'
      write(output_unit, '(A)') ''
      write(output_unit, '(A)') '  Tokens are bytes: token id = byte value, vocabulary 256.'
      write(output_unit, '(A)') '  --bytes 0 (default) consumes the whole file.'
      write(output_unit, '(A)') '  --max-updates 0 and --max-seconds 0 mean "no such bound".'
      stop 0_int32
   end subroutine print_usage_and_stop

   subroutine read_command(name)
      character(len=*), intent(out) :: name
      character(len=32) :: buffer
      integer :: received

      name = ''
      if (command_argument_count() < 1) return
      call get_command_argument(1, buffer, received)
      name = buffer(1:received)
   end subroutine read_command

   subroutine parse_options()
      character(len=64) :: key
      integer(int32) :: index, count

      count = command_argument_count()
      index = 2_int32
      do while (index <= count)
         call get_arg(index, key)
         select case (trim(key))
         case ('--corpus')
            index = index + 1_int32
            call get_arg(index, corpus_path)
         case ('--checkpoint')
            index = index + 1_int32
            call get_arg(index, checkpoint_path)
         case ('--probe')
            index = index + 1_int32
            call get_arg(index, probe_path)
         case ('--prompt')
            index = index + 1_int32
            call get_arg(index, prompt_text)
            prompt_bytes = argument_bytes
         case ('--window')
            index = index + 1_int32
            call get_arg(index, key)
            config%logical_horizon = parse_int64(key, '--window')
         case ('--local-window')
            index = index + 1_int32
            call get_arg(index, key)
            config%local_window = int(parse_int64(key, '--local-window'), int32)
         case ('--memory-scales')
            index = index + 1_int32
            call get_arg(index, key)
            config%memory_scales = int(parse_int64(key, '--memory-scales'), int32)
         case ('--memory-slots')
            index = index + 1_int32
            call get_arg(index, key)
            config%memory_slots = int(parse_int64(key, '--memory-slots'), int32)
         case ('--train-chunk-tokens')
            index = index + 1_int32
            call get_arg(index, key)
            chunk_tokens = int(parse_int64(key, '--train-chunk-tokens'), int32)
         case ('--width')
            index = index + 1_int32
            call get_arg(index, key)
            config%width = int(parse_int64(key, '--width'), int32)
         case ('--heads')
            index = index + 1_int32
            call get_arg(index, key)
            config%heads = int(parse_int64(key, '--heads'), int32)
         case ('--ffn')
            index = index + 1_int32
            call get_arg(index, key)
            config%ffn_width = int(parse_int64(key, '--ffn'), int32)
         case ('--bytes')
            index = index + 1_int32
            call get_arg(index, key)
            byte_limit = parse_int64(key, '--bytes')
         case ('--start-byte')
            index = index + 1_int32
            call get_arg(index, key)
            start_byte = parse_int64(key, '--start-byte')
         case ('--max-updates')
            index = index + 1_int32
            call get_arg(index, key)
            max_updates = parse_int64(key, '--max-updates')
         case ('--progress-every')
            index = index + 1_int32
            call get_arg(index, key)
            progress_every = parse_int64(key, '--progress-every')
         case ('--max-seconds')
            index = index + 1_int32
            call get_arg(index, key)
            max_seconds = parse_int64(key, '--max-seconds')
         case ('--max-new')
            index = index + 1_int32
            call get_arg(index, key)
            max_new = int(parse_int64(key, '--max-new'), int32)
         case ('--seed')
            index = index + 1_int32
            call get_arg(index, key)
            seed = parse_int64(key, '--seed')
         case ('--lr')
            index = index + 1_int32
            call get_arg(index, key)
            update_config%learning_rate = parse_real64(key, '--lr')
            learning_rate_given = .true.
         case ('--resume')
            index = index + 1_int32
            call get_arg(index, resume_path)
         case ('--temperature')
            index = index + 1_int32
            call get_arg(index, key)
            temperature = parse_real64(key, '--temperature')
         case ('--warmup-bytes')
            index = index + 1_int32
            call get_arg(index, key)
            warmup_bytes = parse_int64(key, '--warmup-bytes')
         case ('--warmup-from')
            index = index + 1_int32
            call get_arg(index, warmup_path)
         case ('--reset-every')
            index = index + 1_int32
            call get_arg(index, key)
            reset_every = parse_int64(key, '--reset-every')
         case default
            call usage_and_stop('unknown option: '//trim(key))
         end select
         index = index + 1_int32
      end do

      if (max_new < 1 .or. max_new > MAX_RESPONSE) call usage_and_stop('--max-new is out of range')
      if (chunk_tokens < 1 .or. chunk_tokens > FBC_MAX_CHUNK_TOKENS) &
         call usage_and_stop('--train-chunk-tokens is out of range')
      if (warmup_bytes < 0_int64 .or. warmup_bytes > MAX_WARMUP) &
         call usage_and_stop('--warmup-bytes is out of range')
      if (reset_every < 0_int64) call usage_and_stop('--reset-every is out of range')
   end subroutine parse_options

   subroutine get_arg(index, value)
      integer(int32), intent(in) :: index
      character(len=*), intent(out) :: value
      integer :: received

      value = ''
      if (index > command_argument_count()) call usage_and_stop('missing option value')
      call get_command_argument(index, value, received)
      argument_bytes = int(received, int32)
   end subroutine get_arg

   function parse_int64(text, option) result(value)
      character(len=*), intent(in) :: text, option
      integer(int64) :: value
      integer :: io_status

      value = 0_int64
      if (len_trim(text) == 0) call usage_and_stop('empty value for '//trim(option))
      if (text(1:1) == '-' .or. text(1:1) == '+') &
         call usage_and_stop('signed value for '//trim(option))
      read(text, *, iostat=io_status) value
      if (io_status /= 0 .or. value < 0_int64) call usage_and_stop('not a non-negative integer: '//trim(option))
   end function parse_int64

   function parse_real64(text, option) result(value)
      character(len=*), intent(in) :: text, option
      real(real64) :: value
      integer :: io_status

      value = 0.0_real64
      if (len_trim(text) == 0) call usage_and_stop('empty value for '//trim(option))
      read(text, *, iostat=io_status) value
      if (io_status /= 0 .or. value < 0.0_real64) call usage_and_stop('not a non-negative number: '//trim(option))
   end function parse_real64

   !---------------------------------------------------------------------------
   !  train -- stream a corpus file as bytes
   !---------------------------------------------------------------------------
   subroutine command_train(config, update_config, seed, corpus, checkpoint, resume, chunk, start_byte, &
                            byte_limit, max_updates, max_seconds, progress_every, learning_rate_given, &
                            reset_every)
      type(fbc_config_t), intent(in) :: config
      type(fbc_update_config_t), intent(inout) :: update_config
      integer(int64), intent(in) :: seed, start_byte, byte_limit, max_updates, max_seconds, progress_every
      !> Reset the recurrent state every this many tokens (0 = never).  Training
      !> only ever saw a warm state -- the memory channels are empty for the
      !> first tokens of a stream and filled afterwards -- while every prompt is
      !> answered from a cold one.  Resetting on a fixed period puts both regimes
      !> in the training distribution instead of leaving the cold one unseen.
      integer(int64), intent(in) :: reset_every
      character(len=*), intent(in) :: corpus, checkpoint, resume
      logical, intent(in) :: learning_rate_given
      integer(int32), intent(in) :: chunk
      type(fbc_update_config_t) :: loaded_config
      type(fbc_model_t) :: model
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      type(fbc_gradient_t) :: gradient
      type(fbc_optimizer_t) :: optimizer
      type(fbc_metrics_t) :: metrics
      integer(int32) :: status, unit_number, io_status, block_bytes, block_tokens
      integer(int32), allocatable :: tokens(:)
      integer(int8), allocatable :: raw(:)
      integer(int64) :: consumed, file_bytes, stopped_at, started, now, elapsed, since_reset
      integer(int64) :: tokens_before, updates_before, session_tokens, session_updates
      real(real64) :: loss, rate, session_rate, session_loss
      logical :: bounded, first_field

      if (len_trim(corpus) == 0) call usage_and_stop('train needs --corpus')
      if (len_trim(checkpoint) == 0) call usage_and_stop('train needs --checkpoint')
      if (.not. fbc_config_is_valid(config)) call usage_and_stop('configuration is not valid')

      file_bytes = file_size(corpus)
      bounded = byte_limit > 0_int64 .and. byte_limit < file_bytes
      if (bounded) file_bytes = byte_limit
      if (start_byte > file_bytes) call usage_and_stop('--start-byte is past the training end')
      ! A range of zero or one byte has nothing to predict.  That is not an
      ! error: it is what a chained session sees once it has reached the end,
      ! and reporting an empty session is more useful than failing the run.

      if (len_trim(resume) > 0) then
         ! Continue an existing run.  The checkpoint carries the model, the
         ! optimiser moments and the cumulative counters, so a resumed session
         ! is a continuation of the same run, not a fresh one.
         call load_model(trim(resume), model, optimizer, loaded_config, metrics)
         ! The checkpoint's optimiser settings win unless --lr was given on the
         ! command line, in which case the caller is deliberately overriding.
         if (.not. learning_rate_given) update_config = loaded_config
         call ensure_config_matches(model, config)
      else
         call fbc_model_init(model, config, seed, status)
         if (status /= FBC_OK) call usage_and_stop('model initialisation failed')
      end if
      call fbc_state_init(state, model, status)
      if (status /= FBC_OK) call usage_and_stop('stream state allocation failed')
      call fbc_workspace_init(workspace, model, status)
      if (status /= FBC_OK) call usage_and_stop('workspace allocation failed')
      call fbc_gradient_init(gradient, model, status)
      if (status /= FBC_OK) call usage_and_stop('gradient allocation failed')
      if (len_trim(resume) == 0) then
         call fbc_optimizer_init(optimizer, config, status)
         if (status /= FBC_OK) call usage_and_stop('optimizer allocation failed')
      end if

      open(newunit=unit_number, file=corpus, access='stream', form='unformatted', &
           action='read', status='old', iostat=io_status)
      if (io_status /= 0) call usage_and_stop('cannot open the corpus: '//trim(corpus))

      allocate(raw(chunk + 1_int32), tokens(chunk + 1_int32))
      consumed = start_byte
      since_reset = 0_int64
      tokens_before = metrics%tokens
      updates_before = metrics%updates
      metrics%numeric_ok = .true.
      call wall_clock(started)
      do while (consumed + 1_int64 < file_bytes)
         ! The last block of a range may be shorter than --train-chunk-tokens:
         ! consuming the tail is what makes a chained set of sessions cover the
         ! whole file instead of stopping one block short of the end.
         block_tokens = int(min(int(chunk, int64), file_bytes - consumed - 1_int64), int32)
         if (max_updates > 0_int64 .and. metrics%updates >= max_updates) exit
         if (max_seconds > 0_int64) then
            call wall_clock(now)
            if (now - started >= max_seconds) exit
         end if
         ! Explicit position: the block overlap is expressed once, here.
         read(unit_number, pos=consumed + 1_int64, iostat=io_status) raw(1_int32:block_tokens + 1_int32)
         if (io_status /= 0) exit
         do block_bytes = 1_int32, block_tokens + 1_int32
            tokens(block_bytes) = iand(int(raw(block_bytes), int32), 255_int32)
         end do
         call fbc_train_stream(model, optimizer, state, tokens, block_tokens, block_tokens, &
                              update_config, workspace, gradient, metrics, status)
         if (status /= FBT_OK) then
            write(error_unit, '(A,I0,A,I0)') 'training stopped at byte ', consumed, ' with status ', status
            exit
         end if
         consumed = consumed + int(block_tokens, int64)
         since_reset = since_reset + int(block_tokens, int64)
         if (reset_every > 0_int64 .and. since_reset >= reset_every) then
            call fbc_state_reset(state, model, status)
            if (status /= FBC_OK) then
               write(error_unit, '(A,I0)') 'stream reset failed, status ', status
               exit
            end if
            since_reset = 0_int64
         end if
         if (progress_every > 0_int64 .and. mod(metrics%updates, progress_every) == 0_int64) then
            call wall_clock(now)
            write(output_unit, '(A,I0,A,I0,A,F0.4,A,F0.1,A,I0)') 'progress: bytes ', consumed, &
               ' updates ', metrics%updates, ' mean_nll ', &
               metrics%loss_sum / real(max(1_int64, metrics%tokens), real64), ' tok/s ', &
               real(metrics%tokens, real64) / real(max(1_int64, now - started), real64), &
               ' reach ', nint(fbc_learned_horizon(model))
            flush(output_unit)
         end if
      end do
      close(unit_number)
      call wall_clock(now)
      elapsed = max(1_int64, now - started)
      stopped_at = consumed

      call fbc_checkpoint_save(checkpoint, model, optimizer, update_config, metrics, status)
      if (status /= FBT_OK) then
         write(error_unit, '(A,I0)') 'checkpoint write failed, status ', status
         stop 1_int32
      end if

      loss = metrics%loss_sum / real(max(1_int64, metrics%tokens), real64)
      session_tokens = metrics%tokens - tokens_before
      session_updates = metrics%updates - updates_before
      ! Throughput is this session's tokens over this session's seconds; the
      ! cumulative counters would understate nothing here but would mix runs.
      rate = real(session_tokens, real64) / real(elapsed, real64)
      session_rate = rate
      session_loss = metrics%loss_sum / real(max(1_int64, metrics%tokens), real64)
      first_field = .true.
      write(output_unit, '(A)') 'CORPUS_TRAIN_JSON {'
      call field_text('corpus', trim(corpus), first_field)
      call field_int('corpus_bytes', file_size(corpus), first_field)
      call field_int('start_byte', start_byte, first_field)
      call field_int('end_byte', stopped_at, first_field)
      call field_int('consumed_bytes', stopped_at - start_byte, first_field)
      call field_real('file_fraction', real(stopped_at, real64) / &
                      real(max(1_int64, file_size(corpus)), real64), first_field)
      call field_bool('bounded_by_bytes', bounded, first_field)
      call field_bool('whole_file', stopped_at >= file_size(corpus), first_field)
      call field_int('updates', metrics%updates, first_field)
      call field_int('tokens', metrics%tokens, first_field)
      call field_int('session_updates', session_updates, first_field)
      call field_int('session_tokens', session_tokens, first_field)
      call field_real('mean_nll', loss, first_field)
      call field_real('session_mean_nll', session_loss, first_field)
      call field_real('tokens_per_second', session_rate, first_field)
      call field_int('seconds', elapsed, first_field)
      call field_real('learned_reach', fbc_learned_horizon(model), first_field)
      call field_int('parameters', fbc_parameter_count(config), first_field)
      call field_int('reset_every', reset_every, first_field)
      call field_bool('numeric_ok', metrics%numeric_ok, first_field)
      write(output_unit, '(A)') ''
      write(output_unit, '(A)') '}'
      if (stopped_at < file_size(corpus)) then
         write(output_unit, '(A,I0,A,I0)') 'note: partial pass: reached byte ', stopped_at, ' of ', &
            file_size(corpus)
      end if

      call fbc_model_destroy(model)
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
      call fbc_gradient_destroy(gradient)
      call fbc_optimizer_destroy(optimizer)
      deallocate(raw, tokens)
   end subroutine command_train

   !> A resumed run must describe the same shape as the checkpoint; otherwise
   !> the payloads would be read into arrays of the wrong extent.
   subroutine ensure_config_matches(model, config)
      type(fbc_model_t), intent(in) :: model
      type(fbc_config_t), intent(in) :: config

      if (model%config%width /= config%width .or. model%config%heads /= config%heads .or. &
          model%config%ffn_width /= config%ffn_width .or. &
          model%config%local_window /= config%local_window .or. &
          model%config%memory_scales /= config%memory_scales .or. &
          model%config%memory_slots /= config%memory_slots .or. &
          model%config%vocabulary_size /= config%vocabulary_size) then
         call usage_and_stop('--resume checkpoint shape does not match the requested options')
      end if
   end subroutine ensure_config_matches

   !---------------------------------------------------------------------------
   !  ask -- one prompt in, one response out
   !---------------------------------------------------------------------------
   subroutine command_ask(checkpoint, prompt, max_new, temperature, seed)
      character(len=*), intent(in) :: checkpoint, prompt
      integer(int32), intent(in) :: max_new
      real(real64), intent(in) :: temperature
      integer(int64), intent(in) :: seed
      type(fbc_model_t) :: model
      type(fbc_optimizer_t) :: optimizer
      type(fbc_update_config_t) :: loaded_config
      type(fbc_metrics_t) :: counters
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      character(len=MAX_RESPONSE) :: response
      integer(int32) :: status, length
      integer(int64) :: random_state
      logical :: first_field

      if (len_trim(checkpoint) == 0) call usage_and_stop('ask needs --checkpoint')
      if (len_trim(prompt) == 0) call usage_and_stop('ask needs --prompt')
      if (len_trim(prompt) > MAX_PROMPT) call usage_and_stop('prompt is too long')

      call load_model(checkpoint, model, optimizer, loaded_config, counters)
      call fbc_state_init(state, model, status)
      if (status /= FBC_OK) call usage_and_stop('stream state allocation failed')
      call fbc_workspace_init(workspace, model, status)
      if (status /= FBC_OK) call usage_and_stop('workspace allocation failed')
      call load_warmup(trim(warmup_path), trim(corpus_path), warmup_bytes)

      random_state = seed * 6364136223846793005_int64 + 1442695040888963407_int64
      call respond(model, state, workspace, prompt, prompt_bytes, max_new, temperature, &
                   random_state, response, length)

      first_field = .true.
      write(output_unit, '(A)') 'PROMPT_RESPONSE_JSON {'
      call field_text('prompt', trim(prompt), first_field)
      call field_text('response', response(1_int32:length), first_field)
      call field_int('warmup_bytes', int(warmup_count, int64), first_field)
      call field_real('learned_reach', fbc_learned_horizon(model), first_field)
      write(output_unit, '(A)') ''
      write(output_unit, '(A)') '}'

      call fbc_model_destroy(model)
      call fbc_optimizer_destroy(optimizer)
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
   end subroutine command_ask

   !---------------------------------------------------------------------------
   !  probe -- score a prompt/response file
   !---------------------------------------------------------------------------
   subroutine command_probe(checkpoint, probe_file, max_new, temperature, seed)
      character(len=*), intent(in) :: checkpoint, probe_file
      integer(int32), intent(in) :: max_new
      real(real64), intent(in) :: temperature
      integer(int64), intent(in) :: seed
      type(fbc_model_t) :: model
      type(fbc_optimizer_t) :: optimizer
      type(fbc_update_config_t) :: loaded_config
      type(fbc_metrics_t) :: counters
      type(fbc_state_t) :: state
      type(fbc_workspace_t) :: workspace
      character(len=4096) :: line
      character(len=32) :: kind
      character(len=MAX_RESPONSE) :: expected, response
      character(len=MAX_PROMPT) :: prompt
      integer(int32) :: status, unit_number, io_status, length, rows
      integer(int32) :: fact_total, fact_hit, copy_total, copy_hit, next_total, next_hit
      integer(int32) :: fact_bytes, fact_byte_hits, copy_bytes, copy_byte_hits
      integer(int32) :: next_bytes, next_byte_hits
      real(real64) :: fact_nll, copy_nll, next_nll
      integer(int64) :: random_state
      logical :: first_field
      character(len=MAX_RESPONSE) :: scoring
      integer(int32) :: scoring_bytes, scoring_hits, prompt_length
      real(real64) :: scoring_nll

      if (len_trim(checkpoint) == 0) call usage_and_stop('probe needs --checkpoint')
      call load_model(checkpoint, model, optimizer, loaded_config, counters)
      call fbc_state_init(state, model, status)
      if (status /= FBC_OK) call usage_and_stop('stream state allocation failed')
      call fbc_workspace_init(workspace, model, status)
      if (status /= FBC_OK) call usage_and_stop('workspace allocation failed')
      call load_warmup(trim(warmup_path), trim(corpus_path), warmup_bytes)

      open(newunit=unit_number, file=probe_file, action='read', status='old', iostat=io_status)
      if (io_status /= 0) call usage_and_stop('cannot open the probe file: '//trim(probe_file))

      fact_total = 0; fact_hit = 0
      copy_total = 0; copy_hit = 0
      next_total = 0; next_hit = 0
      fact_bytes = 0; fact_byte_hits = 0; fact_nll = 0.0_real64
      copy_bytes = 0; copy_byte_hits = 0; copy_nll = 0.0_real64
      next_bytes = 0; next_byte_hits = 0; next_nll = 0.0_real64
      rows = 0_int32
      random_state = seed * 6364136223846793005_int64 + 1442695040888963407_int64
      write(output_unit, '(A)') 'PROBE_EXAMPLES ['
      do
         read(unit_number, '(A)', iostat=io_status) line
         if (io_status /= 0) exit
         if (len_trim(line) == 0) cycle
         if (line(1:1) == '#') cycle
         if (rows >= MAX_PROBE_ROWS) exit
         call split_row(line, kind, prompt, expected, prompt_length)
         if (len_trim(prompt) == 0 .or. len_trim(expected) == 0) cycle
         rows = rows + 1_int32
         call respond(model, state, workspace, prompt, prompt_length, max_new, temperature, &
                      random_state, response, length)
         ! Exact match on a greedy continuation is the headline number, but a
         ! tiny model that emits one wrong byte scores zero there while clearly
         ! ranking the right bytes above the rest.  The teacher-forced figures
         ! below say how often the *next byte* was right and how surprised the
         ! model was by the expected continuation, so the probe is informative
         ! even when the exact-match rate is zero.
         call score_continuation(model, state, workspace, prompt, prompt_length, expected, &
                                 scoring, scoring_bytes, scoring_hits, scoring_nll)
         select case (trim(kind))
         case ('fact')
            fact_total = fact_total + 1_int32
            if (trim(response(1_int32:length)) == trim(expected)) fact_hit = fact_hit + 1_int32
            fact_bytes = fact_bytes + scoring_bytes
            fact_byte_hits = fact_byte_hits + scoring_hits
            fact_nll = fact_nll + scoring_nll
         case ('copy')
            copy_total = copy_total + 1_int32
            if (trim(response(1_int32:length)) == trim(expected)) copy_hit = copy_hit + 1_int32
            copy_bytes = copy_bytes + scoring_bytes
            copy_byte_hits = copy_byte_hits + scoring_hits
            copy_nll = copy_nll + scoring_nll
         case ('next')
            next_total = next_total + 1_int32
            if (trim(response(1_int32:length)) == trim(expected)) next_hit = next_hit + 1_int32
            next_bytes = next_bytes + scoring_bytes
            next_byte_hits = next_byte_hits + scoring_hits
            next_nll = next_nll + scoring_nll
         end select
         if (rows <= 8_int32) then
            if (rows > 1_int32) write(output_unit, '(A)') ','
            write(output_unit, '(A,A,A,A,A,A,A)', advance='no') '  {"kind": "', trim(kind), '", "prompt": "', &
               trim(json_escape(trim(prompt))), '", "expected": "', trim(json_escape(trim(expected))), '", '
            write(output_unit, '(A,A,A,A,A)', advance='no') '"response": "', &
               trim(json_escape(response(1_int32:length))), '", "greedy": "', &
               trim(json_escape(trim(scoring))), '"}'
         end if
      end do
      close(unit_number)
      write(output_unit, '(A)') ''
      write(output_unit, '(A)') ']'

      first_field = .true.
      write(output_unit, '(A)') 'PROBE_JSON {'
      call field_text('probe_file', trim(probe_file), first_field)
      call field_int('warmup_bytes', int(warmup_count, int64), first_field)
      call field_int('rows', int(rows, int64), first_field)
      call field_family('fact', fact_hit, fact_total, fact_byte_hits, fact_bytes, fact_nll, first_field)
      call field_family('copy', copy_hit, copy_total, copy_byte_hits, copy_bytes, copy_nll, first_field)
      call field_family('next', next_hit, next_total, next_byte_hits, next_bytes, next_nll, first_field)
      call field_real('accuracy', score(fact_hit + copy_hit + next_hit, &
                      fact_total + copy_total + next_total), first_field)
      call field_real('byte_accuracy', score(fact_byte_hits + copy_byte_hits + next_byte_hits, &
                      fact_bytes + copy_bytes + next_bytes), first_field)
      call field_real('learned_reach', fbc_learned_horizon(model), first_field)
      write(output_unit, '(A)') ''
      write(output_unit, '(A)') '}'

      call fbc_model_destroy(model)
      call fbc_optimizer_destroy(optimizer)
      call fbc_state_destroy(state)
      call fbc_workspace_destroy(workspace)
   end subroutine command_probe

   !---------------------------------------------------------------------------
   !  Shared helpers
   !---------------------------------------------------------------------------
   subroutine load_model(checkpoint, model, optimizer, update_config, counters)
      character(len=*), intent(in) :: checkpoint
      type(fbc_model_t), intent(inout) :: model
      type(fbc_optimizer_t), intent(inout) :: optimizer
      type(fbc_update_config_t), intent(out) :: update_config
      type(fbc_metrics_t), intent(out) :: counters
      integer(int32) :: status

      call fbc_checkpoint_load(checkpoint, model, optimizer, update_config, counters, status)
      if (status == FBT_ERR_LEGACY) call usage_and_stop('legacy VGNR checkpoint: use vegai_neural_pretrain')
      if (status == FBT_ERR_VERSION) call usage_and_stop('older checkpoint format: retrain it')
      if (status /= FBT_OK) then
         write(error_unit, '(A,I0)') 'checkpoint load failed, status ', status
         stop 1_int32
      end if
   end subroutine load_model

   !> Feed `prompt` byte by byte, then continue the stream.
   subroutine respond(model, state, workspace, prompt, prompt_length, max_new, temperature, &
                      random_state, response, length)
      type(fbc_model_t), intent(in) :: model
      type(fbc_state_t), intent(inout) :: state
      type(fbc_workspace_t), intent(inout) :: workspace
      character(len=*), intent(in) :: prompt
      integer(int32), intent(in) :: prompt_length
      integer(int32), intent(in) :: max_new
      real(real64), intent(in) :: temperature
      integer(int64), intent(inout) :: random_state
      character(len=*), intent(out) :: response
      integer(int32), intent(out) :: length
      integer(int32) :: status, index, produced, token
      real(real64), allocatable :: logits(:), hidden(:)

      allocate(logits(model%config%vocabulary_size), hidden(model%config%width))
      call fbc_state_reset(state, model, status)
      response = ''
      length = 0_int32

      do index = 1_int32, warmup_count
         call fbc_step(model, state, workspace, warmup_tokens(index), hidden, logits, status)
      end do

      ! The caller's byte count, not len(): the dummy is bound to a fixed-length
      ! buffer, so len() would stream the padding, and len_trim() would silently
      ! drop the trailing space that every frame in this corpus ends with.
      do index = 1_int32, min(prompt_length, len(prompt))
         call fbc_step(model, state, workspace, byte_of(prompt(index:index)), hidden, logits, status)
      end do

      produced = 0_int32
      do while (produced < max_new)
         if (temperature <= 0.0_real64) then
            token = argmax_token(logits)
         else
            token = sample_token(logits, temperature, random_state)
         end if
         ! Stop at a full stop or a line break: the corpus frames every response
         ! that way, so a response is directly comparable with its label.
         if (token == DOT .or. token == NEWLINE) exit
         if (token < 32_int32) exit
         if (length < len(response)) then
            length = length + 1_int32
            response(length:length) = byte_character(token)
         end if
         call fbc_step(model, state, workspace, token, hidden, logits, status)
         produced = produced + 1_int32
      end do
      deallocate(logits, hidden)
   end subroutine respond

   !> Raw byte value of a one-character string, without assuming ASCII implies
   !> 7 bits: UTF-8 continuation bytes are 0x80..0xBF and must survive.
   pure function byte_of(text) result(token)
      character(len=*), intent(in) :: text
      integer(int32) :: token
      integer(int8) :: signed

      signed = transfer(text(1:1), signed)
      token = iand(int(signed, int32), 255_int32)
   end function byte_of

   !> The inverse of `byte_of`: one raw byte as a one-character string.
   pure function byte_character(token) result(text)
      integer(int32), intent(in) :: token
      character(len=1) :: text
      integer(int8) :: signed

      signed = int(iand(token, 255_int32), int8)
      text = transfer(signed, text)
   end function byte_character

   !> Teacher-forced score of one expected response given the prompt.
   !>
   !> The prompt is fed first, so the logits after its last byte are the model's
   !> prediction for the first expected byte.  Then each expected byte is fed
   !> back (teacher forcing) and the next prediction is scored, so the outputs
   !> are: the continuation the model would have written greedily, how many
   !> bytes it ranked first, and the total negative log-likelihood of the true
   !> continuation.  The greedy text repeats what `respond` produced for the
   !> same prompt, which is the exact-match number, and the other two say how
   !> close the model was when it did not match exactly.
   subroutine score_continuation(model, state, workspace, prompt, prompt_length, expected, greedy, &
                                 byte_total, byte_hits, nll)
      type(fbc_model_t), intent(in) :: model
      type(fbc_state_t), intent(inout) :: state
      type(fbc_workspace_t), intent(inout) :: workspace
      character(len=*), intent(in) :: prompt, expected
      integer(int32), intent(in) :: prompt_length
      character(len=*), intent(out) :: greedy
      integer(int32), intent(out) :: byte_total, byte_hits
      real(real64), intent(out) :: nll
      integer(int32) :: status, index, target, produced
      real(real64), allocatable :: logits(:), hidden(:), dlogits(:)
      real(real64) :: loss

      allocate(logits(model%config%vocabulary_size), hidden(model%config%width), &
               dlogits(model%config%vocabulary_size))
      call fbc_state_reset(state, model, status)
      greedy = ''
      byte_total = 0_int32
      byte_hits = 0_int32
      nll = 0.0_real64
      produced = 0_int32
      do index = 1_int32, warmup_count
         call fbc_step(model, state, workspace, warmup_tokens(index), hidden, logits, status)
      end do
      do index = 1_int32, min(prompt_length, len(prompt))
         call fbc_step(model, state, workspace, byte_of(prompt(index:index)), hidden, logits, status)
      end do
      do index = 1_int32, len_trim(expected)
         target = byte_of(expected(index:index))
         call fbc_loss_and_gradient(logits, target, dlogits, loss, status)
         if (status /= FBC_OK) exit
         nll = nll + loss
         byte_total = byte_total + 1_int32
         if (argmax_token(logits) == target) byte_hits = byte_hits + 1_int32
         if (produced < len(greedy)) then
            produced = produced + 1_int32
            greedy(produced:produced) = byte_character(argmax_token(logits))
         end if
         call fbc_step(model, state, workspace, target, hidden, logits, status)
      end do
      deallocate(logits, hidden, dlogits)
   end subroutine score_continuation

   pure function argmax_token(logits) result(token)
      real(real64), intent(in) :: logits(:)
      integer(int32) :: token, index

      token = 1_int32
      do index = 2_int32, size(logits)
         if (logits(index) > logits(token)) token = index
      end do
      token = token - 1_int32
   end function argmax_token

   !> Temperature sampling over the logits, from a 64-bit xorshift stream.
   function sample_token(logits, temperature, random_state) result(token)
      real(real64), intent(in) :: logits(:)
      real(real64), intent(in) :: temperature
      integer(int64), intent(inout) :: random_state
      integer(int32) :: token, index
      real(real64) :: maximum, total, draw, threshold

      maximum = maxval(logits)
      total = 0.0_real64
      do index = 1_int32, size(logits)
         total = total + exp((logits(index) - maximum) / temperature)
      end do
      draw = next_uniform(random_state) * total
      threshold = 0.0_real64
      token = int(size(logits), int32) - 1_int32
      do index = 1_int32, size(logits)
         threshold = threshold + exp((logits(index) - maximum) / temperature)
         if (draw <= threshold) then
            token = index - 1_int32
            exit
         end if
      end do
   end function sample_token

   function next_uniform(random_state) result(value)
      integer(int64), intent(inout) :: random_state
      real(real64) :: value

      random_state = ieor(random_state, shiftl(random_state, 13_int64))
      random_state = ieor(random_state, shiftr(random_state, 7_int64))
      random_state = ieor(random_state, shiftl(random_state, 17_int64))
      value = real(iand(abs(random_state), int(z'7FFFFFFF', int64)), real64) / 2147483648.0_real64
   end function next_uniform

   pure function score(hits, total) result(value)
      integer(int32), intent(in) :: hits, total
      real(real64) :: value

      if (total <= 0_int32) then
         value = 0.0_real64
      else
         value = real(hits, real64) / real(total, real64)
      end if
   end function score

   !> One `"name": value` member of the enclosing object.  The separators are
   !> emitted here rather than appended to every value, because a missing comma
   !> is invisible to Fortran and fatal to every JSON reader.
   subroutine json_member(name, value_text, first)
      character(len=*), intent(in) :: name, value_text
      logical, intent(inout) :: first

      ! MERGE(TSOURCE, FSOURCE, MASK) returns TSOURCE when MASK is true, so the
      ! comma belongs to the *not first* branch.
      write(output_unit, '(A,A,A,A)') trim(merge(' ', ',', first)), '  "', trim(name)//'": ', value_text
      first = .false.
   end subroutine json_member

   subroutine field_int(name, value, first)
      character(len=*), intent(in) :: name
      integer(int64), intent(in) :: value
      logical, intent(inout) :: first
      character(len=32) :: text

      write(text, '(I0)') value
      call json_member(name, trim(text), first)
   end subroutine field_int

   !> Fortran writes 0.5 as ".500000" and JSON needs a leading zero; a
   !> non-finite value is not representable in JSON at all, so it becomes null.
   subroutine json_real_text(value, text)
      real(real64), intent(in) :: value
      character(len=*), intent(out) :: text
      character(len=64) :: scratch

      if (.not. ieee_is_finite(value)) then
         text = 'null'
         return
      end if
      write(scratch, '(F0.6)') value
      scratch = adjustl(scratch)
      if (scratch(1:1) == '.') then
         text = '0'//trim(scratch)
      else if (scratch(1:2) == '-.') then
         text = '-0'//trim(scratch(2:))
      else
         text = trim(scratch)
      end if
   end subroutine json_real_text

   function int_text(value) result(text)
      integer(int32), intent(in) :: value
      character(len=16) :: text

      write(text, '(I0)') value
   end function int_text

   subroutine field_real(name, value, first)
      character(len=*), intent(in) :: name
      real(real64), intent(in) :: value
      logical, intent(inout) :: first
      character(len=32) :: text

      call json_real_text(value, text)
      call json_member(name, trim(text), first)
   end subroutine field_real

   subroutine field_bool(name, value, first)
      character(len=*), intent(in) :: name
      logical, intent(in) :: value
      logical, intent(inout) :: first

      if (value) then
         call json_member(name, 'true', first)
      else
         call json_member(name, 'false', first)
      end if
   end subroutine field_bool

   !> `"fact": {"correct": n, "accuracy": r, "n": n}` -- one probe family.
   subroutine field_family(name, hits, total, byte_hits, byte_total, nll_total, first)
      character(len=*), intent(in) :: name
      integer(int32), intent(in) :: hits, total, byte_hits, byte_total
      real(real64), intent(in) :: nll_total
      logical, intent(inout) :: first
      character(len=256) :: text, byte_text
      character(len=32) :: score_text, byte_score_text, nll_text

      write(byte_text, '(I0)') byte_total
      call json_real_text(score(hits, total), score_text)
      call json_real_text(score(byte_hits, byte_total), byte_score_text)
      if (byte_total > 0_int32) then
         call json_real_text(nll_total / real(byte_total, real64), nll_text)
      else
         nll_text = 'null'
      end if
      text = '{"correct": '//int_text(hits)//', "accuracy": '//trim(score_text)// &
             ', "n": '//int_text(total)// &
             ', "correct_bytes": '//trim(byte_text)// &
             ', "byte_accuracy": '//trim(byte_score_text)// &
             ', "mean_nll": '//trim(nll_text)//'}'
      call json_member(name, trim(text), first)
   end subroutine field_family

   subroutine field_text(name, value, first)
      character(len=*), intent(in) :: name, value
      logical, intent(inout) :: first

      call json_member(name, '"'//trim(json_escape(value))//'"', first)
   end subroutine field_text

   pure function json_escape(text) result(escaped)
      character(len=*), intent(in) :: text
      character(len=4 * max(1, len(text))) :: escaped
      integer(int32) :: index, out
      character(len=1) :: ch

      escaped = ''
      out = 0_int32
      do index = 1_int32, len_trim(text)
         ch = text(index:index)
         select case (ch)
         case ('"')
            out = out + 1_int32; escaped(out:out) = '\'
            out = out + 1_int32; escaped(out:out) = '"'
         case ('\')
            out = out + 1_int32; escaped(out:out) = '\'
            out = out + 1_int32; escaped(out:out) = '\'
         case default
            if (iachar(ch) < 32 .and. iachar(ch) >= 0) then
               out = out + 1_int32; escaped(out:out) = ' '
            else
               out = out + 1_int32; escaped(out:out) = ch
            end if
         end select
      end do
   end function json_escape

   !> `kind<TAB>prompt<TAB>expected`.  The prompt may not contain a tab.
   subroutine split_row(line, kind, prompt, expected, prompt_length)
      character(len=*), intent(in) :: line
      character(len=*), intent(out) :: kind, prompt, expected
      integer(int32), intent(out) :: prompt_length
      integer(int32) :: first, second

      kind = ''
      prompt = ''
      expected = ''
      prompt_length = 0_int32
      first = index(line, achar(9))
      if (first <= 1_int32) return
      second = index(line(first + 1_int32:), achar(9))
      if (second <= 1_int32) return
      second = second + first
      if (second >= len_trim(line)) return
      kind = line(1_int32:first - 1_int32)
      prompt = line(first + 1_int32:second - 1_int32)
      ! The field as written, including a trailing space that belongs to it.
      prompt_length = second - first - 1_int32
      expected = line(second + 1_int32:len_trim(line))
   end subroutine split_row

   !> Reads the bytes a prompt is preceded by.  A streaming model is trained
   !> with a state that has already consumed text; a prompt presented to an
   !> empty state asks it a question in a regime it never saw, and the answers
   !> differ by orders of magnitude in likelihood.  `--warmup-bytes N` streams N
   !> bytes of a corpus prefix into the state before every prompt, so the
   !> measurement is taken in the training regime.
   subroutine load_warmup(path, fallback_path, nbytes)
      character(len=*), intent(in) :: path, fallback_path
      integer(int64), intent(in) :: nbytes
      character(len=1024) :: source
      integer(int32) :: unit_number, io_status, index, wanted
      integer(int8), allocatable :: raw(:)

      warmup_count = 0_int32
      if (nbytes <= 0_int64) return
      source = path
      if (len_trim(source) == 0) source = fallback_path
      if (len_trim(source) == 0) call usage_and_stop('--warmup-bytes needs --warmup-from or --corpus')
      wanted = int(min(nbytes, file_size(trim(source))), int32)
      if (wanted <= 0_int32) return
      allocate(raw(wanted), warmup_tokens(wanted))
      open(newunit=unit_number, file=trim(source), access='stream', form='unformatted', &
           action='read', status='old', iostat=io_status)
      if (io_status /= 0) call usage_and_stop('cannot open the warm-up text: '//trim(source))
      read(unit_number, iostat=io_status) raw
      close(unit_number)
      if (io_status /= 0) call usage_and_stop('cannot read the warm-up text: '//trim(source))
      do index = 1_int32, wanted
         warmup_tokens(index) = iand(int(raw(index), int32), 255_int32)
      end do
      warmup_count = wanted
      deallocate(raw)
   end subroutine load_warmup

   function file_size(path) result(bytes)
      character(len=*), intent(in) :: path
      integer(int64) :: bytes
      integer(int64) :: size_field
      integer :: status

      inquire(file=path, size=size_field, iostat=status)
      if (status /= 0) call usage_and_stop('cannot stat the corpus: '//trim(path))
      bytes = size_field
   end function file_size

   subroutine wall_clock(seconds)
      integer(int64), intent(out) :: seconds
      integer(int64) :: ticks, ticks_per_second, maximum

      call system_clock(ticks, ticks_per_second, maximum)
      if (ticks_per_second > 0_int64) then
         seconds = ticks / ticks_per_second
      else
         seconds = 0_int64
      end if
   end subroutine wall_clock

end program prompt_engine
