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
!  The corpus is UTF-8 text and the engine vocabulary is 256. Bytes 1..255 map
!  directly to those token IDs. V6 maps byte 0 to the otherwise unused ID 256;
!  V5 retains its old mapping and therefore rejects byte 0. There is no separate
!  tokenizer or vocabulary file.
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
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real64, output_unit, error_unit, iostat_end, iostat_eor
   use portable_cli, only: CLI_OK, cli_parse_integer64, cli_parse_real64, cli_real64_is_finite, cli_valid_utf8
   use utf8_generation, only: utf8_generation_state_t, utf8_generation_reset, utf8_generation_allows, &
                              utf8_generation_advance, utf8_generation_complete
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
   character(len=1024) :: corpus_path, checkpoint_path, probe_path, resume_path
   character(len=MAX_PROMPT) :: prompt_text
   logical :: learning_rate_given, training_rule_given, training_profile_given, text_output_given, text_output_utf8
   integer(int64) :: seed, byte_limit, start_byte, max_updates, max_seconds, progress_every, reset_every
   integer(int32) :: chunk_tokens, max_new, requested_training_rule
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
   training_rule_given = .false.
   training_profile_given = .false.
   text_output_given = .false.
   text_output_utf8 = .false. ! Preserve the historical raw-byte decoder unless explicitly requested.
   requested_training_rule = FBC_TRAINING_RULE_V6

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
      write(output_unit, '(A)') '         [--reset-every N] [--training-rule 6|7]'
      write(output_unit, '(A)') '         [--training-profile turkish-short --max-updates 1..163]'
      write(output_unit, '(A)') '  ask    --checkpoint FILE --prompt TEXT  [--max-new N] [--temperature R]'
      write(output_unit, '(A)') '  probe  --checkpoint FILE --probe FILE   [--max-new N] [--temperature R]'
      write(output_unit, '(A)') '         [--warmup-bytes N] [--warmup-from FILE]'
      write(output_unit, '(A)') '  ask/probe: [--text-output bytes|utf8] (default bytes; utf8 constrains generation)'
      write(output_unit, '(A)') ''
      write(output_unit, '(A)') '  Bytes 1..255 map to their IDs; v6/v7 map byte 0 to ID 256 (v5 rejects it).'
      write(output_unit, '(A)') '  Fresh default shape/chunk without resets: lr 0.006; other cases: 0.003 unless --lr.'
      write(output_unit, '(A)') '  --resume uses checkpoint optimizer settings unless --lr explicitly overrides the rate.'
      write(output_unit, '(A)') '  --training-rule applies only to fresh training; resume preserves the checkpoint rule.'
      write(output_unit, '(A)') '  turkish-short is experimental, V6/default shape/chunk, no resets and fresh runs only.'
      write(output_unit, '(A)') '  It cannot be combined with --lr; its selected rate is preserved on resume.'
      write(output_unit, '(A)') '  --bytes 0 (default) consumes the whole file.'
      write(output_unit, '(A)') '  --max-updates 0 and --max-seconds 0 mean "no such bound".'
      stop 0_int32
   end subroutine print_usage_and_stop

   subroutine read_command(name)
      character(len=*), intent(out) :: name

      name = ''
      if (command_argument_count() < 1) return
      ! Apply the same length/status checks to the command as to options; an
      ! overlong first argument must not create an out-of-bounds substring.
      call get_arg(1_int32, name)
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
            call get_arg(index, corpus_path, path=.true.)
         case ('--checkpoint')
            index = index + 1_int32
            call get_arg(index, checkpoint_path, path=.true.)
         case ('--probe')
            index = index + 1_int32
            call get_arg(index, probe_path, path=.true.)
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
            config%local_window = parse_int32(key, '--local-window')
         case ('--memory-scales')
            index = index + 1_int32
            call get_arg(index, key)
            config%memory_scales = parse_int32(key, '--memory-scales')
         case ('--memory-slots')
            index = index + 1_int32
            call get_arg(index, key)
            config%memory_slots = parse_int32(key, '--memory-slots')
         case ('--train-chunk-tokens')
            index = index + 1_int32
            call get_arg(index, key)
            chunk_tokens = parse_int32(key, '--train-chunk-tokens')
         case ('--width')
            index = index + 1_int32
            call get_arg(index, key)
            config%width = parse_int32(key, '--width')
         case ('--heads')
            index = index + 1_int32
            call get_arg(index, key)
            config%heads = parse_int32(key, '--heads')
         case ('--ffn')
            index = index + 1_int32
            call get_arg(index, key)
            config%ffn_width = parse_int32(key, '--ffn')
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
            max_new = parse_int32(key, '--max-new')
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
            call get_arg(index, resume_path, path=.true.)
         case ('--training-rule')
            index = index + 1_int32
            call get_arg(index, key)
            requested_training_rule = parse_int32(key, '--training-rule')
            training_rule_given = .true.
         case ('--training-profile')
            index = index + 1_int32
            call get_arg(index, key)
            if (trim(key) /= 'turkish-short') call usage_and_stop('--training-profile must be turkish-short')
            training_profile_given = .true.
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
            call get_arg(index, warmup_path, path=.true.)
         case ('--reset-every')
            index = index + 1_int32
            call get_arg(index, key)
            reset_every = parse_int64(key, '--reset-every')
         case ('--text-output')
            index = index + 1_int32
            call get_arg(index, key)
            select case (trim(key))
            case ('bytes')
               text_output_utf8 = .false.
            case ('utf8')
               text_output_utf8 = .true.
            case default
               call usage_and_stop('--text-output must be bytes or utf8')
            end select
            text_output_given = .true.
         case default
            call usage_and_stop('unknown option: '//trim(key))
         end select
         index = index + 1_int32
      end do

      if (text_output_given) then
         if (trim(command) /= 'ask' .and. trim(command) /= 'probe') &
            call usage_and_stop('--text-output is only valid for ask or probe')
      end if
      if (training_profile_given) then
         if (trim(command) /= 'train' .or. len_trim(resume_path) > 0) &
            call usage_and_stop('--training-profile is only valid for fresh training')
         if (learning_rate_given) call usage_and_stop('--training-profile cannot be combined with --lr')
         if (requested_training_rule /= FBC_TRAINING_RULE_V6) &
            call usage_and_stop('turkish-short requires training rule 6')
         if (max_updates < 1_int64 .or. max_updates > 163_int64) &
            call usage_and_stop('turkish-short requires --max-updates in 1..163')
      end if
      if (training_rule_given) then
         if (trim(command) /= 'train') call usage_and_stop('--training-rule is only valid for fresh training')
         if (len_trim(resume_path) > 0) call usage_and_stop('--training-rule cannot change a resumed checkpoint')
         if (requested_training_rule /= FBC_TRAINING_RULE_V6 .and. &
             requested_training_rule /= FBC_TRAINING_RULE_V7) call usage_and_stop('--training-rule must be 6 or 7')
      end if
      if (max_new < 1 .or. max_new > MAX_RESPONSE) call usage_and_stop('--max-new is out of range')
      if (chunk_tokens < 1 .or. chunk_tokens > FBC_MAX_CHUNK_TOKENS) &
         call usage_and_stop('--train-chunk-tokens is out of range')
      if (warmup_bytes < 0_int64 .or. warmup_bytes > MAX_WARMUP) &
         call usage_and_stop('--warmup-bytes is out of range')
      if (reset_every < 0_int64) call usage_and_stop('--reset-every is out of range')
   end subroutine parse_options

   subroutine get_arg(index, value, path)
      integer(int32), intent(in) :: index
      character(len=*), intent(out) :: value
      logical, intent(in), optional :: path
      integer :: received, argument_status
      logical :: is_path

      value = ''
      if (index > command_argument_count()) call usage_and_stop('missing option value')
      call get_command_argument(index, value, received, argument_status)
      if (argument_status /= 0) call usage_and_stop('argument is too long or unavailable')
      if (received < 0 .or. received > len(value)) call usage_and_stop('argument is too long or unavailable')
      if (.not. cli_valid_utf8(value(1:received))) call usage_and_stop('argument is not valid UTF-8')
      is_path = .false.
      if (present(path)) is_path = path
      if (is_path .and. received > 0) then
         ! Most file APIs use TRIM, which would otherwise silently name a
         ! different file and could replace an unrelated checkpoint.
         if (value(received:received) == ' ') call usage_and_stop('file path must not end in a space')
      end if
      argument_bytes = int(received, int32)
   end subroutine get_arg

   function parse_int64(text, option) result(value)
      character(len=*), intent(in) :: text, option
      integer(int64) :: value
      integer(int32) :: parse_status

      if (len_trim(text) == 0) call usage_and_stop('empty value for '//trim(option))
      if (text(1:1) == '-' .or. text(1:1) == '+') &
         call usage_and_stop('signed value for '//trim(option))
      call cli_parse_integer64(text, value, parse_status)
      if (parse_status /= CLI_OK .or. value < 0_int64) &
         call usage_and_stop('not a non-negative integer: '//trim(option))
   end function parse_int64

   function parse_int32(text, option) result(value)
      character(len=*), intent(in) :: text, option
      integer(int32) :: value
      integer(int64) :: wide

      wide = parse_int64(text, option)
      if (wide > int(huge(value), int64)) call usage_and_stop('integer is out of range: '//trim(option))
      value = int(wide, int32)
   end function parse_int32

   function parse_real64(text, option) result(value)
      character(len=*), intent(in) :: text, option
      real(real64) :: value
      integer(int32) :: parse_status

      call cli_parse_real64(text, value, parse_status)
      if (parse_status /= CLI_OK) call usage_and_stop('not a finite non-negative number: '//trim(option))
      if (value < 0.0_real64) call usage_and_stop('not a finite non-negative number: '//trim(option))
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
      integer(int64) :: consumed, file_bytes, corpus_bytes, stopped_at, started, now, elapsed, since_reset
      integer(int64) :: tokens_before, updates_before, session_tokens, session_updates, prediction_count
      integer(int64) :: precise_start, precise_end, precise_rate, precise_max
      real(real64) :: loss, rate, session_rate, session_loss, loss_sum_before, requested_learning_rate, file_fraction
      real(real64) :: session_seconds_precise, session_rate_precise
      logical :: bounded, first_field, reached_end, default_shape, precise_time_valid

      if (len_trim(corpus) == 0) call usage_and_stop('train needs --corpus')
      if (len_trim(checkpoint) == 0) call usage_and_stop('train needs --checkpoint')
      ! Protect the obvious identical path before opening either file. Standard
      ! Fortran cannot prove that distinct names are not symlink/hardlink aliases.
      if (trim(corpus) == trim(checkpoint)) call usage_and_stop('checkpoint cannot be the training corpus')
      if (.not. fbc_config_is_valid(config)) call usage_and_stop('configuration is not valid')
      default_shape = chunk == 512_int32 .and. reset_every == 0_int64 .and. &
                      config%vocabulary_size == 256_int32 .and. config%width == 32_int32 .and. &
                      config%heads == 2_int32 .and. config%ffn_width == 64_int32 .and. &
                      config%local_window == 32_int32 .and. config%memory_scales == 4_int32 .and. &
                      config%memory_slots == 64_int32 .and. config%logical_horizon == 8192_int64
      if (training_profile_given .and. .not. default_shape) &
         call usage_and_stop('turkish-short requires the default shape, chunk 512 and no resets')

      corpus_bytes = file_size(corpus)
      file_bytes = corpus_bytes
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
         ! An explicit --lr overrides only the learning rate. Every other
         ! optimizer setting still comes from the checkpoint; replacing the
         ! entire configuration would silently change momentum, clipping,
         ! decay and epsilon when the user requested only a new rate.
         requested_learning_rate = update_config%learning_rate
         update_config = loaded_config
         if (learning_rate_given) update_config%learning_rate = requested_learning_rate
         call ensure_config_matches(model, config)
      else
         ! Preserve existing defaults. The explicitly selected short Turkish
         ! profile only changes this scalar; no per-token work is added. A
         ! resume always loads the stored rate, not the current default.
         if (.not. learning_rate_given .and. default_shape) update_config%learning_rate = 6.0e-3_real64
         if (training_profile_given) update_config%learning_rate = 1.2e-2_real64
         call fbc_model_init(model, config, seed, status)
         if (status /= FBC_OK) call usage_and_stop('model initialisation failed')
         if (training_rule_given) model%training_rule = requested_training_rule
      end if
      if (.not. fbc_update_config_is_valid(update_config)) call usage_and_stop('optimizer configuration is invalid')
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
      loss_sum_before = metrics%loss_sum
      metrics%numeric_ok = .true.
      ! Two high-resolution clock reads per session, never per token/update.
      ! Keep the legacy integer timer and --max-seconds semantics unchanged.
      call system_clock(precise_start, precise_rate, precise_max)
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
         if (io_status /= 0) then
            close(unit_number)
            write(error_unit, '(A,I0)') 'corpus read failed at byte ', consumed
            stop 1_int32
         end if
         do block_bytes = 1_int32, block_tokens + 1_int32
            tokens(block_bytes) = byte_to_token(iand(int(raw(block_bytes), int32), 255_int32), model%training_rule)
            if (tokens(block_bytes) == 0_int32) then
               close(unit_number)
               write(error_unit, '(A,I0)') 'v5 checkpoint cannot encode NUL at byte ', &
                  consumed + int(block_bytes - 1_int32, int64)
               stop 1_int32
            end if
         end do
         call fbc_train_stream(model, optimizer, state, tokens, block_tokens, block_tokens, &
                              update_config, workspace, gradient, metrics, status)
         if (status /= FBT_OK) then
            close(unit_number)
            write(error_unit, '(A,I0,A,I0)') 'training stopped at byte ', consumed, ' with status ', status
            stop 1_int32
         end if
         consumed = consumed + int(block_tokens, int64)
         since_reset = since_reset + int(block_tokens, int64)
         if (reset_every > 0_int64 .and. since_reset >= reset_every) then
            call fbc_state_reset(state, model, status)
            if (status /= FBC_OK) then
               close(unit_number)
               write(error_unit, '(A,I0)') 'stream reset failed, status ', status
               stop 1_int32
            end if
            since_reset = 0_int64
         end if
         ! Fortran does not promise short-circuit evaluation: never evaluate
         ! MOD with a zero divisor when progress output is disabled.
         if (progress_every > 0_int64) then
            if (mod(metrics%updates, progress_every) == 0_int64) then
               call wall_clock(now)
               write(output_unit, '(A,I0,A,I0,A,F0.4,A,F0.1,A,I0)') 'progress: bytes ', consumed, &
                  ' updates ', metrics%updates, ' mean_nll ', &
                  metrics%loss_sum / real(max(1_int64, metrics%tokens), real64), ' tok/s ', &
                  real(metrics%tokens, real64) / real(max(1_int64, now - started), real64), &
                  ' reach ', nint(fbc_learned_horizon(model))
               flush(output_unit)
            end if
         end if
      end do
      close(unit_number, iostat=io_status)
      if (io_status /= 0) then
         write(error_unit, '(A)') 'could not close the training corpus; checkpoint was not written'
         stop 1_int32
      end if
      call wall_clock(now)
      call system_clock(precise_end)
      elapsed = max(1_int64, now - started)
      stopped_at = consumed
      ! A wrap, unavailable clock, or zero ticks is reported as null rather
      ! than an invented rate. Init and checkpoint I/O are outside this span.
      precise_time_valid = precise_rate > 0_int64 .and. precise_max > 0_int64 .and. &
                           precise_start >= 0_int64 .and. precise_end > precise_start
      session_seconds_precise = 0.0_real64
      session_rate_precise = 0.0_real64
      if (precise_time_valid) then
         session_seconds_precise = real(precise_end - precise_start, real64) / real(precise_rate, real64)
         precise_time_valid = session_seconds_precise > 0.0_real64
      end if

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
      if (precise_time_valid) session_rate_precise = real(session_tokens, real64) / session_seconds_precise
      ! The session's own mean, not the run's.  `loss_sum` is cumulative across
      ! a resumed chain, so it has to be rebased on the sum the session started
      ! with; dividing the cumulative sum by the session's token count mixes two
      ! different intervals and, on a resumed checkpoint, reports a number that
      ! belongs to no session at all.
      session_loss = (metrics%loss_sum - loss_sum_before) / &
                     real(max(1_int64, session_tokens), real64)
      ! `end_byte` is the next overlapping input offset, not a byte count:
      ! the last of N bytes is already the target of prediction N-1. Reporting
      ! EOF only at N would therefore mark every completed pass as partial.
      ! Reaching the end is not proof that earlier resumed offsets were covered.
      prediction_count = max(0_int64, corpus_bytes - 1_int64)
      reached_end = stopped_at >= prediction_count
      file_fraction = 1.0_real64
      if (prediction_count > 0_int64) &
         file_fraction = real(min(stopped_at, prediction_count), real64) / real(prediction_count, real64)
      first_field = .true.
      write(output_unit, '(A)') 'CORPUS_TRAIN_JSON {'
      call field_text('corpus', trim(corpus), first_field)
      call field_int('corpus_bytes', corpus_bytes, first_field)
      call field_int('start_byte', start_byte, first_field)
      call field_int('end_byte', stopped_at, first_field)
      call field_int('consumed_bytes', stopped_at - start_byte, first_field)
      call field_real('file_fraction', file_fraction, first_field)
      call field_bool('bounded_by_bytes', bounded, first_field)
      call field_bool('whole_file', reached_end, first_field)
      call field_int('updates', metrics%updates, first_field)
      call field_int('tokens', metrics%tokens, first_field)
      call field_int('session_updates', session_updates, first_field)
      call field_int('session_tokens', session_tokens, first_field)
      call field_real_full('learning_rate', update_config%learning_rate, first_field)
      call field_bool('training_profile_applied', training_profile_given, first_field)
      call field_real('mean_nll', loss, first_field)
      call field_real('session_mean_nll', session_loss, first_field)
      call field_real('tokens_per_second', session_rate, first_field)
      call field_int('seconds', elapsed, first_field)
      if (precise_time_valid) then
         call field_real_full('session_seconds_precise', session_seconds_precise, first_field)
         call field_real_full('session_tokens_per_second_precise', session_rate_precise, first_field)
      else
         call json_member('session_seconds_precise', 'null', first_field)
         call json_member('session_tokens_per_second_precise', 'null', first_field)
      end if
      call field_real('learned_reach', fbc_learned_horizon(model), first_field)
      call field_int('parameters', fbc_parameter_count(config), first_field)
      call field_int('training_rule', int(model%training_rule, int64), first_field)
      call field_int('reset_every', reset_every, first_field)
      call field_bool('numeric_ok', metrics%numeric_ok, first_field)
      write(output_unit, '(A)') ''
      write(output_unit, '(A)') '}'
      if (.not. reached_end) then
         write(output_unit, '(A,I0,A,I0)') 'note: partial pass: reached prediction offset ', stopped_at, ' of ', &
            prediction_count
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
      if (prompt_bytes > MAX_PROMPT) call usage_and_stop('prompt is too long')
      if (.not. cli_valid_utf8(prompt(1_int32:prompt_bytes))) call usage_and_stop('prompt is not valid UTF-8')

      call load_model(checkpoint, model, optimizer, loaded_config, counters)
      call fbc_state_init(state, model, status)
      if (status /= FBC_OK) call usage_and_stop('stream state allocation failed')
      call fbc_workspace_init(workspace, model, status)
      if (status /= FBC_OK) call usage_and_stop('workspace allocation failed')
      call load_warmup(trim(warmup_path), trim(corpus_path), warmup_bytes)

      random_state = initialize_random_state(seed)
      call respond(model, state, workspace, prompt, prompt_bytes, max_new, temperature, &
                   random_state, response, length)

      first_field = .true.
      write(output_unit, '(A)') 'PROMPT_RESPONSE_JSON {'
      call field_text('prompt', prompt(1_int32:prompt_bytes), first_field)
      ! A byte model may emit incomplete or invalid UTF-8. Keep the JSON valid
      ! and expose lossless bytes instead of silently inventing replacement text.
      call field_text('response', response(1_int32:length), first_field)
      call field_bool('response_valid_utf8', cli_valid_utf8(response(1_int32:length)), first_field)
      call field_text('response_bytes_hex', bytes_hex(response(1_int32:length)), first_field)
      if (text_output_utf8) then
         call field_text('text_output', 'utf8', first_field)
      else
         call field_text('text_output', 'bytes', first_field)
      end if
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
      integer(int32) :: status, unit_number, io_status, length, rows, line_length
      integer(int32) :: fact_total, fact_hit, copy_total, copy_hit, next_total, next_hit
      integer(int32) :: fact_bytes, fact_byte_hits, copy_bytes, copy_byte_hits
      integer(int32) :: next_bytes, next_byte_hits
      integer(int32) :: valid_response_rows, empty_response_rows
      real(real64) :: fact_nll, copy_nll, next_nll
      integer(int64) :: random_state
      logical :: first_field, probe_truncated
      character(len=MAX_RESPONSE) :: scoring
      integer(int32) :: scoring_bytes, scoring_hits, prompt_length, expected_length
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
      valid_response_rows = 0_int32
      empty_response_rows = 0_int32
      probe_truncated = .false.
      random_state = initialize_random_state(seed)
      write(output_unit, '(A)') 'PROBE_EXAMPLES ['
      do
         ! Non-advancing read exposes the real record length and rejects an
         ! overlong line instead of silently discarding its remaining bytes.
         read(unit_number, '(A)', advance='no', size=line_length, iostat=io_status) line
         if (io_status == iostat_end) exit
         if (io_status == 0) call usage_and_stop('probe line is too long')
         if (io_status /= iostat_eor) call usage_and_stop('probe file read failed')
         if (len_trim(line(1_int32:line_length)) == 0) cycle
         if (line(1:1) == '#') cycle
         if (rows >= MAX_PROBE_ROWS) then
            probe_truncated = .true.
            exit
         end if
         call split_row(line(1_int32:line_length), kind, prompt, expected, prompt_length, expected_length, status)
         if (status /= 0_int32) call usage_and_stop('malformed or overlong probe row')
         if (.not. cli_valid_utf8(prompt(1_int32:prompt_length)) .or. &
             .not. cli_valid_utf8(expected(1_int32:expected_length))) &
            call usage_and_stop('probe row is not valid UTF-8')
         rows = rows + 1_int32
         call respond(model, state, workspace, prompt, prompt_length, max_new, temperature, &
                      random_state, response, length)
         if (cli_valid_utf8(response(1_int32:length))) valid_response_rows = valid_response_rows + 1_int32
         if (length == 0_int32) empty_response_rows = empty_response_rows + 1_int32
         ! Exact match on a greedy continuation is the headline number, but a
         ! tiny model that emits one wrong byte scores zero there while clearly
         ! ranking the right bytes above the rest.  The teacher-forced figures
         ! below say how often the *next byte* was right and how surprised the
         ! model was by the expected continuation, so the probe is informative
         ! even when the exact-match rate is zero.
         call score_continuation(model, state, workspace, prompt, prompt_length, expected, expected_length, &
                                 scoring, scoring_bytes, scoring_hits, scoring_nll)
         select case (trim(kind))
         case ('fact')
            fact_total = fact_total + 1_int32
            if (length == expected_length) then
               if (response(1_int32:length) == expected(1_int32:expected_length)) fact_hit = fact_hit + 1_int32
            end if
            fact_bytes = fact_bytes + scoring_bytes
            fact_byte_hits = fact_byte_hits + scoring_hits
            call add_probe_loss(fact_nll, scoring_nll)
         case ('copy')
            copy_total = copy_total + 1_int32
            if (length == expected_length) then
               if (response(1_int32:length) == expected(1_int32:expected_length)) copy_hit = copy_hit + 1_int32
            end if
            copy_bytes = copy_bytes + scoring_bytes
            copy_byte_hits = copy_byte_hits + scoring_hits
            call add_probe_loss(copy_nll, scoring_nll)
         case ('next')
            next_total = next_total + 1_int32
            if (length == expected_length) then
               if (response(1_int32:length) == expected(1_int32:expected_length)) next_hit = next_hit + 1_int32
            end if
            next_bytes = next_bytes + scoring_bytes
            next_byte_hits = next_byte_hits + scoring_hits
            call add_probe_loss(next_nll, scoring_nll)
         end select
         if (rows <= 8_int32) then
            if (rows > 1_int32) write(output_unit, '(A)') ','
            write(output_unit, '(A)', advance='no') '  {"kind": '//json_string_or_null(trim(kind))// &
               ', "prompt": '//json_string_or_null(prompt(1_int32:prompt_length))// &
               ', "expected": '//json_string_or_null(expected(1_int32:expected_length))// &
               ', "response": '//json_string_or_null(response(1_int32:length))// &
               ', "response_bytes_hex": "'//bytes_hex(response(1_int32:length))//'"'// &
               ', "greedy": '//json_string_or_null(scoring(1_int32:min(scoring_bytes, len(scoring))))// &
               ', "greedy_bytes_hex": "'//bytes_hex(scoring(1_int32:min(scoring_bytes, len(scoring))))//'"}'
         end if
      end do
      close(unit_number)
      write(output_unit, '(A)') ''
      write(output_unit, '(A)') ']'

      first_field = .true.
      write(output_unit, '(A)') 'PROBE_JSON {'
      call field_text('probe_file', trim(probe_file), first_field)
      if (text_output_utf8) then
         call field_text('text_output', 'utf8', first_field)
      else
         call field_text('text_output', 'bytes', first_field)
      end if
      call field_int('warmup_bytes', int(warmup_count, int64), first_field)
      call field_int('rows', int(rows, int64), first_field)
      call field_int('valid_utf8_responses', int(valid_response_rows, int64), first_field)
      call field_int('invalid_utf8_responses', int(rows - valid_response_rows, int64), first_field)
      call field_int('empty_responses', int(empty_response_rows, int64), first_field)
      call field_bool('truncated', probe_truncated, first_field)
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
      type(utf8_generation_state_t) :: text_state
      logical :: accepted

      allocate(logits(model%config%vocabulary_size), hidden(model%config%width))
      call fbc_state_reset(state, model, status)
      call utf8_generation_reset(text_state)
      response = ''
      length = 0_int32

      if (warmup_count == 0_int32 .and. prompt_length <= 0_int32) &
         call usage_and_stop('a prompt or warm-up byte is required')
      do index = 1_int32, warmup_count
         call fbc_step(model, state, workspace, byte_to_token(warmup_tokens(index), model%training_rule), &
                       hidden, logits, status)
         if (status /= FBC_OK) call usage_and_stop('warm-up byte is unsupported by this checkpoint')
      end do

      ! The caller's byte count, not len(): the dummy is bound to a fixed-length
      ! buffer, so len() would stream the padding, and len_trim() would silently
      ! drop the trailing space that every frame in this corpus ends with.
      do index = 1_int32, min(prompt_length, len(prompt))
         call fbc_step(model, state, workspace, byte_of(prompt(index:index), model%training_rule), hidden, logits, status)
         if (status /= FBC_OK) call usage_and_stop('prompt byte is unsupported by this checkpoint')
      end do

      produced = 0_int32
      do while (produced < max_new)
         if (text_output_utf8) then
            if (temperature <= 0.0_real64) then
               token = utf8_argmax_token(logits, text_state, max_new - produced)
            else
               token = utf8_sample_token(logits, temperature, random_state, text_state, max_new - produced)
            end if
         else if (temperature <= 0.0_real64) then
            token = argmax_token(logits)
         else
            token = sample_token(logits, temperature, random_state)
         end if
         if (token == 0_int32) call usage_and_stop('non-finite logits or no permitted byte cannot be decoded')
         ! Stop at a full stop or a line break: the corpus frames every response
         ! that way, so a response is directly comparable with its label.
         if (token == DOT .or. token == NEWLINE) exit
         ! Token 256 represents NUL in v6 and was unassigned in v5; neither is
         ! a printable response byte. Never inject it into JSON output.
         if (token < 32_int32 .or. token == 256_int32) exit
         if (text_output_utf8) then
            call utf8_generation_advance(text_state, token, max_new - produced, accepted)
            if (.not. accepted) call usage_and_stop('invalid UTF-8 decoding state')
         end if
         if (length < len(response)) then
            length = length + 1_int32
            response(length:length) = byte_character(token)
         end if
         call fbc_step(model, state, workspace, token, hidden, logits, status)
         if (status /= FBC_OK) call usage_and_stop('response decoding failed')
         produced = produced + 1_int32
      end do
      if (text_output_utf8) then
         if (.not. utf8_generation_complete(text_state)) call usage_and_stop('incomplete UTF-8 decoding state')
         if (.not. cli_valid_utf8(response(1_int32:length))) call usage_and_stop('invalid UTF-8 response')
      end if
      deallocate(logits, hidden)
   end subroutine respond

   !> Legacy tokens 1..255 retain their byte IDs. V6 and V7 assign the
   !> previously unused ID 256 to NUL without changing any other byte.
   pure integer(int32) function byte_to_token(byte, rule) result(token)
      integer(int32), intent(in) :: byte, rule
      token = byte
      if (byte == 0_int32 .and. &
          (rule == FBC_TRAINING_RULE_V6 .or. rule == FBC_TRAINING_RULE_V7)) token = 256_int32
   end function byte_to_token

   !> Raw octets include UTF-8 continuation bytes 0x80..0xBF.
   pure function byte_of(text, rule) result(token)
      character(len=*), intent(in) :: text
      integer(int32), intent(in) :: rule
      integer(int32) :: token
      integer(int8) :: signed

      signed = transfer(text(1:1), signed)
      token = byte_to_token(iand(int(signed, int32), 255_int32), rule)
   end function byte_of

   !> The inverse byte mapping; 256 decodes to NUL. Avoid narrowing an
   !> out-of-range positive integer to int8 on processors with signed int8.
   pure function byte_character(token) result(text)
      integer(int32), intent(in) :: token
      character(len=1) :: text
      integer(int8) :: signed
      integer(int32) :: byte

      byte = iand(token, 255_int32)
      if (byte >= 128_int32) byte = byte - 256_int32
      signed = int(byte, int8)
      text = transfer(signed, text)
   end function byte_character

   !> Teacher-forced score of one expected response given the prompt.
   !>
   !> The prompt is fed first, so the logits after its last byte are the model's
   !> prediction for the first expected byte.  Then each expected byte is fed
   !> back (teacher forcing) and the next prediction is scored, so the outputs
   !> are: the greedy byte at each teacher-forced position, how many bytes it
   !> ranked first, and the total negative log-likelihood of the continuation.
   !> This teacher-forced greedy sequence is not a free-running completion;
   !> the separately sampled `response` is used for the exact-match number.
   subroutine score_continuation(model, state, workspace, prompt, prompt_length, expected, expected_length, greedy, &
                                 byte_total, byte_hits, nll)
      type(fbc_model_t), intent(in) :: model
      type(fbc_state_t), intent(inout) :: state
      type(fbc_workspace_t), intent(inout) :: workspace
      character(len=*), intent(in) :: prompt, expected
      integer(int32), intent(in) :: prompt_length, expected_length
      character(len=*), intent(out) :: greedy
      integer(int32), intent(out) :: byte_total, byte_hits
      real(real64), intent(out) :: nll
      integer(int32) :: status, index, target, produced, guess
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
      if (warmup_count == 0_int32 .and. prompt_length <= 0_int32) &
         call usage_and_stop('a prompt or warm-up byte is required')
      do index = 1_int32, warmup_count
         call fbc_step(model, state, workspace, byte_to_token(warmup_tokens(index), model%training_rule), &
                       hidden, logits, status)
         if (status /= FBC_OK) call usage_and_stop('warm-up byte is unsupported by this checkpoint')
      end do
      do index = 1_int32, min(prompt_length, len(prompt))
         call fbc_step(model, state, workspace, byte_of(prompt(index:index), model%training_rule), hidden, logits, status)
         if (status /= FBC_OK) call usage_and_stop('prompt byte is unsupported by this checkpoint')
      end do
      do index = 1_int32, expected_length
         target = byte_of(expected(index:index), model%training_rule)
         call fbc_loss_and_gradient(logits, target, dlogits, loss, status)
         if (status /= FBC_OK) call usage_and_stop('continuation byte is unsupported by this checkpoint')
         call add_probe_loss(nll, loss)
         byte_total = byte_total + 1_int32
         guess = argmax_token(logits)
         if (guess == 0_int32) call usage_and_stop('non-finite logits cannot be decoded')
         if (guess == target) byte_hits = byte_hits + 1_int32
         if (produced < len(greedy)) then
            produced = produced + 1_int32
            greedy(produced:produced) = byte_character(guess)
         end if
         call fbc_step(model, state, workspace, target, hidden, logits, status)
         if (status /= FBC_OK) call usage_and_stop('continuation scoring failed')
      end do
      deallocate(logits, hidden, dlogits)
   end subroutine score_continuation

   ! Only probe aggregation uses this scalar guard; no extra training scans or
   ! updates are introduced. A finite per-byte loss can still overflow its sum.
   subroutine add_probe_loss(total, value)
      real(real64), intent(inout) :: total
      real(real64), intent(in) :: value

      if (.not. cli_real64_is_finite(total) .or. .not. cli_real64_is_finite(value)) &
         call usage_and_stop('probe loss is not finite')
      if (total < 0.0_real64 .or. value < 0.0_real64) call usage_and_stop('probe loss is negative')
      if (total > 0.0_real64 .and. value > 0.0_real64) then
         ! Conservatively reject the rounded boundary as well; rounding of
         ! huge-value must not permit an overflow in the subsequent addition.
         if (total >= huge(total) - value) call usage_and_stop('probe loss sum is not representable')
      end if
      total = total + value
   end subroutine add_probe_loss

   pure function argmax_token(logits) result(token)
      real(real64), intent(in) :: logits(:)
      integer(int32) :: token, index

      ! Return the one-based token index, not index minus one. Bytes 1..255
      ! retain their IDs; v6 assigns ID 256 to byte 0. Conversion to a byte
      ! occurs only at the input/output boundary.
      token = 0_int32
      do index = 1_int32, size(logits)
         if (.not. cli_real64_is_finite(logits(index))) then
            token = 0_int32
            return
         end if
         if (token == 0_int32) then
            token = index
         else if (logits(index) > logits(token)) then
            token = index
         end if
      end do
   end function argmax_token

   !> Select only bytes that keep the output a complete UTF-8 prefix. Inspect
   !> every logit for finiteness, including masked entries, just as raw decoding
   !> does. Ties use the same lowest-index rule as argmax_token.
   pure function utf8_argmax_token(logits, text_state, bytes_left) result(token)
      real(real64), intent(in) :: logits(:)
      type(utf8_generation_state_t), intent(in) :: text_state
      integer(int32), intent(in) :: bytes_left
      integer(int32) :: token, index

      token = 0_int32
      do index = 1_int32, size(logits)
         if (.not. cli_real64_is_finite(logits(index))) then
            token = 0_int32
            return
         end if
         if (.not. utf8_generation_allows(text_state, index, bytes_left)) cycle
         if (token == 0_int32) then
            token = index
         else if (logits(index) > logits(token)) then
            token = index
         end if
      end do
   end function utf8_argmax_token

   !> Renormalize the original categorical distribution over legal bytes; this
   !> affects free generation only. Teacher-forced NLL remains unmasked.
   function utf8_sample_token(logits, temperature, random_state, text_state, bytes_left) result(token)
      real(real64), intent(in) :: logits(:), temperature
      integer(int64), intent(inout) :: random_state
      type(utf8_generation_state_t), intent(in) :: text_state
      integer(int32), intent(in) :: bytes_left
      integer(int32) :: token, index, top
      real(real64) :: maximum, total, draw, threshold

      token = 0_int32
      top = utf8_argmax_token(logits, text_state, bytes_left)
      if (top == 0_int32) return
      maximum = logits(top)
      total = 0.0_real64
      do index = 1_int32, size(logits)
         if (.not. utf8_generation_allows(text_state, index, bytes_left)) cycle
         total = total + sampling_weight(logits(index), maximum, temperature)
      end do
      draw = next_uniform(random_state) * total
      threshold = 0.0_real64
      ! `top` is a legal fallback even if rounding leaves draw at the end.
      token = top
      do index = 1_int32, size(logits)
         if (.not. utf8_generation_allows(text_state, index, bytes_left)) cycle
         threshold = threshold + sampling_weight(logits(index), maximum, temperature)
         if (draw < threshold) then
            token = index
            exit
         end if
      end do
   end function utf8_sample_token

   !> Temperature sampling over the logits, from a 64-bit xorshift stream.
   function sample_token(logits, temperature, random_state) result(token)
      real(real64), intent(in) :: logits(:)
      real(real64), intent(in) :: temperature
      integer(int64), intent(inout) :: random_state
      integer(int32) :: token, index, top
      real(real64) :: maximum, total, draw, threshold

      top = argmax_token(logits)
      if (top == 0_int32) call usage_and_stop('non-finite logits cannot be sampled')
      maximum = logits(top)
      total = 0.0_real64
      do index = 1_int32, size(logits)
         total = total + sampling_weight(logits(index), maximum, temperature)
      end do
      draw = next_uniform(random_state) * total
      threshold = 0.0_real64
      ! Strict comparison avoids selecting a zero-weight prefix if draw is 0.
      token = int(size(logits), int32)
      do index = 1_int32, size(logits)
         threshold = threshold + sampling_weight(logits(index), maximum, temperature)
         if (draw < threshold) then
            token = index
            exit
         end if
      end do
   end function sample_token

   !> Compute exp((logit-maximum)/temperature) without overflowing a finite
   !> gap or a tiny-temperature division. Values below the normal range are
   !> deliberately truncated to zero, avoiding underflow on different real64
   !> models (this affects only probabilities smaller than tiny(real64)).
   pure real(real64) function sampling_weight(logit, maximum, temperature) result(weight)
      real(real64), intent(in) :: logit, maximum, temperature
      real(real64), parameter :: cutoff = -log(tiny(0.0_real64))
      real(real64) :: gap, ratio
      logical :: wide

      wide = .false.
      if (maximum > 0.0_real64 .and. logit < 0.0_real64) &
         wide = maximum >= huge(0.0_real64) + logit
      if (wide) then
         gap = 0.5_real64 * maximum - 0.5_real64 * logit
         if (temperature < 1.0_real64) then
            if (gap > (0.5_real64 * cutoff) * temperature) then
               weight = 0.0_real64
               return
            end if
         end if
         ratio = gap / temperature
         if (ratio > 0.5_real64 * cutoff) then
            weight = 0.0_real64
         else
            weight = exp(-2.0_real64 * ratio)
         end if
      else
         gap = maximum - logit
         if (temperature < 1.0_real64) then
            if (gap > cutoff * temperature) then
               weight = 0.0_real64
               return
            end if
         end if
         ratio = gap / temperature
         if (ratio > cutoff) then
            weight = 0.0_real64
         else
            weight = exp(-ratio)
         end if
      end if
   end function sampling_weight

   !> Preserve the historical modulo-2**64 LCG seed mixing without signed
   !> overflow. This runs once per ask/probe, not in the training loop.
   pure integer(int64) function initialize_random_state(seed) result(state)
      integer(int64), intent(in) :: seed
      integer(int64), parameter :: multiplier = 6364136223846793005_int64
      integer(int64), parameter :: increment = 1442695040888963407_int64
      integer(int64) :: term
      integer :: bit

      state = 0_int64
      term = seed
      do bit = 0, bit_size(seed) - 1
         if (btest(multiplier, bit)) state = add_modulo_bits(state, term)
         term = shiftl(term, 1)
      end do
      state = add_modulo_bits(state, increment)
   end function initialize_random_state

   pure integer(int64) function add_modulo_bits(left, right) result(sum_bits)
      integer(int64), intent(in) :: left, right
      integer(int64) :: carry, next_carry

      sum_bits = left
      carry = right
      do while (carry /= 0_int64)
         next_carry = shiftl(iand(sum_bits, carry), 1)
         sum_bits = ieor(sum_bits, carry)
         carry = next_carry
      end do
   end function add_modulo_bits

   function next_uniform(random_state) result(value)
      integer(int64), intent(inout) :: random_state
      real(real64) :: value

      random_state = ieor(random_state, shiftl(random_state, 13_int64))
      random_state = ieor(random_state, shiftr(random_state, 7_int64))
      random_state = ieor(random_state, shiftl(random_state, 17_int64))
      ! abs(min_int64) is unrepresentable. Its low 31 bits are zero in the
      ! two's-complement stream, so handle that point without signed overflow.
      if (random_state < -huge(random_state)) then
         value = 0.0_real64
      else
         value = real(iand(abs(random_state), int(z'7FFFFFFF', int64)), real64) / 2147483648.0_real64
      end if
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
   subroutine json_real_text(value, text, full_precision)
      real(real64), intent(in) :: value
      character(len=*), intent(out) :: text
      logical, intent(in), optional :: full_precision
      character(len=64) :: scratch
      integer :: io_status
      logical :: precise

      precise = .false.
      if (present(full_precision)) precise = full_precision
      if (.not. cli_real64_is_finite(value)) then
         text = 'null'
         return
      end if
      ! Fixed decimal F0.6 can exceed even a 64-character record for large
      ! finite numbers and rounds tiny nonzero progress to zero. Scientific
      ! notation is compact, valid JSON, and covers the binary64 exponent range.
      if (precise .or. abs(value) >= 1.0e22_real64 .or. (value /= 0.0_real64 .and. abs(value) < 1.0e-6_real64)) then
         write(scratch, '(ES24.16E3)', iostat=io_status) value
         if (io_status /= 0) then
            text = 'null'
         else if (index(scratch, '*') /= 0) then
            text = 'null'
         else
            text = trim(adjustl(scratch))
         end if
         return
      end if
      write(scratch, '(F0.6)', iostat=io_status) value
      if (io_status /= 0) then
         text = 'null'
         return
      end if
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

   ! Full precision avoids rounding a selected/checkpoint-loaded learning rate
   ! or a measured session duration to six decimal places.
   subroutine field_real_full(name, value, first)
      character(len=*), intent(in) :: name
      real(real64), intent(in) :: value
      logical, intent(inout) :: first
      character(len=32) :: text

      call json_real_text(value, text, full_precision=.true.)
      call json_member(name, trim(text), first)
   end subroutine field_real_full

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
      character(len=256) :: text
      character(len=32) :: score_text, byte_score_text, nll_text

      call json_real_text(score(hits, total), score_text)
      call json_real_text(score(byte_hits, byte_total), byte_score_text)
      if (byte_total > 0_int32) then
         ! Preserve binary64-scale differences in evaluation results instead of
         ! rounding ordinary NLLs to six decimal places. JSON numbers may use E.
         call json_real_text(nll_total / real(byte_total, real64), nll_text, full_precision=.true.)
      else
         nll_text = 'null'
      end if
      text = '{"correct": '//int_text(hits)//', "accuracy": '//trim(score_text)// &
             ', "n": '//int_text(total)// &
             ', "correct_bytes": '//int_text(byte_hits)// &
             ', "total_bytes": '//int_text(byte_total)// &
             ', "byte_accuracy": '//trim(byte_score_text)// &
             ', "mean_nll": '//trim(nll_text)//'}'
      call json_member(name, trim(text), first)
   end subroutine field_family

   subroutine field_text(name, value, first)
      character(len=*), intent(in) :: name, value
      logical, intent(inout) :: first

      call json_member(name, json_string_or_null(value), first)
   end subroutine field_text

   !> JSON strings must be UTF-8; invalid model bytes are represented by null
   !> and accompanying hexadecimal fields, never by lossy replacement text.
   pure function json_string_or_null(text) result(rendered)
      character(len=*), intent(in) :: text
      character(len=:), allocatable :: rendered

      if (cli_valid_utf8(text)) then
         rendered = '"'//json_escape(text)//'"'
      else
         rendered = 'null'
      end if
   end function json_string_or_null

   pure function json_escape(text) result(escaped)
      character(len=*), intent(in) :: text
      character(len=:), allocatable :: escaped
      character(len=6 * len(text)) :: buffer
      character(len=*), parameter :: hex = '0123456789abcdef'
      integer(int32) :: index, out, byte
      character(len=1) :: ch

      out = 0_int32
      do index = 1_int32, len(text)
         ch = text(index:index)
         select case (ch)
         case ('"')
            buffer(out+1:out+2) = '\"'; out = out + 2_int32
         case ('\')
            buffer(out+1:out+2) = '\\'; out = out + 2_int32
         case default
            byte = byte_of(ch, FBC_TRAINING_RULE_V5)
            if (byte < 32_int32) then
               buffer(out+1:out+4) = '\u00'
               buffer(out+5:out+5) = hex(byte/16_int32+1_int32:byte/16_int32+1_int32)
               buffer(out+6:out+6) = hex(modulo(byte,16_int32)+1_int32:modulo(byte,16_int32)+1_int32)
               out = out + 6_int32
            else
               out = out + 1_int32; buffer(out:out) = ch
            end if
         end select
      end do
      escaped = buffer(1:out)
   end function json_escape

   pure function bytes_hex(text) result(encoded)
      character(len=*), intent(in) :: text
      character(len=2 * len(text)) :: encoded
      character(len=*), parameter :: hex = '0123456789abcdef'
      integer(int32) :: index, byte

      do index = 1_int32, len(text)
         byte = byte_of(text(index:index), FBC_TRAINING_RULE_V5)
         encoded(2*index-1:2*index-1) = hex(byte/16_int32+1_int32:byte/16_int32+1_int32)
         encoded(2*index:2*index) = hex(modulo(byte,16_int32)+1_int32:modulo(byte,16_int32)+1_int32)
      end do
   end function bytes_hex

   !> `kind<TAB>prompt<TAB>expected`; exactly three non-empty fields, without
   !> silently truncating the line or discarding meaningful trailing spaces.
   subroutine split_row(line, kind, prompt, expected, prompt_length, expected_length, status)
      character(len=*), intent(in) :: line
      character(len=*), intent(out) :: kind, prompt, expected
      integer(int32), intent(out) :: prompt_length, expected_length, status
      integer(int32) :: first, second

      kind = ''
      prompt = ''
      expected = ''
      prompt_length = 0_int32
      expected_length = 0_int32
      status = 1_int32
      first = index(line, achar(9))
      if (first <= 1_int32) return
      second = index(line(first + 1_int32:), achar(9))
      if (second <= 1_int32) return
      second = second + first
      if (second >= len(line)) return
      if (index(line(second+1_int32:), achar(9)) /= 0_int32) return
      if (first-1_int32 > len(kind) .or. second-first-1_int32 > len(prompt) .or. &
          len(line)-second > len(expected)) return
      kind = line(1_int32:first - 1_int32)
      select case (trim(kind))
      case ('fact', 'copy', 'next')
         continue
      case default
         return
      end select
      prompt_length = second - first - 1_int32
      expected_length = len(line) - second
      prompt = line(first + 1_int32:second - 1_int32)
      expected = line(second + 1_int32:)
      status = 0_int32
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
      ! Fortran does not promise short-circuit evaluation: SIZE is undefined
      ! after a failed inquiry, so never inspect it until IOSTAT succeeded.
      if (status /= 0) call usage_and_stop('cannot stat the corpus: '//trim(path))
      if (size_field < 0_int64) call usage_and_stop('cannot stat the corpus: '//trim(path))
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
