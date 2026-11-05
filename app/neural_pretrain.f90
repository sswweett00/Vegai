! SPDX-License-Identifier: MIT
! Native teacher-forced pretraining utility for the neural retention decoder.
program vegai_neural_pretrain
   use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64, output_unit, error_unit
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use, intrinsic :: ieee_exceptions, only: ieee_status_type, ieee_get_status, ieee_set_status, &
      ieee_support_halting, ieee_set_halting_mode, ieee_set_flag, ieee_get_flag, &
      ieee_overflow, ieee_underflow, ieee_invalid
   use subword_bpe
   use subword_bpe_training
   use neural_retention_decoder
   use neural_pretraining
   use calibrated_decoder
   implicit none

   character(len=32) :: command
   integer :: argument_count

   ! Host-associated CLI options. Defaults retain the model shape while using
   ! NOVA and the warmup/inverse-square-root schedule for new runs. A loaded
   ! checkpoint retains its stored config unless CLI options explicitly override it.
   integer(int32) :: opt_width = 64_int32
   integer(int32) :: opt_heads = 4_int32
   integer(int32) :: opt_window = 32_int32
   integer(int32) :: opt_ffn = 256_int32
   integer(int64) :: opt_seed = 20261025_int64
   logical :: opt_lr_set = .false.
   logical :: opt_width_set = .false., opt_heads_set = .false., opt_window_set = .false., opt_ffn_set = .false.
   logical :: opt_max_tokens_set = .false., opt_accumulation_set = .false.
   logical :: opt_optimizer_set = .false., opt_schedule_set = .false.
   logical :: opt_warmup_set = .false., opt_total_set = .false.
   logical :: opt_beta1_set = .false., opt_beta2_set = .false., opt_adam_eps_set = .false.
   logical :: opt_grad_clip_set = .false., opt_embedding_limit_set = .false.
   logical :: opt_label_smoothing_set = .false., opt_min_lr_ratio_set = .false.
   real(real64) :: opt_lr = 0.0_real64
   real(real64) :: opt_weight_decay = 0.01_real64
   logical :: opt_weight_decay_set = .false.
   real(real64) :: opt_beta1 = 0.9_real64
   real(real64) :: opt_beta2 = 0.999_real64
   real(real64) :: opt_adam_eps = 1.0e-8_real64
   real(real64) :: opt_grad_clip = 4.0_real64
   real(real64) :: opt_embedding_limit = 8.0_real64
   real(real64) :: opt_label_smoothing = 0.0_real64
   real(real64) :: opt_min_lr_ratio = 0.1_real64
   integer(int32) :: opt_optimizer = NPT_OPTIMIZER_NOVA
   integer(int32) :: opt_schedule = NPT_SCHEDULE_WARMUP_INVSQRT
   integer(int32) :: opt_max_tokens_per_record = 8192_int32
   ! Mini-batch accumulation for the tied-output update.  1 is the exact
   ! one-step-per-token path; larger values average that many per-token
   ! gradients into one optimizer step (see neural_pretraining).
   integer(int32) :: opt_gradient_accumulation = 1_int32
   integer(int64) :: opt_warmup_updates = 1000_int64
   integer(int64) :: opt_total_updates = 0_int64

   argument_count = command_argument_count()
   if (argument_count < 1) call usage_and_stop()
   call get_positional_argument(1, command)
   select case (trim(command))
   case ('smoke')
      call run_smoke()
   case ('tokenizer-train')
      call run_tokenizer_train(.false.)
   case ('tokenizer-train-fast')
      call run_tokenizer_train(.true.)
   case ('train')
      call run_train(.false.)
   case ('resume')
      call run_train(.true.)
   case ('init-checkpoint')
      call run_init_checkpoint()
   case ('evaluate')
      call run_evaluate()
   case ('generate')
      call run_generate()
   case ('train-chunk')
      call run_train_chunk()
   case ('plan')
      call run_training_plan()
   case ('help', '--help', '-h')
      call print_help()
   case default
      call usage_and_stop()
   end select

contains

   ! GET_COMMAND_ARGUMENT otherwise truncates silently when STATUS is ignored.
   ! Paths are trimmed by native file APIs, so reject a trailing blank rather
   ! than accidentally opening or replacing a different file.
   subroutine get_positional_argument(index, value, is_path, actual_length)
      integer, intent(in) :: index
      character(len=*), intent(out) :: value
      logical, intent(in), optional :: is_path
      integer, intent(out), optional :: actual_length
      integer :: argument_length, argument_status

      call get_command_argument(index, value, length=argument_length, status=argument_status)
      if (argument_status /= 0 .or. argument_length > len(value)) then
         write(error_unit, '(A,I0)') 'vegai_neural_pretrain: missing or truncated argument ', index
         error stop 2
      end if
      if (present(is_path)) then
         if (is_path) then
            if (argument_length == 0 .or. len_trim(value) /= argument_length) then
               write(error_unit, '(A,I0)') 'vegai_neural_pretrain: empty or trailing-blank path argument ', index
               error stop 2
            end if
         end if
      end if
      if (present(actual_length)) actual_length = argument_length
   end subroutine get_positional_argument

   subroutine reject_matching_paths(first, second)
      character(len=*), intent(in) :: first, second

      ! Exact spelling only: portable Fortran cannot establish inode identity
      ! or resolve all symlinks/hardlinks. Operators must also avoid aliases.
      if (trim(first) == trim(second)) then
         write(error_unit, '(A)') 'vegai_neural_pretrain: input and output paths must be distinct'
         error stop 2
      end if
   end subroutine reject_matching_paths

   ! Reject list-directed separators, repetition and slash before conversion;
   ! otherwise e.g. "16,17" reads as 16 and "/" may leave an old value intact.
   subroutine parse_integer64(raw, parsed, parse_status)
      character(len=*), intent(in) :: raw
      integer(int64), intent(out) :: parsed
      integer(int32), intent(out) :: parse_status
      character(len=len(raw)) :: token
      integer :: position, last

      parsed = 0_int64
      parse_status = 1_int32
      token = adjustl(raw)
      last = len_trim(token)
      if (last == 0) return
      position = 1
      if (token(position:position) == '+' .or. token(position:position) == '-') position = position + 1
      if (position > last) return
      do while (position <= last)
         if (token(position:position) < '0' .or. token(position:position) > '9') return
         position = position + 1
      end do
      read(token, *, iostat=parse_status) parsed
      if (parse_status /= 0_int32) parsed = 0_int64
   end subroutine parse_integer64

   subroutine parse_integer32(raw, parsed, parse_status)
      character(len=*), intent(in) :: raw
      integer(int32), intent(out) :: parsed
      integer(int32), intent(out) :: parse_status
      integer(int64) :: wide

      parsed = 0_int32
      call parse_integer64(raw, wide, parse_status)
      if (parse_status /= 0_int32) return
      if (wide < -int(huge(0_int32), int64) - 1_int64 .or. wide > int(huge(0_int32), int64)) then
         parse_status = 1_int32
         return
      end if
      parsed = int(wide, int32)
   end subroutine parse_integer32

   ! Parse one decimal option without letting a library conversion of 1e309
   ! trap before IOSTAT can report an error. Parsing is CLI-only, not per token.
   ! The caller's IEEE modes and flags are restored even when conversion fails.
   subroutine parse_real_option(raw, parsed, parse_status)
      character(len=*), intent(in) :: raw
      real(real64), intent(out) :: parsed
      integer(int32), intent(out) :: parse_status
      type(ieee_status_type) :: saved_status
      character(len=len(raw)) :: token
      character(len=1) :: digit
      integer :: position, last, digit_count
      integer(int64) :: bits
      logical :: nonzero_mantissa, overflowed, invalid

      parsed = 0.0_real64
      parse_status = 1_int32
      token = adjustl(raw)
      last = len_trim(token)
      if (last == 0) return
      position = 1
      if (token(position:position) == '+' .or. token(position:position) == '-') position = position + 1
      digit_count = 0
      nonzero_mantissa = .false.
      do while (position <= last)
         digit = token(position:position)
         if (digit < '0' .or. digit > '9') exit
         digit_count = digit_count + 1
         nonzero_mantissa = nonzero_mantissa .or. digit /= '0'
         position = position + 1
      end do
      if (position <= last) then
         if (token(position:position) == '.') then
            position = position + 1
            do while (position <= last)
               digit = token(position:position)
               if (digit < '0' .or. digit > '9') exit
               digit_count = digit_count + 1
               nonzero_mantissa = nonzero_mantissa .or. digit /= '0'
               position = position + 1
            end do
         end if
      end if
      if (digit_count == 0) return
      if (position <= last) then
         digit = token(position:position)
         if (digit == 'e' .or. digit == 'E' .or. digit == 'd' .or. digit == 'D') then
            position = position + 1
            if (position <= last) then
               if (token(position:position) == '+' .or. token(position:position) == '-') position = position + 1
            end if
            digit_count = 0
            do while (position <= last)
               digit = token(position:position)
               if (digit < '0' .or. digit > '9') exit
               digit_count = digit_count + 1
               position = position + 1
            end do
            if (digit_count == 0) return
         end if
      end if
      ! No list-directed separators, repeat counts, slash, or trailing tokens.
      if (position <= last) return

      call ieee_get_status(saved_status)
      if (ieee_support_halting(ieee_overflow)) call ieee_set_halting_mode(ieee_overflow, .false.)
      if (ieee_support_halting(ieee_underflow)) call ieee_set_halting_mode(ieee_underflow, .false.)
      if (ieee_support_halting(ieee_invalid)) call ieee_set_halting_mode(ieee_invalid, .false.)
      call ieee_set_flag(ieee_overflow, .false.)
      call ieee_set_flag(ieee_invalid, .false.)
      read(token, *, iostat=parse_status) parsed
      call ieee_get_flag(ieee_overflow, overflowed)
      call ieee_get_flag(ieee_invalid, invalid)
      call ieee_set_status(saved_status)
      if (parse_status /= 0_int32 .or. overflowed .or. invalid) then
         parse_status = 1_int32
         return
      end if
      if (storage_size(parsed) == 64 .and. radix(parsed) == 2 .and. digits(parsed) == 53 .and. &
          maxexponent(parsed) == 1024 .and. minexponent(parsed) == -1021) then
         bits = transfer(parsed, 0_int64)
         if (iand(bits, int(z'7FF0000000000000', int64)) == int(z'7FF0000000000000', int64)) then
            parse_status = 1_int32
            return
         end if
      else
         if (.not. ieee_is_finite(parsed)) then
            parse_status = 1_int32
            return
         end if
      end if
      ! A nonzero decimal rounded entirely to zero must not silently disable an
      ! option such as weight decay. Representable subnormals remain valid.
      if (parsed == 0.0_real64 .and. nonzero_mantissa) parse_status = 1_int32
   end subroutine parse_real_option


   subroutine require_ok(status, label)
      integer(int32), intent(in) :: status
      character(len=*), intent(in) :: label

      if (status /= NPT_OK .and. status /= BPE_OK .and. status /= NRD_OK) then
         write(error_unit, '(A,A,A,I0)') 'vegai_neural_pretrain: ', trim(label), ' failed status=', status
         error stop 1
      end if
   end subroutine require_ok

   subroutine require_true(condition, label)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: label

      if (.not. condition) then
         write(error_unit, '(A,A)') 'vegai_neural_pretrain: ', trim(label)
         error stop 1
      end if
   end subroutine require_true

   logical function starts_with_dashes(text) result(is_option)
      character(len=*), intent(in) :: text

      is_option = len_trim(text) >= 2 .and. text(1:2) == '--'
   end function starts_with_dashes

   ! Parses trailing `--key value` pairs into the host option variables.
   ! Unknown keys and malformed values are usage errors, never ignored.
   subroutine parse_cli_options(first_index)
      integer, intent(in) :: first_index
      character(len=256) :: key, value
      integer :: index, argument_status
      integer(int32) :: int32_value, parse_status
      integer(int64) :: int64_value
      real(real64) :: real_value

      index = first_index
      do while (index <= command_argument_count())
         call get_command_argument(index, key, status=argument_status)
         if (argument_status /= 0) call usage_and_stop()
         if (.not. starts_with_dashes(key) .or. index + 1 > command_argument_count()) call usage_and_stop()
         call get_command_argument(index + 1, value, status=argument_status)
         if (argument_status /= 0) call usage_and_stop()
         select case (trim(key))
         case ('--width', '--heads', '--window', '--ffn', '--max-tokens-per-record', '--gradient-accumulation')
            call parse_integer32(value, int32_value, parse_status)
            if (parse_status /= 0) call usage_and_stop()
            select case (trim(key))
            case ('--width'); opt_width = int32_value; opt_width_set = .true.
            case ('--heads'); opt_heads = int32_value; opt_heads_set = .true.
            case ('--window'); opt_window = int32_value; opt_window_set = .true.
            case ('--ffn'); opt_ffn = int32_value; opt_ffn_set = .true.
            case ('--max-tokens-per-record'); opt_max_tokens_per_record = int32_value; opt_max_tokens_set = .true.
            case ('--gradient-accumulation'); opt_gradient_accumulation = int32_value; opt_accumulation_set = .true.
            end select
         case ('--seed', '--warmup-updates', '--total-updates')
            call parse_integer64(value, int64_value, parse_status)
            if (parse_status /= 0) call usage_and_stop()
            select case (trim(key))
            case ('--seed'); opt_seed = int64_value
            case ('--warmup-updates'); opt_warmup_updates = int64_value; opt_warmup_set = .true.
            case ('--total-updates'); opt_total_updates = int64_value; opt_total_set = .true.
            end select
         case ('--lr', '--weight-decay', '--beta1', '--beta2', '--adam-eps', '--grad-clip', &
               '--embedding-limit', '--label-smoothing', '--min-lr-ratio')
            call parse_real_option(value, real_value, parse_status)
            if (parse_status /= 0) then
               write(error_unit, '(A,A)') 'vegai_neural_pretrain: invalid real option ', trim(key)
               call usage_and_stop()
            end if
            select case (trim(key))
            case ('--lr'); opt_lr = real_value; opt_lr_set = .true.
            case ('--weight-decay'); opt_weight_decay = real_value; opt_weight_decay_set = .true.
            case ('--beta1'); opt_beta1 = real_value; opt_beta1_set = .true.
            case ('--beta2'); opt_beta2 = real_value; opt_beta2_set = .true.
            case ('--adam-eps'); opt_adam_eps = real_value; opt_adam_eps_set = .true.
            case ('--grad-clip'); opt_grad_clip = real_value; opt_grad_clip_set = .true.
            case ('--embedding-limit'); opt_embedding_limit = real_value; opt_embedding_limit_set = .true.
            case ('--label-smoothing'); opt_label_smoothing = real_value; opt_label_smoothing_set = .true.
            case ('--min-lr-ratio'); opt_min_lr_ratio = real_value; opt_min_lr_ratio_set = .true.
            end select
         case ('--optimizer')
            select case (trim(value))
            case ('nova'); opt_optimizer = NPT_OPTIMIZER_NOVA
            case ('adamw'); opt_optimizer = NPT_OPTIMIZER_ADAMW
            case ('sgd'); opt_optimizer = NPT_OPTIMIZER_SGD
            case default; call usage_and_stop()
            end select
            opt_optimizer_set = .true.
         case ('--schedule')
            select case (trim(value))
            case ('constant'); opt_schedule = NPT_SCHEDULE_CONSTANT
            case ('invsqrt'); opt_schedule = NPT_SCHEDULE_WARMUP_INVSQRT
            case ('cosine'); opt_schedule = NPT_SCHEDULE_WARMUP_COSINE
            case default; call usage_and_stop()
            end select
            opt_schedule_set = .true.
         case default
            call usage_and_stop()
         end select
         index = index + 2
      end do
   end subroutine parse_cli_options

   subroutine build_train_config(train_config)
      type(neural_train_config_t), intent(out) :: train_config
      real(real64) :: peak_lr, effective_weight_decay

      if (opt_lr_set) then
         peak_lr = opt_lr
      else if (opt_optimizer == NPT_OPTIMIZER_NOVA .or. opt_optimizer == NPT_OPTIMIZER_ADAMW) then
         ! Both normalize the step by a gradient scale, so their step magnitude is
         ! the learning rate itself; SGD multiplies the raw gradient instead and
         ! needs the historical, much larger rate.
         peak_lr = 0.003_real64
      else
         peak_lr = 0.025_real64
      end if
      ! Decoupled weight decay exists in the NOVA and AdamW updates. An
      ! explicit non-zero value with SGD would otherwise be reported as active
      ! while doing nothing, so reject it and drop the default for SGD.
      effective_weight_decay = opt_weight_decay
      if (opt_optimizer == NPT_OPTIMIZER_SGD) then
         if (opt_weight_decay_set .and. opt_weight_decay /= 0.0_real64) then
            write(error_unit, '(A)') 'vegai_neural_pretrain: --weight-decay requires '// &
                                      '--optimizer nova or adamw (the SGD update has no decay term)'
            call usage_and_stop()
         end if
         effective_weight_decay = 0.0_real64
      end if
      if (opt_gradient_accumulation < 1_int32 .or. opt_gradient_accumulation > opt_max_tokens_per_record .or. &
          opt_gradient_accumulation > NPT_MAX_GRADIENT_ACCUMULATION) then
         write(error_unit, '(A,I0,A,I0,A)') 'vegai_neural_pretrain: --gradient-accumulation must be in [1, ', &
                                               min(opt_max_tokens_per_record, NPT_MAX_GRADIENT_ACCUMULATION), &
                                               '] (requested ', opt_gradient_accumulation, ')'
         call usage_and_stop()
      end if
      train_config = neural_train_config_t(learning_rate=peak_lr, gradient_clip_norm=opt_grad_clip, &
                                           embedding_value_limit=opt_embedding_limit, &
                                           maximum_tokens_per_record=opt_max_tokens_per_record, &
                                           optimizer=opt_optimizer, adam_beta1=opt_beta1, adam_beta2=opt_beta2, &
                                           adam_epsilon=opt_adam_eps, weight_decay=effective_weight_decay, &
                                           label_smoothing=opt_label_smoothing, lr_schedule=opt_schedule, &
                                           warmup_updates=opt_warmup_updates, total_updates=opt_total_updates, &
                                           minimum_lr_ratio=opt_min_lr_ratio, &
                                           gradient_accumulation=opt_gradient_accumulation)
      if (.not. neural_train_config_is_valid(train_config)) then
         ! Report the offending combination here instead of surfacing a bare
         ! library status from deep inside the first training update.  The
         ! library rejects e.g. lr*weight_decay >= 1 (a decoupled decay factor
         ! that would zero or invert every embedding), beta >= 1, and a cosine
         ! schedule without a horizon above the warmup.
         write(error_unit, '(A)') 'vegai_neural_pretrain: invalid training hyperparameters '// &
                                   '(need lr > 0, 0 <= beta1,beta2 < 1, adam-eps > 0, '// &
                                   '0 <= weight-decay, lr*weight-decay < 1, '// &
                                   '0 <= label-smoothing < 0.5, and cosine total-updates > warmup-updates)'
         call usage_and_stop()
      end if
   end subroutine build_train_config

   subroutine build_decoder_config(decoder_config, vocabulary_size)
      type(neural_decoder_config_t), intent(out) :: decoder_config
      integer(int32), intent(in) :: vocabulary_size

      decoder_config = neural_decoder_config_t(vocabulary_size=vocabulary_size, width=opt_width, heads=opt_heads, &
                                               local_window=opt_window, ffn_width=opt_ffn)
   end subroutine build_decoder_config

   ! Resume from the persisted config and override only explicitly supplied
   ! fields. A shape-only option or an unrelated flag must not silently reset
   ! the checkpoint's learning rate, optimizer, schedule or accumulation.
   subroutine apply_options_to_train_config(train_config, train_state)
      type(neural_train_config_t), intent(inout) :: train_config
      type(neural_train_state_t), intent(in) :: train_state
      type(neural_train_config_t) :: fresh

      if (opt_optimizer_set .and. opt_optimizer /= train_config%optimizer .and. &
          train_state%updates_completed > 0_int64) then
         write(error_unit, '(A)') 'vegai_neural_pretrain: cannot change the optimizer of a trained checkpoint; '// &
                                  'start a new run instead'
         call usage_and_stop()
      end if
      fresh = train_config
      if (opt_lr_set) fresh%learning_rate = opt_lr
      if (opt_optimizer_set) then
         fresh%optimizer = opt_optimizer
         if (opt_optimizer == NPT_OPTIMIZER_SGD .and. .not. opt_weight_decay_set) fresh%weight_decay = 0.0_real64
      end if
      if (opt_weight_decay_set) fresh%weight_decay = opt_weight_decay
      if (opt_beta1_set) fresh%adam_beta1 = opt_beta1
      if (opt_beta2_set) fresh%adam_beta2 = opt_beta2
      if (opt_adam_eps_set) fresh%adam_epsilon = opt_adam_eps
      if (opt_grad_clip_set) fresh%gradient_clip_norm = opt_grad_clip
      if (opt_embedding_limit_set) fresh%embedding_value_limit = opt_embedding_limit
      if (opt_label_smoothing_set) fresh%label_smoothing = opt_label_smoothing
      if (opt_min_lr_ratio_set) fresh%minimum_lr_ratio = opt_min_lr_ratio
      if (opt_max_tokens_set) fresh%maximum_tokens_per_record = opt_max_tokens_per_record
      if (opt_accumulation_set) fresh%gradient_accumulation = opt_gradient_accumulation
      if (opt_schedule_set) fresh%lr_schedule = opt_schedule
      if (opt_warmup_set) fresh%warmup_updates = opt_warmup_updates
      if (opt_total_set) fresh%total_updates = opt_total_updates
      if (.not. neural_train_config_is_valid(fresh)) then
         write(error_unit, '(A)') 'vegai_neural_pretrain: invalid explicit overrides for checkpoint training config'
         call usage_and_stop()
      end if
      train_config = fresh
   end subroutine apply_options_to_train_config

   pure logical function requested_shape_matches(config) result(matches)
      type(neural_decoder_config_t), intent(in) :: config

      matches = (.not. opt_width_set .or. opt_width == config%width) .and. &
                (.not. opt_heads_set .or. opt_heads == config%heads) .and. &
                (.not. opt_window_set .or. opt_window == config%local_window) .and. &
                (.not. opt_ffn_set .or. opt_ffn == config%ffn_width)
   end function requested_shape_matches

   function optimizer_name(optimizer) result(name)
      integer(int32), intent(in) :: optimizer
      character(len=6) :: name

      if (optimizer == NPT_OPTIMIZER_NOVA) then
         name = 'nova  '
      else if (optimizer == NPT_OPTIMIZER_ADAMW) then
         name = 'adamw '
      else
         name = 'sgd   '
      end if
   end function optimizer_name

   function schedule_name(schedule) result(name)
      integer(int32), intent(in) :: schedule
      character(len=9) :: name

      select case (schedule)
      case (NPT_SCHEDULE_WARMUP_INVSQRT)
         name = 'invsqrt  '
      case (NPT_SCHEDULE_WARMUP_COSINE)
         name = 'cosine   '
      case default
         name = 'constant '
      end select
   end function schedule_name

   function ratio_text(numerator, denominator) result(rendered)
      integer(int64), intent(in) :: numerator, denominator
      character(len=48) :: rendered

      if (denominator > 0_int64) then
         write(rendered, '(ES24.16E3)') real(numerator, real64) / real(denominator, real64)
      else
         write(rendered, '(ES24.16E3)') 0.0_real64
      end if
   end function ratio_text

   function perplexity_text(mean_nll) result(rendered)
      real(real64), intent(in) :: mean_nll
      character(len=48) :: rendered

      if (mean_nll < 700.0_real64) then
         write(rendered, '(ES24.16E3)') exp(mean_nll)
      else
         rendered = 'null                                            '
      end if
   end function perplexity_text

   subroutine initialize_byte_model(model, bpe, status)
      type(neural_decoder_t), intent(inout) :: model
      type(bpe_model_t), intent(inout) :: bpe
      integer(int32), intent(out) :: status
      type(neural_decoder_config_t) :: decoder_config

      call bpe_init(bpe, 16_int32, status)
      if (status /= BPE_OK) return
      decoder_config = neural_decoder_config_t(vocabulary_size=bpe%token_count, width=16_int32, heads=2_int32, &
                                               local_window=4_int32, ffn_width=32_int32)
      call neural_decoder_init(model, decoder_config, 20261025_int64, status)
   end subroutine initialize_byte_model

   subroutine run_smoke()
      type(bpe_model_t) :: bpe, loaded_bpe
      type(neural_decoder_t) :: model, loaded_model
      type(neural_decoder_state_t) :: state_a, state_b
      type(neural_train_config_t) :: train_config, loaded_config
      type(neural_train_state_t) :: train_state, loaded_state
      type(neural_train_metrics_t) :: before, after, epoch_metrics
      integer(int32) :: tokens(24), status, index
      real(real32) :: hidden_a(16), hidden_b(16), logits_a(256), logits_b(256)
      character(len=*), parameter :: tokenizer_path = 'vegai-neural-smoke.bpe'
      character(len=*), parameter :: checkpoint_path = 'vegai-neural-smoke.vgnr'
      integer :: unit, io_status

      call initialize_byte_model(model, bpe, status)
      call require_ok(status, 'byte model initialization')
      do index = 1_int32, size(tokens, kind=int32), 2_int32
         tokens(index) = bpe_base_token_for_byte(int(iachar('a'), int32))
         tokens(index + 1_int32) = bpe_base_token_for_byte(int(iachar('b'), int32))
      end do
      train_config = neural_train_config_t(learning_rate=0.050_real64, gradient_clip_norm=4.0_real64, &
                                           embedding_value_limit=4.0_real64, maximum_tokens_per_record=64_int32)
      call neural_train_state_init(train_state)
      call neural_train_metrics_reset(before)
      call neural_evaluate_token_sequence(model, tokens, size(tokens, kind=int32), train_config, before, status)
      call require_ok(status, 'initial toy NLL evaluation')
      do index = 1_int32, 80_int32
         call neural_train_metrics_reset(epoch_metrics)
         call neural_train_token_sequence(model, tokens, size(tokens, kind=int32), train_config, train_state, epoch_metrics, status)
         call require_ok(status, 'tied-output toy update')
      end do
      call neural_train_metrics_reset(after)
      call neural_evaluate_token_sequence(model, tokens, size(tokens, kind=int32), train_config, after, status)
      call require_ok(status, 'post-update toy NLL evaluation')
      call require_true(after%mean_nll < before%mean_nll, 'toy held-out NLL did not improve')

      call bpe_save(bpe, tokenizer_path, status)
      call require_ok(status, 'tokenizer checkpoint save')
      call neural_checkpoint_save(checkpoint_path, model, bpe, train_config, train_state, status)
      call require_ok(status, 'neural checkpoint save')
      call bpe_load(loaded_bpe, tokenizer_path, status)
      call require_ok(status, 'tokenizer checkpoint load')
      call neural_checkpoint_load(checkpoint_path, loaded_bpe, loaded_model, loaded_config, loaded_state, status)
      call require_ok(status, 'neural checkpoint load')
      call neural_decoder_state_init(state_a, model, status)
      call require_ok(status, 'source state init')
      call neural_decoder_state_init(state_b, loaded_model, status)
      call require_ok(status, 'restored state init')
      call neural_decoder_step(model, state_a, tokens(1), hidden_a, logits_a, status)
      call require_ok(status, 'source checkpoint inference')
      call neural_decoder_step(loaded_model, state_b, tokens(1), hidden_b, logits_b, status)
      call require_ok(status, 'restored checkpoint inference')
      call require_true(maxval(abs(logits_a - logits_b)) == 0.0_real32 .and. &
                        loaded_state%updates_completed == train_state%updates_completed, &
                        'checkpoint does not reproduce model or cursor exactly')

      write(output_unit, '(A)') 'NEURAL_PRETRAIN_SMOKE_JSON: {'// &
         '"schema_version":"vegai-neural-pretraining-smoke/v1",'// &
         '"training_mode":"tied-output-cross-entropy-adapter/v1",'// &
         '"initial_mean_nll":'//trim(r64_text(before%mean_nll))//','// &
         '"final_mean_nll":'//trim(r64_text(after%mean_nll))//','// &
         '"nll_improved":true,'// &
         '"checkpoint_exact":true,'// &
         '"updates":'//trim(i64_text(train_state%updates_completed))//'}'

      call neural_decoder_state_destroy(state_a)
      call neural_decoder_state_destroy(state_b)
      call neural_decoder_destroy(model)
      call neural_decoder_destroy(loaded_model)
      call bpe_destroy(bpe)
      call bpe_destroy(loaded_bpe)
      open(newunit=unit, file=tokenizer_path, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
      open(newunit=unit, file=checkpoint_path, status='old', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
   end subroutine run_smoke

   subroutine run_tokenizer_train(in_memory)
      logical, intent(in) :: in_memory
      type(bpe_model_t) :: bpe
      type(bpe_training_config_t) :: tokenizer_config
      type(bpe_training_metrics_t) :: tokenizer_metrics
      character(len=1024) :: corpus_path, tokenizer_path, target_text, frequency_text, capacity_text
      integer(int32) :: status, target_vocabulary, minimum_frequency, parse_status
      integer(int32) :: maximum_merges
      integer(int64) :: maximum_tokens
      integer(int64) :: clock_start, clock_end, clock_rate

      if (command_argument_count() < 4) call usage_and_stop()
      call get_positional_argument(2, corpus_path, is_path=.true.)
      call get_positional_argument(3, tokenizer_path, is_path=.true.)
      call reject_matching_paths(corpus_path, tokenizer_path)
      call get_positional_argument(4, target_text)
      call parse_integer32(target_text, target_vocabulary, parse_status)
      if (parse_status /= 0 .or. target_vocabulary < BPE_BASE_VOCABULARY) call usage_and_stop()
      minimum_frequency = 2_int32
      if (command_argument_count() >= 5) then
         call get_positional_argument(5, frequency_text)
         call parse_integer32(frequency_text, minimum_frequency, parse_status)
         if (parse_status /= 0 .or. minimum_frequency <= 0_int32) call usage_and_stop()
      end if
      maximum_merges = max(1_int32, target_vocabulary - BPE_BASE_VOCABULARY)
      call bpe_init(bpe, maximum_merges, status)
      call require_ok(status, 'tokenizer model initialization')
      tokenizer_config = bpe_training_config_t(target_vocabulary_size=target_vocabulary, &
                                               minimum_pair_frequency=int(minimum_frequency, int64))
      if (in_memory) then
         ! Diverse real text needs a larger exact pair table than the smoke
         ! default; capacity only bounds memory, never merge choice.
         tokenizer_config%pair_table_capacity = 1048583_int32
         if (command_argument_count() >= 6) then
            call get_positional_argument(6, capacity_text)
            call parse_integer64(capacity_text, maximum_tokens, parse_status)
            if (parse_status /= 0 .or. maximum_tokens < 2_int64) call usage_and_stop()
            tokenizer_config%maximum_in_memory_tokens = maximum_tokens
         end if
      end if
      call bpe_training_metrics_reset(tokenizer_metrics)
      call system_clock(count=clock_start, count_rate=clock_rate)
      if (in_memory) then
         call bpe_train_utf8_file_in_memory(bpe, trim(corpus_path), tokenizer_config, tokenizer_metrics, status)
      else
         call bpe_train_utf8_file(bpe, trim(corpus_path), tokenizer_config, tokenizer_metrics, status)
      end if
      call system_clock(count=clock_end)
      if (status /= BPET_OK) then
         write(error_unit, '(A,I0)') 'vegai_neural_pretrain: tokenizer training failed status=', status
         error stop 1
      end if
      call bpe_save(bpe, trim(tokenizer_path), status)
      call require_ok(status, 'trained tokenizer save')
      if (in_memory) then
         write(output_unit, '(A)') 'BPE_TRAIN_JSON: {'// &
            '"schema_version":"vegai-byte-bpe-train-fast/v1",'// &
            '"algorithm":"in-memory-apply-merge/v1",'// &
            '"target_vocabulary_size":'//trim(i32_text(target_vocabulary))//','// &
            '"actual_vocabulary_size":'//trim(i32_text(bpe%token_count))//','// &
            '"merges_added":'//trim(i64_text(tokenizer_metrics%merges_added))//','// &
            '"corpus_passes":'//trim(i64_text(tokenizer_metrics%corpus_passes))//','// &
            '"records_scanned":'//trim(i64_text(tokenizer_metrics%records_scanned))//','// &
            '"retained_input_bytes":'//trim(i64_text(tokenizer_metrics%retained_input_bytes))//','// &
            '"oversized_records":'//trim(i64_text(tokenizer_metrics%oversized_records))//','// &
            '"wall_seconds":'// &
               trim(r64_text(real(clock_end - clock_start, real64) / real(max(1_int64, clock_rate), real64)))//','// &
            '"exhausted_before_target":'//logical_json(tokenizer_metrics%exhausted_before_target)//'}'
      else
         write(output_unit, '(A)') 'BPE_TRAIN_JSON: {'// &
            '"schema_version":"vegai-byte-bpe-train/v1",'// &
            '"target_vocabulary_size":'//trim(i32_text(target_vocabulary))//','// &
            '"actual_vocabulary_size":'//trim(i32_text(bpe%token_count))//','// &
            '"merges_added":'//trim(i64_text(tokenizer_metrics%merges_added))//','// &
            '"corpus_passes":'//trim(i64_text(tokenizer_metrics%corpus_passes))//','// &
            '"records_scanned":'//trim(i64_text(tokenizer_metrics%records_scanned))//','// &
            '"exhausted_before_target":'//logical_json(tokenizer_metrics%exhausted_before_target)//'}'
      end if
      call bpe_destroy(bpe)
   end subroutine run_tokenizer_train

   subroutine run_train(resume)
      logical, intent(in) :: resume
      type(bpe_model_t) :: bpe
      type(neural_decoder_t) :: model
      type(neural_decoder_config_t) :: decoder_config
      type(neural_train_config_t) :: train_config
      type(neural_train_state_t) :: train_state
      type(neural_train_metrics_t) :: metrics
      character(len=1024) :: corpus_path, checkpoint_path, tokenizer_path, epochs_text
      integer(int32) :: status, epochs, epoch, parse_status
      integer :: options_start
      integer(int64) :: clock_start, clock_end, clock_rate
      real(real64) :: epoch_seconds, tokens_per_second
      logical :: tokenizer_exists, options_present

      if (command_argument_count() < 4) call usage_and_stop()
      call get_positional_argument(2, corpus_path, is_path=.true.)
      call get_positional_argument(3, checkpoint_path, is_path=.true.)
      call get_positional_argument(4, tokenizer_path, is_path=.true.)
      call reject_matching_paths(corpus_path, checkpoint_path)
      call reject_matching_paths(corpus_path, tokenizer_path)
      call reject_matching_paths(tokenizer_path, checkpoint_path)
      epochs = 1_int32
      options_start = 5
      if (command_argument_count() >= 5) then
         call get_positional_argument(5, epochs_text)
         if (starts_with_dashes(epochs_text)) then
            options_start = 5
         else
            call parse_integer32(epochs_text, epochs, parse_status)
            if (parse_status /= 0 .or. epochs <= 0_int32) call usage_and_stop()
            options_start = 6
         end if
      end if
      options_present = command_argument_count() >= options_start
      if (options_present) call parse_cli_options(options_start)
      if (resume) then
         call bpe_load(bpe, trim(tokenizer_path), status)
         call require_ok(status, 'resume tokenizer load')
         tokenizer_exists = .true.
         call neural_checkpoint_load(trim(checkpoint_path), bpe, model, train_config, train_state, status)
         call require_ok(status, 'resume neural checkpoint load')
         if (options_present) then
            ! Hyperparameters may be re-specified at an epoch boundary, but
            ! the serialized model shape can never change under a resume.
            call require_true(requested_shape_matches(model%config), &
                              'resume options must not change checkpoint model dimensions')
            call apply_options_to_train_config(train_config, train_state)
         end if
      else
         inquire(file=trim(tokenizer_path), exist=tokenizer_exists)
         if (tokenizer_exists) then
            call bpe_load(bpe, trim(tokenizer_path), status)
            call require_ok(status, 'prebuilt tokenizer load')
         else
            call bpe_init(bpe, 16_int32, status)
            call require_ok(status, 'new byte-fallback tokenizer initialization')
         end if
         call build_decoder_config(decoder_config, bpe%token_count)
         call neural_decoder_init(model, decoder_config, opt_seed, status)
         call require_ok(status, 'new neural model initialization')
         call build_train_config(train_config)
         call neural_train_state_init(train_state)
      end if
      do epoch = 1_int32, epochs
         call neural_train_metrics_reset(metrics)
         ! This command resumes at an epoch boundary. A partially-consumed
         ! source needs its separately attested record cursor passed to the
         ! library API; we never silently infer provenance from a path.
         call system_clock(count=clock_start, count_rate=clock_rate)
         call neural_train_utf8_file(model, bpe, trim(corpus_path), train_config, train_state, metrics, status)
         call system_clock(count=clock_end)
         call require_ok(status, 'UTF-8 byte-BPE pretraining pass')
         epoch_seconds = real(clock_end - clock_start, real64) / real(max(1_int64, clock_rate), real64)
         tokens_per_second = 0.0_real64
         if (epoch_seconds > 0.0_real64) then
            tokens_per_second = real(metrics%prediction_tokens, real64) / epoch_seconds
         end if
         write(output_unit, '(A)') 'NEURAL_PRETRAIN_EPOCH_JSON: {'// &
            '"epoch_in_invocation":'//trim(i32_text(epoch))//','// &
            '"mode":"tied-output-cross-entropy-adapter/v1",'// &
            '"optimizer":"'//trim(optimizer_name(train_config%optimizer))//'",'// &
            '"lr_schedule":"'//trim(schedule_name(train_config%lr_schedule))//'",'// &
            '"peak_learning_rate":'//trim(r64_text(train_config%learning_rate))//','// &
            '"final_learning_rate":'// &
               trim(r64_text(neural_scheduled_learning_rate(train_config, &
                             max(1_int64, train_state%updates_completed))))//','// &
            '"label_smoothing":'//trim(r64_text(train_config%label_smoothing))//','// &
            '"gradient_accumulation":'//trim(i32_text(train_config%gradient_accumulation))//','// &
            '"eligible_records":'//trim(i64_text(metrics%records))//','// &
            '"prediction_tokens":'//trim(i64_text(metrics%prediction_tokens))//','// &
            '"mean_nll":'//trim(r64_text(metrics%mean_nll))//','// &
            '"perplexity":'//trim(perplexity_text(metrics%mean_nll))//','// &
            '"top1_accuracy":'//trim(ratio_text(metrics%top1_correct, metrics%prediction_tokens))//','// &
            '"top5_accuracy":'//trim(ratio_text(metrics%top5_correct, metrics%prediction_tokens))//','// &
            '"epoch_wall_seconds":'//trim(r64_text(epoch_seconds))//','// &
            '"prediction_tokens_per_second":'//trim(r64_text(tokens_per_second))//','// &
            '"phase_seconds":{"decode":'//trim(r64_text(metrics%decode_seconds))//','// &
            '"project":'//trim(r64_text(metrics%project_seconds))//','// &
            '"score":'//trim(r64_text(metrics%score_seconds))//','// &
            '"update":'//trim(r64_text(metrics%update_seconds))//'},'// &
            '"skipped_records":'//trim(i64_text(metrics%skipped_records))//','// &
            '"tokenizer_truncations":'//trim(i64_text(metrics%tokenizer_truncations))//'}'
      end do
      ! Tokenization only reads the vocabulary. Rewriting an existing tokenizer
      ! cannot improve the model and risks corrupting it on a failed save.
      if (.not. tokenizer_exists) then
         call bpe_save(bpe, trim(tokenizer_path), status)
         call require_ok(status, 'tokenizer checkpoint save')
      end if
      call neural_checkpoint_save(trim(checkpoint_path), model, bpe, train_config, train_state, status)
      call require_ok(status, 'neural checkpoint save')
      write(output_unit, '(A)') 'NEURAL_PRETRAIN_COMPLETE_JSON: {'// &
         '"checkpoint":'//json_quote(trim(checkpoint_path))//','// &
         '"tokenizer":'//json_quote(trim(tokenizer_path))//','// &
         '"records_completed":'//trim(i64_text(train_state%records_completed))//','// &
         '"source_records_completed":'//trim(i64_text(train_state%source_records_completed))//','// &
         '"tokens_seen":'//trim(i64_text(train_state%tokens_seen))//','// &
         '"gradient_accumulation":'//trim(i32_text(train_config%gradient_accumulation))//','// &
         '"updates_completed":'//trim(i64_text(train_state%updates_completed))//'}'
      call neural_decoder_destroy(model)
      call bpe_destroy(bpe)
   end subroutine run_train

   ! Creates an untrained checkpoint with the exact CLI configuration so an
   ! operator can measure a held-out baseline before spending an epoch.
   subroutine run_init_checkpoint()
      type(bpe_model_t) :: bpe
      type(neural_decoder_t) :: model
      type(neural_decoder_config_t) :: decoder_config
      type(neural_train_config_t) :: train_config
      type(neural_train_state_t) :: train_state
      character(len=1024) :: checkpoint_path, tokenizer_path
      integer(int32) :: status
      logical :: tokenizer_exists

      if (command_argument_count() < 3) call usage_and_stop()
      call get_positional_argument(2, tokenizer_path, is_path=.true.)
      call get_positional_argument(3, checkpoint_path, is_path=.true.)
      call reject_matching_paths(tokenizer_path, checkpoint_path)
      if (command_argument_count() >= 4) call parse_cli_options(4)
      ! Configuration rejection must precede creation of a fallback tokenizer.
      call build_train_config(train_config)
      inquire(file=trim(tokenizer_path), exist=tokenizer_exists)
      if (tokenizer_exists) then
         call bpe_load(bpe, trim(tokenizer_path), status)
         call require_ok(status, 'init tokenizer load')
      else
         call bpe_init(bpe, 16_int32, status)
         call require_ok(status, 'init byte-fallback tokenizer initialization')
         call bpe_save(bpe, trim(tokenizer_path), status)
         call require_ok(status, 'init tokenizer save')
      end if
      call build_decoder_config(decoder_config, bpe%token_count)
      call neural_decoder_init(model, decoder_config, opt_seed, status)
      call require_ok(status, 'init neural model initialization')
      call neural_train_state_init(train_state)
      call neural_checkpoint_save(trim(checkpoint_path), model, bpe, train_config, train_state, status)
      call require_ok(status, 'init neural checkpoint save')
      write(output_unit, '(A)') 'NEURAL_INIT_JSON: {'// &
         '"schema_version":"vegai-neural-init-checkpoint/v1",'// &
         '"checkpoint":'//json_quote(trim(checkpoint_path))//','// &
         '"tokenizer":'//json_quote(trim(tokenizer_path))//','// &
         '"vocabulary_size":'//trim(i32_text(bpe%token_count))//','// &
         '"width":'//trim(i32_text(decoder_config%width))//','// &
         '"heads":'//trim(i32_text(decoder_config%heads))//','// &
         '"local_window":'//trim(i32_text(decoder_config%local_window))//','// &
         '"ffn_width":'//trim(i32_text(decoder_config%ffn_width))//','// &
         '"optimizer":"'//trim(optimizer_name(train_config%optimizer))//'",'// &
         '"lr_schedule":"'//trim(schedule_name(train_config%lr_schedule))//'",'// &
         '"peak_learning_rate":'//trim(r64_text(train_config%learning_rate))//','// &
         '"seed":'//trim(i64_text(opt_seed))//'}'
      call neural_decoder_destroy(model)
      call bpe_destroy(bpe)
   end subroutine run_init_checkpoint

   subroutine run_evaluate()
      type(bpe_model_t) :: bpe
      type(neural_decoder_t) :: model
      type(neural_train_config_t) :: train_config
      type(neural_train_state_t) :: train_state
      type(neural_train_metrics_t) :: metrics
      character(len=1024) :: checkpoint_path, tokenizer_path, corpus_path, limit_text
      integer(int32) :: status, parse_status
      integer(int64) :: record_limit, clock_start, clock_end, clock_rate
      real(real64) :: wall_seconds, tokens_per_second

      if (command_argument_count() < 4) call usage_and_stop()
      call get_positional_argument(2, checkpoint_path, is_path=.true.)
      call get_positional_argument(3, tokenizer_path, is_path=.true.)
      call get_positional_argument(4, corpus_path, is_path=.true.)
      record_limit = 0_int64
      if (command_argument_count() >= 5) then
         call get_positional_argument(5, limit_text)
         call parse_integer64(limit_text, record_limit, parse_status)
         if (parse_status /= 0 .or. record_limit < 1_int64) call usage_and_stop()
      end if
      call bpe_load(bpe, trim(tokenizer_path), status)
      call require_ok(status, 'evaluate tokenizer load')
      call neural_checkpoint_load(trim(checkpoint_path), bpe, model, train_config, train_state, status)
      call require_ok(status, 'evaluate neural checkpoint load')
      call neural_train_metrics_reset(metrics)
      call system_clock(count=clock_start, count_rate=clock_rate)
      if (record_limit > 0_int64) then
         call neural_evaluate_utf8_file(model, bpe, trim(corpus_path), train_config, metrics, status, &
                                        maximum_source_records=record_limit)
      else
         call neural_evaluate_utf8_file(model, bpe, trim(corpus_path), train_config, metrics, status)
      end if
      call system_clock(count=clock_end)
      call require_ok(status, 'held-out UTF-8 evaluation pass')
      wall_seconds = real(clock_end - clock_start, real64) / real(max(1_int64, clock_rate), real64)
      tokens_per_second = 0.0_real64
      if (wall_seconds > 0.0_real64) tokens_per_second = real(metrics%prediction_tokens, real64) / wall_seconds
      write(output_unit, '(A)') 'NEURAL_EVALUATE_JSON: {'// &
         '"schema_version":"vegai-neural-evaluate/v1",'// &
         '"mode":"teacher-forced-no-update/v1",'// &
         '"updates_completed_at_checkpoint":'//trim(i64_text(train_state%updates_completed))//','// &
         '"eligible_records":'//trim(i64_text(metrics%records))//','// &
         '"prediction_tokens":'//trim(i64_text(metrics%prediction_tokens))//','// &
         '"mean_nll":'//trim(r64_text(metrics%mean_nll))//','// &
         '"perplexity":'//trim(perplexity_text(metrics%mean_nll))//','// &
         '"top1_accuracy":'//trim(ratio_text(metrics%top1_correct, metrics%prediction_tokens))//','// &
         '"top5_accuracy":'//trim(ratio_text(metrics%top5_correct, metrics%prediction_tokens))//','// &
         '"wall_seconds":'//trim(r64_text(wall_seconds))//','// &
         '"prediction_tokens_per_second":'//trim(r64_text(tokens_per_second))//','// &
         '"skipped_records":'//trim(i64_text(metrics%skipped_records))//','// &
         '"tokenizer_truncations":'//trim(i64_text(metrics%tokenizer_truncations))//'}'
      call neural_decoder_destroy(model)
      call bpe_destroy(bpe)
   end subroutine run_evaluate

   subroutine run_training_plan()
      type(bpe_model_t) :: bpe
      type(neural_decoder_t) :: model
      type(neural_train_config_t) :: train_config
      type(neural_train_state_t) :: train_state
      type(neural_training_resource_plan_t) :: plan
      character(len=1024) :: checkpoint_path, tokenizer_path
      integer(int32) :: status

      if (command_argument_count() /= 3) call usage_and_stop()
      call get_positional_argument(2, checkpoint_path, is_path=.true.)
      call get_positional_argument(3, tokenizer_path, is_path=.true.)
      call bpe_load(bpe, trim(tokenizer_path), status)
      call require_ok(status, 'training-plan tokenizer load')
      call neural_checkpoint_load(trim(checkpoint_path), bpe, model, train_config, train_state, status)
      call require_ok(status, 'training-plan checkpoint load')
      call neural_training_resource_plan(model%config, bpe, train_config, plan, status)
      call require_ok(status, 'training resource plan')
      write(output_unit, '(A)') 'NEURAL_TRAINING_PLAN_JSON: {'// &
         '"schema_version":"vegai-neural-training-resource-plan/v1",'// &
         '"decoder_parameter_bytes_real32":'//trim(i64_text(plan%decoder_parameter_bytes_real32))//','// &
         '"tokenizer_bytes":'//trim(i64_text(plan%tokenizer_bytes))//','// &
         '"decoder_state_bytes_real32":'//trim(i64_text(plan%decoder_state_bytes_real32))//','// &
         '"record_token_buffer_bytes":'//trim(i64_text(plan%record_token_buffer_bytes))//','// &
         '"per_step_working_bytes":'//trim(i64_text(plan%per_step_working_bytes))//','// &
         '"optimizer_state_bytes_real64":'//trim(i64_text(plan%optimizer_state_bytes_real64))//','// &
         '"gradient_accumulation":'//trim(i32_text(train_config%gradient_accumulation))//','// &
         '"minimum_resident_bytes":'//trim(i64_text(plan%minimum_resident_bytes))//'}'
      call neural_decoder_destroy(model)
      call bpe_destroy(bpe)
   end subroutine run_training_plan

   ! Bounded record chunks are the CLI surface for continuous neural
   ! pretraining. The caller must attest that a resumed corpus is byte/order
   ! identical; this command never treats a filename as provenance evidence.
   subroutine run_train_chunk()
      type(bpe_model_t) :: bpe
      type(neural_decoder_t) :: model
      type(neural_decoder_config_t) :: decoder_config
      type(neural_train_config_t) :: train_config
      type(neural_train_state_t) :: train_state
      type(neural_train_metrics_t) :: metrics
      character(len=1024) :: corpus_path, checkpoint_path, tokenizer_path, limit_text, start_text
      integer(int32) :: status, parse_status
      integer(int64) :: record_limit, start_record
      integer :: options_start
      logical :: checkpoint_exists, tokenizer_exists, options_present

      if (command_argument_count() < 5) call usage_and_stop()
      call get_positional_argument(2, corpus_path, is_path=.true.)
      call get_positional_argument(3, checkpoint_path, is_path=.true.)
      call get_positional_argument(4, tokenizer_path, is_path=.true.)
      call reject_matching_paths(corpus_path, checkpoint_path)
      call reject_matching_paths(corpus_path, tokenizer_path)
      call reject_matching_paths(tokenizer_path, checkpoint_path)
      call get_positional_argument(5, limit_text)
      call parse_integer64(limit_text, record_limit, parse_status)
      if (parse_status /= 0 .or. record_limit < 1_int64) call usage_and_stop()
      options_start = 0
      start_record = -1_int64
      if (command_argument_count() >= 6) then
         call get_positional_argument(6, start_text)
         if (starts_with_dashes(start_text)) then
            options_start = 6
         else
            call parse_integer64(start_text, start_record, parse_status)
            if (parse_status /= 0 .or. start_record < 0_int64) call usage_and_stop()
            options_start = 7
         end if
      end if
      options_present = options_start > 0 .and. command_argument_count() >= options_start
      if (options_present) call parse_cli_options(options_start)
      inquire(file=trim(checkpoint_path), exist=checkpoint_exists)
      if (checkpoint_exists) then
         call bpe_load(bpe, trim(tokenizer_path), status)
         call require_ok(status, 'continuous tokenizer load')
         tokenizer_exists = .true.
         call neural_checkpoint_load(trim(checkpoint_path), bpe, model, train_config, train_state, status)
         call require_ok(status, 'continuous checkpoint load')
         if (options_present) then
            call require_true(requested_shape_matches(model%config), &
                              'train-chunk options must not change checkpoint model dimensions')
            call apply_options_to_train_config(train_config, train_state)
         end if
         if (start_record < 0_int64) start_record = train_state%source_records_completed
      else
         inquire(file=trim(tokenizer_path), exist=tokenizer_exists)
         if (tokenizer_exists) then
            call bpe_load(bpe, trim(tokenizer_path), status)
            call require_ok(status, 'continuous prebuilt tokenizer load')
         else
            call bpe_init(bpe, 16_int32, status)
            call require_ok(status, 'continuous byte-fallback tokenizer initialization')
         end if
         call build_decoder_config(decoder_config, bpe%token_count)
         call neural_decoder_init(model, decoder_config, opt_seed, status)
         call require_ok(status, 'continuous neural model initialization')
         call build_train_config(train_config)
         call neural_train_state_init(train_state)
         if (start_record < 0_int64) start_record = 0_int64
      end if
      call neural_train_metrics_reset(metrics)
      call neural_train_utf8_file(model, bpe, trim(corpus_path), train_config, train_state, metrics, status, &
                                  skip_source_records=start_record, maximum_new_source_records=record_limit)
      call require_ok(status, 'continuous UTF-8 byte-BPE chunk')
      if (.not. tokenizer_exists) then
         call bpe_save(bpe, trim(tokenizer_path), status)
         call require_ok(status, 'continuous tokenizer save')
      end if
      call neural_checkpoint_save(trim(checkpoint_path), model, bpe, train_config, train_state, status)
      call require_ok(status, 'continuous checkpoint save')
      write(output_unit, '(A)') 'NEURAL_PRETRAIN_CHUNK_JSON: {'// &
         '"schema_version":"vegai-neural-pretraining-chunk/v1",'// &
         '"mode":"tied-output-cross-entropy-adapter/v1",'// &
         '"start_source_record":'//trim(i64_text(start_record))//','// &
         '"new_source_record_limit":'//trim(i64_text(record_limit))//','// &
         '"source_records_completed":'//trim(i64_text(train_state%source_records_completed))//','// &
         '"eligible_records":'//trim(i64_text(metrics%records))//','// &
         '"prediction_tokens":'//trim(i64_text(metrics%prediction_tokens))//','// &
         '"mean_nll":'//trim(r64_text(metrics%mean_nll))//','// &
         '"perplexity":'//trim(perplexity_text(metrics%mean_nll))//','// &
         '"top1_accuracy":'//trim(ratio_text(metrics%top1_correct, metrics%prediction_tokens))//','// &
         '"top5_accuracy":'//trim(ratio_text(metrics%top5_correct, metrics%prediction_tokens))//','// &
         '"updates_completed":'//trim(i64_text(train_state%updates_completed))//'}'
      call neural_decoder_destroy(model)
      call bpe_destroy(bpe)
   end subroutine run_train_chunk

   subroutine run_generate()
      type(bpe_model_t) :: bpe
      type(neural_decoder_t) :: model
      type(neural_decoder_state_t) :: decoder_state
      type(neural_train_config_t) :: train_config
      type(neural_train_state_t) :: train_state
      type(decoding_policy_t) :: policy
      character(len=1024) :: checkpoint_path, tokenizer_path, count_text, seed_text
      character(len=BPE_MAX_INPUT_BYTES) :: prompt, decoded_token
      integer(int32) :: prompt_tokens(BPE_MAX_INPUT_BYTES), history(BPE_MAX_INPUT_BYTES + 512_int32), one_token(1)
      integer(int32) :: prompt_count, history_count, decoded_bytes, status, max_generated, generation, selected, parse_status
      integer(int64) :: random_state, initial_random_state
      real(real32), allocatable :: hidden(:), logits(:)
      real(real64), allocatable :: logits64(:), probabilities(:)
      integer :: allocation_status, prompt_length

      if (command_argument_count() < 4) call usage_and_stop()
      call get_positional_argument(2, checkpoint_path, is_path=.true.)
      call get_positional_argument(3, tokenizer_path, is_path=.true.)
      call get_positional_argument(4, prompt, actual_length=prompt_length)
      max_generated = 32_int32
      random_state = 1_int64
      if (command_argument_count() >= 5) then
         call get_positional_argument(5, count_text)
         call parse_integer32(count_text, max_generated, parse_status)
         if (parse_status /= 0 .or. max_generated < 1_int32 .or. max_generated > 512_int32) call usage_and_stop()
      end if
      if (command_argument_count() >= 6) then
         call get_positional_argument(6, seed_text)
         call parse_integer64(seed_text, random_state, parse_status)
         if (parse_status /= 0) call usage_and_stop()
      end if
      initial_random_state = random_state
      call bpe_load(bpe, trim(tokenizer_path), status)
      call require_ok(status, 'generation tokenizer load')
      call neural_checkpoint_load(trim(checkpoint_path), bpe, model, train_config, train_state, status)
      call require_ok(status, 'generation model checkpoint load')
      call bpe_encode(bpe, prompt, int(prompt_length, int32), prompt_tokens, prompt_count, status)
      call require_ok(status, 'generation prompt byte-BPE encode')
      call require_true(prompt_count > 0_int32, 'generation requires a non-empty prompt')
      allocate(hidden(model%config%width), logits(model%config%vocabulary_size), &
               logits64(model%config%vocabulary_size), probabilities(model%config%vocabulary_size), stat=allocation_status)
      if (allocation_status /= 0) then
         write(error_unit, '(A)') 'vegai_neural_pretrain: generation allocation failed'
         error stop 1
      end if
      call neural_decoder_state_init(decoder_state, model, status)
      call require_ok(status, 'generation decoder state init')
      do generation = 1_int32, prompt_count
         call neural_decoder_step(model, decoder_state, prompt_tokens(generation), hidden, logits, status)
         call require_ok(status, 'generation prompt causal step')
      end do
      history = 0_int32
      history(:prompt_count) = prompt_tokens(:prompt_count)
      history_count = prompt_count
      policy = decoding_policy_t(temperature=0.85_real64, frequency_penalty=0.10_real64, presence_penalty=0.10_real64, &
                                 immediate_repeat_penalty=0.20_real64, no_repeat_ngram=3_int32, top_k=64_int32, top_p=0.95_real64)
      do generation = 1_int32, max_generated
         logits64 = real(logits, real64)
         call decoder_distribution(logits64, history, history_count, policy, probabilities, status)
         call require_ok(status, 'generation calibrated distribution')
         call decoder_select(probabilities, policy, random_state, selected, status)
         call require_ok(status, 'generation calibrated selection')
         history_count = history_count + 1_int32
         history(history_count) = selected
         if (generation < max_generated) then
            call neural_decoder_step(model, decoder_state, selected, hidden, logits, status)
            call require_ok(status, 'generation causal continuation step')
         end if
      end do
      write(output_unit, '(A)') 'NEURAL_GENERATE_JSON: {'// &
         '"schema_version":"vegai-neural-generate/v1",'// &
         '"mode":"causal-byte-bpe-retention-calibrated/v1",'// &
         '"prompt_tokens":'//trim(i32_text(prompt_count))//','// &
         '"generated_tokens":'//trim(i32_text(max_generated))//','// &
         '"initial_seed":'//trim(i64_text(initial_random_state))//','// &
         '"final_rng_state":'//trim(i64_text(random_state))//'}'
      write(output_unit, '(A)') 'NEURAL_GENERATION_BEGIN'
      do generation = 1_int32, max_generated
         one_token(1) = history(prompt_count + generation)
         call bpe_decode(bpe, one_token, 1_int32, decoded_token, decoded_bytes, status)
         call require_ok(status, 'generation byte-BPE decode')
         if (decoded_bytes > 0_int32) write(output_unit, '(A)', advance='no') decoded_token(:decoded_bytes)
      end do
      write(output_unit, '(A)') ''
      write(output_unit, '(A)') 'NEURAL_GENERATION_END'
      call neural_decoder_state_destroy(decoder_state)
      call neural_decoder_destroy(model)
      call bpe_destroy(bpe)
      deallocate(hidden, logits, logits64, probabilities)
   end subroutine run_generate

   ! Quote CLI paths as JSON strings, including quotes, backslashes and all
   ! ASCII control bytes. Non-ASCII UTF-8 bytes are passed through unchanged.
   function json_quote(value) result(rendered)
      character(len=*), intent(in) :: value
      character(len=:), allocatable :: rendered
      character(len=2 + 6 * len(value)) :: buffer
      character(len=*), parameter :: hex_digits = '0123456789abcdef'
      integer :: i, position, code, high, low

      buffer = ''
      buffer(1:1) = '"'
      position = 2
      do i = 1, len(value)
         code = iand(iachar(value(i:i)), 255)
         if (code == 34 .or. code == 92) then
            buffer(position:position) = achar(92)
            buffer(position + 1:position + 1) = value(i:i)
            position = position + 2
         else if (code < 32) then
            high = code / 16 + 1
            low = mod(code, 16) + 1
            buffer(position:position + 3) = achar(92)//'u00'
            buffer(position + 4:position + 4) = hex_digits(high:high)
            buffer(position + 5:position + 5) = hex_digits(low:low)
            position = position + 6
         else
            buffer(position:position) = value(i:i)
            position = position + 1
         end if
      end do
      buffer(position:position) = '"'
      rendered = buffer(:position)
   end function json_quote

   function i64_text(value) result(rendered)
      integer(int64), intent(in) :: value
      character(len=32) :: rendered
      write(rendered, '(I0)') value
   end function i64_text

   function i32_text(value) result(rendered)
      integer(int32), intent(in) :: value
      character(len=16) :: rendered
      write(rendered, '(I0)') value
   end function i32_text

   function r64_text(value) result(rendered)
      real(real64), intent(in) :: value
      character(len=48) :: rendered
      write(rendered, '(ES24.16E3)') value
   end function r64_text

   function logical_json(value) result(rendered)
      logical, intent(in) :: value
      character(len=5) :: rendered

      if (value) then
         rendered = 'true '
      else
         rendered = 'false'
      end if
   end function logical_json

   subroutine usage_and_stop()
      call print_help()
      error stop 2
   end subroutine usage_and_stop

   subroutine print_help()
      write(output_unit, '(A)') 'Usage:'
      write(output_unit, '(A)') '  vegai_neural_pretrain smoke'
      write(output_unit, '(A)') '  vegai_neural_pretrain tokenizer-train CORPUS TOKENIZER TARGET_VOCAB [MIN_PAIR_FREQUENCY]'
      write(output_unit, '(A)') '  vegai_neural_pretrain tokenizer-train-fast CORPUS TOKENIZER TARGET_VOCAB '// &
                                '[MIN_PAIR_FREQUENCY] [MAX_IN_MEMORY_TOKENS]'
      write(output_unit, '(A)') '  vegai_neural_pretrain train  CORPUS CHECKPOINT TOKENIZER [EPOCHS] [--key value ...]'
      write(output_unit, '(A)') '  vegai_neural_pretrain resume CORPUS CHECKPOINT TOKENIZER [EPOCHS] [--key value ...]'
      write(output_unit, '(A)') '  vegai_neural_pretrain init-checkpoint TOKENIZER CHECKPOINT [--key value ...]'
      write(output_unit, '(A)') '  vegai_neural_pretrain evaluate CHECKPOINT TOKENIZER CORPUS [MAX_RECORDS]'
      write(output_unit, '(A)') '  vegai_neural_pretrain generate CHECKPOINT TOKENIZER PROMPT [TOKENS] [SEED]'
      write(output_unit, '(A)') '  vegai_neural_pretrain train-chunk CORPUS CHECKPOINT TOKENIZER NEW_RECORDS '// &
                                '[START_RECORD] [--key value ...]'
      write(output_unit, '(A)') '  vegai_neural_pretrain plan CHECKPOINT TOKENIZER'
      write(output_unit, '(A)') 'Options: --width --heads --window --ffn --seed --lr --optimizer nova|adamw|sgd '// &
                                '--schedule constant|invsqrt|cosine --warmup-updates --total-updates --min-lr-ratio'
      write(output_unit, '(A)') '         --weight-decay (nova/adamw only) --beta1 --beta2 --adam-eps --grad-clip '// &
                                '--embedding-limit --label-smoothing --max-tokens-per-record '// &
                                '--gradient-accumulation'
      write(output_unit, '(A)') '--total-updates and --warmup-updates count optimizer steps; with '// &
                                '--gradient-accumulation N one step consumes up to N prediction tokens.'
      write(output_unit, '(A)') 'Resume/chunk retain saved hyperparameters except explicit overrides; '// &
                                '--seed is used only for initialization.'
      write(output_unit, '(A)') 'Changing optimizer after a checkpoint has updates is rejected; start a new run.'
      write(output_unit, '(A)') 'The current trainer updates only tied output embeddings; it is not full BPTT.'
      write(output_unit, '(A)') 'NOVA and AdamW apply decoupled decay to embeddings (never to the output bias) '// &
                                'as theta <- (1 - lr*weight_decay) * theta.'
   end subroutine print_help

end program vegai_neural_pretrain
