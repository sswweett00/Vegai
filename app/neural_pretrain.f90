! SPDX-License-Identifier: MIT
! Native teacher-forced pretraining utility for the neural retention decoder.
program vegai_neural_pretrain
   use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64, output_unit, error_unit
   use subword_bpe
   use neural_retention_decoder
   use neural_pretraining
   implicit none

   character(len=32) :: command
   integer :: argument_count

   argument_count = command_argument_count()
   if (argument_count < 1) call usage_and_stop()
   call get_command_argument(1, command)
   select case (trim(command))
   case ('smoke')
      call run_smoke()
   case ('train')
      call run_train(.false.)
   case ('resume')
      call run_train(.true.)
   case ('help', '--help', '-h')
      call print_help()
   case default
      call usage_and_stop()
   end select

contains

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

      if (command_argument_count() < 4) call usage_and_stop()
      call get_command_argument(2, corpus_path)
      call get_command_argument(3, checkpoint_path)
      call get_command_argument(4, tokenizer_path)
      epochs = 1_int32
      if (command_argument_count() >= 5) then
         call get_command_argument(5, epochs_text)
         read(epochs_text, *, iostat=parse_status) epochs
         if (parse_status /= 0 .or. epochs <= 0_int32) call usage_and_stop()
      end if
      if (resume) then
         call bpe_load(bpe, trim(tokenizer_path), status)
         call require_ok(status, 'resume tokenizer load')
         call neural_checkpoint_load(trim(checkpoint_path), bpe, model, train_config, train_state, status)
         call require_ok(status, 'resume neural checkpoint load')
      else
         call bpe_init(bpe, 16_int32, status)
         call require_ok(status, 'new byte tokenizer initialization')
         decoder_config = neural_decoder_config_t(vocabulary_size=bpe%token_count, width=64_int32, heads=4_int32, &
                                                  local_window=32_int32, ffn_width=256_int32)
         call neural_decoder_init(model, decoder_config, 20261025_int64, status)
         call require_ok(status, 'new neural model initialization')
         train_config = neural_train_config_t(maximum_tokens_per_record=8192_int32)
         call neural_train_state_init(train_state)
      end if
      do epoch = 1_int32, epochs
         call neural_train_metrics_reset(metrics)
         ! This command resumes at an epoch boundary. A partially-consumed
         ! source needs its separately attested record cursor passed to the
         ! library API; we never silently infer provenance from a path.
         call neural_train_utf8_file(model, bpe, trim(corpus_path), train_config, train_state, metrics, status)
         call require_ok(status, 'UTF-8 byte-BPE pretraining pass')
         write(output_unit, '(A)') 'NEURAL_PRETRAIN_EPOCH_JSON: {'// &
            '"epoch_in_invocation":'//trim(i32_text(epoch))//','// &
            '"mode":"tied-output-cross-entropy-adapter/v1",'// &
            '"eligible_records":'//trim(i64_text(metrics%records))//','// &
            '"prediction_tokens":'//trim(i64_text(metrics%prediction_tokens))//','// &
            '"mean_nll":'//trim(r64_text(metrics%mean_nll))//','// &
            '"skipped_records":'//trim(i64_text(metrics%skipped_records))//','// &
            '"tokenizer_truncations":'//trim(i64_text(metrics%tokenizer_truncations))//'}'
      end do
      call bpe_save(bpe, trim(tokenizer_path), status)
      call require_ok(status, 'tokenizer checkpoint save')
      call neural_checkpoint_save(trim(checkpoint_path), model, bpe, train_config, train_state, status)
      call require_ok(status, 'neural checkpoint save')
      write(output_unit, '(A)') 'NEURAL_PRETRAIN_COMPLETE_JSON: {'// &
         '"checkpoint":"'//trim(checkpoint_path)//'",'// &
         '"tokenizer":"'//trim(tokenizer_path)//'",'// &
         '"records_completed":'//trim(i64_text(train_state%records_completed))//','// &
         '"source_records_completed":'//trim(i64_text(train_state%source_records_completed))//','// &
         '"updates_completed":'//trim(i64_text(train_state%updates_completed))//'}'
      call neural_decoder_destroy(model)
      call bpe_destroy(bpe)
   end subroutine run_train

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

   subroutine usage_and_stop()
      call print_help()
      error stop 2
   end subroutine usage_and_stop

   subroutine print_help()
      write(output_unit, '(A)') 'Usage:'
      write(output_unit, '(A)') '  vegai_neural_pretrain smoke'
      write(output_unit, '(A)') '  vegai_neural_pretrain train  CORPUS CHECKPOINT TOKENIZER [EPOCHS]'
      write(output_unit, '(A)') '  vegai_neural_pretrain resume CORPUS CHECKPOINT TOKENIZER [EPOCHS]'
      write(output_unit, '(A)') 'The current trainer updates only tied output embeddings; it is not full BPTT.'
   end subroutine print_help

end program vegai_neural_pretrain
