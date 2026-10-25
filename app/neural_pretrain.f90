! SPDX-License-Identifier: MIT
! Native teacher-forced pretraining utility for the neural retention decoder.
program vegai_neural_pretrain
   use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64, output_unit, error_unit
   use subword_bpe
   use subword_bpe_training
   use neural_retention_decoder
   use neural_pretraining
   use calibrated_decoder
   implicit none

   character(len=32) :: command
   integer :: argument_count

   argument_count = command_argument_count()
   if (argument_count < 1) call usage_and_stop()
   call get_command_argument(1, command)
   select case (trim(command))
   case ('smoke')
      call run_smoke()
   case ('tokenizer-train')
      call run_tokenizer_train()
   case ('train')
      call run_train(.false.)
   case ('resume')
      call run_train(.true.)
   case ('generate')
      call run_generate()
   case ('train-chunk')
      call run_train_chunk()
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

   subroutine run_tokenizer_train()
      type(bpe_model_t) :: bpe
      type(bpe_training_config_t) :: tokenizer_config
      type(bpe_training_metrics_t) :: tokenizer_metrics
      character(len=1024) :: corpus_path, tokenizer_path, target_text, frequency_text
      integer(int32) :: status, target_vocabulary, minimum_frequency, parse_status
      integer(int32) :: maximum_merges

      if (command_argument_count() < 4) call usage_and_stop()
      call get_command_argument(2, corpus_path)
      call get_command_argument(3, tokenizer_path)
      call get_command_argument(4, target_text)
      read(target_text, *, iostat=parse_status) target_vocabulary
      if (parse_status /= 0 .or. target_vocabulary < BPE_BASE_VOCABULARY) call usage_and_stop()
      minimum_frequency = 2_int32
      if (command_argument_count() >= 5) then
         call get_command_argument(5, frequency_text)
         read(frequency_text, *, iostat=parse_status) minimum_frequency
         if (parse_status /= 0 .or. minimum_frequency <= 0_int32) call usage_and_stop()
      end if
      maximum_merges = max(1_int32, target_vocabulary - BPE_BASE_VOCABULARY)
      call bpe_init(bpe, maximum_merges, status)
      call require_ok(status, 'tokenizer model initialization')
      tokenizer_config = bpe_training_config_t(target_vocabulary_size=target_vocabulary, &
                                               minimum_pair_frequency=int(minimum_frequency, int64))
      call bpe_training_metrics_reset(tokenizer_metrics)
      call bpe_train_utf8_file(bpe, trim(corpus_path), tokenizer_config, tokenizer_metrics, status)
      if (status /= BPET_OK) then
         write(error_unit, '(A,I0)') 'vegai_neural_pretrain: tokenizer training failed status=', status
         error stop 1
      end if
      call bpe_save(bpe, trim(tokenizer_path), status)
      call require_ok(status, 'trained tokenizer save')
      write(output_unit, '(A)') 'BPE_TRAIN_JSON: {'// &
         '"schema_version":"vegai-byte-bpe-train/v1",'// &
         '"target_vocabulary_size":'//trim(i32_text(target_vocabulary))//','// &
         '"actual_vocabulary_size":'//trim(i32_text(bpe%token_count))//','// &
         '"merges_added":'//trim(i64_text(tokenizer_metrics%merges_added))//','// &
         '"corpus_passes":'//trim(i64_text(tokenizer_metrics%corpus_passes))//','// &
         '"records_scanned":'//trim(i64_text(tokenizer_metrics%records_scanned))//','// &
         '"exhausted_before_target":'//logical_json(tokenizer_metrics%exhausted_before_target)//'}'
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
      logical :: tokenizer_exists

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
         inquire(file=trim(tokenizer_path), exist=tokenizer_exists)
         if (tokenizer_exists) then
            call bpe_load(bpe, trim(tokenizer_path), status)
            call require_ok(status, 'prebuilt tokenizer load')
         else
            call bpe_init(bpe, 16_int32, status)
            call require_ok(status, 'new byte-fallback tokenizer initialization')
         end if
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
      logical :: checkpoint_exists, tokenizer_exists

      if (command_argument_count() < 5) call usage_and_stop()
      call get_command_argument(2, corpus_path)
      call get_command_argument(3, checkpoint_path)
      call get_command_argument(4, tokenizer_path)
      call get_command_argument(5, limit_text)
      read(limit_text, *, iostat=parse_status) record_limit
      if (parse_status /= 0 .or. record_limit < 1_int64) call usage_and_stop()
      inquire(file=trim(checkpoint_path), exist=checkpoint_exists)
      if (checkpoint_exists) then
         call bpe_load(bpe, trim(tokenizer_path), status)
         call require_ok(status, 'continuous tokenizer load')
         call neural_checkpoint_load(trim(checkpoint_path), bpe, model, train_config, train_state, status)
         call require_ok(status, 'continuous checkpoint load')
         start_record = train_state%source_records_completed
      else
         inquire(file=trim(tokenizer_path), exist=tokenizer_exists)
         if (tokenizer_exists) then
            call bpe_load(bpe, trim(tokenizer_path), status)
            call require_ok(status, 'continuous prebuilt tokenizer load')
         else
            call bpe_init(bpe, 16_int32, status)
            call require_ok(status, 'continuous byte-fallback tokenizer initialization')
         end if
         decoder_config = neural_decoder_config_t(vocabulary_size=bpe%token_count, width=64_int32, heads=4_int32, &
                                                  local_window=32_int32, ffn_width=256_int32)
         call neural_decoder_init(model, decoder_config, 20261025_int64, status)
         call require_ok(status, 'continuous neural model initialization')
         train_config = neural_train_config_t(maximum_tokens_per_record=8192_int32)
         call neural_train_state_init(train_state)
         start_record = 0_int64
      end if
      if (command_argument_count() >= 6) then
         call get_command_argument(6, start_text)
         read(start_text, *, iostat=parse_status) start_record
         if (parse_status /= 0 .or. start_record < 0_int64) call usage_and_stop()
      end if
      call neural_train_metrics_reset(metrics)
      call neural_train_utf8_file(model, bpe, trim(corpus_path), train_config, train_state, metrics, status, &
                                  skip_source_records=start_record, maximum_new_source_records=record_limit)
      call require_ok(status, 'continuous UTF-8 byte-BPE chunk')
      call bpe_save(bpe, trim(tokenizer_path), status)
      call require_ok(status, 'continuous tokenizer save')
      call neural_checkpoint_save(trim(checkpoint_path), model, bpe, train_config, train_state, status)
      call require_ok(status, 'continuous checkpoint save')
      write(output_unit, '(A)') 'NEURAL_PRETRAIN_CHUNK_JSON: {'// &
         '"schema_version":"vegai-neural-pretraining-chunk/v1",'// &
         '"mode":"tied-output-cross-entropy-adapter/v1",'// &
         '"start_source_record":'//trim(i64_text(start_record))//','// &
         '"new_source_record_limit":'//trim(i64_text(record_limit))//','// &
         '"source_records_completed":'//trim(i64_text(train_state%source_records_completed))//','// &
         '"eligible_records":'//trim(i64_text(metrics%records))//','// &
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
      integer :: allocation_status

      if (command_argument_count() < 4) call usage_and_stop()
      call get_command_argument(2, checkpoint_path)
      call get_command_argument(3, tokenizer_path)
      call get_command_argument(4, prompt)
      max_generated = 32_int32
      random_state = 1_int64
      if (command_argument_count() >= 5) then
         call get_command_argument(5, count_text)
         read(count_text, *, iostat=parse_status) max_generated
         if (parse_status /= 0 .or. max_generated < 1_int32 .or. max_generated > 512_int32) call usage_and_stop()
      end if
      if (command_argument_count() >= 6) then
         call get_command_argument(6, seed_text)
         read(seed_text, *, iostat=parse_status) random_state
         if (parse_status /= 0) call usage_and_stop()
      end if
      initial_random_state = random_state
      call bpe_load(bpe, trim(tokenizer_path), status)
      call require_ok(status, 'generation tokenizer load')
      call neural_checkpoint_load(trim(checkpoint_path), bpe, model, train_config, train_state, status)
      call require_ok(status, 'generation model checkpoint load')
      call bpe_encode(bpe, prompt, int(len_trim(prompt), int32), prompt_tokens, prompt_count, status)
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
      write(output_unit, '(A)') '  vegai_neural_pretrain train  CORPUS CHECKPOINT TOKENIZER [EPOCHS]'
      write(output_unit, '(A)') '  vegai_neural_pretrain resume CORPUS CHECKPOINT TOKENIZER [EPOCHS]'
      write(output_unit, '(A)') '  vegai_neural_pretrain generate CHECKPOINT TOKENIZER PROMPT [TOKENS] [SEED]'
      write(output_unit, '(A)') '  vegai_neural_pretrain train-chunk CORPUS CHECKPOINT TOKENIZER NEW_RECORDS [START_RECORD]'
      write(output_unit, '(A)') 'The current trainer updates only tied output embeddings; it is not full BPTT.'
   end subroutine print_help

end program vegai_neural_pretrain
