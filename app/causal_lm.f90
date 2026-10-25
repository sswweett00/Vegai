! SPDX-License-Identifier: MIT
! Native command-line interface for Vegai's bounded causal Turkish n-gram LM.
program vegai_causal_lm
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64, output_unit, error_unit
   use causal_ngram_lm
   implicit none

   character(len=512) :: command, corpus_path, checkpoint_path, prompt, value
   type(causal_lm_t) :: model
   type(causal_lm_config_t) :: config
   type(causal_lm_metrics_t) :: vocabulary_metrics, training_metrics
   type(causal_lm_evaluation_t) :: evaluation
   character(len=4096) :: completion
   integer(int32) :: status, max_tokens
   integer(int64) :: seed, epoch_clock_start, epoch_clock_end, epoch_clock_rate, epoch_clock_max, epoch_clock_delta
   real(real64) :: temperature, epoch_seconds
   integer :: argument_count

   argument_count = command_argument_count()
   if (argument_count < 1) call usage_and_stop()
   call get_command_argument(1, command)

   select case (trim(command))
   case ('train')
      if (argument_count < 3) call usage_and_stop()
      call get_command_argument(2, corpus_path)
      call get_command_argument(3, checkpoint_path)
      config = causal_lm_config_t()
      call parse_train_options(4, argument_count, config, status)
      if (status /= CLM_OK) call fail('invalid train option', status)
      call causal_lm_init(model, config, status)
      if (status /= CLM_OK) call fail('causal model allocation failed', status)
      call causal_lm_build_vocabulary(model, trim(corpus_path), vocabulary_metrics, status)
      if (status /= CLM_OK) call fail('causal vocabulary pass failed', status)
      ! This timer is deliberately adjacent to the exactly-one causal update
      ! pass. Vocabulary selection, checkpoint serialization, CLI startup,
      ! compilation, corpus construction, held-out evaluation, and CI work are
      ! all outside its scope.
      call system_clock(count=epoch_clock_start, count_rate=epoch_clock_rate, count_max=epoch_clock_max)
      if (epoch_clock_rate <= 0_int64) call fail('native training clock is unavailable', CLM_ERR_IO)
      call causal_lm_train_file(model, trim(corpus_path), training_metrics, status)
      call system_clock(count=epoch_clock_end)
      if (status /= CLM_OK) call fail('causal training epoch failed', status)
      if (epoch_clock_end >= epoch_clock_start) then
         epoch_clock_delta = epoch_clock_end - epoch_clock_start
      else
         epoch_clock_delta = epoch_clock_max - epoch_clock_start + 1_int64 + epoch_clock_end
      end if
      epoch_seconds = real(epoch_clock_delta, real64) / real(epoch_clock_rate, real64)
      if (epoch_seconds <= 0.0_real64) call fail('native training clock did not advance', CLM_ERR_IO)
      call causal_lm_save(model, trim(checkpoint_path), status)
      if (status /= CLM_OK) call fail('causal checkpoint save failed', status)
      call print_train_report(config, vocabulary_metrics, training_metrics, causal_lm_model_bytes(model), epoch_seconds)

   case ('generate')
      if (argument_count < 3) call usage_and_stop()
      call get_command_argument(2, checkpoint_path)
      call get_command_argument(3, prompt)
      max_tokens = 48_int32
      temperature = 0.80_real64
      seed = 20261025_int64
      call parse_generate_options(4, argument_count, max_tokens, temperature, seed, status)
      if (status /= CLM_OK) call fail('invalid generate option', status)
      call causal_lm_load(trim(checkpoint_path), model, status)
      if (status /= CLM_OK) call fail('causal checkpoint load failed', status)
      call causal_lm_generate(model, trim(prompt), max_tokens, temperature, seed, completion, status)
      if (status /= CLM_OK) call fail('causal generation failed', status)
      write(output_unit, '(A)') 'CAUSAL_PROMPT: '//trim(prompt)
      write(output_unit, '(A)') 'CAUSAL_RESPONSE: '//trim(completion)

   case ('evaluate')
      if (argument_count /= 3) call usage_and_stop()
      call get_command_argument(2, checkpoint_path)
      call get_command_argument(3, corpus_path)
      call causal_lm_load(trim(checkpoint_path), model, status)
      if (status /= CLM_OK) call fail('causal checkpoint load failed', status)
      call causal_lm_evaluate_file(model, trim(corpus_path), evaluation, status)
      if (status /= CLM_OK) call fail('causal held-out evaluation failed', status)
      call print_evaluation_report(evaluation)

   case ('inspect')
      if (argument_count /= 2) call usage_and_stop()
      call get_command_argument(2, checkpoint_path)
      call causal_lm_load(trim(checkpoint_path), model, status)
      if (status /= CLM_OK) call fail('causal checkpoint load failed', status)
      write(output_unit, '(A)') 'CAUSAL_LM_JSON: {'// &
         '"schema_version":"vegai-causal-ngram-report/v1",'// &
         '"active_vocabulary":'//trim(i32_text(model%active_vocabulary))//','// &
         '"max_vocabulary":'//trim(i32_text(model%config%max_vocabulary))//','// &
         '"successors_per_context":'//trim(i32_text(model%config%successors_per_context))//','// &
         '"trigram_buckets":'//trim(i32_text(model%config%trigram_buckets))//','// &
         '"trigram_ways":'//trim(i32_text(model%config%trigram_ways))//','// &
         '"model_bytes":'//trim(i64_text(causal_lm_model_bytes(model)))//'}'

   case ('help', '--help', '-h')
      call print_help()

   case default
      call usage_and_stop()
   end select

contains

   subroutine parse_train_options(first, last, parsed, parse_status)
      integer, intent(in) :: first, last
      type(causal_lm_config_t), intent(inout) :: parsed
      integer(int32), intent(out) :: parse_status
      integer :: index, io_status
      character(len=512) :: option, option_value

      parse_status = CLM_OK
      index = first
      do while (index <= last)
         if (index == last) then
            parse_status = CLM_ERR_INVALID_ARGUMENT
            return
         end if
         call get_command_argument(index, option)
         call get_command_argument(index + 1, option_value)
         io_status = 0
         select case (trim(option))
         case ('--max-vocab')
            read(option_value, *, iostat=io_status) parsed%max_vocabulary
         case ('--top-successors')
            read(option_value, *, iostat=io_status) parsed%successors_per_context
         case ('--trigram-buckets')
            read(option_value, *, iostat=io_status) parsed%trigram_buckets
         case ('--trigram-ways')
            read(option_value, *, iostat=io_status) parsed%trigram_ways
         case ('--min-estimated-frequency')
            read(option_value, *, iostat=io_status) parsed%min_estimated_frequency
         case default
            io_status = 1
         end select
         if (io_status /= 0) then
            parse_status = CLM_ERR_INVALID_ARGUMENT
            return
         end if
         index = index + 2
      end do
   end subroutine parse_train_options

   subroutine parse_generate_options(first, last, parsed_tokens, parsed_temperature, parsed_seed, parse_status)
      integer, intent(in) :: first, last
      integer(int32), intent(inout) :: parsed_tokens
      real(real64), intent(inout) :: parsed_temperature
      integer(int64), intent(inout) :: parsed_seed
      integer(int32), intent(out) :: parse_status
      integer :: index, io_status
      character(len=512) :: option, option_value

      parse_status = CLM_OK
      index = first
      do while (index <= last)
         if (index == last) then
            parse_status = CLM_ERR_INVALID_ARGUMENT
            return
         end if
         call get_command_argument(index, option)
         call get_command_argument(index + 1, option_value)
         io_status = 0
         select case (trim(option))
         case ('--tokens')
            read(option_value, *, iostat=io_status) parsed_tokens
         case ('--temperature')
            read(option_value, *, iostat=io_status) parsed_temperature
         case ('--seed')
            read(option_value, *, iostat=io_status) parsed_seed
         case default
            io_status = 1
         end select
         if (io_status /= 0) then
            parse_status = CLM_ERR_INVALID_ARGUMENT
            return
         end if
         index = index + 2
      end do
      if (parsed_tokens <= 0_int32 .or. parsed_tokens > 512_int32 .or. parsed_temperature <= 0.0_real64) then
         parse_status = CLM_ERR_INVALID_ARGUMENT
      end if
   end subroutine parse_generate_options

   subroutine print_train_report(train_config, vocabulary, training, model_bytes, training_epoch_seconds)
      type(causal_lm_config_t), intent(in) :: train_config
      type(causal_lm_metrics_t), intent(in) :: vocabulary, training
      integer(int64), intent(in) :: model_bytes
      real(real64), intent(in) :: training_epoch_seconds

      write(output_unit, '(A,I0)') 'vocabulary pass raw tokens: ', vocabulary%raw_tokens
      write(output_unit, '(A,I0)') 'active vocabulary: ', training%active_vocabulary
      write(output_unit, '(A,I0)') 'causal training raw tokens: ', training%raw_tokens
      write(output_unit, '(A,I0)') 'causal known training tokens: ', training%known_training_tokens
      write(output_unit, '(A,I0)') 'causal unknown training tokens: ', training%unknown_training_tokens
      write(output_unit, '(A,I0)') 'causal bigram events: ', training%bigram_events
      write(output_unit, '(A,I0)') 'causal trigram events: ', training%trigram_events
      write(output_unit, '(A)') 'CAUSAL_LM_JSON: {'// &
         '"schema_version":"vegai-causal-ngram-report/v1",'// &
         '"phase":"train",'// &
         '"corpus_bytes":'//trim(i64_text(training%input_bytes))//','// &
         '"raw_tokens":'//trim(i64_text(training%raw_tokens))//','// &
         '"known_training_tokens":'//trim(i64_text(training%known_training_tokens))//','// &
         '"unknown_training_tokens":'//trim(i64_text(training%unknown_training_tokens))//','// &
         '"bigram_events":'//trim(i64_text(training%bigram_events))//','// &
         '"trigram_events":'//trim(i64_text(training%trigram_events))//','// &
         '"training_epoch_wall_seconds":'//trim(real_text(training_epoch_seconds))//','// &
         '"active_vocabulary":'//trim(i32_text(int(training%active_vocabulary, int32)))//','// &
         '"max_vocabulary":'//trim(i32_text(train_config%max_vocabulary))//','// &
         '"successors_per_context":'//trim(i32_text(train_config%successors_per_context))//','// &
         '"trigram_buckets":'//trim(i32_text(train_config%trigram_buckets))//','// &
         '"trigram_ways":'//trim(i32_text(train_config%trigram_ways))//','// &
         '"model_bytes":'//trim(i64_text(model_bytes))//'}'
   end subroutine print_train_report

   subroutine print_evaluation_report(result)
      type(causal_lm_evaluation_t), intent(in) :: result

      write(output_unit, '(A)') 'CAUSAL_LM_JSON: {'// &
         '"schema_version":"vegai-causal-ngram-report/v1",'// &
         '"phase":"evaluate",'// &
         '"records":'//trim(i64_text(result%records))//','// &
         '"eligible_transitions":'//trim(i64_text(result%eligible_transitions))//','// &
         '"top1_hits":'//trim(i64_text(result%top1_hits))//','// &
         '"successor_coverage_hits":'//trim(i64_text(result%successor_coverage_hits))//','// &
         '"trigram_context_hits":'//trim(i64_text(result%trigram_context_hits))//','// &
         '"bigram_context_hits":'//trim(i64_text(result%bigram_context_hits))//','// &
         '"top1_accuracy":'//trim(real_text(result%top1_accuracy))//','// &
         '"successor_coverage":'//trim(real_text(result%successor_coverage))//'}'
   end subroutine print_evaluation_report

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

   function real_text(value) result(rendered)
      real(real64), intent(in) :: value
      character(len=48) :: rendered
      write(rendered, '(ES24.16E3)') value
   end function real_text

   subroutine fail(label, failure_status)
      character(len=*), intent(in) :: label
      integer(int32), intent(in) :: failure_status
      write(error_unit, '(A,A,A,I0)') 'vegai_causal_lm: ', trim(label), ' (status ', failure_status
      error stop 1
   end subroutine fail

   subroutine usage_and_stop()
      call print_help()
      error stop 2
   end subroutine usage_and_stop

   subroutine print_help()
      write(output_unit, '(A)') 'Usage:'
      write(output_unit, '(A)') '  vegai_causal_lm train CORPUS CHECKPOINT [--max-vocab N --top-successors N --trigram-buckets N --trigram-ways N --min-estimated-frequency N]'
      write(output_unit, '(A)') '  vegai_causal_lm generate CHECKPOINT "Türkçe istem" [--tokens N --temperature X --seed N]'
      write(output_unit, '(A)') '  vegai_causal_lm evaluate CHECKPOINT HELDOUT_CORPUS'
      write(output_unit, '(A)') '  vegai_causal_lm inspect CHECKPOINT'
      write(output_unit, '(A)') 'The learner is a bounded causal heavy-hitter n-gram model, not an HDC retriever or a neural LLM.'
   end subroutine print_help

end program vegai_causal_lm
