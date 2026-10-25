! SPDX-License-Identifier: MIT
! Native inspection/self-test utility for the neural subword-retention design.
program vegai_neural_lm
   use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64, output_unit, error_unit
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use subword_bpe
   use neural_retention_decoder
   use calibrated_decoder
   implicit none

   character(len=64) :: command
   integer :: arguments

   arguments = command_argument_count()
   if (arguments < 1) call usage_and_stop()
   call get_command_argument(1, command)
   select case (trim(command))
   case ('self-test')
      call run_self_test()
   case ('plan')
      call print_default_plan()
   case ('help', '--help', '-h')
      call print_help()
   case default
      call usage_and_stop()
   end select

contains

   subroutine require_ok(status, label)
      integer(int32), intent(in) :: status
      character(len=*), intent(in) :: label

      if (status /= BPE_OK .and. status /= NRD_OK .and. status /= DECODER_OK) then
         write(error_unit, '(A,A,A,I0)') 'vegai_neural_lm: ', trim(label), ' failed status=', status
         error stop 1
      end if
   end subroutine require_ok

   subroutine require_true(condition, label)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: label

      if (.not. condition) then
         write(error_unit, '(A,A)') 'vegai_neural_lm self-test failed: ', trim(label)
         error stop 1
      end if
   end subroutine require_true

   subroutine run_self_test()
      type(bpe_model_t) :: bpe
      type(neural_decoder_t) :: model
      type(neural_decoder_state_t) :: state_a, state_b, state_c
      type(neural_decoder_config_t) :: config
      type(neural_resource_plan_t) :: plan
      type(decoding_policy_t) :: policy
      type(temperature_calibration_t) :: calibration
      character(len=64) :: input, decoded
      integer(int32) :: tokens(64), token_count, output_bytes, status, first_merge, second_merge, position
      integer(int32) :: history(8), selected
      integer(int64) :: random_state
      real(real32) :: hidden_a(16), hidden_b(16), logits_a(64), logits_b(64), prefix_logits(64)
      real(real64) :: decoder_logits(4), probabilities(4), calibration_logits(3, 4)
      integer(int32) :: targets(4)

      input = ''
      input = 'Türkçe'
      call bpe_init(bpe, 8_int32, status)
      call require_ok(status, 'BPE init')
      call bpe_add_merge(bpe, bpe_base_token_for_byte(int(iachar(input(1:1)), int32)), &
                          bpe_base_token_for_byte(int(iachar(input(2:2)), int32)), first_merge, status)
      call require_ok(status, 'BPE first UTF-8 merge')
      call bpe_add_merge(bpe, first_merge, bpe_base_token_for_byte(int(iachar(input(3:3)), int32)), second_merge, status)
      call require_ok(status, 'BPE chained UTF-8 merge')
      call bpe_encode(bpe, input, int(len_trim(input), int32), tokens, token_count, status)
      call require_ok(status, 'BPE encode')
      call bpe_decode(bpe, tokens, token_count, decoded, output_bytes, status)
      call require_ok(status, 'BPE decode')
      call require_true(output_bytes == len_trim(input) .and. decoded(:output_bytes) == input(:output_bytes), &
                        'BPE lossless Turkish UTF-8 round trip')

      config = neural_decoder_config_t(vocabulary_size=64_int32, width=16_int32, heads=2_int32, &
                                      local_window=4_int32, ffn_width=32_int32)
      call neural_decoder_init(model, config, 20261025_int64, status)
      call require_ok(status, 'neural decoder init')
      call neural_decoder_state_init(state_a, model, status)
      call require_ok(status, 'neural state A init')
      call neural_decoder_state_init(state_b, model, status)
      call require_ok(status, 'neural state B init')
      call neural_decoder_state_init(state_c, model, status)
      call require_ok(status, 'neural state C init')
      do position = 1_int32, 4_int32
         call neural_decoder_step(model, state_a, position, hidden_a, logits_a, status)
         call require_ok(status, 'neural full-prefix step')
      end do
      call neural_decoder_step(model, state_b, 1_int32, hidden_b, logits_b, status)
      call require_ok(status, 'neural streamed step 1')
      call neural_decoder_step(model, state_b, 2_int32, hidden_b, logits_b, status)
      call require_ok(status, 'neural streamed step 2')
      call neural_decoder_step(model, state_b, 3_int32, hidden_b, logits_b, status)
      call require_ok(status, 'neural streamed step 3')
      call neural_decoder_step(model, state_b, 4_int32, hidden_b, logits_b, status)
      call require_ok(status, 'neural streamed step 4')
      call require_true(maxval(abs(state_a%retention_numerator - state_b%retention_numerator)) < 1.0e-6_real32, &
                        'retention recurrence is chunk-equivalent')
      call require_true(maxval(abs(logits_a - logits_b)) < 1.0e-6_real32, 'streamed logits equal monolithic prefix logits')
      call neural_decoder_step(model, state_c, 1_int32, hidden_b, logits_b, status)
      call require_ok(status, 'causal prefix step 1')
      call neural_decoder_step(model, state_c, 2_int32, hidden_b, prefix_logits, status)
      call require_ok(status, 'causal prefix step 2')
      call neural_decoder_step(model, state_c, 63_int32, hidden_b, logits_b, status)
      call require_ok(status, 'causal future token step')
      call require_true(all(ieee_is_finite(prefix_logits)) .and. all(ieee_is_finite(logits_b)), &
                        'causal logits are finite before and after future append')
      call neural_decoder_plan(config, plan, status)
      call require_ok(status, 'neural resource plan')
      call require_true(plan%parameter_count > 0_int64 .and. plan%per_stream_state_bytes_real32 > 0_int64, &
                        'neural resource plan is positive')

      decoder_logits = [7.0_real64, 2.0_real64, 1.0_real64, 0.0_real64]
      history = [1_int32, 2_int32, 1_int32, 2_int32, 0_int32, 0_int32, 0_int32, 0_int32]
      policy = decoding_policy_t(temperature=1.0_real64, frequency_penalty=0.0_real64, presence_penalty=0.0_real64, &
                                 immediate_repeat_penalty=0.0_real64, no_repeat_ngram=3_int32, top_k=4_int32, top_p=1.0_real64)
      call decoder_distribution(decoder_logits, history, 4_int32, policy, probabilities, status)
      call require_ok(status, 'calibrated distribution')
      call require_true(abs(sum(probabilities) - 1.0_real64) < 1.0e-12_real64, 'decoder distribution normalizes exactly')
      call require_true(probabilities(1) == 0.0_real64, 'no-repeat trigram masks repeated high-logit token')
      random_state = 99_int64
      call decoder_select(probabilities, policy, random_state, selected, status)
      call require_ok(status, 'calibrated selection')
      call require_true(selected /= 1_int32, 'decoder selection honors no-repeat mask')

      calibration_logits(:, 1) = [2.0_real64, 0.0_real64, -1.0_real64]
      calibration_logits(:, 2) = [1.5_real64, 0.0_real64, -1.0_real64]
      calibration_logits(:, 3) = [0.5_real64, 0.0_real64, -1.0_real64]
      calibration_logits(:, 4) = [1.0_real64, 0.0_real64, -1.0_real64]
      targets = [1_int32, 1_int32, 2_int32, 1_int32]
      call decoder_temperature_calibrate(calibration_logits, targets, 4_int32, calibration, status)
      call require_ok(status, 'temperature calibration')
      call require_true(calibration%mean_nll_after <= calibration%mean_nll_before + 1.0e-12_real64, &
                        'calibration retains non-worse held-out NLL')

      write(output_unit, '(A)') 'NEURAL_SELF_TEST_JSON: {'// &
         '"schema_version":"vegai-neural-retention-self-test/v1",'// &
         '"bpe_roundtrip":true,'// &
         '"causal_chunk_equivalence":true,'// &
         '"decoder_normalization":true,'// &
         '"calibration_non_worse":true,'// &
         '"parameters":'//trim(i64_text(plan%parameter_count))//','// &
         '"state_bytes_real32":'//trim(i64_text(plan%per_stream_state_bytes_real32))//'}'
   end subroutine run_self_test

   subroutine print_default_plan()
      type(neural_decoder_config_t) :: config
      type(neural_resource_plan_t) :: plan
      integer(int32) :: status

      config = neural_decoder_config_t()
      call neural_decoder_plan(config, plan, status)
      call require_ok(status, 'default neural plan')
      write(output_unit, '(A)') 'NEURAL_PLAN_JSON: {'// &
         '"schema_version":"vegai-neural-retention-plan/v1",'// &
         '"vocabulary_size":'//trim(i32_text(config%vocabulary_size))//','// &
         '"width":'//trim(i32_text(config%width))//','// &
         '"heads":'//trim(i32_text(config%heads))//','// &
         '"local_window":'//trim(i32_text(config%local_window))//','// &
         '"parameters":'//trim(i64_text(plan%parameter_count))//','// &
         '"parameter_bytes_real32":'//trim(i64_text(plan%parameter_bytes_real32))//','// &
         '"per_stream_state_bytes_real32":'//trim(i64_text(plan%per_stream_state_bytes_real32))//','// &
         '"approximate_forward_flops_per_token":'//trim(i64_text(plan%approximate_forward_flops_per_token))//'}'
   end subroutine print_default_plan

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

   subroutine usage_and_stop()
      call print_help()
      error stop 2
   end subroutine usage_and_stop

   subroutine print_help()
      write(output_unit, '(A)') 'Usage:'
      write(output_unit, '(A)') '  vegai_neural_lm self-test'
      write(output_unit, '(A)') '  vegai_neural_lm plan'
      write(output_unit, '(A)') 'Native reference kernel for byte-BPE, causal hybrid retention attention, and calibrated decoding.'
   end subroutine print_help

end program vegai_neural_lm
