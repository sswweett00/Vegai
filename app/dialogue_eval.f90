! SPDX-License-Identifier: MIT
!==============================================================================
!  dialogue_eval.f90
!  Native batch evaluator for bounded grounded-dialogue retrieval.
!
!  This executable loads one checkpoint/TSV index once, then evaluates a
!  strict three-column UTF-8 TSV case stream. It is deliberately separate from
!  the Python privacy/provenance evaluator: it provides efficient native batch
!  measurement for calibrated source-disjoint dialogue evidence without
!  claiming response generation or semantic truth.
!==============================================================================
program vegai_dialogue_eval
   use, intrinsic :: iso_fortran_env, only: int32, int64, output_unit, error_unit, iostat_end
   use hdc_pretraining, only: PT_OK, hdc_pretrain_model_t, pretrain_checkpoint_load, pretrain_model_destroy
   use dialogue_retrieval
   use fortran_integrity, only: FI_OK, sha256_file
   implicit none

   integer(int32), parameter :: EVAL_OK = 0_int32
   integer(int32), parameter :: EVAL_ERR_INVALID_ARGUMENT = -760_int32
   integer(int32), parameter :: EVAL_ERR_IO = -761_int32
   integer(int32), parameter :: EVAL_ERR_FORMAT = -762_int32
   integer(int32), parameter :: EVAL_ERR_RECORD_TOO_LONG = -763_int32
   integer(int32), parameter :: EVAL_ERR_TOO_MANY_CASES = -764_int32
   integer(int32), parameter :: EVAL_ERR_DIALOGUE = -765_int32
   integer(int32), parameter :: EVAL_MAX_CASE_ID_BYTES = 256_int32
   integer(int32), parameter :: EVAL_MAX_RECORD_BYTES = 8192_int32

   type :: file_identity_t
      character(len=64) :: sha256 = ''
      integer(int64) :: bytes = 0_int64
   end type file_identity_t

   type :: evaluation_identities_t
      type(file_identity_t) :: checkpoint
      type(file_identity_t) :: knowledge
      type(file_identity_t) :: evaluation
   end type evaluation_identities_t

   type :: evaluation_metrics_t
      integer(int64) :: total_cases = 0_int64
      integer(int64) :: expected_grounded_cases = 0_int64
      integer(int64) :: expected_abstain_cases = 0_int64
      integer(int64) :: actual_grounded_cases = 0_int64
      integer(int64) :: rendered_abstain_cases = 0_int64
      integer(int64) :: correct_grounded_prompt_cases = 0_int64
      integer(int64) :: wrong_grounded_prompt_cases = 0_int64
      integer(int64) :: false_abstain_cases = 0_int64
      integer(int64) :: false_grounded_cases = 0_int64
      integer(int64) :: true_abstain_cases = 0_int64
      integer(int64) :: oov_rejected_abstain_cases = 0_int64
      integer(int64) :: oov_rejected_expected_grounded_cases = 0_int64
   end type evaluation_metrics_t

   character(len=32) :: command
   character(len=512) :: checkpoint_path, knowledge_path, evaluation_path
   type(hdc_pretrain_model_t) :: model
   type(dialogue_config_t) :: config
   type(dialogue_index_t) :: index
   type(dialogue_session_t) :: session
   type(evaluation_metrics_t) :: metrics
   type(evaluation_identities_t) :: identities
   integer(int32) :: status, entries_loaded
   integer(int64) :: maximum_cases
   integer :: argument_count

   argument_count = command_argument_count()
   if (argument_count < 1) call fail_usage()
   call read_cli_argument(1, command, status)
   if (status /= EVAL_OK) call fail_usage()

   select case (trim(command))
   case ('evaluate')
      if (argument_count < 5) call fail_usage()
      call read_cli_argument(2, checkpoint_path, status)
      if (status /= EVAL_OK) call fail_usage()
      call read_cli_argument(3, knowledge_path, status)
      if (status /= EVAL_OK) call fail_usage()
      call read_cli_argument(4, evaluation_path, status)
      if (status /= EVAL_OK) call fail_usage()
      config = dialogue_config_t()
      ! Batch evidence must be stateless: no source-file ordering can change a
      ! score through conversational history.
      config%max_history_turns = 0_int32
      maximum_cases = 100000_int64
      call parse_options(5, argument_count, config, maximum_cases, status)
      if (status /= EVAL_OK) call fail_usage()
      call load_dialogue(trim(checkpoint_path), trim(knowledge_path), config, model, index, session, entries_loaded, status)
      if (status /= EVAL_OK) call fail_with_status('dialogue initialization failed', status)
      call evaluate_tsv(trim(evaluation_path), index, model, session, maximum_cases, metrics, status)
      if (status /= EVAL_OK) then
         call cleanup(model, index, session)
         call fail_with_status('native dialogue evaluation failed', status)
      end if
      call collect_identities(trim(checkpoint_path), trim(knowledge_path), trim(evaluation_path), identities, status)
      if (status /= EVAL_OK) then
         call cleanup(model, index, session)
         call fail_with_status('native dialogue identity collection failed', status)
      end if
      call print_json_report(config, entries_loaded, maximum_cases, identities, metrics)
      call cleanup(model, index, session)

   case ('help', '--help', '-h')
      call print_usage()

   case default
      call fail_usage()
   end select

contains

   subroutine cleanup(active_model, active_index, active_session)
      type(hdc_pretrain_model_t), intent(inout) :: active_model
      type(dialogue_index_t), intent(inout) :: active_index
      type(dialogue_session_t), intent(inout) :: active_session

      call dialogue_session_destroy(active_session)
      call dialogue_index_destroy(active_index)
      call pretrain_model_destroy(active_model)
   end subroutine cleanup

   subroutine read_cli_argument(position, value, read_status)
      integer, intent(in) :: position
      character(len=*), intent(out) :: value
      integer(int32), intent(out) :: read_status
      integer :: argument_length, command_status

      value = ''
      read_status = EVAL_OK
      call get_command_argument(position, length=argument_length, status=command_status)
      if (command_status /= 0 .or. argument_length < 0 .or. argument_length > len(value)) then
         read_status = EVAL_ERR_INVALID_ARGUMENT
         return
      end if
      call get_command_argument(position, value=value, status=command_status)
      if (command_status /= 0) read_status = EVAL_ERR_INVALID_ARGUMENT
   end subroutine read_cli_argument

   subroutine load_dialogue(requested_checkpoint, requested_knowledge, requested_config, active_model, active_index, &
                            active_session, loaded_entries, load_status)
      character(len=*), intent(in) :: requested_checkpoint, requested_knowledge
      type(dialogue_config_t), intent(in) :: requested_config
      type(hdc_pretrain_model_t), intent(inout) :: active_model
      type(dialogue_index_t), intent(inout) :: active_index
      type(dialogue_session_t), intent(inout) :: active_session
      integer(int32), intent(out) :: loaded_entries, load_status

      loaded_entries = 0_int32
      call pretrain_checkpoint_load(requested_checkpoint, active_model, load_status)
      if (load_status /= PT_OK) then
         load_status = DIALOGUE_ERR_MODEL
         return
      end if
      call dialogue_index_init(active_index, requested_config, load_status)
      if (load_status /= DIALOGUE_OK) return
      call dialogue_index_load_tsv(active_index, active_model, requested_knowledge, loaded_entries, load_status)
      if (load_status /= DIALOGUE_OK) return
      call dialogue_session_init(active_session, 0_int32, load_status)
   end subroutine load_dialogue

   subroutine collect_file_identity(path, identity, identity_status)
      character(len=*), intent(in) :: path
      type(file_identity_t), intent(out) :: identity
      integer(int32), intent(out) :: identity_status

      identity = file_identity_t()
      call sha256_file(path, identity%sha256, identity%bytes, identity_status)
      if (identity_status /= FI_OK) identity_status = EVAL_ERR_IO
   end subroutine collect_file_identity

   subroutine collect_identities(checkpoint, knowledge, evaluation, collected, identity_status)
      character(len=*), intent(in) :: checkpoint, knowledge, evaluation
      type(evaluation_identities_t), intent(out) :: collected
      integer(int32), intent(out) :: identity_status

      collected = evaluation_identities_t()
      call collect_file_identity(checkpoint, collected%checkpoint, identity_status)
      if (identity_status /= EVAL_OK) return
      call collect_file_identity(knowledge, collected%knowledge, identity_status)
      if (identity_status /= EVAL_OK) return
      call collect_file_identity(evaluation, collected%evaluation, identity_status)
   end subroutine collect_identities

   subroutine parse_options(first_argument, last_argument, parsed_config, parsed_maximum_cases, parse_status)
      integer, intent(in) :: first_argument, last_argument
      type(dialogue_config_t), intent(inout) :: parsed_config
      integer(int64), intent(inout) :: parsed_maximum_cases
      integer(int32), intent(out) :: parse_status
      character(len=1024) :: option, value
      integer :: index, io_status

      parse_status = EVAL_OK
      index = first_argument
      do while (index <= last_argument)
         call read_cli_argument(index, option, parse_status)
         if (parse_status /= EVAL_OK) return
         select case (trim(option))
         case ('--require-all-known')
            parsed_config%require_all_prompt_tokens_known = .true.
            index = index + 1
         case ('--max-entries', '--max-cases', '--min-known-tokens', '--min-similarity', '--min-margin', &
               '--semantic-weight', '--lexical-weight', '--min-hybrid-score', '--min-hybrid-margin')
            if (index == last_argument) then
               parse_status = EVAL_ERR_INVALID_ARGUMENT
               return
            end if
            call read_cli_argument(index + 1, value, parse_status)
            if (parse_status /= EVAL_OK) return
            select case (trim(option))
            case ('--max-entries')
               read(value, *, iostat=io_status) parsed_config%max_entries
            case ('--max-cases')
               read(value, *, iostat=io_status) parsed_maximum_cases
            case ('--min-known-tokens')
               read(value, *, iostat=io_status) parsed_config%min_known_tokens
            case ('--min-similarity')
               read(value, *, iostat=io_status) parsed_config%minimum_similarity
            case ('--min-margin')
               read(value, *, iostat=io_status) parsed_config%minimum_margin
            case ('--semantic-weight')
               read(value, *, iostat=io_status) parsed_config%semantic_weight
            case ('--lexical-weight')
               read(value, *, iostat=io_status) parsed_config%lexical_weight
            case ('--min-hybrid-score')
               read(value, *, iostat=io_status) parsed_config%minimum_hybrid_score
            case ('--min-hybrid-margin')
               read(value, *, iostat=io_status) parsed_config%minimum_hybrid_margin
            end select
            if (io_status /= 0) then
               parse_status = EVAL_ERR_INVALID_ARGUMENT
               return
            end if
            index = index + 2
         case default
            parse_status = EVAL_ERR_INVALID_ARGUMENT
            return
         end select
      end do
      if (parsed_maximum_cases <= 0_int64) parse_status = EVAL_ERR_INVALID_ARGUMENT
   end subroutine parse_options

   subroutine read_evaluation_record(unit, line, eof, overlong, read_status)
      integer, intent(in) :: unit
      character(len=*), intent(out) :: line
      logical, intent(out) :: eof, overlong
      integer(int32), intent(out) :: read_status
      character(len=1) :: byte
      integer :: io_status, line_length

      line = ''
      eof = .false.
      overlong = .false.
      read_status = EVAL_OK
      line_length = 0
      do
         read(unit, iostat=io_status) byte
         if (io_status == iostat_end) then
            if (line_length == 0 .and. .not. overlong) eof = .true.
            return
         end if
         if (io_status /= 0) then
            read_status = EVAL_ERR_IO
            return
         end if
         if (byte == achar(10)) return
         if (overlong) cycle
         if (line_length >= len(line)) then
            overlong = .true.
            cycle
         end if
         line_length = line_length + 1
         line(line_length:line_length) = byte
      end do
   end subroutine read_evaluation_record

   subroutine parse_evaluation_row(line, case_id, prompt, expected_prompt, expected_abstain, row_status)
      character(len=*), intent(in) :: line
      character(len=EVAL_MAX_CASE_ID_BYTES), intent(out) :: case_id
      character(len=DIALOGUE_MAX_PROMPT_CHARS), intent(out) :: prompt, expected_prompt
      logical, intent(out) :: expected_abstain
      integer(int32), intent(out) :: row_status
      integer :: line_length, first_separator, second_separator, position

      case_id = ''
      prompt = ''
      expected_prompt = ''
      expected_abstain = .false.
      row_status = EVAL_OK
      line_length = len_trim(line)
      if (line_length <= 0) then
         row_status = EVAL_ERR_FORMAT
         return
      end if
      do position = 1, line_length
         if (iachar(line(position:position)) < 32 .and. line(position:position) /= achar(9)) then
            row_status = EVAL_ERR_FORMAT
            return
         end if
      end do
      first_separator = index(line(:line_length), achar(9))
      if (first_separator <= 1 .or. first_separator >= line_length) then
         row_status = EVAL_ERR_FORMAT
         return
      end if
      second_separator = index(line(first_separator + 1:line_length), achar(9))
      if (second_separator <= 0) then
         row_status = EVAL_ERR_FORMAT
         return
      end if
      second_separator = first_separator + second_separator
      if (second_separator <= first_separator + 1) then
         row_status = EVAL_ERR_FORMAT
         return
      end if
      if (second_separator < line_length) then
         if (index(line(second_separator + 1:line_length), achar(9)) /= 0) then
            row_status = EVAL_ERR_FORMAT
            return
         end if
      end if
      if (first_separator - 1 > EVAL_MAX_CASE_ID_BYTES .or. &
          second_separator - first_separator - 1 > DIALOGUE_MAX_PROMPT_CHARS .or. &
          line_length - second_separator > DIALOGUE_MAX_PROMPT_CHARS) then
         row_status = EVAL_ERR_RECORD_TOO_LONG
         return
      end if
      case_id = line(:first_separator - 1)
      prompt = line(first_separator + 1:second_separator - 1)
      if (len_trim(case_id) == 0 .or. len_trim(prompt) == 0) then
         row_status = EVAL_ERR_FORMAT
         return
      end if
      if (second_separator == line_length) then
         expected_abstain = .true.
      else
         expected_prompt = line(second_separator + 1:line_length)
         if (len_trim(expected_prompt) == 0) then
            expected_abstain = .true.
         end if
      end if
   end subroutine parse_evaluation_row

   subroutine evaluate_tsv(path, active_index, active_model, active_session, maximum_cases, result, evaluation_status)
      character(len=*), intent(in) :: path
      type(dialogue_index_t), intent(in) :: active_index
      type(hdc_pretrain_model_t), intent(in) :: active_model
      type(dialogue_session_t), intent(inout) :: active_session
      integer(int64), intent(in) :: maximum_cases
      type(evaluation_metrics_t), intent(out) :: result
      integer(int32), intent(out) :: evaluation_status
      character(len=EVAL_MAX_RECORD_BYTES) :: line
      character(len=EVAL_MAX_CASE_ID_BYTES) :: case_id
      character(len=DIALOGUE_MAX_PROMPT_CHARS) :: prompt, expected_prompt
      type(dialogue_answer_t) :: answer
      logical :: eof, overlong, expected_abstain
      integer :: unit, io_status
      integer(int32) :: dialogue_status

      result = evaluation_metrics_t()
      evaluation_status = EVAL_OK
      if (len_trim(path) == 0) then
         evaluation_status = EVAL_ERR_INVALID_ARGUMENT
         return
      end if
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', form='unformatted', &
           iostat=io_status)
      if (io_status /= 0) then
         evaluation_status = EVAL_ERR_IO
         return
      end if
      do
         call read_evaluation_record(unit, line, eof, overlong, evaluation_status)
         if (evaluation_status /= EVAL_OK .or. eof) exit
         if (overlong) then
            evaluation_status = EVAL_ERR_RECORD_TOO_LONG
            exit
         end if
         call parse_evaluation_row(line, case_id, prompt, expected_prompt, expected_abstain, evaluation_status)
         if (evaluation_status /= EVAL_OK) exit
         result%total_cases = result%total_cases + 1_int64
         if (result%total_cases > maximum_cases) then
            evaluation_status = EVAL_ERR_TOO_MANY_CASES
            exit
         end if
         call dialogue_ask(active_index, active_model, active_session, prompt, answer, dialogue_status)
         if (dialogue_status == DIALOGUE_ERR_NO_KNOWN_TOKENS) then
            if (expected_abstain) then
               result%expected_abstain_cases = result%expected_abstain_cases + 1_int64
               result%oov_rejected_abstain_cases = result%oov_rejected_abstain_cases + 1_int64
            else
               result%expected_grounded_cases = result%expected_grounded_cases + 1_int64
               result%false_abstain_cases = result%false_abstain_cases + 1_int64
               result%oov_rejected_expected_grounded_cases = result%oov_rejected_expected_grounded_cases + 1_int64
            end if
            cycle
         end if
         if (dialogue_status /= DIALOGUE_OK) then
            evaluation_status = EVAL_ERR_DIALOGUE
            exit
         end if
         if (expected_abstain) then
            result%expected_abstain_cases = result%expected_abstain_cases + 1_int64
            if (answer%found) then
               result%actual_grounded_cases = result%actual_grounded_cases + 1_int64
               result%false_grounded_cases = result%false_grounded_cases + 1_int64
            else
               result%rendered_abstain_cases = result%rendered_abstain_cases + 1_int64
               result%true_abstain_cases = result%true_abstain_cases + 1_int64
            end if
         else
            result%expected_grounded_cases = result%expected_grounded_cases + 1_int64
            if (.not. answer%found) then
               result%rendered_abstain_cases = result%rendered_abstain_cases + 1_int64
               result%false_abstain_cases = result%false_abstain_cases + 1_int64
            else if (trim(answer%matched_prompt) == trim(expected_prompt)) then
               result%actual_grounded_cases = result%actual_grounded_cases + 1_int64
               result%correct_grounded_prompt_cases = result%correct_grounded_prompt_cases + 1_int64
            else
               result%actual_grounded_cases = result%actual_grounded_cases + 1_int64
               result%wrong_grounded_prompt_cases = result%wrong_grounded_prompt_cases + 1_int64
            end if
         end if
      end do
      close(unit)
      if (evaluation_status == EVAL_OK .and. result%total_cases == 0_int64) evaluation_status = EVAL_ERR_FORMAT
   end subroutine evaluate_tsv

   subroutine write_json_count(name, value, trailing_comma)
      character(len=*), intent(in) :: name
      integer(int64), intent(in) :: value
      logical, intent(in) :: trailing_comma

      if (trailing_comma) then
         write(output_unit, '(A,A,A,I0,A)') '    "', trim(name), '": ', value, ','
      else
         write(output_unit, '(A,A,A,I0)') '    "', trim(name), '": ', value
      end if
   end subroutine write_json_count

   subroutine write_identity_json(name, identity, trailing_comma)
      character(len=*), intent(in) :: name
      type(file_identity_t), intent(in) :: identity
      logical, intent(in) :: trailing_comma

      if (trailing_comma) then
         write(output_unit, '(A,A,A,A,A,I0,A)') '    "', trim(name), '": {"sha256":"', trim(identity%sha256), '","bytes":', identity%bytes, '},'
      else
         write(output_unit, '(A,A,A,A,A,I0,A)') '    "', trim(name), '": {"sha256":"', trim(identity%sha256), '","bytes":', identity%bytes, '}'
      end if
   end subroutine write_identity_json

   subroutine print_json_report(active_config, loaded_entries, maximum_cases, collected_identities, result)
      type(dialogue_config_t), intent(in) :: active_config
      integer(int32), intent(in) :: loaded_entries
      integer(int64), intent(in) :: maximum_cases
      type(evaluation_identities_t), intent(in) :: collected_identities
      type(evaluation_metrics_t), intent(in) :: result

      write(output_unit, '(A)') '{'
      write(output_unit, '(A)') '  "schema": "vegai-native-dialogue-evaluation-report/v1",'
      write(output_unit, '(A)') '  "contract": "stateless native Fortran TSV evaluation; literal retrieval only",'
      write(output_unit, '(A)') '  "identities": {'
      call write_identity_json('checkpoint', collected_identities%checkpoint, .true.)
      call write_identity_json('knowledge', collected_identities%knowledge, .true.)
      call write_identity_json('evaluation', collected_identities%evaluation, .false.)
      write(output_unit, '(A)') '  },'
      write(output_unit, '(A)') '  "policy": {'
      call write_json_count('knowledge_entries', int(loaded_entries, int64), .true.)
      call write_json_count('maximum_cases', maximum_cases, .true.)
      call write_json_count('minimum_known_tokens', int(active_config%min_known_tokens, int64), .true.)
      call write_json_count('minimum_similarity_millionths', int(nint(active_config%minimum_similarity * 1000000.0), int64), .true.)
      call write_json_count('minimum_margin_millionths', int(nint(active_config%minimum_margin * 1000000.0), int64), .true.)
      call write_json_count('semantic_weight_millionths', int(nint(active_config%semantic_weight * 1000000.0), int64), .true.)
      call write_json_count('lexical_weight_millionths', int(nint(active_config%lexical_weight * 1000000.0), int64), .true.)
      call write_json_count('minimum_hybrid_score_millionths', int(nint(active_config%minimum_hybrid_score * 1000000.0), int64), .true.)
      call write_json_count('minimum_hybrid_margin_millionths', int(nint(active_config%minimum_hybrid_margin * 1000000.0), int64), .false.)
      write(output_unit, '(A)') '  },'
      write(output_unit, '(A)') '  "metrics": {'
      call write_json_count('total_cases', result%total_cases, .true.)
      call write_json_count('expected_grounded_cases', result%expected_grounded_cases, .true.)
      call write_json_count('expected_abstain_cases', result%expected_abstain_cases, .true.)
      call write_json_count('actual_grounded_cases', result%actual_grounded_cases, .true.)
      call write_json_count('rendered_abstain_cases', result%rendered_abstain_cases, .true.)
      call write_json_count('correct_grounded_prompt_cases', result%correct_grounded_prompt_cases, .true.)
      call write_json_count('wrong_grounded_prompt_cases', result%wrong_grounded_prompt_cases, .true.)
      call write_json_count('false_abstain_cases', result%false_abstain_cases, .true.)
      call write_json_count('false_grounded_cases', result%false_grounded_cases, .true.)
      call write_json_count('true_abstain_cases', result%true_abstain_cases, .true.)
      call write_json_count('oov_rejected_abstain_cases', result%oov_rejected_abstain_cases, .true.)
      call write_json_count('oov_rejected_expected_grounded_cases', result%oov_rejected_expected_grounded_cases, .false.)
      write(output_unit, '(A)') '  }'
      write(output_unit, '(A)') '}'
   end subroutine print_json_report

   subroutine print_usage()
      write(output_unit, '(A)') 'Native Vegai grounded-dialogue batch evaluator (retrieval only; not a generator)'
      write(output_unit, '(A)') 'Usage:'
      write(output_unit, '(A)') '  vegai_dialogue_eval evaluate CHECKPOINT KNOWLEDGE.tsv EVALUATION.tsv [options]'
      write(output_unit, '(A)') 'Evaluation TSV: case-id<TAB>prompt<TAB>expected-prompt; empty expected prompt means abstain.'
      write(output_unit, '(A)') 'Options:'
      write(output_unit, '(A)') '  --max-entries N --max-cases N --min-known-tokens N --min-similarity R --min-margin R'
      write(output_unit, '(A)') '  --semantic-weight R --lexical-weight R --min-hybrid-score R --min-hybrid-margin R'
      write(output_unit, '(A)') '  --require-all-known'
   end subroutine print_usage

   subroutine fail_usage()
      call print_usage()
      error stop 2
   end subroutine fail_usage

   subroutine fail_with_status(stage, failure_status)
      character(len=*), intent(in) :: stage
      integer(int32), intent(in) :: failure_status

      write(error_unit, '(A,A,A,I0,A)') 'ERROR: ', trim(stage), ' (status ', failure_status, ')'
      select case (failure_status)
      case (EVAL_ERR_IO)
         write(error_unit, '(A)') '       evaluation TSV I/O failed'
      case (EVAL_ERR_FORMAT)
         write(error_unit, '(A)') '       evaluation TSV must have one case-id TAB prompt TAB expected-prompt row per LF record'
      case (EVAL_ERR_RECORD_TOO_LONG)
         write(error_unit, '(A)') '       evaluation TSV record exceeds a bounded native field/record limit'
      case (EVAL_ERR_TOO_MANY_CASES)
         write(error_unit, '(A)') '       evaluation TSV exceeds --max-cases; no silent prefix measurement occurred'
      case (EVAL_ERR_DIALOGUE)
         write(error_unit, '(A)') '       native dialogue query failed outside the explicit no-known-token classification'
      case default
         write(error_unit, '(A)') '       invalid evaluator configuration or native dialogue initialization failure'
      end select
      error stop 1
   end subroutine fail_with_status

end program vegai_dialogue_eval
