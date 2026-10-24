! SPDX-License-Identifier: MIT
!==============================================================================
!  chat.f90
!  Command-line interface for data-driven grounded HDC dialogue retrieval.
!
!  Responses are retrieved verbatim from a user-owned TSV knowledge base.  This
!  executable is intentionally not a free-form LLM/Transformer generator.
!==============================================================================
program vegai_chat
   use, intrinsic :: iso_fortran_env, only: int32, input_unit, output_unit, error_unit, iostat_end, iostat_eor
   use hdc_pretraining, only: PT_OK, hdc_pretrain_model_t, pretrain_checkpoint_load, pretrain_model_destroy
   use dialogue_retrieval
   implicit none

   character(len=32) :: command
   character(len=512) :: checkpoint_path, knowledge_path
   character(len=DIALOGUE_MAX_PROMPT_CHARS) :: prompt
   type(hdc_pretrain_model_t) :: model
   type(dialogue_config_t) :: config
   type(dialogue_index_t) :: index
   type(dialogue_session_t) :: session
   integer(int32) :: status, entries_loaded
   integer :: argument_count

   argument_count = command_argument_count()
   if (argument_count < 1) call fail_usage()
   call read_cli_argument(1, command, status)
   if (status /= DIALOGUE_OK) call fail_usage()

   select case (trim(command))
   case ('ask')
      if (argument_count < 4) call fail_usage()
      call read_cli_argument(2, checkpoint_path, status)
      if (status /= DIALOGUE_OK) call fail_usage()
      call read_cli_argument(3, knowledge_path, status)
      if (status /= DIALOGUE_OK) call fail_usage()
      config = dialogue_config_t()
      prompt = ''
      call parse_options(4, argument_count, config, prompt, .true., status)
      if (status /= DIALOGUE_OK .or. len_trim(prompt) == 0) call fail_usage()
      call load_dialogue(trim(checkpoint_path), trim(knowledge_path), config, model, index, session, entries_loaded, status)
      if (status /= DIALOGUE_OK) call fail_with_status('dialogue initialization failed', status)
      call print_index_summary(index, entries_loaded)
      call answer_one_prompt(model, index, session, prompt, status)
      if (status /= DIALOGUE_OK) call fail_with_status('dialogue prompt failed', status)
      call cleanup(model, index, session)

   case ('repl')
      if (argument_count < 3) call fail_usage()
      call read_cli_argument(2, checkpoint_path, status)
      if (status /= DIALOGUE_OK) call fail_usage()
      call read_cli_argument(3, knowledge_path, status)
      if (status /= DIALOGUE_OK) call fail_usage()
      config = dialogue_config_t()
      prompt = ''
      call parse_options(4, argument_count, config, prompt, .false., status)
      if (status /= DIALOGUE_OK) call fail_usage()
      call load_dialogue(trim(checkpoint_path), trim(knowledge_path), config, model, index, session, entries_loaded, status)
      if (status /= DIALOGUE_OK) call fail_with_status('dialogue initialization failed', status)
      call run_repl(model, index, session, entries_loaded, status)
      call cleanup(model, index, session)
      if (status /= DIALOGUE_OK) call fail_with_status('dialogue session failed', status)

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
      read_status = DIALOGUE_OK
      call get_command_argument(position, length=argument_length, status=command_status)
      if (command_status /= 0 .or. argument_length < 0 .or. argument_length > len(value)) then
         read_status = DIALOGUE_ERR_INVALID_ARGUMENT
         return
      end if
      call get_command_argument(position, value=value, status=command_status)
      if (command_status /= 0) read_status = DIALOGUE_ERR_INVALID_ARGUMENT
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
      call dialogue_session_init(active_session, requested_config%max_history_turns, load_status)
   end subroutine load_dialogue

   subroutine answer_one_prompt(active_model, active_index, active_session, requested_prompt, ask_status)
      type(hdc_pretrain_model_t), intent(in) :: active_model
      type(dialogue_index_t), intent(in) :: active_index
      type(dialogue_session_t), intent(inout) :: active_session
      character(len=*), intent(in) :: requested_prompt
      integer(int32), intent(out) :: ask_status
      type(dialogue_answer_t) :: answer

      call dialogue_ask(active_index, active_model, active_session, requested_prompt, answer, ask_status)
      if (ask_status /= DIALOGUE_OK) return
      call print_answer(answer)
   end subroutine answer_one_prompt

   subroutine print_index_summary(active_index, loaded_entries)
      type(dialogue_index_t), intent(in) :: active_index
      integer(int32), intent(in) :: loaded_entries

      write(output_unit, '(A,I0)') 'knowledge entries: ', loaded_entries
      write(output_unit, '(A,I0)') 'knowledge index resident bytes: ', dialogue_index_footprint_bytes(active_index)
      write(output_unit, '(A,I0)') 'conversation history turns: ', active_index%config%max_history_turns
      write(output_unit, '(A,F8.5)') 'minimum grounding similarity: ', active_index%config%minimum_similarity
      write(output_unit, '(A,F8.5)') 'minimum grounding margin: ', active_index%config%minimum_margin
      write(output_unit, '(A,L1)') 'live prompt requires all known tokens: ', active_index%config%require_all_prompt_tokens_known
   end subroutine print_index_summary

   subroutine print_answer(answer)
      type(dialogue_answer_t), intent(in) :: answer

      write(output_unit, '(A,I0)') 'query known tokens: ', answer%query_known_tokens
      write(output_unit, '(A,I0)') 'query ignored tokens: ', answer%query_ignored_tokens
      write(output_unit, '(A,I0)') 'candidates examined: ', answer%candidates_examined
      write(output_unit, '(A,F8.5)') 'best bipolar similarity: ', answer%similarity
      write(output_unit, '(A,F8.5)') 'runner-up bipolar similarity: ', answer%runner_up_similarity
      write(output_unit, '(A,F8.5)') 'grounding margin: ', answer%margin
      if (answer%found) then
         write(output_unit, '(A)') 'grounded: true'
      else
         write(output_unit, '(A)') 'grounded: false'
         write(output_unit, '(A)') 'answer unavailable: no knowledge-base response met the grounding policy'
      end if
      if (answer%found) then
         write(output_unit, '(A,A)') 'matched prompt: ', trim(answer%matched_prompt)
         ! This is exact operator-supplied TSV content, never a source-coded or
         ! generated reply template.
         write(output_unit, '(A,A)') 'answer: ', trim(answer%response)
      end if
   end subroutine print_answer

   subroutine run_repl(active_model, active_index, active_session, loaded_entries, repl_status)
      type(hdc_pretrain_model_t), intent(in) :: active_model
      type(dialogue_index_t), intent(in) :: active_index
      type(dialogue_session_t), intent(inout) :: active_session
      integer(int32), intent(in) :: loaded_entries
      integer(int32), intent(out) :: repl_status
      character(len=DIALOGUE_MAX_PROMPT_CHARS) :: line
      logical :: end_of_input
      integer(int32) :: read_status

      repl_status = DIALOGUE_OK
      write(output_unit, '(A)') 'Vegai grounded HDC dialogue retrieval'
      call print_index_summary(active_index, loaded_entries)
      write(output_unit, '(A)') 'Enter /quit to end; replies come only from the TSV knowledge base.'
      do
         write(output_unit, '(A)', advance='no') 'you> '
         call read_repl_line(line, end_of_input, read_status)
         if (end_of_input) exit
         if (read_status /= DIALOGUE_OK) then
            repl_status = read_status
            return
         end if
         if (trim(line) == '/quit') exit
         if (len_trim(line) == 0) cycle
         call answer_one_prompt(active_model, active_index, active_session, line, repl_status)
         if (repl_status /= DIALOGUE_OK) return
      end do
   end subroutine run_repl

   subroutine read_repl_line(line, eof, read_status)
      character(len=*), intent(out) :: line
      logical, intent(out) :: eof
      integer(int32), intent(out) :: read_status
      character(len=1) :: probe
      character(len=256) :: discard_buffer
      integer :: io_status, transferred

      line = ''
      eof = .false.
      read_status = DIALOGUE_OK
      transferred = 0
      read(input_unit, '(A)', advance='no', size=transferred, iostat=io_status) line
      if (io_status == iostat_end) then
         eof = .true.
         return
      end if
      if (io_status == iostat_eor) return
      if (io_status /= 0) then
         read_status = DIALOGUE_ERR_IO
         return
      end if

      ! A zero-status first read filled the bounded buffer.  Probe and consume
      ! the physical record so a long interactive prompt cannot be truncated
      ! into a different valid query or spill into the next turn.
      probe = ''
      transferred = 0
      read(input_unit, '(A)', advance='no', size=transferred, iostat=io_status) probe
      if (transferred == 0) then
         if (io_status == iostat_eor .or. io_status == iostat_end) return
         read_status = DIALOGUE_ERR_IO
         return
      end if
      do while (io_status == 0)
         discard_buffer = ''
         transferred = 0
         read(input_unit, '(A)', advance='no', size=transferred, iostat=io_status) discard_buffer
      end do
      if (io_status /= iostat_eor .and. io_status /= iostat_end) then
         read_status = DIALOGUE_ERR_IO
      else
         read_status = DIALOGUE_ERR_RECORD_TOO_LONG
      end if
   end subroutine read_repl_line

   subroutine parse_options(first_argument, last_argument, parsed_config, parsed_prompt, allow_prompt, parse_status)
      integer, intent(in) :: first_argument, last_argument
      type(dialogue_config_t), intent(inout) :: parsed_config
      character(len=*), intent(out) :: parsed_prompt
      logical, intent(in) :: allow_prompt
      integer(int32), intent(out) :: parse_status
      character(len=1024) :: option, value
      integer :: index, io_status
      logical :: prompt_seen

      parsed_prompt = ''
      prompt_seen = .false.
      parse_status = DIALOGUE_OK
      index = first_argument
      do while (index <= last_argument)
         call read_cli_argument(index, option, parse_status)
         if (parse_status /= DIALOGUE_OK) return
         select case (trim(option))
         case ('--require-all-known')
            parsed_config%require_all_prompt_tokens_known = .true.
            index = index + 1
         case ('--max-entries', '--history-turns', '--min-known-tokens', '--current-weight', &
               '--history-weight', '--min-similarity', '--min-margin', '--prompt')
            if (index == last_argument) then
               parse_status = DIALOGUE_ERR_INVALID_ARGUMENT
               return
            end if
            call read_cli_argument(index + 1, value, parse_status)
            if (parse_status /= DIALOGUE_OK) return
            select case (trim(option))
            case ('--max-entries')
               read(value, *, iostat=io_status) parsed_config%max_entries
            case ('--history-turns')
               read(value, *, iostat=io_status) parsed_config%max_history_turns
            case ('--min-known-tokens')
               read(value, *, iostat=io_status) parsed_config%min_known_tokens
            case ('--current-weight')
               read(value, *, iostat=io_status) parsed_config%current_prompt_weight
            case ('--history-weight')
               read(value, *, iostat=io_status) parsed_config%history_turn_weight
            case ('--min-similarity')
               read(value, *, iostat=io_status) parsed_config%minimum_similarity
            case ('--min-margin')
               read(value, *, iostat=io_status) parsed_config%minimum_margin
            case ('--prompt')
               if (.not. allow_prompt .or. prompt_seen) then
                  parse_status = DIALOGUE_ERR_INVALID_ARGUMENT
                  return
               end if
               parsed_prompt = value
               prompt_seen = .true.
               io_status = 0
            end select
            if (io_status /= 0) then
               parse_status = DIALOGUE_ERR_INVALID_ARGUMENT
               return
            end if
            index = index + 2
         case default
            parse_status = DIALOGUE_ERR_INVALID_ARGUMENT
            return
         end select
      end do
   end subroutine parse_options

   subroutine print_usage()
      write(output_unit, '(A)') 'Vegai grounded HDC dialogue retrieval (data-driven, not an LLM generator)'
      write(output_unit, '(A)') 'Usage:'
      write(output_unit, '(A)') '  vegai_chat ask CHECKPOINT KNOWLEDGE.tsv --prompt "ASCII prompt" [options]'
      write(output_unit, '(A)') '  vegai_chat repl CHECKPOINT KNOWLEDGE.tsv [options]'
      write(output_unit, '(A)') 'Knowledge TSV: exactly one ASCII TAB per nonblank row: prompt<TAB>response'
      write(output_unit, '(A)') 'Options:'
      write(output_unit, '(A)') '  --max-entries N --history-turns N --min-known-tokens N'
      write(output_unit, '(A)') '  --current-weight N --history-weight N --min-similarity R --min-margin R'
      write(output_unit, '(A)') '  --require-all-known'
      write(output_unit, '(A)') 'The answer is returned verbatim from the knowledge TSV only when grounded.'
   end subroutine print_usage

   subroutine fail_usage()
      call print_usage()
      error stop 2
   end subroutine fail_usage

   subroutine fail_with_status(stage, failure_status)
      character(len=*), intent(in) :: stage
      integer(int32), intent(in) :: failure_status

      write(error_unit, '(A,A,A,I0,A)') 'ERROR: ', trim(stage), ' (status ', failure_status, ')'
      write(error_unit, '(A,A)') '       ', trim(dialogue_status_description(failure_status))
      error stop 1
   end subroutine fail_with_status

   pure function dialogue_status_description(code) result(description)
      integer(int32), intent(in) :: code
      character(len=80) :: description

      select case (code)
      case (DIALOGUE_OK)
         description = 'success'
      case (DIALOGUE_ERR_INVALID_ARGUMENT)
         description = 'invalid dialogue configuration or API state'
      case (DIALOGUE_ERR_ALLOCATION)
         description = 'bounded dialogue index/session allocation failed'
      case (DIALOGUE_ERR_IO)
         description = 'dialogue knowledge file I/O failed'
      case (DIALOGUE_ERR_FORMAT)
         description = 'knowledge rows must be one nonblank prompt TAB response pair'
      case (DIALOGUE_ERR_CAPACITY)
         description = 'dialogue knowledge exceeds configured max entries'
      case (DIALOGUE_ERR_TOKENIZATION)
         description = 'dialogue input violates the strict ASCII text contract'
      case (DIALOGUE_ERR_MODEL)
         description = 'checkpoint is unavailable or incompatible with dialogue retrieval'
      case (DIALOGUE_ERR_NO_KNOWN_TOKENS)
         description = 'prompt has too few active checkpoint vocabulary tokens'
      case (DIALOGUE_ERR_UNKNOWN_TOKEN)
         description = 'prompt has unknown checkpoint tokens under live or knowledge-base strict policy'
      case (DIALOGUE_ERR_DUPLICATE_PROMPT)
         description = 'knowledge base has an ambiguous duplicate prompt embedding'
      case (DIALOGUE_ERR_RECORD_TOO_LONG)
         description = 'dialogue physical record exceeds its bounded byte contract'
      case (DIALOGUE_ERR_EMPTY_INDEX)
         description = 'knowledge base contains no usable dialogue rows'
      case default
         description = 'unknown dialogue retrieval failure'
      end select
   end function dialogue_status_description

end program vegai_chat
