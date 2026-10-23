! SPDX-License-Identifier: MIT
! Command-line entry point for deterministic HDC/VSA corpus pretraining.
program vegai_pretrain
   use, intrinsic :: iso_fortran_env, only: int32, int64, output_unit, error_unit, real32, iostat_end
   use hyper_types, only: hv_t
   use text_tokenizer, only: TOKEN_LEN, MAX_TOKENS_PER_LINE, tokenize_ascii_line
   use hdc_pretraining
   implicit none

   character(len=512) :: command, corpus_path, checkpoint_path, query, query_file
   type(hdc_pretrain_model_t) :: model
   type(hdc_pretrain_config_t) :: config
   type(pretrain_metrics_t) :: vocabulary_metrics, training_metrics
   type(pretrain_execution_report_t) :: execution_report
   integer(int32) :: status
   integer :: argument_count

   argument_count = command_argument_count()
   if (argument_count < 1) call fail_usage()
   call get_command_argument(1, command)

   select case (trim(command))
   case ('train')
      if (argument_count < 3) call fail_usage()
      call get_command_argument(2, corpus_path)
      call get_command_argument(3, checkpoint_path)
      query = ''
      config = hdc_pretrain_config_t()
      call parse_train_options(4, argument_count, config, query, status)
      if (status /= PT_OK) call fail_with_status('invalid train option', status)

      call pretrain_build_vocabulary(model, trim(corpus_path), config, vocabulary_metrics, status)
      if (status /= PT_OK) call fail_with_status('vocabulary pass failed', status)
      call print_vocabulary_summary(model, vocabulary_metrics)

      call pretrain_train_file(model, trim(corpus_path), config, training_metrics, status, execution_report)
      if (status /= PT_OK) call fail_with_status('training pass failed', status)
      call print_training_summary(model, training_metrics, config, execution_report)

      call pretrain_checkpoint_save(model, trim(checkpoint_path), status)
      if (status /= PT_OK) call fail_with_status('checkpoint save failed', status)
      write(output_unit, '(A,A)') 'checkpoint: ', trim(checkpoint_path)
      call maybe_query(model, query, status)
      if (status /= PT_OK) call fail_with_status('query failed', status)

   case ('resume')
      if (argument_count < 3) call fail_usage()
      call get_command_argument(2, corpus_path)
      call get_command_argument(3, checkpoint_path)
      call pretrain_checkpoint_load(trim(checkpoint_path), model, status)
      if (status /= PT_OK) call fail_with_status('checkpoint load failed', status)
      call pretrain_config_from_model(model, config, status)
      if (status /= PT_OK) call fail_with_status('checkpoint configuration failed', status)
      query = ''
      call parse_resume_options(4, argument_count, config, query, status)
      if (status /= PT_OK) call fail_with_status('invalid resume option', status)

      call pretrain_train_file(model, trim(corpus_path), config, training_metrics, status, execution_report)
      if (status /= PT_OK) call fail_with_status('training pass failed', status)
      call print_training_summary(model, training_metrics, config, execution_report)
      call pretrain_checkpoint_save(model, trim(checkpoint_path), status)
      if (status /= PT_OK) call fail_with_status('checkpoint save failed', status)
      write(output_unit, '(A,A)') 'checkpoint: ', trim(checkpoint_path)
      call maybe_query(model, query, status)
      if (status /= PT_OK) call fail_with_status('query failed', status)

   case ('continue')
      if (argument_count < 3) call fail_usage()
      call get_command_argument(2, corpus_path)
      call get_command_argument(3, checkpoint_path)
      call pretrain_checkpoint_load(trim(checkpoint_path), model, status)
      if (status /= PT_OK) call fail_with_status('checkpoint load failed', status)
      call pretrain_config_from_model(model, config, status)
      if (status /= PT_OK) call fail_with_status('checkpoint configuration failed', status)
      query = ''
      call parse_resume_options(4, argument_count, config, query, status)
      if (status /= PT_OK) call fail_with_status('invalid continue option', status)

      call pretrain_extend_vocabulary(model, trim(corpus_path), config, vocabulary_metrics, status)
      if (status /= PT_OK) call fail_with_status('vocabulary extension failed', status)
      call print_vocabulary_summary(model, vocabulary_metrics, 'vocabulary extension')

      call pretrain_train_file(model, trim(corpus_path), config, training_metrics, status, execution_report)
      if (status /= PT_OK) call fail_with_status('training pass failed', status)
      call print_training_summary(model, training_metrics, config, execution_report)
      call pretrain_checkpoint_save(model, trim(checkpoint_path), status)
      if (status /= PT_OK) call fail_with_status('checkpoint save failed', status)
      write(output_unit, '(A,A)') 'checkpoint: ', trim(checkpoint_path)
      call maybe_query(model, query, status)
      if (status /= PT_OK) call fail_with_status('query failed', status)

   case ('inspect')
      if (argument_count < 2) call fail_usage()
      call get_command_argument(2, checkpoint_path)
      query = ''
      query_file = ''
      call parse_inspect_options(3, argument_count, query, query_file, status)
      if (status /= PT_OK) call fail_with_status('invalid inspect option', status)
      call pretrain_checkpoint_load(trim(checkpoint_path), model, status)
      if (status /= PT_OK) call fail_with_status('checkpoint load failed', status)
      call print_model_summary(model)
      if (len_trim(query_file) > 0) then
         call maybe_query_file(model, trim(query_file), status)
      else
         call maybe_query(model, query, status)
      end if
      if (status /= PT_OK) call fail_with_status('query failed', status)

   case ('help', '--help', '-h')
      call print_usage()

   case default
      call fail_usage()
   end select

   call pretrain_model_destroy(model)

contains

   subroutine parse_train_options(first_argument, last_argument, parsed_config, parsed_query, parse_status)
      integer, intent(in) :: first_argument, last_argument
      type(hdc_pretrain_config_t), intent(inout) :: parsed_config
      character(len=*), intent(out) :: parsed_query
      integer(int32), intent(out) :: parse_status
      character(len=512) :: option, value
      integer :: index, io_status

      parsed_query = ''
      parse_status = PT_OK
      index = first_argument
      do while (index <= last_argument)
         call get_command_argument(index, option)
         select case (trim(option))
         case ('--strict')
            parsed_config%strict_tokenization = .true.
            index = index + 1
         case ('--serial-context-build')
            parsed_config%parallel_context_build = .false.
            index = index + 1
         case ('--parallel-context-build')
            parsed_config%parallel_context_build = .true.
            index = index + 1
         case ('--reference-context-kernel')
            parsed_config%reference_context_kernel = .true.
            index = index + 1
         case ('--openvino-required')
            parsed_config%context_provider = PT_CONTEXT_PROVIDER_OPENVINO
            parsed_config%openvino_fallback_cpu = .false.
            index = index + 1
         case ('--openvino-require-accelerator')
            parsed_config%context_provider = PT_CONTEXT_PROVIDER_OPENVINO
            parsed_config%openvino_require_accelerator = .true.
            parsed_config%openvino_fallback_cpu = .false.
            index = index + 1
         case ('--openvino-fallback-cpu')
            parsed_config%context_provider = PT_CONTEXT_PROVIDER_OPENVINO
            parsed_config%openvino_fallback_cpu = .true.
            parsed_config%openvino_require_accelerator = .false.
            index = index + 1
         case ('--max-vocab', '--min-frequency', '--window', '--position-shift', '--context-mode', '--epochs', &
               '--initial-plasticity', '--minimum-plasticity', '--decay-events', '--query', '--context-provider', &
               '--parallel-min-contexts', '--openvino-worker', '--openvino-python', '--openvino-device', &
               '--openvino-work-dir', '--openvino-min-contexts', '--openvino-timeout-ms')
            if (index == last_argument) then
               parse_status = PT_ERR_INVALID_ARGUMENT
               return
            end if
            call get_command_argument(index + 1, value)
            select case (trim(option))
            case ('--max-vocab')
               read(value, *, iostat=io_status) parsed_config%max_vocabulary
            case ('--min-frequency')
               read(value, *, iostat=io_status) parsed_config%min_frequency
            case ('--window')
               read(value, *, iostat=io_status) parsed_config%context_window
            case ('--position-shift')
               read(value, *, iostat=io_status) parsed_config%position_shift_bits
            case ('--context-mode')
               select case (trim(value))
               case ('legacy')
                  parsed_config%context_mode = PT_CONTEXT_LEGACY_POSITIONAL
                  io_status = 0
               case ('positional')
                  parsed_config%context_mode = PT_CONTEXT_POSITIONAL
                  io_status = 0
               case ('hybrid')
                  parsed_config%context_mode = PT_CONTEXT_HYBRID
                  io_status = 0
               case ('hierarchical')
                  parsed_config%context_mode = PT_CONTEXT_HIERARCHICAL
                  io_status = 0
               case default
                  io_status = 1
               end select
            case ('--epochs')
               read(value, *, iostat=io_status) parsed_config%epochs
            case ('--initial-plasticity')
               read(value, *, iostat=io_status) parsed_config%initial_plasticity_percent
            case ('--minimum-plasticity')
               read(value, *, iostat=io_status) parsed_config%minimum_plasticity_percent
            case ('--decay-events')
               read(value, *, iostat=io_status) parsed_config%plasticity_decay_events
            case ('--query')
               parsed_query = value
               io_status = 0
            case ('--context-provider')
               select case (trim(value))
               case ('cpu')
                  parsed_config%context_provider = PT_CONTEXT_PROVIDER_CPU
                  io_status = 0
               case ('openvino')
                  parsed_config%context_provider = PT_CONTEXT_PROVIDER_OPENVINO
                  io_status = 0
               case default
                  io_status = 1
               end select
            case ('--parallel-min-contexts')
               read(value, *, iostat=io_status) parsed_config%parallel_min_contexts
            case ('--openvino-worker')
               parsed_config%openvino_worker = value
               io_status = 0
            case ('--openvino-python')
               parsed_config%openvino_python = value
               io_status = 0
            case ('--openvino-device')
               parsed_config%openvino_device = value
               io_status = 0
            case ('--openvino-work-dir')
               parsed_config%openvino_work_dir = value
               io_status = 0
            case ('--openvino-min-contexts')
               read(value, *, iostat=io_status) parsed_config%openvino_min_contexts
            case ('--openvino-timeout-ms')
               read(value, *, iostat=io_status) parsed_config%openvino_timeout_milliseconds
            end select
            if (io_status /= 0) then
               parse_status = PT_ERR_INVALID_ARGUMENT
               return
            end if
            index = index + 2
         case default
            parse_status = PT_ERR_INVALID_ARGUMENT
            return
         end select
      end do
   end subroutine parse_train_options

   subroutine parse_resume_options(first_argument, last_argument, parsed_config, parsed_query, parse_status)
      integer, intent(in) :: first_argument, last_argument
      type(hdc_pretrain_config_t), intent(inout) :: parsed_config
      character(len=*), intent(out) :: parsed_query
      integer(int32), intent(out) :: parse_status
      character(len=512) :: option, value
      integer :: index, io_status

      parsed_query = ''
      parse_status = PT_OK
      index = first_argument
      do while (index <= last_argument)
         call get_command_argument(index, option)
         select case (trim(option))
         case ('--strict')
            parsed_config%strict_tokenization = .true.
            index = index + 1
         case ('--serial-context-build')
            parsed_config%parallel_context_build = .false.
            index = index + 1
         case ('--parallel-context-build')
            parsed_config%parallel_context_build = .true.
            index = index + 1
         case ('--reference-context-kernel')
            parsed_config%reference_context_kernel = .true.
            index = index + 1
         case ('--openvino-required')
            parsed_config%context_provider = PT_CONTEXT_PROVIDER_OPENVINO
            parsed_config%openvino_fallback_cpu = .false.
            index = index + 1
         case ('--openvino-require-accelerator')
            parsed_config%context_provider = PT_CONTEXT_PROVIDER_OPENVINO
            parsed_config%openvino_require_accelerator = .true.
            parsed_config%openvino_fallback_cpu = .false.
            index = index + 1
         case ('--openvino-fallback-cpu')
            parsed_config%context_provider = PT_CONTEXT_PROVIDER_OPENVINO
            parsed_config%openvino_fallback_cpu = .true.
            parsed_config%openvino_require_accelerator = .false.
            index = index + 1
         case ('--epochs', '--query', '--context-provider', '--parallel-min-contexts', '--openvino-worker', &
               '--openvino-python', '--openvino-device', '--openvino-work-dir', '--openvino-min-contexts', &
               '--openvino-timeout-ms')
            if (index == last_argument) then
               parse_status = PT_ERR_INVALID_ARGUMENT
               return
            end if
            call get_command_argument(index + 1, value)
            select case (trim(option))
            case ('--epochs')
               read(value, *, iostat=io_status) parsed_config%epochs
            case ('--query')
               parsed_query = value
               io_status = 0
            case ('--context-provider')
               select case (trim(value))
               case ('cpu')
                  parsed_config%context_provider = PT_CONTEXT_PROVIDER_CPU
                  io_status = 0
               case ('openvino')
                  parsed_config%context_provider = PT_CONTEXT_PROVIDER_OPENVINO
                  io_status = 0
               case default
                  io_status = 1
               end select
            case ('--parallel-min-contexts')
               read(value, *, iostat=io_status) parsed_config%parallel_min_contexts
            case ('--openvino-worker')
               parsed_config%openvino_worker = value
               io_status = 0
            case ('--openvino-python')
               parsed_config%openvino_python = value
               io_status = 0
            case ('--openvino-device')
               parsed_config%openvino_device = value
               io_status = 0
            case ('--openvino-work-dir')
               parsed_config%openvino_work_dir = value
               io_status = 0
            case ('--openvino-min-contexts')
               read(value, *, iostat=io_status) parsed_config%openvino_min_contexts
            case ('--openvino-timeout-ms')
               read(value, *, iostat=io_status) parsed_config%openvino_timeout_milliseconds
            end select
            if (io_status /= 0) then
               parse_status = PT_ERR_INVALID_ARGUMENT
               return
            end if
            index = index + 2
         case default
            parse_status = PT_ERR_INVALID_ARGUMENT
            return
         end select
      end do
   end subroutine parse_resume_options

   subroutine parse_inspect_options(first_argument, last_argument, parsed_query, parsed_query_file, parse_status)
      integer, intent(in) :: first_argument, last_argument
      character(len=*), intent(out) :: parsed_query, parsed_query_file
      integer(int32), intent(out) :: parse_status
      character(len=512) :: option, value
      integer :: index

      parsed_query = ''
      parsed_query_file = ''
      parse_status = PT_OK
      index = first_argument
      do while (index <= last_argument)
         if (index == last_argument) then
            parse_status = PT_ERR_INVALID_ARGUMENT
            return
         end if
         call get_command_argument(index, option)
         call get_command_argument(index + 1, value)
         select case (trim(option))
         case ('--query')
            if (len_trim(parsed_query_file) > 0 .or. len_trim(parsed_query) > 0) then
               parse_status = PT_ERR_INVALID_ARGUMENT
               return
            end if
            parsed_query = value
         case ('--query-file')
            if (len_trim(parsed_query) > 0 .or. len_trim(parsed_query_file) > 0) then
               parse_status = PT_ERR_INVALID_ARGUMENT
               return
            end if
            parsed_query_file = value
         case default
            parse_status = PT_ERR_INVALID_ARGUMENT
            return
         end select
         index = index + 2
      end do
   end subroutine parse_inspect_options

   subroutine print_vocabulary_summary(trained_model, metrics, phase_name)
      type(hdc_pretrain_model_t), intent(in) :: trained_model
      type(pretrain_metrics_t), intent(in) :: metrics
      character(len=*), intent(in), optional :: phase_name
      character(len=64) :: rendered_phase

      rendered_phase = 'vocabulary'
      if (present(phase_name)) rendered_phase = phase_name
      write(output_unit, '(A,A)') 'phase: ', trim(rendered_phase)
      write(output_unit, '(A,I0)') '  corpus lines: ', metrics%corpus_lines
      write(output_unit, '(A,I0)') '  observed tokens: ', metrics%raw_tokens
      write(output_unit, '(A,I0)') '  unique tokens: ', trained_model%vocabulary%count
      write(output_unit, '(A,I0)') '  active tokens: ', trained_model%vocabulary%active_count
      write(output_unit, '(A,I0)') '  tokenizer truncations: ', metrics%tokenizer_truncations
      write(output_unit, '(A,I0)') '  non-ASCII input bytes: ', metrics%non_ascii_bytes
      write(output_unit, '(A,I0)') '  dropped overlong records: ', metrics%overlong_records
   end subroutine print_vocabulary_summary

   subroutine print_training_summary(trained_model, metrics, active_config, execution)
      type(hdc_pretrain_model_t), intent(in) :: trained_model
      type(pretrain_metrics_t), intent(in) :: metrics
      type(hdc_pretrain_config_t), intent(in) :: active_config
      type(pretrain_execution_report_t), intent(in) :: execution

      write(output_unit, '(A)') 'phase: streaming HDC context learning'
      write(output_unit, '(A,I0)') '  completed epochs: ', trained_model%training_epochs_completed
      write(output_unit, '(A,I0)') '  corpus lines: ', metrics%corpus_lines
      write(output_unit, '(A,I0)') '  raw tokens: ', metrics%raw_tokens
      write(output_unit, '(A,I0)') '  active tokens: ', metrics%active_tokens
      write(output_unit, '(A,I0)') '  out-of-vocabulary tokens: ', metrics%oov_tokens
      write(output_unit, '(A,I0)') '  ordered context pairs: ', metrics%context_pairs
      write(output_unit, '(A,I0)') '  prototype updates: ', metrics%center_updates
      write(output_unit, '(A,I0)') '  tokenizer truncations: ', metrics%tokenizer_truncations
      write(output_unit, '(A,I0)') '  non-ASCII input bytes: ', metrics%non_ascii_bytes
      write(output_unit, '(A,I0)') '  dropped overlong records: ', metrics%overlong_records
      write(output_unit, '(A,I0)') '  adaptive parallel minimum contexts: ', active_config%parallel_min_contexts
      write(output_unit, '(A,I0)') '  CPU serial context batches: ', execution%cpu_serial_batches
      write(output_unit, '(A,I0)') '  CPU parallel context batches: ', execution%cpu_parallel_batches
      write(output_unit, '(A,I0)') '  CPU serial-built contexts: ', execution%cpu_serial_contexts
      write(output_unit, '(A,I0)') '  CPU parallel-built contexts: ', execution%cpu_parallel_contexts
      if (active_config%context_provider == PT_CONTEXT_PROVIDER_OPENVINO) then
         write(output_unit, '(A,A)') '  requested context provider: ', &
                                    trim(pretrain_context_provider_name(active_config%context_provider))
         write(output_unit, '(A,I0)') '  CPU-built contexts: ', execution%cpu_contexts
         write(output_unit, '(A,I0)') '  OpenVINO-dispatched contexts: ', execution%openvino_contexts
         write(output_unit, '(A,I0)') '  OpenVINO resident worker sessions: ', execution%openvino_worker_sessions
         write(output_unit, '(A,I0)') '  OpenVINO worker requests: ', execution%openvino_worker_invocations
         write(output_unit, '(A,I0)') '  OpenVINO native-CPU bypass contexts: ', execution%openvino_cpu_bypass_contexts
         write(output_unit, '(A,I0)') '  OpenVINO CPU-fallback contexts: ', execution%openvino_fallback_contexts
         write(output_unit, '(A,L1,A,L1,A,L1)') '  OpenVINO execution devices: CPU=', &
              execution%openvino_cpu_execution, ' GPU=', execution%openvino_gpu_execution, &
              ' NPU=', execution%openvino_npu_execution
      else
         write(output_unit, '(A)') '  context execution backend: CPU/OpenMP'
      end if
   end subroutine print_training_summary

   subroutine print_model_summary(trained_model)
      type(hdc_pretrain_model_t), intent(in) :: trained_model

      write(output_unit, '(A)') 'Vegai HDC pretraining checkpoint'
      write(output_unit, '(A,I0)') '  vocabulary capacity: ', trained_model%vocabulary%capacity
      write(output_unit, '(A,I0)') '  unique tokens: ', trained_model%vocabulary%count
      write(output_unit, '(A,I0)') '  active tokens: ', trained_model%vocabulary%active_count
      write(output_unit, '(A,I0)') '  min frequency: ', trained_model%vocabulary%min_frequency
      write(output_unit, '(A,I0)') '  context window: ', trained_model%context_window
      write(output_unit, '(A,I0)') '  positional shift: ', trained_model%position_shift_bits
      write(output_unit, '(A,A)') '  context mode: ', trim(pretrain_context_mode_name(trained_model%context_mode))
      write(output_unit, '(A,I0)') '  completed epochs: ', trained_model%training_epochs_completed
      write(output_unit, '(A,I0)') '  total prototype updates: ', trained_model%totals%center_updates
      write(output_unit, '(A,I0)') '  estimated resident bytes: ', pretrain_model_footprint_bytes(trained_model)
   end subroutine print_model_summary

   ! Batch diagnostic path for benchmark and operator workflows.  Each nonblank
   ! physical record supplies exactly one tokenizer-valid query token; output
   ! remains the same three-line query block used by --query.
   subroutine maybe_query_file(trained_model, requested_query_file, query_status)
      type(hdc_pretrain_model_t), intent(in) :: trained_model
      character(len=*), intent(in) :: requested_query_file
      integer(int32), intent(out) :: query_status
      character(len=512) :: query_line
      integer :: unit, io_status

      query_status = PT_OK
      if (len_trim(requested_query_file) == 0) return
      open(newunit=unit, file=trim(requested_query_file), status='old', action='read', &
           form='formatted', iostat=io_status)
      if (io_status /= 0) then
         query_status = PT_ERR_IO
         return
      end if
      do
         read(unit, '(A)', iostat=io_status) query_line
         if (io_status == iostat_end) exit
         if (io_status /= 0) then
            query_status = PT_ERR_IO
            exit
         end if
         if (len_trim(query_line) == 0) cycle
         call maybe_query(trained_model, trim(query_line), query_status)
         if (query_status /= PT_OK) exit
      end do
      close(unit)
   end subroutine maybe_query_file

   subroutine maybe_query(trained_model, requested_query, query_status)
      type(hdc_pretrain_model_t), intent(in) :: trained_model
      character(len=*), intent(in) :: requested_query
      integer(int32), intent(out) :: query_status
      character(len=TOKEN_LEN) :: canonical_query, nearest_token
      character(len=TOKEN_LEN) :: token_buffer(MAX_TOKENS_PER_LINE)
      type(hv_t) :: semantic_hv
      integer(int32) :: n_tokens, slot, nearest_slot, lookup_status
      logical :: truncated, found
      real(real32) :: similarity

      query_status = PT_OK
      if (len_trim(requested_query) == 0) return
      call tokenize_ascii_line(trim(requested_query), token_buffer, n_tokens, truncated)
      if (truncated .or. n_tokens /= 1_int32) then
         query_status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      canonical_query = token_buffer(1)
      call pretrain_lookup_token(trained_model, canonical_query, slot, found, lookup_status)
      if (lookup_status /= PT_OK .or. .not. found) then
         query_status = lookup_status
         return
      end if
      call pretrain_get_semantic_hv(trained_model, slot, semantic_hv, query_status)
      if (query_status /= PT_OK) return
      call pretrain_nearest_semantic(trained_model, semantic_hv, nearest_slot, nearest_token, &
                                     similarity, query_status, slot)
      if (query_status == PT_ERR_NOT_FOUND) then
         write(output_unit, '(A,A)') 'query: ', trim(canonical_query)
         write(output_unit, '(A)') '  nearest semantic neighbour: unavailable (one active token)'
         query_status = PT_OK
         return
      end if
      if (query_status /= PT_OK) return
      write(output_unit, '(A,A)') 'query: ', trim(canonical_query)
      write(output_unit, '(A,A)') '  nearest semantic neighbour: ', trim(nearest_token)
      write(output_unit, '(A,F8.5)') '  bipolar similarity: ', similarity
   end subroutine maybe_query

   subroutine print_usage()
      write(output_unit, '(A)') 'Vegai deterministic HDC/VSA corpus pretraining'
      write(output_unit, '(A)') 'Usage:'
      write(output_unit, '(A)') '  vegai_pretrain train CORPUS CHECKPOINT [options]'
      write(output_unit, '(A)') '  vegai_pretrain resume CORPUS CHECKPOINT [--epochs N] [--strict] [--query TOKEN]'
      write(output_unit, '(A)') '    [--reference-context-kernel]'
      write(output_unit, '(A)') '  vegai_pretrain continue SHARD CHECKPOINT [--epochs N] [--strict] [--query TOKEN]'
      write(output_unit, '(A)') '    [--reference-context-kernel]'
      write(output_unit, '(A)') '  vegai_pretrain inspect CHECKPOINT [--query TOKEN | --query-file PATH]'
      write(output_unit, '(A)') 'Train options:'
      write(output_unit, '(A)') '  --max-vocab N --min-frequency N --window N --position-shift N --context-mode MODE --epochs N'
      write(output_unit, '(A)') '  MODE is legacy, positional, hybrid, or hierarchical (default hierarchical)'
      write(output_unit, '(A)') '  hierarchical adds bounded whole-record topical evidence for long records'
      write(output_unit, '(A)') '  continue first extends vocabulary from one immutable shard, then trains it'
      write(output_unit, '(A)') '  --initial-plasticity N --minimum-plasticity N --decay-events N --strict --query TOKEN'
      write(output_unit, '(A)') '  --parallel-context-build (default) builds immutable per-record contexts with OpenMP when worthwhile'
      write(output_unit, '(A)') '  --parallel-min-contexts N launches OpenMP only at N record updates (default 32; execution-only)'
      write(output_unit, '(A)') '  --serial-context-build disables that execution-only parallel stage for diagnosis'
      write(output_unit, '(A)') '  --reference-context-kernel is an exact benchmark/regression implementation switch'
      write(output_unit, '(A)') '  --context-provider cpu|openvino --openvino-worker PATH --openvino-python CMD --openvino-device DEVICE'
      write(output_unit, '(A)') '  --openvino-work-dir DIR --openvino-min-contexts N --openvino-timeout-ms N [--openvino-fallback-cpu|--openvino-required]'
      write(output_unit, '(A)') '  --openvino-require-accelerator rejects OpenVINO CPU fallback; immutable contexts only, commits stay ordered on CPU'
      write(output_unit, '(A)') 'This is streaming distributional HDC learning, not Transformer/LLM pretraining.'
   end subroutine print_usage

   subroutine fail_usage()
      call print_usage()
      error stop 2
   end subroutine fail_usage

   subroutine fail_with_status(stage, failure_status)
      character(len=*), intent(in) :: stage
      integer(int32), intent(in) :: failure_status

      write(error_unit, '(A,A,A,I0,A)') 'ERROR: ', trim(stage), ' (status ', failure_status, ')'
      write(error_unit, '(A,A)') '       ', trim(status_description(failure_status))
      error stop 1
   end subroutine fail_with_status

   pure function status_description(failure_status) result(description)
      integer(int32), intent(in) :: failure_status
      character(len=64) :: description

      select case (failure_status)
      case (PT_OK)
         description = 'success'
      case (PT_ERR_INVALID_ARGUMENT)
         description = 'invalid argument or configuration'
      case (PT_ERR_IO)
         description = 'file open, read, or write failure'
      case (PT_ERR_VOCABULARY)
         description = 'vocabulary capacity or integrity failure'
      case (PT_ERR_ALLOCATION)
         description = 'memory allocation failure'
      case (PT_ERR_MODEL_NOT_READY)
         description = 'model is not initialized'
      case (PT_ERR_TOKENIZATION)
         description = 'strict tokenizer policy rejected corpus data'
      case (PT_ERR_INCOMPATIBLE_CONFIG)
         description = 'resume semantics differ from checkpoint configuration'
      case (PT_ERR_CHECKPOINT)
         description = 'checkpoint schema or integrity failure'
      case (PT_ERR_COUNTER_OVERFLOW)
         description = 'counter overflow prevented'
      case (PT_ERR_NOT_FOUND)
         description = 'token was not found in active vocabulary'
      case (PT_ERR_EMPTY_VOCABULARY)
         description = 'no token met the minimum frequency threshold'
      case (PT_ERR_HYPERVECTOR)
         description = 'hypervector operation failed'
      case (PT_ERR_RECORD_TOO_LONG)
         description = 'strict record-length policy rejected corpus data'
      case (PT_ERR_CONTEXT_PROVIDER)
         description = 'optional context provider failed and CPU fallback was disabled'
      case default
         description = 'unknown pretraining status'
      end select
   end function status_description

end program vegai_pretrain
