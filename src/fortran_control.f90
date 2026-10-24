! SPDX-License-Identifier: MIT
!==============================================================================
!  fortran_control.f90
!  Native Fortran control-plane primitives for portable Vegai operation.
!
!  This module deliberately uses only standard Fortran facilities for its
!  profile, bounded resource arithmetic, report writing, and same-host train /
!  checkpoint smoke path. It never shells out to Python or a platform-specific
!  inventory command. Host RAM, free disk, CPU model, and accelerator details
!  therefore remain explicit operator inputs at this layer rather than guessed
!  from an OS-specific command.
!==============================================================================
module fortran_control
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64, output_unit, error_unit, &
                                          compiler_version, compiler_options
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use hdc_pretraining, only: PT_OK, PT_CONTEXT_LEGACY_POSITIONAL, PT_CONTEXT_POSITIONAL, &
                              PT_CONTEXT_HYBRID, PT_CONTEXT_HIERARCHICAL, PT_CONTEXT_COHERENT, &
                              PT_CONTEXT_DEBIASED, PT_CONTEXT_RESONANT, hdc_pretrain_model_t, &
                              hdc_pretrain_config_t, pretrain_context_mode_name, &
                              pretrain_metrics_t, pretrain_execution_report_t, &
                              pretrain_build_vocabulary, pretrain_train_file, &
                              pretrain_checkpoint_save, pretrain_checkpoint_load, &
                              pretrain_model_destroy, pretrain_model_is_ready
   use fortran_integrity, only: FI_OK, sha256_file
   use fortran_quality_gate, only: QG_OK, QG_ERR_IO, quality_case_t, quality_policy_t, &
                                   quality_report_t, quality_case_file_load, quality_gate_evaluate, &
                                   quality_policy_is_valid
   implicit none
   private

   integer(int32), parameter, public :: FC_OK = 0_int32
   integer(int32), parameter, public :: FC_ERR_INVALID_ARGUMENT = -900_int32
   integer(int32), parameter, public :: FC_ERR_IO = -901_int32
   integer(int32), parameter, public :: FC_ERR_NOT_READY = -902_int32
   integer(int32), parameter, public :: FC_ERR_TRAINING = -903_int32
   integer(int32), parameter, public :: FC_ERR_QUALITY = -904_int32
   integer(int32), parameter, public :: FC_ERR_ATTESTATION = -905_int32

   ! These estimates intentionally match the documented portable HDC model
   ! contract. They are a preflight reservation, not a claim about allocator,
   ! OpenMP runtime, page cache, or any operating-system memory manager.
   integer(int64), parameter, public :: FC_MODEL_BYTES_PER_SLOT = 4104_int64
   integer(int64), parameter, public :: FC_VOCABULARY_ALLOWANCE_PER_SLOT = 256_int64
   integer(int64), parameter, public :: FC_CONTEXT_STAGING_BYTES = 2097152_int64
   integer(int64), parameter, public :: FC_LINE_UPDATE_SCRATCH_PER_SLOT = 4_int64
   integer(int64), parameter, public :: FC_LOOKUP_CACHE_BYTES = 278528_int64
   ! RESONANT holds one frozen copy of the 16,384-bit semantic prototype plane
   ! for the duration of each epoch: 256 int64 words = 2,048 bytes per slot.
   integer(int64), parameter, public :: FC_RESONANT_SNAPSHOT_BYTES_PER_SLOT = 2048_int64
   integer(int64), parameter, public :: FC_CHECKPOINT_BYTES_PER_SLOT = 4168_int64
   integer(int64), parameter, public :: FC_CHECKPOINT_FIXED_BYTES = 65536_int64

   type, public :: fortran_resource_plan_t
      integer(int64) :: max_vocabulary = 0_int64
      integer(int32) :: context_mode = PT_CONTEXT_HIERARCHICAL
      real(real64) :: reserve_fraction = 0.25_real64
      integer(int64) :: model_payload_bytes = 0_int64
      integer(int64) :: resonant_snapshot_bytes = 0_int64
      integer(int64) :: vocabulary_allowance_bytes = 0_int64
      integer(int64) :: context_staging_bytes = 0_int64
      integer(int64) :: line_update_scratch_bytes = 0_int64
      integer(int64) :: lookup_cache_workspace_bytes = 0_int64
      integer(int64) :: reserve_bytes = 0_int64
      integer(int64) :: required_available_bytes = 0_int64
      integer(int64) :: estimated_checkpoint_bytes = 0_int64
      integer(int32) :: recommended_omp_threads = 1_int32
      logical :: ready = .false.
      logical :: available_memory_known = .false.
      logical :: available_storage_known = .false.
      integer(int64) :: available_memory_bytes = 0_int64
      integer(int64) :: available_storage_bytes = 0_int64
   end type fortran_resource_plan_t

   public :: fortran_control_main, fortran_resource_plan_estimate

contains

   subroutine fortran_resource_plan_estimate(max_vocabulary, reserve_fraction, requested_threads, plan, status, &
                                             context_mode)
      integer(int64), intent(in) :: max_vocabulary
      real(real64), intent(in) :: reserve_fraction
      integer(int32), intent(in) :: requested_threads
      type(fortran_resource_plan_t), intent(out) :: plan
      integer(int32), intent(out) :: status
      integer(int32), intent(in), optional :: context_mode
      integer(int64) :: resident_before_reserve
      integer(int32) :: effective_context_mode

      plan = fortran_resource_plan_t()
      status = FC_ERR_INVALID_ARGUMENT
      if (max_vocabulary <= 0_int64) return
      if (.not. ieee_is_finite(reserve_fraction)) return
      if (reserve_fraction < 0.0_real64 .or. reserve_fraction > 4.0_real64) return
      ! Keep each documented multiplication inside signed int64 with ample
      ! headroom. A larger request cannot be allocated by the bounded model in
      ! practice and must be rejected before it silently wraps.
      if (max_vocabulary > 100000000_int64) return
      if (requested_threads < 0_int32) return
      effective_context_mode = PT_CONTEXT_HIERARCHICAL
      if (present(context_mode)) effective_context_mode = context_mode
      select case (effective_context_mode)
      case (PT_CONTEXT_LEGACY_POSITIONAL, PT_CONTEXT_POSITIONAL, PT_CONTEXT_HYBRID, &
            PT_CONTEXT_HIERARCHICAL, PT_CONTEXT_COHERENT, PT_CONTEXT_DEBIASED, PT_CONTEXT_RESONANT)
         continue
      case default
         return
      end select

      plan%max_vocabulary = max_vocabulary
      plan%context_mode = effective_context_mode
      plan%reserve_fraction = reserve_fraction
      plan%model_payload_bytes = max_vocabulary * FC_MODEL_BYTES_PER_SLOT
      if (effective_context_mode == PT_CONTEXT_RESONANT) then
         plan%resonant_snapshot_bytes = max_vocabulary * FC_RESONANT_SNAPSHOT_BYTES_PER_SLOT
      end if
      plan%vocabulary_allowance_bytes = max_vocabulary * FC_VOCABULARY_ALLOWANCE_PER_SLOT
      plan%context_staging_bytes = FC_CONTEXT_STAGING_BYTES
      plan%line_update_scratch_bytes = max_vocabulary * FC_LINE_UPDATE_SCRATCH_PER_SLOT
      plan%lookup_cache_workspace_bytes = FC_LOOKUP_CACHE_BYTES
      resident_before_reserve = plan%model_payload_bytes + plan%resonant_snapshot_bytes + &
                                 plan%vocabulary_allowance_bytes + plan%context_staging_bytes + &
                                 plan%line_update_scratch_bytes + plan%lookup_cache_workspace_bytes
      plan%reserve_bytes = int(real(resident_before_reserve, real64) * reserve_fraction, int64)
      plan%required_available_bytes = resident_before_reserve + plan%reserve_bytes
      plan%estimated_checkpoint_bytes = FC_CHECKPOINT_FIXED_BYTES + &
                                        max_vocabulary * FC_CHECKPOINT_BYTES_PER_SLOT
      if (requested_threads > 0_int32) then
         plan%recommended_omp_threads = requested_threads
      else
         plan%recommended_omp_threads = 1_int32
      end if
      status = FC_OK
   end subroutine fortran_resource_plan_estimate

   subroutine fortran_control_main(exit_status)
      integer(int32), intent(out) :: exit_status
      integer :: argument_count
      character(len=512) :: command

      exit_status = FC_OK
      argument_count = command_argument_count()
      if (argument_count < 1) then
         call print_usage(error_unit)
         exit_status = FC_ERR_INVALID_ARGUMENT
         return
      end if
      call get_command_argument(1, command)
      select case (trim(command))
      case ('profile')
         call profile_command(argument_count, exit_status)
      case ('resource-plan')
         call resource_plan_command(argument_count, exit_status)
      case ('smoke')
         call smoke_command(argument_count, exit_status)
      case ('quality-gate')
         call quality_gate_command(argument_count, exit_status)
      case ('attest')
         call attest_command(argument_count, exit_status)
      case ('help', '--help', '-h')
         call print_usage(output_unit)
      case default
         write(error_unit, '(A,A)') 'vegai_control: unknown command: ', trim(command)
         call print_usage(error_unit)
         exit_status = FC_ERR_INVALID_ARGUMENT
      end select
   end subroutine fortran_control_main

   subroutine profile_command(argument_count, status)
      integer, intent(in) :: argument_count
      integer(int32), intent(out) :: status
      character(len=512) :: format, output_path, option, value
      integer :: index

      format = 'text'
      output_path = ''
      status = FC_OK
      index = 2
      do while (index <= argument_count)
         call get_command_argument(index, option)
         select case (trim(option))
         case ('--format')
            if (index == argument_count) then
               status = FC_ERR_INVALID_ARGUMENT
               return
            end if
            call get_command_argument(index + 1, value)
            format = trim(value)
            index = index + 2
         case ('--output')
            if (index == argument_count) then
               status = FC_ERR_INVALID_ARGUMENT
               return
            end if
            call get_command_argument(index + 1, output_path)
            index = index + 2
         case default
            write(error_unit, '(A,A)') 'vegai_control profile: unsupported option: ', trim(option)
            status = FC_ERR_INVALID_ARGUMENT
            return
         end select
      end do
      if (trim(format) /= 'text' .and. trim(format) /= 'json') then
         write(error_unit, '(A)') 'vegai_control profile: --format must be text or json'
         status = FC_ERR_INVALID_ARGUMENT
         return
      end if
      call emit_profile(output_unit, trim(format))
      if (len_trim(output_path) > 0) then
         call emit_profile_file(trim(output_path), trim(format), status)
      end if
   end subroutine profile_command

   subroutine resource_plan_command(argument_count, status)
      integer, intent(in) :: argument_count
      integer(int32), intent(out) :: status
      type(fortran_resource_plan_t) :: plan
      character(len=512) :: corpus_path, checkpoint_path, output_path, format
      character(len=512) :: option, value
      integer(int64) :: corpus_bytes, maximum_vocabulary
      integer(int64) :: available_memory, available_storage
      integer(int32) :: requested_threads, context_mode
      integer :: io_status
      real(real64) :: reserve_fraction
      logical :: exists, require_ready, memory_supplied, storage_supplied
      integer :: index

      status = FC_ERR_INVALID_ARGUMENT
      if (argument_count < 2) then
         write(error_unit, '(A)') 'vegai_control resource-plan: CORPUS is required'
         return
      end if
      call get_command_argument(2, corpus_path)
      checkpoint_path = ''
      output_path = ''
      format = 'text'
      maximum_vocabulary = 20000_int64
      reserve_fraction = 0.25_real64
      requested_threads = 0_int32
      context_mode = PT_CONTEXT_HIERARCHICAL
      available_memory = 0_int64
      available_storage = 0_int64
      memory_supplied = .false.
      storage_supplied = .false.
      require_ready = .false.
      index = 3
      do while (index <= argument_count)
         call get_command_argument(index, option)
         select case (trim(option))
         case ('--checkpoint')
            call option_value(argument_count, index, checkpoint_path, status)
         case ('--max-vocab')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               read(value, *, iostat=io_status) maximum_vocabulary
               if (io_status /= 0) status = FC_ERR_INVALID_ARGUMENT
            end if
         case ('--context-mode')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               select case (trim(value))
               case ('legacy')
                  context_mode = PT_CONTEXT_LEGACY_POSITIONAL
               case ('positional')
                  context_mode = PT_CONTEXT_POSITIONAL
               case ('hybrid')
                  context_mode = PT_CONTEXT_HYBRID
               case ('hierarchical')
                  context_mode = PT_CONTEXT_HIERARCHICAL
               case ('coherent')
                  context_mode = PT_CONTEXT_COHERENT
               case ('debiased')
                  context_mode = PT_CONTEXT_DEBIASED
               case ('resonant')
                  context_mode = PT_CONTEXT_RESONANT
               case default
                  status = FC_ERR_INVALID_ARGUMENT
               end select
            end if
         case ('--reserve-fraction')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               read(value, *, iostat=io_status) reserve_fraction
               if (io_status /= 0) status = FC_ERR_INVALID_ARGUMENT
            end if
         case ('--omp-threads')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               read(value, *, iostat=io_status) requested_threads
               if (io_status /= 0 .or. requested_threads <= 0_int32) status = FC_ERR_INVALID_ARGUMENT
            end if
         case ('--available-memory-bytes')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               read(value, *, iostat=io_status) available_memory
               if (io_status /= 0 .or. available_memory < 0_int64) status = FC_ERR_INVALID_ARGUMENT
            end if
            memory_supplied = status == FC_OK
         case ('--available-storage-bytes')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               read(value, *, iostat=io_status) available_storage
               if (io_status /= 0 .or. available_storage < 0_int64) status = FC_ERR_INVALID_ARGUMENT
            end if
            storage_supplied = status == FC_OK
         case ('--format')
            call option_value(argument_count, index, format, status)
         case ('--output')
            call option_value(argument_count, index, output_path, status)
         case ('--require-ready')
            require_ready = .true.
            index = index + 1
         case default
            write(error_unit, '(A,A)') 'vegai_control resource-plan: unsupported option: ', trim(option)
            status = FC_ERR_INVALID_ARGUMENT
         end select
         if (status /= FC_OK) then
            write(error_unit, '(A,A)') 'vegai_control resource-plan: invalid value for ', trim(option)
            return
         end if
      end do
      if (trim(format) /= 'text' .and. trim(format) /= 'json') then
         write(error_unit, '(A)') 'vegai_control resource-plan: --format must be text or json'
         status = FC_ERR_INVALID_ARGUMENT
         return
      end if
      inquire(file=trim(corpus_path), exist=exists, size=corpus_bytes, iostat=io_status)
      if (.not. exists .or. io_status /= 0) then
         write(error_unit, '(A,A)') 'vegai_control resource-plan: cannot inspect corpus: ', trim(corpus_path)
         status = FC_ERR_IO
         return
      end if
      call fortran_resource_plan_estimate(maximum_vocabulary, reserve_fraction, requested_threads, plan, status, &
                                          context_mode=context_mode)
      if (status /= FC_OK) then
         write(error_unit, '(A)') 'vegai_control resource-plan: invalid resource bounds'
         return
      end if
      plan%available_memory_known = memory_supplied
      plan%available_storage_known = storage_supplied
      plan%available_memory_bytes = available_memory
      plan%available_storage_bytes = available_storage
      plan%ready = .true.
      if (memory_supplied .and. available_memory < plan%required_available_bytes) plan%ready = .false.
      if (storage_supplied .and. available_storage < plan%estimated_checkpoint_bytes) plan%ready = .false.
      if (require_ready .and. (.not. memory_supplied .or. .not. storage_supplied)) plan%ready = .false.

      call emit_resource_plan(output_unit, trim(format), trim(corpus_path), corpus_bytes, trim(checkpoint_path), &
                              require_ready, plan)
      if (len_trim(output_path) > 0) then
         call emit_resource_plan_file(trim(output_path), trim(format), trim(corpus_path), corpus_bytes, &
                                      trim(checkpoint_path), require_ready, plan, status)
         if (status /= FC_OK) return
      end if
      if (require_ready .and. .not. plan%ready) then
         write(error_unit, '(A)') 'vegai_control resource-plan: NOT READY'
         status = FC_ERR_NOT_READY
      else
         status = FC_OK
      end if
   end subroutine resource_plan_command

   subroutine smoke_command(argument_count, status)
      integer, intent(in) :: argument_count
      integer(int32), intent(out) :: status
      character(len=512) :: corpus_path, checkpoint_path
      type(hdc_pretrain_model_t) :: model, loaded
      type(hdc_pretrain_config_t) :: config
      type(pretrain_metrics_t) :: vocabulary_metrics, training_metrics
      type(pretrain_execution_report_t) :: execution

      status = FC_ERR_INVALID_ARGUMENT
      if (argument_count /= 3) then
         write(error_unit, '(A)') 'vegai_control smoke: expected CORPUS CHECKPOINT'
         return
      end if
      call get_command_argument(2, corpus_path)
      call get_command_argument(3, checkpoint_path)
      config = hdc_pretrain_config_t()
      config%max_vocabulary = 64_int32
      config%min_frequency = 2_int64
      config%context_window = 2_int32
      config%position_shift_bits = 17_int32
      config%epochs = 1_int32
      config%strict_tokenization = .true.

      call pretrain_build_vocabulary(model, trim(corpus_path), config, vocabulary_metrics, status)
      if (status /= PT_OK) then
         write(error_unit, '(A,I0)') 'vegai_control smoke: vocabulary pass failed, status=', status
         status = FC_ERR_TRAINING
         call pretrain_model_destroy(model)
         return
      end if
      call pretrain_train_file(model, trim(corpus_path), config, training_metrics, status, execution)
      if (status /= PT_OK) then
         write(error_unit, '(A,I0)') 'vegai_control smoke: training pass failed, status=', status
         status = FC_ERR_TRAINING
         call pretrain_model_destroy(model)
         return
      end if
      call pretrain_checkpoint_save(model, trim(checkpoint_path), status)
      if (status /= PT_OK) then
         write(error_unit, '(A,I0)') 'vegai_control smoke: checkpoint save failed, status=', status
         status = FC_ERR_TRAINING
         call pretrain_model_destroy(model)
         return
      end if
      call pretrain_checkpoint_load(trim(checkpoint_path), loaded, status)
      if (status /= PT_OK .or. .not. pretrain_model_is_ready(loaded)) then
         write(error_unit, '(A,I0)') 'vegai_control smoke: checkpoint load failed, status=', status
         status = FC_ERR_TRAINING
         call pretrain_model_destroy(model)
         call pretrain_model_destroy(loaded)
         return
      end if
      if (loaded%training_epochs_completed /= 1_int64 .or. &
          loaded%totals%raw_tokens /= training_metrics%raw_tokens) then
         write(error_unit, '(A)') 'vegai_control smoke: loaded checkpoint metrics do not match native train'
         status = FC_ERR_TRAINING
         call pretrain_model_destroy(model)
         call pretrain_model_destroy(loaded)
         return
      end if
      write(output_unit, '(A)') 'vegai_control native Fortran smoke: PASS'
      write(output_unit, '(A,I0)') '  raw tokens: ', training_metrics%raw_tokens
      write(output_unit, '(A,I0)') '  active tokens: ', training_metrics%active_tokens
      write(output_unit, '(A,I0)') '  prototype updates: ', training_metrics%center_updates
      write(output_unit, '(A,I0)') '  CPU contexts: ', execution%cpu_contexts
      write(output_unit, '(A,A)') '  checkpoint: ', trim(checkpoint_path)
      call pretrain_model_destroy(model)
      call pretrain_model_destroy(loaded)
      status = FC_OK
   end subroutine smoke_command

   subroutine quality_gate_command(argument_count, status)
      integer, intent(in) :: argument_count
      integer(int32), intent(out) :: status
      type(hdc_pretrain_model_t) :: model
      type(quality_case_t), allocatable :: cases(:)
      type(quality_policy_t) :: policy
      type(quality_report_t) :: report
      character(len=512) :: checkpoint_path, cases_path, output_path, format, option, value
      integer(int32) :: gate_status
      integer :: index, io_status
      logical :: have_minimum_cases, have_minimum_accuracy, have_minimum_wilson
      logical :: have_minimum_mean, have_minimum_case, have_maximum_incorrect

      status = FC_ERR_INVALID_ARGUMENT
      if (argument_count < 3) then
         write(error_unit, '(A)') 'vegai_control quality-gate: expected CHECKPOINT CASES'
         return
      end if
      call get_command_argument(2, checkpoint_path)
      call get_command_argument(3, cases_path)
      output_path = ''
      format = 'text'
      policy = quality_policy_t()
      have_minimum_cases = .false.
      have_minimum_accuracy = .false.
      have_minimum_wilson = .false.
      have_minimum_mean = .false.
      have_minimum_case = .false.
      have_maximum_incorrect = .false.
      status = FC_OK
      index = 4
      do while (index <= argument_count)
         call get_command_argument(index, option)
         select case (trim(option))
         case ('--minimum-cases')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               read(value, *, iostat=io_status) policy%minimum_cases
               if (io_status /= 0) status = FC_ERR_INVALID_ARGUMENT
            end if
            have_minimum_cases = status == FC_OK
         case ('--minimum-top1-accuracy')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               read(value, *, iostat=io_status) policy%minimum_top1_accuracy
               if (io_status /= 0) status = FC_ERR_INVALID_ARGUMENT
            end if
            have_minimum_accuracy = status == FC_OK
         case ('--minimum-top1-wilson-lower-bound-95')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               read(value, *, iostat=io_status) policy%minimum_top1_wilson_lower_bound_95
               if (io_status /= 0) status = FC_ERR_INVALID_ARGUMENT
            end if
            have_minimum_wilson = status == FC_OK
         case ('--minimum-mean-similarity')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               read(value, *, iostat=io_status) policy%minimum_mean_similarity
               if (io_status /= 0) status = FC_ERR_INVALID_ARGUMENT
            end if
            have_minimum_mean = status == FC_OK
         case ('--minimum-case-similarity')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               read(value, *, iostat=io_status) policy%minimum_case_similarity
               if (io_status /= 0) status = FC_ERR_INVALID_ARGUMENT
            end if
            have_minimum_case = status == FC_OK
         case ('--maximum-incorrect-cases')
            call option_value(argument_count, index, value, status)
            if (status == FC_OK) then
               read(value, *, iostat=io_status) policy%maximum_incorrect_cases
               if (io_status /= 0) status = FC_ERR_INVALID_ARGUMENT
            end if
            have_maximum_incorrect = status == FC_OK
         case ('--format')
            call option_value(argument_count, index, format, status)
         case ('--output')
            call option_value(argument_count, index, output_path, status)
         case default
            write(error_unit, '(A,A)') 'vegai_control quality-gate: unsupported option: ', trim(option)
            status = FC_ERR_INVALID_ARGUMENT
         end select
         if (status /= FC_OK) then
            write(error_unit, '(A,A)') 'vegai_control quality-gate: invalid value for ', trim(option)
            return
         end if
      end do
      if (.not. have_minimum_cases .or. .not. have_minimum_accuracy .or. .not. have_minimum_wilson .or. &
          .not. have_minimum_mean .or. .not. have_minimum_case .or. .not. have_maximum_incorrect) then
         write(error_unit, '(A)') 'vegai_control quality-gate: every policy threshold must be explicit'
         status = FC_ERR_INVALID_ARGUMENT
         return
      end if
      if (trim(format) /= 'text' .and. trim(format) /= 'json') then
         write(error_unit, '(A)') 'vegai_control quality-gate: --format must be text or json'
         status = FC_ERR_INVALID_ARGUMENT
         return
      end if
      if (.not. quality_policy_is_valid(policy)) then
         write(error_unit, '(A)') 'vegai_control quality-gate: policy bounds are invalid'
         status = FC_ERR_INVALID_ARGUMENT
         return
      end if

      call pretrain_checkpoint_load(trim(checkpoint_path), model, gate_status)
      if (gate_status /= PT_OK .or. .not. pretrain_model_is_ready(model)) then
         write(error_unit, '(A,I0)') 'vegai_control quality-gate: checkpoint load failed, status=', gate_status
         call pretrain_model_destroy(model)
         status = FC_ERR_TRAINING
         return
      end if
      call quality_case_file_load(trim(cases_path), cases, gate_status)
      if (gate_status /= QG_OK) then
         write(error_unit, '(A,I0)') 'vegai_control quality-gate: cases load failed, status=', gate_status
         call pretrain_model_destroy(model)
         if (allocated(cases)) deallocate(cases)
         if (gate_status == QG_ERR_IO) then
            status = FC_ERR_IO
         else
            status = FC_ERR_QUALITY
         end if
         return
      end if
      call quality_gate_evaluate(model, cases, policy, report, gate_status)
      call emit_quality_report(output_unit, trim(format), trim(checkpoint_path), trim(cases_path), policy, report)
      if (len_trim(output_path) > 0) then
         call emit_quality_report_file(trim(output_path), trim(format), trim(checkpoint_path), trim(cases_path), &
                                       policy, report, status)
         if (status /= FC_OK) then
            call pretrain_model_destroy(model)
            deallocate(cases)
            return
         end if
      end if
      call pretrain_model_destroy(model)
      deallocate(cases)
      if (gate_status == QG_OK) then
         status = FC_OK
      else
         write(error_unit, '(A)') 'vegai_control quality-gate: REJECTED'
         status = FC_ERR_QUALITY
      end if
   end subroutine quality_gate_command

   subroutine attest_command(argument_count, status)
      integer, intent(in) :: argument_count
      integer(int32), intent(out) :: status
      type(hdc_pretrain_model_t) :: model
      character(len=512) :: corpus_path, checkpoint_path, quality_report_path, output_path, format, option
      character(len=64) :: corpus_digest, checkpoint_digest, quality_report_digest
      integer(int64) :: corpus_bytes, checkpoint_bytes, quality_report_bytes
      integer(int32) :: integrity_status, checkpoint_status
      integer :: index

      status = FC_ERR_INVALID_ARGUMENT
      if (argument_count < 3) then
         write(error_unit, '(A)') 'vegai_control attest: expected CORPUS CHECKPOINT'
         return
      end if
      call get_command_argument(2, corpus_path)
      call get_command_argument(3, checkpoint_path)
      quality_report_path = ''
      output_path = ''
      format = 'text'
      status = FC_OK
      index = 4
      do while (index <= argument_count)
         call get_command_argument(index, option)
         select case (trim(option))
         case ('--quality-report')
            call option_value(argument_count, index, quality_report_path, status)
         case ('--format')
            call option_value(argument_count, index, format, status)
         case ('--output')
            call option_value(argument_count, index, output_path, status)
         case default
            write(error_unit, '(A,A)') 'vegai_control attest: unsupported option: ', trim(option)
            status = FC_ERR_INVALID_ARGUMENT
         end select
         if (status /= FC_OK) then
            write(error_unit, '(A,A)') 'vegai_control attest: invalid value for ', trim(option)
            return
         end if
      end do
      if (trim(format) /= 'text' .and. trim(format) /= 'json') then
         write(error_unit, '(A)') 'vegai_control attest: --format must be text or json'
         status = FC_ERR_INVALID_ARGUMENT
         return
      end if
      if (len_trim(output_path) > 0 .and. &
          (trim(output_path) == trim(corpus_path) .or. trim(output_path) == trim(checkpoint_path) .or. &
           trim(output_path) == trim(quality_report_path))) then
         write(error_unit, '(A)') 'vegai_control attest: output must not overwrite an attested input'
         status = FC_ERR_INVALID_ARGUMENT
         return
      end if

      call sha256_file(trim(corpus_path), corpus_digest, corpus_bytes, integrity_status)
      if (integrity_status /= FI_OK) then
         write(error_unit, '(A,I0)') 'vegai_control attest: corpus hash failed, status=', integrity_status
         status = FC_ERR_ATTESTATION
         return
      end if
      call sha256_file(trim(checkpoint_path), checkpoint_digest, checkpoint_bytes, integrity_status)
      if (integrity_status /= FI_OK) then
         write(error_unit, '(A,I0)') 'vegai_control attest: checkpoint hash failed, status=', integrity_status
         status = FC_ERR_ATTESTATION
         return
      end if
      ! Bind the identity to a checkpoint that the native reader actually
      ! accepts; a byte hash alone does not establish Vegai checkpoint validity.
      call pretrain_checkpoint_load(trim(checkpoint_path), model, checkpoint_status)
      if (checkpoint_status /= PT_OK .or. .not. pretrain_model_is_ready(model)) then
         write(error_unit, '(A,I0)') 'vegai_control attest: checkpoint validation failed, status=', checkpoint_status
         call pretrain_model_destroy(model)
         status = FC_ERR_ATTESTATION
         return
      end if
      call pretrain_model_destroy(model)
      quality_report_digest = ''
      quality_report_bytes = 0_int64
      if (len_trim(quality_report_path) > 0) then
         call sha256_file(trim(quality_report_path), quality_report_digest, quality_report_bytes, integrity_status)
         if (integrity_status /= FI_OK) then
            write(error_unit, '(A,I0)') 'vegai_control attest: quality report hash failed, status=', integrity_status
            status = FC_ERR_ATTESTATION
            return
         end if
      end if
      call emit_attestation_report(output_unit, trim(format), trim(corpus_path), corpus_digest, corpus_bytes, &
                                   trim(checkpoint_path), checkpoint_digest, checkpoint_bytes, &
                                   trim(quality_report_path), quality_report_digest, quality_report_bytes)
      if (len_trim(output_path) > 0) then
         call emit_attestation_report_file(trim(output_path), trim(format), trim(corpus_path), corpus_digest, corpus_bytes, &
                                           trim(checkpoint_path), checkpoint_digest, checkpoint_bytes, &
                                           trim(quality_report_path), quality_report_digest, quality_report_bytes, status)
         if (status /= FC_OK) return
      end if
      status = FC_OK
   end subroutine attest_command

   subroutine option_value(argument_count, index, value, status)
      integer, intent(in) :: argument_count
      integer, intent(inout) :: index
      character(len=*), intent(out) :: value
      integer(int32), intent(out) :: status

      value = ''
      status = FC_ERR_INVALID_ARGUMENT
      if (index >= argument_count) return
      call get_command_argument(index + 1, value)
      index = index + 2
      status = FC_OK
   end subroutine option_value

   subroutine emit_profile(unit, format)
      integer, intent(in) :: unit
      character(len=*), intent(in) :: format
      character(len=512) :: omp_threads
      integer :: value_length, environment_status

      omp_threads = ''
      call get_environment_variable('OMP_NUM_THREADS', omp_threads, length=value_length, status=environment_status)
      if (environment_status /= 0 .or. value_length <= 0) omp_threads = 'not-set'
      if (trim(format) == 'json') then
         write(unit, '(A)') '{'
         write(unit, '(A)') '  "schema": "vegai-fortran-native-profile/v1",'
         write(unit, '(A)') '  "language": "Fortran 2023",'
         write(unit, '(A)') '  "compiler_version": ' // json_quote(compiler_version()) // ','
         write(unit, '(A)') '  "compiler_options": ' // json_quote(compiler_options()) // ','
         write(unit, '(A)') '  "omp_num_threads_environment": ' // json_quote(trim(omp_threads)) // ','
         write(unit, '(A)') '  "execution_backend": "portable-cpu",'
         write(unit, '(A)') '  "gpu_compute_used": false,'
         write(unit, '(A)') '  "npu_compute_used": false,'
         write(unit, '(A)') '  "host_inventory": "operator-supplied; standard Fortran performs no OS shell probe",'
         write(unit, '(A)') '  "checkpoint_contract": "same compiler/ABI/endianness family only"'
         write(unit, '(A)') '}'
      else
         write(unit, '(A)') 'Vegai native Fortran profile'
         write(unit, '(A,A)') 'Language: ', 'Fortran 2023'
         write(unit, '(A,A)') 'Compiler: ', trim(compiler_version())
         write(unit, '(A,A)') 'Compiler options: ', trim(compiler_options())
         write(unit, '(A,A)') 'OMP_NUM_THREADS: ', trim(omp_threads)
         write(unit, '(A)') 'Execution backend: portable CPU'
         write(unit, '(A)') 'GPU/NPU compute: not claimed by this native profile'
         write(unit, '(A)') 'Host inventory: provide operator-audited values; no OS shell probe is used'
         write(unit, '(A)') 'Checkpoint contract: same compiler/ABI/endianness family only'
      end if
   end subroutine emit_profile

   subroutine emit_profile_file(path, format, status)
      character(len=*), intent(in) :: path, format
      integer(int32), intent(out) :: status
      integer :: unit, io_status

      open(newunit=unit, file=trim(path), status='replace', action='write', form='formatted', iostat=io_status)
      if (io_status /= 0) then
         write(error_unit, '(A,A)') 'vegai_control profile: cannot write output: ', trim(path)
         status = FC_ERR_IO
         return
      end if
      call emit_profile(unit, format)
      close(unit)
      status = FC_OK
   end subroutine emit_profile_file

   subroutine emit_resource_plan(unit, format, corpus_path, corpus_bytes, checkpoint_path, require_ready, plan)
      integer, intent(in) :: unit
      character(len=*), intent(in) :: format, corpus_path, checkpoint_path
      integer(int64), intent(in) :: corpus_bytes
      logical, intent(in) :: require_ready
      type(fortran_resource_plan_t), intent(in) :: plan
      character(len=:), allocatable :: checkpoint_rendered

      if (len_trim(checkpoint_path) > 0) then
         checkpoint_rendered = json_quote(checkpoint_path)
      else
         checkpoint_rendered = 'null'
      end if
      if (trim(format) == 'json') then
         write(unit, '(A)') '{'
         write(unit, '(A)') '  "schema": "vegai-fortran-resource-plan/v1",'
         write(unit, '(A)') '  "language": "Fortran 2023",'
         write(unit, '(A)') '  "corpus_path": ' // json_quote(corpus_path) // ','
         write(unit, '(A)') '  "corpus_bytes": ' // integer_text(corpus_bytes) // ','
         write(unit, '(A)') '  "checkpoint_path": ' // checkpoint_rendered // ','
         write(unit, '(A)') '  "max_vocabulary": ' // integer_text(plan%max_vocabulary) // ','
         write(unit, '(A)') '  "context_mode": ' // json_quote(trim(pretrain_context_mode_name(plan%context_mode))) // ','
         write(unit, '(A)') '  "resonant_snapshot_bytes": ' // integer_text(plan%resonant_snapshot_bytes) // ','
         write(unit, '(A)') '  "required_available_bytes": ' // integer_text(plan%required_available_bytes) // ','
         write(unit, '(A)') '  "estimated_checkpoint_bytes": ' // integer_text(plan%estimated_checkpoint_bytes) // ','
         write(unit, '(A)') '  "reserve_bytes": ' // integer_text(plan%reserve_bytes) // ','
         write(unit, '(A)') '  "recommended_omp_threads": ' // integer32_text(plan%recommended_omp_threads) // ','
         write(unit, '(A)') '  "available_memory_bytes": ' // optional_integer_text(plan%available_memory_known, &
                              plan%available_memory_bytes) // ','
         write(unit, '(A)') '  "available_storage_bytes": ' // optional_integer_text(plan%available_storage_known, &
                              plan%available_storage_bytes) // ','
         write(unit, '(A)') '  "require_ready": ' // logical_text(require_ready) // ','
         write(unit, '(A)') '  "ready": ' // logical_text(plan%ready) // ','
         write(unit, '(A)') '  "inventory_contract": "RAM and free storage are explicit operator inputs in the pure Fortran path"'
         write(unit, '(A)') '}'
      else
         write(unit, '(A)') 'Vegai native Fortran resource plan'
         write(unit, '(A,A)') 'Corpus: ', trim(corpus_path)
         write(unit, '(A,I0)') 'Corpus bytes: ', corpus_bytes
         if (len_trim(checkpoint_path) > 0) write(unit, '(A,A)') 'Checkpoint: ', trim(checkpoint_path)
         write(unit, '(A,I0)') 'Max vocabulary: ', plan%max_vocabulary
         write(unit, '(A,A)') 'Context mode: ', trim(pretrain_context_mode_name(plan%context_mode))
         write(unit, '(A,I0)') 'Model payload bytes: ', plan%model_payload_bytes
         write(unit, '(A,I0)') 'Resonant snapshot bytes: ', plan%resonant_snapshot_bytes
         write(unit, '(A,I0)') 'Vocabulary allowance bytes: ', plan%vocabulary_allowance_bytes
         write(unit, '(A,I0)') 'Context staging bytes: ', plan%context_staging_bytes
         write(unit, '(A,I0)') 'Line-update scratch bytes: ', plan%line_update_scratch_bytes
         write(unit, '(A,I0)') 'Lookup-cache workspace bytes: ', plan%lookup_cache_workspace_bytes
         write(unit, '(A,I0)') 'Reserve bytes: ', plan%reserve_bytes
         write(unit, '(A,I0)') 'Required available RAM bytes: ', plan%required_available_bytes
         write(unit, '(A,I0)') 'Estimated checkpoint bytes: ', plan%estimated_checkpoint_bytes
         write(unit, '(A,I0)') 'Recommended OMP threads: ', plan%recommended_omp_threads
         if (plan%available_memory_known) then
            write(unit, '(A,I0)') 'Operator available RAM bytes: ', plan%available_memory_bytes
         else
            write(unit, '(A)') 'Operator available RAM bytes: unknown'
         end if
         if (plan%available_storage_known) then
            write(unit, '(A,I0)') 'Operator available storage bytes: ', plan%available_storage_bytes
         else
            write(unit, '(A)') 'Operator available storage bytes: unknown'
         end if
         write(unit, '(A,L1)') 'Require ready: ', require_ready
         write(unit, '(A,L1)') 'Ready: ', plan%ready
         write(unit, '(A)') 'Inventory contract: RAM/free storage are explicit operator inputs in the pure Fortran path'
      end if
   end subroutine emit_resource_plan

   subroutine emit_resource_plan_file(path, format, corpus_path, corpus_bytes, checkpoint_path, require_ready, plan, status)
      character(len=*), intent(in) :: path, format, corpus_path, checkpoint_path
      integer(int64), intent(in) :: corpus_bytes
      logical, intent(in) :: require_ready
      type(fortran_resource_plan_t), intent(in) :: plan
      integer(int32), intent(out) :: status
      integer :: unit, io_status

      open(newunit=unit, file=trim(path), status='replace', action='write', form='formatted', iostat=io_status)
      if (io_status /= 0) then
         write(error_unit, '(A,A)') 'vegai_control resource-plan: cannot write output: ', trim(path)
         status = FC_ERR_IO
         return
      end if
      call emit_resource_plan(unit, format, corpus_path, corpus_bytes, checkpoint_path, require_ready, plan)
      close(unit)
      status = FC_OK
   end subroutine emit_resource_plan_file

   subroutine emit_quality_report(unit, format, checkpoint_path, cases_path, policy, report)
      integer, intent(in) :: unit
      character(len=*), intent(in) :: format, checkpoint_path, cases_path
      type(quality_policy_t), intent(in) :: policy
      type(quality_report_t), intent(in) :: report

      if (trim(format) == 'json') then
         write(unit, '(A)') '{'
         write(unit, '(A)') '  "schema": "vegai-fortran-quality-gate/v1",'
         write(unit, '(A)') '  "checkpoint_path": ' // json_quote(checkpoint_path) // ','
         write(unit, '(A)') '  "cases_path": ' // json_quote(cases_path) // ','
         write(unit, '(A)') '  "policy": {'
         write(unit, '(A)') '    "minimum_cases": ' // integer32_text(policy%minimum_cases) // ','
         write(unit, '(A)') '    "minimum_top1_accuracy": ' // real_text(policy%minimum_top1_accuracy) // ','
         write(unit, '(A)') '    "minimum_top1_wilson_lower_bound_95": ' // &
                              real_text(policy%minimum_top1_wilson_lower_bound_95) // ','
         write(unit, '(A)') '    "minimum_mean_similarity": ' // real_text(policy%minimum_mean_similarity) // ','
         write(unit, '(A)') '    "minimum_case_similarity": ' // real_text(policy%minimum_case_similarity) // ','
         write(unit, '(A)') '    "maximum_incorrect_cases": ' // &
                              integer32_text(policy%maximum_incorrect_cases)
         write(unit, '(A)') '  },'
         write(unit, '(A)') '  "result": {'
         write(unit, '(A)') '    "declared_cases": ' // integer32_text(report%declared_cases) // ','
         write(unit, '(A)') '    "evaluated_cases": ' // integer32_text(report%evaluated_cases) // ','
         write(unit, '(A)') '    "correct_cases": ' // integer32_text(report%correct_cases) // ','
         write(unit, '(A)') '    "incorrect_cases": ' // integer32_text(report%incorrect_cases) // ','
         write(unit, '(A)') '    "unavailable_cases": ' // integer32_text(report%unavailable_cases) // ','
         write(unit, '(A)') '    "top1_accuracy": ' // real_text(report%top1_accuracy) // ','
         write(unit, '(A)') '    "top1_wilson_lower_bound_95": ' // &
                              real_text(report%top1_wilson_lower_bound_95) // ','
         write(unit, '(A)') '    "mean_similarity": ' // real_text(report%mean_similarity) // ','
         write(unit, '(A)') '    "minimum_observed_similarity": ' // &
                              real_text(report%minimum_observed_similarity) // ','
         write(unit, '(A)') '    "passed": ' // logical_text(report%passed)
         write(unit, '(A)') '  },'
         write(unit, '(A)') '  "scope": "held-out top-1 HDC semantic-neighbour gate; not a general LMM quality claim"'
         write(unit, '(A)') '}'
      else
         write(unit, '(A)') 'Vegai native Fortran quality gate'
         write(unit, '(A,A)') 'Checkpoint: ', trim(checkpoint_path)
         write(unit, '(A,A)') 'Cases: ', trim(cases_path)
         write(unit, '(A,I0)') 'Declared cases: ', report%declared_cases
         write(unit, '(A,I0)') 'Evaluated cases: ', report%evaluated_cases
         write(unit, '(A,I0)') 'Correct cases: ', report%correct_cases
         write(unit, '(A,I0)') 'Incorrect cases: ', report%incorrect_cases
         write(unit, '(A,I0)') 'Unavailable cases: ', report%unavailable_cases
         write(unit, '(A,ES14.6E3)') 'Top-1 accuracy: ', report%top1_accuracy
         write(unit, '(A,ES14.6E3)') 'Top-1 Wilson lower bound 95%: ', report%top1_wilson_lower_bound_95
         write(unit, '(A,ES14.6E3)') 'Mean similarity: ', report%mean_similarity
         write(unit, '(A,ES14.6E3)') 'Minimum observed similarity: ', report%minimum_observed_similarity
         write(unit, '(A,L1)') 'Passed: ', report%passed
         write(unit, '(A)') 'Scope: held-out top-1 HDC semantic-neighbour gate; not a general LMM quality claim'
      end if
   end subroutine emit_quality_report

   subroutine emit_quality_report_file(path, format, checkpoint_path, cases_path, policy, report, status)
      character(len=*), intent(in) :: path, format, checkpoint_path, cases_path
      type(quality_policy_t), intent(in) :: policy
      type(quality_report_t), intent(in) :: report
      integer(int32), intent(out) :: status
      integer :: unit, io_status

      open(newunit=unit, file=trim(path), status='replace', action='write', form='formatted', iostat=io_status)
      if (io_status /= 0) then
         write(error_unit, '(A,A)') 'vegai_control quality-gate: cannot write output: ', trim(path)
         status = FC_ERR_IO
         return
      end if
      call emit_quality_report(unit, format, checkpoint_path, cases_path, policy, report)
      close(unit)
      status = FC_OK
   end subroutine emit_quality_report_file

   subroutine emit_attestation_report(unit, format, corpus_path, corpus_digest, corpus_bytes, checkpoint_path, &
                                      checkpoint_digest, checkpoint_bytes, quality_report_path, quality_report_digest, &
                                      quality_report_bytes)
      integer, intent(in) :: unit
      character(len=*), intent(in) :: format, corpus_path, corpus_digest, checkpoint_path, checkpoint_digest
      character(len=*), intent(in) :: quality_report_path, quality_report_digest
      integer(int64), intent(in) :: corpus_bytes, checkpoint_bytes, quality_report_bytes
      character(len=:), allocatable :: quality_report_rendered

      if (len_trim(quality_report_path) > 0) then
         quality_report_rendered = '{"path": ' // json_quote(quality_report_path) // ', "bytes": ' // &
                                    integer_text(quality_report_bytes) // ', "sha256": ' // &
                                    json_quote(quality_report_digest) // '}'
      else
         quality_report_rendered = 'null'
      end if
      if (trim(format) == 'json') then
         write(unit, '(A)') '{'
         write(unit, '(A)') '  "schema": "vegai-fortran-attestation/v1",'
         write(unit, '(A)') '  "corpus": {"path": ' // json_quote(corpus_path) // ', "bytes": ' // &
                            integer_text(corpus_bytes) // ', "sha256": ' // json_quote(corpus_digest) // '},'
         write(unit, '(A)') '  "checkpoint": {"path": ' // json_quote(checkpoint_path) // ', "bytes": ' // &
                            integer_text(checkpoint_bytes) // ', "sha256": ' // json_quote(checkpoint_digest) // '},'
         write(unit, '(A)') '  "quality_report": ' // quality_report_rendered // ','
         write(unit, '(A)') '  "checkpoint_validation": "loaded by the native Fortran checkpoint reader",'
         write(unit, '(A)') '  "integrity_scope": "SHA-256 only; no signature, provenance, lock, or atomic publish claim"'
         write(unit, '(A)') '}'
      else
         write(unit, '(A)') 'Vegai native Fortran attestation identity'
         write(unit, '(A,A)') 'Corpus: ', trim(corpus_path)
         write(unit, '(A,I0)') 'Corpus bytes: ', corpus_bytes
         write(unit, '(A,A)') 'Corpus SHA-256: ', trim(corpus_digest)
         write(unit, '(A,A)') 'Checkpoint: ', trim(checkpoint_path)
         write(unit, '(A,I0)') 'Checkpoint bytes: ', checkpoint_bytes
         write(unit, '(A,A)') 'Checkpoint SHA-256: ', trim(checkpoint_digest)
         if (len_trim(quality_report_path) > 0) then
            write(unit, '(A,A)') 'Quality report: ', trim(quality_report_path)
            write(unit, '(A,I0)') 'Quality report bytes: ', quality_report_bytes
            write(unit, '(A,A)') 'Quality report SHA-256: ', trim(quality_report_digest)
         end if
         write(unit, '(A)') 'Checkpoint validation: loaded by the native Fortran checkpoint reader'
         write(unit, '(A)') 'Integrity scope: SHA-256 only; no signature, provenance, lock, or atomic publish claim'
      end if
   end subroutine emit_attestation_report

   subroutine emit_attestation_report_file(path, format, corpus_path, corpus_digest, corpus_bytes, checkpoint_path, &
                                           checkpoint_digest, checkpoint_bytes, quality_report_path, quality_report_digest, &
                                           quality_report_bytes, status)
      character(len=*), intent(in) :: path, format, corpus_path, corpus_digest, checkpoint_path, checkpoint_digest
      character(len=*), intent(in) :: quality_report_path, quality_report_digest
      integer(int64), intent(in) :: corpus_bytes, checkpoint_bytes, quality_report_bytes
      integer(int32), intent(out) :: status
      integer :: unit, io_status

      open(newunit=unit, file=trim(path), status='replace', action='write', form='formatted', iostat=io_status)
      if (io_status /= 0) then
         write(error_unit, '(A,A)') 'vegai_control attest: cannot write output: ', trim(path)
         status = FC_ERR_IO
         return
      end if
      call emit_attestation_report(unit, format, corpus_path, corpus_digest, corpus_bytes, checkpoint_path, &
                                   checkpoint_digest, checkpoint_bytes, quality_report_path, quality_report_digest, &
                                   quality_report_bytes)
      close(unit)
      status = FC_OK
   end subroutine emit_attestation_report_file

   function integer_text(value) result(rendered)
      integer(int64), intent(in) :: value
      character(len=:), allocatable :: rendered
      character(len=64) :: buffer

      write(buffer, '(I0)') value
      rendered = trim(buffer)
   end function integer_text

   function integer32_text(value) result(rendered)
      integer(int32), intent(in) :: value
      character(len=:), allocatable :: rendered
      character(len=32) :: buffer

      write(buffer, '(I0)') value
      rendered = trim(buffer)
   end function integer32_text

   function real_text(value) result(rendered)
      real(real64), intent(in) :: value
      character(len=:), allocatable :: rendered
      character(len=64) :: buffer

      write(buffer, '(ES24.16E3)') value
      rendered = trim(adjustl(buffer))
   end function real_text

   function optional_integer_text(known, value) result(rendered)
      logical, intent(in) :: known
      integer(int64), intent(in) :: value
      character(len=:), allocatable :: rendered

      if (known) then
         rendered = integer_text(value)
      else
         rendered = 'null'
      end if
   end function optional_integer_text

   function logical_text(value) result(rendered)
      logical, intent(in) :: value
      character(len=:), allocatable :: rendered

      if (value) then
         rendered = 'true'
      else
         rendered = 'false'
      end if
   end function logical_text

   function json_quote(value) result(rendered)
      character(len=*), intent(in) :: value
      character(len=:), allocatable :: rendered
      integer :: index
      character(len=1) :: character

      rendered = achar(34)
      do index = 1, len_trim(value)
         character = value(index:index)
         select case (character)
         case (achar(34))
            rendered = rendered // achar(92) // achar(34)
         case (achar(92))
            rendered = rendered // achar(92) // achar(92)
         case (achar(10))
            rendered = rendered // achar(92) // 'n'
         case (achar(13))
            rendered = rendered // achar(92) // 'r'
         case (achar(9))
            rendered = rendered // achar(92) // 't'
         case default
            if (iachar(character) < 32) then
               rendered = rendered // '?'
            else
               rendered = rendered // character
            end if
         end select
      end do
      rendered = rendered // achar(34)
   end function json_quote

   subroutine print_usage(unit)
      integer, intent(in) :: unit

      write(unit, '(A)') 'Vegai native Fortran control plane'
      write(unit, '(A)') 'Usage:'
      write(unit, '(A)') '  vegai_control profile [--format text|json] [--output PATH]'
      write(unit, '(A)') '  vegai_control resource-plan CORPUS [--checkpoint PATH] [--max-vocab N] [--context-mode MODE]'
      write(unit, '(A)') '      [--reserve-fraction R] [--omp-threads N]; resonant reserves an epoch semantic snapshot'
      write(unit, '(A)') '      [--available-memory-bytes N] [--available-storage-bytes N]'
      write(unit, '(A)') '      [--require-ready] [--format text|json] [--output PATH]'
      write(unit, '(A)') '  vegai_control smoke CORPUS CHECKPOINT'
      write(unit, '(A)') '  vegai_control quality-gate CHECKPOINT CASES --minimum-cases N'
      write(unit, '(A)') '      --minimum-top1-accuracy R --minimum-top1-wilson-lower-bound-95 R'
      write(unit, '(A)') '      --minimum-mean-similarity R --minimum-case-similarity R'
      write(unit, '(A)') '      --maximum-incorrect-cases N [--format text|json] [--output PATH]'
      write(unit, '(A)') '  vegai_control attest CORPUS CHECKPOINT [--quality-report PATH]'
      write(unit, '(A)') '      [--format text|json] [--output PATH]'
      write(unit, '(A)') 'The native resource path does not invoke Python or OS shell inventory commands.'
   end subroutine print_usage

end module fortran_control
