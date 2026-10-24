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
   use hdc_pretraining, only: PT_OK, hdc_pretrain_model_t, hdc_pretrain_config_t, &
                              pretrain_metrics_t, pretrain_execution_report_t, &
                              pretrain_build_vocabulary, pretrain_train_file, &
                              pretrain_checkpoint_save, pretrain_checkpoint_load, &
                              pretrain_model_destroy, pretrain_model_is_ready
   implicit none
   private

   integer(int32), parameter, public :: FC_OK = 0_int32
   integer(int32), parameter, public :: FC_ERR_INVALID_ARGUMENT = -900_int32
   integer(int32), parameter, public :: FC_ERR_IO = -901_int32
   integer(int32), parameter, public :: FC_ERR_NOT_READY = -902_int32
   integer(int32), parameter, public :: FC_ERR_TRAINING = -903_int32

   ! These estimates intentionally match the documented portable HDC model
   ! contract. They are a preflight reservation, not a claim about allocator,
   ! OpenMP runtime, page cache, or any operating-system memory manager.
   integer(int64), parameter, public :: FC_MODEL_BYTES_PER_SLOT = 4104_int64
   integer(int64), parameter, public :: FC_VOCABULARY_ALLOWANCE_PER_SLOT = 256_int64
   integer(int64), parameter, public :: FC_CONTEXT_STAGING_BYTES = 2097152_int64
   integer(int64), parameter, public :: FC_LINE_UPDATE_SCRATCH_PER_SLOT = 4_int64
   integer(int64), parameter, public :: FC_LOOKUP_CACHE_BYTES = 278528_int64
   integer(int64), parameter, public :: FC_CHECKPOINT_BYTES_PER_SLOT = 4168_int64
   integer(int64), parameter, public :: FC_CHECKPOINT_FIXED_BYTES = 65536_int64

   type, public :: fortran_resource_plan_t
      integer(int64) :: max_vocabulary = 0_int64
      real(real64) :: reserve_fraction = 0.25_real64
      integer(int64) :: model_payload_bytes = 0_int64
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

   subroutine fortran_resource_plan_estimate(max_vocabulary, reserve_fraction, requested_threads, plan, status)
      integer(int64), intent(in) :: max_vocabulary
      real(real64), intent(in) :: reserve_fraction
      integer(int32), intent(in) :: requested_threads
      type(fortran_resource_plan_t), intent(out) :: plan
      integer(int32), intent(out) :: status
      integer(int64) :: resident_before_reserve

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

      plan%max_vocabulary = max_vocabulary
      plan%reserve_fraction = reserve_fraction
      plan%model_payload_bytes = max_vocabulary * FC_MODEL_BYTES_PER_SLOT
      plan%vocabulary_allowance_bytes = max_vocabulary * FC_VOCABULARY_ALLOWANCE_PER_SLOT
      plan%context_staging_bytes = FC_CONTEXT_STAGING_BYTES
      plan%line_update_scratch_bytes = max_vocabulary * FC_LINE_UPDATE_SCRATCH_PER_SLOT
      plan%lookup_cache_workspace_bytes = FC_LOOKUP_CACHE_BYTES
      resident_before_reserve = plan%model_payload_bytes + plan%vocabulary_allowance_bytes + &
                                 plan%context_staging_bytes + plan%line_update_scratch_bytes + &
                                 plan%lookup_cache_workspace_bytes
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
      integer(int32) :: requested_threads
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
      call fortran_resource_plan_estimate(maximum_vocabulary, reserve_fraction, requested_threads, plan, status)
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
         write(unit, '(A,I0)') 'Model payload bytes: ', plan%model_payload_bytes
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
      write(unit, '(A)') '  vegai_control resource-plan CORPUS [--checkpoint PATH] [--max-vocab N]'
      write(unit, '(A)') '      [--reserve-fraction R] [--omp-threads N]'
      write(unit, '(A)') '      [--available-memory-bytes N] [--available-storage-bytes N]'
      write(unit, '(A)') '      [--require-ready] [--format text|json] [--output PATH]'
      write(unit, '(A)') '  vegai_control smoke CORPUS CHECKPOINT'
      write(unit, '(A)') 'The native resource path does not invoke Python or OS shell inventory commands.'
   end subroutine print_usage

end module fortran_control
