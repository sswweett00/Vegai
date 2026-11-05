! SPDX-License-Identifier: MIT
! Bounded, no-update, per-record evaluation of a byte-model checkpoint.
program natural_eval
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real64, error_unit, output_unit, file_storage_size
   use portable_cli, only: CLI_OK, cli_get_argument, cli_valid_utf8
   use utf8_record_stream, only: UTF8_STREAM_OK, utf8_stream_reader_t, &
                                 utf8_stream_reader_init, utf8_stream_read_record
   use fixed_budget_context
   use fixed_budget_optimizer, only: fbc_update_config_t, fbc_optimizer_t, fbc_optimizer_destroy
   use fixed_budget_training, only: FBT_OK, fbc_metrics_t, fbc_checkpoint_load
   use fixed_budget_natural_eval
   implicit none (type, external)

   integer(int32), parameter :: max_record_bytes = 4096_int32
   character(len=1024) :: checkpoint_path, corpus_path, argument
   character(len=max_record_bytes) :: record
   integer(int32) :: tokens(max_record_bytes), prefix_limit, record_bytes, status
   integer(int32) :: i, byte
   integer(int64) :: path_size, initial_size, final_size, records, scored_records, predictions, prefix_predictions
   integer :: unit, io_status
   logical :: end_of_file, oversized, path_exists
   real(real64) :: loss_sum, prefix_loss_sum
   type(fbc_model_t) :: model
   type(fbc_optimizer_t) :: optimizer
   type(fbc_update_config_t) :: update_config
   type(fbc_metrics_t) :: counters
   type(fbc_record_metrics_t) :: record_metrics
   type(fbc_state_t) :: state
   type(fbc_workspace_t) :: workspace
   type(utf8_stream_reader_t) :: reader

   if (command_argument_count() < 2 .or. command_argument_count() > 3) &
      call fail('usage: vegai_natural_eval CHECKPOINT HELDOUT [PREFIX_BYTES]')
   if (file_storage_size /= 8 .or. storage_size('A') /= 8 .or. storage_size(0_int8) /= 8) &
      call fail('an ASCII-compatible octet platform is required')
   if (octet('A') /= 65_int32 .or. octet(achar(10)) /= 10_int32 .or. octet(achar(195)) /= 195_int32) &
      call fail('an ASCII-compatible octet platform is required')
   call cli_get_argument(1, checkpoint_path, status, path=.true.)
   if (status /= CLI_OK .or. len_trim(checkpoint_path) == 0) call fail('invalid checkpoint path')
   call cli_get_argument(2, corpus_path, status, path=.true.)
   if (status /= CLI_OK .or. len_trim(corpus_path) == 0) call fail('invalid held-out path')
   prefix_limit = 32_int32
   if (command_argument_count() == 3) then
      call cli_get_argument(3, argument, status)
      if (status /= CLI_OK) call fail('invalid prefix length')
      call fbc_parse_count(trim(argument), 1_int32, max_record_bytes, prefix_limit, status)
      if (status /= FBC_OK) call fail('prefix length must be in 1..4096 bytes')
   end if

   call fbc_checkpoint_load(trim(checkpoint_path), model, optimizer, update_config, counters, status)
   if (status /= FBT_OK) call fail('checkpoint load failed')
   if (model%config%vocabulary_size /= 256_int32) call fail('checkpoint must use the 256-byte vocabulary')
   call fbc_state_init(state, model, status)
   if (status /= FBC_OK) call fail('state allocation failed')
   call fbc_workspace_init(workspace, model, status)
   if (status /= FBC_OK) call fail('workspace allocation failed')
   ! On common runtimes a FIFO/device reports zero or unknown size. Reject it
   ! before OPEN so an evaluator does not wait indefinitely for a writer.
   ! This preflight is not a lock or a defense against path replacement.
   inquire(file=trim(corpus_path), exist=path_exists, size=path_size, iostat=io_status)
   if (io_status /= 0) call fail('cannot inspect held-out file')
   if (.not. path_exists) call fail('cannot open held-out file')
   if (path_size <= 0_int64) call fail('held-out file size is invalid')
   open(newunit=unit, file=trim(corpus_path), status='old', action='read', access='stream', &
        form='unformatted', iostat=io_status)
   if (io_status /= 0) call fail('cannot open held-out file')
   inquire(unit=unit, size=initial_size, iostat=io_status)
   if (io_status /= 0) call fail('held-out file size is invalid')
   if (initial_size /= path_size) call fail('held-out file changed before evaluation')
   call utf8_stream_reader_init(reader, unit, status)
   if (status /= UTF8_STREAM_OK) call fail('record reader allocation failed')

   records = 0_int64
   scored_records = 0_int64
   predictions = 0_int64
   prefix_predictions = 0_int64
   loss_sum = 0.0_real64
   prefix_loss_sum = 0.0_real64
   do
      call utf8_stream_read_record(unit, record, record_bytes, end_of_file, oversized, status, reader)
      if (status /= UTF8_STREAM_OK) call fail('held-out record read failed')
      if (end_of_file) exit
      if (oversized) call fail('held-out record exceeds 4096 bytes')
      if (.not. cli_valid_utf8(record(1_int32:record_bytes))) call fail('held-out record is not valid UTF-8')
      if (records == huge(records)) call fail('record count overflow')
      records = records + 1_int64
      do i = 1_int32, record_bytes
         byte = octet(record(i:i))
         if (byte == 0_int32) call fail('held-out record contains NUL')
         tokens(i) = byte
      end do
      if (record_bytes < 2_int32) cycle
      call fbc_evaluate_lf_records(model, state, workspace, tokens(1:record_bytes), prefix_limit, record_metrics, status)
      if (status /= FBT_OK) call fail('held-out model evaluation failed')
      if (scored_records == huge(scored_records) .or. &
          predictions > huge(predictions) - record_metrics%predictions .or. &
          prefix_predictions > huge(prefix_predictions) - record_metrics%prefix_predictions) &
         call fail('prediction count overflow')
      call add_loss(loss_sum, record_metrics%loss_sum)
      call add_loss(prefix_loss_sum, record_metrics%prefix_loss_sum)
      predictions = predictions + record_metrics%predictions
      prefix_predictions = prefix_predictions + record_metrics%prefix_predictions
      scored_records = scored_records + 1_int64
   end do
   inquire(unit=unit, size=final_size, iostat=io_status)
   if (io_status /= 0) call fail('cannot inspect held-out file after evaluation')
   if (final_size /= initial_size) call fail('held-out file changed during evaluation')
   close(unit, iostat=io_status)
   if (io_status /= 0) call fail('cannot close held-out file')
   if (predictions == 0_int64 .or. prefix_predictions == 0_int64) &
      call fail('held-out file has no within-record predictions')

   write(output_unit, '(A)') '{"schema":"vegai-cold-record-eval/v1",'
   write(output_unit, '(A,I0,A)') '"training_rule":', model%training_rule, ','
   write(output_unit, '(A,I0,A)') '"prefix_bytes":', prefix_limit, ','
   write(output_unit, '(A,I0,A)') '"records":', records, ','
   write(output_unit, '(A,I0,A)') '"scored_records":', scored_records, ','
   write(output_unit, '(A,I0,A)') '"predictions":', predictions, ','
   write(output_unit, '(A,I0,A)') '"prefix_predictions":', prefix_predictions, ','
   write(output_unit, '(A,ES24.16E3,A)') '"mean_nll":', loss_sum / real(predictions, real64), ','
   write(output_unit, '(A,ES24.16E3,A)') '"prefix_mean_nll":', prefix_loss_sum / real(prefix_predictions, real64), '}'
   call fbc_workspace_destroy(workspace)
   call fbc_state_destroy(state)
   call fbc_optimizer_destroy(optimizer)
   call fbc_model_destroy(model)

contains
   pure integer(int32) function octet(character_byte) result(value)
      character(len=1), intent(in) :: character_byte
      integer(int8) :: signed_byte
      signed_byte = transfer(character_byte, 0_int8)
      value = iand(int(signed_byte, int32), 255_int32)
   end function octet

   subroutine add_loss(total, value)
      real(real64), intent(inout) :: total
      real(real64), intent(in) :: value
      if (.not. fbc_real64_is_finite(total) .or. .not. fbc_real64_is_finite(value)) call fail('non-finite loss')
      if (total < 0.0_real64 .or. value < 0.0_real64) call fail('negative loss')
      if (total > 0.0_real64 .and. value > 0.0_real64) then
         if (total >= huge(total) - value) call fail('loss sum overflow')
      end if
      total = total + value
   end subroutine add_loss

   subroutine fail(message)
      character(len=*), intent(in) :: message
      write(error_unit, '(A)') 'vegai_natural_eval: ' // message
      stop 2, quiet=.true.
   end subroutine fail
end program natural_eval
