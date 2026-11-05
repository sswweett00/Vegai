! SPDX-License-Identifier: MIT
! Extract a bounded number of UD CoNLL-U `# text = ` records as raw UTF-8
! sentences, without a Python data-preparation dependency. This is an offline
! preparation tool; training does not call it or incur its input scan.
program ud_text_extract
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, output_unit, error_unit, file_storage_size
   use portable_cli, only: CLI_OK, cli_get_argument, cli_parse_integer32, cli_valid_utf8
   use utf8_record_stream, only: UTF8_STREAM_OK, utf8_stream_reader_t, &
                                 utf8_stream_reader_init, utf8_stream_read_record
   implicit none (type, external)

   integer, parameter :: max_path_bytes = 4096, max_record_bytes = 65536
   integer(int32), parameter :: max_sentences = 10000000_int32
   character(len=*), parameter :: prefix = '# text = '
   character(len=max_path_bytes) :: input_path, output_path
   character(len=64) :: limit_text
   character(len=256) :: failure
   character(len=max_record_bytes) :: record
   type(utf8_stream_reader_t) :: reader
   integer(int32) :: argument_status, limit, reader_status, record_bytes, text_end, extracted
   integer(int64) :: input_path_size, input_size, final_size, physical_lines, payload_bytes, output_bytes
   integer :: input_unit, output_file, io_status, close_status, delete_status
   logical :: exists, end_of_file, oversized, failed

   if (command_argument_count() /= 3) call fail_argument('usage: vegai_ud_text_extract INPUT OUTPUT COUNT')
   if (.not. octet_platform_supported()) call fail_argument('ASCII-compatible octet file and character storage required')
   call cli_get_argument(1, input_path, argument_status, path=.true.)
   if (argument_status /= CLI_OK) call fail_argument('invalid input path')
   call cli_get_argument(2, output_path, argument_status, path=.true.)
   if (argument_status /= CLI_OK) call fail_argument('invalid output path')
   call cli_get_argument(3, limit_text, argument_status)
   if (argument_status /= CLI_OK) call fail_argument('invalid sentence count')
   call cli_parse_integer32(trim(limit_text), limit, argument_status)
   if (argument_status /= CLI_OK) call fail_argument('invalid sentence count')
   if (limit < 1_int32 .or. limit > max_sentences) call fail_argument('sentence count must be in 1..10000000')

   ! Preflight before opening to avoid blocking on common unknown-size devices.
   ! This is not a lock or a defense against hostile path replacement.
   inquire(file=trim(input_path), exist=exists, size=input_path_size, iostat=io_status)
   if (io_status /= 0) call fail_argument('cannot inspect input')
   if (.not. exists) call fail_argument('input does not exist')
   if (input_path_size <= 0_int64) call fail_argument('empty or unknown-size input')
   open(newunit=input_unit, file=trim(input_path), status='old', action='read', &
        access='stream', form='unformatted', iostat=io_status)
   if (io_status /= 0) call fail_argument('cannot open input')
   inquire(unit=input_unit, size=input_size, iostat=io_status)
   if (io_status /= 0) then
      close(input_unit)
      call fail_argument('cannot determine opened input size')
   end if
   if (input_size /= input_path_size) then
      close(input_unit)
      call fail_argument('input size changed before reading')
   end if
   call utf8_stream_reader_init(reader, input_unit, reader_status)
   if (reader_status /= UTF8_STREAM_OK) then
      close(input_unit)
      call fail_argument('cannot allocate input buffer')
   end if
   ! Never intentionally replace an output or a source alias. Standard
   ! Fortran cannot guarantee hostile-directory race freedom on every OS.
   open(newunit=output_file, file=trim(output_path), status='new', action='write', &
        access='stream', form='unformatted', iostat=io_status)
   if (io_status /= 0) then
      close(input_unit)
      call fail_argument('output already exists or cannot be created')
   end if

   failed = .false.
   failure = ''
   extracted = 0_int32
   physical_lines = 0_int64
   payload_bytes = 0_int64
   output_bytes = 0_int64
   do
      call utf8_stream_read_record(input_unit, record, record_bytes, end_of_file, oversized, reader_status, reader)
      if (reader_status /= UTF8_STREAM_OK) then
         failed = .true.
         failure = 'input read error'
         exit
      end if
      if (end_of_file) exit
      if (oversized) then
         failed = .true.
         failure = 'input record exceeds 65536 bytes'
         exit
      end if
      ! With immutable input, physical records and their payload bytes cannot
      ! exceed the original size. This bounds some growing-input cases without
      ! claiming an atomic snapshot or content-level mutation detection.
      if (physical_lines >= input_size .or. int(record_bytes, int64) > input_size - payload_bytes) then
         failed = .true.
         failure = 'input grew while reading'
         exit
      end if
      physical_lines = physical_lines + 1_int64
      payload_bytes = payload_bytes + int(record_bytes, int64)
      text_end = record_bytes
      if (text_end > 0_int32) then
         ! A trailing CR is interpreted as the CRLF record terminator. Preserve
         ! every other payload byte, including spaces and tabs.
         if (record(text_end:text_end) == achar(13)) text_end = text_end - 1_int32
      end if
      if (text_end < len(prefix, kind=int32)) cycle
      if (record(1:len(prefix)) /= prefix) cycle
      if (text_end == len(prefix, kind=int32)) then
         failed = .true.
         failure = 'empty text record'
         exit
      end if
      if (index(record(len(prefix) + 1:text_end), achar(0)) /= 0 .or. &
          .not. cli_valid_utf8(record(len(prefix) + 1:text_end))) then
         failed = .true.
         failure = 'text record contains NUL or invalid UTF-8'
         exit
      end if
      write(output_file, iostat=io_status) record(len(prefix) + 1:text_end), achar(10)
      if (io_status /= 0) then
         failed = .true.
         failure = 'output write error'
         exit
      end if
      extracted = extracted + 1_int32
      output_bytes = output_bytes + int(text_end - len(prefix, kind=int32), int64) + 1_int64
      if (extracted == limit) exit
   end do
   if (.not. failed .and. extracted /= limit) then
      failed = .true.
      failure = 'fewer text records than requested'
   end if
   inquire(unit=input_unit, size=final_size, iostat=io_status)
   if (io_status /= 0) then
      failed = .true.
      failure = 'cannot recheck input size'
   else if (final_size /= input_size) then
      failed = .true.
      failure = 'input size changed while reading'
   end if
   close(input_unit, iostat=close_status)
   if (close_status /= 0) then
      failed = .true.
      failure = 'input close error'
   end if
   if (failed) then
      close(output_file, status='delete', iostat=delete_status)
      write(error_unit, '(A)') 'error: ' // trim(failure)
      if (delete_status /= 0) write(error_unit, '(A)') 'warning: partial output could not be deleted'
      stop 2, quiet=.true.
   end if
   close(output_file, iostat=close_status)
   if (close_status /= 0) then
      ! Once CLOSE fails, standard Fortran cannot promise a safe unlink or an
      ! atomic rollback. Report the path as potentially incomplete.
      write(error_unit, '(A)') 'error: output close failed; output may be incomplete'
      stop 2, quiet=.true.
   end if
   write(output_unit, '(A,I0,A,I0,A)') '{"schema":"vegai-ud-text/v1","records":', extracted, &
      ',"output_bytes":', output_bytes, '}'

contains

   subroutine fail_argument(message)
      character(len=*), intent(in) :: message
      write(error_unit, '(A)') 'error: ' // trim(message)
      stop 2, quiet=.true.
   end subroutine fail_argument

   logical function octet_platform_supported() result(supported)
      integer(int8) :: signed_byte
      character(len=1) :: character_byte
      integer :: index_byte

      supported = .false.
      if (file_storage_size /= 8 .or. storage_size('A') /= 8 .or. storage_size(0_int8) /= 8) return
      do index_byte = 0, 255
         if (index_byte < 128) then
            signed_byte = int(index_byte, int8)
         else
            signed_byte = int(index_byte - 256, int8)
         end if
         character_byte = transfer(signed_byte, 'a')
         if (iachar(character_byte) /= index_byte) return
      end do
      supported = .true.
   end function octet_platform_supported
end program ud_text_extract
