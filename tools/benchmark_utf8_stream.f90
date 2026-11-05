! SPDX-License-Identifier: MIT
! Compare only native record-reading overhead; not an end-to-end training benchmark.
program benchmark_utf8_stream
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64
   use utf8_record_stream
   implicit none
   type(utf8_stream_reader_t) :: reader
   character(len=4096) :: path
   character(len=16) :: mode
   character(len=8192) :: record
   integer :: unit, io_status
   integer(int32) :: record_bytes, status
   integer(int64) :: records, retained_bytes, oversized_records, checksum, started, finished, rate
   logical :: end_of_file, oversized

   call get_command_argument(1, path)
   call get_command_argument(2, mode)
   if (len_trim(path) == 0 .or. (mode /= 'legacy' .and. mode /= 'buffered')) then
      write(*, '(A)') 'Usage: benchmark_utf8_stream CORPUS legacy|buffered'
      error stop 2
   end if
   open(newunit=unit, file=trim(path), status='old', action='read', access='stream', &
        form='unformatted', iostat=io_status)
   if (io_status /= 0) error stop 'cannot open corpus'
   if (mode == 'buffered') then
      call utf8_stream_reader_init(reader, unit, status)
      if (status /= UTF8_STREAM_OK) error stop 'cannot allocate stream reader'
   end if
   records = 0_int64
   retained_bytes = 0_int64
   oversized_records = 0_int64
   checksum = 0_int64
   call system_clock(started, rate)
   if (rate <= 0_int64) error stop 'system clock unavailable'
   do
      if (mode == 'buffered') then
         call utf8_stream_read_record(unit, record, record_bytes, end_of_file, oversized, status, reader)
      else
         call utf8_stream_read_record(unit, record, record_bytes, end_of_file, oversized, status)
      end if
      if (status /= UTF8_STREAM_OK) error stop 'corpus read failed'
      if (end_of_file) exit
      records = records + 1_int64
      retained_bytes = retained_bytes + int(record_bytes, int64)
      if (oversized) oversized_records = oversized_records + 1_int64
      checksum = ishftc(checksum, 1)
      if (record_bytes > 0_int32) then
         checksum = ieor(checksum, int(record_bytes + iachar(record(1:1)) + &
                                      iachar(record(record_bytes:record_bytes)), int64))
      end if
   end do
   call system_clock(finished)
   close(unit)
   write(*, '(A,A,A,I0,A,I0,A,I0,A,I0,A,F0.6)') &
      'mode=', trim(mode), ' records=', records, ' retained_bytes=', retained_bytes, &
      ' oversized_records=', oversized_records, ' checksum=', checksum, &
      ' wall_seconds=', real(finished - started, real64) / real(rate, real64)
end program benchmark_utf8_stream
