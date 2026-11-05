! SPDX-License-Identifier: MIT
! Standalone byte-level contract for the buffered and legacy stream readers.
program test_utf8_record_stream
   use, intrinsic :: iso_fortran_env, only: int32
   use utf8_record_stream
   implicit none
   integer :: failures
   character(len=32) :: mode

   failures = 0
   call get_command_argument(1, mode)
   if (mode == '--fifo') then
      call fifo_contract()
      if (failures /= 0) error stop 1
      write(*, '(A)') 'PASS: non-seekable UTF-8 record stream'
      stop
   end if
   call compare_scenario(.true.)
   call compare_scenario(.false.)
   call compare_empty()
   if (failures /= 0) then
      write(*, '(A,I0)') 'UTF-8 stream contract failures: ', failures
      error stop 1
   end if
   write(*, '(A)') 'PASS: buffered UTF-8 record stream contracts'

contains

   subroutine check(condition, message)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: message
      if (.not. condition) then
         failures = failures + 1
         write(*, '(A,A)') 'FAIL: ', message
      end if
   end subroutine check

   function payload(length, seed) result(bytes)
      integer, intent(in) :: length, seed
      character(len=:), allocatable :: bytes
      integer :: i, code
      allocate(character(len=length) :: bytes)
      do i = 1, length
         code = modulo(i * 53 + seed * 19, 256)
         if (code == 10) code = 0
         bytes(i:i) = char(code)
      end do
   end function payload

   subroutine make_fixture(unit, trailing_lf, lengths)
      integer, intent(out) :: unit
      logical, intent(in) :: trailing_lf
      integer, intent(in) :: lengths(:)
      character(len=:), allocatable :: bytes
      integer :: i, io_status
      open(newunit=unit, status='scratch', action='readwrite', access='stream', form='unformatted', iostat=io_status)
      if (io_status /= 0) error stop 'scratch stream open failed'
      do i = 1, size(lengths)
         bytes = payload(lengths(i), i)
         write(unit, iostat=io_status) bytes
         if (io_status /= 0) error stop 'scratch stream write failed'
         if (i < size(lengths) .or. trailing_lf) then
            write(unit, iostat=io_status) achar(10)
            if (io_status /= 0) error stop 'scratch delimiter write failed'
         end if
      end do
      rewind(unit, iostat=io_status)
      if (io_status /= 0) error stop 'scratch stream rewind failed'
   end subroutine make_fixture

   subroutine compare_record(reference_unit, buffered_unit, reader, length, capacity, seed)
      integer, intent(in) :: reference_unit, buffered_unit, length, capacity, seed
      type(utf8_stream_reader_t), intent(inout) :: reader
      character(len=:), allocatable :: expected, reference, buffered
      integer(int32) :: reference_bytes, buffered_bytes, reference_status, buffered_status
      logical :: reference_eof, buffered_eof, reference_oversized, buffered_oversized
      integer :: retained

      allocate(character(len=capacity) :: reference, buffered)
      expected = payload(length, seed)
      call utf8_stream_read_record(reference_unit, reference, reference_bytes, &
                                   reference_eof, reference_oversized, reference_status)
      call utf8_stream_read_record(buffered_unit, buffered, buffered_bytes, &
                                   buffered_eof, buffered_oversized, buffered_status, reader)
      call check(reference_status == UTF8_STREAM_OK .and. buffered_status == UTF8_STREAM_OK, 'record I/O status')
      call check(.not. reference_eof .and. .not. buffered_eof, 'record is not EOF')
      call check(reference_bytes == buffered_bytes .and. reference == buffered, 'buffered record equals legacy bytes')
      call check(reference_oversized .eqv. buffered_oversized, 'buffered oversized flag equals legacy flag')
      retained = min(length, capacity)
      call check(reference_bytes == int(retained, int32), 'record length honors output capacity')
      call check(reference_oversized .eqv. (length > capacity), 'only records beyond capacity are oversized')
      if (retained > 0) call check(reference(:retained) == expected(:retained), 'arbitrary source bytes retained')
   end subroutine compare_record

   subroutine compare_eof(reference_unit, buffered_unit, reader)
      integer, intent(in) :: reference_unit, buffered_unit
      type(utf8_stream_reader_t), intent(inout) :: reader
      character(len=8) :: reference, buffered
      integer(int32) :: reference_bytes, buffered_bytes, reference_status, buffered_status
      logical :: reference_eof, buffered_eof, reference_oversized, buffered_oversized
      call utf8_stream_read_record(reference_unit, reference, reference_bytes, &
                                   reference_eof, reference_oversized, reference_status)
      call utf8_stream_read_record(buffered_unit, buffered, buffered_bytes, &
                                   buffered_eof, buffered_oversized, buffered_status, reader)
      call check(reference_status == UTF8_STREAM_OK .and. buffered_status == UTF8_STREAM_OK, 'EOF I/O status')
      call check(reference_eof .and. buffered_eof, 'EOF follows the final complete or partial record')
      call check(reference_bytes == 0_int32 .and. buffered_bytes == 0_int32, 'EOF has zero bytes')
      call check(.not. reference_oversized .and. .not. buffered_oversized, 'EOF is not oversized')
   end subroutine compare_eof

   subroutine compare_scenario(trailing_lf)
      logical, intent(in) :: trailing_lf
      ! Place LF exactly at the end of a 64 KiB buffer, immediately after a
      ! full buffer, and beyond multiple buffers. Include empty, NUL, CR,
      ! high-octet and oversized records followed by ordinary records.
      integer, parameter :: lengths(*) = [65535, 65536, 65537, 0, 1, 3, 8, 8192, 131079, 0, 4, 70000, 3, 9]
      integer, parameter :: capacities(*) = [65535, 65536, 65536, 0, 0, 3, 7, 8192, 0, 0, 3, 10, 10, 140000]
      type(utf8_stream_reader_t) :: first_reader, second_reader, uninitialized
      integer :: reference_a, buffered_a, reference_b, buffered_b, i, io_status
      integer(int32) :: status, record_bytes
      logical :: end_of_file, oversized
      character(len=1) :: record

      call make_fixture(reference_a, trailing_lf, lengths)
      call make_fixture(buffered_a, trailing_lf, lengths)
      call make_fixture(reference_b, trailing_lf, lengths)
      call make_fixture(buffered_b, trailing_lf, lengths)
      call utf8_stream_read_record(buffered_a, record, record_bytes, end_of_file, oversized, status, uninitialized)
      call check(status == UTF8_STREAM_ERR_IO, 'uninitialized reader is rejected')
      call utf8_stream_reader_init(first_reader, buffered_a, status)
      call check(status == UTF8_STREAM_OK, 'first reader initializes')
      call utf8_stream_reader_init(second_reader, buffered_b, status)
      call check(status == UTF8_STREAM_OK, 'second reader initializes')
      call utf8_stream_read_record(buffered_b, record, record_bytes, end_of_file, oversized, status, first_reader)
      call check(status == UTF8_STREAM_ERR_IO, 'reader cannot accidentally consume a different unit')
      do i = 1, size(lengths)
         call compare_record(reference_a, buffered_a, first_reader, lengths(i), capacities(i), i)
         call compare_record(reference_b, buffered_b, second_reader, lengths(i), capacities(i), i)
      end do
      call compare_eof(reference_a, buffered_a, first_reader)
      call compare_eof(reference_b, buffered_b, second_reader)

      rewind(reference_a, iostat=io_status)
      call check(io_status == 0, 'legacy stream rewinds')
      rewind(buffered_a, iostat=io_status)
      call check(io_status == 0, 'buffered stream rewinds')
      call utf8_stream_reader_init(first_reader, buffered_a, status)
      call check(status == UTF8_STREAM_OK, 'reinitialization discards old read-ahead')
      call compare_record(reference_a, buffered_a, first_reader, lengths(1), capacities(1), 1)
      close(reference_a)
      close(buffered_a)
      close(reference_b)
      close(buffered_b)
   end subroutine compare_scenario

   ! Called by the POSIX native Make verifier with a real FIFO. Empty or
   ! unknown-size streams must fall back without further position inquiries.
   subroutine fifo_contract()
      type(utf8_stream_reader_t) :: reader
      character(len=4096) :: path
      character(len=3) :: record
      integer :: unit, io_status
      integer(int32) :: status, record_bytes
      logical :: end_of_file, oversized
      call get_command_argument(2, path)
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      call check(io_status == 0, 'FIFO opens')
      if (io_status /= 0) return
      call utf8_stream_reader_init(reader, unit, status)
      call check(status == UTF8_STREAM_OK, 'FIFO reader initializes')
      if (status /= UTF8_STREAM_OK) then
         close(unit)
         return
      end if
      call utf8_stream_read_record(unit, record, record_bytes, end_of_file, oversized, status, reader)
      call check(status == UTF8_STREAM_OK .and. .not. end_of_file .and. oversized .and. &
                 record_bytes == 3_int32 .and. record == 'abc', 'FIFO oversized record')
      call utf8_stream_read_record(unit, record, record_bytes, end_of_file, oversized, status, reader)
      call check(status == UTF8_STREAM_OK .and. .not. end_of_file .and. .not. oversized .and. &
                 record_bytes == 2_int32 .and. record == 'xy ', 'FIFO final unterminated record')
      call utf8_stream_read_record(unit, record, record_bytes, end_of_file, oversized, status, reader)
      call check(status == UTF8_STREAM_OK .and. end_of_file, 'FIFO EOF')
      close(unit)
   end subroutine fifo_contract

   subroutine compare_empty()
      type(utf8_stream_reader_t) :: reader
      integer :: reference_unit, buffered_unit, io_status
      integer(int32) :: status
      open(newunit=reference_unit, status='scratch', action='readwrite', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) error stop 'empty reference stream open failed'
      open(newunit=buffered_unit, status='scratch', action='readwrite', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) error stop 'empty buffered stream open failed'
      call utf8_stream_reader_init(reader, buffered_unit, status)
      call check(status == UTF8_STREAM_OK, 'empty reader initializes')
      call compare_eof(reference_unit, buffered_unit, reader)
      close(reference_unit)
      close(buffered_unit)
   end subroutine compare_empty
end program test_utf8_record_stream
