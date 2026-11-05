! SPDX-License-Identifier: MIT
!==============================================================================
!  utf8_record_stream.f90
!  Byte-preserving LF-record reader for native UTF-8 corpus adapters.
!
!  `character` storage is used as a raw octet buffer: this module does not
!  validate UTF-8 or normalize text. LF (byte 10) is framing only; every other
!  byte, including trailing spaces and CR, is retained verbatim. An overlong
!  record is consumed through its LF boundary and reported to the caller rather
!  than partially returned.
!
!  The optional reader owns a bounded read-ahead buffer. While it is in use the
!  caller must not independently read, reposition, close/reopen, or share its
!  unit with another reader. Reinitialize it after reopening or repositioning.
!  Omitting the reader retains the original unbuffered interface and exact
!  stream-position behavior for existing callers and unusual stream devices.
!==============================================================================
module utf8_record_stream
   use, intrinsic :: iso_fortran_env, only: int32, int64, iostat_end
   implicit none
   private

   integer(int32), parameter, public :: UTF8_STREAM_OK = 0_int32
   integer(int32), parameter, public :: UTF8_STREAM_ERR_IO = -760_int32
   integer(int32), parameter, public :: UTF8_STREAM_ERR_ALLOCATION = -761_int32
   integer, parameter :: BUFFER_BYTES = 65536

   type, public :: utf8_stream_reader_t
      private
      character(len=:), allocatable :: buffer
      integer :: unit = 0
      integer :: next = 1
      integer :: available = 0
      logical :: ready = .false.
      logical :: byte_fallback = .false.
   end type utf8_stream_reader_t

   public :: utf8_stream_reader_init, utf8_stream_read_record

contains

   subroutine utf8_stream_reader_init(reader, unit, status)
      type(utf8_stream_reader_t), intent(out) :: reader
      integer, intent(in) :: unit
      integer(int32), intent(out) :: status
      integer :: allocation_status

      status = UTF8_STREAM_OK
      allocate(character(len=BUFFER_BYTES) :: reader%buffer, stat=allocation_status)
      if (allocation_status /= 0) then
         status = UTF8_STREAM_ERR_ALLOCATION
         return
      end if
      reader%unit = unit
      reader%next = 1
      reader%available = 0
      reader%ready = .true.
      reader%byte_fallback = .false.
   end subroutine utf8_stream_reader_init

   subroutine utf8_stream_read_record(unit, record, record_bytes, end_of_file, oversized, status, reader)
      integer, intent(in) :: unit
      character(len=*), intent(out) :: record
      integer(int32), intent(out) :: record_bytes
      logical, intent(out) :: end_of_file, oversized
      integer(int32), intent(out) :: status
      type(utf8_stream_reader_t), intent(inout), optional :: reader
      character(len=1) :: byte
      integer :: io_status

      record = ''
      record_bytes = 0_int32
      end_of_file = .false.
      oversized = .false.
      status = UTF8_STREAM_OK
      if (present(reader)) then
         call read_buffered(unit, record, record_bytes, end_of_file, oversized, status, reader)
         return
      end if
      do
         read(unit, iostat=io_status) byte
         if (io_status == iostat_end) then
            if (record_bytes == 0_int32 .and. .not. oversized) end_of_file = .true.
            return
         end if
         if (io_status /= 0) then
            status = UTF8_STREAM_ERR_IO
            return
         end if
         if (byte == achar(10)) return
         if (record_bytes >= len(record, kind=int32)) then
            oversized = .true.
            cycle
         end if
         record_bytes = record_bytes + 1_int32
         record(record_bytes:record_bytes) = byte
      end do
   end subroutine utf8_stream_read_record

   subroutine read_buffered(unit, record, record_bytes, end_of_file, oversized, status, reader)
      integer, intent(in) :: unit
      character(len=*), intent(inout) :: record
      integer(int32), intent(inout) :: record_bytes
      logical, intent(inout) :: end_of_file, oversized
      integer(int32), intent(inout) :: status
      type(utf8_stream_reader_t), intent(inout) :: reader
      integer :: delimiter, segment_bytes, retained_bytes
      logical :: exhausted

      if (.not. reader%ready .or. reader%unit /= unit .or. .not. allocated(reader%buffer)) then
         status = UTF8_STREAM_ERR_IO
         return
      end if
      do
         if (reader%next > reader%available) then
            call refill(reader, exhausted, status)
            if (status /= UTF8_STREAM_OK) return
            if (exhausted) then
               if (record_bytes == 0_int32 .and. .not. oversized) end_of_file = .true.
               return
            end if
         end if
         delimiter = index(reader%buffer(reader%next:reader%available), achar(10))
         if (delimiter == 0) then
            segment_bytes = reader%available - reader%next + 1
         else
            segment_bytes = delimiter - 1
         end if
         retained_bytes = min(segment_bytes, max(0, len(record) - int(record_bytes)))
         if (retained_bytes > 0) then
            record(record_bytes + 1:record_bytes + retained_bytes) = &
               reader%buffer(reader%next:reader%next + retained_bytes - 1)
            record_bytes = record_bytes + int(retained_bytes, int32)
         end if
         if (retained_bytes < segment_bytes) oversized = .true.
         reader%next = reader%next + segment_bytes
         if (delimiter /= 0) then
            reader%next = reader%next + 1
            return
         end if
      end do
   end subroutine read_buffered

   subroutine refill(reader, exhausted, status)
      type(utf8_stream_reader_t), intent(inout) :: reader
      logical, intent(out) :: exhausted
      integer(int32), intent(out) :: status
      integer(int64) :: file_bytes, position
      integer :: count, io_status

      exhausted = .false.
      status = UTF8_STREAM_OK
      count = 1
      position = -1_int64
      file_bytes = -1_int64
      if (.not. reader%byte_fallback) then
         ! Avoid a partial unformatted read: its transferred byte count is not
         ! specified by Fortran. For unknown-size/empty streams read one byte,
         ! retaining the old behavior for FIFOs and other special files.
         inquire(unit=reader%unit, pos=position, size=file_bytes, iostat=io_status)
         if (io_status == 0 .and. position > 0_int64 .and. file_bytes >= position) then
            count = int(min(int(BUFFER_BYTES, int64), file_bytes - position + 1_int64))
         else
            ! Repeated position inquiries can disturb non-seekable units on
            ! some runtimes. Once size/position is unknown or exhausted, use
            ! the original one-byte path without further INQUIRE operations.
            reader%byte_fallback = .true.
         end if
      end if
      read(reader%unit, iostat=io_status) reader%buffer(1:count)
      if (io_status == iostat_end .and. count > 1) then
         ! A source can shrink or report an unreliable size after INQUIRE.
         ! Retry at the original position one byte at a time; never return a
         ! partially transferred buffer as a complete or silently lost record.
         reader%byte_fallback = .true.
         count = 1
         read(reader%unit, pos=position, iostat=io_status) reader%buffer(1:1)
      end if
      if (io_status == iostat_end) then
         exhausted = .true.
      else if (io_status /= 0) then
         status = UTF8_STREAM_ERR_IO
      else
         reader%next = 1
         reader%available = count
      end if
   end subroutine refill

end module utf8_record_stream
