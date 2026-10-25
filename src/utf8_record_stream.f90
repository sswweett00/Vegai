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
!==============================================================================
module utf8_record_stream
   use, intrinsic :: iso_fortran_env, only: int32, iostat_end
   implicit none
   private

   integer(int32), parameter, public :: UTF8_STREAM_OK = 0_int32
   integer(int32), parameter, public :: UTF8_STREAM_ERR_IO = -760_int32

   public :: utf8_stream_read_record

contains

   subroutine utf8_stream_read_record(unit, record, record_bytes, end_of_file, oversized, status)
      integer, intent(in) :: unit
      character(len=*), intent(out) :: record
      integer(int32), intent(out) :: record_bytes
      logical, intent(out) :: end_of_file, oversized
      integer(int32), intent(out) :: status
      character(len=1) :: byte
      integer :: io_status

      record = ''
      record_bytes = 0_int32
      end_of_file = .false.
      oversized = .false.
      status = UTF8_STREAM_OK
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

end module utf8_record_stream
