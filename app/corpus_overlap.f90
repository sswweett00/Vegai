! SPDX-License-Identifier: MIT
! Optional, bounded exact-record leakage check. It is deliberately separate
! from training: running this program reads both inputs, but training itself
! performs no additional scan, allocation, checkpoint copy, or update.
program corpus_overlap
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, error_unit, output_unit, file_storage_size
   use fortran_integrity, only: sha256_platform_supported
   use portable_cli, only: CLI_OK, cli_get_argument, cli_valid_utf8
   use utf8_record_stream, only: UTF8_STREAM_OK, utf8_stream_reader_t, &
                                 utf8_stream_reader_init, utf8_stream_read_record
   implicit none (type, external)

   integer, parameter :: max_path_bytes = 4096, max_record_bytes = 4096
   integer(int32), parameter :: max_unique_records = 100000_int32, bucket_count = 131071_int32
   integer(int32), parameter :: max_chain = 1024_int32
   integer(int64), parameter :: max_held_bytes = 67108864_int64

   type :: held_record_t
      character(len=:), allocatable :: text
      integer(int64) :: hash = 0_int64
      integer(int64) :: first_line = 0_int64
      integer(int32) :: next = 0_int32
      logical :: matched = .false.
   end type held_record_t

   type :: overlap_table_t
      type(held_record_t), allocatable :: records(:)
      integer(int32), allocatable :: buckets(:)
      integer(int32) :: count = 0_int32
      integer(int64) :: bytes = 0_int64
      integer(int64) :: held_records = 0_int64
      integer(int64) :: train_records = 0_int64
      integer(int64) :: overlapping_train_records = 0_int64
      integer(int64) :: overlapping_held_unique = 0_int64
      integer(int64) :: first_train_line = 0_int64
      integer(int64) :: first_held_line = 0_int64
   end type overlap_table_t

   type(overlap_table_t) :: table
   character(len=max_path_bytes) :: train_path, held_path
   character(len=256) :: message
   integer(int32) :: argument_status
   integer :: allocation_status, status

   if (command_argument_count() /= 2) then
      write(error_unit, '(A)') 'usage: vegai_corpus_overlap TRAIN HELDOUT'
      stop 2, quiet=.true.
   end if
   if (.not. octet_platform_supported()) then
      write(error_unit, '(A)') 'error: ASCII-compatible octet file and character storage required'
      stop 2, quiet=.true.
   end if
   call cli_get_argument(1, train_path, argument_status, path=.true.)
   if (argument_status /= CLI_OK .or. len_trim(train_path) == 0) then
      write(error_unit, '(A)') 'error: invalid train path'
      stop 2, quiet=.true.
   end if
   call cli_get_argument(2, held_path, argument_status, path=.true.)
   if (argument_status /= CLI_OK .or. len_trim(held_path) == 0) then
      write(error_unit, '(A)') 'error: invalid held-out path'
      stop 2, quiet=.true.
   end if
   allocate(table%records(max_unique_records), table%buckets(bucket_count), stat=allocation_status)
   if (allocation_status /= 0) then
      write(error_unit, '(A)') 'error: cannot allocate overlap index'
      stop 2, quiet=.true.
   end if
   table%buckets = 0_int32
   call scan_file(trim(held_path), .true., table, status, message)
   if (status /= 0) then
      write(error_unit, '(A)') 'error: held-out file: ' // trim(message)
      stop 2, quiet=.true.
   end if
   call scan_file(trim(train_path), .false., table, status, message)
   if (status /= 0) then
      write(error_unit, '(A)') 'error: train file: ' // trim(message)
      stop 2, quiet=.true.
   end if

   write(output_unit, '(A)') '{"schema":"vegai-exact-record-overlap/v1",'
   write(output_unit, '(A,I0,A)') '"train_records":', table%train_records, ','
   write(output_unit, '(A,I0,A)') '"heldout_records":', table%held_records, ','
   write(output_unit, '(A,I0,A)') '"heldout_unique_records":', table%count, ','
   write(output_unit, '(A,I0,A)') '"overlapping_train_records":', table%overlapping_train_records, ','
   write(output_unit, '(A,I0,A)') '"overlapping_heldout_unique_records":', table%overlapping_held_unique, ','
   write(output_unit, '(A,I0,A)') '"first_train_line":', table%first_train_line, ','
   write(output_unit, '(A,I0,A)') '"first_heldout_line":', table%first_held_line, ','
   if (table%overlapping_train_records > 0_int64) then
      write(output_unit, '(A)') '"overlap":true}'
      stop 1, quiet=.true.
   end if
   write(output_unit, '(A)') '"overlap":false}'

contains

   logical function octet_platform_supported() result(supported)
      integer(int8) :: signed_byte
      character(len=1) :: c
      integer :: i

      supported = .false.
      if (.not. sha256_platform_supported()) return
      if (file_storage_size /= 8 .or. storage_size('A') /= 8) return
      do i = 0, 255
         if (i < 128) then
            signed_byte = int(i, int8)
         else
            signed_byte = int(i - 256, int8)
         end if
         c = transfer(signed_byte, 'a')
         if (iachar(c) /= i) return
      end do
      supported = .true.
   end function octet_platform_supported

   subroutine scan_file(path, is_held, index_table, result, error_message)
      character(len=*), intent(in) :: path
      logical, intent(in) :: is_held
      type(overlap_table_t), intent(inout) :: index_table
      integer, intent(out) :: result
      character(len=*), intent(out) :: error_message
      type(utf8_stream_reader_t) :: reader
      character(len=max_record_bytes) :: record
      integer(int64) :: path_size, initial_size, final_size, physical_line, payload_bytes, count_before
      integer(int32) :: record_bytes, reader_status
      integer :: unit, io_status, close_status
      logical :: end_of_file, oversized, exists

      result = 1
      error_message = 'cannot inspect file'
      inquire(file=path, size=path_size, exist=exists, iostat=io_status)
      if (io_status /= 0) return
      if (.not. exists) then
         error_message = 'cannot open file'
         return
      end if
      ! Inspect before OPEN: a normal POSIX FIFO with no writer would otherwise
      ! block here. This is not a guarantee against hostile path replacement.
      if (path_size <= 0_int64) then
         error_message = 'empty or unknown-size input'
         return
      end if
      error_message = 'cannot open file'
      open(newunit=unit, file=path, status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) return
      error_message = 'cannot determine file size'
      inquire(unit=unit, size=initial_size, iostat=io_status)
      if (io_status /= 0) goto 900
      if (initial_size /= path_size) then
         error_message = 'file size changed before reading'
         goto 900
      end if
      call utf8_stream_reader_init(reader, unit, reader_status)
      if (reader_status /= UTF8_STREAM_OK) then
         error_message = 'cannot allocate input buffer'
         goto 900
      end if
      physical_line = 0_int64
      payload_bytes = 0_int64
      if (is_held) then
         count_before = index_table%held_records
      else
         count_before = index_table%train_records
      end if
      do
         call utf8_stream_read_record(unit, record, record_bytes, end_of_file, oversized, reader_status, reader)
         if (reader_status /= UTF8_STREAM_OK) then
            error_message = 'read error'
            goto 900
         end if
         if (end_of_file) exit
         if (oversized) then
            error_message = 'record exceeds 4096 bytes'
            goto 900
         end if
         ! For an immutable file, each physical record needs at least one
         ! storage byte; payload cannot exceed its original size. These checks
         ! also bound some growing-input cases, though they are not a lock.
         if (physical_line >= initial_size) then
            error_message = 'input grew during reading'
            goto 900
         end if
         physical_line = physical_line + 1_int64
         if (int(record_bytes, int64) > initial_size) then
            error_message = 'input grew during reading'
            goto 900
         end if
         if (payload_bytes > initial_size - int(record_bytes, int64)) then
            error_message = 'input grew during reading'
            goto 900
         end if
         payload_bytes = payload_bytes + int(record_bytes, int64)
         if (record_bytes > 0_int32) then
            ! Strip exactly one terminal CR, so LF and CRLF records compare
            ! identically. Spaces and every other byte remain significant.
            if (record(record_bytes:record_bytes) == achar(13)) record_bytes = record_bytes - 1_int32
         end if
         if (record_bytes == 0_int32) cycle
         if (index(record(1:record_bytes), achar(0)) /= 0) then
            error_message = 'NUL byte in text record'
            goto 900
         end if
         if (.not. cli_valid_utf8(record(1:record_bytes))) then
            error_message = 'invalid UTF-8 text record'
            goto 900
         end if
         call process_record(index_table, record(1:record_bytes), physical_line, is_held, result, error_message)
         if (result /= 0) goto 900
      end do
      result = 1
      error_message = 'cannot recheck file size'
      inquire(unit=unit, size=final_size, iostat=io_status)
      if (io_status /= 0) goto 900
      if (final_size /= initial_size) then
         error_message = 'file size changed during reading'
         goto 900
      end if
      if (is_held) then
         if (index_table%held_records == count_before) then
            error_message = 'no nonempty records'
            goto 900
         end if
      else
         if (index_table%train_records == count_before) then
            error_message = 'no nonempty records'
            goto 900
         end if
      end if
      error_message = 'cannot close file'
      close(unit, iostat=close_status)
      if (close_status /= 0) return
      result = 0
      error_message = ''
      return
900   close(unit, iostat=close_status)
      result = 1
   end subroutine scan_file

   subroutine process_record(index_table, text, line_number, is_held, result, error_message)
      type(overlap_table_t), intent(inout) :: index_table
      character(len=*), intent(in) :: text
      integer(int64), intent(in) :: line_number
      logical, intent(in) :: is_held
      integer, intent(out) :: result
      character(len=*), intent(out) :: error_message
      integer(int64) :: digest
      integer(int32) :: bucket, cursor, chain_length
      integer :: allocation_status

      result = 1
      digest = record_hash(text)
      bucket = hash_bucket(digest)
      cursor = index_table%buckets(bucket)
      chain_length = 0_int32
      do while (cursor /= 0_int32)
         chain_length = chain_length + 1_int32
         if (chain_length > max_chain) then
            error_message = 'hash bucket limit exceeded'
            return
         end if
         if (index_table%records(cursor)%hash == digest) then
            if (len(index_table%records(cursor)%text) == len(text)) then
               if (index_table%records(cursor)%text == text) exit
            end if
         end if
         cursor = index_table%records(cursor)%next
      end do
      if (is_held) then
         index_table%held_records = index_table%held_records + 1_int64
         if (cursor == 0_int32) then
            if (chain_length >= max_chain) then
               error_message = 'hash bucket limit exceeded'
               return
            end if
            if (index_table%count >= max_unique_records) then
               error_message = 'held-out unique record limit exceeded'
               return
            end if
            if (int(len(text), int64) > max_held_bytes - index_table%bytes) then
               error_message = 'held-out byte limit exceeded'
               return
            end if
            cursor = index_table%count + 1_int32
            allocate(character(len=len(text)) :: index_table%records(cursor)%text, stat=allocation_status)
            if (allocation_status /= 0) then
               error_message = 'cannot allocate held-out record'
               return
            end if
            index_table%records(cursor)%text = text
            index_table%records(cursor)%hash = digest
            index_table%records(cursor)%first_line = line_number
            index_table%records(cursor)%next = index_table%buckets(bucket)
            index_table%buckets(bucket) = cursor
            index_table%count = cursor
            index_table%bytes = index_table%bytes + int(len(text), int64)
         end if
      else
         index_table%train_records = index_table%train_records + 1_int64
         if (cursor /= 0_int32) then
            index_table%overlapping_train_records = index_table%overlapping_train_records + 1_int64
            if (.not. index_table%records(cursor)%matched) then
               index_table%records(cursor)%matched = .true.
               index_table%overlapping_held_unique = index_table%overlapping_held_unique + 1_int64
            end if
            if (index_table%first_train_line == 0_int64) then
               index_table%first_train_line = line_number
               index_table%first_held_line = index_table%records(cursor)%first_line
            end if
         end if
      end if
      result = 0
      error_message = ''
   end subroutine process_record

   pure integer(int32) function hash_bucket(value) result(bucket)
      integer(int64), intent(in) :: value
      integer(int64), parameter :: word_mask = int(z'00000000FFFFFFFF', int64)
      integer(int64), parameter :: base_mod = mod(4294967296_int64, int(bucket_count, int64))
      integer(int64) :: upper, lower, remainder

      ! Treat the bit pattern as two unsigned words. MODULO on the most
      ! negative int64 can overflow in some checked compiler runtimes; every
      ! operand and intermediate here is nonnegative and below 2**34.
      lower = iand(value, word_mask)
      upper = iand(shiftr(value, 32), word_mask)
      remainder = mod(upper, int(bucket_count, int64))
      remainder = mod(remainder * base_mod + lower, int(bucket_count, int64))
      bucket = int(remainder, int32) + 1_int32
   end function hash_bucket

   pure integer(int64) function record_hash(text) result(value)
      character(len=*), intent(in) :: text
      integer :: i
      ! Rotations and XOR are defined for all bit patterns and do not rely on
      ! signed integer overflow. A hash match is always followed by an exact
      ! length-and-byte comparison, so collisions cannot count as overlap.
      value = int(z'6A09E667F3BCC909', int64)
      do i = 1, len(text)
         value = ieor(ishftc(value, 7), int(iachar(text(i:i)), int64))
      end do
   end function record_hash
end program corpus_overlap
