! SPDX-License-Identifier: MIT
! Single-pass, bounded-memory SHA-256 and ASCII HDC corpus preflight.
! This is a control-plane operation; it is never inserted into training loops.
module corpus_preflight
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64
   use fortran_integrity, only: FI_OK, sha256_state_t, sha256_init, sha256_update_bytes, sha256_final_hex, &
                                sha256_platform_supported
   use text_tokenizer, only: MAX_LINE_CHARS, TOKEN_LEN, MAX_TOKENS_PER_LINE
   implicit none (type, external)
   private

   integer(int32), parameter, public :: CP_OK = 0_int32
   integer(int32), parameter, public :: CP_ERR_ARGUMENT = -970_int32
   integer(int32), parameter, public :: CP_ERR_IO = -971_int32
   integer(int32), parameter, public :: CP_ERR_SIZE = -972_int32
   integer(int32), parameter, public :: CP_ERR_CHANGED = -973_int32
   integer(int32), parameter, public :: CP_ERR_PLATFORM = -974_int32

   type, public :: corpus_preflight_limits_t
      integer(int64) :: max_line_bytes = int(MAX_LINE_CHARS, int64)
      integer(int64) :: token_length_bytes = int(TOKEN_LEN, int64)
      integer(int64) :: max_tokens_per_line = int(MAX_TOKENS_PER_LINE, int64)
   end type corpus_preflight_limits_t

   type, public :: corpus_preflight_stats_t
      integer(int64) :: bytes = 0_int64
      integer(int64) :: records = 0_int64
      integer(int64) :: overlong_records = 0_int64
      integer(int64) :: records_with_tokenizer_truncation = 0_int64
      integer(int64) :: accepted_tokens = 0_int64
      integer(int64) :: dropped_tokens = 0_int64
      integer(int64) :: maximum_record_bytes = 0_int64
      integer(int64) :: non_ascii_bytes = 0_int64
      integer(int64) :: crlf_records = 0_int64
      character(len=64) :: sha256 = ''
   end type corpus_preflight_stats_t

   type :: record_state_t
      integer(int64) :: byte_count = 0_int64
      integer(int64) :: accepted = 0_int64
      integer(int64) :: dropped = 0_int64
      integer(int64) :: non_ascii = 0_int64
      integer(int64) :: token_length = 0_int64
      integer(int32) :: last_byte = -1_int32
      logical :: token_too_long = .false.
   end type record_state_t

   public :: corpus_preflight_scan, corpus_preflight_strict_ok

contains

   subroutine corpus_preflight_scan(path, stats, status, limits, native_strict_ok)
      character(len=*), intent(in) :: path
      type(corpus_preflight_stats_t), intent(out) :: stats
      integer(int32), intent(out) :: status
      type(corpus_preflight_limits_t), intent(in), optional :: limits
      logical, intent(out), optional :: native_strict_ok
      type(corpus_preflight_limits_t) :: effective, native
      type(corpus_preflight_stats_t) :: native_stats
      type(sha256_state_t) :: hash
      type(record_state_t) :: record, native_record
      integer(int8) :: chunk(65536)
      integer(int64) :: file_size, opened_size, ending_size, remaining
      integer(int32) :: hash_status, byte_value
      integer :: unit, io_status, close_status, index, read_count
      logical :: exists, custom_limits

      stats = corpus_preflight_stats_t()
      if (present(native_strict_ok)) native_strict_ok = .false.
      status = CP_ERR_PLATFORM
      if (.not. sha256_platform_supported()) return
      status = CP_ERR_ARGUMENT
      if (len_trim(path) == 0) return
      native = corpus_preflight_limits_t()
      effective = native
      if (present(limits)) effective = limits
      if (effective%max_line_bytes <= 0_int64 .or. effective%token_length_bytes <= 0_int64 .or. &
          effective%max_tokens_per_line <= 0_int64) return
      custom_limits = effective%max_line_bytes /= native%max_line_bytes .or. &
                      effective%token_length_bytes /= native%token_length_bytes .or. &
                      effective%max_tokens_per_line /= native%max_tokens_per_line
      ! Standard Fortran cannot portably identify every special file or detect
      ! same-length mutations; inputs must be operator-provided immutable files.
      exists = .false.
      file_size = -1_int64
      inquire(file=trim(path), exist=exists, size=file_size, iostat=io_status)
      if (io_status /= 0 .or. .not. exists) then
         status = CP_ERR_IO
         return
      end if
      if (file_size < 0_int64 .or. file_size > huge(0_int64) / 8_int64) then
         status = CP_ERR_SIZE
         return
      end if
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = CP_ERR_IO
         return
      end if
      opened_size = -1_int64
      inquire(unit=unit, size=opened_size, iostat=io_status)
      if (io_status /= 0 .or. opened_size /= file_size) then
         close(unit)
         status = CP_ERR_CHANGED
         return
      end if

      call sha256_init(hash)
      remaining = file_size
      record = record_state_t()
      native_record = record_state_t()
      native_stats = corpus_preflight_stats_t()
      do while (remaining > 0_int64)
         read_count = int(min(remaining, int(size(chunk), int64)))
         read(unit, iostat=io_status) chunk(:read_count)
         if (io_status /= 0) then
            close(unit)
            status = CP_ERR_IO
            return
         end if
         call sha256_update_bytes(hash, chunk(:read_count), hash_status)
         if (hash_status /= FI_OK) then
            close(unit)
            status = CP_ERR_SIZE
            return
         end if
         do index = 1, read_count
            byte_value = iand(int(chunk(index), int32), 255_int32)
            call consume_byte(byte_value, record, stats, effective)
            if (custom_limits) call consume_byte(byte_value, native_record, native_stats, native)
         end do
         remaining = remaining - int(read_count, int64)
      end do
      if (record%byte_count > 0_int64) call finish_record(record, stats, effective, had_lf=.false.)
      if (custom_limits .and. native_record%byte_count > 0_int64) &
         call finish_record(native_record, native_stats, native, had_lf=.false.)
      ending_size = -1_int64
      inquire(unit=unit, size=ending_size, iostat=io_status)
      close(unit, iostat=close_status)
      if (io_status /= 0 .or. close_status /= 0 .or. ending_size /= file_size) then
         status = CP_ERR_CHANGED
         return
      end if
      call sha256_final_hex(hash, stats%sha256, hash_status)
      if (hash_status /= FI_OK) then
         status = CP_ERR_SIZE
         return
      end if
      stats%bytes = file_size
      if (present(native_strict_ok)) then
         if (custom_limits) then
            native_strict_ok = corpus_preflight_strict_ok(native_stats)
         else
            native_strict_ok = corpus_preflight_strict_ok(stats)
         end if
      end if
      status = CP_OK
   end subroutine corpus_preflight_scan

   pure subroutine consume_byte(byte_value, record, stats, limits)
      integer(int32), intent(in) :: byte_value
      type(record_state_t), intent(inout) :: record
      type(corpus_preflight_stats_t), intent(inout) :: stats
      type(corpus_preflight_limits_t), intent(in) :: limits

      if (byte_value == 10_int32) then
         call finish_record(record, stats, limits, had_lf=.true.)
         record = record_state_t()
      else
         record%byte_count = record%byte_count + 1_int64
         record%last_byte = byte_value
         if (byte_value > 127_int32) record%non_ascii = record%non_ascii + 1_int64
         if (is_token_byte(byte_value)) then
            if (record%token_length < limits%token_length_bytes) then
               record%token_length = record%token_length + 1_int64
            else
               record%token_too_long = .true.
            end if
         else
            call finish_token(record, limits)
         end if
      end if
   end subroutine consume_byte

   pure logical function is_token_byte(value)
      integer(int32), intent(in) :: value
      is_token_byte = (value >= 65_int32 .and. value <= 90_int32) .or. &
                      (value >= 97_int32 .and. value <= 122_int32) .or. &
                      (value >= 48_int32 .and. value <= 57_int32) .or. &
                      value == 95_int32 .or. value == 39_int32
   end function is_token_byte

   pure subroutine finish_token(record, limits)
      type(record_state_t), intent(inout) :: record
      type(corpus_preflight_limits_t), intent(in) :: limits
      if (record%token_length == 0_int64 .and. .not. record%token_too_long) return
      if (record%token_too_long .or. record%accepted >= limits%max_tokens_per_line) then
         record%dropped = record%dropped + 1_int64
      else
         record%accepted = record%accepted + 1_int64
      end if
      record%token_length = 0_int64
      record%token_too_long = .false.
   end subroutine finish_token

   pure subroutine finish_record(record, stats, limits, had_lf)
      type(record_state_t), intent(inout) :: record
      type(corpus_preflight_stats_t), intent(inout) :: stats
      type(corpus_preflight_limits_t), intent(in) :: limits
      logical, intent(in) :: had_lf
      call finish_token(record, limits)
      stats%records = stats%records + 1_int64
      stats%maximum_record_bytes = max(stats%maximum_record_bytes, record%byte_count)
      if (had_lf .and. record%last_byte == 13_int32) stats%crlf_records = stats%crlf_records + 1_int64
      ! Match the HDC trainer: oversized physical records are dropped as a
      ! whole. Per-token and non-ASCII counters include only retained records.
      if (record%byte_count > limits%max_line_bytes) then
         stats%overlong_records = stats%overlong_records + 1_int64
         return
      end if
      stats%accepted_tokens = stats%accepted_tokens + record%accepted
      stats%dropped_tokens = stats%dropped_tokens + record%dropped
      stats%non_ascii_bytes = stats%non_ascii_bytes + record%non_ascii
      if (record%dropped > 0_int64) &
         stats%records_with_tokenizer_truncation = stats%records_with_tokenizer_truncation + 1_int64
   end subroutine finish_record

   pure logical function corpus_preflight_strict_ok(stats)
      type(corpus_preflight_stats_t), intent(in) :: stats
      corpus_preflight_strict_ok = stats%overlong_records == 0_int64 .and. &
         stats%records_with_tokenizer_truncation == 0_int64 .and. stats%non_ascii_bytes == 0_int64
   end function corpus_preflight_strict_ok

end module corpus_preflight
