! SPDX-License-Identifier: MIT
! Portable, fail-closed command-line numeric and argument parsing.
! These routines are control-plane only; no training-loop work is added.
module portable_cli
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use, intrinsic :: ieee_exceptions, only: ieee_status_type, ieee_get_status, ieee_set_status, &
      ieee_support_halting, ieee_support_flag, ieee_set_halting_mode, ieee_set_flag, ieee_get_flag, &
      ieee_overflow, ieee_underflow, ieee_invalid
   implicit none (type, external)
   private

   integer(int32), parameter, public :: CLI_OK = 0_int32
   integer(int32), parameter, public :: CLI_ERR_INVALID = 1_int32
   public :: cli_get_argument, cli_parse_integer64, cli_parse_integer32, cli_parse_real64, cli_valid_utf8
   public :: cli_real64_is_finite

contains

   subroutine cli_get_argument(index, value, status, path)
      integer, intent(in) :: index
      character(len=*), intent(out) :: value
      integer(int32), intent(out) :: status
      logical, intent(in), optional :: path
      integer :: argument_length, io_status
      logical :: is_path

      value = ''
      status = CLI_ERR_INVALID
      is_path = .false.
      if (present(path)) is_path = path
      call get_command_argument(index, length=argument_length, status=io_status)
      ! Length may be undefined after a failed inquiry and Fortran does not
      ! require short-circuit evaluation of logical expressions.
      if (io_status /= 0) return
      if (argument_length <= 0 .or. argument_length > len(value)) return
      call get_command_argument(index, value, status=io_status)
      if (io_status /= 0) return
      if (.not. cli_valid_utf8(value(:argument_length))) return
      if (is_path) then
         ! Fortran fixed-length path APIs use trim(); a trailing blank would
         ! otherwise name a different file. Inquire first, before any output.
         if (value(argument_length:argument_length) == ' ') return
      end if
      status = CLI_OK
   end subroutine cli_get_argument

   ! Character values can be signed or use non-ASCII collating codes on some
   ! processors. UTF-8 validation is about the stored octets, not collating order.
   pure integer function cli_octet(value) result(byte)
      character(len=1), intent(in) :: value
      integer(int8) :: signed_byte

      signed_byte = transfer(value, 0_int8)
      byte = iand(int(signed_byte), 255)
   end function cli_octet

   pure logical function cli_valid_utf8(value) result(valid)
      character(len=*), intent(in) :: value
      integer :: position, first, second, third, fourth, count

      valid = .false.
      position = 1
      count = len(value)
      do while (position <= count)
         first = cli_octet(value(position:position))
         if (first <= 127) then
            position = position + 1
         else if (first >= 194 .and. first <= 223) then
            if (position + 1 > count) return
            second = cli_octet(value(position + 1:position + 1))
            if (second < 128 .or. second > 191) return
            position = position + 2
         else if (first >= 224 .and. first <= 239) then
            if (position + 2 > count) return
            second = cli_octet(value(position + 1:position + 1))
            third = cli_octet(value(position + 2:position + 2))
            if (second < 128 .or. second > 191 .or. third < 128 .or. third > 191) return
            if ((first == 224 .and. second < 160) .or. (first == 237 .and. second > 159)) return
            position = position + 3
         else if (first >= 240 .and. first <= 244) then
            if (position + 3 > count) return
            second = cli_octet(value(position + 1:position + 1))
            third = cli_octet(value(position + 2:position + 2))
            fourth = cli_octet(value(position + 3:position + 3))
            if (second < 128 .or. second > 191 .or. third < 128 .or. third > 191 .or. &
                fourth < 128 .or. fourth > 191) return
            if ((first == 240 .and. second < 144) .or. (first == 244 .and. second > 143)) return
            position = position + 4
         else
            return
         end if
      end do
      valid = .true.
   end function cli_valid_utf8

   subroutine cli_parse_integer64(raw, parsed, status)
      character(len=*), intent(in) :: raw
      integer(int64), intent(out) :: parsed
      integer(int32), intent(out) :: status
      character(len=len(raw)) :: token
      integer :: position, last, io_status

      parsed = 0_int64
      status = CLI_ERR_INVALID
      token = adjustl(raw)
      last = len_trim(token)
      if (last == 0) return
      position = 1
      if (token(position:position) == '+' .or. token(position:position) == '-') position = position + 1
      if (position > last) return
      do while (position <= last)
         if (token(position:position) < '0' .or. token(position:position) > '9') return
         position = position + 1
      end do
      read(token, *, iostat=io_status) parsed
      if (io_status /= 0) then
         parsed = 0_int64
         return
      end if
      status = CLI_OK
   end subroutine cli_parse_integer64

   subroutine cli_parse_integer32(raw, parsed, status)
      character(len=*), intent(in) :: raw
      integer(int32), intent(out) :: parsed
      integer(int32), intent(out) :: status
      integer(int64) :: wide

      parsed = 0_int32
      call cli_parse_integer64(raw, wide, status)
      if (status /= CLI_OK) return
      if (wide < -int(huge(0_int32), int64) - 1_int64 .or. wide > int(huge(0_int32), int64)) then
         status = CLI_ERR_INVALID
         return
      end if
      parsed = int(wide, int32)
   end subroutine cli_parse_integer32

   pure elemental logical function cli_real64_is_finite(value) result(finite)
      real(real64), intent(in) :: value
      integer(int64) :: bits
      logical :: binary64_layout

      ! Inspect the exponent without evaluating a signaling NaN only when both
      ! the model and representative encodings match IEEE binary64. Inquiring
      ! the numeric model alone does not establish byte/bit layout. Other
      ! formats use the intrinsic and need their own platform FPE validation.
      binary64_layout = storage_size(value) == 64 .and. bit_size(0_int64) == 64 .and. &
          radix(value) == 2 .and. digits(value) == 53 .and. &
          maxexponent(value) == 1024 .and. minexponent(value) == -1021
      if (binary64_layout) then
         binary64_layout = transfer(1.0_real64, 0_int64) == int(z'3FF0000000000000', int64) .and. &
                           transfer(huge(0.0_real64), 0_int64) == int(z'7FEFFFFFFFFFFFFF', int64) .and. &
                           transfer(-0.0_real64, 0_int64) == ibset(0_int64, 63)
      end if
      if (binary64_layout) then
         bits = transfer(value, 0_int64)
         finite = iand(bits, int(z'7FF0000000000000', int64)) /= int(z'7FF0000000000000', int64)
      else
         finite = ieee_is_finite(value)
      end if
   end function cli_real64_is_finite

   subroutine cli_parse_real64(raw, parsed, status)
      character(len=*), intent(in) :: raw
      real(real64), intent(out) :: parsed
      integer(int32), intent(out) :: status
      type(ieee_status_type) :: saved_status
      character(len=len(raw)) :: token
      character(len=1) :: digit
      integer :: position, last, digit_count, io_status
      logical :: nonzero_mantissa, overflowed, invalid

      parsed = 0.0_real64
      status = CLI_ERR_INVALID
      token = adjustl(raw)
      last = len_trim(token)
      if (last == 0) return
      position = 1
      if (token(position:position) == '+' .or. token(position:position) == '-') position = position + 1
      digit_count = 0
      nonzero_mantissa = .false.
      do while (position <= last)
         digit = token(position:position)
         if (digit < '0' .or. digit > '9') exit
         digit_count = digit_count + 1
         nonzero_mantissa = nonzero_mantissa .or. digit /= '0'
         position = position + 1
      end do
      if (position <= last) then
         if (token(position:position) == '.') then
            position = position + 1
            do while (position <= last)
               digit = token(position:position)
               if (digit < '0' .or. digit > '9') exit
               digit_count = digit_count + 1
               nonzero_mantissa = nonzero_mantissa .or. digit /= '0'
               position = position + 1
            end do
         end if
      end if
      if (digit_count == 0) return
      if (position <= last) then
         digit = token(position:position)
         if (digit == 'e' .or. digit == 'E' .or. digit == 'd' .or. digit == 'D') then
            position = position + 1
            if (position <= last) then
               if (token(position:position) == '+' .or. token(position:position) == '-') position = position + 1
            end if
            digit_count = 0
            do while (position <= last)
               digit = token(position:position)
               if (digit < '0' .or. digit > '9') exit
               digit_count = digit_count + 1
               position = position + 1
            end do
            if (digit_count == 0) return
         end if
      end if
      if (position <= last) return

      ! Conversion can signal before IOSTAT reports overflow when FPE traps are
      ! enabled. If overflow/invalid flags cannot be queried, fail closed rather
      ! than guessing; otherwise temporarily suppress traps and restore modes.
      if (.not. ieee_support_flag(ieee_overflow) .or. .not. ieee_support_flag(ieee_invalid)) return
      call ieee_get_status(saved_status)
      if (ieee_support_halting(ieee_overflow)) call ieee_set_halting_mode(ieee_overflow, .false.)
      if (ieee_support_halting(ieee_underflow)) call ieee_set_halting_mode(ieee_underflow, .false.)
      if (ieee_support_halting(ieee_invalid)) call ieee_set_halting_mode(ieee_invalid, .false.)
      call ieee_set_flag(ieee_overflow, .false.)
      call ieee_set_flag(ieee_invalid, .false.)
      read(token, *, iostat=io_status) parsed
      call ieee_get_flag(ieee_overflow, overflowed)
      call ieee_get_flag(ieee_invalid, invalid)
      call ieee_set_status(saved_status)
      if (io_status /= 0 .or. overflowed .or. invalid) return
      if (.not. cli_real64_is_finite(parsed)) return
      ! A nonzero literal that rounds to zero must not silently disable an option.
      if (parsed == 0.0_real64 .and. nonzero_mantissa) return
      status = CLI_OK
   end subroutine cli_parse_real64

end module portable_cli
