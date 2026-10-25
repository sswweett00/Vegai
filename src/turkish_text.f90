! SPDX-License-Identifier: MIT
!==============================================================================
!  turkish_text.f90
!  Strict UTF-8 Turkish dialogue normalization for the native Fortran surface.
!
!  The HDC checkpoint vocabulary remains its documented ASCII contract.  This
!  module does not pretend to be a universal Unicode tokenizer: it validates
!  UTF-8 display text and deterministically folds the Turkish Latin characters
!  needed to query an ASCII-trained checkpoint.  Responses remain byte-faithful
!  UTF-8 data from the knowledge base.
!==============================================================================
module turkish_text
   use, intrinsic :: iso_fortran_env, only: int32
   implicit none
   private

   integer(int32), parameter, public :: TURKISH_TEXT_OK = 0_int32
   integer(int32), parameter, public :: TURKISH_TEXT_ERR_EMPTY = -720_int32
   integer(int32), parameter, public :: TURKISH_TEXT_ERR_LENGTH = -721_int32
   integer(int32), parameter, public :: TURKISH_TEXT_ERR_UTF8 = -722_int32
   integer(int32), parameter, public :: TURKISH_TEXT_ERR_UNSUPPORTED = -723_int32
   integer(int32), parameter, public :: TURKISH_TEXT_ERR_CONTROL = -724_int32

   public :: turkish_fold_prompt_utf8, turkish_validate_display_utf8

contains

   pure logical function utf8_continuation(byte_value) result(valid)
      integer, intent(in) :: byte_value

      valid = byte_value >= 128 .and. byte_value <= 191
   end function utf8_continuation

   subroutine append_ascii(output, output_length, codepoint, status)
      character(len=*), intent(inout) :: output
      integer, intent(inout) :: output_length
      integer, intent(in) :: codepoint
      integer(int32), intent(inout) :: status

      if (status /= TURKISH_TEXT_OK) return
      if (output_length >= len(output)) then
         status = TURKISH_TEXT_ERR_LENGTH
         return
      end if
      output_length = output_length + 1
      output(output_length:output_length) = achar(codepoint)
   end subroutine append_ascii

   subroutine turkish_fold_prompt_utf8(input, output, status)
      ! Fold the explicitly supported Turkish dialogue input alphabet into the
      ! ASCII projection used by native checkpoint training. ASCII punctuation
      ! is retained for the existing tokenizer to handle. Unsupported Unicode
      ! is rejected rather than silently changed into a different query.
      character(len=*), intent(in) :: input
      character(len=*), intent(out) :: output
      integer(int32), intent(out) :: status
      integer :: input_length, position, output_length, first, second, third

      output = ''
      status = TURKISH_TEXT_OK
      input_length = len_trim(input)
      if (input_length <= 0) then
         status = TURKISH_TEXT_ERR_EMPTY
         return
      end if
      output_length = 0
      position = 1
      do while (position <= input_length)
         first = iachar(input(position:position))
         if (first >= 32 .and. first <= 126) then
            call append_ascii(output, output_length, first, status)
            position = position + 1
            cycle
         end if
         if (first < 128) then
            status = TURKISH_TEXT_ERR_CONTROL
            return
         end if
         if (position + 1 > input_length) then
            status = TURKISH_TEXT_ERR_UTF8
            return
         end if
         second = iachar(input(position + 1:position + 1))

         ! ç Ç ö Ö ü Ü, and Turkish orthographic circumflex variants
         ! â Â î Î û Û. The latter retain a display distinction in source TSV
         ! but map to the current ASCII checkpoint's lexical projection.
         if (first == 195) then
            select case (second)
            case (130, 162)
               call append_ascii(output, output_length, iachar('a'), status)
            case (135, 167)
               call append_ascii(output, output_length, iachar('c'), status)
            case (142, 174)
               call append_ascii(output, output_length, iachar('i'), status)
            case (150, 182)
               call append_ascii(output, output_length, iachar('o'), status)
            case (155, 187)
               call append_ascii(output, output_length, iachar('u'), status)
            case (156, 188)
               call append_ascii(output, output_length, iachar('u'), status)
            case default
               status = TURKISH_TEXT_ERR_UNSUPPORTED
            end select
            position = position + 2
            if (status /= TURKISH_TEXT_OK) return
            cycle
         end if

         ! Ğ ğ İ ı
         if (first == 196) then
            select case (second)
            case (158, 159)
               call append_ascii(output, output_length, iachar('g'), status)
            case (176, 177)
               call append_ascii(output, output_length, iachar('i'), status)
            case default
               status = TURKISH_TEXT_ERR_UNSUPPORTED
            end select
            position = position + 2
            if (status /= TURKISH_TEXT_OK) return
            cycle
         end if

         ! Ş ş
         if (first == 197) then
            select case (second)
            case (158, 159)
               call append_ascii(output, output_length, iachar('s'), status)
            case default
               status = TURKISH_TEXT_ERR_UNSUPPORTED
            end select
            position = position + 2
            if (status /= TURKISH_TEXT_OK) return
            cycle
         end if

         ! Some public Turkish material contains Romanian comma-below s
         ! instead of Turkish cedilla s. Fold it explicitly rather than either
         ! rejecting an otherwise useful licensed row or silently dropping it.
         if (first == 200) then
            select case (second)
            case (152, 153)
               call append_ascii(output, output_length, iachar('s'), status)
            case default
               status = TURKISH_TEXT_ERR_UNSUPPORTED
            end select
            position = position + 2
            if (status /= TURKISH_TEXT_OK) return
            cycle
         end if

         ! Non-breaking space is safely a normal space in a prompt.
         if (first == 194 .and. second == 160) then
            call append_ascii(output, output_length, iachar(' '), status)
            position = position + 2
            if (status /= TURKISH_TEXT_OK) return
            cycle
         end if

         ! Fold common typographic apostrophes and dashes rather than making a
         ! visually equivalent Turkish prompt fail an ASCII checkpoint lookup.
         if (first == 226) then
            if (position + 2 > input_length) then
               status = TURKISH_TEXT_ERR_UTF8
               return
            end if
            third = iachar(input(position + 2:position + 2))
            if (second == 128 .and. (third == 152 .or. third == 153)) then
               call append_ascii(output, output_length, iachar("'"), status)
            else if (second == 128 .and. (third == 147 .or. third == 148)) then
               call append_ascii(output, output_length, iachar('-'), status)
            else
               status = TURKISH_TEXT_ERR_UNSUPPORTED
            end if
            position = position + 3
            if (status /= TURKISH_TEXT_OK) return
            cycle
         end if

         status = TURKISH_TEXT_ERR_UNSUPPORTED
         return
      end do
      if (output_length <= 0) status = TURKISH_TEXT_ERR_EMPTY
   end subroutine turkish_fold_prompt_utf8

   subroutine turkish_validate_display_utf8(input, maximum_bytes, status)
      ! Validate printable UTF-8 response/prompt display data without changing
      ! its bytes. The TSV separator is handled by the caller and is forbidden
      ! in fields; CR/LF/control characters cannot hide a second record.
      character(len=*), intent(in) :: input
      integer(int32), intent(in) :: maximum_bytes
      integer(int32), intent(out) :: status
      integer :: input_length, position, first, second, third, fourth

      status = TURKISH_TEXT_OK
      input_length = len_trim(input)
      if (input_length <= 0) then
         status = TURKISH_TEXT_ERR_EMPTY
         return
      end if
      if (maximum_bytes <= 0_int32 .or. input_length > int(maximum_bytes)) then
         status = TURKISH_TEXT_ERR_LENGTH
         return
      end if
      position = 1
      do while (position <= input_length)
         first = iachar(input(position:position))
         if (first <= 127) then
            if (first < 32 .or. first == 127) then
               status = TURKISH_TEXT_ERR_CONTROL
               return
            end if
            position = position + 1
            cycle
         end if

         if (first >= 194 .and. first <= 223) then
            if (position + 1 > input_length) then
               status = TURKISH_TEXT_ERR_UTF8
               return
            end if
            second = iachar(input(position + 1:position + 1))
            if (.not. utf8_continuation(second)) then
               status = TURKISH_TEXT_ERR_UTF8
               return
            end if
            ! C2 80..9F encodes Unicode C1 controls. They must not become
            ! invisible display characters merely because their UTF-8 syntax
            ! is valid; NBSP (C2 A0) remains permitted.
            if (first == 194 .and. second < 160) then
               status = TURKISH_TEXT_ERR_CONTROL
               return
            end if
            position = position + 2
            cycle
         end if

         if (first >= 224 .and. first <= 239) then
            if (position + 2 > input_length) then
               status = TURKISH_TEXT_ERR_UTF8
               return
            end if
            second = iachar(input(position + 1:position + 1))
            third = iachar(input(position + 2:position + 2))
            if (.not. utf8_continuation(second) .or. .not. utf8_continuation(third)) then
               status = TURKISH_TEXT_ERR_UTF8
               return
            end if
            if ((first == 224 .and. second < 160) .or. &
                (first == 237 .and. second > 159)) then
               status = TURKISH_TEXT_ERR_UTF8
               return
            end if
            position = position + 3
            cycle
         end if

         if (first >= 240 .and. first <= 244) then
            if (position + 3 > input_length) then
               status = TURKISH_TEXT_ERR_UTF8
               return
            end if
            second = iachar(input(position + 1:position + 1))
            third = iachar(input(position + 2:position + 2))
            fourth = iachar(input(position + 3:position + 3))
            if (.not. utf8_continuation(second) .or. .not. utf8_continuation(third) .or. &
                .not. utf8_continuation(fourth)) then
               status = TURKISH_TEXT_ERR_UTF8
               return
            end if
            if ((first == 240 .and. second < 144) .or. &
                (first == 244 .and. second > 143)) then
               status = TURKISH_TEXT_ERR_UTF8
               return
            end if
            position = position + 4
            cycle
         end if

         status = TURKISH_TEXT_ERR_UTF8
         return
      end do
   end subroutine turkish_validate_display_utf8

end module turkish_text
