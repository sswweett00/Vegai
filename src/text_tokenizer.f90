! SPDX-License-Identifier: MIT
!==============================================================================
!  text_tokenizer.f90
!  Deterministic, allocation-free ASCII tokenizer for streaming HDC pretraining.
!
!  Input is treated as bytes.  ASCII letters, digits, underscore and apostrophe
!  are token characters; ASCII letters are lower-cased.  UTF-8 multibyte bytes
!  are delimiters by design, so multilingual training should be pre-tokenized
!  into an ASCII-safe representation or use a future Unicode tokenizer.
!==============================================================================
module text_tokenizer
   use, intrinsic :: iso_fortran_env, only: int32
   implicit none
   private

   integer(int32), parameter, public :: TOKEN_LEN = 48_int32
   integer(int32), parameter, public :: MAX_LINE_CHARS = 8192_int32
   integer(int32), parameter, public :: MAX_TOKENS_PER_LINE = 1024_int32

   public :: tokenize_ascii_line

contains

   pure logical function is_token_character(character)
      character(len=1), intent(in) :: character
      integer(int32) :: code

      code = int(iachar(character), int32)
      is_token_character = (code >= iachar('A') .and. code <= iachar('Z')) .or. &
                           (code >= iachar('a') .and. code <= iachar('z')) .or. &
                           (code >= iachar('0') .and. code <= iachar('9')) .or. &
                           character == '_' .or. character == "'"
   end function is_token_character

   pure function lowercase_ascii(character) result(lowercase)
      character(len=1), intent(in) :: character
      character(len=1) :: lowercase
      integer(int32) :: code

      code = int(iachar(character), int32)
      if (code >= iachar('A') .and. code <= iachar('Z')) then
         lowercase = achar(code + iachar('a') - iachar('A'))
      else
         lowercase = character
      end if
   end function lowercase_ascii

   ! Tokenizes a single record into caller-owned fixed buffers.  A long token
   ! or excess token is dropped rather than truncated, avoiding vocabulary-key
   ! aliasing.  The caller can turn truncated=.true. into a strict-data error.
   pure subroutine tokenize_ascii_line(line, tokens, n_tokens, truncated)
      character(len=*), intent(in) :: line
      character(len=TOKEN_LEN), intent(out) :: tokens(:)
      integer(int32), intent(out) :: n_tokens
      logical, intent(out) :: truncated
      character(len=TOKEN_LEN) :: current
      integer(int32) :: position, current_length
      logical :: token_too_long

      tokens = ''
      n_tokens = 0_int32
      truncated = .false.
      current = ''
      current_length = 0_int32
      token_too_long = .false.

      do position = 1, len_trim(line)
         if (is_token_character(line(position:position))) then
            if (current_length < TOKEN_LEN) then
               current_length = current_length + 1_int32
               current(current_length:current_length) = lowercase_ascii(line(position:position))
            else
               token_too_long = .true.
            end if
         else
            call emit_current_token(current, current_length, token_too_long, tokens, n_tokens, truncated)
            current = ''
            current_length = 0_int32
            token_too_long = .false.
         end if
      end do
      call emit_current_token(current, current_length, token_too_long, tokens, n_tokens, truncated)
   end subroutine tokenize_ascii_line

   pure subroutine emit_current_token(current, current_length, token_too_long, tokens, n_tokens, truncated)
      character(len=TOKEN_LEN), intent(in) :: current
      integer(int32), intent(in) :: current_length
      logical, intent(in) :: token_too_long
      character(len=TOKEN_LEN), intent(inout) :: tokens(:)
      integer(int32), intent(inout) :: n_tokens
      logical, intent(inout) :: truncated

      if (current_length == 0_int32 .and. .not. token_too_long) return
      if (token_too_long) then
         truncated = .true.
         return
      end if
      if (n_tokens >= size(tokens, kind=int32)) then
         truncated = .true.
         return
      end if
      n_tokens = n_tokens + 1_int32
      tokens(n_tokens) = current
   end subroutine emit_current_token

end module text_tokenizer
