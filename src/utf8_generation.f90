! SPDX-License-Identifier: MIT
! Incremental UTF-8 constraint for byte-model text generation. This module is
! not used by training or teacher-forced scoring and does not change a model.
module utf8_generation
   use, intrinsic :: iso_fortran_env, only: int32
   implicit none (type, external)
   private

   ! The decoder always starts at a code-point boundary. Keep its invariants
   ! private: callers can inspect completion but can only advance legal bytes.
   type, public :: utf8_generation_state_t
      private
      integer(int32) :: pending = 0_int32
      integer(int32) :: low = 128_int32
      integer(int32) :: high = 191_int32
   end type utf8_generation_state_t

   public :: utf8_generation_reset, utf8_generation_allows
   public :: utf8_generation_advance, utf8_generation_complete

contains

   pure subroutine utf8_generation_reset(state)
      type(utf8_generation_state_t), intent(out) :: state

      state = utf8_generation_state_t()
   end subroutine utf8_generation_reset

   ! `bytes_left` is the remaining output budget, including this byte. Starting
   ! a multibyte scalar is forbidden unless its entire encoding fits. Newline
   ! is a permitted stop token at a boundary, not a byte emitted by the CLI.
   ! Other C0 controls, DEL, NUL/token 256, overlong leads, lone continuations,
   ! surrogates and code points greater than U+10FFFF are excluded.
   pure logical function utf8_generation_allows(state, token, bytes_left) result(allowed)
      type(utf8_generation_state_t), intent(in) :: state
      integer(int32), intent(in) :: token, bytes_left

      allowed = .false.
      if (bytes_left < 1_int32 .or. token < 1_int32 .or. token > 255_int32) return
      if (state%pending > 0_int32) then
         allowed = bytes_left >= state%pending .and. token >= state%low .and. token <= state%high
      else if (token == 10_int32 .or. (token >= 32_int32 .and. token <= 126_int32)) then
         allowed = .true.
      else if (token >= 194_int32 .and. token <= 223_int32) then
         allowed = bytes_left >= 2_int32
      else if (token >= 224_int32 .and. token <= 239_int32) then
         allowed = bytes_left >= 3_int32
      else if (token >= 240_int32 .and. token <= 244_int32) then
         allowed = bytes_left >= 4_int32
      end if
   end function utf8_generation_allows

   ! Reject an illegal byte without modifying the state. The CLI handles the
   ! full-stop and newline stop tokens before advancing the generation state.
   pure subroutine utf8_generation_advance(state, token, bytes_left, accepted)
      type(utf8_generation_state_t), intent(inout) :: state
      integer(int32), intent(in) :: token, bytes_left
      logical, intent(out) :: accepted

      accepted = utf8_generation_allows(state, token, bytes_left)
      if (.not. accepted) return
      if (state%pending > 0_int32) then
         state%pending = state%pending - 1_int32
         state%low = 128_int32
         state%high = 191_int32
      else if (token >= 194_int32 .and. token <= 223_int32) then
         state%pending = 1_int32
      else if (token >= 224_int32 .and. token <= 239_int32) then
         state%pending = 2_int32
         if (token == 224_int32) state%low = 160_int32   ! no overlong U+0000..07FF
         if (token == 237_int32) state%high = 159_int32  ! no UTF-16 surrogates
      else if (token >= 240_int32 .and. token <= 244_int32) then
         state%pending = 3_int32
         if (token == 240_int32) state%low = 144_int32   ! no overlong U+0000..FFFF
         if (token == 244_int32) state%high = 143_int32  ! at most U+10FFFF
      end if
   end subroutine utf8_generation_advance

   pure logical function utf8_generation_complete(state) result(complete)
      type(utf8_generation_state_t), intent(in) :: state

      complete = state%pending == 0_int32
   end function utf8_generation_complete
end module utf8_generation
