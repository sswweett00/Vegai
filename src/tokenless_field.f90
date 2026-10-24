! SPDX-License-Identifier: MIT
!==============================================================================
!  tokenless_field.f90
!  Tokenizer-free byte-event fields with exact finite Walsh-spectrum checks.
!
!  This module deliberately distinguishes formal guarantees from semantic
!  claims.  It accepts raw octets directly, without a vocabulary, BPE/Unigram
!  segmentation, normalization, or learned tokenizer.  The resulting fixed
!  16,384-bit fields are deterministic sketches, not lossless encodings of an
!  unbounded byte stream.  Raw source identity must remain in the governed data
!  plane.
!
!  The Walsh-Hadamard routines are exact int64 finite-signal operations for the
!  bounded binary field.  For an unnormalised transform H with H^T H = D I,
!  Parseval gives sum(Hx)^2 = D * sum(x)^2.  Here D=16,384 and x is bipolar,
!  so the expected spectral energy is D^2, represented exactly below.
!==============================================================================
module tokenless_field
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64
   use hyper_types, only: WORD_BITS, HV_BITS, HV_WORDS, hv_t, hv_zero, hv_from_seed, hv_bind, &
                          hv_acc_t, hv_acc_reset, hv_acc_add, hv_acc_finalize, HV_OK
   implicit none
   private

   integer(int32), parameter, public :: FIELD_OK                    = 0_int32
   integer(int32), parameter, public :: FIELD_ERR_INVALID_ARGUMENT  = -700_int32
   integer(int32), parameter, public :: FIELD_ERR_EMPTY_INPUT       = -701_int32
   integer(int32), parameter, public :: FIELD_ERR_CAPACITY          = -702_int32
   integer(int32), parameter, public :: FIELD_ERR_INVALID_SPECTRUM  = -703_int32
   integer(int32), parameter, public :: FIELD_ERR_IO                = -704_int32

   ! The signed HDC accumulator is intentionally the limiting resource.  This
   ! exact bound prevents counter wraparound; larger raw streams must be split
   ! upstream with a governed boundary rather than silently truncated.
   integer(int32), parameter, public :: FIELD_MAX_BYTES = 32767_int32
   integer(int64), parameter, public :: FIELD_WALSH_EXPECTED_ENERGY = &
      int(HV_BITS, int64) * int(HV_BITS, int64)

   type, public :: field_encoding_t
      ! Position-sensitive XOR composition of raw byte events.  It makes byte
      ! position part of a fixed-dimensional sketch, but cannot be injective
      ! for arbitrary streams.
      type(hv_t) :: order_field
      ! Deterministic bipolar bundle of byte-only events.  This is explicitly
      ! order-insensitive and exposes distributional rather than sequence data.
      type(hv_t) :: distribution_field
      integer(int32) :: byte_count = 0_int32
   end type field_encoding_t

   public :: field_encode_bytes
   public :: field_walsh_forward, field_walsh_inverse
   public :: field_bipolar_energy, field_spectral_energy, field_parseval_residual
   public :: field_fingerprint

   ! Domain-separated constants are representation parameters, not data,
   ! capabilities, prompts, or learned weights.
   integer(int64), parameter :: FIELD_BYTE_TAG = int(z'BA5EBA11C0DEC0DE', int64)
   integer(int64), parameter :: FIELD_ORDER_TAG = int(z'F13D0FD300000001', int64)
   integer(int64), parameter :: FIELD_TIE_TAG = int(z'71E5EED5A17F1E1D', int64)

contains

   pure function unsigned_octet(value) result(octet)
      integer(int8), intent(in) :: value
      integer(int32) :: octet

      octet = int(value, int32)
      if (octet < 0_int32) octet = octet + 256_int32
   end function unsigned_octet

   pure function byte_atom(octet) result(atom)
      integer(int32), intent(in) :: octet
      type(hv_t) :: atom
      integer(int64) :: seed, octet64

      octet64 = int(octet, int64)
      seed = ieor(FIELD_BYTE_TAG, ishftc(octet64, 17))
      seed = ieor(seed, ishftc(octet64, -9))
      atom = hv_from_seed(seed)
   end function byte_atom

   pure function ordered_byte_atom(octet, position) result(atom)
      integer(int32), intent(in) :: octet, position
      type(hv_t) :: atom
      integer(int64) :: seed, octet64, position64

      octet64 = int(octet, int64)
      position64 = int(position, int64)
      seed = ieor(FIELD_ORDER_TAG, ishftc(position64, 23))
      seed = ieor(seed, ishftc(octet64, -17))
      seed = ieor(seed, ishftc(ieor(position64, octet64), 41))
      atom = hv_from_seed(seed)
   end function ordered_byte_atom

   pure subroutine reset_encoding(encoding)
      type(field_encoding_t), intent(out) :: encoding

      encoding%order_field = hv_zero()
      encoding%distribution_field = hv_zero()
      encoding%byte_count = 0_int32
   end subroutine reset_encoding

   ! Direct byte-event encoding.  No text tokenization, vocabulary lookup,
   ! normalization, or subword merge is performed.  A byte stream is distinct
   ! from a semantic language representation; callers retain the immutable raw
   ! source digest when exact source identity matters.
   pure subroutine field_encode_bytes(bytes, encoding, status)
      integer(int8), intent(in) :: bytes(:)
      type(field_encoding_t), intent(out) :: encoding
      integer(int32), intent(out) :: status
      type(hv_acc_t) :: accumulator
      type(hv_t) :: distribution_atom, order_atom
      integer(int32) :: byte_index, byte_count, octet, add_status

      call reset_encoding(encoding)
      status = FIELD_OK
      byte_count = size(bytes, kind=int32)
      if (byte_count == 0_int32) then
         status = FIELD_ERR_EMPTY_INPUT
         return
      end if
      if (byte_count < 0_int32 .or. byte_count > FIELD_MAX_BYTES) then
         status = FIELD_ERR_CAPACITY
         return
      end if

      call hv_acc_reset(accumulator)
      encoding%order_field = hv_zero()
      do byte_index = 1_int32, byte_count
         octet = unsigned_octet(bytes(byte_index))
         distribution_atom = byte_atom(octet)
         order_atom = ordered_byte_atom(octet, byte_index)
         call hv_acc_add(accumulator, distribution_atom, add_status)
         if (add_status /= HV_OK) then
            call reset_encoding(encoding)
            status = FIELD_ERR_CAPACITY
            return
         end if
         encoding%order_field = hv_bind(encoding%order_field, order_atom)
      end do

      encoding%distribution_field = hv_acc_finalize(accumulator, FIELD_TIE_TAG)
      encoding%byte_count = byte_count
   end subroutine field_encode_bytes

   pure subroutine unpack_bipolar(field, values)
      type(hv_t), intent(in) :: field
      integer(int64), intent(out) :: values(HV_BITS)
      integer(int32) :: word, bit, index

      do word = 1_int32, HV_WORDS
         do bit = 0_int32, WORD_BITS - 1_int32
            index = (word - 1_int32) * WORD_BITS + bit + 1_int32
            if (btest(field%bits(word), bit)) then
               values(index) = 1_int64
            else
               values(index) = -1_int64
            end if
         end do
      end do
   end subroutine unpack_bipolar

   ! Unnormalised in-place Walsh-Hadamard transform.  The public forward path
   ! starts at +/-1, hence every intermediate is bounded by HV_BITS.  The
   ! inverse accepts only valid bounded spectra, so its largest intermediate is
   ! HV_BITS**2 (268,435,456), far inside signed int64 range.
   pure subroutine walsh_in_place(values)
      integer(int64), intent(inout) :: values(:)
      integer(int32) :: span, start_index, offset, left_index, right_index
      integer(int64) :: left_value, right_value

      span = 1_int32
      do while (span < size(values, kind=int32))
         do start_index = 1_int32, size(values, kind=int32), 2_int32 * span
            do offset = 0_int32, span - 1_int32
               left_index = start_index + offset
               right_index = left_index + span
               left_value = values(left_index)
               right_value = values(right_index)
               values(left_index) = left_value + right_value
               values(right_index) = left_value - right_value
            end do
         end do
         span = span * 2_int32
      end do
   end subroutine walsh_in_place

   pure subroutine field_walsh_forward(field, spectrum, status)
      type(hv_t), intent(in) :: field
      integer(int64), intent(out) :: spectrum(HV_BITS)
      integer(int32), intent(out) :: status

      call unpack_bipolar(field, spectrum)
      call walsh_in_place(spectrum)
      status = FIELD_OK
   end subroutine field_walsh_forward

   pure subroutine field_walsh_inverse(spectrum, field, status)
      integer(int64), intent(in) :: spectrum(HV_BITS)
      type(hv_t), intent(out) :: field
      integer(int32), intent(out) :: status
      integer(int64) :: work(HV_BITS), amplitude
      integer(int32) :: index, word, bit

      field = hv_zero()
      status = FIELD_OK
      if (any(spectrum < -int(HV_BITS, int64)) .or. any(spectrum > int(HV_BITS, int64))) then
         status = FIELD_ERR_INVALID_SPECTRUM
         return
      end if

      work = spectrum
      call walsh_in_place(work)
      do index = 1_int32, HV_BITS
         if (modulo(work(index), int(HV_BITS, int64)) /= 0_int64) then
            field = hv_zero()
            status = FIELD_ERR_INVALID_SPECTRUM
            return
         end if
         amplitude = work(index) / int(HV_BITS, int64)
         if (amplitude /= -1_int64 .and. amplitude /= 1_int64) then
            field = hv_zero()
            status = FIELD_ERR_INVALID_SPECTRUM
            return
         end if
         if (amplitude == 1_int64) then
            word = (index - 1_int32) / WORD_BITS + 1_int32
            bit = modulo(index - 1_int32, WORD_BITS)
            field%bits(word) = ibset(field%bits(word), bit)
         end if
      end do
   end subroutine field_walsh_inverse

   pure function field_bipolar_energy(field) result(energy)
      type(hv_t), intent(in) :: field
      integer(int64) :: energy

      ! Every packed binary field maps to exactly one +/-1 value per bit.
      energy = int(size(field%bits, kind=int32), int64) * int(WORD_BITS, int64)
   end function field_bipolar_energy

   pure subroutine field_spectral_energy(spectrum, energy, status)
      integer(int64), intent(in) :: spectrum(HV_BITS)
      integer(int64), intent(out) :: energy
      integer(int32), intent(out) :: status
      integer(int32) :: index

      energy = 0_int64
      status = FIELD_OK
      do index = 1_int32, HV_BITS
         ! This is the valid range of a forward transform of a bipolar field.
         ! Restricting it makes the square and accumulated sum demonstrably
         ! int64-safe instead of relying on overflow behaviour.
         if (spectrum(index) < -int(HV_BITS, int64) .or. spectrum(index) > int(HV_BITS, int64)) then
            energy = 0_int64
            status = FIELD_ERR_INVALID_SPECTRUM
            return
         end if
         energy = energy + spectrum(index) * spectrum(index)
      end do
   end subroutine field_spectral_energy

   pure subroutine field_parseval_residual(spectrum, residual, status)
      integer(int64), intent(in) :: spectrum(HV_BITS)
      integer(int64), intent(out) :: residual
      integer(int32), intent(out) :: status
      integer(int64) :: energy

      call field_spectral_energy(spectrum, energy, status)
      if (status /= FIELD_OK) then
         residual = 0_int64
         return
      end if
      residual = energy - FIELD_WALSH_EXPECTED_ENERGY
   end subroutine field_parseval_residual

   ! A compact diagnostic only: this rolling XOR/rotation fingerprint is not a
   ! cryptographic source identity and must never replace a source SHA-256.
   pure function field_fingerprint(field) result(fingerprint)
      type(hv_t), intent(in) :: field
      integer(int64) :: fingerprint
      integer(int32) :: word

      fingerprint = 0_int64
      do word = 1_int32, HV_WORDS
         fingerprint = ieor(ishftc(fingerprint, 7), field%bits(word))
      end do
   end function field_fingerprint

end module tokenless_field
