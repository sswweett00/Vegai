! SPDX-License-Identifier: MIT
!==============================================================================
!  fortran_integrity.f90
!  Streaming SHA-256 identities for native Fortran control-plane artifacts.
!
!  This module implements FIPS 180-4 SHA-256 with unsigned 32-bit words held
!  in nonnegative int64 values. Keeping modular sums below 5*2^32 avoids
!  signed int32 overflow while retaining standard Fortran bit intrinsics.
!==============================================================================
module fortran_integrity
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64
   implicit none
   private

   integer(int32), parameter, public :: FI_OK = 0_int32
   integer(int32), parameter, public :: FI_ERR_INVALID_ARGUMENT = -960_int32
   integer(int32), parameter, public :: FI_ERR_IO = -961_int32
   integer(int32), parameter, public :: FI_ERR_SIZE = -962_int32
   integer(int64), parameter :: MASK32 = int(z'00000000FFFFFFFF', int64)
   integer(int64), parameter :: INITIAL_HASH(8) = [ &
      int(z'6A09E667', int64), int(z'BB67AE85', int64), int(z'3C6EF372', int64), int(z'A54FF53A', int64), &
      int(z'510E527F', int64), int(z'9B05688C', int64), int(z'1F83D9AB', int64), int(z'5BE0CD19', int64) ]
   integer(int64), parameter :: ROUND_CONSTANTS(64) = [ &
      int(z'428A2F98', int64), int(z'71374491', int64), int(z'B5C0FBCF', int64), int(z'E9B5DBA5', int64), &
      int(z'3956C25B', int64), int(z'59F111F1', int64), int(z'923F82A4', int64), int(z'AB1C5ED5', int64), &
      int(z'D807AA98', int64), int(z'12835B01', int64), int(z'243185BE', int64), int(z'550C7DC3', int64), &
      int(z'72BE5D74', int64), int(z'80DEB1FE', int64), int(z'9BDC06A7', int64), int(z'C19BF174', int64), &
      int(z'E49B69C1', int64), int(z'EFBE4786', int64), int(z'0FC19DC6', int64), int(z'240CA1CC', int64), &
      int(z'2DE92C6F', int64), int(z'4A7484AA', int64), int(z'5CB0A9DC', int64), int(z'76F988DA', int64), &
      int(z'983E5152', int64), int(z'A831C66D', int64), int(z'B00327C8', int64), int(z'BF597FC7', int64), &
      int(z'C6E00BF3', int64), int(z'D5A79147', int64), int(z'06CA6351', int64), int(z'14292967', int64), &
      int(z'27B70A85', int64), int(z'2E1B2138', int64), int(z'4D2C6DFC', int64), int(z'53380D13', int64), &
      int(z'650A7354', int64), int(z'766A0ABB', int64), int(z'81C2C92E', int64), int(z'92722C85', int64), &
      int(z'A2BFE8A1', int64), int(z'A81A664B', int64), int(z'C24B8B70', int64), int(z'C76C51A3', int64), &
      int(z'D192E819', int64), int(z'D6990624', int64), int(z'F40E3585', int64), int(z'106AA070', int64), &
      int(z'19A4C116', int64), int(z'1E376C08', int64), int(z'2748774C', int64), int(z'34B0BCB5', int64), &
      int(z'391C0CB3', int64), int(z'4ED8AA4A', int64), int(z'5B9CCA4F', int64), int(z'682E6FF3', int64), &
      int(z'748F82EE', int64), int(z'78A5636F', int64), int(z'84C87814', int64), int(z'8CC70208', int64), &
      int(z'90BEFFFA', int64), int(z'A4506CEB', int64), int(z'BEF9A3F7', int64), int(z'C67178F2', int64) ]

   type, public :: sha256_state_t
      integer(int64) :: hash_words(8) = INITIAL_HASH
      integer(int8) :: buffer(64) = 0_int8
      integer(int32) :: buffer_length = 0_int32
      integer(int64) :: total_bytes = 0_int64
      logical :: finalized = .false.
   end type sha256_state_t

   public :: sha256_init, sha256_update_bytes, sha256_final_hex, sha256_file

contains

   subroutine sha256_init(state)
      type(sha256_state_t), intent(out) :: state

      state = sha256_state_t()
   end subroutine sha256_init

   subroutine sha256_update_bytes(state, bytes, status)
      type(sha256_state_t), intent(inout) :: state
      integer(int8), intent(in) :: bytes(:)
      integer(int32), intent(out) :: status
      integer :: input_index, take_count, available
      integer(int64) :: input_count

      status = FI_OK
      if (state%finalized) then
         status = FI_ERR_INVALID_ARGUMENT
         return
      end if
      input_count = int(size(bytes), int64)
      if (input_count > huge(0_int64) - state%total_bytes) then
         status = FI_ERR_SIZE
         return
      end if
      input_index = 1
      do while (input_index <= size(bytes))
         available = 64 - int(state%buffer_length)
         take_count = min(available, size(bytes) - input_index + 1)
         state%buffer(int(state%buffer_length) + 1:int(state%buffer_length) + take_count) = &
            bytes(input_index:input_index + take_count - 1)
         state%buffer_length = state%buffer_length + int(take_count, int32)
         input_index = input_index + take_count
         if (state%buffer_length == 64_int32) then
            call sha256_compress_block(state%hash_words, state%buffer)
            state%buffer_length = 0_int32
         end if
      end do
      state%total_bytes = state%total_bytes + input_count
   end subroutine sha256_update_bytes

   subroutine sha256_final_hex(state, digest_hex, status)
      type(sha256_state_t), intent(inout) :: state
      character(len=64), intent(out) :: digest_hex
      integer(int32), intent(out) :: status
      integer(int64) :: bit_length
      integer :: buffer_length, index, output_position

      digest_hex = ''
      status = FI_OK
      if (state%finalized) then
         status = FI_ERR_INVALID_ARGUMENT
         return
      end if
      if (state%total_bytes > huge(0_int64) / 8_int64) then
         status = FI_ERR_SIZE
         return
      end if
      bit_length = state%total_bytes * 8_int64
      buffer_length = int(state%buffer_length)
      state%buffer(buffer_length + 1) = int(z'80', int8)
      if (buffer_length + 1 > 56) then
         if (buffer_length + 2 <= 64) state%buffer(buffer_length + 2:64) = 0_int8
         call sha256_compress_block(state%hash_words, state%buffer)
         state%buffer = 0_int8
         buffer_length = 0
      end if
      if (buffer_length + 2 <= 56) state%buffer(buffer_length + 2:56) = 0_int8
      do index = 1, 8
         state%buffer(56 + index) = sha256_u8_to_i8(iand(shiftr(bit_length, 8 * (8 - index)), 255_int64))
      end do
      call sha256_compress_block(state%hash_words, state%buffer)
      state%buffer_length = 0_int32
      state%finalized = .true.
      output_position = 1
      do index = 1, 8
         call sha256_word_to_hex(state%hash_words(index), digest_hex(output_position:output_position + 7))
         output_position = output_position + 8
      end do
   end subroutine sha256_final_hex

   subroutine sha256_file(path, digest_hex, byte_count, status)
      character(len=*), intent(in) :: path
      character(len=64), intent(out) :: digest_hex
      integer(int64), intent(out) :: byte_count
      integer(int32), intent(out) :: status
      type(sha256_state_t) :: state
      integer(int8) :: chunk(65536)
      integer :: unit, io_status, read_count
      integer(int64) :: remaining, file_size
      logical :: exists

      digest_hex = ''
      byte_count = 0_int64
      status = FI_ERR_INVALID_ARGUMENT
      if (len_trim(path) == 0) return
      inquire(file=trim(path), exist=exists, size=file_size, iostat=io_status)
      if (io_status /= 0 .or. .not. exists) then
         status = FI_ERR_IO
         return
      end if
      if (file_size < 0_int64 .or. file_size > huge(0_int64) / 8_int64) then
         status = FI_ERR_SIZE
         return
      end if
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = FI_ERR_IO
         return
      end if
      call sha256_init(state)
      remaining = file_size
      do while (remaining > 0_int64)
         read_count = int(min(remaining, int(size(chunk), int64)))
         read(unit, iostat=io_status) chunk(:read_count)
         if (io_status /= 0) then
            close(unit)
            status = FI_ERR_IO
            return
         end if
         call sha256_update_bytes(state, chunk(:read_count), status)
         if (status /= FI_OK) then
            close(unit)
            return
         end if
         remaining = remaining - int(read_count, int64)
      end do
      close(unit)
      call sha256_final_hex(state, digest_hex, status)
      if (status == FI_OK) byte_count = file_size
   end subroutine sha256_file

   subroutine sha256_compress_block(hash_words, block)
      integer(int64), intent(inout) :: hash_words(8)
      integer(int8), intent(in) :: block(64)
      integer(int64) :: words(64), a, b, c, d, e, f, g, h, first_term, second_term
      integer :: index, byte_index

      do index = 1, 16
         byte_index = 4 * (index - 1)
         words(index) = ior(ior(shiftl(sha256_byte_to_u64(block(byte_index + 1)), 24), &
                                shiftl(sha256_byte_to_u64(block(byte_index + 2)), 16)), &
                            ior(shiftl(sha256_byte_to_u64(block(byte_index + 3)), 8), &
                                sha256_byte_to_u64(block(byte_index + 4))))
      end do
      do index = 17, 64
         words(index) = sha256_add4(sha256_small_sigma0(words(index - 15)), words(index - 7), &
                                    sha256_small_sigma1(words(index - 2)), words(index - 16))
      end do
      a = hash_words(1)
      b = hash_words(2)
      c = hash_words(3)
      d = hash_words(4)
      e = hash_words(5)
      f = hash_words(6)
      g = hash_words(7)
      h = hash_words(8)
      do index = 1, 64
         first_term = sha256_add5(h, sha256_big_sigma1(e), sha256_choose(e, f, g), ROUND_CONSTANTS(index), words(index))
         second_term = sha256_add2(sha256_big_sigma0(a), sha256_majority(a, b, c))
         h = g
         g = f
         f = e
         e = sha256_add2(d, first_term)
         d = c
         c = b
         b = a
         a = sha256_add2(first_term, second_term)
      end do
      hash_words(1) = sha256_add2(hash_words(1), a)
      hash_words(2) = sha256_add2(hash_words(2), b)
      hash_words(3) = sha256_add2(hash_words(3), c)
      hash_words(4) = sha256_add2(hash_words(4), d)
      hash_words(5) = sha256_add2(hash_words(5), e)
      hash_words(6) = sha256_add2(hash_words(6), f)
      hash_words(7) = sha256_add2(hash_words(7), g)
      hash_words(8) = sha256_add2(hash_words(8), h)
   end subroutine sha256_compress_block

   pure integer(int8) function sha256_u8_to_i8(unsigned_value) result(byte_value)
      integer(int64), intent(in) :: unsigned_value

      if (unsigned_value <= 127_int64) then
         byte_value = int(unsigned_value, int8)
      else
         byte_value = int(unsigned_value - 256_int64, int8)
      end if
   end function sha256_u8_to_i8

   pure integer(int64) function sha256_byte_to_u64(byte_value) result(unsigned_value)
      integer(int8), intent(in) :: byte_value

      unsigned_value = iand(int(byte_value, int64), 255_int64)
   end function sha256_byte_to_u64

   pure integer(int64) function sha256_add2(first, second) result(sum)
      integer(int64), intent(in) :: first, second

      sum = iand(first + second, MASK32)
   end function sha256_add2

   pure integer(int64) function sha256_add4(first, second, third, fourth) result(sum)
      integer(int64), intent(in) :: first, second, third, fourth

      sum = iand(first + second + third + fourth, MASK32)
   end function sha256_add4

   pure integer(int64) function sha256_add5(first, second, third, fourth, fifth) result(sum)
      integer(int64), intent(in) :: first, second, third, fourth, fifth

      sum = iand(first + second + third + fourth + fifth, MASK32)
   end function sha256_add5

   pure integer(int64) function sha256_rotr(value, shift_count) result(rotated)
      integer(int64), intent(in) :: value
      integer, intent(in) :: shift_count

      rotated = iand(ior(shiftr(value, shift_count), shiftl(value, 32 - shift_count)), MASK32)
   end function sha256_rotr

   pure integer(int64) function sha256_small_sigma0(value) result(result_value)
      integer(int64), intent(in) :: value

      result_value = ieor(ieor(sha256_rotr(value, 7), sha256_rotr(value, 18)), shiftr(value, 3))
   end function sha256_small_sigma0

   pure integer(int64) function sha256_small_sigma1(value) result(result_value)
      integer(int64), intent(in) :: value

      result_value = ieor(ieor(sha256_rotr(value, 17), sha256_rotr(value, 19)), shiftr(value, 10))
   end function sha256_small_sigma1

   pure integer(int64) function sha256_big_sigma0(value) result(result_value)
      integer(int64), intent(in) :: value

      result_value = ieor(ieor(sha256_rotr(value, 2), sha256_rotr(value, 13)), sha256_rotr(value, 22))
   end function sha256_big_sigma0

   pure integer(int64) function sha256_big_sigma1(value) result(result_value)
      integer(int64), intent(in) :: value

      result_value = ieor(ieor(sha256_rotr(value, 6), sha256_rotr(value, 11)), sha256_rotr(value, 25))
   end function sha256_big_sigma1

   pure integer(int64) function sha256_choose(first, second, third) result(result_value)
      integer(int64), intent(in) :: first, second, third

      result_value = ieor(iand(first, second), iand(not(first), third))
   end function sha256_choose

   pure integer(int64) function sha256_majority(first, second, third) result(result_value)
      integer(int64), intent(in) :: first, second, third

      result_value = ieor(ieor(iand(first, second), iand(first, third)), iand(second, third))
   end function sha256_majority

   subroutine sha256_word_to_hex(word, rendered)
      integer(int64), intent(in) :: word
      character(len=8), intent(out) :: rendered
      character(len=*), parameter :: digits = '0123456789abcdef'
      integer :: index, nibble

      do index = 1, 8
         nibble = int(iand(shiftr(word, 4 * (8 - index)), 15_int64))
         rendered(index:index) = digits(nibble + 1:nibble + 1)
      end do
   end subroutine sha256_word_to_hex

end module fortran_integrity
