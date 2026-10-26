! SPDX-License-Identifier: MIT
!==============================================================================
!  hyper_types.f90
!  Bit-packed 16,384-dimensional binary / bipolar Vector Symbolic Algebra.
!
!  Bit convention:
!    bit 0 -> bipolar -1
!    bit 1 -> bipolar +1
!
!  Binding is XOR in the binary representation, which is equivalent to
!  component-wise multiplication in the bipolar representation.
!==============================================================================
module hyper_types
   use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real32, real64
   implicit none
   private

   integer(int32), parameter, public :: WORD_BITS = 64_int32
   integer(int32), parameter, public :: HV_WORDS  = 256_int32
   integer(int32), parameter, public :: HV_BITS   = HV_WORDS * WORD_BITS
   integer(int32), parameter, public :: HV_BYTES  = HV_WORDS * 8_int32

   ! Status codes shared by the low-level algebra API.
   integer(int32), parameter, public :: HV_OK                      = 0_int32
   integer(int32), parameter, public :: HV_ERR_INVALID_ARGUMENT    = -1_int32
   integer(int32), parameter, public :: HV_ERR_ACCUMULATOR_OVERFLOW = -2_int32

   ! A vector occupies exactly HV_WORDS contiguous int64 elements (2 KiB).
   type, public :: hv_t
      integer(int64) :: bits(HV_WORDS) = 0_int64
   end type hv_t

   ! The bit-sliced accumulator stores a binary counter for every HDC bit.  A
   ! 1-bit input vector can therefore be added with word-wise XOR/AND carries
   ! rather than one branch per hypervector dimension.  Eleven planes represent
   ! up to 2,047 context components, which exceeds the tokenizer's largest
   ! possible hybrid context (2 * (1,024 - 1) = 2,046).
   integer(int32), parameter, public :: HV_BITACC_PLANES = 11_int32
   integer(int32), parameter, public :: HV_BITACC_MAX_ITEMS = 2047_int32
   type, public :: hv_bitacc_t
      integer(int64) :: planes(HV_BITACC_PLANES, HV_WORDS) = 0_int64
      integer(int32) :: sample_count = 0_int32
      ! `initialized_planes` makes the bounded reset path safe: callers may
      ! clear only the planes their next bundle can reach, and an add lazily
      ! clears a newly required higher plane before it is observed.  The public
      ! no-argument reset still clears all planes for ordinary API callers.
      integer(int32) :: initialized_planes = HV_BITACC_PLANES
   end type hv_bitacc_t

   ! 64 x 256 signed counters = 32 KiB.  This intentionally fits in a typical
   ! L1 data cache, while the checked add routine prevents int16 wraparound.
   type, public :: hv_acc_t
      integer(int16) :: counts(WORD_BITS, HV_WORDS) = 0_int16
      integer(int32) :: sample_count = 0_int32
   end type hv_acc_t

   public :: hv_zero, hv_from_seed, hv_seed_from_id
   public :: hv_bind, hv_bundle3, hv_permute
   public :: hv_hamming, hv_similarity, hv_similarity_from_distance, hv_hamming_zscore, hv_equal
   public :: hv_acc_reset, hv_acc_add, hv_acc_finalize
   public :: hv_bitacc_reset, hv_bitacc_add, hv_bitacc_add_words
   public :: hv_bitacc_add_permuted_words, hv_bitacc_add_permuted_pair_words
   public :: hv_bitacc_add_permuted_mixed_pair_words, hv_bitacc_finalize
   public :: hv_hebbian_update, hv_hebbian_update_percent, hv_hebbian_update_tenths_percent
   public :: hv_inject_noise_exact, hv_inject_noise_percent

   ! SplitMix64 constants are represented as BOZ values so their bit patterns
   ! are preserved even when a value is negative as a signed Fortran int64.
   integer(int64), parameter :: SM64_GAMMA = int(z'9E3779B97F4A7C15', int64)
   integer(int64), parameter :: SM64_MUL1  = int(z'BF58476D1CE4E5B9', int64)
   integer(int64), parameter :: SM64_MUL2  = int(z'94D049BB133111EB', int64)
   integer(int64), parameter :: ID_SEED_TAG = int(z'D1B54A32D192ED03', int64)
   integer(int64), parameter :: LOW32_MASK = int(z'00000000FFFFFFFF', int64)
   integer(int32), parameter :: ACC_MIN = -32768_int32
   integer(int32), parameter :: ACC_MAX =  32767_int32

contains

   !---------------------------------------------------------------------------
   ! Modular helpers
   !---------------------------------------------------------------------------

   ! Addition modulo 2^64 implemented only with bit operations.  This avoids
   ! relying on signed-integer overflow, which is not portable Fortran.
   pure function add_mod64(a, b) result(sum64)
      integer(int64), intent(in) :: a, b
      integer(int64) :: sum64, carry, x, y

      x = a
      y = b
      do while (y /= 0_int64)
         carry = iand(x, y)
         x = ieor(x, y)
         y = ishft(carry, 1)
      end do
      sum64 = x
   end function add_mod64

   ! Product modulo 2^64 using four base-2^16 limbs.  Every intermediate is
   ! below 2^35, so no signed overflow is used to implement SplitMix64.
   pure function mul_mod64(a, b) result(product)
      integer(int64), intent(in) :: a, b
      integer(int64) :: product
      integer(int64) :: a0, a1, a2, a3, b0, b1, b2, b3
      integer(int64) :: digit0, digit1, digit2, digit3, carry, total
      integer(int64), parameter :: BASE16 = 65536_int64

      a0 = ibits(a,  0, 16)
      a1 = ibits(a, 16, 16)
      a2 = ibits(a, 32, 16)
      a3 = ibits(a, 48, 16)
      b0 = ibits(b,  0, 16)
      b1 = ibits(b, 16, 16)
      b2 = ibits(b, 32, 16)
      b3 = ibits(b, 48, 16)

      total = a0 * b0
      digit0 = modulo(total, BASE16)
      carry = total / BASE16

      total = carry + a0 * b1 + a1 * b0
      digit1 = modulo(total, BASE16)
      carry = total / BASE16

      total = carry + a0 * b2 + a1 * b1 + a2 * b0
      digit2 = modulo(total, BASE16)
      carry = total / BASE16

      ! Higher limbs are intentionally discarded: arithmetic is modulo 2^64.
      total = carry + a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0
      digit3 = modulo(total, BASE16)

      product = ior(ior(ishft(digit0, 0), ishft(digit1, 16)), &
                    ior(ishft(digit2, 32), ishft(digit3, 48)))
   end function mul_mod64

   ! One standards-safe SplitMix64 step.  SplitMix is used only as a seeded
   ! source of reproducible, high-quality random hypervector components.
   pure subroutine splitmix64_next(state, value)
      integer(int64), intent(inout) :: state
      integer(int64), intent(out) :: value
      integer(int64) :: z

      state = add_mod64(state, SM64_GAMMA)
      z = state
      z = mul_mod64(ieor(z, ishft(z, -30)), SM64_MUL1)
      z = mul_mod64(ieor(z, ishft(z, -27)), SM64_MUL2)
      value = ieor(z, ishft(z, -31))
   end subroutine splitmix64_next

   !---------------------------------------------------------------------------
   ! Hypervector construction and core algebra
   !---------------------------------------------------------------------------

   pure function hv_zero() result(res)
      type(hv_t) :: res
      res%bits = 0_int64
   end function hv_zero

   ! Converts every signed int32 identifier injectively to a 64-bit seed before
   ! mixing.  In particular, negative IDs and -1 are valid symbol identifiers.
   pure function hv_seed_from_id(symbol_id) result(seed)
      integer(int32), intent(in) :: symbol_id
      integer(int64) :: seed, low_bits

      low_bits = iand(int(symbol_id, int64), LOW32_MASK)
      seed = ieor(ishftc(low_bits, 17), ID_SEED_TAG)
      seed = ieor(seed, ishftc(low_bits, -11))
   end function hv_seed_from_id

   ! Deterministic 16,384-bit random-looking hypervector generation.
   pure function hv_from_seed(seed) result(res)
      integer(int64), intent(in) :: seed
      type(hv_t) :: res
      integer(int64) :: state
      integer(int32) :: word

      state = seed
      do word = 1, HV_WORDS
         call splitmix64_next(state, res%bits(word))
      end do
   end function hv_from_seed

   ! Binary XOR binding.  It is self-inverse:
   ! hv_bind(a, hv_bind(a, b)) == b.
   pure function hv_bind(a, b) result(c)
      type(hv_t), intent(in) :: a, b
      type(hv_t) :: c
      integer(int32) :: word

      !$omp simd
      do word = 1, HV_WORDS
         c%bits(word) = ieor(a%bits(word), b%bits(word))
      end do
   end function hv_bind

   ! Bit-parallel majority gate for a three-item bundle.  Odd cardinality means
   ! every bit has a deterministic majority and no tie mask is needed.
   pure function hv_bundle3(a, b, d) result(c)
      type(hv_t), intent(in) :: a, b, d
      type(hv_t) :: c
      integer(int32) :: word

      !$omp simd
      do word = 1, HV_WORDS
         c%bits(word) = ior(ior(iand(a%bits(word), b%bits(word)), &
                             iand(b%bits(word), d%bits(word))), &
                             iand(a%bits(word), d%bits(word)))
      end do
   end function hv_bundle3

   ! Circularly rotates the complete 16,384-bit bitstring.  A positive shift
   ! moves a bit from global position p to p + shift modulo HV_BITS; a negative
   ! shift is therefore its exact inverse.
   pure function hv_permute(v, shift_bits) result(res)
      type(hv_t), intent(in) :: v
      integer(int32), intent(in) :: shift_bits
      type(hv_t) :: res
      integer(int32) :: total_shift, word_shift, bit_shift
      integer(int32) :: word, src_hi, src_lo

      total_shift = modulo(shift_bits, HV_BITS)
      if (total_shift == 0_int32) then
         res = v
         return
      end if

      word_shift = total_shift / WORD_BITS
      bit_shift = modulo(total_shift, WORD_BITS)

      if (bit_shift == 0_int32) then
         !$omp simd
         do word = 1, HV_WORDS
            src_hi = word - word_shift
            if (src_hi <= 0_int32) src_hi = src_hi + HV_WORDS
            res%bits(word) = v%bits(src_hi)
         end do
      else
         do word = 1, HV_WORDS
            src_hi = word - word_shift
            if (src_hi <= 0_int32) src_hi = src_hi + HV_WORDS
            src_lo = src_hi - 1_int32
            if (src_lo <= 0_int32) src_lo = src_lo + HV_WORDS
            res%bits(word) = ior(ishft(v%bits(src_hi), bit_shift), &
                                  ishft(v%bits(src_lo), bit_shift - WORD_BITS))
         end do
      end if
   end function hv_permute

   pure function hv_hamming(a, b) result(distance)
      type(hv_t), intent(in) :: a, b
      integer(int32) :: distance
      integer(int32) :: word

      distance = 0_int32
      !$omp simd reduction(+:distance)
      do word = 1, HV_WORDS
         distance = distance + popcnt(ieor(a%bits(word), b%bits(word)))
      end do
   end function hv_hamming

   ! Maps a valid Hamming distance to the bipolar dot-product normalization.
   ! HV_BITS is 2**14, so every in-range result k / 8192 is represented exactly
   ! in binary real32.  Keeping this mapping in one primitive prevents subtle
   ! divergence between cleanup, dialogue, resonator, and pretraining reports.
   pure function hv_similarity_from_distance(distance) result(similarity)
      integer(int32), intent(in) :: distance
      real(real32) :: similarity

      similarity = 1.0_real32 - 2.0_real32 * real(distance, real32) / real(HV_BITS, real32)
   end function hv_similarity_from_distance

   ! Maps Hamming distance to the bipolar dot-product normalization [-1, +1].
   pure function hv_similarity(a, b) result(similarity)
      type(hv_t), intent(in) :: a, b
      real(real32) :: similarity

      similarity = hv_similarity_from_distance(hv_hamming(a, b))
   end function hv_similarity

   ! Number of standard deviations from the expected D/2 distance of two
   ! independent random binary hypervectors.  Useful for diagnostics/tests.
   pure function hv_hamming_zscore(a, b) result(zscore)
      type(hv_t), intent(in) :: a, b
      real(real64) :: zscore
      real(real64), parameter :: MEAN_DISTANCE = real(HV_BITS, real64) / 2.0_real64
      real(real64), parameter :: STD_DISTANCE = sqrt(real(HV_BITS, real64) / 4.0_real64)

      zscore = (real(hv_hamming(a, b), real64) - MEAN_DISTANCE) / STD_DISTANCE
   end function hv_hamming_zscore

   pure function hv_equal(a, b) result(is_equal)
      type(hv_t), intent(in) :: a, b
      logical :: is_equal
      integer(int32) :: word

      is_equal = .true.
      do word = 1, HV_WORDS
         if (a%bits(word) /= b%bits(word)) then
            is_equal = .false.
            return
         end if
      end do
   end function hv_equal

   !---------------------------------------------------------------------------
   ! Checked Hebbian superposition / bundling
   !---------------------------------------------------------------------------

   pure subroutine hv_acc_reset(acc)
      type(hv_acc_t), intent(inout) :: acc
      acc%counts = 0_int16
      acc%sample_count = 0_int32
   end subroutine hv_acc_reset

   ! Adds a bipolar vector with an optional signed integer weight.  The entire
   ! update is preflighted before mutation, making overflow failure atomic.
   pure subroutine hv_acc_add(acc, v, status, weight)
      type(hv_acc_t), intent(inout) :: acc
      type(hv_t), intent(in) :: v
      integer(int32), intent(out) :: status
      integer(int16), intent(in), optional :: weight
      integer(int32) :: word, bit, w32, delta, candidate

      status = HV_OK
      if (acc%sample_count == huge(acc%sample_count)) then
         status = HV_ERR_ACCUMULATOR_OVERFLOW
         return
      end if

      w32 = 1_int32
      if (present(weight)) w32 = int(weight, int32)

      ! Preflight all 16,384 counters so an error cannot leave a partial add.
      do word = 1, HV_WORDS
         do bit = 0, WORD_BITS - 1
            if (btest(v%bits(word), bit)) then
               delta = w32
            else
               delta = -w32
            end if
            candidate = int(acc%counts(bit + 1, word), int32) + delta
            if (candidate < ACC_MIN .or. candidate > ACC_MAX) then
               status = HV_ERR_ACCUMULATOR_OVERFLOW
               return
            end if
         end do
      end do

      do word = 1, HV_WORDS
         do bit = 0, WORD_BITS - 1
            if (btest(v%bits(word), bit)) then
               delta = w32
            else
               delta = -w32
            end if
            candidate = int(acc%counts(bit + 1, word), int32) + delta
            acc%counts(bit + 1, word) = int(candidate, int16)
         end do
      end do
      acc%sample_count = acc%sample_count + 1_int32
   end subroutine hv_acc_add

   ! Thresholds signed Hebbian counters.  Equal votes are resolved by a seeded
   ! independent hypervector, making even-cardinality bundles reproducible.
   pure function hv_acc_finalize(acc, tie_seed) result(res)
      type(hv_acc_t), intent(in) :: acc
      integer(int64), intent(in), optional :: tie_seed
      type(hv_t) :: res, tie_breaker
      integer(int64) :: seed, out_word
      integer(int32) :: word, bit

      if (acc%sample_count == 0_int32) then
         res = hv_zero()
         return
      end if

      seed = int(z'000000000466F45D', int64)
      if (present(tie_seed)) seed = tie_seed
      tie_breaker = hv_from_seed(seed)

      do word = 1, HV_WORDS
         out_word = 0_int64
         do bit = 0, WORD_BITS - 1
            if (acc%counts(bit + 1, word) > 0_int16) then
               out_word = ibset(out_word, bit)
            else if (acc%counts(bit + 1, word) == 0_int16) then
               if (btest(tie_breaker%bits(word), bit)) out_word = ibset(out_word, bit)
            end if
         end do
         res%bits(word) = out_word
      end do
   end function hv_acc_finalize

   !---------------------------------------------------------------------------
   ! Bit-sliced majority accumulator
   !---------------------------------------------------------------------------

   ! A streaming, word-parallel equivalent of an unsigned per-bit population
   ! counter.  It is intended for bounded context bundling where every input has
   ! unit weight; weighted Hebbian accumulation continues to use hv_acc_t.
   !
   ! Context construction normally knows its maximum item count before the
   ! first add.  `maximum_items` lets that hot path clear only the population
   ! planes it can reach instead of writing all eleven 2 KiB planes for every
   ! small center context.  The no-argument public contract remains a complete
   ! reset.  Later adds safely initialize a newly needed plane if a caller's
   ! bound was conservative or underestimated.
   pure subroutine hv_bitacc_reset(acc, maximum_items)
      type(hv_bitacc_t), intent(inout) :: acc
      integer(int32), intent(in), optional :: maximum_items
      integer(int32) :: planes_to_clear

      planes_to_clear = HV_BITACC_PLANES
      if (present(maximum_items)) then
         if (maximum_items > 0_int32 .and. maximum_items <= HV_BITACC_MAX_ITEMS) then
            planes_to_clear = hv_bitacc_required_planes(maximum_items)
         end if
      end if
      acc%planes(1:planes_to_clear, :) = 0_int64
      acc%sample_count = 0_int32
      acc%initialized_planes = planes_to_clear
   end subroutine hv_bitacc_reset

   pure function hv_bitacc_required_planes(item_count) result(plane_count)
      integer(int32), intent(in) :: item_count
      integer(int32) :: plane_count, remaining

      plane_count = 0_int32
      if (item_count <= 0_int32) return
      remaining = item_count
      do while (remaining > 0_int32)
         plane_count = plane_count + 1_int32
         remaining = remaining / 2_int32
      end do
   end function hv_bitacc_required_planes

   pure subroutine hv_bitacc_ensure_planes(acc, required_planes)
      type(hv_bitacc_t), intent(inout) :: acc
      integer(int32), intent(in) :: required_planes

      if (required_planes <= acc%initialized_planes) return
      if (required_planes > HV_BITACC_PLANES) return
      acc%planes(acc%initialized_planes + 1_int32:required_planes, :) = 0_int64
      acc%initialized_planes = required_planes
   end subroutine hv_bitacc_ensure_planes

   ! Adds a binary hypervector to each independent bit counter.  The carry
   ! chain is a ripple adder over bit planes, but each plane contains 64 HDC
   ! dimensions at once.  The raw-word entry point lets high-throughput callers
   ! consume a column of a structure-of-arrays model without creating a 2 KiB
   ! temporary hv_t.
   pure subroutine hv_bitacc_add(acc, vector, status)
      type(hv_bitacc_t), intent(inout) :: acc
      type(hv_t), intent(in) :: vector
      integer(int32), intent(out) :: status

      call hv_bitacc_add_words(acc, vector%bits, status)
   end subroutine hv_bitacc_add

   pure subroutine hv_bitacc_add_words(acc, vector_words, status)
      type(hv_bitacc_t), intent(inout) :: acc
      integer(int64), intent(in) :: vector_words(HV_WORDS)
      integer(int32), intent(out) :: status
      integer(int64) :: carry, previous
      integer(int32) :: word, plane, required_planes

      status = HV_OK
      if (acc%sample_count >= HV_BITACC_MAX_ITEMS) then
         status = HV_ERR_ACCUMULATOR_OVERFLOW
         return
      end if
      required_planes = hv_bitacc_required_planes(acc%sample_count + 1_int32)
      call hv_bitacc_ensure_planes(acc, required_planes)

      do word = 1, HV_WORDS
         carry = vector_words(word)
         do plane = 1, required_planes
            previous = acc%planes(plane, word)
            acc%planes(plane, word) = ieor(previous, carry)
            carry = iand(previous, carry)
            ! Once a word's carry is zero, untouched higher planes are already
            ! correct.  This is especially common for small local contexts.
            if (carry == 0_int64) exit
         end do
      end do
      acc%sample_count = acc%sample_count + 1_int32
   end subroutine hv_bitacc_add_words

   ! Fuses hv_permute(vector, shift_bits) with one bit-sliced add.  It is
   ! deliberately bit-identical to the two public primitives, but avoids an
   ! intermediate hypervector allocation/copy for each ordered context edge.
   pure subroutine hv_bitacc_add_permuted_words(acc, vector_words, shift_bits, status)
      type(hv_bitacc_t), intent(inout) :: acc
      integer(int64), intent(in) :: vector_words(HV_WORDS)
      integer(int32), intent(in) :: shift_bits
      integer(int32), intent(out) :: status
      integer(int64) :: carry, previous, permuted_word
      integer(int32) :: total_shift, word_shift, bit_shift, word, plane, source_hi, source_lo, required_planes

      status = HV_OK
      if (acc%sample_count >= HV_BITACC_MAX_ITEMS) then
         status = HV_ERR_ACCUMULATOR_OVERFLOW
         return
      end if
      required_planes = hv_bitacc_required_planes(acc%sample_count + 1_int32)
      call hv_bitacc_ensure_planes(acc, required_planes)

      total_shift = modulo(shift_bits, HV_BITS)
      word_shift = total_shift / WORD_BITS
      bit_shift = modulo(total_shift, WORD_BITS)
      do word = 1, HV_WORDS
         ! Equivalent to modulo(word - 1 - word_shift, HV_WORDS) + 1,
         ! without an integer division in this inner packed-HV loop.
         source_hi = word - word_shift
         if (source_hi <= 0_int32) source_hi = source_hi + HV_WORDS
         if (bit_shift == 0_int32) then
            permuted_word = vector_words(source_hi)
         else
            source_lo = source_hi - 1_int32
            if (source_lo <= 0_int32) source_lo = source_lo + HV_WORDS
            permuted_word = ior(ishft(vector_words(source_hi), bit_shift), &
                                ishft(vector_words(source_lo), bit_shift - WORD_BITS))
         end if
         carry = permuted_word
         do plane = 1, required_planes
            previous = acc%planes(plane, word)
            acc%planes(plane, word) = ieor(previous, carry)
            carry = iand(previous, carry)
            if (carry == 0_int64) exit
         end do
      end do
      acc%sample_count = acc%sample_count + 1_int32
   end subroutine hv_bitacc_add_permuted_words

   ! Adds a position-permuted vector and its unpermuted identity view in one
   ! binary-counter pass.  For plane zero the carry is the three-input majority
   ! of old_count, permuted bit and topical bit; higher planes receive that
   ! carry normally.  The result is exactly equivalent to two sequential adds.
   pure subroutine hv_bitacc_add_permuted_pair_words(acc, vector_words, shift_bits, status)
      type(hv_bitacc_t), intent(inout) :: acc
      integer(int64), intent(in) :: vector_words(HV_WORDS)
      integer(int32), intent(in) :: shift_bits
      integer(int32), intent(out) :: status
      integer(int64) :: carry, previous, permuted_word, topical_word
      integer(int32) :: total_shift, word_shift, bit_shift, word, plane, source_hi, source_lo, required_planes

      status = HV_OK
      if (acc%sample_count > HV_BITACC_MAX_ITEMS - 2_int32) then
         status = HV_ERR_ACCUMULATOR_OVERFLOW
         return
      end if
      required_planes = hv_bitacc_required_planes(acc%sample_count + 2_int32)
      call hv_bitacc_ensure_planes(acc, required_planes)

      total_shift = modulo(shift_bits, HV_BITS)
      word_shift = total_shift / WORD_BITS
      bit_shift = modulo(total_shift, WORD_BITS)
      do word = 1, HV_WORDS
         source_hi = word - word_shift
         if (source_hi <= 0_int32) source_hi = source_hi + HV_WORDS
         if (bit_shift == 0_int32) then
            permuted_word = vector_words(source_hi)
         else
            source_lo = source_hi - 1_int32
            if (source_lo <= 0_int32) source_lo = source_lo + HV_WORDS
            permuted_word = ior(ishft(vector_words(source_hi), bit_shift), &
                                ishft(vector_words(source_lo), bit_shift - WORD_BITS))
         end if
         topical_word = vector_words(word)
         previous = acc%planes(1, word)
         acc%planes(1, word) = ieor(ieor(previous, permuted_word), topical_word)
         carry = ior(iand(previous, permuted_word), &
                     ior(iand(previous, topical_word), iand(permuted_word, topical_word)))
         do plane = 2, required_planes
            previous = acc%planes(plane, word)
            acc%planes(plane, word) = ieor(previous, carry)
            carry = iand(previous, carry)
            if (carry == 0_int64) exit
         end do
      end do
      acc%sample_count = acc%sample_count + 2_int32
   end subroutine hv_bitacc_add_permuted_pair_words

   ! Adds a position-permuted immutable identity vector and a separately
   ! supplied unpermuted semantic vector in one bit-sliced counter pass.  This
   ! is exactly equivalent to `add(permuted(position_words)); add(topic_words)`
   ! but avoids a temporary and second carry traversal.  Pretraining uses it
   ! for epoch-frozen semantic feedback: position remains an immutable token
   ! identity, while the topical channel can carry a learned snapshot without
   ! introducing a read-after-write dependency between source-ordered updates.
   pure subroutine hv_bitacc_add_permuted_mixed_pair_words(acc, position_words, topic_words, shift_bits, status)
      type(hv_bitacc_t), intent(inout) :: acc
      integer(int64), intent(in) :: position_words(HV_WORDS), topic_words(HV_WORDS)
      integer(int32), intent(in) :: shift_bits
      integer(int32), intent(out) :: status
      integer(int64) :: carry, previous, permuted_word, topical_word
      integer(int32) :: total_shift, word_shift, bit_shift, word, plane, source_hi, source_lo, required_planes

      status = HV_OK
      if (acc%sample_count > HV_BITACC_MAX_ITEMS - 2_int32) then
         status = HV_ERR_ACCUMULATOR_OVERFLOW
         return
      end if
      required_planes = hv_bitacc_required_planes(acc%sample_count + 2_int32)
      call hv_bitacc_ensure_planes(acc, required_planes)

      total_shift = modulo(shift_bits, HV_BITS)
      word_shift = total_shift / WORD_BITS
      bit_shift = modulo(total_shift, WORD_BITS)
      do word = 1, HV_WORDS
         source_hi = word - word_shift
         if (source_hi <= 0_int32) source_hi = source_hi + HV_WORDS
         if (bit_shift == 0_int32) then
            permuted_word = position_words(source_hi)
         else
            source_lo = source_hi - 1_int32
            if (source_lo <= 0_int32) source_lo = source_lo + HV_WORDS
            permuted_word = ior(ishft(position_words(source_hi), bit_shift), &
                                ishft(position_words(source_lo), bit_shift - WORD_BITS))
         end if
         topical_word = topic_words(word)
         previous = acc%planes(1, word)
         acc%planes(1, word) = ieor(ieor(previous, permuted_word), topical_word)
         carry = ior(iand(previous, permuted_word), &
                     ior(iand(previous, topical_word), iand(permuted_word, topical_word)))
         do plane = 2, required_planes
            previous = acc%planes(plane, word)
            acc%planes(plane, word) = ieor(previous, carry)
            carry = iand(previous, carry)
            if (carry == 0_int64) exit
         end do
      end do
      acc%sample_count = acc%sample_count + 2_int32
   end subroutine hv_bitacc_add_permuted_mixed_pair_words

   ! Thresholds bit-sliced binary population counters without unpacking any of
   ! the 16,384 dimensions.  For even cardinalities equality with N/2 gets the
   ! same seeded tie breaker as hv_acc_finalize; for odd cardinalities strictly
   ! greater than floor(N/2) is the ordinary majority.  Thus unit-weight output
   ! is bit-identical to the reference signed accumulator.
   pure function hv_bitacc_finalize(acc, tie_seed) result(res)
      type(hv_bitacc_t), intent(in) :: acc
      integer(int64), intent(in), optional :: tie_seed
      type(hv_t) :: res, tie_breaker
      integer(int64) :: seed, equal_mask, greater_mask
      integer(int32) :: word, plane, threshold, required_planes
      logical :: needs_tie_breaker

      if (acc%sample_count == 0_int32) then
         res = hv_zero()
         return
      end if

      required_planes = hv_bitacc_required_planes(acc%sample_count)
      needs_tie_breaker = modulo(acc%sample_count, 2_int32) == 0_int32
      if (needs_tie_breaker) then
         seed = int(z'000000000466F45D', int64)
         if (present(tie_seed)) seed = tie_seed
         tie_breaker = hv_from_seed(seed)
      end if
      threshold = acc%sample_count / 2_int32

      do word = 1, HV_WORDS
         equal_mask = not(0_int64)
         greater_mask = 0_int64
         do plane = required_planes, 1, -1
            if (btest(threshold, plane - 1_int32)) then
               equal_mask = iand(equal_mask, acc%planes(plane, word))
            else
               greater_mask = ior(greater_mask, iand(equal_mask, acc%planes(plane, word)))
               equal_mask = iand(equal_mask, not(acc%planes(plane, word)))
            end if
         end do
         if (needs_tie_breaker) then
            res%bits(word) = ior(greater_mask, iand(equal_mask, tie_breaker%bits(word)))
         else
            res%bits(word) = greater_mask
         end if
      end do
   end function hv_bitacc_finalize

   ! Plastic bit replacement.  The AND of two independent SplitMix masks
   ! selects approximately 25 percent of dimensions without floating point.
   pure function hv_hebbian_update(prototype, new_state, plasticity_seed) result(updated)
      type(hv_t), intent(in) :: prototype, new_state
      integer(int64), intent(in) :: plasticity_seed
      type(hv_t) :: updated, mask_a, mask_b, plastic_mask
      integer(int32) :: word

      mask_a = hv_from_seed(plasticity_seed)
      mask_b = hv_from_seed(not(plasticity_seed))
      !$omp simd
      do word = 1, HV_WORDS
         plastic_mask%bits(word) = iand(mask_a%bits(word), mask_b%bits(word))
         updated%bits(word) = ior(iand(prototype%bits(word), not(plastic_mask%bits(word))), &
                                  iand(new_state%bits(word), plastic_mask%bits(word)))
      end do
   end function hv_hebbian_update

   ! Exact-rate plasticity for streaming learners.  Unlike hv_hebbian_update,
   ! this routine replaces an exact percentage of dimensions, which makes a
   ! learning-rate schedule reproducible and independently auditable.
   pure subroutine hv_hebbian_update_percent(prototype, new_state, plasticity_percent, &
                                             plasticity_seed, updated, status, legacy_selection_mask)
      type(hv_t), intent(in) :: prototype, new_state
      integer(int32), intent(in) :: plasticity_percent
      integer(int64), intent(in) :: plasticity_seed
      type(hv_t), intent(out) :: updated
      integer(int32), intent(out) :: status
      logical, intent(in), optional :: legacy_selection_mask

      updated = prototype
      status = HV_OK
      if (plasticity_percent < 0_int32 .or. plasticity_percent > 100_int32) then
         status = HV_ERR_INVALID_ARGUMENT
         return
      end if
      call hv_hebbian_update_tenths_percent(prototype, new_state, 10_int32 * plasticity_percent, &
                                            plasticity_seed, updated, status, legacy_selection_mask)
   end subroutine hv_hebbian_update_percent

   ! Fixed-point exact-rate plasticity. `plasticity_tenths_percent=6` means
   ! 0.6%; integer arithmetic preserves deterministic tie-breaking and allows
   ! sub-percent schedules without binary floating-point drift.
   pure subroutine hv_hebbian_update_tenths_percent(prototype, new_state, plasticity_tenths_percent, &
                                                    plasticity_seed, updated, status, legacy_selection_mask)
      type(hv_t), intent(in) :: prototype, new_state
      integer(int32), intent(in) :: plasticity_tenths_percent
      integer(int64), intent(in) :: plasticity_seed
      type(hv_t), intent(out) :: updated
      integer(int32), intent(out) :: status
      logical, intent(in), optional :: legacy_selection_mask
      type(hv_t) :: plastic_mask
      integer(int32) :: flip_count, word
      integer(int64) :: rounded_flip_count
      logical :: use_legacy_selection_mask

      updated = prototype
      status = HV_OK
      if (plasticity_tenths_percent < 0_int32 .or. plasticity_tenths_percent > 1000_int32) then
         status = HV_ERR_INVALID_ARGUMENT
         return
      end if
      rounded_flip_count = (int(HV_BITS, int64) * int(plasticity_tenths_percent, int64) + 500_int64) / 1000_int64
      flip_count = int(rounded_flip_count, int32)
      use_legacy_selection_mask = .false.
      if (present(legacy_selection_mask)) use_legacy_selection_mask = legacy_selection_mask
      if (use_legacy_selection_mask) then
         call hv_exact_selection_mask_legacy(flip_count, plasticity_seed, plastic_mask, status)
      else
         call hv_exact_selection_mask(flip_count, plasticity_seed, plastic_mask, status)
      end if
      if (status /= HV_OK) return

      !$omp simd
      do word = 1, HV_WORDS
         updated%bits(word) = ior(iand(prototype%bits(word), not(plastic_mask%bits(word))), &
                                  iand(new_state%bits(word), plastic_mask%bits(word)))
      end do
   end subroutine hv_hebbian_update_tenths_percent

   ! Draws a uniformly bounded position from the low 31 bits of one SplitMix
   ! word.  Rejection removes modulo bias for every bound up to HV_BITS while
   ! retaining only signed-int64-safe arithmetic.  SplitMix's deterministic
   ! stream makes this reproducible for a fixed selection seed.
   pure subroutine hv_uniform_position(state, inclusive_upper, position)
      integer(int64), intent(inout) :: state
      integer(int32), intent(in) :: inclusive_upper
      integer(int32), intent(out) :: position
      integer(int64), parameter :: RANDOM31_DOMAIN = 2147483648_int64
      integer(int64) :: random_word, draw, range, accepted_range

      range = int(inclusive_upper, int64) + 1_int64
      accepted_range = (RANDOM31_DOMAIN / range) * range
      do
         call splitmix64_next(state, random_word)
         draw = ibits(random_word, 0, 31)
         if (draw < accepted_range) exit
      end do
      position = int(modulo(draw, range), int32)
   end subroutine hv_uniform_position

   ! Produces an exact-cardinality binary mask without heap allocation.  Floyd
   ! sampling chooses an unbiased subset of the smaller side of the requested
   ! set; for rates above 50 percent it samples the excluded complement and
   ! flips that mask.  This avoids the old high-rate coupon-collector fallback,
   ! whose deterministic low-bit fill skewed masks toward early dimensions.
   pure subroutine hv_exact_selection_mask(selected_bits, selection_seed, mask, status)
      integer(int32), intent(in) :: selected_bits
      integer(int64), intent(in) :: selection_seed
      type(hv_t), intent(out) :: mask
      integer(int32), intent(out) :: status
      type(hv_t) :: working_mask
      integer(int64) :: state
      integer(int32) :: draw_count, position, drawn_position, selected_position, word, bit
      logical :: select_complement

      mask = hv_zero()
      status = HV_OK
      if (selected_bits < 0_int32 .or. selected_bits > HV_BITS) then
         status = HV_ERR_INVALID_ARGUMENT
         return
      end if

      select_complement = selected_bits > HV_BITS / 2_int32
      if (select_complement) then
         draw_count = HV_BITS - selected_bits
      else
         draw_count = selected_bits
      end if
      if (draw_count == 0_int32) then
         if (select_complement) mask%bits = not(0_int64)
         return
      end if

      working_mask = hv_zero()
      state = selection_seed
      do position = HV_BITS - draw_count, HV_BITS - 1_int32
         call hv_uniform_position(state, position, drawn_position)
         selected_position = drawn_position
         word = ishft(selected_position, -6) + 1_int32
         bit = iand(selected_position, WORD_BITS - 1_int32)
         if (btest(working_mask%bits(word), bit)) selected_position = position
         word = ishft(selected_position, -6) + 1_int32
         bit = iand(selected_position, WORD_BITS - 1_int32)
         working_mask%bits(word) = ibset(working_mask%bits(word), bit)
      end do

      if (select_complement) then
         mask%bits = not(working_mask%bits)
      else
         mask = working_mask
      end if
   end subroutine hv_exact_selection_mask

   ! Reproduces the v4-v6 selection stream solely for checkpoint continuation.
   ! Do not use this for new work: at high densities its bounded retry fallback
   ! completes the subset in ascending position order and therefore is not an
   ! unbiased sample. Keeping it private confines that historical contract to
   ! the explicit compatibility path in hdc_pretraining.
   pure subroutine hv_exact_selection_mask_legacy(selected_bits, selection_seed, mask, status)
      integer(int32), intent(in) :: selected_bits
      integer(int64), intent(in) :: selection_seed
      type(hv_t), intent(out) :: mask
      integer(int32), intent(out) :: status
      integer(int64) :: state, random_word
      integer(int32) :: selected, position, word, bit, attempts

      mask = hv_zero()
      status = HV_OK
      if (selected_bits < 0_int32 .or. selected_bits > HV_BITS) then
         status = HV_ERR_INVALID_ARGUMENT
         return
      end if
      if (selected_bits == 0_int32) return

      state = selection_seed
      selected = 0_int32
      attempts = 0_int32
      do while (selected < selected_bits)
         call splitmix64_next(state, random_word)
         position = int(iand(random_word, int(HV_BITS - 1_int32, int64)), int32)
         word = ishft(position, -6) + 1_int32
         bit = iand(position, WORD_BITS - 1_int32)
         if (.not. btest(mask%bits(word), bit)) then
            mask%bits(word) = ibset(mask%bits(word), bit)
            selected = selected + 1_int32
         end if

         attempts = attempts + 1_int32
         if (attempts >= HV_BITS * 16_int32 .and. selected < selected_bits) then
            do position = 0, HV_BITS - 1_int32
               word = ishft(position, -6) + 1_int32
               bit = iand(position, WORD_BITS - 1_int32)
               if (.not. btest(mask%bits(word), bit)) then
                  mask%bits(word) = ibset(mask%bits(word), bit)
                  selected = selected + 1_int32
                  if (selected == selected_bits) exit
               end if
            end do
            exit
         end if
      end do
   end subroutine hv_exact_selection_mask_legacy

   !---------------------------------------------------------------------------
   ! Exact deterministic noise injection for resilience tests
   !---------------------------------------------------------------------------

   ! Flips exactly flip_count distinct bit positions.  Invalid counts return
   ! HV_ERR_INVALID_ARGUMENT and leave noisy equal to v.
   pure subroutine hv_inject_noise_exact(v, flip_count, noise_seed, noisy, status)
      type(hv_t), intent(in) :: v
      integer(int32), intent(in) :: flip_count
      integer(int64), intent(in) :: noise_seed
      type(hv_t), intent(out) :: noisy
      integer(int32), intent(out) :: status
      type(hv_t) :: flip_mask

      noisy = v
      call hv_exact_selection_mask(flip_count, noise_seed, flip_mask, status)
      if (status /= HV_OK) return
      noisy = hv_bind(v, flip_mask)
   end subroutine hv_inject_noise_exact

   ! Converts an integral percentage to its nearest exact bit count.  For
   ! example, 25 percent maps to exactly 4,096 flips at D = 16,384.
   pure subroutine hv_inject_noise_percent(v, percent, noise_seed, noisy, status)
      type(hv_t), intent(in) :: v
      integer(int32), intent(in) :: percent
      integer(int64), intent(in) :: noise_seed
      type(hv_t), intent(out) :: noisy
      integer(int32), intent(out) :: status
      integer(int32) :: flip_count

      noisy = v
      status = HV_OK
      if (percent < 0_int32 .or. percent > 100_int32) then
         status = HV_ERR_INVALID_ARGUMENT
         return
      end if
      flip_count = (HV_BITS * percent + 50_int32) / 100_int32
      call hv_inject_noise_exact(v, flip_count, noise_seed, noisy, status)
   end subroutine hv_inject_noise_percent

end module hyper_types
