! SPDX-License-Identifier: MIT
!==============================================================================
!  fixed_budget_backward.f90
!
!  Exact gradients of one step of fixed_budget_context, for every learnable
!  array in fbc_model_t: the tied embedding, the three head projections, the
!  attention output projection, both RMSNorm scales, the three FFN matrices, the
!  relative-distance bias, the multi-scale readout weights, the channel-mixing
!  logits, all four slot projections (query, write key, write value, read
!  output) and the slot salience weight.
!
!  Scope of the truncation, stated exactly
!  --------------------------------------
!  The recurrent state (retention numerator/normaliser, slot key/value/salience,
!  local ring buffer) is differentiated within the current step only:
!
!      L = sum_t loss_t(parameters ; state before t held constant)
!
!  so the term carrying a gradient back through state_t -> state_{t-1} is
!  deliberately dropped (bounded truncated BPTT, depth 1).  Consequences,
!  intended and exercised by the test suite:
!
!    * keys/values of *earlier* tokens receive no gradient; the current query
!      still receives one, because it is scored against them;
!    * both *writes* that happen before a read inside the same step are fully
!      trained: the slot write (key, value, gate, salience) and the multi-scale
!      retention update (phi(k), v).  Only the previous steps' writes are frozen.
!
!  For v6 models this is the derivative of the implemented computation with
!  pre-step state frozen. A full local ring requires fbc_step's deferred write,
!  followed by fbc_finish_local_step after backward. V5 checkpoints deliberately
!  retain their historical, inexact full-ring/clamped-decay training rule.
!==============================================================================
module fixed_budget_backward
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use fixed_budget_context
   implicit none (type, external)
   private

   !> One gradient array per learnable array, mirroring fbc_model_t.
   type, public :: fbc_gradient_t
      logical :: initialized = .false.
      real(real64), allocatable :: token_embedding(:, :)
      real(real64), allocatable :: output_bias(:)
      real(real64), allocatable :: query(:, :, :)
      real(real64), allocatable :: key(:, :, :)
      real(real64), allocatable :: value(:, :, :)
      real(real64), allocatable :: attention_output(:, :)
      real(real64), allocatable :: attention_rms(:)
      real(real64), allocatable :: ffn_rms(:)
      real(real64), allocatable :: ffn_gate(:, :)
      real(real64), allocatable :: ffn_value(:, :)
      real(real64), allocatable :: ffn_output(:, :)
      real(real64), allocatable :: distance_bias(:)
      real(real64), allocatable :: scale_logit(:)
      real(real64), allocatable :: decay_logit(:)
      real(real64), allocatable :: write_strength_logit(:)
      real(real64), allocatable :: channel_logit(:)
      real(real64), allocatable :: channel_gate_proj(:, :)
      real(real64), allocatable :: channel_router(:, :)
      real(real64), allocatable :: induction_mix(:)
      real(real64), allocatable :: scale_gate_proj(:, :)
      real(real64), allocatable :: slot_query_proj(:, :)
      real(real64), allocatable :: slot_key_proj(:, :)
      real(real64), allocatable :: slot_value_proj(:, :)
      real(real64), allocatable :: slot_read_proj(:, :)
      real(real64) :: slot_salience_weight = 0.0_real64
      integer(int32) :: layout(7) = 0_int32
   end type fbc_gradient_t

   public :: fbc_gradient_init, fbc_gradient_destroy, fbc_gradient_zero, fbc_gradient_norm
   public :: fbc_step_backward, fbc_outer

contains

   subroutine fbc_gradient_init(gradient, model, status)
      type(fbc_gradient_t), intent(inout) :: gradient
      type(fbc_model_t), intent(in) :: model
      integer(int32), intent(out) :: status
      integer(int32) :: allocation_status

      call fbc_gradient_destroy(gradient)
      status = FBC_OK
      if (.not. fbc_model_is_ready(model)) then
         status = FBC_ERR_NOT_READY
         return
      end if
      allocate(gradient%token_embedding(size(model%token_embedding, 1), size(model%token_embedding, 2)), &
               gradient%output_bias(size(model%output_bias)), &
               gradient%query(size(model%query, 1), size(model%query, 2), size(model%query, 3)), &
               gradient%key(size(model%key, 1), size(model%key, 2), size(model%key, 3)), &
               gradient%value(size(model%value, 1), size(model%value, 2), size(model%value, 3)), &
               gradient%attention_output(size(model%attention_output, 1), size(model%attention_output, 2)), &
               gradient%attention_rms(size(model%attention_rms)), &
               gradient%ffn_rms(size(model%ffn_rms)), &
               gradient%ffn_gate(size(model%ffn_gate, 1), size(model%ffn_gate, 2)), &
               gradient%ffn_value(size(model%ffn_value, 1), size(model%ffn_value, 2)), &
               gradient%ffn_output(size(model%ffn_output, 1), size(model%ffn_output, 2)), &
               gradient%distance_bias(size(model%distance_bias)), &
               gradient%scale_logit(size(model%scale_logit)), &
               gradient%decay_logit(size(model%decay_logit)), &
               gradient%write_strength_logit(size(model%write_strength_logit)), &
               gradient%channel_logit(size(model%channel_logit)), &
               gradient%channel_gate_proj(size(model%channel_gate_proj, 1), &
                                          size(model%channel_gate_proj, 2)), &
               gradient%channel_router(size(model%channel_router, 1), size(model%channel_router, 2)), &
               gradient%induction_mix(size(model%induction_mix)), &
               gradient%scale_gate_proj(size(model%scale_gate_proj, 1), &
                                        size(model%scale_gate_proj, 2)), &
               gradient%slot_query_proj(size(model%slot_query_proj, 1), size(model%slot_query_proj, 2)), &
               gradient%slot_key_proj(size(model%slot_key_proj, 1), size(model%slot_key_proj, 2)), &
               gradient%slot_value_proj(size(model%slot_value_proj, 1), size(model%slot_value_proj, 2)), &
               gradient%slot_read_proj(size(model%slot_read_proj, 1), size(model%slot_read_proj, 2)), &
               stat=allocation_status)
      if (allocation_status /= 0) then
         status = FBC_ERR_ALLOCATION
         call fbc_gradient_destroy(gradient)
         return
      end if
      gradient%layout = fbc_config_layout(model%config)
      gradient%initialized = .true.
      call fbc_gradient_zero(gradient)
   end subroutine fbc_gradient_init

   subroutine fbc_gradient_zero(gradient)
      type(fbc_gradient_t), intent(inout) :: gradient

      if (.not. gradient%initialized) return
      gradient%token_embedding = 0.0_real64
      gradient%output_bias = 0.0_real64
      gradient%query = 0.0_real64
      gradient%key = 0.0_real64
      gradient%value = 0.0_real64
      gradient%attention_output = 0.0_real64
      gradient%attention_rms = 0.0_real64
      gradient%ffn_rms = 0.0_real64
      gradient%ffn_gate = 0.0_real64
      gradient%ffn_value = 0.0_real64
      gradient%ffn_output = 0.0_real64
      gradient%distance_bias = 0.0_real64
      gradient%scale_logit = 0.0_real64
      gradient%decay_logit = 0.0_real64
      gradient%write_strength_logit = 0.0_real64
      gradient%channel_logit = 0.0_real64
      gradient%channel_gate_proj = 0.0_real64
      gradient%channel_router = 0.0_real64
      gradient%induction_mix = 0.0_real64
      gradient%scale_gate_proj = 0.0_real64
      gradient%slot_query_proj = 0.0_real64
      gradient%slot_key_proj = 0.0_real64
      gradient%slot_value_proj = 0.0_real64
      gradient%slot_read_proj = 0.0_real64
      gradient%slot_salience_weight = 0.0_real64
   end subroutine fbc_gradient_zero

   subroutine fbc_gradient_destroy(gradient)
      type(fbc_gradient_t), intent(inout) :: gradient

      if (allocated(gradient%token_embedding)) deallocate(gradient%token_embedding)
      if (allocated(gradient%output_bias)) deallocate(gradient%output_bias)
      if (allocated(gradient%query)) deallocate(gradient%query)
      if (allocated(gradient%key)) deallocate(gradient%key)
      if (allocated(gradient%value)) deallocate(gradient%value)
      if (allocated(gradient%attention_output)) deallocate(gradient%attention_output)
      if (allocated(gradient%attention_rms)) deallocate(gradient%attention_rms)
      if (allocated(gradient%ffn_rms)) deallocate(gradient%ffn_rms)
      if (allocated(gradient%ffn_gate)) deallocate(gradient%ffn_gate)
      if (allocated(gradient%ffn_value)) deallocate(gradient%ffn_value)
      if (allocated(gradient%ffn_output)) deallocate(gradient%ffn_output)
      if (allocated(gradient%distance_bias)) deallocate(gradient%distance_bias)
      if (allocated(gradient%scale_logit)) deallocate(gradient%scale_logit)
      if (allocated(gradient%decay_logit)) deallocate(gradient%decay_logit)
      if (allocated(gradient%write_strength_logit)) deallocate(gradient%write_strength_logit)
      if (allocated(gradient%channel_logit)) deallocate(gradient%channel_logit)
      if (allocated(gradient%channel_gate_proj)) deallocate(gradient%channel_gate_proj)
      if (allocated(gradient%channel_router)) deallocate(gradient%channel_router)
      if (allocated(gradient%induction_mix)) deallocate(gradient%induction_mix)
      if (allocated(gradient%scale_gate_proj)) deallocate(gradient%scale_gate_proj)
      if (allocated(gradient%slot_query_proj)) deallocate(gradient%slot_query_proj)
      if (allocated(gradient%slot_key_proj)) deallocate(gradient%slot_key_proj)
      if (allocated(gradient%slot_value_proj)) deallocate(gradient%slot_value_proj)
      if (allocated(gradient%slot_read_proj)) deallocate(gradient%slot_read_proj)
      gradient = fbc_gradient_t()
   end subroutine fbc_gradient_destroy

   ! Local copy of the IEEE exponent check: the norm walks every gradient
   ! element, so cross-module calls here would prevent normal inlining.
   pure elemental logical function gradient_value_is_finite(value) result(finite)
      real(real64), intent(in) :: value
      integer(int64) :: bits

      if (radix(value) == 2 .and. digits(value) == 53 .and. maxexponent(value) == 1024 .and. &
          minexponent(value) == -1021 .and. storage_size(value) == 64) then
         bits = transfer(value, 0_int64)
         finite = iand(bits, int(z'7FF0000000000000', int64)) /= int(z'7FF0000000000000', int64)
      else
         finite = ieee_is_finite(value)
      end if
   end function gradient_value_is_finite

   !> Global L2 norm in one pass per array. A negative result is an explicit
   !> error sentinel for an uninitialized/nonfinite gradient or a norm outside
   !> real64's representable range. The ordinary path keeps inexpensive sums of
   !> squares; very large or subnormal-scale values switch to a scaled norm
   !> without rescanning the elements already consumed.
   pure real(real64) function fbc_gradient_norm(gradient) result(norm)
      type(fbc_gradient_t), intent(in) :: gradient
      real(real64) :: squares, scale, scaled_squares, upper, lower, root
      logical :: scaled, valid
      integer(int64) :: total_elements

      norm = -1.0_real64
      if (.not. gradient%initialized) return
      total_elements = size(gradient%token_embedding, kind=int64) + &
                       size(gradient%output_bias, kind=int64) + &
                       size(gradient%query, kind=int64) + &
                       size(gradient%key, kind=int64) + &
                       size(gradient%value, kind=int64) + &
                       size(gradient%attention_output, kind=int64) + &
                       size(gradient%attention_rms, kind=int64) + &
                       size(gradient%ffn_rms, kind=int64) + &
                       size(gradient%ffn_gate, kind=int64) + &
                       size(gradient%ffn_value, kind=int64) + &
                       size(gradient%ffn_output, kind=int64) + &
                       size(gradient%distance_bias, kind=int64) + &
                       size(gradient%decay_logit, kind=int64) + &
                       size(gradient%write_strength_logit, kind=int64) + &
                       size(gradient%scale_logit, kind=int64) + &
                       size(gradient%channel_logit, kind=int64) + &
                       size(gradient%channel_gate_proj, kind=int64) + &
                       size(gradient%scale_gate_proj, kind=int64) + &
                       size(gradient%channel_router, kind=int64) + &
                       size(gradient%induction_mix, kind=int64) + &
                       size(gradient%slot_query_proj, kind=int64) + &
                       size(gradient%slot_key_proj, kind=int64) + &
                       size(gradient%slot_value_proj, kind=int64) + &
                       size(gradient%slot_read_proj, kind=int64) + 1_int64
      ! With |x| <= upper the sum of all x*x is bounded by huge/4, with
      ! headroom for rounding. Values below lower would lose precision when
      ! squared and therefore use the scaled path too.
      upper = sqrt(huge(0.0_real64) / (4.0_real64 * real(total_elements, real64)))
      lower = sqrt(tiny(0.0_real64))
      squares = 0.0_real64
      scale = 0.0_real64
      scaled_squares = 0.0_real64
      scaled = .false.
      valid = .true.
      call add_norm_values(gradient%token_embedding, size(gradient%token_embedding, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%output_bias, size(gradient%output_bias, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%query, size(gradient%query, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%key, size(gradient%key, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%value, size(gradient%value, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%attention_output, size(gradient%attention_output, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%attention_rms, size(gradient%attention_rms, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%ffn_rms, size(gradient%ffn_rms, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%ffn_gate, size(gradient%ffn_gate, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%ffn_value, size(gradient%ffn_value, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%ffn_output, size(gradient%ffn_output, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%distance_bias, size(gradient%distance_bias, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%decay_logit, size(gradient%decay_logit, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%write_strength_logit, size(gradient%write_strength_logit, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%scale_logit, size(gradient%scale_logit, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%channel_logit, size(gradient%channel_logit, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%channel_gate_proj, size(gradient%channel_gate_proj, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%scale_gate_proj, size(gradient%scale_gate_proj, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%channel_router, size(gradient%channel_router, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%induction_mix, size(gradient%induction_mix, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%slot_query_proj, size(gradient%slot_query_proj, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%slot_key_proj, size(gradient%slot_key_proj, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%slot_value_proj, size(gradient%slot_value_proj, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values(gradient%slot_read_proj, size(gradient%slot_read_proj, kind=int64), &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      call add_norm_values([gradient%slot_salience_weight], 1_int64, &
                           upper, lower, squares, scale, scaled_squares, scaled, valid)
      if (.not. valid) return
      if (.not. scaled) then
         norm = sqrt(squares)
      else if (scale == 0.0_real64) then
         norm = 0.0_real64
      else
         root = sqrt(scaled_squares)
         if (root == 1.0_real64) then
            norm = scale
         else
            ! Equality is rejected conservatively because huge/root can round
            ! upward before the subsequent multiplication.
            if (scale >= huge(0.0_real64) / root) return
            norm = scale * root
         end if
      end if
   end function fbc_gradient_norm

   ! Sequence association lets rank-1/2/3 contiguous allocatables share this
   ! single streaming kernel without materializing a flattened copy.
   pure subroutine add_norm_values(values, count, upper, lower, squares, scale, scaled_squares, scaled, valid)
      real(real64), intent(in) :: values(*)
      integer(int64), intent(in) :: count
      real(real64), intent(in) :: upper, lower
      real(real64), intent(inout) :: squares, scale, scaled_squares
      logical, intent(inout) :: scaled, valid
      integer, parameter :: block_size = 256
      integer(int64), parameter :: magnitude_mask = int(z'7FFFFFFFFFFFFFFF', int64)
      integer(int64) :: first, index, bits, magnitude_bits, upper_bits, lower_bits
      integer :: index_in_block, block_length
      real(real64) :: safe_value, block_squares, magnitude, ratio
      logical :: exceptional, ordinary, binary64

      if (.not. valid) return
      binary64 = storage_size(upper) == 64 .and. radix(upper) == 2 .and. digits(upper) == 53 .and. &
                 maxexponent(upper) == 1024 .and. minexponent(upper) == -1021
      if (binary64) then
         upper_bits = transfer(upper, 0_int64)
         lower_bits = transfer(lower, 0_int64)
      else
         upper_bits = 0_int64
         lower_bits = 0_int64
      end if
      first = 1_int64
      do while (first <= count)
         block_length = int(min(int(block_size, int64), count - first + 1_int64))
         if (.not. scaled .and. binary64) then
            ! Mask unsafe bit patterns to zero before floating arithmetic;
            ! both the finite check and squared reduction share this pass.
            exceptional = .false.
            block_squares = 0.0_real64
            do index_in_block = 1, block_length
               index = first + int(index_in_block - 1, int64)
               bits = transfer(values(index), 0_int64)
               magnitude_bits = iand(bits, magnitude_mask)
               ordinary = magnitude_bits <= upper_bits .and. &
                          (magnitude_bits == 0_int64 .or. magnitude_bits >= lower_bits)
               safe_value = transfer(merge(bits, 0_int64, ordinary), 0.0_real64)
               block_squares = block_squares + safe_value * safe_value
               exceptional = exceptional .or. .not. ordinary
            end do
            squares = squares + block_squares
            if (exceptional) then
               ! Reread only an exceptional block, not a full gradient array.
               ! Normal elements in it have already contributed above.
               do index_in_block = 1, block_length
                  index = first + int(index_in_block - 1, int64)
                  bits = transfer(values(index), 0_int64)
                  magnitude_bits = iand(bits, magnitude_mask)
                  ordinary = magnitude_bits <= upper_bits .and. &
                             (magnitude_bits == 0_int64 .or. magnitude_bits >= lower_bits)
                  if (ordinary) cycle
                  if (.not. gradient_value_is_finite(values(index))) then
                     valid = .false.
                     return
                  end if
                  magnitude = abs(values(index))
                  if (.not. scaled) then
                     scale = sqrt(squares)
                     if (scale > 0.0_real64) scaled_squares = 1.0_real64
                     scaled = .true.
                  end if
                  if (scale < magnitude) then
                     ratio = scale / magnitude
                     scaled_squares = 1.0_real64 + scaled_squares * ratio * ratio
                     scale = magnitude
                  else
                     ratio = magnitude / scale
                     scaled_squares = scaled_squares + ratio * ratio
                  end if
               end do
            end if
         else
            ! A prior exceptional value, or a non-IEEE representation, uses
            ! the scalar scaled recurrence without any additional array pass.
            do index = first, first + int(block_length - 1, int64)
               if (.not. gradient_value_is_finite(values(index))) then
                  valid = .false.
                  return
               end if
               magnitude = abs(values(index))
               if (.not. scaled) then
                  if (magnitude > upper .or. (magnitude > 0.0_real64 .and. magnitude < lower)) then
                     scale = sqrt(squares)
                     if (scale > 0.0_real64) scaled_squares = 1.0_real64
                     scaled = .true.
                  else
                     squares = squares + values(index) * values(index)
                     cycle
                  end if
               end if
               if (magnitude == 0.0_real64) cycle
               if (scale < magnitude) then
                  ratio = scale / magnitude
                  scaled_squares = 1.0_real64 + scaled_squares * ratio * ratio
                  scale = magnitude
               else
                  ratio = magnitude / scale
                  scaled_squares = scaled_squares + ratio * ratio
               end if
            end do
         end if
         first = first + int(block_length, int64)
      end do
   end subroutine add_norm_values

   !> dLoss/dParameters for the step whose activations are in `workspace`.
   !>
   !> `dlogits` is dLoss/dlogits for that single step.  The state is read (the
   !> slot read needs the post-write keys) but never written, and no gradient is
   !> stored into it: that is the depth-1 truncation described in the header.
   subroutine fbc_step_backward(model, state, workspace, dlogits, gradient, status)
      type(fbc_model_t), intent(in) :: model
      type(fbc_state_t), intent(in) :: state
      type(fbc_workspace_t), intent(in) :: workspace
      real(real64), intent(in) :: dlogits(:)
      type(fbc_gradient_t), intent(inout) :: gradient
      integer(int32), intent(out) :: status
      integer(int32) :: width, heads, head_width, scales, slots, ffn
      integer(int32) :: head, scale, component, slot, ordinal, channel, position, row, column
      integer(int32) :: column_index, channel_source
      real(real64) :: context_scale, slot_scale, moment, weight_sum, slope
      real(real64) :: gate, tanh_value, sigmoid_value, write_slope, denominator, salience_moment
      ! Automatic arrays, sized from the *dummy* model and workspace rather than
      ! from the local copies below: a local variable is not a specification
      ! expression, so it cannot bound a declaration.  Nothing here allocates on
      ! the heap, so one step costs no allocator traffic.
      real(real64) :: dn(model%config%width), dc(model%config%width)
      real(real64) :: dr(model%config%width), dx(model%config%width)
      real(real64) :: dv_ffn(model%config%ffn_width), dh_ffn(model%config%ffn_width)
      real(real64) :: du_ffn(model%config%ffn_width), dn2(model%config%width)
      real(real64) :: dq(model%head_width, model%config%heads)
      real(real64) :: dk(model%head_width, model%config%heads)
      real(real64) :: dvv(model%head_width, model%config%heads)
      real(real64) :: dloc(model%head_width), dmem(model%head_width)
      real(real64) :: dphi(model%head_width), dnum(model%head_width)
      real(real64) :: dlocal_weight(workspace%local_count + 1_int32)
      real(real64) :: dchannel(FBC_CHANNEL_COUNT), dcontext(FBC_CHANNEL_COUNT)
      ! d(combined)/d(channel_context) arriving through the context router.
      real(real64) :: dcontext_source(model%config%width, FBC_CHANNEL_COUNT)
      real(real64) :: dslot_channel(model%config%width)
      real(real64) :: dind(model%head_width)
      real(real64) :: shifted
      real(real64) :: dslot_query(model%config%width), dslot_key(model%config%width)
      real(real64) :: dslot_value(model%config%width), dslot_read(model%config%width)
      real(real64) :: dslot_read_score(model%config%memory_slots)
      real(real64) :: dsalience(model%config%memory_slots)
      real(real64) :: dscale_alpha(model%config%memory_scales)
      real(real64) :: dscale_logit(model%config%memory_scales)
      real(real64) :: dslot_key_after(model%config%width)
      real(real64) :: dslot_value_after(model%config%width)
      real(real64) :: key_before(model%config%width)
      real(real64) :: dffn_scaled(model%config%width), dattn_scaled(model%config%width)
      ! `workspace%normalized` is reused by the FFN inside the same step, so by
      ! the time the backward runs it holds the *FFN* normalisation.  The
      ! Q/K/V and slot projections were computed from the attention one, so it
      ! is rebuilt here rather than read back.
      real(real64) :: attention_input(model%config%width)
      ! Accumulated dL/d(normalized) contributed by the input-dependent gates.
      ! Both mixtures are softmax(logit + W . normalized), so the gate has two
      ! gradients: one towards its own rows (W) and one towards the shared input.
      real(real64) :: dn_gate(model%config%width)
      ! Scratch for the intra-step retention write (see stage 8).
      real(real64) :: dphi_key(model%head_width), dvalue(model%head_width)
      real(real64) :: dretention(model%head_width, model%head_width)
      real(real64) :: ddenominator, decay_slope
      real(real64) :: ddecay, dwrite, dgate_raw, lambda, gate_cached, write_signoid

      status = FBC_OK
      if (.not. fbc_state_is_ready(state, model) .or. .not. workspace%initialized .or. .not. gradient%initialized) then
         status = FBC_ERR_NOT_READY
         return
      end if
      if (.not. fbc_layout_matches(workspace%layout, model%config) .or. &
          .not. fbc_layout_matches(gradient%layout, model%config)) then
         status = FBC_ERR_NOT_READY
         return
      end if
      if (size(dlogits) /= model%config%vocabulary_size) then
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      if (model%training_rule /= FBC_TRAINING_RULE_V5 .and. model%training_rule /= FBC_TRAINING_RULE_V6 .and. &
          model%training_rule /= FBC_TRAINING_RULE_V7) then
         status = FBC_ERR_INVALID_ARGUMENT
         return
      end if
      if (state%anchored_slots .neqv. &
          (model%training_rule == FBC_TRAINING_RULE_V7 .and. model%config%memory_slots > 1_int32)) then
         status = FBC_ERR_ORDER
         return
      end if
      if (workspace%token < 1_int32 .or. workspace%token > model%config%vocabulary_size .or. &
          workspace%local_count < 0_int32 .or. workspace%local_count > state%local_window .or. &
          workspace%forward_step <= 0_int64 .or. state%steps < 0_int64) then
         status = FBC_ERR_ORDER
         return
      end if
      if (state%cache_count < 0_int32) then
         if (state%steps == huge(0_int64)) then
            status = FBC_ERR_ORDER
            return
         end if
         if (workspace%forward_step /= state%steps + 1_int64) then
            status = FBC_ERR_ORDER
            return
         end if
         if (model%training_rule == FBC_TRAINING_RULE_V5 .or. &
             -1_int32 - state%cache_count /= workspace%local_count) then
            status = FBC_ERR_ORDER
            return
         end if
      else
         if (state%cache_count > state%local_window .or. workspace%forward_step /= state%steps) then
            status = FBC_ERR_ORDER
            return
         end if
         if (model%training_rule /= FBC_TRAINING_RULE_V5 .and. &
             workspace%local_count == model%config%local_window) then
            ! The evicted key/value are no longer available: returning a plausible
            ! but wrong gradient would be worse than failing explicitly.
            status = FBC_ERR_ORDER
            return
         end if
      end if
      attention_input = workspace%rhat_attention * model%attention_rms
      width = model%config%width
      heads = model%config%heads
      head_width = model%head_width
      scales = model%config%memory_scales
      slots = model%config%memory_slots
      ffn = model%config%ffn_width

      ! ---- 1. tied logits ----------------------------------------------------
      ! logits(j) = hidden . E(:, j) + bias(j), so the tied matrix takes the FULL
      ! outer product hidden (x) dlogits: column j is touched for every j in the
      ! vocabulary, not only for the column of the token being read.  Writing
      ! just the current token's column here (which is what an earlier revision
      ! did) silently drops d(logit_j)/dE(:, j) for every j != token: the model
      ! can then never learn to suppress a wrong byte, only to move the vector of
      ! a byte it is currently reading.  The input side below adds the other
      ! term (d x) to the current token's column, and the two must be summed, not
      ! confused.
      gradient%output_bias = gradient%output_bias + dlogits
      do column = 1_int32, size(dlogits, kind=int32)
         gradient%token_embedding(:, column) = gradient%token_embedding(:, column) + &
                                              workspace%hidden * dlogits(column)
      end do
      dn_gate = 0.0_real64
      dn = matmul(model%token_embedding, dlogits)

      ! ---- 2. residual stream and the gated FFN -----------------------------
      dr = dn
      if (model%training_rule == FBC_TRAINING_RULE_V5) then
         gradient%ffn_output = gradient%ffn_output + fbc_outer(dr, workspace%ffn_gated)
      else
         call fbc_outer_add(gradient%ffn_output, dr, workspace%ffn_gated)
      end if
      dv_ffn = matmul(dr, model%ffn_output)
      do component = 1_int32, ffn
         sigmoid_value = fbc_sigmoid(workspace%ffn_pre_gate(component))
         tanh_value = tanh(workspace%ffn_pre_value(component))
         dh_ffn(component) = dv_ffn(component) * tanh_value * sigmoid_value * (1.0_real64 - sigmoid_value)
         du_ffn(component) = dv_ffn(component) * sigmoid_value * (1.0_real64 - tanh_value * tanh_value)
      end do
      if (model%training_rule == FBC_TRAINING_RULE_V5) then
         gradient%ffn_gate = gradient%ffn_gate + fbc_outer(dh_ffn, workspace%normalized)
         gradient%ffn_value = gradient%ffn_value + fbc_outer(du_ffn, workspace%normalized)
      else
         call fbc_outer_add(gradient%ffn_gate, dh_ffn, workspace%normalized)
         call fbc_outer_add(gradient%ffn_value, du_ffn, workspace%normalized)
      end if
      dn2 = matmul(dh_ffn, model%ffn_gate) + matmul(du_ffn, model%ffn_value)

      ! ---- 3. second RMSNorm -------------------------------------------------
      gradient%ffn_rms = gradient%ffn_rms + dn2 * workspace%rhat_ffn
      do component = 1_int32, width
         dffn_scaled(component) = dn2(component) * model%ffn_rms(component)
      end do
      call rmsnorm_backward(dffn_scaled, workspace%rhat_ffn, workspace%inverse_rms_ffn, dr)
      ! hidden = attention_residual + ffn_output * gated, so attention_residual
      ! receives the direct d(hidden) *and* the through-FFN gradient.  Step 2
      ! stored the former in `dr`; step 3 overwrote it, so add it back here.
      dr = dr + dn

      ! ---- 4. attention residual and the output projection ------------------
      if (model%training_rule == FBC_TRAINING_RULE_V5) then
         gradient%attention_output = gradient%attention_output + fbc_outer(dr, workspace%combined)
      else
         call fbc_outer_add(gradient%attention_output, dr, workspace%combined)
      end if
      dc = matmul(dr, model%attention_output)
      dx = dr

      ! ---- 5. channel mixing -------------------------------------------------
      do channel = 1_int32, FBC_CHANNEL_COUNT
         dcontext(channel) = dot_product(dc, workspace%channel_context(:, channel))
      end do
      moment = dot_product(workspace%channel_alpha, dcontext)
      do channel = 1_int32, FBC_CHANNEL_COUNT
         dchannel(channel) = workspace%channel_alpha(channel) * (dcontext(channel) - moment)
      end do
      gradient%channel_logit = gradient%channel_logit + dchannel
      ! The gate rows and the shared input both see the same dL/d(logit_eff).
      do channel = 1_int32, FBC_CHANNEL_COUNT
         gradient%channel_gate_proj(channel, :) = gradient%channel_gate_proj(channel, :) + &
                                                  dchannel(channel) * attention_input
         dn_gate = dn_gate + dchannel(channel) * model%channel_gate_proj(channel, :)
      end do
      ! The router reads the three contexts, so each context vector receives the
      ! share of the loss that arrives by changing the mixture weights, on top
      ! of the share it receives as the vector being mixed.
      dcontext_source = 0.0_real64
      do channel_source = 1_int32, FBC_CHANNEL_COUNT
         do component = 1_int32, width
            column_index = (component - 1_int32) * FBC_CHANNEL_COUNT + channel_source
            gradient%channel_router(:, column_index) = gradient%channel_router(:, column_index) + &
               dchannel * workspace%channel_context(component, channel_source)
            dcontext_source(component, channel_source) = &
               dot_product(dchannel, model%channel_router(:, column_index))
         end do
      end do

      ! ---- 6. slot channel: read side ---------------------------------------
      dslot_channel = workspace%channel_alpha(FBC_CHANNEL_SLOTS) * dc + &
                      dcontext_source(:, FBC_CHANNEL_SLOTS)
      dslot_read = matmul(dslot_channel, model%slot_read_proj)
      if (model%training_rule == FBC_TRAINING_RULE_V5) then
         gradient%slot_read_proj = gradient%slot_read_proj + fbc_outer(dslot_channel, workspace%slot_read)
      else
         call fbc_outer_add(gradient%slot_read_proj, dslot_channel, workspace%slot_read)
      end if
      do slot = 1_int32, slots
         dslot_read_score(slot) = dot_product(dslot_read, state%slot_value(:, slot))
      end do
      weight_sum = 0.0_real64
      do slot = 1_int32, slots
         weight_sum = weight_sum + workspace%slot_read_weight(slot) * dslot_read_score(slot)
      end do
      do slot = 1_int32, slots
         dslot_read_score(slot) = workspace%slot_read_weight(slot) * (dslot_read_score(slot) - weight_sum)
      end do
      slot_scale = 1.0_real64 / sqrt(real(width, real64))
      dsalience = 0.0_real64
      dslot_query = 0.0_real64
      do slot = 1_int32, slots
         dslot_query = dslot_query + dslot_read_score(slot) * state%slot_key(:, slot) * slot_scale
         dsalience(slot) = model%slot_salience_weight * dslot_read_score(slot)
         gradient%slot_salience_weight = gradient%slot_salience_weight + &
                                         dslot_read_score(slot) * state%slot_salience(slot)
      end do

      ! ---- 7. slot channel: write side (live thanks to write-then-read) -----
      dslot_key = 0.0_real64
      dslot_value = 0.0_real64
      do slot = 1_int32, slots
         gate = workspace%slot_write_gate(slot)
         dslot_key_after = dslot_read_score(slot) * workspace%slot_query * slot_scale
         dslot_value_after = workspace%slot_read_weight(slot) * dslot_read
         ! g = sigmoid(score / T); the state update is affine in (key, value, g).
         write_slope = (dot_product(dslot_key_after, workspace%slot_key_delta(:, slot)) + &
                        dot_product(dslot_value_after, workspace%slot_value_delta(:, slot)) + &
                        (1.0_real64 - model%config%slot_decay) * dsalience(slot))
         write_slope = gate * (1.0_real64 - gate) * write_slope / model%config%slot_temperature
         ! The anchored component is not written from slot_write_key. Keep its
         ! gate/read derivatives, but omit the direct write derivative only.
         if (state%anchored_slots) dslot_key_after(1_int32) = 0.0_real64
         dslot_key = dslot_key + gate * dslot_key_after
         dslot_value = dslot_value + gate * dslot_value_after
         ! K_before is recovered exactly: K_after = K_before + g * delta.
         key_before = state%slot_key(:, slot) - gate * workspace%slot_key_delta(:, slot)
         dslot_key = dslot_key + write_slope * key_before * slot_scale
      end do

      do component = 1_int32, width
         gradient%slot_query_proj(component, :) = gradient%slot_query_proj(component, :) + &
            dslot_query(component) * attention_input
         gradient%slot_key_proj(component, :) = gradient%slot_key_proj(component, :) + &
            dslot_key(component) * attention_input
         gradient%slot_value_proj(component, :) = gradient%slot_value_proj(component, :) + &
            dslot_value(component) * attention_input
      end do
      dn = matmul(dslot_query, model%slot_query_proj) + matmul(dslot_key, model%slot_key_proj) + &
           matmul(dslot_value, model%slot_value_proj)

      ! ---- 8. multi-scale recurrent memory: read, then the intra-step write --
      ! R <- lambda R + (1-lambda) phi(k) v^T and z <- lambda z + (1-lambda)
      ! phi(k), and the *same* step then reads R and z.  This token's phi(k) and
      ! v are therefore on the differentiable path and must receive gradient;
      ! earlier tokens' writes stay frozen (depth-1 truncated BPTT), which is
      ! the documented approximation.  Omitting this block trains a model that
      ! silently never learns what it writes.
      dscale_alpha = 0.0_real64
      dq = 0.0_real64
      dk = 0.0_real64
      dvv = 0.0_real64
      do head = 1_int32, heads
         dmem = workspace%channel_alpha(FBC_CHANNEL_RETENTION) * &
                dc((head - 1_int32) * head_width + 1_int32:head * head_width) + &
                dcontext_source((head - 1_int32) * head_width + 1_int32:head * head_width, &
                                FBC_CHANNEL_RETENTION)
         dphi = 0.0_real64
         dphi_key = 0.0_real64
         dvalue = 0.0_real64
         do scale = 1_int32, scales
            denominator = workspace%read_denominator(head, scale)
            dscale_alpha(scale) = dscale_alpha(scale) + &
                                  dot_product(dmem, workspace%read_scale(:, head, scale))
            ! read = phi(q) . R / (phi(q) . z + epsilon)
            dnum = workspace%scale_alpha(scale) * dmem / denominator
            ddenominator = -workspace%scale_alpha(scale) * &
                           dot_product(dmem, workspace%read_scale(:, head, scale)) / denominator
            dphi = dphi + matmul(state%retention_numerator(:, :, head, scale), dnum) + &
                   ddenominator * state%retention_normalizer(:, head, scale)
            ! dR(i, c) = phi(q)(i) * dnum(c) and dz(i) = ddenominator * phi(q)(i).
            do component = 1_int32, head_width
               dretention(:, component) = workspace%phi_q(:, head) * dnum(component)
            end do
            ! R_after = lambda R_before + g phi(k) v^T
            ! z_after = lambda z_before + g phi(k)
            ! so dR_after/dg = phi(k) v^T and dR_after/dlambda = R_before.  The
            ! state dummy holds the post-write state, so R_before is recovered
            ! from the deltas, exactly as the slot write does below.
            lambda = model%scale_decay(scale)
            gate_cached = model%write_strength(scale)
            dphi_key = dphi_key + gate_cached * (matmul(dretention, workspace%v(:, head)) + &
                                                 ddenominator * workspace%phi_q(:, head))
            dvalue = dvalue + gate_cached * matmul(workspace%phi_k(:, head), dretention)
            ddecay = 0.0_real64
            dwrite = 0.0_real64
            do component = 1_int32, head_width
               ! Columns of the numerator: rank-one write on column `component`.
               ddecay = ddecay + dot_product(dretention(:, component), &
                  state%retention_numerator(:, component, head, scale) - &
                  gate_cached * workspace%phi_k(:, head) * workspace%v(component, head))
               dwrite = dwrite + dot_product(dretention(:, component), &
                  workspace%phi_k(:, head)) * workspace%v(component, head)
            end do
            ddecay = ddecay + dot_product(ddenominator * workspace%phi_q(:, head), &
               state%retention_normalizer(:, head, scale) - gate_cached * workspace%phi_k(:, head))
            dwrite = dwrite + dot_product(ddenominator * workspace%phi_q(:, head), workspace%phi_k(:, head))

            ! R_before was recovered as lambda * R_before, so undo the factor:
            ! dR_after/dlambda = R_before exactly, not lambda * R_before.
            ddecay = ddecay / max(lambda, 1.0e-12_real64)

            ! The two logits are coupled: g = (1 - lambda) * 2 sigmoid(w), so a
            ! change in decay_logit moves lambda *and*, through it, the write
            ! strength.  dg/dlambda = -2 sigmoid(w); omitting this term makes the
            ! analytic decay gradient disagree with a finite difference that
            ! refreshes the cache, which is exactly what the probe caught.
            write_signoid = fbc_sigmoid(model%write_strength_logit(scale))
            dgate_raw = 2.0_real64 * write_signoid
            ! lambda is clamped in fbc_refresh_dynamics. Outside the open
            ! interval it is locally constant, so the v6 derivative is zero
            ! (also chosen at the two nondifferentiable boundary points). The
            ! v5 branch is retained exactly for legacy checkpoint continuation.
            if (model%training_rule == FBC_TRAINING_RULE_V5 .or. &
                (lambda > FBC_MIN_DECAY .and. lambda < FBC_MAX_DECAY)) then
               gradient%decay_logit(scale) = gradient%decay_logit(scale) + &
                  (ddecay - dwrite * dgate_raw) * lambda * (1.0_real64 - lambda)
            end if
            ! d g / d write_logit = (1 - lambda) * 2 s (1 - s).
            gradient%write_strength_logit(scale) = gradient%write_strength_logit(scale) + &
               dwrite * (1.0_real64 - lambda) * dgate_raw * (1.0_real64 - write_signoid)
         end do
         do component = 1_int32, head_width
            dq(component, head) = dq(component, head) + &
                                  fbc_softplus_slope(workspace%q(component, head)) * dphi(component)
            dk(component, head) = dk(component, head) + &
                                  fbc_softplus_slope(workspace%k(component, head)) * dphi_key(component)
            dvv(component, head) = dvv(component, head) + dvalue(component)
         end do
      end do
      moment = dot_product(workspace%scale_alpha, dscale_alpha)
      dscale_logit = workspace%scale_alpha * (dscale_alpha - moment)
      gradient%scale_logit = gradient%scale_logit + dscale_logit
      do scale = 1_int32, scales
         gradient%scale_gate_proj(scale, :) = gradient%scale_gate_proj(scale, :) + &
                                              dscale_logit(scale) * attention_input
         dn_gate = dn_gate + dscale_logit(scale) * model%scale_gate_proj(scale, :)
      end do

      ! ---- 9. local attention ------------------------------------------------
      context_scale = 1.0_real64 / sqrt(real(head_width, real64))
      do head = 1_int32, heads
         dloc = workspace%channel_alpha(FBC_CHANNEL_LOCAL) * &
                dc((head - 1_int32) * head_width + 1_int32:head * head_width) + &
                dcontext_source((head - 1_int32) * head_width + 1_int32:head * head_width, &
                                FBC_CHANNEL_LOCAL)
         dlocal_weight(1_int32) = dot_product(dloc, workspace%v(:, head))
         weight_sum = workspace%local_weight(1_int32, head) * dlocal_weight(1_int32)
         do ordinal = 1_int32, workspace%local_count
            position = workspace%local_position(ordinal)
            dlocal_weight(ordinal + 1_int32) = dot_product(dloc, state%local_value(:, head, position))
            weight_sum = weight_sum + workspace%local_weight(ordinal + 1_int32, head) * &
                                      dlocal_weight(ordinal + 1_int32)
         end do
         ! The shifted (induction) read reuses these weights, so its derivative
         ! adds to the same d(weight) entries before the softmax backward runs.
         ! Its own weight sees the read it mixes in, and it reads values that the
         ! previous steps wrote, so no gradient crosses the step boundary.
         gradient%induction_mix(head) = gradient%induction_mix(head) + &
                                        dot_product(dloc, workspace%induction_context(:, head))
         dind = model%induction_mix(head) * dloc
         do ordinal = 1_int32, workspace%local_count
            if (ordinal == 1_int32) then
               shifted = dot_product(dind, workspace%v(:, head))
               dlocal_weight(2_int32) = dlocal_weight(2_int32) + shifted
               weight_sum = weight_sum + workspace%local_weight(2_int32, head) * shifted
            else
               position = workspace%local_position(ordinal - 1_int32)
               shifted = dot_product(dind, state%local_value(:, head, position))
               dlocal_weight(ordinal + 1_int32) = dlocal_weight(ordinal + 1_int32) + shifted
               weight_sum = weight_sum + workspace%local_weight(ordinal + 1_int32, head) * shifted
            end if
         end do
         do ordinal = 1_int32, workspace%local_count + 1_int32
            dlocal_weight(ordinal) = workspace%local_weight(ordinal, head) * (dlocal_weight(ordinal) - weight_sum)
         end do
         gradient%distance_bias(1_int32) = gradient%distance_bias(1_int32) + dlocal_weight(1_int32)
         do ordinal = 1_int32, workspace%local_count
            position = workspace%local_position(ordinal)
            dq(:, head) = dq(:, head) + dlocal_weight(ordinal + 1_int32) * &
                          state%local_key(:, head, position) * context_scale
            gradient%distance_bias(ordinal + 1_int32) = gradient%distance_bias(ordinal + 1_int32) + &
                                                        dlocal_weight(ordinal + 1_int32)
         end do
         dq(:, head) = dq(:, head) + dlocal_weight(1_int32) * workspace%k(:, head) * context_scale
         dk(:, head) = dk(:, head) + dlocal_weight(1_int32) * workspace%q(:, head) * context_scale
         dvv(:, head) = dvv(:, head) + workspace%local_weight(1_int32, head) * dloc
         if (workspace%local_count >= 1_int32) then
            ! The successor of the most recent cached position is this step's own
            ! value, so the shifted read touches v as well as the frozen cache.
            dvv(:, head) = dvv(:, head) + workspace%local_weight(2_int32, head) * dind
         end if
      end do

      ! ---- 10. head projections ---------------------------------------------
      do head = 1_int32, heads
         do row = 1_int32, head_width
            gradient%query(row, :, head) = gradient%query(row, :, head) + dq(row, head) * attention_input
            gradient%key(row, :, head) = gradient%key(row, :, head) + dk(row, head) * attention_input
            gradient%value(row, :, head) = gradient%value(row, :, head) + dvv(row, head) * attention_input
         end do
         dn = dn + matmul(dq(:, head), model%query(:, :, head)) + matmul(dk(:, head), model%key(:, :, head)) + &
              matmul(dvv(:, head), model%value(:, :, head))
      end do

      ! The gate reads the same normalised input as Q/K/V, the slot
      ! projections and the FFN, so its input-side gradient joins theirs before
      ! the RMSNorm backward splits the sum between the norm and the embedding.
      dn = dn + dn_gate

      ! ---- 11. input RMSNorm and the embedding row --------------------------
      gradient%attention_rms = gradient%attention_rms + dn * workspace%rhat_attention
      do component = 1_int32, width
         dattn_scaled(component) = dn(component) * model%attention_rms(component)
      end do
      ! `dx` already holds d(x) from the residual branch; the normalization
      ! branch adds to it, so the RMSNorm backward writes into a scratch array
      ! (`dn2` is dead by now) and the two are summed below.
      call rmsnorm_backward(dattn_scaled, workspace%rhat_attention, workspace%inverse_rms_attention, dn2)
      do component = 1_int32, width
         gradient%token_embedding(component, workspace%token) = &
            gradient%token_embedding(component, workspace%token) + dx(component) + dn2(component)
      end do
   end subroutine fbc_step_backward

   !----------------------------------------------------------------------------
   !  Helpers
   !----------------------------------------------------------------------------

   !> dL/dr for out = r * inv * g with rhat = r * inv.
   pure subroutine rmsnorm_backward(scaled_gradient, rhat, inverse_rms, output)
      real(real64), intent(in) :: scaled_gradient(:), rhat(:)
      real(real64), intent(in) :: inverse_rms
      real(real64), intent(out) :: output(:)
      real(real64) :: projection

      projection = dot_product(rhat, scaled_gradient) / real(size(rhat, kind=int32), real64)
      output = (scaled_gradient - rhat * projection) * inverse_rms
   end subroutine rmsnorm_backward

   !> outer(a, b) with the model's exact ranks, so no allocatable temporary is
   !> created per token.
   ! V6 accumulates the outer product by columns directly into the existing
   ! gradient, avoiding a fresh matrix temporary for each token. Each element
   ! still sees one multiplication and one addition; V5 keeps the original
   ! expression to avoid changing legacy continuation on other compilers.
   pure subroutine fbc_outer_add(accumulator, left, right)
      real(real64), intent(inout) :: accumulator(:, :)
      real(real64), intent(in) :: left(:), right(:)
      integer(int32) :: column

      do column = 1_int32, size(right, kind=int32)
         accumulator(:, column) = accumulator(:, column) + left * right(column)
      end do
   end subroutine fbc_outer_add

   pure function fbc_outer(left, right) result(product)
      real(real64), intent(in) :: left(:), right(:)
      real(real64) :: product(size(left, kind=int32), size(right, kind=int32))
      integer(int32) :: column

      do column = 1_int32, size(right, kind=int32)
         product(:, column) = left * right(column)
      end do
   end function fbc_outer

end module fixed_budget_backward
