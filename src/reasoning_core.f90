! SPDX-License-Identifier: MIT
!==============================================================================
!  reasoning_core.f90
!  Safe compositional encoders and one-pass symbolic inference on VSA/HDC
!  hypervectors.  No dense matrices or gradient-based learning are used.
!==============================================================================
module reasoning_core
   use, intrinsic :: iso_fortran_env, only: int32, int64
   use hyper_types
   use symbolic_memory
   implicit none
   private

   integer(int32), parameter, public :: REASON_OK = 0_int32
   integer(int32), parameter, public :: REASON_ERR_INVALID_SHAPE = -200_int32
   integer(int32), parameter, public :: REASON_ERR_ACCUMULATOR = -201_int32

   integer(int64), parameter :: FRAME_TIE_SEED = int(z'5A17F00D3C4B2E19', int64)
   integer(int64), parameter :: RULE_TIE_SEED = int(z'13C0FFEE7A5E1D42', int64)
   integer(int64), parameter :: ANALOGY_TIE_SEED = int(z'6E624EB7D3A91F05', int64)

   public :: encode_role_filler_frame, query_frame_role
   public :: encode_sequence_ngram
   public :: encode_causal_rule, encode_causal_rule_bank, infer_causal_step
   public :: encode_analogy_mapping, solve_analogy_zero_shot

contains

   ! Frame = bundle_i(role_i XOR filler_i).
   ! The caller receives a status instead of an out-of-bounds access when the
   ! two aligned input lists are empty or have different cardinalities.
   pure subroutine encode_role_filler_frame(roles, fillers, frame_hv, status, tie_seed)
      type(hv_t), intent(in) :: roles(:), fillers(:)
      type(hv_t), intent(out) :: frame_hv
      integer(int32), intent(out) :: status
      integer(int64), intent(in), optional :: tie_seed
      type(hv_acc_t) :: accumulator
      type(hv_t) :: bound_pair
      integer(int32) :: pair, add_status
      integer(int64) :: seed

      frame_hv = hv_zero()
      status = REASON_OK
      if (size(roles) == 0 .or. size(roles) /= size(fillers)) then
         status = REASON_ERR_INVALID_SHAPE
         return
      end if

      seed = FRAME_TIE_SEED
      if (present(tie_seed)) seed = tie_seed
      call hv_acc_reset(accumulator)
      do pair = 1, size(roles)
         bound_pair = hv_bind(roles(pair), fillers(pair))
         call hv_acc_add(accumulator, bound_pair, add_status)
         if (add_status /= HV_OK) then
            status = REASON_ERR_ACCUMULATOR
            frame_hv = hv_zero()
            return
         end if
      end do
      frame_hv = hv_acc_finalize(accumulator, seed)
   end subroutine encode_role_filler_frame

   ! Unbinds a role and asks the exact cleanup memory to select the nearest
   ! registered filler.  Crosstalk is expected for multi-pair frames.
   subroutine query_frame_role(mem, frame_hv, target_role, result, status)
      type(hds_memory_t), intent(in) :: mem
      type(hv_t), intent(in) :: frame_hv, target_role
      type(cleanup_result_t), intent(out) :: result
      integer(int32), intent(out) :: status
      type(hv_t) :: unbound_probe

      unbound_probe = hv_bind(frame_hv, target_role)
      call mem_cleanup(mem, unbound_probe, result, status)
   end subroutine query_frame_role

   ! Ordered n-gram encoder:
   ! sequence = XOR_i Pi^(N-i)(token_i).
   ! Binding is deliberately used here (not bundling) so each position is part
   ! of an invertible compositional key.
   pure subroutine encode_sequence_ngram(tokens, sequence_hv, status)
      type(hv_t), intent(in) :: tokens(:)
      type(hv_t), intent(out) :: sequence_hv
      integer(int32), intent(out) :: status
      type(hv_t) :: rotated
      integer(int32) :: token, shift_step, token_count

      sequence_hv = hv_zero()
      status = REASON_OK
      if (size(tokens) == 0) then
         status = REASON_ERR_INVALID_SHAPE
         return
      end if

      token_count = size(tokens, kind=int32)
      do token = 1, token_count
         shift_step = token_count - token
         rotated = hv_permute(tokens(token), shift_step)
         sequence_hv = hv_bind(sequence_hv, rotated)
      end do
   end subroutine encode_sequence_ngram

   ! One directed rule is cause XOR Pi(+1, effect).  Directionality comes from
   ! the reversible permutation; binding alone is symmetric.
   pure function encode_causal_rule(cause_hv, effect_hv) result(rule_hv)
      type(hv_t), intent(in) :: cause_hv, effect_hv
      type(hv_t) :: rule_hv

      rule_hv = hv_bind(cause_hv, hv_permute(effect_hv, 1_int32))
   end function encode_causal_rule

   ! Bundles aligned cause/effect rules into one rule-bank hypervector.
   pure subroutine encode_causal_rule_bank(causes, effects, rule_bank_hv, status, tie_seed)
      type(hv_t), intent(in) :: causes(:), effects(:)
      type(hv_t), intent(out) :: rule_bank_hv
      integer(int32), intent(out) :: status
      integer(int64), intent(in), optional :: tie_seed
      type(hv_acc_t) :: accumulator
      integer(int32) :: rule, add_status
      integer(int64) :: seed

      rule_bank_hv = hv_zero()
      status = REASON_OK
      if (size(causes) == 0 .or. size(causes) /= size(effects)) then
         status = REASON_ERR_INVALID_SHAPE
         return
      end if

      seed = RULE_TIE_SEED
      if (present(tie_seed)) seed = tie_seed
      call hv_acc_reset(accumulator)
      do rule = 1, size(causes)
         call hv_acc_add(accumulator, encode_causal_rule(causes(rule), effects(rule)), &
                         add_status)
         if (add_status /= HV_OK) then
            status = REASON_ERR_ACCUMULATOR
            rule_bank_hv = hv_zero()
            return
         end if
      end do
      rule_bank_hv = hv_acc_finalize(accumulator, seed)
   end subroutine encode_causal_rule_bank

   ! Performs one forward causal inference step and cleans the noisy result
   ! against the registered vocabulary before it is used by the next step.
   subroutine infer_causal_step(mem, rule_bank_hv, observed_cause, inferred_hv, result, status)
      type(hds_memory_t), intent(in) :: mem
      type(hv_t), intent(in) :: rule_bank_hv, observed_cause
      type(hv_t), intent(out) :: inferred_hv
      type(cleanup_result_t), intent(out) :: result
      integer(int32), intent(out) :: status
      type(hv_t) :: raw_shifted, noisy_effect
      logical :: found
      integer(int32) :: get_status

      inferred_hv = hv_zero()
      raw_shifted = hv_bind(rule_bank_hv, observed_cause)
      noisy_effect = hv_permute(raw_shifted, -1_int32)
      call mem_cleanup(mem, noisy_effect, result, status)
      if (status /= MEM_OK) return

      call mem_get_by_id(mem, result%id, inferred_hv, found, get_status)
      if (.not. found .or. get_status /= MEM_OK) then
         inferred_hv = hv_zero()
         status = get_status
      end if
   end subroutine infer_causal_step

   ! Creates an aligned analogy relation:
   ! mapping = bundle_i(source_value_i XOR target_value_i).
   ! Unbinding this mapping with source_value_j yields target_value_j plus
   ! bounded superposition crosstalk.  This is the correct VSA construction;
   ! XORing two already-bundled holistic frames does not, in general, isolate a
   ! role correspondence and is therefore intentionally not exposed as exact.
   pure subroutine encode_analogy_mapping(source_values, target_values, mapping_hv, &
                                          status, tie_seed)
      type(hv_t), intent(in) :: source_values(:), target_values(:)
      type(hv_t), intent(out) :: mapping_hv
      integer(int32), intent(out) :: status
      integer(int64), intent(in), optional :: tie_seed
      type(hv_acc_t) :: accumulator
      integer(int32) :: pair, add_status
      integer(int64) :: seed

      mapping_hv = hv_zero()
      status = REASON_OK
      if (size(source_values) == 0 .or. size(source_values) /= size(target_values)) then
         status = REASON_ERR_INVALID_SHAPE
         return
      end if

      seed = ANALOGY_TIE_SEED
      if (present(tie_seed)) seed = tie_seed
      call hv_acc_reset(accumulator)
      do pair = 1, size(source_values)
         call hv_acc_add(accumulator, hv_bind(source_values(pair), target_values(pair)), &
                         add_status)
         if (add_status /= HV_OK) then
            status = REASON_ERR_ACCUMULATOR
            mapping_hv = hv_zero()
            return
         end if
      end do
      mapping_hv = hv_acc_finalize(accumulator, seed)
   end subroutine encode_analogy_mapping

   ! Zero-shot transfer through a previously encoded aligned structural mapping.
   subroutine solve_analogy_zero_shot(mem, mapping_hv, query_entity, result, status)
      type(hds_memory_t), intent(in) :: mem
      type(hv_t), intent(in) :: mapping_hv, query_entity
      type(cleanup_result_t), intent(out) :: result
      integer(int32), intent(out) :: status
      type(hv_t) :: mapped_query

      mapped_query = hv_bind(mapping_hv, query_entity)
      call mem_cleanup(mem, mapped_query, result, status)
   end subroutine solve_analogy_zero_shot

end module reasoning_core
