! SPDX-License-Identifier: MIT
! Cold, LF-record byte evaluation. Separate from training: no additional
! training scans, updates, state snapshots or checkpoint changes are implied.
module fixed_budget_natural_eval
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64
   use fixed_budget_context
   use fixed_budget_training, only: FBT_OK, FBT_ERR_ARGUMENT, FBT_ERR_NUMERIC, fbc_loss_and_gradient
   implicit none (type, external)
   private

   type, public :: fbc_record_metrics_t
      integer(int64) :: records = 0_int64
      integer(int64) :: scored_records = 0_int64
      integer(int64) :: predictions = 0_int64
      integer(int64) :: prefix_predictions = 0_int64
      real(real64) :: mean_nll = 0.0_real64
      real(real64) :: prefix_mean_nll = 0.0_real64
      real(real64) :: loss_sum = 0.0_real64
      real(real64) :: prefix_loss_sum = 0.0_real64
   end type fbc_record_metrics_t

   public :: fbc_evaluate_lf_records

contains

   ! Tokens are byte IDs (NUL is 256 for V6/V7); byte 10 separates records.
   ! Start each nonempty scored record from a fresh state. The first byte has
   ! no BOS prediction and neither newline nor cross-record transitions are
   ! scored. Prefix NLL scores positions with 1..prefix_tokens of left context.
   ! Input encoding, record overlap and source independence are the caller's
   ! responsibility. On error the metrics remain zero; state may have advanced.
   subroutine fbc_evaluate_lf_records(model, state, workspace, tokens, prefix_tokens, metrics, status)
      type(fbc_model_t), intent(in) :: model
      type(fbc_state_t), intent(inout) :: state
      type(fbc_workspace_t), intent(inout) :: workspace
      integer(int32), intent(in) :: tokens(:), prefix_tokens
      type(fbc_record_metrics_t), intent(out) :: metrics
      integer(int32), intent(out) :: status
      type(fbc_record_metrics_t) :: result
      integer(int64) :: first, last, position, token_size
      integer :: allocation_status
      real(real64) :: loss, cold_sum, prefix_sum
      real(real64), allocatable :: logits(:), hidden(:), dlogits(:)

      metrics = fbc_record_metrics_t()
      result = fbc_record_metrics_t()
      status = FBT_ERR_ARGUMENT
      token_size = size(tokens, kind=int64)
      if (token_size <= 0_int64 .or. token_size >= huge(token_size) - 1_int64 .or. prefix_tokens < 1_int32) return
      if (.not. fbc_state_is_ready(state, model)) then
         status = FBC_ERR_NOT_READY
         return
      end if
      allocate(logits(model%config%vocabulary_size), hidden(model%config%width), &
               dlogits(model%config%vocabulary_size), stat=allocation_status)
      if (allocation_status /= 0) then
         status = FBC_ERR_ALLOCATION
         return
      end if
      cold_sum = 0.0_real64
      prefix_sum = 0.0_real64
      first = 1_int64
      do while (first <= token_size)
         last = first
         do while (last <= token_size)
            if (tokens(last) == 10_int32) exit
            last = last + 1_int64
         end do
         result%records = result%records + 1_int64
         if (last - first > 1_int64) then
            result%scored_records = result%scored_records + 1_int64
            call fbc_state_reset(state, model, status)
            if (status /= FBC_OK) return
            do position = first, last - 2_int64
               call fbc_step(model, state, workspace, tokens(position), hidden, logits, status)
               if (status /= FBC_OK) return
               call fbc_loss_and_gradient(logits, tokens(position + 1_int64), dlogits, loss, status)
               if (status /= FBC_OK) return
               if (loss > 0.0_real64 .and. cold_sum > 0.0_real64) then
                  if (cold_sum >= huge(cold_sum) - loss) then
                     status = FBT_ERR_NUMERIC
                     return
                  end if
               end if
               cold_sum = cold_sum + loss
               result%predictions = result%predictions + 1_int64
               if (position - first < int(prefix_tokens, int64)) then
                  if (loss > 0.0_real64 .and. prefix_sum > 0.0_real64) then
                     if (prefix_sum >= huge(prefix_sum) - loss) then
                        status = FBT_ERR_NUMERIC
                        return
                     end if
                  end if
                  prefix_sum = prefix_sum + loss
                  result%prefix_predictions = result%prefix_predictions + 1_int64
               end if
            end do
         end if
         first = last + 1_int64
      end do
      if (result%predictions == 0_int64 .or. result%prefix_predictions == 0_int64) then
         status = FBT_ERR_ARGUMENT
         return
      end if
      result%loss_sum = cold_sum
      result%prefix_loss_sum = prefix_sum
      result%mean_nll = cold_sum / real(result%predictions, real64)
      result%prefix_mean_nll = prefix_sum / real(result%prefix_predictions, real64)
      metrics = result
      status = FBT_OK
   end subroutine fbc_evaluate_lf_records
end module fixed_budget_natural_eval
