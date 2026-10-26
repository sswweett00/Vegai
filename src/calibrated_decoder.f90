! SPDX-License-Identifier: MIT
!==============================================================================
!  calibrated_decoder.f90
!  Stable temperature calibration and repetition-aware causal decoding.
!
!  This module contains no language-specific hard-coded answers.  It transforms
!  a model-supplied logit vector under explicit, testable policy knobs.  The
!  selected distribution is always normalized after penalties/masks; penalties
!  never get applied after sampling where they would only cosmetically change a
!  displayed response.
!==============================================================================
module calibrated_decoder
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   implicit none
   private

   integer(int32), parameter, public :: DECODER_OK = 0_int32
   integer(int32), parameter, public :: DECODER_ERR_INVALID_ARGUMENT = -900_int32
   integer(int32), parameter, public :: DECODER_ERR_NO_SUPPORT = -901_int32
   integer(int32), parameter, public :: DECODER_ERR_NUMERIC = -902_int32

   integer(int32), parameter, public :: DECODER_MAX_TOP_K = 256_int32
   real(real64), parameter :: NEGATIVE_INFINITY = -huge(0.0_real64) / 8.0_real64

   type, public :: decoding_policy_t
      real(real64) :: temperature = 1.0_real64
      real(real64) :: frequency_penalty = 0.0_real64
      real(real64) :: presence_penalty = 0.0_real64
      real(real64) :: immediate_repeat_penalty = 0.0_real64
      integer(int32) :: no_repeat_ngram = 0_int32
      integer(int32) :: top_k = 0_int32
      real(real64) :: top_p = 1.0_real64
   end type decoding_policy_t

   type, public :: temperature_calibration_t
      real(real64) :: temperature = 1.0_real64
      real(real64) :: mean_nll_before = 0.0_real64
      real(real64) :: mean_nll_after = 0.0_real64
      real(real64) :: inverse_temperature_gradient = 0.0_real64
      integer(int32) :: iterations = 0_int32
      logical :: baseline_retained = .true.
   end type temperature_calibration_t

   public :: decoder_distribution, decoder_select, decoder_temperature_calibrate
   public :: decoder_sequence_has_repeat_ngram

contains

   pure logical function policy_is_valid(policy)
      type(decoding_policy_t), intent(in) :: policy

      policy_is_valid = ieee_is_finite(policy%temperature) .and. policy%temperature > 0.0_real64 .and. &
                        ieee_is_finite(policy%frequency_penalty) .and. policy%frequency_penalty >= 0.0_real64 .and. &
                        ieee_is_finite(policy%presence_penalty) .and. policy%presence_penalty >= 0.0_real64 .and. &
                        ieee_is_finite(policy%immediate_repeat_penalty) .and. &
                        policy%immediate_repeat_penalty >= 0.0_real64 .and. policy%no_repeat_ngram >= 0_int32 .and. &
                        policy%top_k >= 0_int32 .and. policy%top_k <= DECODER_MAX_TOP_K .and. &
                        ieee_is_finite(policy%top_p) .and. policy%top_p > 0.0_real64 .and. policy%top_p <= 1.0_real64
   end function policy_is_valid

   pure integer(int32) function token_frequency(token, history, history_count) result(frequency)
      integer(int32), intent(in) :: token
      integer(int32), intent(in) :: history(:)
      integer(int32), intent(in) :: history_count
      integer(int32) :: position

      frequency = 0_int32
      do position = 1_int32, history_count
         if (history(position) == token) frequency = frequency + 1_int32
      end do
   end function token_frequency

   pure logical function candidate_repeats_ngram(candidate, history, history_count, ngram) result(repeats)
      integer(int32), intent(in) :: candidate
      integer(int32), intent(in) :: history(:)
      integer(int32), intent(in) :: history_count, ngram
      integer(int32) :: start_position, offset
      logical :: matches

      repeats = .false.
      if (ngram <= 1_int32 .or. history_count < ngram - 1_int32) return
      do start_position = 1_int32, history_count - ngram + 1_int32
         matches = .true.
         do offset = 0_int32, ngram - 2_int32
            if (history(start_position + offset) /= history(history_count - (ngram - 2_int32) + offset)) then
               matches = .false.
               exit
            end if
         end do
         if (matches .and. history(start_position + ngram - 1_int32) == candidate) then
            repeats = .true.
            return
         end if
      end do
   end function candidate_repeats_ngram

   pure logical function decoder_sequence_has_repeat_ngram(history, history_count, ngram) result(repeats)
      integer(int32), intent(in) :: history(:)
      integer(int32), intent(in) :: history_count, ngram
      integer(int32) :: end_position

      repeats = .false.
      if (ngram <= 1_int32 .or. history_count < ngram) return
      do end_position = ngram, history_count
         if (candidate_repeats_ngram(history(end_position), history, end_position - 1_int32, ngram)) then
            repeats = .true.
            return
         end if
      end do
   end function decoder_sequence_has_repeat_ngram

   subroutine decoder_distribution(logits, history, history_count, policy, probabilities, status)
      real(real64), intent(in) :: logits(:)
      integer(int32), intent(in) :: history(:)
      integer(int32), intent(in) :: history_count
      type(decoding_policy_t), intent(in) :: policy
      real(real64), intent(out) :: probabilities(:)
      integer(int32), intent(out) :: status
      real(real64) :: adjusted(size(logits)), maximum, normalizer
      integer(int32) :: token, frequency
      logical :: any_support

      probabilities = 0.0_real64
      status = DECODER_OK
      if (size(logits) <= 0 .or. size(probabilities) /= size(logits) .or. history_count < 0_int32 .or. &
          history_count > size(history, kind=int32) .or. .not. policy_is_valid(policy)) then
         status = DECODER_ERR_INVALID_ARGUMENT
         return
      end if
      maximum = NEGATIVE_INFINITY
      any_support = .false.
      do token = 1_int32, size(logits, kind=int32)
         if (.not. ieee_is_finite(logits(token))) then
            adjusted(token) = NEGATIVE_INFINITY
            cycle
         end if
         if (policy%no_repeat_ngram > 1_int32 .and. &
             candidate_repeats_ngram(token, history, history_count, policy%no_repeat_ngram)) then
            adjusted(token) = NEGATIVE_INFINITY
            cycle
         end if
         frequency = token_frequency(token, history, history_count)
         adjusted(token) = logits(token) / policy%temperature - &
                           policy%frequency_penalty * real(frequency, real64)
         if (frequency > 0_int32) adjusted(token) = adjusted(token) - policy%presence_penalty
         if (history_count > 0_int32 .and. token == history(history_count)) then
            adjusted(token) = adjusted(token) - policy%immediate_repeat_penalty
         end if
         if (.not. ieee_is_finite(adjusted(token))) then
            adjusted(token) = NEGATIVE_INFINITY
            cycle
         end if
         maximum = max(maximum, adjusted(token))
         any_support = .true.
      end do
      if (.not. any_support) then
         status = DECODER_ERR_NO_SUPPORT
         return
      end if
      normalizer = 0.0_real64
      do token = 1_int32, size(logits, kind=int32)
         if (adjusted(token) <= NEGATIVE_INFINITY / 2.0_real64) cycle
         probabilities(token) = exp(adjusted(token) - maximum)
         normalizer = normalizer + probabilities(token)
      end do
      if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_real64) then
         probabilities = 0.0_real64
         status = DECODER_ERR_NUMERIC
         return
      end if
      probabilities = probabilities / normalizer
      if (.not. all(ieee_is_finite(probabilities))) then
         probabilities = 0.0_real64
         status = DECODER_ERR_NUMERIC
      end if
   end subroutine decoder_distribution

   subroutine insert_top_candidate(token, probability, top_tokens, top_probabilities, selected_count)
      integer(int32), intent(in) :: token
      real(real64), intent(in) :: probability
      integer(int32), intent(inout) :: top_tokens(:)
      real(real64), intent(inout) :: top_probabilities(:)
      integer(int32), intent(inout) :: selected_count
      integer(int32) :: position, limit

      if (probability <= 0.0_real64) return
      limit = size(top_tokens, kind=int32)
      if (selected_count < limit) then
         selected_count = selected_count + 1_int32
         position = selected_count
      else
         if (probability < top_probabilities(limit)) return
         if (probability == top_probabilities(limit) .and. token > top_tokens(limit)) return
         position = limit
      end if
      do while (position > 1_int32)
         if (probability < top_probabilities(position - 1_int32)) exit
         if (probability == top_probabilities(position - 1_int32) .and. token > top_tokens(position - 1_int32)) exit
         if (position <= limit) then
            top_tokens(position) = top_tokens(position - 1_int32)
            top_probabilities(position) = top_probabilities(position - 1_int32)
         end if
         position = position - 1_int32
      end do
      top_tokens(position) = token
      top_probabilities(position) = probability
   end subroutine insert_top_candidate

   subroutine next_random_unit(state, uniform)
      integer(int64), intent(inout) :: state
      real(real64), intent(out) :: uniform
      integer(int64) :: nonnegative

      state = ieor(state, ishftc(state, 13))
      state = ieor(state, shiftr(state, 7))
      state = ieor(state, ishftc(state, 17))
      nonnegative = iand(state, huge(0_int64))
      uniform = real(nonnegative, real64) / real(huge(0_int64), real64)
   end subroutine next_random_unit

   subroutine decoder_select(probabilities, policy, random_state, selected_token, status)
      real(real64), intent(in) :: probabilities(:)
      type(decoding_policy_t), intent(in) :: policy
      integer(int64), intent(inout) :: random_state
      integer(int32), intent(out) :: selected_token
      integer(int32), intent(out) :: status
      integer(int32) :: top_tokens(DECODER_MAX_TOP_K), selected_count, token, limit, index
      real(real64) :: top_probabilities(DECODER_MAX_TOP_K), cumulative, cutoff, uniform, running

      selected_token = 0_int32
      status = DECODER_OK
      if (size(probabilities) <= 0 .or. .not. policy_is_valid(policy) .or. any(probabilities < 0.0_real64) .or. &
          .not. all(ieee_is_finite(probabilities))) then
         status = DECODER_ERR_INVALID_ARGUMENT
         return
      end if
      if (sum(probabilities) <= 0.0_real64) then
         status = DECODER_ERR_NO_SUPPORT
         return
      end if
      ! Exact whole-distribution sampling is O(V) and is the fast path.  A
      ! top-p request explicitly requires a bounded top-k candidate heap so a
      ! 100k-token vocabulary never causes an O(V^2) full sort.
      if (policy%top_p >= 1.0_real64 .and. policy%top_k == 0_int32) then
         call next_random_unit(random_state, uniform)
         running = 0.0_real64
         do token = 1_int32, size(probabilities, kind=int32)
            running = running + probabilities(token)
            if (uniform <= running) then
               selected_token = token
               return
            end if
         end do
         selected_token = maxloc(probabilities, dim=1, kind=int32)
         return
      end if
      if (policy%top_k <= 0_int32) then
         status = DECODER_ERR_INVALID_ARGUMENT
         return
      end if
      limit = min(policy%top_k, size(probabilities, kind=int32))
      top_tokens = 0_int32
      top_probabilities = 0.0_real64
      selected_count = 0_int32
      do token = 1_int32, size(probabilities, kind=int32)
         call insert_top_candidate(token, probabilities(token), top_tokens(:limit), top_probabilities(:limit), selected_count)
      end do
      if (selected_count <= 0_int32) then
         status = DECODER_ERR_NO_SUPPORT
         return
      end if
      cutoff = min(policy%top_p, sum(top_probabilities(:selected_count)))
      cumulative = 0.0_real64
      do index = 1_int32, selected_count
         cumulative = cumulative + top_probabilities(index)
         if (cumulative >= cutoff) exit
      end do
      ! A completed DO loop leaves INDEX one past the last retained candidate.
      ! Rounding can make the incremental sum fall a few ulps short of CUTOFF
      ! even though it equals the total mass, so clamp here instead of letting
      ! the sampling pass and the fallback read one element past the prefix.
      if (index > selected_count) index = selected_count
      selected_count = index
      if (cumulative <= 0.0_real64) then
         status = DECODER_ERR_NO_SUPPORT
         return
      end if
      call next_random_unit(random_state, uniform)
      uniform = uniform * cumulative
      running = 0.0_real64
      do index = 1_int32, selected_count
         running = running + top_probabilities(index)
         if (uniform <= running) then
            selected_token = top_tokens(index)
            return
         end if
      end do
      selected_token = top_tokens(selected_count)
   end subroutine decoder_select

   subroutine nll_and_gradient(logits, targets, case_count, inverse_temperature, mean_nll, gradient, status)
      real(real64), intent(in) :: logits(:, :)
      integer(int32), intent(in) :: targets(:)
      integer(int32), intent(in) :: case_count
      real(real64), intent(in) :: inverse_temperature
      real(real64), intent(out) :: mean_nll, gradient
      integer(int32), intent(out) :: status
      integer(int32) :: case_index, token, target
      real(real64) :: maximum, normalizer, expectation, scaled

      mean_nll = 0.0_real64
      gradient = 0.0_real64
      status = DECODER_OK
      if (case_count <= 0_int32 .or. case_count > size(logits, 2, kind=int32) .or. &
          case_count > size(targets, kind=int32) .or. inverse_temperature <= 0.0_real64 .or. &
          .not. ieee_is_finite(inverse_temperature)) then
         status = DECODER_ERR_INVALID_ARGUMENT
         return
      end if
      do case_index = 1_int32, case_count
         target = targets(case_index)
         if (target < 1_int32 .or. target > size(logits, 1, kind=int32)) then
            status = DECODER_ERR_INVALID_ARGUMENT
            return
         end if
         maximum = NEGATIVE_INFINITY
         do token = 1_int32, size(logits, 1, kind=int32)
            if (.not. ieee_is_finite(logits(token, case_index))) then
               status = DECODER_ERR_NUMERIC
               return
            end if
            maximum = max(maximum, inverse_temperature * logits(token, case_index))
         end do
         normalizer = 0.0_real64
         expectation = 0.0_real64
         do token = 1_int32, size(logits, 1, kind=int32)
            scaled = exp(inverse_temperature * logits(token, case_index) - maximum)
            normalizer = normalizer + scaled
            expectation = expectation + scaled * logits(token, case_index)
         end do
         if (normalizer <= 0.0_real64 .or. .not. ieee_is_finite(normalizer)) then
            status = DECODER_ERR_NUMERIC
            return
         end if
         mean_nll = mean_nll + log(normalizer) + maximum - inverse_temperature * logits(target, case_index)
         gradient = gradient + expectation / normalizer - logits(target, case_index)
      end do
      mean_nll = mean_nll / real(case_count, real64)
      gradient = gradient / real(case_count, real64)
   end subroutine nll_and_gradient

   subroutine decoder_temperature_calibrate(logits, targets, case_count, calibration, status)
      ! NLL is convex in inverse temperature a > 0:
      ! L(a)=mean(logsumexp(a z)-a z_y), L''(a)=mean(Var_p(z)) >= 0.
      ! Bisection on L'(a) therefore finds the bounded global minimizer up to
      ! the explicit numerical tolerance.  Temperature 1 is retained whenever
      ! finite precision would make the fitted point worse.
      real(real64), intent(in) :: logits(:, :)
      integer(int32), intent(in) :: targets(:)
      integer(int32), intent(in) :: case_count
      type(temperature_calibration_t), intent(out) :: calibration
      integer(int32), intent(out) :: status
      real(real64), parameter :: inverse_lower = 0.20_real64 ! T=5
      real(real64), parameter :: inverse_upper = 20.0_real64 ! T=0.05
      integer(int32), parameter :: maximum_iterations = 80_int32
      real(real64) :: lower, upper, middle, nll_lower, gradient_lower, nll_upper, gradient_upper
      real(real64) :: nll_middle, gradient_middle, nll_baseline, gradient_baseline, selected_inverse, nll_selected
      integer(int32) :: iteration, local_status

      calibration = temperature_calibration_t()
      call nll_and_gradient(logits, targets, case_count, 1.0_real64, nll_baseline, gradient_baseline, local_status)
      if (local_status /= DECODER_OK) then
         status = local_status
         return
      end if
      call nll_and_gradient(logits, targets, case_count, inverse_lower, nll_lower, gradient_lower, local_status)
      if (local_status /= DECODER_OK) then; status = local_status; return; end if
      call nll_and_gradient(logits, targets, case_count, inverse_upper, nll_upper, gradient_upper, local_status)
      if (local_status /= DECODER_OK) then; status = local_status; return; end if
      lower = inverse_lower
      upper = inverse_upper
      if (gradient_lower >= 0.0_real64) then
         selected_inverse = lower
         nll_selected = nll_lower
         gradient_middle = gradient_lower
         iteration = 0_int32
      else if (gradient_upper <= 0.0_real64) then
         selected_inverse = upper
         nll_selected = nll_upper
         gradient_middle = gradient_upper
         iteration = 0_int32
      else
         do iteration = 1_int32, maximum_iterations
            middle = 0.5_real64 * (lower + upper)
            call nll_and_gradient(logits, targets, case_count, middle, nll_middle, gradient_middle, local_status)
            if (local_status /= DECODER_OK) then; status = local_status; return; end if
            if (gradient_middle < 0.0_real64) then
               lower = middle
            else
               upper = middle
            end if
            if (upper - lower <= 1.0e-10_real64 * max(1.0_real64, middle)) exit
         end do
         selected_inverse = 0.5_real64 * (lower + upper)
         call nll_and_gradient(logits, targets, case_count, selected_inverse, nll_selected, gradient_middle, local_status)
         if (local_status /= DECODER_OK) then; status = local_status; return; end if
      end if
      calibration%mean_nll_before = nll_baseline
      calibration%iterations = iteration
      calibration%baseline_retained = nll_selected > nll_baseline
      if (calibration%baseline_retained) then
         calibration%temperature = 1.0_real64
         calibration%mean_nll_after = nll_baseline
         calibration%inverse_temperature_gradient = gradient_baseline
      else
         calibration%temperature = 1.0_real64 / selected_inverse
         calibration%mean_nll_after = nll_selected
         calibration%inverse_temperature_gradient = gradient_middle
      end if
      status = DECODER_OK
   end subroutine decoder_temperature_calibrate

end module calibrated_decoder
