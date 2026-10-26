! Reference-only vector optimizer variants for controlled benchmark tasks.
! These routines are isolated from the production Apex module.
module ref_vector_optimizers
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   implicit none
   private

   integer(int32), parameter, public :: REF_SGD = 1_int32
   integer(int32), parameter, public :: REF_MOMENTUM = 2_int32
   integer(int32), parameter, public :: REF_NESTEROV = 3_int32
   integer(int32), parameter, public :: REF_ADAGRAD = 4_int32
   integer(int32), parameter, public :: REF_RMSPROP = 5_int32
   integer(int32), parameter, public :: REF_ADAM = 6_int32
   integer(int32), parameter, public :: REF_ADAMW = 7_int32
   integer(int32), parameter, public :: REF_NADAM = 8_int32
   integer(int32), parameter, public :: REF_RADAM = 9_int32
   integer(int32), parameter, public :: REF_LION = 10_int32
   integer(int32), parameter, public :: REF_ADAFACTOR_VECTOR = 11_int32
   integer(int32), parameter, public :: REF_LAMB = 12_int32
   integer(int32), parameter, public :: REF_LARS = 13_int32
   integer(int32), parameter, public :: REF_SOPHIA_G = 14_int32
   integer(int32), parameter, public :: REF_PRODIGY = 15_int32
   integer(int32), parameter, public :: REF_SCHEDULE_FREE_ADAMW = 16_int32
   integer(int32), parameter, public :: REF_OK = 0_int32
   integer(int32), parameter, public :: REF_INVALID = 1_int32
   integer(int32), parameter, public :: REF_ALLOCATION = 2_int32
   integer(int32), parameter, public :: REF_NUMERICAL = 3_int32

   type, public :: ref_vector_state_t
      private
      integer(int32) :: kind = 0_int32
      integer(int64) :: step = 0_int64
      real(real64) :: nadam_mu_product = 1.0_real64
      real(real64) :: prodigy_d = 1.0e-6_real64
      real(real64) :: prodigy_dmax = 1.0e-6_real64
      real(real64) :: prodigy_numerator = 0.0_real64
      real(real64) :: schedulefree_weight_sum = 0.0_real64
      real(real64) :: schedulefree_lr_max = 0.0_real64
      real(real64), allocatable :: first(:)
      real(real64), allocatable :: second(:)
      real(real64), allocatable :: third(:)
      real(real64), allocatable :: fourth(:)
   end type ref_vector_state_t

   public :: ref_vector_initialize, ref_vector_step, ref_vector_state_bytes, ref_vector_name, &
             ref_vector_eval_parameters

contains

   subroutine ref_vector_initialize(kind, parameter_count, state, status)
      integer(int32), intent(in) :: kind
      integer(int64), intent(in) :: parameter_count
      type(ref_vector_state_t), intent(out) :: state
      integer(int32), intent(out) :: status
      integer :: allocation_status

      status = REF_OK
      if (kind < REF_SGD .or. kind > REF_SCHEDULE_FREE_ADAMW .or. parameter_count <= 0_int64) then
         status = REF_INVALID
         return
      end if
      state%kind = kind
      state%step = 0_int64
      select case (kind)
      case (REF_MOMENTUM, REF_NESTEROV, REF_LION, REF_LARS)
         allocate(state%first(parameter_count), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_ALLOCATION
            return
         end if
         state%first = 0.0_real64
      case (REF_ADAGRAD, REF_RMSPROP, REF_ADAFACTOR_VECTOR)
         allocate(state%second(parameter_count), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_ALLOCATION
            return
         end if
         state%second = 0.0_real64
      case (REF_ADAM, REF_ADAMW, REF_NADAM, REF_RADAM, REF_LAMB, REF_SOPHIA_G)
         allocate(state%first(parameter_count), state%second(parameter_count), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_ALLOCATION
            return
         end if
         state%first = 0.0_real64
         state%second = 0.0_real64
      case (REF_PRODIGY)
         ! Original Prodigy state: first/second are Adam moments, third is
         ! the D-adaptation accumulator, fourth is the initial parameter copy.
         allocate(state%first(parameter_count), state%second(parameter_count), &
                  state%third(parameter_count), state%fourth(parameter_count), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_ALLOCATION
            return
         end if
         state%first = 0.0_real64
         state%second = 0.0_real64
         state%third = 0.0_real64
         state%fourth = 0.0_real64
         state%prodigy_d = 1.0e-6_real64
         state%prodigy_dmax = state%prodigy_d
         state%prodigy_numerator = 0.0_real64
      case (REF_SCHEDULE_FREE_ADAMW)
         ! Schedule-Free AdamW (AdamW base with no inner first moment):
         ! first stores z, second stores the second-moment accumulator.
         allocate(state%first(parameter_count), state%second(parameter_count), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_ALLOCATION
            return
         end if
         state%first = 0.0_real64
         state%second = 0.0_real64
         state%schedulefree_weight_sum = 0.0_real64
         state%schedulefree_lr_max = 0.0_real64
      case (REF_SGD)
         ! Plain SGD has no per-parameter optimizer state.
      end select
   end subroutine ref_vector_initialize

   subroutine ref_vector_step(parameters, gradient, state, learning_rate, beta1, beta2, epsilon, &
                              weight_decay, status, curvature_diagonal, rho)
      real(real64), contiguous, intent(inout) :: parameters(:)
      real(real64), contiguous, intent(in) :: gradient(:)
      type(ref_vector_state_t), intent(inout) :: state
      real(real64), intent(in) :: learning_rate, beta1, beta2, epsilon, weight_decay
      integer(int32), intent(out) :: status
      real(real64), contiguous, optional, intent(in) :: curvature_diagonal(:)
      real(real64), optional, intent(in) :: rho
      integer :: i
      real(real64) :: beta1_power, beta2_power, mhat, vhat, delta, rho_inf, rho_t
      real(real64) :: rectification, parameter_norm, update_norm, trust_ratio, adjusted_norm
      real(real64) :: beta2_t, update_rms, clip_scale, sign_input, g_eff, gradient_norm
      real(real64) :: mu_t, mu_next, mu_product_t, mu_product_next
      real(real64) :: beta3, prodigy_denom, prodigy_dhat, prodigy_dlr, prodigy_delta_num
      real(real64) :: sf_lr, sf_weight, sf_c, sf_bias2, sf_denom, sf_gradient

      status = REF_OK
      if (size(parameters) /= size(gradient) .or. size(parameters) == 0 .or. state%kind == 0_int32 .or. &
          .not. ieee_is_finite(learning_rate) .or. learning_rate <= 0.0_real64 .or. &
          .not. ieee_is_finite(beta1) .or. beta1 < 0.0_real64 .or. beta1 >= 1.0_real64 .or. &
          .not. ieee_is_finite(beta2) .or. beta2 < 0.0_real64 .or. beta2 >= 1.0_real64 .or. &
          .not. ieee_is_finite(epsilon) .or. epsilon <= 0.0_real64 .or. &
          .not. ieee_is_finite(weight_decay) .or. weight_decay < 0.0_real64) then
         status = REF_INVALID
         return
      end if
      select case (state%kind)
      case (REF_MOMENTUM, REF_NESTEROV, REF_LION, REF_LARS)
         if (.not. allocated(state%first)) then
            status = REF_INVALID
            return
         end if
      case (REF_ADAGRAD, REF_RMSPROP, REF_ADAFACTOR_VECTOR)
         if (.not. allocated(state%second)) then
            status = REF_INVALID
            return
         end if
      case (REF_ADAM, REF_ADAMW, REF_NADAM, REF_RADAM, REF_LAMB, REF_SOPHIA_G, &
            REF_SCHEDULE_FREE_ADAMW)
         if (.not. allocated(state%first) .or. .not. allocated(state%second)) then
            status = REF_INVALID
            return
         end if
      case (REF_PRODIGY)
         if (.not. allocated(state%first) .or. .not. allocated(state%second) .or. &
             .not. allocated(state%third) .or. .not. allocated(state%fourth)) then
            status = REF_INVALID
            return
         end if
      case (REF_SGD)
         ! No optimizer arrays are needed.
      case default
         status = REF_INVALID
         return
      end select
      if (allocated(state%first)) then
         if (size(state%first) /= size(parameters)) then
            status = REF_INVALID
            return
         end if
      end if
      if (allocated(state%second)) then
         if (size(state%second) /= size(parameters)) then
            status = REF_INVALID
            return
         end if
      end if
      if (allocated(state%third)) then
         if (size(state%third) /= size(parameters)) then
            status = REF_INVALID
            return
         end if
      end if
      if (allocated(state%fourth)) then
         if (size(state%fourth) /= size(parameters)) then
            status = REF_INVALID
            return
         end if
      end if
      if (state%kind == REF_SCHEDULE_FREE_ADAMW .and. beta1 <= 0.0_real64) then
         status = REF_INVALID
         return
      end if
      if (state%kind == REF_SOPHIA_G) then
         if (.not. present(curvature_diagonal) .or. .not. present(rho)) then
            status = REF_INVALID
            return
         end if
         if (size(curvature_diagonal) /= size(parameters) .or. .not. ieee_is_finite(rho) .or. rho <= 0.0_real64) then
            status = REF_INVALID
            return
         end if
         if (.not. all(ieee_is_finite(curvature_diagonal)) .or. any(curvature_diagonal < 0.0_real64)) then
            status = REF_NUMERICAL
            return
         end if
      end if
      if (.not. all(ieee_is_finite(parameters)) .or. .not. all(ieee_is_finite(gradient))) then
         status = REF_NUMERICAL
         return
      end if
      if (state%step == huge(state%step)) then
         status = REF_INVALID
         return
      end if
      if (state%kind == REF_PRODIGY .and. state%step == 0_int64) state%fourth = parameters
      if (state%kind == REF_SCHEDULE_FREE_ADAMW .and. state%step == 0_int64) state%first = parameters
      state%step = state%step + 1_int64
      beta1_power = beta1**real(state%step, real64)
      beta2_power = beta2**real(state%step, real64)
      if (state%kind == REF_NADAM) then
         mu_t = beta1 * (1.0_real64 - 0.5_real64 * 0.96_real64**(0.004_real64 * real(state%step, real64)))
         mu_next = beta1 * (1.0_real64 - 0.5_real64 * &
                            0.96_real64**(0.004_real64 * real(state%step + 1_int64, real64)))
         mu_product_t = state%nadam_mu_product * mu_t
         mu_product_next = mu_product_t * mu_next
      end if

      select case (state%kind)
      case (REF_SOPHIA_G)
         ! Sophia-G (Liu et al., "Sophia: A Scalable Stochastic Second-order
         ! Optimizer for Language Model Pre-training", arXiv:2305.14342): use
         ! exact task-supplied diagonal curvature, update k=1, and clip m/(rho*h).
         do i = 1, size(parameters)
            state%first(i) = beta1 * state%first(i) + (1.0_real64 - beta1) * gradient(i)
            state%second(i) = beta2 * state%second(i) + &
                              (1.0_real64 - beta2) * curvature_diagonal(i)
            delta = state%first(i) / (rho * state%second(i) + epsilon)
            delta = max(-1.0_real64, min(1.0_real64, delta))
            parameters(i) = (1.0_real64 - learning_rate * weight_decay) * parameters(i) - &
                            learning_rate * delta
         end do

      case (REF_PRODIGY)
         ! Mishchenko & Defazio, "Prodigy: An Expeditiously Adaptive Parameter-Free
         ! Learner", arXiv:2306.06101; mirrors prodigy_reference.py at
         ! https://github.com/konstmish/prodigy, single group, beta3=sqrt(beta2),
         ! no bias correction, D-adaptation on all flattened coordinates.
         beta3 = sqrt(beta2)
         prodigy_dlr = state%prodigy_d * learning_rate
         state%prodigy_numerator = beta3 * state%prodigy_numerator
         prodigy_delta_num = 0.0_real64
         do i = 1, size(parameters)
            prodigy_delta_num = prodigy_delta_num + (state%prodigy_d / 1.0e-6_real64) * prodigy_dlr * &
                                gradient(i) * (state%fourth(i) - parameters(i))
            state%first(i) = beta1 * state%first(i) + state%prodigy_d * (1.0_real64 - beta1) * gradient(i)
            state%second(i) = beta2 * state%second(i) + state%prodigy_d**2 * &
                              (1.0_real64 - beta2) * gradient(i) * gradient(i)
            state%third(i) = beta3 * state%third(i) + &
                             (state%prodigy_d / 1.0e-6_real64) * prodigy_dlr * gradient(i)
         end do
         state%prodigy_numerator = state%prodigy_numerator + prodigy_delta_num
         prodigy_denom = sum(abs(state%third))
         if (prodigy_denom > 0.0_real64) then
            prodigy_dhat = (state%prodigy_numerator / prodigy_denom)
            if (state%prodigy_d == 1.0e-6_real64) state%prodigy_d = max(state%prodigy_d, prodigy_dhat)
            state%prodigy_dmax = max(state%prodigy_dmax, prodigy_dhat)
            state%prodigy_d = state%prodigy_dmax
            do i = 1, size(parameters)
               if (weight_decay > 0.0_real64) parameters(i) = &
                  (1.0_real64 - weight_decay * prodigy_dlr) * parameters(i)
               parameters(i) = parameters(i) - prodigy_dlr * state%first(i) / &
                               (sqrt(state%second(i)) + state%prodigy_d * epsilon)
            end do
         end if

      case (REF_SCHEDULE_FREE_ADAMW)
         ! Defazio et al., "The Road Less Scheduled", NeurIPS 2024,
         ! arXiv:2405.15682; follows facebookresearch/schedule_free's
         ! AdamWScheduleFree, r=0, weight_lr_power=2, no inner EMA.
         sf_lr = learning_rate
         state%schedulefree_lr_max = max(state%schedulefree_lr_max, sf_lr)
         sf_weight = state%schedulefree_lr_max**2
         state%schedulefree_weight_sum = state%schedulefree_weight_sum + sf_weight
         sf_c = sf_weight / state%schedulefree_weight_sum
         sf_bias2 = 1.0_real64 - beta2**real(state%step, real64)
         do i = 1, size(parameters)
            state%second(i) = beta2 * state%second(i) + (1.0_real64 - beta2) * gradient(i) * gradient(i)
            sf_denom = sqrt(state%second(i) / sf_bias2) + epsilon
            sf_gradient = gradient(i) / sf_denom + weight_decay * parameters(i)
            parameters(i) = (1.0_real64 - sf_c) * parameters(i) + sf_c * state%first(i) + &
                            sf_lr * (beta1 * (1.0_real64 - sf_c) - 1.0_real64) * sf_gradient
            state%first(i) = state%first(i) - sf_lr * sf_gradient
         end do

      case (REF_SGD)
         do i = 1, size(parameters)
            g_eff = gradient(i) + weight_decay * parameters(i)
            parameters(i) = parameters(i) - learning_rate * g_eff
         end do

      case (REF_MOMENTUM)
         ! Classical heavy-ball method (Polyak, USSR Computational Mathematics
         ! and Mathematical Physics 4(5), 1964): m_t = beta*m_(t-1) + g_t.
         do i = 1, size(parameters)
            g_eff = gradient(i) + weight_decay * parameters(i)
            state%first(i) = beta1 * state%first(i) + g_eff
            parameters(i) = parameters(i) - learning_rate * state%first(i)
         end do

      case (REF_NESTEROV)
         ! Nesterov accelerated gradient (Nesterov, 1983; Sutskever et al.,
         ! "On the Importance of Initialization and Momentum in Deep Learning", ICML 2013).
         do i = 1, size(parameters)
            g_eff = gradient(i) + weight_decay * parameters(i)
            state%first(i) = beta1 * state%first(i) + g_eff
            parameters(i) = parameters(i) - learning_rate * (g_eff + beta1 * state%first(i))
         end do

      case (REF_ADAGRAD)
         ! Duchi, Hazan & Singer, JMLR 12 (2011), 2121–2159,
         ! "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization".
         do i = 1, size(parameters)
            g_eff = gradient(i) + weight_decay * parameters(i)
            state%second(i) = state%second(i) + g_eff * g_eff
            parameters(i) = parameters(i) - learning_rate * g_eff / (sqrt(state%second(i)) + epsilon)
         end do

      case (REF_RMSPROP)
         ! Tieleman & Hinton, Neural Networks for Machine Learning, lecture 6.5
         ! (2012), uncentered exponential average of squared gradients.
         do i = 1, size(parameters)
            g_eff = gradient(i) + weight_decay * parameters(i)
            state%second(i) = beta2 * state%second(i) + (1.0_real64 - beta2) * g_eff * g_eff
            parameters(i) = parameters(i) - learning_rate * g_eff / (sqrt(state%second(i)) + epsilon)
         end do

      case (REF_ADAM, REF_ADAMW, REF_LAMB, REF_NADAM, REF_RADAM)
         ! Adam: Kingma & Ba (2015), arXiv:1412.6980. AdamW: Loshchilov &
         ! Hutter (2019), arXiv:1711.05101. RAdam: Liu et al. (2020),
         ! arXiv:1908.03265. LAMB: You et al. (2020), arXiv:1904.00962.
         do i = 1, size(parameters)
            g_eff = gradient(i)
            if (state%kind /= REF_ADAMW .and. state%kind /= REF_LAMB) &
               g_eff = g_eff + weight_decay * parameters(i)
            state%first(i) = beta1 * state%first(i) + (1.0_real64 - beta1) * g_eff
            state%second(i) = beta2 * state%second(i) + (1.0_real64 - beta2) * g_eff * g_eff
         end do
         select case (state%kind)
         case (REF_ADAM, REF_ADAMW)
            do i = 1, size(parameters)
               mhat = state%first(i) / (1.0_real64 - beta1_power)
               vhat = state%second(i) / (1.0_real64 - beta2_power)
               delta = mhat / (sqrt(vhat) + epsilon)
               if (state%kind == REF_ADAMW) then
                  parameters(i) = (1.0_real64 - learning_rate * weight_decay) * parameters(i) - learning_rate * delta
               else
                  parameters(i) = parameters(i) - learning_rate * delta
               end if
            end do
         case (REF_NADAM)
            ! Dozat (2016), "Incorporating Nesterov Momentum into Adam", with
            ! the published scheduled-momentum product correction.
            do i = 1, size(parameters)
               g_eff = gradient(i) + weight_decay * parameters(i)
               mhat = (1.0_real64 - mu_t) * g_eff / (1.0_real64 - mu_product_next) + &
                      mu_next * state%first(i) / (1.0_real64 - mu_product_t)
               vhat = state%second(i) / (1.0_real64 - beta2_power)
               parameters(i) = parameters(i) - learning_rate * mhat / (sqrt(vhat) + epsilon)
            end do
            state%nadam_mu_product = mu_product_t
         case (REF_RADAM)
            rho_inf = 2.0_real64 / (1.0_real64 - beta2) - 1.0_real64
            rho_t = rho_inf - 2.0_real64 * real(state%step, real64) * beta2_power / (1.0_real64 - beta2_power)
            if (rho_t > 4.0_real64) then
               rectification = sqrt(((rho_t - 4.0_real64) * (rho_t - 2.0_real64) * rho_inf) / &
                                    ((rho_inf - 4.0_real64) * (rho_inf - 2.0_real64) * rho_t))
               do i = 1, size(parameters)
                  mhat = state%first(i) / (1.0_real64 - beta1_power)
                  vhat = state%second(i) / (1.0_real64 - beta2_power)
                  parameters(i) = parameters(i) - learning_rate * rectification * mhat / (sqrt(vhat) + epsilon)
               end do
            else
               do i = 1, size(parameters)
                  mhat = state%first(i) / (1.0_real64 - beta1_power)
                  parameters(i) = parameters(i) - learning_rate * mhat
               end do
            end if
         case (REF_LAMB)
            parameter_norm = sqrt(sum(parameters * parameters))
            update_norm = 0.0_real64
            do i = 1, size(parameters)
               mhat = state%first(i) / (1.0_real64 - beta1_power)
               vhat = state%second(i) / (1.0_real64 - beta2_power)
               delta = mhat / (sqrt(vhat) + epsilon) + weight_decay * parameters(i)
               update_norm = update_norm + delta * delta
            end do
            update_norm = sqrt(update_norm)
            trust_ratio = 1.0_real64
            if (parameter_norm > 0.0_real64 .and. update_norm > 0.0_real64) &
               trust_ratio = parameter_norm / update_norm
            do i = 1, size(parameters)
               mhat = state%first(i) / (1.0_real64 - beta1_power)
               vhat = state%second(i) / (1.0_real64 - beta2_power)
               delta = mhat / (sqrt(vhat) + epsilon) + weight_decay * parameters(i)
               parameters(i) = parameters(i) - learning_rate * trust_ratio * delta
            end do
         end select

      case (REF_LION)
         ! Chen et al., "Symbolic Discovery of Optimization Algorithms" (2023),
         ! arXiv:2302.06675; sign update with decoupled weight decay.
         do i = 1, size(parameters)
            sign_input = beta1 * state%first(i) + (1.0_real64 - beta1) * gradient(i)
            delta = sign(1.0_real64, sign_input)
            if (sign_input == 0.0_real64) delta = 0.0_real64
            parameters(i) = (1.0_real64 - learning_rate * weight_decay) * parameters(i) - learning_rate * delta
            state%first(i) = beta2 * state%first(i) + (1.0_real64 - beta2) * gradient(i)
         end do

      case (REF_LARS)
         ! You et al. (2017), "Large Batch Training of Convolutional Networks
         ! by Layer-wise Adaptive Rate Scaling", arXiv:1708.03888. The vector is
         ! one complete layer; trust coefficient is fixed at 1e-3.
         parameter_norm = norm2(parameters)
         adjusted_norm = 0.0_real64
         do i = 1, size(parameters)
            g_eff = gradient(i) + weight_decay * parameters(i)
            adjusted_norm = adjusted_norm + g_eff * g_eff
         end do
         gradient_norm = sqrt(adjusted_norm)
         trust_ratio = 1.0_real64
         if (parameter_norm > 0.0_real64 .and. gradient_norm > 0.0_real64) &
            trust_ratio = 1.0e-3_real64 * parameter_norm / (gradient_norm + epsilon)
         do i = 1, size(parameters)
            g_eff = gradient(i) + weight_decay * parameters(i)
            state%first(i) = beta1 * state%first(i) + trust_ratio * g_eff
            parameters(i) = parameters(i) - learning_rate * state%first(i)
         end do

      case (REF_ADAFACTOR_VECTOR)
         ! Shazeer & Stern (2018), "Adafactor: Adaptive Learning Rates with
         ! Sublinear Memory Cost", arXiv:1804.04235. This is the unfactored
         ! vector mode; relative-step mode is disabled and beta2_t=1-t**(-0.8).
         beta2_t = 1.0_real64 - real(state%step, real64)**(-0.8_real64)
         do i = 1, size(parameters)
            g_eff = gradient(i) + weight_decay * parameters(i)
            state%second(i) = beta2_t * state%second(i) + (1.0_real64 - beta2_t) * g_eff * g_eff
         end do
         update_rms = 0.0_real64
         do i = 1, size(parameters)
            delta = (gradient(i) + weight_decay * parameters(i)) / (sqrt(state%second(i)) + epsilon)
            update_rms = update_rms + delta * delta
         end do
         update_rms = sqrt(update_rms / real(size(parameters), real64))
         clip_scale = min(1.0_real64, 1.0_real64 / max(update_rms, tiny(1.0_real64)))
         do i = 1, size(parameters)
            delta = (gradient(i) + weight_decay * parameters(i)) / (sqrt(state%second(i)) + epsilon)
            parameters(i) = parameters(i) - learning_rate * clip_scale * delta
         end do
      end select

      if (.not. all(ieee_is_finite(parameters))) then
         status = REF_NUMERICAL
         return
      end if
      if (allocated(state%first)) then
         if (.not. all(ieee_is_finite(state%first))) status = REF_NUMERICAL
      end if
      if (allocated(state%second)) then
         if (.not. all(ieee_is_finite(state%second))) status = REF_NUMERICAL
      end if
      if (allocated(state%third)) then
         if (.not. all(ieee_is_finite(state%third))) status = REF_NUMERICAL
      end if
      if (allocated(state%fourth)) then
         if (.not. all(ieee_is_finite(state%fourth))) status = REF_NUMERICAL
      end if
      if (state%kind == REF_PRODIGY) then
         if (.not. ieee_is_finite(state%prodigy_d) .or. .not. ieee_is_finite(state%prodigy_dmax) .or. &
             .not. ieee_is_finite(state%prodigy_numerator)) status = REF_NUMERICAL
      end if
      if (state%kind == REF_SCHEDULE_FREE_ADAMW) then
         if (.not. ieee_is_finite(state%schedulefree_weight_sum) .or. &
             .not. ieee_is_finite(state%schedulefree_lr_max)) status = REF_NUMERICAL
      end if
   end subroutine ref_vector_step

   subroutine ref_vector_eval_parameters(parameters, state, beta1, status)
      real(real64), contiguous, intent(inout) :: parameters(:)
      type(ref_vector_state_t), intent(in) :: state
      real(real64), intent(in) :: beta1
      integer(int32), intent(out) :: status
      status = REF_OK
      if (state%kind == REF_SCHEDULE_FREE_ADAMW) then
         if (.not. allocated(state%first)) then
            status = REF_INVALID
            return
         end if
         if (size(state%first) /= size(parameters) .or. state%step <= 0_int64 .or. &
             .not. ieee_is_finite(beta1) .or. beta1 <= 0.0_real64 .or. beta1 >= 1.0_real64) then
            status = REF_INVALID
            return
         end if
         if (state%step == 1_int64) then
            ! At the first step y and z coincide; eval conversion is identity.
            return
         end if
         parameters = parameters / beta1 + (1.0_real64 - 1.0_real64 / beta1) * state%first
      end if
      if (.not. all(ieee_is_finite(parameters))) status = REF_NUMERICAL
   end subroutine ref_vector_eval_parameters

   pure integer(int64) function ref_vector_state_bytes(state) result(byte_count)
      type(ref_vector_state_t), intent(in) :: state
      byte_count = 0_int64
      if (state%kind == REF_NADAM) byte_count = int(storage_size(state%nadam_mu_product) / 8, int64)
      if (state%kind == REF_PRODIGY) byte_count = 3_int64 * int(storage_size(state%prodigy_d) / 8, int64)
      if (state%kind == REF_SCHEDULE_FREE_ADAMW) &
         byte_count = 2_int64 * int(storage_size(state%schedulefree_weight_sum) / 8, int64)
      if (allocated(state%first)) &
         byte_count = byte_count + int(size(state%first), int64) * int(storage_size(0.0_real64) / 8, int64)
      if (allocated(state%second)) &
         byte_count = byte_count + int(size(state%second), int64) * int(storage_size(0.0_real64) / 8, int64)
      if (allocated(state%third)) &
         byte_count = byte_count + int(size(state%third), int64) * int(storage_size(0.0_real64) / 8, int64)
      if (allocated(state%fourth)) &
         byte_count = byte_count + int(size(state%fourth), int64) * int(storage_size(0.0_real64) / 8, int64)
   end function ref_vector_state_bytes

   pure function ref_vector_name(kind) result(name)
      integer(int32), intent(in) :: kind
      character(len=24) :: name
      select case (kind)
      case (REF_SGD); name = 'SGD'
      case (REF_MOMENTUM); name = 'SGD+momentum'
      case (REF_NESTEROV); name = 'Nesterov'
      case (REF_ADAGRAD); name = 'AdaGrad'
      case (REF_RMSPROP); name = 'RMSprop'
      case (REF_ADAM); name = 'Adam'
      case (REF_ADAMW); name = 'AdamW'
      case (REF_NADAM); name = 'NAdam'
      case (REF_RADAM); name = 'RAdam'
      case (REF_LION); name = 'Lion'
      case (REF_ADAFACTOR_VECTOR); name = 'Adafactor-vector'
      case (REF_LAMB); name = 'LAMB'
      case (REF_LARS); name = 'LARS'
      case (REF_SOPHIA_G); name = 'Sophia-G exact-diagonal'
      case (REF_PRODIGY); name = 'Prodigy'
      case (REF_SCHEDULE_FREE_ADAMW); name = 'Schedule-Free AdamW'
      case default; name = 'unknown'
      end select
   end function ref_vector_name

end module ref_vector_optimizers
