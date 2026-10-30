! SPDX-License-Identifier: MIT
!==============================================================================
!  fixed_budget_optimizer.f90
!
!  NOVA-1 shaped optimizer for every learnable array of the fixed-budget
!  context engine.  The rule is the project's, applied per matrix column (and
!  per vector):
!
!      e_c   <- rho e_c + (1 - rho) mean(g_c * g_c)        per-column energy
!      a     <- <g_c, v_c>                                 agreement probe
!      gamma <- momentum  (momentum * restart_damp when a < 0)
!      v_c   <- gamma v_c + (1 - gamma) g_c                momentum as an EMA
!      p_c   <- (1 - lr * wd) p_c - lr * v_c / (sqrt(e_c) + eps)
!
!  The momentum is an exponential moving average, not a running sum: a sum
!  amplifies the displacement by 1/(1 - momentum) = 10x, which makes the
!  learning rate mean ten times what it says.  With the EMA the displacement
!  per update is of the order of the learning rate whatever the gradient scale,
!  because the energy is estimated from the same gradients.
!
!  State is one velocity array plus one energy per column, not the two full
!  moment arrays AdamW needs, and there is no per-element transcendental.
!
!  Every parameter array of the engine is covered, including the memory
!  components: the slot write/query/value/read projections, the salience
!  weight, the multi-scale readout (scale_logit) and the channel logits.
!==============================================================================
module fixed_budget_optimizer
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64
   use fixed_budget_context, only: fbc_model_t, fbc_config_t
   use fixed_budget_backward, only: fbc_gradient_t
   implicit none (type, external)
   private

   !> Optimizer hyper-parameters.  Separate from fbc_config_t on purpose: they
   !> describe the update rule, not the context architecture, so a checkpoint
   !> can carry a new optimiser setting without redefining the model.
   type, public :: fbc_update_config_t
      real(real64) :: learning_rate = 1.0e-3_real64
      real(real64) :: momentum = 0.9_real64
      real(real64) :: energy_decay = 0.95_real64
      real(real64) :: restart_damp = 0.5_real64
      real(real64) :: weight_decay = 0.0_real64
      real(real64) :: epsilon = 1.0e-8_real64
      real(real64) :: gradient_clip = 1.0_real64
   end type fbc_update_config_t

   type, public :: fbc_optimizer_t
      logical :: initialized = .false.
      real(real64), allocatable :: velocity_token_embedding(:, :)
      real(real64), allocatable :: energy_token_embedding(:)
      real(real64), allocatable :: velocity_output_bias(:)
      real(real64) :: energy_output_bias = 0.0_real64
      real(real64), allocatable :: velocity_query(:, :, :)
      real(real64), allocatable :: energy_query(:, :)
      real(real64), allocatable :: velocity_key(:, :, :)
      real(real64), allocatable :: energy_key(:, :)
      real(real64), allocatable :: velocity_value(:, :, :)
      real(real64), allocatable :: energy_value(:, :)
      real(real64), allocatable :: velocity_attention_output(:, :)
      real(real64), allocatable :: energy_attention_output(:)
      real(real64), allocatable :: velocity_attention_rms(:)
      real(real64) :: energy_attention_rms = 0.0_real64
      real(real64), allocatable :: velocity_ffn_rms(:)
      real(real64) :: energy_ffn_rms = 0.0_real64
      real(real64), allocatable :: velocity_ffn_gate(:, :)
      real(real64), allocatable :: energy_ffn_gate(:)
      real(real64), allocatable :: velocity_ffn_value(:, :)
      real(real64), allocatable :: energy_ffn_value(:)
      real(real64), allocatable :: velocity_ffn_output(:, :)
      real(real64), allocatable :: energy_ffn_output(:)
      real(real64), allocatable :: velocity_distance_bias(:)
      real(real64) :: energy_distance_bias = 0.0_real64
      real(real64), allocatable :: velocity_scale_logit(:)
      real(real64), allocatable :: velocity_decay_logit(:)
      real(real64) :: energy_decay_logit = 0.0_real64
      real(real64), allocatable :: velocity_write_strength_logit(:)
      real(real64) :: energy_write_strength_logit = 0.0_real64
      real(real64) :: energy_scale_logit = 0.0_real64
      real(real64), allocatable :: velocity_channel_logit(:)
      real(real64) :: energy_channel_logit = 0.0_real64
      real(real64), allocatable :: velocity_channel_gate_proj(:, :)
      real(real64), allocatable :: energy_channel_gate_proj(:)
      real(real64), allocatable :: velocity_channel_router(:, :)
      real(real64), allocatable :: energy_channel_router(:)
      real(real64), allocatable :: velocity_induction_mix(:)
      real(real64), allocatable :: energy_induction_mix
      real(real64), allocatable :: velocity_scale_gate_proj(:, :)
      real(real64), allocatable :: energy_scale_gate_proj(:)
      real(real64), allocatable :: velocity_slot_query_proj(:, :)
      real(real64), allocatable :: energy_slot_query_proj(:)
      real(real64), allocatable :: velocity_slot_key_proj(:, :)
      real(real64), allocatable :: energy_slot_key_proj(:)
      real(real64), allocatable :: velocity_slot_value_proj(:, :)
      real(real64), allocatable :: energy_slot_value_proj(:)
      real(real64), allocatable :: velocity_slot_read_proj(:, :)
      real(real64), allocatable :: energy_slot_read_proj(:)
      real(real64) :: slot_salience_weight_velocity = 0.0_real64
      real(real64) :: slot_salience_weight_energy = 0.0_real64
   end type fbc_optimizer_t

   public :: fbc_optimizer_init, fbc_optimizer_destroy, fbc_optimizer_reset, fbc_optimizer_apply

contains

   subroutine fbc_optimizer_init(optimizer, config, status)
      type(fbc_optimizer_t), intent(inout) :: optimizer
      type(fbc_config_t), intent(in) :: config
      integer(int32), intent(out) :: status
      integer(int32) :: allocation_status
      integer(int32) :: width, vocab, heads, head_width, ffn, window_plus_one, scales, nchannels

      call fbc_optimizer_destroy(optimizer)
      status = 0_int32
      width = config%width
      vocab = config%vocabulary_size
      heads = config%heads
      head_width = config%width / config%heads
      ffn = config%ffn_width
      window_plus_one = config%local_window + 1_int32
      scales = config%memory_scales
      nchannels = 3_int32
      allocation_status = 0_int32
      allocate(optimizer%velocity_token_embedding(width, vocab), &
               optimizer%energy_token_embedding(vocab), stat=allocation_status)
      allocate(optimizer%velocity_output_bias(vocab), stat=allocation_status)
      allocate(optimizer%velocity_query(head_width, width, heads), &
               optimizer%energy_query(width, heads), stat=allocation_status)
      allocate(optimizer%velocity_key(head_width, width, heads), &
               optimizer%energy_key(width, heads), stat=allocation_status)
      allocate(optimizer%velocity_value(head_width, width, heads), &
               optimizer%energy_value(width, heads), stat=allocation_status)
      allocate(optimizer%velocity_attention_output(width, width), &
               optimizer%energy_attention_output(width), stat=allocation_status)
      allocate(optimizer%velocity_attention_rms(width), stat=allocation_status)
      allocate(optimizer%velocity_ffn_rms(width), stat=allocation_status)
      allocate(optimizer%velocity_ffn_gate(ffn, width), &
               optimizer%energy_ffn_gate(width), stat=allocation_status)
      allocate(optimizer%velocity_ffn_value(ffn, width), &
               optimizer%energy_ffn_value(width), stat=allocation_status)
      allocate(optimizer%velocity_ffn_output(width, ffn), &
               optimizer%energy_ffn_output(ffn), stat=allocation_status)
      allocate(optimizer%velocity_distance_bias(window_plus_one), stat=allocation_status)
      allocate(optimizer%velocity_scale_logit(scales), stat=allocation_status)
      allocate(optimizer%velocity_decay_logit(scales), stat=allocation_status)
      allocate(optimizer%velocity_write_strength_logit(scales), stat=allocation_status)
      allocate(optimizer%velocity_channel_logit(nchannels), stat=allocation_status)
      allocate(optimizer%velocity_channel_gate_proj(nchannels, width), &
               optimizer%energy_channel_gate_proj(width), stat=allocation_status)
      allocate(optimizer%velocity_channel_router(nchannels, nchannels * width), &
               optimizer%energy_channel_router(nchannels * width), stat=allocation_status)
      allocate(optimizer%velocity_induction_mix(heads), stat=allocation_status)
      allocate(optimizer%velocity_scale_gate_proj(scales, width), &
               optimizer%energy_scale_gate_proj(width), stat=allocation_status)
      allocate(optimizer%velocity_slot_query_proj(width, width), &
               optimizer%energy_slot_query_proj(width), stat=allocation_status)
      allocate(optimizer%velocity_slot_key_proj(width, width), &
               optimizer%energy_slot_key_proj(width), stat=allocation_status)
      allocate(optimizer%velocity_slot_value_proj(width, width), &
               optimizer%energy_slot_value_proj(width), stat=allocation_status)
      allocate(optimizer%velocity_slot_read_proj(width, width), &
               optimizer%energy_slot_read_proj(width), stat=allocation_status)
      if (allocation_status /= 0) then
         status = -1_int32
         call fbc_optimizer_destroy(optimizer)
         return
      end if
      optimizer%initialized = .true.
      call fbc_optimizer_reset(optimizer)
   end subroutine fbc_optimizer_init

   subroutine fbc_optimizer_reset(optimizer)
      type(fbc_optimizer_t), intent(inout) :: optimizer

      if (.not. optimizer%initialized) return
      optimizer%velocity_token_embedding = 0.0_real64
      optimizer%energy_token_embedding = 0.0_real64
      optimizer%velocity_output_bias = 0.0_real64
      optimizer%energy_output_bias = 0.0_real64
      optimizer%velocity_query = 0.0_real64
      optimizer%energy_query = 0.0_real64
      optimizer%velocity_key = 0.0_real64
      optimizer%energy_key = 0.0_real64
      optimizer%velocity_value = 0.0_real64
      optimizer%energy_value = 0.0_real64
      optimizer%velocity_attention_output = 0.0_real64
      optimizer%energy_attention_output = 0.0_real64
      optimizer%velocity_attention_rms = 0.0_real64
      optimizer%energy_attention_rms = 0.0_real64
      optimizer%velocity_ffn_rms = 0.0_real64
      optimizer%energy_ffn_rms = 0.0_real64
      optimizer%velocity_ffn_gate = 0.0_real64
      optimizer%energy_ffn_gate = 0.0_real64
      optimizer%velocity_ffn_value = 0.0_real64
      optimizer%energy_ffn_value = 0.0_real64
      optimizer%velocity_ffn_output = 0.0_real64
      optimizer%energy_ffn_output = 0.0_real64
      optimizer%velocity_distance_bias = 0.0_real64
      optimizer%energy_distance_bias = 0.0_real64
      optimizer%velocity_scale_logit = 0.0_real64
      optimizer%velocity_decay_logit = 0.0_real64
      optimizer%energy_decay_logit = 0.0_real64
      optimizer%velocity_write_strength_logit = 0.0_real64
      optimizer%energy_write_strength_logit = 0.0_real64
      optimizer%energy_scale_logit = 0.0_real64
      optimizer%velocity_channel_logit = 0.0_real64
      optimizer%energy_channel_logit = 0.0_real64
      optimizer%velocity_channel_gate_proj = 0.0_real64
      optimizer%energy_channel_gate_proj = 0.0_real64
      optimizer%velocity_channel_router = 0.0_real64
      optimizer%energy_channel_router = 0.0_real64
      optimizer%velocity_induction_mix = 0.0_real64
      optimizer%energy_induction_mix = 0.0_real64
      optimizer%velocity_scale_gate_proj = 0.0_real64
      optimizer%energy_scale_gate_proj = 0.0_real64
      optimizer%velocity_slot_query_proj = 0.0_real64
      optimizer%energy_slot_query_proj = 0.0_real64
      optimizer%velocity_slot_key_proj = 0.0_real64
      optimizer%energy_slot_key_proj = 0.0_real64
      optimizer%velocity_slot_value_proj = 0.0_real64
      optimizer%energy_slot_value_proj = 0.0_real64
      optimizer%velocity_slot_read_proj = 0.0_real64
      optimizer%energy_slot_read_proj = 0.0_real64
   end subroutine fbc_optimizer_reset

   subroutine fbc_optimizer_destroy(optimizer)
      type(fbc_optimizer_t), intent(inout) :: optimizer

      if (allocated(optimizer%velocity_token_embedding)) deallocate(optimizer%velocity_token_embedding)
      if (allocated(optimizer%energy_token_embedding)) deallocate(optimizer%energy_token_embedding)
      if (allocated(optimizer%velocity_output_bias)) deallocate(optimizer%velocity_output_bias)
      if (allocated(optimizer%velocity_query)) deallocate(optimizer%velocity_query)
      if (allocated(optimizer%energy_query)) deallocate(optimizer%energy_query)
      if (allocated(optimizer%velocity_key)) deallocate(optimizer%velocity_key)
      if (allocated(optimizer%energy_key)) deallocate(optimizer%energy_key)
      if (allocated(optimizer%velocity_value)) deallocate(optimizer%velocity_value)
      if (allocated(optimizer%energy_value)) deallocate(optimizer%energy_value)
      if (allocated(optimizer%velocity_attention_output)) deallocate(optimizer%velocity_attention_output)
      if (allocated(optimizer%energy_attention_output)) deallocate(optimizer%energy_attention_output)
      if (allocated(optimizer%velocity_attention_rms)) deallocate(optimizer%velocity_attention_rms)
      if (allocated(optimizer%velocity_ffn_rms)) deallocate(optimizer%velocity_ffn_rms)
      if (allocated(optimizer%velocity_ffn_gate)) deallocate(optimizer%velocity_ffn_gate)
      if (allocated(optimizer%energy_ffn_gate)) deallocate(optimizer%energy_ffn_gate)
      if (allocated(optimizer%velocity_ffn_value)) deallocate(optimizer%velocity_ffn_value)
      if (allocated(optimizer%energy_ffn_value)) deallocate(optimizer%energy_ffn_value)
      if (allocated(optimizer%velocity_ffn_output)) deallocate(optimizer%velocity_ffn_output)
      if (allocated(optimizer%energy_ffn_output)) deallocate(optimizer%energy_ffn_output)
      if (allocated(optimizer%velocity_distance_bias)) deallocate(optimizer%velocity_distance_bias)
      if (allocated(optimizer%velocity_scale_logit)) deallocate(optimizer%velocity_scale_logit)
      if (allocated(optimizer%velocity_decay_logit)) deallocate(optimizer%velocity_decay_logit)
      if (allocated(optimizer%velocity_write_strength_logit)) deallocate(optimizer%velocity_write_strength_logit)
      if (allocated(optimizer%velocity_channel_logit)) deallocate(optimizer%velocity_channel_logit)
      if (allocated(optimizer%velocity_channel_gate_proj)) deallocate(optimizer%velocity_channel_gate_proj)
      if (allocated(optimizer%energy_channel_gate_proj)) deallocate(optimizer%energy_channel_gate_proj)
      if (allocated(optimizer%velocity_channel_router)) deallocate(optimizer%velocity_channel_router)
      if (allocated(optimizer%energy_channel_router)) deallocate(optimizer%energy_channel_router)
      if (allocated(optimizer%velocity_induction_mix)) deallocate(optimizer%velocity_induction_mix)
      if (allocated(optimizer%velocity_scale_gate_proj)) deallocate(optimizer%velocity_scale_gate_proj)
      if (allocated(optimizer%energy_scale_gate_proj)) deallocate(optimizer%energy_scale_gate_proj)
      if (allocated(optimizer%velocity_slot_query_proj)) deallocate(optimizer%velocity_slot_query_proj)
      if (allocated(optimizer%energy_slot_query_proj)) deallocate(optimizer%energy_slot_query_proj)
      if (allocated(optimizer%velocity_slot_key_proj)) deallocate(optimizer%velocity_slot_key_proj)
      if (allocated(optimizer%energy_slot_key_proj)) deallocate(optimizer%energy_slot_key_proj)
      if (allocated(optimizer%velocity_slot_value_proj)) deallocate(optimizer%velocity_slot_value_proj)
      if (allocated(optimizer%energy_slot_value_proj)) deallocate(optimizer%energy_slot_value_proj)
      if (allocated(optimizer%velocity_slot_read_proj)) deallocate(optimizer%velocity_slot_read_proj)
      if (allocated(optimizer%energy_slot_read_proj)) deallocate(optimizer%energy_slot_read_proj)
      optimizer = fbc_optimizer_t()
   end subroutine fbc_optimizer_destroy

   !> Applies the rule to every parameter array.  `parameter_updates` counts the
   !> scalars written, the structural companion of "one update step".
   subroutine fbc_optimizer_apply(optimizer, model, gradient, config, status, parameter_updates)
      type(fbc_optimizer_t), intent(inout) :: optimizer
      type(fbc_model_t), intent(inout) :: model
      type(fbc_gradient_t), intent(in) :: gradient
      type(fbc_update_config_t), intent(in) :: config
      integer(int32), intent(out) :: status
      integer(int64), intent(out) :: parameter_updates
      integer(int32) :: index

      status = 0_int32
      parameter_updates = 0_int64
      if (.not. optimizer%initialized) then
         status = -2_int32
         return
      end if
      call nova_columns(model%token_embedding, gradient%token_embedding, optimizer%velocity_token_embedding, &
                        optimizer%energy_token_embedding, config, parameter_updates)
      call nova_vector(model%output_bias, gradient%output_bias, optimizer%velocity_output_bias, &
                       optimizer%energy_output_bias, config, parameter_updates)
      do index = 1_int32, size(model%query, 3, kind=int32)
         call nova_columns(model%query(:, :, index), gradient%query(:, :, index), &
                           optimizer%velocity_query(:, :, index), &
                           optimizer%energy_query(:, index), config, parameter_updates)
      end do
      do index = 1_int32, size(model%key, 3, kind=int32)
         call nova_columns(model%key(:, :, index), gradient%key(:, :, index), &
                           optimizer%velocity_key(:, :, index), &
                           optimizer%energy_key(:, index), config, parameter_updates)
      end do
      do index = 1_int32, size(model%value, 3, kind=int32)
         call nova_columns(model%value(:, :, index), gradient%value(:, :, index), &
                           optimizer%velocity_value(:, :, index), &
                           optimizer%energy_value(:, index), config, parameter_updates)
      end do
      call nova_columns(model%attention_output, gradient%attention_output, optimizer%velocity_attention_output, &
                        optimizer%energy_attention_output, config, parameter_updates)
      call nova_vector(model%attention_rms, gradient%attention_rms, optimizer%velocity_attention_rms, &
                       optimizer%energy_attention_rms, config, parameter_updates)
      call nova_vector(model%ffn_rms, gradient%ffn_rms, optimizer%velocity_ffn_rms, &
                       optimizer%energy_ffn_rms, config, parameter_updates)
      call nova_columns(model%ffn_gate, gradient%ffn_gate, optimizer%velocity_ffn_gate, &
                        optimizer%energy_ffn_gate, config, parameter_updates)
      call nova_columns(model%ffn_value, gradient%ffn_value, optimizer%velocity_ffn_value, &
                        optimizer%energy_ffn_value, config, parameter_updates)
      call nova_columns(model%ffn_output, gradient%ffn_output, optimizer%velocity_ffn_output, &
                        optimizer%energy_ffn_output, config, parameter_updates)
      call nova_vector(model%distance_bias, gradient%distance_bias, optimizer%velocity_distance_bias, &
                       optimizer%energy_distance_bias, config, parameter_updates)
      call nova_vector(model%scale_logit, gradient%scale_logit, optimizer%velocity_scale_logit, &
                       optimizer%energy_scale_logit, config, parameter_updates)
      call nova_vector(model%decay_logit, gradient%decay_logit, optimizer%velocity_decay_logit, &
                       optimizer%energy_decay_logit, config, parameter_updates)
      call nova_vector(model%write_strength_logit, gradient%write_strength_logit, &
                       optimizer%velocity_write_strength_logit, &
                       optimizer%energy_write_strength_logit, config, parameter_updates)
      call nova_vector(model%channel_logit, gradient%channel_logit, optimizer%velocity_channel_logit, &
                       optimizer%energy_channel_logit, config, parameter_updates)
      call nova_columns(model%channel_gate_proj, gradient%channel_gate_proj, &
                        optimizer%velocity_channel_gate_proj, &
                        optimizer%energy_channel_gate_proj, config, parameter_updates)
      call nova_columns(model%channel_router, gradient%channel_router, &
                        optimizer%velocity_channel_router, optimizer%energy_channel_router, &
                        config, parameter_updates)
      call nova_vector(model%induction_mix, gradient%induction_mix, &
                       optimizer%velocity_induction_mix, &
                       optimizer%energy_induction_mix, config, parameter_updates)
      call nova_columns(model%scale_gate_proj, gradient%scale_gate_proj, &
                        optimizer%velocity_scale_gate_proj, &
                        optimizer%energy_scale_gate_proj, config, parameter_updates)
      call nova_columns(model%slot_query_proj, gradient%slot_query_proj, optimizer%velocity_slot_query_proj, &
                        optimizer%energy_slot_query_proj, config, parameter_updates)
      call nova_columns(model%slot_key_proj, gradient%slot_key_proj, optimizer%velocity_slot_key_proj, &
                        optimizer%energy_slot_key_proj, config, parameter_updates)
      call nova_columns(model%slot_value_proj, gradient%slot_value_proj, optimizer%velocity_slot_value_proj, &
                        optimizer%energy_slot_value_proj, config, parameter_updates)
      call nova_columns(model%slot_read_proj, gradient%slot_read_proj, optimizer%velocity_slot_read_proj, &
                        optimizer%energy_slot_read_proj, config, parameter_updates)
      call nova_scalar(model%slot_salience_weight, gradient%slot_salience_weight, &
                       optimizer%slot_salience_weight_velocity, &
                       optimizer%slot_salience_weight_energy, config, parameter_updates)
   end subroutine fbc_optimizer_apply

   !> Per-column rule for one matrix.  Columns are independent, so the traversal
   !> is a straight column-major stream over the parameter array.
   subroutine nova_columns(parameter, grad, velocity, energy, config, parameter_updates)
      real(real64), intent(inout) :: parameter(:, :)
      real(real64), intent(in) :: grad(:, :)
      real(real64), intent(inout) :: velocity(:, :)
      real(real64), intent(inout) :: energy(:)
      type(fbc_update_config_t), intent(in) :: config
      integer(int64), intent(inout) :: parameter_updates
      integer(int32) :: column, row
      real(real64) :: column_energy, agreement, gamma, step, rows

      do column = 1_int32, size(parameter, 2, kind=int32)
         rows = real(size(parameter, 1, kind=int32), real64)
         column_energy = dot_product(grad(:, column), grad(:, column)) / rows
         energy(column) = config%energy_decay * energy(column) + &
                          (1.0_real64 - config%energy_decay) * column_energy
         agreement = dot_product(grad(:, column), velocity(:, column))
         gamma = config%momentum
         if (agreement < 0.0_real64) gamma = config%momentum * config%restart_damp
         step = config%learning_rate / (sqrt(max(energy(column), 0.0_real64)) + config%epsilon)
         do row = 1_int32, size(parameter, 1, kind=int32)
            velocity(row, column) = gamma * velocity(row, column) + &
                                    (1.0_real64 - config%momentum) * grad(row, column)
            parameter(row, column) = (1.0_real64 - config%learning_rate * config%weight_decay) * &
                                     parameter(row, column) - step * velocity(row, column)
         end do
         parameter_updates = parameter_updates + int(size(parameter, 1, kind=int32), int64)
      end do
   end subroutine nova_columns

   subroutine nova_vector(parameter, grad, velocity, energy, config, parameter_updates)
      real(real64), intent(inout) :: parameter(:)
      real(real64), intent(in) :: grad(:)
      real(real64), intent(inout) :: velocity(:)
      real(real64), intent(inout) :: energy
      type(fbc_update_config_t), intent(in) :: config
      integer(int64), intent(inout) :: parameter_updates
      real(real64) :: column_energy, agreement, gamma, step

      column_energy = dot_product(grad, grad) / real(size(grad, kind=int32), real64)
      energy = config%energy_decay * energy + (1.0_real64 - config%energy_decay) * column_energy
      agreement = dot_product(grad, velocity)
      gamma = config%momentum
      if (agreement < 0.0_real64) gamma = config%momentum * config%restart_damp
      step = config%learning_rate / (sqrt(max(energy, 0.0_real64)) + config%epsilon)
      velocity = gamma * velocity + (1.0_real64 - config%momentum) * grad
      parameter = (1.0_real64 - config%learning_rate * config%weight_decay) * parameter - step * velocity
      parameter_updates = parameter_updates + int(size(parameter, kind=int32), int64)
   end subroutine nova_vector

   subroutine nova_scalar(parameter, grad, velocity, energy, config, parameter_updates)
      real(real64), intent(inout) :: parameter
      real(real64), intent(in) :: grad
      real(real64), intent(inout) :: velocity, energy
      type(fbc_update_config_t), intent(in) :: config
      integer(int64), intent(inout) :: parameter_updates
      real(real64) :: agreement, gamma, step

      energy = config%energy_decay * energy + (1.0_real64 - config%energy_decay) * grad * grad
      agreement = grad * velocity
      gamma = config%momentum
      if (agreement < 0.0_real64) gamma = config%momentum * config%restart_damp
      step = config%learning_rate / (sqrt(max(energy, 0.0_real64)) + config%epsilon)
      velocity = gamma * velocity + (1.0_real64 - config%momentum) * grad
      parameter = (1.0_real64 - config%learning_rate * config%weight_decay) * parameter - step * velocity
      parameter_updates = parameter_updates + 1_int64
   end subroutine nova_scalar

end module fixed_budget_optimizer
