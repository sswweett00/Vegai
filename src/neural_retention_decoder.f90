! SPDX-License-Identifier: MIT
!==============================================================================
!  neural_retention_decoder.f90
!  Native Fortran reference kernel for a hybrid causal neural decoder.
!
!  Architecture, per layer:
!    RMSNorm -> rotary local causal attention + positive-feature retention
!    -> residual -> RMSNorm -> gated feed-forward -> residual.
!
!  The retention state is a finite recurrent sufficient statistic:
!      S_t = lambda S_(t-1) + phi(k_t) v_t^T
!      z_t = lambda z_(t-1) + phi(k_t)
!      g_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + epsilon)
!  where phi(x)=softplus(x)>0.  It gives O(d^2) state per head independent of
!  sequence length, while the local cache preserves exact causal softmax-style
!  attention over a recent window.  No future K/V can enter either component.
!
!  This is an inference/invariant kernel, not yet a claim of a finished
!  corpus-scale neural optimizer.  Its explicit dimensions, state equations,
!  tied output head and resource plan establish the native ABI that an audited
!  blockwise backprop/AdamW trainer must use rather than silently falling back
!  to the older HDC or n-gram models.
!==============================================================================
module neural_retention_decoder
   use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   implicit none
   private

   integer(int32), parameter, public :: NRD_OK = 0_int32
   integer(int32), parameter, public :: NRD_ERR_INVALID_ARGUMENT = -1000_int32
   integer(int32), parameter, public :: NRD_ERR_ALLOCATION = -1001_int32
   integer(int32), parameter, public :: NRD_ERR_NOT_READY = -1002_int32
   integer(int32), parameter, public :: NRD_ERR_NUMERIC = -1003_int32

   type, public :: neural_decoder_config_t
      integer(int32) :: vocabulary_size = 2048_int32
      integer(int32) :: width = 128_int32
      integer(int32) :: heads = 4_int32
      integer(int32) :: local_window = 128_int32
      integer(int32) :: ffn_width = 384_int32
      real(real32) :: memory_decay = 0.985_real32
      real(real32) :: rms_epsilon = 1.0e-5_real32
      integer(int32) :: rotary_period = 1048576_int32
   end type neural_decoder_config_t

   type, public :: neural_decoder_state_t
      logical :: initialized = .false.
      integer(int32) :: head_width = 0_int32
      integer(int32) :: heads = 0_int32
      integer(int32) :: local_window = 0_int32
      integer(int32) :: cache_head = 0_int32
      integer(int32) :: cache_count = 0_int32
      integer(int64) :: steps = 0_int64
      real(real32), allocatable :: retention_numerator(:, :, :)
      real(real32), allocatable :: retention_normalizer(:, :)
      real(real32), allocatable :: local_keys(:, :, :)
      real(real32), allocatable :: local_values(:, :, :)
   end type neural_decoder_state_t

   type, public :: neural_decoder_t
      type(neural_decoder_config_t) :: config
      logical :: initialized = .false.
      integer(int32) :: head_width = 0_int32
      ! Tied input/output embedding matrix reduces model parameters and gives
      ! a direct likelihood geometry shared by encoding and decoding.
      real(real32), allocatable :: token_embedding(:, :)
      real(real32), allocatable :: output_bias(:)
      real(real32), allocatable :: attention_rms(:)
      real(real32), allocatable :: ffn_rms(:)
      real(real32), allocatable :: query(:, :, :)
      real(real32), allocatable :: key(:, :, :)
      real(real32), allocatable :: value(:, :, :)
      real(real32), allocatable :: attention_output(:, :)
      real(real32), allocatable :: ffn_gate(:, :)
      real(real32), allocatable :: ffn_value(:, :)
      real(real32), allocatable :: ffn_output(:, :)
      real(real32), allocatable :: global_mix_logit(:)
   end type neural_decoder_t

   type, public :: neural_resource_plan_t
      integer(int64) :: parameter_count = 0_int64
      integer(int64) :: parameter_bytes_real32 = 0_int64
      integer(int64) :: per_stream_state_bytes_real32 = 0_int64
      integer(int64) :: approximate_forward_flops_per_token = 0_int64
      integer(int64) :: local_attention_score_operations_per_token = 0_int64
   end type neural_resource_plan_t

   public :: neural_decoder_init, neural_decoder_destroy, neural_decoder_is_ready
   public :: neural_decoder_state_init, neural_decoder_state_destroy, neural_decoder_state_reset
   public :: neural_decoder_state_is_ready, neural_decoder_step, neural_decoder_plan
   public :: neural_decoder_step_hidden, neural_decoder_project_logits

   ! Number of staged tokens the block projection consumes at a time.  It is a
   ! compile-time constant so the per-lane accumulators live in vector registers
   ! instead of a runtime-sized stack array: with a dynamic size the fan-out loop
   ! degrades to load/FMA/store per element, which measured ~8x slower than the
   ! register form.  Callers allocate the hidden block with this many extra
   ! zeroed columns, so the last partial group reads defined values.
   integer(int32), parameter, public :: NRD_LOGIT_BLOCK_LANES = 16_int32
   public :: neural_decoder_config_is_valid

contains

   pure logical function neural_decoder_config_is_valid(config)
      type(neural_decoder_config_t), intent(in) :: config
      integer(int32) :: head_width

      if (config%heads <= 0_int32) then
         neural_decoder_config_is_valid = .false.
         return
      end if
      head_width = config%width / config%heads
      neural_decoder_config_is_valid = config%vocabulary_size >= 2_int32 .and. config%width >= 2_int32 .and. &
                        modulo(config%width, config%heads) == 0_int32 .and. modulo(head_width, 2_int32) == 0_int32 .and. &
                        config%local_window >= 1_int32 .and. config%local_window <= 4096_int32 .and. &
                        config%ffn_width >= config%width .and. config%memory_decay > 0.0_real32 .and. &
                        config%memory_decay < 1.0_real32 .and. config%rms_epsilon > 0.0_real32 .and. &
                        config%rotary_period >= 16_int32
   end function neural_decoder_config_is_valid

   pure logical function neural_decoder_is_ready(model)
      type(neural_decoder_t), intent(in) :: model

      neural_decoder_is_ready = model%initialized .and. model%head_width > 0_int32 .and. &
                                allocated(model%token_embedding) .and. allocated(model%output_bias) .and. &
                                allocated(model%attention_rms) .and. allocated(model%ffn_rms) .and. &
                                allocated(model%query) .and. allocated(model%key) .and. &
                                allocated(model%value) .and. allocated(model%attention_output) .and. &
                                allocated(model%ffn_gate) .and. allocated(model%ffn_value) .and. &
                                allocated(model%ffn_output) .and. allocated(model%global_mix_logit)
   end function neural_decoder_is_ready

   pure logical function neural_decoder_state_is_ready(state, model)
      type(neural_decoder_state_t), intent(in) :: state
      type(neural_decoder_t), intent(in) :: model

      neural_decoder_state_is_ready = state%initialized .and. neural_decoder_is_ready(model) .and. &
                                      state%head_width == model%head_width .and. state%heads == model%config%heads .and. &
                                      state%local_window == model%config%local_window .and. &
                                      allocated(state%retention_numerator) .and. &
                                      allocated(state%retention_normalizer) .and. allocated(state%local_keys) .and. &
                                      allocated(state%local_values)
   end function neural_decoder_state_is_ready

   subroutine neural_decoder_destroy(model)
      type(neural_decoder_t), intent(inout) :: model

      if (allocated(model%token_embedding)) deallocate(model%token_embedding)
      if (allocated(model%output_bias)) deallocate(model%output_bias)
      if (allocated(model%attention_rms)) deallocate(model%attention_rms)
      if (allocated(model%ffn_rms)) deallocate(model%ffn_rms)
      if (allocated(model%query)) deallocate(model%query)
      if (allocated(model%key)) deallocate(model%key)
      if (allocated(model%value)) deallocate(model%value)
      if (allocated(model%attention_output)) deallocate(model%attention_output)
      if (allocated(model%ffn_gate)) deallocate(model%ffn_gate)
      if (allocated(model%ffn_value)) deallocate(model%ffn_value)
      if (allocated(model%ffn_output)) deallocate(model%ffn_output)
      if (allocated(model%global_mix_logit)) deallocate(model%global_mix_logit)
      model = neural_decoder_t()
   end subroutine neural_decoder_destroy

   subroutine neural_decoder_state_destroy(state)
      type(neural_decoder_state_t), intent(inout) :: state

      if (allocated(state%retention_numerator)) deallocate(state%retention_numerator)
      if (allocated(state%retention_normalizer)) deallocate(state%retention_normalizer)
      if (allocated(state%local_keys)) deallocate(state%local_keys)
      if (allocated(state%local_values)) deallocate(state%local_values)
      state = neural_decoder_state_t()
   end subroutine neural_decoder_state_destroy

   subroutine next_unit(seed, value)
      integer(int64), intent(inout) :: seed
      real(real32), intent(out) :: value
      integer(int64) :: nonnegative

      seed = ieor(seed, ishftc(seed, 13))
      seed = ieor(seed, shiftr(seed, 7))
      seed = ieor(seed, ishftc(seed, 17))
      nonnegative = iand(seed, huge(0_int64))
      value = real(real(nonnegative, real64) / real(huge(0_int64), real64), real32)
   end subroutine next_unit

   subroutine initialize_matrix(matrix, fan_in, seed)
      real(real32), intent(out) :: matrix(:, :)
      integer(int32), intent(in) :: fan_in
      integer(int64), intent(inout) :: seed
      integer :: row, column
      real(real32) :: uniform, scale

      scale = sqrt(3.0_real32 / real(max(1_int32, fan_in), real32))
      do column = 1, size(matrix, 2)
         do row = 1, size(matrix, 1)
            call next_unit(seed, uniform)
            matrix(row, column) = (2.0_real32 * uniform - 1.0_real32) * scale
         end do
      end do
   end subroutine initialize_matrix

   subroutine neural_decoder_init(model, config, seed, status)
      type(neural_decoder_t), intent(inout) :: model
      type(neural_decoder_config_t), intent(in) :: config
      integer(int64), intent(in) :: seed
      integer(int32), intent(out) :: status
      integer(int32) :: head_width, head
      integer :: allocation_status
      integer(int64) :: random_state

      call neural_decoder_destroy(model)
      status = NRD_OK
      if (.not. neural_decoder_config_is_valid(config)) then
         status = NRD_ERR_INVALID_ARGUMENT
         return
      end if
      head_width = config%width / config%heads
      random_state = seed
      if (random_state == 0_int64) random_state = int(z'6A09E667F3BCC909', int64)
      model%config = config
      model%head_width = head_width
      allocate(model%token_embedding(config%width, config%vocabulary_size), model%output_bias(config%vocabulary_size), &
               model%attention_rms(config%width), model%ffn_rms(config%width), stat=allocation_status)
      if (allocation_status /= 0) then; status = NRD_ERR_ALLOCATION; call neural_decoder_destroy(model); return; end if
      allocate(model%query(head_width, config%width, config%heads), model%key(head_width, config%width, config%heads), &
               model%value(head_width, config%width, config%heads), model%attention_output(config%width, config%width), &
               stat=allocation_status)
      if (allocation_status /= 0) then; status = NRD_ERR_ALLOCATION; call neural_decoder_destroy(model); return; end if
      allocate(model%ffn_gate(config%ffn_width, config%width), model%ffn_value(config%ffn_width, config%width), &
               model%ffn_output(config%width, config%ffn_width), model%global_mix_logit(config%heads), stat=allocation_status)
      if (allocation_status /= 0) then; status = NRD_ERR_ALLOCATION; call neural_decoder_destroy(model); return; end if
      call initialize_matrix(model%token_embedding, config%width, random_state)
      call initialize_matrix(model%attention_output, config%width, random_state)
      call initialize_matrix(model%ffn_gate, config%width, random_state)
      call initialize_matrix(model%ffn_value, config%width, random_state)
      call initialize_matrix(model%ffn_output, config%ffn_width, random_state)
      do head = 1_int32, config%heads
         call initialize_matrix(model%query(:, :, head), config%width, random_state)
         call initialize_matrix(model%key(:, :, head), config%width, random_state)
         call initialize_matrix(model%value(:, :, head), config%width, random_state)
      end do
      model%output_bias = 0.0_real32
      model%attention_rms = 1.0_real32
      model%ffn_rms = 1.0_real32
      model%global_mix_logit = 0.0_real32
      model%initialized = .true.
   end subroutine neural_decoder_init

   subroutine neural_decoder_state_init(state, model, status)
      type(neural_decoder_state_t), intent(inout) :: state
      type(neural_decoder_t), intent(in) :: model
      integer(int32), intent(out) :: status
      integer :: allocation_status

      call neural_decoder_state_destroy(state)
      status = NRD_OK
      if (.not. neural_decoder_is_ready(model)) then
         status = NRD_ERR_NOT_READY
         return
      end if
      state%head_width = model%head_width
      state%heads = model%config%heads
      state%local_window = model%config%local_window
      allocate(state%retention_numerator(state%head_width, state%head_width, state%heads), &
               state%retention_normalizer(state%head_width, state%heads), &
               state%local_keys(state%head_width, state%heads, state%local_window), &
               state%local_values(state%head_width, state%heads, state%local_window), stat=allocation_status)
      if (allocation_status /= 0) then
         status = NRD_ERR_ALLOCATION
         call neural_decoder_state_destroy(state)
         return
      end if
      state%initialized = .true.
      call neural_decoder_state_reset(state, model, status)
   end subroutine neural_decoder_state_init

   subroutine neural_decoder_state_reset(state, model, status)
      type(neural_decoder_state_t), intent(inout) :: state
      type(neural_decoder_t), intent(in) :: model
      integer(int32), intent(out) :: status

      status = NRD_OK
      if (.not. neural_decoder_state_is_ready(state, model)) then
         status = NRD_ERR_NOT_READY
         return
      end if
      state%retention_numerator = 0.0_real32
      state%retention_normalizer = 0.0_real32
      state%local_keys = 0.0_real32
      state%local_values = 0.0_real32
      state%cache_head = 0_int32
      state%cache_count = 0_int32
      state%steps = 0_int64
   end subroutine neural_decoder_state_reset

   pure real(real32) function stable_softplus(value) result(transformed)
      real(real32), intent(in) :: value

      if (value > 20.0_real32) then
         transformed = value
      else if (value < -20.0_real32) then
         transformed = exp(value)
      else
         transformed = log(1.0_real32 + exp(value))
      end if
   end function stable_softplus

   pure real(real32) function stable_sigmoid(value) result(transformed)
      real(real32), intent(in) :: value

      if (value >= 0.0_real32) then
         transformed = 1.0_real32 / (1.0_real32 + exp(-value))
      else
         transformed = exp(value) / (1.0_real32 + exp(value))
      end if
   end function stable_sigmoid

   subroutine rms_normalize(input, scale, epsilon, output, status)
      real(real32), intent(in) :: input(:), scale(:)
      real(real32), intent(in) :: epsilon
      real(real32), intent(out) :: output(:)
      integer(int32), intent(out) :: status
      real(real32) :: mean_square, inverse_rms

      status = NRD_OK
      if (size(input) <= 0 .or. size(scale) /= size(input) .or. size(output) /= size(input) .or. epsilon <= 0.0_real32) then
         status = NRD_ERR_INVALID_ARGUMENT
         return
      end if
      mean_square = sum(input * input) / real(size(input), real32)
      if (.not. ieee_is_finite(mean_square) .or. mean_square < 0.0_real32) then
         status = NRD_ERR_NUMERIC
         return
      end if
      inverse_rms = 1.0_real32 / sqrt(mean_square + epsilon)
      output = input * inverse_rms * scale
      if (.not. all(ieee_is_finite(output))) status = NRD_ERR_NUMERIC
   end subroutine rms_normalize

   subroutine apply_rotary(query, key, step, period)
      real(real32), intent(inout) :: query(:, :), key(:, :)
      integer(int64), intent(in) :: step
      integer(int32), intent(in) :: period
      integer :: head, component, pair, pair_count
      integer(int64) :: reduced_step
      ! Angle depends only on the component pair, never on the head.  The
      ! previous form recomputed exp/log/sin/cos identically for every head
      ! on every step; these tables hold bit-identical values computed with
      ! the same expressions exactly once.
      real(real32) :: cosines(size(query, 1) / 2), sines(size(query, 1) / 2)
      real(real32) :: angle, frequency, first, second

      reduced_step = modulo(step, int(period, int64))
      pair_count = size(query, 1) / 2
      do pair = 1, pair_count
         component = 2 * pair - 1
         frequency = exp(-log(10000.0_real32) * real(component - 1, real32) / real(size(query, 1), real32))
         angle = real(reduced_step, real32) * frequency
         cosines(pair) = cos(angle)
         sines(pair) = sin(angle)
      end do
      do head = 1, size(query, 2)
         do pair = 1, pair_count
            component = 2 * pair - 1
            first = query(component, head)
            second = query(component + 1, head)
            query(component, head) = cosines(pair) * first - sines(pair) * second
            query(component + 1, head) = sines(pair) * first + cosines(pair) * second
            first = key(component, head)
            second = key(component + 1, head)
            key(component, head) = cosines(pair) * first - sines(pair) * second
            key(component + 1, head) = sines(pair) * first + cosines(pair) * second
         end do
      end do
   end subroutine apply_rotary

   subroutine project_heads(weights, input, output)
      real(real32), intent(in) :: weights(:, :, :), input(:)
      real(real32), intent(out) :: output(:, :)
      integer :: head, component

      do head = 1, size(weights, 3)
         do component = 1, size(weights, 1)
            output(component, head) = dot_product(weights(component, :, head), input)
         end do
      end do
   end subroutine project_heads

   pure integer(int32) function cache_position(state, ordinal) result(position)
      type(neural_decoder_state_t), intent(in) :: state
      integer(int32), intent(in) :: ordinal

      position = modulo(state%cache_head - state%cache_count + ordinal - 1_int32, state%local_window) + 1_int32
   end function cache_position

   subroutine local_attention(state, query, key, value, local_context)
      type(neural_decoder_state_t), intent(in) :: state
      real(real32), intent(in) :: query(:, :), key(:, :), value(:, :)
      real(real32), intent(out) :: local_context(:, :)
      integer(int32) :: head, ordinal, position
      integer :: component
      ! Scores are computed once and reused for the maximum and the weights;
      ! the previous form evaluated every dot product twice per step.  The
      ! cached values are bit-identical (same expressions, same order) and
      ! the accumulation order is unchanged.  Index cache_count+1 holds the
      ! score against the current key.
      real(real32) :: scores(state%local_window + 1_int32)
      real(real32) :: maximum, normalizer, weight, scale

      scale = 1.0_real32 / sqrt(real(size(query, 1), real32))
      local_context = 0.0_real32
      do head = 1_int32, state%heads
         scores(state%cache_count + 1_int32) = dot_product(query(:, head), key(:, head)) * scale
         maximum = scores(state%cache_count + 1_int32)
         do ordinal = 1_int32, state%cache_count
            position = cache_position(state, ordinal)
            scores(ordinal) = dot_product(query(:, head), state%local_keys(:, head, position)) * scale
            maximum = max(maximum, scores(ordinal))
         end do
         normalizer = exp(scores(state%cache_count + 1_int32) - maximum)
         local_context(:, head) = normalizer * value(:, head)
         do ordinal = 1_int32, state%cache_count
            position = cache_position(state, ordinal)
            weight = exp(scores(ordinal) - maximum)
            normalizer = normalizer + weight
            local_context(:, head) = local_context(:, head) + weight * state%local_values(:, head, position)
         end do
         local_context(:, head) = local_context(:, head) / normalizer
      end do
   end subroutine local_attention

   subroutine update_retention_and_context(state, model, query, key, value, global_context, status)
      type(neural_decoder_state_t), intent(inout) :: state
      type(neural_decoder_t), intent(in) :: model
      real(real32), intent(in) :: query(:, :), key(:, :), value(:, :)
      real(real32), intent(out) :: global_context(:, :)
      integer(int32), intent(out) :: status
      real(real32) :: phi_query(size(query, 1)), phi_key(size(key, 1)), denominator
      integer(int32) :: head
      integer :: row, column

      status = NRD_OK
      global_context = 0.0_real32
      do head = 1_int32, state%heads
         do row = 1, state%head_width
            phi_query(row) = stable_softplus(query(row, head))
            phi_key(row) = stable_softplus(key(row, head))
         end do
         do column = 1, state%head_width
            do row = 1, state%head_width
               state%retention_numerator(row, column, head) = model%config%memory_decay * &
                  state%retention_numerator(row, column, head) + phi_key(row) * value(column, head)
            end do
         end do
         state%retention_normalizer(:, head) = model%config%memory_decay * state%retention_normalizer(:, head) + phi_key
         denominator = dot_product(phi_query, state%retention_normalizer(:, head)) + model%config%rms_epsilon
         if (.not. ieee_is_finite(denominator) .or. denominator <= 0.0_real32) then
            status = NRD_ERR_NUMERIC
            return
         end if
         do column = 1, state%head_width
            global_context(column, head) = dot_product(phi_query, state%retention_numerator(:, column, head)) / denominator
         end do
      end do
      if (.not. all(ieee_is_finite(global_context)) .or. .not. all(ieee_is_finite(state%retention_numerator))) then
         status = NRD_ERR_NUMERIC
      end if
   end subroutine update_retention_and_context

   subroutine append_local_cache(state, key, value)
      type(neural_decoder_state_t), intent(inout) :: state
      real(real32), intent(in) :: key(:, :), value(:, :)
      integer(int32) :: position

      position = modulo(state%cache_head, state%local_window) + 1_int32
      state%local_keys(:, :, position) = key
      state%local_values(:, :, position) = value
      state%cache_head = position
      state%cache_count = min(state%cache_count + 1_int32, state%local_window)
   end subroutine append_local_cache

   ! Everything a step does except the tied logits projection: the residual
   ! stream, attention, retention update and the gated feed-forward.  Splitting
   ! it out lets the trainer project the logits of a whole accumulated block in
   ! one pass over the tied embedding (see neural_decoder_project_logits), which
   ! is the difference between reading the (vocabulary x width) matrix once per
   ! block instead of once per prediction token.
   subroutine neural_decoder_step_hidden(model, state, token, hidden, status)
      type(neural_decoder_t), intent(in) :: model
      type(neural_decoder_state_t), intent(inout) :: state
      integer(int32), intent(in) :: token
      real(real32), intent(out) :: hidden(:)
      integer(int32), intent(out) :: status
      real(real32) :: normalized(model%config%width), attention_normalized(model%config%width)
      real(real32) :: query(model%head_width, model%config%heads), key(model%head_width, model%config%heads)
      real(real32) :: value(model%head_width, model%config%heads), local_context(model%head_width, model%config%heads)
      real(real32) :: global_context(model%head_width, model%config%heads), mixed_context(model%head_width, model%config%heads)
      real(real32) :: attention_vector(model%config%width), residual(model%config%width)
      real(real32) :: gate(model%config%ffn_width), value_gate(model%config%ffn_width), ffn_input(model%config%ffn_width)
      integer(int32) :: head, component, offset, local_status
      real(real32) :: mixture

      status = NRD_OK
      if (.not. neural_decoder_is_ready(model) .or. .not. neural_decoder_state_is_ready(state, model) .or. &
          token < 1_int32 .or. token > model%config%vocabulary_size .or. size(hidden) /= model%config%width) then
         status = NRD_ERR_INVALID_ARGUMENT
         return
      end if
      call rms_normalize(model%token_embedding(:, token), model%attention_rms, model%config%rms_epsilon, normalized, local_status)
      if (local_status /= NRD_OK) then; status = local_status; return; end if
      call project_heads(model%query, normalized, query)
      call project_heads(model%key, normalized, key)
      call project_heads(model%value, normalized, value)
      call apply_rotary(query, key, state%steps, model%config%rotary_period)
      call local_attention(state, query, key, value, local_context)
      call update_retention_and_context(state, model, query, key, value, global_context, local_status)
      if (local_status /= NRD_OK) then; status = local_status; return; end if
      do head = 1_int32, model%config%heads
         mixture = stable_sigmoid(model%global_mix_logit(head))
         mixed_context(:, head) = mixture * global_context(:, head) + (1.0_real32 - mixture) * local_context(:, head)
      end do
      do head = 1_int32, model%config%heads
         offset = (head - 1_int32) * model%head_width
         do component = 1_int32, model%head_width
            attention_vector(offset + component) = mixed_context(component, head)
         end do
      end do
      ! Keep the intrinsic dense products.  Hand-written alternatives were
      ! measured and are slower, not faster: replacing these four MATMUL calls
      ! with a vectorized matvec loop, the strided project_heads dot products
      ! with a contiguous axpy, and TANH with a one-exponential identity moved
      ! the decoder step from 54.4 to 111.9 microseconds per prediction (CI job
      ! 122344621975; the same job's held-out evaluate pass fell from 6998 to
      ! 4600 tokens/s while the untouched merge-base arm moved only 10%, so the
      ! regression is the code, not the runner).  gfortran lowers MATMUL and
      ! DOT_PRODUCT on contiguous operands to loops it vectorizes better than
      ! these hand-written forms.
      residual = model%token_embedding(:, token) + matmul(model%attention_output, attention_vector)
      call rms_normalize(residual, model%ffn_rms, model%config%rms_epsilon, attention_normalized, local_status)
      if (local_status /= NRD_OK) then; status = local_status; return; end if
      ! 384 sigmoid + 384 tanh per token is the transcendental bill of the
      ! gate.  Fusing them (tanh(x) = 2*sigmoid(2x) - 1, or a one-exponential
      ! tanh) does not pay here: see the note above on the same measurement.
      gate = matmul(model%ffn_gate, attention_normalized)
      value_gate = matmul(model%ffn_value, attention_normalized)
      do component = 1_int32, model%config%ffn_width
         ffn_input(component) = stable_sigmoid(gate(component)) * tanh(value_gate(component))
      end do
      hidden = residual + matmul(model%ffn_output, ffn_input)
      if (.not. all(ieee_is_finite(hidden))) then
         status = NRD_ERR_NUMERIC
         return
      end if
      call append_local_cache(state, key, value)
      state%steps = state%steps + 1_int64
   end subroutine neural_decoder_step_hidden

   subroutine neural_decoder_step(model, state, token, hidden, logits, status)
      type(neural_decoder_t), intent(in) :: model
      type(neural_decoder_state_t), intent(inout) :: state
      integer(int32), intent(in) :: token
      real(real32), intent(out) :: hidden(:)
      real(real32), intent(out) :: logits(:)
      integer(int32), intent(out) :: status

      status = NRD_OK
      if (size(logits) /= model%config%vocabulary_size) then
         status = NRD_ERR_INVALID_ARGUMENT
         return
      end if
      call neural_decoder_step_hidden(model, state, token, hidden, status)
      if (status /= NRD_OK) return
      ! Logits use the vector-matrix matmul form hidden . E, which streams
      ! the tied embedding without materializing a transposed copy (the
      ! earlier matmul(transpose(E), hidden) allocated a full (vocabulary x
      ! width) temporary every token) and stays inside the SIMD matmul
      ! kernel with no extra OpenMP fork per token.  Hand-written column
      ! loops were measured slower on the identical probe slice (2275 and
      ! 3250 vs 5125 token/s for sequential and eight-partial-sum variants,
      ! runs 40301975287/40305596215): scalar FMA chains plus per-token
      ! fork-join wake-ups lose to the vectorized kernel.  The value is the
      ! same embedding^T hidden + bias up to reassociation, gated against
      ! the transpose-matmul reference in the test suite.
      logits = matmul(hidden, model%token_embedding) + model%output_bias
      if (.not. all(ieee_is_finite(logits))) then
         status = NRD_ERR_NUMERIC
         return
      end if
   end subroutine neural_decoder_step

   ! Projects a block of hidden states onto the tied embedding in one pass.
   !
   ! hidden_matrix is (count, width) -- one contiguous row per staged token, so
   ! the lane fan-out below is unit stride -- and logits_matrix is
   ! (count, vocabulary):
   ! one row of the output per staged token, so a row is contiguous and can be
   ! handed straight to stable_softmax.  The outer loop runs over the vocabulary
   ! (parallel, one token per iteration) and the embedding element
   ! E(feature, token) is loaded once and fanned out to every column of the
   ! block, so the (vocabulary x width) matrix is streamed once per block instead
   ! of once per token.  The fan-out vector is the only dependency: its elements
   ! are updated once per feature and there are count of them, which is what
   ! keeps the inner loop at one vector FMA per element instead of a scalar
   ! chain.  Lanes are consumed NRD_LOGIT_BLOCK_LANES at a time and the vector is
   ! fixed size, which is what keeps the fan-out in registers.  Writes of one
   ! token's row step by vocabulary, but each cache line is shared by the
   ! following tokens of the block, so the output stays resident.
   subroutine neural_decoder_project_logits(model, hidden_matrix, count, logits_matrix, status)
      type(neural_decoder_t), intent(in) :: model
      ! CONTIGUOUS is load bearing: with a bare assumed-shape dummy gfortran
      ! cannot prove the inner fan-out has unit stride and leaves the loop
      ! scalar, which measured ~20x slower than the vector form.
      real(real32), contiguous, intent(in) :: hidden_matrix(:, :)
      integer(int32), intent(in) :: count
      real(real32), contiguous, intent(out) :: logits_matrix(:, :)
      integer(int32), intent(out) :: status
      integer(int32) :: width, vocabulary_size, token, feature, lane, group, first, available
      real(real32) :: weight, bias
      ! Exactly one group of lanes: fixed size, therefore register resident.
      real(real32) :: accumulator(NRD_LOGIT_BLOCK_LANES)

      status = NRD_OK
      width = model%config%width
      vocabulary_size = model%config%vocabulary_size
      if (.not. neural_decoder_is_ready(model) .or. count < 1_int32 .or. &
          count > size(hidden_matrix, 1, kind=int32) .or. &
          count > size(logits_matrix, 1, kind=int32) .or. &
          size(hidden_matrix, 2, kind=int32) /= width .or. &
          size(logits_matrix, 2, kind=int32) /= vocabulary_size) then
         status = NRD_ERR_INVALID_ARGUMENT
         return
      end if
      !$omp parallel do private(token, feature, lane, group, first, available, weight, bias, accumulator) &
      !$omp schedule(static) if(vocabulary_size >= 512_int32)
      do token = 1_int32, vocabulary_size
         bias = model%output_bias(token)
         do group = 0_int32, (count - 1_int32) / NRD_LOGIT_BLOCK_LANES
            first = group * NRD_LOGIT_BLOCK_LANES + 1_int32
            available = min(NRD_LOGIT_BLOCK_LANES, count - first + 1_int32)
            accumulator = 0.0_real32
            do feature = 1_int32, width
               weight = model%token_embedding(feature, token)
               !$omp simd
               do lane = 1_int32, NRD_LOGIT_BLOCK_LANES
                  accumulator(lane) = accumulator(lane) + weight * hidden_matrix(first + lane - 1_int32, feature)
               end do
            end do
            do lane = 1_int32, available
               logits_matrix(first + lane - 1_int32, token) = accumulator(lane) + bias
            end do
         end do
      end do
      !$omp end parallel do
      if (.not. all(ieee_is_finite(logits_matrix(1_int32:count, 1_int32:vocabulary_size)))) then
         status = NRD_ERR_NUMERIC
      end if
   end subroutine neural_decoder_project_logits


   subroutine neural_decoder_plan(config, plan, status)
      type(neural_decoder_config_t), intent(in) :: config
      type(neural_resource_plan_t), intent(out) :: plan
      integer(int32), intent(out) :: status
      integer(int32) :: head_width
      integer(int64) :: width, vocab, heads, ffn, window, head_width64

      plan = neural_resource_plan_t()
      status = NRD_OK
      if (.not. neural_decoder_config_is_valid(config)) then
         status = NRD_ERR_INVALID_ARGUMENT
         return
      end if
      head_width = config%width / config%heads
      width = int(config%width, int64)
      vocab = int(config%vocabulary_size, int64)
      heads = int(config%heads, int64)
      ffn = int(config%ffn_width, int64)
      window = int(config%local_window, int64)
      head_width64 = int(head_width, int64)
      ! embedding + tied bias/norm + QKV + output + gated FFN + mixing gates
      plan%parameter_count = width * vocab + vocab + 2_int64 * width + 3_int64 * width * width + width * width + &
                             3_int64 * width * ffn + heads
      plan%parameter_bytes_real32 = 4_int64 * plan%parameter_count
      plan%per_stream_state_bytes_real32 = 4_int64 * (heads * head_width64 * head_width64 + heads * head_width64 + &
                                          2_int64 * width * window)
      ! Q/K/V, output projection, two FFN projections plus output, and tied logits.
      plan%approximate_forward_flops_per_token = 2_int64 * (3_int64 * width * width + width * width + &
                                                        3_int64 * width * ffn + width * vocab)
      plan%local_attention_score_operations_per_token = 2_int64 * heads * head_width64 * (window + 1_int64)
   end subroutine neural_decoder_plan

end module neural_retention_decoder
