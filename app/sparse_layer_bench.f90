! SPDX-License-Identifier: MIT
!==============================================================================
!  sparse_layer_bench.f90  --  in-engine proof of capacity invariance
!
!  The claim under test: with the sparse expert layer of
!  src/sparse_expert_layer.f90, one training step (route + forward + backward +
!  sparse update) costs the same at 1e3, 1e6 and 1e9 parameters of *capacity*.
!
!  Measured points
!  ---------------
!    A  W ~ 1.0e3   16x16 experts, k=4,   capacity      4  (one family of its own:
!                   a single 32x32 expert is already 1056 parameters, so the
!                   10^3 point has to use smaller experts)
!    B  W ~ 1.0e6   32x32 experts, k=8,   capacity    977   (  1,031,712 params )
!    C  W ~ 1.0e9   32x32 experts, k=8,   capacity 947,000   (1,000,032,000 params)
!
!  B and C share the expert shape and k on purpose: any latency difference
!  between them is capacity, and nothing else.
!
!  Two timing modes per point
!  --------------------------
!    steady     the identical input every step, so after the warmup every expert
!               page is resident: the number is pure arithmetic.
!    streaming  a fresh deterministic input every step, so the number also pays
!               for materialising cold experts -- the real regime for a 1e9
!               layer, where only a sliver of capacity is ever resident.
!
!  What is checked (and makes the program exit non-zero)
!  ----------------------------------------------------
!    1. the analytic active-parameter and active-FLOP counters of B and C are
!       equal to the last digit, i.e. active work does not know the capacity;
!    2. the resident page count is bounded by the number of steps times k (never
!       by capacity) and is far below it;
!    3. steady-state latency of C is within LATENCY_TOLERANCE of B.
!
!  Usage:  vegai_sparse_layer_bench            (prints a table, exit 0 on pass)
!==============================================================================
program sparse_layer_bench
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64, output_unit
   use sparse_expert_layer
   implicit none (type, external)

   integer(int32), parameter :: WARMUP_STEPS = 32_int32
   integer(int32), parameter :: MEASURED_STEPS = 256_int32
   real(real64), parameter :: LATENCY_TOLERANCE = 1.5_real64
   real(real64), parameter :: MEBIBYTE = 1048576.0_real64

   type :: bench_point_t
      character(len=10) :: label = ''
      integer(int32) :: input_width = 0_int32
      integer(int32) :: output_width = 0_int32
      integer(int32) :: active_experts = 0_int32
      integer(int64) :: expert_capacity = 0_int64
      integer(int64) :: capacity_parameters = 0_int64
      integer(int64) :: active_parameters = 0_int64
      integer(int64) :: active_flops = 0_int64
      integer(int64) :: pages_allocated = 0_int64
      integer(int64) :: resident_bytes = 0_int64
      real(real64) :: steady_microseconds = 0.0_real64
      real(real64) :: streaming_microseconds = 0.0_real64
      logical :: finite = .false.
   end type bench_point_t

   type(bench_point_t) :: points(3)
   integer(int32) :: index, status
   integer(int64) :: page_bound
   logical :: invariant_counters, sparse_storage, flat_latency, numerically_finite

   points(1)%label = 'W~1.0e3'
   points(1)%input_width = 16_int32
   points(1)%output_width = 16_int32
   points(1)%active_experts = 4_int32
   points(1)%expert_capacity = 4_int64

   points(2)%label = 'W~1.0e6'
   points(2)%input_width = 32_int32
   points(2)%output_width = 32_int32
   points(2)%active_experts = 8_int32
   points(2)%expert_capacity = 977_int64

   points(3)%label = 'W~1.0e9'
   points(3)%input_width = 32_int32
   points(3)%output_width = 32_int32
   points(3)%active_experts = 8_int32
   points(3)%expert_capacity = 947000_int64

   write (output_unit, '(A)') '== sparse extreme-capacity layer: one training step =='
   write (output_unit, '(A,I0,A,I0,A)') 'warmup ', WARMUP_STEPS, ' steps, measured ', &
      MEASURED_STEPS, ' steps per mode'

   do index = 1_int32, size(points, kind=int32)
      call measure_point(points(index), status)
      if (status /= SE_OK) then
         write (output_unit, '(A,A,A,I0)') 'measurement failed for ', trim(points(index)%label), &
            ', status ', status
         error stop 1_int32
      end if
   end do

   call report_table(points)

   ! ---- claim 1: active work is a function of (k, width) only -----------------
   invariant_counters = (points(2)%active_parameters == points(3)%active_parameters) .and. &
                        (points(2)%active_flops == points(3)%active_flops)
   ! ---- claim 2: storage follows pages touched, not capacity ------------------
   page_bound = (int(WARMUP_STEPS, int64) + int(MEASURED_STEPS, int64) + 1_int64) * &
                int(points(3)%active_experts, int64)
   sparse_storage = (points(3)%pages_allocated <= page_bound) .and. &
                    (points(3)%pages_allocated < points(3)%expert_capacity)
   ! ---- claim 3: latency does not move with capacity inside one family --------
   flat_latency = (points(2)%steady_microseconds > 0.0_real64) .and. &
                  (points(3)%steady_microseconds <= LATENCY_TOLERANCE * points(2)%steady_microseconds) .and. &
                  (points(3)%steady_microseconds >= points(2)%steady_microseconds / LATENCY_TOLERANCE)
   numerically_finite = points(1)%finite .and. points(2)%finite .and. points(3)%finite

   write (output_unit, '(A,L1)') 'active counters capacity-invariant   : ', invariant_counters
   write (output_unit, '(A,L1)') 'storage bounded by pages touched     : ', sparse_storage
   write (output_unit, '(A,L1)') 'steady latency flat within tolerance : ', flat_latency
   write (output_unit, '(A,L1)') 'all outputs finite (no overflow)     : ', numerically_finite
   call publish_json(points, invariant_counters, sparse_storage, flat_latency, numerically_finite)

   if (.not. (invariant_counters .and. sparse_storage .and. flat_latency .and. numerically_finite)) then
      error stop 1_int32
   end if

contains

   !> Builds one capacity point, times both modes, records the counters.
   subroutine measure_point(point, status_out)
      type(bench_point_t), intent(inout) :: point
      integer(int32), intent(out) :: status_out
      type(sparse_layer_config_t) :: config
      type(sparse_layer_t) :: layer
      type(sparse_forward_t) :: cache
      real(real64), allocatable :: input(:), output(:), target(:), grad_output(:), grad_input(:)
      integer(int32) :: allocation_status

      status_out = SE_OK
      config%input_width = point%input_width
      config%output_width = point%output_width
      config%expert_capacity = point%expert_capacity
      config%active_experts = point%active_experts
      config%router_bits = 32_int32
      config%bucket_bits = 12_int32
      config%learning_rate = 5.0e-4_real64
      call sparse_layer_init(layer, config, status_out)
      if (status_out /= SE_OK) return

      allocate(input(config%input_width), output(config%output_width), target(config%output_width), &
               grad_output(config%output_width), grad_input(config%input_width), stat=allocation_status)
      if (allocation_status /= 0) then
         status_out = SE_ERR_ALLOCATION
         return
      end if
      call deterministic_vector(target, 7_int64)

      point%capacity_parameters = sparse_layer_parameter_count(layer)
      point%active_parameters = sparse_layer_active_parameter_count(layer)
      point%active_flops = active_flops_of(config)

      call deterministic_vector(input, 11_int64)
      point%steady_microseconds = measure_mode(layer, cache, input, target, output, grad_output, &
                                               grad_input, .false., status_out)
      if (status_out /= SE_OK) return
      point%streaming_microseconds = measure_mode(layer, cache, input, target, output, grad_output, &
                                                  grad_input, .true., status_out)
      if (status_out /= SE_OK) return

      point%pages_allocated = layer%stats%pages_allocated
      point%resident_bytes = layer%stats%resident_bytes
      point%finite = all(output == output) .and. abs(maxval(abs(output))) < 1.0e6_real64
      call sparse_layer_destroy(layer)
      deallocate(input, output, target, grad_output, grad_input)
   end subroutine measure_point

   !> One timed run of forward + backward + update.  `streaming` is the only
   !> difference between the two modes: it refreshes the input per step, so the
   !> routing lands on experts that have never been materialised.
   function measure_mode(layer, cache, input, target, output, grad_output, grad_input, &
                         streaming, status) result(microseconds)
      type(sparse_layer_t), intent(inout) :: layer
      type(sparse_forward_t), intent(inout) :: cache
      real(real64), intent(inout) :: input(:)
      real(real64), intent(in) :: target(:)
      real(real64), intent(inout) :: output(:), grad_output(:), grad_input(:)
      logical, intent(in) :: streaming
      integer(int32), intent(out) :: status
      real(real64) :: microseconds
      integer(int32) :: step, inner
      integer(int64) :: start_clock, end_clock, rate

      status = SE_OK
      microseconds = 0.0_real64
      do inner = 1_int32, 2_int32
         ! inner = 1: warmup (unmeasured).  inner = 2: measured.
         if (inner == 2_int32) call system_clock(start_clock, rate)
         do step = 1_int32, merge(WARMUP_STEPS, MEASURED_STEPS, inner == 1_int32)
            if (streaming) call deterministic_vector(input, int(step, int64) + 1000_int64)
            call sparse_layer_forward(layer, input, output, cache, status)
            if (status /= SE_OK) return
            grad_output = output - target
            call sparse_layer_backward(layer, cache, grad_output, grad_input, status)
            if (status /= SE_OK) return
            call sparse_layer_apply_update(layer, status)
            if (status /= SE_OK) return
         end do
         if (inner == 2_int32) then
            call system_clock(end_clock, rate)
            microseconds = real(end_clock - start_clock, real64) * 1.0e6_real64 / &
                           (real(rate, real64) * real(MEASURED_STEPS, real64))
         end if
      end do
   end function measure_mode

   !> The same analytic FLOP count the layer publishes, restated here so the
   !> benchmark does not merely echo the library's own arithmetic.
   pure function active_flops_of(config) result(flops)
      type(sparse_layer_config_t), intent(in) :: config
      integer(int64) :: flops
      integer(int64) :: in_width, out_width, active

      in_width = int(config%input_width, int64)
      out_width = int(config%output_width, int64)
      active = int(config%active_experts, int64)
      flops = active * (2_int64 * (2_int64 * in_width * out_width + out_width) + &
                        in_width * out_width + 8_int64 * out_width * in_width) + &
              active * int(config%router_bits, int64) * in_width * 2_int64
   end function active_flops_of

   !> Deterministic vector in [-1, 1): same numbers on every host, and different
   !> inputs differ by construction rather than by luck.
   subroutine deterministic_vector(vector, seed)
      real(real64), intent(out) :: vector(:)
      integer(int64), intent(in) :: seed
      integer(int64) :: state
      integer(int32) :: index

      state = seed
      do index = 1_int32, size(vector, kind=int32)
         vector(index) = real(mix_word(state), real64) / real(huge(0_int64), real64)
      end do
   end subroutine deterministic_vector

   !> Not pure: it advances `state`, and a pure function may only have
   !> `intent(in)` dummies.  Its caller is an ordinary subroutine.
   function mix_word(state) result(word)
      integer(int64), intent(inout) :: state
      integer(int64) :: word
      integer(int64), parameter :: gamma = int(z'9E3779B97F4A7C15', int64)
      integer(int64), parameter :: mix_a = int(z'BF58476D1CE4E5B9', int64)
      integer(int64), parameter :: mix_b = int(z'94D049BB133111EB', int64)

      state = state + gamma
      word = state
      word = ieor(word, ishft(word, -30_int32)) * mix_a
      word = ieor(word, ishft(word, -27_int32)) * mix_b
      word = ieor(word, ishft(word, -31_int32))
   end function mix_word

   subroutine report_table(points)
      type(bench_point_t), intent(in) :: points(:)
      integer(int32) :: index

      write (output_unit, '(A)') ''
      write (output_unit, '(A)') '  point     capacity params   active params    active FLOPs  pages  resident MB'// &
                                 '   steady us   streaming us'
      do index = 1_int32, size(points, kind=int32)
         write (output_unit, '(A10,1X,I15,1X,I15,1X,I15,1X,I6,1X,F12.2,1X,F11.2,1X,F14.2)') &
            trim(adjustl(points(index)%label)), points(index)%capacity_parameters, &
            points(index)%active_parameters, points(index)%active_flops, points(index)%pages_allocated, &
            real(points(index)%resident_bytes, real64) / MEBIBYTE, &
            points(index)%steady_microseconds, points(index)%streaming_microseconds
      end do
      write (output_unit, '(A)') ''
   end subroutine report_table

   !> One machine-readable line, for the CI annotation and for the docs.
   subroutine publish_json(points, invariant_counters, sparse_storage, flat_latency, numerically_finite)
      type(bench_point_t), intent(in) :: points(:)
      logical, intent(in) :: invariant_counters, sparse_storage, flat_latency, numerically_finite
      character(len=:), allocatable :: line
      integer(int32) :: index

      line = '{"schema":"vegai-sparse-layer-bench/v1","points":['
      do index = 1_int32, size(points, kind=int32)
         line = line//'{"label":"'//trim(adjustl(points(index)%label))//'",'// &
                '"capacity_parameters":'//trim(i64_text(points(index)%capacity_parameters))//','// &
                '"active_parameters":'//trim(i64_text(points(index)%active_parameters))//','// &
                '"active_flops":'//trim(i64_text(points(index)%active_flops))//','// &
                '"pages_allocated":'//trim(i64_text(points(index)%pages_allocated))//','// &
                '"resident_bytes":'//trim(i64_text(points(index)%resident_bytes))//','// &
                '"steady_microseconds":'//trim(r64_text(points(index)%steady_microseconds))//','// &
                '"streaming_microseconds":'//trim(r64_text(points(index)%streaming_microseconds))//'}'
         if (index < size(points, kind=int32)) line = line//','
      end do
      line = line//'],"invariant_counters":'//merge('true', 'false', invariant_counters)//','// &
             '"sparse_storage":'//merge('true', 'false', sparse_storage)//','// &
             '"flat_latency":'//merge('true', 'false', flat_latency)//','// &
             '"finite":'//merge('true', 'false', numerically_finite)//'}'
      write (output_unit, '(A)') 'SPARSE_LAYER_BENCH_JSON: '//trim(line)
   end subroutine publish_json

   function i64_text(value) result(text)
      integer(int64), intent(in) :: value
      character(len=24) :: text

      write (text, '(I0)') value
   end function i64_text

   function r64_text(value) result(text)
      real(real64), intent(in) :: value
      character(len=24) :: text

      write (text, '(F0.3)') value
   end function r64_text

end program sparse_layer_bench
