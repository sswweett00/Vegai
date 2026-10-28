! SPDX-License-Identifier: MIT
!==============================================================================
!  sparse_expert_layer.f90
!
!  An extreme-capacity sparse layer whose *active* cost is independent of its
!  parameter capacity.
!
!  The problem this solves
!  ----------------------
!  A dense layer with W parameters costs O(W) time, O(W) memory traffic and
!  O(W) FLOPs per step, so capacity and step cost are the same number.  This
!  module separates them: capacity is the product
!
!      W_total = expert_capacity * (input_width * output_width + output_width)
!
!  while a step touches exactly `active_experts` (= k) of those experts, i.e.
!
!      active parameters = k * (input_width * output_width + output_width)
!      active FLOPs      = O(k)   -- independent of expert_capacity
!
!  How capacity can be billions without the memory or the scan
!  ----------------------------------------------------------
!  1. **Routing is locality-sensitive hashing, not a ranking.**  A fixed random
!     hyperplane family produces a `router_bits`-bit signature of the input in
!     O(router_bits * input_width) -- a constant.  The low `bucket_bits` of that
!     signature are the bucket, and the expert is that bucket mapped onto the
!     capacity.  Similar inputs share low bits with high probability, so the map
!     is locality sensitive; it is also a direct address computation, so no
!     candidate list over `expert_capacity` is ever formed, sorted or read.
!     Someone has to pay O(W_total) to *choose* among W_total experts; hashing
!     is what makes that payment unnecessary.
!  2. **Experts are virtual until touched.**  No parameter array of size
!     W_total exists.  An expert's weights are materialised on demand from a
!     deterministic hash of (seed, expert_id) -- pure, no storage, no I/O -- so
!     an expert that is never routed to costs zero bytes and zero time.  Pages
!     that have been touched are kept in an append-only sparse store behind an
!     open-addressing index, so resident memory is O(pages touched) while
!     capacity remains W_total.  This is the paging model of a large server
!     that has cold weights on disk; here the "disk" is a 64-bit hash.
!  3. **Gradients and optimizer state are sparse too.**  Only a touched page
!     owns a velocity array, a per-column energy array and gradient buffers, so
!     the optimizer's memory is also O(k) per step rather than O(W_total).
!
!  The update rule is the project's NOVA-1 (velocity from the rank-1 gradient,
!  per-column energy, agreement-triggered restart, no per-element
!  transcendental); see docs/TURKISH_99B_NEURAL_PRETRAINING.md section 3.6.
!
!  What is deliberately NOT here
!  -----------------------------
!  * No BLAS, LAPACK, C binding or external package: every kernel below is a
!    plain Fortran loop, written column-major so the fast axis is contiguous.
!  * The router is **frozen** (training-free LSH).  A learned router that
!    softmaxes over the full expert set is O(W_total) by construction and would
!    reintroduce the bottleneck this module exists to remove; a learned router
!    with the same O(k) property is possible (hash-then-score) and is left as a
!    documented extension rather than pretended to be implemented.
!  * `simple` (Fortran 2023) is not used even though it is the right attribute
!    for the pure kernels: no shipping compiler accepts it yet (gfortran and
!    flang reject the keyword), so `pure` -- which expresses exactly the same
!    side-effect contract -- is used instead.  One keyword that compilers do not
!    know turns a "compilation-ready" deliverable into one that cannot be built.
!
!  Threading: the hot kernels use `do concurrent`, which is the standard's
!  parallel-loop construct.  With OpenMP enabled the compiler may map it onto
!  threads (automatic `do concurrent` offloading/parallelisation); without it,
!  it is a legal serial loop.  No OpenMP directives appear in this file, so the
!  module compiles with plain `gfortran -std=f2023` as well.
!==============================================================================
module sparse_expert_layer
   use, intrinsic :: iso_fortran_env, only: int32, int64, real64
   implicit none (type, external)
   private

   ! Status codes.  Everything is fail-closed: a routine that cannot prove its
   ! postcondition returns a non-OK status and leaves the layer usable.
   integer(int32), parameter, public :: SE_OK = 0_int32
   integer(int32), parameter, public :: SE_ERR_INVALID_ARGUMENT = -1101_int32
   integer(int32), parameter, public :: SE_ERR_ALLOCATION = -1102_int32
   integer(int32), parameter, public :: SE_ERR_NUMERICAL = -1103_int32
   integer(int32), parameter, public :: SE_ERR_ORDER = -1104_int32

   integer(int32), parameter, public :: SE_MAX_ROUTER_BITS = 64_int32
   integer(int32), parameter, public :: SE_MAX_ACTIVE_EXPERTS = 64_int32
   integer(int32), parameter, public :: SE_MAX_BUCKET_BITS = 32_int32

   ! 64-bit hash constants (splitmix64).  Fits the int64 kind exactly.
   integer(int64), parameter :: SE_GAMMA = int(z'9E3779B97F4A7C15', int64)
   integer(int64), parameter :: SE_MIX_A = int(z'BF58476D1CE4E5B9', int64)
   integer(int64), parameter :: SE_MIX_B = int(z'94D049BB133111EB', int64)

   !> Static description of one layer.
   type, public :: sparse_layer_config_t
      ! Per-expert shape.  Fixed: capacity grows by adding experts, never by
      ! widening them, which is what keeps the active sub-slice constant.
      integer(int32) :: input_width = 32_int32
      integer(int32) :: output_width = 32_int32
      ! Number of experts.  W_total = expert_capacity * (in*out + out).
      integer(int64) :: expert_capacity = 1024_int64
      ! k: how many experts a single activation reaches.  Bound on O(k).
      integer(int32) :: active_experts = 4_int32
      ! Bits of the LSH signature (the hash width of the random hyperplanes).
      integer(int32) :: router_bits = 32_int32
      ! Low bits of the signature used as the bucket.  This is the knob that
      ! trades routing locality (small bucket_bits: many inputs share an expert)
      ! against expert specialisation (large: adults get their own).
      integer(int32) :: bucket_bits = 12_int32
      ! Deterministic seed of both the router hyperplanes and every expert's
      ! virtual initialisation.  Same seed + same expert_id = same weights on
      ! any host, so a checkpoint stores identity, not parameters.
      integer(int64) :: seed = 20261028_int64
      ! NOVA-1 hyperparameters (see module header).
      real(real64) :: learning_rate = 2.0e-3_real64
      real(real64) :: momentum = 0.9_real64
      real(real64) :: energy_decay = 0.9_real64
      real(real64) :: restart_damp = 0.5_real64
      real(real64) :: weight_decay = 0.0_real64
      real(real64) :: epsilon = 1.0e-8_real64
   end type sparse_layer_config_t

   !> Counters that make the invariance claim machine-checkable.  A benchmark
   !> cannot prove "O(k)" from a stopwatch alone; these are exact integers.
   type, public :: sparse_layer_stats_t
      integer(int64) :: steps = 0_int64
      integer(int64) :: pages_allocated = 0_int64
      integer(int64) :: resident_bytes = 0_int64
      integer(int64) :: active_parameters = 0_int64
      integer(int64) :: active_flops = 0_int64
      integer(int64) :: cumulative_active_flops = 0_int64
      integer(int64) :: cumulative_active_parameters = 0_int64
      integer(int64) :: router_evaluations = 0_int64
      integer(int64) :: updates_applied = 0_int64
   end type sparse_layer_stats_t

   !> One resident expert.  Arrays are allocated only when the expert is first
   !> routed to; `state_ready` and `gradient_ready` track the sparse optimizer
   !> and gradient buffers.  The type is public because it is the unit of the
   !> storage contract: a caller (and the contract test) may inspect which experts
   !> exist and what one of them owns, without any accessor that would copy a page.
   type, public :: sparse_page_t
      integer(int64) :: expert_id = -1_int64
      logical :: state_ready = .false.
      logical :: gradient_ready = .false.
      real(real64), allocatable :: weight(:, :)
      real(real64), allocatable :: bias(:)
      real(real64), allocatable :: weight_gradient(:, :)
      real(real64), allocatable :: bias_gradient(:)
      real(real64), allocatable :: velocity(:, :)
      real(real64), allocatable :: energy(:)
   end type sparse_page_t

   !> The layer: frozen router, sparse page store, index, counters.
   type, public :: sparse_layer_t
      type(sparse_layer_config_t) :: config
      ! (input_width, router_bits, active_experts) sign hyperplanes.
      real(real64), allocatable :: router(:, :, :)
      ! Cached L2 norm of each hyperplane, so the gate score costs no extra pass.
      real(real64), allocatable :: router_norm(:, :)
      type(sparse_page_t), allocatable :: pages(:)
      integer(int32) :: page_count = 0_int32
      integer(int32) :: page_capacity = 0_int32
      ! Open-addressing index: expert_id -> 1-based page index (0 = empty).
      integer(int32), allocatable :: slot_table(:)
      integer(int32) :: table_size = 0_int32
      ! Pages that own a pending gradient, in the order they were touched.  This
      ! is what keeps the update pass O(k) instead of O(pages): without it the
      ! optimizer would have to sweep the page store, and a long-running layer
      ! would slow down as cold pages accumulate.
      integer(int32), allocatable :: dirty_slots(:)
      integer(int32) :: dirty_count = 0_int32
      type(sparse_layer_stats_t) :: stats
   end type sparse_layer_t

   !> Exactly the state one backward pass needs.  Nothing here scales with
   !> capacity: the arrays are all of size k (plus one input copy).
   type, public :: sparse_forward_t
      logical :: ready = .false.
      integer(int32) :: active_count = 0_int32
      integer(int64), allocatable :: expert_id(:)
      integer(int32), allocatable :: page_slot(:)
      real(real64), allocatable :: gate(:)
      real(real64), allocatable :: score(:)
      real(real64), allocatable :: expert_output(:, :)
      real(real64), allocatable :: input_copy(:)
   end type sparse_forward_t

   public :: sparse_layer_config_is_valid
   public :: sparse_layer_parameter_count
   public :: sparse_layer_active_parameter_count
   public :: sparse_layer_init
   public :: sparse_layer_destroy
   public :: sparse_layer_forward
   public :: sparse_layer_backward
   public :: sparse_layer_apply_update

contains

   !----------------------------------------------------------------------------
   !  Deterministic, side-effect-free kernels
   !----------------------------------------------------------------------------

   !> splitmix64: a bijective 64-bit mixer.  Used for the router hyperplanes,
   !> for every expert's virtual initial weights and for the benchmark inputs.
   !>
   !> This one is a *stream* generator: it advances its argument, and Fortran
   !> requires every dummy of a pure function to be `intent(in)`, so it cannot be
   !> pure -- a `pure` prefix here would be an illegal interface, not a style
   !> choice.  The side-effect-free variant for use inside `pure` expressions is
   !> `splitmix64_of` below.  All callers are ordinary (non-pure) procedures.
   function splitmix64(state) result(value)
      integer(int64), intent(inout) :: state
      integer(int64) :: value

      state = state + SE_GAMMA
      value = state
      value = ieor(value, ishft(value, -30_int32)) * SE_MIX_A
      value = ieor(value, ishft(value, -27_int32)) * SE_MIX_B
      value = ieor(value, ishft(value, -31_int32))
   end function splitmix64

   !> Uniform in [-1, 1) from a 64-bit word.  Deterministic on every host: the
   !> conversion is exact for |word| < 2^63 and division by 2^63-1 is exact in
   !> real64?  It is not exact for every word, which is why the *same* host
   !> reproduces bit-identical weights and a different host is only required to
   !> reproduce them to rounding -- the checkpoint contract is on identity.
   pure function uniform_signed(word) result(value)
      integer(int64), intent(in) :: word
      real(real64) :: value

      value = real(word, real64) / real(huge(0_int64), real64)
   end function uniform_signed

   !> Uniform magnitude for the initial weights: the usual 1/sqrt(fan_in).
   pure function initial_scale(fan_in) result(scale)
      integer(int32), intent(in) :: fan_in
      real(real64) :: scale

      scale = 1.0_real64 / sqrt(real(max(fan_in, 1_int32), real64))
   end function initial_scale

   !> Validity is part of the contract: a layer that cannot route k distinct
   !> experts, or whose signature has more bits than the hash, is rejected before
   !> anything is allocated.
   pure logical function sparse_layer_config_is_valid(config)
      type(sparse_layer_config_t), intent(in) :: config
      logical :: valid

      valid = config%input_width >= 1_int32 .and. config%output_width >= 1_int32
      valid = valid .and. config%expert_capacity >= 1_int64
      valid = valid .and. config%active_experts >= 1_int32
      valid = valid .and. config%active_experts <= SE_MAX_ACTIVE_EXPERTS
      valid = valid .and. int(config%active_experts, int64) <= config%expert_capacity
      valid = valid .and. config%router_bits >= 1_int32 .and. config%router_bits <= SE_MAX_ROUTER_BITS
      valid = valid .and. config%bucket_bits >= 1_int32 .and. config%bucket_bits <= SE_MAX_BUCKET_BITS
      valid = valid .and. config%bucket_bits <= config%router_bits
      valid = valid .and. config%learning_rate > 0.0_real64 .and. config%learning_rate < 1.0_real64
      valid = valid .and. config%momentum >= 0.0_real64 .and. config%momentum < 1.0_real64
      valid = valid .and. config%energy_decay >= 0.0_real64 .and. config%energy_decay < 1.0_real64
      valid = valid .and. config%restart_damp > 0.0_real64 .and. config%restart_damp <= 1.0_real64
      valid = valid .and. config%weight_decay >= 0.0_real64 .and. config%weight_decay < 1.0_real64
      valid = valid .and. config%epsilon > 0.0_real64
      sparse_layer_config_is_valid = valid
   end function sparse_layer_config_is_valid

   !> Total capacity this layer *represents*.  This number is O(W_total) of
   !> course -- it is the capacity -- and no allocation of this size is ever
   !> made.
   pure integer(int64) function sparse_layer_parameter_count(layer) result(count)
      type(sparse_layer_t), intent(in) :: layer

      count = layer%config%expert_capacity * &
              (int(layer%config%input_width, int64) * int(layer%config%output_width, int64) + &
               int(layer%config%output_width, int64))
   end function sparse_layer_parameter_count

   !> Parameters a single activation can touch: k experts' worth.  Independent
   !> of capacity by construction, and asserted against the live counters in the
   !> test suite.
   pure integer(int64) function sparse_layer_active_parameter_count(layer) result(count)
      type(sparse_layer_t), intent(in) :: layer

      count = int(layer%config%active_experts, int64) * &
              (int(layer%config%input_width, int64) * int(layer%config%output_width, int64) + &
               int(layer%config%output_width, int64))
   end function sparse_layer_active_parameter_count

   !----------------------------------------------------------------------------
   !  Router: O(router_bits * input_width) per probe, independent of capacity
   !----------------------------------------------------------------------------

   !> Signature of one probe: bit b is set when the input lies on the positive
   !> side of hyperplane b.  Cost is bits * input_width multiply-adds.
   pure subroutine lsh_signature(layer, input, probe, signature, status)
      type(sparse_layer_t), intent(in) :: layer
      real(real64), intent(in) :: input(:)
      integer(int32), intent(in) :: probe
      integer(int64), intent(out) :: signature
      integer(int32), intent(out) :: status
      integer(int32) :: bit
      logical :: positive(size(layer%router, 2, kind=int32))

      status = SE_OK
      signature = 0_int64
      if (probe < 1_int32 .or. probe > size(layer%router, 3, kind=int32)) then
         status = SE_ERR_INVALID_ARGUMENT
         return
      end if
      ! The hyperplane tests are independent and vectorise as a `do concurrent`
      ! store into a logical mask; assembling the mask into one 64-bit word is a
      ! sequential reduction by definition (each bit is a different bit of the
      ! same scalar), and it is 64 iterations of bit arithmetic.
      do concurrent (bit=1_int32:size(layer%router, 2, kind=int32))
         positive(bit) = dot_product(layer%router(:, bit, probe), input) > 0.0_real64
      end do
      do bit = 1_int32, size(layer%router, 2, kind=int32)
         if (positive(bit)) signature = ior(signature, ishft(1_int64, bit - 1_int32))
      end do
   end subroutine lsh_signature

   !> Gate score of one probe: the |cos| of the input against hyperplane 1 of
   !> that probe, in [0, 1].  Uses the precomputed hyperplane norm, so it is one
   !> dot product after the signature pass.
   pure function probe_score(layer, input, probe) result(score)
      type(sparse_layer_t), intent(in) :: layer
      real(real64), intent(in) :: input(:)
      integer(int32), intent(in) :: probe
      real(real64) :: score
      real(real64) :: projection, input_norm

      projection = abs(dot_product(layer%router(:, 1_int32, probe), input))
      input_norm = sqrt(dot_product(input, input))
      score = projection / (layer%router_norm(1_int32, probe) * input_norm + 1.0e-12_real64)
      score = min(max(score, 0.0_real64), 1.0_real64)
   end function probe_score

   !> Expert id for the signature of one probe.  Probe p has its own hyperplanes
   !> (`lsh_signature` takes p), so the k probes already address unrelated
   !> experts without a salt term -- and the id stays a pure function of the input.
   pure function routed_expert(signature, config) result(expert_id)
      integer(int64), intent(in) :: signature
      type(sparse_layer_config_t), intent(in) :: config
      integer(int64) :: expert_id
      integer(int64) :: bucket, mixed, mask

      mask = ishft(1_int64, config%bucket_bits) - 1_int64
      bucket = iand(signature, mask)
      ! The bucket goes through the mixer, so experts are spread over the whole
      ! capacity instead of clustering in the low buckets.
      mixed = splitmix64_of(bucket)
      expert_id = modulo(iand(mixed, int(z'7FFFFFFFFFFFFFFF', int64)), config%expert_capacity)
   end function routed_expert

   !> splitmix64 without an inout argument, for use inside `pure` expressions.
   pure function splitmix64_of(word) result(value)
      integer(int64), intent(in) :: word
      integer(int64) :: value

      value = ieor(word, ishft(word, -30_int32)) * SE_MIX_A
      value = ieor(value, ishft(value, -27_int32)) * SE_MIX_B
      value = ieor(value, ishft(value, -31_int32))
   end function splitmix64_of

   !----------------------------------------------------------------------------
   !  Sparse page store: O(1) lookup, memory proportional to pages touched
   !----------------------------------------------------------------------------

   !> Page index for an expert, or 0 when the expert has never been touched.
   !> Open addressing with linear probing: expected O(1), and the table itself is
   !> sized to the number of *pages*, never to the capacity.
   pure function find_page_slot(layer, expert_id) result(slot)
      type(sparse_layer_t), intent(in) :: layer
      integer(int64), intent(in) :: expert_id
      integer(int32) :: slot
      integer(int32) :: probe, entry

      slot = 0_int32
      if (layer%table_size <= 0_int32) return
      probe = probe_index(expert_id, layer%table_size)
      do
         entry = layer%slot_table(probe)
         if (entry == 0_int32) return
         if (layer%pages(entry)%expert_id == expert_id) then
            slot = entry
            return
         end if
         probe = modulo(probe, layer%table_size) + 1_int32
      end do
   end function find_page_slot

   pure function probe_index(expert_id, table_size) result(index)
      integer(int64), intent(in) :: expert_id
      integer(int32), intent(in) :: table_size
      integer(int32) :: index

      index = int(iand(splitmix64_of(ieor(expert_id, SE_GAMMA)), int(table_size - 1, int64)), int32) + &
              1_int32
   end function probe_index

   !> Materialises the virtual weights of one expert.  The hash is seeded by the
   !> expert id, so an expert's initial parameters are a pure function of
   !> (config%seed, expert_id): nothing has to be stored to "have" a billion
   !> experts, and the initialization is reproducible without a checkpoint.
   subroutine materialise_page(page, config, expert_id, status)
      type(sparse_page_t), intent(inout) :: page
      type(sparse_layer_config_t), intent(in) :: config
      integer(int64), intent(in) :: expert_id
      integer(int32), intent(out) :: status
      integer(int64) :: state
      integer(int32) :: allocation_status, row, column
      real(real64) :: scale

      status = SE_OK
      page%expert_id = expert_id
      page%state_ready = .false.
      page%gradient_ready = .false.
      allocate(page%weight(config%output_width, config%input_width), &
               page%bias(config%output_width), &
               page%weight_gradient(config%output_width, config%input_width), &
               page%bias_gradient(config%output_width), &
               page%velocity(config%output_width, config%input_width), &
               page%energy(config%input_width), stat=allocation_status)
      if (allocation_status /= 0) then
         status = SE_ERR_ALLOCATION
         return
      end if
      page%weight_gradient = 0.0_real64
      page%bias_gradient = 0.0_real64
      page%velocity = 0.0_real64
      page%energy = 0.0_real64
      ! Weight i,j of expert e comes from word (e, i, j); the stream is keyed by
      ! the expert so two experts never share an initialisation.
      state = splitmix64_of(ieor(expert_id, config%seed))
      scale = initial_scale(config%input_width)
      do column = 1_int32, config%input_width
         do row = 1_int32, config%output_width
            page%weight(row, column) = scale * uniform_signed(splitmix64(state))
         end do
      end do
      do row = 1_int32, config%output_width
         page%bias(row) = 0.1_real64 * scale * uniform_signed(splitmix64(state))
      end do
   end subroutine materialise_page

   !> Inserts a freshly materialised page and updates the index, growing the page
   !> array and the table geometrically (amortised O(1) per page touched).
   subroutine insert_page(layer, page, status)
      type(sparse_layer_t), intent(inout) :: layer
      type(sparse_page_t), intent(inout) :: page
      integer(int32), intent(out) :: status
      type(sparse_page_t), allocatable :: grown(:)
      integer(int32) :: allocation_status, new_capacity, slot, probe

      status = SE_OK
      if (layer%page_count >= layer%page_capacity) then
         new_capacity = max(8_int32, 2_int32 * layer%page_capacity)
         allocate(grown(new_capacity), stat=allocation_status)
         if (allocation_status /= 0) then
            status = SE_ERR_ALLOCATION
            return
         end if
         if (layer%page_count > 0_int32) grown(1_int32:layer%page_count) = layer%pages(1_int32:layer%page_count)
         call move_alloc(grown, layer%pages)
         layer%page_capacity = new_capacity
      end if
      layer%page_count = layer%page_count + 1_int32
      slot = layer%page_count
      layer%pages(slot) = page
      if (allocate_table_shape(layer, 2_int32 * layer%page_count, status) /= SE_OK) return
      probe = probe_index(page%expert_id, layer%table_size)
      do
         if (layer%slot_table(probe) == 0_int32) then
            layer%slot_table(probe) = slot
            exit
         end if
         probe = modulo(probe, layer%table_size) + 1_int32
      end do
      layer%stats%pages_allocated = int(layer%page_count, int64)
      layer%stats%resident_bytes = resident_bytes_of(layer)
   end subroutine insert_page

   !> (Re)sizes the index so the load factor stays at or below 50%.
   function allocate_table_shape(layer, wanted_pages, status) result(result_status)
      type(sparse_layer_t), intent(inout) :: layer
      integer(int32), intent(in) :: wanted_pages
      integer(int32), intent(out) :: status
      integer(int32) :: result_status
      integer(int32), allocatable :: rebuilt(:)
      integer(int32) :: allocation_status, target, probe, slot

      status = SE_OK
      result_status = SE_OK
      target = 8_int32
      do while (target < 2_int32 * max(wanted_pages, 1_int32))
         target = 2_int32 * target
      end do
      if (target == layer%table_size) return
      allocate(rebuilt(target), stat=allocation_status)
      if (allocation_status /= 0) then
         status = SE_ERR_ALLOCATION
         result_status = SE_ERR_ALLOCATION
         return
      end if
      rebuilt = 0_int32
      do slot = 1_int32, layer%page_count
         probe = probe_index(layer%pages(slot)%expert_id, target)
         do
            if (rebuilt(probe) == 0_int32) then
               rebuilt(probe) = slot
               exit
            end if
            probe = modulo(probe, target) + 1_int32
         end do
      end do
      call move_alloc(rebuilt, layer%slot_table)
      layer%table_size = target
   end function allocate_table_shape

   !> Resident bytes actually owned by this layer.  This is the number that stays
   !> small while `sparse_layer_parameter_count` grows to billions.
   pure integer(int64) function resident_bytes_of(layer) result(bytes)
      type(sparse_layer_t), intent(in) :: layer
      integer(int64) :: per_page, router_bytes

      ! weight + gradient + velocity (3 matrices), bias + gradient (2 vectors),
      ! plus energy (input_width reals).
      per_page = 3_int64 * int(layer%config%output_width, int64) * &
                 int(layer%config%input_width, int64) * real64_bytes() + &
                 2_int64 * int(layer%config%output_width, int64) * real64_bytes() + &
                 int(layer%config%input_width, int64) * real64_bytes()
      ! Closed form: every page has the same shape, so this is O(1) and can be
      ! recomputed on every insertion without turning a streaming run quadratic.
      bytes = int(layer%page_count, int64) * per_page
      router_bytes = int(layer%config%input_width, int64) * int(layer%config%router_bits, int64) * &
                     int(layer%config%active_experts, int64) * real64_bytes()
      bytes = bytes + router_bytes + int(layer%table_size, int64) * 4_int64
   end function resident_bytes_of

   !> Pure, because it has no dummy arguments and no side effect: `bytes` is the
   !> result name and must not be declared again in the specification part.
   pure integer(int64) function real64_bytes() result(bytes)

      bytes = storage_size(1.0_real64) / 8_int64
   end function real64_bytes

   !----------------------------------------------------------------------------
   !  Lifecycle
   !----------------------------------------------------------------------------

   !> Allocates the router and an empty page store.  Note what is *not* allocated
   !> here: there is no W_total-sized array anywhere in this type.
   subroutine sparse_layer_init(layer, config, status)
      type(sparse_layer_t), intent(inout) :: layer
      type(sparse_layer_config_t), intent(in) :: config
      integer(int32), intent(out) :: status
      integer(int32) :: allocation_status, bit, probe, column_index
      integer(int64) :: state

      status = SE_OK
      if (.not. sparse_layer_config_is_valid(config)) then
         status = SE_ERR_INVALID_ARGUMENT
         return
      end if
      call sparse_layer_destroy(layer)
      layer%config = config
      allocate(layer%router(config%input_width, config%router_bits, config%active_experts), &
               layer%router_norm(config%router_bits, config%active_experts), &
               layer%pages(max(8_int32, config%active_experts)), &
               layer%slot_table(16_int32), &
               layer%dirty_slots(max(1_int32, config%active_experts)), stat=allocation_status)
      if (allocation_status /= 0) then
         status = SE_ERR_ALLOCATION
         return
      end if
      layer%page_capacity = max(8_int32, config%active_experts)
      layer%table_size = 16_int32
      layer%slot_table = 0_int32
      layer%page_count = 0_int32
      layer%dirty_count = 0_int32
      layer%dirty_slots = 0_int32
      layer%stats = sparse_layer_stats_t()
      ! Frozen, deterministic hyperplanes.  Cost is O(bits * width) once, at
      ! construction; nothing about it scales with capacity.
      state = splitmix64_of(ieor(config%seed, int(z'5EED5EED', int64)))
      do probe = 1_int32, config%active_experts
         do bit = 1_int32, config%router_bits
            ! Serial: the hash stream is a stateful sequence (splitmix64 advances
            ! its argument), so these writes are ordered, not independent.  It
            ! runs once per layer, not per step.
            do column_index = 1_int32, config%input_width
               layer%router(column_index, bit, probe) = uniform_signed(splitmix64(state))
            end do
            layer%router_norm(bit, probe) = sqrt(dot_product(layer%router(:, bit, probe), &
                                                             layer%router(:, bit, probe)))
            if (layer%router_norm(bit, probe) <= 0.0_real64) then
               status = SE_ERR_NUMERICAL
               return
            end if
         end do
      end do
   end subroutine sparse_layer_init

   !> Releases every page and the router.  A destroyed layer is reusable: all
   !> scalar members are reset, so `init` after `destroy` never sees stale state.
   subroutine sparse_layer_destroy(layer)
      type(sparse_layer_t), intent(inout) :: layer
      integer(int32) :: slot

      do slot = 1_int32, layer%page_count
         call release_page(page=layer%pages(slot))
      end do
      if (allocated(layer%pages)) deallocate(layer%pages)
      if (allocated(layer%slot_table)) deallocate(layer%slot_table)
      if (allocated(layer%dirty_slots)) deallocate(layer%dirty_slots)
      if (allocated(layer%router)) deallocate(layer%router)
      if (allocated(layer%router_norm)) deallocate(layer%router_norm)
      layer%page_count = 0_int32
      layer%page_capacity = 0_int32
      layer%table_size = 0_int32
      layer%dirty_count = 0_int32
      layer%stats = sparse_layer_stats_t()
   end subroutine sparse_layer_destroy

   subroutine release_page(page)
      type(sparse_page_t), intent(inout) :: page

      if (allocated(page%weight)) deallocate(page%weight)
      if (allocated(page%bias)) deallocate(page%bias)
      if (allocated(page%weight_gradient)) deallocate(page%weight_gradient)
      if (allocated(page%bias_gradient)) deallocate(page%bias_gradient)
      if (allocated(page%velocity)) deallocate(page%velocity)
      if (allocated(page%energy)) deallocate(page%energy)
      page%expert_id = -1_int64
      page%state_ready = .false.
      page%gradient_ready = .false.
   end subroutine release_page

   !----------------------------------------------------------------------------
   !  Forward: O(k) routing and O(k) dense work
   !----------------------------------------------------------------------------

   !> Routes one activation to k experts and evaluates only those experts.
   !>
   !> Order of operations, all of it O(k) or O(router_bits * width):
   !>   1. per probe: LSH signature (O(bits*width)) and gate score (O(width));
   !>   2. expert id = f(signature, probe): a direct address, no ranking;
   !>   3. deduplicate within the k selected ids (O(k^2), k <= 64);
   !>   4. materialise the page if this expert was never touched;
   !>   5. softmax over the k scores, then the weighted sum of k expert outputs.
   subroutine sparse_layer_forward(layer, input, output, cache, status)
      type(sparse_layer_t), intent(inout) :: layer
      real(real64), intent(in) :: input(:)
      real(real64), intent(out) :: output(:)
      type(sparse_forward_t), intent(inout) :: cache
      integer(int32), intent(out) :: status
      integer(int32) :: wanted, index, other, out_index, probe, slot, allocation_status
      integer(int64) :: signature, expert_id
      real(real64) :: maximum, normalizer
      logical :: duplicate

      status = SE_OK
      wanted = layer%config%active_experts
      if (size(input, kind=int32) /= layer%config%input_width) then
         status = SE_ERR_INVALID_ARGUMENT
         return
      end if
      if (size(output, kind=int32) /= layer%config%output_width) then
         status = SE_ERR_INVALID_ARGUMENT
         return
      end if
      if (.not. allocated(cache%expert_id)) then
         allocate(cache%expert_id(wanted), cache%page_slot(wanted), cache%gate(wanted), &
                  cache%score(wanted), cache%expert_output(layer%config%output_width, wanted), &
                  cache%input_copy(layer%config%input_width), stat=allocation_status)
         if (allocation_status /= 0) then
            status = SE_ERR_ALLOCATION
            return
         end if
      end if
      if (size(cache%expert_id, kind=int32) /= wanted) then
         deallocate(cache%expert_id, cache%page_slot, cache%gate, cache%score, &
                    cache%expert_output, cache%input_copy)
         allocate(cache%expert_id(wanted), cache%page_slot(wanted), cache%gate(wanted), &
                  cache%score(wanted), cache%expert_output(layer%config%output_width, wanted), &
                  cache%input_copy(layer%config%input_width), stat=allocation_status)
         if (allocation_status /= 0) then
            status = SE_ERR_ALLOCATION
            return
         end if
      end if
      cache%input_copy = input
      cache%active_count = 0_int32
      ! ---- routing: k probes, each O(bits * width), no capacity term ----------
      do index = 1_int32, wanted
         probe = index
         call lsh_signature(layer, input, probe, signature, status)
         if (status /= SE_OK) return
         expert_id = routed_expert(signature, layer%config)
         ! Distinct experts only: a collision inside one activation is resolved by
         ! walking the address space, which is O(1) expected and bounded by k.
         duplicate = .true.
         do while (duplicate)
            duplicate = .false.
            do other = 1_int32, cache%active_count
               if (cache%expert_id(other) == expert_id) then
                  duplicate = .true.
                  exit
               end if
            end do
            if (duplicate) expert_id = modulo(expert_id + 1_int64, layer%config%expert_capacity)
         end do
         slot = find_page_slot(layer, expert_id)
         if (slot == 0_int32) then
            ! Cold expert: this is the only place where a parameter array comes
            ! into existence, and it happens once per expert ever touched.
            block
               type(sparse_page_t) :: fresh
               call materialise_page(fresh, layer%config, expert_id, status)
               if (status /= SE_OK) return
               call insert_page(layer, fresh, status)
               if (status /= SE_OK) then
                  call release_page(fresh)
                  return
               end if
               slot = layer%page_count
            end block
         end if
         cache%active_count = cache%active_count + 1_int32
         cache%expert_id(cache%active_count) = expert_id
         cache%page_slot(cache%active_count) = slot
         cache%score(cache%active_count) = probe_score(layer, input, probe)
         layer%stats%router_evaluations = layer%stats%router_evaluations + 1_int64
      end do
      ! ---- gating: softmax over the k selected scores only --------------------
      maximum = cache%score(1_int32)
      do index = 2_int32, cache%active_count
         maximum = max(maximum, cache%score(index))
      end do
      normalizer = 0.0_real64
      do index = 1_int32, cache%active_count
         cache%gate(index) = exp(cache%score(index) - maximum)
         normalizer = normalizer + cache%gate(index)
      end do
      if (normalizer <= 0.0_real64) then
         status = SE_ERR_NUMERICAL
         return
      end if
      do index = 1_int32, cache%active_count
         cache%gate(index) = cache%gate(index) / normalizer
      end do
      ! ---- active work: k experts only ---------------------------------------
      output = 0.0_real64
      do index = 1_int32, cache%active_count
         slot = cache%page_slot(index)
         call expert_forward(layer%pages(slot), input, cache%expert_output(:, index))
         do concurrent (out_index=1_int32:layer%config%output_width)
            output(out_index) = output(out_index) + &
                                cache%gate(index) * cache%expert_output(out_index, index)
         end do
      end do
      cache%ready = .true.
      layer%stats%steps = layer%stats%steps + 1_int64
      layer%stats%active_parameters = sparse_layer_active_parameter_count(layer)
      layer%stats%active_flops = active_flops_per_step(layer)
      layer%stats%cumulative_active_parameters = layer%stats%cumulative_active_parameters + &
                                                layer%stats%active_parameters
      layer%stats%cumulative_active_flops = layer%stats%cumulative_active_flops + layer%stats%active_flops
   end subroutine sparse_layer_forward

   !> y = W x + b for one expert.  Column-major: the inner (concurrent) loop walks
   !> a column of W, which is contiguous, so both the load and the store are
   !> unit-stride.
   pure subroutine expert_forward(page, input, output)
      type(sparse_page_t), intent(in) :: page
      real(real64), intent(in) :: input(:)
      real(real64), intent(out) :: output(:)
      integer(int32) :: column, row

      output = page%bias
      do column = 1_int32, size(page%weight, 2, kind=int32)
         do concurrent (row=1_int32:size(page%weight, 1, kind=int32))
            output(row) = output(row) + page%weight(row, column) * input(column)
         end do
      end do
   end subroutine expert_forward

   !> Analytic FLOP count of one step, from (k, in, out) alone.  Counted as two
   !> FLOPs per multiply-add, and it is the same expression for every capacity --
   !> which is the whole point.
   pure integer(int64) function active_flops_per_step(layer) result(flops)
      type(sparse_layer_t), intent(in) :: layer
      integer(int64) :: active, in_width, out_width, router

      active = int(layer%config%active_experts, int64)
      in_width = int(layer%config%input_width, int64)
      out_width = int(layer%config%output_width, int64)
      ! forward: out*in MACs + out adds; backward: out*in MACs for dW, out for db,
      ! in*out MACs for dx; update: 4 ops per parameter and one sqrt per column.
      flops = active * (2_int64 * (2_int64 * in_width * out_width + out_width) + &
                        in_width * out_width + 8_int64 * out_width * in_width)
      ! Router: k probes * bits * width MACs.
      router = active * int(layer%config%router_bits, int64) * in_width * 2_int64
      flops = flops + router
   end function active_flops_per_step

   !----------------------------------------------------------------------------
   !  Backward: touches only the pages the forward pass touched
   !----------------------------------------------------------------------------

   !> dL/dx and the parameter gradients of the k active experts.
   !>
   !> The gate is treated as a constant of the routing map (the router is
   !> frozen), which is what makes the sparse backward exact for the parameters it
   !> does own: every expert that contributed to the output receives the exact
   !> gradient, and every other expert is not touched at all.
   subroutine sparse_layer_backward(layer, cache, grad_output, grad_input, status)
      type(sparse_layer_t), intent(inout) :: layer
      type(sparse_forward_t), intent(in) :: cache
      real(real64), intent(in) :: grad_output(:)
      real(real64), intent(out) :: grad_input(:)
      integer(int32), intent(out) :: status
      integer(int32) :: index, slot

      status = SE_OK
      if (.not. cache%ready .or. cache%active_count <= 0_int32) then
         status = SE_ERR_ORDER
         return
      end if
      if (size(grad_output, kind=int32) /= layer%config%output_width .or. &
          size(grad_input, kind=int32) /= layer%config%input_width) then
         status = SE_ERR_INVALID_ARGUMENT
         return
      end if
      grad_input = 0.0_real64
      do index = 1_int32, cache%active_count
         slot = cache%page_slot(index)
         call expert_backward(layer%pages(slot), cache%input_copy, grad_output, &
                              cache%gate(index), grad_input)
         ! Register the page once, even if backward is called twice before an
         ! update: the list stays a set, and it is bounded by k because the
         ! forward pass deduplicated the routed ids.
         if (.not. page_is_dirty(layer, slot)) then
            layer%dirty_count = layer%dirty_count + 1_int32
            layer%dirty_slots(layer%dirty_count) = slot
         end if
      end do
   end subroutine sparse_layer_backward

   !> Parameter gradients and the input gradient of one expert.  Nothing here is
   !> O(capacity): `grad_input` is accumulated in place and the weight gradients
   !> are written into the page's own buffer.
   pure subroutine expert_backward(page, input, grad_output, gate, grad_input)
      type(sparse_page_t), intent(inout) :: page
      real(real64), intent(in) :: input(:)
      real(real64), intent(in) :: grad_output(:)
      real(real64), intent(in) :: gate
      real(real64), intent(inout) :: grad_input(:)
      integer(int32) :: column, row
      real(real64) :: scale

      do column = 1_int32, size(page%weight, 2, kind=int32)
         scale = gate * input(column)
         do concurrent (row=1_int32:size(page%weight, 1, kind=int32))
            page%weight_gradient(row, column) = page%weight_gradient(row, column) + &
                                                scale * grad_output(row)
         end do
         ! dx_j = sum_i W_ij dy_i: one contiguous dot product per column.
         grad_input(column) = grad_input(column) + gate * dot_product(page%weight(:, column), grad_output)
      end do
      do concurrent (row=1_int32:size(page%weight, 1, kind=int32))
         page%bias_gradient(row) = page%bias_gradient(row) + gate * grad_output(row)
      end do
      page%gradient_ready = .true.
      page%state_ready = .true.
   end subroutine expert_backward

   !----------------------------------------------------------------------------
   !  Sparse update: NOVA-1 on the pages that own a gradient
   !----------------------------------------------------------------------------

   !> Applies the NOVA-1 rule to every page with a pending gradient and clears the
   !> gradient.  Untouched pages are not visited, so the optimizer's traffic is
   !> O(k) per step, like the forward and backward passes.
   subroutine sparse_layer_apply_update(layer, status)
      type(sparse_layer_t), intent(inout) :: layer
      integer(int32), intent(out) :: status
      integer(int32) :: index, slot

      status = SE_OK
      do index = 1_int32, layer%dirty_count
         slot = layer%dirty_slots(index)
         call nova_update_page(layer%pages(slot), layer%config)
         layer%pages(slot)%gradient_ready = .false.
         layer%stats%updates_applied = layer%stats%updates_applied + 1_int64
      end do
      layer%dirty_count = 0_int32
   end subroutine sparse_layer_apply_update

   !> Membership test for the dirty list.  O(k) with k <= 64, called k times per
   !> backward pass, so it costs less than one expert's matrix product.
   pure logical function page_is_dirty(layer, slot) result(dirty)
      type(sparse_layer_t), intent(in) :: layer
      integer(int32), intent(in) :: slot
      integer(int32) :: index

      dirty = .false.
      do index = 1_int32, layer%dirty_count
         if (layer%dirty_slots(index) == slot) then
            dirty = .true.
            return
         end if
      end do
   end function page_is_dirty

   !> NOVA-1 shape: per-column energy of the rank-one gradient, one velocity
   !> array, agreement-triggered restart, decoupled decay, and no transcendental
   !> in the element chain.
   pure subroutine nova_update_page(page, config)
      type(sparse_page_t), intent(inout) :: page
      type(sparse_layer_config_t), intent(in) :: config
      integer(int32) :: column, row
      real(real64) :: gradient_norm, agreement, decayed_energy, step_scale, gamma
      real(real64) :: keep, epsilon_scale

      keep = 1.0_real64 - config%learning_rate * config%weight_decay
      epsilon_scale = config%epsilon
      do column = 1_int32, size(page%weight, 2, kind=int32)
         gradient_norm = dot_product(page%weight_gradient(:, column), page%weight_gradient(:, column))
         gradient_norm = gradient_norm / real(max(1_int32, size(page%weight, 1, kind=int32)), real64)
         agreement = dot_product(page%weight_gradient(:, column), page%velocity(:, column))
         decayed_energy = config%energy_decay * page%energy(column) + &
                          (1.0_real64 - config%energy_decay) * gradient_norm
         page%energy(column) = decayed_energy
         gamma = config%momentum
         if (agreement < 0.0_real64) gamma = config%momentum * config%restart_damp
         step_scale = config%learning_rate / (sqrt(max(decayed_energy, 0.0_real64)) + epsilon_scale)
         do concurrent (row=1_int32:size(page%weight, 1, kind=int32))
            ! Every reference below is to element (row, column), so iterations
            ! touch disjoint memory: this is a legal `do concurrent` body with
            ! no shared temporary (a scalar scratch variable here would be a race).
            page%velocity(row, column) = gamma * page%velocity(row, column) + &
                                         page%weight_gradient(row, column)
            page%weight(row, column) = keep * page%weight(row, column) - &
                                       step_scale * page%velocity(row, column)
            page%weight_gradient(row, column) = 0.0_real64
         end do
      end do
      do concurrent (row=1_int32:size(page%weight, 1, kind=int32))
         page%bias(row) = keep * page%bias(row) - config%learning_rate * page%bias_gradient(row)
         page%bias_gradient(row) = 0.0_real64
      end do
   end subroutine nova_update_page

end module sparse_expert_layer
