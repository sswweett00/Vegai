! SPDX-License-Identifier: MIT
!==============================================================================
!  hdc_pretraining.f90
!  Two-pass, streaming, single-pass-Hebbian pretraining pipeline for Vegai.
!
!  This trains a distributional HDC context model, not a Transformer and not a
!  gradient-based language model.  Each active token owns a deterministic base
!  hypervector and a bounded-plasticity context prototype learned from ordered
!  windowed contexts.  The implementation is designed for reproducible corpus
!  passes, bounded memory, checkpoint/resume, and explicit data-loss metrics.
!==============================================================================
module hdc_pretraining
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32, iostat_end, iostat_eor, error_unit
   use hyper_types
   use text_tokenizer
   use vocabulary
   implicit none
   private

   integer(int32), parameter, public :: PT_OK = 0_int32
   integer(int32), parameter, public :: PT_ERR_INVALID_ARGUMENT = -500_int32
   integer(int32), parameter, public :: PT_ERR_IO = -501_int32
   integer(int32), parameter, public :: PT_ERR_VOCABULARY = -502_int32
   integer(int32), parameter, public :: PT_ERR_ALLOCATION = -503_int32
   integer(int32), parameter, public :: PT_ERR_MODEL_NOT_READY = -504_int32
   integer(int32), parameter, public :: PT_ERR_TOKENIZATION = -505_int32
   integer(int32), parameter, public :: PT_ERR_INCOMPATIBLE_CONFIG = -506_int32
   integer(int32), parameter, public :: PT_ERR_CHECKPOINT = -507_int32
   integer(int32), parameter, public :: PT_ERR_COUNTER_OVERFLOW = -508_int32
   integer(int32), parameter, public :: PT_ERR_NOT_FOUND = -509_int32
   integer(int32), parameter, public :: PT_ERR_EMPTY_VOCABULARY = -510_int32
   integer(int32), parameter, public :: PT_ERR_HYPERVECTOR = -511_int32
   integer(int32), parameter, public :: PT_ERR_RECORD_TOO_LONG = -512_int32
   ! Optional execution-provider failure.  This is deliberately distinct from
   ! semantic HDC errors so an explicit CPU fallback can remain truthful.
   integer(int32), parameter, public :: PT_ERR_CONTEXT_PROVIDER = -513_int32

   ! Context encoders share the same deterministic update and checkpoint
   ! contracts.  LEGACY_POSITIONAL retains the original signed-counter path for
   ! regression comparison; POSITIONAL is its bit-identical bit-sliced fast
   ! implementation; HYBRID adds an order-invariant semantic view alongside
   ! the positional view to strengthen topical retrieval; HIERARCHICAL adds a
   ! bounded whole-record view only for sufficiently long records.
   integer(int32), parameter, public :: PT_CONTEXT_LEGACY_POSITIONAL = 1_int32
   integer(int32), parameter, public :: PT_CONTEXT_POSITIONAL = 2_int32
   integer(int32), parameter, public :: PT_CONTEXT_HYBRID = 3_int32
   ! HIERARCHICAL retains hybrid's local positional/topic evidence and adds a
   ! bounded whole-record topical summary for records with many active tokens.
   integer(int32), parameter, public :: PT_CONTEXT_HIERARCHICAL = 4_int32
   ! COHERENT keeps the same bounded item budget as HIERARCHICAL but uses a
   ! cross-fold record view for most global evidence.  The focal token appears
   ! only once in the full-record contribution rather than in every global
   ! contribution, reducing self-identity echo without mutable model reads.
   integer(int32), parameter, public :: PT_CONTEXT_COHERENT = 5_int32
   ! DEBIASED keeps HYBRID local evidence but admits the long-record global
   ! channel only when a record has enough *distinct* active token slots. Its
   ! global majority sees each slot at most once, so boilerplate/repetition
   ! cannot win the whole-record vote solely by occurrence count.
   integer(int32), parameter, public :: PT_CONTEXT_DEBIASED = 6_int32

   ! Execution-only context providers.  CPU is the portable authoritative
   ! backend.  OPENVINO can offload only immutable, pre-positioned context
   ! bundling through an optional external worker; source-ordered prototype
   ! commits always remain in this Fortran module.
   integer(int32), parameter, public :: PT_CONTEXT_PROVIDER_CPU = 1_int32
   integer(int32), parameter, public :: PT_CONTEXT_PROVIDER_OPENVINO = 2_int32

   integer(int32), parameter :: HIERARCHICAL_MIN_ACTIVE_TOKENS = 12_int32
   integer(int32), parameter :: HIERARCHICAL_GLOBAL_CONTEXT_WEIGHT = 3_int32
   ! The de-biased global channel uses the same cardinality threshold, applied
   ! to distinct active vocabulary slots rather than raw occurrences.
   integer(int32), parameter :: DEBIASED_MIN_DISTINCT_ACTIVE_TOKENS = HIERARCHICAL_MIN_ACTIVE_TOKENS
   ! Direct-mapped, exact-token-checked execution cache. It is intentionally
   ! bounded and transient: no cache data can affect checkpoint bytes.
   integer(int32), parameter :: PRETRAIN_LOOKUP_CACHE_SLOTS = 4096_int32
   integer(int64), parameter :: PRETRAIN_LOOKUP_CACHE_MASK = int(PRETRAIN_LOOKUP_CACHE_SLOTS - 1_int32, int64)
   integer(int64), parameter :: OPENVINO_LOW32_MASK = int(z'00000000FFFFFFFF', int64)
   ! Persistent worker control protocol.  These paths and reports are strictly
   ! transient execution infrastructure, never checkpoint/model state.
   character(len=8), parameter :: OPENVINO_SESSION_READY_MAGIC = 'VGHDCW01'
   integer(int32), parameter :: OPENVINO_SESSION_READY_VERSION = 1_int32
   integer(int32), parameter :: OPENVINO_POLL_MILLISECONDS = 1_int32

   character(len=8), parameter :: CHECKPOINT_MAGIC = 'VGHDC001'
   ! v7 binds the unbiased Floyd/complement exact-mask sampler to a checkpoint
   ! lineage. v4-v6 remain readable and explicitly retain their historical
   ! bounded-retry sampler on resume, so an old checkpoint never silently
   ! changes its future deterministic trajectory. The wire layout is stable;
   ! the version is an algorithm/continuation contract as well as a parser key.
   integer(int32), parameter :: CHECKPOINT_VERSION = 7_int32
   integer(int32), parameter :: CHECKPOINT_PRE_COHERENT_VERSION = 5_int32
   integer(int32), parameter :: CHECKPOINT_COHERENT_LEGACY_VERSION = 6_int32
   integer(int32), parameter :: CHECKPOINT_MIN_VERSION = 4_int32
   integer(int32), parameter, public :: PT_PLASTICITY_SAMPLER_LEGACY = 1_int32
   integer(int32), parameter, public :: PT_PLASTICITY_SAMPLER_FLOYD_COMPLEMENT = 2_int32

   type, public :: hdc_pretrain_config_t
      integer(int32) :: max_vocabulary = 20000_int32
      integer(int64) :: min_frequency = 2_int64
      integer(int32) :: context_window = 4_int32
      integer(int32) :: position_shift_bits = 73_int32
      integer(int32) :: context_mode = PT_CONTEXT_HIERARCHICAL
      integer(int32) :: epochs = 1_int32
      integer(int32) :: initial_plasticity_percent = 25_int32
      integer(int32) :: minimum_plasticity_percent = 1_int32
      integer(int32) :: plasticity_decay_events = 32_int32
      integer(int64) :: tie_seed = int(z'6A09E667F3BCC909', int64)
      logical :: strict_tokenization = .false.
      ! Benchmark/regression switch only: both paths are semantically exact.
      ! It is deliberately not checkpoint state, so a fast resume can continue
      ! a checkpoint originally trained with the reference implementation.
      logical :: reference_context_kernel = .false.
      ! Execution-only switch.  Context construction for one input record can
      ! run under OpenMP, while prototype commits remain source-ordered.  This
      ! retains byte-identical semantic state even when repeated tokens occur
      ! on the same line.  A compiler without OpenMP ignores the directives and
      ! remains serial; this flag is intentionally not checkpoint state.
      logical :: parallel_context_build = .true.
      ! A bounded direct-mapped cache avoids repeated collision-safe vocabulary
      ! probes after the first occurrence of an exact token. Cache collisions
      ! always miss and fall back to the canonical lookup, so this affects only
      ! execution cost and is intentionally excluded from checkpoint state.
      logical :: vocabulary_lookup_cache = .true.
      ! Do not launch an OpenMP team for a tiny record: scheduling and staging
      ! overhead can exceed the immutable HDC arithmetic.  This is an
      ! execution-only adaptive threshold, not checkpoint/model semantics.
      integer(int32) :: parallel_min_contexts = 32_int32
      ! Optional OpenVINO execution settings; none of these affect checkpoint
      ! semantics.  The worker receives pre-positioned immutable base-vector
      ! items and returns contexts; CPU applies commits in source order.
      integer(int32) :: context_provider = PT_CONTEXT_PROVIDER_CPU
      character(len=512) :: openvino_worker = ''
      character(len=512) :: openvino_python = 'python3'
      character(len=512) :: openvino_work_dir = '.'
      character(len=256) :: openvino_device = 'auto'
      integer(int32) :: openvino_min_contexts = 64_int32
      ! Upper bound for startup/request/shutdown control-plane waits.  The
      ! previous one-shot process call could block indefinitely; the resident
      ! protocol fails closed after this bounded interval instead.
      integer(int32) :: openvino_timeout_milliseconds = 120000_int32
      logical :: openvino_require_accelerator = .false.
      logical :: openvino_fallback_cpu = .true.
   end type hdc_pretrain_config_t

   ! This is transient execution truth, intentionally excluded from semantic
   ! metrics/checkpoints.  A run can therefore distinguish CPU fallback from
   ! device context dispatch without changing a resume contract.
   type, public :: pretrain_execution_report_t
      integer(int64) :: cpu_contexts = 0_int64
      ! CPU contexts are split by the adaptive scheduler decision.  A
      ! `parallel` value means the OpenMP worksharing path was selected; an
      ! OpenMP-less compiler still executes that directive serially.
      integer(int64) :: cpu_serial_contexts = 0_int64
      integer(int64) :: cpu_parallel_contexts = 0_int64
      integer(int64) :: cpu_serial_batches = 0_int64
      integer(int64) :: cpu_parallel_batches = 0_int64
      ! Bounded-cache telemetry is execution-only. A miss includes a safe
      ! direct-map collision and always falls back to the canonical vocabulary
      ! lookup, so hit/miss counters cannot alter model semantics.
      integer(int64) :: vocabulary_lookup_cache_hits = 0_int64
      integer(int64) :: vocabulary_lookup_cache_misses = 0_int64
      ! DEBIASED context telemetry distinguishes records that contributed a
      ! distinct-token global topic from repeated/boilerplate-heavy records
      ! whose global channel was intentionally suppressed.
      integer(int64) :: debiased_topic_records = 0_int64
      integer(int64) :: debiased_topic_suppressed_records = 0_int64
      integer(int64) :: debiased_topic_distinct_tokens = 0_int64
      integer(int64) :: openvino_contexts = 0_int64
      ! A resident worker session can process many record requests.  Keep the
      ! historic invocation counter as a request counter and report sessions
      ! separately so runtime output does not imply per-line process startup.
      integer(int64) :: openvino_worker_invocations = 0_int64
      integer(int64) :: openvino_worker_sessions = 0_int64
      ! Contexts deliberately kept on native CPU when an optional policy
      ! resolves only to CPU.  This is not an OpenVINO CPU execution claim.
      integer(int64) :: openvino_cpu_bypass_contexts = 0_int64
      integer(int64) :: openvino_fallback_contexts = 0_int64
      logical :: openvino_cpu_execution = .false.
      logical :: openvino_gpu_execution = .false.
      logical :: openvino_npu_execution = .false.
      logical :: openvino_provider_requested = .false.
   end type pretrain_execution_report_t

   ! Internal lifetime for a single optional provider service.  The worker owns
   ! no mutable HDC model state; it only caches compiled immutable-context
   ! graphs.  Keeping this state here prevents a Python/OpenVINO launch and
   ! compile sequence for every corpus record.
   type :: openvino_context_session_t
      logical :: active = .false.
      logical :: launch_attempted = .false.
      logical :: cpu_only_bypass = .false.
      character(len=1024) :: directory = ''
      character(len=1024) :: ready_path = ''
      character(len=1024) :: startup_error_path = ''
      character(len=1024) :: request_path = ''
      character(len=1024) :: request_ready_path = ''
      character(len=1024) :: response_path = ''
      character(len=1024) :: response_ready_path = ''
      character(len=1024) :: response_error_path = ''
      character(len=1024) :: shutdown_path = ''
      character(len=1024) :: stopped_path = ''
      character(len=1024) :: stopped_ack_path = ''
   end type openvino_context_session_t

   type, public :: pretrain_metrics_t
      integer(int64) :: corpus_lines = 0_int64
      integer(int64) :: raw_tokens = 0_int64
      integer(int64) :: active_tokens = 0_int64
      integer(int64) :: oov_tokens = 0_int64
      integer(int64) :: context_pairs = 0_int64
      integer(int64) :: center_updates = 0_int64
      integer(int64) :: tokenizer_truncations = 0_int64
      integer(int64) :: non_ascii_bytes = 0_int64
      integer(int64) :: overlong_records = 0_int64
   end type pretrain_metrics_t

   type, public :: hdc_pretrain_model_t
      type(vocabulary_t) :: vocabulary
      logical :: ready = .false.
      integer(int32) :: context_window = 0_int32
      integer(int32) :: position_shift_bits = 0_int32
      integer(int32) :: context_mode = 0_int32
      integer(int32) :: initial_plasticity_percent = 0_int32
      integer(int32) :: minimum_plasticity_percent = 0_int32
      integer(int32) :: plasticity_decay_events = 0_int32
      ! Kept outside the historical wire layout; the checkpoint format version
      ! selects it on load and is selected from it on save.
      integer(int32) :: plasticity_sampler = PT_PLASTICITY_SAMPLER_FLOYD_COMPLEMENT
      integer(int64) :: tie_seed = 0_int64
      integer(int64) :: training_epochs_completed = 0_int64
      type(pretrain_metrics_t) :: totals
      ! Word-major packed SoA data planes: each column is one hypervector.
      integer(int64), allocatable :: base_bits(:, :)
      integer(int64), allocatable :: semantic_bits(:, :)
      integer(int64), allocatable :: experience_count(:)
   end type hdc_pretrain_model_t

   public :: pretrain_build_vocabulary, pretrain_extend_vocabulary, pretrain_train_file
   public :: pretrain_model_destroy, pretrain_model_is_ready
   public :: pretrain_metrics_reset, pretrain_model_footprint_bytes
   public :: pretrain_checkpoint_save, pretrain_checkpoint_load
   public :: pretrain_lookup_token, pretrain_get_semantic_hv, pretrain_nearest_semantic
   public :: pretrain_config_from_model, pretrain_context_mode_name, pretrain_context_provider_name
   public :: pretrain_plasticity_sampler_name
   public :: pretrain_execution_report_reset

contains

   pure function pretrain_model_is_ready(model) result(ready)
      type(hdc_pretrain_model_t), intent(in) :: model
      logical :: ready

      ready = model%ready .and. vocab_is_initialized(model%vocabulary) .and. &
              allocated(model%base_bits) .and. allocated(model%semantic_bits) .and. &
              allocated(model%experience_count) .and. &
              (model%plasticity_sampler == PT_PLASTICITY_SAMPLER_LEGACY .or. &
               model%plasticity_sampler == PT_PLASTICITY_SAMPLER_FLOYD_COMPLEMENT)
   end function pretrain_model_is_ready

   pure function pretrain_context_mode_name(context_mode) result(name)
      integer(int32), intent(in) :: context_mode
      character(len=18) :: name

      select case (context_mode)
      case (PT_CONTEXT_LEGACY_POSITIONAL)
         name = 'legacy'
      case (PT_CONTEXT_POSITIONAL)
         name = 'positional'
      case (PT_CONTEXT_HYBRID)
         name = 'hybrid'
      case (PT_CONTEXT_HIERARCHICAL)
         name = 'hierarchical'
      case (PT_CONTEXT_COHERENT)
         name = 'coherent'
      case (PT_CONTEXT_DEBIASED)
         name = 'debiased'
      case default
         name = 'invalid'
      end select
   end function pretrain_context_mode_name

   ! Human-readable continuation identity for inspect/report surfaces. This is
   ! selected by checkpoint version, not a user-tunable training option.
   pure function pretrain_plasticity_sampler_name(plasticity_sampler) result(name)
      integer(int32), intent(in) :: plasticity_sampler
      character(len=24) :: name

      select case (plasticity_sampler)
      case (PT_PLASTICITY_SAMPLER_LEGACY)
         name = 'legacy-bounded-retry'
      case (PT_PLASTICITY_SAMPLER_FLOYD_COMPLEMENT)
         name = 'floyd-complement'
      case default
         name = 'invalid'
      end select
   end function pretrain_plasticity_sampler_name

   pure function pretrain_context_provider_name(context_provider) result(name)
      integer(int32), intent(in) :: context_provider
      character(len=10) :: name

      select case (context_provider)
      case (PT_CONTEXT_PROVIDER_CPU)
         name = 'cpu'
      case (PT_CONTEXT_PROVIDER_OPENVINO)
         name = 'openvino'
      case default
         name = 'invalid'
      end select
   end function pretrain_context_provider_name

   subroutine pretrain_execution_report_reset(report)
      type(pretrain_execution_report_t), intent(inout) :: report

      report%cpu_contexts = 0_int64
      report%cpu_serial_contexts = 0_int64
      report%cpu_parallel_contexts = 0_int64
      report%cpu_serial_batches = 0_int64
      report%cpu_parallel_batches = 0_int64
      report%vocabulary_lookup_cache_hits = 0_int64
      report%vocabulary_lookup_cache_misses = 0_int64
      report%debiased_topic_records = 0_int64
      report%debiased_topic_suppressed_records = 0_int64
      report%debiased_topic_distinct_tokens = 0_int64
      report%openvino_contexts = 0_int64
      report%openvino_worker_invocations = 0_int64
      report%openvino_worker_sessions = 0_int64
      report%openvino_cpu_bypass_contexts = 0_int64
      report%openvino_fallback_contexts = 0_int64
      report%openvino_cpu_execution = .false.
      report%openvino_gpu_execution = .false.
      report%openvino_npu_execution = .false.
      report%openvino_provider_requested = .false.
   end subroutine pretrain_execution_report_reset

   subroutine pretrain_metrics_reset(metrics)
      type(pretrain_metrics_t), intent(inout) :: metrics

      metrics%corpus_lines = 0_int64
      metrics%raw_tokens = 0_int64
      metrics%active_tokens = 0_int64
      metrics%oov_tokens = 0_int64
      metrics%context_pairs = 0_int64
      metrics%center_updates = 0_int64
      metrics%tokenizer_truncations = 0_int64
      metrics%non_ascii_bytes = 0_int64
      metrics%overlong_records = 0_int64
   end subroutine pretrain_metrics_reset

   subroutine pretrain_model_destroy(model)
      type(hdc_pretrain_model_t), intent(inout) :: model

      call vocab_destroy(model%vocabulary)
      if (allocated(model%base_bits)) deallocate(model%base_bits)
      if (allocated(model%semantic_bits)) deallocate(model%semantic_bits)
      if (allocated(model%experience_count)) deallocate(model%experience_count)
      model%ready = .false.
      model%context_window = 0_int32
      model%position_shift_bits = 0_int32
      model%context_mode = 0_int32
      model%initial_plasticity_percent = 0_int32
      model%minimum_plasticity_percent = 0_int32
      model%plasticity_decay_events = 0_int32
      model%plasticity_sampler = PT_PLASTICITY_SAMPLER_FLOYD_COMPLEMENT
      model%tie_seed = 0_int64
      model%training_epochs_completed = 0_int64
      call pretrain_metrics_reset(model%totals)
   end subroutine pretrain_model_destroy

   subroutine validate_pretrain_config(config, status)
      type(hdc_pretrain_config_t), intent(in) :: config
      integer(int32), intent(out) :: status
      integer(int64) :: maximum_shift

      status = PT_OK
      if (config%max_vocabulary <= 0_int32 .or. config%min_frequency <= 0_int64) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%context_window <= 0_int32 .or. &
          config%context_window >= MAX_TOKENS_PER_LINE) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%parallel_min_contexts <= 0_int32 .or. &
          config%parallel_min_contexts > MAX_TOKENS_PER_LINE) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%position_shift_bits == 0_int32 .or. config%epochs <= 0_int32) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%context_mode /= PT_CONTEXT_LEGACY_POSITIONAL .and. &
          config%context_mode /= PT_CONTEXT_POSITIONAL .and. &
          config%context_mode /= PT_CONTEXT_HYBRID .and. &
          config%context_mode /= PT_CONTEXT_HIERARCHICAL .and. &
          config%context_mode /= PT_CONTEXT_COHERENT .and. &
          config%context_mode /= PT_CONTEXT_DEBIASED) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%context_provider /= PT_CONTEXT_PROVIDER_CPU .and. &
          config%context_provider /= PT_CONTEXT_PROVIDER_OPENVINO) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%context_provider == PT_CONTEXT_PROVIDER_OPENVINO) then
         ! The lane worker has an exact ABI for positional/hybrid/hierarchical
         ! immutable items. COHERENT's center-dependent cross-fold view and
         ! DEBIASED's distinct-slot global view remain CPU-only until separately
         ! verified worker ABIs exist.
         if (config%context_mode == PT_CONTEXT_LEGACY_POSITIONAL .or. &
             config%context_mode == PT_CONTEXT_COHERENT .or. &
             config%context_mode == PT_CONTEXT_DEBIASED .or. &
             len_trim(config%openvino_worker) == 0 .or. len_trim(config%openvino_python) == 0 .or. &
             len_trim(config%openvino_work_dir) == 0 .or. len_trim(config%openvino_device) == 0 .or. &
             config%openvino_min_contexts <= 0_int32 .or. config%openvino_timeout_milliseconds <= 0_int32 .or. &
             (config%openvino_require_accelerator .and. config%openvino_fallback_cpu)) then
            status = PT_ERR_INVALID_ARGUMENT
            return
         end if
      end if
      if (config%initial_plasticity_percent <= 0_int32 .or. &
          config%initial_plasticity_percent > 100_int32 .or. &
          config%minimum_plasticity_percent <= 0_int32 .or. &
          config%minimum_plasticity_percent > config%initial_plasticity_percent .or. &
          config%plasticity_decay_events <= 0_int32) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if

      maximum_shift = int(config%context_window, int64) * &
                      abs(int(config%position_shift_bits, int64))
      if (maximum_shift > int(huge(0_int32), int64)) status = PT_ERR_INVALID_ARGUMENT
      if (config%context_provider == PT_CONTEXT_PROVIDER_OPENVINO .and. &
          (4_int64 * int(config%context_window, int64) + &
           int(HIERARCHICAL_GLOBAL_CONTEXT_WEIGHT, int64) > int(HV_BITACC_MAX_ITEMS, int64))) then
         status = PT_ERR_INVALID_ARGUMENT
      end if
   end subroutine validate_pretrain_config

   subroutine add_nonnegative(counter, delta, status)
      integer(int64), intent(inout) :: counter
      integer(int64), intent(in) :: delta
      integer(int32), intent(out) :: status

      status = PT_OK
      if (delta < 0_int64 .or. counter > huge(0_int64) - delta) then
         status = PT_ERR_COUNTER_OVERFLOW
         return
      end if
      counter = counter + delta
   end subroutine add_nonnegative

   ! Direct-map index for the transient vocabulary cache. The cache key itself
   ! remains the fingerprint plus the full fixed-width token text, so this
   ! inexpensive mix can only create a miss/eviction, never a false symbol hit.
   pure function pretrain_lookup_cache_index(fingerprint) result(cache_index)
      integer(int64), intent(in) :: fingerprint
      integer(int32) :: cache_index
      integer(int64) :: mixed

      mixed = ieor(fingerprint, ishft(fingerprint, -29))
      mixed = ieor(mixed, ishftc(mixed, 17))
      cache_index = int(iand(mixed, PRETRAIN_LOOKUP_CACHE_MASK), int32) + 1_int32
   end function pretrain_lookup_cache_index

   ! Fast-path an active-vocabulary lookup without weakening collision safety.
   ! Every direct-map hit compares the full padded token; every miss calls the
   ! canonical vocabulary hash table with the already-computed fingerprint.
   subroutine pretrain_lookup_active_cached(model, token, cache_tokens, cache_fingerprints, cache_slots, &
                                            cache_used, slot, found, cache_hit, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      character(len=*), intent(in) :: token
      character(len=TOKEN_LEN), intent(inout) :: cache_tokens(:)
      integer(int64), intent(inout) :: cache_fingerprints(:)
      integer(int32), intent(inout) :: cache_slots(:)
      logical, intent(inout) :: cache_used(:)
      integer(int32), intent(out) :: slot
      logical, intent(out) :: found, cache_hit
      integer(int32), intent(out) :: status
      integer(int64) :: fingerprint
      integer(int32) :: cache_index

      slot = 0_int32
      found = .false.
      cache_hit = .false.
      ! This private helper is called only with train_one_epoch's fixed-size
      ! PRETRAIN_LOOKUP_CACHE_SLOTS scratch arrays. Avoid per-token descriptor
      ! size checks on the hot path; allocation is validated once before the
      ! corpus stream begins.
      status = PT_OK
      fingerprint = vocab_token_fingerprint(token)
      cache_index = pretrain_lookup_cache_index(fingerprint)
      if (cache_used(cache_index) .and. cache_fingerprints(cache_index) == fingerprint .and. &
          cache_tokens(cache_index) == token) then
         cache_hit = .true.
         slot = cache_slots(cache_index)
         found = slot > 0_int32
         if (found) then
            status = VOCAB_OK
         else
            status = VOCAB_ERR_NOT_FOUND
         end if
         return
      end if

      call vocab_lookup_fingerprint(model%vocabulary, token, fingerprint, slot, found, status, .true.)
      if (status == VOCAB_OK .or. status == VOCAB_ERR_NOT_FOUND) then
         cache_used(cache_index) = .true.
         cache_fingerprints(cache_index) = fingerprint
         cache_tokens(cache_index) = token
         if (found) then
            cache_slots(cache_index) = slot
         else
            cache_slots(cache_index) = 0_int32
         end if
      end if
   end subroutine pretrain_lookup_active_cached

   subroutine add_metrics(destination, source, status)
      type(pretrain_metrics_t), intent(inout) :: destination
      type(pretrain_metrics_t), intent(in) :: source
      integer(int32), intent(out) :: status

      status = PT_OK
      if (source%corpus_lines < 0_int64 .or. source%raw_tokens < 0_int64 .or. &
          source%active_tokens < 0_int64 .or. source%oov_tokens < 0_int64 .or. &
          source%context_pairs < 0_int64 .or. source%center_updates < 0_int64 .or. &
          source%tokenizer_truncations < 0_int64 .or. source%non_ascii_bytes < 0_int64 .or. &
          source%overlong_records < 0_int64) then
         status = PT_ERR_COUNTER_OVERFLOW
         return
      end if
      if (destination%corpus_lines > huge(0_int64) - source%corpus_lines .or. &
          destination%raw_tokens > huge(0_int64) - source%raw_tokens .or. &
          destination%active_tokens > huge(0_int64) - source%active_tokens .or. &
          destination%oov_tokens > huge(0_int64) - source%oov_tokens .or. &
          destination%context_pairs > huge(0_int64) - source%context_pairs .or. &
          destination%center_updates > huge(0_int64) - source%center_updates .or. &
          destination%tokenizer_truncations > huge(0_int64) - source%tokenizer_truncations .or. &
          destination%non_ascii_bytes > huge(0_int64) - source%non_ascii_bytes .or. &
          destination%overlong_records > huge(0_int64) - source%overlong_records) then
         status = PT_ERR_COUNTER_OVERFLOW
         return
      end if
      destination%corpus_lines = destination%corpus_lines + source%corpus_lines
      destination%raw_tokens = destination%raw_tokens + source%raw_tokens
      destination%active_tokens = destination%active_tokens + source%active_tokens
      destination%oov_tokens = destination%oov_tokens + source%oov_tokens
      destination%context_pairs = destination%context_pairs + source%context_pairs
      destination%center_updates = destination%center_updates + source%center_updates
      destination%tokenizer_truncations = destination%tokenizer_truncations + source%tokenizer_truncations
      destination%non_ascii_bytes = destination%non_ascii_bytes + source%non_ascii_bytes
      destination%overlong_records = destination%overlong_records + source%overlong_records
   end subroutine add_metrics

   ! Reads one physical formatted record without silently accepting a record
   ! longer than MAX_LINE_CHARS.  A too-long record is fully drained and exposed
   ! to the caller as telemetry, so a non-strict run can drop it deterministically
   ! rather than learn from a prefix.  EOF is distinct from malformed I/O.
   subroutine read_corpus_record(unit, line, eof, overlong, status)
      integer, intent(in) :: unit
      character(len=*), intent(out) :: line
      logical, intent(out) :: eof, overlong
      integer(int32), intent(out) :: status
      character(len=1) :: probe
      character(len=MAX_LINE_CHARS) :: discard_buffer
      integer :: io_status, transferred

      line = ''
      eof = .false.
      overlong = .false.
      status = PT_OK
      transferred = 0
      read(unit, '(A)', advance='no', size=transferred, iostat=io_status) line
      if (io_status == iostat_end) then
         eof = .true.
         return
      end if
      if (io_status /= 0 .and. io_status /= iostat_eor) then
         status = PT_ERR_IO
         return
      end if
      if (io_status == iostat_eor) return

      ! The first transfer filled the fixed record buffer.  Probe once to
      ! distinguish an exactly-sized record from an overlong record.
      probe = ''
      transferred = 0
      read(unit, '(A)', advance='no', size=transferred, iostat=io_status) probe
      if (transferred > 0) overlong = .true.
      if (io_status /= 0 .and. io_status /= iostat_eor .and. io_status /= iostat_end) then
         status = PT_ERR_IO
         return
      end if
      if (.not. overlong) then
         if (io_status == iostat_eor .or. io_status == iostat_end) return
         status = PT_ERR_IO
         return
      end if

      ! If the probe did not finish this record, consume its remaining bytes.
      do while (io_status == 0)
         discard_buffer = ''
         transferred = 0
         read(unit, '(A)', advance='no', size=transferred, iostat=io_status) discard_buffer
         if (io_status /= 0 .and. io_status /= iostat_eor .and. io_status /= iostat_end) then
            status = PT_ERR_IO
            return
         end if
      end do
   end subroutine read_corpus_record

   subroutine allocate_model_storage(model, status)
      type(hdc_pretrain_model_t), intent(inout) :: model
      integer(int32), intent(out) :: status
      integer(int32) :: allocation_status

      status = PT_OK
      if (.not. vocab_is_initialized(model%vocabulary)) then
         status = PT_ERR_VOCABULARY
         return
      end if
      allocate(model%base_bits(HV_WORDS, model%vocabulary%capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         status = PT_ERR_ALLOCATION
         return
      end if
      allocate(model%semantic_bits(HV_WORDS, model%vocabulary%capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         deallocate(model%base_bits)
         status = PT_ERR_ALLOCATION
         return
      end if
      allocate(model%experience_count(model%vocabulary%capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         deallocate(model%base_bits)
         deallocate(model%semantic_bits)
         status = PT_ERR_ALLOCATION
         return
      end if
      model%base_bits = 0_int64
      model%semantic_bits = 0_int64
      model%experience_count = 0_int64
   end subroutine allocate_model_storage

   subroutine initialize_active_hypervectors(model)
      type(hdc_pretrain_model_t), intent(inout) :: model
      type(hv_t) :: vector
      integer(int32) :: slot

      do slot = 1, model%vocabulary%count
         if (.not. model%vocabulary%active(slot)) cycle
         vector = hv_from_seed(vocab_seed_for_slot(model%vocabulary, slot))
         model%base_bits(:, slot) = vector%bits
         model%semantic_bits(:, slot) = vector%bits
      end do
   end subroutine initialize_active_hypervectors

   ! First corpus pass: observe all normalized tokens, finalize min-frequency
   ! activity, then allocate deterministic token identities and semantic traces.
   subroutine pretrain_build_vocabulary(model, corpus_path, config, metrics, status)
      type(hdc_pretrain_model_t), intent(inout) :: model
      character(len=*), intent(in) :: corpus_path
      type(hdc_pretrain_config_t), intent(in) :: config
      type(pretrain_metrics_t), intent(out) :: metrics
      integer(int32), intent(out) :: status
      character(len=MAX_LINE_CHARS) :: line
      character(len=TOKEN_LEN) :: tokens(MAX_TOKENS_PER_LINE)
      integer :: unit, io_status
      integer(int32) :: token_index, n_tokens, vocab_status
      integer(int64) :: non_ascii_count
      logical :: eof, overlong, truncated

      call pretrain_model_destroy(model)
      call pretrain_metrics_reset(metrics)
      call validate_pretrain_config(config, status)
      if (status /= PT_OK) return
      if (len_trim(corpus_path) == 0) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if

      call vocab_init(model%vocabulary, config%max_vocabulary, config%min_frequency, vocab_status)
      if (vocab_status /= VOCAB_OK) then
         status = PT_ERR_VOCABULARY
         call pretrain_model_destroy(model)
         return
      end if

      open(newunit=unit, file=trim(corpus_path), status='old', action='read', &
           form='formatted', iostat=io_status)
      if (io_status /= 0) then
         status = PT_ERR_IO
         call pretrain_model_destroy(model)
         return
      end if

      do
         call read_corpus_record(unit, line, eof, overlong, status)
         if (status /= PT_OK) then
            close(unit)
            call pretrain_model_destroy(model)
            return
         end if
         if (eof) exit
         call add_nonnegative(metrics%corpus_lines, 1_int64, status)
         if (status /= PT_OK) exit
         if (overlong) then
            call add_nonnegative(metrics%overlong_records, 1_int64, status)
            if (status /= PT_OK) exit
            if (config%strict_tokenization) then
               status = PT_ERR_RECORD_TOO_LONG
               exit
            end if
            cycle
         end if
         call tokenize_ascii_line(line, tokens, n_tokens, truncated, non_ascii_count)
         call add_nonnegative(metrics%non_ascii_bytes, non_ascii_count, status)
         if (status /= PT_OK) exit
         if (config%strict_tokenization .and. non_ascii_count > 0_int64) then
            status = PT_ERR_TOKENIZATION
            exit
         end if
         if (truncated) then
            call add_nonnegative(metrics%tokenizer_truncations, 1_int64, status)
            if (status /= PT_OK) exit
            if (config%strict_tokenization) then
               status = PT_ERR_TOKENIZATION
               exit
            end if
         end if
         do token_index = 1, n_tokens
            call add_nonnegative(metrics%raw_tokens, 1_int64, status)
            if (status /= PT_OK) exit
            call vocab_observe(model%vocabulary, tokens(token_index), vocab_status)
            if (vocab_status /= VOCAB_OK) then
               status = PT_ERR_VOCABULARY
               exit
            end if
         end do
         if (status /= PT_OK) exit
      end do
      close(unit)
      if (status /= PT_OK) then
         call pretrain_model_destroy(model)
         return
      end if

      call vocab_finalize(model%vocabulary, vocab_status)
      if (vocab_status /= VOCAB_OK) then
         status = PT_ERR_VOCABULARY
         call pretrain_model_destroy(model)
         return
      end if
      if (model%vocabulary%active_count == 0_int32) then
         status = PT_ERR_EMPTY_VOCABULARY
         call pretrain_model_destroy(model)
         return
      end if

      call allocate_model_storage(model, status)
      if (status /= PT_OK) then
         call pretrain_model_destroy(model)
         return
      end if
      call initialize_active_hypervectors(model)
      model%context_window = config%context_window
      model%position_shift_bits = config%position_shift_bits
      model%context_mode = config%context_mode
      model%initial_plasticity_percent = config%initial_plasticity_percent
      model%minimum_plasticity_percent = config%minimum_plasticity_percent
      model%plasticity_decay_events = config%plasticity_decay_events
      model%plasticity_sampler = PT_PLASTICITY_SAMPLER_FLOYD_COMPLEMENT
      model%tie_seed = config%tie_seed
      model%ready = .true.
   end subroutine pretrain_build_vocabulary

   ! Activates only vocabulary slots that crossed min_frequency during a
   ! continuous shard-extension pass.  Existing semantic prototypes and their
   ! experience schedules are deliberately left untouched.
   subroutine initialize_new_active_hypervectors(model, active_before_extension)
      type(hdc_pretrain_model_t), intent(inout) :: model
      logical, intent(in) :: active_before_extension(:)
      type(hv_t) :: vector
      integer(int32) :: slot

      if (size(active_before_extension, kind=int32) /= model%vocabulary%capacity) return
      do slot = 1, model%vocabulary%count
         if (.not. model%vocabulary%active(slot) .or. active_before_extension(slot)) cycle
         vector = hv_from_seed(vocab_seed_for_slot(model%vocabulary, slot))
         model%base_bits(:, slot) = vector%bits
         model%semantic_bits(:, slot) = vector%bits
         model%experience_count(slot) = 0_int64
      end do
   end subroutine initialize_new_active_hypervectors

   ! Observes one new immutable corpus shard without resetting an already
   ! trained model.  It may append vocabulary slots (up to the fixed capacity)
   ! and deterministically initializes only newly-active identities.  The
   ! caller subsequently invokes pretrain_train_file on the same shard.
   !
   ! The API intentionally exposes extension and learning as distinct phases:
   ! a transactional continuous controller can keep the durable checkpoint
   ! unchanged until both have completed and been integrity-checked.
   subroutine pretrain_extend_vocabulary(model, corpus_path, config, metrics, status)
      type(hdc_pretrain_model_t), intent(inout) :: model
      character(len=*), intent(in) :: corpus_path
      type(hdc_pretrain_config_t), intent(in) :: config
      type(pretrain_metrics_t), intent(out) :: metrics
      integer(int32), intent(out) :: status
      character(len=MAX_LINE_CHARS) :: line
      character(len=TOKEN_LEN) :: tokens(MAX_TOKENS_PER_LINE)
      logical, allocatable :: active_before_extension(:)
      integer :: unit, io_status, allocation_status
      integer(int32) :: token_index, n_tokens, vocab_status
      integer(int64) :: non_ascii_count
      logical :: eof, overlong, truncated

      call pretrain_metrics_reset(metrics)
      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if
      call validate_pretrain_config(config, status)
      if (status /= PT_OK) return
      if (len_trim(corpus_path) == 0) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%max_vocabulary /= model%vocabulary%capacity .or. &
          config%min_frequency /= model%vocabulary%min_frequency .or. &
          config%context_window /= model%context_window .or. &
          config%position_shift_bits /= model%position_shift_bits .or. &
          config%context_mode /= model%context_mode .or. &
          config%initial_plasticity_percent /= model%initial_plasticity_percent .or. &
          config%minimum_plasticity_percent /= model%minimum_plasticity_percent .or. &
          config%plasticity_decay_events /= model%plasticity_decay_events .or. &
          config%tie_seed /= model%tie_seed) then
         status = PT_ERR_INCOMPATIBLE_CONFIG
         return
      end if

      allocate(active_before_extension(model%vocabulary%capacity), stat=allocation_status)
      if (allocation_status /= 0) then
         status = PT_ERR_ALLOCATION
         return
      end if
      active_before_extension = model%vocabulary%active

      open(newunit=unit, file=trim(corpus_path), status='old', action='read', &
           form='formatted', iostat=io_status)
      if (io_status /= 0) then
         deallocate(active_before_extension)
         status = PT_ERR_IO
         return
      end if

      do
         call read_corpus_record(unit, line, eof, overlong, status)
         if (status /= PT_OK) exit
         if (eof) exit
         call add_nonnegative(metrics%corpus_lines, 1_int64, status)
         if (status /= PT_OK) exit
         if (overlong) then
            call add_nonnegative(metrics%overlong_records, 1_int64, status)
            if (status /= PT_OK) exit
            if (config%strict_tokenization) then
               status = PT_ERR_RECORD_TOO_LONG
               exit
            end if
            cycle
         end if
         call tokenize_ascii_line(line, tokens, n_tokens, truncated, non_ascii_count)
         call add_nonnegative(metrics%non_ascii_bytes, non_ascii_count, status)
         if (status /= PT_OK) exit
         if (config%strict_tokenization .and. non_ascii_count > 0_int64) then
            status = PT_ERR_TOKENIZATION
            exit
         end if
         if (truncated) then
            call add_nonnegative(metrics%tokenizer_truncations, 1_int64, status)
            if (status /= PT_OK) exit
            if (config%strict_tokenization) then
               status = PT_ERR_TOKENIZATION
               exit
            end if
         end if
         do token_index = 1, n_tokens
            call add_nonnegative(metrics%raw_tokens, 1_int64, status)
            if (status /= PT_OK) exit
            call vocab_observe(model%vocabulary, tokens(token_index), vocab_status)
            if (vocab_status /= VOCAB_OK) then
               status = PT_ERR_VOCABULARY
               exit
            end if
         end do
         if (status /= PT_OK) exit
      end do
      close(unit)
      if (status /= PT_OK) then
         deallocate(active_before_extension)
         return
      end if

      call vocab_finalize(model%vocabulary, vocab_status)
      if (vocab_status /= VOCAB_OK) then
         deallocate(active_before_extension)
         status = PT_ERR_VOCABULARY
         return
      end if
      if (model%vocabulary%active_count == 0_int32) then
         deallocate(active_before_extension)
         status = PT_ERR_EMPTY_VOCABULARY
         return
      end if
      call initialize_new_active_hypervectors(model, active_before_extension)
      deallocate(active_before_extension)
   end subroutine pretrain_extend_vocabulary

   pure function scheduled_plasticity_percent(model, experiences) result(percent)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int64), intent(in) :: experiences
      integer(int32) :: percent
      integer(int64) :: decay

      decay = int(model%plasticity_decay_events, int64)
      if (experiences < decay) then
         percent = model%initial_plasticity_percent
      else if (experiences < 4_int64 * decay) then
         percent = max(model%minimum_plasticity_percent, &
                       (model%initial_plasticity_percent + 1_int32) / 2_int32)
      else if (experiences < 16_int64 * decay) then
         percent = max(model%minimum_plasticity_percent, &
                       (model%initial_plasticity_percent + 3_int32) / 4_int32)
      else
         percent = model%minimum_plasticity_percent
      end if
   end function scheduled_plasticity_percent

   ! A hierarchical record summary is built once per input line, then shared
   ! by all eligible centers on that line.  It is a bounded topical signal, not
   ! an unbounded document memory: only records with at least twelve active
   ! tokens activate it, and the local update later caps its multiplicity by
   ! the bit-sliced accumulator capacity.
   subroutine pretrain_build_line_topic_context(model, token_slots, n_tokens, active_count, line_context, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int32), intent(in) :: token_slots(:)
      integer(int32), intent(in) :: n_tokens, active_count
      type(hv_t), intent(out) :: line_context
      integer(int32), intent(out) :: status
      type(hv_bitacc_t) :: line_accumulator
      integer(int32) :: token_index, slot, add_status

      line_context = hv_zero()
      status = PT_OK
      ! The caller already counts active vocabulary hits while tokenizing.  Do
      ! not construct/reset a 16,384-bit topic accumulator for short records:
      ! those records cannot consume the hierarchical channel at all.
      if (active_count < HIERARCHICAL_MIN_ACTIVE_TOKENS) return
      if (active_count > n_tokens) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      call hv_bitacc_reset(line_accumulator, active_count)
      do token_index = 1, n_tokens
         slot = token_slots(token_index)
         if (slot == 0_int32) cycle
         call hv_bitacc_add_words(line_accumulator, model%base_bits(:, slot), add_status)
         if (add_status /= HV_OK) then
            status = PT_ERR_HYPERVECTOR
            return
         end if
      end do
      line_context = hv_bitacc_finalize(line_accumulator, model%tie_seed)
   end subroutine pretrain_build_line_topic_context

   ! DEBIASED builds the global long-record signal from unique active slots.
   ! Local positional/topic neighbours remain occurrence-sensitive exactly as
   ! HYBRID does; this routine changes only the bounded whole-record channel.
   ! Reusing the capacity-indexed planner scratch avoids an O(record_length^2)
   ! duplicate scan and clears only slots touched by this record.
   subroutine pretrain_build_debiased_line_topic_context(model, token_slots, n_tokens, active_count, &
                                                         line_context, distinct_active_count, seen_counts, &
                                                         touched_slots, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int32), intent(in) :: token_slots(:)
      integer(int32), intent(in) :: n_tokens, active_count
      type(hv_t), intent(out) :: line_context
      integer(int32), intent(out) :: distinct_active_count
      integer(int32), intent(inout) :: seen_counts(:)
      integer(int32), intent(out) :: touched_slots(:)
      integer(int32), intent(out) :: status
      type(hv_bitacc_t) :: line_accumulator
      integer(int32) :: token_index, slot, touched_count, observed_active_count, add_status

      line_context = hv_zero()
      distinct_active_count = 0_int32
      status = PT_OK
      if (active_count < 0_int32 .or. active_count > n_tokens .or. n_tokens < 0_int32 .or. &
          n_tokens > size(token_slots, kind=int32) .or. size(seen_counts, kind=int32) < model%vocabulary%capacity .or. &
          size(touched_slots, kind=int32) < n_tokens) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      ! A short record can never activate the global channel. Leaving the
      ! scratch untouched is safe because the planner cleared it after the
      ! previous record and no current record has written it yet.
      if (active_count < DEBIASED_MIN_DISTINCT_ACTIVE_TOKENS) return

      touched_count = 0_int32
      observed_active_count = 0_int32
      do token_index = 1, n_tokens
         slot = token_slots(token_index)
         if (slot == 0_int32) cycle
         if (slot < 1_int32 .or. slot > model%vocabulary%capacity) then
            status = PT_ERR_INVALID_ARGUMENT
            goto 700
         end if
         observed_active_count = observed_active_count + 1_int32
         if (seen_counts(slot) == 0_int32) then
            touched_count = touched_count + 1_int32
            touched_slots(touched_count) = slot
            seen_counts(slot) = 1_int32
         else if (seen_counts(slot) < huge(0_int32)) then
            ! The value itself is not semantically used; retaining a bounded
            ! occurrence count makes accidental stale scratch state detectable.
            seen_counts(slot) = seen_counts(slot) + 1_int32
         else
            status = PT_ERR_COUNTER_OVERFLOW
            goto 700
         end if
      end do
      if (observed_active_count /= active_count) then
         status = PT_ERR_INVALID_ARGUMENT
         goto 700
      end if
      distinct_active_count = touched_count
      if (distinct_active_count < DEBIASED_MIN_DISTINCT_ACTIVE_TOKENS) goto 700

      call hv_bitacc_reset(line_accumulator, distinct_active_count)
      do token_index = 1, touched_count
         slot = touched_slots(token_index)
         call hv_bitacc_add_words(line_accumulator, model%base_bits(:, slot), add_status)
         if (add_status /= HV_OK) then
            status = PT_ERR_HYPERVECTOR
            goto 700
         end if
      end do
      line_context = hv_bitacc_finalize(line_accumulator, model%tie_seed)

700   continue
      do token_index = 1, touched_count
         seen_counts(touched_slots(token_index)) = 0_int32
      end do
   end subroutine pretrain_build_debiased_line_topic_context

   ! COHERENT uses three immutable record summaries while preserving the
   ! HIERARCHICAL channel's fixed three-item context budget:
   !   1 x full-record topic + 2 x opposite parity-fold topic.
   ! The parity-fold topic cannot contain the focal token occurrence, so it
   ! contributes cross-position record evidence without repeatedly echoing the center's
   ! own base identity.  A repeated token at another source position remains a
   ! valid contextual occurrence.  This is a deterministic structural context
   ! view, not a learned attention mechanism or an unbounded document memory.
   subroutine pretrain_build_coherent_line_topic_contexts(model, token_slots, n_tokens, active_count, &
                                                          line_context, fold_contexts, fold_active_counts, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int32), intent(in) :: token_slots(:)
      integer(int32), intent(in) :: n_tokens, active_count
      type(hv_t), intent(out) :: line_context, fold_contexts(2)
      integer(int32), intent(out) :: fold_active_counts(2)
      integer(int32), intent(out) :: status
      type(hv_bitacc_t) :: line_accumulator, fold_accumulators(2)
      integer(int32) :: token_index, slot, fold_index, add_status, observed_active_count

      line_context = hv_zero()
      fold_contexts(1) = hv_zero()
      fold_contexts(2) = hv_zero()
      fold_active_counts = 0_int32
      status = PT_OK
      if (active_count < HIERARCHICAL_MIN_ACTIVE_TOKENS) return
      if (active_count > n_tokens .or. n_tokens > size(token_slots, kind=int32)) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if

      do token_index = 1, n_tokens
         if (token_slots(token_index) == 0_int32) cycle
         fold_index = modulo(token_index - 1_int32, 2_int32) + 1_int32
         fold_active_counts(fold_index) = fold_active_counts(fold_index) + 1_int32
      end do
      observed_active_count = fold_active_counts(1) + fold_active_counts(2)
      if (observed_active_count /= active_count) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if

      call hv_bitacc_reset(line_accumulator, active_count)
      do fold_index = 1, 2
         if (fold_active_counts(fold_index) > 0_int32) then
            call hv_bitacc_reset(fold_accumulators(fold_index), fold_active_counts(fold_index))
         end if
      end do
      do token_index = 1, n_tokens
         slot = token_slots(token_index)
         if (slot == 0_int32) cycle
         call hv_bitacc_add_words(line_accumulator, model%base_bits(:, slot), add_status)
         if (add_status /= HV_OK) then
            status = PT_ERR_HYPERVECTOR
            return
         end if
         fold_index = modulo(token_index - 1_int32, 2_int32) + 1_int32
         call hv_bitacc_add_words(fold_accumulators(fold_index), model%base_bits(:, slot), add_status)
         if (add_status /= HV_OK) then
            status = PT_ERR_HYPERVECTOR
            return
         end if
      end do
      line_context = hv_bitacc_finalize(line_accumulator, model%tie_seed)
      do fold_index = 1, 2
         if (fold_active_counts(fold_index) > 0_int32) then
            fold_contexts(fold_index) = hv_bitacc_finalize(fold_accumulators(fold_index), model%tie_seed)
         end if
      end do
   end subroutine pretrain_build_coherent_line_topic_contexts

   ! Determine which centers will update and the exact experience counter that
   ! each update must observe.  Context vectors depend only on immutable base
   ! identities plus this counter-derived tie seed; prototype mutation happens
   ! later in source order.  Preplanning lets context construction run safely in
   ! parallel without changing repeated-token semantics.
   subroutine pretrain_plan_line_updates(model, config, token_slots, n_tokens, update_required, &
                                         neighbor_counts, experiences_before, line_update_counts, touched_slots, &
                                         update_count, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      type(hdc_pretrain_config_t), intent(in) :: config
      integer(int32), intent(in) :: token_slots(:)
      integer(int32), intent(in) :: n_tokens
      logical, intent(out) :: update_required(:)
      integer(int32), intent(out) :: neighbor_counts(:)
      integer(int64), intent(out) :: experiences_before(:)
      ! Reusable capacity-indexed scratch avoids the old O(record_length^2)
      ! backward scan for repeated centers.  Only slots touched by this record
      ! are reset, so the extra bounded storage is not cleared per line.
      integer(int32), intent(inout) :: line_update_counts(:)
      integer(int32), intent(out) :: touched_slots(:)
      integer(int32), intent(out) :: update_count
      integer(int32), intent(out) :: status
      integer(int32) :: center_index, center_slot, touched_count, left_index, right_index
      integer(int32) :: neighbours, active_prefix(0:MAX_TOKENS_PER_LINE)
      integer(int64) :: expected_experience, prior_updates

      status = PT_OK
      update_count = 0_int32
      update_required = .false.
      neighbor_counts = 0_int32
      experiences_before = 0_int64
      if (n_tokens < 0_int32 .or. n_tokens > size(token_slots, kind=int32) .or. &
          n_tokens > size(update_required, kind=int32) .or. n_tokens > size(neighbor_counts, kind=int32) .or. &
          n_tokens > size(experiences_before, kind=int32) .or. &
          size(line_update_counts, kind=int32) < model%vocabulary%capacity .or. &
          size(touched_slots, kind=int32) < n_tokens) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if

      ! Prefix active counts make all bounded-window neighbor cardinalities
      ! O(record_length), including the largest legal window, rather than an
      ! O(record_length * window) nested scan.
      active_prefix(0) = 0_int32
      do center_index = 1, n_tokens
         active_prefix(center_index) = active_prefix(center_index - 1_int32)
         if (token_slots(center_index) /= 0_int32) then
            active_prefix(center_index) = active_prefix(center_index) + 1_int32
         end if
      end do
      do center_index = 1, n_tokens
         center_slot = token_slots(center_index)
         if (center_slot == 0_int32) cycle
         left_index = max(1_int32, center_index - config%context_window)
         right_index = min(n_tokens, center_index + config%context_window)
         ! The active center is included in this closed interval and is removed
         ! explicitly, yielding exactly the prior offset-loop cardinality.
         neighbours = active_prefix(right_index) - active_prefix(left_index - 1_int32) - 1_int32
         if (neighbours > 0_int32) then
            update_required(center_index) = .true.
            neighbor_counts(center_index) = neighbours
            update_count = update_count + 1_int32
         end if
      end do

      touched_count = 0_int32
      do center_index = 1, n_tokens
         if (.not. update_required(center_index)) cycle
         center_slot = token_slots(center_index)
         if (center_slot < 1_int32 .or. center_slot > model%vocabulary%capacity) then
            status = PT_ERR_INVALID_ARGUMENT
            return
         end if
         if (line_update_counts(center_slot) == 0_int32) then
            touched_count = touched_count + 1_int32
            touched_slots(touched_count) = center_slot
         end if
         prior_updates = int(line_update_counts(center_slot), int64)
         if (model%experience_count(center_slot) > huge(0_int64) - prior_updates) then
            status = PT_ERR_COUNTER_OVERFLOW
            return
         end if
         expected_experience = model%experience_count(center_slot) + prior_updates
         if (expected_experience == huge(0_int64)) then
            status = PT_ERR_COUNTER_OVERFLOW
            return
         end if
         experiences_before(center_index) = expected_experience
         line_update_counts(center_slot) = line_update_counts(center_slot) + 1_int32
      end do

      do center_index = 1, touched_count
         line_update_counts(touched_slots(center_index)) = 0_int32
      end do
   end subroutine pretrain_plan_line_updates

   ! Build one center's immutable HDC context only.  It has no writes to model
   ! state and is therefore OpenMP-safe.  Its output is committed separately in
   ! source order by pretrain_apply_center_update.
   subroutine pretrain_construct_center_context(model, config, token_slots, n_tokens, center_index, &
                                                expected_neighbor_count, line_topic_context, line_topic_fold_contexts, &
                                                line_active_tokens, line_topic_fold_active_counts, experience_before, &
                                                context_bits, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      type(hdc_pretrain_config_t), intent(in) :: config
      integer(int32), intent(in) :: token_slots(:)
      integer(int32), intent(in) :: n_tokens, center_index, expected_neighbor_count
      type(hv_t), intent(in) :: line_topic_context, line_topic_fold_contexts(2)
      integer(int32), intent(in) :: line_active_tokens, line_topic_fold_active_counts(2)
      integer(int64), intent(in) :: experience_before
      integer(int64), intent(out) :: context_bits(HV_WORDS)
      integer(int32), intent(out) :: status
      type(hv_acc_t) :: legacy_accumulator
      type(hv_bitacc_t) :: bit_accumulator
      type(hv_t) :: neighbor, positioned, context
      integer(int32) :: center_slot, offset, neighbor_index, neighbor_slot
      integer(int32) :: add_status, shift_bits, neighbors, global_items, global_index, maximum_items
      integer(int32) :: center_fold, opposite_fold
      integer(int64) :: tie_seed
      logical :: reference_kernel, topic_mode, use_global_topic, use_coherent_cross_topic

      status = PT_OK
      context_bits = 0_int64
      if (center_index < 1_int32 .or. center_index > n_tokens .or. &
          expected_neighbor_count <= 0_int32) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      center_slot = token_slots(center_index)
      if (center_slot == 0_int32) return

      ! LEGACY is the signed-counter oracle.  The reference switch retains the
      ! previous bit-sliced rotate-then-add implementation for fair before/after
      ! timing, while the default fuses the same arithmetic directly over the
      ! vocabulary's word-major SoA column.
      reference_kernel = config%reference_context_kernel
      topic_mode = config%context_mode == PT_CONTEXT_HYBRID .or. &
                   config%context_mode == PT_CONTEXT_HIERARCHICAL .or. &
                   config%context_mode == PT_CONTEXT_COHERENT .or. &
                   config%context_mode == PT_CONTEXT_DEBIASED
      use_global_topic = .false.
      use_coherent_cross_topic = .false.
      center_fold = modulo(center_index - 1_int32, 2_int32) + 1_int32
      opposite_fold = 3_int32 - center_fold
      select case (config%context_mode)
      case (PT_CONTEXT_HIERARCHICAL)
         use_global_topic = line_active_tokens >= HIERARCHICAL_MIN_ACTIVE_TOKENS
      case (PT_CONTEXT_COHERENT)
         if (line_active_tokens >= HIERARCHICAL_MIN_ACTIVE_TOKENS) then
            if (line_topic_fold_active_counts(1) < 0_int32 .or. &
                line_topic_fold_active_counts(2) < 0_int32 .or. &
                line_topic_fold_active_counts(1) + line_topic_fold_active_counts(2) /= line_active_tokens) then
               status = PT_ERR_INVALID_ARGUMENT
               return
            end if
            use_global_topic = .true.
            use_coherent_cross_topic = line_topic_fold_active_counts(opposite_fold) > 0_int32
         end if
      case (PT_CONTEXT_DEBIASED)
         use_global_topic = line_active_tokens >= DEBIASED_MIN_DISTINCT_ACTIVE_TOKENS
      end select
      if (config%context_mode == PT_CONTEXT_LEGACY_POSITIONAL) then
         call hv_acc_reset(legacy_accumulator)
      else
         maximum_items = expected_neighbor_count
         if (topic_mode) maximum_items = 2_int32 * maximum_items
         if (use_global_topic) then
            maximum_items = maximum_items + min(HIERARCHICAL_GLOBAL_CONTEXT_WEIGHT, &
                                                HV_BITACC_MAX_ITEMS - maximum_items)
         end if
         if (maximum_items <= 0_int32 .or. maximum_items > HV_BITACC_MAX_ITEMS) then
            status = PT_ERR_HYPERVECTOR
            return
         end if
         call hv_bitacc_reset(bit_accumulator, maximum_items)
      end if
      neighbors = 0_int32
      do offset = -config%context_window, config%context_window
         if (offset == 0_int32) cycle
         neighbor_index = center_index + offset
         if (neighbor_index < 1_int32 .or. neighbor_index > n_tokens) cycle
         neighbor_slot = token_slots(neighbor_index)
         if (neighbor_slot == 0_int32) cycle

         shift_bits = int(int(offset, int64) * int(config%position_shift_bits, int64), int32)
         if (config%context_mode == PT_CONTEXT_LEGACY_POSITIONAL .or. reference_kernel) then
            neighbor%bits = model%base_bits(:, neighbor_slot)
            positioned = hv_permute(neighbor, shift_bits)
            if (config%context_mode == PT_CONTEXT_LEGACY_POSITIONAL) then
               call hv_acc_add(legacy_accumulator, positioned, add_status)
            else
               call hv_bitacc_add(bit_accumulator, positioned, add_status)
            end if
            if (add_status /= HV_OK) then
               status = PT_ERR_HYPERVECTOR
               return
            end if
            if (topic_mode) then
               call hv_bitacc_add(bit_accumulator, neighbor, add_status)
               if (add_status /= HV_OK) then
                  status = PT_ERR_HYPERVECTOR
                  return
               end if
            end if
         else if (config%context_mode == PT_CONTEXT_POSITIONAL) then
            call hv_bitacc_add_permuted_words(bit_accumulator, model%base_bits(:, neighbor_slot), &
                                               shift_bits, add_status)
            if (add_status /= HV_OK) then
               status = PT_ERR_HYPERVECTOR
               return
            end if
         else
            ! HYBRID, HIERARCHICAL, and COHERENT share the exact local
            ! positional+topic representation.  The pair kernel is algebraically identical to
            ! two adds but eliminates both a 2 KiB rotate temporary and a
            ! second carry pass.
            call hv_bitacc_add_permuted_pair_words(bit_accumulator, model%base_bits(:, neighbor_slot), &
                                                    shift_bits, add_status)
            if (add_status /= HV_OK) then
               status = PT_ERR_HYPERVECTOR
               return
            end if
         end if
         neighbors = neighbors + 1_int32
      end do
      if (neighbors == 0_int32 .or. neighbors /= expected_neighbor_count) then
         status = PT_ERR_HYPERVECTOR
         return
      end if

      if (use_global_topic) then
         global_items = min(HIERARCHICAL_GLOBAL_CONTEXT_WEIGHT, &
                            HV_BITACC_MAX_ITEMS - bit_accumulator%sample_count)
         if (config%context_mode == PT_CONTEXT_COHERENT .and. use_coherent_cross_topic) then
            ! Keep one all-record anchor for coverage, then favor the opposite
            ! parity fold for the remaining global weight.  The latter excludes
            ! this token occurrence and therefore reduces self-echo.
            if (global_items > 0_int32) then
               call hv_bitacc_add(bit_accumulator, line_topic_context, add_status)
               if (add_status /= HV_OK) then
                  status = PT_ERR_HYPERVECTOR
                  return
               end if
            end if
            do global_index = 2, global_items
               call hv_bitacc_add(bit_accumulator, line_topic_fold_contexts(opposite_fold), add_status)
               if (add_status /= HV_OK) then
                  status = PT_ERR_HYPERVECTOR
                  return
               end if
            end do
         else
            do global_index = 1, global_items
               call hv_bitacc_add(bit_accumulator, line_topic_context, add_status)
               if (add_status /= HV_OK) then
                  status = PT_ERR_HYPERVECTOR
                  return
               end if
            end do
         end if
      end if

      tie_seed = ieor(vocab_seed_for_slot(model%vocabulary, center_slot), ishftc(experience_before, 19))
      if (config%context_mode == PT_CONTEXT_LEGACY_POSITIONAL) then
         context = hv_acc_finalize(legacy_accumulator, ieor(config%tie_seed, tie_seed))
      else
         context = hv_bitacc_finalize(bit_accumulator, ieor(config%tie_seed, tie_seed))
      end if
      context_bits = context%bits
   end subroutine pretrain_construct_center_context


   ! Number of immutable bundle items sent to the OpenVINO lane graph for one
   ! center.  The worker groups equal cardinalities because seeded majority
   ! thresholds cannot be safely zero-padded.
   pure function openvino_context_item_count(config, neighbor_count, line_active_tokens) result(item_count)
      type(hdc_pretrain_config_t), intent(in) :: config
      integer(int32), intent(in) :: neighbor_count, line_active_tokens
      integer(int32) :: item_count, global_items

      item_count = -1_int32
      select case (config%context_mode)
      case (PT_CONTEXT_POSITIONAL)
         item_count = neighbor_count
      case (PT_CONTEXT_HYBRID)
         item_count = 2_int32 * neighbor_count
      case (PT_CONTEXT_HIERARCHICAL)
         item_count = 2_int32 * neighbor_count
         if (line_active_tokens >= HIERARCHICAL_MIN_ACTIVE_TOKENS) then
            global_items = min(HIERARCHICAL_GLOBAL_CONTEXT_WEIGHT, HV_BITACC_MAX_ITEMS - item_count)
            item_count = item_count + global_items
         end if
      end select
   end function openvino_context_item_count

   ! Explicitly split an int64 bit pattern into low/high int32 lanes.  The
   ! high bit of either lane is created/read with bit intrinsics, never with an
   ! out-of-range numeric conversion.  Stream files still reject big-endian
   ! hosts at dispatch time because the Python protocol is little-endian.
   pure subroutine pack_hv_u32_lanes(vector, lanes)
      type(hv_t), intent(in) :: vector
      integer(int32), intent(out) :: lanes(:)
      integer(int32) :: word, low_lane, high_lane

      lanes = 0_int32
      if (size(lanes, kind=int32) /= 2_int32 * HV_WORDS) return
      do word = 1, HV_WORDS
         low_lane = int(ibits(vector%bits(word), 0, 31), int32)
         if (btest(vector%bits(word), 31)) low_lane = ibset(low_lane, 31)
         high_lane = int(ibits(vector%bits(word), 32, 31), int32)
         if (btest(vector%bits(word), 63)) high_lane = ibset(high_lane, 31)
         lanes(2_int32 * word - 1_int32) = low_lane
         lanes(2_int32 * word) = high_lane
      end do
   end subroutine pack_hv_u32_lanes

   pure subroutine unpack_hv_u32_lanes(lanes, bits)
      integer(int32), intent(in) :: lanes(:)
      integer(int64), intent(out) :: bits(HV_WORDS)
      integer(int32) :: word
      integer(int64) :: low_lane, high_lane

      bits = 0_int64
      if (size(lanes, kind=int32) /= 2_int32 * HV_WORDS) return
      do word = 1, HV_WORDS
         low_lane = iand(int(lanes(2_int32 * word - 1_int32), int64), OPENVINO_LOW32_MASK)
         high_lane = iand(int(lanes(2_int32 * word), int64), OPENVINO_LOW32_MASK)
         bits(word) = ior(low_lane, ishft(high_lane, 32))
      end do
   end subroutine unpack_hv_u32_lanes

   ! Materialize the exact sequence consumed by hv_bitacc_finalize: position
   ! permutations first, then the hybrid topical view, then optional bounded
   ! hierarchical record summary entries.  This has no model writes.
   subroutine pretrain_build_openvino_items(model, config, token_slots, n_tokens, center_index, &
                                            line_topic_context, line_active_tokens, experience_before, &
                                            expected_item_count, item_lanes, tie_lanes, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      type(hdc_pretrain_config_t), intent(in) :: config
      integer(int32), intent(in) :: token_slots(:)
      integer(int32), intent(in) :: n_tokens, center_index, line_active_tokens
      type(hv_t), intent(in) :: line_topic_context
      integer(int64), intent(in) :: experience_before
      integer(int32), intent(in) :: expected_item_count
      integer(int32), intent(out) :: item_lanes(:, :), tie_lanes(:)
      integer(int32), intent(out) :: status
      type(hv_t) :: neighbor, positioned, tie_breaker
      integer(int32) :: offset, neighbor_index, neighbor_slot, shift_bits
      integer(int32) :: item_index, global_items, global_index
      integer(int64) :: tie_seed

      status = PT_OK
      item_lanes = 0_int32
      tie_lanes = 0_int32
      if (center_index < 1_int32 .or. center_index > n_tokens .or. expected_item_count < 1_int32 .or. &
          size(item_lanes, 1, kind=int32) /= 2_int32 * HV_WORDS .or. &
          size(item_lanes, 2, kind=int32) < expected_item_count .or. &
          size(tie_lanes, kind=int32) /= 2_int32 * HV_WORDS) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%context_mode == PT_CONTEXT_LEGACY_POSITIONAL .or. &
          config%context_mode == PT_CONTEXT_COHERENT .or. &
          config%context_mode == PT_CONTEXT_DEBIASED .or. token_slots(center_index) == 0_int32) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if

      item_index = 0_int32
      do offset = -config%context_window, config%context_window
         if (offset == 0_int32) cycle
         neighbor_index = center_index + offset
         if (neighbor_index < 1_int32 .or. neighbor_index > n_tokens) cycle
         neighbor_slot = token_slots(neighbor_index)
         if (neighbor_slot == 0_int32) cycle
         neighbor%bits = model%base_bits(:, neighbor_slot)
         shift_bits = int(int(offset, int64) * int(config%position_shift_bits, int64), int32)
         positioned = hv_permute(neighbor, shift_bits)
         item_index = item_index + 1_int32
         call pack_hv_u32_lanes(positioned, item_lanes(:, item_index))
         if (config%context_mode == PT_CONTEXT_HYBRID .or. config%context_mode == PT_CONTEXT_HIERARCHICAL) then
            item_index = item_index + 1_int32
            call pack_hv_u32_lanes(neighbor, item_lanes(:, item_index))
         end if
      end do
      if (config%context_mode == PT_CONTEXT_HIERARCHICAL .and. &
          line_active_tokens >= HIERARCHICAL_MIN_ACTIVE_TOKENS) then
         global_items = min(HIERARCHICAL_GLOBAL_CONTEXT_WEIGHT, HV_BITACC_MAX_ITEMS - item_index)
         do global_index = 1, global_items
            item_index = item_index + 1_int32
            call pack_hv_u32_lanes(line_topic_context, item_lanes(:, item_index))
         end do
      end if
      if (item_index /= expected_item_count) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      tie_seed = ieor(vocab_seed_for_slot(model%vocabulary, token_slots(center_index)), &
                      ishftc(experience_before, 19))
      tie_breaker = hv_from_seed(ieor(config%tie_seed, tie_seed))
      call pack_hv_u32_lanes(tie_breaker, tie_lanes)
   end subroutine pretrain_build_openvino_items

   pure logical function openvino_host_is_little_endian()
      integer(int32) :: marker
      integer(int8) :: bytes(4)

      marker = 1_int32
      bytes = transfer(marker, bytes)
      openvino_host_is_little_endian = bytes(1) == 1_int8
   end function openvino_host_is_little_endian

   pure logical function openvino_safe_command_component(value)
      character(len=*), intent(in) :: value

      openvino_safe_command_component = len_trim(value) > 0 .and. &
                                       index(value, achar(34)) == 0 .and. &
                                       index(value, achar(10)) == 0 .and. &
                                       index(value, achar(13)) == 0
   end function openvino_safe_command_component

   subroutine openvino_delete_stream_file(path)
      character(len=*), intent(in) :: path
      integer :: unit, io_status

      open(newunit=unit, file=trim(path), status='old', action='readwrite', &
           access='stream', form='unformatted', iostat=io_status)
      if (io_status == 0) close(unit, status='delete')
   end subroutine openvino_delete_stream_file

   pure logical function openvino_explicit_cpu_device(value)
      character(len=*), intent(in) :: value
      character(len=256) :: normalized

      normalized = adjustl(trim(value))
      openvino_explicit_cpu_device = trim(normalized) == 'CPU' .or. trim(normalized) == 'cpu' .or. &
                                     trim(normalized) == 'CPU.0' .or. trim(normalized) == 'cpu.0'
   end function openvino_explicit_cpu_device

   subroutine openvino_join_path(directory, leaf, path)
      character(len=*), intent(in) :: directory, leaf
      character(len=*), intent(out) :: path
      integer :: trailing_index

      path = ''
      trailing_index = len_trim(directory)
      if (trailing_index <= 0 .or. len_trim(leaf) <= 0) return
      if (directory(trailing_index:trailing_index) == '/' .or. directory(trailing_index:trailing_index) == achar(92)) then
         path = trim(directory) // trim(leaf)
      else
         path = trim(directory) // '/' // trim(leaf)
      end if
   end subroutine openvino_join_path

   subroutine openvino_make_session_paths(config, session, status)
      type(hdc_pretrain_config_t), intent(in) :: config
      type(openvino_context_session_t), intent(inout) :: session
      integer(int32), intent(out) :: status
      integer(int64) :: clock_count, clock_rate, clock_max
      character(len=80) :: suffix

      status = PT_OK
      session%directory = ''
      call system_clock(count=clock_count, count_rate=clock_rate, count_max=clock_max)
      write(suffix, '(I0)') clock_count
      call openvino_join_path(trim(config%openvino_work_dir), 'vegai-openvino-session-' // trim(suffix), session%directory)
      if (len_trim(session%directory) == 0) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      call openvino_join_path(trim(session%directory), 'ready.bin', session%ready_path)
      call openvino_join_path(trim(session%directory), 'startup.error', session%startup_error_path)
      call openvino_join_path(trim(session%directory), 'request.bin', session%request_path)
      call openvino_join_path(trim(session%directory), 'request.ready', session%request_ready_path)
      call openvino_join_path(trim(session%directory), 'response.bin', session%response_path)
      call openvino_join_path(trim(session%directory), 'response.ready', session%response_ready_path)
      call openvino_join_path(trim(session%directory), 'response.error', session%response_error_path)
      call openvino_join_path(trim(session%directory), 'shutdown.ready', session%shutdown_path)
      call openvino_join_path(trim(session%directory), 'stopped.ready', session%stopped_path)
      call openvino_join_path(trim(session%directory), 'stopped.ack', session%stopped_ack_path)
      if (len_trim(session%ready_path) == 0 .or. len_trim(session%request_path) == 0 .or. &
          len_trim(session%response_path) == 0 .or. len_trim(session%shutdown_path) == 0) then
         status = PT_ERR_CONTEXT_PROVIDER
      end if
   end subroutine openvino_make_session_paths

   subroutine openvino_write_marker(path, status)
      character(len=*), intent(in) :: path
      integer(int32), intent(out) :: status
      integer :: unit, io_status

      status = PT_OK
      open(newunit=unit, file=trim(path), status='replace', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      write(unit, iostat=io_status) 1_int32
      close(unit)
      if (io_status /= 0) status = PT_ERR_CONTEXT_PROVIDER
   end subroutine openvino_write_marker

   subroutine openvino_read_text_file(path, detail)
      character(len=*), intent(in) :: path
      character(len=*), intent(out) :: detail
      integer :: unit, io_status

      detail = ''
      open(newunit=unit, file=trim(path), status='old', action='read', form='formatted', iostat=io_status)
      if (io_status /= 0) return
      read(unit, '(A)', iostat=io_status) detail
      close(unit)
      if (io_status /= 0) detail = ''
   end subroutine openvino_read_text_file

   subroutine openvino_spin_pause(milliseconds)
      integer(int32), intent(in) :: milliseconds
      integer(int64) :: started, current, rate, maximum, target_ticks

      if (milliseconds <= 0_int32) return
      call system_clock(count=started, count_rate=rate, count_max=maximum)
      if (rate <= 0_int64) return
      target_ticks = max(1_int64, (rate * int(milliseconds, int64) + 999_int64) / 1000_int64)
      do
         call system_clock(count=current)
         if (current - started >= target_ticks) exit
      end do
   end subroutine openvino_spin_pause

   ! Wait for a worker-published success/error marker.  A bounded wait is
   ! essential for an optional external provider: an interrupted worker must
   ! never turn the deterministic trainer into an unbounded hang.
   subroutine openvino_wait_for_marker(success_path, error_path, timeout_milliseconds, &
                                       found_success, detail)
      character(len=*), intent(in) :: success_path, error_path
      integer(int32), intent(in) :: timeout_milliseconds
      logical, intent(out) :: found_success
      character(len=*), intent(out) :: detail
      logical :: exists
      integer(int64) :: started, current, rate, maximum, timeout_ticks

      found_success = .false.
      detail = ''
      call system_clock(count=started, count_rate=rate, count_max=maximum)
      if (rate <= 0_int64) then
         detail = 'system clock is unavailable for provider timeout'
         return
      end if
      timeout_ticks = max(1_int64, (rate * int(timeout_milliseconds, int64) + 999_int64) / 1000_int64)
      do
         inquire(file=trim(success_path), exist=exists)
         if (exists) then
            found_success = .true.
            return
         end if
         inquire(file=trim(error_path), exist=exists)
         if (exists) then
            call openvino_read_text_file(error_path, detail)
            if (len_trim(detail) == 0) detail = 'worker published an unreadable failure marker'
            return
         end if
         call system_clock(count=current)
         if (current - started >= timeout_ticks) exit
         call openvino_spin_pause(OPENVINO_POLL_MILLISECONDS)
      end do
      detail = 'worker marker wait timed out'
   end subroutine openvino_wait_for_marker

   subroutine openvino_prepare_session(config, session, execution, status)
      type(hdc_pretrain_config_t), intent(in) :: config
      type(openvino_context_session_t), intent(inout) :: session
      type(pretrain_execution_report_t), intent(inout), optional :: execution
      integer(int32), intent(out) :: status
      integer(int32) :: command_status, protocol_version, selected_mask, reserved_one, reserved_two
      integer :: unit, io_status
      logical :: ready
      character(len=8) :: ready_magic
      character(len=4096) :: command
      character(len=512) :: detail

      status = PT_OK
      if (session%active .or. session%cpu_only_bypass) return
      if (session%launch_attempted) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      session%launch_attempted = .true.

      ! A caller that explicitly permits CPU fallback and explicitly selects
      ! CPU should use the native fused/OpenMP implementation directly.  It is
      ! both faster and semantically authoritative; launching OpenVINO here
      ! would be a false performance optimization.
      if (config%openvino_fallback_cpu .and. .not. config%openvino_require_accelerator .and. &
          openvino_explicit_cpu_device(config%openvino_device)) then
         session%cpu_only_bypass = .true.
         return
      end if
      if (.not. openvino_host_is_little_endian() .or. &
          .not. openvino_safe_command_component(config%openvino_python) .or. &
          .not. openvino_safe_command_component(config%openvino_worker) .or. &
          .not. openvino_safe_command_component(config%openvino_work_dir) .or. &
          .not. openvino_safe_command_component(config%openvino_device)) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      call openvino_make_session_paths(config, session, status)
      if (status /= PT_OK) return
      command = '"' // trim(config%openvino_python) // '" "' // trim(config%openvino_worker) // &
                '" --serve "' // trim(session%directory) // '" --device "' // trim(config%openvino_device) // '" --quiet'
      if (config%openvino_require_accelerator) command = trim(command) // ' --require-accelerator'
      call execute_command_line(trim(command), wait=.false., cmdstat=command_status)
      if (command_status /= 0) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      call openvino_wait_for_marker(session%ready_path, session%startup_error_path, &
                                    config%openvino_timeout_milliseconds, ready, detail)
      if (.not. ready) then
         if (len_trim(detail) > 0) write(error_unit, '(A,A)') 'OpenVINO context provider: ', trim(detail)
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      open(newunit=unit, file=trim(session%ready_path), status='old', action='read', &
           access='stream', form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      read(unit, iostat=io_status) ready_magic, protocol_version, selected_mask, reserved_one, reserved_two
      close(unit)
      if (io_status /= 0 .or. ready_magic /= OPENVINO_SESSION_READY_MAGIC .or. &
          protocol_version /= OPENVINO_SESSION_READY_VERSION .or. selected_mask < 0_int32 .or. &
          iand(selected_mask, not(7_int32)) /= 0_int32 .or. reserved_one /= 0_int32 .or. reserved_two /= 0_int32) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      ! With auto selection, a CPU-only machine is discoverable only after the
      ! worker imports OpenVINO.  In fallback mode we keep that resident worker
      ! only until train teardown but route contexts to the much faster native
      ! CPU kernel; strict --openvino-required still exercises the graph.
      if (config%openvino_fallback_cpu .and. .not. config%openvino_require_accelerator .and. &
          selected_mask == 1_int32) then
         session%cpu_only_bypass = .true.
      end if
      session%active = .true.
      if (present(execution)) execution%openvino_worker_sessions = execution%openvino_worker_sessions + 1_int64
   end subroutine openvino_prepare_session

   subroutine openvino_stop_session(config, session, status)
      type(hdc_pretrain_config_t), intent(in) :: config
      type(openvino_context_session_t), intent(inout) :: session
      integer(int32), intent(out) :: status
      logical :: stopped
      character(len=512) :: detail

      status = PT_OK
      if (.not. session%active) return
      call openvino_write_marker(session%shutdown_path, status)
      if (status /= PT_OK) return
      call openvino_wait_for_marker(session%stopped_path, session%response_error_path, &
                                    config%openvino_timeout_milliseconds, stopped, detail)
      if (.not. stopped) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      call openvino_write_marker(session%stopped_ack_path, status)
      session%active = .false.
   end subroutine openvino_stop_session

   pure subroutine pretrain_pad_openvino_items(item_lanes, actual_item_count, wire_item_count, tie_lanes, status)
      integer(int32), intent(inout) :: item_lanes(:, :)
      integer(int32), intent(in) :: actual_item_count, wire_item_count
      integer(int32), intent(in) :: tie_lanes(:)
      integer(int32), intent(out) :: status
      integer(int32) :: padding_index

      status = PT_OK
      if (actual_item_count < 1_int32 .or. wire_item_count < actual_item_count .or. &
          wire_item_count > size(item_lanes, 2, kind=int32) .or. &
          mod(wire_item_count - actual_item_count, 2_int32) /= 0_int32 .or. &
          size(item_lanes, 1, kind=int32) /= size(tie_lanes, kind=int32)) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      ! Each (tie, NOT tie) pair contributes exactly one vote per packed bit.
      ! It therefore preserves strict majority and the seeded even-count tie
      ! rule while normalizing all odd/even cardinalities to two reusable graph
      ! shapes.  Zero initialization was done by pretrain_build_openvino_items.
      do padding_index = actual_item_count + 1_int32, wire_item_count, 2_int32
         item_lanes(:, padding_index) = tie_lanes
         item_lanes(:, padding_index + 1_int32) = not(tie_lanes)
      end do
   end subroutine pretrain_pad_openvino_items

   ! Dispatch all sufficiently numerous line contexts through one resident
   ! OpenVINO worker.  Context cardinalities are normalized to two fixed wire
   ! shapes (largest odd/even) by neutral tie/complement pairs.  This preserves
   ! the exact majority contract while allowing the worker to cache just two
   ! dynamic-batch graphs instead of compiling every edge cardinality on every
   ! corpus line.  CPU keeps the serial semantic commit stage.  On any
   ! worker/protocol error no model mutation has occurred, allowing the caller
   ! to choose its explicit CPU fallback.
   subroutine pretrain_construct_line_contexts_openvino(model, config, token_slots, n_tokens, &
                                                         line_topic_context, line_active_tokens, &
                                                         update_required, neighbor_counts, experiences_before, &
                                                         prepared_context_bits, prepared_status, session, used_provider, &
                                                         execution, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      type(hdc_pretrain_config_t), intent(in) :: config
      integer(int32), intent(in) :: token_slots(:), n_tokens, line_active_tokens
      type(hv_t), intent(in) :: line_topic_context
      logical, intent(in) :: update_required(:)
      integer(int32), intent(in) :: neighbor_counts(:)
      integer(int64), intent(in) :: experiences_before(:)
      integer(int64), intent(inout) :: prepared_context_bits(:, :)
      integer(int32), intent(inout) :: prepared_status(:)
      type(openvino_context_session_t), intent(inout) :: session
      logical, intent(out) :: used_provider
      type(pretrain_execution_report_t), intent(inout), optional :: execution
      integer(int32), intent(out) :: status
      integer(int32) :: group_counts(2), group_starts(2), group_next(2), group_wire_items(2)
      integer(int32) :: center_order(MAX_TOKENS_PER_LINE), center_actual_item_counts(MAX_TOKENS_PER_LINE)
      integer(int32) :: center_groups(MAX_TOKENS_PER_LINE)
      integer(int32) :: center_index, actual_item_count, wire_item_count, maximum_items, update_count, group_count
      integer(int32) :: group_index, cursor, local_index, output_groups
      integer(int32) :: protocol_version, lane_count, output_item_count, output_count, execution_mask
      integer(int32), allocatable :: item_lanes(:, :, :), tie_lanes(:, :), context_lanes(:, :)
      integer :: unit, io_status, allocation_status
      character(len=8) :: input_magic, output_magic
      character(len=512) :: failure_detail
      logical :: stream_open, response_ready

      status = PT_OK
      failure_detail = ''
      used_provider = .false.
      stream_open = .false.
      group_counts = 0_int32
      group_starts = 0_int32
      group_next = 0_int32
      group_wire_items = 0_int32
      center_order = 0_int32
      center_actual_item_counts = 0_int32
      center_groups = 0_int32
      if (config%context_provider /= PT_CONTEXT_PROVIDER_OPENVINO) return
      if (config%context_mode == PT_CONTEXT_LEGACY_POSITIONAL .or. &
          config%context_mode == PT_CONTEXT_COHERENT .or. &
          config%context_mode == PT_CONTEXT_DEBIASED) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if

      ! The maximum HDC item count is always odd: 4*window + 3.  Actual odd
      ! contexts pad to that shape; actual even contexts pad to one-less.  The
      ! difference is even in both cases, so tie/complement padding is neutral.
      maximum_items = 4_int32 * config%context_window + HIERARCHICAL_GLOBAL_CONTEXT_WEIGHT
      if (maximum_items < 3_int32 .or. maximum_items > HV_BITACC_MAX_ITEMS .or. &
          mod(maximum_items, 2_int32) == 0_int32) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      group_wire_items(1) = maximum_items
      group_wire_items(2) = maximum_items - 1_int32
      update_count = 0_int32
      do center_index = 1, n_tokens
         if (.not. update_required(center_index)) cycle
         actual_item_count = openvino_context_item_count(config, neighbor_counts(center_index), line_active_tokens)
         if (actual_item_count < 1_int32 .or. actual_item_count > maximum_items .or. &
             actual_item_count > HV_BITACC_MAX_ITEMS) then
            status = PT_ERR_CONTEXT_PROVIDER
            return
         end if
         if (mod(actual_item_count, 2_int32) == 0_int32) then
            group_index = 2_int32
         else
            group_index = 1_int32
         end if
         wire_item_count = group_wire_items(group_index)
         if (wire_item_count < actual_item_count .or. &
             mod(wire_item_count - actual_item_count, 2_int32) /= 0_int32) then
            status = PT_ERR_CONTEXT_PROVIDER
            return
         end if
         update_count = update_count + 1_int32
         center_actual_item_counts(center_index) = actual_item_count
         center_groups(center_index) = group_index
         group_counts(group_index) = group_counts(group_index) + 1_int32
      end do
      if (update_count < config%openvino_min_contexts) return

      group_count = 0_int32
      cursor = 1_int32
      do group_index = 1, 2
         if (group_counts(group_index) == 0_int32) cycle
         group_count = group_count + 1_int32
         group_starts(group_index) = cursor
         group_next(group_index) = cursor
         cursor = cursor + group_counts(group_index)
      end do
      if (cursor - 1_int32 /= update_count .or. group_count <= 0_int32) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      do center_index = 1, n_tokens
         if (.not. update_required(center_index)) cycle
         group_index = center_groups(center_index)
         center_order(group_next(group_index)) = center_index
         group_next(group_index) = group_next(group_index) + 1_int32
      end do

      call openvino_prepare_session(config, session, execution, status)
      if (status /= PT_OK) return
      if (session%cpu_only_bypass) then
         if (present(execution)) execution%openvino_cpu_bypass_contexts = &
                                      execution%openvino_cpu_bypass_contexts + int(update_count, int64)
         return
      end if
      if (.not. session%active) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if

      call openvino_delete_stream_file(session%request_path)
      call openvino_delete_stream_file(session%request_ready_path)
      call openvino_delete_stream_file(session%response_path)
      call openvino_delete_stream_file(session%response_ready_path)
      call openvino_delete_stream_file(session%response_error_path)
      open(newunit=unit, file=trim(session%request_path), status='replace', action='write', &
           access='stream', form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = PT_ERR_CONTEXT_PROVIDER
         return
      end if
      stream_open = .true.

      input_magic = 'VGHDCI01'
      write(unit, iostat=io_status) input_magic, 1_int32, group_count, 2_int32 * HV_WORDS, 0_int32
      if (io_status /= 0) goto 900
      do group_index = 1, 2
         if (group_counts(group_index) == 0_int32) cycle
         wire_item_count = group_wire_items(group_index)
         write(unit, iostat=io_status) wire_item_count, group_counts(group_index)
         if (io_status /= 0) goto 900
         allocate(item_lanes(2_int32 * HV_WORDS, wire_item_count, group_counts(group_index)), &
                  tie_lanes(2_int32 * HV_WORDS, group_counts(group_index)), stat=allocation_status)
         if (allocation_status /= 0) then
            status = PT_ERR_ALLOCATION
            goto 900
         end if
         do local_index = 1, group_counts(group_index)
            center_index = center_order(group_starts(group_index) + local_index - 1_int32)
            actual_item_count = center_actual_item_counts(center_index)
            call pretrain_build_openvino_items(model, config, token_slots, n_tokens, center_index, &
                                               line_topic_context, line_active_tokens, experiences_before(center_index), &
                                               actual_item_count, item_lanes(:, :, local_index), tie_lanes(:, local_index), status)
            if (status /= PT_OK) goto 900
            call pretrain_pad_openvino_items(item_lanes(:, :, local_index), actual_item_count, wire_item_count, &
                                             tie_lanes(:, local_index), status)
            if (status /= PT_OK) goto 900
         end do
         write(unit, iostat=io_status) item_lanes, tie_lanes
         deallocate(item_lanes, tie_lanes)
         if (io_status /= 0) goto 900
      end do
      close(unit)
      stream_open = .false.

      call openvino_write_marker(session%request_ready_path, status)
      if (status /= PT_OK) goto 900
      call openvino_wait_for_marker(session%response_ready_path, session%response_error_path, &
                                    config%openvino_timeout_milliseconds, response_ready, failure_detail)
      if (.not. response_ready) then
         status = PT_ERR_CONTEXT_PROVIDER
         goto 900
      end if

      open(newunit=unit, file=trim(session%response_path), status='old', action='read', &
           access='stream', form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         failure_detail = 'resident worker did not create a readable response file'
         status = PT_ERR_CONTEXT_PROVIDER
         goto 900
      end if
      stream_open = .true.
      read(unit, iostat=io_status) output_magic, protocol_version, output_groups, lane_count, execution_mask
      if (io_status /= 0 .or. output_magic /= 'VGHDCO01' .or. protocol_version /= 1_int32 .or. &
          output_groups /= group_count .or. lane_count /= 2_int32 * HV_WORDS .or. execution_mask <= 0_int32 .or. &
          iand(execution_mask, not(7_int32)) /= 0_int32) then
         write(failure_detail, '(A,A,A,I0,A,I0,A,I0,A,I0)') 'invalid worker response magic=', output_magic, &
              ' version=', protocol_version, ' groups=', output_groups, ' lanes=', lane_count, ' mask=', execution_mask
         status = PT_ERR_CONTEXT_PROVIDER
         goto 910
      end if
      do group_index = 1, 2
         if (group_counts(group_index) == 0_int32) cycle
         wire_item_count = group_wire_items(group_index)
         read(unit, iostat=io_status) output_item_count, output_count
         if (io_status /= 0 .or. output_item_count /= wire_item_count .or. &
             output_count /= group_counts(group_index)) then
            write(failure_detail, '(A,I0,A,I0,A,I0,A,I0)') 'worker group mismatch expected_items=', wire_item_count, &
                 ' expected_count=', group_counts(group_index), ' got_items=', output_item_count, ' got_count=', output_count
            status = PT_ERR_CONTEXT_PROVIDER
            goto 910
         end if
         allocate(context_lanes(2_int32 * HV_WORDS, group_counts(group_index)), stat=allocation_status)
         if (allocation_status /= 0) then
            status = PT_ERR_ALLOCATION
            goto 910
         end if
         read(unit, iostat=io_status) context_lanes
         if (io_status /= 0) then
            deallocate(context_lanes)
            status = PT_ERR_CONTEXT_PROVIDER
            goto 910
         end if
         do local_index = 1, group_counts(group_index)
            center_index = center_order(group_starts(group_index) + local_index - 1_int32)
            call unpack_hv_u32_lanes(context_lanes(:, local_index), prepared_context_bits(:, center_index))
            prepared_status(center_index) = PT_OK
         end do
         deallocate(context_lanes)
      end do
      close(unit, status='delete')
      stream_open = .false.
      call openvino_delete_stream_file(session%response_ready_path)
      call openvino_delete_stream_file(session%request_path)
      used_provider = .true.
      if (present(execution)) then
         execution%openvino_contexts = execution%openvino_contexts + int(update_count, int64)
         execution%openvino_worker_invocations = execution%openvino_worker_invocations + 1_int64
         execution%openvino_cpu_execution = execution%openvino_cpu_execution .or. &
                                           (iand(execution_mask, 1_int32) /= 0_int32)
         execution%openvino_gpu_execution = execution%openvino_gpu_execution .or. &
                                           (iand(execution_mask, 2_int32) /= 0_int32)
         execution%openvino_npu_execution = execution%openvino_npu_execution .or. &
                                           (iand(execution_mask, 4_int32) /= 0_int32)
      end if
      return

910   continue
      if (stream_open) then
         close(unit, status='delete')
         stream_open = .false.
      end if
900   continue
      if (stream_open) then
         close(unit, status='delete')
         stream_open = .false.
      end if
      if (allocated(item_lanes)) deallocate(item_lanes)
      if (allocated(tie_lanes)) deallocate(tie_lanes)
      if (allocated(context_lanes)) deallocate(context_lanes)
      call openvino_delete_stream_file(session%request_ready_path)
      call openvino_delete_stream_file(session%response_ready_path)
      call openvino_delete_stream_file(session%request_path)
      call openvino_delete_stream_file(session%response_path)
      call openvino_delete_stream_file(session%response_error_path)
      if (status == PT_OK) status = PT_ERR_CONTEXT_PROVIDER
      if (len_trim(failure_detail) == 0) failure_detail = 'resident request encoding, worker output, or protocol validation failed'
      write(error_unit, '(A,A)') 'OpenVINO context provider: ', trim(failure_detail)
   end subroutine pretrain_construct_line_contexts_openvino

   ! Apply an already-built context in the exact stream order.  The explicit
   ! experience assertion makes a parallel scheduler bug fail closed rather
   ! than silently changing the deterministic Hebbian schedule.
   subroutine pretrain_apply_center_update(model, center_slot, context_bits, experience_before, metrics, status)
      type(hdc_pretrain_model_t), intent(inout) :: model
      integer(int32), intent(in) :: center_slot
      integer(int64), intent(in) :: context_bits(HV_WORDS)
      integer(int64), intent(in) :: experience_before
      type(pretrain_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      type(hv_t) :: context, prototype, updated
      integer(int32) :: update_status, plasticity_percent
      integer(int64) :: tie_seed

      status = PT_OK
      if (center_slot < 1_int32 .or. center_slot > model%vocabulary%count .or. &
          .not. model%vocabulary%active(center_slot) .or. &
          model%experience_count(center_slot) /= experience_before) then
         status = PT_ERR_HYPERVECTOR
         return
      end if
      if (experience_before == huge(0_int64)) then
         status = PT_ERR_COUNTER_OVERFLOW
         return
      end if
      context%bits = context_bits
      prototype%bits = model%semantic_bits(:, center_slot)
      tie_seed = ieor(vocab_seed_for_slot(model%vocabulary, center_slot), ishftc(experience_before, 19))
      plasticity_percent = scheduled_plasticity_percent(model, experience_before)
      if (model%plasticity_sampler /= PT_PLASTICITY_SAMPLER_LEGACY .and. &
          model%plasticity_sampler /= PT_PLASTICITY_SAMPLER_FLOYD_COMPLEMENT) then
         status = PT_ERR_HYPERVECTOR
         return
      end if
      call hv_hebbian_update_percent(prototype, context, plasticity_percent, tie_seed, updated, update_status, &
                                     legacy_selection_mask = &
                                     model%plasticity_sampler == PT_PLASTICITY_SAMPLER_LEGACY)
      if (update_status /= HV_OK) then
         status = PT_ERR_HYPERVECTOR
         return
      end if
      model%semantic_bits(:, center_slot) = updated%bits
      model%experience_count(center_slot) = experience_before + 1_int64
      call add_nonnegative(metrics%center_updates, 1_int64, status)
   end subroutine pretrain_apply_center_update

   subroutine train_one_epoch(model, corpus_path, config, metrics, status, session, execution)
      type(hdc_pretrain_model_t), intent(inout) :: model
      character(len=*), intent(in) :: corpus_path
      type(hdc_pretrain_config_t), intent(in) :: config
      type(pretrain_metrics_t), intent(inout) :: metrics
      integer(int32), intent(out) :: status
      type(openvino_context_session_t), intent(inout) :: session
      type(pretrain_execution_report_t), intent(inout), optional :: execution
      character(len=MAX_LINE_CHARS) :: line
      character(len=TOKEN_LEN) :: tokens(MAX_TOKENS_PER_LINE)
      integer(int32) :: token_slots(MAX_TOKENS_PER_LINE)
      integer :: unit, io_status, allocation_status
      integer(int32) :: token_index, n_tokens, slot, vocab_status, line_active_tokens, line_topic_active_tokens
      integer(int32) :: debiased_distinct_tokens, update_count
      integer(int64) :: non_ascii_count
      integer(int64), allocatable :: prepared_context_bits(:, :), experiences_before(:)
      integer(int64), allocatable :: lookup_cache_fingerprints(:)
      integer(int32), allocatable :: neighbor_counts(:), prepared_status(:), line_update_counts(:), lookup_cache_slots(:)
      integer(int32) :: touched_slots(MAX_TOKENS_PER_LINE), line_topic_fold_active_counts(2)
      character(len=TOKEN_LEN), allocatable :: lookup_cache_tokens(:)
      logical, allocatable :: update_required(:), lookup_cache_used(:)
      type(hv_t) :: line_topic_context, line_topic_fold_contexts(2)
      logical :: eof, found, lookup_cache_hit, overlong, truncated, provider_used, construct_in_parallel

      status = PT_OK
      allocate(prepared_context_bits(HV_WORDS, MAX_TOKENS_PER_LINE), &
               experiences_before(MAX_TOKENS_PER_LINE), neighbor_counts(MAX_TOKENS_PER_LINE), &
               prepared_status(MAX_TOKENS_PER_LINE), update_required(MAX_TOKENS_PER_LINE), &
               line_update_counts(model%vocabulary%capacity), lookup_cache_tokens(PRETRAIN_LOOKUP_CACHE_SLOTS), &
               lookup_cache_fingerprints(PRETRAIN_LOOKUP_CACHE_SLOTS), lookup_cache_slots(PRETRAIN_LOOKUP_CACHE_SLOTS), &
               lookup_cache_used(PRETRAIN_LOOKUP_CACHE_SLOTS), stat=allocation_status)
      if (allocation_status /= 0) then
         status = PT_ERR_ALLOCATION
         return
      end if
      line_update_counts = 0_int32
      lookup_cache_tokens = ''
      lookup_cache_fingerprints = 0_int64
      lookup_cache_slots = 0_int32
      lookup_cache_used = .false.
      open(newunit=unit, file=trim(corpus_path), status='old', action='read', &
           form='formatted', iostat=io_status)
      if (io_status /= 0) then
         deallocate(prepared_context_bits, experiences_before, neighbor_counts, prepared_status, update_required, &
                    line_update_counts, lookup_cache_tokens, lookup_cache_fingerprints, lookup_cache_slots, &
                    lookup_cache_used)
         status = PT_ERR_IO
         return
      end if

      do
         call read_corpus_record(unit, line, eof, overlong, status)
         if (status /= PT_OK) then
            close(unit)
            return
         end if
         if (eof) exit
         call add_nonnegative(metrics%corpus_lines, 1_int64, status)
         if (status /= PT_OK) exit
         if (overlong) then
            call add_nonnegative(metrics%overlong_records, 1_int64, status)
            if (status /= PT_OK) exit
            if (config%strict_tokenization) then
               status = PT_ERR_RECORD_TOO_LONG
               exit
            end if
            cycle
         end if
         call tokenize_ascii_line(line, tokens, n_tokens, truncated, non_ascii_count)
         call add_nonnegative(metrics%non_ascii_bytes, non_ascii_count, status)
         if (status /= PT_OK) exit
         if (config%strict_tokenization .and. non_ascii_count > 0_int64) then
            status = PT_ERR_TOKENIZATION
            exit
         end if
         if (truncated) then
            call add_nonnegative(metrics%tokenizer_truncations, 1_int64, status)
            if (status /= PT_OK) exit
            if (config%strict_tokenization) then
               status = PT_ERR_TOKENIZATION
               exit
            end if
         end if

         token_slots = 0_int32
         line_active_tokens = 0_int32
         do token_index = 1, n_tokens
            call add_nonnegative(metrics%raw_tokens, 1_int64, status)
            if (status /= PT_OK) exit
            lookup_cache_hit = .false.
            if (config%vocabulary_lookup_cache) then
               call pretrain_lookup_active_cached(model, tokens(token_index), lookup_cache_tokens, &
                                                  lookup_cache_fingerprints, lookup_cache_slots, lookup_cache_used, &
                                                  slot, found, lookup_cache_hit, vocab_status)
               if (present(execution)) then
                  if (lookup_cache_hit) then
                     call add_nonnegative(execution%vocabulary_lookup_cache_hits, 1_int64, status)
                  else
                     call add_nonnegative(execution%vocabulary_lookup_cache_misses, 1_int64, status)
                  end if
                  if (status /= PT_OK) exit
               end if
            else
               call vocab_lookup(model%vocabulary, tokens(token_index), slot, found, vocab_status, .true.)
            end if
            if (found) then
               token_slots(token_index) = slot
               line_active_tokens = line_active_tokens + 1_int32
               call add_nonnegative(metrics%active_tokens, 1_int64, status)
            else if (vocab_status == VOCAB_ERR_NOT_FOUND) then
               call add_nonnegative(metrics%oov_tokens, 1_int64, status)
            else
               status = PT_ERR_VOCABULARY
            end if
            if (status /= PT_OK) exit
         end do
         if (status /= PT_OK) exit

         line_topic_context = hv_zero()
         line_topic_fold_contexts(1) = hv_zero()
         line_topic_fold_contexts(2) = hv_zero()
         line_topic_fold_active_counts = 0_int32
         line_topic_active_tokens = line_active_tokens
         debiased_distinct_tokens = 0_int32
         if (config%context_mode == PT_CONTEXT_HIERARCHICAL .and. &
             line_active_tokens >= HIERARCHICAL_MIN_ACTIVE_TOKENS) then
            call pretrain_build_line_topic_context(model, token_slots, n_tokens, line_active_tokens, &
                                                   line_topic_context, status)
            if (status /= PT_OK) exit
         else if (config%context_mode == PT_CONTEXT_COHERENT .and. &
                  line_active_tokens >= HIERARCHICAL_MIN_ACTIVE_TOKENS) then
            call pretrain_build_coherent_line_topic_contexts(model, token_slots, n_tokens, line_active_tokens, &
                                                             line_topic_context, line_topic_fold_contexts, &
                                                             line_topic_fold_active_counts, status)
            if (status /= PT_OK) exit
         else if (config%context_mode == PT_CONTEXT_DEBIASED .and. &
                  line_active_tokens >= DEBIASED_MIN_DISTINCT_ACTIVE_TOKENS) then
            call pretrain_build_debiased_line_topic_context(model, token_slots, n_tokens, line_active_tokens, &
                                                            line_topic_context, debiased_distinct_tokens, &
                                                            line_update_counts, touched_slots, status)
            if (status /= PT_OK) exit
            line_topic_active_tokens = debiased_distinct_tokens
            if (present(execution)) then
               call add_nonnegative(execution%debiased_topic_distinct_tokens, &
                                    int(debiased_distinct_tokens, int64), status)
               if (status /= PT_OK) exit
               if (debiased_distinct_tokens >= DEBIASED_MIN_DISTINCT_ACTIVE_TOKENS) then
                  call add_nonnegative(execution%debiased_topic_records, 1_int64, status)
               else
                  call add_nonnegative(execution%debiased_topic_suppressed_records, 1_int64, status)
               end if
               if (status /= PT_OK) exit
            end if
         end if

         call pretrain_plan_line_updates(model, config, token_slots, n_tokens, update_required, &
                                         neighbor_counts, experiences_before, line_update_counts, touched_slots, &
                                         update_count, status)
         if (status /= PT_OK) exit
         if (update_count == 0_int32) cycle
         ! Each selected CPU/provider path writes every update column and
         ! status before the ordered commit below.  Do not clear the 2 MiB
         ! staging matrix for every short input line: untouched columns are
         ! never read, and a full wipe was pure memory-bandwidth overhead.
         provider_used = .false.
         if (config%context_provider == PT_CONTEXT_PROVIDER_OPENVINO) then
            if (present(execution)) execution%openvino_provider_requested = .true.
            if (present(execution)) then
               call pretrain_construct_line_contexts_openvino(model, config, token_slots, n_tokens, &
                                                              line_topic_context, line_active_tokens, update_required, &
                                                              neighbor_counts, experiences_before, prepared_context_bits, &
                                                              prepared_status, session, provider_used, execution, status)
            else
               call pretrain_construct_line_contexts_openvino(model, config, token_slots, n_tokens, &
                                                              line_topic_context, line_active_tokens, update_required, &
                                                              neighbor_counts, experiences_before, prepared_context_bits, &
                                                              prepared_status, session, provider_used, status=status)
            end if
            if (status == PT_ERR_CONTEXT_PROVIDER .and. config%openvino_fallback_cpu) then
               status = PT_OK
               if (present(execution)) then
                  execution%openvino_fallback_contexts = execution%openvino_fallback_contexts + &
                                                          int(update_count, int64)
               end if
            else if (status /= PT_OK) then
               exit
            end if
         end if
         if (.not. provider_used) then
            ! Launching a team for each tiny natural-language record is often
            ! slower than the bounded packed-HV work itself.  Large records
            ! still use the immutable OpenMP stage; both paths commit below in
            ! source order and therefore share identical checkpoint semantics.
            construct_in_parallel = config%parallel_context_build .and. &
                                    update_count >= config%parallel_min_contexts
            if (construct_in_parallel) then
               ! Context construction touches immutable base vectors only. Each
               ! iteration owns its output column and status element; commits
               ! below deliberately remain in corpus source order.
!$omp parallel do schedule(static)
               do token_index = 1, n_tokens
                  if (.not. update_required(token_index)) cycle
                  call pretrain_construct_center_context(model, config, token_slots, n_tokens, token_index, &
                                                         neighbor_counts(token_index), line_topic_context, &
                                                         line_topic_fold_contexts, line_topic_active_tokens, &
                                                         line_topic_fold_active_counts, experiences_before(token_index), &
                                                         prepared_context_bits(:, token_index), &
                                                         prepared_status(token_index))
               end do
!$omp end parallel do
            else
               do token_index = 1, n_tokens
                  if (.not. update_required(token_index)) cycle
                  call pretrain_construct_center_context(model, config, token_slots, n_tokens, token_index, &
                                                         neighbor_counts(token_index), line_topic_context, &
                                                         line_topic_fold_contexts, line_topic_active_tokens, &
                                                         line_topic_fold_active_counts, experiences_before(token_index), &
                                                         prepared_context_bits(:, token_index), &
                                                         prepared_status(token_index))
               end do
            end if
            if (present(execution)) then
               execution%cpu_contexts = execution%cpu_contexts + int(update_count, int64)
               if (construct_in_parallel) then
                  execution%cpu_parallel_contexts = execution%cpu_parallel_contexts + int(update_count, int64)
                  execution%cpu_parallel_batches = execution%cpu_parallel_batches + 1_int64
               else
                  execution%cpu_serial_contexts = execution%cpu_serial_contexts + int(update_count, int64)
                  execution%cpu_serial_batches = execution%cpu_serial_batches + 1_int64
               end if
            end if
         end if
         do token_index = 1, n_tokens
            if (.not. update_required(token_index)) cycle
            if (prepared_status(token_index) /= PT_OK) then
               status = prepared_status(token_index)
               exit
            end if
         end do
         if (status /= PT_OK) exit

         do token_index = 1, n_tokens
            if (.not. update_required(token_index)) cycle
            call add_nonnegative(metrics%context_pairs, int(neighbor_counts(token_index), int64), status)
            if (status /= PT_OK) exit
            call pretrain_apply_center_update(model, token_slots(token_index), &
                                              prepared_context_bits(:, token_index), &
                                              experiences_before(token_index), metrics, status)
            if (status /= PT_OK) exit
         end do
         if (status /= PT_OK) exit
      end do
      close(unit)
      deallocate(prepared_context_bits, experiences_before, neighbor_counts, prepared_status, update_required, &
                 line_update_counts, lookup_cache_tokens, lookup_cache_fingerprints, lookup_cache_slots, lookup_cache_used)
   end subroutine train_one_epoch

   subroutine pretrain_train_file(model, corpus_path, config, metrics, status, execution)
      type(hdc_pretrain_model_t), intent(inout) :: model
      character(len=*), intent(in) :: corpus_path
      type(hdc_pretrain_config_t), intent(in) :: config
      type(pretrain_metrics_t), intent(out) :: metrics
      integer(int32), intent(out) :: status
      type(pretrain_execution_report_t), intent(inout), optional :: execution
      type(openvino_context_session_t) :: openvino_session
      integer(int32) :: epoch, tally_status, original_status, stop_status

      openvino_session = openvino_context_session_t()
      call pretrain_metrics_reset(metrics)
      if (present(execution)) call pretrain_execution_report_reset(execution)
      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if
      call validate_pretrain_config(config, status)
      if (status /= PT_OK) return
      if (len_trim(corpus_path) == 0) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      if (config%max_vocabulary /= model%vocabulary%capacity .or. &
          config%min_frequency /= model%vocabulary%min_frequency .or. &
          config%context_window /= model%context_window .or. &
          config%position_shift_bits /= model%position_shift_bits .or. &
          config%context_mode /= model%context_mode .or. &
          config%initial_plasticity_percent /= model%initial_plasticity_percent .or. &
          config%minimum_plasticity_percent /= model%minimum_plasticity_percent .or. &
          config%plasticity_decay_events /= model%plasticity_decay_events .or. &
          config%tie_seed /= model%tie_seed) then
         status = PT_ERR_INCOMPATIBLE_CONFIG
         return
      end if

      do epoch = 1, config%epochs
         if (present(execution)) then
            call train_one_epoch(model, corpus_path, config, metrics, status, openvino_session, execution)
         else
            call train_one_epoch(model, corpus_path, config, metrics, status, openvino_session)
         end if
         if (status /= PT_OK) then
            original_status = status
            call add_metrics(model%totals, metrics, tally_status)
            if (tally_status /= PT_OK) then
               status = tally_status
            else
               status = original_status
            end if
            goto 800
         end if
         if (model%training_epochs_completed == huge(0_int64)) then
            call add_metrics(model%totals, metrics, tally_status)
            if (tally_status /= PT_OK) then
               status = tally_status
            else
               status = PT_ERR_COUNTER_OVERFLOW
            end if
            goto 800
         end if
         model%training_epochs_completed = model%training_epochs_completed + 1_int64
      end do
      call add_metrics(model%totals, metrics, status)

800   continue
      ! A resident worker is strictly execution infrastructure.  Always ask it
      ! to stop before returning, including after an HDC/training error, so a
      ! failed optional run cannot leak an OpenVINO process or SSD session dir.
      call openvino_stop_session(config, openvino_session, stop_status)
      if (stop_status /= PT_OK) then
         write(error_unit, '(A)') 'OpenVINO context provider: resident worker shutdown did not complete'
         if (status == PT_OK .and. .not. config%openvino_fallback_cpu) status = stop_status
      end if
   end subroutine pretrain_train_file

   subroutine pretrain_lookup_token(model, token, slot, found, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      character(len=*), intent(in) :: token
      integer(int32), intent(out) :: slot
      logical, intent(out) :: found
      integer(int32), intent(out) :: status
      integer(int32) :: vocab_status

      slot = 0_int32
      found = .false.
      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if
      call vocab_lookup(model%vocabulary, token, slot, found, vocab_status, .true.)
      if (vocab_status == VOCAB_ERR_NOT_FOUND .or. .not. found) then
         status = PT_ERR_NOT_FOUND
      else if (vocab_status /= VOCAB_OK) then
         status = PT_ERR_VOCABULARY
      end if
   end subroutine pretrain_lookup_token

   subroutine pretrain_get_semantic_hv(model, slot, semantic_hv, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int32), intent(in) :: slot
      type(hv_t), intent(out) :: semantic_hv
      integer(int32), intent(out) :: status

      semantic_hv = hv_zero()
      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if
      if (slot < 1_int32 .or. slot > model%vocabulary%count .or. &
          .not. model%vocabulary%active(slot)) then
         status = PT_ERR_NOT_FOUND
         return
      end if
      semantic_hv%bits = model%semantic_bits(:, slot)
   end subroutine pretrain_get_semantic_hv

   subroutine pretrain_nearest_semantic(model, query_hv, best_slot, best_token, best_similarity, &
                                        status, exclude_slot)
      type(hdc_pretrain_model_t), intent(in) :: model
      type(hv_t), intent(in) :: query_hv
      integer(int32), intent(out) :: best_slot
      character(len=TOKEN_LEN), intent(out) :: best_token
      real(real32), intent(out) :: best_similarity
      integer(int32), intent(out) :: status
      integer(int32), intent(in), optional :: exclude_slot
      type(hv_t) :: candidate
      integer(int32) :: slot, distance, best_distance, excluded

      best_slot = 0_int32
      best_token = ''
      best_similarity = -1.0_real32
      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if

      excluded = 0_int32
      if (present(exclude_slot)) excluded = exclude_slot
      best_distance = HV_BITS + 1_int32
      do slot = 1, model%vocabulary%count
         if (.not. model%vocabulary%active(slot) .or. slot == excluded) cycle
         candidate%bits = model%semantic_bits(:, slot)
         distance = hv_hamming(query_hv, candidate)
         if (distance < best_distance) then
            best_distance = distance
            best_slot = slot
         end if
      end do
      if (best_slot == 0_int32) then
         status = PT_ERR_NOT_FOUND
         return
      end if
      best_token = model%vocabulary%tokens(best_slot)
      best_similarity = hv_similarity_from_distance(best_distance)
   end subroutine pretrain_nearest_semantic

   subroutine pretrain_config_from_model(model, config, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      type(hdc_pretrain_config_t), intent(out) :: config
      integer(int32), intent(out) :: status

      config = hdc_pretrain_config_t()
      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if
      config%max_vocabulary = model%vocabulary%capacity
      config%min_frequency = model%vocabulary%min_frequency
      config%context_window = model%context_window
      config%position_shift_bits = model%position_shift_bits
      config%context_mode = model%context_mode
      config%initial_plasticity_percent = model%initial_plasticity_percent
      config%minimum_plasticity_percent = model%minimum_plasticity_percent
      config%plasticity_decay_events = model%plasticity_decay_events
      config%tie_seed = model%tie_seed
   end subroutine pretrain_config_from_model

   pure function pretrain_model_footprint_bytes(model) result(total_bytes)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int64) :: total_bytes

      total_bytes = vocab_footprint_bytes(model%vocabulary)
      if (allocated(model%base_bits)) then
         total_bytes = total_bytes + size(model%base_bits, kind=int64) * &
                       int(storage_size(model%base_bits(1, 1)) / 8, int64)
      end if
      if (allocated(model%semantic_bits)) then
         total_bytes = total_bytes + size(model%semantic_bits, kind=int64) * &
                       int(storage_size(model%semantic_bits(1, 1)) / 8, int64)
      end if
      if (allocated(model%experience_count)) then
         total_bytes = total_bytes + size(model%experience_count, kind=int64) * &
                       int(storage_size(model%experience_count(1)) / 8, int64)
      end if
   end function pretrain_model_footprint_bytes

   ! A deterministic accidental-corruption detector for a checkpoint payload.
   ! It is not a cryptographic signature and must not be used as an authenticity
   ! mechanism.  Stream binary layout remains compiler/ABI-local by contract.
   pure function checkpoint_mix(state, value) result(mixed)
      integer(int64), intent(in) :: state, value
      integer(int64) :: mixed

      mixed = ieor(ishftc(state, 9), value)
      mixed = ieor(mixed, ishftc(value, -13))
      mixed = ieor(mixed, ishftc(state, -27))
   end function checkpoint_mix

   pure function model_checkpoint_checksum(model, checkpoint_version) result(checksum)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int32), intent(in) :: checkpoint_version
      integer(int64) :: checksum
      integer(int32) :: slot, word, character_index

      checksum = int(z'BB67AE8584CAA73B', int64)
      checksum = checkpoint_mix(checksum, int(checkpoint_version, int64))
      checksum = checkpoint_mix(checksum, int(model%vocabulary%capacity, int64))
      checksum = checkpoint_mix(checksum, int(model%vocabulary%count, int64))
      checksum = checkpoint_mix(checksum, int(model%vocabulary%active_count, int64))
      checksum = checkpoint_mix(checksum, model%vocabulary%min_frequency)
      checksum = checkpoint_mix(checksum, int(model%context_window, int64))
      checksum = checkpoint_mix(checksum, int(model%position_shift_bits, int64))
      checksum = checkpoint_mix(checksum, int(model%context_mode, int64))
      checksum = checkpoint_mix(checksum, int(model%initial_plasticity_percent, int64))
      checksum = checkpoint_mix(checksum, int(model%minimum_plasticity_percent, int64))
      checksum = checkpoint_mix(checksum, int(model%plasticity_decay_events, int64))
      checksum = checkpoint_mix(checksum, model%tie_seed)
      checksum = checkpoint_mix(checksum, model%training_epochs_completed)
      checksum = checkpoint_mix(checksum, model%totals%corpus_lines)
      checksum = checkpoint_mix(checksum, model%totals%raw_tokens)
      checksum = checkpoint_mix(checksum, model%totals%active_tokens)
      checksum = checkpoint_mix(checksum, model%totals%oov_tokens)
      checksum = checkpoint_mix(checksum, model%totals%context_pairs)
      checksum = checkpoint_mix(checksum, model%totals%center_updates)
      checksum = checkpoint_mix(checksum, model%totals%tokenizer_truncations)
      checksum = checkpoint_mix(checksum, model%totals%non_ascii_bytes)
      checksum = checkpoint_mix(checksum, model%totals%overlong_records)
      do slot = 1, model%vocabulary%capacity
         do character_index = 1, TOKEN_LEN
            checksum = checkpoint_mix(checksum, &
                                      int(iachar(model%vocabulary%tokens(slot)(character_index:character_index)), int64))
         end do
         checksum = checkpoint_mix(checksum, model%vocabulary%frequencies(slot))
         if (model%vocabulary%active(slot)) then
            checksum = checkpoint_mix(checksum, 1_int64)
         else
            checksum = checkpoint_mix(checksum, 0_int64)
         end if
         checksum = checkpoint_mix(checksum, model%experience_count(slot))
         do word = 1, HV_WORDS
            checksum = checkpoint_mix(checksum, model%base_bits(word, slot))
            checksum = checkpoint_mix(checksum, model%semantic_bits(word, slot))
         end do
      end do
   end function model_checkpoint_checksum

   pure function pretrain_checkpoint_write_version(model) result(format_version)
      type(hdc_pretrain_model_t), intent(in) :: model
      integer(int32) :: format_version

      ! Keep the result name distinct from CHECKPOINT_VERSION: Fortran names
      ! are case-insensitive, so a same-spelling result would shadow the module
      ! constant. A v7 model is the new unbiased sampler lineage. A loaded
      ! v4-v6 model writes its legacy version so future resumes continue with
      ! its original deterministic selection stream.
      format_version = 0_int32
      select case (model%plasticity_sampler)
      case (PT_PLASTICITY_SAMPLER_FLOYD_COMPLEMENT)
         format_version = CHECKPOINT_VERSION
      case (PT_PLASTICITY_SAMPLER_LEGACY)
         format_version = CHECKPOINT_PRE_COHERENT_VERSION
         if (model%context_mode == PT_CONTEXT_COHERENT) then
            format_version = CHECKPOINT_COHERENT_LEGACY_VERSION
         end if
      end select
   end function pretrain_checkpoint_write_version

   subroutine pretrain_checkpoint_save(model, checkpoint_path, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      character(len=*), intent(in) :: checkpoint_path
      integer(int32), intent(out) :: status
      integer :: unit, io_status
      integer(int32) :: checkpoint_version
      integer(int64) :: checksum

      status = PT_OK
      if (.not. pretrain_model_is_ready(model)) then
         status = PT_ERR_MODEL_NOT_READY
         return
      end if
      if (len_trim(checkpoint_path) == 0) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      ! A de-biased semantic context is only defined for the current v7
      ! Floyd/complement lineage. Public construction always satisfies this;
      ! reject an in-memory forged combination rather than serializing a file
      ! that an older sampler contract could misrepresent.
      if (model%context_mode == PT_CONTEXT_DEBIASED .and. &
          model%plasticity_sampler /= PT_PLASTICITY_SAMPLER_FLOYD_COMPLEMENT) then
         status = PT_ERR_CHECKPOINT
         return
      end if

      checkpoint_version = pretrain_checkpoint_write_version(model)
      open(newunit=unit, file=trim(checkpoint_path), status='replace', action='write', &
           access='stream', form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = PT_ERR_IO
         return
      end if

      write(unit, iostat=io_status) CHECKPOINT_MAGIC, checkpoint_version
      if (io_status == 0) write(unit, iostat=io_status) model%vocabulary%capacity, &
         model%vocabulary%count, model%vocabulary%active_count, model%vocabulary%min_frequency
      if (io_status == 0) write(unit, iostat=io_status) model%context_window, model%position_shift_bits, &
         model%context_mode, model%initial_plasticity_percent, model%minimum_plasticity_percent, &
         model%plasticity_decay_events, model%tie_seed
      if (io_status == 0) write(unit, iostat=io_status) model%training_epochs_completed
      if (io_status == 0) write(unit, iostat=io_status) model%totals%corpus_lines, model%totals%raw_tokens, &
         model%totals%active_tokens, model%totals%oov_tokens, model%totals%context_pairs, &
         model%totals%center_updates, model%totals%tokenizer_truncations, &
         model%totals%non_ascii_bytes, model%totals%overlong_records
      if (io_status == 0) write(unit, iostat=io_status) model%vocabulary%tokens
      if (io_status == 0) write(unit, iostat=io_status) model%vocabulary%frequencies
      if (io_status == 0) write(unit, iostat=io_status) model%vocabulary%active
      if (io_status == 0) write(unit, iostat=io_status) model%base_bits
      if (io_status == 0) write(unit, iostat=io_status) model%semantic_bits
      if (io_status == 0) write(unit, iostat=io_status) model%experience_count
      if (io_status == 0) then
         checksum = model_checkpoint_checksum(model, checkpoint_version)
         write(unit, iostat=io_status) checksum
      end if
      close(unit)
      if (io_status /= 0) status = PT_ERR_CHECKPOINT
   end subroutine pretrain_checkpoint_save

   subroutine pretrain_checkpoint_load(checkpoint_path, model, status)
      character(len=*), intent(in) :: checkpoint_path
      type(hdc_pretrain_model_t), intent(inout) :: model
      integer(int32), intent(out) :: status
      character(len=8) :: magic
      type(hdc_pretrain_config_t) :: checkpoint_config
      integer :: unit, io_status
      integer(int32) :: version, capacity, count, stored_active_count, vocab_status, slot
      integer(int64) :: min_frequency, stored_checksum

      call pretrain_model_destroy(model)
      status = PT_OK
      if (len_trim(checkpoint_path) == 0) then
         status = PT_ERR_INVALID_ARGUMENT
         return
      end if
      open(newunit=unit, file=trim(checkpoint_path), status='old', action='read', &
           access='stream', form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = PT_ERR_IO
         return
      end if

      read(unit, iostat=io_status) magic, version
      if (io_status /= 0) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         return
      end if
      if (magic /= CHECKPOINT_MAGIC .or. version < CHECKPOINT_MIN_VERSION .or. &
          version > CHECKPOINT_VERSION) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         return
      end if
      if (version == CHECKPOINT_VERSION) then
         model%plasticity_sampler = PT_PLASTICITY_SAMPLER_FLOYD_COMPLEMENT
      else
         model%plasticity_sampler = PT_PLASTICITY_SAMPLER_LEGACY
      end if
      read(unit, iostat=io_status) capacity, count, stored_active_count, min_frequency
      if (io_status /= 0) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         return
      end if
      if (capacity <= 0_int32 .or. count <= 0_int32 .or. count > capacity .or. &
          stored_active_count <= 0_int32 .or. stored_active_count > count .or. &
          min_frequency <= 0_int64) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         return
      end if
      call vocab_init(model%vocabulary, capacity, min_frequency, vocab_status)
      if (vocab_status /= VOCAB_OK) then
         close(unit)
         status = PT_ERR_VOCABULARY
         call pretrain_model_destroy(model)
         return
      end if
      model%vocabulary%count = count

      read(unit, iostat=io_status) model%context_window, model%position_shift_bits, &
         model%context_mode, model%initial_plasticity_percent, model%minimum_plasticity_percent, &
         model%plasticity_decay_events, model%tie_seed
      if (io_status == 0) read(unit, iostat=io_status) model%training_epochs_completed
      if (io_status == 0) read(unit, iostat=io_status) model%totals%corpus_lines, model%totals%raw_tokens, &
         model%totals%active_tokens, model%totals%oov_tokens, model%totals%context_pairs, &
         model%totals%center_updates, model%totals%tokenizer_truncations, &
         model%totals%non_ascii_bytes, model%totals%overlong_records
      if (io_status == 0) read(unit, iostat=io_status) model%vocabulary%tokens
      if (io_status == 0) read(unit, iostat=io_status) model%vocabulary%frequencies
      if (io_status == 0) read(unit, iostat=io_status) model%vocabulary%active
      if (io_status /= 0) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         call pretrain_model_destroy(model)
         return
      end if

      checkpoint_config = hdc_pretrain_config_t()
      checkpoint_config%max_vocabulary = capacity
      checkpoint_config%min_frequency = min_frequency
      checkpoint_config%context_window = model%context_window
      checkpoint_config%position_shift_bits = model%position_shift_bits
      checkpoint_config%context_mode = model%context_mode
      checkpoint_config%initial_plasticity_percent = model%initial_plasticity_percent
      checkpoint_config%minimum_plasticity_percent = model%minimum_plasticity_percent
      checkpoint_config%plasticity_decay_events = model%plasticity_decay_events
      checkpoint_config%tie_seed = model%tie_seed
      if ((model%context_mode == PT_CONTEXT_COHERENT .and. &
           version < CHECKPOINT_COHERENT_LEGACY_VERSION) .or. &
          (model%context_mode == PT_CONTEXT_DEBIASED .and. version < CHECKPOINT_VERSION)) then
         ! DEBIASED was introduced under the current v7 sampler lineage. A
         ! forged older checkpoint must not be interpreted as a valid distinct-
         ! topic continuation simply because its wire layout happens to fit.
         close(unit)
         status = PT_ERR_CHECKPOINT
         call pretrain_model_destroy(model)
         return
      end if
      call validate_pretrain_config(checkpoint_config, status)
      if (status /= PT_OK .or. model%training_epochs_completed < 0_int64 .or. &
          model%totals%corpus_lines < 0_int64 .or. model%totals%raw_tokens < 0_int64 .or. &
          model%totals%active_tokens < 0_int64 .or. model%totals%oov_tokens < 0_int64 .or. &
          model%totals%context_pairs < 0_int64 .or. model%totals%center_updates < 0_int64 .or. &
          model%totals%tokenizer_truncations < 0_int64 .or. model%totals%non_ascii_bytes < 0_int64 .or. &
          model%totals%overlong_records < 0_int64 .or. &
          model%totals%active_tokens > model%totals%raw_tokens .or. &
          model%totals%oov_tokens > model%totals%raw_tokens .or. &
          model%totals%center_updates > model%totals%active_tokens) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         call pretrain_model_destroy(model)
         return
      end if

      do slot = 1, count
         if (model%vocabulary%frequencies(slot) <= 0_int64 .or. &
             (model%vocabulary%active(slot) .neqv. &
              (model%vocabulary%frequencies(slot) >= min_frequency))) then
            close(unit)
            status = PT_ERR_CHECKPOINT
            call pretrain_model_destroy(model)
            return
         end if
      end do
      call vocab_rebuild_index(model%vocabulary, vocab_status)
      if (vocab_status /= VOCAB_OK .or. model%vocabulary%active_count /= stored_active_count) then
         close(unit)
         status = PT_ERR_CHECKPOINT
         call pretrain_model_destroy(model)
         return
      end if
      call allocate_model_storage(model, status)
      if (status /= PT_OK) then
         close(unit)
         call pretrain_model_destroy(model)
         return
      end if
      read(unit, iostat=io_status) model%base_bits
      if (io_status == 0) read(unit, iostat=io_status) model%semantic_bits
      if (io_status == 0) read(unit, iostat=io_status) model%experience_count
      if (io_status == 0) read(unit, iostat=io_status) stored_checksum
      close(unit)
      if (io_status /= 0) then
         status = PT_ERR_CHECKPOINT
         call pretrain_model_destroy(model)
         return
      end if
      if (any(model%experience_count < 0_int64) .or. &
          stored_checksum /= model_checkpoint_checksum(model, version)) then
         status = PT_ERR_CHECKPOINT
         call pretrain_model_destroy(model)
         return
      end if
      model%ready = .true.
   end subroutine pretrain_checkpoint_load

end module hdc_pretraining
