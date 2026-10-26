! Deterministic synthetic objective definitions for benchmark-only studies.
! These tasks are synthetic testbeds, not application datasets.
module bench_tasks
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   use bench_contracts, only : bench_seed_state, bench_normal, bench_uniform, BENCH_OK, BENCH_INVALID, BENCH_NUMERICAL
   implicit none
   private

   integer(int32), parameter, public :: TASK_QUADRATIC = 1_int32
   integer(int32), parameter, public :: TASK_ROSENBROCK = 2_int32
   integer(int32), parameter, public :: TASK_RASTRIGIN = 3_int32
   integer(int32), parameter, public :: TASK_LOGISTIC = 4_int32
   integer(int32), parameter, public :: TASK_SPARSE_LOGISTIC = 5_int32
   integer(int32), parameter, public :: TASK_MLP = 6_int32
   integer(int32), parameter, public :: TASK_TRANSFORMER_LM = 7_int32
   integer(int32), parameter, public :: TASK_MATRIX_FACTORIZATION = 8_int32
   integer(int32), parameter, public :: TASK_NOISY_QUADRATIC = 9_int32
   integer(int32), parameter, public :: TASK_PARTITION_FULL = 0_int32
   integer(int32), parameter, public :: TASK_PARTITION_TRAIN = 1_int32
   integer(int32), parameter, public :: TASK_PARTITION_VALIDATION = 2_int32
   integer(int32), parameter, public :: TASK_PARTITION_TEST = 3_int32

   integer(int32), parameter :: MLP_INPUTS = 4_int32
   integer(int32), parameter :: MLP_HIDDEN1 = 8_int32
   integer(int32), parameter :: MLP_HIDDEN2 = 8_int32
   integer(int32), parameter :: MLP_PARAMETERS = 121_int32
   integer(int32), parameter :: TRANSFORMER_WIDTH = 4_int32
   integer(int32), parameter :: TRANSFORMER_HIDDEN = 8_int32
   integer(int32), parameter :: TRANSFORMER_VOCAB = 8_int32
   integer(int32), parameter :: TRANSFORMER_PARAMETERS = 112_int32
   integer(int32), parameter :: MF_ROWS = 8_int32
   integer(int32), parameter :: MF_COLUMNS = 10_int32
   integer(int32), parameter :: MF_RANK = 3_int32
   integer(int32), parameter :: MF_PARAMETERS = MF_ROWS * MF_RANK + MF_RANK * MF_COLUMNS

   type, public :: bench_task_t
      private
      integer(int32) :: kind = 0_int32
      integer(int64) :: seed = 1_int64
      real(real64) :: condition_number = 1.0_real64
      real(real64) :: noise_scale = 0.1_real64
      real(real64), allocatable :: diagonal(:)
      real(real64), allocatable :: features(:, :)
      real(real64), allocatable :: labels(:)
      real(real64), allocatable :: ratings(:, :)
      logical, allocatable :: rating_train_mask(:, :)
      logical, allocatable :: rating_validation_mask(:, :)
      logical, allocatable :: rating_test_mask(:, :)
   end type bench_task_t

   public :: bench_task_initialize, bench_task_evaluate, bench_task_curvature_diagonal, bench_task_name
   public :: bench_task_has_validation_split, bench_task_has_test_split

contains

   subroutine bench_task_initialize(task, kind, parameter_count, seed, status, condition_number, sample_count, noise_scale)
      type(bench_task_t), intent(out) :: task
      integer(int32), intent(in) :: kind
      integer(int64), intent(in) :: parameter_count, seed
      integer(int32), intent(out) :: status
      real(real64), optional, intent(in) :: condition_number, noise_scale
      integer(int32), optional, intent(in) :: sample_count
      integer(int64) :: rng
      integer :: i, sample, allocations, token, position
      integer(int32) :: samples
      real(real64) :: linear_position, target_probability, true_logit, max_score, score_sum
      real(real64) :: embedding(TRANSFORMER_WIDTH, TRANSFORMER_VOCAB), teacher_u(MF_ROWS, MF_RANK)
      real(real64) :: teacher_v(MF_RANK, MF_COLUMNS), scores(4), weights(4), context(TRANSFORMER_WIDTH)
      integer :: tokens(4)

      status = BENCH_OK
      if (parameter_count <= 0_int64 .or. seed < 0_int64 .or. &
          kind < TASK_QUADRATIC .or. kind > TASK_NOISY_QUADRATIC) then
         status = BENCH_INVALID
         return
      end if
      task%kind = kind
      task%seed = seed
      task%condition_number = 1.0_real64
      task%noise_scale = 0.1_real64
      if (present(condition_number)) task%condition_number = condition_number
      if (present(noise_scale)) task%noise_scale = noise_scale
      if (.not. ieee_is_finite(task%noise_scale) .or. task%noise_scale < 0.0_real64) then
         status = BENCH_INVALID
         return
      end if

      if (kind == TASK_QUADRATIC .or. kind == TASK_NOISY_QUADRATIC) then
         if (.not. ieee_is_finite(task%condition_number) .or. task%condition_number < 1.0_real64) then
            status = BENCH_INVALID
            return
         end if
         allocate(task%diagonal(parameter_count), stat=allocations)
         if (allocations /= 0) then
            status = BENCH_INVALID
            return
         end if
         do i = 1, size(task%diagonal)
            if (size(task%diagonal) == 1) then
               linear_position = 0.0_real64
            else
               linear_position = real(i - 1, real64) / real(size(task%diagonal) - 1, real64)
            end if
            task%diagonal(i) = task%condition_number**linear_position
         end do

      else if (kind == TASK_ROSENBROCK .or. kind == TASK_RASTRIGIN) then
         if (parameter_count < 2_int64) status = BENCH_INVALID

      else if (kind == TASK_LOGISTIC .or. kind == TASK_SPARSE_LOGISTIC) then
         samples = 512_int32
         if (present(sample_count)) samples = sample_count
         if (samples < 4_int32) then
            status = BENCH_INVALID
            return
         end if
         allocate(task%features(parameter_count, samples), task%labels(samples), stat=allocations)
         if (allocations /= 0) then
            status = BENCH_INVALID
            return
         end if
         rng = bench_seed_state(seed)
         block
            real(real64) :: true_weights(parameter_count)
            do i = 1, size(true_weights)
               true_weights(i) = 0.5_real64 * bench_normal(rng) / sqrt(real(size(true_weights), real64))
            end do
            do sample = 1, samples
               true_logit = 0.0_real64
               do i = 1, size(true_weights)
                  task%features(i, sample) = bench_normal(rng)
                  if (kind == TASK_SPARSE_LOGISTIC .and. bench_uniform(rng) < 0.8_real64) &
                     task%features(i, sample) = 0.0_real64
                  true_logit = true_logit + true_weights(i) * task%features(i, sample)
               end do
               target_probability = stable_sigmoid(true_logit)
               task%labels(sample) = 0.0_real64
               if (bench_uniform(rng) < target_probability) task%labels(sample) = 1.0_real64
            end do
         end block

      else if (kind == TASK_MLP) then
         if (parameter_count /= int(MLP_PARAMETERS, int64)) then
            status = BENCH_INVALID
            return
         end if
         samples = 128_int32
         if (present(sample_count)) samples = sample_count
         if (samples < 4_int32) then
            status = BENCH_INVALID
            return
         end if
         allocate(task%features(MLP_INPUTS, samples), task%labels(samples), stat=allocations)
         if (allocations /= 0) then
            status = BENCH_INVALID
            return
         end if
         rng = bench_seed_state(seed)
         do sample = 1, samples
            do i = 1, MLP_INPUTS
               task%features(i, sample) = bench_normal(rng)
            end do
            task%labels(sample) = tanh(task%features(1, sample) - 0.5_real64 * task%features(2, sample)) + &
                                  0.5_real64 * task%features(3, sample) * task%features(4, sample)
         end do

      else if (kind == TASK_TRANSFORMER_LM) then
         if (parameter_count /= int(TRANSFORMER_PARAMETERS, int64)) then
            status = BENCH_INVALID
            return
         end if
         samples = 128_int32
         if (present(sample_count)) samples = sample_count
         if (samples < 4_int32) then
            status = BENCH_INVALID
            return
         end if
         allocate(task%features(TRANSFORMER_WIDTH, samples), task%labels(samples), stat=allocations)
         if (allocations /= 0) then
            status = BENCH_INVALID
            return
         end if
         rng = bench_seed_state(seed)
         do token = 1, TRANSFORMER_VOCAB
            do i = 1, TRANSFORMER_WIDTH
               embedding(i, token) = bench_normal(rng) / sqrt(real(TRANSFORMER_WIDTH, real64))
            end do
         end do
         do sample = 1, samples
            do position = 1, 4
               tokens(position) = 1 + int(bench_uniform(rng) * real(TRANSFORMER_VOCAB, real64))
               tokens(position) = min(tokens(position), TRANSFORMER_VOCAB)
            end do
            do position = 1, 4
               scores(position) = dot_product(embedding(:, tokens(4)), embedding(:, tokens(position))) / &
                                  sqrt(real(TRANSFORMER_WIDTH, real64))
            end do
            max_score = maxval(scores)
            weights = exp(scores - max_score)
            score_sum = sum(weights)
            weights = weights / score_sum
            context = 0.0_real64
            do position = 1, 4
               context = context + weights(position) * embedding(:, tokens(position))
            end do
            task%features(:, sample) = context
            task%labels(sample) = real(mod(tokens(4) + tokens(2), TRANSFORMER_VOCAB) + 1, real64)
         end do

      else if (kind == TASK_MATRIX_FACTORIZATION) then
         if (parameter_count /= int(MF_PARAMETERS, int64)) then
            status = BENCH_INVALID
            return
         end if
         allocate(task%ratings(MF_ROWS, MF_COLUMNS), task%rating_train_mask(MF_ROWS, MF_COLUMNS), &
                  task%rating_validation_mask(MF_ROWS, MF_COLUMNS), task%rating_test_mask(MF_ROWS, MF_COLUMNS), &
                  stat=allocations)
         if (allocations /= 0) then
            status = BENCH_INVALID
            return
         end if
         rng = bench_seed_state(seed)
         do sample = 1, MF_RANK
            do i = 1, MF_ROWS
               teacher_u(i, sample) = 0.4_real64 * bench_normal(rng)
            end do
         end do
         do sample = 1, MF_COLUMNS
            do i = 1, MF_RANK
               teacher_v(i, sample) = 0.4_real64 * bench_normal(rng)
            end do
         end do
         task%ratings = matmul(teacher_u, teacher_v)
         do sample = 1, MF_COLUMNS
            do i = 1, MF_ROWS
               position = modulo((sample - 1) + 3 * i + int(modulo(seed, 10_int64)), 10)
               task%rating_train_mask(i, sample) = position < 7
               if (modulo(i, 2) == 0) then
                  task%rating_validation_mask(i, sample) = position == 7
                  task%rating_test_mask(i, sample) = position >= 8
               else
                  task%rating_validation_mask(i, sample) = position >= 7 .and. position <= 8
                  task%rating_test_mask(i, sample) = position == 9
               end if
            end do
         end do
      end if
   end subroutine bench_task_initialize

   subroutine bench_task_evaluate(task, parameters, gradient, loss, status, step, partition)
      type(bench_task_t), intent(in) :: task
      real(real64), contiguous, intent(in) :: parameters(:)
      real(real64), contiguous, intent(out) :: gradient(:)
      real(real64), intent(out) :: loss
      integer(int32), intent(out) :: status
      integer(int64), optional, intent(in) :: step
      integer(int32), optional, intent(in) :: partition
      real(real64) :: residual, z, probability, pair_error, two_pi, noise
      integer(int64) :: eval_step, rng
      integer(int32) :: active_partition
      integer :: i, sample, first_sample, last_sample

      status = BENCH_OK
      loss = 0.0_real64
      if (size(parameters) == 0 .or. size(parameters) /= size(gradient) .or. task%kind == 0_int32) then
         status = BENCH_INVALID
         return
      end if
      active_partition = TASK_PARTITION_FULL
      if (present(partition)) active_partition = partition
      if (active_partition < TASK_PARTITION_FULL .or. active_partition > TASK_PARTITION_TEST) then
         status = BENCH_INVALID
         return
      end if
      if (.not. all(ieee_is_finite(parameters))) then
         status = BENCH_NUMERICAL
         return
      end if
      gradient = 0.0_real64
      select case (task%kind)
      case (TASK_QUADRATIC, TASK_NOISY_QUADRATIC)
         if (.not. allocated(task%diagonal)) then
            status = BENCH_INVALID
            return
         end if
         if (size(parameters) /= size(task%diagonal)) then
            status = BENCH_INVALID
            return
         end if
         eval_step = 0_int64
         if (present(step)) then
            if (step < 0_int64) then
               status = BENCH_INVALID
               return
            end if
            eval_step = step
         end if
         if (task%kind == TASK_NOISY_QUADRATIC) rng = bench_seed_state(ieor(task%seed, eval_step))
         do i = 1, size(parameters)
            loss = loss + 0.5_real64 * task%diagonal(i) * parameters(i) * parameters(i)
            gradient(i) = task%diagonal(i) * parameters(i)
            if (task%kind == TASK_NOISY_QUADRATIC .and. task%noise_scale > 0.0_real64 .and. &
                active_partition /= TASK_PARTITION_VALIDATION .and. active_partition /= TASK_PARTITION_TEST) then
               noise = task%noise_scale * bench_normal(rng)
               gradient(i) = gradient(i) + noise
            end if
         end do

      case (TASK_ROSENBROCK)
         if (size(parameters) < 2) then
            status = BENCH_INVALID
            return
         end if
         do i = 1, size(parameters) - 1
            pair_error = parameters(i + 1) - parameters(i) * parameters(i)
            residual = 1.0_real64 - parameters(i)
            loss = loss + 100.0_real64 * pair_error * pair_error + residual * residual
            gradient(i) = gradient(i) - 400.0_real64 * parameters(i) * pair_error - 2.0_real64 * residual
            gradient(i + 1) = gradient(i + 1) + 200.0_real64 * pair_error
         end do

      case (TASK_RASTRIGIN)
         two_pi = 2.0_real64 * acos(-1.0_real64)
         do i = 1, size(parameters)
            loss = loss + parameters(i) * parameters(i) - 10.0_real64 * cos(two_pi * parameters(i)) + 10.0_real64
            gradient(i) = 2.0_real64 * parameters(i) + 20.0_real64 * acos(-1.0_real64) * &
                          sin(two_pi * parameters(i))
         end do

      case (TASK_LOGISTIC, TASK_SPARSE_LOGISTIC)
         if (.not. allocated(task%features) .or. .not. allocated(task%labels)) then
            status = BENCH_INVALID
            return
         end if
         if (size(parameters) /= size(task%features, 1) .or. size(task%labels) /= size(task%features, 2)) then
            status = BENCH_INVALID
            return
         end if
         call sample_partition_bounds(size(task%labels), active_partition, first_sample, last_sample, status)
         if (status /= BENCH_OK) return
         do sample = first_sample, last_sample
            z = dot_product(parameters, task%features(:, sample))
            probability = stable_sigmoid(z)
            loss = loss + max(z, 0.0_real64) - task%labels(sample) * z + log(1.0_real64 + exp(-abs(z)))
            residual = probability - task%labels(sample)
            gradient = gradient + residual * task%features(:, sample)
         end do
         loss = loss / real(last_sample - first_sample + 1, real64)
         gradient = gradient / real(last_sample - first_sample + 1, real64)

      case (TASK_MLP)
         if (size(parameters) /= MLP_PARAMETERS .or. .not. allocated(task%features) .or. &
             .not. allocated(task%labels)) then
            status = BENCH_INVALID
            return
         end if
         call evaluate_mlp(task, parameters, gradient, loss, status, active_partition)

      case (TASK_TRANSFORMER_LM)
         if (size(parameters) /= TRANSFORMER_PARAMETERS .or. .not. allocated(task%features) .or. &
             .not. allocated(task%labels)) then
            status = BENCH_INVALID
            return
         end if
         call evaluate_transformer_lm(task, parameters, gradient, loss, status, active_partition)

      case (TASK_MATRIX_FACTORIZATION)
         if (size(parameters) /= MF_PARAMETERS .or. .not. allocated(task%ratings)) then
            status = BENCH_INVALID
            return
         end if
         call evaluate_matrix_factorization(task, parameters, gradient, loss, status, active_partition)

      case default
         status = BENCH_INVALID
         return
      end select
      if (.not. ieee_is_finite(loss) .or. .not. all(ieee_is_finite(gradient))) status = BENCH_NUMERICAL
   end subroutine bench_task_evaluate

   subroutine evaluate_mlp(task, parameters, gradient, loss, status, partition)
      type(bench_task_t), intent(in) :: task
      real(real64), intent(in) :: parameters(:)
      real(real64), intent(out) :: gradient(:)
      real(real64), intent(out) :: loss
      integer(int32), intent(out) :: status
      integer(int32), intent(in) :: partition
      real(real64) :: w1(MLP_HIDDEN1, MLP_INPUTS), b1(MLP_HIDDEN1)
      real(real64) :: w2(MLP_HIDDEN2, MLP_HIDDEN1), b2(MLP_HIDDEN2)
      real(real64) :: w3(MLP_HIDDEN2), b3, a1(MLP_HIDDEN1), a2(MLP_HIDDEN2)
      real(real64) :: d1(MLP_HIDDEN1), d2(MLP_HIDDEN2), d3, residual, prediction
      real(real64) :: gw1(MLP_HIDDEN1, MLP_INPUTS), gb1(MLP_HIDDEN1)
      real(real64) :: gw2(MLP_HIDDEN2, MLP_HIDDEN1), gb2(MLP_HIDDEN2), gw3(MLP_HIDDEN2), gb3
      real(real64) :: scale
      integer :: s, cursor, first_sample, last_sample

      status = BENCH_OK
      call sample_partition_bounds(size(task%labels), partition, first_sample, last_sample, status)
      if (status /= BENCH_OK) return
      cursor = 1
      w1 = reshape(parameters(cursor:cursor + MLP_HIDDEN1 * MLP_INPUTS - 1), shape(w1))
      cursor = cursor + MLP_HIDDEN1 * MLP_INPUTS
      b1 = parameters(cursor:cursor + MLP_HIDDEN1 - 1)
      cursor = cursor + MLP_HIDDEN1
      w2 = reshape(parameters(cursor:cursor + MLP_HIDDEN2 * MLP_HIDDEN1 - 1), shape(w2))
      cursor = cursor + MLP_HIDDEN2 * MLP_HIDDEN1
      b2 = parameters(cursor:cursor + MLP_HIDDEN2 - 1)
      cursor = cursor + MLP_HIDDEN2
      w3 = parameters(cursor:cursor + MLP_HIDDEN2 - 1)
      b3 = parameters(cursor + MLP_HIDDEN2)
      gradient = 0.0_real64
      loss = 0.0_real64
      gw1 = 0.0_real64
      gb1 = 0.0_real64
      gw2 = 0.0_real64
      gb2 = 0.0_real64
      gw3 = 0.0_real64
      gb3 = 0.0_real64
      do s = first_sample, last_sample
         a1 = tanh(matmul(w1, task%features(:, s)) + b1)
         a2 = tanh(matmul(w2, a1) + b2)
         prediction = dot_product(w3, a2) + b3
         residual = prediction - task%labels(s)
         loss = loss + 0.5_real64 * residual * residual
         d3 = residual
         gw3 = gw3 + d3 * a2
         gb3 = gb3 + d3
         d2 = (d3 * w3) * (1.0_real64 - a2 * a2)
         gw2 = gw2 + spread(d2, 2, MLP_HIDDEN1) * spread(a1, 1, MLP_HIDDEN2)
         gb2 = gb2 + d2
         d1 = matmul(transpose(w2), d2) * (1.0_real64 - a1 * a1)
         gw1 = gw1 + spread(d1, 2, MLP_INPUTS) * spread(task%features(:, s), 1, MLP_HIDDEN1)
         gb1 = gb1 + d1
      end do
      scale = 1.0_real64 / real(last_sample - first_sample + 1, real64)
      loss = loss * scale
      cursor = 1
      gradient(cursor:cursor + MLP_HIDDEN1 * MLP_INPUTS - 1) = reshape(gw1 * scale, [MLP_HIDDEN1 * MLP_INPUTS])
      cursor = cursor + MLP_HIDDEN1 * MLP_INPUTS
      gradient(cursor:cursor + MLP_HIDDEN1 - 1) = gb1 * scale
      cursor = cursor + MLP_HIDDEN1
      gradient(cursor:cursor + MLP_HIDDEN2 * MLP_HIDDEN1 - 1) = reshape(gw2 * scale, [MLP_HIDDEN2 * MLP_HIDDEN1])
      cursor = cursor + MLP_HIDDEN2 * MLP_HIDDEN1
      gradient(cursor:cursor + MLP_HIDDEN2 - 1) = gb2 * scale
      cursor = cursor + MLP_HIDDEN2
      gradient(cursor:cursor + MLP_HIDDEN2 - 1) = gw3 * scale
      gradient(cursor + MLP_HIDDEN2) = gb3 * scale
   end subroutine evaluate_mlp

   subroutine evaluate_transformer_lm(task, parameters, gradient, loss, status, partition)
      type(bench_task_t), intent(in) :: task
      real(real64), intent(in) :: parameters(:)
      real(real64), intent(out) :: gradient(:)
      real(real64), intent(out) :: loss
      integer(int32), intent(out) :: status
      integer(int32), intent(in) :: partition
      real(real64) :: w1(TRANSFORMER_HIDDEN, TRANSFORMER_WIDTH), b1(TRANSFORMER_HIDDEN)
      real(real64) :: w2(TRANSFORMER_VOCAB, TRANSFORMER_HIDDEN), b2(TRANSFORMER_VOCAB)
      real(real64) :: hidden(TRANSFORMER_HIDDEN), logits(TRANSFORMER_VOCAB), probs(TRANSFORMER_VOCAB)
      real(real64) :: dlogits(TRANSFORMER_VOCAB), dhidden(TRANSFORMER_HIDDEN), max_logit
      real(real64) :: gw1(TRANSFORMER_HIDDEN, TRANSFORMER_WIDTH), gb1(TRANSFORMER_HIDDEN)
      real(real64) :: gw2(TRANSFORMER_VOCAB, TRANSFORMER_HIDDEN), gb2(TRANSFORMER_VOCAB), scale
      integer :: cursor, s, target, first_sample, last_sample

      status = BENCH_OK
      call sample_partition_bounds(size(task%labels), partition, first_sample, last_sample, status)
      if (status /= BENCH_OK) return
      cursor = 1
      w1 = reshape(parameters(cursor:cursor + TRANSFORMER_HIDDEN * TRANSFORMER_WIDTH - 1), shape(w1))
      cursor = cursor + TRANSFORMER_HIDDEN * TRANSFORMER_WIDTH
      b1 = parameters(cursor:cursor + TRANSFORMER_HIDDEN - 1)
      cursor = cursor + TRANSFORMER_HIDDEN
      w2 = reshape(parameters(cursor:cursor + TRANSFORMER_VOCAB * TRANSFORMER_HIDDEN - 1), shape(w2))
      cursor = cursor + TRANSFORMER_VOCAB * TRANSFORMER_HIDDEN
      b2 = parameters(cursor:cursor + TRANSFORMER_VOCAB - 1)
      gradient = 0.0_real64
      loss = 0.0_real64
      gw1 = 0.0_real64
      gb1 = 0.0_real64
      gw2 = 0.0_real64
      gb2 = 0.0_real64
      do s = first_sample, last_sample
         hidden = tanh(matmul(w1, task%features(:, s)) + b1)
         logits = matmul(w2, hidden) + b2
         max_logit = maxval(logits)
         probs = exp(logits - max_logit)
         probs = probs / sum(probs)
         target = int(task%labels(s))
         loss = loss - log(max(probs(target), tiny(1.0_real64)))
         dlogits = probs
         dlogits(target) = dlogits(target) - 1.0_real64
         gw2 = gw2 + spread(dlogits, 2, TRANSFORMER_HIDDEN) * spread(hidden, 1, TRANSFORMER_VOCAB)
         gb2 = gb2 + dlogits
         dhidden = matmul(transpose(w2), dlogits) * (1.0_real64 - hidden * hidden)
         gw1 = gw1 + spread(dhidden, 2, TRANSFORMER_WIDTH) * spread(task%features(:, s), 1, TRANSFORMER_HIDDEN)
         gb1 = gb1 + dhidden
      end do
      scale = 1.0_real64 / real(last_sample - first_sample + 1, real64)
      loss = loss * scale
      cursor = 1
      gradient(cursor:cursor + TRANSFORMER_HIDDEN * TRANSFORMER_WIDTH - 1) = &
         reshape(gw1 * scale, [TRANSFORMER_HIDDEN * TRANSFORMER_WIDTH])
      cursor = cursor + TRANSFORMER_HIDDEN * TRANSFORMER_WIDTH
      gradient(cursor:cursor + TRANSFORMER_HIDDEN - 1) = gb1 * scale
      cursor = cursor + TRANSFORMER_HIDDEN
      gradient(cursor:cursor + TRANSFORMER_VOCAB * TRANSFORMER_HIDDEN - 1) = &
         reshape(gw2 * scale, [TRANSFORMER_VOCAB * TRANSFORMER_HIDDEN])
      cursor = cursor + TRANSFORMER_VOCAB * TRANSFORMER_HIDDEN
      gradient(cursor:cursor + TRANSFORMER_VOCAB - 1) = gb2 * scale
   end subroutine evaluate_transformer_lm

   subroutine evaluate_matrix_factorization(task, parameters, gradient, loss, status, partition)
      type(bench_task_t), intent(in) :: task
      real(real64), intent(in) :: parameters(:)
      real(real64), intent(out) :: gradient(:)
      real(real64), intent(out) :: loss
      integer(int32), intent(out) :: status
      integer(int32), intent(in) :: partition
      real(real64) :: user_factors(MF_ROWS, MF_RANK), item_factors(MF_RANK, MF_COLUMNS)
      real(real64) :: residuals(MF_ROWS, MF_COLUMNS), grad_u(MF_ROWS, MF_RANK), grad_v(MF_RANK, MF_COLUMNS)
      logical :: selected(MF_ROWS, MF_COLUMNS)
      integer :: cursor, observed_count

      status = BENCH_OK
      cursor = MF_ROWS * MF_RANK
      user_factors = reshape(parameters(1:cursor), shape(user_factors))
      item_factors = reshape(parameters(cursor + 1:MF_PARAMETERS), shape(item_factors))
      select case (partition)
      case (TASK_PARTITION_FULL)
         selected = .true.
      case (TASK_PARTITION_TRAIN)
         if (.not. allocated(task%rating_train_mask)) then
            status = BENCH_INVALID
            return
         end if
         selected = task%rating_train_mask
      case (TASK_PARTITION_VALIDATION)
         if (.not. allocated(task%rating_validation_mask)) then
            status = BENCH_INVALID
            return
         end if
         selected = task%rating_validation_mask
      case (TASK_PARTITION_TEST)
         if (.not. allocated(task%rating_test_mask)) then
            status = BENCH_INVALID
            return
         end if
         selected = task%rating_test_mask
      case default
         status = BENCH_INVALID
         return
      end select
      observed_count = count(selected)
      if (observed_count <= 0) then
         status = BENCH_INVALID
         return
      end if
      residuals = 0.0_real64
      where (selected) residuals = matmul(user_factors, item_factors) - task%ratings
      loss = 0.5_real64 * sum(residuals * residuals) / real(observed_count, real64)
      grad_u = matmul(residuals, transpose(item_factors)) / real(observed_count, real64)
      grad_v = matmul(transpose(user_factors), residuals) / real(observed_count, real64)
      gradient(1:cursor) = reshape(grad_u, [cursor])
      gradient(cursor + 1:MF_PARAMETERS) = reshape(grad_v, [MF_RANK * MF_COLUMNS])
   end subroutine evaluate_matrix_factorization

   subroutine bench_task_curvature_diagonal(task, diagonal, status)
      type(bench_task_t), intent(in) :: task
      real(real64), allocatable, intent(out) :: diagonal(:)
      integer(int32), intent(out) :: status
      integer :: allocation_status
      status = BENCH_INVALID
      if (task%kind /= TASK_QUADRATIC .and. task%kind /= TASK_NOISY_QUADRATIC) return
      if (.not. allocated(task%diagonal)) return
      allocate(diagonal(size(task%diagonal)), stat=allocation_status)
      if (allocation_status /= 0) then
         status = BENCH_INVALID
         return
      end if
      diagonal = task%diagonal
      status = BENCH_OK
   end subroutine bench_task_curvature_diagonal

   subroutine sample_partition_bounds(sample_count, partition, first_sample, last_sample, status)
      integer, intent(in) :: sample_count
      integer(int32), intent(in) :: partition
      integer, intent(out) :: first_sample, last_sample
      integer(int32), intent(out) :: status
      integer :: training_count, validation_count
      status = BENCH_OK
      first_sample = 1
      last_sample = sample_count
      if (sample_count < 4 .or. partition < TASK_PARTITION_FULL .or. partition > TASK_PARTITION_TEST) then
         status = BENCH_INVALID
         return
      end if
      training_count = max(1, (7 * sample_count) / 10)
      validation_count = max(1, (15 * sample_count) / 100)
      select case (partition)
      case (TASK_PARTITION_FULL)
         return
      case (TASK_PARTITION_TRAIN)
         last_sample = training_count
      case (TASK_PARTITION_VALIDATION)
         first_sample = training_count + 1
         last_sample = min(sample_count, training_count + validation_count)
      case (TASK_PARTITION_TEST)
         first_sample = training_count + validation_count + 1
      end select
      if (first_sample > last_sample) status = BENCH_INVALID
   end subroutine sample_partition_bounds

   pure logical function bench_task_has_validation_split(task)
      type(bench_task_t), intent(in) :: task
      select case (task%kind)
      case (TASK_LOGISTIC, TASK_SPARSE_LOGISTIC, TASK_MLP, TASK_TRANSFORMER_LM, TASK_MATRIX_FACTORIZATION)
         bench_task_has_validation_split = .true.
      case default
         bench_task_has_validation_split = .false.
      end select
   end function bench_task_has_validation_split

   pure logical function bench_task_has_test_split(task)
      type(bench_task_t), intent(in) :: task
      bench_task_has_test_split = bench_task_has_validation_split(task)
   end function bench_task_has_test_split

   pure real(real64) function stable_sigmoid(value)
      real(real64), intent(in) :: value
      if (value >= 0.0_real64) then
         stable_sigmoid = 1.0_real64 / (1.0_real64 + exp(-value))
      else
         stable_sigmoid = exp(value) / (1.0_real64 + exp(value))
      end if
   end function stable_sigmoid

   pure function bench_task_name(kind) result(name)
      integer(int32), intent(in) :: kind
      character(len=24) :: name
      select case (kind)
      case (TASK_QUADRATIC); name = 'quadratic'
      case (TASK_ROSENBROCK); name = 'Rosenbrock'
      case (TASK_RASTRIGIN); name = 'Rastrigin'
      case (TASK_LOGISTIC); name = 'logistic'
      case (TASK_SPARSE_LOGISTIC); name = 'sparse-logistic'
      case (TASK_MLP); name = 'synthetic-MLP-2-hidden'
      case (TASK_TRANSFORMER_LM); name = 'synthetic-transformer-LM'
      case (TASK_MATRIX_FACTORIZATION); name = 'low-rank-matrix-factorization'
      case (TASK_NOISY_QUADRATIC); name = 'noisy-quadratic'
      case default; name = 'unknown'
      end select
   end function bench_task_name

end module bench_tasks
