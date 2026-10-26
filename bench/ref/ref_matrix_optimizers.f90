! Matrix-shaped benchmark references, isolated from production Apex.
module ref_matrix_optimizers
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   implicit none
   private

   integer(int32), parameter, public :: REF_SHAMPOO = 1_int32
   integer(int32), parameter, public :: REF_MUON = 2_int32
   integer(int32), parameter, public :: REF_ADAFACTOR_MATRIX = 3_int32
   integer(int32), parameter, public :: REF_MATRIX_OK = 0_int32
   integer(int32), parameter, public :: REF_MATRIX_INVALID = 1_int32
   integer(int32), parameter, public :: REF_MATRIX_ALLOC = 2_int32
   integer(int32), parameter, public :: REF_MATRIX_NUMERICAL = 3_int32

   type, public :: ref_matrix_state_t
      private
      integer(int32) :: kind = 0_int32
      integer(int64) :: step = 0_int64
      real(real64), allocatable :: momentum(:, :)
      real(real64), allocatable :: left_covariance(:, :)
      real(real64), allocatable :: right_covariance(:, :)
      real(real64), allocatable :: row_statistic(:)
      real(real64), allocatable :: column_statistic(:)
   end type ref_matrix_state_t

   public :: ref_matrix_initialize, ref_matrix_step, ref_matrix_state_bytes, ref_matrix_name

contains

   subroutine ref_matrix_initialize(kind, rows, columns, state, status)
      integer(int32), intent(in) :: kind
      integer(int64), intent(in) :: rows, columns
      type(ref_matrix_state_t), intent(out) :: state
      integer(int32), intent(out) :: status
      integer :: allocation_status

      status = REF_MATRIX_OK
      if ((kind /= REF_SHAMPOO .and. kind /= REF_MUON .and. kind /= REF_ADAFACTOR_MATRIX) .or. &
          rows <= 0_int64 .or. columns <= 0_int64) then
         status = REF_MATRIX_INVALID
         return
      end if
      state%kind = kind
      state%step = 0_int64
      if (kind == REF_ADAFACTOR_MATRIX) then
         allocate(state%row_statistic(rows), state%column_statistic(columns), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_MATRIX_ALLOC
            return
         end if
         state%row_statistic = 0.0_real64
         state%column_statistic = 0.0_real64
      else
         allocate(state%momentum(rows, columns), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_MATRIX_ALLOC
            return
         end if
         state%momentum = 0.0_real64
         if (kind == REF_SHAMPOO) then
            allocate(state%left_covariance(rows, rows), state%right_covariance(columns, columns), &
                     stat=allocation_status)
            if (allocation_status /= 0) then
               status = REF_MATRIX_ALLOC
               return
            end if
            state%left_covariance = 0.0_real64
            state%right_covariance = 0.0_real64
         end if
      end if
   end subroutine ref_matrix_initialize

   subroutine ref_matrix_step(parameters, gradient, state, learning_rate, beta1, epsilon, status)
      real(real64), contiguous, intent(inout) :: parameters(:, :)
      real(real64), contiguous, intent(in) :: gradient(:, :)
      type(ref_matrix_state_t), intent(inout) :: state
      real(real64), intent(in) :: learning_rate, beta1, epsilon
      integer(int32), intent(out) :: status
      real(real64), allocatable :: left_root(:, :), right_root(:, :)
      real(real64), allocatable :: preconditioned(:, :), orthogonalized(:, :)
      real(real64) :: beta2_t, row_average, column_average, mean_row, estimate, update_rms, clip_scale
      integer :: allocation_status, i, j

      status = REF_MATRIX_OK
      if (any(shape(parameters) /= shape(gradient)) .or. size(parameters) == 0 .or. &
          state%kind == 0_int32 .or. .not. ieee_is_finite(learning_rate) .or. learning_rate <= 0.0_real64 .or. &
          .not. ieee_is_finite(beta1) .or. beta1 < 0.0_real64 .or. beta1 >= 1.0_real64 .or. &
          .not. ieee_is_finite(epsilon) .or. epsilon <= 0.0_real64) then
         status = REF_MATRIX_INVALID
         return
      end if
      if (.not. all(ieee_is_finite(parameters)) .or. .not. all(ieee_is_finite(gradient))) then
         status = REF_MATRIX_NUMERICAL
         return
      end if
      if (state%kind == REF_ADAFACTOR_MATRIX) then
         if (.not. allocated(state%row_statistic) .or. .not. allocated(state%column_statistic)) then
            status = REF_MATRIX_INVALID
            return
         end if
         if (size(state%row_statistic) /= size(parameters, 1) .or. &
             size(state%column_statistic) /= size(parameters, 2)) then
            status = REF_MATRIX_INVALID
            return
         end if
      else
         if (.not. allocated(state%momentum)) then
            status = REF_MATRIX_INVALID
            return
         end if
         if (any(shape(state%momentum) /= shape(parameters))) then
            status = REF_MATRIX_INVALID
            return
         end if
      end if
      if (state%step == huge(state%step)) then
         status = REF_MATRIX_INVALID
         return
      end if
      state%step = state%step + 1_int64

      select case (state%kind)
      case (REF_ADAFACTOR_MATRIX)
         ! Shazeer & Stern (2018), arXiv:1804.04235: factored row/column
         ! second moments, relative-step disabled and update-RMS clipping at 1.
         beta2_t = 1.0_real64 - real(state%step, real64)**(-0.8_real64)
         do i = 1, size(parameters, 1)
            row_average = sum(gradient(i, :) * gradient(i, :)) / real(size(parameters, 2), real64)
            state%row_statistic(i) = beta2_t * state%row_statistic(i) + (1.0_real64 - beta2_t) * row_average
         end do
         do j = 1, size(parameters, 2)
            column_average = sum(gradient(:, j) * gradient(:, j)) / real(size(parameters, 1), real64)
            state%column_statistic(j) = beta2_t * state%column_statistic(j) + &
                                        (1.0_real64 - beta2_t) * column_average
         end do
         mean_row = sum(state%row_statistic) / real(size(state%row_statistic), real64)
         allocate(preconditioned(size(parameters, 1), size(parameters, 2)), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_MATRIX_ALLOC
            return
         end if
         update_rms = 0.0_real64
         do j = 1, size(parameters, 2)
            do i = 1, size(parameters, 1)
               estimate = state%row_statistic(i) * state%column_statistic(j) / max(mean_row, epsilon)
               preconditioned(i, j) = gradient(i, j) / (sqrt(estimate) + epsilon)
               update_rms = update_rms + preconditioned(i, j) * preconditioned(i, j)
            end do
         end do
         update_rms = sqrt(update_rms / real(size(parameters), real64))
         clip_scale = min(1.0_real64, 1.0_real64 / max(update_rms, tiny(1.0_real64)))
         parameters = parameters - learning_rate * clip_scale * preconditioned

      case (REF_SHAMPOO)
         ! Gupta et al., "Shampoo: Preconditioned Stochastic Tensor Optimization",
         ! ICML 2018; rank-2 Kronecker factors, cumulative covariance and p=4 roots.
         if (.not. allocated(state%left_covariance) .or. .not. allocated(state%right_covariance)) then
            status = REF_MATRIX_INVALID
            return
         end if
         if (size(state%left_covariance, 1) /= size(parameters, 1) .or. &
             size(state%right_covariance, 1) /= size(parameters, 2)) then
            status = REF_MATRIX_INVALID
            return
         end if
         state%left_covariance = state%left_covariance + matmul(gradient, transpose(gradient))
         state%right_covariance = state%right_covariance + matmul(transpose(gradient), gradient)
         allocate(left_root(size(parameters, 1), size(parameters, 1)), &
                  right_root(size(parameters, 2), size(parameters, 2)), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_MATRIX_ALLOC
            return
         end if
         call symmetric_inverse_fourth_root(state%left_covariance, epsilon, left_root, status)
         if (status /= REF_MATRIX_OK) return
         call symmetric_inverse_fourth_root(state%right_covariance, epsilon, right_root, status)
         if (status /= REF_MATRIX_OK) return
         allocate(preconditioned(size(parameters, 1), size(parameters, 2)), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_MATRIX_ALLOC
            return
         end if
         preconditioned = matmul(matmul(left_root, gradient), right_root)
         state%momentum = beta1 * state%momentum + preconditioned
         parameters = parameters - learning_rate * state%momentum

      case (REF_MUON)
         ! Muon (Keller Jordan's public reference implementation, 2024):
         ! Nesterov matrix momentum and five Newton–Schulz polar iterations
         ! using the published implementation polynomial coefficients.
         state%momentum = beta1 * state%momentum + (1.0_real64 - beta1) * gradient
         allocate(preconditioned(size(parameters, 1), size(parameters, 2)), &
                  orthogonalized(size(parameters, 1), size(parameters, 2)), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_MATRIX_ALLOC
            return
         end if
         preconditioned = beta1 * state%momentum + (1.0_real64 - beta1) * gradient
         call muon_orthogonalize(preconditioned, orthogonalized, status)
         if (status /= REF_MATRIX_OK) return
         parameters = parameters - learning_rate * orthogonalized
      end select

      if (.not. all(ieee_is_finite(parameters)) .or. .not. all(ieee_is_finite(state%momentum))) &
         status = REF_MATRIX_NUMERICAL
      if (allocated(state%left_covariance)) then
         if (.not. all(ieee_is_finite(state%left_covariance))) status = REF_MATRIX_NUMERICAL
      end if
      if (allocated(state%right_covariance)) then
         if (.not. all(ieee_is_finite(state%right_covariance))) status = REF_MATRIX_NUMERICAL
      end if
      if (allocated(state%row_statistic)) then
         if (.not. all(ieee_is_finite(state%row_statistic))) status = REF_MATRIX_NUMERICAL
      end if
      if (allocated(state%column_statistic)) then
         if (.not. all(ieee_is_finite(state%column_statistic))) status = REF_MATRIX_NUMERICAL
      end if
   end subroutine ref_matrix_step

   subroutine symmetric_inverse_fourth_root(matrix, epsilon, root, status)
      real(real64), contiguous, intent(in) :: matrix(:, :)
      real(real64), intent(in) :: epsilon
      real(real64), contiguous, intent(out) :: root(:, :)
      integer(int32), intent(out) :: status
      real(real64) :: diagonalized(size(matrix, 1), size(matrix, 2))
      real(real64) :: eigenvectors(size(matrix, 1), size(matrix, 2))
      real(real64) :: app, aqq, apq, tau, tangent, cosine, sine, value_p, value_q, off_diagonal
      real(real64) :: temp_p, temp_q
      integer :: n, p, q, k, sweep

      status = REF_MATRIX_OK
      n = size(matrix, 1)
      if (size(matrix, 2) /= n .or. size(root, 1) /= n .or. size(root, 2) /= n .or. &
          .not. ieee_is_finite(epsilon) .or. epsilon <= 0.0_real64) then
         status = REF_MATRIX_INVALID
         return
      end if
      diagonalized = matrix
      eigenvectors = 0.0_real64
      do k = 1, n
         diagonalized(k, k) = diagonalized(k, k) + epsilon
         eigenvectors(k, k) = 1.0_real64
      end do

      do sweep = 1, max(12, 8 * n)
         off_diagonal = 0.0_real64
         do p = 1, n - 1
            do q = p + 1, n
               off_diagonal = max(off_diagonal, abs(diagonalized(p, q)))
               apq = diagonalized(p, q)
               if (abs(apq) <= epsilon * 1.0e-6_real64) cycle
               app = diagonalized(p, p)
               aqq = diagonalized(q, q)
               tau = (aqq - app) / (2.0_real64 * apq)
               tangent = sign(1.0_real64, tau) / (abs(tau) + sqrt(1.0_real64 + tau * tau))
               cosine = 1.0_real64 / sqrt(1.0_real64 + tangent * tangent)
               sine = tangent * cosine
               do k = 1, n
                  if (k == p .or. k == q) cycle
                  temp_p = diagonalized(k, p)
                  temp_q = diagonalized(k, q)
                  diagonalized(k, p) = cosine * temp_p - sine * temp_q
                  diagonalized(p, k) = diagonalized(k, p)
                  diagonalized(k, q) = sine * temp_p + cosine * temp_q
                  diagonalized(q, k) = diagonalized(k, q)
                  temp_p = eigenvectors(k, p)
                  temp_q = eigenvectors(k, q)
                  eigenvectors(k, p) = cosine * temp_p - sine * temp_q
                  eigenvectors(k, q) = sine * temp_p + cosine * temp_q
               end do
               diagonalized(p, p) = app - tangent * apq
               diagonalized(q, q) = aqq + tangent * apq
               diagonalized(p, q) = 0.0_real64
               diagonalized(q, p) = 0.0_real64
            end do
         end do
         if (off_diagonal <= epsilon * 1.0e-8_real64) exit
      end do
      if (.not. all(ieee_is_finite(diagonalized)) .or. .not. all(ieee_is_finite(eigenvectors))) then
         status = REF_MATRIX_NUMERICAL
         return
      end if
      root = 0.0_real64
      do k = 1, n
         value_p = max(diagonalized(k, k), epsilon)**(-0.25_real64)
         do p = 1, n
            do q = 1, n
               root(p, q) = root(p, q) + eigenvectors(p, k) * value_p * eigenvectors(q, k)
            end do
         end do
      end do
      if (.not. all(ieee_is_finite(root))) status = REF_MATRIX_NUMERICAL
   end subroutine symmetric_inverse_fourth_root

   subroutine muon_orthogonalize(matrix, result, status)
      real(real64), contiguous, intent(in) :: matrix(:, :)
      real(real64), contiguous, intent(out) :: result(:, :)
      integer(int32), intent(out) :: status
      real(real64), allocatable :: x(:, :), gram(:, :), gram_square(:, :), polynomial(:, :)
      real(real64) :: matrix_norm
      integer :: rows, columns, iteration, allocation_status, i
      logical :: transposed

      status = REF_MATRIX_OK
      rows = size(matrix, 1)
      columns = size(matrix, 2)
      if (size(result, 1) /= rows .or. size(result, 2) /= columns .or. &
          .not. all(ieee_is_finite(matrix))) then
         status = REF_MATRIX_INVALID
         return
      end if
      transposed = rows > columns
      if (transposed) then
         allocate(x(columns, rows), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_MATRIX_ALLOC
            return
         end if
         x = transpose(matrix)
      else
         allocate(x(rows, columns), stat=allocation_status)
         if (allocation_status /= 0) then
            status = REF_MATRIX_ALLOC
            return
         end if
         x = matrix
      end if
      matrix_norm = sqrt(sum(x * x))
      if (matrix_norm <= tiny(1.0_real64)) then
         result = 0.0_real64
         return
      end if
      x = x / matrix_norm
      allocate(gram(size(x, 1), size(x, 1)), gram_square(size(x, 1), size(x, 1)), &
               polynomial(size(x, 1), size(x, 1)), stat=allocation_status)
      if (allocation_status /= 0) then
         status = REF_MATRIX_ALLOC
         return
      end if
      do iteration = 1, 5
         gram = matmul(x, transpose(x))
         gram_square = matmul(gram, gram)
         polynomial = 2.0315_real64 * gram_square - 4.7750_real64 * gram
         do i = 1, size(x, 1)
            polynomial(i, i) = polynomial(i, i) + 3.4445_real64
         end do
         x = matmul(polynomial, x)
      end do
      if (transposed) then
         result = transpose(x)
      else
         result = x
      end if
      if (.not. all(ieee_is_finite(result))) status = REF_MATRIX_NUMERICAL
   end subroutine muon_orthogonalize

   pure integer(int64) function ref_matrix_state_bytes(state) result(byte_count)
      type(ref_matrix_state_t), intent(in) :: state
      byte_count = 0_int64
      if (allocated(state%momentum)) byte_count = byte_count + &
         int(size(state%momentum), int64) * int(storage_size(0.0_real64) / 8, int64)
      if (allocated(state%left_covariance)) byte_count = byte_count + &
         int(size(state%left_covariance), int64) * int(storage_size(0.0_real64) / 8, int64)
      if (allocated(state%right_covariance)) byte_count = byte_count + &
         int(size(state%right_covariance), int64) * int(storage_size(0.0_real64) / 8, int64)
      if (allocated(state%row_statistic)) byte_count = byte_count + &
         int(size(state%row_statistic), int64) * int(storage_size(0.0_real64) / 8, int64)
      if (allocated(state%column_statistic)) byte_count = byte_count + &
         int(size(state%column_statistic), int64) * int(storage_size(0.0_real64) / 8, int64)
   end function ref_matrix_state_bytes

   pure function ref_matrix_name(kind) result(name)
      integer(int32), intent(in) :: kind
      character(len=24) :: name
      select case (kind)
      case (REF_ADAFACTOR_MATRIX); name = 'Adafactor-matrix'
      case (REF_SHAMPOO); name = 'Shampoo'
      case (REF_MUON); name = 'Muon'
      case default; name = 'unknown'
      end select
   end function ref_matrix_name

end module ref_matrix_optimizers
