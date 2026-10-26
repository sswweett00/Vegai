! Fortran-only paired inference, bootstrap confidence intervals and Pareto utilities.
module bench_statistics
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   use bench_contracts, only : bench_seed_state, bench_uniform
   implicit none
   private

   integer(int32), parameter, public :: STATS_OK = 0_int32
   integer(int32), parameter, public :: STATS_INVALID = 1_int32

   public :: bench_paired_summary, bench_bootstrap_mean_ci, bench_wilcoxon_signed_rank
   public :: bench_holm_adjust, bench_pareto_front

contains

   subroutine bench_paired_summary(differences, mean_difference, sample_stddev, status)
      real(real64), intent(in) :: differences(:)
      real(real64), intent(out) :: mean_difference, sample_stddev
      integer(int32), intent(out) :: status
      integer :: n

      status = STATS_INVALID
      mean_difference = 0.0_real64
      sample_stddev = 0.0_real64
      n = size(differences)
      if (n < 2 .or. .not. all(ieee_is_finite(differences))) return
      mean_difference = sum(differences) / real(n, real64)
      sample_stddev = sqrt(sum((differences - mean_difference)**2) / real(n - 1, real64))
      status = STATS_OK
   end subroutine bench_paired_summary

   subroutine bench_bootstrap_mean_ci(differences, seed, replicates, alpha, lower, upper, status)
      real(real64), intent(in) :: differences(:)
      integer(int64), intent(in) :: seed
      integer(int32), intent(in) :: replicates
      real(real64), intent(in) :: alpha
      real(real64), intent(out) :: lower, upper
      integer(int32), intent(out) :: status
      real(real64), allocatable :: bootstrap_means(:)
      integer(int64) :: rng
      integer :: n, b, draw, index, lower_index, upper_index, allocation_status
      real(real64) :: sum_sample

      status = STATS_INVALID
      lower = 0.0_real64
      upper = 0.0_real64
      n = size(differences)
      if (n < 2 .or. seed < 0_int64 .or. replicates < 1000_int32 .or. &
          .not. ieee_is_finite(alpha) .or. alpha <= 0.0_real64 .or. alpha >= 1.0_real64 .or. &
          .not. all(ieee_is_finite(differences))) return
      allocate(bootstrap_means(replicates), stat=allocation_status)
      if (allocation_status /= 0) return
      rng = bench_seed_state(seed)
      do b = 1, replicates
         sum_sample = 0.0_real64
         do draw = 1, n
            index = 1 + int(bench_uniform(rng) * real(n, real64))
            index = min(n, index)
            sum_sample = sum_sample + differences(index)
         end do
         bootstrap_means(b) = sum_sample / real(n, real64)
      end do
      call sort_values(bootstrap_means)
      lower_index = max(1, int(floor((alpha / 2.0_real64) * real(replicates - 1, real64))) + 1)
      upper_index = min(replicates, int(ceiling((1.0_real64 - alpha / 2.0_real64) * &
                                                real(replicates - 1, real64))) + 1)
      lower = bootstrap_means(lower_index)
      upper = bootstrap_means(upper_index)
      status = STATS_OK
   end subroutine bench_bootstrap_mean_ci

   subroutine bench_wilcoxon_signed_rank(differences, statistic, p_value, effective_n, status)
      real(real64), intent(in) :: differences(:)
      real(real64), intent(out) :: statistic, p_value
      integer(int32), intent(out) :: effective_n, status
      real(real64), allocatable :: ranks(:), absolute_values(:)
      real(real64) :: total_rank, observed_distance, candidate_sum, mean_rank, variance, z, tie_count
      integer(int64) :: mask, combinations, extreme
      integer :: i, j, n, allocation_status

      status = STATS_INVALID
      statistic = 0.0_real64
      p_value = 1.0_real64
      effective_n = 0_int32
      if (size(differences) == 0 .or. .not. all(ieee_is_finite(differences))) return
      n = count(differences /= 0.0_real64)
      if (n == 0) then
         status = STATS_OK
         return
      end if
      allocate(ranks(n), absolute_values(n), stat=allocation_status)
      if (allocation_status /= 0) return
      j = 0
      do i = 1, size(differences)
         if (differences(i) == 0.0_real64) cycle
         j = j + 1
         absolute_values(j) = abs(differences(i))
      end do
      do i = 1, n
         tie_count = real(count(absolute_values == absolute_values(i)), real64)
         ranks(i) = 1.0_real64 + real(count(absolute_values < absolute_values(i)), real64) + &
                    0.5_real64 * (tie_count - 1.0_real64)
      end do
      statistic = 0.0_real64
      j = 0
      do i = 1, size(differences)
         if (differences(i) == 0.0_real64) cycle
         j = j + 1
         if (differences(i) > 0.0_real64) statistic = statistic + ranks(j)
      end do
      total_rank = sum(ranks)
      mean_rank = total_rank / 2.0_real64
      observed_distance = abs(statistic - mean_rank)
      effective_n = int(n, int32)
      if (n <= 20) then
         combinations = ishft(1_int64, n)
         extreme = 0_int64
         do mask = 0_int64, combinations - 1_int64
            candidate_sum = 0.0_real64
            do i = 1, n
               if (btest(mask, i - 1)) candidate_sum = candidate_sum + ranks(i)
            end do
            if (abs(candidate_sum - mean_rank) >= observed_distance - 1.0e-12_real64) &
               extreme = extreme + 1_int64
         end do
         p_value = real(extreme, real64) / real(combinations, real64)
      else
         variance = sum(ranks * ranks) / 4.0_real64
         if (variance <= 0.0_real64) then
            status = STATS_OK
            return
         end if
         z = max(0.0_real64, observed_distance - 0.5_real64) / sqrt(variance)
         p_value = erfc(z / sqrt(2.0_real64))
      end if
      p_value = max(0.0_real64, min(1.0_real64, p_value))
      status = STATS_OK
   end subroutine bench_wilcoxon_signed_rank

   subroutine bench_holm_adjust(p_values, adjusted, status)
      real(real64), intent(in) :: p_values(:)
      real(real64), intent(out) :: adjusted(:)
      integer(int32), intent(out) :: status
      logical, allocatable :: selected(:)
      integer :: n, rank, index, allocation_status
      real(real64) :: smallest, previous, candidate

      status = STATS_INVALID
      n = size(p_values)
      if (n == 0 .or. size(adjusted) /= n .or. .not. all(ieee_is_finite(p_values))) return
      if (any(p_values < 0.0_real64) .or. any(p_values > 1.0_real64)) return
      allocate(selected(n), stat=allocation_status)
      if (allocation_status /= 0) return
      selected = .false.
      adjusted = 1.0_real64
      previous = 0.0_real64
      do rank = 1, n
         smallest = huge(1.0_real64)
         index = 0
         block
            integer :: i
            do i = 1, n
               if (.not. selected(i) .and. p_values(i) < smallest) then
                  smallest = p_values(i)
                  index = i
               end if
            end do
         end block
         if (index == 0) return
         selected(index) = .true.
         candidate = min(1.0_real64, real(n - rank + 1, real64) * smallest)
         previous = max(previous, candidate)
         adjusted(index) = previous
      end do
      status = STATS_OK
   end subroutine bench_holm_adjust

   subroutine bench_pareto_front(metrics, minimize, front, status)
      real(real64), intent(in) :: metrics(:, :)
      logical, intent(in) :: minimize(:)
      logical, intent(out) :: front(:)
      integer(int32), intent(out) :: status
      integer :: method_count, metric_count, i, j, metric
      logical :: no_worse, strictly_better

      status = STATS_INVALID
      method_count = size(metrics, 2)
      metric_count = size(metrics, 1)
      if (method_count == 0 .or. metric_count == 0 .or. size(minimize) /= metric_count .or. &
          size(front) /= method_count) return
      front = .false.
      do i = 1, method_count
         if (.not. all(ieee_is_finite(metrics(:, i)))) cycle
         front(i) = .true.
         do j = 1, method_count
            if (j == i .or. .not. all(ieee_is_finite(metrics(:, j)))) cycle
            no_worse = .true.
            strictly_better = .false.
            do metric = 1, metric_count
               if (minimize(metric)) then
                  no_worse = no_worse .and. metrics(metric, j) <= metrics(metric, i)
                  strictly_better = strictly_better .or. metrics(metric, j) < metrics(metric, i)
               else
                  no_worse = no_worse .and. metrics(metric, j) >= metrics(metric, i)
                  strictly_better = strictly_better .or. metrics(metric, j) > metrics(metric, i)
               end if
            end do
            if (no_worse .and. strictly_better) then
               front(i) = .false.
               exit
            end if
         end do
      end do
      status = STATS_OK
   end subroutine bench_pareto_front

   recursive subroutine quicksort(values, first, last)
      real(real64), intent(inout) :: values(:)
      integer, intent(in) :: first, last
      integer :: left, right
      real(real64) :: pivot, swap
      if (first >= last) return
      left = first
      right = last
      pivot = values((first + last) / 2)
      do
         do while (values(left) < pivot)
            left = left + 1
         end do
         do while (values(right) > pivot)
            right = right - 1
         end do
         if (left <= right) then
            swap = values(left)
            values(left) = values(right)
            values(right) = swap
            left = left + 1
            right = right - 1
         end if
         if (left > right) exit
      end do
      if (first < right) call quicksort(values, first, right)
      if (left < last) call quicksort(values, left, last)
   end subroutine quicksort

   subroutine sort_values(values)
      real(real64), intent(inout) :: values(:)
      if (size(values) > 1) call quicksort(values, 1, size(values))
   end subroutine sort_values

end module bench_statistics
