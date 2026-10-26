program test_bench_statistics
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   use bench_statistics
   implicit none

   real(real64) :: differences(5), mean_difference, sample_stddev, lower, upper, lower_repeat, upper_repeat
   real(real64) :: p_values(3), adjusted(3), metrics(2, 3)
   logical :: minimize(2), front(3)
   integer(int32) :: status, effective_n
   real(real64) :: statistic, p_value

   differences = [1.0_real64, 2.0_real64, 3.0_real64, 4.0_real64, 5.0_real64]
   call bench_paired_summary(differences, mean_difference, sample_stddev, status)
   call require(status == STATS_OK, 'paired summary succeeds')
   call require(abs(mean_difference - 3.0_real64) < 1.0e-14_real64, 'paired mean matches hand calculation')
   call require(abs(sample_stddev - sqrt(2.5_real64)) < 1.0e-14_real64, &
                'sample standard deviation matches hand calculation')

   call bench_bootstrap_mean_ci(differences, 91_int64, 10000_int32, 0.05_real64, lower, upper, status)
   call require(status == STATS_OK .and. lower <= upper, 'bootstrap confidence interval is ordered')
   call require(lower <= mean_difference .and. upper >= mean_difference, 'bootstrap interval covers sample mean')
   call bench_bootstrap_mean_ci(differences, 91_int64, 10000_int32, 0.05_real64, &
                                lower_repeat, upper_repeat, status)
   call require(lower == lower_repeat .and. upper == upper_repeat, 'bootstrap is reproducible from fixed seed')

   call bench_wilcoxon_signed_rank(differences, statistic, p_value, effective_n, status)
   call require(status == STATS_OK .and. effective_n == 5_int32, 'paired Wilcoxon uses five nonzero pairs')
   call require(abs(statistic - 15.0_real64) < 1.0e-14_real64, 'Wilcoxon positive-rank sum is exact')
   call require(abs(p_value - 0.0625_real64) < 1.0e-14_real64, 'small-sample exact Wilcoxon p-value is exact')

   p_values = [0.01_real64, 0.04_real64, 0.03_real64]
   call bench_holm_adjust(p_values, adjusted, status)
   call require(status == STATS_OK, 'Holm correction succeeds')
   call require(all(abs(adjusted - [0.03_real64, 0.06_real64, 0.06_real64]) < 1.0e-14_real64), &
                'Holm adjusted p-values match hand calculation')

   metrics = reshape([0.2_real64, 0.6_real64, 0.3_real64, 0.4_real64, 0.4_real64, 0.8_real64], [2, 3])
   minimize = [.true., .true.]
   call bench_pareto_front(metrics, minimize, front, status)
   call require(status == STATS_OK .and. all(front .eqv. [.true., .true., .false.]), &
                'Pareto front retains two non-dominated points and excludes dominated point')
   call require(all(ieee_is_finite([lower, upper, statistic, p_value])), 'reported statistics remain finite')

contains

   subroutine require(condition, message)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: message
      if (.not. condition) then
         write(*, '(a,a)') 'FAIL: ', trim(message)
         stop 1
      end if
   end subroutine require

end program test_bench_statistics
