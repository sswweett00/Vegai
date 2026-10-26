! Fortran-only paired analysis of Phase 2 search CSV output.
program bench_phase3_analysis
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64, error_unit
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite, ieee_value, ieee_quiet_nan
   use bench_tuning, only : BENCH_TUNING_TRIALS
   use bench_statistics, only : STATS_OK, bench_paired_summary, bench_bootstrap_mean_ci, &
                                bench_wilcoxon_signed_rank, bench_holm_adjust, bench_pareto_front
   implicit none

   integer, parameter :: max_tasks = 10, max_methods = 24, seed_count = 5, csv_fields = 23
   integer(int32), parameter :: bootstrap_replicates = 10000_int32
   real(real64), parameter :: alpha = 0.05_real64
   integer(int64), parameter :: expected_seeds(seed_count) = [1_int64, 7_int64, 19_int64, 101_int64, 2026_int64]
   character(len=512) :: input_path, output_path, line
   character(len=48) :: task_names(max_tasks), method_names(max_methods)
   character(len=512) :: fields(csv_fields)
   integer(int64) :: seeds(seed_count)
   integer(int32) :: candidate_count(max_tasks, max_methods, seed_count)
   integer(int32) :: selected_count(max_tasks, max_methods, seed_count)
   integer(int32) :: selected_candidate_id(max_tasks, max_methods, seed_count)
   integer(int32) :: task_count, method_count, task_index, method_index, seed_index
   integer(int32) :: candidate_id, row_status, row_test_status, stats_status, apex_index
   integer(int32) :: pair_count(max_tasks, max_methods), selected_n(max_tasks, max_methods)
   integer(int32) :: unit_in, unit_out, io_status, arg_status, field_count
   integer(int32) :: test_methods(max_methods), test_count
   integer(int32) :: valid_methods(max_methods), valid_count
   real(real64) :: selected_loss(max_tasks, max_methods, seed_count)
   real(real64) :: selected_validation_loss(max_tasks, max_methods, seed_count)
   real(real64) :: candidate_validation_loss(max_tasks, max_methods, seed_count, BENCH_TUNING_TRIALS)
   real(real64) :: selected_state_bytes(max_tasks, max_methods, seed_count)
   real(real64) :: total_search_seconds(max_tasks, max_methods, seed_count)
   real(real64) :: paired_mean(max_tasks, max_methods), paired_sd(max_tasks, max_methods)
   real(real64) :: ci_lower(max_tasks, max_methods), ci_upper(max_tasks, max_methods)
   real(real64) :: wilcoxon_stat(max_tasks, max_methods), p_raw(max_tasks, max_methods)
   real(real64) :: p_holm(max_tasks, max_methods)
   real(real64) :: mean_loss(max_tasks, max_methods), mean_validation(max_tasks, max_methods)
   real(real64) :: mean_search(max_tasks, max_methods), mean_state(max_tasks, max_methods)
   integer(int32) :: validation_n(max_tasks, max_methods)
   real(real64) :: differences(seed_count), p_buffer(max_methods), adjusted(max_methods)
   real(real64) :: pareto_metrics(3, max_methods), mean_value
   logical :: selected_seen(max_tasks, max_methods, seed_count)
   logical :: holdout_seen(max_tasks, max_methods, seed_count)
   logical :: candidate_seen(max_tasks, max_methods, seed_count, BENCH_TUNING_TRIALS)
   logical :: input_open, output_open, row_holdout, row_test_holdout
   logical :: has_holdout(max_tasks, max_methods), has_test_holdout(max_tasks, max_methods)
   logical :: pareto(max_tasks, max_methods)
   logical :: pareto_buffer(max_methods), minimize(3)
   integer(int64) :: current_seed, bootstrap_seed, trial_max_steps, trial_steps
   integer :: i, j, k, parsed_integer, available_seeds, finite_candidates, best_candidate

   unit_in = 0
   unit_out = 0
   input_open = .false.
   output_open = .false.
   input_path = 'bench/results/phase2/validation_search.csv'
   output_path = 'bench/results/phase2/paired_analysis.csv'
   call get_command_argument(1, input_path, status=arg_status)
   if (arg_status /= 0) call fatal('input-path argument is too long')
   if (len_trim(input_path) == 0) input_path = 'bench/results/phase2/validation_search.csv'
   call get_command_argument(2, output_path, status=arg_status)
   if (arg_status /= 0) call fatal('output-path argument is too long')
   if (len_trim(output_path) == 0) output_path = 'bench/results/phase2/paired_analysis.csv'

   task_names = ''
   method_names = ''
   task_count = 0_int32
   method_count = 0_int32
   seeds = expected_seeds
   candidate_count = 0_int32
   selected_count = 0_int32
   selected_candidate_id = 0_int32
   selected_seen = .false.
   holdout_seen = .false.
   candidate_seen = .false.
   selected_loss = ieee_value(0.0_real64, ieee_quiet_nan)
   selected_validation_loss = ieee_value(0.0_real64, ieee_quiet_nan)
   candidate_validation_loss = ieee_value(0.0_real64, ieee_quiet_nan)
   selected_state_bytes = ieee_value(0.0_real64, ieee_quiet_nan)
   total_search_seconds = 0.0_real64
   has_holdout = .false.
   has_test_holdout = .false.

   open(newunit=unit_in, file=trim(input_path), status='old', action='read', iostat=io_status)
   if (io_status /= 0) call fatal('cannot open Phase 2 search CSV')
   input_open = .true.
   read(unit_in, '(a)', iostat=io_status) line
   if (io_status /= 0) call fatal('search CSV has no header')
   if (trim(line) /= 'task,optimizer,seed,candidate_id,selected,validation_is_holdout,test_is_holdout,'// &
       'learning_rate,beta1,beta2,epsilon,weight_decay,rho,max_steps,steps,training_initial_loss,'// &
       'validation_loss,test_loss,test_status,elapsed_seconds,test_elapsed_seconds,state_bytes,status') &
      call fatal('unrecognized Phase 2 CSV header')

   do
      read(unit_in, '(a)', iostat=io_status) line
      if (io_status < 0) exit
      if (io_status /= 0) call fatal('error reading search CSV')
      call split_csv_line(trim(line), fields, field_count)
      if (field_count /= csv_fields) call fatal('malformed search CSV row')
      call locate_name(trim(fields(1)), task_names, task_count, task_index)
      call locate_name(trim(fields(2)), method_names, method_count, method_index)
      read(fields(3), *, iostat=io_status) current_seed
      if (io_status /= 0) call fatal('invalid seed in search CSV')
      seed_index = find_seed(current_seed, seeds)
      if (seed_index == 0_int32) call fatal('unexpected seed in search CSV')
      candidate_id = 0_int32
      read(fields(4), *, iostat=io_status) candidate_id
      if (io_status /= 0 .or. candidate_id < 1_int32 .or. candidate_id > BENCH_TUNING_TRIALS) &
         call fatal('invalid candidate ID in search CSV')
      trial_max_steps = 0_int64
      read(fields(14), *, iostat=io_status) trial_max_steps
      if (io_status /= 0 .or. trial_max_steps <= 0_int64) call fatal('invalid max_steps in search CSV')
      trial_steps = -1_int64
      read(fields(15), *, iostat=io_status) trial_steps
      if (io_status /= 0 .or. trial_steps < 0_int64 .or. trial_steps > trial_max_steps) &
         call fatal('invalid steps in search CSV')
      mean_value = 0.0_real64
      read(fields(20), *, iostat=io_status) mean_value
      if (io_status /= 0 .or. .not. ieee_is_finite(mean_value) .or. mean_value < 0.0_real64) &
         call fatal('invalid elapsed time in search CSV')
      total_search_seconds(task_index, method_index, seed_index) = &
         total_search_seconds(task_index, method_index, seed_index) + mean_value
      if (candidate_seen(task_index, method_index, seed_index, candidate_id)) &
         call fatal('duplicate candidate ID in search CSV')
      candidate_seen(task_index, method_index, seed_index, candidate_id) = .true.
      candidate_count(task_index, method_index, seed_index) = &
         candidate_count(task_index, method_index, seed_index) + 1_int32
      if (candidate_count(task_index, method_index, seed_index) > BENCH_TUNING_TRIALS) &
         call fatal('duplicate candidate row in search CSV')

      read(fields(23), *, iostat=io_status) row_status
      if (io_status /= 0) call fatal('invalid candidate status in search CSV')
      if (row_status == 0_int32) then
         call parse_finite_real(fields(17), mean_value, 'candidate validation loss')
         candidate_validation_loss(task_index, method_index, seed_index, candidate_id) = mean_value
      else
         call require_nan(fields(17), 'failed candidate validation loss')
      end if
      if (len_trim(fields(5)) == 0 .or. len_trim(fields(6)) == 0 .or. len_trim(fields(7)) == 0) &
         call fatal('missing logical CSV field')
      if (.not. valid_logical(fields(5)) .or. .not. valid_logical(fields(6)) .or. &
          .not. valid_logical(fields(7))) call fatal('invalid logical CSV field')
      row_holdout = fields(6)(1:1) == 'T' .or. fields(6)(1:1) == 't'
      row_test_holdout = fields(7)(1:1) == 'T' .or. fields(7)(1:1) == 't'
      if (row_holdout .neqv. row_test_holdout) call fatal('validation/test holdout flag mismatch')
      if (any(holdout_seen(task_index, method_index, :))) then
         if (has_holdout(task_index, method_index) .neqv. row_holdout) &
            call fatal('validation holdout flag changes within a search cell')
         if (has_test_holdout(task_index, method_index) .neqv. row_test_holdout) &
            call fatal('test holdout flag changes within a search cell')
      else
         has_holdout(task_index, method_index) = row_holdout
         has_test_holdout(task_index, method_index) = row_test_holdout
      end if
      holdout_seen(task_index, method_index, seed_index) = .true.
      read(fields(19), *, iostat=io_status) row_test_status
      if (io_status /= 0) call fatal('invalid test status in search CSV')
      call parse_finite_real(fields(21), mean_value, 'test evaluation elapsed time')
      if (mean_value < 0.0_real64) call fatal('negative test evaluation elapsed time')
      if (fields(5)(1:1) == 'T' .or. fields(5)(1:1) == 't') then
         selected_count(task_index, method_index, seed_index) = &
            selected_count(task_index, method_index, seed_index) + 1_int32
         if (selected_seen(task_index, method_index, seed_index)) call fatal('multiple selected rows for one search')
         selected_seen(task_index, method_index, seed_index) = .true.
         if (row_status /= 0_int32) call fatal('selected candidate has failed training/validation status')
         selected_candidate_id(task_index, method_index, seed_index) = candidate_id
         selected_validation_loss(task_index, method_index, seed_index) = &
            candidate_validation_loss(task_index, method_index, seed_index, candidate_id)
         call parse_finite_real(fields(22), selected_state_bytes(task_index, method_index, seed_index), &
                                'selected state bytes')
         if (row_test_status == 0_int32) then
            call parse_finite_real(fields(18), selected_loss(task_index, method_index, seed_index), &
                                   'selected test loss')
         else
            call require_nan(fields(18), 'failed selected test loss')
         end if
      else
         if (row_test_status == 0_int32) call fatal('test loss is only allowed on the selected row')
         call require_nan(fields(18), 'non-selected test loss')
      end if
   end do
   close(unit_in)
   input_open = .false.

   if (task_count == 0 .or. method_count == 0) call fatal('search CSV contains no candidates')
   do i = 1, task_count
      do j = 1, method_count
         available_seeds = count(candidate_count(i, j, :) > 0)
         if (available_seeds > 0) then
            do k = 1, seed_count
               if (candidate_count(i, j, k) /= BENCH_TUNING_TRIALS) &
                  call fatal('incomplete five-seed 20-candidate search grid')
            end do
         end if
         do k = 1, seed_count
            if (selected_count(i, j, k) > 1) call fatal('multiple selected outcomes for task/method/seed')
            if (candidate_count(i, j, k) == 0) cycle
            finite_candidates = count(ieee_is_finite(candidate_validation_loss(i, j, k, :)))
            if (finite_candidates == 0) then
               if (selected_count(i, j, k) /= 0) call fatal('all-failed grid unexpectedly selects a candidate')
            else
               if (selected_count(i, j, k) /= 1) call fatal('finite grid has no unique selected candidate')
               best_candidate = minloc(candidate_validation_loss(i, j, k, :), dim=1, &
                                      mask=ieee_is_finite(candidate_validation_loss(i, j, k, :)))
               if (selected_candidate_id(i, j, k) /= best_candidate) &
                  call fatal('selected candidate is not the minimum finite validation loss')
            end if
         end do
      end do
   end do

   pair_count = 0_int32
   selected_n = 0_int32
   paired_mean = ieee_value(0.0_real64, ieee_quiet_nan)
   paired_sd = ieee_value(0.0_real64, ieee_quiet_nan)
   ci_lower = ieee_value(0.0_real64, ieee_quiet_nan)
   ci_upper = ieee_value(0.0_real64, ieee_quiet_nan)
   wilcoxon_stat = ieee_value(0.0_real64, ieee_quiet_nan)
   p_raw = ieee_value(0.0_real64, ieee_quiet_nan)
   p_holm = ieee_value(0.0_real64, ieee_quiet_nan)
   mean_loss = ieee_value(0.0_real64, ieee_quiet_nan)
   mean_validation = ieee_value(0.0_real64, ieee_quiet_nan)
   validation_n = 0_int32
   mean_search = ieee_value(0.0_real64, ieee_quiet_nan)
   mean_state = ieee_value(0.0_real64, ieee_quiet_nan)
   pareto = .false.

   do i = 1, task_count
      apex_index = find_method('Apex', method_names, method_count)
      if (apex_index == 0_int32) cycle
      do j = 1, method_count
         selected_n(i, j) = count(ieee_is_finite(selected_loss(i, j, :)))
         if (selected_n(i, j) > 0) then
            mean_loss(i, j) = sum(pack(selected_loss(i, j, :), &
                                       ieee_is_finite(selected_loss(i, j, :)))) / real(selected_n(i, j), real64)
         end if
         validation_n(i, j) = count(ieee_is_finite(selected_validation_loss(i, j, :)))
         if (validation_n(i, j) > 0) then
            mean_validation(i, j) = sum(pack(selected_validation_loss(i, j, :), &
                                              ieee_is_finite(selected_validation_loss(i, j, :)))) / &
                                    real(validation_n(i, j), real64)
            mean_state(i, j) = sum(pack(selected_state_bytes(i, j, :), &
                                        ieee_is_finite(selected_validation_loss(i, j, :)))) / &
                               real(validation_n(i, j), real64)
         end if
         available_seeds = count(candidate_count(i, j, :) > 0)
         if (available_seeds > 0) then
            mean_search(i, j) = sum(total_search_seconds(i, j, :)) / real(available_seeds, real64)
         end if
         differences = 0.0_real64
         pair_count(i, j) = 0_int32
         do k = 1, seed_count
            if (.not. ieee_is_finite(selected_loss(i, j, k)) .or. &
                .not. ieee_is_finite(selected_loss(i, apex_index, k))) cycle
            pair_count(i, j) = pair_count(i, j) + 1_int32
            differences(pair_count(i, j)) = selected_loss(i, j, k) - selected_loss(i, apex_index, k)
         end do
         if (pair_count(i, j) < 2) cycle
         call bench_paired_summary(differences(1:pair_count(i, j)), paired_mean(i, j), paired_sd(i, j), stats_status)
         if (stats_status /= STATS_OK) cycle
         bootstrap_seed = int(i * 100000 + j * 101 + 43, int64)
         call bench_bootstrap_mean_ci(differences(1:pair_count(i, j)), bootstrap_seed, &
                                      bootstrap_replicates, alpha, ci_lower(i, j), ci_upper(i, j), stats_status)
         if (stats_status /= STATS_OK) cycle
         call bench_wilcoxon_signed_rank(differences(1:pair_count(i, j)), wilcoxon_stat(i, j), &
                                         p_raw(i, j), parsed_integer, stats_status)
         if (stats_status /= STATS_OK) p_raw(i, j) = ieee_value(0.0_real64, ieee_quiet_nan)
      end do

      test_count = 0_int32
      do j = 1, method_count
         if (j == apex_index .or. .not. ieee_is_finite(p_raw(i, j))) cycle
         test_count = test_count + 1_int32
         test_methods(test_count) = int(j, int32)
         p_buffer(test_count) = p_raw(i, j)
      end do
      if (test_count > 0) then
         call bench_holm_adjust(p_buffer(1:test_count), adjusted(1:test_count), stats_status)
         if (stats_status == STATS_OK) then
            do j = 1, test_count
               p_holm(i, test_methods(j)) = adjusted(j)
            end do
         end if
      end if

      valid_count = 0_int32
      do j = 1, method_count
         if (.not. ieee_is_finite(mean_loss(i, j)) .or. .not. ieee_is_finite(mean_search(i, j)) .or. &
             .not. ieee_is_finite(mean_state(i, j))) cycle
         valid_count = valid_count + 1_int32
         valid_methods(valid_count) = int(j, int32)
         pareto_metrics(:, valid_count) = [mean_loss(i, j), mean_search(i, j), mean_state(i, j)]
      end do
      if (valid_count > 0) then
         minimize = [.true., .true., .true.]
         call bench_pareto_front(pareto_metrics(:, 1:valid_count), minimize, &
                                 pareto_buffer(1:valid_count), stats_status)
         if (stats_status == STATS_OK) then
            do j = 1, valid_count
               pareto(i, valid_methods(j)) = pareto_buffer(j)
            end do
         end if
      end if
   end do

   open(newunit=unit_out, file=trim(output_path), status='replace', action='write', iostat=io_status)
   if (io_status /= 0) call fatal('cannot open paired-analysis CSV')
   output_open = .true.
   write(unit_out, '(a)', iostat=io_status) &
      'task,optimizer,validation_is_holdout,test_is_holdout,selected_test_n,validation_selected_n,paired_n,'// &
      'mean_test_loss,mean_validation_loss,mean_search_seconds,mean_state_bytes,mean_delta_vs_apex,'// &
      'sample_sd_delta,bootstrap_95_low,bootstrap_95_high,'// &
      'wilcoxon_wplus,p_raw,p_holm,pareto_front,analysis_status'
   if (io_status /= 0) call fatal('cannot write paired-analysis CSV header')
   do i = 1, task_count
      do j = 1, method_count
         if (sum(candidate_count(i, j, :)) == 0) cycle
         call write_analysis_row(unit_out, trim(task_names(i)), trim(method_names(j)), has_holdout(i, j), &
                                 has_test_holdout(i, j), selected_n(i, j), validation_n(i, j), pair_count(i, j), &
                                 mean_loss(i, j), mean_validation(i, j), mean_search(i, j), mean_state(i, j), &
                                 paired_mean(i, j), paired_sd(i, j), ci_lower(i, j), ci_upper(i, j), &
                                 wilcoxon_stat(i, j), p_raw(i, j), p_holm(i, j), pareto(i, j), stats_status)
         if (stats_status /= STATS_OK) call fatal('paired-analysis CSV row write failed')
      end do
   end do
   close(unit_out, iostat=io_status)
   output_open = .false.
   if (io_status /= 0) call fatal('paired-analysis CSV close failed')

contains

   subroutine split_csv_line(text, output_fields, count_fields)
      character(len=*), intent(in) :: text
      character(len=*), intent(out) :: output_fields(:)
      integer(int32), intent(out) :: count_fields
      integer :: position, start, length_text
      output_fields = ''
      count_fields = 1_int32
      start = 1
      length_text = len_trim(text)
      do position = 1, length_text
         if (text(position:position) /= ',') cycle
         if (count_fields > size(output_fields)) call fatal('too many CSV fields')
         output_fields(count_fields) = text(start:position - 1)
         count_fields = count_fields + 1_int32
         start = position + 1
      end do
      if (count_fields > size(output_fields)) call fatal('too many CSV fields')
      output_fields(count_fields) = text(start:length_text)
   end subroutine split_csv_line

   subroutine locate_name(name, names, name_count, index)
      character(len=*), intent(in) :: name
      character(len=*), intent(inout) :: names(:)
      integer(int32), intent(inout) :: name_count
      integer(int32), intent(out) :: index
      integer :: n
      index = 0_int32
      do n = 1, name_count
         if (trim(names(n)) == trim(name)) then
            index = int(n, int32)
            return
         end if
      end do
      if (name_count >= size(names)) call fatal('too many unique task or optimizer labels')
      name_count = name_count + 1_int32
      names(name_count) = trim(name)
      index = name_count
   end subroutine locate_name

   integer(int32) function find_seed(value, seed_values) result(index)
      integer(int64), intent(in) :: value, seed_values(:)
      integer :: n
      index = 0_int32
      do n = 1, size(seed_values)
         if (seed_values(n) == value) then
            index = int(n, int32)
            return
         end if
      end do
   end function find_seed

   integer(int32) function find_method(name, names, name_count) result(index)
      character(len=*), intent(in) :: name, names(:)
      integer(int32), intent(in) :: name_count
      integer :: n
      index = 0_int32
      do n = 1, name_count
         if (trim(names(n)) == trim(name)) then
            index = int(n, int32)
            return
         end if
      end do
   end function find_method

   pure logical function valid_logical(text)
      character(len=*), intent(in) :: text
      valid_logical = .false.
      if (len_trim(text) == 0) return
      valid_logical = text(1:1) == 'T' .or. text(1:1) == 't' .or. text(1:1) == 'F' .or. text(1:1) == 'f'
   end function valid_logical

   subroutine require_nan(text, label)
      character(len=*), intent(in) :: text, label
      if (index(text, 'NaN') == 0 .and. index(text, 'nan') == 0 .and. index(text, 'NAN') == 0) &
         call fatal(trim(label)//' should be NaN for a failed trial')
   end subroutine require_nan

   subroutine parse_finite_real(text, value, label)
      character(len=*), intent(in) :: text, label
      real(real64), intent(out) :: value
      integer :: parse_status
      if (index(text, 'NaN') > 0 .or. index(text, 'nan') > 0 .or. index(text, 'NAN') > 0) then
         value = ieee_value(0.0_real64, ieee_quiet_nan)
         call fatal(trim(label)//' is not finite')
      end if
      value = ieee_value(0.0_real64, ieee_quiet_nan)
      read(text, *, iostat=parse_status) value
      if (parse_status /= 0) call fatal('invalid '//trim(label))
      if (.not. ieee_is_finite(value)) call fatal('invalid '//trim(label))
   end subroutine parse_finite_real

   subroutine write_analysis_row(output_unit, task_name, method_name, validation_holdout, test_holdout, &
                                 n_test, n_validation, n_paired, test_loss, validation_loss, search_time, &
                                 state_bytes, delta, delta_sd, low, high, wplus, p_value, adjusted_p, &
                                 is_pareto, write_status)
      integer, intent(in) :: output_unit
      character(len=*), intent(in) :: task_name, method_name
      logical, intent(in) :: validation_holdout, test_holdout, is_pareto
      integer(int32), intent(in) :: n_test, n_validation, n_paired
      real(real64), intent(in) :: test_loss, validation_loss, search_time, state_bytes
      real(real64), intent(in) :: delta, delta_sd, low, high, wplus, p_value, adjusted_p
      integer(int32), intent(out) :: write_status
      integer(int32) :: analysis_status
      analysis_status = STATS_OK
      if (n_paired < 2) analysis_status = 1_int32
      if (.not. ieee_is_finite(delta) .or. .not. ieee_is_finite(p_value)) analysis_status = 2_int32
      write(output_unit, '(a,",",a,",",l1,",",l1,",",i0,",",i0,",",i0,",", &
                          11(es24.16,","),l1,",",i0)', iostat=write_status) &
         trim(task_name), trim(method_name), validation_holdout, test_holdout, n_test, n_validation, n_paired, &
         test_loss, validation_loss, search_time, state_bytes, delta, delta_sd, low, high, wplus, p_value, &
         adjusted_p, is_pareto, analysis_status
   end subroutine write_analysis_row

   subroutine fatal(message)
      character(len=*), intent(in) :: message
      if (input_open) close(unit_in)
      if (output_open) close(unit_out)
      write(error_unit, '(a)') trim(message)
      stop 1
   end subroutine fatal

end program bench_phase3_analysis
