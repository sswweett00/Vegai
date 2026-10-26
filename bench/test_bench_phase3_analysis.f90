! End-to-end contract test for the Phase 3 CSV analyzer. The fixture is
! generated under the build directory and is never a benchmark observation.
program test_bench_phase3_analysis
   use, intrinsic :: iso_fortran_env, only : int32, int64, real64, error_unit
   use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan
   implicit none

   integer(int64), parameter :: seeds(5) = [1_int64, 7_int64, 19_int64, 101_int64, 2026_int64]
   integer, parameter :: field_capacity = 20
   character(len=512) :: analyzer_path, input_path, output_path, command, line
   character(len=128) :: fields(field_capacity)
   integer :: io_status, command_status, exit_status, unit, field_count, row_count
   real(real64) :: value

   command_status = -1
   exit_status = -1
   call get_command_argument(1, analyzer_path, status=command_status)
   if (command_status /= 0 .or. len_trim(analyzer_path) == 0) call fail('analyzer executable argument required')
   input_path = 'phase3_test_search.csv'
   output_path = 'phase3_test_analysis.csv'
   call write_fixture(trim(input_path), .false., .false.)
   command = '"'//trim(analyzer_path)//'" "'//trim(input_path)//'" "'//trim(output_path)//'"'
   call execute_command_line(trim(command), wait=.true., exitstat=exit_status, cmdstat=command_status)
   if (command_status /= 0 .or. exit_status /= 0) call fail('Phase 3 analyzer rejected the generated fixture')

   open(newunit=unit, file=trim(output_path), status='old', action='read', iostat=io_status)
   if (io_status /= 0) call fail('analyzer did not create its output CSV')
   read(unit, '(a)', iostat=io_status) line
   if (io_status /= 0) call fail('analysis CSV has no header')
   if (trim(line) /= 'task,optimizer,validation_is_holdout,test_is_holdout,selected_test_n,'// &
       'validation_selected_n,paired_n,mean_test_loss,mean_validation_loss,mean_search_seconds,'// &
       'mean_state_bytes,mean_delta_vs_apex,sample_sd_delta,bootstrap_95_low,bootstrap_95_high,'// &
       'wilcoxon_wplus,p_raw,p_holm,pareto_front,analysis_status') call fail('unexpected analysis CSV header')

   row_count = 0
   do
      read(unit, '(a)', iostat=io_status) line
      if (io_status < 0) exit
      if (io_status /= 0) call fail('error reading analysis CSV')
      call split_line(trim(line), fields, field_count)
      if (field_count /= field_capacity) call fail('analysis row has the wrong number of fields')
      row_count = row_count + 1
      if (trim(fields(1)) == 'logistic-dense') then
         if (fields(3)(1:1) /= 'T' .or. fields(4)(1:1) /= 'T') &
            call fail('sample-based holdout flags lost')
      else if (trim(fields(1)) == 'Rosenbrock') then
         if (fields(3)(1:1) /= 'F' .or. fields(4)(1:1) /= 'F') &
            call fail('analytic task incorrectly marked as held out')
      else
         call fail('unexpected task label')
      end if
      if (trim(fields(2)) == 'Apex') then
         call assert_integer(fields(5), 5, 'Apex selected test count')
         call assert_integer(fields(6), 5, 'Apex selected validation count')
         call assert_integer(fields(7), 5, 'Apex paired count')
         call assert_real(fields(8), 3.0_real64, 'Apex mean test loss')
         call assert_real(fields(12), 0.0_real64, 'Apex paired delta')
         if (fields(19)(1:1) /= 'T') call fail('Apex should be on the Pareto front')
         call assert_integer(fields(20), 0, 'Apex analysis status')
      else if (trim(fields(2)) == 'SGD') then
         call assert_integer(fields(5), 5, 'SGD selected test count')
         call assert_integer(fields(6), 5, 'SGD selected validation count')
         call assert_integer(fields(7), 5, 'SGD paired count')
         call assert_real(fields(8), 3.5_real64, 'SGD mean test loss')
         call assert_real(fields(12), 0.5_real64, 'SGD paired delta')
         call assert_real(fields(13), 0.0_real64, 'SGD paired standard deviation')
         call assert_real(fields(14), 0.5_real64, 'SGD bootstrap lower bound')
         call assert_real(fields(15), 0.5_real64, 'SGD bootstrap upper bound')
         call assert_real(fields(16), 15.0_real64, 'SGD Wilcoxon W-plus')
         call assert_real(fields(17), 0.0625_real64, 'SGD exact Wilcoxon p-value')
         call assert_real(fields(18), 0.0625_real64, 'SGD Holm-adjusted p-value')
         if (fields(19)(1:1) /= 'F') call fail('dominated SGD should not be on the Pareto front')
         call assert_integer(fields(20), 0, 'SGD analysis status')
      else
         call fail('unexpected optimizer label')
      end if
   end do
   close(unit, iostat=io_status)
   if (io_status /= 0 .or. row_count /= 4) call fail('analysis CSV should contain two methods for both tasks')
   call remove_file(trim(input_path))
   call remove_file(trim(output_path))
   call expect_rejected(.true., .false., 'wrong validation-selected candidate')
   call expect_rejected(.false., .true., 'non-selected candidate has test score')
   write(*, '(a)') 'PASS: Phase 3 CSV analysis end-to-end and rejection contracts'

contains

   subroutine write_fixture(path, corrupt_selection, leak_test_score)
      character(len=*), intent(in) :: path
      logical, intent(in) :: corrupt_selection, leak_test_score
      character(len=8), parameter :: method_names(2) = [character(len=8) :: 'Apex', 'SGD']
      character(len=24), parameter :: task_names(2) = [character(len=24) :: 'logistic-dense', 'Rosenbrock']
      integer :: output_unit, status, task_index, method_index, seed_index, candidate_id
      integer(int32) :: test_status
      integer(int64) :: steps, max_steps, state_bytes
      logical :: selected, has_holdout
      real(real64) :: validation_loss, test_loss, test_elapsed

      open(newunit=output_unit, file=path, status='replace', action='write', iostat=status)
      if (status /= 0) call fail('cannot create temporary Phase 2 CSV fixture')
      write(output_unit, '(a)', iostat=status) &
         'task,optimizer,seed,candidate_id,selected,validation_is_holdout,test_is_holdout,'// &
         'learning_rate,beta1,beta2,epsilon,weight_decay,rho,max_steps,steps,training_initial_loss,'// &
         'validation_loss,test_loss,test_status,elapsed_seconds,test_elapsed_seconds,state_bytes,status'
      if (status /= 0) call fail('cannot write temporary CSV header')
      max_steps = 100_int64
      steps = max_steps
      do task_index = 1, size(task_names)
         has_holdout = task_index == 1
         do method_index = 1, size(method_names)
            do seed_index = 1, size(seeds)
               do candidate_id = 1, 20
                  selected = candidate_id == 20
                  validation_loss = real(21 - candidate_id, real64) + &
                                    0.01_real64 * real(method_index, real64) + &
                                    0.001_real64 * real(seed_index, real64)
                  test_loss = ieee_value(0.0_real64, ieee_quiet_nan)
                  test_status = 1_int32
                  test_elapsed = 0.0_real64
                  if (leak_test_score .and. method_index == 1 .and. candidate_id == 1) then
                     test_loss = 0.0_real64
                     test_status = 0_int32
                     test_elapsed = 0.001_real64
                  end if
                  if (selected) then
                     test_loss = real(seed_index, real64)
                     if (method_index == 2) test_loss = test_loss + 0.5_real64
                     test_status = 0_int32
                     test_elapsed = 0.001_real64
                  end if
                  if (corrupt_selection .and. method_index == 1 .and. candidate_id == 20) &
                     validation_loss = 100.0_real64
                  state_bytes = 128_int64
                  if (method_index == 2) state_bytes = 256_int64
                  write(output_unit, '(a,",",a,",",i0,",",i0,",",l1,",",l1,",",l1,",", &
                      6(es24.16,","),i0,",",i0,",",3(es24.16,","),i0,",",es24.16,",", &
                      es24.16,",",i0,",",i0)', iostat=status) &
                     trim(task_names(task_index)), trim(method_names(method_index)), seeds(seed_index), &
                     candidate_id, selected, has_holdout, has_holdout, 0.01_real64, 0.9_real64, 0.999_real64, &
                     1.0e-8_real64, 0.0_real64, 0.9_real64, max_steps, steps, 1.0_real64, validation_loss, &
                     test_loss, test_status, &
                     0.01_real64, test_elapsed, state_bytes, 0_int32
                  if (status /= 0) call fail('cannot write temporary candidate row')
               end do
            end do
         end do
      end do
      close(output_unit, iostat=status)
      if (status /= 0) call fail('cannot close temporary candidate fixture')
   end subroutine write_fixture

   subroutine expect_rejected(corrupt_selection, leak_test_score, label)
      logical, intent(in) :: corrupt_selection, leak_test_score
      character(len=*), intent(in) :: label
      integer :: local_command_status, local_exit_status
      logical :: output_file_exists
      call remove_file(trim(output_path))
      call write_fixture(trim(input_path), corrupt_selection, leak_test_score)
      command = '"'//trim(analyzer_path)//'" "'//trim(input_path)//'" "'//trim(output_path)//'"'
      local_command_status = -1
      local_exit_status = -1
      call execute_command_line(trim(command), wait=.true., exitstat=local_exit_status, &
                                cmdstat=local_command_status)
      if (local_command_status /= 0) call fail('could not launch analyzer for invalid fixture: '//trim(label))
      if (local_exit_status == 0) call fail('analyzer accepted invalid fixture: '//trim(label))
      inquire(file=trim(output_path), exist=output_file_exists)
      if (output_file_exists) call fail('analyzer emitted output for invalid fixture: '//trim(label))
      call remove_file(trim(input_path))
   end subroutine expect_rejected

   subroutine split_line(text, output_fields, count_fields)
      character(len=*), intent(in) :: text
      character(len=*), intent(out) :: output_fields(:)
      integer, intent(out) :: count_fields
      integer :: position, start, text_length
      output_fields = ''
      count_fields = 1
      start = 1
      text_length = len_trim(text)
      do position = 1, text_length
         if (text(position:position) /= ',') cycle
         if (count_fields > size(output_fields)) call fail('too many output CSV fields')
         output_fields(count_fields) = text(start:position - 1)
         count_fields = count_fields + 1
         start = position + 1
      end do
      if (count_fields > size(output_fields)) call fail('too many output CSV fields')
      output_fields(count_fields) = text(start:text_length)
   end subroutine split_line

   subroutine assert_real(text, expected, label)
      character(len=*), intent(in) :: text, label
      real(real64), intent(in) :: expected
      real(real64) :: actual
      integer :: status
      read(text, *, iostat=status) actual
      if (status /= 0) call fail('invalid numeric output: '//trim(label))
      if (abs(actual - expected) > 1.0e-12_real64) call fail('wrong numeric output: '//trim(label))
   end subroutine assert_real

   subroutine assert_integer(text, expected, label)
      character(len=*), intent(in) :: text, label
      integer, intent(in) :: expected
      integer :: actual, status
      actual = -huge(actual)
      read(text, *, iostat=status) actual
      if (status /= 0 .or. actual /= expected) call fail('wrong integer output: '//trim(label))
   end subroutine assert_integer

   subroutine remove_file(path)
      character(len=*), intent(in) :: path
      integer :: file_unit, status
      open(newunit=file_unit, file=path, status='old', iostat=status)
      if (status == 0) close(file_unit, status='delete', iostat=status)
   end subroutine remove_file

   subroutine fail(message)
      character(len=*), intent(in) :: message
      write(error_unit, '(a)') trim(message)
      stop 1
   end subroutine fail

end program test_bench_phase3_analysis
