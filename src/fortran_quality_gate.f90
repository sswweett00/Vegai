! SPDX-License-Identifier: MIT
!==============================================================================
!  fortran_quality_gate.f90
!  A small, deterministic, native quality-gate primitive for HDC checkpoints.
!
!  This is intentionally a bounded structural/top-1 semantic-neighbour gate.
!  It evaluates explicit operator-supplied ASCII token pairs against the loaded
!  Fortran model; held-out provenance is not verified here. It does not claim
!  general language-model quality, semantic safety, or
!  replace governed provenance/cryptographic attestation adapters.
!==============================================================================
module fortran_quality_gate
   use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64, iostat_end
   use portable_cli, only: cli_real64_is_finite
   use hyper_types, only: hv_t
   use text_tokenizer, only: TOKEN_LEN
   use hdc_pretraining, only: PT_OK, hdc_pretrain_model_t, pretrain_model_is_ready, &
                              pretrain_lookup_token, pretrain_get_semantic_hv, &
                              pretrain_nearest_semantic
   implicit none
   private

   integer(int32), parameter, public :: QG_OK = 0_int32
   integer(int32), parameter, public :: QG_ERR_INVALID_ARGUMENT = -940_int32
   integer(int32), parameter, public :: QG_ERR_IO = -941_int32
   integer(int32), parameter, public :: QG_ERR_CASE = -942_int32
   integer(int32), parameter, public :: QG_ERR_MODEL = -943_int32
   integer(int32), parameter, public :: QG_ERR_REJECTED = -944_int32
   integer(int32), parameter, public :: QG_MAX_CASES = 4096_int32

   type, public :: quality_case_t
      character(len=TOKEN_LEN) :: query_token = ''
      character(len=TOKEN_LEN) :: expected_neighbour = ''
   end type quality_case_t

   type, public :: quality_policy_t
      integer(int32) :: minimum_cases = 1_int32
      real(real64) :: minimum_top1_accuracy = 1.0_real64
      real(real64) :: minimum_top1_wilson_lower_bound_95 = 0.0_real64
      real(real64) :: minimum_mean_similarity = -1.0_real64
      real(real64) :: minimum_case_similarity = -1.0_real64
      integer(int32) :: maximum_incorrect_cases = 0_int32
   end type quality_policy_t

   type, public :: quality_report_t
      integer(int32) :: declared_cases = 0_int32
      integer(int32) :: evaluated_cases = 0_int32
      integer(int32) :: correct_cases = 0_int32
      integer(int32) :: incorrect_cases = 0_int32
      integer(int32) :: unavailable_cases = 0_int32
      real(real64) :: top1_accuracy = 0.0_real64
      real(real64) :: top1_wilson_lower_bound_95 = 0.0_real64
      real(real64) :: mean_similarity = 0.0_real64
      real(real64) :: minimum_observed_similarity = -1.0_real64
      logical :: passed = .false.
   end type quality_report_t

   public :: quality_case_file_load, quality_gate_evaluate, quality_policy_is_valid

contains

   pure logical function quality_policy_is_valid(policy) result(valid)
      type(quality_policy_t), intent(in) :: policy

      valid = .false.
      if (policy%minimum_cases <= 0_int32 .or. policy%minimum_cases > QG_MAX_CASES .or. &
          policy%maximum_incorrect_cases < 0_int32) return
      if (.not. cli_real64_is_finite(policy%minimum_top1_accuracy) .or. &
          .not. cli_real64_is_finite(policy%minimum_top1_wilson_lower_bound_95) .or. &
          .not. cli_real64_is_finite(policy%minimum_mean_similarity) .or. &
          .not. cli_real64_is_finite(policy%minimum_case_similarity)) return
      if (policy%minimum_top1_accuracy < 0.0_real64 .or. policy%minimum_top1_accuracy > 1.0_real64) return
      if (policy%minimum_top1_wilson_lower_bound_95 < 0.0_real64 .or. &
          policy%minimum_top1_wilson_lower_bound_95 > 1.0_real64) return
      if (policy%minimum_mean_similarity < -1.0_real64 .or. policy%minimum_mean_similarity > 1.0_real64) return
      if (policy%minimum_case_similarity < -1.0_real64 .or. policy%minimum_case_similarity > 1.0_real64) return
      valid = .true.
   end function quality_policy_is_valid

   subroutine quality_case_file_load(path, cases, status)
      character(len=*), intent(in) :: path
      type(quality_case_t), allocatable, intent(out) :: cases(:)
      integer(int32), intent(out) :: status
      type(quality_case_t), allocatable :: work(:)
      integer :: unit, io_status, count, record_bytes
      character(len=2 * TOKEN_LEN + 2) :: line
      logical :: eof

      status = QG_ERR_INVALID_ARGUMENT
      if (len_trim(path) == 0) return
      allocate(work(QG_MAX_CASES), stat=io_status)
      if (io_status /= 0) then
         status = QG_ERR_IO
         return
      end if
      open(newunit=unit, file=trim(path), status='old', action='read', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) then
         status = QG_ERR_IO
         return
      end if

      ! A single bounded pass avoids counting one version of a file and then
      ! indexing a changed second version without revalidation. It also rejects
      ! overlong records rather than accepting their truncated prefixes.
      count = 0
      do
         call quality_read_record(unit, line, record_bytes, eof, status)
         if (status /= QG_OK) then
            close(unit)
            return
         end if
         if (eof) exit
         if (record_bytes <= 0) then
            close(unit)
            status = QG_ERR_CASE
            return
         end if
         if (.not. quality_case_line_is_valid(line(:record_bytes))) then
            close(unit)
            status = QG_ERR_CASE
            return
         end if
         count = count + 1
         if (count > int(QG_MAX_CASES)) then
            close(unit)
            status = QG_ERR_CASE
            return
         end if
         call quality_case_from_line(line(:record_bytes), work(count))
      end do
      close(unit, iostat=io_status)
      if (io_status /= 0) then
         status = QG_ERR_IO
         return
      end if
      if (count == 0) then
         status = QG_ERR_CASE
         return
      end if
      allocate(cases(count), stat=io_status)
      if (io_status /= 0) then
         status = QG_ERR_IO
         return
      end if
      cases = work(:count)
      status = QG_OK
   end subroutine quality_case_file_load

   subroutine quality_read_record(unit, line, record_bytes, eof, status)
      integer, intent(in) :: unit
      character(len=*), intent(out) :: line
      integer, intent(out) :: record_bytes
      logical, intent(out) :: eof
      integer(int32), intent(out) :: status
      character(len=1) :: byte
      integer :: io_status

      line = ''
      record_bytes = 0
      eof = .false.
      status = QG_OK
      do
         read(unit, iostat=io_status) byte
         if (io_status == iostat_end) then
            if (record_bytes == 0) then
               eof = .true.
            else if (line(record_bytes:record_bytes) == achar(13)) then
               ! A bare final CR is not a CRLF delimiter.
               status = QG_ERR_CASE
            end if
            return
         end if
         if (io_status /= 0) then
            status = QG_ERR_IO
            return
         end if
         if (byte == achar(10)) then
            if (record_bytes > 0) then
               if (line(record_bytes:record_bytes) == achar(13)) record_bytes = record_bytes - 1
            end if
            return
         end if
         if (record_bytes >= len(line)) then
            status = QG_ERR_CASE
            return
         end if
         record_bytes = record_bytes + 1
         line(record_bytes:record_bytes) = byte
      end do
   end subroutine quality_read_record

   subroutine quality_gate_evaluate(model, cases, policy, report, status)
      type(hdc_pretrain_model_t), intent(in) :: model
      type(quality_case_t), intent(in) :: cases(:)
      type(quality_policy_t), intent(in) :: policy
      type(quality_report_t), intent(out) :: report
      integer(int32), intent(out) :: status
      type(hv_t) :: query_hv
      integer(int32) :: case_index, query_slot, neighbour_slot, operation_status
      logical :: found
      character(len=TOKEN_LEN) :: neighbour_token
      real(real32) :: similarity
      real(real64) :: similarity_sum

      report = quality_report_t()
      status = QG_ERR_INVALID_ARGUMENT
      if (.not. pretrain_model_is_ready(model)) then
         status = QG_ERR_MODEL
         return
      end if
      if (.not. quality_policy_is_valid(policy)) return
      if (size(cases) <= 0 .or. size(cases) > int(QG_MAX_CASES)) return
      report%declared_cases = int(size(cases), int32)
      similarity_sum = 0.0_real64
      report%minimum_observed_similarity = 1.0_real64

      do case_index = 1, size(cases)
         if (.not. quality_case_is_valid(cases(case_index))) then
            report%unavailable_cases = report%unavailable_cases + 1_int32
            cycle
         end if
         call pretrain_lookup_token(model, trim(cases(case_index)%query_token), query_slot, found, operation_status)
         if (operation_status /= PT_OK .or. .not. found) then
            report%unavailable_cases = report%unavailable_cases + 1_int32
            cycle
         end if
         call pretrain_get_semantic_hv(model, query_slot, query_hv, operation_status)
         if (operation_status /= PT_OK) then
            report%unavailable_cases = report%unavailable_cases + 1_int32
            cycle
         end if
         call pretrain_nearest_semantic(model, query_hv, neighbour_slot, neighbour_token, similarity, &
                                        operation_status, exclude_slot=query_slot)
         if (operation_status /= PT_OK .or. neighbour_slot <= 0_int32) then
            report%unavailable_cases = report%unavailable_cases + 1_int32
            cycle
         end if
         report%evaluated_cases = report%evaluated_cases + 1_int32
         similarity_sum = similarity_sum + real(similarity, real64)
         report%minimum_observed_similarity = min(report%minimum_observed_similarity, real(similarity, real64))
         if (trim(neighbour_token) == trim(cases(case_index)%expected_neighbour)) then
            report%correct_cases = report%correct_cases + 1_int32
         else
            report%incorrect_cases = report%incorrect_cases + 1_int32
         end if
      end do

      if (report%evaluated_cases > 0_int32) then
         report%top1_accuracy = real(report%correct_cases, real64) / real(report%evaluated_cases, real64)
         report%mean_similarity = similarity_sum / real(report%evaluated_cases, real64)
         report%top1_wilson_lower_bound_95 = quality_wilson_lower_bound(report%correct_cases, report%evaluated_cases)
      else
         report%minimum_observed_similarity = -1.0_real64
      end if

      report%passed = report%unavailable_cases == 0_int32 .and. &
                      report%evaluated_cases >= policy%minimum_cases .and. &
                      report%top1_accuracy >= policy%minimum_top1_accuracy .and. &
                      report%top1_wilson_lower_bound_95 >= policy%minimum_top1_wilson_lower_bound_95 .and. &
                      report%mean_similarity >= policy%minimum_mean_similarity .and. &
                      report%minimum_observed_similarity >= policy%minimum_case_similarity .and. &
                      report%incorrect_cases <= policy%maximum_incorrect_cases
      if (report%passed) then
         status = QG_OK
      else
         status = QG_ERR_REJECTED
      end if
   end subroutine quality_gate_evaluate

   pure real(real64) function quality_wilson_lower_bound(correct, total) result(bound)
      integer(int32), intent(in) :: correct, total
      real(real64), parameter :: z95 = 1.959963984540054_real64
      real(real64) :: observations, proportion, denominator, variance

      bound = 0.0_real64
      if (total <= 0_int32 .or. correct < 0_int32 .or. correct > total) return
      observations = real(total, real64)
      proportion = real(correct, real64) / observations
      denominator = 1.0_real64 + z95 * z95 / observations
      variance = proportion * (1.0_real64 - proportion) / observations + &
                 z95 * z95 / (4.0_real64 * observations * observations)
      bound = (proportion + z95 * z95 / (2.0_real64 * observations) - z95 * sqrt(variance)) / denominator
      bound = max(0.0_real64, min(1.0_real64, bound))
   end function quality_wilson_lower_bound

   pure logical function quality_case_is_valid(case_data) result(valid)
      type(quality_case_t), intent(in) :: case_data

      valid = quality_token_is_valid(trim(case_data%query_token)) .and. &
              quality_token_is_valid(trim(case_data%expected_neighbour)) .and. &
              case_data%query_token /= case_data%expected_neighbour
   end function quality_case_is_valid

   pure logical function quality_case_line_is_valid(line) result(valid)
      character(len=*), intent(in) :: line
      integer :: tab_position, extra_tab_position

      valid = .false.
      tab_position = index(line, achar(9))
      if (tab_position <= 1 .or. tab_position >= len(line)) return
      extra_tab_position = index(line(tab_position + 1:), achar(9))
      if (extra_tab_position /= 0) return
      valid = quality_token_is_valid(line(:tab_position - 1)) .and. &
              quality_token_is_valid(line(tab_position + 1:)) .and. &
              line(:tab_position - 1) /= line(tab_position + 1:)
   end function quality_case_line_is_valid

   subroutine quality_case_from_line(line, case_data)
      character(len=*), intent(in) :: line
      type(quality_case_t), intent(out) :: case_data
      integer :: tab_position

      case_data = quality_case_t()
      tab_position = index(line, achar(9))
      case_data%query_token = trim(line(:tab_position - 1))
      case_data%expected_neighbour = trim(line(tab_position + 1:))
   end subroutine quality_case_from_line

   pure logical function quality_token_is_valid(token) result(valid)
      character(len=*), intent(in) :: token
      integer :: index, token_length
      character(len=1) :: character

      valid = .false.
      token_length = len(token)
      if (token_length <= 0 .or. token_length > TOKEN_LEN) return
      do index = 1, token_length
         character = token(index:index)
         if ((character >= 'a' .and. character <= 'z') .or. &
             (character >= '0' .and. character <= '9') .or. character == '_' .or. character == "'") then
            cycle
         end if
         return
      end do
      valid = .true.
   end function quality_token_is_valid

end module fortran_quality_gate
