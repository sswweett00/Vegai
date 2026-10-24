! SPDX-License-Identifier: MIT
!==============================================================================
!  Bounded inspection surface for the native tokenizer-free byte-event field.
!  It deliberately reports representation invariants, not a semantic answer,
!  language interpretation, a prediction, or a claim about external reality.
!==============================================================================
program vegai_field
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, output_unit, error_unit
   use hyper_types, only: HV_BITS, hv_t, hv_equal
   use tokenless_field
   implicit none

   character(len=1024) :: command, source_path
   integer :: argument_count
   integer(int32) :: status

   argument_count = command_argument_count()
   if (argument_count < 1) call fail_usage()
   call read_cli_argument(1, command, status)
   if (status /= FIELD_OK) call fail_usage()

   select case (trim(command))
   case ('inspect')
      if (argument_count /= 2) call fail_usage()
      call read_cli_argument(2, source_path, status)
      if (status /= FIELD_OK) call fail_usage()
      call inspect_file(trim(source_path), status)
      if (status /= FIELD_OK) call fail_with_status('inspection failed', status)
   case ('help', '--help', '-h')
      call print_usage()
   case default
      call fail_usage()
   end select

contains

   subroutine read_cli_argument(position, value, read_status)
      integer, intent(in) :: position
      character(len=*), intent(out) :: value
      integer(int32), intent(out) :: read_status
      integer :: argument_length, command_status

      value = ''
      read_status = FIELD_OK
      call get_command_argument(position, length=argument_length, status=command_status)
      if (command_status /= 0 .or. argument_length < 0 .or. argument_length > len(value)) then
         read_status = FIELD_ERR_INVALID_ARGUMENT
         return
      end if
      call get_command_argument(position, value=value, status=command_status)
      if (command_status /= 0) read_status = FIELD_ERR_INVALID_ARGUMENT
   end subroutine read_cli_argument

   subroutine inspect_file(requested_path, inspect_status)
      character(len=*), intent(in) :: requested_path
      integer(int32), intent(out) :: inspect_status
      integer :: unit, io_status, close_status
      logical :: exists
      integer(int64) :: file_bytes, order_energy, parseval_residual
      integer(int64) :: order_spectrum(HV_BITS)
      integer(int8), allocatable :: bytes(:)
      type(field_encoding_t) :: encoding
      type(hv_t) :: reconstructed

      inspect_status = FIELD_OK
      inquire(file=requested_path, exist=exists, iostat=io_status)
      if (io_status /= 0) then
         inspect_status = FIELD_ERR_IO
         return
      end if
      if (.not. exists) then
         inspect_status = FIELD_ERR_IO
         return
      end if
      inquire(file=requested_path, size=file_bytes, iostat=io_status)
      if (io_status /= 0) then
         inspect_status = FIELD_ERR_IO
         return
      end if
      if (file_bytes <= 0_int64) then
         inspect_status = FIELD_ERR_EMPTY_INPUT
         return
      end if
      if (file_bytes > int(FIELD_MAX_BYTES, int64)) then
         inspect_status = FIELD_ERR_CAPACITY
         return
      end if

      allocate(bytes(int(file_bytes)), stat=io_status)
      if (io_status /= 0) then
         inspect_status = FIELD_ERR_CAPACITY
         return
      end if
      open(newunit=unit, file=requested_path, access='stream', form='unformatted', status='old', &
           action='read', iostat=io_status)
      if (io_status /= 0) then
         deallocate(bytes)
         inspect_status = FIELD_ERR_IO
         return
      end if
      read(unit, iostat=io_status) bytes
      close(unit, iostat=close_status)
      if (io_status /= 0 .or. close_status /= 0) then
         deallocate(bytes)
         inspect_status = FIELD_ERR_IO
         return
      end if

      call field_encode_bytes(bytes, encoding, inspect_status)
      deallocate(bytes)
      if (inspect_status /= FIELD_OK) return
      call field_walsh_forward(encoding%order_field, order_spectrum, inspect_status)
      if (inspect_status /= FIELD_OK) return
      call field_spectral_energy(order_spectrum, order_energy, inspect_status)
      if (inspect_status /= FIELD_OK) return
      call field_parseval_residual(order_spectrum, parseval_residual, inspect_status)
      if (inspect_status /= FIELD_OK) return
      call field_walsh_inverse(order_spectrum, reconstructed, inspect_status)
      if (inspect_status /= FIELD_OK) return
      if (.not. hv_equal(reconstructed, encoding%order_field)) then
         inspect_status = FIELD_ERR_INVALID_SPECTRUM
         return
      end if

      call print_report(encoding, order_energy, parseval_residual, reconstructed)
   end subroutine inspect_file

   subroutine print_report(encoding, order_energy, parseval_residual, reconstructed)
      type(field_encoding_t), intent(in) :: encoding
      integer(int64), intent(in) :: order_energy, parseval_residual
      type(hv_t), intent(in) :: reconstructed

      write(output_unit, '(A)') 'schema: vegai-tokenless-signal-field-inspection/v1'
      write(output_unit, '(A)') 'ingestion: raw-octet-event/no-vocabulary'
      write(output_unit, '(A)') 'learned_tokenizer_used: false'
      write(output_unit, '(A)') 'raw_input_lossless_in_field: false'
      write(output_unit, '(A)') 'raw_source_identity_requirement: external-digest-and-provenance'
      write(output_unit, '(A,I0)') 'bytes: ', encoding%byte_count
      write(output_unit, '(A,I0)') 'field_dimension_bits: ', HV_BITS
      write(output_unit, '(A,I0)') 'max_input_bytes: ', FIELD_MAX_BYTES
      write(output_unit, '(A,Z16.16)') 'order_field_fingerprint_xor64: ', field_fingerprint(encoding%order_field)
      write(output_unit, '(A,Z16.16)') 'distribution_field_fingerprint_xor64: ', &
         field_fingerprint(encoding%distribution_field)
      write(output_unit, '(A,I0)') 'order_walsh_energy: ', order_energy
      write(output_unit, '(A,I0)') 'order_parseval_expected_energy: ', FIELD_WALSH_EXPECTED_ENERGY
      write(output_unit, '(A,I0)') 'order_parseval_residual: ', parseval_residual
      if (hv_equal(reconstructed, encoding%order_field)) then
         write(output_unit, '(A)') 'order_inverse_exact: true'
      else
         write(output_unit, '(A)') 'order_inverse_exact: false'
      end if
      write(output_unit, '(A)') 'scope: finite representation invariants only; no semantic or physical-world claim'
   end subroutine print_report

   subroutine print_usage()
      write(output_unit, '(A)') 'Vegai Tokenless Signal Field inspector'
      write(output_unit, '(A)') 'Usage: vegai_field inspect <raw-byte-file>'
      write(output_unit, '(A)') 'Reads 1 to 32767 raw bytes directly, without BPE/subword tokenization.'
      write(output_unit, '(A)') 'The field is a bounded deterministic sketch; retain source digest/provenance externally.'
   end subroutine print_usage

   subroutine fail_with_status(message, failed_status)
      character(len=*), intent(in) :: message
      integer(int32), intent(in) :: failed_status

      write(error_unit, '(A,A,A,I0,A)') 'vegai_field: ', trim(message), ' (status ', failed_status, ')'
      error stop 1
   end subroutine fail_with_status

   subroutine fail_usage()
      call print_usage()
      error stop 2
   end subroutine fail_usage

end program vegai_field
