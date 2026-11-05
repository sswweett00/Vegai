! SPDX-License-Identifier: MIT
! Portable native tests for bounded corpus preflight and strict CLI parsing.
program corpus_preflight_tests
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real64, output_unit
   use, intrinsic :: ieee_exceptions, only: ieee_invalid, ieee_overflow, ieee_support_halting, ieee_get_halting_mode
   use corpus_preflight, only: CP_OK, CP_ERR_ARGUMENT, corpus_preflight_limits_t, corpus_preflight_stats_t, &
                               corpus_preflight_scan, corpus_preflight_strict_ok
   use fortran_integrity, only: FI_OK, FI_ERR_INVALID_ARGUMENT, sha256_state_t, sha256_file, &
                                sha256_init, sha256_update_bytes, sha256_final_hex, sha256_platform_supported
   use portable_cli, only: CLI_OK, cli_parse_integer32, cli_parse_integer64, cli_parse_real64, cli_valid_utf8
   implicit none (type, external)
   type(corpus_preflight_stats_t) :: stats
   type(corpus_preflight_limits_t) :: limits
   type(sha256_state_t) :: hash
   logical :: native_ok, invalid_before, invalid_after, overflow_before, overflow_after
   integer(int8) :: sample_bytes(3), sample_stride(6), sha_block(8192)
   integer, parameter :: sha_lengths(5) = [55, 56, 63, 64, 65]
   character(len=64), parameter :: sha_expected(5) = [character(len=64) :: &
      '9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318', &
      'b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a', &
      '7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34', &
      'ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb', &
      '635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0']
   character(len=128) :: path
   character(len=64) :: digest
   integer(int64) :: count, wide
   integer(int32) :: status, parsed
   integer :: unit, io_status, checks, failures, index
   real(real64) :: value

   checks = 0
   failures = 0
   call check(sha256_platform_supported(), 'current processor supports the native octet SHA-256 contract')
   if (.not. sha256_platform_supported()) error stop 'unsupported SHA-256 file and integer representation'
   call reserve_fixture(path, unit, io_status)
   call check(io_status == 0, 'reserve a unique writable fixture')
   if (io_status /= 0) error stop 1
   close(unit, iostat=io_status)
   call check(io_status == 0, 'close empty fixture')
   call corpus_preflight_scan(path, stats, status)
   call check(status == CP_OK, 'scan empty file')
   call check(stats%bytes == 0_int64 .and. stats%records == 0_int64, 'empty counts')
   call check(stats%sha256 == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', &
              'empty SHA-256 FIPS vector')
   call check(corpus_preflight_strict_ok(stats), 'empty input has no lexical violations')

   open(newunit=unit, file=trim(path), status='replace', action='write', access='stream', &
        form='unformatted', iostat=io_status)
   if (io_status /= 0) error stop 2
   write(unit) 'abc'
   close(unit)
   call corpus_preflight_scan(path, stats, status)
   call check(status == CP_OK .and. stats%records == 1_int64, 'unterminated final record counted')
   call check(stats%sha256 == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad', &
              'abc SHA-256 FIPS vector')
   call check(stats%accepted_tokens == 1_int64 .and. corpus_preflight_strict_ok(stats), 'simple strict ASCII')

   open(newunit=unit, file=trim(path), status='replace', action='write', access='stream', &
        form='unformatted', iostat=io_status)
   if (io_status /= 0) error stop 3
   write(unit) 'Alpha beta' // achar(13) // achar(10)
   write(unit) repeat('a', 49) // ' ' // repeat('b ', 1025) // achar(10)
   write(unit) repeat('x', 70000) // achar(13) // achar(10)
   write(unit) achar(195) // achar(182) // 'z'
   close(unit)
   call corpus_preflight_scan(path, stats, status)
   call check(status == CP_OK, 'multi-chunk mixed input scan')
   call check(stats%bytes == 72118_int64 .and. stats%records == 4_int64, 'physical byte and record counts')
   call check(stats%overlong_records == 1_int64 .and. stats%maximum_record_bytes == 70001_int64, &
              'overlong record spans a chunk boundary and includes CR')
   call check(stats%records_with_tokenizer_truncation == 1_int64, 'long and excess token record is flagged once')
   call check(stats%accepted_tokens == 1027_int64 .and. stats%dropped_tokens == 2_int64, 'token counters')
   call check(stats%non_ascii_bytes == 2_int64 .and. stats%crlf_records == 2_int64, 'non-ASCII and CRLF counts')
   call check(.not. corpus_preflight_strict_ok(stats), 'mixed violations reject strict contract')
   call sha256_file(path, digest, count, status)
   call check(status == FI_OK .and. digest == stats%sha256 .and. count == stats%bytes, &
              'preflight fingerprint matches independent file interface')

   open(newunit=unit, file=trim(path), status='replace', action='write', access='stream', &
        form='unformatted', iostat=io_status)
   if (io_status /= 0) error stop 4
   write(unit) achar(195) // achar(182) // 'z' // achar(13)
   close(unit)
   call corpus_preflight_scan(path, stats, status)
   call check(status == CP_OK .and. stats%non_ascii_bytes == 2_int64 .and. &
              stats%overlong_records == 0_int64 .and. stats%records_with_tokenizer_truncation == 0_int64, &
              'non-ASCII-only input is recognized without unrelated violations')
   call check(.not. corpus_preflight_strict_ok(stats), 'native strict mode rejects non-ASCII bytes')
   call check(stats%crlf_records == 0_int64, 'a final CR without LF is not CRLF')

   open(newunit=unit, file=trim(path), status='replace', action='write', access='stream', &
        form='unformatted', iostat=io_status)
   if (io_status /= 0) error stop 5
   write(unit) repeat(' ', 8192) // achar(10) // repeat(' ', 8193)
   close(unit)
   call corpus_preflight_scan(path, stats, status)
   call check(status == CP_OK .and. stats%records == 2_int64 .and. stats%overlong_records == 1_int64, &
              'exact 8192-byte record accepted and 8193-byte record rejected')
   limits = corpus_preflight_limits_t()
   limits%max_line_bytes = 9000_int64
   call corpus_preflight_scan(path, stats, status, limits, native_ok)
   call check(status == CP_OK .and. stats%overlong_records == 0_int64 .and. .not. native_ok, &
              'looser custom record limit does not hide native HDC overlength')

   open(newunit=unit, file=trim(path), status='replace', action='write', access='stream', &
        form='unformatted', iostat=io_status)
   if (io_status /= 0) error stop 6
   write(unit) repeat('a', 49) // ' b' // achar(10)
   close(unit)
   limits = corpus_preflight_limits_t()
   limits%token_length_bytes = 64_int64
   call corpus_preflight_scan(path, stats, status, limits, native_ok)
   call check(status == CP_OK .and. stats%accepted_tokens == 2_int64 .and. &
              stats%dropped_tokens == 0_int64 .and. corpus_preflight_strict_ok(stats) .and. .not. native_ok, &
              'looser custom token limit cannot mask native HDC truncation')
   limits%token_length_bytes = 0_int64
   call corpus_preflight_scan(path, stats, status, limits, native_ok)
   call check(status == CP_ERR_ARGUMENT .and. .not. native_ok, 'non-positive custom limits fail closed')

   open(newunit=unit, file=trim(path), status='replace', action='write', access='stream', &
        form='unformatted', iostat=io_status)
   if (io_status /= 0) error stop 7
   write(unit) 'a b' // achar(10)
   close(unit)
   limits = corpus_preflight_limits_t()
   limits%max_tokens_per_line = 1_int64
   call corpus_preflight_scan(path, stats, status, limits, native_ok)
   call check(status == CP_OK .and. stats%accepted_tokens == 1_int64 .and. stats%dropped_tokens == 1_int64 .and. &
              .not. corpus_preflight_strict_ok(stats) .and. native_ok, &
              'tighter custom token limit reports its own violation separately from native readiness')
   open(newunit=unit, file=trim(path), status='old', iostat=io_status)
   if (io_status == 0) close(unit, status='delete', iostat=io_status)
   call check(io_status == 0, 'fixture is removed')

   sha_block = 97_int8
   do index = 1, size(sha_lengths)
      call sha256_init(hash)
      call sha256_update_bytes(hash, sha_block(:sha_lengths(index)), status)
      call sha256_final_hex(hash, digest, status)
      call check(status == FI_OK .and. digest == sha_expected(index), 'SHA-256 padding/block boundary vector')
   end do
   call sha256_init(hash)
   do index = 1, 122
      call sha256_update_bytes(hash, sha_block, status)
      if (status /= FI_OK) exit
   end do
   if (status == FI_OK) call sha256_update_bytes(hash, sha_block(:576), status)
   if (status == FI_OK) call sha256_final_hex(hash, digest, status)
   call check(status == FI_OK .and. digest == &
              'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0', &
              'SHA-256 million-a vector across many updates')

   sample_bytes = [97_int8, 98_int8, 99_int8]
   call sha256_init(hash)
   call sha256_update_bytes(hash, sample_bytes(:1), status)
   call check(status == FI_OK, 'incremental SHA-256 first fragment')
   call sha256_update_bytes(hash, sample_bytes(2:), status)
   call sha256_final_hex(hash, digest, status)
   call check(status == FI_OK .and. digest == &
              'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad', &
              'incremental SHA-256 matches the FIPS vector')
   call sha256_update_bytes(hash, sample_bytes, status)
   call check(status == FI_ERR_INVALID_ARGUMENT, 'finalized SHA-256 state cannot be updated')
   sample_stride = [97_int8, 0_int8, 98_int8, 0_int8, 99_int8, 0_int8]
   call sha256_init(hash)
   call sha256_update_bytes(hash, sample_bytes(:0), status)
   call check(status == FI_OK .and. hash%total_bytes == 0_int64, 'empty SHA-256 update is a no-op')
   call sha256_update_bytes(hash, sample_stride(::2), status)
   call sha256_final_hex(hash, digest, status)
   call check(status == FI_OK .and. digest == &
              'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad', &
              'non-contiguous SHA-256 input uses logical element order')
   call sha256_init(hash)
   hash%buffer_length = -1_int32
   call sha256_update_bytes(hash, sample_bytes, status)
   call check(status == FI_ERR_INVALID_ARGUMENT, 'negative SHA-256 buffer length rejected')
   call sha256_init(hash)
   hash%buffer_length = 64_int32
   call sha256_final_hex(hash, digest, status)
   call check(status == FI_ERR_INVALID_ARGUMENT, 'oversized SHA-256 buffer length rejected')
   call sha256_init(hash)
   hash%total_bytes = 1_int64
   call sha256_final_hex(hash, digest, status)
   call check(status == FI_ERR_INVALID_ARGUMENT, 'inconsistent SHA-256 buffered length rejected')
   call sha256_init(hash)
   hash%hash_words(1) = -1_int64
   call sha256_update_bytes(hash, sample_bytes, status)
   call check(status == FI_ERR_INVALID_ARGUMENT, 'out-of-range SHA-256 word rejected')

   call cli_parse_integer64('9223372036854775807', wide, status)
   call check(status == CLI_OK .and. wide == huge(wide), 'maximum int64 is accepted')
   call cli_parse_integer64('-9223372036854775808', wide, status)
   call check(status == CLI_OK .and. wide == -huge(wide)-1_int64, 'minimum int64 is accepted')
   call cli_parse_integer64('9223372036854775808', wide, status)
   call check(status /= CLI_OK, 'integer overflow rejected')
   call cli_parse_integer64('1,2', wide, status)
   call check(status /= CLI_OK, 'list-directed comma integer rejected')
   call cli_parse_integer64('1/2', wide, status)
   call check(status /= CLI_OK, 'list-directed slash integer rejected')
   call cli_parse_integer32('2147483648', parsed, status)
   call check(status /= CLI_OK, 'int32 overflow rejected')
   call cli_parse_real64('3D-3', value, status)
   call check(status == CLI_OK .and. abs(value - 0.003_real64) < 1.0e-15_real64, 'scientific real accepted')
   if (ieee_support_halting(ieee_invalid)) call ieee_get_halting_mode(ieee_invalid, invalid_before)
   if (ieee_support_halting(ieee_overflow)) call ieee_get_halting_mode(ieee_overflow, overflow_before)
   call cli_parse_real64('1e309', value, status)
   call check(status /= CLI_OK, 'overflowing real rejected under FPE traps')
   if (ieee_support_halting(ieee_invalid)) then
      call ieee_get_halting_mode(ieee_invalid, invalid_after)
      call check(invalid_before .eqv. invalid_after, 'invalid-operation halting mode restored after parse')
   end if
   if (ieee_support_halting(ieee_overflow)) then
      call ieee_get_halting_mode(ieee_overflow, overflow_after)
      call check(overflow_before .eqv. overflow_after, 'overflow halting mode restored after parse')
   end if
   call cli_parse_real64('1e-9999', value, status)
   call check(status /= CLI_OK, 'underflow-to-zero real rejected')
   call cli_parse_real64('NaN', value, status)
   call check(status /= CLI_OK, 'NaN real rejected')
   call cli_parse_real64('inf', value, status)
   call check(status /= CLI_OK, 'infinite real rejected')
   call cli_parse_real64('0.1,0.2', value, status)
   call check(status /= CLI_OK, 'list-directed comma real rejected')
   call cli_parse_real64('0.1/0.2', value, status)
   call check(status /= CLI_OK, 'list-directed slash real rejected')
   call cli_parse_real64('1e', value, status)
   call check(status /= CLI_OK, 'missing exponent rejected')
   call check(cli_valid_utf8('Turkish ' // achar(195) // achar(182)), 'valid UTF-8 accepted')
   call check(.not. cli_valid_utf8(achar(192) // achar(128)), 'overlong UTF-8 rejected')
   call check(.not. cli_valid_utf8(achar(237) // achar(160) // achar(128)), 'UTF-8 surrogate rejected')
   call check(.not. cli_valid_utf8(achar(244) // achar(144) // achar(128) // achar(128)), &
              'out-of-range Unicode codepoint rejected')
   call check(.not. cli_valid_utf8(achar(195)), 'truncated UTF-8 rejected')
   call check(.not. cli_valid_utf8(transfer(-1_int8, 'a')), '0xff octet rejected as UTF-8')
   call check(.not. cli_valid_utf8(transfer(-127_int8 - 1_int8, 'a')), 'isolated UTF-8 continuation rejected')
   call check(cli_valid_utf8(achar(240)//achar(159)//achar(152)//achar(128)), 'four-byte UTF-8 accepted')

   write(output_unit, '(A,I0,A,I0)') 'native preflight/parser checks: ', checks, ', failures: ', failures
   if (failures /= 0) error stop 1
contains
   subroutine check(condition, description)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: description
      checks = checks + 1
      if (.not. condition) then
         failures = failures + 1
         write(output_unit, '(A,A)') 'FAIL: ', description
      end if
   end subroutine check

   subroutine reserve_fixture(file_path, file_unit, result)
      character(len=*), intent(out) :: file_path
      integer, intent(out) :: file_unit, result
      integer(int64) :: clock_value
      integer :: attempt
      call system_clock(count=clock_value)
      do attempt = 1, 1000
         write(file_path, '(A,I0,A,I0,A)') 'native-preflight-', clock_value, '-', attempt, '.bin'
         open(newunit=file_unit, file=trim(file_path), status='new', action='write', &
              access='stream', form='unformatted', iostat=result)
         if (result == 0) return
      end do
   end subroutine reserve_fixture
end program corpus_preflight_tests
