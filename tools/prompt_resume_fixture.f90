! SPDX-License-Identifier: MIT
! Native CTest helper: a checkpoint with non-default optimizer settings, and
! a strict check that a resumed prompt run retained those settings.
program prompt_resume_fixture
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real64, error_unit
   use fixed_budget_context
   use fixed_budget_optimizer
   use fixed_budget_training
   use portable_cli, only: CLI_OK, cli_parse_integer64
   implicit none (type, external)
   type(fbc_config_t) :: config
   type(fbc_model_t) :: model
   type(fbc_optimizer_t) :: optimizer
   type(fbc_update_config_t) :: update_config
   type(fbc_metrics_t) :: counters
   character(len=1024) :: action, path, rule_text, seed_text
   integer(int32) :: status, rule
   integer(int64) :: fixture_seed
   integer :: argument_status, unit, io_status, i, bit, position, fixture_limit
   integer(int8) :: byte_fixture(33)
   character(len=20) :: collision_line
   character(len=4096) :: large_line
   real(real64) :: expected_rate

   if (command_argument_count() < 3) error stop 'expected action, path and arguments'
   call get_command_argument(1, action, status=argument_status)
   if (argument_status /= 0) error stop 'invalid action'
   call get_command_argument(2, path, status=argument_status)
   if (argument_status /= 0 .or. len_trim(path) == 0) error stop 'invalid path'
   if (trim(action) == 'bytes') then
      ! CMake's file(WRITE) emits native line endings (CRLF on Windows), so
      ! byte-exact test corpora are written here from hexadecimal tokens.
      ! A token is either hex text or "<one hex byte>*<repeat count>".
      call write_hex_tokens(trim(path), 3)
      stop 0
   end if
   if (command_argument_count() /= 3 .and. command_argument_count() /= 4) &
      error stop 'expected action, path, training rule and optional seed'
   fixture_seed = 77_int64
   if (command_argument_count() == 4) then
      call get_command_argument(4, seed_text, status=argument_status)
      if (argument_status /= 0) error stop 'invalid seed'
      call cli_parse_integer64(trim(seed_text), fixture_seed, status)
      if (status /= CLI_OK) error stop 'invalid seed'
   end if
   call get_command_argument(3, rule_text, status=argument_status)
   if (argument_status /= 0) error stop 'invalid rule'
   select case (trim(rule_text))
   case ('5')
      rule = FBC_TRAINING_RULE_V5
   case ('6')
      rule = FBC_TRAINING_RULE_V6
   case ('7')
      rule = FBC_TRAINING_RULE_V7
   case default
      error stop 'invalid rule'
   end select
   select case (trim(action))
   case ('nul-only')
      open(newunit=unit, file=trim(path), status='new', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) error stop 'NUL fixture open failed'
      write(unit, iostat=io_status) [65_int8, 0_int8, 66_int8, 10_int8]
      if (io_status /= 0) error stop 'NUL fixture write failed'
      close(unit, iostat=io_status)
      if (io_status /= 0) error stop 'NUL fixture close failed'
   case ('corpus')
      byte_fixture = 65_int8
      byte_fixture(1) = 0_int8
      byte_fixture(2) = -1_int8
      open(newunit=unit, file=trim(path), status='new', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) error stop 'byte fixture open failed'
      write(unit, iostat=io_status) byte_fixture
      if (io_status /= 0) error stop 'byte fixture write failed'
      close(unit, iostat=io_status)
      if (io_status /= 0) error stop 'byte fixture close failed'
   case ('overlap-unique', 'overlap-unique-exact', 'overlap-collisions', 'overlap-collisions-exact')
      open(newunit=unit, file=trim(path), status='new', action='write', form='formatted', iostat=io_status)
      if (io_status /= 0) error stop 'overlap fixture open failed'
      if (index(trim(action), 'overlap-unique') == 1) then
         fixture_limit = 100001
         if (trim(action) == 'overlap-unique-exact') fixture_limit = 100000
         do i = 1, fixture_limit
            write(unit, '(I0)', iostat=io_status) i
            if (io_status /= 0) error stop 'overlap fixture write failed'
         end do
      else
         ! A nine-character gap rotates by 63 bits, aligning bit 1 with bit 0.
         ! Eleven independent pairs yield 1,025 distinct ASCII strings with
         ! exactly the same complete 64-bit overlap-index hash.
         fixture_limit = 1024
         if (trim(action) == 'overlap-collisions-exact') fixture_limit = 1023
         do i = 0, fixture_limit
            collision_line = repeat('A', len(collision_line))
            do bit = 0, 10
               if (.not. btest(i, bit)) cycle
               position = bit + 1
               collision_line(position:position) = achar(ieor(iachar(collision_line(position:position)), 2))
               position = position + 9
               collision_line(position:position) = achar(ieor(iachar(collision_line(position:position)), 1))
            end do
            write(unit, '(A)', iostat=io_status) collision_line
            if (io_status /= 0) error stop 'collision fixture write failed'
         end do
      end if
      close(unit, iostat=io_status)
      if (io_status /= 0) error stop 'overlap fixture close failed'
   case ('overlap-bytes', 'overlap-bytes-exact')
      open(newunit=unit, file=trim(path), status='new', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) error stop 'overlap bytes fixture open failed'
      large_line = repeat('x', len(large_line))
      fixture_limit = 16385
      if (trim(action) == 'overlap-bytes-exact') fixture_limit = 16384
      do i = 1, fixture_limit
         write(large_line(1:8), '(I8.8)', iostat=io_status) i
         if (io_status /= 0) error stop 'overlap fixture numbering failed'
         write(unit, iostat=io_status) large_line, achar(10)
         if (io_status /= 0) error stop 'overlap bytes fixture write failed'
      end do
      close(unit, iostat=io_status)
      if (io_status /= 0) error stop 'overlap bytes fixture close failed'
   case ('write', 'force-a', 'force-nul', 'force-ff', 'force-c3', 'force-e0', 'force-ed', 'force-f0', &
         'force-f4', 'force-huge', 'force-overflow')
      config = fbc_config_t(vocabulary_size=256_int32, width=32_int32, heads=2_int32, ffn_width=64_int32, &
                            local_window=32_int32, memory_scales=4_int32, memory_slots=64_int32, &
                            logical_horizon=8192_int64)
      call fbc_model_init(model, config, fixture_seed, status)
      if (status /= FBC_OK) error stop 'model init failed'
      model%training_rule = rule
      if (trim(action) == 'force-a') model%output_bias(65_int32) = 100.0_real64
      if (trim(action) == 'force-nul') model%output_bias(256_int32) = 100.0_real64
      if (trim(action) == 'force-ff') model%output_bias(255_int32) = 100.0_real64
      if (trim(action) == 'force-c3') model%output_bias(195_int32) = 100.0_real64
      if (trim(action) == 'force-e0') model%output_bias(224_int32) = 100.0_real64
      if (trim(action) == 'force-ed') model%output_bias(237_int32) = 100.0_real64
      if (trim(action) == 'force-f0') model%output_bias(240_int32) = 100.0_real64
      if (trim(action) == 'force-f4') model%output_bias(244_int32) = 100.0_real64
      if (trim(action) == 'force-huge') model%output_bias(65_int32) = 1.0e100_real64
      if (trim(action) == 'force-overflow') model%output_bias(65_int32) = 1.0e308_real64
      call fbc_optimizer_init(optimizer, config, status)
      if (status /= 0_int32) error stop 'optimizer init failed'
      update_config%learning_rate = 0.002_real64
      update_config%momentum = 0.71_real64
      update_config%energy_decay = 0.81_real64
      update_config%restart_damp = 0.33_real64
      update_config%weight_decay = 0.02_real64
      update_config%epsilon = 2.0e-7_real64
      update_config%gradient_clip = 0.9_real64
      counters = fbc_metrics_t()
      call fbc_checkpoint_save(trim(path), model, optimizer, update_config, counters, status)
      if (status /= FBT_OK) error stop 'fixture save failed'
   case ('corrupt-loss')
      ! Compute the stream position using this compiler's own file-storage
      ! units, not a hard-coded byte offset or assumed endianness.
      inquire(iolength=position) FBC_CHECKPOINT_MAGIC, rule, config%vocabulary_size, config%width, &
         config%heads, config%ffn_width, config%local_window, config%memory_scales, config%memory_slots, &
         config%logical_horizon, config%rms_epsilon, config%slot_decay, config%slot_temperature, &
         model%effective_horizon, 0_int32, 0_int64, 0_int64
      open(newunit=unit, file=trim(path), status='old', action='readwrite', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) error stop 'corrupt fixture open failed'
      write(unit, pos=int(position, int64) + 1_int64, iostat=io_status) &
         transfer(int(z'7FF0000000000001', int64), 0.0_real64)
      if (io_status /= 0) error stop 'corrupt fixture write failed'
      close(unit, iostat=io_status)
      if (io_status /= 0) error stop 'corrupt fixture close failed'
   case ('check-new-default', 'check-new-custom', 'check-new-explicit')
      call fbc_checkpoint_load(trim(path), model, optimizer, update_config, counters, status)
      if (status /= FBT_OK) error stop 'new checkpoint load failed'
      expected_rate = 0.006_real64
      if (trim(action) == 'check-new-custom') expected_rate = 0.003_real64
      if (trim(action) == 'check-new-explicit') expected_rate = 0.007_real64
      if (model%training_rule /= FBC_TRAINING_RULE_V6 .or. update_config%learning_rate /= expected_rate) &
         error stop 'new checkpoint selected the wrong rate or training rule'
      if (counters%updates /= 1_int64 .or. counters%tokens < 1_int64) error stop 'new checkpoint was not trained'
   case ('check-rule')
      call fbc_checkpoint_load(trim(path), model, optimizer, update_config, counters, status)
      if (status /= FBT_OK .or. model%training_rule /= rule) error stop 'checkpoint selected the wrong rule'
   case ('check-original', 'check-override')
      call fbc_checkpoint_load(trim(path), model, optimizer, update_config, counters, status)
      if (status /= FBT_OK) error stop 'fixture load failed'
      expected_rate = 0.002_real64
      if (trim(action) == 'check-override') expected_rate = 0.007_real64
      if (model%training_rule /= rule .or. update_config%learning_rate /= expected_rate .or. &
          update_config%momentum /= 0.71_real64 .or. update_config%energy_decay /= 0.81_real64 .or. &
          update_config%restart_damp /= 0.33_real64 .or. update_config%weight_decay /= 0.02_real64 .or. &
          update_config%epsilon /= 2.0e-7_real64 .or. update_config%gradient_clip /= 0.9_real64) then
         write(error_unit, '(A)') 'resumed checkpoint did not preserve its selected optimizer settings'
         error stop 1
      end if
      if (counters%updates /= 1_int64 .or. counters%tokens /= 32_int64) error stop 'resume budget mismatch'
   case default
      error stop 'unknown action'
   end select

contains

   subroutine write_hex_tokens(target, first_token)
      character(len=*), intent(in) :: target
      integer, intent(in) :: first_token
      character(len=2048) :: token
      integer :: index, argument_status, file_unit, io_status, star
      integer(int64) :: count

      open(newunit=file_unit, file=target, status='new', action='write', access='stream', &
           form='unformatted', iostat=io_status)
      if (io_status /= 0) error stop 'byte fixture open failed'
      do index = first_token, command_argument_count()
         call get_command_argument(index, token, status=argument_status)
         if (argument_status /= 0) error stop 'invalid byte fixture token'
         star = index(token, '*')
         if (star > 0) then
            call cli_parse_integer64(trim(token(star + 1:)), count, argument_status)
            if (argument_status /= CLI_OK) error stop 'invalid byte fixture repeat count'
            if (count < 0_int64 .or. count > 67108864_int64) error stop 'byte fixture repeat out of range'
            call write_repeated_byte(file_unit, trim(token(:star - 1)), count)
         else
            call write_hex_text(file_unit, trim(token))
         end if
      end do
      close(file_unit, iostat=io_status)
      if (io_status /= 0) error stop 'byte fixture close failed'
   end subroutine write_hex_tokens

   subroutine write_hex_text(file_unit, text)
      integer, intent(in) :: file_unit
      character(len=*), intent(in) :: text
      character(len=8192) :: chunk
      integer :: position, produced, high, low, io_status

      if (len(text) == 0 .or. mod(len(text), 2) /= 0) error stop 'byte fixture hex length is invalid'
      position = 1
      produced = 0
      do while (position < len(text))
         high = hex_value(text(position:position))
         low = hex_value(text(position + 1:position + 1))
         if (high < 0 .or. low < 0) error stop 'byte fixture token is not hexadecimal'
         produced = produced + 1
         chunk(produced:produced) = octet(high * 16 + low)
         position = position + 2
         if (produced == len(chunk)) then
            write(file_unit, iostat=io_status) chunk
            if (io_status /= 0) error stop 'byte fixture write failed'
            produced = 0
         end if
      end do
      if (produced > 0) then
         write(file_unit, iostat=io_status) chunk(:produced)
         if (io_status /= 0) error stop 'byte fixture write failed'
      end if
   end subroutine write_hex_text

   subroutine write_repeated_byte(file_unit, pair, count)
      integer, intent(in) :: file_unit
      character(len=*), intent(in) :: pair
      integer(int64), intent(in) :: count
      character(len=8192) :: block
      integer :: high, low, io_status
      integer(int64) :: produced, remaining

      if (len(pair) /= 2) error stop 'byte fixture repeat needs exactly one byte'
      high = hex_value(pair(1:1))
      low = hex_value(pair(2:2))
      if (high < 0 .or. low < 0) error stop 'byte fixture repeat byte is not hexadecimal'
      block = repeat(octet(high * 16 + low), len(block))
      remaining = count
      do while (remaining > 0_int64)
         produced = min(remaining, int(len(block), int64))
         write(file_unit, iostat=io_status) block(1:int(produced))
         if (io_status /= 0) error stop 'byte fixture write failed'
         remaining = remaining - produced
      end do
   end subroutine write_repeated_byte

   integer function hex_value(digit) result(value)
      character(len=1), intent(in) :: digit

      select case (digit)
      case ('0':'9')
         value = iachar(digit) - iachar('0')
      case ('a':'f')
         value = iachar(digit) - iachar('a') + 10
      case ('A':'F')
         value = iachar(digit) - iachar('A') + 10
      case default
         value = -1
      end select
   end function hex_value

   character(len=1) function octet(value) result(byte)
      integer, intent(in) :: value

      byte = transfer(int(value, int8), 'a')
   end function octet
end program prompt_resume_fixture
