# SPDX-License-Identifier: MIT
# Regression for resumed optimizer settings, byte-zero handling and error
# propagation. WORK_ROOT is trusted and each invocation owns a random scratch.
if(NOT DEFINED ENGINE OR NOT DEFINED FIXTURE OR NOT DEFINED WORK_ROOT)
  message(FATAL_ERROR "ENGINE, FIXTURE and WORK_ROOT are required")
endif()
# CMake writes native line endings (CRLF on Windows); the native helper
# writes the exact bytes the contract asserts.
include("${CMAKE_CURRENT_LIST_DIR}/native_exact_text.cmake")
string(RANDOM LENGTH 24 ALPHABET 0123456789abcdef nonce)
set(work "${WORK_ROOT}/native-prompt-${nonce}")
file(MAKE_DIRECTORY "${work}")
string(REPEAT "A Turkish byte training line for the resume fixture.\n" 3 text)
native_write_text("${work}/text.txt" "${text}")
native_write_text("${work}/probe.tsv" "fact\tx\tA\n")
execute_process(COMMAND "${FIXTURE}" corpus "${work}/bytes.bin" 6 RESULT_VARIABLE result ERROR_VARIABLE err)
if(NOT "${result}" STREQUAL "0")
  message(FATAL_ERROR "cannot create byte fixture: ${result}: ${err}")
endif()

foreach(rule IN ITEMS 5 6 7)
  execute_process(COMMAND "${FIXTURE}" write "${work}/input-${rule}.vgfb" "${rule}"
    RESULT_VARIABLE result ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "cannot create v${rule} checkpoint: ${result}: ${err}")
  endif()
  foreach(mode IN ITEMS original override)
    set(extra)
    if(mode STREQUAL "override")
      set(extra --lr 0.007)
    endif()
    execute_process(COMMAND "${ENGINE}" train --corpus "${work}/text.txt"
      --checkpoint "${work}/output-${rule}-${mode}.vgfb" --resume "${work}/input-${rule}.vgfb"
      --train-chunk-tokens 32 --max-updates 1 ${extra}
      RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT "${result}" STREQUAL "0")
      message(FATAL_ERROR "v${rule} ${mode} resume failed: ${result}: ${out}: ${err}")
    endif()
    execute_process(COMMAND "${FIXTURE}" "check-${mode}" "${work}/output-${rule}-${mode}.vgfb" "${rule}"
      RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT "${result}" STREQUAL "0")
      message(FATAL_ERROR "v${rule} ${mode} resume changed optimizer state: ${result}: ${out}: ${err}")
    endif()
  endforeach()

  execute_process(COMMAND "${FIXTURE}" force-a "${work}/force-a-${rule}.vgfb" "${rule}"
    RESULT_VARIABLE result ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "force-a fixture failed: ${result}: ${err}")
  endif()
  execute_process(COMMAND "${ENGINE}" ask --checkpoint "${work}/force-a-${rule}.vgfb"
    --prompt x --max-new 1 RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0" OR NOT out MATCHES "\"response\": \"A\"")
    message(FATAL_ERROR "v${rule} byte decoding failed: ${result}: ${out}: ${err}")
  endif()
  execute_process(COMMAND "${ENGINE}" ask --checkpoint "${work}/force-a-${rule}.vgfb"
    --prompt x --max-new 1 --temperature 1e-310 --seed 9223372036854775807
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0" OR NOT out MATCHES "\"response\": \"A\"")
    message(FATAL_ERROR "v${rule} tiny-temperature/high-seed sampling failed: ${result}: ${out}: ${err}")
  endif()
  execute_process(COMMAND "${ENGINE}" probe --checkpoint "${work}/force-a-${rule}.vgfb"
    --probe "${work}/probe.tsv" --max-new 1 RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0" OR NOT out MATCHES "\"byte_accuracy\": 1\\.000000")
    message(FATAL_ERROR "v${rule} teacher-forced byte scoring failed: ${result}: ${out}: ${err}")
  endif()
  execute_process(COMMAND "${FIXTURE}" force-nul "${work}/force-nul-${rule}.vgfb" "${rule}"
    RESULT_VARIABLE result ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "force-nul fixture failed: ${result}: ${err}")
  endif()
  execute_process(COMMAND "${ENGINE}" ask --checkpoint "${work}/force-nul-${rule}.vgfb"
    --prompt x --max-new 1 RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0" OR NOT out MATCHES "\"response\": \"\"")
    message(FATAL_ERROR "v${rule} emitted an invalid NUL response: ${result}: ${out}: ${err}")
  endif()
endforeach()

# A malformed numeric checkpoint must never be resumed or used to overwrite an
# existing destination. The fixture computes the offset using the same ABI.
execute_process(COMMAND "${FIXTURE}" write "${work}/corrupt.vgfb" 6 RESULT_VARIABLE result ERROR_VARIABLE err)
if(NOT "${result}" STREQUAL "0")
  message(FATAL_ERROR "cannot create corrupt-header fixture: ${result}: ${err}")
endif()
execute_process(COMMAND "${FIXTURE}" corrupt-loss "${work}/corrupt.vgfb" 6 RESULT_VARIABLE result ERROR_VARIABLE err)
if(NOT "${result}" STREQUAL "0")
  message(FATAL_ERROR "cannot corrupt loss header: ${result}: ${err}")
endif()
native_write_text("${work}/corrupt-destination.vgfb" "sentinel must survive\n")
execute_process(COMMAND "${ENGINE}" train --corpus "${work}/text.txt"
  --checkpoint "${work}/corrupt-destination.vgfb" --resume "${work}/corrupt.vgfb"
  --train-chunk-tokens 32 --max-updates 1 RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0")
  message(FATAL_ERROR "corrupt checkpoint must be rejected by train: ${result}: ${out}: ${err}")
endif()
file(READ "${work}/corrupt-destination.vgfb" preserved)
if(NOT preserved STREQUAL "sentinel must survive\n")
  message(FATAL_ERROR "rejected checkpoint overwrote the destination")
endif()
execute_process(COMMAND "${ENGINE}" ask --checkpoint "${work}/corrupt.vgfb" --prompt x --max-new 1
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0")
  message(FATAL_ERROR "corrupt checkpoint must be rejected by ask: ${result}: ${out}: ${err}")
endif()

# Only a fresh default-shape/default-chunk run without resets adopts the
# dev-selected and held-out-tested 0.006 rate. Overrides and other shapes keep
# their explicit/previous values.
foreach(mode IN ITEMS default custom-chunk custom-shape custom-reset explicit)
  set(extra)
  set(check check-new-custom)
  if(mode STREQUAL "default")
    set(check check-new-default)
  elseif(mode STREQUAL "custom-chunk")
    set(extra --train-chunk-tokens 32)
  elseif(mode STREQUAL "custom-shape")
    set(extra --width 16)
  elseif(mode STREQUAL "custom-reset")
    set(extra --reset-every 32)
  else()
    set(extra --lr 0.007)
    set(check check-new-explicit)
  endif()
  execute_process(COMMAND "${ENGINE}" train --corpus "${work}/text.txt" --checkpoint "${work}/new-${mode}.vgfb"
    --max-updates 1 ${extra} RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "fresh ${mode} training failed: ${result}: ${out}: ${err}")
  endif()
  if(mode STREQUAL "default")
    if(NOT out MATCHES "\"whole_file\": true" OR NOT out MATCHES "\"file_fraction\": 1\\.000000" OR
       out MATCHES "note: partial pass")
      message(FATAL_ERROR "completed N-1 prediction pairs were reported partial: ${out}")
    endif()
  elseif(mode STREQUAL "custom-chunk")
    if(NOT out MATCHES "\"whole_file\": false" OR NOT out MATCHES "note: partial pass")
      message(FATAL_ERROR "a genuinely partial run was reported as complete: ${out}")
    endif()
  endif()
  execute_process(COMMAND "${FIXTURE}" "${check}" "${work}/new-${mode}.vgfb" 6
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "fresh ${mode} rate selection failed: ${result}: ${out}: ${err}")
  endif()
endforeach()

# Fresh users may select either rule explicitly; the resulting checkpoint
# and JSON must report it, rather than silently using the default.
foreach(rule IN ITEMS 6 7)
  execute_process(COMMAND "${ENGINE}" train --corpus "${work}/text.txt" --checkpoint "${work}/fresh-rule-${rule}.vgfb"
    --training-rule "${rule}" --train-chunk-tokens 32 --max-updates 1
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0" OR NOT out MATCHES "\"training_rule\": ${rule}")
    message(FATAL_ERROR "fresh v${rule} selection failed: ${result}: ${out}: ${err}")
  endif()
  execute_process(COMMAND "${FIXTURE}" check-rule "${work}/fresh-rule-${rule}.vgfb" "${rule}"
    RESULT_VARIABLE result ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "fresh v${rule} checkpoint lost its selected rule: ${result}: ${err}")
  endif()
endforeach()

# V6/V7 can train NUL and byte 255; V5 cannot encode NUL and must fail
# before replacing an existing output checkpoint, not merely log and succeed.
foreach(rule IN ITEMS 6 7)
  execute_process(COMMAND "${ENGINE}" train --corpus "${work}/bytes.bin" --checkpoint "${work}/bytes-v${rule}.vgfb"
    --resume "${work}/input-${rule}.vgfb" --train-chunk-tokens 32 --max-updates 1
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "v${rule} could not train the 0x00/0xFF fixture: ${result}: ${out}: ${err}")
  endif()
  execute_process(COMMAND "${FIXTURE}" check-original "${work}/bytes-v${rule}.vgfb" "${rule}"
    RESULT_VARIABLE result ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "v${rule} byte training did not complete its update: ${result}: ${err}")
  endif()
endforeach()
native_write_text("${work}/protected.vgfb" "user-sentinel\n")
execute_process(COMMAND "${ENGINE}" train --corpus "${work}/bytes.bin" --checkpoint "${work}/protected.vgfb"
  --resume "${work}/input-5.vgfb" --train-chunk-tokens 32 --max-updates 1
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "cannot encode NUL")
  message(FATAL_ERROR "v5 NUL was not rejected normally: ${result}: ${out}: ${err}")
endif()
file(READ "${work}/protected.vgfb" sentinel)
if(NOT sentinel STREQUAL "user-sentinel\n")
  message(FATAL_ERROR "failed training replaced an existing checkpoint")
endif()
# Rule changes on resume would silently change model semantics. Reject even
# an explicit matching rule, and leave an existing destination untouched.
foreach(rule IN ITEMS 6 7)
  execute_process(COMMAND "${ENGINE}" train --corpus "${work}/text.txt" --checkpoint "${work}/protected.vgfb"
    --resume "${work}/input-6.vgfb" --training-rule "${rule}"
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "cannot change")
    message(FATAL_ERROR "resume accepted an explicit rule: ${result}: ${out}: ${err}")
  endif()
endforeach()
foreach(rule IN ITEMS 5 8)
  execute_process(COMMAND "${ENGINE}" train --corpus "${work}/text.txt" --checkpoint "${work}/protected.vgfb"
    --training-rule "${rule}" RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "must be 6 or 7")
    message(FATAL_ERROR "fresh training accepted invalid rule ${rule}: ${result}: ${out}: ${err}")
  endif()
endforeach()
file(READ "${work}/protected.vgfb" sentinel)
if(NOT sentinel STREQUAL "user-sentinel\n")
  message(FATAL_ERROR "invalid training rule replaced an existing checkpoint")
endif()
execute_process(COMMAND "${ENGINE}" ask --checkpoint "${work}/input-6.vgfb" --prompt x --training-rule 7
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "only valid")
  message(FATAL_ERROR "ask accepted a training-only rule: ${result}: ${out}: ${err}")
endif()
# Invalid numeric inputs must fail normally before touching any checkpoint,
# even with compiler floating-point traps enabled. In particular int32 options
# must not wrap 2**32 + a plausible value into a different valid setting.
foreach(bad IN ITEMS "1e309" "1e-9999" "NaN" "0.003,1")
  execute_process(COMMAND "${ENGINE}" train --corpus "${work}/text.txt" --checkpoint "${work}/protected.vgfb"
    --lr "${bad}" RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "--lr")
    message(FATAL_ERROR "invalid learning rate '${bad}' was not rejected normally: ${result}: ${out}: ${err}")
  endif()
endforeach()
foreach(option IN ITEMS --width --heads --local-window --memory-scales --memory-slots --ffn
                        --train-chunk-tokens --max-new --training-rule)
  execute_process(COMMAND "${ENGINE}" train --corpus "${work}/text.txt" --checkpoint "${work}/protected.vgfb"
    "${option}" 4294967328 RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "out of range")
    message(FATAL_ERROR "int32 narrowing '${option}' was not rejected: ${result}: ${out}: ${err}")
  endif()
endforeach()
string(REPEAT x 1100 long_arg)
execute_process(COMMAND "${ENGINE}" "${long_arg}" RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "argument is too long")
  message(FATAL_ERROR "overlong command was not rejected normally: ${result}: ${out}: ${err}")
endif()
execute_process(COMMAND "${ENGINE}" train --corpus "${long_arg}" --checkpoint "${work}/protected.vgfb"
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "argument is too long")
  message(FATAL_ERROR "truncated argument was not rejected: ${result}: ${out}: ${err}")
endif()
function(expect_bad_argument expected)
  execute_process(COMMAND "${ENGINE}" ${ARGN} RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "${expected}")
    message(FATAL_ERROR "invalid argument was not rejected: ${result}: ${out}: ${err}")
  endif()
endfunction()
# A trailing path blank is significant on some systems, but the file APIs use
# TRIM. Reject it before opening the similarly named file or checkpoint.
expect_bad_argument("file path must not end" train --corpus "${work}/text.txt " --checkpoint "${work}/protected.vgfb")
expect_bad_argument("file path must not end" train --corpus "${work}/text.txt" --checkpoint "${work}/protected.vgfb ")
expect_bad_argument("file path must not end" train --corpus "${work}/text.txt" --checkpoint "${work}/protected.vgfb"
  --resume "${work}/input-6.vgfb ")
expect_bad_argument("file path must not end" probe --checkpoint "${work}/force-a-6.vgfb" --probe "${work}/probe.tsv ")
expect_bad_argument("file path must not end" ask --checkpoint "${work}/force-a-6.vgfb" --prompt x
  --warmup-from "${work}/text.txt ")
# POSIX argv can contain arbitrary octets; Windows process arguments are Unicode.
if(NOT WIN32)
  string(ASCII 255 invalid_argument)
  expect_bad_argument("argument is not valid UTF-8" train --corpus "${work}/text.txt"
    --checkpoint "${work}/protected${invalid_argument}")
endif()
file(READ "${work}/protected.vgfb" sentinel)
if(NOT sentinel STREQUAL "user-sentinel\n")
  message(FATAL_ERROR "invalid options replaced an existing checkpoint")
endif()
native_write_text("${work}/same-file.txt" "training corpus sentinel\n")
execute_process(COMMAND "${ENGINE}" train --corpus "${work}/same-file.txt" --checkpoint "${work}/same-file.txt"
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "cannot be the training corpus")
  message(FATAL_ERROR "identical input/output was not rejected: ${result}: ${out}: ${err}")
endif()
file(READ "${work}/same-file.txt" sentinel)
if(NOT sentinel STREQUAL "training corpus sentinel\n")
  message(FATAL_ERROR "identical-path input corpus was overwritten")
endif()

# Empty/one-byte files and bounded one-byte prefixes have no predictions;
# exact-end and zero-update resumes must never claim they trained extra bytes.
native_write_text("${work}/empty.txt" "")
native_write_text("${work}/one.txt" "A")
native_write_text("${work}/two.txt" "AB")
native_write_text("${work}/three.txt" "ABC")
function(check_tiny_report name corpus expected_tokens expected_updates expected_end expected_whole expected_bound expected_fraction)
  execute_process(COMMAND "${ENGINE}" train --corpus "${work}/${corpus}.txt" --checkpoint "${work}/${name}.vgfb"
    ${ARGN} RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "tiny ${name} training failed: ${result}: ${out}: ${err}")
  endif()
  string(FIND "${out}" "CORPUS_TRAIN_JSON " at)
  string(FIND "${out}" "\n}" close)
  if(at LESS 0 OR close LESS at)
    message(FATAL_ERROR "tiny ${name} report is missing: ${out}")
  endif()
  string(LENGTH "CORPUS_TRAIN_JSON " prefix)
  math(EXPR begin "${at} + ${prefix}")
  math(EXPR length "${close} - ${begin} + 2")
  string(SUBSTRING "${out}" ${begin} ${length} json)
  foreach(key IN ITEMS session_tokens session_updates end_byte whole_file bounded_by_bytes file_fraction)
    string(JSON ${key} ERROR_VARIABLE error GET "${json}" "${key}")
    if(NOT error STREQUAL "NOTFOUND")
      message(FATAL_ERROR "tiny ${name} invalid report ${key}: ${error}: ${out}")
    endif()
  endforeach()
  if(NOT session_tokens EQUAL expected_tokens OR NOT session_updates EQUAL expected_updates OR
     NOT end_byte EQUAL expected_end OR NOT "${whole_file}" STREQUAL "${expected_whole}" OR
     NOT "${bounded_by_bytes}" STREQUAL "${expected_bound}" OR NOT file_fraction EQUAL expected_fraction)
    message(FATAL_ERROR "tiny ${name} report has an incorrect boundary or budget: ${out}")
  endif()
  string(JSON time_type ERROR_VARIABLE error TYPE "${json}" session_seconds_precise)
  string(JSON rate_type TYPE "${json}" session_tokens_per_second_precise)
  if(NOT error STREQUAL "NOTFOUND" OR NOT time_type STREQUAL rate_type OR
     (NOT time_type STREQUAL "NUMBER" AND NOT time_type STREQUAL "NULL"))
    message(FATAL_ERROR "tiny ${name} lacks paired precise timing fields: ${out}")
  endif()
  if(time_type STREQUAL "NUMBER" AND expected_tokens EQUAL 0)
    string(JSON rate GET "${json}" session_tokens_per_second_precise)
    if(NOT rate EQUAL 0)
      message(FATAL_ERROR "empty ${name} session must report zero precise throughput: ${out}")
    endif()
  endif()
endfunction()
check_tiny_report(empty empty 0 0 0 ON OFF 1.0)
check_tiny_report(one one 0 0 0 ON OFF 1.0)
check_tiny_report(two two 1 1 1 ON OFF 1.0)
check_tiny_report(limit-one two 0 0 0 OFF ON 0.0 --bytes 1)
check_tiny_report(limit-two three 1 1 1 OFF ON 0.5 --bytes 2)
string(REPEAT "A" 2000001 long_corpus)
native_write_text("${work}/large-denominator.txt" "${long_corpus}")
check_tiny_report(tiny-fraction large-denominator 1 1 1 OFF ON 5e-7 --bytes 2)
check_tiny_report(at-end two 0 0 1 ON OFF 1.0 --start-byte 1)
# --max-updates is an absolute cumulative checkpoint limit, not an additional
# per-session allowance; the previous checkpoint already has one update.
check_tiny_report(resume-capped two 0 0 0 OFF OFF 0.0 --resume "${work}/two.vgfb" --max-updates 1)
execute_process(COMMAND "${ENGINE}" train --corpus "${work}/does-not-exist.txt" --checkpoint "${work}/protected.vgfb"
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "cannot stat")
  message(FATAL_ERROR "missing corpus was not rejected: ${result}: ${out}: ${err}")
endif()
file(READ "${work}/protected.vgfb" sentinel)
if(NOT sentinel STREQUAL "user-sentinel\n")
  message(FATAL_ERROR "missing corpus replaced an existing checkpoint")
endif()

# The same byte mapping is used in warm-up, not only during training.
execute_process(COMMAND "${ENGINE}" ask --checkpoint "${work}/force-a-6.vgfb" --prompt x --max-new 1
  --warmup-from "${work}/bytes.bin" --warmup-bytes 2
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" STREQUAL "0" OR NOT out MATCHES "\"response\": \"A\"")
  message(FATAL_ERROR "v6 byte warm-up failed: ${result}: ${out}: ${err}")
endif()
execute_process(COMMAND "${ENGINE}" ask --checkpoint "${work}/force-a-5.vgfb" --prompt x --max-new 1
  --warmup-from "${work}/bytes.bin" --warmup-bytes 2
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR NOT err MATCHES "warm-up byte")
  message(FATAL_ERROR "v5 NUL warm-up was not rejected normally: ${result}: ${out}: ${err}")
endif()
# Linux reports a FIFO as zero-size. It must be rejected before OPEN rather
# than waiting forever for a writer; preserve an existing output on failure.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  find_program(mkfifo_program mkfifo)
  if(mkfifo_program)
    execute_process(COMMAND "${mkfifo_program}" "${work}/checkpoint.fifo" RESULT_VARIABLE fifo_result)
    if("${fifo_result}" STREQUAL "0")
      execute_process(COMMAND "${ENGINE}" train --corpus "${work}/text.txt"
        --checkpoint "${work}/protected.vgfb" --resume "${work}/checkpoint.fifo" --max-updates 1
        RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 10)
      if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0")
        message(FATAL_ERROR "FIFO checkpoint was accepted or blocked: ${result}: ${out}: ${err}")
      endif()
      file(READ "${work}/protected.vgfb" sentinel)
      if(NOT sentinel STREQUAL "user-sentinel\n")
        message(FATAL_ERROR "rejected FIFO checkpoint overwrote the destination")
      endif()
      if(DEFINED CONTEXT)
        execute_process(COMMAND "${CONTEXT}" inspect --checkpoint "${work}/checkpoint.fifo"
          RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 10)
        if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0")
          message(FATAL_ERROR "FIFO checkpoint inspect was accepted or blocked: ${result}: ${out}: ${err}")
        endif()
      endif()
    endif()
  endif()
endif()
file(REMOVE_RECURSE "${work}")
message(STATUS "native prompt resume, byte boundaries and fail-closed writes passed")
