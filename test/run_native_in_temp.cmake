# SPDX-License-Identifier: MIT
# Many older tests use fixed fixture names. Run them only inside a fresh test
# directory instead of risking user files or concurrent invocations' fixtures.
if(NOT DEFINED PROGRAM OR NOT DEFINED WORK_ROOT)
  message(FATAL_ERROR "PROGRAM and WORK_ROOT are required")
endif()
string(RANDOM LENGTH 24 ALPHABET 0123456789abcdef nonce)
set(work "${WORK_ROOT}/native-properties-${nonce}")
file(MAKE_DIRECTORY "${work}")
file(WRITE "${work}/.vegai-test-scratch" "${nonce}\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "VEGAI_TEST_SCRATCH_TOKEN=${nonce}" "${PROGRAM}"
  WORKING_DIRECTORY "${work}" RESULT_VARIABLE result)
file(REMOVE_RECURSE "${work}")
if(NOT "${result}" STREQUAL "0")
  message(FATAL_ERROR "native property test failed: ${result}")
endif()
