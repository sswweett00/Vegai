# SPDX-License-Identifier: MIT
# Runner for `fpm test --runner "cmake -P test/fpm_scratch_runner.cmake"`.
# fpm appends one test executable as the argument after this script. Resolve it
# before changing directory, then reuse the same isolated runner as CTest.
if(NOT CMAKE_ARGC EQUAL 4)
  message(FATAL_ERROR "fpm scratch runner expects exactly one test executable argument")
endif()
get_filename_component(PROGRAM "${CMAKE_ARGV3}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
if(NOT EXISTS "${PROGRAM}")
  message(FATAL_ERROR "fpm test executable not found: ${PROGRAM}")
endif()
set(WORK_ROOT "${CMAKE_CURRENT_BINARY_DIR}/build")
file(MAKE_DIRECTORY "${WORK_ROOT}")
include("${CMAKE_CURRENT_LIST_DIR}/run_native_in_temp.cmake")
