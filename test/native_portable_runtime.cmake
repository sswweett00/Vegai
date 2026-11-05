# SPDX-License-Identifier: MIT
# Same-host launch/train/inspect check for the real Fortran executables.
# CMake is build/test infrastructure, not a runtime dependency of the engine.
if(NOT DEFINED ENGINE OR NOT DEFINED PRETRAIN OR NOT DEFINED CORPUS OR NOT DEFINED WORK_ROOT)
  message(FATAL_ERROR "ENGINE, PRETRAIN, CORPUS and trusted WORK_ROOT are required")
endif()
foreach(input IN ITEMS ENGINE PRETRAIN CORPUS)
  if(NOT EXISTS "${${input}}" OR IS_DIRECTORY "${${input}}")
    message(FATAL_ERROR "${input} is not a file: ${${input}}")
  endif()
endforeach()
string(RANDOM LENGTH 24 ALPHABET 0123456789abcdef nonce)
set(work "${WORK_ROOT}/native-portable-${nonce}")
file(MAKE_DIRECTORY "${work}")
set(checkpoint "${work}/smoke.vgai")

execute_process(COMMAND "${CMAKE_COMMAND}" -E env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE "${ENGINE}"
  WORKING_DIRECTORY "${work}" TIMEOUT 90 RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" STREQUAL "0" OR NOT out MATCHES "VERIFICATION RESULT: PASS")
  message(FATAL_ERROR "engine launch/verification failed: ${result}: ${out}: ${err}")
endif()
execute_process(COMMAND "${PRETRAIN}" help WORKING_DIRECTORY "${work}" TIMEOUT 90
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" STREQUAL "0" OR NOT out MATCHES "Vegai deterministic HDC/VSA corpus pretraining")
  message(FATAL_ERROR "pretrain help failed: ${result}: ${out}: ${err}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE "${PRETRAIN}" train
  "${CORPUS}" "${checkpoint}" --max-vocab 64 --min-frequency 2 --window 2 --position-shift 17
  --context-mode hierarchical --epochs 1 --strict --query dog
  WORKING_DIRECTORY "${work}" TIMEOUT 90 RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" STREQUAL "0" OR NOT EXISTS "${checkpoint}" OR
   NOT out MATCHES "phase: streaming HDC context learning" OR NOT out MATCHES "completed epochs: 1")
  message(FATAL_ERROR "pretrain smoke failed: ${result}: ${out}: ${err}")
endif()
file(SIZE "${checkpoint}" checkpoint_bytes)
if(checkpoint_bytes LESS 1)
  message(FATAL_ERROR "pretrain produced an empty checkpoint")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE "${PRETRAIN}"
  inspect "${checkpoint}" --query dog WORKING_DIRECTORY "${work}" TIMEOUT 90
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${result}" STREQUAL "0" OR NOT out MATCHES "Vegai HDC pretraining checkpoint")
  message(FATAL_ERROR "same-host checkpoint inspect failed: ${result}: ${out}: ${err}")
endif()
file(SHA256 "${ENGINE}" engine_hash)
file(SHA256 "${PRETRAIN}" pretrain_hash)
file(SHA256 "${checkpoint}" checkpoint_hash)
message(STATUS "same-host portable runtime PASS; cross-OS/ABI checkpoint interchange is not asserted")
message(STATUS "engine SHA-256: ${engine_hash}; pretrain SHA-256: ${pretrain_hash}")
message(STATUS "checkpoint bytes: ${checkpoint_bytes}; SHA-256: ${checkpoint_hash}")
file(REMOVE_RECURSE "${work}")
