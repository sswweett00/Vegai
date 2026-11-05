# SPDX-License-Identifier: MIT
# Native CLI regression in a fresh test-owned directory. No Python or shell.
if(NOT DEFINED ENGINE OR NOT DEFINED MODE OR NOT DEFINED WORK_ROOT)
  message(FATAL_ERROR "ENGINE, MODE and WORK_ROOT are required")
endif()
string(RANDOM LENGTH 24 ALPHABET 0123456789abcdef nonce)
set(work "${WORK_ROOT}/native-context-${nonce}")
file(MAKE_DIRECTORY "${work}")
set(checkpoint "${work}/checkpoint.vgfb")

if(MODE STREQUAL "partial-reject")
  execute_process(COMMAND "${ENGINE}" train --tokens 5 --train-chunk-tokens 4 --task markov
    --checkpoint "${checkpoint}" RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" MATCHES "^-?[0-9]+$" OR "${result}" STREQUAL "0" OR EXISTS "${checkpoint}"
     OR NOT err MATCHES "training tokens must be a multiple")
    message(FATAL_ERROR "partial budget must fail normally before publishing a checkpoint: ${result}: ${err}")
  endif()
elseif(MODE STREQUAL "partial-allow")
  execute_process(COMMAND "${ENGINE}" train --tokens 5 --train-chunk-tokens 4 --task markov
    --allow-partial-chunk true --width 8 --heads 2 --ffn 16 --vocab 16
    --local-window 8 --memory-scales 2 --memory-slots 8 --checkpoint "${checkpoint}"
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0" OR NOT out MATCHES "trained 5 tokens in 2 updates" OR NOT EXISTS "${checkpoint}")
    message(FATAL_ERROR "partial opt-in failed: ${result}: ${out}: ${err}")
  endif()
else()
  if(NOT MODE STREQUAL "train" AND NOT MODE STREQUAL "inspect" AND NOT MODE STREQUAL "evaluate")
    message(FATAL_ERROR "unknown native context test mode: ${MODE}")
  endif()
  execute_process(COMMAND "${ENGINE}" train --tokens 256 --train-chunk-tokens 32
    --width 16 --heads 2 --ffn 32 --vocab 16 --local-window 8
    --memory-scales 2 --memory-slots 8 --window 1000000 --checkpoint "${checkpoint}"
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0" OR NOT EXISTS "${checkpoint}")
    message(FATAL_ERROR "context fixture train failed: ${result}: ${out}: ${err}")
  endif()
  if(MODE STREQUAL "inspect")
    execute_process(COMMAND "${ENGINE}" inspect --checkpoint "${checkpoint}"
      RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT "${result}" STREQUAL "0")
      message(FATAL_ERROR "context checkpoint inspect failed: ${result}: ${err}")
    endif()
  elseif(MODE STREQUAL "evaluate")
    execute_process(COMMAND "${ENGINE}" evaluate --tokens 256 --task needle --distance 32
      --checkpoint "${checkpoint}" RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT "${result}" STREQUAL "0")
      message(FATAL_ERROR "context checkpoint evaluation failed: ${result}: ${err}")
    endif()
  endif()
endif()
file(REMOVE_RECURSE "${work}")
message(STATUS "native context ${MODE} contract passed")
