# SPDX-License-Identifier: MIT
# Byte-exact text fixtures for the Python-free contracts.
#
# CMake's file(WRITE) writes text in the host's native line-ending mode, so on
# Windows an LF byte becomes CRLF. Contracts that assert exact bytes therefore
# create their fixtures through the compiled Fortran helper, which writes a
# stream-unformatted file from hexadecimal tokens. No Python is involved.
#
# native_write_text(<path> <literal text>) writes the literal bytes.
# native_write_hex(<path> <hex-token>...) accepts hex text tokens and
# "<byte>*<count>" repetition tokens, so large records stay off the command line.
#
# An operating system bounds a single command-line argument, so native_write_text
# writes long texts as several appended pieces. The chunk stays well below the
# smallest Windows limit (32767 characters for the whole command line).
if(NOT DEFINED FIXTURE OR "${FIXTURE}" STREQUAL "")
  message(FATAL_ERROR "FIXTURE is required for byte-exact fixtures")
endif()
set(NATIVE_EXACT_CHUNK_HEX 16000)

function(native_write_bytes action path)
  # The helper only creates the file; empty fixtures (a zero-byte corpus) are
  # legal and the destination directory may not exist yet.
  get_filename_component(parent "${path}" DIRECTORY)
  if(NOT "${parent}" STREQUAL "")
    file(MAKE_DIRECTORY "${parent}")
  endif()
  execute_process(COMMAND "${FIXTURE}" "${action}" "${path}" ${ARGN}
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "byte fixture ${path} failed: ${result}: ${out}: ${err}")
  endif()
endfunction()

function(native_write_hex path)
  native_write_bytes(bytes "${path}" ${ARGN})
endfunction()

function(native_write_text path text)
  string(HEX "${text}" hex)
  string(LENGTH "${hex}" length)
  if(length EQUAL 0)
    native_write_hex("${path}")
    return()
  endif()
  set(offset 0)
  set(action bytes)
  while(offset LESS length)
    math(EXPR remaining "${length} - ${offset}")
    if(remaining LESS NATIVE_EXACT_CHUNK_HEX)
      set(take ${remaining})
    else()
      set(take ${NATIVE_EXACT_CHUNK_HEX})
    endif()
    string(SUBSTRING "${hex}" ${offset} ${take} piece)
    native_write_bytes(${action} "${path}" "${piece}")
    set(action bytes-append)
    math(EXPR offset "${offset} + ${take}")
  endwhile()
endfunction()
