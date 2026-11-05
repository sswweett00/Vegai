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
if(NOT DEFINED FIXTURE OR "${FIXTURE}" STREQUAL "")
  message(FATAL_ERROR "FIXTURE is required for byte-exact fixtures")
endif()

function(native_write_hex path)
  execute_process(COMMAND "${FIXTURE}" bytes "${path}" ${ARGN}
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "byte fixture ${path} failed: ${result}: ${out}: ${err}")
  endif()
endfunction()

function(native_write_text path text)
  string(HEX "${text}" hex)
  native_write_hex("${path}" "${hex}")
endfunction()
