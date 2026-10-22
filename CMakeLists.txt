cmake_minimum_required(VERSION 3.20)

project(VegaiHDS VERSION 0.1.0 LANGUAGES Fortran)

option(VEGAI_ENABLE_OPENMP "Enable OpenMP parallel cleanup scans" ON)
option(VEGAI_BUILD_TESTS "Build property and integration tests" ON)

set(CMAKE_Fortran_STANDARD 2018)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)
set(CMAKE_Fortran_EXTENSIONS OFF)

add_library(vegai_hds STATIC
    src/hyper_types.f90
    src/symbolic_memory.f90
    src/reasoning_core.f90
)

set_target_properties(vegai_hds PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules"
)

target_include_directories(vegai_hds PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/modules>
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(vegai_hds PRIVATE
        -Wall -Wextra -Wimplicit-interface -Wcharacter-truncation
        -ffree-line-length-none
    )
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    target_compile_options(vegai_hds PRIVATE -warn all)
endif()

if(VEGAI_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS Fortran)
    if(OpenMP_Fortran_FOUND)
        target_link_libraries(vegai_hds PUBLIC OpenMP::OpenMP_Fortran)
        target_compile_definitions(vegai_hds PUBLIC VEGAI_OPENMP=1)
    else()
        message(WARNING "OpenMP Fortran was not found; cleanup scans remain serial.")
    endif()
endif()

add_executable(hds_engine app/main.f90)
target_link_libraries(hds_engine PRIVATE vegai_hds)

if(VEGAI_BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_executable(vegai_tests
        test/test_support.f90
        test/test_all.f90
    )
    target_link_libraries(vegai_tests PRIVATE vegai_hds)

    if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
        target_compile_options(vegai_tests PRIVATE -ffree-line-length-none)
    endif()

    add_test(NAME vegai_fortran_properties COMMAND vegai_tests)
endif()
