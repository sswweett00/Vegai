cmake_minimum_required(VERSION 3.20)

project(VegaiHDS VERSION 0.1.0 LANGUAGES Fortran)

option(VEGAI_ENABLE_OPENMP "Enable OpenMP parallel cleanup scans" ON)
option(VEGAI_BUILD_TESTS "Build property and integration tests" ON)
option(VEGAI_ENABLE_LEGACY_PYTHON_TESTS
       "Enable migration-era Python integration/control-plane tests (not required by native Fortran runtime)" OFF)
set(VEGAI_CPU_TARGET "portable" CACHE STRING
    "CPU code-generation profile: portable (no host ISA) or native (host-tuned)")
set_property(CACHE VEGAI_CPU_TARGET PROPERTY STRINGS portable native)

if(NOT VEGAI_CPU_TARGET STREQUAL "portable" AND NOT VEGAI_CPU_TARGET STREQUAL "native")
    message(FATAL_ERROR "VEGAI_CPU_TARGET must be portable or native, got: ${VEGAI_CPU_TARGET}")
endif()

# Vegai is specified and continuously compiled as a Fortran 2023 project.
# CMake has no universally portable feature abstraction for every compiler, so
# known compiler families receive their explicit conformance flags below.
set(CMAKE_Fortran_STANDARD 2023)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)
set(CMAKE_Fortran_EXTENSIONS OFF)

add_library(vegai_hds STATIC
    src/hyper_types.f90
    src/tokenless_field.f90
    src/text_tokenizer.f90
    src/vocabulary.f90
    src/symbolic_memory.f90
    src/resonator_network.f90
    src/reasoning_core.f90
    src/hdc_pretraining.f90
    src/turkish_text.f90
    src/dialogue_retrieval.f90
    src/fortran_integrity.f90
    src/fortran_quality_gate.f90
    src/fortran_control.f90
)

set_target_properties(vegai_hds PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules"
)

target_include_directories(vegai_hds PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/modules>
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(vegai_hds PUBLIC -std=f2023)
    target_compile_options(vegai_hds PRIVATE
        -Wall -Wextra -Wimplicit-interface -Wcharacter-truncation
        -ffree-line-length-none
    )
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    if(WIN32)
        target_compile_options(vegai_hds PUBLIC /stand:f23)
        target_compile_options(vegai_hds PRIVATE /warn:all)
    else()
        target_compile_options(vegai_hds PUBLIC -stand f23)
        target_compile_options(vegai_hds PRIVATE -warn all)
    endif()
else()
    message(WARNING
        "Compiler-specific Fortran 2023 conformance flag is unknown; configure it explicitly.")
endif()

# The portable profile intentionally emits no host-ISA flags.  A binary built
# this way relies only on the compiler's architecture baseline and can be moved
# among compatible CPUs (for example Core Ultra, older x86-64, and supported
# non-x86 targets when compiled by an appropriate Fortran compiler).  The
# native profile is opt-in and must only be distributed to the build host's
# compatible CPU family.
if(VEGAI_CPU_TARGET STREQUAL "native")
    if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
        target_compile_options(vegai_hds PRIVATE -march=native -mtune=native)
    elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
        if(WIN32)
            target_compile_options(vegai_hds PRIVATE /QxHost)
        else()
            target_compile_options(vegai_hds PRIVATE -xHost)
        endif()
    else()
        message(WARNING
            "Native CPU profile requested, but no safe compiler-specific host-tuning flag is known; using portable code generation.")
    endif()
endif()
message(STATUS "Vegai CPU target profile: ${VEGAI_CPU_TARGET}")
message(STATUS "Vegai execution backend: portable CPU; GPU/NPU inventory is not accelerator dispatch and no vendor runtime is required")

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

add_executable(vegai_pretrain app/pretrain.f90)
target_link_libraries(vegai_pretrain PRIVATE vegai_hds)

# Raw-byte field inspection remains separate from text-token pretraining.  It
# exposes finite representation checks only and has no generative surface.
add_executable(vegai_field app/field.f90)
target_link_libraries(vegai_field PRIVATE vegai_hds)

# A separate executable keeps grounded dialogue retrieval modular: model
# training/checkpoint semantics remain in vegai_pretrain, while this surface
# only retrieves operator-supplied knowledge-base responses.
add_executable(vegai_chat app/chat.f90)
target_link_libraries(vegai_chat PRIVATE vegai_hds)

# Stateless native batch evaluator. It reads a strict bounded TSV contract and
# keeps source-disjoint retrieval evidence out of process-per-prompt wrappers.
add_executable(vegai_dialogue_eval app/dialogue_eval.f90)
target_link_libraries(vegai_dialogue_eval PRIVATE vegai_hds)

# Pure-Fortran control plane: portable compiler profile, explicit bounded
# resource arithmetic, and a same-host native train/checkpoint smoke path.
# It deliberately does not require Python or invoke OS-specific shell probes.
add_executable(vegai_control app/control.f90)
target_link_libraries(vegai_control PRIVATE vegai_hds)

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
    # The default native control-plane contract contains no Python process:
    # profile/report rendering, bounded resource arithmetic, and a real
    # same-host vocabulary/train/checkpoint/load smoke all run in Fortran.
    add_test(
        NAME vegai_fortran_control_profile
        COMMAND $<TARGET_FILE:vegai_control> profile --format json
    )
    add_test(
        NAME vegai_fortran_control_resource_plan
        COMMAND $<TARGET_FILE:vegai_control> resource-plan
                ${CMAKE_CURRENT_SOURCE_DIR}/examples/corpus/toy_hdc_corpus.txt
                --checkpoint ${CMAKE_CURRENT_BINARY_DIR}/fortran-control-plan.vgai
                --max-vocab 64 --available-memory-bytes 8589934592
                --available-storage-bytes 8589934592 --require-ready --format json
    )
    add_test(
        NAME vegai_fortran_control_smoke
        COMMAND $<TARGET_FILE:vegai_control> smoke
                ${CMAKE_CURRENT_SOURCE_DIR}/examples/corpus/toy_hdc_corpus.txt
                ${CMAKE_CURRENT_BINARY_DIR}/fortran-control-smoke.vgai
    )
    set_tests_properties(vegai_fortran_control_smoke PROPERTIES TIMEOUT 120)
    add_test(
        NAME vegai_fortran_control_attest
        COMMAND $<TARGET_FILE:vegai_control> attest
                ${CMAKE_CURRENT_SOURCE_DIR}/examples/corpus/toy_hdc_corpus.txt
                ${CMAKE_CURRENT_BINARY_DIR}/fortran-control-smoke.vgai --format json
    )
    set_tests_properties(vegai_fortran_control_attest PROPERTIES DEPENDS vegai_fortran_control_smoke)

    # Python remains available only as a migration-era verifier/control-plane
    # adapter. The default CMake build/test path is self-contained Fortran.
    if(VEGAI_ENABLE_LEGACY_PYTHON_TESTS)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
        add_test(
            NAME vegai_tokenless_signal_field
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_tokenless_field.py
                    --field-bin $<TARGET_FILE:vegai_field>
        )
        add_test(
            NAME vegai_evidential_optimizer
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_evidential_optimizer.py
        )
        add_test(
            NAME vegai_hardware_compatibility
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_hardware_compatibility.py
        )
        # A platform-neutral smoke check uses CMake target-file paths (including
        # the .exe suffix on Windows), runs the actual engine, then creates and
        # inspects a same-host checkpoint. It deliberately does not imply that
        # stream-unformatted checkpoints are portable across compiler/ABI pairs.
        add_test(
            NAME vegai_portable_runtime_smoke
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_portable_runtime.py
                    --engine-bin $<TARGET_FILE:hds_engine>
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
        )
        set_tests_properties(vegai_portable_runtime_smoke PROPERTIES TIMEOUT 120)
        add_test(
            NAME vegai_training_resource_plan
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_training_resource_plan.py
        )
        # This creates/streams an exact 9 MiB synthetic fixture and exercises
        # report lineage with a fake executable. The separate GitHub workflow
        # runs the same contract against the real compiled Fortran binary.
        add_test(
            NAME vegai_9mib_pretraining_contract
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_9mib_pretraining.py
        )
        set_tests_properties(vegai_9mib_pretraining_contract PROPERTIES TIMEOUT 120)
        # The fixed 900MB -> 90MB orchestrator is exercised with tiny exact
        # byte budgets and a hermetic native ABI fixture here; its separate
        # manual workflow owns the real large native measurement.
        add_test(
            NAME vegai_900mb_continuous_pretraining_contract
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_900mb_continuous_pretraining.py
        )
        set_tests_properties(vegai_900mb_continuous_pretraining_contract PROPERTIES TIMEOUT 180)
        # A small instance of the same schema verifies that the bounded runner
        # parses the actual compiled Fortran train/inspect ABI. The dedicated
        # workflow owns the full 9 MiB real-binary measurement.
        add_test(
            NAME vegai_9mib_pretraining_real_binary
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_9mib_pretraining_real_binary.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
        )
        set_tests_properties(vegai_9mib_pretraining_real_binary PROPERTIES TIMEOUT 240)
        # Generic direct CLI users receive the same fail-closed corpus/binary/
        # checkpoint provenance boundary as the fixture runner: this invokes a
        # real compiled train, validates staged publication, and then verifies
        # corpus/manifest/checkpoint/report tamper rejection without retraining.
        # Its hermetic optional-provider subcase binds worker/interpreter files
        # without claiming an OpenVINO device/runtime is available in CTest.
        add_test(
            NAME vegai_attested_pretraining_real_binary
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_attested_pretrain.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
        )
        set_tests_properties(vegai_attested_pretraining_real_binary PROPERTIES TIMEOUT 300)
        # Both cache arms execute real compiled train processes and must produce
        # byte-identical checkpoints. The report's timer excludes compilation,
        # fixture construction, and comparator/report work.
        add_test(
            NAME vegai_pretraining_lookup_cache
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/compare_pretraining_lookup_cache.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain> --repetitions 2
        )
        set_tests_properties(vegai_pretraining_lookup_cache PROPERTIES TIMEOUT 240)
        add_test(
            NAME vegai_openvino_provider_contract
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_openvino_provider.py
        )
        add_test(
            NAME vegai_openvino_context_worker_contract
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_openvino_context_worker.py
        )
        add_test(
            NAME vegai_openvino_context_provider
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_openvino_context_provider.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
        )
        add_test(
            NAME vegai_example_pretraining
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_example_corpus.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
        )
        add_test(
            NAME vegai_dialogue_retrieval
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_dialogue_retrieval.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
                    --chat-bin $<TARGET_FILE:vegai_chat>
        )
        add_test(
            NAME vegai_dialogue_evaluation
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_dialogue_evaluation.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
                    --chat-bin $<TARGET_FILE:vegai_chat>
        )
        set_tests_properties(vegai_dialogue_evaluation PROPERTIES TIMEOUT 180)
        add_test(
            NAME vegai_native_dialogue_evaluator
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_native_dialogue_evaluator.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
                    --native-eval-bin $<TARGET_FILE:vegai_dialogue_eval>
        )
        set_tests_properties(vegai_native_dialogue_evaluator PROPERTIES TIMEOUT 180)
        add_test(
            NAME vegai_dialogue_abstention_calibration
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_dialogue_abstention_calibration.py
        )
        add_test(
            NAME vegai_turkish_dialogue_corpus
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_turkish_dialogue_retrieval_corpus.py
        )
        add_test(
            NAME vegai_lmm_runtime
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_lmm_runtime.py
        )
        set_tests_properties(vegai_lmm_runtime PROPERTIES TIMEOUT 180)
        add_test(
            NAME vegai_lmm_tool_orchestrator
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_lmm_tool_orchestrator.py
        )
        set_tests_properties(vegai_lmm_tool_orchestrator PROPERTIES TIMEOUT 180)
        add_test(
            NAME vegai_lmm_release_governor
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_lmm_release_governor.py
        )
        set_tests_properties(vegai_lmm_release_governor PROPERTIES TIMEOUT 180)
        add_test(
            NAME vegai_lmm_release_registry
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_lmm_release_registry.py
        )
        set_tests_properties(vegai_lmm_release_registry PROPERTIES TIMEOUT 180)
        add_test(
            NAME vegai_lmm_portfolio
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_lmm_portfolio.py
        )
        set_tests_properties(vegai_lmm_portfolio PROPERTIES TIMEOUT 180)
        add_test(
            NAME vegai_continuous_pretraining
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_continuous_pretraining.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
        )
        # Real native anchor/candidate inspection plus a hermetic catastrophic-
        # forgetting subcase. A continuation candidate is never journaled when
        # its immutable held-out regression policy fails.
        add_test(
            NAME vegai_continuous_quality_regression
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_continuous_quality_regression.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
        )
        set_tests_properties(vegai_continuous_quality_regression PROPERTIES TIMEOUT 240)
        add_test(
            NAME vegai_production_pretraining
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_production_pretraining.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
        )
        add_test(
            NAME vegai_pretraining_quality_evaluation
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_pretraining_quality_evaluation.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
        )
        set_tests_properties(vegai_pretraining_quality_evaluation PROPERTIES TIMEOUT 180)
        add_test(
            NAME vegai_capability_curriculum
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_capability_curriculum.py
        )
        set_tests_properties(vegai_capability_curriculum PROPERTIES TIMEOUT 180)
        add_test(
            NAME vegai_example_production_dataset_preflight
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/production_pretrain.py
                    --pretrain-bin $<TARGET_FILE:vegai_pretrain>
                    --dataset ${CMAKE_CURRENT_SOURCE_DIR}/examples/production_dataset/mini_turkish_ascii_hdc/dataset.json
                    --run-dir ${CMAKE_CURRENT_BINARY_DIR}/example-production-preflight
                    --max-vocab 128 --preflight-only
        )
        else()
            message(WARNING "Python3 was not found; requested legacy Python integration tests are not registered.")
        endif()
    else()
        message(STATUS "Vegai legacy Python integration tests are disabled; native Fortran tests remain enabled.")
    endif()
endif()
