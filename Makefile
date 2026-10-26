# SPDX-License-Identifier: MIT
# GNU Fortran convenience build.  The default is deliberately ISA-portable;
# use `make native` only for a binary that stays on the build host's CPU family.
# For Windows, macOS, non-GNU compilers, and cross-platform builds, prefer CMake.

# Command-line overrides still win over this default (for example: make FC=ifx).
FC = gfortran
BUILD_DIR ?= build
OPENMP_FLAGS ?= -fopenmp
# Do not use -march=native by default: it can emit instructions unavailable on
# another x86 CPU and is not meaningful for ARM or other architectures.
OPT_FLAGS ?= -O3
WARN_FLAGS ?= -std=f2023 -Wall -Wextra -Wimplicit-interface -Wcharacter-truncation
COMMON_FLAGS ?= -ffree-line-length-none
FFLAGS ?= $(OPT_FLAGS) $(WARN_FLAGS) $(COMMON_FLAGS) $(OPENMP_FLAGS)
LDFLAGS ?= $(OPENMP_FLAGS)

PORTABLE_BUILD_DIR ?= build-portable
NATIVE_BUILD_DIR ?= build-native
NATIVE_OPT_FLAGS ?= -O3 -march=native -mtune=native -funroll-loops
# Keep optional PyArrow outside the system interpreter: Debian/PEP 668 hosts
# commonly reject global pip installs. Override DATAIO_PYTHON for an existing
# managed environment if desired.
DATAIO_VENV ?= .venv-dataio
DATAIO_PYTHON ?= $(DATAIO_VENV)/bin/python
# Optional Rust Unicode/subword tokenizer integration stays outside the default
# HDC toolchain, just like optional Arrow support.
LMM_VENV ?= .venv-lmm
LMM_PYTHON ?= $(LMM_VENV)/bin/python

SOURCES := src/hyper_types.f90 src/tokenless_field.f90 src/text_tokenizer.f90 src/utf8_record_stream.f90 \
           src/subword_bpe.f90 \
           src/subword_bpe_training.f90 src/calibrated_decoder.f90 src/neural_retention_decoder.f90 \
           src/apex_optimizer.f90 src/apex_optimizer_impl.f90 src/neural_pretraining.f90 \
           src/causal_ngram_lm.f90 src/vocabulary.f90 \
           src/symbolic_memory.f90 src/resonator_network.f90 src/reasoning_core.f90 \
           src/hdc_pretraining.f90 src/turkish_text.f90 src/dialogue_retrieval.f90 src/fortran_integrity.f90 \
           src/fortran_quality_gate.f90 \
           src/fortran_control.f90
APP := app/main.f90
PRETRAIN_APP := app/pretrain.f90
CAUSAL_LM_APP := app/causal_lm.f90
NEURAL_LM_APP := app/neural_lm.f90
NEURAL_PRETRAIN_APP := app/neural_pretrain.f90
FIELD_APP := app/field.f90
CHAT_APP := app/chat.f90
DIALOGUE_EVAL_APP := app/dialogue_eval.f90
CONTROL_APP := app/control.f90
TEST_SOURCES := test/test_support.f90 test/test_all.f90
APEX_TEST_SOURCE := test/test_apex.f90
BENCH_CONTRACTS := bench/bench_contracts.f90
BENCH_APEX_SMOKE_SOURCE := bench/bench_apex_smoke.f90
BENCH_APEX_SMOKE := $(BUILD_DIR)/vegai_bench_apex_smoke
BENCH_VECTOR_BASELINE_SOURCE := bench/bench_vector_baseline.f90
BENCH_VECTOR_BASELINE := $(BUILD_DIR)/vegai_bench_vector_baseline
BENCH_PHASE1_BASELINE_SOURCE := bench/bench_phase1_baseline.f90
BENCH_PHASE1_BASELINE := $(BUILD_DIR)/vegai_bench_phase1_baseline
BENCH_PHASE2_SEARCH_SOURCE := bench/bench_phase2_search.f90
BENCH_PHASE2_SEARCH := $(BUILD_DIR)/vegai_bench_phase2_search
BENCH_PHASE3_ANALYSIS_SOURCE := bench/bench_phase3_analysis.f90
BENCH_PHASE3_ANALYSIS := $(BUILD_DIR)/vegai_bench_phase3_analysis
BENCH_CONTRACT_TEST_SOURCE := bench/test_bench_contracts.f90
BENCH_CONTRACT_TEST := $(BUILD_DIR)/vegai_bench_contract_tests
BENCH_REF_SOURCE := bench/ref/ref_vector_optimizers.f90
BENCH_REF_TEST_SOURCE := bench/ref/test_ref_vector_optimizers.f90
BENCH_REF_TEST := $(BUILD_DIR)/vegai_ref_vector_tests
BENCH_MATRIX_REF_SOURCE := bench/ref/ref_matrix_optimizers.f90
BENCH_MATRIX_REF_TEST_SOURCE := bench/ref/test_ref_matrix_optimizers.f90
BENCH_MATRIX_REF_TEST := $(BUILD_DIR)/vegai_ref_matrix_tests
BENCH_TASK_SOURCE := bench/bench_tasks.f90
BENCH_TASK_TEST_SOURCE := bench/test_bench_tasks.f90
BENCH_TASK_TEST := $(BUILD_DIR)/vegai_bench_task_tests
BENCH_TUNING_SOURCE := bench/bench_tuning.f90
BENCH_TUNING_TEST_SOURCE := bench/test_bench_tuning.f90
BENCH_TUNING_TEST := $(BUILD_DIR)/vegai_bench_tuning_tests
BENCH_STATISTICS_SOURCE := bench/bench_statistics.f90
BENCH_STATISTICS_TEST_SOURCE := bench/test_bench_statistics.f90
BENCH_STATISTICS_TEST := $(BUILD_DIR)/vegai_bench_statistics_tests
BENCH_PHASE3_TEST_SOURCE := bench/test_bench_phase3_analysis.f90
BENCH_PHASE3_TEST := $(BUILD_DIR)/vegai_bench_phase3_tests
ENGINE := $(BUILD_DIR)/hds_engine
PRETRAIN := $(BUILD_DIR)/vegai_pretrain
CAUSAL_LM := $(BUILD_DIR)/vegai_causal_lm
NEURAL_LM := $(BUILD_DIR)/vegai_neural_lm
NEURAL_PRETRAIN := $(BUILD_DIR)/vegai_neural_pretrain
APEX_BENCHMARK := $(BUILD_DIR)/vegai_apex_benchmark
APEX_DEMO := $(BUILD_DIR)/vegai_apex_demo
FIELD := $(BUILD_DIR)/vegai_field
CHAT := $(BUILD_DIR)/vegai_chat
DIALOGUE_EVAL := $(BUILD_DIR)/vegai_dialogue_eval
CONTROL := $(BUILD_DIR)/vegai_control
TEST_BINARY := $(BUILD_DIR)/vegai_tests
APEX_TEST_BINARY := $(BUILD_DIR)/vegai_apex_tests
# Runtime-only exact-size benchmark artifacts are intentionally ignored by Git.
FIXTURE_9MIB_CORPUS ?= corpora/synthetic_hdc_9mib.txt
FIXTURE_9MIB_MANIFEST ?= corpora/synthetic_hdc_9mib.manifest.json
FIXTURE_9MIB_CHECKPOINT ?= checkpoints/synthetic_hdc_9mib.vgai
FIXTURE_9MIB_REPORT ?= pretraining-runs/synthetic_hdc_9mib.json
PRETRAIN_9MIB_CONTEXT_PROVIDER ?= cpu
PRETRAIN_9MIB_OPENVINO_DEVICE ?= GPU
# Exact decimal 900MB + 90MB controlled continuation artifacts are large and
# remain ignored. Prefer the dedicated GitHub workflow for CI-hosted evidence.
CONTINUOUS_900_RUN_DIR ?= pretraining-runs/synthetic_900mb_then_90mb

.PHONY: all engine pretrain causal-lm neural-lm neural-pretrain apex-benchmark apex-demo bench-apex-smoke bench-vector-baseline bench-phase1-baseline bench-phase2-search bench-phase3-analysis bench-test bench-contract-test bench-task-test bench-tuning-test bench-statistics-test bench-phase3-test bench-ref-test bench-matrix-ref-test field chat dialogue-eval control test fortran-check portable-runtime-check run run-pretrain run-causal-lm run-neural-lm run-neural-pretrain run-field run-chat run-control check field-check optimizer-check tool-check release-check release-registry-check data-check capability-check lmm-check dialogue-check turkish-dialogue-data-check quality-check continuous-quality-check attested-pretrain-check lookup-cache-compare neural-quality-check fixture-9mib verify-fixture-9mib pretrain-9mib pretrain-900mb-continuous clean clean-all debug portable native portable-test native-test

# Default target: portable optimized GNU binaries without host-only ISA flags.
all: engine pretrain causal-lm neural-lm neural-pretrain field chat dialogue-eval control

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/mod

engine: $(ENGINE)

$(ENGINE): $(SOURCES) $(APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(APP) -o $@ $(LDFLAGS)

pretrain: $(PRETRAIN)

$(PRETRAIN): $(SOURCES) $(PRETRAIN_APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(PRETRAIN_APP) -o $@ $(LDFLAGS)

causal-lm: $(CAUSAL_LM)

$(CAUSAL_LM): $(SOURCES) $(CAUSAL_LM_APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(CAUSAL_LM_APP) -o $@ $(LDFLAGS)

neural-lm: $(NEURAL_LM)

$(NEURAL_LM): $(SOURCES) $(NEURAL_LM_APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(NEURAL_LM_APP) -o $@ $(LDFLAGS)

neural-pretrain: $(NEURAL_PRETRAIN)

$(NEURAL_PRETRAIN): $(SOURCES) $(NEURAL_PRETRAIN_APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(NEURAL_PRETRAIN_APP) -o $@ $(LDFLAGS)

apex-benchmark: $(APEX_BENCHMARK)

$(APEX_BENCHMARK): $(SOURCES) app/apex_benchmark.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) app/apex_benchmark.f90 -o $@ $(LDFLAGS)

apex-demo: $(APEX_DEMO)

$(APEX_DEMO): $(SOURCES) app/apex_demo.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) app/apex_demo.f90 -o $@ $(LDFLAGS)

bench-apex-smoke: $(BENCH_APEX_SMOKE)
	OMP_NUM_THREADS=1 $(BENCH_APEX_SMOKE)

$(BENCH_APEX_SMOKE): $(SOURCES) $(BENCH_CONTRACTS) $(BENCH_APEX_SMOKE_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(BENCH_CONTRACTS) \
	  $(BENCH_APEX_SMOKE_SOURCE) -o $@ $(LDFLAGS)

bench-vector-baseline: $(BENCH_VECTOR_BASELINE)
	OMP_NUM_THREADS=1 $(BENCH_VECTOR_BASELINE)

$(BENCH_VECTOR_BASELINE): $(SOURCES) $(BENCH_CONTRACTS) $(BENCH_REF_SOURCE) $(BENCH_VECTOR_BASELINE_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(BENCH_CONTRACTS) \
	  $(BENCH_REF_SOURCE) $(BENCH_VECTOR_BASELINE_SOURCE) -o $@ $(LDFLAGS)

bench-phase1-baseline: $(BENCH_PHASE1_BASELINE)
	OMP_NUM_THREADS=1 $(BENCH_PHASE1_BASELINE)

$(BENCH_PHASE1_BASELINE): $(SOURCES) $(BENCH_CONTRACTS) $(BENCH_TASK_SOURCE) $(BENCH_REF_SOURCE) \
                           $(BENCH_MATRIX_REF_SOURCE) $(BENCH_PHASE1_BASELINE_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(BENCH_CONTRACTS) \
	  $(BENCH_TASK_SOURCE) $(BENCH_REF_SOURCE) $(BENCH_MATRIX_REF_SOURCE) $(BENCH_PHASE1_BASELINE_SOURCE) \
	  -o $@ $(LDFLAGS)

bench-phase2-search: $(BENCH_PHASE2_SEARCH)
	OMP_NUM_THREADS=1 $(BENCH_PHASE2_SEARCH)

$(BENCH_PHASE2_SEARCH): $(SOURCES) $(BENCH_CONTRACTS) $(BENCH_TASK_SOURCE) $(BENCH_TUNING_SOURCE) \
                        $(BENCH_REF_SOURCE) $(BENCH_MATRIX_REF_SOURCE) $(BENCH_PHASE2_SEARCH_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(BENCH_CONTRACTS) \
	  $(BENCH_TASK_SOURCE) $(BENCH_TUNING_SOURCE) $(BENCH_REF_SOURCE) $(BENCH_MATRIX_REF_SOURCE) \
	  $(BENCH_PHASE2_SEARCH_SOURCE) -o $@ $(LDFLAGS)

bench-phase3-analysis: $(BENCH_PHASE3_ANALYSIS)
	OMP_NUM_THREADS=1 $(BENCH_PHASE3_ANALYSIS)

$(BENCH_PHASE3_ANALYSIS): $(BENCH_CONTRACTS) $(BENCH_TUNING_SOURCE) $(BENCH_STATISTICS_SOURCE) \
                          $(BENCH_PHASE3_ANALYSIS_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(BENCH_CONTRACTS) $(BENCH_TUNING_SOURCE) \
	  $(BENCH_STATISTICS_SOURCE) $(BENCH_PHASE3_ANALYSIS_SOURCE) -o $@ $(LDFLAGS)

bench-contract-test: $(BENCH_CONTRACT_TEST)
	$(BENCH_CONTRACT_TEST)

$(BENCH_CONTRACT_TEST): $(BENCH_CONTRACTS) $(BENCH_CONTRACT_TEST_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(BENCH_CONTRACTS) \
	  $(BENCH_CONTRACT_TEST_SOURCE) -o $@ $(LDFLAGS)

bench-task-test: $(BENCH_TASK_TEST)
	$(BENCH_TASK_TEST)

$(BENCH_TASK_TEST): $(BENCH_CONTRACTS) $(BENCH_TASK_SOURCE) $(BENCH_TASK_TEST_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(BENCH_CONTRACTS) $(BENCH_TASK_SOURCE) \
	  $(BENCH_TASK_TEST_SOURCE) -o $@ $(LDFLAGS)

bench-tuning-test: $(BENCH_TUNING_TEST)
	$(BENCH_TUNING_TEST)

$(BENCH_TUNING_TEST): $(BENCH_CONTRACTS) $(BENCH_TUNING_SOURCE) $(BENCH_TUNING_TEST_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(BENCH_CONTRACTS) $(BENCH_TUNING_SOURCE) \
	  $(BENCH_TUNING_TEST_SOURCE) -o $@ $(LDFLAGS)

bench-statistics-test: $(BENCH_STATISTICS_TEST)
	$(BENCH_STATISTICS_TEST)

$(BENCH_STATISTICS_TEST): $(BENCH_CONTRACTS) $(BENCH_STATISTICS_SOURCE) $(BENCH_STATISTICS_TEST_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(BENCH_CONTRACTS) $(BENCH_STATISTICS_SOURCE) \
	  $(BENCH_STATISTICS_TEST_SOURCE) -o $@ $(LDFLAGS)

bench-phase3-test: $(BENCH_PHASE3_TEST) $(BENCH_PHASE3_ANALYSIS)
	cd $(BUILD_DIR) && OMP_NUM_THREADS=1 ./vegai_bench_phase3_tests ./vegai_bench_phase3_analysis

$(BENCH_PHASE3_TEST): $(BENCH_PHASE3_TEST_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(BENCH_PHASE3_TEST_SOURCE) -o $@ $(LDFLAGS)

.NOTPARALLEL: bench-test
bench-test: bench-contract-test bench-task-test bench-tuning-test bench-statistics-test bench-phase3-test \
            bench-ref-test bench-matrix-ref-test

bench-ref-test: $(BENCH_REF_TEST)
	$(BENCH_REF_TEST)

$(BENCH_REF_TEST): $(BENCH_REF_SOURCE) $(BENCH_REF_TEST_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(BENCH_REF_SOURCE) \
	  $(BENCH_REF_TEST_SOURCE) -o $@ $(LDFLAGS)

bench-matrix-ref-test: $(BENCH_MATRIX_REF_TEST)
	$(BENCH_MATRIX_REF_TEST)

$(BENCH_MATRIX_REF_TEST): $(BENCH_MATRIX_REF_SOURCE) $(BENCH_MATRIX_REF_TEST_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(BENCH_MATRIX_REF_SOURCE) \
	  $(BENCH_MATRIX_REF_TEST_SOURCE) -o $@ $(LDFLAGS)

field: $(FIELD)

$(FIELD): $(SOURCES) $(FIELD_APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(FIELD_APP) -o $@ $(LDFLAGS)

chat: $(CHAT)

$(CHAT): $(SOURCES) $(CHAT_APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(CHAT_APP) -o $@ $(LDFLAGS)

dialogue-eval: $(DIALOGUE_EVAL)

$(DIALOGUE_EVAL): $(SOURCES) $(DIALOGUE_EVAL_APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(DIALOGUE_EVAL_APP) -o $@ $(LDFLAGS)

control: $(CONTROL)

$(CONTROL): $(SOURCES) $(CONTROL_APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(CONTROL_APP) -o $@ $(LDFLAGS)

$(TEST_BINARY): $(SOURCES) $(TEST_SOURCES) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(TEST_SOURCES) -o $@ $(LDFLAGS)

$(APEX_TEST_BINARY): $(SOURCES) $(APEX_TEST_SOURCE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(APEX_TEST_SOURCE) -o $@ $(LDFLAGS)

test: $(TEST_BINARY) $(APEX_TEST_BINARY)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(TEST_BINARY)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(APEX_TEST_BINARY)

# Default Fortran-only verification: no Python interpreter, shell inventory,
# or optional accelerator SDK is invoked by these executable paths.
fortran-check: $(TEST_BINARY) $(APEX_TEST_BINARY) $(CONTROL)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(TEST_BINARY)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(APEX_TEST_BINARY)
	$(CONTROL) profile --format json
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(CONTROL) smoke examples/corpus/toy_hdc_corpus.txt \
	  "$(BUILD_DIR)/fortran-control-smoke.vgai"
	$(CONTROL) attest examples/corpus/toy_hdc_corpus.txt "$(BUILD_DIR)/fortran-control-smoke.vgai" --format json

# Same-host binary launch/train/inspect smoke contract. CMake is the preferred
# cross-platform entry point; this GNU Make convenience target mirrors it on
# POSIX hosts without claiming checkpoint interchange across ABI boundaries.
portable-runtime-check: $(ENGINE) $(PRETRAIN)
	python3 tools/verify_portable_runtime.py --engine-bin "$(ENGINE)" --pretrain-bin "$(PRETRAIN)"

run: $(ENGINE)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(ENGINE)

run-pretrain: $(PRETRAIN)
	$(PRETRAIN) help

run-causal-lm: $(CAUSAL_LM)
	$(CAUSAL_LM) help

run-neural-lm: $(NEURAL_LM)
	$(NEURAL_LM) self-test

run-neural-pretrain: $(NEURAL_PRETRAIN)
	$(NEURAL_PRETRAIN) smoke

run-field: $(FIELD)
	$(FIELD) help

run-chat: $(CHAT)
	$(CHAT) help

run-control: $(CONTROL)
	$(CONTROL) help

# Explicit profiles keep portable and host-tuned binaries in distinct folders,
# avoiding accidental reuse of incompatible module/object files.
portable:
	$(MAKE) all BUILD_DIR=$(PORTABLE_BUILD_DIR) OPT_FLAGS='-O3'

native:
	$(MAKE) all BUILD_DIR=$(NATIVE_BUILD_DIR) OPT_FLAGS='$(NATIVE_OPT_FLAGS)'

portable-test:
	$(MAKE) test BUILD_DIR=$(PORTABLE_BUILD_DIR) OPT_FLAGS='-O3'

native-test:
	$(MAKE) test BUILD_DIR=$(NATIVE_BUILD_DIR) OPT_FLAGS='$(NATIVE_OPT_FLAGS)'

# Debug build with bounds, undefined floating-point, and backtrace checks.
debug:
	$(MAKE) clean
	$(MAKE) test OPT_FLAGS='-O0 -g -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'

check:
	python3 tools/verify_reference.py
	python3 tools/verify_tokenless_field.py
	python3 tools/verify_evidential_optimizer.py
	python3 tools/verify_lmm_tool_orchestrator.py
	python3 tools/verify_lmm_release_governor.py
	python3 tools/verify_lmm_release_registry.py
	python3 tools/verify_example_corpus.py
	python3 tools/verify_hardware_compatibility.py
	python3 tools/verify_training_resource_plan.py
	python3 tools/verify_9mib_pretraining.py
	python3 tools/verify_900mb_continuous_pretraining.py
	python3 tools/verify_openvino_provider.py
	python3 tools/verify_openvino_context_worker.py
	python3 tools/verify_pretraining_revision_comparison.py
	python3 tools/verify_capability_curriculum.py
	python3 tools/verify_turkish_dialogue_retrieval_corpus.py
	python3 tools/verify_neural_turkish_epoch1.py

# Independent Python oracle plus compiled CLI integration for the finite
# byte-event field.  This verifies representation identities, not semantics.
field-check: $(FIELD)
	python3 tools/verify_tokenless_field.py --field-bin $(FIELD)

# Algorithm/optimizer candidate selection is evidence-bound rather than a
# claim that any one optimizer is universally best.  It needs no model weights.
optimizer-check:
	python3 tools/verify_evidential_optimizer.py

# Typed, no-shell, independently checked read-only tool execution for actual
# LMM/operator plans. This verifies the control plane, not a model's reasoning.
tool-check:
	python3 tools/verify_lmm_tool_orchestrator.py

# Immutable shadow/canary release state, aggregate-only telemetry, conservative
# promotion/rollback, privacy and artifact-provenance control-plane contract.
release-check:
	python3 tools/verify_lmm_release_governor.py
	python3 tools/verify_lmm_release_registry.py

# Single-host authoritative CAS head for verified governor artifacts. This is
# intentionally not a substitute for a signed distributed release registry.
release-registry-check:
	python3 tools/verify_lmm_release_registry.py

# Strict stdlib-only verification of policy-driven reasoning/problem-solving/
# coding (or operator-defined) curriculum packaging.  It uses only temporary
# protocol fixtures and does not install data adapters or train a model.
capability-check:
	python3 tools/verify_capability_curriculum.py

# Real binary integration checks for the privacy-oriented per-case evaluator
# and the all-Fortran stateless batch evaluator used for calibrated evidence.
dialogue-check: $(PRETRAIN) $(CHAT) $(DIALOGUE_EVAL)
	python3 tools/verify_dialogue_evaluation.py --pretrain-bin $(PRETRAIN) --chat-bin $(CHAT)
	python3 tools/verify_native_dialogue_evaluator.py --pretrain-bin $(PRETRAIN) --native-eval-bin $(DIALOGUE_EVAL)
	python3 tools/verify_dialogue_abstention_calibration.py

# Strict local-source data compiler contract: validates UTF-8 dialogue rows,
# deterministic native ASCII projection, source-disjoint partitions, licence
# metadata and immutable output package publication. It performs no download
# and does not itself claim native Fortran model execution.
turkish-dialogue-data-check:
	python3 tools/verify_turkish_dialogue_retrieval_corpus.py

# Corpus manifest, split integrity, validation-based checkpoint selection,
# atomic selected-checkpoint publication, and a native one-epoch integration run.
neural-quality-check: $(NEURAL_PRETRAIN)
	VEGAI_NEURAL_BIN=$(NEURAL_PRETRAIN) python3 tools/verify_neural_turkish_epoch1.py

# Real compiled checkpoint inspection plus strict held-out case/policy,
# HMAC-redaction, provenance and intentional failed-gate verification. This
# measures a narrow semantic-neighbour contract, not general model intelligence.
quality-check: $(PRETRAIN)
	python3 tools/verify_pretraining_quality_evaluation.py --pretrain-bin $(PRETRAIN)
	python3 tools/verify_continuous_quality_regression.py --pretrain-bin $(PRETRAIN)

# Two-generation native checkpoint gate: the first checkpoint becomes the
# immutable held-out anchor, and a forgotten continuation is rejected before
# it can enter the exactly-once journal. This is not a general-quality claim.
continuous-quality-check: $(PRETRAIN)
	python3 tools/verify_continuous_quality_regression.py --pretrain-bin $(PRETRAIN)

# Real direct-CLI attestation integration: a reviewed generic corpus manifest,
# strict native train, same-filesystem staged publication, independent verify,
# tamper rejection, and failed-overwrite preservation. This fixture validates
# control/provenance behaviour only; it is not a semantic-quality benchmark.
attested-pretrain-check: $(PRETRAIN)
	python3 tools/verify_attested_pretrain.py --pretrain-bin $(PRETRAIN)

# Real train child-process cache on/off comparator. Fixture/report/checkpoint
# comparison are outside the timed interval; checkpoint byte equality is the
# semantic-preservation gate, not a minimum host-specific speedup threshold.
lookup-cache-compare: $(PRETRAIN)
	python3 tools/compare_pretraining_lookup_cache.py --pretrain-bin $(PRETRAIN) --repetitions 3

# A deterministic, provenance-labelled 9 MiB structural fixture. It carries no
# third-party text and must not be interpreted as domain or language knowledge.
fixture-9mib:
	python3 tools/synthetic_pretraining_fixture.py generate \
		--output "$(FIXTURE_9MIB_CORPUS)" --manifest "$(FIXTURE_9MIB_MANIFEST)" \
		--target-bytes 9MiB --overwrite

verify-fixture-9mib:
	python3 tools/synthetic_pretraining_fixture.py verify \
		--corpus "$(FIXTURE_9MIB_CORPUS)" --manifest "$(FIXTURE_9MIB_MANIFEST)"

# The runner's elapsed metric encloses only this one train child process;
# generation, preflight, inspection, report formatting, and compilation are
# deliberately outside it. Outputs remain ignored runtime artifacts.
pretrain-9mib: pretrain fixture-9mib
	python3 tools/run_9mib_pretraining.py \
		--pretrain-bin "$(PRETRAIN)" --corpus "$(FIXTURE_9MIB_CORPUS)" \
		--manifest "$(FIXTURE_9MIB_MANIFEST)" --checkpoint "$(FIXTURE_9MIB_CHECKPOINT)" \
		--report "$(FIXTURE_9MIB_REPORT)" --max-vocab 32768 --min-frequency 3 \
		--window 6 --position-shift 41 --context-mode hierarchical --epochs 2 \
		--initial-plasticity 11.0 --minimum-plasticity 0.6 --decay-events 3000 \
		--context-provider "$(PRETRAIN_9MIB_CONTEXT_PROVIDER)" \
		--openvino-device "$(PRETRAIN_9MIB_OPENVINO_DEVICE)" --overwrite

# This performs a real exact-decimal 900,000,000-byte initial train and a
# 90,000,000-byte native continuation. The train-only timings and held-out
# topology regression decision are written below the ignored run directory.
pretrain-900mb-continuous: pretrain
	python3 tools/run_900mb_continuous_pretraining.py \
		--pretrain-bin "$(PRETRAIN)" --run-dir "$(CONTINUOUS_900_RUN_DIR)" \
		--omp-threads 2 --train-timeout-seconds 2400

# Rich JSON/Parquet/Arrow data formats are optional rather than a dependency of
# the portable Fortran runtime. This explicit target installs only the adapter
# dependency, then runs streaming round-trip/production-adapter and optional
# Parquet/Arrow curriculum-adapter checks.
data-check:
	@test -x "$(DATAIO_PYTHON)" || python3 -m venv "$(DATAIO_VENV)"
	$(DATAIO_PYTHON) -m pip install --disable-pip-version-check -r requirements-dataio.txt
	$(DATAIO_PYTHON) tools/verify_training_data_io.py
	$(DATAIO_PYTHON) tools/verify_capability_curriculum.py

# Optional real Unicode/BPE tokenizer path plus stdlib-only worker/context,
# evidence, session, multimodal assets, held-out citation/abstention release,
# measured domain-expert portfolio-routing and verified read-only tool-plan
# contracts. No model weights are pulled.
lmm-check:
	@test -x "$(LMM_PYTHON)" || python3 -m venv "$(LMM_VENV)"
	$(LMM_PYTHON) -m pip install --disable-pip-version-check -r requirements-lmm.txt
	$(LMM_PYTHON) tools/verify_lmm_runtime.py
	$(LMM_PYTHON) tools/verify_lmm_portfolio.py
	$(LMM_PYTHON) tools/verify_lmm_tool_orchestrator.py
	$(LMM_PYTHON) tools/verify_lmm_release_governor.py
	$(LMM_PYTHON) tools/verify_lmm_release_registry.py

clean:
	rm -rf $(BUILD_DIR)

clean-all:
	rm -rf build $(PORTABLE_BUILD_DIR) $(NATIVE_BUILD_DIR)
