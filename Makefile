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

SOURCES := src/hyper_types.f90 src/tokenless_field.f90 src/text_tokenizer.f90 src/vocabulary.f90 \
           src/symbolic_memory.f90 src/resonator_network.f90 src/reasoning_core.f90 \
           src/hdc_pretraining.f90 src/dialogue_retrieval.f90
APP := app/main.f90
PRETRAIN_APP := app/pretrain.f90
FIELD_APP := app/field.f90
CHAT_APP := app/chat.f90
TEST_SOURCES := test/test_support.f90 test/test_all.f90
ENGINE := $(BUILD_DIR)/hds_engine
PRETRAIN := $(BUILD_DIR)/vegai_pretrain
FIELD := $(BUILD_DIR)/vegai_field
CHAT := $(BUILD_DIR)/vegai_chat
TEST_BINARY := $(BUILD_DIR)/vegai_tests

.PHONY: all engine pretrain field chat test run run-pretrain run-field run-chat check field-check optimizer-check tool-check release-check release-registry-check data-check capability-check lmm-check dialogue-check clean clean-all debug portable native portable-test native-test

# Default target: portable optimized GNU binaries without host-only ISA flags.
all: engine pretrain field chat

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/mod

engine: $(ENGINE)

$(ENGINE): $(SOURCES) $(APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(APP) -o $@ $(LDFLAGS)

pretrain: $(PRETRAIN)

$(PRETRAIN): $(SOURCES) $(PRETRAIN_APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(PRETRAIN_APP) -o $@ $(LDFLAGS)

field: $(FIELD)

$(FIELD): $(SOURCES) $(FIELD_APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(FIELD_APP) -o $@ $(LDFLAGS)

chat: $(CHAT)

$(CHAT): $(SOURCES) $(CHAT_APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(CHAT_APP) -o $@ $(LDFLAGS)

$(TEST_BINARY): $(SOURCES) $(TEST_SOURCES) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(TEST_SOURCES) -o $@ $(LDFLAGS)

test: $(TEST_BINARY)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(TEST_BINARY)

run: $(ENGINE)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(ENGINE)

run-pretrain: $(PRETRAIN)
	$(PRETRAIN) help

run-field: $(FIELD)
	$(FIELD) help

run-chat: $(CHAT)
	$(CHAT) help

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
	python3 tools/verify_openvino_provider.py
	python3 tools/verify_openvino_context_worker.py
	python3 tools/verify_capability_curriculum.py

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

# Real binary integration check for held-out retrieval grounding/abstention,
# strict evaluator schema, input provenance and an intentional failed gate.
dialogue-check: $(PRETRAIN) $(CHAT)
	python3 tools/verify_dialogue_evaluation.py --pretrain-bin $(PRETRAIN) --chat-bin $(CHAT)

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
