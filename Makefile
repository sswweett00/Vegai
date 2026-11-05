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
# Fortran 2018 is the portable baseline; opt in to f2023 on newer compilers.
FORTRAN_STANDARD ?= f2018
WARN_FLAGS ?= -std=$(FORTRAN_STANDARD) -Wall -Wextra -Wimplicit-interface -Wcharacter-truncation
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
           src/subword_bpe_training.f90 src/calibrated_decoder.f90 src/neural_retention_decoder.f90 src/neural_pretraining.f90 \
           src/causal_ngram_lm.f90 src/vocabulary.f90 \
           src/symbolic_memory.f90 src/resonator_network.f90 src/reasoning_core.f90 \
           src/hdc_pretraining.f90 src/turkish_text.f90 src/dialogue_retrieval.f90 src/fortran_integrity.f90 \
           src/corpus_preflight.f90 src/portable_cli.f90 src/fortran_quality_gate.f90 \
           src/fortran_control.f90 \
           src/sparse_expert_layer.f90
APP := app/main.f90
PRETRAIN_APP := app/pretrain.f90
CAUSAL_LM_APP := app/causal_lm.f90
NEURAL_LM_APP := app/neural_lm.f90
NEURAL_PRETRAIN_APP := app/neural_pretrain.f90
FIELD_APP := app/field.f90
CHAT_APP := app/chat.f90
DIALOGUE_EVAL_APP := app/dialogue_eval.f90
CONTROL_APP := app/control.f90
CORPUS_OVERLAP_APP := app/corpus_overlap.f90
UD_TEXT_EXTRACT_APP := app/ud_text_extract.f90
SPARSE_APP := app/sparse_layer_bench.f90
TEST_SOURCES := test/test_support.f90 test/test_all.f90
UTF8_STREAM_TEST_SOURCE := test/test_utf8_record_stream.f90
PREFLIGHT_TEST_SOURCE := test/corpus_preflight_tests.f90
SPARSE_TEST_SOURCES := test/sparse_layer_tests.f90
# Fixed-budget context engine: its own library, its own CLI, its own contract
# suite.  Kept separate from the HDC stack so it can be built and diagnosed on
# its own (`make context-engine context-tests`).
CONTEXT_SOURCES := src/portable_cli.f90 src/utf8_record_stream.f90 src/utf8_generation.f90 src/fixed_budget_context.f90 \
                   src/fixed_budget_backward.f90 src/fixed_budget_optimizer.f90 src/fixed_budget_training.f90 \
                   src/fixed_budget_natural_eval.f90
CONTEXT_APP := app/context_engine.f90
# Byte-level corpus trainer + prompt/response CLI: the same library, a
# streaming data path instead of the synthetic stream.
PROMPT_APP := app/prompt_engine.f90
NATURAL_EVAL_APP := app/natural_eval.f90
PROMPT_FIXTURE_SOURCE := tools/prompt_resume_fixture.f90
CONTEXT_TEST_SOURCES := test/fixed_budget_context_tests.f90
CONTEXT_QUALITY_AB_SOURCE := tools/benchmark_context_quality.f90
SLOT_QUALITY_AB_SOURCE := tools/benchmark_slot_quality.f90
NATURAL_QUALITY_AB_SOURCE := tools/benchmark_natural_quality.f90
ENGINE := $(BUILD_DIR)/hds_engine
PRETRAIN := $(BUILD_DIR)/vegai_pretrain
CAUSAL_LM := $(BUILD_DIR)/vegai_causal_lm
NEURAL_LM := $(BUILD_DIR)/vegai_neural_lm
NEURAL_PRETRAIN := $(BUILD_DIR)/vegai_neural_pretrain
FIELD := $(BUILD_DIR)/vegai_field
CHAT := $(BUILD_DIR)/vegai_chat
DIALOGUE_EVAL := $(BUILD_DIR)/vegai_dialogue_eval
CONTROL := $(BUILD_DIR)/vegai_control
CORPUS_OVERLAP := $(BUILD_DIR)/vegai_corpus_overlap
UD_TEXT_EXTRACT := $(BUILD_DIR)/vegai_ud_text_extract
SPARSE_BENCH := $(BUILD_DIR)/vegai_sparse_layer_bench
SPARSE_TESTS := $(BUILD_DIR)/vegai_sparse_layer_tests
CONTEXT_ENGINE := $(BUILD_DIR)/vegai_context_engine
PROMPT_ENGINE := $(BUILD_DIR)/vegai_prompt_engine
NATURAL_EVAL := $(BUILD_DIR)/vegai_natural_eval
PROMPT_FIXTURE := $(BUILD_DIR)/vegai_prompt_resume_fixture
CONTEXT_TESTS := $(BUILD_DIR)/vegai_context_tests
CONTEXT_QUALITY_AB := $(BUILD_DIR)/vegai_context_quality_ab
SLOT_QUALITY_AB := $(BUILD_DIR)/vegai_slot_quality_ab
NATURAL_QUALITY_AB := $(BUILD_DIR)/vegai_natural_quality_ab
TEST_BINARY := $(BUILD_DIR)/vegai_tests
UTF8_STREAM_TESTS := $(BUILD_DIR)/vegai_utf8_stream_tests
PREFLIGHT_TESTS := $(BUILD_DIR)/vegai_corpus_preflight_tests
# Runtime-only exact-size benchmark artifacts are intentionally ignored by Git.
FIXTURE_9MIB_CORPUS ?= corpora/synthetic_hdc_9mib.txt
FIXTURE_9MIB_MANIFEST ?= corpora/synthetic_hdc_9mib.manifest.json
FIXTURE_9MIB_CHECKPOINT ?= checkpoints/synthetic_hdc_9mib.vgai
FIXTURE_9MIB_REPORT ?= pretraining-runs/synthetic_hdc_9mib.json
# Exact decimal 900MB + 90MB controlled continuation artifacts are large and
# remain ignored. Prefer the dedicated GitHub workflow for CI-hosted evidence.
CONTINUOUS_900_RUN_DIR ?= pretraining-runs/synthetic_900mb_then_90mb

.PHONY: all engine pretrain causal-lm neural-lm neural-pretrain field chat dialogue-eval control context-engine prompt-engine context-tests utf8-stream-tests test fortran-check portable-runtime-check run run-pretrain run-causal-lm run-neural-lm run-neural-pretrain run-field run-chat run-control check legacy-python-check field-check sparse-bench sparse-tests optimizer-check tool-check release-check release-registry-check data-check capability-check lmm-check dialogue-check turkish-dialogue-data-check quality-check continuous-quality-check prompt-corpus-check attested-pretrain-check lookup-cache-compare fixture-9mib verify-fixture-9mib pretrain-9mib pretrain-900mb-continuous clean clean-all debug portable native portable-test native-test

# Default target: portable optimized GNU binaries without host-only ISA flags.
all: engine pretrain causal-lm neural-lm neural-pretrain field chat dialogue-eval control corpus-overlap ud-text-extract sparse-bench context-engine prompt-engine natural-eval

# Each source is compiled once per library and independent modules can build
# in parallel. Module dependencies below must follow the USE statements; the
# build-graph regression checks that this list does not drift from the sources.
HDS_LIBRARY := $(BUILD_DIR)/libvegai_hds.a
CONTEXT_LIBRARY := $(BUILD_DIR)/libvegai_context.a
HDS_OBJ_DIR := $(BUILD_DIR)/obj-hds
CONTEXT_OBJ_DIR := $(BUILD_DIR)/obj-context
HDS_OBJECTS := $(addprefix $(HDS_OBJ_DIR)/,$(notdir $(SOURCES:.f90=.o)))
CONTEXT_OBJECTS := $(addprefix $(CONTEXT_OBJ_DIR)/,$(notdir $(CONTEXT_SOURCES:.f90=.o)))
HDS_MOD_DIR := $(BUILD_DIR)/mod-hds
CONTEXT_MOD_DIR := $(BUILD_DIR)/mod-context
COMPILE_CONFIG := $(BUILD_DIR)/.compile-config
LINK_CONFIG := $(BUILD_DIR)/.link-config

# Delete a partially written target after a failed compile/link. The module
# repair check below then regenerates any missing producer before a consumer.
.DELETE_ON_ERROR:
.PHONY: FORCE
FORCE:

$(BUILD_DIR):
	mkdir -p "$@"

$(HDS_OBJ_DIR) $(CONTEXT_OBJ_DIR): | $(BUILD_DIR)
	mkdir -p "$@"

# Make does not otherwise notice compiler/flag changes. Keep timestamps stable
# for identical configuration so no-op builds stay incremental. Each profile
# still has its own build directory; replacing a compiler in place requires a
# clean rebuild, since command/flags alone cannot attest its binary identity.
sh_quote = $(subst ','"'"',$(1))
$(COMPILE_CONFIG): FORCE | $(BUILD_DIR)
	@printf '%s\n' 'FC=$(call sh_quote,$(FC))' 'FFLAGS=$(call sh_quote,$(FFLAGS))' 'AR=$(call sh_quote,$(AR))' > "$@.tmp"
	@if ! cmp -s "$@.tmp" "$@"; then mv "$@.tmp" "$@"; else rm -f "$@.tmp"; fi

$(LINK_CONFIG): FORCE | $(BUILD_DIR)
	@printf '%s\n' 'FC=$(call sh_quote,$(FC))' 'FFLAGS=$(call sh_quote,$(FFLAGS))' 'LDFLAGS=$(call sh_quote,$(LDFLAGS))' > "$@.tmp"
	@if ! cmp -s "$@.tmp" "$@"; then mv "$@.tmp" "$@"; else rm -f "$@.tmp"; fi

# A missing individual .mod (not just a missing directory) must invalidate the
# corresponding library. Every listed source defines one same-named module.
# Touching the stable marker only on damage forces all producers to recover,
# including after a failed/incomplete previous compilation.
$(HDS_MOD_DIR)/.ready: FORCE | $(BUILD_DIR)
	@mkdir -p "$(HDS_MOD_DIR)"
	@missing=0; for module in $(basename $(notdir $(SOURCES))); do \
	  test -f "$(HDS_MOD_DIR)/$$module.mod" || missing=1; \
	done; if test ! -f "$@" || test "$$missing" = 1; then touch "$@"; fi

$(CONTEXT_MOD_DIR)/.ready: FORCE | $(BUILD_DIR)
	@mkdir -p "$(CONTEXT_MOD_DIR)"
	@missing=0; for module in $(basename $(notdir $(CONTEXT_SOURCES))); do \
	  test -f "$(CONTEXT_MOD_DIR)/$$module.mod" || missing=1; \
	done; if test ! -f "$@" || test "$$missing" = 1; then touch "$@"; fi

$(HDS_OBJ_DIR)/%.o: src/%.f90 $(COMPILE_CONFIG) $(HDS_MOD_DIR)/.ready Makefile | $(HDS_OBJ_DIR)
	$(FC) $(FFLAGS) -J"$(abspath $(HDS_MOD_DIR))" -I"$(abspath $(HDS_MOD_DIR))" -c "src/$*.f90" -o "$@"
	@test -f "$(HDS_MOD_DIR)/$*.mod" || { echo "expected module $*.mod was not produced" >&2; exit 1; }

$(CONTEXT_OBJ_DIR)/%.o: src/%.f90 $(COMPILE_CONFIG) $(CONTEXT_MOD_DIR)/.ready Makefile | $(CONTEXT_OBJ_DIR)
	$(FC) $(FFLAGS) -J"$(abspath $(CONTEXT_MOD_DIR))" -I"$(abspath $(CONTEXT_MOD_DIR))" -c "src/$*.f90" -o "$@"
	@test -f "$(CONTEXT_MOD_DIR)/$*.mod" || { echo "expected module $*.mod was not produced" >&2; exit 1; }

# Fortran USE edges (object prerequisites provide module build ordering).
$(HDS_OBJ_DIR)/tokenless_field.o: $(HDS_OBJ_DIR)/hyper_types.o
$(HDS_OBJ_DIR)/subword_bpe_training.o: $(HDS_OBJ_DIR)/subword_bpe.o $(HDS_OBJ_DIR)/utf8_record_stream.o
$(HDS_OBJ_DIR)/neural_pretraining.o: $(HDS_OBJ_DIR)/subword_bpe.o $(HDS_OBJ_DIR)/utf8_record_stream.o $(HDS_OBJ_DIR)/neural_retention_decoder.o
$(HDS_OBJ_DIR)/vocabulary.o: $(HDS_OBJ_DIR)/text_tokenizer.o
$(HDS_OBJ_DIR)/symbolic_memory.o: $(HDS_OBJ_DIR)/hyper_types.o
$(HDS_OBJ_DIR)/resonator_network.o: $(HDS_OBJ_DIR)/hyper_types.o $(HDS_OBJ_DIR)/symbolic_memory.o
$(HDS_OBJ_DIR)/reasoning_core.o: $(HDS_OBJ_DIR)/hyper_types.o $(HDS_OBJ_DIR)/symbolic_memory.o
$(HDS_OBJ_DIR)/hdc_pretraining.o: $(HDS_OBJ_DIR)/hyper_types.o $(HDS_OBJ_DIR)/text_tokenizer.o $(HDS_OBJ_DIR)/vocabulary.o
$(HDS_OBJ_DIR)/dialogue_retrieval.o: $(HDS_OBJ_DIR)/hyper_types.o $(HDS_OBJ_DIR)/text_tokenizer.o $(HDS_OBJ_DIR)/turkish_text.o $(HDS_OBJ_DIR)/hdc_pretraining.o
$(HDS_OBJ_DIR)/fortran_quality_gate.o: $(HDS_OBJ_DIR)/hyper_types.o $(HDS_OBJ_DIR)/text_tokenizer.o $(HDS_OBJ_DIR)/hdc_pretraining.o $(HDS_OBJ_DIR)/portable_cli.o
$(HDS_OBJ_DIR)/corpus_preflight.o: $(HDS_OBJ_DIR)/fortran_integrity.o $(HDS_OBJ_DIR)/text_tokenizer.o
$(HDS_OBJ_DIR)/fortran_control.o: $(HDS_OBJ_DIR)/hdc_pretraining.o $(HDS_OBJ_DIR)/fortran_integrity.o $(HDS_OBJ_DIR)/corpus_preflight.o $(HDS_OBJ_DIR)/fortran_quality_gate.o $(HDS_OBJ_DIR)/portable_cli.o
$(HDS_OBJ_DIR)/sparse_expert_layer.o: $(HDS_OBJ_DIR)/hyper_types.o
$(CONTEXT_OBJ_DIR)/fixed_budget_backward.o: $(CONTEXT_OBJ_DIR)/fixed_budget_context.o
$(CONTEXT_OBJ_DIR)/fixed_budget_optimizer.o: $(CONTEXT_OBJ_DIR)/fixed_budget_context.o $(CONTEXT_OBJ_DIR)/fixed_budget_backward.o
$(CONTEXT_OBJ_DIR)/fixed_budget_training.o: $(CONTEXT_OBJ_DIR)/fixed_budget_context.o $(CONTEXT_OBJ_DIR)/fixed_budget_backward.o $(CONTEXT_OBJ_DIR)/fixed_budget_optimizer.o
$(CONTEXT_OBJ_DIR)/fixed_budget_natural_eval.o: $(CONTEXT_OBJ_DIR)/fixed_budget_context.o $(CONTEXT_OBJ_DIR)/fixed_budget_training.o

$(HDS_LIBRARY): $(HDS_OBJECTS) Makefile
	rm -f "$@.tmp"
	$(AR) rcs "$@.tmp" $(HDS_OBJECTS)
	mv "$@.tmp" "$@"

$(CONTEXT_LIBRARY): $(CONTEXT_OBJECTS) Makefile
	rm -f "$@.tmp"
	$(AR) rcs "$@.tmp" $(CONTEXT_OBJECTS)
	mv "$@.tmp" "$@"

engine: $(ENGINE)

$(ENGINE): $(APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

pretrain: $(PRETRAIN)

$(PRETRAIN): $(PRETRAIN_APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(PRETRAIN_APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

causal-lm: $(CAUSAL_LM)

$(CAUSAL_LM): $(CAUSAL_LM_APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(CAUSAL_LM_APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

neural-lm: $(NEURAL_LM)

$(NEURAL_LM): $(NEURAL_LM_APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(NEURAL_LM_APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

neural-pretrain: $(NEURAL_PRETRAIN)

$(NEURAL_PRETRAIN): $(NEURAL_PRETRAIN_APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(NEURAL_PRETRAIN_APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

field: $(FIELD)

$(FIELD): $(FIELD_APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(FIELD_APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

chat: $(CHAT)

$(CHAT): $(CHAT_APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(CHAT_APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

dialogue-eval: $(DIALOGUE_EVAL)

$(DIALOGUE_EVAL): $(DIALOGUE_EVAL_APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(DIALOGUE_EVAL_APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

control: $(CONTROL)

$(CONTROL): $(CONTROL_APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(CONTROL_APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

.PHONY: corpus-overlap corpus-overlap-check
corpus-overlap: $(CORPUS_OVERLAP)

$(CORPUS_OVERLAP): $(CORPUS_OVERLAP_APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(CORPUS_OVERLAP_APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

corpus-overlap-check: $(CORPUS_OVERLAP) $(PROMPT_FIXTURE)
	cmake -DOVERLAP="$(abspath $(CORPUS_OVERLAP))" -DFIXTURE="$(abspath $(PROMPT_FIXTURE))" \
	  -DWORK_ROOT="$(abspath $(BUILD_DIR))" -P test/native_corpus_overlap.cmake

.PHONY: ud-text-extract ud-text-extract-check
ud-text-extract: $(UD_TEXT_EXTRACT)

$(UD_TEXT_EXTRACT): $(UD_TEXT_EXTRACT_APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(UD_TEXT_EXTRACT_APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

ud-text-extract-check: $(UD_TEXT_EXTRACT) $(PROMPT_FIXTURE)
	cmake -DEXTRACT="$(abspath $(UD_TEXT_EXTRACT))" -DFIXTURE="$(abspath $(PROMPT_FIXTURE))" \
	  -DWORK_ROOT="$(abspath $(BUILD_DIR))" -P test/native_ud_text_extract.cmake

sparse-bench: $(SPARSE_BENCH)

$(SPARSE_BENCH): $(SPARSE_APP) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(SPARSE_APP) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

sparse-tests: $(SPARSE_TESTS)
	$(SPARSE_TESTS)

$(SPARSE_TESTS): $(SPARSE_TEST_SOURCES) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(SPARSE_TEST_SOURCES) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

context-engine: $(CONTEXT_ENGINE)

$(CONTEXT_ENGINE): $(CONTEXT_APP) $(CONTEXT_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(CONTEXT_MOD_DIR) -I$(CONTEXT_MOD_DIR) $(CONTEXT_APP) $(CONTEXT_LIBRARY) -o $@ $(LDFLAGS)

prompt-engine: $(PROMPT_ENGINE)

$(PROMPT_ENGINE): $(PROMPT_APP) $(CONTEXT_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(CONTEXT_MOD_DIR) -I$(CONTEXT_MOD_DIR) $(PROMPT_APP) $(CONTEXT_LIBRARY) -o $@ $(LDFLAGS)

.PHONY: natural-eval natural-eval-check
natural-eval: $(NATURAL_EVAL)

natural-eval-check: $(NATURAL_EVAL) $(PROMPT_ENGINE) $(PROMPT_FIXTURE)
	cmake -DEVAL="$(abspath $(NATURAL_EVAL))" -DENGINE="$(abspath $(PROMPT_ENGINE))" \
	  -DFIXTURE="$(abspath $(PROMPT_FIXTURE))" -DWORK_ROOT="$(abspath $(BUILD_DIR))" -P test/native_natural_eval.cmake

$(NATURAL_EVAL): $(NATURAL_EVAL_APP) $(CONTEXT_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(CONTEXT_MOD_DIR) -I$(CONTEXT_MOD_DIR) $(NATURAL_EVAL_APP) $(CONTEXT_LIBRARY) -o $@ $(LDFLAGS)

$(PROMPT_FIXTURE): $(PROMPT_FIXTURE_SOURCE) $(CONTEXT_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(CONTEXT_MOD_DIR) -I$(CONTEXT_MOD_DIR) $(PROMPT_FIXTURE_SOURCE) $(CONTEXT_LIBRARY) -o $@ $(LDFLAGS)

# Native CMake/Fortran tests; no Python interpreter is used.
.PHONY: prompt-contract-check
prompt-contract-check: $(PROMPT_ENGINE) $(PROMPT_FIXTURE) $(CONTEXT_ENGINE)
	cmake -DENGINE="$(abspath $(PROMPT_ENGINE))" -DCONTEXT="$(abspath $(CONTEXT_ENGINE))" \
	  -DFIXTURE="$(abspath $(PROMPT_FIXTURE))" -DWORK_ROOT="$(abspath $(BUILD_DIR))" -P test/native_prompt_resume.cmake
	cmake -DENGINE="$(abspath $(PROMPT_ENGINE))" -DFIXTURE="$(abspath $(PROMPT_FIXTURE))" \
	  -DWORK_ROOT="$(abspath $(BUILD_DIR))" -P test/native_prompt_json.cmake
	cmake -DENGINE="$(abspath $(PROMPT_ENGINE))" -DWORK_ROOT="$(abspath $(BUILD_DIR))" -P test/native_natural_profile.cmake

context-tests: $(CONTEXT_TESTS)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(CONTEXT_TESTS)

$(CONTEXT_TESTS): $(CONTEXT_TEST_SOURCES) $(CONTEXT_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(CONTEXT_MOD_DIR) -I$(CONTEXT_MOD_DIR) $(CONTEXT_TEST_SOURCES) $(CONTEXT_LIBRARY) -o $@ $(LDFLAGS)

.PHONY: context-quality-ab
context-quality-ab: $(CONTEXT_QUALITY_AB)

$(CONTEXT_QUALITY_AB): $(CONTEXT_QUALITY_AB_SOURCE) $(CONTEXT_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(CONTEXT_MOD_DIR) -I$(CONTEXT_MOD_DIR) $(CONTEXT_QUALITY_AB_SOURCE) $(CONTEXT_LIBRARY) -o $@ $(LDFLAGS)

.PHONY: slot-quality-ab slot-quality-contract-check
slot-quality-ab: $(SLOT_QUALITY_AB)

slot-quality-contract-check: $(SLOT_QUALITY_AB) $(PROMPT_FIXTURE)
	cmake -DBENCH="$(abspath $(SLOT_QUALITY_AB))" -DFIXTURE="$(abspath $(PROMPT_FIXTURE))" \
	  -DWORK_ROOT="$(abspath $(BUILD_DIR))" -P test/native_slot_quality_cli.cmake

$(SLOT_QUALITY_AB): $(SLOT_QUALITY_AB_SOURCE) $(CONTEXT_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(CONTEXT_MOD_DIR) -I$(CONTEXT_MOD_DIR) $(SLOT_QUALITY_AB_SOURCE) $(CONTEXT_LIBRARY) -o $@ $(LDFLAGS)

.PHONY: natural-quality-ab natural-quality-contract-check
natural-quality-ab: $(NATURAL_QUALITY_AB)

natural-quality-contract-check: $(NATURAL_QUALITY_AB) $(PROMPT_FIXTURE)
	cmake -DBENCH="$(abspath $(NATURAL_QUALITY_AB))" -DFIXTURE="$(abspath $(PROMPT_FIXTURE))" \
	  -DWORK_ROOT="$(abspath $(BUILD_DIR))" -P test/native_natural_quality_cli.cmake

$(NATURAL_QUALITY_AB): $(NATURAL_QUALITY_AB_SOURCE) $(CONTEXT_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(CONTEXT_MOD_DIR) -I$(CONTEXT_MOD_DIR) $(NATURAL_QUALITY_AB_SOURCE) $(CONTEXT_LIBRARY) -o $@ $(LDFLAGS)

$(TEST_BINARY): $(TEST_SOURCES) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(TEST_SOURCES) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

$(UTF8_STREAM_TESTS): $(UTF8_STREAM_TEST_SOURCE) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(UTF8_STREAM_TEST_SOURCE) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

$(PREFLIGHT_TESTS): $(PREFLIGHT_TEST_SOURCE) $(HDS_LIBRARY) $(LINK_CONFIG) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(HDS_MOD_DIR) -I$(HDS_MOD_DIR) $(PREFLIGHT_TEST_SOURCE) $(HDS_LIBRARY) -o $@ $(LDFLAGS)

utf8-stream-tests: $(UTF8_STREAM_TESTS)
	$(UTF8_STREAM_TESTS)

test: $(TEST_BINARY) $(UTF8_STREAM_TESTS)
	@set -eu; work=$$(mktemp -d "$(abspath $(BUILD_DIR))/vegai-properties-XXXXXXXX"); \
	  trap 'rm -rf "$$work"' EXIT; token=$${work##*/}; \
	  printf '%s\n' "$$token" > "$$work/.vegai-test-scratch"; cd "$$work"; \
	  VEGAI_TEST_SCRATCH_TOKEN="$$token" OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} "$(abspath $(TEST_BINARY))"; \
	  "$(abspath $(UTF8_STREAM_TESTS))"

# Default Fortran-only verification: no Python interpreter, shell inventory,
# or optional accelerator SDK is invoked by these executable paths.
fortran-check: $(TEST_BINARY) $(UTF8_STREAM_TESTS) $(PREFLIGHT_TESTS) $(CONTROL) $(SPARSE_TESTS)
	@set -eu; work=$$(mktemp -d "$(abspath $(BUILD_DIR))/vegai-control-XXXXXXXX"); \
	  trap 'rm -rf "$$work"' EXIT; token=$${work##*/}; \
	  printf '%s\n' "$$token" > "$$work/.vegai-test-scratch"; cd "$$work"; \
	  VEGAI_TEST_SCRATCH_TOKEN="$$token" OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} "$(abspath $(TEST_BINARY))"; \
	  "$(abspath $(UTF8_STREAM_TESTS))"; \
	  "$(abspath $(PREFLIGHT_TESTS))"; \
	  OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} "$(abspath $(SPARSE_TESTS))"; \
	  "$(abspath $(CONTROL))" profile --format json; \
	  "$(abspath $(CONTROL))" corpus-manifest "$(abspath examples/corpus/toy_hdc_corpus.txt)" --strict-contract; \
	  OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} "$(abspath $(CONTROL))" smoke \
	    "$(abspath examples/corpus/toy_hdc_corpus.txt)" "$$work/smoke.vgai"; \
	  "$(abspath $(CONTROL))" attest "$(abspath examples/corpus/toy_hdc_corpus.txt)" "$$work/smoke.vgai" --format json

# Same-host binary launch/train/inspect smoke. CMake is test infrastructure,
# not a runtime dependency; Python is only needed for the legacy verifier.
portable-runtime-check: $(ENGINE) $(PRETRAIN)
	cmake -DENGINE="$(abspath $(ENGINE))" -DPRETRAIN="$(abspath $(PRETRAIN))" \
	  -DCORPUS="$(abspath examples/corpus/toy_hdc_corpus.txt)" -DWORK_ROOT="$(abspath $(BUILD_DIR))" \
	  -P test/native_portable_runtime.cmake

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
	$(MAKE) test OPT_FLAGS='-O0 -g -fcheck=all -finit-real=snan -fbacktrace -ffpe-trap=invalid,zero,overflow'

# The default verification path is native Fortran only. The legacy suite is
# retained explicitly until its individual contracts have native replacements.
check: fortran-check context-tests

# Expanded native CLI contracts use CMake as a test runner. Keep the lighter
# `make check` independent of CMake for GNU Make/Fortran-only environments.
.PHONY: native-contract-check
native-contract-check: check prompt-contract-check natural-eval-check corpus-overlap-check ud-text-extract-check slot-quality-contract-check natural-quality-contract-check

legacy-python-check:
	python3 tools/verify_bounded_process.py
	python3 tools/verify_lmm_endpoint_transport.py
	python3 tools/verify_lmm_source_binding.py
	python3 tools/verify_system_contracts.py
	python3 tools/verify_quality_runner_binding.py
	python3 tools/verify_continuous_progress.py
	python3 tools/verify_training_streaming.py
	python3 tools/verify_neural_quality_budget.py
	python3 tools/verify_context_numerics.py
	python3 tools/verify_neural_turkish_epoch1.py
	python3 tools/verify_make_build.py
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
	python3 tools/verify_prompt_response_corpus.py

# The 600 MB prompt/response corpus generator is pure Python, so its contract
# (exact byte count, determinism, manifest, probe families) is checked without a
# Fortran toolchain.
prompt-corpus-check:
	python3 tools/verify_prompt_response_corpus.py

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

# Measured A/B of the tied-output update: per-token steps versus averaged
# mini-batch steps over a byte-identical synthetic corpus. Reports tokens/s,
# held-out NLL at equal tokens, and NLL improvement per second, and fails the
# target when the measured ratios miss the gates.
neural-throughput-check: $(NEURAL_PRETRAIN)
	python3 tools/benchmark_neural_batch_throughput.py --neural-bin $(NEURAL_PRETRAIN) \
		--output $(BUILD_DIR)/neural-throughput.json

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
		--report "$(FIXTURE_9MIB_REPORT)" --max-vocab 1024 --min-frequency 1 \
		--window 2 --context-mode hierarchical --epochs 1 --omp-threads 2 --overwrite

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
