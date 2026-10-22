# SPDX-License-Identifier: MIT
# Portable GNU Fortran convenience build.  CMake is recommended for IDE/CI use.

# Command-line overrides still win over this default (for example: make FC=ifx).
FC = gfortran
BUILD_DIR ?= build
OPENMP_FLAGS ?= -fopenmp
OPT_FLAGS ?= -O3 -march=native -mtune=native -funroll-loops
WARN_FLAGS ?= -std=f2023 -Wall -Wextra -Wimplicit-interface -Wcharacter-truncation
COMMON_FLAGS ?= -ffree-line-length-none
FFLAGS ?= $(OPT_FLAGS) $(WARN_FLAGS) $(COMMON_FLAGS) $(OPENMP_FLAGS)
LDFLAGS ?= $(OPENMP_FLAGS)

SOURCES := src/hyper_types.f90 src/symbolic_memory.f90 src/resonator_network.f90 src/reasoning_core.f90
APP := app/main.f90
TEST_SOURCES := test/test_support.f90 test/test_all.f90
ENGINE := $(BUILD_DIR)/hds_engine
TEST_BINARY := $(BUILD_DIR)/vegai_tests

.PHONY: all engine test run check clean debug

all: engine

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/mod

engine: $(ENGINE)

$(ENGINE): $(SOURCES) $(APP) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(APP) -o $@ $(LDFLAGS)

$(TEST_BINARY): $(SOURCES) $(TEST_SOURCES) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR)/mod -I$(BUILD_DIR)/mod $(SOURCES) $(TEST_SOURCES) -o $@ $(LDFLAGS)

test: $(TEST_BINARY)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(TEST_BINARY)

run: $(ENGINE)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-2} $(ENGINE)

# Debug build with bounds, undefined floating-point, and backtrace checks.
debug:
	$(MAKE) clean
	$(MAKE) test OPT_FLAGS='-O0 -g -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'

check:
	python3 tools/verify_reference.py

clean:
	rm -rf $(BUILD_DIR)
