# Makefile for gotch - PyTorch 2.10.0 Go bindings
# Architecture-specific configuration is loaded from arch/ directory

SHELL := /bin/bash

# Detect OS and architecture
OS    := $(shell uname -s | tr '[:upper:]' '[:lower:]')
ARCH  := $(shell uname -m)

# Normalize architecture names
ifeq ($(ARCH),x86_64)
	ARCH := amd64
else ifeq ($(ARCH),aarch64)
	ARCH := arm64
endif

# Allow override of architecture configuration
# Examples:
#   make ARCH_CONFIG=linux-amd64-cuda build
#   make ARCH_CONFIG=darwin-arm64 test
ARCH_CONFIG ?= $(OS)-$(ARCH)

# Include architecture-specific configuration
ARCH_FILE := arch/$(ARCH_CONFIG).mk
ifeq ($(wildcard $(ARCH_FILE)),)
	$(error Architecture config file not found: $(ARCH_FILE). Available: $(wildcard arch/*.mk))
endif
include $(ARCH_FILE)

# Go test flags
TEST_FLAGS := -v
TEST_TIMEOUT := 5m

.PHONY: all build test test-nn test-ts clean help ffi-validate

# Default target
all: build

# Build all core packages
build:
	@echo "Building gotch core packages..."
	@go build -v . ./ts ./nn ./vision

# Run all tests
test: test-nn test-ts ffi-validate
	@echo "All tests completed"

# Run nn package tests
# Running with -p 1 to force sequential execution (PyTorch 2.10.0 thread-local gradient state)
test-nn:
	@echo "Running nn package tests..."
	@go test $(TEST_FLAGS) -timeout $(TEST_TIMEOUT) -p 1 -parallel 1 ./nn

# Run ts package tests
test-ts:
	@echo "Running ts package tests..."
	@go test $(TEST_FLAGS) -timeout $(TEST_TIMEOUT) ./ts

# Run specific test in nn package
# Usage: make test-nn-specific TEST=TestInitTensor_Memcheck
test-nn-specific:
	@echo "Running specific nn test: $(TEST)..."
	@go test $(TEST_FLAGS) -timeout $(TEST_TIMEOUT) -run $(TEST) ./nn

# Run specific test in ts package
# Usage: make test-ts-specific TEST=TestTensor
test-ts-specific:
	@echo "Running specific ts test: $(TEST)..."
	@go test $(TEST_FLAGS) -timeout $(TEST_TIMEOUT) -run $(TEST) ./ts

# Run tests with coverage
test-nn-coverage:
	@echo "Running nn tests with coverage..."
	@go test -v -timeout $(TEST_TIMEOUT) -coverprofile=coverage-nn.out ./nn
	@go tool cover -html=coverage-nn.out -o coverage-nn.html
	@echo "Coverage report saved to coverage-nn.html"

test-ts-coverage:
	@echo "Running ts tests with coverage..."
	@go test -v -timeout $(TEST_TIMEOUT) -coverprofile=coverage-ts.out ./ts
	@go tool cover -html=coverage-ts.out -o coverage-ts.html
	@echo "Coverage report saved to coverage-ts.html"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@go clean -cache -testcache
	@rm -f coverage-*.out coverage-*.html

# Display build environment
env:
	@echo "Build Environment:"
	@echo "  Platform:           $(PLATFORM_DESC)"
	@echo "  Arch Config:        $(ARCH_CONFIG) ($(ARCH_FILE))"
	@echo "  LIBTORCH_PATH:      $(LIBTORCH_PATH)"
	@echo "  $(RUNTIME_LIB_VAR): $($(RUNTIME_LIB_VAR))"
	@echo ""
	@echo "CGO Flags:"
	@echo "  CGO_CFLAGS:         $(CGO_CFLAGS)"
	@echo "  CGO_LDFLAGS:        $(CGO_LDFLAGS)"
	@echo "  CGO_CXXFLAGS:       $(CGO_CXXFLAGS)"
	@echo ""
	@echo "Go version:"
	@go version
	@echo ""
	@echo "LibTorch version: 2.10.0"

# Check if MPS is available
check-mps:
	@echo "Checking MPS availability..."
	@go run -exec 'env DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH)' tools/check_device.go || echo "Create tools/check_device.go to test device availability"

# Validate FFI type conversions
ffi-validate:
	@echo "Validating FFI type conversions..."
	@go run tools/ffi-validation/main.go

# Help target
help:
	@echo "Gotch Makefile - PyTorch 2.10.0 Go Bindings"
	@echo ""
	@echo "Current Configuration:"
	@echo "  Platform:      $(PLATFORM_DESC)"
	@echo "  Arch Config:   $(ARCH_CONFIG)"
	@echo "  LibTorch:      $(LIBTORCH_PATH)"
	@echo ""
	@echo "Targets:"
	@echo "  make build              - Build all core packages"
	@echo "  make test               - Run all tests (nn + ts)"
	@echo "  make test-nn            - Run nn package tests"
	@echo "  make test-ts            - Run ts package tests"
	@echo "  make test-nn-specific   - Run specific nn test (TEST=TestName)"
	@echo "  make test-ts-specific   - Run specific ts test (TEST=TestName)"
	@echo "  make test-nn-coverage   - Run nn tests with coverage report"
	@echo "  make test-ts-coverage   - Run ts tests with coverage report"
	@echo "  make clean              - Clean build artifacts and caches"
	@echo "  make env                - Display build environment"
	@echo "  make check-mps          - Check MPS device availability"
	@echo "  make ffi-validate       - Validate FFI type conversions (C <-> Go)"
	@echo "  make help               - Show this help message"
	@echo ""
	@echo "Architecture Configuration:"
	@echo "  Default: auto-detected (current: $(ARCH_CONFIG))"
	@echo "  Override: make ARCH_CONFIG=linux-amd64-cuda build"
	@echo "  Available configs: $(notdir $(basename $(wildcard arch/*.mk)))"
	@echo ""
	@echo "Examples:"
	@echo "  make test-nn"
	@echo "  make test-nn-specific TEST=TestInitTensor_Memcheck"
	@echo "  make ARCH_CONFIG=linux-amd64-cuda build"
	@echo "  LIBTORCH_PATH=/custom/path make build"
