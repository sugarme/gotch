# Architecture configuration for macOS ARM64 (Apple Silicon)
# This file is included by the main Makefile based on detected OS and architecture

# LibTorch installation path (can be overridden via LIBTORCH environment variable)
LIBTORCH_PATH ?= $(shell [ -d "$(HOME)/src/gotch/libtorch-2.10.0-macos" ] && echo "$(HOME)/src/gotch/libtorch-2.10.0-macos" || echo "$(CURDIR)/libtorch-2.10.0-macos")

# Runtime library path variable for macOS
RUNTIME_LIB_VAR := DYLD_LIBRARY_PATH

# CGO flags for compilation
export CGO_CFLAGS := -I$(LIBTORCH_PATH)/include/torch/csrc/api/include -I$(LIBTORCH_PATH)/include
export CGO_CFLAGS += -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function

export CGO_LDFLAGS := -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -lc10
export CGO_LDFLAGS += -Wl,-rpath,$(LIBTORCH_PATH)/lib

export CGO_CXXFLAGS := -std=c++17 -I$(LIBTORCH_PATH)/include/torch/csrc/api/include -I$(LIBTORCH_PATH)/include
export CGO_CXXFLAGS += -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function

# Environment for runtime (tests, etc.)
export CPATH := $(LIBTORCH_PATH)/include/torch/csrc/api/include:$(LIBTORCH_PATH)/include
export LIBRARY_PATH := $(LIBTORCH_PATH)/lib
export DYLD_LIBRARY_PATH := $(LIBTORCH_PATH)/lib

# Platform-specific notes
PLATFORM_DESC := macOS ARM64 (Apple Silicon) with MPS support
