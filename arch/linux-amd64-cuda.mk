# Architecture configuration for Linux x86_64 with CUDA
# This file is included by the main Makefile based on detected OS and architecture

# LibTorch installation path (can be overridden via LIBTORCH environment variable)
LIBTORCH_PATH ?= $(shell [ -d "/opt/libtorch-cuda" ] && echo "/opt/libtorch-cuda" || echo "$(CURDIR)/libtorch")

# Runtime library path variable for Linux
RUNTIME_LIB_VAR := LD_LIBRARY_PATH

# CGO flags for compilation
export CGO_CFLAGS := -I$(LIBTORCH_PATH)/include/torch/csrc/api/include -I$(LIBTORCH_PATH)/include
export CGO_CFLAGS += -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function

export CGO_LDFLAGS := -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda
export CGO_LDFLAGS += -Wl,-rpath,$(LIBTORCH_PATH)/lib

export CGO_CXXFLAGS := -std=c++17 -I$(LIBTORCH_PATH)/include/torch/csrc/api/include -I$(LIBTORCH_PATH)/include
export CGO_CXXFLAGS += -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function

# Environment for runtime (tests, etc.)
export CPATH := $(LIBTORCH_PATH)/include/torch/csrc/api/include:$(LIBTORCH_PATH)/include
export LIBRARY_PATH := $(LIBTORCH_PATH)/lib
export LD_LIBRARY_PATH := $(LIBTORCH_PATH)/lib

# Platform-specific notes
PLATFORM_DESC := Linux x86_64 with CUDA support
