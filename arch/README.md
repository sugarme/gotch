# Architecture-Specific Configuration

This directory contains Makefile configuration files for different operating systems and architectures. The main Makefile automatically detects your platform and includes the appropriate configuration file.

## Available Configurations

- **darwin-arm64.mk** - macOS on Apple Silicon (M1/M2/M3)
- **darwin-amd64.mk** - macOS on Intel processors
- **linux-amd64.mk** - Linux x86_64 (CPU-only)
- **linux-amd64-cuda.mk** - Linux x86_64 with CUDA support

## How It Works

1. The main `Makefile` detects your OS and architecture
2. It includes the corresponding `.mk` file from this directory
3. The included file sets:
   - `LIBTORCH_PATH` - Path to LibTorch installation
   - `CGO_CFLAGS` - C compiler flags for CGO
   - `CGO_LDFLAGS` - Linker flags for CGO
   - `CGO_CXXFLAGS` - C++ compiler flags for CGO
   - Runtime library paths (DYLD_LIBRARY_PATH for macOS, LD_LIBRARY_PATH for Linux)

## Usage

### Default (Auto-Detection)

```bash
make build
make test
```

### Override Architecture

```bash
# Use CUDA configuration on Linux
make ARCH_CONFIG=linux-amd64-cuda build

# Use specific configuration
make ARCH_CONFIG=darwin-arm64 test
```

### Override LibTorch Path

```bash
# Temporary override
LIBTORCH_PATH=/custom/path make build

# Or set in environment
export LIBTORCH_PATH=/opt/libtorch
make build
```

### View Current Configuration

```bash
make env
```

## Creating a New Configuration

To add support for a new platform:

1. Create a new `.mk` file in this directory (e.g., `linux-arm64.mk`)
2. Copy an existing configuration as a template
3. Modify the paths and flags as needed
4. Key variables to set:
   - `LIBTORCH_PATH` - Default path to LibTorch
   - `RUNTIME_LIB_VAR` - Runtime library path variable name
   - `CGO_CFLAGS` - C compilation flags
   - `CGO_LDFLAGS` - Linker flags
   - `CGO_CXXFLAGS` - C++ flags
   - `PLATFORM_DESC` - Human-readable platform description

## Configuration Priority

1. Command-line: `LIBTORCH_PATH=/path make build`
2. Environment: `export LIBTORCH_PATH=/path`
3. Arch file default: `LIBTORCH_PATH ?= ...` in the `.mk` file

## Example Configuration File

```makefile
# Architecture configuration for Custom Platform

LIBTORCH_PATH ?= /opt/libtorch
RUNTIME_LIB_VAR := LD_LIBRARY_PATH

export CGO_CFLAGS := -I$(LIBTORCH_PATH)/include/torch/csrc/api/include -I$(LIBTORCH_PATH)/include
export CGO_CFLAGS += -O3 -Wall -Wno-deprecated-declarations

export CGO_LDFLAGS := -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -lc10
export CGO_LDFLAGS += -Wl,-rpath,$(LIBTORCH_PATH)/lib

export CGO_CXXFLAGS := -std=c++17

export CPATH := $(LIBTORCH_PATH)/include/torch/csrc/api/include:$(LIBTORCH_PATH)/include
export LIBRARY_PATH := $(LIBTORCH_PATH)/lib
export LD_LIBRARY_PATH := $(LIBTORCH_PATH)/lib

PLATFORM_DESC := Custom Platform Description
```

## Notes

- The `libtch/lib.go` file no longer contains hardcoded paths
- All CGO configuration is managed through the Makefile and these arch files
- This allows the same codebase to work across different platforms without code changes
- Always use `make build` instead of `go build` directly to ensure CGO flags are properly set
