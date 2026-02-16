package libtch

/*
CGO configuration is now managed via Makefile and arch-specific files in arch/ directory.
Build with 'make build' to ensure proper CGO flags are set.

The Makefile exports:
  CGO_CFLAGS   - C compilation flags including LibTorch include paths
  CGO_LDFLAGS  - Linker flags including LibTorch library paths and rpath
  CGO_CXXFLAGS - C++ compilation flags (e.g., -std=c++17)

To override configuration:
  LIBTORCH_PATH=/custom/path make build
  make ARCH_CONFIG=linux-amd64-cuda build
*/

// #cgo CFLAGS: -I${SRCDIR}
import "C"
