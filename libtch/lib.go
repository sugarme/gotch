package libtch

// CGO flags are set via environment variables exported by the arch/*.mk
// Makefile includes (e.g., arch/darwin-arm64.mk). Run builds through
// `make` or export CGO_CFLAGS/CGO_LDFLAGS/CGO_CXXFLAGS manually.
//
// The only hardcoded flag is -I${SRCDIR} so CGO can find the local
// torch_api.h header next to the .cpp source.

// #cgo CFLAGS: -I${SRCDIR}
// #cgo CXXFLAGS: -I${SRCDIR}
import "C"
