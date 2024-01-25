//go:build gotch_gpu

package libtch

// #cgo CFLAGS: -I${SRCDIR} -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -lstdc++ -ltorch -ltorch_cpu -ltorch_cuda -lc10
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcuda
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
// #cgo CXXFLAGS: -std=c++17
import "C"
