package libtch

// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR} -g -O3
// #cgo CFLAGS: -I${SRCDIR} -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CFLAGS: -I/usr/local/include -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include
// #cgo LDFLAGS: -lstdc++ -ltorch -lc10 -ltorch_cpu
// #cgo LDFLAGS: -L/opt/libtorch/lib -L/lib64
// #cgo CXXFLAGS: -isystem /opt/libtorch/lib
// #cgo CXXFLAGS: -isystem /opt/libtorch/include
// #cgo CXXFLAGS: -isystem /opt/libtorch/include/torch/csrc/api/include
// #cgo CXXFLAGS: -isystem /opt/libtorch/include/torch/csrc
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
import "C"
