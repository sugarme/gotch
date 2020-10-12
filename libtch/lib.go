package libtch

// #cgo LDFLAGS: -lstdc++ -ltorch -lc10 -ltorch_cpu -L${SRCDIR}/libtorch/lib
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn -lcaffe2_nvrtc -lnvrtc-builtins -lnvrtc -lnvToolsExt -lc10_cuda -ltorch_cuda
// #cgo CFLAGS: -I${SRCDIR} -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR} -g -O3
// #cgo CXXFLAGS: -I${SRCDIR}/libtorch/lib -I${SRCDIR}/libtorch/include -I${SRCDIR}/libtorch/include/torch/csrc/api/include
import "C"
