// NOTE: this is a sample for OCaml generated code for `tensor-generated.go`
package wrapper

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
// #cgo linux,amd64,!nogpu CFLAGS: -I/usr/local/cuda/include
// #cgo linux,amd64,!nogpu LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn -lcaffe2_nvrtc -lnvrtc-builtins -lnvrtc -lnvToolsExt -L/opt/libtorch/lib -lc10_cuda -ltorch_cuda
// # include <cuda.h>
import "C"

import (
	// "fmt"
	"log"
	"unsafe"

	gt "github.com/sugarme/gotch"
	lib "github.com/sugarme/gotch/libtch"
)

func (ts Tensor) To(device gt.Device) Tensor {

	// TODO: how to get pointer to CUDA memory???
	// Something like `C.cudaMalloc()`???
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	// var cudaPtr unsafe.Pointer
	// C.cuMemAlloc((*C.ulonglong)(cudaPtr), 1)
	// fmt.Printf("Cuda Pointer: %v\n", &cudaPtr)
	// ptr := (*lib.Ctensor)(unsafe.Pointer(C.cuMemAlloc(device.CInt(), 0)))

	// var ptr unsafe.Pointer
	// lib.AtgTo(ptr, ts.ctensor, int(device.CInt()))
	lib.AtgTo((*lib.Ctensor)(ptr), ts.ctensor, int(device.CInt()))
	// lib.AtgTo((*lib.Ctensor)(cudaPtr), ts.ctensor, int(device.CInt()))

	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}

	// return Tensor{ctensor: *(*lib.Ctensor)(unsafe.Pointer(&ptr))}
	// return Tensor{ctensor: (lib.Ctensor)(cudaPtr)}

	return Tensor{ctensor: *ptr}
}

func (ts Tensor) Device() gt.Device {
	cInt := lib.AtDevice(ts.ctensor)

	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}

	var device gt.Device

	return device.OfCInt(int32(cInt))
}
