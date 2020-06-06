// NOTE: this is a sample for OCaml generated code for `tensor-generated.go`
package wrapper

// #include "stdlib.h"
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
	// C.cuMemAlloc((*C.ulonglong)(cudaPtr), 1) // 0 byte is invalid
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgTo((*lib.Ctensor)(ptr), ts.ctensor, int(device.CInt()))

	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}

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
