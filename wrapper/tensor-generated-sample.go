// NOTE: this is a sample for OCaml generated code for `tensor-generated.go`
package wrapper

// #include "stdlib.h"
import "C"

import (
	"unsafe"

	gt "github.com/sugarme/gotch"
	lib "github.com/sugarme/gotch/libtch"
)

func (ts Tensor) To(device gt.Device) (retVal Tensor, err error) {

	// TODO: how to get pointer to CUDA memory???
	// C.cuMemAlloc((*C.ulonglong)(cudaPtr), 1) // 0 byte is invalid
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgTo((*lib.Ctensor)(ptr), ts.ctensor, int(device.CInt()))

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) Device() (retVal gt.Device, err error) {
	cInt := lib.AtDevice(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	var device gt.Device

	return device.OfCInt(int32(cInt)), nil
}

func (ts Tensor) Matmul(other Tensor) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgMatmul(ptr, ts.ctensor, other.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}
