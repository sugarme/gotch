// NOTE: this is a sample for OCaml generated code for `tensor-generated.go`
package wrapper

// #include "stdlib.h"
import "C"

import (
	"log"
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

func (ts Tensor) Matmul(other Tensor) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgMatmul(ptr, ts.ctensor, other.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustMatMul(other Tensor) (retVal Tensor) {
	retVal, err := ts.Matmul(other)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Grad() (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgGrad(ptr, ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustGrad() (retVal Tensor) {
	retVal, err := ts.Grad()
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Detach_() (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgDetach_(ptr, ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustDetach_() (retVal Tensor) {
	retVal, err := ts.Detach_()
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Zero_() (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgZero_(ptr, ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustZero_() (retVal Tensor) {
	retVal, err := ts.Zero_()
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) SetRequiresGrad(rb bool) (retVal Tensor, err error) {
	var r int = 0
	if rb {
		r = 1
	}

	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgSetRequiresGrad(ptr, ts.ctensor, r)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustSetRequiresGrad(rb bool) (retVal Tensor) {
	retVal, err := ts.SetRequiresGrad(rb)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Mul(other Tensor) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgMul(ptr, ts.ctensor, other.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustMul(other Tensor) (retVal Tensor) {
	retVal, err := ts.Mul(other)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Add(other Tensor) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgAdd(ptr, ts.ctensor, other.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustAdd(other Tensor) (retVal Tensor) {
	retVal, err := ts.Add(other)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts *Tensor) AddG(other Tensor) (err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgAdd(ptr, ts.ctensor, other.ctensor)

	if err = TorchErr(); err != nil {
		return err
	}

	ts = &Tensor{ctensor: *ptr}

	return nil
}

func (ts *Tensor) MustAddG(other Tensor) {
	err := ts.AddG(other)
	if err != nil {
		log.Fatal(err)
	}
}
