// NOTE: this is a sample for OCaml generated code for `tensor-generated.go`
package tensor

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

func (ts Tensor) MustTo(device gt.Device) (retVal Tensor) {
	var err error
	retVal, err = ts.To(device)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
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

func (ts Tensor) AddG(other Tensor) (err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgAdd(ptr, ts.ctensor, other.ctensor)

	if err = TorchErr(); err != nil {
		return err
	}

	ts = Tensor{ctensor: *ptr}

	return nil
}

func (ts Tensor) MustAddG(other Tensor) {
	err := ts.AddG(other)
	if err != nil {
		log.Fatal(err)
	}
}

// Totype casts type of tensor to a new tensor with specified DType
func (ts Tensor) Totype(dtype gt.DType) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	cint, err := gt.DType2CInt(dtype)
	if err != nil {
		return retVal, err
	}

	lib.AtgTotype(ptr, ts.ctensor, cint)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

// Totype casts type of tensor to a new tensor with specified DType. It will
// panic if error
func (ts Tensor) MustTotype(dtype gt.DType) (retVal Tensor) {
	retVal, err := ts.Totype(dtype)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// Unsqueeze unsqueezes tensor to specified dimension.
func (ts Tensor) Unsqueeze(dim int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgUnsqueeze(ptr, ts.ctensor, dim)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

// Select creates a new tensor from current tensor given dim and index.
func (ts Tensor) Select(dim int64, index int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgSelect(ptr, ts.ctensor, dim, index)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

// Narrow creates a new tensor from current tensor given dim and start index
// and length.
func (ts Tensor) Narrow(dim int64, start int64, length int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgNarrow(ptr, ts.ctensor, dim, start, length)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

// IndexSelect creates a new tensor from current tensor given dim and index
// tensor.
func (ts Tensor) IndexSelect(dim int64, index Tensor) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgIndexSelect(ptr, ts.ctensor, dim, index.ctensor)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) Zeros(size []int64, optionsKind, optionsDevice int32) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgZeros(ptr, size, len(size), optionsKind, optionsDevice)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) Ones(size []int64, optionsKind, optionsDevice int32) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgOnes(ptr, size, len(size), optionsKind, optionsDevice)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

// NOTE: `_` denotes "in-place".
func (ts Tensor) Uniform_(from float64, to float64) {
	var err error
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgUniform_(ptr, ts.ctensor, from, to)
	if err = TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) ZerosLike() (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgZerosLike(ptr, ts.ctensor)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) Fill_(value Scalar) {
	var err error
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgFill_(ptr, ts.ctensor, value.cscalar)

	if err = TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) RandnLike() (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgRandnLike(ptr, ts.ctensor)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) Permute(dims []int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgPermute(ptr, ts.ctensor, dims, len(dims))

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) Squeeze1(dim int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgSqueeze1(ptr, ts.ctensor, dim)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustSqueeze1(dim int64) (retVal Tensor) {
	var err error
	retVal, err = ts.Squeeze1(dim)
	if err != nil {
		log.Fatal(err)
	}
	return retVal
}

func (ts Tensor) Squeeze_() {
	var err error
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgSqueeze_(ptr, ts.ctensor)

	if err = TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func Stack(tensors []Tensor, dim int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	var ctensors []lib.Ctensor
	for _, t := range tensors {
		ctensors = append(ctensors, t.ctensor)
	}

	lib.AtgStack(ptr, ctensors, len(tensors), dim)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}
