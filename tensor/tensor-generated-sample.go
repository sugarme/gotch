// NOTE: this is a sample for OCaml generated code for `tensor-generated.go`
package tensor

// #include "stdlib.h"
import "C"

import (
	"log"
	"unsafe"

	"github.com/sugarme/gotch"
	lib "github.com/sugarme/gotch/libtch"
)

func (ts Tensor) To(device gotch.Device) (retVal Tensor, err error) {

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

func (ts Tensor) MustTo(device gotch.Device) (retVal Tensor) {
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

func (ts Tensor) Zero_() (err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgZero_(ptr, ts.ctensor)

	if err = TorchErr(); err != nil {
		return err
	}

	return nil
}

func (ts Tensor) MustZero_() {
	err := ts.Zero_()
	if err != nil {
		log.Fatal(err)
	}
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

func (ts Tensor) Mul1(other Scalar) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgMul1(ptr, ts.ctensor, other.cscalar)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustMul1(other Scalar) (retVal Tensor) {
	retVal, err := ts.Mul1(other)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Mul_(other Tensor) (err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgMul_(ptr, ts.ctensor, other.ctensor)

	if err = TorchErr(); err != nil {
		return err
	}

	return nil
}

func (ts Tensor) MustMul_(other Tensor) {
	err := ts.Mul_(other)
	if err != nil {
		log.Fatal(err)
	}
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

func (ts Tensor) Add_(other Tensor) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgAdd_(ptr, ts.ctensor, other.ctensor)

	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) Add1(other Scalar) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	lib.AtgAdd1(ptr, ts.ctensor, other.cscalar)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustAdd1(other Scalar) (retVal Tensor) {
	retVal, err := ts.Add1(other)

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
func (ts Tensor) Totype(dtype gotch.DType) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))
	cint, err := gotch.DType2CInt(dtype)
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
func (ts Tensor) MustTotype(dtype gotch.DType) (retVal Tensor) {
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
func (ts Tensor) MustIndexSelect(dim int64, index Tensor) (retVal Tensor) {
	retVal, err := ts.IndexSelect(dim, index)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func Zeros(size []int64, optionsKind, optionsDevice int32) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgZeros(ptr, size, len(size), optionsKind, optionsDevice)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func MustZeros(size []int64, optionsKind, optionsDevice int32) (retVal Tensor) {
	retVal, err := Zeros(size, optionsKind, optionsDevice)
	if err != nil {
		log.Fatal(err)
	}
	return retVal
}

func Ones(size []int64, optionsKind, optionsDevice int32) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgOnes(ptr, size, len(size), optionsKind, optionsDevice)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func MustOnes(size []int64, optionsKind, optionsDevice int32) (retVal Tensor) {
	retVal, err := Ones(size, optionsKind, optionsDevice)
	if err != nil {
		log.Fatal(err)
	}
	return retVal
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

func (ts Tensor) Mm(mat2 Tensor) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgMm(ptr, ts.ctensor, mat2.ctensor)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustMm(mat2 Tensor) (retVal Tensor) {
	retVal, err := ts.Mm(mat2)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) LogSoftmax(dim int64, dtype int32) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgLogSoftmax(ptr, ts.ctensor, dim, dtype)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustLogSoftmax(dim int64, dtype int32) (retVal Tensor) {
	retVal, err := ts.LogSoftmax(dim, dtype)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) NllLoss(target Tensor) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	// NOTE: uncomment this causes panic
	// defer C.free(unsafe.Pointer(ptr))

	weight := NewTensor()

	reduction := int64(1) // Mean of loss
	ignoreIndex := int64(-100)
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgNllLoss(ptr, ts.ctensor, target.ctensor, weight.ctensor, reduction, ignoreIndex)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustNllLoss(target Tensor) (retVal Tensor) {
	retVal, err := ts.NllLoss(target)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Argmax(dim int64, keepDim bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	var ckeepDim int = 0
	if keepDim {
		ckeepDim = 1
	}

	lib.AtgArgmax(ptr, ts.ctensor, dim, ckeepDim)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustArgmax(dim int64, keepDim bool) (retVal Tensor) {
	retVal, err := ts.Argmax(dim, keepDim)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Mean(dtype int32) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgMean(ptr, ts.ctensor, dtype)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustMean(dtype int32) (retVal Tensor) {
	retVal, err := ts.Mean(dtype)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) View(sizeData []int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgView(ptr, ts.ctensor, sizeData, len(sizeData))
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustView(sizeData []int64) (retVal Tensor) {
	retVal, err := ts.View(sizeData)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Div1(other Scalar) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgDiv1(ptr, ts.ctensor, other.cscalar)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustDiv1(other Scalar) (retVal Tensor) {
	retVal, err := ts.Div1(other)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func Randperm(n int64, optionKind gotch.DType, optionDevice gotch.Device) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgRandperm(ptr, n, optionKind.CInt(), optionDevice.CInt())
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func MustRandperm(n int64, optionKind gotch.DType, optionDevice gotch.Device) (retVal Tensor) {
	retVal, err := Randperm(n, optionKind, optionDevice)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Clamp_(min Scalar, max Scalar) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgClamp_(ptr, ts.ctensor, min.cscalar, max.cscalar)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) Relu_() {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgRelu_(ptr, ts.ctensor)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) Relu() (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgRelu(ptr, ts.ctensor)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustRelu() (retVal Tensor) {
	retVal, err := ts.Relu()
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) T() (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgT(ptr, ts.ctensor)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustT() (retVal Tensor) {
	retVal, err := ts.T()
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) T_() {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgT_(ptr, ts.ctensor)
	err := TorchErr()
	if err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) MseLoss(target Tensor, reduction int) (retVal Tensor, err error) {

	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgMseLoss(ptr, ts.ctensor, target.ctensor, reduction)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustMseLoss(target Tensor, reduction int) (retVal Tensor) {
	retVal, err := ts.MseLoss(target, reduction)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Exp() (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgExp(ptr, ts.ctensor)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustExp() (retVal Tensor) {
	retVal, err := ts.Exp()

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Exp_() {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgExp(ptr, ts.ctensor)
	err := TorchErr()
	if err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) Pow(exponent Scalar) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgPow(ptr, ts.ctensor, exponent.cscalar)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustPow(exponent Scalar) (retVal Tensor) {
	retVal, err := ts.Pow(exponent)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Sum(dtype int32) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgSum(ptr, ts.ctensor, dtype)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustSum(dtype int32) (retVal Tensor) {
	retVal, err := ts.Sum(dtype)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Sub(other Tensor) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgSub(ptr, ts.ctensor, other.ctensor)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustSub(other Tensor) (retVal Tensor) {
	retVal, err := ts.Sub(other)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Sub1(other Scalar) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgSub1(ptr, ts.ctensor, other.cscalar)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustSub1(other Scalar) (retVal Tensor) {
	retVal, err := ts.Sub1(other)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Sub_(other Tensor) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgSub_(ptr, ts.ctensor, other.ctensor)
	err := TorchErr()
	if err != nil {
		log.Fatal(err)
	}
}
