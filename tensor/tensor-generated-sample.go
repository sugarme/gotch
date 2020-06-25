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

func (ts Tensor) To(device gotch.Device, del bool) (retVal Tensor, err error) {

	// TODO: how to get pointer to CUDA memory???
	// C.cuMemAlloc((*C.ulonglong)(cudaPtr), 1) // 0 byte is invalid
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	if del {
		defer ts.MustDrop()
	}

	lib.AtgTo((*lib.Ctensor)(ptr), ts.ctensor, int(device.CInt()))

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustTo(device gotch.Device, del bool) (retVal Tensor) {
	var err error
	retVal, err = ts.To(device, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Matmul(other Tensor, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}
	lib.AtgMatmul(ptr, ts.ctensor, other.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustMatMul(other Tensor, del bool) (retVal Tensor) {
	retVal, err := ts.Matmul(other, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Grad() (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
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

func (ts Tensor) Detach_() {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	lib.AtgDetach_(ptr, ts.ctensor)

	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) Zero_() {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	lib.AtgZero_(ptr, ts.ctensor)

	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) SetRequiresGrad(rb bool) (retVal Tensor, err error) {
	var r int = 0
	if rb {
		r = 1
	}

	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

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

func (ts Tensor) Mul(other Tensor, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}
	lib.AtgMul(ptr, ts.ctensor, other.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustMul(other Tensor, del bool) (retVal Tensor) {
	retVal, err := ts.Mul(other, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Mul1(other Scalar, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}
	lib.AtgMul1(ptr, ts.ctensor, other.cscalar)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustMul1(other Scalar, del bool) (retVal Tensor) {
	retVal, err := ts.Mul1(other, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Mul_(other Tensor) (err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
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

func (ts Tensor) Add(other Tensor, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}
	lib.AtgAdd(ptr, ts.ctensor, other.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustAdd(other Tensor, del bool) (retVal Tensor) {
	retVal, err := ts.Add(other, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Add_(other Tensor) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	lib.AtgAdd_(ptr, ts.ctensor, other.ctensor)

	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) Add1(other Scalar, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}
	lib.AtgAdd1(ptr, ts.ctensor, other.cscalar)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil
}

func (ts Tensor) MustAdd1(other Scalar, del bool) (retVal Tensor) {
	retVal, err := ts.Add1(other, del)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) AddG(other Tensor) (err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
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
func (ts Tensor) Totype(dtype gotch.DType, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}
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
func (ts Tensor) MustTotype(dtype gotch.DType, del bool) (retVal Tensor) {
	retVal, err := ts.Totype(dtype, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// Unsqueeze unsqueezes tensor to specified dimension.
func (ts Tensor) Unsqueeze(dim int64, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}
	lib.AtgUnsqueeze(ptr, ts.ctensor, dim)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustUnsqueeze(dim int64, del bool) (retVal Tensor) {
	retVal, err := ts.Unsqueeze(dim, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// Select creates a new tensor from current tensor given dim and index.
func (ts Tensor) Select(dim int64, index int64, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}
	lib.AtgSelect(ptr, ts.ctensor, dim, index)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

// Narrow creates a new tensor from current tensor given dim and start index
// and length.
func (ts Tensor) Narrow(dim int64, start int64, length int64, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgNarrow(ptr, ts.ctensor, dim, start, length)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

// IndexSelect creates a new tensor from current tensor given dim and index
// tensor.
func (ts Tensor) IndexSelect(dim int64, index Tensor, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgIndexSelect(ptr, ts.ctensor, dim, index.ctensor)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}
func (ts Tensor) MustIndexSelect(dim int64, index Tensor, del bool) (retVal Tensor) {
	retVal, err := ts.IndexSelect(dim, index, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func Zeros(size []int64, optionsKind, optionsDevice int32) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

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

	lib.AtgUniform_(ptr, ts.ctensor, from, to)
	if err = TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) ZerosLike(del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}
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
	lib.AtgFill_(ptr, ts.ctensor, value.cscalar)

	if err = TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) RandnLike(del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}
	lib.AtgRandnLike(ptr, ts.ctensor)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) Permute(dims []int64, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}
	lib.AtgPermute(ptr, ts.ctensor, dims, len(dims))

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) Squeeze1(dim int64, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgSqueeze1(ptr, ts.ctensor, dim)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustSqueeze1(dim int64, del bool) (retVal Tensor) {
	var err error
	retVal, err = ts.Squeeze1(dim, del)
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

func (ts Tensor) Mm(mat2 Tensor, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgMm(ptr, ts.ctensor, mat2.ctensor)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustMm(mat2 Tensor, del bool) (retVal Tensor) {
	retVal, err := ts.Mm(mat2, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) LogSoftmax(dim int64, dtype int32, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgLogSoftmax(ptr, ts.ctensor, dim, dtype)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustLogSoftmax(dim int64, dtype int32, del bool) (retVal Tensor) {
	retVal, err := ts.LogSoftmax(dim, dtype, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) NllLoss(target Tensor, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

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

func (ts Tensor) MustNllLoss(target Tensor, del bool) (retVal Tensor) {
	retVal, err := ts.NllLoss(target, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Argmax(dim int64, keepDim bool, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

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

func (ts Tensor) MustArgmax(dim int64, keepDim bool, del bool) (retVal Tensor) {
	retVal, err := ts.Argmax(dim, keepDim, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Mean(dtype int32, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgMean(ptr, ts.ctensor, dtype)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustMean(dtype int32, del bool) (retVal Tensor) {
	retVal, err := ts.Mean(dtype, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) View(sizeData []int64, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgView(ptr, ts.ctensor, sizeData, len(sizeData))
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustView(sizeData []int64, del bool) (retVal Tensor) {
	retVal, err := ts.View(sizeData, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Div1(other Scalar, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgDiv1(ptr, ts.ctensor, other.cscalar)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustDiv1(other Scalar, del bool) (retVal Tensor) {
	retVal, err := ts.Div1(other, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func Randperm(n int64, optionKind gotch.DType, optionDevice gotch.Device) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

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

	lib.AtgClamp_(ptr, ts.ctensor, min.cscalar, max.cscalar)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) Relu_() {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	lib.AtgRelu_(ptr, ts.ctensor)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

func (ts Tensor) Relu(del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgRelu(ptr, ts.ctensor)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustRelu(del bool) (retVal Tensor) {
	retVal, err := ts.Relu(del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) T(del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgT(ptr, ts.ctensor)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustT(del bool) (retVal Tensor) {
	retVal, err := ts.T(del)
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

func (ts Tensor) MseLoss(target Tensor, reduction int, del bool) (retVal Tensor, err error) {

	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgMseLoss(ptr, ts.ctensor, target.ctensor, reduction)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustMseLoss(target Tensor, reduction int, del bool) (retVal Tensor) {
	retVal, err := ts.MseLoss(target, reduction, del)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Exp(del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgExp(ptr, ts.ctensor)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustExp(del bool) (retVal Tensor) {
	retVal, err := ts.Exp(del)

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

func (ts Tensor) Pow(exponent Scalar, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgPow(ptr, ts.ctensor, exponent.cscalar)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustPow(exponent Scalar, del bool) (retVal Tensor) {
	retVal, err := ts.Pow(exponent, del)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Sum(dtype int32, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgSum(ptr, ts.ctensor, dtype)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustSum(dtype int32, del bool) (retVal Tensor) {
	retVal, err := ts.Sum(dtype, del)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Sub(other Tensor, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgSub(ptr, ts.ctensor, other.ctensor)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustSub(other Tensor, del bool) (retVal Tensor) {
	retVal, err := ts.Sub(other, del)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Sub1(other Scalar, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgSub1(ptr, ts.ctensor, other.cscalar)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustSub1(other Scalar, del bool) (retVal Tensor) {
	retVal, err := ts.Sub1(other, del)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Sub_(other Tensor) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	lib.AtgSub_(ptr, ts.ctensor, other.ctensor)
	err := TorchErr()
	if err != nil {
		log.Fatal(err)
	}
}

func Conv1D(input, weight, bias Tensor, stride, padding, dilation []int64, groups int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	lib.AtgConv1d(ptr, input.ctensor, weight.ctensor, bias.ctensor, stride, len(stride), padding, len(padding), dilation, len(dilation), groups)

	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func MustConv1D(input, weight, bias Tensor, stride, padding, dilation []int64, groups int64) (retVal Tensor) {
	retVal, err := Conv1D(input, weight, bias, stride, padding, dilation, groups)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func Conv2D(input, weight, bias Tensor, stride, padding, dilation []int64, groups int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	lib.AtgConv2d(ptr, input.ctensor, weight.ctensor, bias.ctensor, stride, len(stride), padding, len(padding), dilation, len(dilation), groups)

	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func MustConv2D(input, weight, bias Tensor, stride, padding, dilation []int64, groups int64) (retVal Tensor) {
	retVal, err := Conv2D(input, weight, bias, stride, padding, dilation, groups)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func Conv3D(input, weight, bias Tensor, stride, padding, dilation []int64, groups int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	lib.AtgConv3d(ptr, input.ctensor, weight.ctensor, bias.ctensor, stride, len(stride), padding, len(padding), dilation, len(dilation), groups)

	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func MustConv3D(input, weight, bias Tensor, stride, padding, dilation []int64, groups int64) (retVal Tensor) {
	retVal, err := Conv3D(input, weight, bias, stride, padding, dilation, groups)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) MaxPool2D(kernel []int64, stride []int64, padding []int64, dilation []int64, ceil bool, del bool) (retVal Tensor, err error) {

	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	var ceilMode int
	switch ceil {
	case true:
		ceilMode = 1
	case false:
		ceilMode = 0
	}

	lib.AtgMaxPool2d(ptr, ts.ctensor, kernel, len(kernel), stride, len(stride), padding, len(padding), dilation, len(dilation), ceilMode)

	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustMaxPool2D(kernel []int64, stride []int64, padding []int64, dilation []int64, ceil bool, del bool) (retVal Tensor) {
	retVal, err := ts.MaxPool2D(kernel, stride, padding, dilation, ceil, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func Dropout(input Tensor, p float64, train bool) (retVal Tensor, err error) {

	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	var ctrain int
	switch train {
	case true:
		ctrain = 1
	case false:
		ctrain = 0
	}

	lib.AtgDropout(ptr, input.ctensor, p, ctrain)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil

}

func MustDropout(input Tensor, p float64, train bool) (retVal Tensor) {
	retVal, err := Dropout(input, p, train)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Dropout_(p float64, train bool) {

	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	var ctrain int
	switch train {
	case true:
		ctrain = 1
	case false:
		ctrain = 0
	}
	lib.AtgDropout_(ptr, ts.ctensor, p, ctrain)
	err := TorchErr()
	if err != nil {
		log.Fatal(err)
	}
}

func ConvTranspose1D(input, weight, bias Tensor, stride, padding, outputPadding, dilation []int64, groups int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	lib.AtgConvTranspose1d(ptr, input.ctensor, weight.ctensor, bias.ctensor, stride, len(stride), padding, len(padding), outputPadding, len(outputPadding), dilation, len(dilation), groups)

	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func MustConvTranspose1D(input, weight, bias Tensor, stride, padding, outputPadding, dilation []int64, groups int64) (retVal Tensor) {
	retVal, err := ConvTranspose1D(input, weight, bias, stride, padding, outputPadding, dilation, groups)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func ConvTranspose2D(input, weight, bias Tensor, stride, padding, outputPadding, dilation []int64, groups int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	lib.AtgConvTranspose2d(ptr, input.ctensor, weight.ctensor, bias.ctensor, stride, len(stride), padding, len(padding), outputPadding, len(outputPadding), dilation, len(dilation), groups)

	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func MustConvTranspose2D(input, weight, bias Tensor, stride, padding, outputPadding, dilation []int64, groups int64) (retVal Tensor) {
	retVal, err := ConvTranspose2D(input, weight, bias, stride, padding, outputPadding, dilation, groups)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func ConvTranspose3D(input, weight, bias Tensor, stride, padding, outputPadding, dilation []int64, groups int64) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	lib.AtgConvTranspose3d(ptr, input.ctensor, weight.ctensor, bias.ctensor, stride, len(stride), padding, len(padding), outputPadding, len(outputPadding), dilation, len(dilation), groups)

	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func MustConvTranspose3D(input, weight, bias Tensor, stride, padding, outputPadding, dilation []int64, groups int64) (retVal Tensor) {
	retVal, err := ConvTranspose3D(input, weight, bias, stride, padding, outputPadding, dilation, groups)

	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) LSTM(hxData []Tensor, paramsData []Tensor, hasBiases bool, numLayers int64, dropout float64, train bool, bidirectional bool, batchFirst bool) (output, h, c Tensor, err error) {

	// NOTE: `atg_lstm` will create 3 consecutive Ctensors in memory of C land. The first
	// Ctensor will have address given by `ctensorPtr1` here.
	// The next pointers can be calculated based on `ctensorPtr1`
	ctensorPtr1 := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	ctensorPtr2 := (*lib.Ctensor)(unsafe.Pointer(uintptr(unsafe.Pointer(ctensorPtr1)) + unsafe.Sizeof(ctensorPtr1)))
	ctensorPtr3 := (*lib.Ctensor)(unsafe.Pointer(uintptr(unsafe.Pointer(ctensorPtr2)) + unsafe.Sizeof(ctensorPtr1)))

	var chxData []lib.Ctensor
	for _, t := range hxData {
		chxData = append(chxData, t.ctensor)
	}

	var cparamsData []lib.Ctensor
	for _, t := range paramsData {
		cparamsData = append(cparamsData, t.ctensor)
	}

	chasBiases := 0
	if hasBiases {
		chasBiases = 1
	}
	ctrain := 0
	if train {
		ctrain = 1
	}
	cbidirectional := 0
	if bidirectional {
		cbidirectional = 1
	}
	cbatchFirst := 0
	if batchFirst {
		cbatchFirst = 1
	}

	lib.AtgLstm(ctensorPtr1, ts.ctensor, chxData, len(hxData), cparamsData, len(paramsData), chasBiases, numLayers, dropout, ctrain, cbidirectional, cbatchFirst)
	err = TorchErr()
	if err != nil {
		return output, h, c, err
	}

	return Tensor{ctensor: *ctensorPtr1}, Tensor{ctensor: *ctensorPtr2}, Tensor{ctensor: *ctensorPtr3}, nil

}

func (ts Tensor) MustLSTM(hxData []Tensor, paramsData []Tensor, hasBiases bool, numLayers int64, dropout float64, train bool, bidirectional bool, batchFirst bool) (output, h, c Tensor) {
	output, h, c, err := ts.LSTM(hxData, paramsData, hasBiases, numLayers, dropout, train, bidirectional, batchFirst)

	if err != nil {
		log.Fatal(err)
	}

	return output, h, c
}

func (ts Tensor) GRU(hx Tensor, paramsData []Tensor, hasBiases bool, numLayers int64, dropout float64, train bool, bidirectional bool, batchFirst bool) (output, h Tensor, err error) {

	// NOTE: `atg_gru` will create 2 consecutive Ctensors in memory of C land.
	// The first Ctensor will have address given by `ctensorPtr1` here.
	// The next pointer can be calculated based on `ctensorPtr1`
	ctensorPtr1 := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	ctensorPtr2 := (*lib.Ctensor)(unsafe.Pointer(uintptr(unsafe.Pointer(ctensorPtr1)) + unsafe.Sizeof(ctensorPtr1)))

	var cparamsData []lib.Ctensor
	for _, t := range paramsData {
		cparamsData = append(cparamsData, t.ctensor)
	}

	chasBiases := 0
	if hasBiases {
		chasBiases = 1
	}
	ctrain := 0
	if train {
		ctrain = 1
	}
	cbidirectional := 0
	if bidirectional {
		cbidirectional = 1
	}
	cbatchFirst := 0
	if batchFirst {
		cbatchFirst = 1
	}

	lib.AtgGru(ctensorPtr1, ts.ctensor, hx.ctensor, cparamsData, len(paramsData), chasBiases, numLayers, dropout, ctrain, cbidirectional, cbatchFirst)
	err = TorchErr()
	if err != nil {
		return output, h, err
	}

	return Tensor{ctensor: *ctensorPtr1}, Tensor{ctensor: *ctensorPtr2}, nil

}

func (ts Tensor) MustGRU(hx Tensor, paramsData []Tensor, hasBiases bool, numLayers int64, dropout float64, train bool, bidirectional bool, batchFirst bool) (output, h Tensor) {
	output, h, err := ts.GRU(hx, paramsData, hasBiases, numLayers, dropout, train, bidirectional, batchFirst)
	if err != nil {
		log.Fatal(err)
	}

	return output, h
}

func Randn(sizeData []int64, optionsKind gotch.DType, optionsDevice gotch.Device) (retVal Tensor, err error) {

	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))

	lib.AtgRandn(ptr, sizeData, len(sizeData), optionsKind.CInt(), optionsDevice.CInt())
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func MustRandn(sizeData []int64, optionsKind gotch.DType, optionsDevice gotch.Device) (retVal Tensor) {

	retVal, err := Randn(sizeData, optionsKind, optionsDevice)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}
