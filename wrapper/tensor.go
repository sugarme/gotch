package wrapper

// #include "stdlib.h"
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"reflect"
	"unsafe"

	gotch "github.com/sugarme/gotch"
	lib "github.com/sugarme/gotch/libtch"
)

type Tensor struct {
	ctensor lib.Ctensor
}

// NewTensor creates a new tensor
func NewTensor() Tensor {
	ctensor := lib.AtNewTensor()
	return Tensor{ctensor}
}

func (ts Tensor) Dim() uint64 {
	retVal := lib.AtDim(ts.ctensor)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
	return retVal
}

// Size return shape of the tensor
//
// NOTE: C++ libtorch calls at_shape() -> t.sizes()
// And returns a slice of sizes or shape using given pointer
// to that slice.
func (ts Tensor) Size() (retVal []int64, err error) {
	dim := lib.AtDim(ts.ctensor)
	sz := make([]int64, dim)
	szPtr, err := DataAsPtr(sz)
	if err != nil {
		return retVal, err
	}
	defer C.free(unsafe.Pointer(szPtr))

	lib.AtShape(ts.ctensor, szPtr)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = decodeSize(szPtr, dim)
	return retVal, nil
}

func (ts Tensor) MustSize() (retVal []int64) {
	retVal, err := ts.Size()
	if err != nil {
		log.Fatal(err)
	}
	return retVal
}

// Size1 returns the tensor size for 1D tensors.
func (ts Tensor) Size1() (retVal int64, err error) {
	shape, err := ts.Size()
	if err != nil {
		return retVal, err
	}

	if len(shape) != 1 {
		err = fmt.Errorf("Expected one dim, got %v\n", len(shape))
		return retVal, err
	}

	return shape[0], nil
}

// Size2 returns the tensor size for 2D tensors.
func (ts Tensor) Size2() (retVal []int64, err error) {
	shape, err := ts.Size()
	if err != nil {
		return retVal, err
	}

	if len(shape) != 2 {
		err = fmt.Errorf("Expected two dims, got %v\n", len(shape))
		return retVal, err
	}

	return shape, nil
}

// Size3 returns the tensor size for 3D tensors.
func (ts Tensor) Size3() (retVal []int64, err error) {
	shape, err := ts.Size()
	if err != nil {
		return retVal, err
	}

	if len(shape) != 3 {
		err = fmt.Errorf("Expected three dims, got %v\n", len(shape))
		return retVal, err
	}

	return shape, nil
}

// Size4 returns the tensor size for 4D tensors.
func (ts Tensor) Size4() (retVal []int64, err error) {
	shape, err := ts.Size()
	if err != nil {
		return retVal, err
	}

	if len(shape) != 4 {
		err = fmt.Errorf("Expected four dims, got %v\n", len(shape))
		return retVal, err
	}

	return shape, nil
}

func decodeSize(ptr unsafe.Pointer, nsize uint64) []int64 {
	// Decode sz
	// 1. Count number of elements in data
	elementNum := nsize
	// 2. Element size in bytes
	eltSizeInBytes, err := gotch.DTypeSize(gotch.Int64)
	if err != nil {
		log.Fatal(err)
	}
	nbytes := int(eltSizeInBytes) * int(elementNum)
	dataSlice := (*[1 << 30]byte)(ptr)[:nbytes:nbytes]
	r := bytes.NewReader(dataSlice)
	dataIn := make([]int64, nsize)
	if err := binary.Read(r, nativeEndian, dataIn); err != nil {
		log.Fatal(err)
	}

	return dataIn
}

// OfSlice creates tensor from a slice data
func OfSlice(data interface{}) (retVal Tensor, err error) {

	typ, dataLen, err := DataCheck(data)
	if err != nil {
		return retVal, err
	}

	dtype, err := gotch.ToDType(typ)
	if err != nil {
		return retVal, err
	}

	shape := []int64{int64(dataLen)}
	elementNum := ElementCount(shape)

	eltSizeInBytes, err := gotch.DTypeSize(dtype)
	if err != nil {
		return retVal, err
	}

	nbytes := int(eltSizeInBytes) * int(elementNum)

	dataPtr, buff := CMalloc(nbytes)
	defer C.free(unsafe.Pointer(dataPtr))

	if err = EncodeTensor(buff, reflect.ValueOf(data), shape); err != nil {
		return retVal, err
	}

	cint, err := gotch.DType2CInt(dtype)
	if err != nil {
		return retVal, err
	}

	ctensor := lib.AtTensorOfData(dataPtr, shape, uint(len(shape)), uint(eltSizeInBytes), int(cint))
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor}

	return retVal, nil
}

func TensorFrom(data interface{}) (retVal Tensor) {
	retVal, err := OfSlice(data)
	if err != nil {
		log.Fatal(err)
	}
	return retVal
}

// Print prints tensor values to console.
//
// NOTE: it is printed from C and will print ALL elements of tensor
// with no truncation at all.
func (ts Tensor) Print() {
	lib.AtPrint(ts.ctensor)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

// NewTensorFromData creates tensor from given data and shape
func NewTensorFromData(data interface{}, shape []int64) (retVal Tensor, err error) {
	// 1. Check whether data and shape match
	elementNum, err := DataDim(data)
	if err != nil {
		return retVal, err
	}

	nflattend := FlattenDim(shape)

	if elementNum != nflattend {
		err = fmt.Errorf("Number of data elements (%v) and flatten shape (%v) dimension mismatched.\n", elementNum, nflattend)
		return retVal, err
	}

	// 2. Write raw data to C memory and get C pointer
	dataPtr, err := DataAsPtr(data)
	defer C.free(unsafe.Pointer(dataPtr))
	if err != nil {
		return retVal, err
	}

	// 3. Create tensor with pointer and shape
	dtype, err := gotch.DTypeFromData(data)
	if err != nil {
		return retVal, err
	}

	eltSizeInBytes, err := gotch.DTypeSize(dtype)
	if err != nil {
		return retVal, err
	}

	cint, err := gotch.DType2CInt(dtype)
	if err != nil {
		return retVal, err
	}

	ctensor := lib.AtTensorOfData(dataPtr, shape, uint(len(shape)), uint(eltSizeInBytes), int(cint))
	// defer C.free(unsafe.Pointer(ctensor))
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor}

	return retVal, nil

}

func (ts Tensor) DType() gotch.DType {
	cint := lib.AtScalarType(ts.ctensor)

	dtype, err := gotch.CInt2DType(cint)
	if err != nil {
		log.Fatalf("Tensor DType error: %v\n", err)
	}

	return dtype
}

func (ts Tensor) Device() (retVal gotch.Device, err error) {
	cInt := lib.AtDevice(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	var device gotch.Device

	return device.OfCInt(int32(cInt)), nil
}

func (ts Tensor) Eq1(other Tensor) (retVal Tensor, err error) {

	// Get a C null pointer
	// https://stackoverflow.com/a/2022369
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	defer C.free(unsafe.Pointer(ptr))

	lib.AtgEq1(ptr, ts.ctensor, other.ctensor)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil

}

// DoubleValue returns a float value on tensors holding a single element.
// An error is returned otherwise.
// double at_double_value_at_indexes(tensor, int64_t *indexes, int indexes_len);
func (ts Tensor) Float64Value(idx []int64) (retVal float64, err error) {

	idxPtr, err := DataAsPtr(idx)
	if err != nil {
		return retVal, err
	}
	defer C.free(unsafe.Pointer(idxPtr))

	retVal = lib.AtDoubleValueAtIndexes(ts.ctensor, idxPtr, len(idx))
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, err
}

// Int64Value returns an int value on tensors holding a single element. An error is
// returned otherwise.
func (ts Tensor) Int64Value(idx []int64) (retVal int64, err error) {

	idxPtr, err := DataAsPtr(idx)
	if err != nil {
		return retVal, err
	}
	defer C.free(unsafe.Pointer(idxPtr))

	retVal = lib.AtInt64ValueAtIndexes(ts.ctensor, idxPtr, len(idx))
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, err
}

// RequiresGrad returns true if gradient are currently tracked for this tensor.
func (ts Tensor) RequiresGrad() (retVal bool, err error) {
	retVal = lib.AtRequiresGrad(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, nil
}

// DataPtr returns the address of the first element of this tensor.
func (ts Tensor) DataPtr() (retVal unsafe.Pointer, err error) {

	retVal = lib.AtDataPtr(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, nil
}

// Defined returns true is the tensor is defined.
func (ts Tensor) Defined() (retVal bool, err error) {
	retVal = lib.AtDefined(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, nil
}

func (ts Tensor) MustDefined() (retVal bool) {
	retVal, err := ts.Defined()
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// IsSparse returns true is the tensor is spare.
func (ts Tensor) IsSparse() (retVal bool, err error) {
	retVal = lib.AtIsSparse(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, nil
}

// ZeroGrad zeroes the gradient tensor attached to this tensor if defined.
func (ts Tensor) ZeroGrad() {
	grad := ts.MustGrad()
	if grad.MustDefined() {
		// TODO: can we chain them?
		// grad.MustDetach_().MustZero_()
		// https://www.calhoun.io/using-functional-options-instead-of-method-chaining-in-go/
		detach := grad.MustDetach_()
		detach.MustZero_()
	}
}

// Backward runs the backward pass, populating the gradient tensors for tensors
// which gradients are tracked.
//
// Gradients tracking can be turned on via `SetRequiresGrad`.
func (ts Tensor) Backward() (err error) {
	lib.AtBackward(ts.ctensor, 0, 0)
	if err = TorchErr(); err != nil {
		return err
	}

	return nil
}

func (ts Tensor) MustBackward() {
	if err := ts.Backward(); err != nil {
		log.Fatal(err)
	}
}

// RunBackward runs the backward ...
func RunBackward(tensors []Tensor, inputs []Tensor, keepGraphB bool, createGraphB bool) (retVal []Tensor, err error) {
	// NOTE: outputs is a slice of tensors with length = len(inputs)
	var outputsPtr []*lib.Ctensor
	// TODO: Are they allocated continouslly???
	for i := 0; i < len(inputs); i++ {
		outputPtr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
		// defer C.free(unsafe.Pointer(outputPtr))
		outputsPtr = append(outputsPtr, outputPtr)
	}

	// Get first element pointer
	ctensor := tensors[0].ctensor
	cinput := inputs[0].ctensor
	tensorsPtr := (*lib.Ctensor)(unsafe.Pointer(&ctensor))
	inputsPtr := (*lib.Ctensor)(unsafe.Pointer(&cinput))
	var keepGraph int = 0
	if keepGraphB {
		keepGraph = 1
	}
	var createGraph int = 0
	if createGraphB {
		createGraph = 1
	}

	lib.AtRunBackward(tensorsPtr, len(tensors), inputsPtr, len(inputs), outputsPtr[0], keepGraph, createGraph)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	for i := 0; i < len(inputs); i++ {
		outputPtr := outputsPtr[i]
		retVal = append(retVal, Tensor{ctensor: *outputPtr})
	}

	return retVal, nil
}

// CopyDataUint8 copies `numel` elements from `self` to `dst`.
//
// NOTE: `dst` located in Go memory. Should it be?
func (ts Tensor) CopyDataUint8(dst []uint8, numel uint) (err error) {

	// NOTE: we must make sure that `dst` has same len as `numel`. Otherwise,
	// there will be memory leak and or out of range error.
	if len(dst) < int(numel) {
		err = fmt.Errorf("CopyDataUint8 Error: length of destination slice data (%v) is smaller than \nnumber of elements to be copied (%v)", len(dst), numel)
		return err
	}

	vs := unsafe.Pointer(&dst[0])
	elt_size_in_bytes, err := gotch.DTypeSize(gotch.Uint8)
	if err != nil {
		return err
	}
	lib.AtCopyData(ts.ctensor, vs, numel, elt_size_in_bytes)
	if err = TorchErr(); err != nil {
		return err
	}

	return nil
}

func (ts Tensor) MustCopyDataUint8(dst []uint8, numel uint) {
	err := ts.CopyDataUint8(dst, numel)
	if err != nil {
		log.Fatal(err)
	}
}

// CopyData copies `numel` elements from `self` to `dst`.
// `dst` should be a slice of Go type equivalent to tensor type.
//
// NOTE: `dst` located in Go memory. Should it be?
func (ts Tensor) CopyData(dst interface{}, numel uint) (err error) {

	gotype, dlen, err := DataCheck(dst)
	if err != nil {
		return err
	}

	dtype, err := gotch.ToDType(gotype)
	if err != nil {
		return err
	}

	if dlen < int(numel) {
		err = fmt.Errorf("CopyDataUint8 Error: length of destination slice data (%v) is smaller than \nnumber of elements to be copied (%v)", dlen, numel)
		return err
	}

	if ts.DType() != dtype {
		err = fmt.Errorf("Type mismatched: `dst` type: %v, tensor DType: %v", dtype, ts.DType())
		return err
	}

	var vs unsafe.Pointer
	switch dtype {
	case gotch.Uint8:
		vs = unsafe.Pointer(&dst.([]uint8)[0])
	case gotch.Int8:
		vs = unsafe.Pointer(&dst.([]int8)[0])
	case gotch.Int16:
		vs = unsafe.Pointer(&dst.([]int16)[0])
	case gotch.Int:
		vs = unsafe.Pointer(&dst.([]int32)[0])
	case gotch.Int64:
		vs = unsafe.Pointer(&dst.([]int64)[0])
	case gotch.Float:
		vs = unsafe.Pointer(&dst.([]float32)[0])
	case gotch.Double:
		vs = unsafe.Pointer(&dst.([]float64)[0])
	case gotch.Bool:
		vs = unsafe.Pointer(&dst.([]bool)[0])
	default:
		err = fmt.Errorf("Unsupported type: `dst` type: %v, tensor DType: %v", dtype, ts.DType())
		return err
	}

	elt_size_in_bytes, err := gotch.DTypeSize(dtype)
	if err != nil {
		return err
	}
	lib.AtCopyData(ts.ctensor, vs, numel, elt_size_in_bytes)
	if err = TorchErr(); err != nil {
		return err
	}

	return nil
}

func (ts Tensor) MustCopyData(dst interface{}, numel uint) {
	err := ts.CopyData(dst, numel)
	if err != nil {
		log.Fatal(err)
	}
}

// Numel returns the total number of elements stored in a tensor.
func (ts Tensor) Numel() (retVal uint) {
	var shape []int64
	shape = ts.MustSize()
	return uint(FlattenDim(shape))
}
