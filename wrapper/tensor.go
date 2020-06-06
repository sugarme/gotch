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
