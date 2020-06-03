package wrapper

// #include <stdlib.h>
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
	ctensor *lib.C_tensor
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
func (ts Tensor) Size() []int64 {
	dim := lib.AtDim(ts.ctensor)
	sz := make([]int64, dim)
	szPtr, err := DataAsPtr(sz)
	if err != nil {
		log.Fatal(err)
	}

	// TODO: should we free C memory here or at `DataAsPtr` func
	defer C.free(unsafe.Pointer(szPtr))

	lib.AtShape(ts.ctensor, szPtr)

	retVal := decodeSize(szPtr, dim)
	return retVal
}

// Size1 returns the tensor size for 1D tensors.
func (ts Tensor) Size1() (retVal int64, err error) {
	shape := ts.Size()
	if len(shape) != 1 {
		err = fmt.Errorf("Expected one dim, got %v\n", len(shape))
		return 0, err
	}

	return shape[0], nil
}

// Size2 returns the tensor size for 2D tensors.
func (ts Tensor) Size2() (retVal []int64, err error) {
	shape := ts.Size()
	if len(shape) != 2 {
		err = fmt.Errorf("Expected two dims, got %v\n", len(shape))
		return nil, err
	}

	return shape, nil
}

// Size3 returns the tensor size for 3D tensors.
func (ts Tensor) Size3() (retVal []int64, err error) {
	shape := ts.Size()
	if len(shape) != 3 {
		err = fmt.Errorf("Expected three dims, got %v\n", len(shape))
		return nil, err
	}

	return shape, nil
}

// Size4 returns the tensor size for 4D tensors.
func (ts Tensor) Size4() (retVal []int64, err error) {
	shape := ts.Size()
	if len(shape) != 4 {
		err = fmt.Errorf("Expected four dims, got %v\n", len(shape))
		return nil, err
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

// Size1 returns the tensor size for single dimension tensor
// func (ts Tensor) Size1() {
//
// shape := ts.Size()
//
// fmt.Printf("shape: %v\n", shape)
//
// }

// FOfSlice creates tensor from a slice data
func (ts Tensor) FOfSlice(data interface{}, dtype gotch.DType) (retVal *Tensor, err error) {

	if ok, msg := gotch.TypeCheck(data, dtype); !ok {
		err = fmt.Errorf("data type and DType are mismatched: %v\n", msg)
		return nil, err
	}

	dataLen := reflect.ValueOf(data).Len()
	shape := []int64{int64(dataLen)}
	elementNum := ElementCount(shape)

	eltSizeInBytes, err := gotch.DTypeSize(dtype)
	if err != nil {
		return nil, err
	}

	nbytes := int(eltSizeInBytes) * int(elementNum)

	dataPtr, buff := CMalloc(nbytes)

	if err = EncodeTensor(buff, reflect.ValueOf(data), shape); err != nil {
		return nil, err
	}

	cint, err := gotch.DType2CInt(dtype)
	if err != nil {
		return nil, err
	}

	ctensor := lib.AtTensorOfData(dataPtr, shape, uint(len(shape)), uint(eltSizeInBytes), int(cint))

	retVal = &Tensor{ctensor}

	return retVal, nil
}

// Print prints tensor values to console.
//
// NOTE: it is printed from C and will print ALL elements of tensor
// with no truncation at all.
func (ts Tensor) Print() {
	lib.AtPrint(ts.ctensor)
}

// NewTensorFromData creates tensor from given data and shape
func NewTensorFromData(data interface{}, shape []int64) (retVal *Tensor, err error) {
	// 1. Check whether data and shape match
	elementNum, err := DataDim(data)
	if err != nil {
		return nil, err
	}

	nflattend := FlattenDim(shape)

	if elementNum != nflattend {
		err = fmt.Errorf("Number of data elements (%v) and flatten shape (%v) dimension mismatched.\n", elementNum, nflattend)
		return nil, err
	}

	// 2. Write raw data to C memory and get C pointer
	dataPtr, err := DataAsPtr(data)
	if err != nil {
		return nil, err
	}

	// 3. Create tensor with pointer and shape
	dtype, err := gotch.DTypeFromData(data)
	if err != nil {
		return nil, err
	}

	eltSizeInBytes, err := gotch.DTypeSize(dtype)
	if err != nil {
		return nil, err
	}

	cint, err := gotch.DType2CInt(dtype)
	if err != nil {
		return nil, err
	}

	ctensor := lib.AtTensorOfData(dataPtr, shape, uint(len(shape)), uint(eltSizeInBytes), int(cint))

	retVal = &Tensor{ctensor}

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
