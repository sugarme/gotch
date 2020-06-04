package wrapper

// #include <stdlib.h>
//#include "stdbool.h"
// #include "../libtch/torch_api.h"
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
} // Size1 returns the tensor size for 1D tensors.
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

// OfSlice creates tensor from a slice data
func OfSlice(data interface{}) (retVal *Tensor, err error) {

	typ, dataLen, err := DataCheck(data)
	if err != nil {
		return nil, err
	}

	dtype, err := gotch.ToDType(typ)
	if err != nil {
		return nil, err
	}

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

func (ts Tensor) Eq1(other Tensor) {

	// var ptr unsafe.Pointer
	// NOTE:
	// This will cause panic: runtime error: cgo argument has Go pointer to Go pointer
	// ptr = NewTensor()
	// lib.Atg_eq1(unsafe.Pointer(&ptr), ts.ctensor, other.ctensor)

	// C pointer to [1]uintptr (Go pointer)
	// ctensorsPtr := C.malloc(C.size_t(1) * C.size_t(unsafe.Sizeof(uintptr(0))))

	// TODO: create C pointer to a slice of tensors [1]C.tensor using C.malloc
	// Slice with 1 element type C.tensor
	// nbytes := C.size_t(1) * C.size_t(unsafe.Sizeof(C.tensor))
	// ctensorsPtr := C.malloc(nbytes)
	// ctensorsPtr := C.malloc(C.size_t(1) * C.size_t(unsafe.Sizeof(C.tensor)))

	// C null pointer C.tensor * = null
	ctensorPtr := lib.NewTensor()
	fmt.Printf("Out tensor BEFORE: %v\n", &ctensorPtr)
	fmt.Printf("Out tensor address: %v\n", *(*int)(unsafe.Pointer(&ctensorPtr)))

	ctensorAddr := *(*int64)(unsafe.Pointer(&ctensorPtr))
	var data []int64
	data = append(data, ctensorAddr)

	// lib.AtPrint((*lib.C_tensor)(unsafe.Pointer(ctensorPtr)))

	// nullPtr := (*C.tensor)(unsafe.Pointer(uintptr(0)))
	// fmt.Printf("Null pointer: %v\n", &nullPtr)
	//
	// data := []*C.tensor{nullPtr}
	// fmt.Printf("data: %v\n", data)
	// // Calculate number of bytes for a slice of one element of C null pointer
	// nbytes := 1 * unsafe.Sizeof(uintptr(0))
	// fmt.Printf("Nbytes: %v\n", nbytes)
	//
	// cptr := C.malloc(C.size_t(nbytes))
	// ctensorsPtr := (*[1 << 30]byte)(cptr)[:nbytes:nbytes]
	// buf := bytes.NewBuffer(ctensorsPtr[:0:nbytes])
	// // ctensorsPtr := (*[1 << 30]C.tensor)(unsafe.Pointer(uintptr(0)))[:nbytes:nbytes]
	// fmt.Printf("ctensorsPtr 1: %v\n", &ctensorsPtr[0])
	// fmt.Printf("Type of ctensorsPtr: %v\n", reflect.TypeOf(ctensorsPtr))
	// // buff := bytes.NewBuffer(dataSlice[:0:nbytes])
	// // Write to memory
	// err := binary.Write(buf, nativeEndian, data)
	// if err != nil {
	// log.Fatal(err)
	// }

	// lib.Atg_eq1(unsafe.Pointer(cptr), ts.ctensor, other.ctensor)
	lib.Atg_eq1(unsafe.Pointer(&ctensorPtr), ts.ctensor, other.ctensor)

	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Out tensor AFTER: %v\n", &ctensorPtr)

	lib.AtPrint((*lib.C_tensor)(unsafe.Pointer(&ctensorPtr)))

}
