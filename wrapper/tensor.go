package wrapper

// #include <stdlib.h>
import "C"

import (
	"fmt"
	"reflect"

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
