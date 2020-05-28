package wrapper

//#include <stdlib.h>
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"reflect"
	"unsafe"

	lib "github.com/sugarme/gotch/libtch"
)

type Tensor struct {
	ctensor *t.C_tensor
}

// NewTensor creates a new tensor
func NewTensor() Tensor {
	ctensor := lib.AtNewTensor()
	return Tensor{ctensor}
}

// FOfSlice creates tensor from a slice data
func(ts Tensor) FOfSlice(data []inteface{}) (retVal Tensor, err error) {

	data := []int{0, 0, 0, 0}
	shape := []int64{int64(len(data))}
	nflattened := numElements(shape)
	dtype := 3          // Kind.Int
	eltSizeInBytes := 4 // Element Size in Byte for Int dtype

	nbytes := eltSizeInBytes * int(uintptr(nflattened))

	// NOTE: dataPrt is type of `*void` in C or type of `unsafe.Pointer` in Go
	// data should be allocated to memory BY `C` side
	dataPtr := C.malloc(C.size_t(nbytes))

	// Recall: 1 << 30 = 1 * 2 * 30
	// Ref. See more at https://stackoverflow.com/questions/48756732
	dataSlice := (*[1 << 30]byte)(dataPtr)[:nbytes:nbytes]

	buf := bytes.NewBuffer(dataSlice[:0:nbytes])

	EncodeTensor(buf, reflect.ValueOf(data), shape)

	c_tensor := lib.AtTensorOfData(dataPtr, shape, uint(len(shape)), uint(eltSizeInBytes), int(dtype))

	retVal = Tensor{c_tensor}

	// Read back created tensor values by C libtorch
	readDataPtr := lib.AtDataPtr(retVal.c_tensor)
	readDataSlice := (*[1 << 30]byte)(readDataPtr)[:nbytes:nbytes]
	// typ := typeOf(dtype, shape)
	typ := reflect.TypeOf(int32(0)) // C. type `int` ~ Go type `int32`
	val := reflect.New(typ)
	if err := DecodeTensor(bytes.NewReader(readDataSlice), shape, typ, val); err != nil {
		panic(fmt.Sprintf("unable to decode Tensor of type %v and shape %v - %v", dtype, shape, err))
	}

	tensorData := reflect.Indirect(val).Interface()

	fmt.Println("%v", tensorData)

	return retVal, nil
}
