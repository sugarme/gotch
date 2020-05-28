package libtch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import (
	"fmt"
	"reflect"
	"unsafe"
)

// type c_void unsafe.Pointer
// type size_t uint
// type c_int int32

type C_tensor struct {
	private unsafe.Pointer
}

func AtNewTensor() *C_tensor {
	t := C.at_new_tensor()
	return &C_tensor{private: unsafe.Pointer(t)}
}

func AtTensorOfData(vs unsafe.Pointer, dims []int64, ndims uint, elt_size_in_bytes uint, kind int) *C_tensor {

	// just get pointer of the first element of shape
	c_dims := (*C.int64_t)(unsafe.Pointer(&dims[0]))
	c_ndims := *(*C.size_t)(unsafe.Pointer(&ndims))
	c_elt_size_in_bytes := *(*C.size_t)(unsafe.Pointer(&elt_size_in_bytes))
	c_kind := *(*C.int)(unsafe.Pointer(&kind))

	// t is of type `unsafe.Pointer` in Go and `*void` in C
	t := C.at_tensor_of_data(vs, c_dims, c_ndims, c_elt_size_in_bytes, c_kind)
	fmt.Printf("t type: %v\n", reflect.TypeOf(t).Kind())
	fmt.Printf("1. C.tensor AtTensorOfData returned from C call: %v\n", t)
	// Keep C pointer value tin Go struct
	cTensorPtrVal := unsafe.Pointer(t)
	fmt.Printf("2. cTensorPtrVal: %v\n", cTensorPtrVal)

	var retVal *C_tensor
	retVal = &C_tensor{private: cTensorPtrVal}
	fmt.Printf("3. C_tensor.private: %v\n", (*retVal).private)

	// test call C.at_print to print out tensor
	// C.at_print(*(*C.tensor)(unsafe.Pointer(&t)))
	AtPrint(retVal)

	return retVal
}

func AtPrint(t *C_tensor) {
	fmt.Printf("4. C_tensor.private AtPrint: %v\n", (*t).private)
	cTensor := (C.tensor)((*t).private)
	fmt.Printf("5. C.tensor AtPrint: %v\n", cTensor)

	C.at_print(cTensor)
}

func AtDataPtr(t *C_tensor) unsafe.Pointer {
	cTensor := (C.tensor)((*t).private)
	return C.at_data_ptr(cTensor)
}
