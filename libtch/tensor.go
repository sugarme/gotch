package libtch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import (
	"unsafe"
)

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

	return &C_tensor{private: unsafe.Pointer(t)}
}

func AtPrint(t *C_tensor) {
	c_tensor := (C.tensor)((*t).private)
	C.at_print(c_tensor)
}

func AtDataPtr(t *C_tensor) unsafe.Pointer {
	c_tensor := (C.tensor)((*t).private)
	return C.at_data_ptr(c_tensor)
}

func AtDim(t *C_tensor) uint64 {
	c_tensor := (C.tensor)((*t).private)
	c_result := C.at_dim(c_tensor)
	return *(*uint64)(unsafe.Pointer(&c_result))
}

func AtShape(t *C_tensor, ptr unsafe.Pointer) {
	cTensor := (C.tensor)((*t).private)
	c_ptr := (*C.long)(ptr)
	C.at_shape(cTensor, c_ptr)
}

func AtScalarType(t *C_tensor) int32 {
	c_tensor := (C.tensor)((*t).private)
	c_result := C.at_scalar_type(c_tensor)
	return *(*int32)(unsafe.Pointer(&c_result))
}
