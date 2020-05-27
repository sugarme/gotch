package libtch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import (
	"unsafe"
)

type c_void unsafe.Pointer
type size_t uint
type c_int int32

type C_tensor struct {
	private uint8
}

func NewTensor() *C_tensor {
	t := C.at_new_tensor()
	return &C_tensor{private: *(*uint8)(unsafe.Pointer(&t))}
}

func AtTensorOfData(vs unsafe.Pointer, dims int64, ndims uint, elt_size_in_bytes uint, kind int32) *C_tensor {
	c_dims := (*C.long)(unsafe.Pointer(&dims))
	c_ndims := *(*C.ulong)(unsafe.Pointer(&ndims))
	c_elt_size_in_bytes := *(*C.ulong)(unsafe.Pointer(&elt_size_in_bytes))
	c_kind := *(*C.int)(unsafe.Pointer(&kind))

	t := C.at_tensor_of_data(vs, c_dims, c_ndims, c_elt_size_in_bytes, c_kind)
	return &C_tensor{private: *(*uint8)(unsafe.Pointer(&t))}
}
