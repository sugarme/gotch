package libtch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import (
	"unsafe"
)

type c_void unsafe.Pointer
type size_t uint

type C_tensor struct {
	_private uint8
}

func NewTensor() *C_tensor {
	t := C.at_new_tensor()
	return &C_tensor{_private: *(*uint8)(unsafe.Pointer(&t))}
}

func AtTensorOfData(vs c_void, dims int64, ndims size_t, elt_size_in_bytes size_t, kind c_int) *C_tensor {
	t := C.at_tensor_of_data(vs, dims, ndims, elt_size_in_bytes, kind)
	return &C_tensor{_private: *(*uint8)(unsafe.Pointer(&t))}
}
