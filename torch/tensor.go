package torch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import (
	// "fmt"
	// "reflect"
	"unsafe"
)

type C_tensor struct {
	_private uint8
}

func NewTensor() *C_tensor {
	ct := C.at_new_tensor()
	return &C_tensor{_private: *(*uint8)(unsafe.Pointer(&ct))}
}
