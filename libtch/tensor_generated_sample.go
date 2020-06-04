// NOTE: this is a sample for OCaml generated code for `tensor_generated.go`
package libtch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import (
	"unsafe"
)

// void atg_eq1(tensor *, tensor self, tensor other);
func Atg_eq1(ptr unsafe.Pointer, self *C_tensor, other *C_tensor) {
	// // func Atg_eq1(ptr unsafe.Pointer, self *C_tensor, other *C_tensor) {
	//
	// // t := C.malloc(C.size_t(1) * C.size_t(unsafe.Sizeof(uintptr(C.tensor{}))))
	// var ctensor C.tensor
	// t := C.malloc(C.size_t(3) * C.size_t(unsafe.Sizeof(uintptr(ctensor))))
	// // t := C.malloc(1000)
	// // t := C.at_new_tensor()
	c_self := (C.tensor)((*self).private)
	c_other := (C.tensor)((*other).private)

	C.atg_eq1((*C.tensor)(ptr), c_self, c_other)
	// cptr := (*C.tensor)(ptr)
	// C.atg_eq1(cptr, c_self, c_other)
}
