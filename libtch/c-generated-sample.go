// NOTE: this is a sample for OCaml generated code for `c-generated.go`
package libtch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import (
	"unsafe"
)

// void atg_eq1(tensor *, tensor self, tensor other);
func AtgEq1(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_eq1(ptr, self, other)
}

// void atg_matmul(tensor *, tensor self, tensor other);
func AtgMatmul(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_matmul(ptr, self, other)
}

// void atg_to(tensor *, tensor self, int device);
func AtgTo(ptr *Ctensor, self Ctensor, device int) {
	cdevice := *(*C.int)(unsafe.Pointer(&device))
	C.atg_to(ptr, self, cdevice)
}

// int at_device(tensor);
func AtDevice(ts Ctensor) int {
	cint := C.at_device(ts)
	return *(*int)(unsafe.Pointer(&cint))
}
