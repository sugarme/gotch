package tensor

// #include <stdlib.h>
import "C"

import (
	"fmt"
	"unsafe"

	lib "github.com/sugarme/gotch/libtch"
)

// ptrToString check C pointer for null. If not null, get value
// the pointer points to and frees up C memory. It is used for
// getting error message C pointer points to and clean up C memory.
//
// NOTE: C does not have exception design. C++ throws exception
// to stderr. This code to check stderr for any err message,
// if it exists, takes it and frees up C memory.
func ptrToString(cptr *C.char) string {
	var str string = ""

	if cptr != nil {
		str = C.GoString(cptr)
		C.free(unsafe.Pointer(cptr))
	}

	return str
}

// TorchErr checks and retrieves last error message from
// C `thread_local` if existing and frees up C memory the C pointer
// points to.
//
// NOTE: Go language atm does not have generic function something
// similar to `macro` in Rust language, does it? So we have to
// wrap this function to any Libtorch C function call to check error
// instead of doing the other way around.
// See Go2 proposal: https://github.com/golang/go/issues/32620
func TorchErr() error {
	cptr := (*C.char)(lib.GetAndResetLastErr())
	errStr := ptrToString(cptr)
	if errStr != "" {
		return fmt.Errorf("Libtorch API Error: %v\n", errStr)
	}

	return nil
}
