package wrapper

// #include <stdlib.h>
import "C"

import (
	"fmt"
	"unsafe"

	lib "github.com/sugarme/gotch/libtch"
)

// ptrToString returns nil on the null pointer. If not null,
// the pointer gets freed.
// NOTE: C does not have exception design. C++ throws exception
// to stderr. This code to check stderr for any err message,
// if it exists, takes it and frees up C pointer.
func ptrToString(cptr *C.char) string {
	var str string

	str = *(*string)(unsafe.Pointer(&cptr))
	fmt.Printf("Err Msg from C: %v\n", str)
	if str != "" {
		// Free up C memory
		C.free(unsafe.Pointer(cptr))
		return str
	} else {
		return ""
	}
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
		return fmt.Errorf("Libtorch API Err: %v\n", errStr)
	}

	return nil
}
