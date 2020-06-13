package libtch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import (
	"unsafe"
)

// scalar ats_int(int64_t);
func AtsInt(v int64) Cscalar {
	cv := *(*C.int64_t)(unsafe.Pointer(&v))
	return C.ats_int(cv)
}

// scalar ats_float(double);
func AtsFloat(v float64) Cscalar {
	cv := *(*C.double)(unsafe.Pointer(&v))
	return C.ats_float(cv)
}

// int64_t ats_to_int(scalar);
func AtsToInt(cscalar Cscalar) int64 {
	cint := C.ats_to_int(cscalar)
	return *(*int64)(unsafe.Pointer(&cint))
}

// double ats_to_float(scalar);
func AtsToFloat(cscalar Cscalar) float64 {
	cfloat := C.ats_to_float(cscalar)
	return *(*float64)(unsafe.Pointer(&cfloat))
}

// char *ats_to_string(scalar);
func AtsToString(cscalar Cscalar) string {
	charPtr := C.ats_to_string(cscalar)
	return C.GoString(charPtr)
}

// void ats_free(scalar);
func AtsFree(cscalar Cscalar) {
	C.ats_free(cscalar)
}
