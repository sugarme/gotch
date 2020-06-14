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

// void atg_grad(tensor *, tensor self);
func AtgGrad(ptr *Ctensor, self Ctensor) {
	C.atg_grad(ptr, self)
}

// void atg_detach_(tensor *, tensor self);
func AtgDetach_(ptr *Ctensor, self Ctensor) {
	C.atg_detach_(ptr, self)
}

// void atg_zero_(tensor *, tensor self);
func AtgZero_(ptr *Ctensor, self Ctensor) {
	C.atg_zero_(ptr, self)
}

// void atg_set_requires_grad(tensor *, tensor self, int r);
func AtgSetRequiresGrad(ptr *Ctensor, self Ctensor, r int) {
	cr := *(*C.int)(unsafe.Pointer(&r))
	C.atg_set_requires_grad(ptr, self, cr)
}

// void atg_mul(tensor *, tensor self, tensor other);
func AtgMul(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_mul(ptr, self, other)
}

// void atg_add(tensor *, tensor self, tensor other);
func AtgAdd(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_add(ptr, self, other)
}

// void atg_totype(tensor *, tensor self, int scalar_type);
func AtgTotype(ptr *Ctensor, self Ctensor, scalar_type int32) {
	cscalar_type := *(*C.int)(unsafe.Pointer(&scalar_type))
	C.atg_totype(ptr, self, cscalar_type)
}

// void atg_unsqueeze(tensor *, tensor self, int64_t dim);
func AtgUnsqueeze(ptr *Ctensor, self Ctensor, dim int64) {
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	C.atg_unsqueeze(ptr, self, cdim)
}

// void atg_select(tensor *, tensor self, int64_t dim, int64_t index);
func AtgSelect(ptr *Ctensor, self Ctensor, dim int64, index int64) {
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	cindex := *(*C.int64_t)(unsafe.Pointer(&index))
	C.atg_select(ptr, self, cdim, cindex)
}

// void atg_narrow(tensor *, tensor self, int64_t dim, int64_t start, int64_t length);
func AtgNarrow(ptr *Ctensor, self Ctensor, dim int64, start int64, length int64) {
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	cstart := *(*C.int64_t)(unsafe.Pointer(&start))
	clength := *(*C.int64_t)(unsafe.Pointer(&length))
	C.atg_narrow(ptr, self, cdim, cstart, clength)
}

// void atg_index_select(tensor *, tensor self, int64_t dim, tensor index);
func AtgIndexSelect(ptr *Ctensor, self Ctensor, dim int64, index Ctensor) {
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	C.atg_index_select(ptr, self, cdim, index)
}

// void atg_zeros(tensor *, int64_t *size_data, int size_len, int options_kind, int options_device);
func AtgZeros(ptr *Ctensor, sizeData []int64, sizeLen int, optionsKind, optionsDevice int32) {
	// just get pointer of the first element of the shape(sizeData)
	csizeDataPtr := (*C.int64_t)(unsafe.Pointer(&sizeData[0]))
	csizeLen := *(*C.int)(unsafe.Pointer(&sizeLen))
	coptionsKind := *(*C.int)(unsafe.Pointer(&optionsKind))
	coptionsDevice := *(*C.int)(unsafe.Pointer(&optionsDevice))

	C.atg_zeros(ptr, csizeDataPtr, csizeLen, coptionsKind, coptionsDevice)
}

// void atg_ones(tensor *, int64_t *size_data, int size_len, int options_kind, int options_device);
func AtgOnes(ptr *Ctensor, sizeData []int64, sizeLen int, optionsKind, optionsDevice int32) {
	// just get pointer of the first element of the shape(sizeData)
	csizeDataPtr := (*C.int64_t)(unsafe.Pointer(&sizeData[0]))
	csizeLen := *(*C.int)(unsafe.Pointer(&sizeLen))
	coptionsKind := *(*C.int)(unsafe.Pointer(&optionsKind))
	coptionsDevice := *(*C.int)(unsafe.Pointer(&optionsDevice))

	C.atg_ones(ptr, csizeDataPtr, csizeLen, coptionsKind, coptionsDevice)
}

// void atg_uniform_(tensor *, tensor self, double from, double to);
func AtgUniform_(ptr *Ctensor, self Ctensor, from float64, to float64) {
	cfrom := *(*C.double)(unsafe.Pointer(&from))
	cto := *(*C.double)(unsafe.Pointer(&to))

	C.atg_uniform_(ptr, self, cfrom, cto)
}

// void atg_zeros_like(tensor *, tensor self);
func AtgZerosLike(ptr *Ctensor, self Ctensor) {
	C.atg_zeros_like(ptr, self)
}

// void atg_fill_(tensor *, tensor self, scalar value);
func AtgFill_(ptr *Ctensor, self Ctensor, value Cscalar) {
	C.atg_fill_(ptr, self, value)
}

// void atg_randn_like(tensor *, tensor self);
func AtgRandnLike(ptr *Ctensor, self Ctensor) {
	C.atg_rand_like(ptr, self)
}
