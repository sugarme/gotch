package libtch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import (
	"unsafe"
)

// NOTE: C.tensor is a C pointer to torch::Tensor
type Ctensor = C.tensor
type Cscalar = C.scalar

func AtNewTensor() Ctensor {
	return C.at_new_tensor()
}

// tensor at_new_tensor();
func NewTensor() Ctensor {
	return C.at_new_tensor()
}

// tensor at_tensor_of_data(void *vs, int64_t *dims, size_t ndims, size_t element_size_in_bytes, int type);
func AtTensorOfData(vs unsafe.Pointer, dims []int64, ndims uint, elt_size_in_bytes uint, kind int) Ctensor {

	// just get pointer of the first element of shape
	c_dims := (*C.int64_t)(unsafe.Pointer(&dims[0]))
	c_ndims := *(*C.size_t)(unsafe.Pointer(&ndims))
	c_elt_size_in_bytes := *(*C.size_t)(unsafe.Pointer(&elt_size_in_bytes))
	c_kind := *(*C.int)(unsafe.Pointer(&kind))

	return C.at_tensor_of_data(vs, c_dims, c_ndims, c_elt_size_in_bytes, c_kind)

}

// void at_print(tensor);
func AtPrint(t Ctensor) {
	C.at_print(t)
}

// void *at_data_ptr(tensor);
func AtDataPtr(t Ctensor) unsafe.Pointer {
	return C.at_data_ptr(t)
}

// size_t at_dim(tensor);
func AtDim(t Ctensor) uint64 {
	result := C.at_dim(t)
	return *(*uint64)(unsafe.Pointer(&result))
}

// void at_shape(tensor, int64_t *);
func AtShape(t Ctensor, ptr unsafe.Pointer) {
	c_ptr := (*C.long)(ptr)
	C.at_shape(t, c_ptr)
}

// int at_scalar_type(tensor);
func AtScalarType(t Ctensor) int32 {
	result := C.at_scalar_type(t)
	return *(*int32)(unsafe.Pointer(&result))
}

func GetAndResetLastErr() *C.char {
	return C.get_and_reset_last_err()
}

// int atc_cuda_device_count();
func AtcCudaDeviceCount() int {
	result := C.atc_cuda_device_count()
	return *(*int)(unsafe.Pointer(&result))
}

// int atc_cuda_is_available();
func AtcCudaIsAvailable() bool {
	result := C.atc_cuda_is_available()
	return *(*bool)(unsafe.Pointer(&result))
}

// int atc_cudnn_is_available();
func AtcCudnnIsAvailable() bool {
	result := C.atc_cudnn_is_available()
	return *(*bool)(unsafe.Pointer(&result))
}

// void atc_set_benchmark_cudnn(int b);
func AtcSetBenchmarkCudnn(b int) {
	cb := *(*C.int)(unsafe.Pointer(&b))
	C.atc_set_benchmark_cudnn(cb)
}

// double at_double_value_at_indexes(tensor, int64_t *indexes, int indexes_len);
func AtDoubleValueAtIndexes(ts Ctensor, indexes unsafe.Pointer, indexesLen int) float64 {
	ctensor := (C.tensor)(ts)
	cindexes := (*C.long)(indexes)
	cindexesLen := *(*C.int)(unsafe.Pointer(&indexesLen))
	retVal := C.at_double_value_at_indexes(ctensor, cindexes, cindexesLen)
	return *(*float64)(unsafe.Pointer(&retVal))
}

// int64_t at_int64_value_at_indexes(tensor, int64_t *indexes, int indexes_len);
func AtInt64ValueAtIndexes(ts Ctensor, indexes unsafe.Pointer, indexesLen int) int64 {
	ctensor := (C.tensor)(ts)
	cindexes := (*C.long)(indexes)
	cindexesLen := *(*C.int)(unsafe.Pointer(&indexesLen))
	retVal := C.at_int64_value_at_indexes(ctensor, cindexes, cindexesLen)
	return *(*int64)(unsafe.Pointer(&retVal))
}

// int at_requires_grad(tensor);
func AtRequiresGrad(ts Ctensor) bool {
	retVal := C.at_requires_grad((C.tensor)(ts))
	return *(*bool)(unsafe.Pointer(&retVal))
}

// int at_defined(tensor);
func AtDefined(ts Ctensor) bool {
	retVal := C.at_defined((C.tensor)(ts))
	return *(*bool)(unsafe.Pointer(&retVal))
}

// int at_is_sparse(tensor);
func AtIsSparse(ts Ctensor) bool {
	retVal := C.at_is_sparse((C.tensor)(ts))
	return *(*bool)(unsafe.Pointer(&retVal))
}

// void at_backward(tensor, int, int);
func AtBackward(ts Ctensor, keepGraph int, createGraph int) {
	ctensor := (C.tensor)(ts)
	ckeepGraph := *(*C.int)(unsafe.Pointer(&keepGraph))
	ccreateGraph := *(*C.int)(unsafe.Pointer(&createGraph))

	C.at_backward(ctensor, ckeepGraph, ccreateGraph)
}

/*
 * void at_run_backward(tensor *tensors,
 *                       int ntensors,
 *                       tensor *inputs,
 *                       int ninputs,
 *                       tensor *outputs,
 *                       int keep_graph,
 *                       int create_graph);
 *  */
func AtRunBackward(tensorsPtr *Ctensor, ntensors int, inputsPtr *Ctensor, ninputs int, outputsPtr *Ctensor, keepGraph int, createGraph int) {
	cntensors := *(*C.int)(unsafe.Pointer(&ntensors))
	cninputs := *(*C.int)(unsafe.Pointer(&ninputs))
	ckeepGraph := *(*C.int)(unsafe.Pointer(&keepGraph))
	ccreateGraph := *(*C.int)(unsafe.Pointer(&createGraph))
	C.at_run_backward(tensorsPtr, cntensors, inputsPtr, cninputs, outputsPtr, ckeepGraph, ccreateGraph)
}

// void at_copy_data(tensor tensor, void *vs, size_t numel, size_t element_size_in_bytes);
func AtCopyData(tensor Ctensor, vs unsafe.Pointer, numel uint, element_size_in_bytes uint) {
	ctensor := (C.tensor)(tensor)
	cnumel := *(*C.size_t)(unsafe.Pointer(&numel))
	celement_size_in_bytes := *(*C.size_t)(unsafe.Pointer(&element_size_in_bytes))
	C.at_copy_data(ctensor, vs, cnumel, celement_size_in_bytes)
}
