package libtch

//#include "stddef.h"
//#include "stdbool.h"
//#include "torch_api.h"
//#include "stdlib.h"
//void callback_fn(void *, char *, tensor);
//typedef void (*f)(void *, char *, tensor);
import "C"

import (
	"unsafe"
)

// NOTE: C.tensor is a C pointer to torch::Tensor
type Ctensor = C.tensor
type Cscalar = C.scalar

type NamedCtensor struct {
	Name    string
	Ctensor C.tensor
}

type LoadData struct {
	NamedCtensors []NamedCtensor
}

var PStore = NewPointerStore()

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

// tensor at_shallow_clone(tensor);
func AtShallowClone(ts Ctensor) Ctensor {
	ctensor := (C.tensor)(ts)
	return C.at_shallow_clone(ctensor)
}

// tensor at_get(tensor, int index);
func AtGet(ts Ctensor, index int) Ctensor {
	ctensor := (C.tensor)(ts)
	cindex := *(*C.int)(unsafe.Pointer(&index))
	return C.at_get(ctensor, cindex)
}

// void at_copy_(tensor dst, tensor src);
func AtCopy_(dst Ctensor, src Ctensor) {
	cdst := (C.tensor)(dst)
	csrc := (C.tensor)(src)
	C.at_copy_(cdst, csrc)
}

// void at_save(tensor, char *filename);
func AtSave(ts Ctensor, path string) {
	ctensor := (C.tensor)(ts)
	cstringPtr := C.CString(path)
	defer C.free(unsafe.Pointer(cstringPtr))
	C.at_save(ctensor, cstringPtr)
}

// tensor at_load(char *filename);
func AtLoad(path string) Ctensor {
	cstringPtr := C.CString(path)
	defer C.free(unsafe.Pointer(cstringPtr))
	return C.at_load(cstringPtr)
}

// void at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
func AtSaveMulti(tensors []Ctensor, tensor_names []string, ntensors int, filename string) {

	var ctensors []C.tensor
	for i := 0; i < len(tensors); i++ {
		ctensors = append(ctensors, (C.tensor)(tensors[i]))
	}

	cpointerSize := 4
	cnamesPtr := (*[1 << 30]**C.char)(C.malloc(C.size_t(cpointerSize * len(tensor_names))))
	for i := 0; i < len(tensor_names); i++ {
		cname := C.CString(tensor_names[i])
		cnamesPtr[i] = &cname
		// defer C.free(unsafe.Pointer(cnamesPtr[i]))
	}
	cntensors := *(*C.int)(unsafe.Pointer(&ntensors))
	cfilename := C.CString(filename)

	C.at_save_multi(&ctensors[0], cnamesPtr[0], cntensors, cfilename)
}

/* [at_load_multi] takes as input an array of nullptr for [tensors]. */
// void at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
func AtLoadMulti(tensors []Ctensor, tensor_names []string, ntensors int, filename string) {
	// TODO: implement this
}

// void at_load_callback(char *filename, void *data, void (*f)(void *, char *, tensor));
/*
 * void at_load_callback(char *filename, void *data, void (*f)(void *, char *, tensor)) {
 *   PROTECT(
 *     auto module = torch::jit::load(filename);
 *     for (const auto &p : module.named_parameters()) {
 *       auto v = p.value;
 *       f(data, (char*)p.name.c_str(), new torch::Tensor(v));
 *     }
 *   )
 * }
 *  */
func AtLoadCallback(filename string, dataPtr unsafe.Pointer) {
	cfilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cfilename))
	C.at_load_callback(cfilename, dataPtr, C.f(C.callback_fn))
}

//TODO: move `callback_fn` to wrapper package???

//export callback_fn
func callback_fn(dataPtr unsafe.Pointer, name *C.char, ctensor C.tensor) {
	tsName := C.GoString(name)
	namedCtensor := NamedCtensor{
		Name:    tsName,
		Ctensor: ctensor,
	}

	data := PStore.Get(dataPtr).(*LoadData)
	data.NamedCtensors = append(data.NamedCtensors, namedCtensor)
}

/*
 * void at_load_callback_with_device(char *filename, void *data, void (*f)(void *, char *, tensor), int device_id) {
 *   PROTECT(
 *     auto module = torch::jit::load(filename, device_of_int(device_id));
 *     for (const auto &p : module.named_parameters()) {
 *       auto v = p.value;
 *       f(data, (char*)p.name.c_str(), new torch::Tensor(v));
 *     }
 *   )
 * }
 *  */
func AtLoadCallbackWithDevice(filename string, dataPtr unsafe.Pointer, device int32) {
	cfilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cfilename))
	cdevice := *(*C.int)(unsafe.Pointer(&device))
	C.at_load_callback_with_device(cfilename, dataPtr, C.f(C.callback_fn), cdevice)
}

/*
 * char *at_to_string(tensor t, int line_size) {
 *   PROTECT(
 *     std::ostringstream oss;
 *     torch::print(oss, *t, line_size);
 *     return strdup(oss.str().c_str());
 *   )
 *   return nullptr;
 * }
 *  */
func AtToString(ts Ctensor, lineSize int64) string {
	ctensor := (C.tensor)(ts)
	clineSize := *(*C.int)(unsafe.Pointer(&lineSize))
	charPtr := C.at_to_string(ctensor, clineSize)
	goString := C.GoString(charPtr)

	return goString
}

// void at_free(tensor);
func AtFree(ts Ctensor) {
	ctensor := (C.tensor)(ts)
	C.at_free(ctensor)
}

//int at_grad_set_enabled(int b);
func AtGradSetEnabled(b int) int {
	cbool := *(*C.int)(unsafe.Pointer(&b))
	cretVal := C.at_grad_set_enabled(cbool)
	return *(*int)(unsafe.Pointer(&cretVal))
}
