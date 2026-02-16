package libtch

//#include "stddef.h"
//#include "stdbool.h"
//#include "torch_api.h"
//#include "stdlib.h"
//void callback_fn(void *, char *, tensor);
//typedef void (*f)(void *, char *, tensor);
import "C"

import (
	"bytes"
	"encoding/binary"
	"log"
	"strings"
	"unsafe"
)

var nativeEndian binary.ByteOrder

func init() {
	buf := [2]byte{}
	*(*uint16)(unsafe.Pointer(&buf[0])) = uint16(0xABCD)

	switch buf {
	case [2]byte{0xCD, 0xAB}:
		nativeEndian = binary.LittleEndian
	case [2]byte{0xAB, 0xCD}:
		nativeEndian = binary.BigEndian
	default:
		panic("Could not determine native endianness.")
	}
}

// NOTE: C.tensor is a C pointer to torch::Tensor
type Ctensor = C.tensor
type Cscalar = C.scalar
type Coptimizer = C.optimizer
type Civalue = C.ivalue
type Cmodule = C.module

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

// int at_defined(tensor);
func AtDefined(ts Ctensor) bool {
	retVal := C.at_defined(ts)
	return *(*bool)(unsafe.Pointer(&retVal))
}

// int at_is_mkldnn(tensor);
func AtIsMkldnn(ts Ctensor) bool {
	retVal := C.at_is_mkldnn(ts)
	return *(*bool)(unsafe.Pointer(&retVal))
}

// int at_is_sparse(tensor);
func AtIsSparse(ts Ctensor) bool {
	retVal := C.at_is_sparse(ts)
	return *(*bool)(unsafe.Pointer(&retVal))
}

// int at_device(tensor);
// PyTorch 2.10.0: Use direct type conversion instead of unsafe pointer casting
func AtDevice(ts Ctensor) int {
	cint := C.at_device(ts)
	return int(cint)
}

// size_t at_dim(tensor);
func AtDim(t Ctensor) uint64 {
	result := C.at_dim(t)
	return *(*uint64)(unsafe.Pointer(&result))
}

// void at_shape(tensor, int64_t *);
func AtShape(t Ctensor, ptr unsafe.Pointer) {
	c_ptr := (*C.int64_t)(ptr)
	C.at_shape(t, c_ptr)
}

// void at_stride(tensor, int64_t *);
func AtStride(t Ctensor, ptr unsafe.Pointer) {
	c_ptr := (*C.int64_t)(ptr)
	C.at_stride(t, c_ptr)
}

// int at_scalar_type(tensor);
func AtScalarType(t Ctensor) int32 {
	result := C.at_scalar_type(t)
	return *(*int32)(unsafe.Pointer(&result))
}

// int at_is_contiguous(tensor);
func AtIsContiguous(ts Ctensor) bool {
	retVal := C.at_is_contiguous(ts)
	return *(*bool)(unsafe.Pointer(&retVal))
}

func GetAndResetLastErr() *C.char {
	return C.get_and_reset_last_err()
}

// int atc_cuda_device_count();
func AtcCudaDeviceCount() int32 {
	result := C.atc_cuda_device_count()
	return *(*int32)(unsafe.Pointer(&result))
}

// int atc_cuda_is_available();
func AtcCudaIsAvailable() bool {
	result := C.atc_cuda_is_available()
	return *(*bool)(unsafe.Pointer(&result))
}

// int atc_mps_is_available();
func AtcMpsIsAvailable() bool {
	result := C.atc_mps_is_available()
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

// void atc_synchronize(int64_t device_index);
func AtcSynchronize(deviceIndex int64) {
	cDeviceIndex := *(*C.int64_t)(unsafe.Pointer(&deviceIndex))
	C.atc_synchronize(cDeviceIndex)
}

// int atc_get_device();
func AtcGetDevice() int {
	cDeviceIndex := C.atc_get_device()
	return int(cDeviceIndex)
}

// int atc_set_device(int device_index);
func AtcSetDevice(deviceIndex int) int {
	cDeviceIndex := C.int(deviceIndex)
	return int(cDeviceIndex)
}

// double at_double_value_at_indexes(tensor, int64_t *indexes, int indexes_len);
func AtDoubleValueAtIndexes(ts Ctensor, indexes unsafe.Pointer, indexesLen int) float64 {
	ctensor := (C.tensor)(ts)
	cindexes := (*C.int64_t)(indexes)
	cindexesLen := *(*C.int)(unsafe.Pointer(&indexesLen))
	retVal := C.at_double_value_at_indexes(ctensor, cindexes, cindexesLen)
	return *(*float64)(unsafe.Pointer(&retVal))
}

// int64_t at_int64_value_at_indexes(tensor, int64_t *indexes, int indexes_len);
func AtInt64ValueAtIndexes(ts Ctensor, indexes unsafe.Pointer, indexesLen int) int64 {
	ctensor := (C.tensor)(ts)
	cindexes := (*C.int64_t)(indexes)
	cindexesLen := *(*C.int)(unsafe.Pointer(&indexesLen))
	retVal := C.at_int64_value_at_indexes(ctensor, cindexes, cindexesLen)
	return *(*int64)(unsafe.Pointer(&retVal))
}

// int at_requires_grad(tensor);
func AtRequiresGrad(ts Ctensor) bool {
	retVal := C.at_requires_grad(ts)
	return *(*bool)(unsafe.Pointer(&retVal))
}

// void at_backward(tensor, int, int);
func AtBackward(ts Ctensor, keepGraph int, createGraph int) {
	ckeepGraph := *(*C.int)(unsafe.Pointer(&keepGraph))
	ccreateGraph := *(*C.int)(unsafe.Pointer(&createGraph))

	C.at_backward(ts, ckeepGraph, ccreateGraph)
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
func AtCopyData(ts Ctensor, vs unsafe.Pointer, numel uint, element_size_in_bytes uint) {
	cnumel := *(*C.size_t)(unsafe.Pointer(&numel))
	celement_size_in_bytes := *(*C.size_t)(unsafe.Pointer(&element_size_in_bytes))
	C.at_copy_data(ts, vs, cnumel, celement_size_in_bytes)
}

// tensor at_shallow_clone(tensor);
func AtShallowClone(ts Ctensor) Ctensor {
	return C.at_shallow_clone(ts)
}

// tensor at_get(tensor, int index);
func AtGet(ts Ctensor, index int) Ctensor {
	cindex := *(*C.int)(unsafe.Pointer(&index))
	return C.at_get(ts, cindex)
}

// void at_copy_(tensor dst, tensor src);
func AtCopy_(dst Ctensor, src Ctensor) {
	C.at_copy_(dst, src)
}

// void at_save(tensor, char *filename);
func AtSave(ts Ctensor, path string) {
	cstringPtr := C.CString(path)
	defer C.free(unsafe.Pointer(cstringPtr))
	C.at_save(ts, cstringPtr)
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
	// defer C.free(unsafe.Pointer(cnamesPtr))
	for i := 0; i < len(tensor_names); i++ {
		cname := C.CString(tensor_names[i])
		cnamesPtr[i] = &cname
		// defer C.free(unsafe.Pointer(cnamesPtr[i]))
	}
	cntensors := *(*C.int)(unsafe.Pointer(&ntensors))
	cfilename := C.CString(filename)

	C.at_save_multi(&ctensors[0], cnamesPtr[0], cntensors, cfilename)
}

// void at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
func AtSaveMultiNew(tensors []Ctensor, names []string, filename string) {
	// NOTE. namedTensors is slice of tensors which wrap Ctensor pointer.
	// However, they are not neccessary consecutive Ctensor pointer.
	// The following code will create 2 arrays to contain ctensors and cnames
	// 1. Calculate memory size for the array.
	// 2. Copy C pointer values to the arrays.

	ntensors := len(tensors)
	// number of bytes for each array of pointers.
	nbytes := C.size_t(ntensors) * C.size_t(unsafe.Sizeof(uintptr(0)))

	cnamesPtr := make([]*C.char, ntensors)
	ctensorsPtr := (*[1 << 30]C.tensor)(C.malloc(nbytes))
	defer C.free(unsafe.Pointer(ctensorsPtr))

	for i := 0; i < ntensors; i++ {
		cname := C.CString(names[i])
		defer C.free(unsafe.Pointer(cname))
		cnamesPtr[i] = cname
		ctensorsPtr[i] = (C.tensor)(tensors[i])
	}

	cntensors := *(*C.int)(unsafe.Pointer(&ntensors))
	cfilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cfilename))

	C.at_save_multi(&ctensorsPtr[0], &cnamesPtr[0], cntensors, cfilename)
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
	tsName = strings.ReplaceAll(tsName, "|", ".")
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
	clineSize := *(*C.int)(unsafe.Pointer(&lineSize))
	charPtr := C.at_to_string(ts, clineSize)
	goString := C.GoString(charPtr)

	return goString
}

// void at_free(tensor);
func AtFree(ts Ctensor) {
	C.at_free(ts)
}

// int at_grad_set_enabled(int b);
// PyTorch 2.10.0: Use direct type conversion instead of unsafe pointer casting
// to avoid reading garbage memory (C.int is 32-bit, Go int is 64-bit)
func AtGradSetEnabled(b int) int {
	cbool := C.int(b)
	cretVal := C.at_grad_set_enabled(cbool)
	return int(cretVal)
}

/*
 * optimizer ato_adam(double learning_rate,
 *                    double beta1,
 *                    double beta2,
 *                    double weight_decay);
 *  */
func AtoAdam(learningRate, beta1, beta2, weightDecay float64) Coptimizer {
	clearningRate := *(*C.double)(unsafe.Pointer(&learningRate))
	cbeta1 := *(*C.double)(unsafe.Pointer(&beta1))
	cbeta2 := *(*C.double)(unsafe.Pointer(&beta2))
	cweightDecay := *(*C.double)(unsafe.Pointer(&weightDecay))

	return C.ato_adam(clearningRate, cbeta1, cbeta2, cweightDecay)
}

func AtoAdamW(learningRate, beta1, beta2, weightDecay float64) Coptimizer {
	clearningRate := *(*C.double)(unsafe.Pointer(&learningRate))
	cbeta1 := *(*C.double)(unsafe.Pointer(&beta1))
	cbeta2 := *(*C.double)(unsafe.Pointer(&beta2))
	cweightDecay := *(*C.double)(unsafe.Pointer(&weightDecay))

	return C.ato_adamw(clearningRate, cbeta1, cbeta2, cweightDecay)
}

/*
 * optimizer ato_rms_prop(double learning_rate,
 *                        double alpha,
 *                        double eps,
 *                        double weight_decay,
 *                        double momentum,
 *                        int centered);
 *  */
func AtoRmsProp(learningRate, alpha, eps, weightDecay, momentum float64, centered int) Coptimizer {
	clearningRate := *(*C.double)(unsafe.Pointer(&learningRate))
	calpha := *(*C.double)(unsafe.Pointer(&alpha))
	ceps := *(*C.double)(unsafe.Pointer(&eps))
	cweightDecay := *(*C.double)(unsafe.Pointer(&weightDecay))
	cmomentum := *(*C.double)(unsafe.Pointer(&momentum))
	ccentered := *(*C.int)(unsafe.Pointer(&centered))

	return C.ato_rms_prop(clearningRate, calpha, ceps, cweightDecay, cmomentum, ccentered)
}

/*
 * optimizer ato_sgd(double learning_rate,
 *                   double momentum,
 *                   double dampening,
 *                   double weight_decay,
 *                   int nesterov);
 *  */
func AtoSgd(learningRate, momentum, dampening, weightDecay float64, nesterov int) Coptimizer {
	clearningRate := *(*C.double)(unsafe.Pointer(&learningRate))
	cmomentum := *(*C.double)(unsafe.Pointer(&momentum))
	cdampening := *(*C.double)(unsafe.Pointer(&dampening))
	cweightDecay := *(*C.double)(unsafe.Pointer(&weightDecay))
	cnesterov := *(*C.int)(unsafe.Pointer(&nesterov))

	return C.ato_sgd(clearningRate, cmomentum, cdampening, cweightDecay, cnesterov)
}

// NOTE. Backward compat for param group not updated (#261)
// void ato_add_parameters(optimizer, tensor *, int ntensors);
func AtoAddParametersOld(coptimizer Coptimizer, tensors []Ctensor, ntensors int) {

	var ctensors []C.tensor
	for i := 0; i < len(tensors); i++ {
		ctensors = append(ctensors, (C.tensor)(tensors[i]))
	}

	cntensors := *(*C.int)(unsafe.Pointer(&ntensors))

	// Just give pointer to the first element of ctensors slice
	C.ato_add_parameters_old(coptimizer, &ctensors[0], cntensors)
}

func AtoAddParameter(coptimizer Coptimizer, tensor Ctensor, group uint) {
	cgroup := *(*C.size_t)(unsafe.Pointer(&group))
	C.ato_add_parameter(coptimizer, tensor, cgroup)
}

// void ato_set_learning_rate(optimizer, double learning_rate);
func AtoSetLearningRate(coptimizer Coptimizer, learningRate float64) {
	clearningRate := *(*C.double)(unsafe.Pointer(&learningRate))
	C.ato_set_learning_rate(coptimizer, clearningRate)
}

func AtoGetLearningRates(coptimizer Coptimizer) []float64 {
	cLRsPtr := (*C.double)(unsafe.Pointer(C.malloc(0)))
	cngroup := (*C.int)(unsafe.Pointer(C.malloc(0)))

	C.ato_get_learning_rates(coptimizer, cLRsPtr, cngroup)
	ngroup := *(*int)(unsafe.Pointer(cngroup))

	// NOTE. temp fix `panic: runtime error: makeslice: len out of range`
	// due to error in C side with ngroup ridiculous huge number (ie. ngroup: 139745350909953)
	if ngroup > 1000 {
		ngroup = 1
	}

	var lrs []float64 = make([]float64, ngroup)
	var currPtr *C.double = cLRsPtr
	for i := 0; i < ngroup; i++ {
		lrs[i] = *(*float64)(unsafe.Pointer(currPtr))
		nextPtr := (*C.double)(unsafe.Pointer(uintptr(unsafe.Pointer(currPtr)) + unsafe.Sizeof(currPtr)))
		currPtr = nextPtr
	}

	return lrs
}

func AtoSetLearningRates(coptimizer Coptimizer, lrs []float64) {
	elementNum := len(lrs)
	eltSizeInBytes := 8 // float64 takes 8 Bytes
	nbytes := int(eltSizeInBytes) * int(elementNum)
	dataPtr := C.malloc(C.size_t(nbytes))
	defer C.free(unsafe.Pointer(dataPtr))
	dataSlice := (*[1 << 32]byte)(dataPtr)[:nbytes:nbytes] // 4294967296
	buf := bytes.NewBuffer(dataSlice[:0:nbytes])
	if err := binary.Write(buf, nativeEndian, lrs); err != nil {
		log.Fatal(err)
	}

	clrs := (*C.double)(dataPtr)
	lrsNum := len(lrs)
	clrsNum := *(*C.int)(unsafe.Pointer(&lrsNum))
	C.ato_set_learning_rates(coptimizer, clrs, clrsNum)
}

func AtoParamGroupNum(coptimizer Coptimizer) int64 {
	cpgNum := C.ato_param_group_num(coptimizer)

	pgNum := *(*int64)(unsafe.Pointer(&cpgNum))
	return pgNum
}

func AtoAddParamGroup(coptimizer Coptimizer, tensors []Ctensor, ntensors int) {
	var ctensors []C.tensor
	for i := 0; i < len(tensors); i++ {
		ctensors = append(ctensors, (C.tensor)(tensors[i]))
	}
	cntensors := *(*C.int)(unsafe.Pointer(&ntensors))

	C.ato_add_param_group(coptimizer, &ctensors[0], cntensors)
}

// void ato_set_momentum(optimizer, double momentum);
func AtoSetMomentum(coptimizer Coptimizer, momentum float64) {
	cmomentum := *(*C.double)(unsafe.Pointer(&momentum))

	C.ato_set_momentum(coptimizer, cmomentum)
}

// void ato_zero_grad(optimizer);
func AtoZeroGrad(coptimizer Coptimizer) {

	C.ato_zero_grad(coptimizer)
}

// void ato_step(optimizer);
func AtoStep(coptimizer Coptimizer) {

	C.ato_step(coptimizer)
}

// void ato_free(optimizer);
func AtoFree(coptimizer Coptimizer) {
	C.ato_free(coptimizer)
}

// tensor at_load_image(char *filename);
func AtLoadImage(path string) Ctensor {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	return C.at_load_image(cpath)
}

// int at_save_image(tensor, char *filename);
func AtSaveImage(ts Ctensor, path string) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	// TODO: we don't take the return value
	// as we handle error with `TochErr()` anyway
	_ = C.at_save_image(ts, cpath)
}

// tensor at_resize_image(tensor, int w, int h);
func AtResizeImage(ts Ctensor, w, h int64) Ctensor {

	cw := *(*C.int)(unsafe.Pointer(&w))
	ch := *(*C.int)(unsafe.Pointer(&h))

	return C.at_resize_image(ts, cw, ch)
}

// ivalue ati_none();
func AtiNone() Civalue {
	return C.ati_none()
}

// ivalue ati_tensor(tensor);
func AtiTensor(ts Ctensor) Civalue {
	return C.ati_tensor(ts)
}

// ivalue ati_int(int64_t);
func AtiInt(val int64) Civalue {
	cval := *(*C.int64_t)(unsafe.Pointer(&val))
	result := C.ati_int(cval)
	return result
}

// ivalue ati_double(double);
func AtiDouble(val float64) Civalue {
	cval := *(*C.double)(unsafe.Pointer(&val))
	return C.ati_double(cval)
}

// ivalue ati_bool(int);
func AtiBool(val bool) Civalue {
	ival := 0
	if val {
		ival = 1
	}
	cval := *(*C.int)(unsafe.Pointer(&ival))
	return C.ati_bool(cval)
}

// ivalue ati_string(char *);
func AtiString(val string) Civalue {
	cval := C.CString(val)
	return C.ati_string(cval)
}

// ivalue ati_tuple(ivalue *, int);
func AtiTuple(tupleData []Civalue, tupleLen int) Civalue {
	ctupleDataPtr := (*C.ivalue)(unsafe.Pointer(&tupleData[0]))
	ctupleLen := *(*C.int)(unsafe.Pointer(&tupleLen))

	return C.ati_tuple(ctupleDataPtr, ctupleLen)
}

// ivalue ati_generic_list(ivalue *, int);
func AtiGenericList(genericListData []Civalue, genericListLen int) Civalue {
	cgenericListDataPtr := (*C.ivalue)(unsafe.Pointer(&genericListData[0]))
	cgenericListLen := *(*C.int)(unsafe.Pointer(&genericListLen))

	return C.ati_generic_list(cgenericListDataPtr, cgenericListLen)
}

// ivalue ati_generic_dict(ivalue *, int);
func AtiGenericDict(genericDictData []Civalue, genericDictLen int) Civalue {
	cgenericDictDataPtr := (*C.ivalue)(unsafe.Pointer(&genericDictData[0]))
	cgenericDictLen := *(*C.int)(unsafe.Pointer(&genericDictLen))

	return C.ati_generic_dict(cgenericDictDataPtr, cgenericDictLen)
}

// ivalue ati_int_list(int64_t *, int);
func AtiIntList(intListData []int64, intListLen int) Civalue {
	cintListDataPtr := (*C.int64_t)(unsafe.Pointer(&intListData[0]))
	cintListLen := *(*C.int)(unsafe.Pointer(&intListLen))

	return C.ati_int_list(cintListDataPtr, cintListLen)
}

// ivalue ati_double_list(double *, int);
func AtiDoubleList(doubleListData []float64, doubleListLen int) Civalue {
	cdoubleListDataPtr := (*C.double)(unsafe.Pointer(&doubleListData[0]))
	cdoubleListLen := *(*C.int)(unsafe.Pointer(&doubleListLen))

	return C.ati_double_list(cdoubleListDataPtr, cdoubleListLen)
}

// ivalue ati_bool_list(char *, int);
func AtiBoolList(boolListData []bool, boolListLen int) Civalue {
	var cboolListData []int
	for _, v := range boolListData {
		item := 0
		if v {
			item = 1
		}
		cboolListData = append(cboolListData, item)
	}
	cboolListDataPtr := (*C.char)(unsafe.Pointer(&cboolListData[0]))
	cboolListLen := *(*C.int)(unsafe.Pointer(&boolListLen))

	return C.ati_bool_list(cboolListDataPtr, cboolListLen)
}

// ivalue ati_tensor_list(tensor *, int);
func AtiTensorList(tensorListData []Ctensor, tensorListLen int) Civalue {
	ctensorListDataPtr := (*C.tensor)(unsafe.Pointer(&tensorListData[0]))
	ctensorListLen := *(*C.int)(unsafe.Pointer(&tensorListLen))

	return C.ati_tensor_list(ctensorListDataPtr, ctensorListLen)
}

// tensor ati_to_tensor(ivalue);
func AtiToTensor(val Civalue) Ctensor {
	return C.ati_to_tensor(val)
}

// int64_t ati_to_int(ivalue);
func AtiToInt(val Civalue) int64 {
	cval := C.ati_to_int(val)
	return *(*int64)(unsafe.Pointer(&cval))
}

// double ati_to_double(ivalue);
func AtiToDouble(val Civalue) float64 {
	cval := C.ati_to_double(val)
	return *(*float64)(unsafe.Pointer(&cval))
}

// char *ati_to_string(ivalue);
func AtiToString(val Civalue) string {
	cval := C.ati_to_string(val)
	return C.GoString(cval)
}

// int ati_to_bool(ivalue);
func AtiToBool(val Civalue) bool {
	cval := C.ati_to_bool(val)
	goval := *(*int32)(unsafe.Pointer(&cval))
	return goval == 1
}

// int ati_length(ivalue);
func AtiLength(val Civalue) int32 {
	cval := C.ati_length(val)
	return *(*int32)(unsafe.Pointer(&cval))
}

// int ati_tuple_length(ivalue);
func AtiTupleLength(val Civalue) int32 {
	cval := C.ati_tuple_length(val)
	return *(*int32)(unsafe.Pointer(&cval))
}

// void ati_to_tuple(ivalue, ivalue *, int);
func AtiToTuple(val Civalue, ptr *Civalue, tupleLen int) {

	ctupleLen := *(*C.int)(unsafe.Pointer(&tupleLen))
	C.ati_to_tuple(val, ptr, ctupleLen)
}

// void ati_to_generic_list(ivalue, ivalue *, int);
func AtiToGenericList(val Civalue, ptr *Civalue, genericListLen int) {

	cgenericListLen := *(*C.int)(unsafe.Pointer(&genericListLen))
	C.ati_to_generic_list(val, ptr, cgenericListLen)
}

// void ati_to_generic_dict(ivalue, ivalue *, int);
func AtiToGenericDict(val Civalue, ptr *Civalue, genericDictLen int) {

	cgenericDictLen := *(*C.int)(unsafe.Pointer(&genericDictLen))
	C.ati_to_generic_dict(val, ptr, cgenericDictLen)
}

// void ati_to_int_list(ivalue, int64_t *, int);
func AtiToIntList(val Civalue, ptr unsafe.Pointer, intListLen int) {

	cptr := (*C.int64_t)(ptr)
	cintListLen := *(*C.int)(unsafe.Pointer(&intListLen))
	C.ati_to_int_list(val, cptr, cintListLen)
}

// void ati_to_double_list(ivalue, double *, int);
func AtiToDoubleList(val Civalue, ptr unsafe.Pointer, doubleListLen int) {

	cptr := (*C.double)(ptr)
	cdoubleListLen := *(*C.int)(unsafe.Pointer(&doubleListLen))
	C.ati_to_double_list(val, cptr, cdoubleListLen)
}

// void ati_to_bool_list(ivalue, char *, int);
func AtiToBoolList(val Civalue, ptr unsafe.Pointer, boolListLen int) {

	cptr := (*C.char)(ptr)
	cboolListLen := *(*C.int)(unsafe.Pointer(&boolListLen))

	C.ati_to_bool_list(val, cptr, cboolListLen)
}

// void ati_to_tensor_list(ivalue, tensor *, int);
func AtiToTensorList(val Civalue, ptr *Ctensor, tensorListLen int) {
	ctensorListLen := *(*C.int)(unsafe.Pointer(&tensorListLen))

	C.ati_to_tensor_list(val, ptr, ctensorListLen)
}

// int ati_tag(ivalue);
func AtiTag(val Civalue) int32 {

	ctag := C.ati_tag(val)
	return *(*int32)(unsafe.Pointer(&ctag))
}

// void ati_free(ivalue);
func AtiFree(val Civalue) {
	C.ati_free(val)
}

// module atm_load(char *);
func AtmLoad(path string) Cmodule {
	ptr := C.CString(path)
	return C.atm_load(ptr)
}

// module atm_load_on_device(char *, int device);
func AtmLoadOnDevice(path string, device int32) Cmodule {
	ptr := C.CString(path)
	cdevice := *(*C.int)(unsafe.Pointer(&device))
	return C.atm_load_on_device(ptr, cdevice)
}

// module atm_load_str(char *, size_t sz);
func AtmLoadStr(val string, sz int) Cmodule {
	ptr := C.CString(val)
	csz := *(*C.size_t)(unsafe.Pointer(&sz))

	return C.atm_load_str(ptr, csz)
}

// module atm_load_str_on_device(char *, size_t sz, int device);
func AtmLoadStrOnDevice(val string, sz int, device int32) Cmodule {
	ptr := C.CString(val)
	csz := *(*C.size_t)(unsafe.Pointer(&sz))
	cdevice := *(*C.int)(unsafe.Pointer(&device))

	return C.atm_load_str_on_device(ptr, csz, cdevice)
}

// void atm_save(module m, char *);
func AtmSave(m Cmodule, path string) {
	ptr := C.CString(path)
	C.atm_save(m, ptr)
}

// void atm_named_parameters(module, void *data, void (*f)(void *, char *, tensor));
func AtmNamedParameters(m Cmodule, dataPtr unsafe.Pointer) {
	C.atm_named_parameters(m, dataPtr, C.f(C.callback_fn))
}

// tensor atm_forward(module, tensor *tensors, int ntensors);
func AtmForward(m Cmodule, tensors *Ctensor, ntensors int) Ctensor {
	cntensors := *(*C.int)(unsafe.Pointer(&ntensors))
	return C.atm_forward(m, tensors, cntensors)
}

// ivalue atm_forward_(module, ivalue *ivalues, int nivalues);
func AtmForward_(m Cmodule, ivalues *Civalue, nivalues int) Civalue {
	cnivalues := *(*C.int)(unsafe.Pointer(&nivalues))
	return C.atm_forward_(m, ivalues, cnivalues)
}

// void atm_free(module);
func AtmFree(m Cmodule) {
	C.atm_free(m)
}

// void atm_to(module m, int device, int dtype, bool non_blocking);
func AtmTo(m Cmodule, device int32, dtype int32, nonBlocking bool) {
	cdevice := *(*C.int)(unsafe.Pointer(&device))
	cdtype := *(*C.int)(unsafe.Pointer(&dtype))
	cnonBlocking := *(*C.bool)(unsafe.Pointer(&nonBlocking))

	C.atm_to(m, cdevice, cdtype, cnonBlocking)
}

// int atm_get_profiling_mode();
func AtmGetProfilingMode() bool {
	retVal := C.atm_get_profiling_mode()
	return *(*bool)(unsafe.Pointer(&retVal))
}

// void atm_set_profiling_mode(int);
func AtmSetProfilingMode(b bool) {
	cbool := *(*C.int)(unsafe.Pointer(&b))
	C.atm_set_profiling_mode(cbool)
}

// void atm_eval(module);
func AtmEval(m Cmodule) {
	C.atm_eval(m)
}

// void atm_train(module);
func AtmTrain(m Cmodule) {
	C.atm_train(m)
}

func AtoConstantPadNd(ptr *Ctensor, self Ctensor, padData []int64, padLen int, value Cscalar) {
	cpadDataPtr := (*C.int64_t)(unsafe.Pointer(&padData[0]))
	cpadLen := *(*C.int)(unsafe.Pointer(&padLen))
	C.ato_constant_pad_nd(ptr, self, cpadDataPtr, cpadLen, value)
}

// // NOTE. TT. added to test new API generated
// func AtgRandn1(sizeData []int64, sizeLen int, optionsKind int32, optionsDevice int32) Ctensor {
// csizeDataPtr := (*C.int64_t)(unsafe.Pointer(&sizeData[0]))
// csizeLen := *(*C.int)(unsafe.Pointer(&sizeLen))
// coptionsKind := *(*C.int)(unsafe.Pointer(&optionsKind))
// coptionsDevice := *(*C.int)(unsafe.Pointer(&optionsDevice))
//
// return C.atg_randn1(csizeDataPtr, csizeLen, coptionsKind, coptionsDevice)
// }
