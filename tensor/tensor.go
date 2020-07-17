package tensor

//#include "stdlib.h"
//#include "stdbool.h"
//#include<stdio.h>
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"reflect"
	"unsafe"

	gotch "github.com/sugarme/gotch"
	lib "github.com/sugarme/gotch/libtch"
)

type Tensor struct {
	ctensor lib.Ctensor
}

// NewTensor creates a new tensor
func NewTensor() Tensor {
	ctensor := lib.AtNewTensor()
	return Tensor{ctensor}
}

func (ts Tensor) Dim() uint64 {
	retVal := lib.AtDim(ts.ctensor)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
	return retVal
}

// Size return shape of the tensor
//
// NOTE: C++ libtorch calls at_shape() -> t.sizes()
// And returns a slice of sizes or shape using given pointer
// to that slice.
func (ts Tensor) Size() (retVal []int64, err error) {
	dim := lib.AtDim(ts.ctensor)
	sz := make([]int64, dim)
	szPtr, err := DataAsPtr(sz)
	if err != nil {
		return retVal, err
	}
	defer C.free(unsafe.Pointer(szPtr))

	lib.AtShape(ts.ctensor, szPtr)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = decodeSize(szPtr, dim)
	return retVal, nil
}

func (ts Tensor) MustSize() (retVal []int64) {
	retVal, err := ts.Size()
	if err != nil {
		log.Fatal(err)
	}
	return retVal
}

// Size1 returns the tensor size for 1D tensors.
func (ts Tensor) Size1() (retVal int64, err error) {
	shape, err := ts.Size()
	if err != nil {
		return retVal, err
	}

	if len(shape) != 1 {
		err = fmt.Errorf("Expected one dim, got %v\n", len(shape))
		return retVal, err
	}

	return shape[0], nil
}

// Size2 returns the tensor size for 2D tensors.
func (ts Tensor) Size2() (retVal []int64, err error) {
	shape, err := ts.Size()
	if err != nil {
		return retVal, err
	}

	if len(shape) != 2 {
		err = fmt.Errorf("Expected two dims, got %v\n", len(shape))
		return retVal, err
	}

	return shape, nil
}

// Size3 returns the tensor size for 3D tensors.
func (ts Tensor) Size3() (retVal []int64, err error) {
	shape, err := ts.Size()
	if err != nil {
		return retVal, err
	}

	if len(shape) != 3 {
		err = fmt.Errorf("Expected three dims, got %v\n", len(shape))
		return retVal, err
	}

	return shape, nil
}

// Size4 returns the tensor size for 4D tensors.
func (ts Tensor) Size4() (retVal []int64, err error) {
	shape, err := ts.Size()
	if err != nil {
		return retVal, err
	}

	if len(shape) != 4 {
		err = fmt.Errorf("Expected four dims, got %v\n", len(shape))
		return retVal, err
	}

	return shape, nil
}

func decodeSize(ptr unsafe.Pointer, nsize uint64) []int64 {
	// Decode sz
	// 1. Count number of elements in data
	elementNum := nsize
	// 2. Element size in bytes
	eltSizeInBytes, err := gotch.DTypeSize(gotch.Int64)
	if err != nil {
		log.Fatal(err)
	}
	nbytes := int(eltSizeInBytes) * int(elementNum)
	dataSlice := (*[1 << 30]byte)(ptr)[:nbytes:nbytes]
	r := bytes.NewReader(dataSlice)
	dataIn := make([]int64, nsize)
	if err := binary.Read(r, nativeEndian, dataIn); err != nil {
		log.Fatal(err)
	}

	return dataIn
}

// OfSlice creates tensor from a slice data
func OfSlice(data interface{}) (retVal Tensor, err error) {

	typ, dataLen, err := DataCheck(data)
	if err != nil {
		return retVal, err
	}

	dtype, err := gotch.ToDType(typ)
	if err != nil {
		return retVal, err
	}

	shape := []int64{int64(dataLen)}
	elementNum := ElementCount(shape)

	eltSizeInBytes, err := gotch.DTypeSize(dtype)
	if err != nil {
		return retVal, err
	}

	nbytes := int(eltSizeInBytes) * int(elementNum)

	dataPtr, buff := CMalloc(nbytes)
	defer C.free(unsafe.Pointer(dataPtr))

	if err = EncodeTensor(buff, reflect.ValueOf(data), shape); err != nil {
		return retVal, err
	}

	cint, err := gotch.DType2CInt(dtype)
	if err != nil {
		return retVal, err
	}

	ctensor := lib.AtTensorOfData(dataPtr, shape, uint(len(shape)), uint(eltSizeInBytes), int(cint))
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor}

	return retVal, nil
}

// MustOfSlice create a tensor from slice of data. It will be panic if error.
func MustOfSlice(data interface{}) (retVal Tensor) {
	retVal, err := OfSlice(data)
	if err != nil {
		log.Fatal(err)
	}

	return retVal

}

// TensorFrom create a tensor from slice of data. It will be panic if error.
func TensorFrom(data interface{}) (retVal Tensor) {
	retVal, err := OfSlice(data)
	if err != nil {
		log.Fatal(err)
	}
	return retVal
}

// Print prints tensor values to console.
//
// NOTE: it is printed from C and will print ALL elements of tensor
// with no truncation at all.
func (ts Tensor) Print() {
	lib.AtPrint(ts.ctensor)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

// NewTensorFromData creates tensor from given data and shape
func NewTensorFromData(data interface{}, shape []int64) (retVal Tensor, err error) {
	// 1. Check whether data and shape match
	elementNum, err := DataDim(data)
	if err != nil {
		return retVal, err
	}

	nflattend := FlattenDim(shape)

	if elementNum != nflattend {
		err = fmt.Errorf("Number of data elements (%v) and flatten shape (%v) dimension mismatched.\n", elementNum, nflattend)
		return retVal, err
	}

	// 2. Write raw data to C memory and get C pointer
	dataPtr, err := DataAsPtr(data)
	defer C.free(unsafe.Pointer(dataPtr))
	if err != nil {
		return retVal, err
	}

	// 3. Create tensor with pointer and shape
	dtype, err := gotch.DTypeFromData(data)
	if err != nil {
		return retVal, err
	}

	eltSizeInBytes, err := gotch.DTypeSize(dtype)
	if err != nil {
		return retVal, err
	}

	cint, err := gotch.DType2CInt(dtype)
	if err != nil {
		return retVal, err
	}

	ctensor := lib.AtTensorOfData(dataPtr, shape, uint(len(shape)), uint(eltSizeInBytes), int(cint))
	// defer C.free(unsafe.Pointer(ctensor))
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor}

	return retVal, nil

}

func (ts Tensor) DType() gotch.DType {
	cint := lib.AtScalarType(ts.ctensor)

	dtype, err := gotch.CInt2DType(cint)
	if err != nil {
		log.Fatalf("Tensor DType error: %v\n", err)
	}

	return dtype
}

func (ts Tensor) Device() (retVal gotch.Device, err error) {
	cInt := lib.AtDevice(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	var device gotch.Device

	return device.OfCInt(int32(cInt)), nil
}

func (ts Tensor) Eq1(other Tensor, del bool) (retVal Tensor, err error) {

	// Get a C null pointer
	// https://stackoverflow.com/a/2022369
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	lib.AtgEq1(ptr, ts.ctensor, other.ctensor)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return Tensor{ctensor: *ptr}, nil

}

func (ts Tensor) MustEq1(other Tensor, del bool) (retVal Tensor) {
	retVal, err := ts.Eq1(other, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// Float64Value returns a float value on tensors holding a single element.
// An error is returned otherwise.
// double at_double_value_at_indexes(tensor, int64_t *indexes, int indexes_len);
func (ts Tensor) Float64Value(idx []int64) (retVal float64, err error) {

	idxPtr, err := DataAsPtr(idx)
	if err != nil {
		return retVal, err
	}
	defer C.free(unsafe.Pointer(idxPtr))

	retVal = lib.AtDoubleValueAtIndexes(ts.ctensor, idxPtr, len(idx))
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, err
}

func (ts Tensor) MustFloat64Value(idx []int64) (retVal float64) {
	retVal, err := ts.Float64Value(idx)
	if err != nil {
		log.Fatal(err)
	}
	return retVal
}

// Int64Value returns an int value on tensors holding a single element. An error is
// returned otherwise.
func (ts Tensor) Int64Value(idx []int64) (retVal int64, err error) {

	idxPtr, err := DataAsPtr(idx)
	if err != nil {
		return retVal, err
	}
	defer C.free(unsafe.Pointer(idxPtr))

	retVal = lib.AtInt64ValueAtIndexes(ts.ctensor, idxPtr, len(idx))
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, err
}

func (ts Tensor) MustInt64Value(idx []int64) (retVal int64) {
	retVal, err := ts.Int64Value(idx)
	if err != nil {
		log.Fatal(err)
	}
	return retVal
}

// RequiresGrad returns true if gradient are currently tracked for this tensor.
func (ts Tensor) RequiresGrad() (retVal bool, err error) {
	retVal = lib.AtRequiresGrad(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, nil
}

func (ts Tensor) MustRequiresGrad() (retVal bool) {
	retVal, err := ts.RequiresGrad()
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// DataPtr returns the address of the first element of this tensor.
func (ts Tensor) DataPtr() (retVal unsafe.Pointer, err error) {

	retVal = lib.AtDataPtr(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, nil
}

// Defined returns true is the tensor is defined.
func (ts Tensor) Defined() (retVal bool, err error) {
	retVal = lib.AtDefined(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, nil
}

func (ts Tensor) MustDefined() (retVal bool) {
	retVal, err := ts.Defined()
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// IsSparse returns true is the tensor is spare.
func (ts Tensor) IsSparse() (retVal bool, err error) {
	retVal = lib.AtIsSparse(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, nil
}

// ZeroGrad zeroes the gradient tensor attached to this tensor if defined.
func (ts Tensor) ZeroGrad() {
	grad := ts.MustGrad()
	if grad.MustDefined() {
		grad.Detach_()
		grad.Zero_()
	}
}

// Backward runs the backward pass, populating the gradient tensors for tensors
// which gradients are tracked.
//
// Gradients tracking can be turned on via `SetRequiresGrad`.
func (ts Tensor) Backward() (err error) {
	lib.AtBackward(ts.ctensor, 0, 0)
	if err = TorchErr(); err != nil {
		return err
	}

	return nil
}

func (ts Tensor) MustBackward() {
	if err := ts.Backward(); err != nil {
		log.Fatal(err)
	}
}

// RunBackward runs the backward ...
func RunBackward(tensors []Tensor, inputs []Tensor, keepGraphB bool, createGraphB bool) (retVal []Tensor, err error) {
	// NOTE: outputs is a slice of tensors with length = len(inputs)
	var outputsPtr []*lib.Ctensor
	// TODO: Are they allocated continouslly???
	for i := 0; i < len(inputs); i++ {
		outputPtr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
		defer C.free(unsafe.Pointer(outputPtr))
		outputsPtr = append(outputsPtr, outputPtr)
	}

	// Get first element pointer
	ctensor := tensors[0].ctensor
	cinput := inputs[0].ctensor
	tensorsPtr := (*lib.Ctensor)(unsafe.Pointer(&ctensor))
	inputsPtr := (*lib.Ctensor)(unsafe.Pointer(&cinput))
	var keepGraph int = 0
	if keepGraphB {
		keepGraph = 1
	}
	var createGraph int = 0
	if createGraphB {
		createGraph = 1
	}

	lib.AtRunBackward(tensorsPtr, len(tensors), inputsPtr, len(inputs), outputsPtr[0], keepGraph, createGraph)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	for i := 0; i < len(inputs); i++ {
		outputPtr := outputsPtr[i]
		retVal = append(retVal, Tensor{ctensor: *outputPtr})
	}

	return retVal, nil
}

// CopyDataUint8 copies `numel` elements from `self` to `dst`.
//
// NOTE: `dst` located in Go memory. Should it be?
func (ts Tensor) CopyDataUint8(dst []uint8, numel uint) (err error) {

	// NOTE: we must make sure that `dst` has same len as `numel`. Otherwise,
	// there will be memory leak and or out of range error.
	if len(dst) < int(numel) {
		err = fmt.Errorf("CopyDataUint8 Error: length of destination slice data (%v) is smaller than \nnumber of elements to be copied (%v)", len(dst), numel)
		return err
	}

	vs := unsafe.Pointer(&dst[0])
	elt_size_in_bytes, err := gotch.DTypeSize(gotch.Uint8)
	if err != nil {
		return err
	}
	lib.AtCopyData(ts.ctensor, vs, numel, elt_size_in_bytes)
	if err = TorchErr(); err != nil {
		return err
	}

	return nil
}

func (ts Tensor) MustCopyDataUint8(dst []uint8, numel uint) {
	err := ts.CopyDataUint8(dst, numel)
	if err != nil {
		log.Fatal(err)
	}
}

// CopyData copies `numel` elements from `self` to `dst`.
// `dst` should be a slice of Go type equivalent to tensor type.
//
// NOTE: `dst` located in Go memory. Should it be?
// We will render Go pointer of first element of `dst` slice
// and number of elements to C land. This may break in the future
// if Go policy changes.
func (ts Tensor) CopyData(dst interface{}, numel uint) (err error) {

	gotype, dlen, err := DataCheck(dst)
	if err != nil {
		return err
	}

	dtype, err := gotch.ToDType(gotype)
	if err != nil {
		return err
	}

	if dlen < int(numel) {
		err = fmt.Errorf("CopyData Error: length of destination slice data (%v) is smaller than \nnumber of elements to be copied (%v)", dlen, numel)
		return err
	}

	if ts.DType() != dtype {
		err = fmt.Errorf("Type mismatched: `dst` type: %v, tensor DType: %v", dtype, ts.DType())
		return err
	}

	var vs unsafe.Pointer
	switch dtype {
	case gotch.Uint8:
		vs = unsafe.Pointer(&dst.([]uint8)[0])
	case gotch.Int8:
		vs = unsafe.Pointer(&dst.([]int8)[0])
	case gotch.Int16:
		vs = unsafe.Pointer(&dst.([]int16)[0])
	case gotch.Int:
		vs = unsafe.Pointer(&dst.([]int32)[0])
	case gotch.Int64:
		vs = unsafe.Pointer(&dst.([]int64)[0])
	case gotch.Float:
		vs = unsafe.Pointer(&dst.([]float32)[0])
	case gotch.Double:
		vs = unsafe.Pointer(&dst.([]float64)[0])
	case gotch.Bool:
		vs = unsafe.Pointer(&dst.([]bool)[0])
	default:
		err = fmt.Errorf("Unsupported type: `dst` type: %v, tensor DType: %v", dtype, ts.DType())
		return err
	}

	elt_size_in_bytes, err := gotch.DTypeSize(dtype)
	if err != nil {
		return err
	}
	lib.AtCopyData(ts.ctensor, vs, numel, elt_size_in_bytes)
	if err = TorchErr(); err != nil {
		return err
	}

	return nil
}

// MustCopyData copies number of elements from tensor to a slice of data
//
// NOTE: `dst` is a slice with length = numel and Go type equavalent to tensor
// DType
func (ts Tensor) MustCopyData(dst interface{}, numel uint) {
	err := ts.CopyData(dst, numel)
	if err != nil {
		log.Fatal(err)
	}
}

// Numel returns the total number of elements stored in a tensor.
func (ts Tensor) Numel() (retVal uint) {
	var shape []int64
	shape = ts.MustSize()
	return uint(FlattenDim(shape))
}

// ShallowCopy returns a new tensor that share storage with the input tensor.
func (ts Tensor) ShallowClone() (retVal Tensor, err error) {

	ctensor := lib.AtShallowClone(ts.ctensor)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor}

	return retVal, nil
}

// MustShallowClone returns a new tensor that share storage with the input
// tensor. It will panic if error occurred
func (ts Tensor) MustShallowClone() (retVal Tensor) {
	retVal, err := ts.ShallowClone()
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// Get gets the sub-tensor at the given index.
func (ts Tensor) Get(index int) (retVal Tensor, err error) {

	ctensor := lib.AtGet(ts.ctensor, index)
	if err = TorchErr(); err != nil {
		return retVal, err
	}
	retVal = Tensor{ctensor}

	return retVal, nil
}

// MustGet gets the sub-tensor at the given index. It will panic if error
// occurred.
func (ts Tensor) MustGet(index int) (retVal Tensor) {
	retVal, err := ts.Get(index)
	if err != nil {
		log.Fatal(err)
	}
	return retVal
}

// Copy_ copies in-place values from the argument tensor to the input tensor.
func Copy_(self, src Tensor) {
	var err error
	lib.AtCopy_(self.ctensor, src.ctensor)

	if err = TorchErr(); err != nil {
		log.Fatal(err)
	}

}

// Copy_ copies in-place values from the argument tensor to existing tensor
func (ts Tensor) Copy_(src Tensor) {
	var err error
	lib.AtCopy_(ts.ctensor, src.ctensor)

	if err = TorchErr(); err != nil {
		log.Fatal(err)
	}
}

// Save saves a tensor to a file.
func (ts Tensor) Save(path string) (err error) {
	lib.AtSave(ts.ctensor, path)

	if err = TorchErr(); err != nil {
		return err
	}

	return nil
}

// MustSave saves a tensor to a file. It will panic if error
func (ts Tensor) MustSave(path string) {
	if err := ts.Save(path); err != nil {
		log.Fatal(err)
	}
}

// Load loads a tensor from a file.
func Load(path string) (retVal Tensor, err error) {
	ctensor := lib.AtLoad(path)

	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor}

	return retVal, nil
}

// MustLoad loads a tensor to a file. It will panic if error
func MustLoad(path string) (retVal Tensor) {
	retVal, err := Load(path)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

type NamedTensor struct {
	Name   string
	Tensor Tensor
}

// SaveMulti saves some named tensors to a file
//
// The file format is the same as the one used by the PyTorch C++ API.
func SaveMulti(namedTensors []NamedTensor, path string) (err error) {
	var ctensors []lib.Ctensor
	var names []string

	for _, ts := range namedTensors {
		ctensors = append(ctensors, ts.Tensor.ctensor)
		names = append(names, ts.Name)
	}

	lib.AtSaveMulti(ctensors, names, len(namedTensors), path)
	if err = TorchErr(); err != nil {
		return err
	}

	return nil
}

// MustSaveMulti saves some named tensors to a file. It will panic if error
func MustSaveMulti(namedTensors []NamedTensor, path string) {
	err := SaveMulti(namedTensors, path)
	if err != nil {
		log.Fatal(err)
	}
}

// LoadMulti loads some named tensors from a file
//
// The file format is the same as the one used by the PyTorch C++ API.
func LoadMulti(path string) (retVal []NamedTensor, err error) {

	var data lib.LoadData
	dataPtr := lib.PStore.Set(&data)
	lib.AtLoadCallback(path, dataPtr)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	for _, v := range data.NamedCtensors {
		namedTensor := NamedTensor{
			Name:   v.Name,
			Tensor: Tensor{v.Ctensor},
		}

		retVal = append(retVal, namedTensor)
	}

	return retVal, nil
}

// MustLoadMulti loads some named tensors from a file. It will panic if error
func MustLoadMulti(path string) (retVal []NamedTensor) {
	retVal, err := LoadMulti(path)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// LoadMultiWithDevice loads some named tensors from a file to a given device
//
// The file format is the same as the one used by the PyTorch C++ API.
func LoadMultiWithDevice(path string, device gotch.Device) (retVal []NamedTensor, err error) {
	var data lib.LoadData
	dataPtr := lib.PStore.Set(&data)

	lib.AtLoadCallbackWithDevice(path, dataPtr, device.CInt())
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	for _, v := range data.NamedCtensors {
		namedTensor := NamedTensor{
			Name:   v.Name,
			Tensor: Tensor{v.Ctensor},
		}

		retVal = append(retVal, namedTensor)
	}

	return retVal, nil
}

// MustLoadMulti loads some named tensors from a file. It will panic if error
func MustLoadMultiWithDevice(path string, device gotch.Device) (retVal []NamedTensor) {
	retVal, err := LoadMultiWithDevice(path, device)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// ToString returns a string representation for the tensor.
//
// lw : line width (size)
// NOTE: The representation will contain all the tensor element hence may be huge for
// large tensors.
func (ts Tensor) ToString(lw int64) (retVal string, err error) {
	retVal = lib.AtToString(ts.ctensor, lw)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	return retVal, nil
}

// MustToString returns a string representation for the tensor. It will be panic
// if error.
// lw : line width (size)
func (ts Tensor) MustToString(lw int64) (retVal string) {
	retVal, err := ts.ToString(lw)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// Drop drops (frees) the tensor
func (ts Tensor) Drop() (err error) {
	lib.AtFree(ts.ctensor)
	if err = TorchErr(); err != nil {
		return err
	}

	return nil
}

// MustDrop drops the tensor. It will be panic if error
func (ts Tensor) MustDrop() {
	if err := ts.Drop(); err != nil {
		log.Fatal(err)
	}
}

// GradSetEnabled sets globally whether GradMode gradient accumulation is enable or not.
// It returns PREVIOUS state of Grad before setting.
func GradSetEnabled(b bool) (retVal bool, err error) {

	var cbool, cretVal int
	switch b {
	case true:
		cbool = 1
	case false:
		cbool = 0
	}

	cretVal = lib.AtGradSetEnabled(cbool)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	switch cretVal {
	case 0:
		retVal = false
		break
	case 1:
		retVal = true
		break
		// case -1: // should be unreachable as error is captured above with TorchrErr()
		// err = fmt.Errorf("Cannot set grad enable. \n")
		// return retVal, err
		// default: // should be unreachable as error is captured above with TorchrErr()
		// err = fmt.Errorf("Cannot set grad enable. \n")
		// return retVal, err
	}

	return retVal, nil
}

// MustGradSetEnabled sets globally whether GradMode gradient accumuation is enable or not.
// It returns PREVIOUS state of Grad before setting. It will be panic if error
func MustGradSetEnabled(b bool) (retVal bool) {
	retVal, err := GradSetEnabled(b)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// NoGrad runs a closure without keeping track of gradients.
func NoGrad(fn interface{}) {

	// TODO: This is weird but somehow we need to trigger C++ print
	// to get loss function updated. Probably it is related to
	// C++ cache clearing.
	// Next step would be creating a Go func that trigger C++ cache clean
	// instead of this ugly hacky way.
	newTs := NewTensor()
	newTs.Drop()

	// Switch off Grad
	prev := MustGradSetEnabled(false)

	// Analyze input as function. If not, throw error
	f, err := NewFunc(fn)
	if err != nil {
		log.Fatal(err)
	}

	// invokes the function
	f.Invoke()

	// Switch on Grad
	_ = MustGradSetEnabled(prev)

}

func NoGrad1(fn func() interface{}) (retVal interface{}) {
	newTs := NewTensor()
	newTs.Drop()

	// Switch off Grad
	prev := MustGradSetEnabled(false)

	retVal = fn()

	// Switch on Grad
	_ = MustGradSetEnabled(prev)

	return retVal
}

// NoGradGuard is a RAII guard that prevents gradient tracking until deallocated.
// It actually sets a global flag that is checked by the backend whenever an op is done on a variable.
// The guard itself saved the current status and set it to false in the constructor.
// And restore the saved status in it’s destructor.
// That way it is similar to a with torch.no_grad(): block in python.
// Ref. https://discuss.pytorch.org/t/how-does-nogradguard-works-in-cpp/34960/2
//
// TODO: should we implement Go `mutex` here???
type NoGradGuard struct {
	enabled bool
}

// Init NoGradGuard and disables gradient tracking
func NewNoGradGuard() NoGradGuard {
	return noGradGuardInit()
}

// Disables gradient tracking, this will be enabled back when the
// returned value gets deallocated.
func noGradGuardInit() NoGradGuard {
	return NoGradGuard{enabled: MustGradSetEnabled(false)}
}

// Drop drops the NoGradGuard state.
func (ngg *NoGradGuard) Drop() {
	ngg.enabled = true
	_ = MustGradSetEnabled(ngg.enabled)
}

func (ngg *NoGradGuard) Enable() {
	ngg.enabled = false
	_ = MustGradSetEnabled(ngg.enabled)
}

// Reduction type is an enum-like type
type Reduction int

const (
	// Do not reduce
	ReductionNone Reduction = iota
	// Mean of losses
	ReductionMean
	// Sum of losses
	ReductionSum
	// Escape hatch in case new options become available
	ReductionOther
)

func (r Reduction) ToInt() (retVal int) {
	switch r {
	case ReductionNone:
		return 0
	case ReductionMean:
		return 1
	case ReductionSum:
		return 2
	case ReductionOther:
		return 3
	}
	return
}

// Values returns values of tensor in a slice of float64.
func (ts Tensor) Values() []float64 {
	numel := ts.Numel()
	vec := make([]float64, numel)

	float64Ts := ts.MustTotype(gotch.Double, false)

	float64Ts.MustCopyData(vec, numel)
	float64Ts.MustDrop()

	return vec
}

// Vals returns tensor values in a slice
// NOTE: need a type insersion to get runtime type
// E.g. res := xs.Vals().([]int64)
func (ts Tensor) Vals() (retVal interface{}) {
	dtype := ts.DType()
	numel := ts.Numel()

	switch dtype.Name() {
	case "uint8":
		retVal = make([]uint8, numel)
	case "int8":
		retVal = make([]int8, numel)
	case "int16":
		retVal = make([]int16, numel)
	case "int32":
		retVal = make([]int32, numel)
	case "int64":
		retVal = make([]int64, numel)
	case "float32":
		retVal = make([]float32, numel)
	case "float64":
		retVal = make([]float64, numel)
	case "bool":
		retVal = make([]bool, numel)
	default:
		log.Fatalf("Unsupported dtype (%v)", dtype)
	}

	ts.CopyData(retVal, numel)
	return retVal
}

// FlatView flattens a tensor.
//
// This returns a flattened version of the given tensor. The first dimension
// is preserved as it is assumed to be the mini-batch dimension.
func (ts Tensor) FlatView() (retVal Tensor) {
	batchSize := ts.MustSize()[0]
	return ts.MustView([]int64{batchSize, -1}, false)
}

func (ts Tensor) ZeroPad2d(left, right, top, bottom int64, del bool) (retVal Tensor, err error) {
	if ts.Dim() != 4 {
		err = fmt.Errorf("Expected a 4 dimension tensor, got %v\n", ts.MustSize())
		return retVal, err
	}

	return ts.ConstantPadNd([]int64{left, right, top, bottom}, del)
}

func (ts Tensor) MustZeroPad2d(left, right, top, bottom int64, del bool) (retVal Tensor) {
	retVal, err := ts.ZeroPad2d(left, right, top, bottom, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (ts Tensor) Swish() (retVal Tensor) {
	sig := ts.MustSigmoid(false)
	retVal = ts.MustMul(sig, false)
	sig.MustDrop()
	return retVal
}

func (ts Tensor) AvgPool2DDefault(ksize int64, del bool) (retVal Tensor) {
	return ts.MustAvgPool2D([]int64{ksize, ksize}, []int64{ksize, ksize}, []int64{0, 0}, false, true, 1, del)
}
