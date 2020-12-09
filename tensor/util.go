package tensor

// #include <stdlib.h>
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	// "log"
	"reflect"
	"unsafe"

	gotch "github.com/sugarme/gotch"
)

// nativeEndian is a ByteOrder for local platform.
// Ref. https://stackoverflow.com/a/53286786
// Ref. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/tensor.go#L488-L505
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

// CMalloc allocates a given number of bytes to C side memory.
// It returns
// - dataPtr: a C pointer type of `*void` (`unsafe.Pointer` in Go).
// - buf : a Go pointer points to a given bytes of buffer (empty) in C memory
// allocated by C waiting for writing data to.
//
// NOTE:
// 1. Go pointer is a pointer to Go memory. C pointer is a pointer to C memory.
// 2. General rule is Go code can use C pointers. Go code may pass Go pointer to C
// provided that the Go memory to which it points does NOT contain any Go
// pointers. BUT C code must not store any Go pointers in Go memory, even
// temporarily.
// 3. Some Go values contain Go pointers IMPLICITLY: strings, slices, maps,
// channels and function values. Thus, pointers to these values should not be
// passed to C side. Instead, data should be allocated to C memory and return a
// C pointer to it using `C.malloc`.
// Ref: https://github.com/golang/proposal/blob/master/design/12416-cgo-pointers.md
func CMalloc(nbytes int) (dataPtr unsafe.Pointer, buf *bytes.Buffer) {

	dataPtr = C.malloc(C.size_t(nbytes))
	// NOTE: uncomment this cause panic!
	// defer C.free(unsafe.Pointer(dataPtr))

	// Recall: 1 << 30 = 1 * 2 * 30 = 1073741824
	dataSlice := (*[1 << 32]byte)(dataPtr)[:nbytes:nbytes] // 4294967296
	buf = bytes.NewBuffer(dataSlice[:0:nbytes])

	return dataPtr, buf
}

// EncodeTensor loads tensor data to C memory and returns a C pointer.
func EncodeTensor(w *bytes.Buffer, v reflect.Value, shape []int64) error {
	switch v.Kind() {
	case reflect.Bool:
		b := byte(0)
		if v.Bool() {
			b = 1
		}
		if err := w.WriteByte(b); err != nil {
			return err
		}
	case reflect.Uint8, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Float32, reflect.Float64:
		if err := binary.Write(w, nativeEndian, v.Interface()); err != nil {
			return err
		}

	case reflect.Array, reflect.Slice:
		// If current dimension is a slice, verify that it has the expected size
		// Go's type system makes that guarantee for arrays.
		if v.Kind() == reflect.Slice {
			expected := int(shape[0])
			if v.Len() != expected {
				return fmt.Errorf("mismatched slice lengths: %d and %d", v.Len(), expected)
			}
		}

		// Optimisation: if only one dimension is left we can use binary.Write() directly for this slice
		if len(shape) == 1 && v.Len() > 0 {
			switch v.Index(0).Kind() {
			case reflect.Uint8, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Float32, reflect.Float64:
				return binary.Write(w, nativeEndian, v.Interface())
			}
		}

		subShape := shape[1:]
		for i := 0; i < v.Len(); i++ {
			err := EncodeTensor(w, v.Index(i), subShape)
			if err != nil {
				return err
			}
		}

	default:
		return fmt.Errorf("unsupported type %v", v.Type())
	}
	return nil
}

// DecodeTensor decodes tensor value from a C memory buffer given
// C pointer, data type and shape and returns data value of type interface
func DecodeTensor(r *bytes.Reader, shape []int64, typ reflect.Type, ptr reflect.Value) error {
	switch typ.Kind() {
	case reflect.Bool:
		b, err := r.ReadByte()
		if err != nil {
			return err
		}
		ptr.Elem().SetBool(b == 1)
	case reflect.Uint8, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Float32, reflect.Float64:
		if err := binary.Read(r, nativeEndian, ptr.Interface()); err != nil {
			return err
		}

	case reflect.Slice:
		val := reflect.Indirect(ptr)
		val.Set(reflect.MakeSlice(typ, int(shape[0]), int(shape[0])))

		// Optimization: if only one dimension is left we can use binary.Read() directly for this slice
		if len(shape) == 1 && val.Len() > 0 {
			switch val.Index(0).Kind() {
			case reflect.Uint8, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Float32, reflect.Float64:
				return binary.Read(r, nativeEndian, val.Interface())
			}
		}

		for i := 0; i < val.Len(); i++ {
			if err := DecodeTensor(r, shape[1:], typ.Elem(), val.Index(i).Addr()); err != nil {
				return err
			}
		}

	default:
		return fmt.Errorf("unsupported type %v", typ)
	}
	return nil
}

// ElementCount counts number of element in the tensor given a shape
func ElementCount(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

// DataDim returns number of elements in data
// NOTE: only support scalar and (nested) slice/array of scalar type
func DataDim(data interface{}) (retVal int, err error) {

	_, count, err := dataCheck(reflect.ValueOf(data).Interface(), 0)

	return count, err
}

// DataCheck checks the input data for element Go type and number of elements.
// It will return errors if element type is not supported.
func DataCheck(data interface{}) (k reflect.Type, n int, err error) {

	return dataCheck(reflect.ValueOf(data).Interface(), 0)
}

// NOTE: 0 is reflect.Kind() of Invalid
// See: https://golang.org/pkg/reflect/#Kind
func dataCheck(data interface{}, count int) (k reflect.Type, n int, err error) {
	v := reflect.ValueOf(data)
	var goType reflect.Type = reflect.TypeOf(data)
	var total int = count
	var round = 0

	switch v.Kind() {
	case reflect.Slice, reflect.Array:
		if round == 0 {
			round = v.Len()
		}
		for i := 0; i < v.Len(); i++ {
			round--
			goType, total, err = dataCheck(v.Index(i).Interface(), total)

			if err != nil {
				return reflect.TypeOf(reflect.Zero), 0, err
			}
		}

		return goType, total, nil

	case reflect.Uint8, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Float32, reflect.Float64, reflect.Bool:
		total++
		if goType.String() != "invalid" {
			goType = v.Type()
		}
	default:
		err = fmt.Errorf("Input Data: unsupported data structure or type: %v\n", v.Kind())
		return reflect.TypeOf(reflect.Zero), 0, err
	}

	return goType, total, nil
}

// DataAsPtr write to C memory and returns a C pointer.
//
// NOTE:
// Supported data types are scalar, slice/array of scalar type equivalent to
// DType.
func DataAsPtr(data interface{}) (dataPtr unsafe.Pointer, err error) {

	// 1. Count number of elements in data
	elementNum, err := DataDim(data)
	if err != nil {
		return nil, err
	}

	// 2. Element size in bytes
	dtype, err := gotch.DTypeFromData(data)
	if err != nil {
		return nil, err
	}

	eltSizeInBytes, err := gotch.DTypeSize(dtype)
	if err != nil {
		return nil, err
	}

	nbytes := int(eltSizeInBytes) * int(elementNum)

	// 3. Get C pointer and prepare C memory buffer for writing
	dataPtr, buff := CMalloc(nbytes)

	// 4. Write data to C memory
	// NOTE: data should be **fixed size** values so that binary.Write can work
	// A fixed-size value is either a fixed-size arithmetic type (bool, int8, uint8,
	// int16, float32, complex64, ...) or an array or struct containing only fixed-size values.
	// See more: https://golang.org/pkg/encoding/binary/
	// Therefore, we will need to flatten data to `[]T`
	fData, err := FlattenData(data)
	if err != nil {
		return nil, err
	}

	err = binary.Write(buff, nativeEndian, fData)
	if err != nil {
		return nil, err
	}

	return dataPtr, nil
}

// FlattenDim counts number of elements with given shape
func FlattenDim(shape []int64) int {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}

	return int(n)
}

// FlattenData flattens data to 1D array ([]T)
func FlattenData(data interface{}) (fData interface{}, err error) {

	// If data is 1D already, just return it.
	dataVal := reflect.ValueOf(data)
	dataTyp := reflect.TypeOf(data)
	if dataVal.Kind() == reflect.Slice {
		eleVal := dataTyp.Elem()
		if eleVal.Kind() != reflect.Slice {
			return data, nil
		}
	}

	flat, err := flattenData(reflect.ValueOf(data).Interface(), 0, []interface{}{})
	if err != nil {
		return nil, err
	}

	ele := flat[0]

	// Boring task. Convert interface to specific type.
	// Any good way to do???
	switch reflect.ValueOf(ele).Kind() {
	case reflect.Uint8:
		var retVal []uint8
		for _, v := range flat {
			retVal = append(retVal, v.(uint8))
		}
		return retVal, nil
	case reflect.Int8:
		var retVal []int8
		for _, v := range flat {
			retVal = append(retVal, v.(int8))
		}
		return retVal, nil
	case reflect.Int16:
		var retVal []int16
		for _, v := range flat {
			retVal = append(retVal, v.(int16))
		}
		return retVal, nil
	case reflect.Int32:
		var retVal []int32
		for _, v := range flat {
			retVal = append(retVal, v.(int32))
		}
		return retVal, nil
	case reflect.Int64:
		var retVal []int64
		for _, v := range flat {
			retVal = append(retVal, v.(int64))
		}
		return retVal, nil
	case reflect.Float32:
		var retVal []float32
		for _, v := range flat {
			retVal = append(retVal, v.(float32))
		}
		return retVal, nil
	case reflect.Float64:
		var retVal []float64
		for _, v := range flat {
			retVal = append(retVal, v.(float64))
		}
		return retVal, nil
	case reflect.Bool:
		var retVal []bool
		for _, v := range flat {
			retVal = append(retVal, v.(bool))
		}
		return retVal, nil

	default:
		err = fmt.Errorf("Unsupport type for input data: %v\n", reflect.ValueOf(ele).Kind())
		return nil, err
	}

	return nil, err

}

func flattenData(data interface{}, round int, flat []interface{}) (f []interface{}, err error) {
	v := reflect.ValueOf(data)
	var flatData []interface{} = flat

	switch v.Kind() {
	case reflect.Slice, reflect.Array:
		if round == 0 {
			round = v.Len()
		}
		for i := 0; i < v.Len(); i++ {
			round--
			flatData, err = flattenData(v.Index(i).Interface(), round, flatData)
			if err != nil {
				return nil, err
			}
		}

		return flatData, nil

	case reflect.Uint8, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Float32, reflect.Float64, reflect.Bool:
		flatData = append(flatData, data)
	}

	return flatData, nil
}

// InvokeFn reflects and invokes a function of interface type.
func InvokeFnWithArgs(fn interface{}, args ...string) {
	v := reflect.ValueOf(fn)
	rargs := make([]reflect.Value, len(args))
	for i, a := range args {
		rargs[i] = reflect.ValueOf(a)
	}
	v.Call(rargs)
}

// Func struct contains information of a function
type FuncInfo struct {
	Signature  string
	InArgs     []reflect.Value
	OutArgs    []reflect.Value
	IsVariadic bool
}

type Func struct {
	typ  reflect.Type
	val  reflect.Value
	meta FuncInfo
}

func NewFunc(fn interface{}) (retVal Func, err error) {
	meta, err := getFuncInfo(fn)
	if err != nil {
		return retVal, err
	}

	retVal = Func{
		typ:  reflect.TypeOf(fn),
		val:  reflect.ValueOf(fn),
		meta: meta,
	}
	return retVal, nil
}

// getFuncInfo analyzes input of interface type and returns function information
// in FuncInfo struct. It returns error if input is not a function type under
// the hood.
func getFuncInfo(fn interface{}) (retVal FuncInfo, err error) {
	fnVal := reflect.ValueOf(fn)
	fnTyp := reflect.TypeOf(fn)

	// First, check whether input is a function type
	if fnVal.Kind() != reflect.Func {
		err = fmt.Errorf("Input is not a function.")
		return retVal, err
	}

	// get number of input and output arguments of function
	numIn := fnTyp.NumIn()           // inbound parameters
	numOut := fnTyp.NumOut()         // outbound parameters
	isVariadic := fnTyp.IsVariadic() // whether function is a variadic func
	fnSig := fnTyp.String()          // function signature

	// get input and ouput arguments values (reflect.Value type)
	var inArgs []reflect.Value
	var outArgs []reflect.Value

	for i := 0; i < numIn; i++ {
		t := fnTyp.In(i) // reflect.Type

		inArgs = append(inArgs, reflect.ValueOf(t))
	}

	for i := 0; i < numOut; i++ {
		t := fnTyp.Out(i) // reflect.Type
		outArgs = append(outArgs, reflect.ValueOf(t))
	}

	retVal = FuncInfo{
		Signature:  fnSig,
		InArgs:     inArgs,
		OutArgs:    outArgs,
		IsVariadic: isVariadic,
	}

	return retVal, nil
}

// Info analyzes input of interface type and returns function information
// in FuncInfo struct. It returns error if input is not a function type under
// the hood. It will be panic if input is not a function
func (f *Func) Info() (retVal FuncInfo) {
	return f.meta
}

func (f *Func) Invoke() interface{} {
	// call function with input parameters
	// TODO: return vals are []reflect.Value
	// How do we match them to output order of signature function
	return f.val.Call(f.meta.InArgs)
}

// Must is a helper to unwrap function it wraps. If having error,
// it will cause panic.
func Must(ts Tensor, err error) (retVal Tensor) {
	if err != nil {
		panic(err)
	}
	return ts
}
