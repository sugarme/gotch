package main

//#include <stdlib.h>
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"reflect"
	"unsafe"

	t "github.com/sugarme/gotch/torch/libtch"
)

type Tensor struct {
	c_tensor *t.C_tensor
}

func FnOfSlice() (retVal Tensor, err error) {

	data := []int{1, 2, 3, 4, 5, 6}
	nflattened := len(data)
	dtype := 3          // Kind.Int
	eltSizeInBytes := 4 // Element Size in Byte for Int dtype

	nbytes := eltSizeInBytes * int(uintptr(nflattened))

	dataPtr := C.malloc(C.size_t(nbytes))

	// Recall: 1 << 30 = 1 * 2 * 30
	// Ref. See more at https://stackoverflow.com/questions/48756732
	dataSlice := (*[1 << 30]byte)(dataPtr)[:nbytes:nbytes]

	buf := bytes.NewBuffer(dataSlice[:0:nbytes])

	encodeTensor(buf, reflect.ValueOf(data), []int64{1})

	c_tensor := t.AtTensorOfData(dataPtr, int64(nflattened), 1, uint(eltSizeInBytes), int32(dtype))

	retVal = Tensor{c_tensor}

	return retVal, nil
}

func numElements(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func main() {

	t := t.NewTensor()

	fmt.Printf("Type of t: %v\n", reflect.TypeOf(t))

	res, err := FnOfSlice()
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println(res)
}

func encodeTensor(w *bytes.Buffer, v reflect.Value, shape []int64) error {
	switch v.Kind() {
	case reflect.Bool:
		b := byte(0)
		if v.Bool() {
			b = 1
		}
		if err := w.WriteByte(b); err != nil {
			return err
		}
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
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
			case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
				return binary.Write(w, nativeEndian, v.Interface())
			}
		}

		subShape := shape[1:]
		for i := 0; i < v.Len(); i++ {
			err := encodeTensor(w, v.Index(i), subShape)
			if err != nil {
				return err
			}
		}

	default:
		return fmt.Errorf("unsupported type %v", v.Type())
	}
	return nil
}

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
