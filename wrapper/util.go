package wrapper

// #include <stdlib.h>
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"reflect"
	"unsafe"
	// gotch "github.com/sugarme/gotch"
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

	// Recall: 1 << 30 = 1 * 2 * 30
	// Ref. See more at https://stackoverflow.com/questions/48756732
	dataSlice := (*[1 << 30]byte)(dataPtr)[:nbytes:nbytes]
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
