package wrapper

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"unsafe"

	gotch "github.com/sugarme/gotch"
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

// DecodeTensor decodes the Tensor from the buffer to ptr using the format
// specified in c_api.h. Use stringDecoder for String tensors.
func DecodeTensor(r *bytes.Reader, shape []int64, typ reflect.Type, ptr reflect.Value) error {
	switch typ.Kind() {
	case reflect.Bool:
		b, err := r.ReadByte()
		if err != nil {
			return err
		}
		ptr.Elem().SetBool(b == 1)
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		if err := binary.Read(r, nativeEndian, ptr.Interface()); err != nil {
			return err
		}

	case reflect.Slice:
		val := reflect.Indirect(ptr)
		val.Set(reflect.MakeSlice(typ, int(shape[0]), int(shape[0])))

		// Optimization: if only one dimension is left we can use binary.Read() directly for this slice
		if len(shape) == 1 && val.Len() > 0 {
			switch val.Index(0).Kind() {
			case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
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

func numElements(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

// GetKind returns data type `Kind` (a element of tensor can hold)
// v - a value of a data element
func GetKind(v interface{}) (retVal gotch.Kind, err error) {

	switch {
	case reflect.TypeOf(v) == int:
		retVal = gotch.Int
	case reflect.TypeOf(v) == uint8:
		retVal = gotch.Uint8

	default:
		err = fmt.Errorf("Unsupported data type for %v\n", reflect.TypeOf(v))
		return retVal, err
	}

	return retVal, nil
}

// // TypeOf converts from a DType and Shape to the equivalent Go type.
// func TypeOf(dt DType, shape []int64) reflect.Type {
// var ret reflect.Type
// for _, t := range types {
// if dt == DType(t.dataType) {
// ret = t.typ
// break
// }
// }
// if ret == nil {
// // TODO get tensor name
// panic(fmt.Sprintf("Unsupported DType %d", int(dt)))
// }
// for range shape {
// ret = reflect.SliceOf(ret)
// }
// return ret
// }
