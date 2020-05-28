package tensor

import (
	"log"
	"reflect"
	"unsafe"
)

// CInt is equal to C type int. Go type is int32
type CInt = int32

// Kind is 'enum' like type. It represents different kind of elements
// that a Tensor can hold.
type Kind int

const (
	Uint8         Kind = iota // 0
	Int8                      // 1
	Int16                     // 2
	Int                       // 3
	Int64                     // 4
	Half                      // 5
	Float                     // 6
	Double                    // 7
	ComplexHalf               // 8
	ComplexFloat              // 9
	ComplexDouble             // 10
	Bool                      // 11
)

// ToCInt converts Kind to CInt type value which is `C int`
func (k Kind) ToCInt() CInt {
	return CInt(k)
}

// OfCInt converts a value of type CInt to Kind type value
func (k Kind) OfCInt(v CInt) Kind {
	switch v {
	case 0:
		return Uint8
	case 1:
		return Int8
	case 2:
		return Int16
	case 3:
		return Int
	case 4:
		return Int64
	case 5:
		return Half
	case 6:
		return Float
	case 7:
		return Double
	case 8:
		return ComplexHalf
	case 9:
		return ComplexFloat
	case 10:
		return ComplexDouble
	case 11:
		return Bool
	default:
		log.Fatalf("Unexpected kind %v\n", v)
	}
	return Kind(0)
}

// EltSizeInBytes converts a Kind value to number of bytes
// This is a ELement Size In Byte in Libtorch.
// Has it been deprecated?
func (k Kind) EltSizeInBytes() uint {
	switch {
	case k.ToCInt() == int32(Uint8):
		return 1
	case k.ToCInt() == int32(Int8):
		return 1
	case k.ToCInt() == int32(Int16):
		return 2
	case k.ToCInt() == int32(Int):
		return 4
	case k.ToCInt() == int32(Int64):
		return 8
	case k.ToCInt() == int32(Half):
		return 2
	case k.ToCInt() == int32(Float):
		return 4
	case k.ToCInt() == int32(Double):
		return 8
	case k.ToCInt() == int32(ComplexHalf):
		return 4
	case k.ToCInt() == int32(ComplexDouble):
		return 16
	case k.ToCInt() == int32(Bool):
		return 1
	default:
		log.Fatalf("Unreachable")
	}
	return uint(0)
}

// TODO: continue with devices...
