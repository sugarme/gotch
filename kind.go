package gotch

import (
	"log"
	"reflect"
)

// CInt is equal to C type int. Go type is int32
type CInt = int32

// Kind represents different kind of element that a tensor can hold.
// It has an embedded `reflect.Type` for type reflection.
type Kind struct {
	reflect.Type
}

// TODO: double check these Torch DType to Go type
var (
	Uint8         = Kind{reflect.TypeOf(uint8(1))}      // 0
	Int8          = Kind{reflect.TypeOf(int8(1))}       // 1
	Int16         = Kind{reflect.TypeOf(int16(1))}      // 2
	Int           = Kind{reflect.TypeOf(int(1))}        // 3
	Int64         = Kind{reflect.TypeOf(int64(1))}      // 4
	Half          = Kind{reflect.TypeOf(float32(1))}    // 5
	Float         = Kind{reflect.TypeOf(float64(1))}    // 6
	Double        = Kind{reflect.TypeOf(float64(1))}    // 7
	ComplexHalf   = kind{reflect.TypeOf(complex(1))}    // 8
	ComplexFloat  = Kind{reflect.TypeOf(complex64(1))}  // 9
	ComplexDouble = kind{reflect.TypeOf(complex128(1))} // 10
	Bool          = kind{reflect.TypeOf(true)}          // 11
)

// ToCInt converts Kind to CInt type value which is `C int`
func (k Kind) ToCInt() CInt {
	switch {
	case k.Kind() == uint8:
		return 0
	case k.Kind() == int8:
		return 1
	case k.Kind() == int16:
		return 2
	case k.Kind() == int:
		return 3
	case k.Kind() == int64:
		return 4
	case k.Kind() == float32:
		return 5
	default:
		log.Fatalf("Unsupported type.")
	}

	// unreachable
	return CInt(-1)
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
	return Kind{reflect.TypeOf(false)}
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

type KindDevice struct {
	Kind   Kind
	Device Device
}

var (
	FloatCPU  KindDevice = KindDevice{Float, CPU}
	DoubleCPU KindDevice = KindDevice{Double, CPU}
	Int64CPU  KindDevice = KindDevice{Int64, CPU}

	FloatCUDA  KindDevice = KindDevice{Float, CudaBuilder(0)}
	DoubleCUDA KindDevice = KindDevice{Double, CudaBuilder(0)}
	Int64CUDA  KindDevice = KindDevice{Int64, CudaBuilder(0)}
)
