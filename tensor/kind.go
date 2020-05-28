package tensor

import (
	"log"

	gotch "github.com/sugarme/gotch"
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

type KindDevice struct {
	Kind   Kind
	Device gotch.Device
}

var (
	FloatCPU  KindDevice = KindDevice{Float, gotch.CPU}
	DoubleCPU KindDevice = KindDevice{Double, gotch.CPU}
	Int64CPU  KindDevice = KindDevice{Int64, gotch.CPU}

	FloatCUDA  KindDevice = KindDevice{Float, gotch.CudaBuilder(0)}
	DoubleCUDA KindDevice = KindDevice{Double, gotch.CudaBuilder(0)}
	Int64CUDA  KindDevice = KindDevice{Int64, gotch.CudaBuilder(0)}
)

type KindTrait interface {
	GetKind() Kind
}

type KindUint8 struct{}

func (k KindUint8) GetKind() Kind { return Uint8 }

type KindInt8 struct{}

func (k KindInt8) GetKind() Kind { return Int8 }

type KindInt16 struct{}

func (k KindInt16) GetKind() Kind { return Int16 }

type KindInt64 struct{}

func (k KindInt64) GetKind() Kind { return Int64 }

type KindFloat32 struct{}

func (k KindFloat32) GetKind() Kind { return Float }

type KindFloat64 struct{}

func (k KindFloat64) GetKind() Kind { return Double }

type KindBool struct{}

func (k KindBool) GetKind() Kind { return Bool }
