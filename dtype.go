package gotch

import (
	"fmt"
	// "log"
	"reflect"
)

// CInt is equal to C type int. Go type is int32
type CInt = int32

// DType represents different kind of element that a tensor can hold.
// It has an embedded `reflect.Type` for type reflection.
type DType struct {
	reflect.Type
}

/*
 * // Custom-made Float16 as not exist in Go
 * // Ref: https://github.com/golang/go/issues/32022
 * type GoFloat16 = int16 // not implemented yet
 * type GoComplexHalf = interface{} // not implemented yet!
 *  */

// TODO: double check these Torch DType to Go type
var (
	Uint8 DType = DType{reflect.TypeOf(uint8(1))} // 0
	Int8  DType = DType{reflect.TypeOf(int8(1))}  // 1
	Int16 DType = DType{reflect.TypeOf(int16(1))} // 2
	Int   DType = DType{reflect.TypeOf(int32(1))} // 3
	Int64 DType = DType{reflect.TypeOf(int64(1))} // 4
	// Half       DType   = DType{reflect.TypeOf(GoFloat16(1))}     // 5
	Float  DType = DType{reflect.TypeOf(float32(1))} // 6
	Double DType = DType{reflect.TypeOf(float64(1))} // 7
	// ComplexHalf DType  = DType{reflect.TypeOf(GoComplexHalf(1))} // 8
	// ComplexFloat DType  = DType{reflect.TypeOf(complex64(1))}  // 9
	// ComplexDouble DType = DType{reflect.TypeOf(complex128(1))} // 10
	Bool DType = DType{reflect.TypeOf(true)} // 11
)

/*
 * // ToCInt converts DType to CInt type value which is `C int`
 * func (dt DType) ToCInt() CInt {
 *   switch dt.Kind() {
 *   case reflect.Uint8:
 *     return 0
 *   case reflect.Int8:
 *     return 1
 *   case reflect.Int16:
 *     return 2
 *   case reflect.Int32:
 *     return 3
 *   case reflect.Int64:
 *     return 4
 *   case reflect.Float32:
 *     return 6
 *   case reflect.Float64:
 *     return 7
 *   case reflect.Bool:
 *     return 11
 *   default:
 *     log.Fatalf("Unsupported type.")
 *   }
 *
 *   // unreachable
 *   return CInt(-1)
 * }
 *
 * // OfCInt converts a value of type CInt to DType type value
 * func (dt DType) OfCInt(v CInt) DType {
 *   switch v {
 *   case 0:
 *     return Uint8
 *   case 1:
 *     return Int8
 *   case 2:
 *     return Int16
 *   case 3:
 *     return Int
 *   case 4:
 *     return Int64
 *   case 6:
 *     return Float
 *   case 7:
 *     return Double
 *   case 8:
 *   case 11:
 *     return Bool
 *   default:
 *     log.Fatalf("Unexpected DType %v\n", v)
 *   }
 *   return DType{reflect.TypeOf(false)}
 * }
 *
 * // EltSizeInBytes converts a DType value to number of bytes
 * // This is a ELement Size In Bytes in Libtorch.
 * // Has it been deprecated?
 * func (dt DType) EltSizeInBytes() uint {
 *   switch dt.Kind() {
 *   case reflect.Uint8:
 *     return 1
 *   case reflect.Int8:
 *     return 1
 *   case reflect.Int16:
 *     return 2
 *   case reflect.Int:
 *     return 4
 *   case reflect.Int64:
 *     return 8
 *   case reflect.Float32:
 *     return 4
 *   case reflect.Float64:
 *     return 8
 *   case reflect.Bool:
 *     return 1
 *   default:
 *     log.Fatalf("Unsupported Type %v\n", dt.Type)
 *   }
 *   return uint(0)
 * }
 *  */

// ToGoType converts DType to Go type
func (dt DType) ToGoType() reflect.Type {
	return dt.Type
}

var dtypeCInt = map[DType]CInt{
	Uint8:  0,
	Int8:   1,
	Int16:  2,
	Int:    3,
	Int64:  4,
	Float:  6,
	Double: 7,
	Bool:   11,
}

func DType2CInt(dt DType) CInt {
	return dtypeCInt[dt]
}

func CInt2DType(v CInt) (dtype DType, err error) {
	var found = false
	for key, val := range dtypeCInt {
		if val == v {
			dtype = key
			break
		}
	}

	if !found {
		err = fmt.Errorf("Unsuported DType for CInt %v\n", v)
		return DType{}, err
	}

	return dtype, nil

}

// dtypeSize is a map of DType and its size in Bytes
var dtypeSize = map[DType]uint{
	Uint8:  1,
	Int8:   1,
	Int16:  2,
	Int:    4,
	Int64:  8,
	Float:  4,
	Double: 8,
	Bool:   1,
}

// DTypeSize returns DType size in Bytes
func DTypeSize(dt DType) uint {
	return dtypeSize[dt]
}

type DTypeDevice struct {
	DType  DType
	Device Device
}

var (
	FloatCPU  DTypeDevice = DTypeDevice{Float, CPU}
	DoubleCPU DTypeDevice = DTypeDevice{Double, CPU}
	Int64CPU  DTypeDevice = DTypeDevice{Int64, CPU}

	FloatCUDA  DTypeDevice = DTypeDevice{Float, CudaBuilder(0)}
	DoubleCUDA DTypeDevice = DTypeDevice{Double, CudaBuilder(0)}
	Int64CUDA  DTypeDevice = DTypeDevice{Int64, CudaBuilder(0)}
)

// Type Inferring:
// ===============

// DataDType infers and returns data type of tensor data
func DataDType(v interface{}, shape []int64) (retVal DType, err error) {
	// assuming that all elements in data have the same type
	switch {
	case len(shape) == 0:
		retVal, err = ElementDType(v)
	case len(shape) > 0:
		return ElementDType(v.([]interface{})[0])
	default:
		err = fmt.Errorf("Unsupported data type for %v\n", reflect.TypeOf(v))
		return DType{}, err
	}
	return DType{}, nil
}

// ElementDType infers and returns its own tensor data type
func ElementDType(v interface{}) (retVal DType, err error) {
	switch v.(type) {
	case uint8:
		retVal = Uint8
	case int8:
		retVal = Int8
	case int16:
		retVal = Int16
	case int32:
		retVal = Int
	case int64:
		retVal = Int64
	case float32:
		retVal = Float
	case float64:
		retVal = Double
	case bool:
		retVal = Bool
	default:
		err = fmt.Errorf("Unsupported data type for %v\n", reflect.TypeOf(v))
	}

	return retVal, nil
}

// TypeOf infers and returns element Go type from given tensor DType and shape
func TypeOf(dt DType, shape []int64) (retVal reflect.Type, err error) {
	typ := dt.ToGoType()

	switch {
	case len(shape) == 0:
		return typ, nil
	case len(shape) > 0:
		return reflect.SliceOf(typ), nil
	default:
		err = fmt.Errorf("Unsupported data type.")
		return nil, err
	}
}
