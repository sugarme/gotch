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

var dtypeGoType = map[DType]reflect.Type{
	Uint8:  reflect.TypeOf(uint8(1)),
	Int8:   reflect.TypeOf(int8(1)),
	Int16:  reflect.TypeOf(int16(1)),
	Int:    reflect.TypeOf(int32(1)),
	Int64:  reflect.TypeOf(int64(1)),
	Float:  reflect.TypeOf(float32(1)),
	Double: reflect.TypeOf(float64(1)),
	Bool:   reflect.TypeOf(true),
}

// ToDType infers and returns supported equivalent DType from given Go type
func ToDType(typ reflect.Type) (retVal DType, err error) {
	var found = false
	for key, val := range dtypeGoType {
		if val == typ {
			retVal = key
			found = true
			break
		}
	}

	if !found {
		err = fmt.Errorf("Unsupported Go type: %v", typ)
		return DType{}, err
	}

	return retVal, nil
}

// ToGoType infers and returns supported equivalent Go type from given DType
func ToGoType(dtype DType) (retVal reflect.Type, err error) {
	if _, ok := dtypeGoType[dtype]; !ok {
		err = fmt.Errorf("Unsupported DType %v", dtype)
		return nil, err
	}

	retVal = dtypeGoType[dtype]

	return retVal, nil
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

func DType2CInt(dt DType) (retVal CInt, err error) {
	if _, ok := dtypeCInt[dt]; !ok {
		err = fmt.Errorf("Unsupported CInt conversion from DType: %v\n", dt)
	}

	retVal = dtypeCInt[dt]

	return retVal, nil
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
func DTypeSize(dt DType) (retVal uint, err error) {
	if _, ok := dtypeSize[dt]; !ok {
		err = fmt.Errorf("Unsupported conversion DType size in Byte for DType: %v\n", dt)
		return 99, err
	}

	retVal = dtypeSize[dt]

	return retVal, nil
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

// DTypeFromData infers returns equavalent DType from given data
func DTypeFromData(data interface{}) (retVal DType, err error) {
	dataKind := reflect.ValueOf(data).Kind()
	var dataType reflect.Type
	switch dataKind {
	case reflect.Uint8, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Float32, reflect.Float64, reflect.Bool:
		dataType = reflect.TypeOf(data)
	case reflect.Slice:
		dataType = reflect.TypeOf(data).Elem()
	default:
		err = fmt.Errorf("Unsupported type for data type %v\n", dataType)
		return DType{}, err
	}

	retVal = DType{reflect.TypeOf(dataType)}

	return retVal, nil

}

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
	var typ reflect.Type
	if typ, err = ToGoType(dt); err != nil {
		return nil, err
	}

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

// TypeCheck checks whether data Go type matching DType
func TypeCheck(data interface{}, dtype DType) (matched bool, msg string) {

	dataKind := reflect.ValueOf(data).Kind()
	dataType := reflect.TypeOf(data)

	switch dataKind {
	case reflect.Slice:
		dataEleType := reflect.TypeOf(data).Elem()
		matched = dataEleType == dtype.Type
		msg = fmt.Sprintf("data type: %v, DType: %v", dataEleType, dtype.Kind())
	default:
		matched = dataType == dtype.Type
		msg = fmt.Sprintf("data type: %v, DType: %v", dataType, dtype.Kind())
	}

	return matched, msg

}
