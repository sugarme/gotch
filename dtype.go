package gotch

import (
	"fmt"
	"log"
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

func (dt DType) CInt() (retVal CInt) {
	retVal, err := DType2CInt(dt)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func CInt2DType(v CInt) (dtype DType, err error) {
	var found = false
	for key, val := range dtypeCInt {
		if val == v {
			dtype = key
			found = true
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

	// NOTE: call `Interface()` to get data type back to interface{} type
	typ, _, err := dataCheck(reflect.ValueOf(data).Interface(), 0)
	if err != nil {
		return retVal, err
	}

	if typ.Kind() == reflect.Slice {
		return ToDType(typ.Elem())
	}

	return ToDType(typ)
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

// ElementGoType infers and returns Go type of element in given data
func ElementGoType(data interface{}) (retVal reflect.Type, err error) {
	dataValue := reflect.ValueOf(data)
	return elementType(dataValue)
}

func elementType(data reflect.Value) (dataType reflect.Type, err error) {
	dataKind := data.Kind()
	switch dataKind {
	case reflect.Uint8, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Float32, reflect.Float64, reflect.Bool:
		dataType = data.Type()
	case reflect.Slice, reflect.Array:
		data = data.Elem()
		dataType, err = elementType(data) // recursively type inferring
	default:
		err = fmt.Errorf("Unsupported type for data type %v\n", dataType)
		return DType{}, err
	}

	return dataType, nil
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

/*
 * // TypeCheck checks whether data Go type matching DType
 * func TypeCheck(data interface{}, dtype DType) (matched bool, msg string) {
 *   dataValue := reflect.ValueOf(data)
 *   var dataType reflect.Type
 *   var err error
 *   dataType, err = elementType(dataValue)
 *   if err != nil {
 *     msg = fmt.Sprintf("data type: %v, DType: %v\n", dataType, dtype.Kind())
 *     msg += err.Error()
 *     return false, msg
 *   }
 *
 *   matched = dataType == dtype.Type
 *   msg = fmt.Sprintf("data type: %v, DType: %v\n", dataType, dtype.Kind())
 *
 *   return matched, msg
 * }
 *  */

var supportedTypes = map[reflect.Kind]bool{
	reflect.Uint8:   true,
	reflect.Int8:    true,
	reflect.Int16:   true,
	reflect.Int32:   true,
	reflect.Int64:   true,
	reflect.Float32: true,
	reflect.Float64: true,
	reflect.Bool:    true,
}

var scalarTypes = map[reflect.Kind]bool{
	reflect.Bool:       true,
	reflect.Int:        true,
	reflect.Int8:       true,
	reflect.Int16:      true,
	reflect.Int32:      true,
	reflect.Int64:      true,
	reflect.Uint:       true,
	reflect.Uint8:      true,
	reflect.Uint16:     true,
	reflect.Uint32:     true,
	reflect.Uint64:     true,
	reflect.Uintptr:    true,
	reflect.Float32:    true,
	reflect.Float64:    true,
	reflect.Complex64:  true,
	reflect.Complex128: true,
}

// IsSupportedScalar checks whether given SCALAR type is supported
// TODO: check input is a scalar.
func IsSupportedScalar(k reflect.Kind) bool {
	// if _, ok := scalarTypes[k]; !ok {
	// log.Fatalf("Input type: %v is not a Go scalar type.", k)
	// }

	_, retVal := supportedTypes[k]

	return retVal
}
