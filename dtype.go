package gotch

import (
	"fmt"
	"log"
	"reflect"
)

// CInt is equal to C type int. Go type is int32
type CInt = int32

// DType represents different kind of element that a tensor can hold.
// Ref. https://github.com/pytorch/pytorch/blob/a290cbf32b0c282aa60fa521ca5c6cd19c7f779f/c10/core/ScalarType.h
type DType int

const (
	Invalid       DType = -1
	Uint8         DType = 0
	Int8          DType = 1
	Int16         DType = 2
	Int           DType = 3
	Int64         DType = 4
	Half          DType = 5
	Float         DType = 6
	Double        DType = 7
	ComplexHalf   DType = 8
	ComplexFloat  DType = 9
	ComplexDouble DType = 10
	Bool          DType = 11
	QInt8         DType = 12
	QUInt8        DType = 13
	QInt32        DType = 14
	BFloat16      DType = 15
	// ---not implemented ---
	QUInt4x2 DType = 16
	QUInt2x4 DType = 17
	Bits1x8  DType = 18
	Bits2x4  DType = 19
	Bits4x2  DType = 20
	Bits8    DType = 21
	Bits16   DType = 22
)

var dtype2CKind = map[DType]CInt{
	Uint8:         0,
	Int8:          1,
	Int16:         2,
	Int:           3,
	Int64:         4,
	Half:          5,
	Float:         6,
	Double:        7,
	ComplexHalf:   8,
	ComplexFloat:  9,
	ComplexDouble: 10,
	Bool:          11,
	QInt8:         12,
	QUInt8:        13,
	QInt32:        14,
	BFloat16:      15,
	// ---not implemented ---
	QUInt4x2: 16,
	QUInt2x4: 17,
	Bits1x8:  18,
	Bits2x4:  19,
	Bits4x2:  20,
	Bits8:    21,
	Bits16:   22,
}

func (dt DType) CKind() CInt {
	if cint, ok := dtype2CKind[dt]; ok {
		return cint
	}

	if Debug {
		log.Printf("WARNING: dt.CKind() failed: no corresponding CKind to this DType %v\n", dt)
	}
	return -1 // invalid
}

// Back compat
func (dt DType) CInt() CInt {
	return dt.CKind()
}

var ckind2DType map[CInt]DType = map[CInt]DType{
	0:  Uint8,
	1:  Int8,
	2:  Int16,
	3:  Int,
	4:  Int64,
	5:  Half,
	6:  Float,
	7:  Double,
	8:  ComplexHalf,
	9:  ComplexFloat,
	10: ComplexDouble,
	11: Bool,
	12: QInt8,
	13: QUInt8,
	14: QInt32,
	15: BFloat16,
	// ---not implemented ---
	16: QUInt4x2,
	17: QUInt2x4,
	18: Bits1x8,
	19: Bits2x4,
	20: Bits4x2,
	21: Bits8,
	22: Bits16,
}

func CKind2DType(ckind int32) DType {
	if dtype, ok := ckind2DType[ckind]; ok {
		return dtype
	}

	if Debug {
		log.Printf("WARNING: CKind2DType() failed: no corresponding DType to input CInt %v\n", ckind)
	}
	return -1 // invalid
}

var dtypeSize map[DType]uint = map[DType]uint{
	Uint8:         1,
	Int8:          1,
	Int16:         2,
	Int:           4,
	Int64:         8,
	Half:          2,
	Float:         4,
	Double:        8,
	ComplexHalf:   4,
	ComplexFloat:  8,
	ComplexDouble: 16,
	Bool:          1,
	QInt8:         1,
	QUInt8:        1,
	QInt32:        4,
	BFloat16:      2,
	QUInt4x2:      2,
	QUInt2x4:      1,
	// ---not implemented ---
	Bits1x8: 1,
	Bits2x4: 1,
	Bits4x2: 1,
	Bits8:   1,
	Bits16:  2,
}

// Size returns dtype size in Bytes.
func (dt DType) Size() uint {
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

var dtype2GoKind map[DType]reflect.Kind = map[DType]reflect.Kind{
	Uint8:         reflect.Uint8,
	Int8:          reflect.Int8,
	Int16:         reflect.Int16,
	Int:           reflect.Int32,
	Int64:         reflect.Int64,
	Half:          reflect.Uint16, // <- just uint16
	Float:         reflect.Float32,
	Double:        reflect.Float64,
	ComplexHalf:   reflect.Invalid, // no equivalent in Go. Would it be reflect.Float64?
	ComplexFloat:  reflect.Complex64,
	ComplexDouble: reflect.Complex128,
	Bool:          reflect.Bool,
	QInt8:         reflect.Int8,
	QUInt8:        reflect.Uint8,
	QInt32:        reflect.Int32,
	BFloat16:      reflect.Uint16, // <- just uint16
	// ---not implemented ---
	QUInt4x2: reflect.Invalid,
	QUInt2x4: reflect.Invalid,
	Bits1x8:  reflect.Invalid,
	Bits2x4:  reflect.Invalid,
	Bits4x2:  reflect.Invalid,
	Bits8:    reflect.Invalid,
	Bits16:   reflect.Invalid,
}

func (dt DType) GoKind() reflect.Kind {
	if kind, ok := dtype2GoKind[dt]; ok && kind != reflect.Invalid {
		return kind
	}

	if Debug {
		log.Printf("WARNING: DType.GoKind() failed: no corresponding Go reflect.Kind to given DType %v\n", dt)
	}

	return reflect.Invalid
}

const (
	// NOTE. reflect.Kind 0-26
	QUInt8Kind      reflect.Kind = 27
	QInt8Kind       reflect.Kind = 28
	QInt32Kind      reflect.Kind = 29
	Float16Kind     reflect.Kind = 30
	BFloat16Kind    reflect.Kind = 31
	QUInt4x2Kind    reflect.Kind = 32
	QUInt2x4Kind    reflect.Kind = 33
	Bits1x8Kind     reflect.Kind = 34
	Bits2x4Kind     reflect.Kind = 35
	Bits4x2Kind     reflect.Kind = 36
	Bits8Kind       reflect.Kind = 37
	Bits16Kind      reflect.Kind = 38
	ComplexHalfKind reflect.Kind = 39
)

var goKind2DType map[reflect.Kind]DType = map[reflect.Kind]DType{
	reflect.Uint8:      Uint8,
	reflect.Int8:       Int8,
	reflect.Int16:      Int16,
	reflect.Int32:      Int,
	reflect.Int64:      Int64,
	reflect.Float32:    Float,
	reflect.Float64:    Double,
	reflect.Complex64:  ComplexFloat,
	reflect.Complex128: ComplexDouble,
	reflect.Bool:       Bool,
	reflect.Uint16:     Half,

	// Added Kinds
	QUInt8Kind: QUInt8,
	QInt8Kind:  QInt8,
	QInt32Kind: QInt32,
	// Float16Kind:     Half,
	BFloat16Kind:    BFloat16,
	QUInt4x2Kind:    QUInt4x2,
	QUInt2x4Kind:    QUInt2x4,
	Bits1x8Kind:     Bits1x8,
	Bits2x4Kind:     Bits2x4,
	Bits4x2Kind:     Bits4x2,
	Bits8Kind:       Bits8,
	Bits16Kind:      Bits16,
	ComplexHalfKind: ComplexHalf,
}

type DTypeOptions struct {
	HalfDTypePref DType
	Quantized     bool
}

type DTypeOpt func(*DTypeOptions)

func DefaultDTypeOptions() *DTypeOptions {
	return &DTypeOptions{
		HalfDTypePref: Half,
		Quantized:     false,
	}
}

func HalfDTypePref(v DType) DTypeOpt {
	if v != Half && v != BFloat16 {
		if Debug {
			log.Printf("WARNING: HalfDTypePref(): Ignoring invalid HalfDTypePref. HalfDTypePref either 'gotch.Half' or 'gotch.BFloat16'. Got %v\n", v)
		}
	}

	return func(o *DTypeOptions) {
		o.HalfDTypePref = v
	}
}

func WithQuantized(v bool) DTypeOpt {
	return func(o *DTypeOptions) {
		o.Quantized = v
	}
}

func GoKind2DType(kind reflect.Kind, opts ...DTypeOpt) (DType, error) {
	o := DefaultDTypeOptions()
	for _, opt := range opts {
		opt(o)
	}

	switch {
	case kind == reflect.Uint16 && o.HalfDTypePref == Half:
		return Half, nil
	case kind == reflect.Uint16 && o.HalfDTypePref == BFloat16:
		return BFloat16, nil
	case kind == reflect.Int8 && o.Quantized:
		return QInt8, nil
	case kind == reflect.Uint8 && o.Quantized:
		return QUInt8, nil
	case kind == reflect.Int32 && o.Quantized:
		return QInt32, nil

	default:
		dtype, ok := goKind2DType[kind]
		if !ok {
			err := fmt.Errorf("GoKind2DType() failed: no corresponding DType to given Go reflect.Kind %v\n", kind)
			return Invalid, err
		}
		return dtype, nil
	}
}

var dtype2GoType map[DType]reflect.Type = map[DType]reflect.Type{
	Uint8:  reflect.TypeOf(uint8(0)),
	Int8:   reflect.TypeOf(int8(0)),
	Int16:  reflect.TypeOf(int16(0)),
	Int:    reflect.TypeOf(int(0)),
	Int64:  reflect.TypeOf(int64(0)),
	Half:   reflect.TypeOf(uint16(0)), // <- just uint16
	Float:  reflect.TypeOf(float32(0)),
	Double: reflect.TypeOf(float64(0)),
	// ComplexHalf:   reflect.Invalid, // no equivalent in Go. Would it be reflect.Float64?
	ComplexFloat:  reflect.TypeOf(complex64(0)),
	ComplexDouble: reflect.TypeOf(complex128(0)),
	Bool:          reflect.TypeOf(true),
	QInt8:         reflect.TypeOf(int8(0)),
	QUInt8:        reflect.TypeOf(uint8(0)),
	QInt32:        reflect.TypeOf(int32(0)),
	BFloat16:      reflect.TypeOf(uint16(0)), // <- just uint16
	// ---not implemented ---
	QUInt4x2: reflect.TypeOf(int8(0)),
	QUInt2x4: reflect.TypeOf(uint8(0)),
	Bits1x8:  reflect.TypeOf(uint8(0)),
	Bits2x4:  reflect.TypeOf(uint8(0)),
	Bits4x2:  reflect.TypeOf(uint8(0)),
	Bits8:    reflect.TypeOf(uint8(0)),
	Bits16:   reflect.TypeOf(uint16(0)),
}

func (dt DType) GoType() (reflect.Type, error) {
	typ, ok := dtype2GoType[dt]
	if !ok {
		err := fmt.Errorf("DType.GoType() failed: no corresponding Go type to given DType %v\n", typ.String())
		return nil, err
	}

	return typ, nil
}

var dtypeNames map[DType]string = map[DType]string{
	Uint8:  "Uint8",
	Int8:   "Int8",
	Int16:  "Int16",
	Int:    "Int",
	Int64:  "Int64",
	Half:   "Half", // <- just uint16
	Float:  "Float",
	Double: "Double",
	// ComplexHalf:   reflect.Invalid, // no equivalent in Go. Would it be reflect.Float64?
	ComplexFloat:  "ComplexFloat",
	ComplexDouble: "ComplexDouble",
	Bool:          "Bool",
	QInt8:         "QInt8",
	QUInt8:        "QUInt8",
	QInt32:        "QInt32",
	BFloat16:      "BFloat16", // <- just uint16
	// ---not implemented ---
	QUInt4x2: "QUInt4x2",
	QUInt2x4: "QUInt2x4",
	Bits1x8:  "Bits1x8",
	Bits2x4:  "Bits2x4",
	Bits4x2:  "Bits4x2",
	Bits8:    "Bits8",
	Bits16:   "Bits16",
}

func (dt DType) String() string {
	return dtypeNames[dt]
}

func DTypeFromData(data interface{}) (DType, error) {
	dataKind := reflect.TypeOf(data).Kind()

	// Data is a slice/array
	if dataKind == reflect.Slice || dataKind == reflect.Array {
		elementKind := reflect.TypeOf(data).Elem().Kind()
		return GoKind2DType(elementKind)
	}

	// single element
	return GoKind2DType(dataKind)
}
