package torch

import (
	"reflect"
	"unsafe"
)

type Kind struct {
	reflect.Type
}

type CInt = int32

/*
 *     Uint8,
 *     Int8,
 *     Int16,
 *     Int,
 *     Int64,
 *     Half,
 *     Float,
 *     Double,
 *     ComplexHalf,
 *     ComplexFloat,
 *     ComplexDouble,
 *     Bool,
 *  */

// TODO: recode these types

var (
	Bool       = Kind{reflect.TypeOf(true)}
	Int        = Kind{reflect.TypeOf(int(1))}
	Int8       = Kind{reflect.TypeOf(int8(1))}
	Int16      = Kind{reflect.TypeOf(int16(1))}
	Int32      = Kind{reflect.TypeOf(int32(1))}
	Int64      = Kind{reflect.TypeOf(int64(1))}
	Uint       = Kind{reflect.TypeOf(uint(1))}
	Uint8      = Kind{reflect.TypeOf(uint8(1))}
	Uint16     = Kind{reflect.TypeOf(uint16(1))}
	Uint32     = Kind{reflect.TypeOf(uint32(1))}
	Uint64     = Kind{reflect.TypeOf(uint64(1))}
	Float32    = Kind{reflect.TypeOf(float32(1))}
	Float64    = Kind{reflect.TypeOf(float64(1))}
	Complex64  = Kind{reflect.TypeOf(complex64(1))}
	Complex128 = Kind{reflect.TypeOf(complex128(1))}
	String     = Kind{reflect.TypeOf("")}

	// aliases
	Byte = Uint8

	// extras
	Uintptr       = Kind{reflect.TypeOf(uintptr(0))}
	UnsafePointer = Kind{reflect.TypeOf(unsafe.Pointer(&Uintptr))}
)
