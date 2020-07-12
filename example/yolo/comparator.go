package main

import (
	"log"
	"reflect"
)

type Comparator struct{}

// Init initiates a Comparator object
func (c Comparator) Init() Comparator {
	return Comparator{}
}

// Comparator compares 2 input a and b of generic type.
// It returns an interger result
// - 0: a == b
// - negative: a < b
// - positive: a > b
// Will be panic if a or be is not of asserted type.
// func (c Comparator) Compare(a, b interface{}) (retVal int) {
func (c Comparator) Compare(a, b interface{}) (retVal int) {
	if reflect.TypeOf(a) != reflect.TypeOf(b) {
		log.Fatalf("Expected a and b of the same type. Got a type: %v, b type: %v", reflect.TypeOf(a).Kind(), reflect.TypeOf(b).Kind())
	}

	typ := reflect.TypeOf(a)

	switch typ.Name() {
	case "string":
		return compareString(a.(string), b.(string))
	case "uint":
		return compareUint(a.(uint), b.(uint))
	case "uint8": // including `Byte` type
		return compareUint8(a.(uint8), b.(uint8))
	case "uint16":
		return compareUint16(a.(uint16), b.(uint16))
	case "uint32": // including `Rune` type
		return compareUint32(a.(uint32), b.(uint32))
	case "uint64":
		return compareUint64(a.(uint64), b.(uint64))
	case "int":
		return compareInt(a.(int), b.(int))
	case "int8":
		return compareInt8(a.(int8), b.(int8))
	case "int16":
		return compareInt16(a.(int16), b.(int16))
	case "int32":
		return compareInt32(a.(int32), b.(int32))
	case "int64":
		return compareInt64(a.(int64), b.(int64))
	case "float32":
		return compareFloat32(a.(float32), b.(float32))
	case "float64":
		return compareFloat64(a.(float64), b.(float64))
	default:
		log.Fatalf("Unsupported a type(%v) or b type(%v).\n", reflect.TypeOf(a).Kind(), reflect.TypeOf(b).Kind())
	}

	return retVal
}

func compareString(a, b string) (retVal int) {
	min := len(b)
	if len(a) < len(b) {
		min = len(a)
	}
	diff := 0
	for i := 0; i < min && diff == 0; i++ {
		diff = int(a[i]) - int(b[i])
	}
	if diff == 0 {
		diff = len(a) - len(b)
	}
	if diff < 0 {
		return -1
	}
	if diff > 0 {
		return 1
	}
	return 0
}

func compareUint(a, b uint) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}

func compareUint8(a, b uint8) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}

func compareUint16(a, b uint16) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}

func compareUint32(a, b uint32) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}

func compareUint64(a, b uint64) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}

func compareInt(a, b int) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}

func compareInt8(a, b int8) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}

func compareInt16(a, b int16) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}

func compareInt32(a, b int32) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}

func compareInt64(a, b int64) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}

func compareFloat32(a, b float32) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}

func compareFloat64(a, b float64) (retVal int) {
	switch {
	case a > b:
		return 1
	case a < b:
		return -1
	default:
		return 0
	}
}
