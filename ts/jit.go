package ts

// JIT interface to run model trained/saved using PyTorch Python API.

// #include "stdlib.h"
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"reflect"
	"unsafe"

	"github.com/sugarme/gotch"
	lib "github.com/sugarme/gotch/libtch"
)

type CIValue struct {
	civalue lib.Civalue
}

type IValueKind struct {
	reflect.Type
}

var (
	NoneVal        IValueKind = IValueKind{reflect.TypeOf(nil)}
	TensorVal      IValueKind = IValueKind{reflect.TypeOf(Tensor{})}
	DoubleVal      IValueKind = IValueKind{reflect.TypeOf(float64(1))}
	IntVal         IValueKind = IValueKind{reflect.TypeOf(int64(1))}
	BoolVal        IValueKind = IValueKind{reflect.TypeOf(true)}
	TupleVal       IValueKind = IValueKind{reflect.TypeOf([]IValue{})}
	IntListVal     IValueKind = IValueKind{reflect.TypeOf([]int64{})}
	DoubleListVal  IValueKind = IValueKind{reflect.TypeOf([]float64{})}
	BoolListVal    IValueKind = IValueKind{reflect.TypeOf([]bool{})}
	StringVal      IValueKind = IValueKind{reflect.TypeOf("")}
	TensorListVal  IValueKind = IValueKind{reflect.TypeOf([]Tensor{})}
	GenericListVal IValueKind = IValueKind{reflect.TypeOf([]IValue{})}
	GenericDictVal IValueKind = IValueKind{reflect.TypeOf(map[IValue]IValue{})} // 2 elements. ? map[IValue]IValue
	GenericVal     IValueKind = IValueKind{reflect.TypeOf(IValue{})}
)

type IValue struct {
	value interface{}
	kind  IValueKind
	name  string
}

// NewIValue creates a new IValue from given value of various types.
func NewIValue(v interface{}) *IValue {

	retVal := &IValue{value: v}
	if v == nil {
		retVal.kind = NoneVal
		retVal.name = "None"
		return retVal
	}

	inputTypeStr := reflect.TypeOf(v).Kind().String()

	switch inputTypeStr {
	case "Tensor":
		retVal.kind = TensorVal
		retVal.name = "Tensor"
	case "float64":
		retVal.kind = DoubleVal
		retVal.name = "Double"
	case "float32":
		retVal.kind = GenericVal
		retVal.name = "Generic"
	case "int64":
		retVal.kind = IntVal
		retVal.name = "Int"
	case "int":
		retVal.kind = GenericVal
		retVal.name = "Generic"
	case "int32":
		retVal.kind = GenericVal
		retVal.name = "Generic"
	case "bool":
		retVal.kind = BoolVal
		retVal.name = "Bool"
	case "string":
		retVal.kind = StringVal
		retVal.name = "String"
	case "slice":
		fmt.Printf("slice elem type: %q\n", reflect.TypeOf(v).Elem().Kind().String())
		switch reflect.TypeOf(v).Elem().Kind().String() {
		case "IValue":
			switch len(v.([]IValue)) {
			case 2:
				retVal.kind = TupleVal
				retVal.name = "Tuple"
			default:
				retVal.kind = GenericListVal
				retVal.name = "GenericList"
			}
		case "int64":
			retVal.kind = IntListVal
			retVal.name = "IntList"
		case "float64":
			retVal.kind = DoubleListVal
			retVal.name = "DoubleList"
		case "float32":
			retVal.kind = GenericListVal
			retVal.name = "GenericList"
		case "int32":
			retVal.kind = GenericListVal
			retVal.name = "GenericList"
		case "int":
			retVal.kind = GenericListVal
			retVal.name = "GenericList"
		case "string":
			retVal.kind = GenericListVal
			retVal.name = "GenericList"
		case "bool":
			retVal.kind = BoolListVal
			retVal.name = "BoolList"
		case "struct": // NOTE: only supported `Tensor` type
			val := reflect.Indirect(reflect.ValueOf(v))
			switch {
			// 1. Tuple (Tensor, Tensor)
			case val.Type() == reflect.TypeOf([]Tensor{}) && val.Len() == 2:
				retVal.kind = TensorListVal
				retVal.name = "Tuple"
				retVal.value = v.([]Tensor)

				// 2. List (Tensor, Tensor, ...)
			case val.Type() == reflect.TypeOf([]Tensor{}) && val.Len() > 2:
				retVal.kind = TensorListVal
				retVal.name = "TensorList"
				retVal.value = v.([]Tensor)
			default:
				log.Fatalf("NewIValue method call - 'slice -> struct' case - Unsupported type (%v)\n", reflect.TypeOf(v).Kind().String())
			}
		}
	case "map":
		// TODO: exclude map of type other than IValue type
		retVal.kind = GenericDictVal
		retVal.name = "GenericDict"
	case "struct":
		val := reflect.Indirect(reflect.ValueOf(v))
		fieldName := val.Type().Field(0).Name
		switch fieldName {
		case "ctensor":
			retVal.kind = TensorVal
			retVal.name = "Tensor"
		default:
			log.Fatalf("NewIValue method call - 'struct' case - Unsupported type (%v)\n", reflect.TypeOf(v).Kind().String())
		}
	default:
		log.Fatalf("NewIValue method call - Unsupported type (%v)\n", reflect.TypeOf(v).Kind().String())
	}

	return retVal
}

// IValue methods:
// ===============

func (iv *IValue) ToCIValue() (*CIValue, error) {
	switch iv.name {
	case "None":
		cval := lib.AtiNone()
		if err := TorchErr(); err != nil {
			return nil, err
		}

		return &CIValue{civalue: cval}, nil

	case "Tensor":
		cval := lib.AtiTensor(iv.value.(Tensor).ctensor)
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &CIValue{civalue: cval}, nil

	case "Int":
		cval := lib.AtiInt(iv.value.(int64))
		if err := TorchErr(); err != nil {
			return nil, err
		}

		return &CIValue{civalue: cval}, nil

	case "Double":
		cval := lib.AtiDouble(iv.value.(float64))
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &CIValue{civalue: cval}, nil

	case "Bool":
		cval := lib.AtiBool(iv.value.(bool))
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &CIValue{civalue: cval}, nil

	case "Tuple":
		val := reflect.Indirect(reflect.ValueOf(iv.value))
		switch {
		// 1. Tuple is (Tensor, Tensor)
		case val.Type() == reflect.TypeOf([]Tensor{}):
			var v []Tensor = iv.value.([]Tensor)
			var cvals []lib.Civalue
			for _, tensor := range v {
				ival := NewIValue(tensor)
				cval, err := ival.ToCIValue()
				if err != nil {
					err = fmt.Errorf("ToCIValue method call err - Tuple case: %v\n", err)
					return nil, err
				}
				cvals = append(cvals, cval.civalue)
			}

			tuple := lib.AtiTuple(cvals, len(cvals))
			if err := TorchErr(); err != nil {
				return nil, err
			}
			return &CIValue{civalue: tuple}, nil

		// 2. Tuple is (IValue, IValue)
		default:
			var v []IValue = iv.value.([]IValue)
			var cvals []lib.Civalue
			for _, i := range v {
				cval, err := i.ToCIValue()
				if err != nil {
					err = fmt.Errorf("ToCIValue method call err - Tuple case: %v\n", err)
					return nil, err
				}
				cvals = append(cvals, cval.civalue)
			}

			tuple := lib.AtiTuple(cvals, len(cvals))
			if err := TorchErr(); err != nil {
				return nil, err
			}
			return &CIValue{civalue: tuple}, nil
		}

	case "GenericList":
		// GenericList can be: string, int, int32, float32
		// TODO: refactor to function
		// NOTE: atm, `GenericList` are all unsupported cases
		var cvals []lib.Civalue
		vtyp := reflect.TypeOf(iv.value).Elem().Kind().String()
		switch vtyp {
		case "string":
			var v []string = iv.value.([]string)
			for _, i := range v {
				ival := NewIValue(i)
				cval, err := ival.ToCIValue()
				if err != nil {
					err = fmt.Errorf("ToCIValue method call err - GenericList case: %v\n", err)
					return nil, err
				}
				cvals = append(cvals, cval.civalue)
			}

		case "int":
			var v []int = iv.value.([]int)
			for _, i := range v {
				ival := NewIValue(i)
				cval, err := ival.ToCIValue()
				if err != nil {
					log.Fatalf("ToCIValue method call err - int case: %v\n", err)
				}
				cvals = append(cvals, cval.civalue)
			}
		case "int32":
			var v []int32 = iv.value.([]int32)
			for _, i := range v {
				ival := NewIValue(i)
				cval, err := ival.ToCIValue()
				if err != nil {
					log.Fatalf("ToCIValue method call err - int32 case: %v\n", err)
				}
				cvals = append(cvals, cval.civalue)
			}
		case "float32":
			var v []float32 = iv.value.([]float32)
			for _, i := range v {
				ival := NewIValue(i)
				cval, err := ival.ToCIValue()
				if err != nil {
					log.Fatalf("ToCIValue method call err - float32 case: %v\n", err)
				}
				cvals = append(cvals, cval.civalue)
			}
		default:
			log.Fatalf("ToCIValue method call err - Default case: Unsupport type (%v)\n", vtyp)

		}

		list := lib.AtiGenericList(cvals, len(cvals))
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &CIValue{civalue: list}, nil

	case "IntList":
		var vals []int64 = iv.value.([]int64)
		cval := lib.AtiIntList(vals, len(vals))
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &CIValue{civalue: cval}, nil

	case "DoubleList":
		var vals []float64 = iv.value.([]float64)
		cval := lib.AtiDoubleList(vals, len(vals))
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &CIValue{civalue: cval}, nil

	case "BoolList":
		var vals []bool = iv.value.([]bool)
		cval := lib.AtiBoolList(vals, len(vals))
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &CIValue{civalue: cval}, nil

	case "TensorList":
		var vals []Tensor = iv.value.([]Tensor)
		var cvals []lib.Ctensor
		for _, i := range vals {
			cvals = append(cvals, i.ctensor)
		}
		list := lib.AtiTensorList(cvals, len(cvals))
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &CIValue{civalue: list}, nil

	case "String":
		cval := lib.AtiString(iv.value.(string))
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &CIValue{civalue: cval}, nil

	case "GenericDict":
		var cvals []lib.Civalue
		keyType := reflect.TypeOf(iv.value).Key().Kind().String()
		valType := reflect.TypeOf(iv.value).Elem().Kind().String()

		// 1. Create key and value lists seperately
		switch {
		case keyType == "int64" && valType == "int64":
			var m map[int64]int64 = iv.value.(map[int64]int64)
			var vals []int64
			for k, v := range m {
				vals = append(vals, k, v)
			}
			for _, v := range vals {
				ival := NewIValue(v)
				cval, err := ival.ToCIValue()
				if err != nil {
					log.Fatalf("ToCIValue method call err - GenericDict case: %v\n", err)
				}
				cvals = append(cvals, cval.civalue)
			}

		case keyType == "float64" && valType == "float64":
			var m map[float64]float64 = iv.value.(map[float64]float64)
			var vals []float64
			for k, v := range m {
				vals = append(vals, k, v)
			}
			for _, v := range vals {
				ival := NewIValue(v)
				cval, err := ival.ToCIValue()
				if err != nil {
					log.Fatalf("ToCIValue method call err - GenericDict case: %v\n", err)
				}
				cvals = append(cvals, cval.civalue)
			}

		case keyType == "float32" && valType == "float32":
			var m map[float32]float32 = iv.value.(map[float32]float32)
			var vals []float32
			for k, v := range m {
				vals = append(vals, k, v)
			}
			for _, v := range vals {
				ival := NewIValue(v)
				cval, err := ival.ToCIValue()
				if err != nil {
					log.Fatalf("ToCIValue method call err - GenericDict case: %v\n", err)
				}
				cvals = append(cvals, cval.civalue)
			}

		// TODO: map[int64]Tensor
		// TODO: map[float64]Tensor
		// TODO: map[string]Tensor
		// TODO: map[bool]Tensor
		// ...

		default:
			log.Fatalf("ToCIValue method call - GenericDict case: unsupported key type(%v) or value type(%v) \n", keyType, valType)
		}

		// 2. Pairing key and value in a slice (cvals)
		dict := lib.AtiGenericDict(cvals, len(cvals)/2)
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &CIValue{civalue: dict}, nil

	case "Generic":
		err := fmt.Errorf("ToCIValue method call - Generic case: unsupport type(%v)\n", reflect.TypeOf(iv.value).Kind().String())
		return nil, err

	default:
		err := fmt.Errorf("ToCIValue method call - Generic case: unsupport type(%v)\n", reflect.TypeOf(iv.value).Kind().String())
		return nil, err
	}

	panic("Shouldn't reached here.")
}

// IValueFromC returns an IValue from given CIValue.
//
// It consumes the pointer and frees the associated memory.
func IValueFromC(cval *CIValue) (*IValue, error) {
	// tag will be a value of int32
	tag := lib.AtiTag(cval.civalue)
	if err := TorchErr(); err != nil {
		return nil, err
	}

	switch tag {
	case 0:
		return &IValue{
			value: nil,
			kind:  NoneVal,
			name:  "None",
		}, nil
	case 1:
		tensor := lib.AtiToTensor(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &IValue{
			value: Tensor{tensor},
			kind:  TensorVal,
			name:  "Tensor",
		}, nil
	case 2:
		v := lib.AtiToDouble(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &IValue{
			value: v,
			kind:  DoubleVal,
			name:  "Double",
		}, nil
	case 3:
		v := lib.AtiToInt(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &IValue{
			value: v,
			kind:  IntVal,
			name:  "Int",
		}, nil

	case 4:
		v := lib.AtiToBool(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &IValue{
			value: v,
			kind:  BoolVal,
			name:  "Bool",
		}, nil

	case 5: // Tuple []IValue 2 elements
		// 1. Determine tuple length
		len := lib.AtiTupleLength(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}
		// 2. Call with first pointer and length
		ptr1 := (*lib.Civalue)(unsafe.Pointer(C.malloc(0)))
		lib.AtiToTuple(cval.civalue, ptr1, int(len))
		if err := TorchErr(); err != nil {
			return nil, err
		}

		// 3. Get list of Civalue tuple elements
		var civalues []CIValue
		civalues = append(civalues, CIValue{civalue: *ptr1})
		currPtr := ptr1
		for i := 1; i < int(len); i++ {
			nextPtr := (*lib.Civalue)(unsafe.Pointer(uintptr(unsafe.Pointer(currPtr)) + unsafe.Sizeof(ptr1)))
			civalues = append(civalues, CIValue{civalue: *nextPtr})
			currPtr = nextPtr
		}

		// 4. Get Ivalue from Civalue for each tuple element
		// Determine element kind
		v, err := IValueFromC(&civalues[0])
		if err != nil {
			return nil, err
		}
		elemName := v.Name()
		switch elemName {
		case "Tensor":
			var vals []Tensor
			for _, civalue := range civalues {
				v, err := IValueFromC(&civalue)
				if err != nil {
					return nil, err
				}

				vals = append(vals, v.Value().(Tensor))
			}
			if len == 2 {
				return &IValue{
					value: vals,
					kind:  TensorListVal,
					name:  "Tuple",
				}, nil
			} else {
				return &IValue{
					value: vals,
					kind:  TensorListVal,
					name:  "TensorList",
				}, nil
			}
		case "IntList":
			var vals []int64
			for _, civalue := range civalues {
				v, err := IValueFromC(&civalue)
				if err != nil {
					return nil, err
				}
				vals = append(vals, v.Value().(int64))
			}
			return &IValue{
				value: vals,
				kind:  IntListVal,
				name:  "IntList",
			}, nil
		case "BoolList":
			var vals []bool
			for _, civalue := range civalues {
				v, err := IValueFromC(&civalue)
				if err != nil {
					return nil, err
				}
				vals = append(vals, v.Value().(bool))
			}
			return &IValue{
				value: vals,
				kind:  BoolListVal,
				name:  "BoolList",
			}, nil
		case "DoubleList":
			var vals []float64
			for _, civalue := range civalues {
				v, err := IValueFromC(&civalue)
				if err != nil {
					return nil, err
				}
				vals = append(vals, v.Value().(float64))
			}
			return &IValue{
				value: vals,
				kind:  DoubleListVal,
				name:  "DoubleList",
			}, nil

		default:
			var vals []interface{}
			for _, civalue := range civalues {
				v, err := IValueFromC(&civalue)
				if err != nil {
					return nil, err
				}
				vals = append(vals, v)
			}

			return &IValue{
				value: vals,
				kind:  TupleVal,
				name:  "Tuple",
			}, nil
		}

	case 6: // IntList
		// 1. Len
		len := lib.AtiLength(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}

		// 2. Call
		ptr1 := unsafe.Pointer(C.malloc(0))
		lib.AtiToIntList(cval.civalue, ptr1, int(len))
		if err := TorchErr(); err != nil {
			return nil, err
		}

		// 3. Get int list
		var intVals []int64
		intVals = append(intVals, *(*int64)(unsafe.Pointer(ptr1)))
		currPtr := ptr1
		for i := 1; i < int(len); i++ {
			nextPtr := unsafe.Pointer(uintptr(unsafe.Pointer(currPtr)) + unsafe.Sizeof(ptr1))
			intVals = append(intVals, *(*int64)(unsafe.Pointer(nextPtr)))
			currPtr = nextPtr
		}

		return &IValue{
			value: intVals,
			kind:  IntListVal,
			name:  "IntList",
		}, nil

	case 7: // DoubleList
		// 1. Len
		len := lib.AtiLength(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}

		// 2. Call
		ptr1 := unsafe.Pointer(C.malloc(0))
		lib.AtiToDoubleList(cval.civalue, ptr1, int(len))
		if err := TorchErr(); err != nil {
			return nil, err
		}

		// 3. Get int list
		var floatVals []float64
		floatVals = append(floatVals, *(*float64)(unsafe.Pointer(ptr1)))
		currPtr := ptr1
		for i := 1; i < int(len); i++ {
			nextPtr := unsafe.Pointer(uintptr(unsafe.Pointer(currPtr)) + unsafe.Sizeof(ptr1))
			floatVals = append(floatVals, *(*float64)(unsafe.Pointer(nextPtr)))
			currPtr = nextPtr
		}

		return &IValue{
			value: floatVals,
			kind:  DoubleListVal,
			name:  "DoubleList",
		}, nil

	case 8: // BoolList
		// 1. Len
		len := lib.AtiLength(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}

		// 2. Call
		ptr1 := unsafe.Pointer(C.malloc(0))
		lib.AtiToBoolList(cval.civalue, ptr1, int(len))
		if err := TorchErr(); err != nil {
			return nil, err
		}

		// 3. Get values
		var vals []int32
		var bvals []bool
		vals = append(vals, *(*int32)(unsafe.Pointer(ptr1)))
		currPtr := ptr1
		for i := 1; i < int(len); i++ {
			nextPtr := unsafe.Pointer(uintptr(unsafe.Pointer(currPtr)) + unsafe.Sizeof(ptr1))
			vals = append(vals, *(*int32)(unsafe.Pointer(nextPtr)))
			currPtr = nextPtr
		}

		for _, i := range vals {
			bval := false
			if i == 1 {
				bval = true
			}
			bvals = append(bvals, bval)
		}

		return &IValue{
			value: bvals,
			kind:  BoolListVal,
			name:  "BoolList",
		}, nil

	case 9: // String
		v := lib.AtiToString(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}
		return &IValue{
			value: v,
			kind:  StringVal,
			name:  "String",
		}, nil

	case 10: // TensorList
		// 1. Len
		len := lib.AtiLength(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}

		// 2. Call
		ptr1 := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
		lib.AtiToTensorList(cval.civalue, ptr1, int(len))
		if err := TorchErr(); err != nil {
			return nil, err
		}

		// 3. Get values
		var tensors []Tensor
		tensors = append(tensors, Tensor{ctensor: *ptr1})
		currPtr := ptr1
		for i := 1; i < int(len); i++ {
			nextPtr := (*lib.Ctensor)(unsafe.Pointer(uintptr(unsafe.Pointer(currPtr)) + unsafe.Sizeof(ptr1)))
			tensors = append(tensors, Tensor{ctensor: *nextPtr})
			currPtr = nextPtr
		}

		return &IValue{
			value: tensors,
			kind:  TensorListVal,
			name:  "TensorList",
		}, nil

	case 12: // GenericList []IValue
		// NOTE: atm, all these cases are unsupported.
		// 1. Len
		len := lib.AtiLength(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}
		// 2. Call with first pointer and length
		ptr1 := (*lib.Civalue)(unsafe.Pointer(C.malloc(0)))
		lib.AtiToGenericList(cval.civalue, ptr1, int(len))
		if err := TorchErr(); err != nil {
			return nil, err
		}

		// 3. Get values
		var civalues []CIValue
		civalues = append(civalues, CIValue{civalue: *ptr1})
		currPtr := ptr1
		for i := 1; i < int(len); i++ {
			nextPtr := (*lib.Civalue)(unsafe.Pointer(uintptr(unsafe.Pointer(currPtr)) + unsafe.Sizeof(ptr1)))
			civalues = append(civalues, CIValue{civalue: *nextPtr})
			currPtr = nextPtr
		}

		// 4. Get Ivalue from Civalue for each tuple element
		var vals []interface{}
		var itemTyp string
		for _, civalue := range civalues {
			v, err := IValueFromC(&civalue)
			if err != nil {
				return nil, err
			}
			itemTyp = reflect.TypeOf(v.value).Kind().String()
			vals = append(vals, v.value)
		}

		switch itemTyp {
		case "string":
			var specVals []string
			for _, v := range vals {
				specVals = append(specVals, v.(string))
			}
			return &IValue{
				value: specVals,
				kind:  GenericListVal,
				name:  "GenericList",
			}, nil
		case "int":
			var specVals []int
			for _, v := range vals {
				specVals = append(specVals, v.(int))
			}
			return &IValue{
				value: specVals,
				kind:  GenericListVal,
				name:  "GenericList",
			}, nil
		case "int32":
			var specVals []int32
			for _, v := range vals {
				specVals = append(specVals, v.(int32))
			}
			return &IValue{
				value: vals,
				kind:  GenericListVal,
				name:  "GenericList",
			}, nil
		case "float32":
			var specVals []float32
			for _, v := range vals {
				specVals = append(specVals, v.(float32))
			}
			return &IValue{
				value: vals,
				kind:  GenericListVal,
				name:  "GenericList",
			}, nil

		default:
			log.Fatalf("IValueFromC method call - GenericList case: Unsupported item type (%v)\n", itemTyp)
		}

	case 13: // GenericDict map[IValue]IValue
		// 1. Len
		numVals := lib.AtiLength(cval.civalue)
		if err := TorchErr(); err != nil {
			return nil, err
		}
		// 2. Call with first pointer and length
		ptr1 := (*lib.Civalue)(unsafe.Pointer(C.malloc(0)))
		lib.AtiToGenericDict(cval.civalue, ptr1, int(numVals))
		if err := TorchErr(); err != nil {
			return nil, err
		}

		// 3. Get values

		// TODO: Need to drill down a specific type
		var civalues []CIValue
		civalues = append(civalues, CIValue{civalue: *ptr1})
		currPtr := ptr1
		for i := 1; i < int(numVals)*2; i++ {
			nextPtr := (*lib.Civalue)(unsafe.Pointer(uintptr(unsafe.Pointer(currPtr)) + unsafe.Sizeof(ptr1)))
			civalues = append(civalues, CIValue{civalue: *nextPtr})
			currPtr = nextPtr
		}

		// 4. Get Ivalue from Civalue for each element
		var vals []interface{}
		var itemTyp string
		for _, civalue := range civalues {
			v, err := IValueFromC(&civalue)
			if err != nil {
				return nil, err
			}
			itemTyp = reflect.TypeOf(v.value).Kind().String()
			vals = append(vals, v.value)
		}

		switch itemTyp {
		case "string":
			var specVals map[string]string = make(map[string]string)
			for i := 0; i < len(vals); i += 2 {
				specVals[vals[i].(string)] = vals[i+1].(string)
			}
			return &IValue{
				value: specVals,
				kind:  GenericDictVal,
				name:  "GenericDict",
			}, nil
		case "int":
			var specVals map[int]int = make(map[int]int)
			for i := 0; i < len(vals); i += 2 {
				specVals[vals[i].(int)] = vals[i+1].(int)
			}
			return &IValue{
				value: specVals,
				kind:  GenericDictVal,
				name:  "GenericDict",
			}, nil
		case "int32":
			var specVals map[int32]int32 = make(map[int32]int32)
			for i := 0; i < len(vals); i += 2 {
				specVals[vals[i].(int32)] = vals[i+1].(int32)
			}
			return &IValue{
				value: specVals,
				kind:  GenericDictVal,
				name:  "GenericDict",
			}, nil
		case "int64":
			var specVals map[int64]int64 = make(map[int64]int64)
			for i := 0; i < len(vals); i += 2 {
				specVals[vals[i].(int64)] = vals[i+1].(int64)
			}
			return &IValue{
				value: specVals,
				kind:  GenericDictVal,
				name:  "GenericDict",
			}, nil
		case "float32":
			var specVals map[float32]float32 = make(map[float32]float32)
			for i := 0; i < len(vals); i += 2 {
				specVals[vals[i].(float32)] = vals[i+1].(float32)
			}
			return &IValue{
				value: specVals,
				kind:  GenericDictVal,
				name:  "GenericDict",
			}, nil
		case "float64":
			var specVals map[float64]float64 = make(map[float64]float64)
			for i := 0; i < len(vals); i += 2 {
				specVals[vals[i].(float64)] = vals[i+1].(float64)
			}
			return &IValue{
				value: specVals,
				kind:  GenericDictVal,
				name:  "GenericDict",
			}, nil
		}

	default:
		err := fmt.Errorf("IValueFromC - Unsupported type (tag value: %v)\n", tag)
		return nil, err
	}

	panic("Shouldn't reach here.")
}

func (iv *IValue) Value() interface{} {
	return iv.value
}

func (iv *IValue) Name() string {
	return iv.name
}

func (iv *IValue) Kind() IValueKind {
	return iv.kind
}

// A JIT PyTorch module.
//
// These modules can be created via the
// [TorchScript python api](https://pytorch.org/docs/stable/jit.html).
type CModule struct {
	Cmodule lib.Cmodule
}

func (cm *CModule) Drop() {
	lib.AtmFree(cm.Cmodule)
	if err := TorchErr(); err != nil {
		log.Fatalf("CModule Drop method err: %v\n", err)
	}
}

// Loads a PyTorch saved JIT model from a file.
func ModuleLoad(path string) (*CModule, error) {
	cmodule := lib.AtmLoad(path)
	if err := TorchErr(); err != nil {
		return nil, err
	}

	return &CModule{cmodule}, nil

}

// Loads a PyTorch saved JIT model from a file onto the given device.
//
// This function loads the model directly on the specified device,
// which means it also allows loading a GPU model on the CPU without having a CUDA enabled GPU.
func ModuleLoadOnDevice(path string, device gotch.Device) (*CModule, error) {
	cmodule := lib.AtmLoadOnDevice(path, device.CInt())
	if err := TorchErr(); err != nil {
		return nil, err
	}

	return &CModule{cmodule}, nil
}

// Loads a PyTorch saved JIT model from a read instance.
func ModuleLoadData(stream io.Reader) (*CModule, error) {

	buf := new(bytes.Buffer)
	buf.ReadFrom(stream)

	bufString := buf.String()

	cmodule := lib.AtmLoadStr(bufString, len(bufString))
	if err := TorchErr(); err != nil {
		return nil, err
	}

	return &CModule{cmodule}, nil

}

// Loads a PyTorch saved JIT model from a read instance.
//
// This function loads the model directly on the specified device,
// which means it also allows loading a GPU model on the CPU without having a CUDA enabled GPU.
func ModuleLoadDataOnDevice(stream io.Reader, device gotch.Device) (*CModule, error) {
	buf := new(bytes.Buffer)
	buf.ReadFrom(stream)

	bufString := buf.String()

	cmodule := lib.AtmLoadStrOnDevice(bufString, len(bufString), device.CInt())
	if err := TorchErr(); err != nil {
		return nil, err
	}

	return &CModule{cmodule}, nil
}

// ForwardTs performs the forward pass for a model on some specified tensor inputs.
func (cm *CModule) ForwardTs(tensors []Tensor) (*Tensor, error) {
	var ctensors []lib.Ctensor
	for _, t := range tensors {
		ctensors = append(ctensors, t.ctensor)
	}

	// NOTE: Write a slice of ctensors to C memory and get the pointer
	// 1. Calculate buffer size
	cptrSize := int(unsafe.Sizeof(ctensors[0])) // 8 bytes
	nbytes := cptrSize * len(ctensors)
	dataPtr := C.malloc(C.size_t(nbytes))
	defer C.free(dataPtr)
	dataSlice := (*[1 << 30]byte)(dataPtr)[:nbytes:nbytes]

	// 2. Convert C pointers to []byte
	var data []byte
	for _, ctensor := range ctensors {
		b := make([]byte, cptrSize)
		u := uintptr(unsafe.Pointer(ctensor))
		switch cptrSize {
		case 4:
			binary.LittleEndian.PutUint32(b, uint32(u))
		case 8:
			binary.LittleEndian.PutUint64(b, uint64(u))
		default:
			panic(fmt.Sprintf("unknown uintptr size: %v", cptrSize))
		}

		data = append(data, b...)
	}

	// 3. Copy data to buffer
	copy(dataSlice[:], data)

	// 4. Call C func with slice data pointer and number of ctensor pointers
	// NOTE:
	// - `dataPtr` is the pointer to slice of ctensor pointers
	// - `nsize` is number of ctensor pointers encoded in binary data.
	ctensorsPtr := (*lib.Ctensor)(dataPtr)
	ctensor := lib.AtmForward(cm.Cmodule, ctensorsPtr, len(ctensors))
	if err := TorchErr(); err != nil {
		return nil, err
	}

	return &Tensor{ctensor}, nil
}

// ForwardIs performs the forward pass for a model on some specified ivalue input.
func (cm *CModule) ForwardIs(ivalues []IValue) (*IValue, error) {

	var civalues []lib.Civalue
	for _, i := range ivalues {
		civalue, err := i.ToCIValue()
		if err != nil {
			return nil, err
		}
		civalues = append(civalues, civalue.civalue)
	}

	// NOTE: Write a slice of civalues to C memory and get the pointer
	// 1. Calculate buffer size
	cptrSize := int(unsafe.Sizeof(civalues[0])) // 8 bytes
	nbytes := cptrSize * len(civalues)
	dataPtr := C.malloc(C.size_t(nbytes))
	defer C.free(dataPtr)
	dataSlice := (*[1 << 30]byte)(dataPtr)[:nbytes:nbytes]

	// 2. Convert C pointers to []byte
	var data []byte
	for _, civalue := range civalues {
		b := make([]byte, cptrSize)
		u := uintptr(unsafe.Pointer(civalue))
		switch cptrSize {
		case 4:
			binary.LittleEndian.PutUint32(b, uint32(u))
		case 8:
			binary.LittleEndian.PutUint64(b, uint64(u))
		default:
			panic(fmt.Sprintf("unknown uintptr size: %v", cptrSize))
		}

		data = append(data, b...)
	}

	// 3. Copy data to buffer
	copy(dataSlice[:], data)

	// 4. Call C func with slice data pointer and number of civalue pointers
	// NOTE:
	// - `dataPtr` is the pointer to slice of civalue pointers
	// - `nsize` is number of civalue pointers encoded in binary data.
	civaluesPtr := (*lib.Civalue)(dataPtr)

	civ := lib.AtmForward_(cm.Cmodule, civaluesPtr, len(civalues))
	if err := TorchErr(); err != nil {
		return nil, err
	}

	return IValueFromC(&CIValue{civ})
}

// To moves CModule to specified device.
func (cm *CModule) To(device gotch.Device, kind gotch.DType, nonBlocking bool) {
	lib.AtmTo(cm.Cmodule, device.CInt(), kind.CInt(), nonBlocking)
	if err := TorchErr(); err != nil {
		log.Fatalf("CModule To method call err: %v\n", err)
	}
}

// Save save CModule to a specified path.
func (cm *CModule) Save(file string) error {
	lib.AtmSave(cm.Cmodule, file)
	return TorchErr()
}

// NamedParameters loads some named tensors from a module.
func (cm *CModule) NamedParameters() ([]NamedTensor, error) {
	var data lib.LoadData
	dataPtr := lib.PStore.Set(&data)
	lib.AtmNamedParameters(cm.Cmodule, dataPtr)
	if err := TorchErr(); err != nil {
		return nil, err
	}

	var namedTensors []NamedTensor
	for _, v := range data.NamedCtensors {
		namedTensor := NamedTensor{
			Name:   v.Name,
			Tensor: &Tensor{v.Ctensor},
		}

		namedTensors = append(namedTensors, namedTensor)
	}

	return namedTensors, nil
}

// GetProfilingMode get CModule profiling mode
func (cm *CModule) GetProfilingMode() bool {
	retVal := lib.AtmGetProfilingMode()
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}

	return retVal
}

// SetProfilingMode set CModule profiling mode
func (cm *CModule) SetProfilingMode(b bool) {
	lib.AtmSetProfilingMode(b)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

// SetTrain set CModule to train mode
func (cm *CModule) SetTrain() {
	lib.AtmTrain(cm.Cmodule)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

// SetEval set CModule to inference mode
func (cm *CModule) SetEval() {
	lib.AtmEval(cm.Cmodule)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}

// Implement Module for CModule:
// =============================

// Forwad implements Module interface for CModule.
func (cm *CModule) Forward(tensor *Tensor) (*Tensor, error) {

	var tensors []Tensor = []Tensor{*tensor}
	return cm.ForwardTs(tensors)
}

// Tensor methods for CModule:
// ======================================

// Apply forwards tensor itself through a module.
func (ts *Tensor) ApplyCModule(m *CModule) *Tensor {
	retVal, err := m.Forward(ts)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}
