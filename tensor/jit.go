package tensor

// JIT interface to run model trained/saved using PyTorch Python API.

// #include "stdlib.h"
import "C"

import (
	"fmt"
	"log"
	"reflect"
	"unsafe"

	// "github.com/sugarme/gotch"
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
)

type IValue struct {
	value interface{}
	kind  IValueKind
	name  string
}

// NewIValue creates a new IValue from given value of various types.
func NewIValue(v interface{}) (retVal IValue) {

	retVal = IValue{value: v}
	if v == nil {
		retVal.kind = NoneVal
		retVal.name = "None"
		return retVal
	}

	switch reflect.TypeOf(v).Kind().String() {
	case "Tensor":
		retVal.kind = TensorVal
		retVal.name = "Tensor"
	case "float64":
		retVal.kind = DoubleVal
		retVal.name = "Double"
	case "int64":
		retVal.kind = IntVal
		retVal.name = "Int"
	case "int":
		retVal.value = int64(v.(int))
		retVal.kind = IntVal
		retVal.name = "Int"
	case "int32":
		retVal.value = int64(v.(int32))
		retVal.kind = IntVal
		retVal.name = "Int"
	case "bool":
		retVal.kind = BoolVal
		retVal.name = "Bool"
	case "string":
		retVal.kind = StringVal
		retVal.name = "String"
	case "slice":
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
		case "bool":
			retVal.kind = BoolListVal
			retVal.name = "BoolList"
		case "Tensor":
			retVal.kind = TensorListVal
			retVal.name = "TensorList"
		}
	case "map":
		// TODO: exclude map of type other than IValue type
		retVal.kind = GenericDictVal
		retVal.name = "GenericDict"
	default:
		log.Fatalf("NewIValue method call - Unsupport type(%v)\n", reflect.TypeOf(v).Kind().String())
	}

	return retVal
}

// IValue methods:
// ===============

func (iv IValue) ToCIValue() (retVal CIValue, err error) {

	switch iv.name {
	case "None":
		cval := lib.AtiNone()
		if err = TorchErr(); err != nil {
			return retVal, err
		}

		return CIValue{civalue: cval}, nil

	case "Tensor":
		cval := lib.AtiTensor(iv.value.(Tensor).ctensor)
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		return CIValue{civalue: cval}, nil

	case "Int":
		cval := lib.AtiInt(iv.value.(int64))
		if err = TorchErr(); err != nil {
			return retVal, err
		}

		return CIValue{civalue: cval}, nil

	case "Double":
		cval := lib.AtiDouble(iv.value.(float64))
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		return CIValue{civalue: cval}, nil

	case "Bool":
		cval := lib.AtiBool(iv.value.(bool))
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		return CIValue{civalue: cval}, nil

	case "Tuple":
		var v []IValue = iv.value.([]IValue)
		var cvals []lib.Civalue
		for _, i := range v {
			cval, err := i.ToCIValue()
			if err != nil {
				err = fmt.Errorf("ToCIValue method call err - Tuple case: %v\n", err)
				return retVal, err
			}
			cvals = append(cvals, cval.civalue)
		}

		tuple := lib.AtiTuple(cvals, len(cvals))
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		return CIValue{civalue: tuple}, nil

	case "GenericList":
		var v []IValue = iv.value.([]IValue)
		var cvals []lib.Civalue
		for _, i := range v {
			cval, err := i.ToCIValue()
			if err != nil {
				err = fmt.Errorf("ToCIValue method call err - GenericList case: %v\n", err)
				return retVal, err
			}
			cvals = append(cvals, cval.civalue)
		}

		list := lib.AtiGenericList(cvals, len(cvals))
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		return CIValue{civalue: list}, nil

	case "IntList":
		var vals []int64 = iv.value.([]int64)
		cval := lib.AtiIntList(vals, len(vals))
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		return CIValue{civalue: cval}, nil

	case "DoubleList":
		var vals []float64 = iv.value.([]float64)
		cval := lib.AtiDoubleList(vals, len(vals))
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		return CIValue{civalue: cval}, nil

	case "BoolList":
		var vals []bool = iv.value.([]bool)
		cval := lib.AtiBoolList(vals, len(vals))
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		return CIValue{civalue: cval}, nil

	case "TensorList":
		var vals []Tensor = iv.value.([]Tensor)
		var cvals []lib.Ctensor
		for _, i := range vals {
			cvals = append(cvals, i.ctensor)
		}
		list := lib.AtiTensorList(cvals, len(cvals))
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		return CIValue{civalue: list}, nil

	case "String":
		cval := lib.AtiString(iv.value.(string))
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		return CIValue{civalue: cval}, nil

	case "GenericDict":
		var m map[IValue]IValue = iv.value.(map[IValue]IValue)
		var vals []IValue
		for k, v := range m {
			vals = append(vals, k, v)
		}
		var cvals []lib.Civalue
		for _, v := range vals {
			cval, err := v.ToCIValue()
			if err != nil {
				err = fmt.Errorf("ToCIValue method call err - GenericList case: %v\n", err)
				return retVal, err
			}
			cvals = append(cvals, cval.civalue)
		}

		dict := lib.AtiGenericDict(cvals, len(cvals))
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		return CIValue{civalue: dict}, nil
	}

	return retVal, nil
}

// IValueFromC returns an IValue from given CIValue.
//
// It consumes the pointer and frees the associated memory.
func IValueFromC(cval CIValue) (retVal IValue, err error) {

	// tag will be a value of int32
	tag := lib.AtiTag(cval.civalue)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	fmt.Printf("tag value: %v\n", tag)

	switch tag {
	case 0:
		retVal = IValue{
			value: nil,
			kind:  NoneVal,
			name:  "None",
		}
	case 1:
		tensor := lib.AtiToTensor(cval.civalue)
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		retVal = IValue{
			value: tensor,
			kind:  TensorVal,
			name:  "Tensor",
		}
	case 2:
		v := lib.AtiToDouble(cval.civalue)
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		retVal = IValue{
			value: v,
			kind:  DoubleVal,
			name:  "Double",
		}
	case 3:
		v := lib.AtiToInt(cval.civalue)
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		retVal = IValue{
			value: v,
			kind:  IntVal,
			name:  "Int",
		}

	case 4:
		v := lib.AtiToBool(cval.civalue)
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		retVal = IValue{
			value: v,
			kind:  BoolVal,
			name:  "Bool",
		}

	case 5: // Tuple []IValue 2 elements
		// 1. Determine tuple length
		len := lib.AtiTupleLength(cval.civalue)
		if err = TorchErr(); err != nil {
			return retVal, err
		}
		// 2. Call with first pointer and length
		ptr1 := (*lib.Civalue)(unsafe.Pointer(C.malloc(0)))
		lib.AtiToTuple(cval.civalue, ptr1, int(len))
		if err = TorchErr(); err != nil {
			return retVal, err
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
		var vals []interface{}
		for _, civalue := range civalues {
			v, err := IValueFromC(civalue)
			if err != nil {
				return retVal, err
			}
			vals = append(vals, v)
		}

		retVal = IValue{
			value: vals,
			kind:  TupleVal,
			name:  "Tuple",
		}

	case 6: // IntList
		// 1. Len
		len := lib.AtiLength(cval.civalue)
		if err = TorchErr(); err != nil {
			return retVal, err
		}

		// 2. Call
		ptr1 := unsafe.Pointer(C.malloc(0))
		lib.AtiToIntList(cval.civalue, ptr1, int(len))
		if err = TorchErr(); err != nil {
			return retVal, err
		}

		// 3. Get int list
		var intVals []int64
		intVals = append(intVals, *(*int64)(unsafe.Pointer(ptr1)))
		fmt.Printf("intVal: %v\n", intVals)
		currPtr := ptr1
		for i := 1; i < int(len); i++ {
			nextPtr := unsafe.Pointer(uintptr(unsafe.Pointer(currPtr)) + unsafe.Sizeof(ptr1))
			intVals = append(intVals, *(*int64)(unsafe.Pointer(nextPtr)))
			currPtr = nextPtr
		}

		retVal = IValue{
			value: intVals,
			kind:  IntListVal,
			name:  "IntList",
		}

		// TODO: continue

	}

	return
}
