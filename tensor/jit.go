package tensor

// JIT interface to run model trained/saved using PyTorch Python API.

import (
	"fmt"
	"log"
	"reflect"

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
		fmt.Printf("cval: %v\n", cval)
		if err = TorchErr(); err != nil {
			return retVal, err
		}

		return CIValue{civalue: cval}, nil

		// TODO: continue...
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

		// TODO: continue

	}

	return
}
