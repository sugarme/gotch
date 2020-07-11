package tensor

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
)

type Iterator interface {
	Next() (item interface{}, ok bool)
}

type Iterable struct {
	Index    int64
	Len      int64
	Content  Tensor
	ItemKind gotch.DType
}

// Next implements Iterator interface
func (it *Iterable) Next() (retVal interface{}, ok bool) {

	if it.Index == it.Len {
		return retVal, false
	}

	var err error
	switch it.ItemKind.Kind().String() {
	case "int64":
		retVal, err = it.Content.Int64Value([]int64{it.Index})
		if err != nil {
			log.Fatal(err)
		}
		it.Index += 1
	case "float64":
		retVal, err = it.Content.Float64Value([]int64{it.Index})
		if err != nil {
			log.Fatal(err)
		}
		it.Index += 1
	default:
		err := fmt.Errorf("Iterator error: unsupported item kind (%v).\n", it.ItemKind)
		log.Fatal(err)
	}

	return retVal, true
}

// Iter creates an iterable object with specified item type.
func (ts Tensor) Iter(dtype gotch.DType) (retVal Iterable, err error) {
	num, err := ts.Size1() // size for 1D tensor
	if err != nil {
		return retVal, err
	}
	tmp, err := ts.ShallowClone()
	if err != nil {
		return retVal, err
	}
	content := tmp.MustTotype(dtype, true)

	return Iterable{
		Index:    0,
		Len:      num,
		Content:  content,
		ItemKind: dtype,
	}, nil
}
