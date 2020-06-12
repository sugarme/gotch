package wrapper

import (
	"fmt"
	"log"
	"reflect"
)

type Iterator interface {
	Next() interface{}
}

type Iterable struct {
	Index    int64
	Len      int64
	Content  Tensor
	ItemKind reflect.Kind
}

// Next implements Iterator interface
func (it *Iterable) Next() (retVal interface{}) {
	var err error
	switch it.ItemKind {
	case reflect.Int64:
		retVal, err = it.Content.Int64Value([]int64{it.Index})
		if err != nil {
			log.Fatal(err)
		}
		it.Index += 1
	case reflect.Float64:
		retVal, err = it.Content.Float64Value([]int64{it.Index})
		if err != nil {
			log.Fatal(err)
		}
		it.Index += 1
	default:
		err := fmt.Errorf("Iterator error: unsupported item kind (%v).\n", it.ItemKind)
		log.Fatal(err)
	}

	return retVal
}

// Iter creates an iterable object with specified item type.
func (ts Tensor) Iter(kind reflect.Kind) (retVal Iterable, err error) {
	num, err := ts.Size1() // size for 1D tensor
	if err != nil {
		return retVal, err
	}
	content, err := ts.ShallowClone()
	if err != nil {
		return retVal, err
	}

	return Iterable{
		Index:    0,
		Len:      num,
		Content:  content,
		ItemKind: kind,
	}, nil
}
