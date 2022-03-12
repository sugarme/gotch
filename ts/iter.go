package ts

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
	Content  *Tensor
	ItemKind gotch.DType
}

// Next implements Iterator interface
func (it *Iterable) Next() (item interface{}, ok bool) {

	if it.Index == it.Len {
		return nil, false
	}

	var err error
	switch it.ItemKind.Kind().String() {
	case "int64":
		item, err = it.Content.Int64Value([]int64{it.Index})
		if err != nil {
			log.Fatal(err)
		}
		it.Index += 1
	case "float64":
		item, err = it.Content.Float64Value([]int64{it.Index})
		if err != nil {
			log.Fatal(err)
		}
		it.Index += 1
	default:
		err := fmt.Errorf("Iterator error: unsupported item kind (%v).\n", it.ItemKind)
		log.Fatal(err)
	}

	return item, true
}

// Iter creates an iterable object with specified item type.
func (ts *Tensor) Iter(dtype gotch.DType) (*Iterable, error) {
	num, err := ts.Size1() // size for 1D tensor
	if err != nil {
		return nil, err
	}
	tmp, err := ts.ShallowClone()
	if err != nil {
		return nil, err
	}
	content := tmp.MustTotype(dtype, true)

	return &Iterable{
		Index:    0,
		Len:      num,
		Content:  content,
		ItemKind: dtype,
	}, nil
}
