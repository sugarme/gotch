package dutil

import (
	"fmt"
	"reflect"

	ts "github.com/sugarme/gotch/tensor"
)

// DataLoader combines a dataset and a sampler and provides
// an iterable over the given dataset.
type DataLoader struct {
	dataset   Dataset
	indexes   []int // order of samples in dataset for interation.
	batchSize int
	currIdx   int
	sampler   Sampler
}

func NewDataLoader(data Dataset, s Sampler) (*DataLoader, error) {
	dkind, err := checkDKind(data)
	if err != nil {
		return nil, err
	}

	// Use default Sampler if no specified
	if s == nil {
		switch dkind {
		case SliceDKind:
			s = NewSequentialSampler(data.Len())
		case MapDKind:
			s, err = NewRandomSampler(data.Len())
			if err != nil {
				return nil, err
			}
		}
	}

	return &DataLoader{
		dataset:   data,
		indexes:   s.Sample(),
		batchSize: s.BatchSize(),
		currIdx:   0,
		sampler:   s,
	}, nil
}

func checkDKind(data Dataset) (DatasetKind, error) {
	dtyp := data.DType()
	dkind := dtyp.Kind().String()

	switch dkind {
	case "slice":
		return SliceDKind, nil
	case "map":
		if dtyp.Key().Kind().String() != "string" {
			err := fmt.Errorf("Invalid Dataset. Dataset should be a collection data of type '[]interface{}' or 'map[string]interface{}'")
			return InvalidDKind, err
		}
		return MapDKind, nil

	default: // other types are invalid
		err := fmt.Errorf("Invalid Dataset. Dataset should be a collection data of type '[]interface{}' or 'map[string]interface{}'")
		return InvalidDKind, err
	}
}

// Next acts as iterator to return next sample(s) from dataset.
func (dl *DataLoader) Next() (interface{}, error) {
	if !dl.HasNext() {
		err := fmt.Errorf("Next Error: no more item to iterate.\n")
		return nil, err
	}

	// determine element dtype
	elem, err := dl.dataset.Item(0)
	if err != nil {
		return nil, err
	}

	elemType := reflect.TypeOf(elem)
	// Free up memory if element is Tensor
	switch elemType.String() {
	case "[]tensor.Tensor":
		for _, el := range elem.([]ts.Tensor) {
			el.MustDrop()
		}
	case "*tensor.Tensor":
		elem.(*ts.Tensor).MustDrop()
	}

	// Get a batch based on batch size
	items := reflect.MakeSlice(reflect.SliceOf(elemType), 0, dl.dataset.Len())
	nextIndex := dl.currIdx + dl.batchSize

	// NOTE. length of indexes can be shorter than dataset length
	if nextIndex >= len(dl.indexes) {
		nextIndex = len(dl.indexes)
	}
	for i := dl.currIdx; i < nextIndex; i++ {
		item, err := dl.dataset.Item(i)
		if err != nil {
			return nil, err
		}
		items = reflect.Append(items, reflect.ValueOf(item))
	}

	dl.currIdx = nextIndex
	return items.Interface(), nil
}

// HasNext returns whether there is a next item in the iteration.
func (dl *DataLoader) HasNext() bool {
	return dl.currIdx < len(dl.indexes)
}

// Reset reset index to start position.
func (dl *DataLoader) Reset(shuffleOpt ...bool) {
	shuffle := false
	if len(shuffleOpt) > 0 {
		shuffle = shuffleOpt[0]
	}
	if shuffle {
		dl.indexes = dl.sampler.Sample()
	}
	dl.currIdx = 0
}

// Len returns number of samples to be iterated.
func (dl *DataLoader) Len() int {
	return len(dl.indexes)
}
