package dutil

import (
	"fmt"
	"reflect"
)

// DataLoader combines a dataset and a sampler and provides
// an iterable over the given dataset.
type DataLoader struct {
	dataset   Dataset
	indexes   []int // order of samples in dataset for interation.
	batchSize int
	currIdx   int
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

	// Non-batching
	if dl.batchSize == 1 {
		item, err := dl.dataset.Item(dl.currIdx)
		if err != nil {
			return nil, err
		}

		dl.currIdx += 1
		return item, nil
	}

	// Batch sampling
	elem, err := dl.dataset.Item(0)
	if err != nil {
		return nil, err
	}

	elemType := reflect.TypeOf(elem)

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
func (dl *DataLoader) Reset() {
	dl.currIdx = 0
}
