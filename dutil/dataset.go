package dutil

import (
	"fmt"
	"reflect"
)

// Dataset represents a set of samples and
// how to access a sample by its index by implementing
// `Item()` method.
type Dataset interface {
	Item(idx int) (interface{}, error)
	DType() reflect.Type
	Len() int
}

type DatasetKind int

const (
	SliceDKind DatasetKind = iota
	MapDKind
	InvalidDKind
)

// SliceDataset is a slice of samples.
type SliceDataset struct {
	data interface{}
}

// NewSliceDataset creates a new SliceDataset.
func NewSliceDataset(data interface{}) (*SliceDataset, error) {
	kind := reflect.TypeOf(data).Kind().String()
	if kind != "slice" {
		err := fmt.Errorf("Invalid Type: expected data of slice type. Got '%v'.\n", kind)
		return nil, err
	}
	return &SliceDataset{
		data: data,
	}, nil
}

// Item implements Dataset interface to get a sample by its index.
func (ds *SliceDataset) Item(idx int) (interface{}, error) {
	if idx < 0 || idx >= reflect.ValueOf(ds.data).Len() {
		err := fmt.Errorf("Idx is out of range.")
		return nil, err
	}

	return reflect.ValueOf(ds.data).Index(idx).Interface(), nil
}

func (ds *SliceDataset) Len() int {
	return reflect.ValueOf(ds.data).Len()
}

func (ds *SliceDataset) DType() reflect.Type {
	return reflect.TypeOf(ds.data)
}

// MapDataset holds samples in a map.
type MapDataset struct {
	// data map[string]interface{}
	data interface{}
	keys []string // keys to access elements in map
}

// NewMapDataset creates a new MapDataset.
// func NewMapDataset(data map[string]interface{}) *MapDataset {
func NewMapDataset(data interface{}) (*MapDataset, error) {
	// validate map type
	dtype := reflect.TypeOf(data).Kind().String()
	if dtype != "map" {
		err := fmt.Errorf("Expected data of map type. Got: %v\n", dtype)
		return nil, err
	}

	// validate key string type
	keyType := reflect.TypeOf(data).Key().Kind().String()
	if keyType != "string" {
		err := fmt.Errorf("Expected data with map key of string type. Got '%v'\n", keyType)
		return nil, err
	}

	var keys []string
	mapIter := reflect.ValueOf(data).MapRange()
	for mapIter.Next() {
		key := mapIter.Key().Interface()
		keys = append(keys, key.(string))
	}

	return &MapDataset{
		data: data,
		keys: keys,
	}, nil
}

// Item implements Dataset interface.
func (ds *MapDataset) Item(idx int) (interface{}, error) {
	if idx < 0 || idx >= len(ds.keys) {
		err := fmt.Errorf("idx is out of range.")
		return nil, err
	}

	key := ds.keys[idx]
	item := reflect.ValueOf(ds.data).MapIndex(reflect.ValueOf(key)).Interface()
	return item, nil
}

func (ds *MapDataset) Len() int {
	return reflect.ValueOf(ds.data).Len()
}

func (ds *MapDataset) DType() reflect.Type {
	return reflect.TypeOf(ds.data)
}

// NOTE. To make this package agnostic, we don't add TensorDataset here.
// A end-user can create a custom dataset by implementing `Item()` method.
