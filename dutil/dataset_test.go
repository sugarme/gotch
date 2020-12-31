package dutil_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/gotch/dutil"
)

func TestNewSliceDataset(t *testing.T) {
	// Error case: non `slice` type
	invalidData := 1
	_, err := dutil.NewSliceDataset(invalidData)
	if err == nil {
		t.Errorf("Expected invalid data type error: %v.\n", err)
	}

	// Valid case
	validData := []int{0, 1, 2, 3}
	_, err = dutil.NewSliceDataset(validData)
	if err != nil {
		t.Errorf("Unexpected error. Got: %v.\n", err)
	}
}

func TestSliceDataset_Len(t *testing.T) {
	data := []int{0, 1, 2, 3}
	ds, err := dutil.NewSliceDataset(data)
	if err != nil {
		t.Error(err)
	}

	want := 4
	got := ds.Len()

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want data length: %v\n", want)
		t.Errorf("Got data length: %v\n", got)
	}
}

func TestSliceDataset_Item(t *testing.T) {
	data := []int{0, 1, 2, 3}
	ds, err := dutil.NewSliceDataset(data)
	if err != nil {
		t.Error(err)
	}

	want := 2
	got, err := ds.Item(2)
	if err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want item value: %v\n", want)
		t.Errorf("Got item value: %v\n", got)
	}
}

func TestNewMapDataset(t *testing.T) {
	// Invalid data type
	invalidData := []int{0, 1, 2, 3}
	_, err := dutil.NewMapDataset(invalidData)
	if err == nil {
		t.Errorf("Expected Invalid data type. Got nil.")
	}

	// Invalid map key type
	invalidKey := make(map[int]int, 0)
	invalidKey[1] = 1
	invalidKey[2] = 2
	_, err = dutil.NewMapDataset(invalidKey)
	if err == nil {
		t.Errorf("Expected Invalid map key type. Got nil.")
	}

	// Valid data
	validData := make(map[string]int)
	validData["one"] = 1
	validData["two"] = 2
	_, err = dutil.NewMapDataset(validData)
	if err != nil {
		t.Errorf("Unexpected error. Got: %v\n", err)
	}
}

func TestMaptDataset_Len(t *testing.T) {
	var data map[string]int = map[string]int{"one": 1, "two": 2}
	ds, err := dutil.NewMapDataset(data)
	if err != nil {
		t.Error(err)
	}

	want := 2
	got := ds.Len()
	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want data length: %v\n", want)
		t.Errorf("Got data length: %v\n", got)
	}
}

func TestMapDataset_Item(t *testing.T) {
	var data map[string]int = map[string]int{"three": 3, "one": 1, "two": 2}
	ds, err := dutil.NewMapDataset(data)
	if err != nil {
		t.Error(err)
	}

	want := 3
	got, err := ds.Item(0)
	if err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}
}
