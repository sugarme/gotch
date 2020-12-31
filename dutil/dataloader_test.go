package dutil_test

import (
	// "reflect"
	"reflect"
	"testing"

	"github.com/sugarme/gotch/dutil"
)

func TestNewDataLoader(t *testing.T) {
	data, err := dutil.NewSliceDataset([]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	if err != nil {
		t.Error(err)
	}

	_, err = dutil.NewDataLoader(data, nil)
	if err != nil {
		t.Errorf("Unexpected error. Got: %v\n", err)
	}
}

func TestDataLoader_Next(t *testing.T) {
	data, err := dutil.NewSliceDataset([]int{100, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	if err != nil {
		t.Error(err)
	}

	dl, err := dutil.NewDataLoader(data, nil)
	if err != nil {
		t.Errorf("Unexpected error. Got: %v\n", err)
	}

	got, err := dl.Next()
	if err != nil {
		t.Error(err)
	}
	want := 100

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}
}
