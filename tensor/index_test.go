package tensor_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func TestNewInsertAxis(t *testing.T) {

	tensor := ts.MustArange1(ts.IntScalar(0), ts.IntScalar(2*3), gotch.Int64, gotch.CPU).MustView([]int64{2, 3}, true)

	var idxs1 []ts.TensorIndexer = []ts.TensorIndexer{
		ts.NewInsertNewAxis(),
	}

	result1 := tensor.Idx(idxs1)

	want1 := []int64{1, 2, 3}
	got1 := result1.MustSize()

	if !reflect.DeepEqual(want1, got1) {
		t.Errorf("Expected a tensor shape: %v\n", want1)
		t.Errorf("Got a tensor shape: %v\n", got1)
	}

	var idxs2 []ts.TensorIndexer = []ts.TensorIndexer{
		ts.NewNarrow(0, tensor.MustSize()[0]),
		ts.NewInsertNewAxis(),
	}

	result2 := tensor.Idx(idxs2)

	want2 := []int64{2, 1, 3}
	got2 := result2.MustSize()

	if !reflect.DeepEqual(want2, got2) {
		t.Errorf("Expected a tensor shape: %v\n", want2)
		t.Errorf("Got a tensor shape: %v\n", got2)
	}

	var idxs3 []ts.TensorIndexer = []ts.TensorIndexer{
		ts.NewNarrow(0, tensor.MustSize()[0]),
		ts.NewNarrow(0, tensor.MustSize()[1]),
		ts.NewInsertNewAxis(),
	}

	result3 := tensor.Idx(idxs3)

	want3 := []int64{2, 3, 1}
	got3 := result3.MustSize()

	if !reflect.DeepEqual(want3, got3) {
		t.Errorf("Expected a tensor shape: %v\n", want3)
		t.Errorf("Got a tensor shape: %v\n", got3)
	}
}
