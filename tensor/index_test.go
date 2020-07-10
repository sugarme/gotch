package tensor_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func TestIntegerIndex(t *testing.T) {
	// [ 0 1 2
	//   3 4 5 ]
	tensor := ts.MustArange1(ts.IntScalar(0), ts.IntScalar(2*3), gotch.Int64, gotch.CPU).MustView([]int64{2, 3}, true)
	// tensor, err := ts.NewTensorFromData([]bool{true, false, false, false, false, false}, []int64{2, 3})
	// if err != nil {
	// panic(err)
	// }
	idx1 := []ts.TensorIndexer{
		ts.NewSelect(1),
	}
	result1 := tensor.Idx(idx1)
	want1 := []int64{3, 4, 5}
	want1Shape := []int64{3}
	got1 := result1.Vals()
	got1Shape := result1.MustSize()
	if !reflect.DeepEqual(want1, got1) {
		t.Errorf("Expected tensor values: %v\n", want1)
		t.Errorf("Got tensor values: %v\n", got1)
	}
	if !reflect.DeepEqual(want1Shape, got1Shape) {
		t.Errorf("Expected tensor values: %v\n", want1Shape)
		t.Errorf("Got tensor values: %v\n", got1Shape)
	}

	idx2 := []ts.TensorIndexer{
		ts.NewNarrow(0, tensor.MustSize()[0]),
		ts.NewSelect(2),
	}
	result2 := tensor.Idx(idx2)
	want2 := []int64{2, 5}
	want2Shape := []int64{2}
	got2 := result2.Vals()
	got2Shape := result2.MustSize()
	if !reflect.DeepEqual(want2, got2) {
		t.Errorf("Expected tensor values: %v\n", want2)
		t.Errorf("Got tensor values: %v\n", got2)
	}
	if !reflect.DeepEqual(want2Shape, got2Shape) {
		t.Errorf("Expected tensor values: %v\n", want2Shape)
		t.Errorf("Got tensor values: %v\n", got2Shape)
	}

	idx3 := []ts.TensorIndexer{
		ts.NewNarrow(0, tensor.MustSize()[0]),
		ts.NewSelect(-2),
	}
	result3 := tensor.Idx(idx3)
	want3 := []int64{1, 4}
	want3Shape := []int64{2}
	got3 := result3.Vals()
	got3Shape := result3.MustSize()
	if !reflect.DeepEqual(want3, got3) {
		t.Errorf("Expected tensor values: %v\n", want3)
		t.Errorf("Got tensor values: %v\n", got3)
	}
	if !reflect.DeepEqual(want3Shape, got3Shape) {
		t.Errorf("Expected tensor values: %v\n", want3Shape)
		t.Errorf("Got tensor values: %v\n", got3Shape)
	}
}

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

func TestRangeIndex(t *testing.T) {

	// Range
	tensor1 := ts.MustArange1(ts.IntScalar(0), ts.IntScalar(4*3), gotch.Int64, gotch.CPU).MustView([]int64{4, 3}, true)
	idx1 := []ts.TensorIndexer{
		ts.NewNarrow(1, 3),
	}
	result1 := tensor1.Idx(idx1)
	want1 := []int64{3, 4, 5, 6, 7, 8}
	want1Shape := []int64{2, 3}
	got1 := result1.Vals()
	got1Shape := result1.MustSize()
	if !reflect.DeepEqual(want1, got1) {
		t.Errorf("Expected tensor values: %v\n", want1)
		t.Errorf("Got tensor values: %v\n", got1)
	}
	if !reflect.DeepEqual(want1Shape, got1Shape) {
		t.Errorf("Expected tensor values: %v\n", want1Shape)
		t.Errorf("Got tensor values: %v\n", got1Shape)
	}

	// Full range
	tensor2 := ts.MustArange1(ts.IntScalar(0), ts.IntScalar(2*3), gotch.Int64, gotch.CPU).MustView([]int64{2, 3}, true)
	idx2 := []ts.TensorIndexer{
		ts.NewNarrow(0, tensor2.MustSize()[0]),
	}
	result2 := tensor2.Idx(idx2)
	want2 := []int64{0, 1, 2, 3, 4, 5}
	want2Shape := []int64{2, 3}
	got2 := result2.Vals()
	got2Shape := result2.MustSize()
	if !reflect.DeepEqual(want2, got2) {
		t.Errorf("Expected tensor values: %v\n", want2)
		t.Errorf("Got tensor values: %v\n", got2)
	}
	if !reflect.DeepEqual(want2Shape, got2Shape) {
		t.Errorf("Expected tensor values: %v\n", want2Shape)
		t.Errorf("Got tensor values: %v\n", got2Shape)
	}

	// Range from
	tensor3 := ts.MustArange1(ts.IntScalar(0), ts.IntScalar(4*3), gotch.Int64, gotch.CPU).MustView([]int64{4, 3}, true)
	idx3 := []ts.TensorIndexer{
		ts.NewNarrow(2, tensor3.MustSize()[0]),
	}
	result3 := tensor3.Idx(idx3)
	want3 := []int64{6, 7, 8, 9, 10, 11}
	want3Shape := []int64{2, 3}
	got3 := result3.Vals()
	got3Shape := result3.MustSize()
	if !reflect.DeepEqual(want3, got3) {
		t.Errorf("Expected tensor values: %v\n", want3)
		t.Errorf("Got tensor values: %v\n", got3)
	}
	if !reflect.DeepEqual(want3Shape, got3Shape) {
		t.Errorf("Expected tensor values: %v\n", want3Shape)
		t.Errorf("Got tensor values: %v\n", got3Shape)
	}

	// Range to
	tensor4 := ts.MustArange1(ts.IntScalar(0), ts.IntScalar(4*3), gotch.Int64, gotch.CPU).MustView([]int64{4, 3}, true)
	idx4 := []ts.TensorIndexer{
		ts.NewNarrow(0, 2),
	}
	result4 := tensor4.Idx(idx4)
	want4 := []int64{0, 1, 2, 3, 4, 5}
	want4Shape := []int64{2, 3}
	got4 := result4.Vals()
	got4Shape := result4.MustSize()
	if !reflect.DeepEqual(want4, got4) {
		t.Errorf("Expected tensor values: %v\n", want4)
		t.Errorf("Got tensor values: %v\n", got4)
	}
	if !reflect.DeepEqual(want4Shape, got4Shape) {
		t.Errorf("Expected tensor values: %v\n", want4Shape)
		t.Errorf("Got tensor values: %v\n", got4Shape)
	}
}

func TestSliceIndex(t *testing.T) {
	// TODO
}
