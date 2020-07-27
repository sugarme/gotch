package tensor_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func TestTensorInit(t *testing.T) {
	tensor := ts.MustArange1(ts.IntScalar(1), ts.IntScalar(5), gotch.Int64, gotch.CPU)

	want := []float64{1, 2, 3, 4}
	got := tensor.Float64Values()

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Expected tensor values: %v\n", want)
		t.Errorf("Got tensor values: %v\n", got)
	}
}

func TestInplaceAssign(t *testing.T) {
	tensor := ts.MustOfSlice([]int64{3, 1, 4, 1, 5})

	tensor.MustAdd1_(ts.IntScalar(1))
	tensor.MustMul1_(ts.IntScalar(2))
	tensor.MustSub1_(ts.IntScalar(1))

	want := []int64{7, 3, 9, 3, 11}
	got := tensor.Vals()

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Expected tensor values: %v\n", want)
		t.Errorf("Got tensor values: %v\n", got)
	}
}

func TestConstantOp(t *testing.T) {
	tensor := ts.MustOfSlice([]int64{3, 9, 3, 11})
	resTs1 := tensor.MustMul1(ts.IntScalar(-1), true)

	want1 := []int64{-3, -9, -3, -11}
	got1 := resTs1.Vals()
	if !reflect.DeepEqual(want1, got1) {
		t.Errorf("Expected tensor values: %v\n", want1)
		t.Errorf("Got tensor values: %v\n", got1)
	}

	// TODO: more ops

}

func TestIter(t *testing.T) {

	tensor := ts.MustOfSlice([]int64{3, 9, 3, 11})

	iter, err := tensor.Iter(gotch.Int64)
	if err != nil {
		panic(err)
	}

	want := []int64{3, 9, 3, 11}
	var got []int64

	for {
		item, ok := iter.Next()
		if !ok {
			break
		}
		got = append(got, item.(int64))
	}

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Expected tensor values: %v\n", want)
		t.Errorf("Got tensor values: %v\n", got)
	}

	tensor1 := ts.MustOfSlice([]float64{3.14, 15.926, 5.3589, 79.0})
	iter1, err := tensor1.Iter(gotch.Double)

	want1 := []float64{3.14, 15.926, 5.3589, 79.0}
	var got1 []float64
	for {
		item, ok := iter1.Next()
		if !ok {
			break
		}

		got1 = append(got1, item.(float64))
	}

	if !reflect.DeepEqual(want1, got1) {
		t.Errorf("Expected tensor values: %v\n", want1)
		t.Errorf("Got tensor values: %v\n", got1)
	}
}

func TestOnehot(t *testing.T) {
	xs := ts.MustOfSlice([]int64{0, 1, 2, 3}).MustView([]int64{2, 2}, true)
	onehot := xs.Onehot(4)

	want := []float64{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}
	got := onehot.Float64Values()

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Expected onehot tensor values: %v\n", want)
		t.Errorf("Got onehot tensor values: %v\n", got)
	}

}

/*
 *     let xs = Tensor::of_slice(&[0, 1, 2, 3]);
 *     let onehot = xs.onehot(4);
 *     assert_eq!(
 *         Vec::<f64>::from(&onehot),
 *         vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
 *     );
 *     assert_eq!(onehot.size(), vec![4, 4]) */
