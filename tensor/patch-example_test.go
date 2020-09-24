package tensor_test

import (
	"testing"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func ExampleTensor_Split(t *testing.T) {
	tensor := ts.MustArange1(ts.FloatScalar(0), ts.FloatScalar(10), gotch.Float, gotch.CPU).MustView([]int64{5, 2}, true)
	splitTensors := tensor.MustSplit(2, 0, false)

	for _, t := range splitTensors {
		t.Print()
	}

	//Output:
	// 0  1
	// 2  3
	// [ CPUFloatType{2,2} ]
	// 4  5
	// 6  7
	// [ CPUFloatType{2,2} ]
	// 8  9
	// [ CPUFloatType{1,2} ]
}

func ExampleTensorSplitWithSizes(t *testing.T) {
	tensor := ts.MustArange1(ts.FloatScalar(0), ts.FloatScalar(10), gotch.Float, gotch.CPU).MustView([]int64{5, 2}, true)
	splitTensors := tensor.MustSplitWithSizes([]int64{1, 4}, 0, false)

	for _, t := range splitTensors {
		t.Print()
	}

	//Output:
	// 0  1
	// [ CPUFloatType{1,2} ]
	// 2  3
	// 4  5
	// 6  7
	// 8  9
	// [ CPUFloatType{4,2} ]
}
