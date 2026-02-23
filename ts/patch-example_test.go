package ts_test

import (
	"testing"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

func ExampleTensor_Split() {
	tensor := ts.MustArange(ts.FloatScalar(10), gotch.Float, gotch.CPU).MustView([]int64{5, 2}, true)
	splitTensors := tensor.MustSplit(2, 0, false)

	for _, st := range splitTensors {
		st.Print()
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

func ExampleTensor_SplitWithSizes() {
	tensor := ts.MustArange(ts.FloatScalar(10), gotch.Float, gotch.CPU).MustView([]int64{5, 2}, true)
	splitTensors := tensor.MustSplitWithSizes([]int64{1, 4}, 0, false)

	for _, st := range splitTensors {
		st.Print()
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

// Test Unbind op specific for BFloat16/Half
func TestTensorUnbind(t *testing.T) {
	// device := gotch.CudaIfAvailable()
	device := gotch.CPU

	dtype := gotch.BFloat16
	// dtype := gotch.Half // <- NOTE. Libtorch API Error: "arange_cpu" not implemented for 'Half'

	x := ts.MustArange(ts.IntScalar(60), dtype, device).MustView([]int64{3, 4, 5}, true)

	out := x.MustUnbind(0, true)

	if len(out) != 3 {
		t.Errorf("Want 3, got %v\n", len(out))
	}
}
