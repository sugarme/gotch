package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func main() {

	tensor := ts.MustArange1(ts.IntScalar(0), ts.IntScalar(2*3), gotch.Int64, gotch.CPU).MustView([]int64{2, 3}, true)

	var idxs []ts.TensorIndexer = []ts.TensorIndexer{
		// ts.NewNarrow(0, tensor.MustSize()[0]),
		// ts.NewNarrow(0, tensor.MustSize()[1]),
		ts.NewInsertNewAxis(),
	}

	result := tensor.Idx(idxs)

	fmt.Printf("Original Ts shape: %v\n", tensor.MustSize())
	fmt.Printf("Result Ts shape: %v\n", result.MustSize())

}
