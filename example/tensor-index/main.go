package main

import (
	"github.com/sugarme/gotch/tensor"
)

func main() {
	data := [][]int64{
		{1, 1, 1, 2, 2, 2, 3, 3},
		{1, 1, 1, 2, 2, 2, 4, 4},
	}
	shape := []int64{2, 8}
	// shape := []int64{2, 2, 4}

	ts, err := tensor.NewTensorFromData(data, shape)
	if err != nil {
		panic(err)
	}

	ts.Print()

	// Select
	s := tensor.NewSelect(7)
	// selectedTs := ts.Idx(s)
	// selectedTs.Print()

	// Narrow (start inclusive, end exclusive)
	n := tensor.NewNarrow(0, 1)
	// narrowedTs := ts.Idx(n)
	// narrowedTs.Print()

	// InsertNewAxis
	// i := tensor.NewInsertNewAxis()
	// newAxisTs := ts.Idx(i)
	// newAxisTs.Print()

	// IndexSelect
	// idxTensor := tensor.MustOfSlice([]int64{0, 1})
	// is := tensor.NewIndexSelect(idxTensor)
	// isTs := ts.Idx(is)
	// isTs.Print()

	// Combined
	var tsIndexes []tensor.TensorIndexer = []tensor.TensorIndexer{n, s}
	combinedTs := ts.Idx(tsIndexes)

	combinedTs.Print()

}
