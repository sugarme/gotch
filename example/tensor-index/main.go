package main

import (
	wrapper "github.com/sugarme/gotch/wrapper"
)

func main() {
	data := [][]int64{
		{1, 1, 1, 2, 2, 2, 3, 3},
		{1, 1, 1, 2, 2, 2, 4, 4},
	}
	shape := []int64{2, 8}
	// shape := []int64{2, 2, 4}

	ts, err := wrapper.NewTensorFromData(data, shape)
	if err != nil {
		panic(err)
	}

	ts.Print()

	// Select
	s := wrapper.NewSelect(7)
	// selectedTs := ts.Idx(s)
	// selectedTs.Print()

	// Narrow (start inclusive, end exclusive)
	n := wrapper.NewNarrow(0, 1)
	// narrowedTs := ts.Idx(n)
	// narrowedTs.Print()

	// InsertNewAxis
	// i := wrapper.NewInsertNewAxis()
	// newAxisTs := ts.Idx(i)
	// newAxisTs.Print()

	// IndexSelect
	// idxTensor := wrapper.MustOfSlice([]int64{0, 1})
	// is := wrapper.NewIndexSelect(idxTensor)
	// isTs := ts.Idx(is)
	// isTs.Print()

	// Combined
	var tsIndexes []wrapper.TensorIndexer = []wrapper.TensorIndexer{n, s}
	combinedTs := ts.Idx(tsIndexes)

	combinedTs.Print()

}
