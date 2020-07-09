package main

import (
	// "github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func main() {
	data := [][]int64{
		{1, 1, 1, 2, 2, 2, 3, 3},
		{1, 1, 1, 2, 2, 2, 4, 4},
	}
	// shape := []int64{2, 8}
	shape := []int64{2, 2, 4}

	t, err := ts.NewTensorFromData(data, shape)
	if err != nil {
		panic(err)
	}

	t.Print()

	idx := ts.NewNarrow(0, 3)

	selTs := t.Idx(idx)
	selTs.Print()
}
