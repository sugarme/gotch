package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/tensor"
)

func main() {

	// TODO: Check Go type of data and tensor DType
	// For. if data is []int and DType is Bool
	// It is still running but get wrong result.
	data := [][]int64{
		{1, 1, 1, 2, 2, 2, 3, 3},
		{1, 1, 1, 2, 2, 2, 4, 4},
	}
	shape := []int64{2, 8}

	ts, err := tensor.NewTensorFromData(data, shape)
	if err != nil {
		log.Fatal(err)
	}

	ts, err = ts.To(gotch.CPU)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Tensor value BEFORE: %v\n", ts)
	ts.Print()

	scalarVal := tensor.IntScalar(int64(5))

	ts.Fill_(scalarVal)

	fmt.Printf("Tensor value AFTER: %v\n", ts)
	ts.Print()
}
