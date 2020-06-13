package main

import (
	"fmt"
	"log"

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

	// data := []int16{1, 1, 1, 2, 2, 2, 3, 3}
	// shape := []int64{1, 8}

	ts, err := tensor.NewTensorFromData(data, shape)
	if err != nil {
		log.Fatal(err)
	}

	ts.Print()

	numel := uint(6)
	// dst := make([]uint8, numel)
	var dst = make([]int64, 6)

	ts.MustCopyData(dst, numel)

	fmt.Println(dst)

}
