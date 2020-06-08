package main

import (
	"fmt"
	"log"

	wrapper "github.com/sugarme/gotch/wrapper"
)

func main() {

	// TODO: Check Go type of data and tensor DType
	// For. if data is []int and DType is Bool
	// It is still running but get wrong result.
	// data := [][]int16{
	// {1, 1, 1, 2, 2, 2, 3, 3},
	// {1, 1, 1, 2, 2, 2, 4, 4},
	// }
	// shape := []int64{2, 8}

	data := []int16{1, 1, 1, 2, 2, 2, 3, 3}
	shape := []int64{1, 8}

	ts, err := wrapper.NewTensorFromData(data, shape)
	if err != nil {
		log.Fatal(err)
	}

	ts.Print()

	numel := uint(11)
	// dst := make([]uint8, numel)
	var dst = make([]uint8, 1)

	ts.MustCopyData(dst, numel)

	fmt.Println(dst)

}
