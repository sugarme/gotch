package main

import (
	// "fmt"
	"log"

	// gotch "github.com/sugarme/gotch"
	wrapper "github.com/sugarme/gotch/wrapper"
)

func main() {

	// TODO: Check Go type of data and tensor DType
	// For. if data is []int and DType is Bool
	// It is still running but get wrong result.
	data := []int32{1, 1, 1, 2, 2, 2}
	shape := []int64{2, 3}

	// dtype := gotch.Int
	// ts := wrapper.NewTensor()
	// sliceTensor, err := ts.FOfSlice(data, dtype)
	// if err != nil {
	// log.Fatal(err)
	// }

	ts, err := wrapper.NewTensorFromData(data, shape)
	if err != nil {
		log.Fatal(err)
	}

	ts.Print()

}
