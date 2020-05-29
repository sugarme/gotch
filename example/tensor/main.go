package main

import (
	"log"

	gotch "github.com/sugarme/gotch"
	wrapper "github.com/sugarme/gotch/wrapper"
)

func main() {

	// TODO: Check Go type of data and tensor DType
	// For. if data is []int and DType is Bool
	// It is still running but get wrong result.
	data := []bool{true, true, false}
	dtype := gotch.Bool

	ts := wrapper.NewTensor()
	sliceTensor, err := ts.FOfSlice(data, dtype)
	if err != nil {
		log.Fatal(err)
	}

	sliceTensor.Print()
}
