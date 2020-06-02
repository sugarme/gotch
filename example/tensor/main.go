package main

import (
	"fmt"
	"log"

	// gotch "github.com/sugarme/gotch"
	wrapper "github.com/sugarme/gotch/wrapper"
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
	// shape := []int64{2, 2, 4}

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

	// fmt.Printf("Dim: %v\n", ts.Dim())

	// ts.Size()
	// fmt.Println(ts.Size())

	sz, err := ts.Size2()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Shape: %v\n", sz)

	// typ, count, err := wrapper.DataCheck(data)
	// if err != nil {
	// log.Fatal(err)
	// }
	//
	// fmt.Printf("typ: %v\n", typ)
	// fmt.Printf("Count: %v\n", count)

	fmt.Printf("DType: %v\n", ts.DType())

}
