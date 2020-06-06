package main

import (
	"fmt"
	"log"
	"time"

	gotch "github.com/sugarme/gotch"
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

	sz, err := ts.Size2()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Shape: %v\n", sz)

	fmt.Printf("DType: %v\n", ts.DType())

	dx := [][]float64{
		{1, 1},
		{1, 1},
		{1, 1},
	}

	dy := [][]float64{
		{1, 2, 3},
		{1, 1, 1},
	}

	xs, err := wrapper.NewTensorFromData(dx, []int64{3, 2})
	if err != nil {
		log.Fatal(err)
	}
	ys, err := wrapper.NewTensorFromData(dy, []int64{2, 3})
	if err != nil {
		log.Fatal(err)
	}

	// CPU
	startCPUTime := time.Now()
	for i := 1; i < 100000; i++ {
		xs.Matmul(ys)
	}
	fmt.Printf("CPU time: %v\n", time.Since(startCPUTime))

	// Cuda
	device := gotch.NewCuda()
	startGPUTime := time.Now()
	for i := 1; i < 100000; i++ {
		cx, err := xs.To(device)
		if err != nil {
			log.Fatal(err)
		}
		cy, err := ys.To(device)
		if err != nil {
			log.Fatal(err)
		}
		cx.Matmul(cy)
	}

	fmt.Printf("GPU time: %v\n", time.Since(startGPUTime))
}
