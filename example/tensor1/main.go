package main

import (
	"fmt"
	"log"

	wrapper "github.com/sugarme/gotch/wrapper"
)

func main() {

	ts, err := wrapper.OfSlice([]float64{1.3, 29.7})
	if err != nil {
		log.Fatal(err)
	}

	res, err := ts.Float64Value([]int64{1})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(res)

	resInt64, err := ts.Int64Value([]int64{1})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resInt64)

	grad, err := ts.RequiresGrad()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Requires Grad: %v\n", grad)

	ele1, err := ts.DataPtr()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("First element address: %v\n", ele1)

	fmt.Printf("Number of tensor elements: %v\n", ts.Numel())

	clone := ts.MustShallowClone()
	clone.Print()

	atGet := ts.MustGet(1)
	atGet.Print() // 29.7

	atGet = ts.MustGet(0)
	atGet.Print() // 1.3

	dst, err := wrapper.NewTensorFromData([]int64{1, 2}, []int64{1, 2})
	if err != nil {
		panic(err)
	}

	dst = dst.MustTotype(ts.DType())

	wrapper.MustCopy_(dst, ts)
	dst.Print()

	ts.MustDrop()
	// The below statement will be panic as `ts` has been dropped.
	// ts.Print()

}
