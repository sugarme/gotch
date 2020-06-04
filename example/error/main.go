package main

import (
	"fmt"
	"log"

	wrapper "github.com/sugarme/gotch/wrapper"
)

func main() {

	// Try to compare 2 tensor with incompatible dimensions
	// and check this returns an error
	dx := []int32{1, 2, 3}
	// dy := []int32{1, 2, 3, 4}
	dy := []int32{1, 2, 5}

	xs, err := wrapper.OfSlice(dx)
	if err != nil {
		log.Fatal(err)
	}
	ys, err := wrapper.OfSlice(dy)
	if err != nil {
		log.Fatal(err)
	}

	xs.Print()
	ys.Print()

	fmt.Printf("xs num of dimensions: %v\n", xs.Dim())
	fmt.Printf("ys num of dimensions: %v\n", ys.Dim())

	fmt.Printf("xs shape: %v\n", xs.Size())
	fmt.Printf("ys shape: %v\n", ys.Size())

	xs.Eq1(*ys)
}
