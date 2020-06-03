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
	dy := []int32{1, 2, 3, 4}

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

	fmt.Printf("xs dim: %v\n", xs.Dim())
	fmt.Printf("ys dim: %v\n", ys.Dim())

}
