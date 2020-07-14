package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func main() {

	// Create a tensor [2,3,4]
	tensor := ts.MustArange(ts.IntScalar(2*3*4), gotch.Int64, gotch.CPU).MustView([]int64{2, 3, 4}, true)

	tensor.Print()

	nilTs := ts.NewTensor()

	fmt.Printf("nilTs val: %v", nilTs)

}
