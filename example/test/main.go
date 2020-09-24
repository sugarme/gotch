package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
	// "github.com/sugarme/gotch/nn"
)

func main() {
	// init := nn.NewKaimingUniformInit()
	//
	// tensor := init.InitTensor([]int64{10}, gotch.CPU)
	//
	// tensor.Print()

	tensor := ts.MustArange1(ts.FloatScalar(0), ts.FloatScalar(10), gotch.Float, gotch.CPU).MustView([]int64{5, 2}, true)

	tensor.Print()

	splitTensors := tensor.MustSplit(2, 0, false)

	fmt.Printf("length of splitTensors: %v\n", len(splitTensors))

	for _, t := range splitTensors {
		t.Print()
	}

	// SplitWithSizes
	splitTensors1 := tensor.MustSplitWithSizes([]int64{1, 4}, 0, false)
	fmt.Printf("length of splitTensors1: %v\n", len(splitTensors))

	for _, t := range splitTensors1 {
		t.Print()
	}
}
