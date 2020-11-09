package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func main() {
	// intTensor()
	floatTensor()
}

func intTensor() {
	xs := ts.MustArange(ts.IntScalar(7*3*4*5*6), gotch.Int64, gotch.CPU).MustView([]int64{7, 3, 4, 5, 6}, true)
	fmt.Printf("%v\n", xs)
}

func floatTensor() {
	xs := ts.MustRand([]int64{7, 3, 4, 5, 6}, gotch.Double, gotch.CPU)
	fmt.Printf("%v\n", xs)
}
