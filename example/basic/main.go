package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

func main() {
	// intTensor()
	floatTensor()
}

func intTensor() {
	xs := ts.MustArange(ts.IntScalar(7*3*4*5*6), gotch.Int64, gotch.CPU).MustView([]int64{7, 3, 4, 5, 6}, true)
	fmt.Printf("%4d\n", xs)
}

func floatTensor() {
	// xs := ts.MustRand([]int64{7, 3, 4, 5, 6}, gotch.Double, gotch.CPU)
	xs := ts.MustRand([]int64{3, 5, 6}, gotch.Float, gotch.CPU)
	fmt.Printf("%8.3f\n", xs)
	fmt.Printf("%i", xs)
}
