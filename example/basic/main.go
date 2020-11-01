package main

import (
	// "fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func main() {

	// Create a tensor [2,3,4]
	tensor := ts.MustArange(ts.IntScalar(2*3*4), gotch.Int64, gotch.CPU).MustView([]int64{2, 3, 4}, true)
	tensor.Print()

	mul := ts.MustOnes([]int64{4, 5}, gotch.Int64, gotch.CPU)
	res := tensor.MustMatmul(mul, false)

	res.Print()
}
