package ts_test

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

func ExampleMustArange() {
	tensor := ts.MustArange(ts.FloatScalar(12), gotch.Int64, gotch.CPU).MustView([]int64{3, 4}, true)

	fmt.Printf("%v", tensor)

	// output
	// 0   1   2   3
	// 4   5   6   7
	// 8   9   10  11
}

func ExampleTensor_Matmul() {
	// Basic tensor operations
	ts1 := ts.MustArange(ts.IntScalar(6), gotch.Int64, gotch.CPU).MustView([]int64{2, 3}, true)
	defer ts1.MustDrop()
	ts2 := ts.MustOnes([]int64{3, 4}, gotch.Int64, gotch.CPU)
	defer ts2.MustDrop()

	mul := ts1.MustMatmul(ts2, false)
	defer mul.MustDrop()
	fmt.Println("ts1: ")
	ts1.Print()
	fmt.Println("ts2: ")
	ts2.Print()
	fmt.Println("mul tensor (ts1 x ts2): ")
	mul.Print()

	//ts1:
	// 0  1  2
	// 3  4  5
	//[ CPULongType{2,3} ]
	//ts2:
	// 1  1  1  1
	// 1  1  1  1
	// 1  1  1  1
	//[ CPULongType{3,4} ]
	//mul tensor (ts1 x ts2):
	//  3   3   3   3
	// 12  12  12  12
	//[ CPULongType{2,4} ]

}

func ExampleTensor_AddScalar() {
	// In-place operation
	ts3 := ts.MustOnes([]int64{2, 3}, gotch.Float, gotch.CPU)
	fmt.Println("Before:")
	ts3.Print()
	ts3.MustAddScalar_(ts.FloatScalar(2.0))
	fmt.Printf("After (ts3 + 2.0): \n")
	ts3.Print()
	ts3.MustDrop()

	//Before:
	// 1  1  1
	// 1  1  1
	//[ CPUFloatType{2,3} ]
	//After (ts3 + 2.0):
	// 3  3  3
	// 3  3  3
	//[ CPUFloatType{2,3} ]
}
