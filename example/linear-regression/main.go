package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func main() {

	// mockup data
	var (
		n      int = 20
		xvals  []float32
		yvals  []float32
		epochs = 10
	)

	for i := 0; i < n; i++ {
		xvals = append(xvals, float32(i))
		yvals = append(yvals, float32(2*i+1))
	}

	xtrain, err := ts.NewTensorFromData(xvals, []int64{int64(n), 1})
	if err != nil {
		log.Fatal(err)
	}
	ytrain, err := ts.NewTensorFromData(yvals, []int64{int64(n), 1})
	if err != nil {
		log.Fatal(err)
	}

	ws := ts.MustZeros([]int64{1, int64(n)}, gotch.Float.CInt(), gotch.CPU.CInt())
	bs := ts.MustZeros([]int64{1, int64(n)}, gotch.Float.CInt(), gotch.CPU.CInt())

	for epoch := 0; epoch < epochs; epoch++ {

		logit := ws.MustMatMul(xtrain).MustAdd(bs)
		loss := ts.NewTensor().MustLogSoftmax(-1, gotch.Float.CInt())

		ws.MustGrad()
		bs.MustGrad()
		loss.MustBackward()

	}
}
