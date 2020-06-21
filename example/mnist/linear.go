package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

const (
	ImageDim int64  = 784
	Label    int64  = 10
	MnistDir string = "../../data/mnist"

	epochs = 200
)

func runLinear() {
	var ds vision.Dataset
	ds = vision.LoadMNISTDir(MnistDir)

	device := (gotch.CPU).CInt()
	dtype := (gotch.Float).CInt()

	ws := ts.MustZeros([]int64{ImageDim, Label}, dtype, device).MustSetRequiresGrad(true)
	bs := ts.MustZeros([]int64{Label}, dtype, device).MustSetRequiresGrad(true)

	for epoch := 0; epoch < epochs; epoch++ {

		logits := ds.TrainImages.MustMm(ws).MustAdd(bs)
		loss := logits.MustLogSoftmax(-1, dtype).MustNllLoss(ds.TrainLabels)

		ws.ZeroGrad()
		bs.ZeroGrad()
		loss.MustBackward()

		ts.NoGrad(func() {
			ws.Add_(ws.MustGrad().MustMul1(ts.FloatScalar(-1.0)))
			bs.Add_(bs.MustGrad().MustMul1(ts.FloatScalar(-1.0)))
		})

		testLogits := ds.TestImages.MustMm(ws).MustAdd(bs)
		testAccuracy := testLogits.MustArgmax(-1, false).MustEq1(ds.TestLabels).MustTotype(gotch.Float).MustMean(gotch.Float.CInt()).MustView([]int64{-1}).MustFloat64Value([]int64{0})

		fmt.Printf("Epoch: %v - Loss: %.3f - Test accuracy: %.2f%%\n", epoch, loss.Values()[0], testAccuracy*100)

	}
}
