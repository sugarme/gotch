package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

const (
	ImageDim int64 = 784
	Label    int64 = 10

	epochs = 200
)

func runLinear() {
	var ds *vision.Dataset
	ds = vision.LoadMNISTDir(MnistDirNN)
	trainImages := ds.TrainImages.MustTo(device, true)
	trainLabels := ds.TrainLabels.MustTo(device, true)
	testImages := ds.TestImages.MustTo(device, true)
	testLabels := ds.TestLabels.MustTo(device, true)

	dtype := gotch.Float
	ws := ts.MustZeros([]int64{ImageDim, Label}, dtype, device).MustSetRequiresGrad(true, false)
	bs := ts.MustZeros([]int64{Label}, dtype, device).MustSetRequiresGrad(true, false)

	// NOTE(TT). if initiating with random float, result is worse.
	// ws := ts.MustRandn([]int64{ImageDim, Label}, dtype, device)
	// bs := ts.MustRandn([]int64{Label}, dtype, device)
	// ws.MustRequiresGrad_(true)
	// bs.MustRequiresGrad_(true)

	for epoch := 0; epoch < epochs; epoch++ {
		weight := ts.NewTensor()
		reduction := int64(1) // Mean of loss
		ignoreIndex := int64(-100)

		logits := trainImages.MustMm(ws, false).MustAdd(bs, true)
		loss := logits.MustLogSoftmax(-1, dtype, true).MustNllLoss(trainLabels, weight, reduction, ignoreIndex, true)

		ws.ZeroGrad()
		bs.ZeroGrad()
		loss.MustBackward()

		ts.NoGrad(func() {
			ws.Add_(ws.MustGrad(false).MustMulScalar(ts.FloatScalar(-1.0), true))
			bs.Add_(bs.MustGrad(false).MustMulScalar(ts.FloatScalar(-1.0), true))
		}, 100) // 100 msec sleeping time. Adjustable to available GPU RAM.

		testLogits := testImages.MustMm(ws, false).MustAdd(bs, true)
		testAccuracy := testLogits.MustArgmax([]int64{-1}, false, true).MustEqTensor(testLabels, true).MustTotype(gotch.Float, true).MustMean(gotch.Float, true).MustView([]int64{-1}, true).MustFloat64Value([]int64{0})

		fmt.Printf("Epoch: %v - Loss: %.3f - Test accuracy: %.2f%%\n", epoch, loss.Float64Values()[0], testAccuracy*100)
	}
}
