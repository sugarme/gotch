package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

const (
	ImageDim int64  = 784
	Label    int64  = 10
	MnistDir string = "../../data/mnist"

	epochs = 200
)

func runLinear() {
	var ds *vision.Dataset
	ds = vision.LoadMNISTDir(MnistDir)

	device := gotch.CPU
	dtype := gotch.Float

	ws := ts.MustZeros([]int64{ImageDim, Label}, dtype, device).MustSetRequiresGrad(true, false)
	bs := ts.MustZeros([]int64{Label}, dtype, device).MustSetRequiresGrad(true, false)

	for epoch := 0; epoch < epochs; epoch++ {

		weight := ts.NewTensor()
		reduction := int64(1) // Mean of loss
		ignoreIndex := int64(-100)

		logits := ds.TrainImages.MustMm(ws, false).MustAdd(bs, true)
		loss := logits.MustLogSoftmax(-1, dtype, true).MustNllLoss(ds.TrainLabels, weight, reduction, ignoreIndex, true)

		ws.ZeroGrad()
		bs.ZeroGrad()
		loss.MustBackward()

		ts.NoGrad(func() {
			ws.Add_(ws.MustGrad(false).MustMulScalar(ts.FloatScalar(-1.0), true))
			bs.Add_(bs.MustGrad(false).MustMulScalar(ts.FloatScalar(-1.0), true))
		})

		testLogits := ds.TestImages.MustMm(ws, false).MustAdd(bs, true)
		testAccuracy := testLogits.MustArgmax([]int64{-1}, false, true).MustEqTensor(ds.TestLabels, true).MustTotype(gotch.Float, true).MustMean(gotch.Float, true).MustView([]int64{-1}, true).MustFloat64Value([]int64{0})

		fmt.Printf("Epoch: %v - Loss: %.3f - Test accuracy: %.2f%%\n", epoch, loss.Float64Values()[0], testAccuracy*100)

		loss.MustDrop()
	}
}
