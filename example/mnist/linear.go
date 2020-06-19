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

	epochs    = 200
	batchSize = 256
)

func runLinear() {
	var ds vision.Dataset
	ds = vision.LoadMNISTDir(MnistDir)

	// fmt.Printf("Train image size: %v\n", ds.TrainImages.MustSize())
	// fmt.Printf("Train label size: %v\n", ds.TrainLabels.MustSize())
	// fmt.Printf("Test image size: %v\n", ds.TestImages.MustSize())
	// fmt.Printf("Test label size: %v\n", ds.TestLabels.MustSize())

	device := (gotch.CPU).CInt()
	dtype := (gotch.Float).CInt()

	ws := ts.MustZeros([]int64{ImageDim, Label}, dtype, device).MustSetRequiresGrad(true)
	bs := ts.MustZeros([]int64{Label}, dtype, device).MustSetRequiresGrad(true)

	for epoch := 0; epoch < epochs; epoch++ {
		/*
		 *     totalSize := ds.TrainImages.MustSize()[0]
		 *     samples := int(totalSize)
		 *     index := ts.MustRandperm(int64(totalSize), gotch.Int64, gotch.CPU)
		 *     imagesTs := ds.TrainImages.MustIndexSelect(0, index)
		 *     labelsTs := ds.TrainLabels.MustIndexSelect(0, index)
		 *
		 *     batches := samples / batchSize
		 *     batchIndex := 0
		 *     var loss ts.Tensor
		 *     for i := 0; i < batches; i++ {
		 *       start := batchIndex * batchSize
		 *       size := batchSize
		 *       if samples-start < batchSize {
		 *         // size = samples - start
		 *         break
		 *       }
		 *       batchIndex += 1
		 *
		 *       // Indexing
		 *       narrowIndex := ts.NewNarrow(int64(start), int64(start+size))
		 *       bImages := ds.TrainImages.Idx(narrowIndex)
		 *       bLabels := ds.TrainLabels.Idx(narrowIndex)
		 *       // bImages := imagesTs.Idx(narrowIndex)
		 *       // bLabels := labelsTs.Idx(narrowIndex)
		 *
		 *       logits := bImages.MustMm(ws).MustAdd(bs)
		 *       loss = logits.MustLogSoftmax(-1, dtype).MustNllLoss(bLabels).MustSetRequiresGrad(true)
		 *
		 *       ws.ZeroGrad()
		 *       bs.ZeroGrad()
		 *       loss.MustBackward()
		 *
		 *       ts.NoGrad(func() {
		 *         ws.MustAdd_(ws.MustGrad().MustMul1(ts.FloatScalar(-1.0)))
		 *         bs.MustAdd_(bs.MustGrad().MustMul1(ts.FloatScalar(-1.0)))
		 *       })
		 *     }
		 *
		 *     imagesTs.MustDrop()
		 *     labelsTs.MustDrop()
		 *  */

		logits := ds.TrainImages.MustMm(ws).MustAdd(bs)
		loss := logits.MustLogSoftmax(-1, dtype).MustNllLoss(ds.TrainLabels).MustSetRequiresGrad(true)

		ws.ZeroGrad()
		bs.ZeroGrad()
		loss.MustBackward()

		ts.NoGrad(func() {
			ws.Add_(ws.MustGrad().MustMul1(ts.FloatScalar(-1.0)))
			bs.Add_(bs.MustGrad().MustMul1(ts.FloatScalar(-1.0)))
		})

		testLogits := ds.TestImages.MustMm(ws).MustAdd(bs)
		testAccuracy := testLogits.MustArgmax(-1, false).MustEq1(ds.TestLabels).MustTotype(gotch.Float).MustMean(gotch.Float.CInt()).MustView([]int64{-1}).MustFloat64Value([]int64{0})

		lossVal := loss.MustShallowClone().MustView([]int64{-1}).MustFloat64Value([]int64{0})
		fmt.Printf("Epoch: %v - Loss: %.3f - Test accuracy: %.2f%%\n", epoch, lossVal, testAccuracy*100)
	}
}
