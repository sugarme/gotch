package main

import (
	"fmt"
	"log"
	"time"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

const (
	MnistDirCNN string = "../../data/mnist"

	epochsCNN = 100
	batchCNN  = 256
	batchSize = 256

	LrCNN = 1e-4
)

type Net struct {
	conv1 nn.Conv2D
	conv2 nn.Conv2D
	fc1   nn.Linear
	fc2   nn.Linear
}

func newNet(vs *nn.Path) Net {
	conv1 := nn.NewConv2D(vs, 1, 32, 5, nn.DefaultConv2DConfig())
	conv2 := nn.NewConv2D(vs, 32, 64, 5, nn.DefaultConv2DConfig())
	fc1 := nn.NewLinear(*vs, 1024, 1024, nn.DefaultLinearConfig())
	fc2 := nn.NewLinear(*vs, 1024, 10, nn.DefaultLinearConfig())

	return Net{
		conv1,
		conv2,
		fc1,
		fc2}
}

func (n Net) ForwardT(xs ts.Tensor, train bool) (retVal ts.Tensor) {
	outView1 := xs.MustView([]int64{-1, 1, 28, 28}, false)
	defer outView1.MustDrop()

	outC1 := outView1.Apply(n.conv1)
	// defer outC1.MustDrop()

	outMP1 := outC1.MaxPool2DDefault(2, true)
	defer outMP1.MustDrop()

	outC2 := outMP1.Apply(n.conv2)
	// defer outC2.MustDrop()

	outMP2 := outC2.MaxPool2DDefault(2, true)
	// defer outMP2.MustDrop()

	outView2 := outMP2.MustView([]int64{-1, 1024}, true)
	defer outView2.MustDrop()

	outFC1 := outView2.Apply(&n.fc1)
	// defer outFC1.MustDrop()

	outRelu := outFC1.MustRelu(true)
	defer outRelu.MustDrop()
	// outRelu.Dropout_(0.5, train)
	outDropout := ts.MustDropout(outRelu, 0.5, train)
	defer outDropout.MustDrop()

	return outDropout.Apply(&n.fc2)

}

func runCNN1() {

	var ds vision.Dataset
	ds = vision.LoadMNISTDir(MnistDirNN)
	testImages := ds.TestImages
	testLabels := ds.TestLabels

	cuda := gotch.CudaBuilder(0)
	vs := nn.NewVarStore(cuda.CudaIfAvailable())
	// vs := nn.NewVarStore(gotch.CPU)
	path := vs.Root()
	net := newNet(&path)
	opt, err := nn.DefaultAdamConfig().Build(vs, LrCNN)
	if err != nil {
		log.Fatal(err)
	}

	startTime := time.Now()

	for epoch := 0; epoch < epochsCNN; epoch++ {

		totalSize := ds.TrainImages.MustSize()[0]
		samples := int(totalSize)
		index := ts.MustRandperm(int64(totalSize), gotch.Int64, gotch.CPU)
		imagesTs := ds.TrainImages.MustIndexSelect(0, index, false)
		labelsTs := ds.TrainLabels.MustIndexSelect(0, index, false)

		batches := samples / batchSize
		batchIndex := 0
		var epocLoss ts.Tensor
		// var loss ts.Tensor
		for i := 0; i < batches; i++ {
			start := batchIndex * batchSize
			size := batchSize
			if samples-start < batchSize {
				// size = samples - start
				break
			}
			batchIndex += 1

			// Indexing
			narrowIndex := ts.NewNarrow(int64(start), int64(start+size))
			// bImages := ds.TrainImages.Idx(narrowIndex)
			// bLabels := ds.TrainLabels.Idx(narrowIndex)
			bImages := imagesTs.Idx(narrowIndex)
			bLabels := labelsTs.Idx(narrowIndex)

			bImages = bImages.MustTo(vs.Device(), true)
			bLabels = bLabels.MustTo(vs.Device(), true)

			logits := net.ForwardT(bImages, true)
			loss := logits.CrossEntropyForLogits(bLabels)

			opt.BackwardStep(loss)

			epocLoss = loss.MustShallowClone()
			epocLoss.Detach_()

			// fmt.Printf("completed \t %v batches\t %.2f\n", i, loss.Values()[0])

			bImages.MustDrop()
			bLabels.MustDrop()
			// logits.MustDrop()
			// loss.MustDrop()
		}

		// testAccuracy := ts.BatchAccuracyForLogitsIdx(net, testImages, testLabels, vs.Device(), 1024)
		// fmt.Printf("Epoch: %v\t Loss: %.2f \t Test accuracy: %.2f%%\n", epoch, epocLoss.Values()[0], testAccuracy*100)

		fmt.Printf("Epoch:\t %v\tLoss: \t %.2f\n", epoch, epocLoss.Values()[0])
		epocLoss.MustDrop()
		imagesTs.MustDrop()
		labelsTs.MustDrop()
	}

	testAccuracy := ts.BatchAccuracyForLogitsIdx(net, testImages, testLabels, vs.Device(), 1024)
	fmt.Printf("Test accuracy: %.2f%%\n", testAccuracy*100)

	fmt.Printf("Taken time:\t%.2f mins\n", time.Since(startTime).Minutes())
}

func runCNN2() {

	var ds vision.Dataset
	ds = vision.LoadMNISTDir(MnistDirNN)

	cuda := gotch.CudaBuilder(0)
	vs := nn.NewVarStore(cuda.CudaIfAvailable())
	path := vs.Root()
	net := newNet(&path)
	opt, err := nn.DefaultAdamConfig().Build(vs, LrNN)
	if err != nil {
		log.Fatal(err)
	}

	startTime := time.Now()

	var lossVal float64
	for epoch := 0; epoch < epochsCNN; epoch++ {

		iter := ts.MustNewIter2(ds.TrainImages, ds.TrainLabels, batchCNN)
		// iter.Shuffle()

		for {
			item, ok := iter.Next()
			if !ok {
				break
			}

			bImages := item.Data.MustTo(vs.Device(), true)
			bLabels := item.Label.MustTo(vs.Device(), true)

			// _ = ts.MustGradSetEnabled(true)

			logits := net.ForwardT(bImages, true)
			loss := logits.CrossEntropyForLogits(bLabels)

			opt.BackwardStep(loss)

			lossVal = loss.Values()[0]

			bImages.MustDrop()
			bLabels.MustDrop()
			loss.MustDrop()
		}

		fmt.Printf("Epoch:\t %v\tLoss: \t %.2f\n", epoch, lossVal)

		// testAcc := ts.BatchAccuracyForLogits(net, ds.TestImages, ds.TestLabels, vs.Device(), batchCNN)
		// fmt.Printf("Epoch:\t %v\tLoss: \t %.2f\t Accuracy: %.2f\n", epoch, lossVal, testAcc*100)
	}

	testAcc := ts.BatchAccuracyForLogits(net, ds.TestImages, ds.TestLabels, vs.Device(), batchCNN)
	fmt.Printf("Loss: \t %.2f\t Accuracy: %.2f\n", lossVal, testAcc*100)
	fmt.Printf("Taken time:\t%.2f mins\n", time.Since(startTime).Minutes())
}
