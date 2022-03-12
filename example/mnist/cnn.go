package main

import (
	"fmt"
	"log"
	"time"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
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
	conv1 *nn.Conv2D
	conv2 *nn.Conv2D
	fc1   *nn.Linear
	fc2   *nn.Linear
}

func newNet(vs *nn.Path) *Net {
	conv1 := nn.NewConv2D(vs, 1, 32, 5, nn.DefaultConv2DConfig())
	conv2 := nn.NewConv2D(vs, 32, 64, 5, nn.DefaultConv2DConfig())
	fc1 := nn.NewLinear(vs, 1024, 1024, nn.DefaultLinearConfig())
	fc2 := nn.NewLinear(vs, 1024, 10, nn.DefaultLinearConfig())

	return &Net{
		conv1,
		conv2,
		fc1,
		fc2}
}

func (n *Net) ForwardT(xs *ts.Tensor, train bool) *ts.Tensor {
	outView1 := xs.MustView([]int64{-1, 1, 28, 28}, false)
	defer outView1.MustDrop()

	outC1 := outView1.Apply(n.conv1)

	outMP1 := outC1.MaxPool2DDefault(2, true)
	defer outMP1.MustDrop()

	outC2 := outMP1.Apply(n.conv2)

	outMP2 := outC2.MaxPool2DDefault(2, true)

	outView2 := outMP2.MustView([]int64{-1, 1024}, true)
	defer outView2.MustDrop()

	outFC1 := outView2.Apply(n.fc1)

	outRelu := outFC1.MustRelu(true)
	defer outRelu.MustDrop()
	outDropout := ts.MustDropout(outRelu, 0.5, train)
	defer outDropout.MustDrop()

	return outDropout.Apply(n.fc2)
}

func runCNN1() {

	var ds *vision.Dataset
	ds = vision.LoadMNISTDir(MnistDirNN)
	// ds.TrainImages [60000, 784]
	// ds.TrainLabels [60000, 784]
	testImages := ds.TestImages // [10000, 784]
	testLabels := ds.TestLabels // [10000, 784]

	fmt.Printf("testImages: %v\n", testImages.MustSize())
	fmt.Printf("testLabels: %v\n", testLabels.MustSize())

	device := gotch.CudaIfAvailable()
	vs := nn.NewVarStore(device)

	net := newNet(vs.Root())
	opt, err := nn.DefaultAdamConfig().Build(vs, LrCNN)
	if err != nil {
		log.Fatal(err)
	}

	var bestAccuracy float64 = 0.0
	startTime := time.Now()

	for epoch := 0; epoch < epochsCNN; epoch++ {
		totalSize := ds.TrainImages.MustSize()[0]
		samples := int(totalSize)
		// Shuffling
		index := ts.MustRandperm(int64(totalSize), gotch.Int64, gotch.CPU)
		imagesTs := ds.TrainImages.MustIndexSelect(0, index, false)
		labelsTs := ds.TrainLabels.MustIndexSelect(0, index, false)
		index.MustDrop()

		batches := samples / batchSize
		batchIndex := 0
		var epocLoss float64
		for i := 0; i < batches; i++ {
			start := batchIndex * batchSize
			size := batchSize
			if samples-start < batchSize {
				break
			}
			batchIndex += 1

			// Indexing
			bImages := imagesTs.MustNarrow(0, int64(start), int64(size), false)
			bLabels := labelsTs.MustNarrow(0, int64(start), int64(size), false)

			bImages = bImages.MustTo(vs.Device(), true)
			bLabels = bLabels.MustTo(vs.Device(), true)

			logits := net.ForwardT(bImages, true)
			bImages.MustDrop()
			loss := logits.CrossEntropyForLogits(bLabels)
			logits.MustDrop()
			bLabels.MustDrop()

			loss = loss.MustSetRequiresGrad(true, true)
			opt.BackwardStep(loss)

			epocLoss = loss.Float64Values()[0]
			loss.MustDrop()
		}

		ts.NoGrad(func() {
			testAccuracy := nn.BatchAccuracyForLogits(vs, net, testImages, testLabels, vs.Device(), 1024)
			fmt.Printf("Epoch: %v\t Loss: %.2f \t Test accuracy: %.2f%%\n", epoch, epocLoss, testAccuracy*100.0)
			if testAccuracy > bestAccuracy {
				bestAccuracy = testAccuracy
			}
		})

		imagesTs.MustDrop()
		labelsTs.MustDrop()
	}

	fmt.Printf("Best test accuracy: %.2f%%\n", bestAccuracy*100.0)
	fmt.Printf("Taken time:\t%.2f mins\n", time.Since(startTime).Minutes())
}
