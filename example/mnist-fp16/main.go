package main

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

func main() {
	runCNN()
}

const (
	MnistDir string = "/mnt/projects/numbat/data/mnist"

	epochsCNN = 30
	batchCNN  = 256
	// batchSize = 256
	batchSize = 32

	LrCNN = 3 * 1e-4
)

var mu sync.Mutex

// var device gotch.Device = gotch.CPU
var device gotch.Device = gotch.CudaIfAvailable()

// var dtype gotch.DType = gotch.BFloat16
// var dtype gotch.DType = gotch.Half
var dtype gotch.DType = gotch.Float

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
	outC1 := outView1.Apply(n.conv1)

	outMP1 := outC1.MaxPool2DDefault(2, true)
	outC2 := outMP1.Apply(n.conv2)

	outMP2 := outC2.MaxPool2DDefault(2, true)
	outView2 := outMP2.MustView([]int64{-1, 1024}, true)

	outFC1 := outView2.Apply(n.fc1)
	outRelu := outFC1.MustRelu(false)
	outDropout := ts.MustDropout(outRelu, 0.5, train)
	return outDropout.Apply(n.fc2)
}

func runCNN() {
	var ds *vision.Dataset
	ds = vision.LoadMNISTDir(MnistDir)
	trainImages := ds.TrainImages.MustTo(device, false)                       //[60000, 784]
	trainLabels := ds.TrainLabels.MustTo(device, false)                       // [60000, 784]
	testImages := ds.TestImages.MustTo(device, false).MustTotype(dtype, true) // [10000, 784]
	testLabels := ds.TestLabels.MustTo(device, false).MustTotype(dtype, true) // [10000, 784]

	fmt.Printf("testImages: %v\n", testImages.MustSize())
	fmt.Printf("testLabels: %v\n", testLabels.MustSize())

	odtype := gotch.SetDefaultDType(dtype)
	vs := nn.NewVarStore(device)
	net := newNet(vs.Root())
	gotch.SetDefaultDType(odtype)

	opt, err := nn.DefaultAdamConfig().Build(vs, LrCNN)
	// opt, err := nn.DefaultSGDConfig().Build(vs, LrCNN)
	if err != nil {
		log.Fatal(err)
	}

	var bestAccuracy float64 = 0.0
	startTime := time.Now()

	for epoch := 0; epoch < epochsCNN; epoch++ {
		totalSize := ds.TrainImages.MustSize()[0]
		samples := int(totalSize)
		// Shuffling
		index := ts.MustRandperm(int64(totalSize), gotch.Int64, device)
		imagesTs := trainImages.MustIndexSelect(0, index, false).MustTotype(dtype, true)
		labelsTs := trainLabels.MustIndexSelect(0, index, false)

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
			logits := net.ForwardT(bImages, true)

			bLabels := labelsTs.MustNarrow(0, int64(start), int64(size), false)
			loss := logits.CrossEntropyForLogits(bLabels)

			loss = loss.MustSetRequiresGrad(true, true)
			opt.BackwardStep(loss)
			epocLoss = loss.Float64Values()[0]

			runtime.GC()
		}

		ts.NoGrad(func() {
			fmt.Printf("Start eval...")
			testAccuracy := nn.BatchAccuracyForLogits(vs, net, testImages, testLabels, vs.Device(), 1000)
			fmt.Printf("Epoch: %v\t Loss: %.2f \t Test accuracy: %.2f%%\n", epoch, epocLoss, testAccuracy*100.0)
			if testAccuracy > bestAccuracy {
				bestAccuracy = testAccuracy
			}
		})
	}

	fmt.Printf("Best test accuracy: %.2f%%\n", bestAccuracy*100.0)
	fmt.Printf("Taken time:\t%.2f mins\n", time.Since(startTime).Minutes())

	ts.CleanUp()
}
