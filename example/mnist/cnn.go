package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

const (
	MnistDirCNN string = "../../data/mnist"

	epochsCNN = 10
	batchCNN  = 256

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
	fc1 := nn.NewLinear(*vs, 1024, 1024, *nn.DefaultLinearConfig())
	fc2 := nn.NewLinear(*vs, 1024, 10, *nn.DefaultLinearConfig())

	return Net{
		conv1,
		conv2,
		*fc1,
		*fc2}
}

func (n Net) ForwardT(xs ts.Tensor, train bool) ts.Tensor {
	out := xs.MustView([]int64{-1, 1, 28, 28}).Apply(n.conv1).MaxPool2DDefault(2, true)
	out = out.Apply(n.conv2).MaxPool2DDefault(2, true)
	out = out.MustView([]int64{-1, 1024}).Apply(&n.fc1).MustRelu(true)
	out.Dropout_(0.5, train)
	return out.Apply(&n.fc2)
}

func runCNN() {

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

	for epoch := 0; epoch < epochsCNN; epoch++ {
		var count = 0
		for {
			iter := ds.TrainIter(batchCNN).Shuffle()
			item, ok := iter.Next()
			if !ok {
				break
			}

			loss := net.ForwardT(item.Data.MustTo(vs.Device(), true), true).CrossEntropyForLogits(item.Label.MustTo(vs.Device(), true))
			opt.BackwardStep(loss)
			loss.MustDrop()
			count++
			if count == 50 {
				break
			}
			fmt.Printf("completed \t %v batches\n", count)
		}

		// testAccuracy := ts.BatchAccuracyForLogits(net, ds.TestImages, ds.TestLabels, vs.Device(), 1024)
		//
		// fmt.Printf("Epoch: %v \t Test accuracy: %.2f%%\n", epoch, testAccuracy*100)
	}

}
