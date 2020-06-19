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
	ImageDimNN    int64  = 784
	HiddenNodesNN int64  = 128
	LabelNN       int64  = 10
	MnistDirNN    string = "../../data/mnist"

	epochsNN    = 50
	batchSizeNN = 256

	LrNN = 1e-3
)

var l nn.Linear

func netInit(vs nn.Path) ts.Module {
	n := nn.Seq()

	l = nn.NewLinear(vs.Sub("layer1"), ImageDimNN, HiddenNodesNN, nn.DefaultLinearConfig())

	n.Add(l)

	n.AddFn(nn.ForwardWith(func(xs ts.Tensor) ts.Tensor {
		return xs.MustRelu()
	}))

	n.Add(nn.NewLinear(vs, HiddenNodesNN, LabelNN, nn.DefaultLinearConfig()))

	return n
}

func runNN() {
	var ds vision.Dataset
	ds = vision.LoadMNISTDir(MnistDirNN)

	vs := nn.NewVarStore(gotch.CPU)
	net := netInit(vs.Root())
	opt, err := nn.DefaultAdamConfig().Build(vs, LrNN)
	if err != nil {
		log.Fatal(err)
	}

	for epoch := 0; epoch < epochsNN; epoch++ {

		loss := net.Forward(ds.TrainImages).CrossEntropyForLogits(ds.TrainLabels)

		opt.BackwardStep(loss)
		lossVal := loss.MustShallowClone().MustView([]int64{-1}).MustFloat64Value([]int64{0})
		testAccuracy := net.Forward(ds.TestImages).AccuracyForLogits(ds.TestLabels).MustView([]int64{-1}).MustFloat64Value([]int64{0})
		fmt.Printf("Epoch: %v - Loss: %.3f - Test accuracy: %.2f%%\n", epoch, lossVal, testAccuracy*100)

		fmt.Printf("Loss: %v\n", lossVal)
	}

}
