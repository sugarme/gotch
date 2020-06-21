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

func train(trainX, trainY, testX, testY ts.Tensor, m ts.Module, opt nn.Optimizer, epoch int) {
	loss := m.Forward(trainX).CrossEntropyForLogits(trainY)

	opt.BackwardStep(loss)

	testAccuracy := m.Forward(testX).AccuracyForLogits(testY).Values()[0]
	fmt.Printf("Epoch: %v \t Loss: %.3f \t Test accuracy: %.2f%%\n", epoch, loss.Values()[0], testAccuracy*100)

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

		train(ds.TrainImages, ds.TrainLabels, ds.TestImages, ds.TestLabels, net, opt, epoch)

	}

}
