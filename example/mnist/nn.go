package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

const (
	ImageDimNN    int64  = 784
	HiddenNodesNN int64  = 128
	LabelNN       int64  = 10
	MnistDirNN    string = "../../data/mnist"

	epochsNN = 200

	LrNN = 1e-3
)

var l nn.Linear

func netInit(vs *nn.Path) ts.Module {
	n := nn.Seq()

	n.Add(nn.NewLinear(vs, ImageDimNN, HiddenNodesNN, nn.DefaultLinearConfig()))

	n.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	n.Add(nn.NewLinear(vs, HiddenNodesNN, LabelNN, nn.DefaultLinearConfig()))

	return n
}

func train(trainX, trainY, testX, testY *ts.Tensor, m ts.Module, opt *nn.Optimizer, epoch int) {

	logits := m.Forward(trainX)
	loss := logits.CrossEntropyForLogits(trainY)

	opt.BackwardStep(loss)

	testLogits := m.Forward(testX)
	testAccuracy := testLogits.AccuracyForLogits(testY)
	accuracy := testAccuracy.Float64Values()[0] * 100
	testAccuracy.MustDrop()
	lossVal := loss.Float64Values()[0]
	loss.MustDrop()

	fmt.Printf("Epoch: %v \t Loss: %.3f \t Test accuracy: %.2f%%\n", epoch, lossVal, accuracy)
}

func runNN() {

	var ds *vision.Dataset
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
