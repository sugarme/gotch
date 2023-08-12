package main

import (
	"fmt"
	"log"
	"runtime"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

const (
	ImageDimNN    int64 = 784
	HiddenNodesNN int64 = 128
	LabelNN       int64 = 10

	epochsNN = 200

	LrNN = 1e-3
)

var MnistDirNN string = fmt.Sprintf("%s/%s", gotch.CachedDir, "mnist")
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
	lossVal := loss.Float64Values()[0]

	fmt.Printf("Epoch: %v \t Loss: %.3f \t Test accuracy: %.2f%%\n", epoch, lossVal, accuracy)

	runtime.GC()
}

func runNN() {
	var ds *vision.Dataset
	ds = vision.LoadMNISTDir(MnistDirNN)
	vs := nn.NewVarStore(device)
	net := netInit(vs.Root())
	opt, err := nn.DefaultAdamConfig().Build(vs, LrNN)
	if err != nil {
		log.Fatal(err)
	}

	trainImages := ds.TrainImages.MustTo(device, true)
	trainLabels := ds.TrainLabels.MustTo(device, true)
	testImages := ds.TestImages.MustTo(device, true)
	testLabels := ds.TestLabels.MustTo(device, true)

	for epoch := 0; epoch < epochsNN; epoch++ {
		train(trainImages, trainLabels, testImages, testLabels, net, opt, epoch)
	}
}
