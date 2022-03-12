package main

// This example illustrates how to use transfer learning to fine tune a pre-trained
// imagenet model on another dataset.

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

var (
	datasetDir string
	weights    string
)

func init() {
	flag.StringVar(&datasetDir, "dataset", "../../data/hymenoptera-data", "full path to dataset directory")
	flag.StringVar(&weights, "weights", "../../data/pretrained/resnet18.pt", "resnet18 pretrained weights file")
}

func main() {
	flag.Parse()

	// Load the dataset and resize it to the usual imagenet dimension of 224x224.
	imageNet := vision.NewImageNet()
	datasetPath, err := filepath.Abs(datasetDir)
	if err != nil {
		log.Fatal(err)
	}
	dataset, err := imageNet.LoadFromDir(datasetPath)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Dataset loaded")

	// Create the model and load the weights from the file.
	vs := nn.NewVarStore(gotch.CPU)
	net := vision.ResNet18NoFinalLayer(vs.Root())

	err = vs.Load(weights)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Weights loaded")

	// Pre-compute the final activations.

	linear := nn.NewLinear(vs.Root(), 512, dataset.Labels, nn.DefaultLinearConfig())
	sgd, err := nn.DefaultSGDConfig().Build(vs, 1e-3)
	if err != nil {
		log.Fatal(err)
	}

	trainImages := ts.NoGrad1(func() (retVal interface{}) {
		return dataset.TrainImages.ApplyT(net, true)
	}).(*ts.Tensor)

	testImages := ts.NoGrad1(func() (retVal interface{}) {
		return dataset.TestImages.ApplyT(net, true)
	}).(*ts.Tensor)

	fmt.Println("start training...")

	for epoch := 1; epoch <= 1000; epoch++ {

		predicted := trainImages.ApplyT(linear, true)
		loss := predicted.CrossEntropyForLogits(dataset.TrainLabels)
		sgd.BackwardStep(loss)
		loss.MustDrop()

		ts.NoGrad(func() {
			testAccuracy := testImages.Apply(linear).AccuracyForLogits(dataset.TestLabels)
			fmt.Printf("Epoch %v\t Accuracy: %5.2f%%\n", epoch, testAccuracy.Float64Values()[0]*100)
		})
	}
}
