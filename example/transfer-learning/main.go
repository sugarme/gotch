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

	fmt.Printf("Dataset: %v\n", dataset)
	fmt.Printf("Train shape: %v\n", dataset.TrainImages.MustSize())
	fmt.Printf("Train shape: %v\n", dataset.TrainLabels.MustSize())
	fmt.Printf("Test shape: %v\n", dataset.TestImages.MustSize())
	fmt.Printf("Test shape: %v\n", dataset.TestLabels.MustSize())

	// Create the model and load the weights from the file.
	vs := nn.NewVarStore(gotch.CPU)
	net := vision.ResNet18NoFinalLayer(vs.Root())

	// for k, _ := range vs.Vars.NamedVariables {
	// fmt.Printf("First variable name: %v\n", k)
	// }
	fmt.Printf("vs variables: %v\n", vs.Variables())
	fmt.Printf("vs num of variables: %v\n", vs.Len())

	err = vs.Load(weights)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Net infor: %v\n", net)

	panic("stop")

}
