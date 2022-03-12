package main

// This example illustrates how to use a PyTorch model trained and exported using the
// Python JIT API.
// See https://pytorch.org/tutorials/advanced/cpp_export.html for more details.

import (
	"flag"
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

var (
	modelPath string
	imageFile string
)

func init() {
	flag.StringVar(&modelPath, "modelpath", "model.pt", "full path to exported pytorch model.")
	flag.StringVar(&imageFile, "image", "image.jpg", "full path to image file.")
}

func main() {
	flag.Parse()

	imageNet := vision.NewImageNet()

	// Load the image file and resize it to the usual imagenet dimension of 224x224.
	image, err := imageNet.LoadImageAndResize224(imageFile)
	if err != nil {
		log.Fatal(err)
	}

	// Load the Python saved module.
	model, err := ts.ModuleLoad(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	// Apply the forward pass of the model to get the logits.
	output := image.MustUnsqueeze(int64(0), false).ApplyCModule(model).MustSoftmax(-1, gotch.Float, true)

	// Print the top 5 categories for this image.
	var top5 []vision.TopItem

	top5 = imageNet.Top(output, int64(5))

	for _, i := range top5 {
		fmt.Printf("%-80v %5.2f%%\n", i.Label, i.Pvalue*100)
	}
}
