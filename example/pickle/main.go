package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/pickle"
	"github.com/sugarme/gotch/vision"
)

func main() {
	device := gotch.CPU
	vs := nn.NewVarStore(device)
	net := vision.VGG16(vs.Root(), 1000)

	modelName := "vgg16"
	modelUrl, ok := gotch.ModelUrls[modelName]
	if !ok {
		log.Fatal("model name %q not found.", modelName)
	}

	modelFile, err := gotch.CachedPath(modelUrl)
	if err != nil {
		log.Fatal(err)
	}

	err = pickle.LoadAll(vs, modelFile)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%v\n", net)
	vs.Summary()
}
