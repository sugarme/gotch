package main

import (
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/pickle"
)

func main() {
	// modelName := "vgg16"
	// modelName := "mobilenet_v2"
	// modelName := "resnet18"
	// modelName := "alexnet"
	// modelName := "squeezenet1_1"
	// modelName := "inception_v3_google"
	modelName := "efficientnet_b4"

	url, ok := gotch.ModelUrls[modelName]
	if !ok {
		log.Fatalf("Unsupported model name %q\n", modelName)
	}
	modelFile, err := gotch.CachedPath(url)
	if err != nil {
		panic(err)
	}

	err = pickle.LoadInfo(modelFile)
	if err != nil {
		log.Fatal(err)
	}
}
