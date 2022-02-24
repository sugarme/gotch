package pickle_test

import (
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/pickle"
)

func ExampleLoadInfo() {
	modelName := "vgg16"
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

	// Output:
	// classifier.0.bias - [4096]
	// classifier.0.weight - [4096 25088]
	// classifier.3.bias - [4096]
	// classifier.3.weight - [4096 4096]
	// classifier.6.bias - [1000]
	// classifier.6.weight - [1000 4096]
	// features.0.bias - [64]
	// features.0.weight - [64 3 3 3]
	// features.10.bias - [256]
	// features.10.weight - [256 128 3 3]
	// features.12.bias - [256]
	// features.12.weight - [256 256 3 3]
	// features.14.bias - [256]
	// features.14.weight - [256 256 3 3]
	// features.17.bias - [512]
	// features.17.weight - [512 256 3 3]
	// features.19.bias - [512]
	// features.19.weight - [512 512 3 3]
	// features.2.bias - [64]
	// features.2.weight - [64 64 3 3]
	// features.21.bias - [512]
	// features.21.weight - [512 512 3 3]
	// features.24.bias - [512]
	// features.24.weight - [512 512 3 3]
	// features.26.bias - [512]
	// features.26.weight - [512 512 3 3]
	// features.28.bias - [512]
	// features.28.weight - [512 512 3 3]
	// features.5.bias - [128]
	// features.5.weight - [128 64 3 3]
	// features.7.bias - [128]
	// features.7.weight - [128 128 3 3]
	// Num of variables: 32
}
