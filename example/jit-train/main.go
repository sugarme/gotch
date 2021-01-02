package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	// ts "github.com/sugarme/gotch/tensor"
	// "github.com/sugarme/gotch/vision"
)

func main() {
	runTrainAndSaveModel(gotch.CudaIfAvailable())
}

func runTrainAndSaveModel(device gotch.Device) {

	file := "./model.pt"
	vs := nn.NewVarStore(device)
	trainable, err := nn.TrainableCModuleLoad(vs.Root(), file)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Trainable JIT model loaded.\n")

	namedTensors, err := trainable.Inner.NamedParameters()
	if err != nil {
		log.Fatal(err)
	}

	for _, x := range namedTensors {
		fmt.Println(x.Name)
	}
}
