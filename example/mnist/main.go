package main

import (
	"flag"

	"github.com/sugarme/gotch"
)

var (
	model     string
	deviceOpt string
	device    gotch.Device
)

func init() {
	flag.StringVar(&model, "model", "linear", "specify a model to run")
	flag.StringVar(&deviceOpt, "device", "cpu", "specify device to run on. Eitheir 'cpu' or 'cuda'")
}

func main() {

	flag.Parse()

	if deviceOpt == "cuda" {
		device = gotch.CudaIfAvailable()
	} else {
		device = gotch.CPU
	}

	switch model {
	case "linear":
		runLinear()
	case "nn":
		runNN()
	case "cnn":
		// runCNN2()
		runCNN1()
	default:
		panic("No specified model to run")
	}

}
