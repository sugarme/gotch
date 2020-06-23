package main

import (
	"flag"
)

var model string

func init() {
	flag.StringVar(&model, "model", "linear", "specify a model to run")

}

func main() {

	flag.Parse()

	switch model {
	case "linear":
		runLinear()
	case "nn":
		runNN()
	case "cnn":
		runCNN()
	default:
		panic("No specified model to run")
	}

}