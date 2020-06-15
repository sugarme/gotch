package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

const (
	ImageDim int64  = 784
	Label    int64  = 10
	MnistDir string = "../../data/mnist"

	epochs = 200
)

func runLinear() {
	var ds vision.Dataset
	ds = vision.LoadMNISTDir(MnistDir)

	fmt.Printf("Train image size: %v\n", ds.TrainImages.MustSize())
	fmt.Printf("Train label size: %v\n", ds.TrainLabels.MustSize())
	fmt.Printf("Test image size: %v\n", ds.TestImages.MustSize())
	fmt.Printf("Test label size: %v\n", ds.TestLabels.MustSize())

	device := (gotch.CPU).CInt()
	dtype := (gotch.Double).CInt()

	ws := ts.MustZeros([]int64{ImageDim, Label}, dtype, device).MustSetRequiresGrad(true)

	bs := ts.MustZeros([]int64{Label}, dtype, device).MustSetRequiresGrad(true)

	fmt.Println(ws.MustSize())
	fmt.Println(bs.MustSize())

	for epoch := 0; epoch < epochs; epoch++ {
	}
}

func handleError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
