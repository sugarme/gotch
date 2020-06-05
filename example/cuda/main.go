package main

import (
	"fmt"

	"github.com/sugarme/gotch"
)

func main() {

	var d gotch.Cuda
	fmt.Printf("Cuda device count: %v\n", d.DeviceCount())
	fmt.Printf("Cuda is available: %v\n", d.IsAvailable())
	fmt.Printf("Cudnn is available: %v\n", d.CudnnIsAvailable())

}
