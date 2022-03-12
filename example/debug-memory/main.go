package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

var device string

func createTensors(samples int) []ts.Tensor {
	n := int(10e6)
	var data []float64
	for i := 0; i < n; i++ {
		data = append(data, float64(i))
	}

	var tensors []ts.Tensor
	s := ts.FloatScalar(float64(0.23))

	for i := 0; i < 1; i++ {
		t := ts.MustOfSlice(data).MustMulScalar(s, true)

		tensors = append(tensors, *t)
	}

	return tensors
}

func dropTensors(tensors []ts.Tensor) {
	for _, t := range tensors {
		t.MustDrop()
	}
}

func init() {
	flag.StringVar(&device, "device", "CPU", "Select CPU or GPU to use")

}

func main() {
	// TODO: create flags to load tensor to device(CPU, GPU) and get CPU or GPU
	// infor accordingly
	flag.Parse()

	switch device {
	case "CPU":
		var si *SI
		si = CPUInfo()
		fmt.Printf("Total RAM (MB):\t %8.2f\n", float64(si.TotalRam)/1024)
		fmt.Printf("Used RAM (MB):\t %8.2f\n", float64(si.TotalRam-si.FreeRam)/1024)
		startRAM := si.TotalRam - si.FreeRam
		epochs := 50
		for i := 0; i < epochs; i++ {
			tensors := createTensors(10000)
			dropTensors(tensors)

			si = CPUInfo()
			fmt.Printf("Epoch %v\t Used: [%8.2f MiB]\n", i, (float64(si.TotalRam-si.FreeRam)-float64(startRAM))/1024)
		}

	case "GPU":
		cuda := gotch.CudaBuilder(0)
		gpu := cuda.CudaIfAvailable()

		epochs := 50
		for i := 0; i < epochs; i++ {

			tensors := createTensors(10000)
			var gpuTensors []ts.Tensor
			for _, t := range tensors {
				gpuTensors = append(gpuTensors, *t.MustTo(gpu, true))
			}

			for _, t := range gpuTensors {
				t.MustDrop()
			}

			fmt.Printf("Epoch %v\n", i)
			GPUInfo()
		}

	default:
		log.Fatalf("Invalid device flag (%v). It should be either CPU or GPU.", device)
	}

}
