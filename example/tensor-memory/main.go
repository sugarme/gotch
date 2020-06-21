package main

import (
	"fmt"
	// "runtime"

	ts "github.com/sugarme/gotch/tensor"
)

func createTensors(samples int) []ts.Tensor {
	n := int(10e6)
	var data []float64
	for i := 0; i < n; i++ {
		data = append(data, float64(i))
	}

	var tensors []ts.Tensor
	s := ts.FloatScalar(float64(0.23))

	// for i := 0; i < samples; i++ {
	for i := 0; i < 1; i++ {
		t := ts.MustOfSlice(data).MustMul1(s, true)

		// t1.MustDrop()
		// t.MustDrop()
		// t1 = ts.Tensor{}
		// t = ts.Tensor{}
		// runtime.GC()

		// fmt.Printf("t values: %v", t.Values())
		// fmt.Printf("t1 values: %v", t1.Values())
		tensors = append(tensors, t)
	}

	return tensors
}

func dropTensors(tensors []ts.Tensor) {
	for _, t := range tensors {
		t.MustDrop()
	}
}

func main() {

	var si *SI
	si = Get()
	fmt.Printf("Total RAM (MB):\t %8.2f\n", float64(si.TotalRam)/1024)
	fmt.Printf("Used RAM (MB):\t %8.2f\n", float64(si.TotalRam-si.FreeRam)/1024)

	startRAM := si.TotalRam - si.FreeRam

	epochs := 50
	// var m runtime.MemStats

	for i := 0; i < epochs; i++ {
		// runtime.ReadMemStats(&m)
		// t0 := float64(m.Sys) / 1024 / 1024

		tensors := createTensors(10000)

		// runtime.ReadMemStats(&m)
		// t1 := float64(m.Sys) / 1024 / 1024

		dropTensors(tensors)

		// runtime.ReadMemStats(&m)
		// t2 := float64(m.Sys) / 1024 / 1024

		// fmt.Printf("Epoch: %v \t Start Mem [%.3f MiB] \t Alloc Mem [%.3f MiB] \t Free Mem [%.3f MiB]\n", i, t0, t1, t2)
		si = Get()
		fmt.Printf("Epoch %v\t Used: [%8.2f MiB]\n", i, (float64(si.TotalRam-si.FreeRam)-float64(startRAM))/1024)
	}
}
