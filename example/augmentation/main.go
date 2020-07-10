package main

// Training various models on the CIFAR-10 dataset.
//
// The dataset can be downloaded from https:www.cs.toronto.edu/~kriz/cifar.html, files
// should be placed in the data/ directory.
//
// The resnet model reaches 95.4% accuracy.

import (
	"fmt"
	// "log"
	// "os/exec"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

func main() {
	dir := "../../data/cifar10"
	ds := vision.CFLoadDir(dir)

	fmt.Printf("TrainImages shape: %v\n", ds.TrainImages.MustSize())
	fmt.Printf("TrainLabel shape: %v\n", ds.TrainLabels.MustSize())
	fmt.Printf("TestImages shape: %v\n", ds.TestImages.MustSize())
	fmt.Printf("TestLabel shape: %v\n", ds.TestLabels.MustSize())
	fmt.Printf("Number of labels: %v\n", ds.Labels)

	// cuda := gotch.CudaBuilder(0)
	// device := cuda.CudaIfAvailable()
	device := gotch.CPU

	var si *gotch.SI
	si = gotch.GetSysInfo()
	fmt.Printf("Total RAM (MB):\t %8.2f\n", float64(si.TotalRam)/1024)
	fmt.Printf("Used RAM (MB):\t %8.2f\n", float64(si.TotalRam-si.FreeRam)/1024)

	startRAM := si.TotalRam - si.FreeRam

	vs := nn.NewVarStore(device)

	for epoch := 0; epoch < 150; epoch++ {

		iter := ts.MustNewIter2(ds.TrainImages, ds.TrainLabels, int64(64))
		iter.Shuffle()

		for {
			item, ok := iter.Next()
			if !ok {
				item.Data.MustDrop()
				item.Label.MustDrop()
				break
			}

			devicedData := item.Data.MustTo(vs.Device(), true)
			devicedLabel := item.Label.MustTo(vs.Device(), true)
			bimages := vision.Augmentation(devicedData, true, 4, 8)

			devicedData.MustDrop()
			devicedLabel.MustDrop()
			bimages.MustDrop()

		}

		iter.Drop()

		si = gotch.GetSysInfo()
		memUsed := (float64(si.TotalRam-si.FreeRam) - float64(startRAM)) / 1024
		fmt.Printf("Epoch:\t %v\t Memory Used:\t [%8.2f MiB]\n", epoch, memUsed)

		/*
		 *     // Print out GPU used
		 *     nvidia := "nvidia-smi"
		 *     cmd := exec.Command(nvidia)
		 *     stdout, err := cmd.Output()
		 *
		 *     if err != nil {
		 *       log.Fatal(err.Error())
		 *     }
		 *
		 *     fmt.Println(string(stdout))
		 *  */
	}

}
