package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

func main() {
	ds := vision.LoadMNISTDir("../../data/mnist")
	dataset := &vision.Dataset{
		TestImages:  ds.TestImages.MustView([]int64{-1, 1, 28, 28}, true),
		TrainImages: ds.TrainImages.MustView([]int64{-1, 1, 28, 28}, true),
		TestLabels:  ds.TestLabels,
		TrainLabels: ds.TrainLabels,
	}
	device := gotch.CudaIfAvailable()

	// runTrainAndSaveModel(dataset, device)
	loadTrainedAndTestAcc(dataset, device)
}

func runTrainAndSaveModel(ds *vision.Dataset, device gotch.Device) {

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

	trainable.SetTrain()
	initialAcc := nn.BatchAccuracyForLogits(vs, trainable, ds.TestImages, ds.TestLabels, device, 1024)
	fmt.Printf("Initial Accuracy: %0.4f\n", initialAcc)
	bestAccuracy := initialAcc

	opt, err := nn.DefaultAdamConfig().Build(vs, 1e-4)
	if err != nil {
		log.Fatal(err)
	}
	batchSize := 128
	for epoch := 0; epoch < 20; epoch++ {

		totalSize := ds.TrainImages.MustSize()[0]
		samples := int(totalSize)
		index := ts.MustRandperm(int64(totalSize), gotch.Int64, gotch.CPU)
		imagesTs := ds.TrainImages.MustIndexSelect(0, index, false)
		labelsTs := ds.TrainLabels.MustIndexSelect(0, index, false)

		batches := samples / batchSize
		batchIndex := 0
		var epocLoss *ts.Tensor
		for i := 0; i < batches; i++ {
			start := batchIndex * batchSize
			size := batchSize
			if samples-start < batchSize {
				break
			}
			batchIndex += 1

			// Indexing
			narrowIndex := ts.NewNarrow(int64(start), int64(start+size))
			bImages := imagesTs.Idx(narrowIndex)
			bLabels := labelsTs.Idx(narrowIndex)

			bImages = bImages.MustTo(vs.Device(), true)
			bLabels = bLabels.MustTo(vs.Device(), true)

			logits := trainable.ForwardT(bImages, true)
			loss := logits.CrossEntropyForLogits(bLabels)

			opt.BackwardStep(loss)

			epocLoss = loss.MustShallowClone()
			epocLoss.Detach_()

			// fmt.Printf("completed \t %v batches\t %.2f\n", i, loss.Float64Values()[0])

			bImages.MustDrop()
			bLabels.MustDrop()
		}

		testAccuracy := nn.BatchAccuracyForLogits(vs, trainable, ds.TestImages, ds.TestLabels, vs.Device(), 1024)
		fmt.Printf("Epoch: %v\t Loss: %.2f \t Test accuracy: %.2f%%\n", epoch, epocLoss.Float64Values()[0], testAccuracy*100.0)
		if testAccuracy > bestAccuracy {
			bestAccuracy = testAccuracy
		}

		epocLoss.MustDrop()
		imagesTs.MustDrop()
		labelsTs.MustDrop()
	}

	err = trainable.Save("trained-model.pt")
	if err != nil {
		log.Fatal(err)
	}
}

func loadTrainedAndTestAcc(ds *vision.Dataset, device gotch.Device) {
	vs := nn.NewVarStore(device)
	m, err := nn.TrainableCModuleLoad(vs.Root(), "./trained-model.pt")
	if err != nil {
		log.Fatal(err)
	}

	m.SetEval()
	acc := nn.BatchAccuracyForLogits(vs, m, ds.TestImages, ds.TestLabels, device, 1024)

	fmt.Printf("Accuracy: %0.4f\n", acc)
}
