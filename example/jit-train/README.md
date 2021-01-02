# Load and train Pytorch Model in Go

This example demonstrates how to load a Python Pytorch model using Torch Script, then train model in Go. 

- Step 1: convert Python Pytorch model to Torch Script. The detail can be found in [Pytorch tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html). Below is an example of a MNIST model.


```python
import torch
from torch.nn import Module
import torch.nn.functional as F

class MNISTModule(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.maxpool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.maxpool2 = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x).view(-1, 1024)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return self.linear2(x)

traced_script_module = torch.jit.script(MNISTModule())
traced_script_module.save("model.pt")

```

- Step 2: Load Torch Model and continue train/fine-tune in Go. After training, model can be saved in Torch Script format so that it can be either loaded in Go, Python, or any supported Pytorch binding languages. 

```go
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
	bestAccuracy := nn.BatchAccuracyForLogits(vs, trainable, ds.TestImages, ds.TestLabels, device, 1024)
	fmt.Printf("Initial Accuracy: %0.4f\n", bestAccuracy)

	opt, err := nn.DefaultAdamConfig().Build(vs, 1e-4)
	if err != nil {
		log.Fatal(err)
	}
	for epoch := 0; epoch < epochs; epoch++ {

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

	fmt.Printf("Completed training. Best accuracy: %0.4f\n", bestAccuracy)
}
```

- Further step: trained model can be loaded and evaluated. 

```go
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
```

See MNIST example for how to access MNIST dataset.

Below is a session of training and evaluate outputs:

```bash
go run .
Trainable JIT model loaded.
conv1.weight
conv1.bias
conv2.weight
conv2.bias
linear1.weight
linear1.bias
linear2.weight
linear2.bias
Initial Accuracy: 0.1122
Epoch: 0         Loss: 0.20      Test accuracy: 93.22%
Epoch: 1         Loss: 0.21      Test accuracy: 96.14%
Epoch: 2         Loss: 0.07      Test accuracy: 97.49%
Epoch: 3         Loss: 0.07      Test accuracy: 98.00%
Epoch: 4         Loss: 0.04      Test accuracy: 98.17%
Epoch: 5         Loss: 0.06      Test accuracy: 98.34%
Epoch: 6         Loss: 0.03      Test accuracy: 98.59%
Epoch: 7         Loss: 0.08      Test accuracy: 98.62%
Epoch: 8         Loss: 0.01      Test accuracy: 98.54%
Epoch: 9         Loss: 0.08      Test accuracy: 98.75%
Epoch: 10        Loss: 0.07      Test accuracy: 98.88%
Epoch: 11        Loss: 0.05      Test accuracy: 98.74%
Epoch: 12        Loss: 0.03      Test accuracy: 98.80%
Epoch: 13        Loss: 0.02      Test accuracy: 98.91%
Epoch: 14        Loss: 0.02      Test accuracy: 98.99%
Epoch: 15        Loss: 0.01      Test accuracy: 98.90%
Epoch: 16        Loss: 0.02      Test accuracy: 98.90%
Epoch: 17        Loss: 0.02      Test accuracy: 98.87%
Epoch: 18        Loss: 0.05      Test accuracy: 99.00%
Epoch: 19        Loss: 0.03      Test accuracy: 98.96%
Epoch: 20        Loss: 0.01      Test accuracy: 98.98%
Epoch: 21        Loss: 0.03      Test accuracy: 99.02%
Epoch: 22        Loss: 0.02      Test accuracy: 98.95%
Epoch: 23        Loss: 0.02      Test accuracy: 98.99%
Epoch: 24        Loss: 0.02      Test accuracy: 98.96%
Epoch: 25        Loss: 0.01      Test accuracy: 99.15%
Epoch: 26        Loss: 0.01      Test accuracy: 98.97%
Epoch: 27        Loss: 0.01      Test accuracy: 99.03%
Epoch: 28        Loss: 0.03      Test accuracy: 99.09%
Epoch: 29        Loss: 0.01      Test accuracy: 99.05%
Epoch: 30        Loss: 0.00      Test accuracy: 98.97%
Epoch: 31        Loss: 0.00      Test accuracy: 99.01%
Epoch: 32        Loss: 0.00      Test accuracy: 99.08%
Epoch: 33        Loss: 0.00      Test accuracy: 98.93%
Epoch: 34        Loss: 0.01      Test accuracy: 98.86%
Epoch: 35        Loss: 0.00      Test accuracy: 98.94%
Epoch: 36        Loss: 0.01      Test accuracy: 98.96%
Epoch: 37        Loss: 0.00      Test accuracy: 99.01%
Epoch: 38        Loss: 0.00      Test accuracy: 99.03%
Epoch: 39        Loss: 0.00      Test accuracy: 99.14%
Epoch: 40        Loss: 0.00      Test accuracy: 99.06%
Epoch: 41        Loss: 0.00      Test accuracy: 99.01%
Epoch: 42        Loss: 0.00      Test accuracy: 99.01%
Epoch: 43        Loss: 0.01      Test accuracy: 99.01%
Epoch: 44        Loss: 0.00      Test accuracy: 98.98%
Epoch: 45        Loss: 0.02      Test accuracy: 99.03%
Epoch: 46        Loss: 0.00      Test accuracy: 99.14%
Epoch: 47        Loss: 0.00      Test accuracy: 99.11%
Epoch: 48        Loss: 0.00      Test accuracy: 98.84%
Epoch: 49        Loss: 0.00      Test accuracy: 98.93%
Completed training. Best accuracy: 0.9915
```

```bash
go run . -task=infer
Accuracy: 0.9915
```




