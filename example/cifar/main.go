package main

// Training various models on the CIFAR-10 dataset.
//
// The dataset can be downloaded from https:www.cs.toronto.edu/~kriz/cifar.html, files
// should be placed in the data/ directory.
//
// The resnet model reaches 95.4% accuracy.

import (
	"fmt"
	"log"
	"time"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

func convBn(p *nn.Path, cIn, cOut int64) *nn.SequentialT {
	config := nn.DefaultConv2DConfig()
	config.Padding = []int64{1, 1}
	config.Bias = false

	seq := nn.SeqT()

	seq.Add(nn.NewConv2D(p, cIn, cOut, 3, config))
	seq.Add(nn.BatchNorm2D(p, cOut, nn.DefaultBatchNormConfig()))
	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	return seq
}

func layer(p *nn.Path, cIn, cOut int64) nn.FuncT {
	pre := convBn(p.Sub("pre"), cIn, cOut)
	block1 := convBn(p.Sub("b1"), cOut, cOut)
	block2 := convBn(p.Sub("b2"), cOut, cOut)

	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		tmp1 := xs.ApplyT(pre, train)
		preTs := tmp1.MaxPool2DDefault(2, true)
		tmp2 := preTs.ApplyT(block1, train)
		ys := tmp2.ApplyT(block2, train)
		tmp2.MustDrop()

		res := preTs.MustAdd(ys, true)
		ys.MustDrop()

		return res
	})
}

func fastResnet(p *nn.Path) *nn.SequentialT {
	seq := nn.SeqT()

	seq.Add(convBn(p.Sub("pre"), 3, 64))
	seq.Add(layer(p.Sub("layer1"), 64, 128))
	seq.Add(convBn(p.Sub("inter"), 128, 256))
	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MaxPool2DDefault(2, false)
	}))
	seq.Add(layer(p.Sub("layer2"), 256, 512))
	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		tmp := xs.MaxPool2DDefault(4, false)
		res := tmp.FlatView()
		tmp.MustDrop()

		return res
	}))

	seq.Add(nn.NewLinear(p.Sub("linear"), 512, 10, nn.DefaultLinearConfig()))
	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustMulScalar(ts.FloatScalar(0.125), false)
	}))

	return seq
}

func main() {
	dir := "../../data/cifar10"
	ds := vision.CFLoadDir(dir)

	fmt.Printf("TrainImages shape: %v\n", ds.TrainImages.MustSize())
	fmt.Printf("TrainLabel shape: %v\n", ds.TrainLabels.MustSize())
	fmt.Printf("TestImages shape: %v\n", ds.TestImages.MustSize())
	fmt.Printf("TestLabel shape: %v\n", ds.TestLabels.MustSize())
	fmt.Printf("Number of labels: %v\n", ds.Labels)

	// device := gotch.CPU
	device := gotch.NewCuda().CudaIfAvailable()

	vs := nn.NewVarStore(device)

	net := fastResnet(vs.Root())

	var lossVal float64
	startTime := time.Now()

	var bestAccuracy float64

	for epoch := 0; epoch < 150; epoch++ {
		optConfig := nn.NewSGDConfig(0.9, 0.0, 5e-4, true)
		var (
			opt *nn.Optimizer
			err error
		)
		switch {
		case epoch < 50:
			opt, err = optConfig.Build(vs, 0.1)
			if err != nil {
				log.Fatal(err)
			}
		case epoch < 100:
			opt, err = optConfig.Build(vs, 0.01)
			if err != nil {
				log.Fatal(err)
			}
		case epoch >= 100:
			opt, err = optConfig.Build(vs, 0.001)
			if err != nil {
				log.Fatal(err)
			}
		}

		iter := ts.MustNewIter2(ds.TrainImages, ds.TrainLabels, int64(64))
		iter.Shuffle()

		for {
			item, ok := iter.Next()
			if !ok {
				break
			}

			devicedData := item.Data.MustTo(vs.Device(), true)
			devicedLabel := item.Label.MustTo(vs.Device(), true)
			bimages := vision.Augmentation(devicedData, true, 4, 8)

			logits := net.ForwardT(bimages, true)

			loss := logits.CrossEntropyForLogits(devicedLabel)
			opt.BackwardStep(loss)

			lossVal = loss.Float64Values()[0]

			devicedData.MustDrop()
			devicedLabel.MustDrop()
			bimages.MustDrop()
			loss.MustDrop()
		}

		testAcc := nn.BatchAccuracyForLogits(vs, net, ds.TestImages, ds.TestLabels, vs.Device(), 512)
		fmt.Printf("Epoch:\t %v\t Loss: \t %.3f \tAcc: %10.2f%%\n", epoch, lossVal, testAcc*100.0)
		if testAcc > bestAccuracy {
			bestAccuracy = testAcc
		}
		iter.Drop()
	}

	fmt.Printf("Best Accuracy: %10.2f%%\n", bestAccuracy*100.0)
	fmt.Printf("Taken time:\t%.2f mins\n", time.Since(startTime).Minutes())
}
