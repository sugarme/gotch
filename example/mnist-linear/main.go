package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

const (
	N         = 60000 // number of rows in train data (training size)
	inDim     = 784   // input features - columns in train data (image data is 28x28pixel matrix)
	outDim    = 10    // output features (probabilities for digits 0-9)
	batchSize = 50
	batches   = N / batchSize
	epochs    = 100
)

var (
	trainX ts.Tensor
	trainY ts.Tensor
	testX  ts.Tensor
	testY  ts.Tensor

	err error
)

func init() {
	// load the train set
	// trainX is input tensor with shape{60000, 784} (image size: 28x28 pixels)
	// trainY is target tensor with shape{6000, 10}
	// (represent probabilities for digit 0-9)
	// E.g. [0.1 0.1 0.1 0.1 0.1 0.9 0.1 0.1 0.1 0.1]
	trainX, trainY, err = Load("train", "../testdata/mnist", gotch.Double)
	handleError(err)
	// load our test set
	testX, testY, err = Load("test", "../testdata/mnist", gotch.Double)
	handleError(err)

}

func main() {

	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < batches; i++ {
			// NOTE: `m.Reset()` does not delete data. It just moves pointer to starting point.
			start := i * batchSize
			end := start + batchSize

			dims, err := trainX.Size()
			handleError(err)

			if start > ts.FlattenDim(dims) || end > ts.FlattenDim(dims) {
				break
			}

			index := ts.NewNarrow(int64(start), int64(end))

			batchX := trainX.Idx(index)
			batchX.Print()

			fmt.Printf("Processed epoch %v - sample %v\n", epoch, i)

			panic("Stop")

			// batchX, err := trainX.Slice(nn.MakeRangedSlice(start, end))
			// handleError(err)
			// batchY, err := trainY.Slice(nn.MakeRangedSlice(start, end))
			// handleError(err)
			// xi := batchX.Data().([]float64)
			// yi := batchY.Data().([]float64)
			//
			// xiT := ts.New(ts.WithShape(batchSize, inDim), ts.WithBacking(xi))
			// yiT := ts.New(ts.WithShape(batchSize, outDim), ts.WithBacking(yi))

		}

	}
}

func handleError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
