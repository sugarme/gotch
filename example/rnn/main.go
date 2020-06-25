package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func rnnTest(rnnConfig nn.RNNConfig) {

	var (
		batchDim int64 = 5
		// seqLen    int64 = 3
		inputDim  int64 = 2
		outputDim int64 = 4
	)

	vs := nn.NewVarStore(gotch.CPU)
	path := vs.Root()

	gru := nn.NewGRU(&path, inputDim, outputDim, rnnConfig)

	numDirections := int64(1)
	if rnnConfig.Bidirectional {
		numDirections = 2
	}
	layerDim := rnnConfig.NumLayers * numDirections

	// Step test
	input := ts.MustRandn([]int64{batchDim, inputDim}, gotch.Float, gotch.CPU)
	output := gru.Step(input, gru.ZeroState(batchDim).(nn.GRUState))

	fmt.Printf("Expected ouput shape: %v\n", []int64{layerDim, batchDim, outputDim})
	fmt.Printf("Got output shape: %v\n", output.(nn.GRUState).Tensor.MustSize())

}

func main() {

	rnnTest(nn.DefaultRNNConfig())
}
