package nn_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func gruTest(rnnConfig *nn.RNNConfig, t *testing.T) {

	var (
		batchDim  int64 = 5
		seqLen    int64 = 3
		inputDim  int64 = 2
		outputDim int64 = 4
	)

	vs := nn.NewVarStore(gotch.CPU)
	path := vs.Root()

	gru := nn.NewGRU(path, inputDim, outputDim, rnnConfig)

	numDirections := int64(1)
	if rnnConfig.Bidirectional {
		numDirections = 2
	}
	layerDim := rnnConfig.NumLayers * numDirections

	// Step test
	input := ts.MustRandn([]int64{batchDim, inputDim}, gotch.Float, gotch.CPU)
	output := gru.Step(input, gru.ZeroState(batchDim).(*nn.GRUState))

	want := []int64{layerDim, batchDim, outputDim}
	got := output.(*nn.GRUState).Tensor.MustSize()

	if !reflect.DeepEqual(want, got) {
		fmt.Println("Step test:")
		t.Errorf("Expected ouput shape: %v\n", want)
		t.Errorf("Got output shape: %v\n", got)
	}

	// seq test
	input = ts.MustRandn([]int64{batchDim, seqLen, inputDim}, gotch.Float, gotch.CPU)
	output, _ = gru.Seq(input)
	wantSeq := []int64{batchDim, seqLen, outputDim * numDirections}
	gotSeq := output.(*ts.Tensor).MustSize()

	if !reflect.DeepEqual(wantSeq, gotSeq) {
		fmt.Println("Seq test:")
		t.Errorf("Expected ouput shape: %v\n", wantSeq)
		t.Errorf("Got output shape: %v\n", gotSeq)

	}
}

func TestGRU(t *testing.T) {

	cfg := nn.DefaultRNNConfig()

	gruTest(cfg, t)

	cfg.Bidirectional = true
	gruTest(cfg, t)

	cfg.NumLayers = 2
	cfg.Bidirectional = false
	gruTest(cfg, t)

	cfg.NumLayers = 2
	cfg.Bidirectional = true
	gruTest(cfg, t)
}

func lstmTest(rnnConfig *nn.RNNConfig, t *testing.T) {

	var (
		batchDim  int64 = 5
		seqLen    int64 = 3
		inputDim  int64 = 2
		outputDim int64 = 4
	)

	vs := nn.NewVarStore(gotch.CPU)
	path := vs.Root()

	lstm := nn.NewLSTM(path, inputDim, outputDim, rnnConfig)

	numDirections := int64(1)
	if rnnConfig.Bidirectional {
		numDirections = 2
	}
	layerDim := rnnConfig.NumLayers * numDirections

	// Step test
	input := ts.MustRandn([]int64{batchDim, inputDim}, gotch.Float, gotch.CPU)
	output := lstm.Step(input, lstm.ZeroState(batchDim).(*nn.LSTMState))

	wantH := []int64{layerDim, batchDim, outputDim}
	gotH := output.(*nn.LSTMState).Tensor1.MustSize()
	wantC := []int64{layerDim, batchDim, outputDim}
	gotC := output.(*nn.LSTMState).Tensor2.MustSize()

	if !reflect.DeepEqual(wantH, gotH) {
		fmt.Println("Step test:")
		t.Errorf("Expected ouput H shape: %v\n", wantH)
		t.Errorf("Got output H shape: %v\n", gotH)
	}

	if !reflect.DeepEqual(wantC, gotC) {
		fmt.Println("Step test:")
		t.Errorf("Expected ouput C shape: %v\n", wantC)
		t.Errorf("Got output C shape: %v\n", gotC)
	}

	// seq test
	input = ts.MustRandn([]int64{batchDim, seqLen, inputDim}, gotch.Float, gotch.CPU)
	output, _ = lstm.Seq(input)

	wantSeq := []int64{batchDim, seqLen, outputDim * numDirections}
	gotSeq := output.(*ts.Tensor).MustSize()

	if !reflect.DeepEqual(wantSeq, gotSeq) {
		fmt.Println("Seq test:")
		t.Errorf("Expected ouput shape: %v\n", wantSeq)
		t.Errorf("Got output shape: %v\n", gotSeq)
	}
}

func TestLSTM(t *testing.T) {

	cfg := nn.DefaultRNNConfig()

	lstmTest(cfg, t)

	cfg.Bidirectional = true
	lstmTest(cfg, t)

	cfg.NumLayers = 2
	cfg.Bidirectional = false
	lstmTest(cfg, t)

	cfg.NumLayers = 2
	cfg.Bidirectional = true
	lstmTest(cfg, t)
}
