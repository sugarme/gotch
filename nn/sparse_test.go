package nn_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

func embeddingTest(embeddingConfig *nn.EmbeddingConfig, t *testing.T) {

	var (
		batchDim  int64 = 5
		seqLen    int64 = 7
		inputDim  int64 = 10
		outputDim int64 = 4
	)

	vs := nn.NewVarStore(gotch.CPU)
	embeddings := nn.NewEmbedding(vs.Root(), inputDim, outputDim, embeddingConfig)

	// Forward test
	input := ts.MustRandint(10, []int64{batchDim, seqLen}, gotch.Int64, gotch.CPU)
	output := embeddings.Forward(input)

	want := []int64{batchDim, seqLen, outputDim}
	got := output.MustSize()

	if !reflect.DeepEqual(got, want) {
		t.Errorf("Forward - Expected output shape: %v\n", want)
		t.Errorf("Forward - Got output shape: %v\n", got)
	}

	// Padding test
	paddingIdx := embeddingConfig.PaddingIdx
	if embeddingConfig.PaddingIdx < 0 {
		paddingIdx = inputDim + embeddingConfig.PaddingIdx
	}

	input = ts.MustOfSlice([]int64{paddingIdx})
	output = embeddings.Forward(input)
	want = []int64{1, outputDim}
	got = output.MustSize()

	if !reflect.DeepEqual(got, want) {
		t.Errorf("Padding - Expected output shape: %v\n", want)
		t.Errorf("Padding - Got output shape: %v\n", got)
	}

}

func TestEmbedding(t *testing.T) {

	cfg := nn.DefaultEmbeddingConfig()
	embeddingTest(cfg, t)

	cfg.PaddingIdx = -1
	embeddingTest(cfg, t)

	cfg.PaddingIdx = 0
	embeddingTest(cfg, t)
}
