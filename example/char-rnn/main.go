package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

const (
	LearningRate float64 = 0.01
	HiddenSize   int64   = 256
	SeqLen       int64   = 180
	BatchSize    int64   = 256
	Epochs       int     = 100
	SamplingLen  int64   = 1024
)

func sample(data *ts.TextData, lstm *nn.LSTM, linear *nn.Linear, device gotch.Device) string {

	labels := data.Labels()
	inState := lstm.ZeroState(1)
	lastLabel := int64(0)
	var runes []rune

	for i := 0; i < int(SamplingLen); i++ {
		input := ts.MustZeros([]int64{1, labels}, gotch.Float, device)
		// NOTE. `Narrow` creates tensor that shares same storage
		inputView := input.MustNarrow(1, lastLabel, 1, false)
		inputView.MustFill_(ts.FloatScalar(1.0))

		state := lstm.Step(input, inState)

		// 1. Delete inState tensors (from C land memory)
		inState.(*nn.LSTMState).Tensor1.MustDrop()
		inState.(*nn.LSTMState).Tensor2.MustDrop()
		// 2. Then update with current state
		inState = state
		// 3. Delete intermediate tensors
		input.MustDrop()
		inputView.MustDrop()

		forwardTs := linear.Forward(state.(*nn.LSTMState).H()).MustSqueezeDim(0, true).MustSoftmax(-1, gotch.Float, true)
		sampledY := forwardTs.MustMultinomial(1, false, true)
		lastLabel = sampledY.Int64Values()[0]
		sampledY.MustDrop()
		char := data.LabelForChar(lastLabel)

		runes = append(runes, char)
	}

	// Delete the last state
	inState.(*nn.LSTMState).Tensor1.MustDrop()
	inState.(*nn.LSTMState).Tensor2.MustDrop()

	return string(runes)
}

func main() {
	cuda := gotch.NewCuda()
	device := cuda.CudaIfAvailable()

	vs := nn.NewVarStore(device)
	data, err := ts.NewTextData("../../data/char-rnn/input.txt")
	if err != nil {
		log.Fatal(err)
	}

	labels := data.Labels()
	fmt.Printf("Dataset loaded, %v labels\n", labels)

	lstm := nn.NewLSTM(vs.Root(), labels, HiddenSize, nn.DefaultRNNConfig())
	linear := nn.NewLinear(vs.Root(), HiddenSize, labels, nn.DefaultLinearConfig())

	optConfig := nn.DefaultAdamConfig()
	opt, err := optConfig.Build(vs, LearningRate)
	if err != nil {
		log.Fatal(err)
	}

	for epoch := 1; epoch <= Epochs; epoch++ {
		sumLoss := 0.0
		cntLoss := 0.0

		dataIter := data.IterShuffle(SeqLen+1, BatchSize)

		batchCount := 0
		for {
			batchTs, ok := dataIter.Next()
			if !ok {
				break
			}

			batchNarrow := batchTs.MustNarrow(1, 0, SeqLen, false)
			xsOnehotTmp := batchNarrow.Onehot(labels)
			xsOnehot := xsOnehotTmp.MustTo(device, true) // shape: [256, 180, 65]
			ysTmp1 := batchTs.MustNarrow(1, 1, SeqLen, true)
			ysTmp2 := ysTmp1.MustTotype(gotch.Int64, true)
			ysTmp3 := ysTmp2.MustTo(device, true)
			ys := ysTmp3.MustView([]int64{BatchSize * SeqLen}, true)

			lstmOut, outState := lstm.Seq(xsOnehot)
			// NOTE. Although outState will not be used. There a hidden memory usage
			// on C land memory that is needed to free up. Don't use `_`
			outState.(*nn.LSTMState).Tensor1.MustDrop()
			outState.(*nn.LSTMState).Tensor2.MustDrop()

			logits := linear.Forward(lstmOut)
			lossView := logits.MustView([]int64{BatchSize * SeqLen, labels}, true)

			loss := lossView.CrossEntropyForLogits(ys)

			opt.BackwardStepClip(loss, 0.5)
			sumLoss += loss.Float64Values()[0]
			cntLoss += 1.0

			// batchTs.MustDrop()
			// batchNarrow.MustDrop()
			// xsOnehotTmp.MustDrop()
			xsOnehot.MustDrop()
			ys.MustDrop()
			lstmOut.MustDrop()
			loss.MustDrop()

			batchCount++
			if batchCount%500 == 0 {
				fmt.Printf("Epoch %v - Batch %v \n", epoch, batchCount)
			}
		}

		sampleStr := sample(data, lstm, linear, device)
		fmt.Printf("Epoch %v - Loss: %v \n", epoch, sumLoss/cntLoss)
		fmt.Println(sampleStr)

		dataIter.Data.MustDrop()
		dataIter.Indexes.MustDrop()
	}

}
