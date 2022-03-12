/* Translation with a Sequence to Sequence Model and Attention.

   This follows the line of the PyTorch tutorial:
   https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
   And trains a Sequence to Sequence (seq2seq) model using attention to
   perform translation between French and English.

   The dataset can be downloaded from the following link:
   https://download.pytorch.org/tutorial/data.zip
   The eng-fra.txt file should be moved in the data directory.
*/

package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

var (
	MaxLength    int64   = 10
	LearningRate float64 = 0.001
	Samples      int64   = 100000
	HiddenSize   int64   = 256
)

type Encoder struct {
	embedding nn.Embedding
	gru       nn.GRU
}

func newEncoder(vs *nn.Path, inDim, hiddenDim int64) *Encoder {

	gru := nn.NewGRU(vs, hiddenDim, hiddenDim, nn.DefaultRNNConfig())

	embedding := nn.NewEmbedding(vs, inDim, hiddenDim, nn.DefaultEmbeddingConfig())

	return &Encoder{*embedding, *gru}
}

func (e *Encoder) forward(xs *ts.Tensor, state *nn.GRUState) (*ts.Tensor, *nn.GRUState) {

	retTs := e.embedding.Forward(xs).MustView([]int64{1, -1}, true)
	retState := e.gru.Step(retTs, state).(*nn.GRUState)

	return retTs, retState
}

type Decoder struct {
	device      gotch.Device
	embedding   *nn.Embedding
	gru         *nn.GRU
	attn        *nn.Linear
	attnCombine *nn.Linear
	linear      *nn.Linear
}

func newDecoder(vs *nn.Path, hiddenDim, outDim int64) *Decoder {

	return &Decoder{
		device:      vs.Device(),
		embedding:   nn.NewEmbedding(vs, outDim, hiddenDim, nn.DefaultEmbeddingConfig()),
		gru:         nn.NewGRU(vs, hiddenDim, hiddenDim, nn.DefaultRNNConfig()),
		attn:        nn.NewLinear(vs, hiddenDim*2, MaxLength, nn.DefaultLinearConfig()),
		attnCombine: nn.NewLinear(vs, hiddenDim*2, hiddenDim, nn.DefaultLinearConfig()),
		linear:      nn.NewLinear(vs, hiddenDim, outDim, nn.DefaultLinearConfig()),
	}
}

func (d *Decoder) forward(xs *ts.Tensor, state *nn.GRUState, encOutputs *ts.Tensor, isTraining bool) (*ts.Tensor, *nn.GRUState) {

	forwardTsTmp := d.embedding.Forward(xs)
	forwardTsTmp.MustDropout_(0.1, isTraining)
	forwardTs := forwardTsTmp.MustView([]int64{1, -1}, true)

	// NOTE. forwardTs shape: [1, 256] state [1, 1, 256]
	// hence, just get state[0] of 3D tensor state
	stateTs := state.Value().MustShallowClone().MustView([]int64{1, -1}, true)
	catTs := ts.MustCat([]ts.Tensor{*forwardTs, *stateTs}, 1)
	stateTs.MustDrop()

	// NOTE. d.attn Ws shape : [512, 10]
	appliedTs := catTs.Apply(d.attn)
	catTs.MustDrop()
	attnWeights := appliedTs.MustUnsqueeze(0, true)

	size3, err := encOutputs.Size3()
	if err != nil {
		log.Fatal(err)
	}
	sz1 := size3[0]
	sz2 := size3[1]
	sz3 := size3[2]

	var encOutputsTs *ts.Tensor
	if sz2 == MaxLength {
		encOutputsTs = encOutputs.MustShallowClone()
	} else {
		shape := []int64{sz1, MaxLength - sz2, sz3}
		zerosTs := ts.MustZeros(shape, gotch.Float, d.device)
		encOutputsTs = ts.MustCat([]ts.Tensor{*encOutputs, *zerosTs}, 1)
		zerosTs.MustDrop()
	}

	attnApplied := attnWeights.MustBmm(encOutputsTs, true).MustSqueeze1(1, true)
	encOutputsTs.MustDrop()

	cTs := ts.MustCat([]ts.Tensor{*forwardTs, *attnApplied}, 1)
	forwardTs.MustDrop()
	attnApplied.MustDrop()
	aTs := cTs.Apply(d.attnCombine)
	cTs.MustDrop()
	xsTs := aTs.MustRelu(true)

	retState := d.gru.Step(xsTs, state).(*nn.GRUState)
	xsTs.MustDrop()

	retTs := d.linear.Forward(retState.Value()).MustLogSoftmax(-1, gotch.Float, true)

	return retTs, retState
}

type Model struct {
	encoder      *Encoder
	decoder      *Decoder
	decoderStart *ts.Tensor
	decoderEos   int64
	device       gotch.Device
}

func newModel(vs *nn.Path, ilang Lang, olang Lang, hiddenDim int64) *Model {
	return &Model{
		encoder:      newEncoder(vs.Sub("enc"), int64(ilang.Len()), hiddenDim),
		decoder:      newDecoder(vs.Sub("dec"), hiddenDim, int64(olang.Len())),
		decoderStart: ts.MustOfSlice([]int64{int64(olang.SosToken())}).MustTo(vs.Device(), true),
		decoderEos:   int64(olang.EosToken()),
		device:       vs.Device(),
	}
}

func (m *Model) trainLoss(input []int, target []int) *ts.Tensor {
	state := m.encoder.gru.ZeroState(1)
	var encOutputs []ts.Tensor

	for _, v := range input {
		s := ts.MustOfSlice([]int64{int64(v)}).MustTo(m.device, true)
		outTs, outState := m.encoder.forward(s, state.(*nn.GRUState))
		s.MustDrop()
		encOutputs = append(encOutputs, *outTs)
		state.(*nn.GRUState).Tensor.MustDrop()
		state = outState
	}

	stackTs := ts.MustStack(encOutputs, 1)
	for _, t := range encOutputs {
		t.MustDrop()
	}

	// TODO: should we implement random here???
	loss := ts.TensorFrom([]float32{0.0}).MustTo(m.device, true)
	prev := m.decoderStart.MustShallowClone()

	for _, s := range target {
		// TODO: fix memory leak at decoder.forward
		outTs, outState := m.decoder.forward(prev, state.(*nn.GRUState), stackTs, true)
		state.(*nn.GRUState).Tensor.MustDrop()
		state = outState

		targetTs := ts.MustOfSlice([]int64{int64(s)}).MustTo(m.device, true)

		outTsView := outTs.MustView([]int64{1, -1}, false)
		currLoss := outTsView.MustNLLLoss(targetTs, true)
		targetTs.MustDrop()

		loss.MustAdd_(currLoss)
		currLoss.MustDrop()

		noUseTs, output := outTs.MustTopK(1, -1, true, true)
		noUseTs.MustDrop()

		if m.decoderEos == outTs.Int64Values()[0] {
			prev.MustDrop()
			prev = output
			outTs.MustDrop()
			break
		}

		prev.MustDrop()
		prev = output
		outTs.MustDrop()
	}

	state.(*nn.GRUState).Tensor.MustDrop()
	stackTs.MustDrop()
	prev.MustDrop()

	return loss

}

func (m *Model) predict(input []int) []int {
	state := m.encoder.gru.ZeroState(1)
	var encOutputs []ts.Tensor

	for _, v := range input {
		s := ts.MustOfSlice([]int64{int64(v)}).MustTo(m.device, true)
		outTs, outState := m.encoder.forward(s, state.(*nn.GRUState))

		encOutputs = append(encOutputs, *outTs)
		state.(*nn.GRUState).Tensor.MustDrop()
		state = outState
	}

	stackTs := ts.MustStack(encOutputs, 1)
	for _, t := range encOutputs {
		t.MustDrop()
	}

	prev := m.decoderStart.MustShallowClone()
	var outputSeq []int

	for i := 0; i < int(MaxLength); i++ {
		outTs, outState := m.decoder.forward(prev, state.(*nn.GRUState), stackTs, true)
		_, output := outTs.MustTopK(1, -1, true, true)
		outputVal := output.Int64Values()[0]
		outputSeq = append(outputSeq, int(outputVal))

		if m.decoderEos == outTs.Int64Values()[0] {
			break
		}

		state.(*nn.GRUState).Tensor.MustDrop()
		state = outState
		prev.MustDrop()
		prev = output
	}

	return outputSeq

}

type LossStats struct {
	totalLoss float64
	samples   int
}

func newLossStats() *LossStats {
	return &LossStats{
		totalLoss: 0.0,
		samples:   0,
	}
}

func (ls *LossStats) update(loss float64) {
	ls.totalLoss += loss
	ls.samples += 1
}

func (ls *LossStats) avgAndReset() float64 {
	avg := ls.totalLoss / float64(ls.samples)
	ls.totalLoss = 0.0
	ls.samples = 0
	return avg
}

func main() {

	dataset := newDataset("eng", "fra", int(MaxLength)).Reverse()

	ilang := dataset.InputLang()
	olang := dataset.OutputLang()
	pairs := dataset.Pairs()

	fmt.Printf("Input: %v %v words\n", ilang.GetName(), ilang.Len())
	fmt.Printf("Output: %v %v words\n", olang.GetName(), olang.Len())
	fmt.Printf("Pairs: %v\n", len(pairs))

	// TODO: should we implement random here??

	cuda := gotch.NewCuda()
	device := cuda.CudaIfAvailable()

	vs := nn.NewVarStore(device)

	model := newModel(vs.Root(), ilang, olang, HiddenSize)

	optConfig := nn.DefaultAdamConfig()
	opt, err := optConfig.Build(vs, LearningRate)
	if err != nil {
		log.Fatal(err)
	}

	lossStats := newLossStats()

	for i := 1; i < int(Samples); i++ {
		// randomly choose a pair
		idx := rand.Intn(len(pairs))
		pair := pairs[idx]
		input := pair.Val1
		target := pair.Val2
		loss := model.trainLoss(input, target)
		opt.BackwardStep(loss)
		lossStats.update(loss.Float64Values()[0] / float64(len(target)))
		loss.MustDrop()

		if i%1000 == 0 {
			fmt.Printf("Trained %v samples -  Avg. Loss: %v\n", i, lossStats.avgAndReset())
			for predIdx := 1; predIdx <= 5; predIdx++ {
				idx := rand.Intn(len(pairs))
				in := pairs[idx].Val1
				tgt := pairs[idx].Val2
				predict := model.predict(in)

				fmt.Printf("input: %v\n", ilang.SeqToString(in))
				fmt.Printf("target: %v\n", olang.SeqToString(tgt))
				fmt.Printf("ouput: %v\n", olang.SeqToString(predict))
			}
		}
	}

}
