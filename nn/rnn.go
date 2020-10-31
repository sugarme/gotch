package nn

import (
	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

type State interface{}

type RNN interface {

	// A zero state from which the recurrent network is usually initialized.
	ZeroState(batchDim int64) State

	// Applies a single step of the recurrent network.
	//
	// The input should have dimensions [batch_size, features].
	Step(input *ts.Tensor, inState State) State

	// Applies multiple steps of the recurrent network.
	//
	// The input should have dimensions [batch_size, seq_len, features].
	// The initial state is the result of applying zero_state.
	Seq(input *ts.Tensor) (*ts.Tensor, State)

	// Applies multiple steps of the recurrent network.
	//
	// The input should have dimensions [batch_size, seq_len, features].
	SeqInit(input *ts.Tensor, inState State) (*ts.Tensor, State)
}

// The state for a LSTM network, this contains two tensors.
type LSTMState struct {
	Tensor1 *ts.Tensor
	Tensor2 *ts.Tensor
}

// The hidden state vector, which is also the output of the LSTM.
func (ls *LSTMState) H() *ts.Tensor {
	return ls.Tensor1.MustShallowClone()
}

// The cell state vector.
func (ls *LSTMState) C() *ts.Tensor {
	return ls.Tensor2.MustShallowClone()
}

// The GRU and LSTM layers share the same config.
// Configuration for the GRU and LSTM layers.
type RNNConfig struct {
	HasBiases     bool
	NumLayers     int64
	Dropout       float64
	Train         bool
	Bidirectional bool
	BatchFirst    bool
}

// Default creates default RNN configuration
func DefaultRNNConfig() *RNNConfig {
	return &RNNConfig{
		HasBiases:     true,
		NumLayers:     1,
		Dropout:       float64(0.0),
		Train:         true,
		Bidirectional: false,
		BatchFirst:    true,
	}
}

// A Long Short-Term Memory (LSTM) layer.
//
// https://en.wikipedia.org/wiki/Long_short-term_memory
type LSTM struct {
	flatWeights []ts.Tensor
	hiddenDim   int64
	config      *RNNConfig
	device      gotch.Device
}

// NewLSTM creates a LSTM layer.
func NewLSTM(vs *Path, inDim, hiddenDim int64, cfg *RNNConfig) *LSTM {

	var numDirections int64 = 1
	if cfg.Bidirectional {
		numDirections = 2
	}

	gateDim := 4 * hiddenDim
	flatWeights := make([]ts.Tensor, 0)

	for i := 0; i < int(cfg.NumLayers); i++ {
		for n := 0; n < int(numDirections); n++ {
			if i != 0 {
				inDim = hiddenDim * numDirections
			}

			wIh := vs.KaimingUniform("w_ih", []int64{gateDim, inDim})
			wHh := vs.KaimingUniform("w_hh", []int64{gateDim, hiddenDim})
			bIh := vs.Zeros("b_ih", []int64{gateDim})
			bHh := vs.Zeros("b_hh", []int64{gateDim})

			flatWeights = append(flatWeights, *wIh, *wHh, *bIh, *bHh)
		}
	}

	// if vs.Device().IsCuda() && gotch.Cuda.CudnnIsAvailable() {
	// TODO: check if Cudnn is available here!!!
	if vs.Device().IsCuda() {
		// NOTE. 2 is for LSTM
		// ref. rnn.cpp in Pytorch
		ts.Must_CudnnRnnFlattenWeight(flatWeights, 4, inDim, 2, hiddenDim, cfg.NumLayers, cfg.BatchFirst, cfg.Bidirectional)
	}

	return &LSTM{
		flatWeights: flatWeights,
		hiddenDim:   hiddenDim,
		config:      cfg,
		device:      vs.Device(),
	}

}

// Implement RNN interface for LSTM:
// =================================

func (l *LSTM) ZeroState(batchDim int64) State {
	var numDirections int64 = 1
	if l.config.Bidirectional {
		numDirections = 2
	}

	layerDim := l.config.NumLayers * numDirections
	shape := []int64{layerDim, batchDim, l.hiddenDim}
	zeros := ts.MustZeros(shape, gotch.Float, l.device)

	retVal := &LSTMState{
		Tensor1: zeros.MustShallowClone(),
		Tensor2: zeros.MustShallowClone(),
	}

	zeros.MustDrop()

	return retVal
}

func (l *LSTM) Step(input *ts.Tensor, inState State) State {
	ip := input.MustUnsqueeze(1, false)

	output, state := l.SeqInit(ip, inState)

	// NOTE: though we won't use `output`, it is a Ctensor created in C land, so
	// it should be cleaned up here to prevent memory hold-up.
	output.MustDrop()

	return state
}

func (l *LSTM) Seq(input *ts.Tensor) (*ts.Tensor, State) {
	batchDim := input.MustSize()[0]
	inState := l.ZeroState(batchDim)

	output, state := l.SeqInit(input, inState)

	// Delete intermediate tensors in inState
	inState.(*LSTMState).Tensor1.MustDrop()
	inState.(*LSTMState).Tensor2.MustDrop()

	return output, state
}

func (l *LSTM) SeqInit(input *ts.Tensor, inState State) (*ts.Tensor, State) {

	output, h, c := input.MustLstm([]ts.Tensor{*inState.(*LSTMState).Tensor1, *inState.(*LSTMState).Tensor2}, l.flatWeights, l.config.HasBiases, l.config.NumLayers, l.config.Dropout, l.config.Train, l.config.Bidirectional, l.config.BatchFirst)

	return output, &LSTMState{
		Tensor1: h,
		Tensor2: c,
	}
}

// GRUState is a GRU state. It contains a single tensor.
type GRUState struct {
	Tensor *ts.Tensor
}

func (gs *GRUState) Value() *ts.Tensor {
	return gs.Tensor
}

// A Gated Recurrent Unit (GRU) layer.
//
// https://en.wikipedia.org/wiki/Gated_recurrent_unit
type GRU struct {
	flatWeights []ts.Tensor
	hiddenDim   int64
	config      *RNNConfig
	device      gotch.Device
}

// NewGRU create a new GRU layer
func NewGRU(vs *Path, inDim, hiddenDim int64, cfg *RNNConfig) (retVal *GRU) {
	var numDirections int64 = 1
	if cfg.Bidirectional {
		numDirections = 2
	}

	gateDim := 3 * hiddenDim
	flatWeights := make([]ts.Tensor, 0)

	for i := 0; i < int(cfg.NumLayers); i++ {
		for n := 0; n < int(numDirections); n++ {
			var inputDim int64
			if i == 0 {
				inputDim = inDim
			} else {
				inputDim = hiddenDim * numDirections
			}

			wIh := vs.KaimingUniform("w_ih", []int64{gateDim, inputDim})
			wHh := vs.KaimingUniform("w_hh", []int64{gateDim, hiddenDim})
			bIh := vs.Zeros("b_ih", []int64{gateDim})
			bHh := vs.Zeros("b_hh", []int64{gateDim})

			flatWeights = append(flatWeights, *wIh, *wHh, *bIh, *bHh)
		}
	}

	if vs.Device().IsCuda() {
		// NOTE. 3 is for GRU
		// ref. rnn.cpp in Pytorch
		ts.Must_CudnnRnnFlattenWeight(flatWeights, 4, inDim, 3, hiddenDim, cfg.NumLayers, cfg.BatchFirst, cfg.Bidirectional)
	}

	return &GRU{
		flatWeights: flatWeights,
		hiddenDim:   hiddenDim,
		config:      cfg,
		device:      vs.Device(),
	}
}

// Implement RNN interface for GRU:
// ================================

func (g *GRU) ZeroState(batchDim int64) State {
	var numDirections int64 = 1
	if g.config.Bidirectional {
		numDirections = 2
	}

	layerDim := g.config.NumLayers * numDirections
	shape := []int64{layerDim, batchDim, g.hiddenDim}

	tensor := ts.MustZeros(shape, gotch.Float, g.device)

	return &GRUState{Tensor: tensor}
}

func (g *GRU) Step(input *ts.Tensor, inState State) State {
	unsqueezedInput := input.MustUnsqueeze(1, false)
	output, state := g.SeqInit(unsqueezedInput, inState)

	// NOTE: though we won't use `output`, it is a Ctensor created in C land, so
	// it should be cleaned up here to prevent memory hold-up.
	output.MustDrop()
	unsqueezedInput.MustDrop()

	return state
}

func (g *GRU) Seq(input *ts.Tensor) (*ts.Tensor, State) {
	batchDim := input.MustSize()[0]
	inState := g.ZeroState(batchDim)

	output, state := g.SeqInit(input, inState)

	// Delete intermediate tensors in inState
	inState.(*GRUState).Tensor.MustDrop()

	return output, state
}

func (g *GRU) SeqInit(input *ts.Tensor, inState State) (*ts.Tensor, State) {

	output, h := input.MustGru(inState.(*GRUState).Tensor, g.flatWeights, g.config.HasBiases, g.config.NumLayers, g.config.Dropout, g.config.Train, g.config.Bidirectional, g.config.BatchFirst)

	return output, &GRUState{Tensor: h}
}
