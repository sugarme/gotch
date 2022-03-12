package nn

// A batch-normalization layer.

import (
	"log"

	"github.com/sugarme/gotch/ts"
)

// Batch-normalization config.
type BatchNormConfig struct {
	CudnnEnable bool
	Eps         float64
	Momentum    float64
	WsInit      Init
	BsInit      Init
}

func DefaultBatchNormConfig() *BatchNormConfig {
	return &BatchNormConfig{
		CudnnEnable: true,
		Eps:         1e-5,
		Momentum:    0.1,
		WsInit:      NewUniformInit(0.0, 1.0),
		BsInit:      NewConstInit(0.0),
	}
}

// A batch-normalization layer.
type BatchNorm struct {
	config      *BatchNormConfig
	RunningMean *ts.Tensor
	RunningVar  *ts.Tensor
	Ws          *ts.Tensor
	Bs          *ts.Tensor
	Nd          uint
}

// NewBatchNorm creates a new BatchNorm layer
func NewBatchNorm(vs *Path, nd uint, outDim int64, config *BatchNormConfig) *BatchNorm {
	return &BatchNorm{
		config:      config,
		RunningMean: vs.MustZerosNoTrain("running_mean", []int64{outDim}),
		RunningVar:  vs.MustOnesNoTrain("running_var", []int64{outDim}),
		Ws:          vs.MustNewVar("weight", []int64{outDim}, config.WsInit),
		Bs:          vs.MustNewVar("bias", []int64{outDim}, config.BsInit),
	}
}

// Applies Batch Normalization over a three dimension input.
//
// The input shape is assumed to be (N, C, L). Normalization
// is performed over the first batch dimension N.
func BatchNorm1D(vs *Path, outDim int64, config *BatchNormConfig) *BatchNorm {
	return NewBatchNorm(vs, 1, outDim, config)
}

// Applies Batch Normalization over a four dimension input.
//
// The input shape is assumed to be (N, C, H, W). Normalization
// is performed over the first batch dimension N.
func BatchNorm2D(vs *Path, outDim int64, config *BatchNormConfig) *BatchNorm {
	return NewBatchNorm(vs, 2, outDim, config)
}

// Applies Batch Normalization over a five dimension input.
//
// The input shape is assumed to be (N, C, D, H, W). Normalization
// is performed over the first batch dimension N.
func BatchNorm3D(vs *Path, outDim int64, config *BatchNormConfig) *BatchNorm {
	return NewBatchNorm(vs, 3, outDim, config)
}

// Implement ModuleT interface for BatchNorm:
// ==========================================

func (bn *BatchNorm) ForwardT(xs *ts.Tensor, train bool) (retVal *ts.Tensor) {

	dim := xs.Dim()

	if bn.Nd == 1 && dim != 2 && dim != 3 {
		log.Fatalf("Expected an input tensor with 2 or 3 dims, got %v\n", xs.MustSize())
	}

	if bn.Nd > 1 && int(dim) != int(bn.Nd)+2 {
		log.Fatalf("Expected an input tensor with %v dims, got %v\n", bn.Nd+2, xs.MustSize())
	}

	return ts.MustBatchNorm(xs, bn.Ws, bn.Bs, bn.RunningMean, bn.RunningVar, train, bn.config.Momentum, bn.config.Eps, bn.config.CudnnEnable)

}

// Forward forwards inputs through the module.
// NOTE.
// This forwarding will update BatchNorm weight by default (training=true).
// Wrap module with tensor.NoGrad() when running model inference mode.
func (bn *BatchNorm) Forward(xs *ts.Tensor) (retVal *ts.Tensor) {
	dim := xs.Dim()

	if bn.Nd == 1 && dim != 2 && dim != 3 {
		log.Fatalf("Expected an input tensor with 2 or 3 dims, got %v\n", xs.MustSize())
	}

	if bn.Nd > 1 && int(dim) != int(bn.Nd)+2 {
		log.Fatalf("Expected an input tensor with %v dims, got %v\n", bn.Nd+2, xs.MustSize())
	}

	return ts.MustBatchNorm(xs, bn.Ws, bn.Bs, bn.RunningMean, bn.RunningVar, true, bn.config.Momentum, bn.config.Eps, bn.config.CudnnEnable)
}
