package nn

// linear is a fully-connected layer

import (
	"math"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// LinearConfig is a configuration for a linear layer
type LinearConfig struct {
	WsInit Init // iniital weights
	BsInit Init // optional initial bias
	Bias   bool
}

// DefaultLinearConfig creates default LinearConfig with
// weights initiated using KaimingUniform and Bias is set to true
func DefaultLinearConfig() *LinearConfig {
	return &LinearConfig{
		WsInit: NewKaimingUniformInit(),
		BsInit: nil,
		Bias:   true,
	}
}

// Linear is a linear fully-connected layer
type Linear struct {
	Ws *ts.Tensor
	Bs *ts.Tensor
}

// NewLinear creates a new linear layer
// y = x*wT + b
// inDim - input dimension (x) [input features - columns]
// outDim - output dimension (y) [output features - columns]
// NOTE: w will have shape{outDim, inDim}; b will have shape{outDim}
func NewLinear(vs *Path, inDim, outDim int64, c *LinearConfig) *Linear {

	var bs *ts.Tensor
	// bs has size of output dimension
	switch c.Bias {
	case false:
		bs = ts.MustZeros([]int64{outDim}, gotch.Float, vs.Device())
	case true:
		switch {
		case c.BsInit == nil:
			bound := 1.0 / math.Sqrt(float64(inDim))
			bsInit := NewUniformInit(-bound, bound)
			bs = vs.NewVar("bias", []int64{outDim}, bsInit)
		case c.BsInit != nil:
			bs = vs.NewVar("bias", []int64{outDim}, c.BsInit)
		}
	}

	return &Linear{
		Ws: vs.NewVar("weight", []int64{outDim, inDim}, c.WsInit).MustT(false),
		Bs: bs,
	}
}

// Implement `Module` for `Linear` struct:
// =======================================

// Forward proceeds input node through linear layer.
// NOTE:
// - It assumes that node has dimensions of 2 (matrix).
// To make it work for matrix multiplication, input node should
// has same number of **column** as number of **column** in
// `LinearLayer` `Ws` property as weights matrix will be
// transposed before multiplied to input node. (They are all used `inDim`)
// - Input node should have shape of `shape{batch size, input features}`.
// (shape{batchSize, inDim}). The input features is `inDim` while the
// output feature is `outDim` in `LinearConfig` struct.
//
// Example:
//
// 	inDim := 3
// 	outDim := 2
// 	batchSize := 4
// 	weights: 2x3
// 	[ 1 1 1
// 		1 1 1 ]
//
// 	input node: 3x4
// 	[ 1 1 1
// 	  1 1 1
// 	  1 1 1
// 		1 1 1 ]
func (l *Linear) Forward(xs *ts.Tensor) (retVal *ts.Tensor) {

	mul := xs.MustMatmul(l.Ws, false)
	return mul.MustAdd(l.Bs, true)
}

// ForwardT implements ModuleT interface for Linear layer.
//
// NOTE: train param will not be used.
func (l *Linear) ForwardT(xs *ts.Tensor, train bool) (retVal *ts.Tensor) {

	mul := xs.MustMatmul(l.Ws, false)
	return mul.MustAdd(l.Bs, true)
}
