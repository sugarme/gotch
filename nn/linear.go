package nn

// linear is a fully-connected layer

import (
	"fmt"
	"math"

	"github.com/sugarme/gotch/ts"
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
	negSlope := math.Sqrt(5)
	return &LinearConfig{
		WsInit: NewKaimingUniformInit(WithKaimingNegativeSlope(negSlope)),
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
	if c.Bias {
		switch {
		case c.BsInit == nil:
			shape := []int64{inDim, outDim}
			fanIn, _, err := CalculateFans(shape)
			if err != nil {
				err := fmt.Errorf("NewLinear() initiate bias failed: %v", err)
				panic(err)
			}
			bound := 0.0
			if fanIn > 0 {
				bound = 1 / math.Sqrt(float64(fanIn))
			}
			bsInit := NewUniformInit(-bound, bound)
			bs = vs.MustNewVar("bias", []int64{outDim}, bsInit)
		case c.BsInit != nil:
			bs = vs.MustNewVar("bias", []int64{outDim}, c.BsInit)
		}
	}

	return &Linear{
		Ws: vs.MustNewVar("weight", []int64{outDim, inDim}, c.WsInit).MustT(false),
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
//	inDim := 3
//	outDim := 2
//	batchSize := 4
//	weights: 2x3
//	[ 1 1 1
//		1 1 1 ]
//
//	input node: 3x4
//	[ 1 1 1
//	  1 1 1
//	  1 1 1
//		1 1 1 ]
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
