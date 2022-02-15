package nn

import (
	ts "github.com/sugarme/gotch/tensor"
)

// Dropout:
// ========

// Dropout represents a neural network dropout layer.
type Dropout struct {
	dropoutProb float64
}

// NewDropout creates a new Dropout layer
func NewDropout(p float64) *Dropout {
	return &Dropout{
		dropoutProb: p,
	}
}

// ForwardT implements ModuleT for Dropout layer.
func (d *Dropout) ForwardT(input *ts.Tensor, train bool) (retVal *ts.Tensor) {
	return ts.MustDropout(input, d.dropoutProb, train)
}

// NewParameter creates a kind of tensor that is considered as a module parameter.
// Ref. https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
func NewParameter(path *Path, name string, x *ts.Tensor, requireGradOpt ...bool) *ts.Tensor {
	requiredGrad := true
	if len(requireGradOpt) > 0 {
		requiredGrad = requireGradOpt[0]
	}

	param := path.Add(name, x, requiredGrad)

	return param
}

// Identity:
// =========

type Identity struct{}

func (m *Identity) Forward(x *ts.Tensor) *ts.Tensor {
	if x == nil {
		return nil
	}
	return x.MustShallowClone()
}

func NewIdentity() *Identity {
	return new(Identity)
}

// MaxPool2D:
// ==========

type MaxPool2D struct {
	Kernel   []int64
	Stride   []int64
	Padding  []int64
	Dilation []int64
	CeilMode bool
}

type MaxPool2DOpts struct {
	Stride   []int64
	Padding  []int64
	Dilation []int64
	CeilMode bool
}

type MaxPool2DOpt func(*MaxPool2DOpts)

func OptStrideMp2D(v []int64) MaxPool2DOpt {
	return func(o *MaxPool2DOpts) {
		o.Stride = v
	}
}

func OptPaddingMp2D(v []int64) MaxPool2DOpt {
	return func(o *MaxPool2DOpts) {
		o.Padding = v
	}
}

func OptDilationMp2D(v []int64) MaxPool2DOpt {
	return func(o *MaxPool2DOpts) {
		o.Dilation = v
	}
}

func OptCeilModeMp2D(v bool) MaxPool2DOpt {
	return func(o *MaxPool2DOpts) {
		o.CeilMode = v
	}
}

func DefaultMaxPool2DOpts() *MaxPool2DOpts {
	return &MaxPool2DOpts{
		Stride:   nil,
		Padding:  []int64{0, 0},
		Dilation: []int64{1, 1},
	}
}

func NewMaxPool2D(kernelSize []int64, opts ...MaxPool2DOpt) *MaxPool2D {
	o := DefaultMaxPool2DOpts()
	for _, opt := range opts {
		opt(o)
	}

	return &MaxPool2D{
		Kernel:   kernelSize,
		Stride:   o.Stride,
		Padding:  o.Padding,
		Dilation: o.Dilation,
		CeilMode: o.CeilMode,
	}
}

func (m *MaxPool2D) Forward(x *ts.Tensor) *ts.Tensor {
	return x.MustMaxPool2d(m.Kernel, m.Stride, m.Padding, m.Dilation, m.CeilMode, false)
}
