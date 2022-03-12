package nn

import (
	"github.com/sugarme/gotch/ts"
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

// Parameter:
// ==========

// NewParameter creates a kind of tensor that is considered as a module parameter.
// Ref. https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
func NewParameter(path *Path, name string, x *ts.Tensor, requireGradOpt ...bool) *ts.Tensor {
	requiredGrad := true
	if len(requireGradOpt) > 0 {
		requiredGrad = requireGradOpt[0]
	}

	param := path.MustAdd(name, x, requiredGrad)

	return param
}

// Buffer:
// =======

// NewBuffer creates new buffer.
//
// Buffer is different from Parameter as its requiredGrad always false.
// - `o.Persistent` param. Default=true. If `true` buffer variable will be saved when `nn.VarStore.Save()` is called.
//
// Ref.
// - https://github.com/pytorch/pytorch/blob/f71eede85a69caed637008e331f5ac5f5b7717ae/torch/nn/modules/module.py#L275
// - https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/2
func NewBuffer(path *Path, name string, x *ts.Tensor, persistentOpt ...bool) *ts.Tensor {
	persistent := true
	if len(persistentOpt) > 0 {
		persistent = persistentOpt[0]
	}
	opts := []AddOpt{
		WithPersistent(persistent),
		WithVarType("buffer"),
	}
	return path.MustAdd(name, x, false, opts...) // requiredGrad always false. Different from parameter.
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
