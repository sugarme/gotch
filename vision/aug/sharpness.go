package aug

import (
	"github.com/sugarme/gotch/ts"
)

// Adjust the sharpness of the image randomly with a given probability. If the image is torch Tensor,
// it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
// Args:
// sharpness_factor (float):  How much to adjust the sharpness. Can be
// any non negative number. 0 gives a blurred image, 1 gives the
// original image while 2 increases the sharpness by a factor of 2.
// p (float): probability of the image being color inverted. Default value is 0.5
type RandomAdjustSharpness struct {
	sharpnessFactor float64
	pvalue          float64
}

type sharpnessOptions struct {
	sharpnessFactor float64
	pvalue          float64
}

type SharpnessOption func(*sharpnessOptions)

func defaultSharpnessOptions() *sharpnessOptions {
	return &sharpnessOptions{
		sharpnessFactor: 1.0,
		pvalue:          0.5,
	}
}

func WithSharpnessPvalue(p float64) SharpnessOption {
	return func(o *sharpnessOptions) {
		o.pvalue = p
	}
}

func WithSharpnessFactor(f float64) SharpnessOption {
	return func(o *sharpnessOptions) {
		o.sharpnessFactor = f
	}
}

func newRandomAdjustSharpness(opts ...SharpnessOption) *RandomAdjustSharpness {
	p := defaultSharpnessOptions()
	for _, o := range opts {
		o(p)
	}
	return &RandomAdjustSharpness{
		sharpnessFactor: p.sharpnessFactor,
		pvalue:          p.pvalue,
	}
}

// NOTE. input img dtype shoule be `uint8` (Byte)
func (ras *RandomAdjustSharpness) Forward(x *ts.Tensor) *ts.Tensor {
	r := randPvalue()
	var out *ts.Tensor
	switch {
	case r < ras.pvalue:
		out = adjustSharpness(x, ras.sharpnessFactor)
	default:
		out = x.MustShallowClone()
	}

	return out
}

func WithRandomAdjustSharpness(opts ...SharpnessOption) Option {
	ras := newRandomAdjustSharpness(opts...)
	return func(o *Options) {
		o.randomAdjustSharpness = ras
	}
}
