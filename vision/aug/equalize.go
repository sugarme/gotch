package aug

import (
	ts "github.com/sugarme/gotch/tensor"
)

// RandomEqualize equalizes the histogram of the given image randomly with a given probability.
// If the image is torch Tensor, it is expected
// to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
// Args:
// - p (float): probability of the image being equalized. Default value is 0.5
// Histogram equalization
// Ref. https://en.wikipedia.org/wiki/Histogram_equalization
type RandomEqualize struct {
	pvalue float64
}

func newRandomEqualize(pOpt ...float64) *RandomEqualize {
	p := 0.5
	if len(pOpt) > 0 {
		p = pOpt[0]
	}

	return &RandomEqualize{p}
}

// NOTE. input image MUST be uint8 dtype otherwise panic!
func (re *RandomEqualize) Forward(x *ts.Tensor) *ts.Tensor {
	r := randPvalue()
	var out *ts.Tensor
	switch {
	case r < re.pvalue:
		out = equalize(x)
	default:
		out = x.MustShallowClone()
	}

	return out
}

func WithRandomEqualize(p ...float64) Option {
	re := newRandomEqualize(p...)
	return func(o *Options) {
		o.randomEqualize = re
	}
}
