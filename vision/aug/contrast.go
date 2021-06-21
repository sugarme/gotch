package aug

import (
	ts "github.com/sugarme/gotch/tensor"
)

// RandomAutocontrast autocontrasts the pixels of the given image randomly with a given probability.
// If the image is torch Tensor, it is expected
// to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
// Args:
// - p (float): probability of the image being autocontrasted. Default value is 0.5
type RandomAutocontrast struct {
	pvalue float64
}

func newRandomAutocontrast(pOpt ...float64) *RandomAutocontrast {
	p := 0.5
	if len(pOpt) > 0 {
		p = pOpt[0]
	}

	return &RandomAutocontrast{p}
}

func (rac *RandomAutocontrast) Forward(x *ts.Tensor) *ts.Tensor {
	fx := Byte2FloatImage(x)

	r := randPvalue()
	var out *ts.Tensor
	switch {
	case r < rac.pvalue:
		out = autocontrast(fx)
	default:
		out = fx.MustShallowClone()
	}

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func WithRandomAutocontrast(p ...float64) Option {
	rac := newRandomAutocontrast(p...)
	return func(o *Options) {
		o.randomAutocontrast = rac
	}
}
