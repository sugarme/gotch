package aug

import (
	ts "github.com/sugarme/gotch/tensor"
)

type RandomInvert struct {
	pvalue float64
}

func newRandomInvert(pOpt ...float64) *RandomInvert {
	p := 0.5
	if len(pOpt) > 0 {
		p = pOpt[0]
	}
	return &RandomInvert{p}
}

func (ri *RandomInvert) Forward(x *ts.Tensor) *ts.Tensor {
	fx := Byte2FloatImage(x)

	r := randPvalue()
	var out *ts.Tensor
	switch {
	case r < ri.pvalue:
		out = invert(fx)
	default:
		out = fx.MustShallowClone()
	}

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func WithRandomInvert(pvalueOpt ...float64) Option {
	ri := newRandomInvert(pvalueOpt...)

	return func(o *Options) {
		o.randomInvert = ri
	}
}
