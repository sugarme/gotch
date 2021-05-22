package aug

import (
	ts "github.com/sugarme/gotch/tensor"
)

// RandomSolarize solarizes the image randomly with a given probability by inverting all pixel
// values above a threshold. If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
// where ... means it can have an arbitrary number of leading dimensions.
// If img is PIL Image, it is expected to be in mode "L" or "RGB".
// Args:
// - threshold (float): all pixels equal or above this value are inverted.
// - p (float): probability of the image being color inverted. Default value is 0.5
// Ref. https://en.wikipedia.org/wiki/Solarization_(photography)
type RandomSolarize struct {
	threshold float64
	pvalue    float64
}

type solarizeOptions struct {
	threshold float64
	pvalue    float64
}

type solarizeOption func(*solarizeOptions)

func defaultSolarizeOptions() *solarizeOptions {
	return &solarizeOptions{
		threshold: 128,
		pvalue:    0.5,
	}
}

func WithSolarizePvalue(p float64) solarizeOption {
	return func(o *solarizeOptions) {
		o.pvalue = p
	}
}

func WithSolarizeThreshold(th float64) solarizeOption {
	return func(o *solarizeOptions) {
		o.threshold = th
	}
}

func newRandomSolarize(opts ...solarizeOption) *RandomSolarize {
	params := defaultSolarizeOptions()

	for _, o := range opts {
		o(params)
	}

	return &RandomSolarize{
		threshold: params.threshold,
		pvalue:    params.pvalue,
	}
}

func (rs *RandomSolarize) Forward(x *ts.Tensor) *ts.Tensor {
	r := randPvalue()

	var out *ts.Tensor
	switch {
	case r < rs.pvalue:
		out = solarize(x, rs.threshold)
	default:
		out = x.MustShallowClone()
	}

	return out
}

func WithRandomSolarize(opts ...solarizeOption) Option {
	rs := newRandomSolarize(opts...)

	return func(o *Options) {
		o.randomSolarize = rs
	}
}
