package aug

import (
	"github.com/sugarme/gotch/ts"
)

// RandomPosterize posterizes the image randomly with a given probability by reducing the
// number of bits for each color channel. If the image is torch Tensor, it should be of type torch.uint8,
// and it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
// Args:
// - bits (int): number of bits to keep for each channel (0-8)
// - p (float): probability of the image being color inverted. Default value is 0.5
// Ref. https://en.wikipedia.org/wiki/Posterization
type RandomPosterize struct {
	pvalue float64
	bits   uint8
}

type posterizeOptions struct {
	pvalue float64
	bits   uint8
}

type PosterizeOption func(*posterizeOptions)

func defaultPosterizeOptions() *posterizeOptions {
	return &posterizeOptions{
		pvalue: 0.5,
		bits:   4,
	}
}

func WithPosterizePvalue(p float64) PosterizeOption {
	return func(o *posterizeOptions) {
		o.pvalue = p
	}
}

func WithPosterizeBits(bits uint8) PosterizeOption {
	return func(o *posterizeOptions) {
		o.bits = bits
	}
}

func newRandomPosterize(opts ...PosterizeOption) *RandomPosterize {
	p := defaultPosterizeOptions()
	for _, o := range opts {
		o(p)
	}

	return &RandomPosterize{
		pvalue: p.pvalue,
		bits:   p.bits,
	}
}

// NOTE. Input image must be uint8 dtype otherwise panic!
func (rp *RandomPosterize) Forward(x *ts.Tensor) *ts.Tensor {
	r := randPvalue()
	var out *ts.Tensor
	switch {
	case r < rp.pvalue:
		out = posterize(x, rp.bits)
	default:
		out = x.MustShallowClone()
	}

	return out
}

func WithRandomPosterize(opts ...PosterizeOption) Option {
	rp := newRandomPosterize(opts...)

	return func(o *Options) {
		o.randomPosterize = rp
	}
}
