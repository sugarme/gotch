package aug

import (
	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// RandomHorizontalFlip horizontally flips the given image randomly with a given probability.
//
// If the image is torch Tensor, it is expected to have [..., H, W] shape,
// where ... means an arbitrary number of leading dimensions
// Args:
// p (float): probability of the image being flipped. Default value is 0.5
type RandomHorizontalFlip struct {
	pvalue float64
}

func newRandomHorizontalFlip(pvalue float64) *RandomHorizontalFlip {
	return &RandomHorizontalFlip{
		pvalue: pvalue,
	}
}

func (hf *RandomHorizontalFlip) Forward(x *ts.Tensor) *ts.Tensor {
	fx := Byte2FloatImage(x)

	randTs := ts.MustRandn([]int64{1}, gotch.Float, gotch.CPU)
	randVal := randTs.Float64Values()[0]
	randTs.MustDrop()
	var out *ts.Tensor
	switch {
	case randVal < hf.pvalue:
		out = hflip(fx)
	default:
		out = fx.MustShallowClone()
	}

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func WithRandomHFlip(pvalue float64) Option {
	return func(o *Options) {
		hf := newRandomHorizontalFlip(pvalue)
		o.randomHFlip = hf
	}
}

// RandomVerticalFlip vertically flips the given image randomly with a given probability.
//
// If the image is torch Tensor, it is expected to have [..., H, W] shape,
// where ... means an arbitrary number of leading dimensions
// Args:
// p (float): probability of the image being flipped. Default value is 0.5
type RandomVerticalFlip struct {
	pvalue float64
}

func newRandomVerticalFlip(pvalue float64) *RandomVerticalFlip {
	return &RandomVerticalFlip{
		pvalue: pvalue,
	}
}

func (vf *RandomVerticalFlip) Forward(x *ts.Tensor) *ts.Tensor {
	fx := Byte2FloatImage(x)

	randTs := ts.MustRandn([]int64{1}, gotch.Float, gotch.CPU)
	randVal := randTs.Float64Values()[0]
	randTs.MustDrop()

	var out *ts.Tensor
	switch {
	case randVal < vf.pvalue:
		out = vflip(fx)
	default:
		out = fx.MustShallowClone()
	}

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func WithRandomVFlip(pvalue float64) Option {
	return func(o *Options) {
		vf := newRandomVerticalFlip(pvalue)
		o.randomVFlip = vf
	}
}
