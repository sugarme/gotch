package aug

import (
	"log"

	"github.com/sugarme/gotch/ts"
)

// GrayScale converts image to grayscale.
// If the image is torch Tensor, it is expected
// to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions
// Args:
// - num_output_channels (int): (1 or 3) number of channels desired for output image
type Grayscale struct {
	outChan int64
}

func (gs *Grayscale) Forward(x *ts.Tensor) *ts.Tensor {
	fx := Byte2FloatImage(x)

	out := rgb2Gray(fx, gs.outChan)

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func newGrayscale(outChanOpt ...int64) *Grayscale {
	var outChan int64 = 3
	if len(outChanOpt) > 0 {
		c := outChanOpt[0]
		switch c {
		case 1:
			outChan = 1
		case 3:
			outChan = 3
		default:
			log.Fatalf("Out channels should be either 1 or 3. Got %v\n", c)
		}
	}
	return &Grayscale{outChan}
}

// RandomGrayscale randomly converts image to grayscale with a probability of p (default 0.1).
// If the image is torch Tensor, it is expected
// to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions
// Args:
// - p (float): probability that image should be converted to grayscale.
type RandomGrayscale struct {
	pvalue float64
}

func newRandomGrayscale(pvalueOpt ...float64) *RandomGrayscale {
	pvalue := 0.1
	if len(pvalueOpt) > 0 {
		pvalue = pvalueOpt[0]
	}
	return &RandomGrayscale{pvalue}
}

func (rgs *RandomGrayscale) Forward(x *ts.Tensor) *ts.Tensor {
	c := getImageChanNum(x)
	r := randPvalue()
	var out *ts.Tensor
	switch {
	case r < rgs.pvalue:
		out = rgb2Gray(x, c)
	default:
		out = x.MustShallowClone()
	}

	return out
}

func WithRandomGrayscale(pvalueOpt ...float64) Option {
	var p float64 = 0.1
	if len(pvalueOpt) > 0 {
		p = pvalueOpt[0]
	}

	rgs := newRandomGrayscale(p)
	return func(o *Options) {
		o.randomGrayscale = rgs
	}
}

func NewGrayscale(outChanOpt ...int64) *Grayscale {
	return newGrayscale(outChanOpt...)
}
