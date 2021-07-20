package aug

import (
	// "fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// RandomPerspective performs a random perspective transformation of the given image with a given probability.
// If the image is torch Tensor, it is expected
// to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
// Args:
// distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
// Default is 0.5.
// p (float): probability of the image being transformed. Default is 0.5.
// interpolation (InterpolationMode): Desired interpolation enum defined by
// :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
// If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
// For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
// fill (sequence or number): Pixel fill value for the area outside the transformed
// image. Default is ``0``. If given a number, the value is used for all bands respectively.
type RandomPerspective struct {
	distortionScale   float64 // range [0, 1]
	pvalue            float64 //  range [0, 1]
	interpolationMode string
	fillValue         []float64
}

type perspectiveOptions struct {
	distortionScale   float64 // range [0, 1]
	pvalue            float64 //  range [0, 1]
	interpolationMode string
	fillValue         []float64
}

func defaultPerspectiveOptions() *perspectiveOptions {
	return &perspectiveOptions{
		distortionScale:   0.5,
		pvalue:            0.5,
		interpolationMode: "bilinear",
		fillValue:         []float64{0.0, 0.0, 0.0},
	}
}

type PerspectiveOption func(*perspectiveOptions)

func WithPerspectivePvalue(p float64) PerspectiveOption {
	return func(o *perspectiveOptions) {
		o.pvalue = p
	}
}

func WithPerspectiveScale(s float64) PerspectiveOption {
	return func(o *perspectiveOptions) {
		o.distortionScale = s
	}
}

func WithPerspectiveMode(m string) PerspectiveOption {
	return func(o *perspectiveOptions) {
		o.interpolationMode = m
	}
}

func WithPerspectiveValue(v []float64) PerspectiveOption {
	return func(o *perspectiveOptions) {
		o.fillValue = v
	}
}

func newRandomPerspective(opts ...PerspectiveOption) *RandomPerspective {
	params := defaultPerspectiveOptions()
	for _, opt := range opts {
		opt(params)
	}

	return &RandomPerspective{
		distortionScale:   params.distortionScale,
		pvalue:            params.pvalue,
		interpolationMode: params.interpolationMode,
		fillValue:         params.fillValue,
	}
}

// Get parameters for ``perspective`` for a random perspective transform.
//
// Args:
// - width (int): width of the image.
// - height (int): height of the image.
// Returns:
// - List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
// - List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
func (rp *RandomPerspective) getParams(w, h int64) ([][]int64, [][]int64) {
	halfH := h / 2
	halfW := w / 2

	var (
		topLeft     []int64
		topRight    []int64
		bottomRight []int64
		bottomLeft  []int64
	)

	// topleft = [
	// int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()),
	// int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())
	// ]
	tlVal1 := int64(rp.distortionScale*float64(halfW)) + 1
	tlTs1 := ts.MustRandint1(0, tlVal1, []int64{1}, gotch.Int64, gotch.CPU)
	tl1 := tlTs1.Int64Values()[0]
	tlTs1.MustDrop()
	tlVal2 := int64(rp.distortionScale*float64(halfH)) + 1
	tlTs2 := ts.MustRandint1(0, tlVal2, []int64{1}, gotch.Int64, gotch.CPU)
	tl2 := tlTs2.Int64Values()[0]
	tlTs2.MustDrop()
	topLeft = []int64{tl1, tl2}

	// topright = [
	// int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()),
	// int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())
	// ]
	trVal1 := w - int64(rp.distortionScale*float64(halfW)) - 1
	trTs1 := ts.MustRandint1(trVal1, w, []int64{1}, gotch.Int64, gotch.CPU)
	tr1 := trTs1.Int64Values()[0]
	trTs1.MustDrop()
	trVal2 := int64(rp.distortionScale*float64(halfH)) + 1
	trTs2 := ts.MustRandint1(0, trVal2, []int64{1}, gotch.Int64, gotch.CPU)
	tr2 := trTs2.Int64Values()[0]
	trTs2.MustDrop()
	topRight = []int64{tr1, tr2}

	// botright = [
	// int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()),
	// int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())
	// ]
	brVal1 := w - int64(rp.distortionScale*float64(halfW)) - 1
	brTs1 := ts.MustRandint1(brVal1, w, []int64{1}, gotch.Int64, gotch.CPU)
	br1 := brTs1.Int64Values()[0]
	brTs1.MustDrop()
	brVal2 := h - int64(rp.distortionScale*float64(halfH)) - 1
	brTs2 := ts.MustRandint1(brVal2, h, []int64{1}, gotch.Int64, gotch.CPU)
	br2 := brTs2.Int64Values()[0]
	brTs2.MustDrop()
	bottomRight = []int64{br1, br2}

	// botleft = [
	// int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()),
	// int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())
	// ]
	blVal1 := int64(rp.distortionScale*float64(halfW)) + 1
	blTs1 := ts.MustRandint1(0, blVal1, []int64{1}, gotch.Int64, gotch.CPU)
	bl1 := blTs1.Int64Values()[0]
	blTs1.MustDrop()
	blVal2 := h - int64(rp.distortionScale*float64(halfH)) - 1
	blTs2 := ts.MustRandint1(blVal2, h, []int64{1}, gotch.Int64, gotch.CPU)
	bl2 := blTs2.Int64Values()[0]
	blTs2.MustDrop()
	bottomLeft = []int64{bl1, bl2}

	startPoints := [][]int64{
		{0, 0},
		{w - 1, 0},
		{w - 1, h - 1},
		{0, h - 1},
	}

	endPoints := [][]int64{
		topLeft,
		topRight,
		bottomRight,
		bottomLeft,
	}

	return startPoints, endPoints
}

func (rp *RandomPerspective) Forward(x *ts.Tensor) *ts.Tensor {
	fx := Byte2FloatImage(x)

	height, width := getImageSize(fx)
	startPoints, endPoints := rp.getParams(height, width)
	out := perspective(fx, startPoints, endPoints, rp.interpolationMode, rp.fillValue)

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func WithRandomPerspective(opts ...PerspectiveOption) Option {
	rp := newRandomPerspective(opts...)
	return func(o *Options) {
		o.randomPerspective = rp
	}
}
