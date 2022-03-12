package aug

import (
	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// RandomAffine is transformation of the image keeping center invariant.
// If the image is torch Tensor, it is expected
// to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
// Args:
// - degrees (sequence or number): Range of degrees to select from.
// If degrees is a number instead of sequence like (min, max), the range of degrees
// will be (-degrees, +degrees). Set to 0 to deactivate rotations.
// - translate (tuple, optional): tuple of maximum absolute fraction for horizontal
// and vertical translations. For example translate=(a, b), then horizontal shift
// is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
// randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
// - scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
// randomly sampled from the range a <= scale <= b. Will keep original scale by default.
// - shear (sequence or number, optional): Range of degrees to select from.
// If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
// will be applied. Else if shear is a sequence of 2 values a shear parallel to the x axis in the
// range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
// a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
// Will not apply shear by default.
// - interpolation (InterpolationMode): Desired interpolation enum defined by
// :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
// If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
// For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
// - fill (sequence or number): Pixel fill value for the area outside the transformed
// image. Default is ``0``. If given a number, the value is used for all bands respectively.
// Please use the ``interpolation`` parameter instead.
// .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
type RandomAffine struct {
	degree            []int64 // degree range
	translate         []float64
	scale             []float64 // scale range
	shear             []float64
	interpolationMode string
	fillValue         []float64
}

func (ra *RandomAffine) getParams(imageSize []int64) (float64, []int64, float64, []float64) {
	angleTs := ts.MustEmpty([]int64{1}, gotch.Float, gotch.CPU)
	angleTs.MustUniform_(float64(ra.degree[0]), float64(ra.degree[1]))
	angle := angleTs.Float64Values()[0]
	angleTs.MustDrop()

	var translations []int64 = []int64{0, 0}
	if ra.translate != nil {
		maxDX := ra.translate[0] * float64(imageSize[0])
		maxDY := ra.translate[1] * float64(imageSize[1])
		dx := ts.MustEmpty([]int64{1}, gotch.Float, gotch.CPU)
		dx.MustUniform_(-maxDX, maxDX)
		tx := dx.Float64Values()[0]
		dx.MustDrop()

		dy := ts.MustEmpty([]int64{1}, gotch.Float, gotch.CPU)
		dy.MustUniform_(-maxDY, maxDY)
		ty := dy.Float64Values()[0]
		dy.MustDrop()

		translations = []int64{int64(tx), int64(ty)} // should we use math.Round here???
	}

	scale := 1.0
	if ra.scale != nil {
		scaleTs := ts.MustEmpty([]int64{1}, gotch.Float, gotch.CPU)
		scaleTs.MustUniform_(ra.scale[0], ra.scale[1])
		scale = scaleTs.Float64Values()[0]
		scaleTs.MustDrop()
	}

	var (
		shearX, shearY float64 = 0.0, 0.0
	)
	if ra.shear != nil {
		shearXTs := ts.MustEmpty([]int64{1}, gotch.Float, gotch.CPU)
		shearXTs.MustUniform_(ra.shear[0], ra.shear[1])
		shearX = shearXTs.Float64Values()[0]
		shearXTs.MustDrop()

		if len(ra.shear) == 4 {
			shearYTs := ts.MustEmpty([]int64{1}, gotch.Float, gotch.CPU)
			shearYTs.MustUniform_(ra.shear[2], ra.shear[3])
			shearY = shearYTs.Float64Values()[0]
			shearYTs.MustDrop()
		}
	}

	var shear []float64 = []float64{shearX, shearY}

	return angle, translations, scale, shear
}

func (ra *RandomAffine) Forward(x *ts.Tensor) *ts.Tensor {
	assertImageTensor(x)
	fx := Byte2FloatImage(x)

	w, h := getImageSize(fx)
	angle, translations, scale, shear := ra.getParams([]int64{w, h})

	out := affine(fx, angle, translations, scale, shear, ra.interpolationMode, ra.fillValue)

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func newRandomAffine(opts ...AffineOption) *RandomAffine {
	p := defaultAffineOptions()
	for _, o := range opts {
		o(p)
	}

	return &RandomAffine{
		degree:            p.degree,
		translate:         p.translate,
		scale:             p.scale,
		shear:             p.shear,
		interpolationMode: p.interpolationMode,
		fillValue:         p.fillValue,
	}
}

type affineOptions struct {
	degree            []int64
	translate         []float64
	scale             []float64
	shear             []float64
	interpolationMode string
	fillValue         []float64
}

type AffineOption func(*affineOptions)

func defaultAffineOptions() *affineOptions {
	return &affineOptions{
		degree:            []int64{0, 0},
		translate:         []float64{0, 0},
		scale:             []float64{1, 1},
		shear:             []float64{0, 0},
		interpolationMode: "nearest",
		fillValue:         []float64{0.0, 0.0, 0.0},
	}
}

func WithAffineDegree(degree []int64) AffineOption {
	return func(o *affineOptions) {
		o.degree = degree
	}
}

func WithAffineTranslate(translate []float64) AffineOption {
	return func(o *affineOptions) {
		o.translate = translate
	}
}

func WithAffineScale(scale []float64) AffineOption {
	return func(o *affineOptions) {
		o.scale = scale
	}
}

func WithAffineShear(shear []float64) AffineOption {
	return func(o *affineOptions) {
		o.shear = shear
	}
}

func WithAffineMode(mode string) AffineOption {
	return func(o *affineOptions) {
		o.interpolationMode = mode
	}
}

func WithAffineFillValue(fillValue []float64) AffineOption {
	return func(o *affineOptions) {
		o.fillValue = fillValue
	}
}

func WithRandomAffine(opts ...AffineOption) Option {
	ra := newRandomAffine(opts...)
	return func(o *Options) {
		o.randomAffine = ra
	}
}
