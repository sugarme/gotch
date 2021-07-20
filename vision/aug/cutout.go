package aug

import (
	"fmt"
	"log"
	"math"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// Randomly selects a rectangle region in an torch Tensor image and erases its pixels.
// This transform does not support PIL Image.
// 'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896
//
// Args:
// p: probability that the random erasing operation will be performed.
// scale: range of proportion of erased area against input image.
// ratio: range of aspect ratio of erased area.
// value: erasing value. Default is 0. If a single int, it is used to
// erase all pixels. If a tuple of length 3, it is used to erase
// R, G, B channels respectively.
// If a str of 'random', erasing each pixel with random values.
type RandomCutout struct {
	pvalue float64
	scale  []float64
	ratio  []float64
	rgbVal []int64 // RGB value
}

type cutoutOptions struct {
	pvalue float64
	scale  []float64
	ratio  []float64
	rgbVal []int64 // RGB value
}

type CutoutOption func(o *cutoutOptions)

func defaultCutoutOptions() *cutoutOptions {
	return &cutoutOptions{
		pvalue: 0.5,
		scale:  []float64{0.02, 0.33},
		ratio:  []float64{0.3, 3.3},
		rgbVal: []int64{0, 0, 0},
	}
}

func newRandomCutout(pvalue float64, scale, ratio []float64, rgbVal []int64) *RandomCutout {
	return &RandomCutout{
		pvalue: pvalue,
		scale:  scale,
		ratio:  ratio,
		rgbVal: rgbVal,
	}
}

func WithCutoutPvalue(p float64) CutoutOption {
	if p < 0 || p > 1 {
		log.Fatalf("Cutout p-value must be in range from 0 to 1. Got %v\n", p)
	}
	return func(o *cutoutOptions) {
		o.pvalue = p
	}
}

func WithCutoutScale(scale []float64) CutoutOption {
	if len(scale) != 2 {
		log.Fatalf("Cutout scale should be in a range of 2 elments. Got %v elements\n", len(scale))
	}
	return func(o *cutoutOptions) {
		o.scale = scale
	}
}

func WithCutoutRatio(ratio []float64) CutoutOption {
	if len(ratio) != 2 {
		log.Fatalf("Cutout ratio should be in a range of 2 elments. Got %v elements\n", len(ratio))
	}
	return func(o *cutoutOptions) {
		o.ratio = ratio
	}
}

func WithCutoutValue(rgb []int64) CutoutOption {
	var rgbVal []int64
	switch len(rgb) {
	case 1:
		rgbVal = []int64{rgb[0], rgb[0], rgb[0]}
	case 3:
		rgbVal = rgb
	default:
		err := fmt.Errorf("Cutout values can be single value or 3-element (RGB) value. Got %v values.", len(rgb))
		log.Fatal(err)
	}
	return func(o *cutoutOptions) {
		o.rgbVal = rgbVal
	}
}

func (rc *RandomCutout) cutoutParams(x *ts.Tensor) (int64, int64, int64, int64, *ts.Tensor) {
	dim := x.MustSize()

	imgH, imgW := dim[len(dim)-2], dim[len(dim)-1]
	area := float64(imgH * imgW)
	logRatio := ts.MustOfSlice(rc.ratio).MustLog(true).Float64Values()

	for i := 0; i < 10; i++ {
		scaleTs := ts.MustEmpty([]int64{1}, gotch.Float, gotch.CPU)
		scaleTs.MustUniform_(rc.scale[0], rc.scale[1])
		scaleVal := scaleTs.Float64Values()[0]
		scaleTs.MustDrop()
		eraseArea := area * scaleVal

		ratioTs := ts.MustEmpty([]int64{1}, gotch.Float, gotch.CPU)
		ratioTs.MustUniform_(logRatio[0], logRatio[1])
		asTs := ratioTs.MustExp(true)
		asVal := asTs.Float64Values()[0] // aspect ratio
		asTs.MustDrop()

		// h = int(round(math.sqrt(erase_area * aspect_ratio)))
		// w = int(round(math.sqrt(erase_area / aspect_ratio)))
		h := int64(math.Round(math.Sqrt(eraseArea * asVal)))
		w := int64(math.Round(math.Sqrt(eraseArea / asVal)))
		if !(h < imgH && w < imgW) {
			continue
		}

		// v = torch.tensor(value)[:, None, None]
		v := ts.MustOfSlice(rc.rgbVal).MustUnsqueeze(1, true).MustUnsqueeze(1, true)

		// i = torch.randint(0, img_h - h + 1, size=(1, )).item()
		iTs := ts.MustRandint1(0, imgH-h+1, []int64{1}, gotch.Int64, gotch.CPU)
		i := iTs.Int64Values()[0]
		iTs.MustDrop()
		// j = torch.randint(0, img_w - w + 1, size=(1, )).item()
		jTs := ts.MustRandint1(0, imgW-w+1, []int64{1}, gotch.Int64, gotch.CPU)
		j := jTs.Int64Values()[0]
		jTs.MustDrop()
		return i, j, h, w, v
	}

	// return original image
	img := x.MustShallowClone()
	return 0, 0, imgH, imgW, img
}

func (rc *RandomCutout) Forward(img *ts.Tensor) *ts.Tensor {
	fx := Byte2FloatImage(img)

	randTs := ts.MustRandn([]int64{1}, gotch.Float, gotch.CPU)
	randVal := randTs.Float64Values()[0]
	randTs.MustDrop()

	var out *ts.Tensor
	switch randVal < rc.pvalue {
	case true:
		x, y, h, w, v := rc.cutoutParams(fx)
		out = cutout(fx, x, y, h, w, rc.rgbVal)
		v.MustDrop()
	case false:
		out = fx.MustShallowClone()
	}

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func WithRandomCutout(opts ...CutoutOption) Option {
	params := defaultCutoutOptions()
	for _, o := range opts {
		o(params)
	}

	return func(o *Options) {
		rc := newRandomCutout(params.pvalue, params.scale, params.ratio, params.rgbVal)
		o.randomCutout = rc
	}
}
