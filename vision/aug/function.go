package aug

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func gaussianKernel1D(ks int64, sigma float64, dtype gotch.DType, device gotch.Device) *ts.Tensor {
	ksHalf := (ks - 1) / 2
	x := ts.MustLinspace(ts.IntScalar(-ksHalf), ts.IntScalar(ksHalf), []int64{ks}, dtype, device)

	// pdf = torch.exp(-0.5 * (x / sigma).pow(2))
	pdf := x.MustDiv1(ts.FloatScalar(sigma), true).MustPow(ts.IntScalar(2), true).MustMul1(ts.FloatScalar(0.5), true).MustExp(true)
	// kernel1d = pdf / pdf.sum()
	pdfSum := pdf.MustSum(dtype, false)
	kernel1d := pdf.MustDiv(pdfSum, true)
	pdfSum.MustDrop()

	return kernel1d
}

func gaussianKernel2D(ks []int64, sigma []float64, dtype gotch.DType, device gotch.Device) *ts.Tensor {
	kernel1dX := gaussianKernel1D(ks[0], sigma[0], dtype, device)
	kernel1dY := gaussianKernel1D(ks[1], sigma[1], dtype, device)

	// dimX := kernel1dX.MustSize()
	kernel1dX.MustUnsqueeze_(0) // kernel1d_x[None, :]
	dimY := kernel1dY.MustSize()
	kernel1dY.MustUnsqueeze_(int64(len(dimY))) // kernel1d_y[:, None]

	kernel2d := kernel1dY.MustMm(kernel1dX, true)
	kernel1dX.MustDrop()
	return kernel2d
}

func containsDType(dtype gotch.DType, dtypes []gotch.DType) bool {
	for _, dt := range dtypes {
		if dtype == dt {
			return true
		}
	}

	return false
}

func castSqueezeIn(x *ts.Tensor, reqDtypes []gotch.DType) (*ts.Tensor, bool, bool, gotch.DType) {
	needSqueeze := false
	xdim := x.MustSize()
	var img *ts.Tensor
	if len(xdim) < 4 {
		img = x.MustUnsqueeze(0, false)
		needSqueeze = true
	} else {
		img = x.MustShallowClone()
	}
	outDtype := x.DType()
	needCast := false
	if !containsDType(outDtype, reqDtypes) {
		needCast = true
		reqDType := reqDtypes[0]
		img1 := img.MustTotype(reqDType, true)
		return img1, needCast, needSqueeze, outDtype
	}
	return img, needCast, needSqueeze, outDtype
}

func castSqueezeOut(x *ts.Tensor, needCast, needSqueeze bool, outDType gotch.DType) *ts.Tensor {
	var (
		squeezeTs, castTs *ts.Tensor
	)
	switch needSqueeze {
	case true:
		squeezeTs = x.MustSqueeze1(0, false)
	case false:
		squeezeTs = x.MustShallowClone()
	}

	switch needCast {
	case true:
		// it is better to round before cast
		if containsDType(outDType, []gotch.DType{gotch.Uint8, gotch.Int8, gotch.Int16, gotch.Int, gotch.Int64}) {
			roundTs := squeezeTs.MustRound(true)
			castTs = roundTs.MustTotype(outDType, true)
		} else {
			castTs = squeezeTs.MustTotype(outDType, true)
		}
	case false:
		castTs = squeezeTs.MustShallowClone()
		squeezeTs.MustDrop()
	}

	return castTs
}

func gaussianBlur(x *ts.Tensor, ks []int64, sigma []float64) *ts.Tensor {
	// dtype := gotch.Float
	dtype := x.DType()
	if x.DType() == gotch.Float || x.DType() == gotch.Double {
		dtype = x.DType()
	}
	device := x.MustDevice()

	assertImageTensor(x)

	kernel := gaussianKernel2D(ks, sigma, dtype, device)
	xdim := x.MustSize()
	kdim := kernel.MustSize()

	// kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])
	kexpand := kernel.MustExpand([]int64{xdim[len(xdim)-3], 1, kdim[0], kdim[1]}, true, true)
	kdtype := kexpand.DType()
	img, needCast, needSqueeze, outDType := castSqueezeIn(x, []gotch.DType{kdtype})

	// padding = (left, right, top, bottom)
	// padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
	left := ks[0] / 2
	right := ks[0] / 2
	top := ks[1] / 2
	bottom := ks[1] / 2
	padding := []int64{left, right, top, bottom}

	// F.pad()
	// https://github.com/pytorch/pytorch/blob/71f4c5c1f436258adc303b710efb3f41b2d50c4e/torch/nn/functional.py#L4070
	// img = torch_pad(img, padding, mode="reflect")
	imgPad := img.MustReflectionPad2d(padding, true) // deleted img

	imgPadDim := imgPad.MustSize()
	// img = conv2d(img, kernel, groups=img.shape[-3])
	// ref. https://github.com/pytorch/pytorch/blob/6060684609ebf66120db5af004b4cdafc5cccbdb/torch/nn/functional.py#L71
	imgConv2d := ts.MustConv2d(imgPad, kexpand, ts.NewTensor(), []int64{1}, []int64{0}, []int64{1}, imgPadDim[len(imgPadDim)-3])
	imgPad.MustDrop()

	// img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
	out := castSqueezeOut(imgConv2d, needCast, needSqueeze, outDType)
	imgConv2d.MustDrop()

	return out
}

func isTorchImage(x *ts.Tensor) bool {
	return x.Dim() >= 2
}

func assertImageTensor(x *ts.Tensor) {
	if !isTorchImage(x) {
		err := fmt.Errorf("Input tensor is not a torch image.")
		log.Fatal(err)
	}
}

func imageChanNum(x *ts.Tensor) int64 {
	ndim := x.Dim()

	switch {
	case ndim == 2:
		return 1
	case ndim > 2:
		return x.MustSize()[0]
	default:
		err := fmt.Errorf("imageChanNum - Input should be 2 or more. Got %v", ndim)
		log.Fatal(err)
		return 0
	}
}

func contains(item int64, list []int64) bool {
	for _, i := range list {
		if item == i {
			return true
		}
	}

	return false
}

func assertChannels(x *ts.Tensor, permitted []int64) {
	c := imageChanNum(x)
	if !contains(c, permitted) {
		err := fmt.Errorf("Input image tensor permitted channels are %+v, but found %v", permitted, c)
		log.Fatal(err)
	}
}

func blend(img1, img2 *ts.Tensor, ratio float64) *ts.Tensor {
	dtype := img1.DType()
	bound := 255.0

	// (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)
	i1 := img1.MustMul1(ts.FloatScalar(ratio), false)
	i2 := img2.MustMul1(ts.FloatScalar(1.0-ratio), false)
	sumTs := i1.MustAdd(i2, true)
	i2.MustDrop()
	out := sumTs.MustClamp(ts.FloatScalar(0), ts.FloatScalar(bound), true).MustTotype(dtype, true)
	return out
}

// brightness should be in range 0.25 - 1.25 for visible view
func adjustBrightness(x *ts.Tensor, brightness float64) *ts.Tensor {
	if brightness < 0 {
		err := fmt.Errorf("adjustBrightness - brightness factor (%v) is not non-negative.", brightness)
		log.Fatal(err)
	}

	assertImageTensor(x)
	assertChannels(x, []int64{1, 3})

	zeros := x.MustZerosLike(false)
	out := blend(x, zeros, brightness)
	zeros.MustDrop()

	return out
}

// randVal generates a value from uniform values from 0 to x
func randVal(from, to float64) float64 {
	v := ts.MustEmpty([]int64{1}, gotch.Float, gotch.CPU)
	v.MustUniform_(from, to)
	randVal := v.Float64Values()[0]
	v.MustDrop()
	return randVal
}

func getMinMax(x float64) (float64, float64) {
	from := 0.0
	if 1-x > 0 {
		from = 1 - x
	}
	to := 1 + x

	return from, to
}

func rgb2Gray(x *ts.Tensor, outChanOpt ...int64) *ts.Tensor {
	var outChannels int64 = 1
	if len(outChanOpt) > 0 {
		outChannels = outChanOpt[0]
	}

	ndim := x.Dim()
	if ndim < 3 {
		err := fmt.Errorf("Input image tensor should have at least 3 dimensions, but found %v", ndim)
		log.Fatal(err)
	}

	assertChannels(x, []int64{3})
	if !contains(outChannels, []int64{1, 3}) {
		err := fmt.Errorf("Number of output channels should be either 1 or 3")
		log.Fatal(err)
	}

	rgbTs := x.MustUnbind(-3, false)
	r := &rgbTs[0]
	g := &rgbTs[1]
	b := &rgbTs[2]

	// This implementation closely follows the TF one:
	// https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
	// l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
	rmul := r.MustMul1(ts.FloatScalar(0.2989), true)
	gmul := g.MustMul1(ts.FloatScalar(0.587), true)
	bmul := b.MustMul1(ts.FloatScalar(0.114), true)
	addTs := rmul.MustAdd(gmul, true).MustAdd(bmul, true)
	gmul.MustDrop()
	bmul.MustDrop()
	lImg := addTs.MustTotype(x.DType(), true).MustUnsqueeze(-3, true)

	if outChannels == 3 {
		return lImg.MustExpand(x.MustSize(), true, true)
	}

	return lImg
}

func adjustContrast(x *ts.Tensor, contrast float64) *ts.Tensor {
	if contrast < 0 {
		err := fmt.Errorf("adjustContrast - contrast factor (%v) is not non-negative.", contrast)
		log.Fatal(err)
	}

	assertImageTensor(x)
	assertChannels(x, []int64{3})

	grayTs := rgb2Gray(x).MustTotype(x.DType(), true)

	mean := grayTs.MustMean1([]int64{-3, -2, -1}, true, gotch.Float, true).MustTotype(x.DType(), true)
	out := blend(x, mean, contrast)
	mean.MustDrop()

	return out
}

func adjustSaturation(x *ts.Tensor, sat float64) *ts.Tensor {
	if sat < 0 {
		err := fmt.Errorf("adjustSaturation - saturation factor (%v) is not non-negative.", sat)
		log.Fatal(err)
	}
	assertImageTensor(x)
	assertChannels(x, []int64{3})
	grayTs := rgb2Gray(x).MustTotype(x.DType(), true)
	out := blend(x, grayTs, sat)
	grayTs.MustDrop()

	return out
}

func rgb2HSV(x *ts.Tensor) *ts.Tensor {
	rgbTs := x.MustUnbind(-3, false)
	r := &rgbTs[0]
	g := &rgbTs[1]
	b := &rgbTs[2]

	// # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
	// # src/libImaging/Convert.c#L330
	// maxc = torch.max(img, dim=-3).values
	// minc = torch.min(img, dim=-3).values
	maxC := x.MustAmax([]int64{-3}, false, false)
	minC := x.MustAmin([]int64{-3}, false, false)

	// # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
	// # from happening in the results, because
	// #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
	// #   + H channel has division by `(maxc - minc)`.
	// #
	// # Instead of overwriting NaN afterwards, we just prevent it from occuring so
	// # we don't need to deal with it in case we save the NaN in a buffer in
	// # backprop, if it is ever supported, but it doesn't hurt to do so.
	// eqc = maxc == minc
	eqC := maxC.MustEq1(minC, false)

	// cr = maxc - minc
	cr := maxC.MustSub(minC, false)

	// # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
	ones := maxC.MustOnesLike(false)

	// s = cr / torch.where(eqc, ones, maxc)
	condMaxC := ones.MustWhere1(eqC, maxC, false)
	s := cr.MustDiv(condMaxC, false)

	// # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
	// # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
	// # would not matter what values `rc`, `gc`, and `bc` have here, and thus
	// # replacing denominator with 1 when `eqc` is fine.
	// cr_divisor = torch.where(eqc, ones, cr)
	// rc = (maxc - r) / cr_divisor
	// gc = (maxc - g) / cr_divisor
	// bc = (maxc - b) / cr_divisor
	crDivisor := ones.MustWhere1(eqC, cr, true) // delete ones
	rc := maxC.MustSub(r, false).MustDiv(crDivisor, true)
	gc := maxC.MustSub(g, false).MustDiv(crDivisor, true)
	bc := maxC.MustSub(b, false).MustDiv(crDivisor, true)

	// hr = (maxc == r) * (bc - gc)
	rSub := bc.MustSub(gc, false)
	hr := maxC.MustEq1(r, false).MustMul(rSub, true)
	rSub.MustDrop()

	// hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
	maxcCond1 := maxC.MustNotEqual1(r, false)
	hgMul := rc.MustSub(bc, false).MustAdd1(ts.FloatScalar(2.0), true)
	hg := maxC.MustEq1(g, false).MustLogicalAnd(maxcCond1, true).MustMul(hgMul, true)
	maxcCond1.MustDrop()
	hgMul.MustDrop()

	// hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
	maxcCond2 := maxC.MustNotEqual1(r, false)
	hbMul := gc.MustSub(rc, false).MustAdd1(ts.FloatScalar(4.0), true)
	hb := maxC.MustNotEqual1(g, false).MustLogicalAnd(maxcCond2, true).MustMul(hbMul, true)
	maxcCond2.MustDrop()
	hbMul.MustDrop()

	// h = (hr + hg + hb)
	h1 := hr.MustAdd(hg, false).MustAdd(hb, true)

	// h = torch.fmod((h / 6.0 + 1.0), 1.0)
	h2 := h1.MustDiv1(ts.FloatScalar(6.0), true).MustAdd1(ts.FloatScalar(1.0), true) // delete h1
	h3 := h2.MustFmod(ts.FloatScalar(1.0), true)                                     // delete h2

	// torch.stack((h, s, maxc), dim=-3)
	out := ts.MustStack([]ts.Tensor{*h3, *s, *maxC}, -3)

	// Delete intermediate tensors
	r.MustDrop()
	g.MustDrop()
	b.MustDrop()
	h3.MustDrop()
	maxC.MustDrop()
	minC.MustDrop()
	eqC.MustDrop()
	s.MustDrop()
	condMaxC.MustDrop()
	cr.MustDrop()
	crDivisor.MustDrop()
	rc.MustDrop()
	gc.MustDrop()
	bc.MustDrop()
	hr.MustDrop()
	hg.MustDrop()
	hb.MustDrop()

	return out
}

func hsv2RGB(x *ts.Tensor) *ts.Tensor {
	hsvTs := x.MustUnbind(-3, false)
	h := &hsvTs[0]
	s := &hsvTs[1]
	v := &hsvTs[2]
	// i = torch.floor(h * 6.0)
	i := h.MustMul1(ts.FloatScalar(6.0), false).MustFloor(true)
	// f = (h * 6.0) - i
	f := h.MustMul1(ts.FloatScalar(6.0), false).MustSub(i, true)

	// p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
	x1 := s.MustMul1(ts.FloatScalar(-1), false).MustAdd1(ts.FloatScalar(1.0), true)
	p := v.MustMul(x1, false).MustClamp(ts.FloatScalar(0.0), ts.FloatScalar(1.0), true)
	x1.MustDrop()

	// q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
	x2 := s.MustMul(f, false).MustMul1(ts.FloatScalar(-1), true).MustAdd1(ts.FloatScalar(1.0), true)
	q := v.MustMul(x2, false).MustClamp(ts.FloatScalar(0.0), ts.FloatScalar(1.0), true)
	x2.MustDrop()

	//t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
	// step1. s * (1.0 - f)
	sub1 := f.MustMul1(ts.FloatScalar(-1), false).MustAdd1(ts.FloatScalar(1.0), true).MustMul(s, true)
	// step 2: v *(1.0 - step1)
	x3 := sub1.MustMul1(ts.FloatScalar(-1), true).MustAdd1(ts.FloatScalar(1.0), true).MustMul(v, true) // deleted sub1
	t := x3.MustClamp(ts.FloatScalar(0.0), ts.FloatScalar(1.0), true)                                  // deleted x3

	// i = i.to(dtype=torch.int32)
	i = i.MustTotype(gotch.Int, true)
	//i = i % 6
	iremainder := i.MustRemainder(ts.IntScalar(6), true) // delete i
	// torch.arange(6, device=i.device).view(-1, 1, 1)
	x4 := ts.MustArange(ts.FloatScalar(6), gotch.Float, iremainder.MustDevice()).MustView([]int64{-1, 1, 1}, true)
	// mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)
	mask := iremainder.MustUnsqueeze(-3, true).MustEq1(x4, true).MustTotype(x.DType(), true) // delete iremainder
	x4.MustDrop()

	// a1 = torch.stack((v, q, p, p, t, v), dim=-3)
	// a2 = torch.stack((t, v, v, q, p, p), dim=-3)
	// a3 = torch.stack((p, p, t, v, v, q), dim=-3)
	// a4 = torch.stack((a1, a2, a3), dim=-4)
	a1 := ts.MustStack([]ts.Tensor{*v, *q, *p, *p, *t, *v}, -3)
	a2 := ts.MustStack([]ts.Tensor{*t, *v, *v, *q, *p, *p}, -3)
	a3 := ts.MustStack([]ts.Tensor{*p, *p, *t, *v, *v, *q}, -3)
	a4 := ts.MustStack([]ts.Tensor{*a1, *a2, *a3}, -4)

	out := ts.MustEinsum("...ijk, ...xijk -> ...xjk", []ts.Tensor{*mask, *a4})

	// Delete intermediate tensors
	h.MustDrop()
	s.MustDrop()
	v.MustDrop()
	f.MustDrop()
	p.MustDrop()
	q.MustDrop()
	t.MustDrop()

	a1.MustDrop()
	a2.MustDrop()
	a3.MustDrop()
	a4.MustDrop()
	mask.MustDrop()

	return out
}

// ref. https://en.wikipedia.org/wiki/HSL_and_HSV
func adjustHue(x *ts.Tensor, hue float64) *ts.Tensor {
	if hue < -0.5 || hue > 0.5 {
		err := fmt.Errorf("hue factor (%v) is not in [-0.5, 0.5]", hue)
		log.Fatal(err)
	}
	assertImageTensor(x)
	assertChannels(x, []int64{1, 3})

	if c := imageChanNum(x); c == 1 {
		out := x.MustShallowClone()
		return out
	}

	imgFl := x.MustTotype(gotch.Float, false).MustDiv1(ts.FloatScalar(255.0), true)
	hsvImg := rgb2HSV(imgFl)

	hsvTs := hsvImg.MustUnbind(-3, true)
	h := &hsvTs[0]
	s := &hsvTs[1]
	v := &hsvTs[2]
	// h = (h + hue_factor) % 1.0
	hAdj := h.MustAdd1(ts.FloatScalar(hue), false).MustRemainder(ts.FloatScalar(1.0), true)

	hsvAdj := ts.MustStack([]ts.Tensor{*hAdj, *s, *v}, -3)

	imgHueAdj := hsv2RGB(hsvAdj)

	out := imgHueAdj.MustMul1(ts.FloatScalar(255.0), true)

	imgFl.MustDrop()
	h.MustDrop()
	s.MustDrop()
	v.MustDrop()
	hAdj.MustDrop()
	hsvAdj.MustDrop()

	return out
}

func adjustGamma(x *ts.Tensor, gamma float64, gainOpt ...int64) *ts.Tensor {
	// var gain int64 = 1
	// if len(gainOpt) > 0 {
	// gain = gainOpt[0]
	// }
	// TODO
	return x
}

func RGB2HSV(x *ts.Tensor) *ts.Tensor {
	return rgb2HSV(x)
}

func HSV2RGB(x *ts.Tensor) *ts.Tensor {
	return hsv2RGB(x)
}

func pad(x *ts.Tensor, padding []int64, paddingMode string) *ts.Tensor {
	switch paddingMode {
	case "reflection":
		return x.MustReflectionPad2d(padding, false)
	case "constant":
		return x.MustConstantPadNd(padding, false)
	case "replicate":
		return x.MustReplicationPad2d(padding, false)
	case "circular":
		// TODO:
		// ref: https://github.com/pytorch/pytorch/blob/71f4c5c1f436258adc303b710efb3f41b2d50c4e/torch/nn/functional.py#L4493
		log.Fatal("Unsupported circular padding.")
	default:
		log.Fatalf("Unrecognized padding mode %q\n", paddingMode)
	}
	return nil
}

func getImageSize(x *ts.Tensor) (width, height int64) {
	assertImageTensor(x)
	dim := x.MustSize()
	return dim[len(dim)-1], dim[len(dim)-2]
}

func makeSlice(from, to int64) []int64 {
	n := from - to
	var out []int64 = make([]int64, n)
	for i := 0; i < int(n); i++ {
		out[i] = from + int64(i)
	}
	return out
}

func crop(x *ts.Tensor, top, left, height, width int64) *ts.Tensor {
	// return img[..., top:top + height, left:left + width]
	dim := x.MustSize()
	c := dim[0]

	var chans []ts.Tensor = make([]ts.Tensor, c)
	hNar := ts.NewNarrow(top, top+height)
	wNar := ts.NewNarrow(left, left+width)
	for i := 0; i < int(c); i++ {
		cx := x.Idx(ts.NewSelect(int64(i)))
		x1 := cx.Idx(hNar)
		cx.MustDrop()
		x1T := x1.MustT(true)
		x2 := x1T.Idx(wNar)
		x1T.MustDrop()
		out := x2.MustT(true)
		chans[i] = *out
	}

	cropTs := ts.MustStack(chans, 0)
	for i := range chans {
		chans[i].MustDrop()
	}
	return cropTs
}

// Crops the given image at the center.
// If the image is torch Tensor, it is expected
// to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
// If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.
func centerCrop(x *ts.Tensor, size []int64) *ts.Tensor {
	imgW, imgH := getImageSize(x)
	cropH, cropW := size[0], size[1]

	var paddedImg *ts.Tensor

	if cropW > imgW || cropH > imgH {
		// (crop_width - image_width) // 2 if crop_width > image_width else 0,
		// (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
		var left, top, right, bottom int64 = 0, 0, 0, 0
		if cropW > imgW {
			left = (cropW - imgW) / 2
			right = (cropW - imgW + 1) / 2
		}
		// (crop_height - image_height) // 2 if crop_height > image_height else 0,
		// (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
		if cropH > imgH {
			top = (cropH - imgH) / 2
			bottom = (cropH - imgH + 1) / 2
		}

		// floatX := x.MustTotype(gotch.Float, false)
		// paddedImg = pad(floatX, []int64{left, right, top, bottom}, "reflection")
		// floatX.MustDrop()

		paddedImg = pad(x, []int64{left, right, top, bottom}, "constant")
		imgW, imgH = getImageSize(paddedImg)
		if cropW == imgW && cropH == imgH {
			return paddedImg
		}
	} else {
		paddedImg = x.MustShallowClone()
	}

	// cropTop := int64(math.Floor(float64(imgH-cropH) / 2.0))
	// cropLeft := int64(math.Floor(float64(imgW-cropW) / 2.0))
	cropTop := (imgH - cropH) / 2
	cropLeft := (imgW - cropW) / 2

	out := crop(paddedImg, cropTop, cropLeft, cropH, cropW)
	paddedImg.MustDrop()

	return out
}

// cutout erases the input Tensor Image with given value
//
// Args:
// img (Tensor Image): Tensor image of size (C, H, W) to be erased
// i (int): i in (i,j) i.e coordinates of the upper left corner.
// j (int): j in (i,j) i.e coordinates of the upper left corner.
// h (int): Height of the erased region.
// w (int): Width of the erased region.
// v: Erasing value.
func cutout(x *ts.Tensor, top, left, height, width int64, rgbVal []int64) *ts.Tensor {
	output := x.MustZerosLike(false)
	output.Copy_(x)
	dim := output.MustSize()
	for i := 0; i < int(dim[0]); i++ {
		cIdx := ts.NewSelect(int64(i))
		hNar := ts.NewNarrow(top, top+height)
		wNar := ts.NewNarrow(left, left+width)
		srcIdx := []ts.TensorIndexer{cIdx, hNar, wNar}
		view := output.Idx(srcIdx)
		oneTs := view.MustOnesLike(false)
		vTs := oneTs.MustMul1(ts.IntScalar(rgbVal[i]), true)
		view.Copy_(vTs)
		vTs.MustDrop()
		view.MustDrop()
	}

	// output.Print()
	return output
}

func hflip(x *ts.Tensor) *ts.Tensor {
	assertImageTensor(x)
	return x.MustFlip([]int64{-1}, false)
}

func vflip(x *ts.Tensor) *ts.Tensor {
	assertImageTensor(x)
	return x.MustFlip([]int64{-2}, false)
}

// Ref. https://stackoverflow.com/questions/64197754
// Ref. https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
func getRotMat(theta float64) (*ts.Tensor, error) {
	grid := []float64{math.Cos(theta), -1 * (math.Sin(theta)), 0, math.Sin(theta), math.Cos(theta), 0}
	t, err := ts.NewTensorFromData(grid, []int64{2, 3})
	if err != nil {
		return nil, err
	}

	return t, nil
}

func rotImg(x *ts.Tensor, theta float64, dtype gotch.DType) (*ts.Tensor, error) {
	rotMat, err := getRotMat(theta)
	if err != nil {
		return nil, err
	}

	size := x.MustSize()
	mat := rotMat.MustUnsqueeze(0, true).MustTotype(dtype, true).MustRepeat([]int64{size[0], 1, 1}, true)
	grid := ts.MustAffineGridGenerator(mat, size, true).MustTo(x.MustDevice(), true)
	mat.MustDrop()

	out, err := ts.GridSampler(x, grid, 1, 1, true)
	if err != nil {
		return nil, err
	}
	grid.MustDrop()
	return out, nil
}

func applyGridTransform(x, gridInput *ts.Tensor, mode string, fillValue []float64) *ts.Tensor {
	dtype := gridInput.DType()
	img, needCast, needSqueeze, outDtype := castSqueezeIn(x, []gotch.DType{dtype})

	imgDim := img.MustSize()
	gridDim := gridInput.MustSize()
	var grid *ts.Tensor
	if imgDim[0] > 1 {
		// Apply same grid to a batch of images
		// grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])
		grid = gridInput.MustExpand([]int64{imgDim[0], gridDim[1], gridDim[2], gridDim[3]}, true, false)
	} else {
		grid = gridInput.MustShallowClone()
	}

	// Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
	// dummy = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype, device=img.device)
	// img = torch.cat((img, dummy), dim=1)
	dummy := ts.MustOnes([]int64{img.MustSize()[0], 1, img.MustSize()[2], img.MustSize()[3]}, img.DType(), img.MustDevice())
	imgCat := ts.MustCat([]ts.Tensor{*img, *dummy}, 1)
	dummy.MustDrop()
	img.MustDrop()

	// imgSample := gridSample(imgCat, grid, mode, "zeros", false)
	var (
		modeInt     int64 = 0
		paddingMode int64 = 0
	)

	imgSample := ts.MustGridSampler(imgCat, grid, modeInt, paddingMode, false)
	imgCat.MustDrop()
	grid.MustDrop()

	// TODO.
	// Fill with required color
	// mask = img[:, -1:, :, :]  # N * 1 * H * W
	// img = img[:, :-1, :, :]  # N * C * H * W
	// mask = mask.expand_as(img)
	// len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
	// fill_img = torch.tensor(fill, dtype=img.dtype, device=img.device).view(1, len_fill, 1, 1).expand_as(img)
	// if mode == 'nearest':
	// mask = mask < 0.5
	// img[mask] = fill_img[mask]
	// else:  # 'bilinear'
	// img = img * mask + (1.0 - mask) * fill_img
	image := imgSample.MustNarrow(0, 0, 1, false).MustNarrow(1, 0, 3, true)
	mask := imgSample.MustNarrow(0, 0, 1, false).MustNarrow(1, -1, 1, true).MustExpandAs(image, true)
	imgSample.MustDrop()
	fillImg := ts.MustOfSlice(fillValue).MustTotype(image.DType(), true).MustTo(image.MustDevice(), true).MustView([]int64{1, 3, 1, 1}, true).MustExpandAs(image, true)

	// img = img * mask + (1.0 - mask) * fill_img
	addTs := mask.MustMul1(ts.FloatScalar(-1), false).MustAdd1(ts.FloatScalar(1.0), true).MustMul(fillImg, true)
	imgOut := image.MustMul(mask, true).MustAdd(addTs, true)
	addTs.MustDrop()
	mask.MustDrop()
	fillImg.MustDrop()

	// out := castSqueezeOut(imgSample, needCast, needSqueeze, outDtype)
	out := castSqueezeOut(imgOut, needCast, needSqueeze, outDtype)
	imgOut.MustDrop()

	return out
}

// Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.
//
// In Perspective Transform each pixel (x, y) in the original image gets transformed as,
// (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )
// Args:
// - startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
// ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
// - endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
// ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.
// Returns:
// - octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
func perspectiveCoeff(startPoints, endPoints [][]int64) []float64 {
	size := int64(2 * len(startPoints))
	aMat := ts.MustZeros([]int64{size, 8}, gotch.Float, gotch.CPU)
	for i := 0; i < len(startPoints); i++ {
		p1 := endPoints[i]
		p2 := startPoints[i]
		// a_matrix[2 * i, :] = torch.tensor([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
		val1 := ts.MustOfSlice([]int64{p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]})
		// a_matrix[2 * i + 1, :] = torch.tensor([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
		val2 := ts.MustOfSlice([]int64{0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]})

		idx1 := ts.NewSelect(int64(2 * i))
		aMatView1 := aMat.Idx(idx1)
		aMatView1.Copy_(val1)
		val1.MustDrop()

		idx2 := ts.NewSelect(int64(2*i + 1))
		aMatView2 := aMat.Idx(idx2)
		aMatView2.Copy_(val2)
		val2.MustDrop()
	}

	var startData []int64
	for _, p := range startPoints {
		startData = append(startData, p[0], p[1])
	}

	// bMat := ts.MustOfSlice(startPoints).MustTotype(gotch.Float, true).MustView([]int64{8}, true)
	bMat := ts.MustOfSlice(startData).MustTotype(gotch.Float, true).MustView([]int64{8}, true)

	res := bMat.MustLstsq(aMat, true)

	aMat.MustDrop()
	outputTs := res.MustSqueeze1(1, true)
	output := outputTs.Float64Values()
	outputTs.MustDrop()

	return output
}

func perspectiveGrid(coef []float64, ow, oh int64, dtype gotch.DType, device gotch.Device) *ts.Tensor {
	// https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
	// src/libImaging/Geometry.c#L394
	// x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
	// y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)

	theta1 := ts.MustOfSlice([]float64{
		coef[0],
		coef[1],
		coef[2],
		coef[3],
		coef[4],
		coef[5],
	}).MustTotype(dtype, true).MustTo(device, true).MustView([]int64{1, 2, 3}, true)

	theta2 := ts.MustOfSlice([]float64{
		coef[6],
		coef[7],
		coef[1.0],
		coef[6],
		coef[7],
		coef[1.0],
	}).MustTotype(dtype, true).MustTo(device, true).MustView([]int64{1, 2, 3}, true)

	d := 0.5

	baseGrid := ts.MustEmpty([]int64{1, oh, ow, 3}, dtype, device)

	// x_grid = torch.linspace(d, ow * 1.0 + d - 1.0, steps=ow, device=device)
	endX := float64(ow) + d - 1.0
	xGrid := ts.MustLinspace(ts.FloatScalar(d), ts.FloatScalar(endX), []int64{ow}, dtype, device)

	// y_grid = torch.linspace(d, oh * 1.0 + d - 1.0, steps=oh, device=device).unsqueeze_(-1)
	endY := float64(oh) + d - 1.0
	yGrid := ts.MustLinspace(ts.FloatScalar(d), ts.FloatScalar(endY), []int64{oh}, dtype, device)

	// base_grid[..., 0].copy_(x_grid)
	// base_grid[..., 1].copy_(y_grid)
	// base_grid[..., 2].fill_(1)
	baseDim := baseGrid.MustSize()
	for i := 0; i < int(baseDim[1]); i++ {
		view := baseGrid.MustSelect(0, 0, false).MustSelect(0, int64(i), true).MustSelect(1, 0, true)
		view.Copy_(xGrid)
		view.MustDrop()
	}
	for i := 0; i < int(baseDim[2]); i++ {
		view := baseGrid.MustSelect(0, 0, false).MustSelect(1, int64(i), true).MustSelect(1, 1, true)
		view.Copy_(yGrid)
		view.MustDrop()
	}

	for i := 0; i < int(baseDim[2]); i++ {
		view := baseGrid.MustSelect(0, 0, false).MustSelect(1, int64(i), true).MustSelect(1, 2, true)
		// view.Fill_(ts.FloatScalar(1.0)) // NOTE. THIS CAUSES MEMORY LEAK!!!
		oneTs := view.MustOnesLike(false)
		view.Copy_(oneTs)
		oneTs.MustDrop()
		view.MustDrop()
	}

	// rescaled_theta1 = theta1.transpose(1, 2) / torch.tensor([0.5 * ow, 0.5 * oh], dtype=dtype, device=device)
	divTs := ts.MustOfSlice([]float64{0.5 * float64(ow), 0.5 * float64(oh)}).MustTotype(dtype, true).MustTo(device, true)
	rescaledTheta1 := theta1.MustTranspose(1, 2, true).MustDiv(divTs, true)
	divTs.MustDrop()
	outputGrid1 := baseGrid.MustView([]int64{1, oh * ow, 3}, false).MustBmm(rescaledTheta1, true)

	// output_grid2 = base_grid.view(1, oh * ow, 3).bmm(theta2.transpose(1, 2))
	rescaledTheta2 := theta2.MustTranspose(1, 2, true)
	outputGrid2 := baseGrid.MustView([]int64{1, oh * ow, 3}, false).MustBmm(rescaledTheta2, true)

	rescaledTheta1.MustDrop()
	rescaledTheta2.MustDrop()

	outputGrid := outputGrid1.MustDiv(outputGrid2, true).MustSub1(ts.FloatScalar(1.0), true).MustView([]int64{1, oh, ow, 2}, true)
	outputGrid2.MustDrop()

	baseGrid.MustDrop()

	return outputGrid
}

func perspective(x *ts.Tensor, startPoints, endPoints [][]int64, mode string, fillValue []float64) *ts.Tensor {
	coef := perspectiveCoeff(startPoints, endPoints)

	assertImageTensor(x)
	// assertGridTransformInputs(x, nil, mode, fillValue, []string{"nearest", "bilinear"}, coef)

	dim := x.MustSize()
	ow, oh := dim[len(dim)-1], dim[len(dim)-2]
	device := x.MustDevice()
	grid := perspectiveGrid(coef, ow, oh, gotch.Float, device)

	output := applyGridTransform(x, grid, mode, fillValue)
	grid.MustDrop()

	return output
}

// Apply affine transformation on the image keeping image center invariant.
//
//If the image is torch Tensor, it is expected
// to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
// Args:
// - img (Tensor): image to transform.
// - angle (number): rotation angle in degrees between -180 and 180, clockwise direction.
// - translate (sequence of integers): horizontal and vertical translations (post-rotation translation)
// - scale (float): overall scale
// - shear (float or sequence): shear angle value in degrees between -180 to 180, clockwise direction.
// If a sequence is specified, the first value corresponds to a shear parallel to the x axis, while
// the second value corresponds to a shear parallel to the y axis.
// - interpolation (InterpolationMode): Desired interpolation enum defined by
// :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
// If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
// - fill (sequence or number, optional): Pixel fill value for the area outside the transformed
// image. If given a number, the value is used for all bands respectively.
func affine(img *ts.Tensor, angle float64, translations []int64, scale float64, shear []float64, interpolationMode string, fillValue []float64) *ts.Tensor {

	var translateF []float64
	for _, v := range translations {
		translateF = append(translateF, float64(v))
	}

	matrix := getInverseAffineMatrix([]float64{0.0, 0.0}, angle, translateF, scale, shear)

	// dtype := gotch.Float
	dtype := img.DType()
	device := img.MustDevice()
	dim := img.MustSize()
	theta := ts.MustOfSlice(matrix).MustTotype(dtype, true).MustTo(device, true).MustReshape([]int64{1, 2, 3}, true)

	// grid will be generated on the same device as theta and img
	w := dim[len(dim)-1]
	h := dim[len(dim)-2]
	ow := w
	oh := h

	// grid = _gen_affine_grid(theta, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2])
	grid := genAffineGrid(theta, w, h, ow, oh)
	// grid := ts.MustEmpty([]int64{1, 512, 512, 2}, dtype, device)

	out := applyGridTransform(img, grid, interpolationMode, fillValue)

	grid.MustDrop()
	theta.MustDrop()

	return out
}

// Helper method to compute inverse matrix for affine transformation
//
// As it is explained in PIL.Image.rotate
// We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
// where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
//       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
//       RSS is rotation with scale and shear matrix
//       RSS(a, s, (sx, sy)) =
//       = R(a) * S(s) * SHy(sy) * SHx(sx)
//       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
//         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
//         [ 0                    , 0                                      , 1 ]
//
// where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
// SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
//          [0, 1      ]              [-tan(s), 1]
//
// Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1
func getInverseAffineMatrix(center []float64, angle float64, translate []float64, scale float64, shear []float64) []float64 {

	// convert to randiants
	rot := angle * math.Pi / 180
	sx := shear[0] * math.Pi / 180
	sy := shear[1] * math.Pi / 180

	cx, cy := center[0], center[1]
	tx, ty := translate[0], translate[1]

	// RSS without scaling
	// a = math.cos(rot - sy) / math.cos(sy)
	a := math.Cos(rot-sy) / math.Cos(sy)
	// b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
	b := -math.Cos(rot-sy)*math.Tan(sx)/math.Cos(sy) - math.Sin(rot)
	// c = math.sin(rot - sy) / math.cos(sy)
	c := math.Sin(rot-sy) / math.Cos(sy)
	// d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)
	d := -math.Sin(rot-sy)*math.Tan(sx)/math.Cos(sy) + math.Cos(rot)

	// Inverted rotation matrix with scale and shear
	// det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
	// matrix = [d, -b, 0.0, -c, a, 0.0]
	var matrix []float64 = []float64{d, -b, 0.0, -c, a, 0.0}
	// matrix = [x / scale for x in matrix]
	var mat []float64
	for _, v := range matrix {
		mat = append(mat, v/scale)
	}

	// Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
	// matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
	mat[2] += mat[0]*(-cx-tx) + mat[1]*(-cy-ty)
	// matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
	mat[5] += mat[3]*(-cx-tx) + mat[4]*(-cy-ty)

	// Apply center translation: C * RSS^-1 * C^-1 * T^-1
	// matrix[2] += cx
	mat[2] += cx
	// matrix[5] += cy
	mat[5] += cy

	return mat
}

// https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
// AffineGridGenerator.cpp#L18
// Difference with AffineGridGenerator is that:
// 1) we normalize grid values after applying theta
// 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate
func genAffineGrid(theta *ts.Tensor, w, h, ow, oh int64) *ts.Tensor {
	d := 0.5
	dtype := theta.DType()
	device := theta.MustDevice()

	// base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
	x := ts.MustEmpty([]int64{oh, ow, 3}, dtype, device)

	startX := float64(-ow)*0.5 + d
	endX := float64(ow)*0.5 + d - 1.0
	xGrid := ts.MustLinspace(ts.FloatScalar(startX), ts.FloatScalar(endX), []int64{ow}, dtype, device)

	startY := float64(-oh)*0.5 + d
	endY := float64(oh)*0.5 + d - 1.0
	yGrid := ts.MustLinspace(ts.FloatScalar(startY), ts.FloatScalar(endY), []int64{oh}, dtype, device).MustUnsqueeze(-1, true)

	oneGrid := ts.MustOnes([]int64{ow}, dtype, device)

	// base_grid[..., 0].copy_(x_grid)
	// base_grid[..., 1].copy_(y_grid)
	// base_grid[..., 2].fill_(1)
	xview := x.MustTranspose(2, 0, false).MustSelect(0, 0, true).MustTranspose(0, 1, true)
	xview.Copy_(xGrid)
	xview.MustDrop()

	yview := x.MustTranspose(2, 0, false).MustSelect(0, 1, true).MustTranspose(0, 1, true)
	yview.Copy_(yGrid)
	yview.MustDrop()

	oview := x.MustTranspose(2, 0, false).MustSelect(0, 2, true).MustTranspose(0, 1, true)
	oview.Copy_(oneGrid)
	oview.MustDrop()

	// rescaled_theta1 = theta1.transpose(1, 2) / torch.tensor([0.5 * ow, 0.5 * oh], dtype=dtype, device=device)
	divTs := ts.MustOfSlice([]float64{0.5 * float64(w), 0.5 * float64(h)}).MustTotype(dtype, true).MustTo(device, true)
	rescaledTheta := theta.MustTranspose(1, 2, false).MustDiv(divTs, true)
	divTs.MustDrop()

	// output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
	outputGrid := x.MustView([]int64{1, oh * ow, 3}, true).MustBmm(rescaledTheta, true).MustView([]int64{1, oh, ow, 2}, true)
	xGrid.MustDrop()
	yGrid.MustDrop()
	rescaledTheta.MustDrop()

	return outputGrid
}

// randPvalue generates a random propability value [0, 1]
func randPvalue() float64 {
	rand.Seed(time.Now().UnixNano())
	var min, max float64 = 0.0, 1.0

	r := min + rand.Float64()*(max-min)
	return r
}

func getImageChanNum(x *ts.Tensor) int64 {
	dim := x.MustSize()
	switch {
	case len(dim) == 2:
		return 1
	case len(dim) > 2:
		return dim[len(dim)-3]
	default:
		log.Fatalf("Input image tensor should have dim of 2 or more. Got %v\n", len(dim))
	}

	log.Fatalf("Input image tensor should have dim of 2 or more. Got %v\n", len(dim))
	return -1
}

// solarize solarizes an RGB/grayscale image by inverting all pixel values above a threshold.
// Args:
// - img (Tensor): Image to have its colors inverted.
// If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
// where ... means it can have an arbitrary number of leading dimensions.
// - threshold (float): All pixels equal or above this value are inverted.
func solarize(img *ts.Tensor, threshold float64) *ts.Tensor {
	assertImageTensor(img)

	dim := img.MustSize()
	if len(dim) < 3 {
		log.Fatalf("Input image tensor should have at least 3 dimensions. Got %v", len(dim))
	}

	assertChannels(img, []int64{1, 3})

	invertedImg := invert(img)

	// return torch.where(img >= threshold, inverted_img, img)
	conditionTs := img.MustGe(ts.FloatScalar(threshold), false)

	out := img.MustWhere1(conditionTs, invertedImg, false)

	invertedImg.MustDrop()
	conditionTs.MustDrop()

	return out
}

// invert inverts image tensor.
func invert(img *ts.Tensor) *ts.Tensor {
	assertImageTensor(img)

	dim := img.MustSize()
	if len(dim) < 3 {
		log.Fatalf("Input image tensor should have at least 3 dimensions. Got %v", len(dim))
	}

	assertChannels(img, []int64{1, 3})

	var bound int64 = 255
	// return bound - img
	out := img.MustMul1(ts.IntScalar(-1), false).MustAdd1(ts.IntScalar(bound), true)
	return out
}

func posterize(img *ts.Tensor, bits uint8) *ts.Tensor {
	assertImageTensor(img)

	dim := img.MustSize()

	if len(dim) < 3 {
		log.Fatalf("Input image tensor should have at least 3 dimensions. Got %v\n", len(dim))
	}

	dtype := img.DType()
	if dtype != gotch.Uint8 {
		log.Fatalf("Only dtype uint8 image tensors are supported. Got %v", dtype)
	}

	assertChannels(img, []int64{1, 3})

	// mask = -int(2**(8 - bits))  # JIT-friendly for: ~(2 ** (8 - bits) - 1)
	// or mask := -int64(1<<(uint8(8) - bits))
	mask := -int64(math.Exp2(float64(uint8(8) - bits)))

	out := img.MustBitwiseAnd(ts.IntScalar(mask), false)
	return out
}

func autocontrast(img *ts.Tensor) *ts.Tensor {
	assertImageTensor(img)

	dim := img.MustSize()

	if len(dim) < 3 {
		log.Fatalf("Input image tensor should have at least 3 dimensions. Got %v\n", len(dim))
	}

	// NOTE. image tensor expected to be float dtype [0,1]
	var bound float64 = 1.0
	dtype := img.DType()

	// minimum = img.amin(dim=(-2, -1), keepdim=True).to(dtype)
	minTs := img.MustAmin([]int64{-2, -1}, true, false).MustTotype(dtype, true)
	// maximum = img.amax(dim=(-2, -1), keepdim=True).to(dtype)
	maxTs := img.MustAmax([]int64{-2, -1}, true, false).MustTotype(dtype, true)

	// eq_idxs = torch.where(minimum == maximum)[0]
	// NOTE. Eq(minTs, maxTs) give [n, c, 1, 1] or [channels, 1, 1]
	eqIdx := minTs.MustEq1(maxTs, false).MustSqueeze1(-1, true).MustSqueeze1(-1, true).MustTotype(gotch.Int64, true)

	// minimum[eq_idxs] = 0
	minTsView := minTs.MustIndexSelect(0, eqIdx, false)
	zerosTs := minTsView.MustZerosLike(false)
	minTsView.Copy_(zerosTs)
	zerosTs.MustDrop()
	minTsView.MustDrop()

	// maximum[eq_idxs] = bound
	maxTsView := maxTs.MustIndexSelect(0, eqIdx, false)
	boundTs := maxTsView.MustOnesLike(false).MustMul1(ts.FloatScalar(bound), true)
	maxTsView.Copy_(boundTs)
	boundTs.MustDrop()
	maxTsView.MustDrop()

	// scale = bound / (maximum - minimum)
	scale := maxTs.MustSub(minTs, false).MustPow(ts.IntScalar(-1), true).MustMul1(ts.FloatScalar(bound), true)
	//
	// return ((img - minimum) * scale).clamp(0, bound).to(img.dtype)
	out := img.MustSub(minTs, false).MustMul(scale, true).MustClamp(ts.IntScalar(0), ts.FloatScalar(bound), true).MustTotype(dtype, true)

	minTs.MustDrop()
	maxTs.MustDrop()
	eqIdx.MustDrop()
	scale.MustDrop()

	return out
}

func adjustSharpness(img *ts.Tensor, factor float64) *ts.Tensor {
	if factor < 0 {
		log.Fatalf("Sharpness factor should not be negative. Got %v", factor)
	}

	assertImageTensor(img)
	assertChannels(img, []int64{1, 3})

	dim := img.MustSize()

	var out *ts.Tensor
	if (dim[len(dim)-1]) <= 2 || (dim[len(dim)-2] <= 2) {
		out = img.MustShallowClone()
		return out
	}

	// return _blend(img, _blurred_degenerate_image(img), sharpness_factor)
	img1 := blurredDegenerateImage(img)
	out = blend(img, img1, factor)

	img1.MustDrop()
	return out
}

func blurredDegenerateImage(img *ts.Tensor) *ts.Tensor {
	dtype := gotch.Float
	device := img.MustDevice()
	dim := img.MustSize()

	// kernel = torch.ones((3, 3), dtype=dtype, device=img.device)
	kernel := ts.MustOnes([]int64{3, 3}, dtype, device)

	// kernel[1, 1] = 5.0
	kernelView := kernel.MustNarrow(1, 1, 1, false).MustNarrow(0, 1, 1, true)
	centerVal := kernelView.MustOnesLike(false).MustMul1(ts.FloatScalar(5.0), true)
	kernelView.Copy_(centerVal) // center kernel value
	centerVal.MustDrop()
	kernelView.MustDrop()

	// kernel /= kernel.sum()
	kernelSum := kernel.MustSum(dtype, false)
	kernelS := kernel.MustDiv(kernelSum, false)
	kernelSum.MustDrop()
	// kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])
	kdim := kernelS.MustSize()
	kdtype := kernelS.DType()
	kernelExp := kernelS.MustExpand([]int64{dim[len(dim)-3], 1, kdim[0], kdim[1]}, true, false)

	// result_tmp, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype, ])
	resTmp, needCast, needSqueeze, outDtype := castSqueezeIn(img, []gotch.DType{kdtype})

	// result_tmp = conv2d(result_tmp, kernel, groups=result_tmp.shape[-3])
	stride := []int64{1, 1}
	padding := []int64{0, 0}
	dilation := []int64{1, 1}
	resTmpDim := resTmp.MustSize()
	group := resTmpDim[len(resTmpDim)-3]
	// resTmp1 shape: [1, 3, h, w]
	resTmp1 := ts.MustConv2d(resTmp, kernelExp, ts.NewTensor(), stride, padding, dilation, group)

	// result_tmp = _cast_squeeze_out(result_tmp, need_cast, need_squeeze, out_dtype)
	// resTmp2 shape: [3, h, w]
	resTmp2 := castSqueezeOut(resTmp1, needCast, needSqueeze, outDtype)

	// result = img.clone()
	// NOTE. out := img.MustShallowClone() doesn't work!
	out := img.MustZerosLike(false)

	// result[..., 1:-1, 1:-1] = result_tmp
	hDim := int64(len(dim) - 2) // second last dim
	wDim := int64(len(dim) - 1) // last dim
	outView := out.MustNarrow(hDim, 1, dim[len(dim)-2]-2, false).MustNarrow(wDim, 1, dim[len(dim)-1]-2, true)

	outView.Copy_(resTmp2)

	outView.MustDrop()
	kernelS.MustDrop()
	kernelExp.MustDrop()
	resTmp.MustDrop()
	resTmp1.MustDrop()
	resTmp2.MustDrop()

	return out
}

func equalize(img *ts.Tensor) *ts.Tensor {
	assertImageTensor(img)

	shape := img.MustSize()
	ndim := len(shape)
	dtype := img.DType()

	if ndim < 3 || ndim > 4 {
		log.Fatalf("Input image should have 3 or 4 dimensions. Got %v", ndim)
	}

	if dtype != gotch.Uint8 {
		log.Fatalf("Only dtype uint8 image tensors are supported. Got %v", dtype)
	}

	assertChannels(img, []int64{1, 3})

	// single image
	if ndim == 3 {
		out := equalizeSingleImage(img)
		return out
	}

	// batched images
	var images []ts.Tensor
	for i := 0; i < int(shape[0]); i++ {
		x := img.MustSelect(0, int64(i), false)
		o := equalizeSingleImage(x)
		images = append(images, *o)
		x.MustDrop()
	}

	out := ts.MustStack(images, 0)
	for _, x := range images {
		x.MustDrop()
	}

	return out
}

func equalizeSingleImage(img *ts.Tensor) *ts.Tensor {
	dim := img.MustSize()
	var scaledChans []ts.Tensor = make([]ts.Tensor, int(dim[0]))
	for i := 0; i < int(dim[0]); i++ {
		cTs := img.MustSelect(0, int64(i), false)
		scaledChan := scaleChannel(cTs)
		cTs.MustDrop()
		scaledChans[i] = *scaledChan
	}

	out := ts.MustStack(scaledChans, 0)

	for _, x := range scaledChans {
		x.MustDrop()
	}

	return out
}

func scaleChannel(imgChan *ts.Tensor) *ts.Tensor {
	// hist = torch.histc(img_chan.to(torch.float32), bins=256, min=0, max=255)
	// NOTE. Use `Bincount` so that result similar to Pytorch. If use `Histc`, results are different!!!
	device := imgChan.MustDevice()
	var histo *ts.Tensor
	if device == gotch.CPU {
		histo = imgChan.MustFlatten(0, -1, false).MustBincount(ts.NewTensor(), 256, true)
	} else {
		histo = imgChan.MustTotype(gotch.Float, false).MustHistc(256, true)
	}

	// nonzero_hist = hist[hist != 0]
	nonzeroHistoIdx := histo.MustNonzero(false).MustFlatten(0, -1, true)
	nonzeroHisto := histo.MustIndexSelect(0, nonzeroHistoIdx, false)
	nonzeroHistoIdx.MustDrop()

	// step = torch.div(nonzero_hist[:-1].sum(), 255, rounding_mode='floor')
	histoLen := nonzeroHisto.MustSize()[0]
	step := nonzeroHisto.MustNarrow(0, 0, histoLen-1, true).MustSum(gotch.Float, true).MustFloorDivide1(ts.FloatScalar(255.0), true)

	stepVal := step.Float64Values()[0]
	if stepVal == 0 {
		histo.MustDrop()
		step.MustDrop()
		out := imgChan.MustShallowClone()
		return out
	}

	// lut = torch.div(torch.cumsum(hist, 0) + torch.div(step, 2, rounding_mode='floor'), step, rounding_mode='floor')
	halfStep := step.MustFloorDivide1(ts.FloatScalar(2.0), false)
	lut := histo.Must_Cumsum(0, true).MustAdd(halfStep, true).MustFloorDivide(step, true)
	step.MustDrop()
	halfStep.MustDrop()

	// lut = torch.nn.functional.pad(lut, [1, 0])[:-1].clamp(0, 255)
	lutLen := lut.MustSize()[0]
	lut = lut.MustConstantPadNd([]int64{1, 0}, true).MustNarrow(0, 0, lutLen, true).MustClamp(ts.FloatScalar(0), ts.FloatScalar(255.0), true)

	// return lut[img_chan.to(torch.int64)].to(torch.uint8)
	// can't index using 2d index. Have to flatten and then reshape
	// result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
	// result = result.reshape_as(im)
	flattenImg := imgChan.MustFlatten(0, -1, false).MustTotype(gotch.Int64, true)
	out := lut.MustIndexSelect(0, flattenImg, true).MustReshapeAs(imgChan, true).MustTotype(gotch.Uint8, true)
	flattenImg.MustDrop()

	return out
}

// Normalize a float tensor image with mean and standard deviation.
//
// Args:
// - tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
// - mean (sequence): Sequence of means for each channel.
// - std (sequence): Sequence of standard deviations for each channel.
// Returns:
// - Tensor: Normalized Tensor image.
func normalize(img *ts.Tensor, mean, std []float64) *ts.Tensor {
	for _, v := range std {
		if v == 0 {
			log.Fatalf("One of std (%v) is zero. This is invalid as it leads to division by zero.", std)
		}
	}

	assertImageTensor(img)

	dim := img.MustSize()
	// dtype := img.DType()
	device := img.MustDevice()
	if len(dim) < 3 {
		log.Fatalf("Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() =%v", dim)
	}

	meanTs := ts.MustOfSlice(mean).MustTotype(gotch.Float, true).MustTo(device, true)
	stdTs := ts.MustOfSlice(std).MustTotype(gotch.Float, true).MustTo(device, true)

	var mTs, sTs *ts.Tensor
	meanSize := meanTs.MustSize()
	stdSize := stdTs.MustSize()
	switch len(meanSize) {
	case 1:
		mTs = meanTs.MustView([]int64{-1, 1, 1}, true)
	case 3:
		mTs = meanTs.MustShallowClone()
		meanTs.MustDrop()
	default:
		log.Fatalf("mean must be 1 or 3 elements. Got %v\n", len(mean))
	}

	switch len(stdSize) {
	case 1:
		sTs = stdTs.MustView([]int64{-1, 1, 1}, true)
	case 3:
		sTs = stdTs.MustShallowClone()
		stdTs.MustDrop()
	default:
		log.Fatalf("std must be 1 or 3 elements. Got %v\n", len(std))
	}

	out := img.MustSub(mTs, false).MustDiv(sTs, true)

	mTs.MustDrop()
	sTs.MustDrop()

	return out
}

// Byte2FloatImage converts uint8 dtype image tensor to float dtype.
// It's panic if input image is not uint8 dtype.
func Byte2FloatImage(x *ts.Tensor) *ts.Tensor {
	dtype := x.DType()
	if dtype != gotch.Uint8 {
		err := fmt.Errorf("Input tensor is not uint8 dtype (%v)", dtype)
		panic(err)
	}

	return x.MustDiv1(ts.FloatScalar(255.0), false)
}

// Float2ByteImage converts float dtype image to uint8 dtype image.
// It's panic if input is not float dtype tensor.
func Float2ByteImage(x *ts.Tensor) *ts.Tensor {
	dtype := x.DType()
	if dtype != gotch.Float && dtype != gotch.Double {
		err := fmt.Errorf("Input tensor is not float dtype (%v)", dtype)
		panic(err)
	}

	return x.MustMul1(ts.IntScalar(255), false).MustTotype(gotch.Uint8, true)
}
