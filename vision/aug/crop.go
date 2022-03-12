package aug

import (
	"fmt"
	"log"

	// "math"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

type RandomCrop struct {
	size            []int64
	padding         []int64
	paddingIfNeeded bool
	paddingMode     string
}

func newRandomCrop(size, padding []int64, paddingIfNeeded bool, paddingMode string) *RandomCrop {
	return &RandomCrop{
		size:            size,
		padding:         padding,
		paddingIfNeeded: paddingIfNeeded,
		paddingMode:     paddingMode,
	}
}

// get parameters for crop
func (c *RandomCrop) params(x *ts.Tensor) (int64, int64, int64, int64) {
	w, h := getImageSize(x)
	th, tw := c.size[0], c.size[1]
	if h+1 < th || w+1 < tw {
		err := fmt.Errorf("Required crop size %v is larger then input image size %v", c.size, []int64{h, w})
		log.Fatal(err)
	}

	if w == tw && h == th {
		return 0, 0, h, w
	}

	iTs := ts.MustRandint(h-th+1, []int64{1}, gotch.Int64, gotch.CPU)
	i := iTs.Int64Values()[0]
	iTs.MustDrop()

	jTs := ts.MustRandint(w-tw+1, []int64{1}, gotch.Int64, gotch.CPU)
	j := jTs.Int64Values()[0]
	jTs.MustDrop()

	return i, j, th, tw
}

func (c *RandomCrop) Forward(x *ts.Tensor) *ts.Tensor {
	fx := Byte2FloatImage(x)

	var img *ts.Tensor
	if c.padding != nil {
		img = pad(fx, c.padding, c.paddingMode)
	} else {
		img = fx.MustShallowClone()
	}

	w, h := getImageSize(fx)

	var (
		paddedW  *ts.Tensor
		paddedWH *ts.Tensor
	)
	// pad width if needed
	if c.paddingIfNeeded && w < c.size[1] {
		padding := []int64{c.size[1] - w, 0}
		paddedW = pad(img, padding, c.paddingMode)
	} else {
		paddedW = img.MustShallowClone()
	}
	img.MustDrop()

	// pad height if needed
	if c.paddingIfNeeded && h < c.size[0] {
		padding := []int64{0, c.size[0] - h}
		paddedWH = pad(paddedW, padding, c.paddingMode)
	} else {
		paddedWH = paddedW.MustShallowClone()
	}

	paddedW.MustDrop()

	// i, j, h, w = self.get_params(img, self.size)
	i, j, h, w := c.params(x)
	out := crop(paddedWH, i, j, h, w)
	paddedWH.MustDrop()

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()
	return bx
}

func WithRandomCrop(size []int64, padding []int64, paddingIfNeeded bool, paddingMode string) Option {
	return func(o *Options) {
		c := newRandomCrop(size, padding, paddingIfNeeded, paddingMode)
		o.randomCrop = c
	}
}

// CenterCrop crops the given image at the center.
// If the image is torch Tensor, it is expected
// to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
// If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.
type CenterCrop struct {
	size []int64
}

func newCenterCrop(size []int64) *CenterCrop {
	if len(size) != 2 {
		err := fmt.Errorf("Expected size of 2 elements. Got %v\n", len(size))
		log.Fatal(err)
	}
	return &CenterCrop{size}
}

func (cc *CenterCrop) Forward(x *ts.Tensor) *ts.Tensor {
	return centerCrop(x, cc.size)
}

func WithCenterCrop(size []int64) Option {
	return func(o *Options) {
		cc := newCenterCrop(size)
		o.centerCrop = cc
	}
}
