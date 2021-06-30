package aug

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

type ResizeModule struct {
	height int64
	width  int64
}

func newResizeModule(h, w int64) *ResizeModule {
	return &ResizeModule{h, w}
}

// Forward implements ts.Module for RandRotateModule
// NOTE. input tensor must be uint8 (Byte) dtype otherwise panic!
func (rs *ResizeModule) Forward(x *ts.Tensor) *ts.Tensor {
	dtype := x.DType()
	if dtype != gotch.Uint8 {
		err := fmt.Errorf("Invalid dtype. Expect uint8 (Byte) dtype. Got %v\n", dtype)
		panic(err)
	}

	device := x.MustDevice()
	var xCPU *ts.Tensor
	if device != gotch.CPU {
		xCPU = x.MustTo(device, false)
	} else {
		xCPU = x.MustShallowClone()
	}

	out, err := vision.Resize(xCPU, rs.width, rs.height)
	if err != nil {
		log.Fatal(err)
	}

	xCPU.MustDrop()

	return out.MustTo(device, true)
}

func WithResize(h, w int64) Option {
	return func(o *Options) {
		rs := newResizeModule(h, w)
		o.resize = rs
	}
}

// TODO.
type RandomResizedCrop struct{}

type DownSample struct{}

func newDownSample(p float64) *DownSample {
	return &DownSample{}
}

// Forward implements ts.Module for RandRotateModule
// NOTE. input tensor must be uint8 (Byte) dtype otherwise panic!
func (rs *DownSample) Forward(x *ts.Tensor) *ts.Tensor {
	dtype := x.DType()
	if dtype != gotch.Uint8 {
		err := fmt.Errorf("Invalid dtype. Expect uint8 (Byte) dtype. Got %v\n", dtype)
		panic(err)
	}

	device := x.MustDevice()
	h := x.MustSize()[1]
	w := x.MustSize()[2]
	var xCPU *ts.Tensor
	if device != gotch.CPU {
		xCPU = x.MustTo(device, false)
	} else {
		xCPU = x.MustShallowClone()
	}

	out, err := vision.Resize(xCPU, w/2, h/2)
	if err != nil {
		log.Fatal(err)
	}

	xCPU.MustDrop()
	return out.MustTo(device, true)
}

type ZoomIn struct {
	v float64 // v should be [0, 0.5]
}

func newZoomIn(v float64) *ZoomIn {
	return &ZoomIn{v: v}
}

func WithZoomIn(v float64) Option {
	if v < 0 || v > 0.5 {
		err := fmt.Errorf("Invalid input value. Expect value in range [0, 0.5]. Got %v\n", v)
		panic(err)
	}
	return func(o *Options) {
		ds := newZoomIn(v)
		o.zoomIn = ds
	}
}

// Forward implements ts.Module for RandRotateModule
// NOTE. input tensor must be uint8 (Byte) dtype otherwise panic!
func (rs *ZoomIn) Forward(x *ts.Tensor) *ts.Tensor {
	dtype := x.DType()
	if dtype != gotch.Uint8 {
		err := fmt.Errorf("Invalid dtype. Expect uint8 (Byte) dtype. Got %v\n", dtype)
		panic(err)
	}

	device := x.MustDevice()
	h := x.MustSize()[1]
	w := x.MustSize()[2]
	var xCPU *ts.Tensor
	if device != gotch.CPU {
		xCPU = x.MustTo(device, false)
	} else {
		xCPU = x.MustShallowClone()
	}

	var out *ts.Tensor
	var err error
	r := randPvalue()
	switch {
	case r < rs.v:
		cropW := int64(rs.v) * w
		cropH := int64(rs.v) * h
		newW := w - cropW
		newH := h - cropH
		// img = PIL.ImageOps.fit(img, size=(new_w,new_h), bleed=v/2, method=Image.BILINEAR)
		fitImg := fitImg(xCPU, newW, newH)
		xCPU.MustDrop()
		// return img.resize((w,h), resample=Image.BILINEAR)
		out, err = vision.Resize(fitImg, w, h)
		if err != nil {
			log.Fatal(err)
		}

		fitImg.MustDrop()
	default:
		out = x.MustShallowClone()
	}

	return out.MustTo(device, true)
}

// TODO.
func fitImg(x *ts.Tensor, w, h int64) *ts.Tensor {

	panic("Not implemented")
}

type ZoomOut struct {
	v float64 // v should be [0, 0.5]
}

func newZoomOut(v float64) *ZoomOut {
	return &ZoomOut{v: v}
}

func WithZoomOut(v float64) Option {
	if v < 0 || v > 0.5 {
		err := fmt.Errorf("Invalid input value. Expect value in range [0, 0.5]. Got %v\n", v)
		panic(err)
	}
	return func(o *Options) {
		ds := newZoomOut(v)
		o.zoomOut = ds
	}
}

// Forward implements ts.Module for RandRotateModule
// NOTE. input tensor must be uint8 (Byte) dtype otherwise panic!
func (rs *ZoomOut) Forward(x *ts.Tensor) *ts.Tensor {
	dtype := x.DType()
	if dtype != gotch.Uint8 {
		err := fmt.Errorf("Invalid dtype. Expect uint8 (Byte) dtype. Got %v\n", dtype)
		panic(err)
	}

	device := x.MustDevice()
	var xCPU *ts.Tensor
	if device != gotch.CPU {
		xCPU = x.MustTo(device, false)
	} else {
		xCPU = x.MustShallowClone()
	}

	Fimg := Byte2FloatImage(xCPU)

	fmt.Printf("Fimg size: %v\n", Fimg.MustSize())
	h := float64(Fimg.MustSize()[1])
	w := float64(Fimg.MustSize()[2])
	padW := int64(rs.v * w)
	padH := int64(rs.v * h)
	fmt.Printf("padH: %v - padW: %v\n", padH, padW)

	// img = np.pad(img, [(pad_h//2,pad_h//2), (pad_w//2,pad_w//2), (0,0)], mode='reflect')
	padding := []int64{padH / 2, padH / 2, padW / 2, padW / 2, 0, 0}
	fmt.Printf("padding: %+v\n", padding)
	padImg := pad(Fimg, padding, "reflection")
	xCPU.MustDrop()
	Fimg.MustDrop()
	// return img.resize((w,h), resample=Image.BILINEAR)
	out, err := vision.Resize(padImg, int64(w), int64(h))
	if err != nil {
		log.Fatal(err)
	}

	padImg.MustDrop()

	return out.MustTo(device, true)
}
