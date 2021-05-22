package aug

import (
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
func (rs *ResizeModule) Forward(x *ts.Tensor) *ts.Tensor {
	imgTs := x.MustTotype(gotch.Uint8, false)
	out, err := vision.Resize(imgTs, rs.width, rs.height)
	if err != nil {
		log.Fatal(err)
	}
	imgTs.MustDrop()
	return out
}

func WithResize(h, w int64) Option {
	return func(o *Options) {
		rs := newResizeModule(h, w)
		o.resize = rs
	}
}

// TODO.
type RandomResizedCrop struct{}
