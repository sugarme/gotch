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
	out, err := vision.Resize(x, rs.width, rs.height)
	if err != nil {
		log.Fatal(err)
	}
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
