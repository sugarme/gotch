package aug

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// RandomRotate randomly rotates a tensor image within a specifed angle range (degree).
func RandomRotate(img *ts.Tensor, min, max float64) (*ts.Tensor, error) {
	if min > max {
		tmp := min
		min = max
		max = tmp
	}
	if min < -360 || min > 360 || max < -360 || max > 360 {
		err := fmt.Errorf("min and max should be in range from -360 to 360. Got %v and %v\n", min, max)
		return nil, err
	}
	// device := img.MustDevice()
	dtype := gotch.Double
	rand.Seed(time.Now().UnixNano())
	angle := min + rand.Float64()*(max-min)

	theta := float64(angle) * (math.Pi / 180)
	input := img.MustUnsqueeze(0, false).MustTotype(dtype, true)
	r, err := rotImg(input, theta, dtype)
	if err != nil {
		return nil, err
	}
	input.MustDrop()
	rotatedImg := r.MustSqueeze(true)
	return rotatedImg, nil
}

func Rotate(img *ts.Tensor, angle float64) (*ts.Tensor, error) {
	if angle < -360 || angle > 360 {
		err := fmt.Errorf("angle must be in range (-360, 360)")
		return nil, err
	}
	dtype := gotch.Double
	theta := float64(angle) * (math.Pi / 180)
	input := img.MustUnsqueeze(0, false).MustTotype(dtype, true)
	r, err := rotImg(input, theta, dtype)
	if err != nil {
		return nil, err
	}
	input.MustDrop()
	rotatedImg := r.MustSqueeze(true)
	return rotatedImg, nil
}

// RotateModule
type RotateModule struct {
	angle float64
}

func newRotate(angle float64) *RotateModule {
	return &RotateModule{angle}
}

// Forward implements ts.Module for RotateModule
func (r *RotateModule) Forward(x *ts.Tensor) *ts.Tensor {
	fx := Byte2FloatImage(x)

	out, err := Rotate(fx, r.angle)
	if err != nil {
		log.Fatal(err)
	}

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func WithRotate(angle float64) Option {
	return func(o *Options) {
		r := newRotate(angle)
		o.rotate = r
	}
}

// RandomRotateModule
type RandRotateModule struct {
	minAngle float64
	maxAngle float64
}

func newRandRotate(min, max float64) *RandRotateModule {
	return &RandRotateModule{min, max}
}

// Forward implements ts.Module for RandRotateModule
func (rr *RandRotateModule) Forward(x *ts.Tensor) *ts.Tensor {
	fx := Byte2FloatImage(x)

	out, err := RandomRotate(fx, rr.minAngle, rr.maxAngle)
	if err != nil {
		log.Fatal(err)
	}

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func WithRandRotate(minAngle, maxAngle float64) Option {
	return func(o *Options) {
		r := newRandRotate(minAngle, maxAngle)
		o.randRotate = r
	}
}
