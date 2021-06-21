package aug

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

type GaussianBlur struct {
	kernelSize []int64   // >= 0 && ks%2 != 0
	sigma      []float64 // [0.1, 2.0] range(min, max)
}

// ks : kernal size. Can be 1-2 element slice
// sigma: minimal and maximal standard deviation that can be chosen for blurring kernel
// range (min, max). Can be 1-2 element slice
func newGaussianBlur(ks []int64, sig []float64) *GaussianBlur {
	if len(ks) == 0 || len(ks) > 2 {
		err := fmt.Errorf("Kernel size should have 1-2 elements. Got %v\n", len(ks))
		log.Fatal(err)
	}
	for _, size := range ks {
		if size <= 0 || size%2 == 0 {
			err := fmt.Errorf("Kernel size should be an odd and positive number.")
			log.Fatal(err)
		}
	}

	if len(sig) == 0 || len(sig) > 2 {
		err := fmt.Errorf("Sigma should have 1-2 elements. Got %v\n", len(sig))
		log.Fatal(err)
	}

	for _, s := range sig {
		if s <= 0 {
			err := fmt.Errorf("Sigma should be a positive number.")
			log.Fatal(err)
		}
	}

	var kernelSize []int64
	switch len(ks) {
	case 1:
		kernelSize = []int64{ks[0], ks[0]}
	case 2:
		kernelSize = ks
	default:
		panic("Shouldn't reach here.")
	}

	var sigma []float64
	switch len(sig) {
	case 1:
		sigma = []float64{sig[0], sig[0]}
	case 2:
		min := sig[0]
		max := sig[1]
		if min > max {
			min = sig[1]
			max = sig[0]
		}
		sigma = []float64{min, max}
	default:
		panic("Shouldn't reach here.")
	}

	return &GaussianBlur{
		kernelSize: kernelSize,
		sigma:      sigma,
	}
}

func (b *GaussianBlur) Forward(x *ts.Tensor) *ts.Tensor {
	assertImageTensor(x)
	fx := Byte2FloatImage(x)

	sigmaTs := ts.MustEmpty([]int64{1}, gotch.Float, gotch.CPU)
	sigmaTs.MustUniform_(b.sigma[0], b.sigma[1])
	sigmaVal := sigmaTs.Float64Values()[0]
	sigmaTs.MustDrop()

	out := gaussianBlur(fx, b.kernelSize, []float64{sigmaVal, sigmaVal})
	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func WithGaussianBlur(ks []int64, sig []float64) Option {
	return func(o *Options) {
		gb := newGaussianBlur(ks, sig)
		o.gaussianBlur = gb
	}
}
