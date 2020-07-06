package vision

// DenseNet implementation.
//
// See "Densely Connected Convolutional Networks", Huang et al 2016.
// https://arxiv.org/abs/1608.06993

import (
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func dnConv2d(p nn.Path, cIn, cOut, ksize, padding, stride int64) (retVal nn.Conv2D) {
	config := nn.DefaultConv2DConfig()
	config.Stride = []int64{stride, stride}
	config.Padding = []int64{padding, padding}
	config.Bias = false

	return nn.NewConv2D(&p, cIn, cOut, ksize, config)
}

func denseLayer(p nn.Path, cIn, bnSize, growth int64) (retVal ts.ModuleT) {
	cInter := bnSize * growth
	bn1 := nn.BatchNorm2D(p.Sub("norm1"), cIn, nn.DefaultBatchNormConfig())
	conv1 := dnConv2d(p.Sub("conv1"), cIn, cInter, 1, 0, 1)
	bn2 := nn.BatchNorm2D(p.Sub("norm2"), cInter, nn.DefaultBatchNormConfig())
	conv2 := dnConv2d(p.Sub("conv2"), cInter, growth, 3, 1, 1)

	return nn.NewFuncT(func(xs ts.Tensor, train bool) ts.Tensor {
		ys1 := xs.ApplyT(bn1, train)
		ys2 := ys1.MustRelu(true)
		ys3 := ys2.Apply(conv1)
		ys2.MustDrop()
		ys4 := ys3.ApplyT(bn2, train)
		ys3.MustDrop()
		ys5 := ys4.MustRelu(true)
		ys := ys5.Apply(conv2)
		ys5.MustDrop()

		res := ts.MustCat([]ts.Tensor{xs, ys}, 1, false)
		ys.MustDrop()

		return res
	})
}
