package vision

import (
	nn "github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

// ResNet implementation.
//
// See "Deep Residual Learning for Image Recognition" He et al. 2015
// https://arxiv.org/abs/1512.03385

func conv2d(path nn.Path, cIn, cOut, ksize, padding, stride int64) (retVal nn.Conv2D) {
	config := nn.DefaultConv2DConfig()
	config.Stride = []int64{stride, stride}
	config.Padding = []int64{padding, padding}
	config.Bias = false

	return nn.NewConv2D(&path, cIn, cOut, ksize, config)
}

func downSample(path nn.Path, cIn, cOut, stride int64) (retVal ts.ModuleT) {

	if stride != 1 || cIn != cOut {
		seq := nn.SeqT()
		seq.Add(conv2d(path, cIn, cOut, 1, 0, stride))

	} else {
		retVal = nn.SeqT()
	}

	return retVal
}

func basicBlock(path nn.Path, cIn, cOut, stride int64) (retVal ts.ModuleT) {

	// TODO: check and make sure delete middle tensors created in C memory
	// otherwise, there will be a memory blow out!
	conv1 := conv2d(path.Sub("conv1"), cIn, cOut, 3, 1, stride)
	bn1 := nn.BatchNorm2D(path.Sub("bn1"), cOut, nn.DefaultBatchNormConfig())
	conv2 := conv2d(path.Sub("conv2"), cOut, cOut, 3, 1, 1)
	bn2 := nn.BatchNorm2D(path.Sub("bn2"), cOut, nn.DefaultBatchNormConfig())
	downsample := downSample(path.Sub("downsample"), cIn, cOut, stride)

	return nn.NewFuncT(func(xs ts.Tensor, train bool) ts.Tensor {
		ys := xs.Apply(conv1).ApplyT(bn1, train).MustRelu(false).Apply(conv2).ApplyT(bn2, train)

		return xs.ApplyT(downsample, train).MustAdd(ys, true).MustRelu(true)
	})
}
