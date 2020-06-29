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

func basicLayer(path nn.Path, cIn, cOut, stride, cnt int64) (retVal ts.ModuleT) {

	layer := nn.SeqT()
	layer.Add(basicBlock(path.Sub("0"), cIn, cOut, stride))

	for blockIndex := 0; blockIndex < int(cnt); blockIndex++ {
		layer.Add(basicBlock(path.Sub(string(blockIndex)), cOut, cOut, 1))
	}

	return layer
}

func resnet(path nn.Path, nclasses int64, c1, c2, c3, c4 int64) (retVal nn.FuncT) {
	conv1 := conv2d(path.Sub("conv1"), 3, 64, 7, 3, 2)
	bn1 := nn.BatchNorm2D(path.Sub("bn1"), 64, nn.DefaultBatchNormConfig())
	layer1 := basicLayer(path.Sub("layer1"), 64, 64, 1, c1)
	layer2 := basicLayer(path.Sub("layer2"), 64, 64, 1, c2)
	layer3 := basicLayer(path.Sub("layer3"), 64, 64, 1, c3)
	layer4 := basicLayer(path.Sub("layer4"), 64, 64, 1, c4)

	if nclasses > 0 {
		linearConfig := nn.DefaultLinearConfig()
		fc := nn.NewLinear(path.Sub("fc"), 512, nclasses, *linearConfig)

		return nn.NewFuncT(func(xs ts.Tensor, train bool) ts.Tensor {
			return xs.Apply(conv1).ApplyT(bn1, train).MustRelu(false).MustMaxPool2D([]int64{3, 3}, []int64{2, 2}, []int64{1, 1}, []int64{1, 1}, false, true).ApplyT(layer1, train).ApplyT(layer2, train).ApplyT(layer3, train).ApplyT(layer4, train).MustAdaptiveAvgPool2D([]int64{1, 1}).FlatView().ApplyOpt(ts.WithModule(fc))
		})

	} else {
		// No final layer
		return nn.NewFuncT(func(xs ts.Tensor, train bool) ts.Tensor {
			return xs.Apply(conv1).ApplyT(bn1, train).MustRelu(false).MustMaxPool2D([]int64{3, 3}, []int64{2, 2}, []int64{1, 1}, []int64{1, 1}, false, true).ApplyT(layer1, train).ApplyT(layer2, train).ApplyT(layer3, train).ApplyT(layer4, train).MustAdaptiveAvgPool2D([]int64{1, 1}).FlatView()
		})
	}
}

// Creates a ResNet-18 model.
func ResNet18(path nn.Path, numClasses int64) (retVal nn.FuncT) {
	return resnet(path, numClasses, 2, 2, 2, 2)
}

func ResNet18NoFinalLayer(path nn.Path) (retVal nn.FuncT) {
	return resnet(path, 0, 2, 2, 2, 2)
}

func ResNet34(path nn.Path, numClasses int64) (retVal nn.FuncT) {
	return resnet(path, numClasses, 3, 4, 6, 3)
}

func ResNet34NoFinalLayer(path nn.Path) (retVal nn.FuncT) {
	return resnet(path, 0, 3, 4, 6, 3)
}

// Bottleneck versions for ResNet 50, 101, and 152.
func bottleneckBlock(path nn.Path, cIn, cOut, stride, e int64) (retVal ts.ModuleT) {

	eDim := e * cOut
	conv1 := conv2d(path.Sub("conv1"), cIn, cOut, 1, 0, 1)
	bn1 := nn.BatchNorm2D(path.Sub("bn1"), cOut, nn.DefaultBatchNormConfig())
	conv2 := conv2d(path.Sub("conv2"), cOut, cOut, 3, 1, stride)
	bn2 := nn.BatchNorm2D(path.Sub("bn2"), cOut, nn.DefaultBatchNormConfig())
	conv3 := conv2d(path.Sub("conv3"), cOut, eDim, 1, 0, 1)
	bn3 := nn.BatchNorm2D(path.Sub("bn3"), eDim, nn.DefaultBatchNormConfig())
	downsample := downSample(path.Sub("downsample"), cIn, eDim, stride)

	return nn.NewFuncT(func(xs ts.Tensor, train bool) ts.Tensor {
		ys := xs.Apply(conv1).ApplyT(bn1, train).MustRelu(true).Apply(conv2).ApplyT(bn2, train).MustRelu(true).Apply(conv3).ApplyT(bn3, train)
		return xs.ApplyT(downsample, train).MustAdd(ys, true).MustRelu(true)
	})
}

func bottleneckLayer(path nn.Path, cIn, cOut, stride, cnt int64) (retVal ts.ModuleT) {

	layer := nn.SeqT()
	layer.Add(bottleneckBlock(path.Sub("0"), cIn, cOut, stride, 4))
	for blockIndex := 0; blockIndex < int(cnt); blockIndex++ {
		layer.Add(bottleneckBlock(path.Sub(string(blockIndex)), (cOut * 4), cOut, 1, 4))
	}

	return layer
}

func bottleneckResnet(path nn.Path, nclasses int64, c1, c2, c3, c4 int64) (retVal ts.ModuleT) {
	conv1 := conv2d(path.Sub("conv1"), 3, 64, 7, 3, 2)
	bn1 := nn.BatchNorm2D(path.Sub("bn1"), 64, nn.DefaultBatchNormConfig())
	layer1 := bottleneckLayer(path.Sub("layer1"), 64, 64, 1, c1)
	layer2 := bottleneckLayer(path.Sub("layer2"), 4*64, 128, 2, c2)
	layer3 := bottleneckLayer(path.Sub("layer3"), 4*128, 256, 2, c3)
	layer4 := bottleneckLayer(path.Sub("layer4"), 4*256, 512, 2, c4)

	if nclasses > 0 {
		fc := nn.NewLinear(path.Sub("fc"), 4*512, nclasses, *nn.DefaultLinearConfig())

		return nn.NewFuncT(func(xs ts.Tensor, train bool) ts.Tensor {
			return xs.Apply(conv1).ApplyT(bn1, train).MustRelu(true).MustMaxPool2D([]int64{3, 3}, []int64{2, 2}, []int64{1, 1}, []int64{1, 1}, false, true).ApplyT(layer1, train).ApplyT(layer2, train).ApplyT(layer3, train).ApplyT(layer4, train).MustAdaptiveAvgPool2D([]int64{1, 1}).FlatView().ApplyOpt(ts.WithModule(fc))
		})
	} else {
		return nn.NewFuncT(func(xs ts.Tensor, train bool) ts.Tensor {
			return xs.Apply(conv1).ApplyT(bn1, train).MustRelu(true).MustMaxPool2D([]int64{3, 3}, []int64{2, 2}, []int64{1, 1}, []int64{1, 1}, false, true).ApplyT(layer1, train).ApplyT(layer2, train).ApplyT(layer3, train).ApplyT(layer4, train).MustAdaptiveAvgPool2D([]int64{1, 1}).FlatView()
		})
	}
}

func ResNet50(path nn.Path, numClasses int64) (retVal ts.ModuleT) {
	return bottleneckResnet(path, numClasses, 3, 4, 6, 3)
}

func ResNet50NoFinalLayer(path nn.Path) (retVal ts.ModuleT) {
	return bottleneckResnet(path, 0, 3, 4, 6, 3)
}

func ResNet101(path nn.Path, numClasses int64) (retVal ts.ModuleT) {
	return bottleneckResnet(path, numClasses, 3, 4, 23, 3)
}

func ResNet101NoFinalLayer(path nn.Path) (retVal ts.ModuleT) {
	return bottleneckResnet(path, 0, 3, 4, 23, 3)
}

func ResNet152(path nn.Path, numClasses int64) (retVal ts.ModuleT) {
	return bottleneckResnet(path, numClasses, 3, 8, 36, 3)
}

func ResNet150NoFinalLayer(path nn.Path) (retVal ts.ModuleT) {
	return bottleneckResnet(path, 0, 3, 8, 36, 3)
}
