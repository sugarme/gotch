package vision

import (
	"fmt"

	nn "github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

// ResNet implementation.
//
// See "Deep Residual Learning for Image Recognition" He et al. 2015
// https://arxiv.org/abs/1512.03385

func layerZero(p *nn.Path) ts.ModuleT {
	conv1 := conv2dNoBias(p.Sub("conv1"), 3, 64, 7, 3, 2)
	bn1 := nn.BatchNorm2D(p.Sub("bn1"), 64, nn.DefaultBatchNormConfig())
	layer0 := nn.SeqT()
	layer0.Add(conv1)
	layer0.Add(bn1)
	layer0.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))
	layer0.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustMaxPool2d([]int64{3, 3}, []int64{2, 2}, []int64{1, 1}, []int64{1, 1}, false, false)
	}))

	return layer0
}

func basicLayer(path *nn.Path, cIn, cOut, stride, cnt int64) ts.ModuleT {
	layer := nn.SeqT()
	layer.Add(newBasicBlock(path.Sub("0"), cIn, cOut, stride))
	for blockIndex := 1; blockIndex < int(cnt); blockIndex++ {
		layer.Add(newBasicBlock(path.Sub(fmt.Sprint(blockIndex)), cOut, cOut, 1))
	}

	return layer
}

func conv2d(p *nn.Path, cIn, cOut, ksize, padding, stride int64) *nn.Conv2D {
	config := nn.DefaultConv2DConfig()
	config.Stride = []int64{stride, stride}
	config.Padding = []int64{padding, padding}

	return nn.NewConv2D(p, cIn, cOut, ksize, config)
}

func conv2dNoBias(p *nn.Path, cIn, cOut, ksize, padding, stride int64) *nn.Conv2D {
	config := nn.DefaultConv2DConfig()
	config.Bias = false
	config.Stride = []int64{stride, stride}
	config.Padding = []int64{padding, padding}

	return nn.NewConv2D(p, cIn, cOut, ksize, config)
}

func downSample(path *nn.Path, cIn, cOut, stride int64) ts.ModuleT {
	if stride != 1 || cIn != cOut {
		seq := nn.SeqT()
		seq.Add(conv2dNoBias(path.Sub("0"), cIn, cOut, 1, 0, stride))
		seq.Add(nn.BatchNorm2D(path.Sub("1"), cOut, nn.DefaultBatchNormConfig()))

		return seq
	}
	return nn.SeqT()
}

type basicBlock struct {
	Conv1      *nn.Conv2D
	Bn1        *nn.BatchNorm
	Conv2      *nn.Conv2D
	Bn2        *nn.BatchNorm
	Downsample ts.ModuleT
}

func newBasicBlock(path *nn.Path, cIn, cOut, stride int64) *basicBlock {
	conv1 := conv2dNoBias(path.Sub("conv1"), cIn, cOut, 3, 1, stride)
	bn1 := nn.BatchNorm2D(path.Sub("bn1"), cOut, nn.DefaultBatchNormConfig())
	conv2 := conv2dNoBias(path.Sub("conv2"), cOut, cOut, 3, 1, 1)
	bn2 := nn.BatchNorm2D(path.Sub("bn2"), cOut, nn.DefaultBatchNormConfig())
	downsample := downSample(path.Sub("downsample"), cIn, cOut, stride)

	return &basicBlock{conv1, bn1, conv2, bn2, downsample}
}

func (bb *basicBlock) ForwardT(x *ts.Tensor, train bool) *ts.Tensor {
	c1 := bb.Conv1.ForwardT(x, train)
	bn1Ts := bb.Bn1.ForwardT(c1, train)
	c1.MustDrop()
	relu := bn1Ts.MustRelu(true)
	c2 := bb.Conv2.ForwardT(relu, train)
	relu.MustDrop()
	bn2Ts := bb.Bn2.ForwardT(c2, train)
	c2.MustDrop()
	dsl := bb.Downsample.ForwardT(x, train)
	dslAdd := dsl.MustAdd(bn2Ts, true)
	bn2Ts.MustDrop()
	res := dslAdd.MustRelu(true)

	return res
}

func resnet(p *nn.Path, nclasses int64, c1, c2, c3, c4 int64) nn.FuncT {
	seq := nn.SeqT()
	layer0 := layerZero(p)
	layer1 := basicLayer(p.Sub("layer1"), 64, 64, 1, 3)
	layer2 := basicLayer(p.Sub("layer2"), 64, 128, 2, 4)
	layer3 := basicLayer(p.Sub("layer3"), 128, 256, 2, 6)
	layer4 := basicLayer(p.Sub("layer4"), 256, 512, 2, 3)
	seq.Add(layer0)
	seq.Add(layer1)
	seq.Add(layer2)
	seq.Add(layer3)
	seq.Add(layer4)

	if nclasses > 0 {
		// With final layer
		linearConfig := nn.DefaultLinearConfig()
		fc := nn.NewLinear(p.Sub("fc"), 512, nclasses, linearConfig)
		return nn.NewFuncT(func(x *ts.Tensor, train bool) *ts.Tensor {
			output := seq.ForwardT(x, train)
			avgpool := output.MustAdaptiveAvgPool2d([]int64{1, 1}, true)
			fv := avgpool.FlatView()
			avgpool.MustDrop()
			retVal := fv.ApplyOpt(ts.WithModule(fc))
			fv.MustDrop()

			return retVal
		})
	} else {
		// no final layer
		return nn.NewFuncT(func(x *ts.Tensor, train bool) *ts.Tensor {
			output := seq.ForwardT(x, train)
			avgpool := output.MustAdaptiveAvgPool2d([]int64{1, 1}, true)
			retVal := avgpool.FlatView()
			avgpool.MustDrop()

			return retVal
		})
	}
}

type bottleneckBlock struct {
	Conv1      *nn.Conv2D
	Bn1        *nn.BatchNorm
	Conv2      *nn.Conv2D
	Bn2        *nn.BatchNorm
	Conv3      *nn.Conv2D
	Bn3        *nn.BatchNorm
	Downsample ts.ModuleT
}

// ForwardT implements ModuleT for bottleneckBlock.
func (b *bottleneckBlock) ForwardT(xs *ts.Tensor, train bool) *ts.Tensor {
	c1 := xs.Apply(b.Conv1)
	bn1 := c1.ApplyT(b.Bn1, train)
	c1.MustDrop()
	relu1 := bn1.MustRelu(true)
	c2 := relu1.Apply(b.Conv2)
	relu1.MustDrop()
	bn2 := c2.ApplyT(b.Bn2, train)
	relu2 := bn2.MustRelu(true)
	c3 := relu2.Apply(b.Conv3)
	relu2.MustDrop()
	bn3 := c3.ApplyT(b.Bn3, train)

	dsl := xs.ApplyT(b.Downsample, train)
	add := dsl.MustAdd(bn3, true)
	bn3.MustDrop()
	res := add.MustRelu(true)
	return res
}

// Bottleneck versions for ResNet 50, 101, and 152.
func newBottleneckBlock(path *nn.Path, cIn, cOut, stride, e int64) *bottleneckBlock {
	eDim := e * cOut
	conv1 := conv2d(path.Sub("conv1"), cIn, cOut, 1, 0, 1)
	bn1 := nn.BatchNorm2D(path.Sub("bn1"), cOut, nn.DefaultBatchNormConfig())
	conv2 := conv2d(path.Sub("conv2"), cOut, cOut, 3, 1, stride)
	bn2 := nn.BatchNorm2D(path.Sub("bn2"), cOut, nn.DefaultBatchNormConfig())
	conv3 := conv2d(path.Sub("conv3"), cOut, eDim, 1, 0, 1)
	bn3 := nn.BatchNorm2D(path.Sub("bn3"), eDim, nn.DefaultBatchNormConfig())
	downsample := downSample(path.Sub("downsample"), cIn, eDim, stride)

	return &bottleneckBlock{
		Conv1:      conv1,
		Bn1:        bn1,
		Conv2:      conv2,
		Bn2:        bn2,
		Conv3:      conv3,
		Bn3:        bn3,
		Downsample: downsample,
	}
}

func bottleneckLayer(path *nn.Path, cIn, cOut, stride, cnt int64) ts.ModuleT {
	layer := nn.SeqT()
	layer.Add(newBottleneckBlock(path.Sub("0"), cIn, cOut, stride, 4))
	for blockIndex := 1; blockIndex < int(cnt); blockIndex++ {
		layer.Add(newBottleneckBlock(path.Sub(fmt.Sprint(blockIndex)), (cOut * 4), cOut, 1, 4))
	}

	return layer
}

func bottleneckResnet(path *nn.Path, nclasses int64, c1, c2, c3, c4 int64) ts.ModuleT {
	conv1 := conv2d(path.Sub("conv1"), 3, 64, 7, 3, 2)
	bn1 := nn.BatchNorm2D(path.Sub("bn1"), 64, nn.DefaultBatchNormConfig())

	layer1 := bottleneckLayer(path.Sub("layer1"), 64, 64, 1, c1)
	layer2 := bottleneckLayer(path.Sub("layer2"), 4*64, 128, 2, c2)
	layer3 := bottleneckLayer(path.Sub("layer3"), 4*128, 256, 2, c3)
	layer4 := bottleneckLayer(path.Sub("layer4"), 4*256, 512, 2, c4)

	seq := nn.SeqT()
	seq.Add(conv1)
	seq.Add(bn1)
	seq.Add(layer1)
	seq.Add(layer2)
	seq.Add(layer3)
	seq.Add(layer4)

	if nclasses > 0 {
		// With final layer
		linearConfig := nn.DefaultLinearConfig()
		fc := nn.NewLinear(path.Sub("fc"), 4*512, nclasses, linearConfig)
		return nn.NewFuncT(func(x *ts.Tensor, train bool) *ts.Tensor {
			output := seq.ForwardT(x, train)
			avgpool := output.MustAdaptiveAvgPool2d([]int64{1, 1}, true)
			fv := avgpool.FlatView()
			avgpool.MustDrop()
			retVal := fv.ApplyOpt(ts.WithModule(fc))
			fv.MustDrop()

			return retVal
		})
	} else {
		// no final layer
		return nn.NewFuncT(func(x *ts.Tensor, train bool) *ts.Tensor {
			output := seq.ForwardT(x, train)
			avgpool := output.MustAdaptiveAvgPool2d([]int64{1, 1}, true)
			retVal := avgpool.FlatView()
			avgpool.MustDrop()

			return retVal
		})
	}
}

// ResNet18 creates a ResNet-18 model.
func ResNet18(path *nn.Path, numClasses int64) nn.FuncT {
	return resnet(path, numClasses, 2, 2, 2, 2)
}

// ResNet18 creates a ResNet-18 model without final fully connfected layer.
func ResNet18NoFinalLayer(path *nn.Path) nn.FuncT {
	return resnet(path, 0, 2, 2, 2, 2)
}

// ResNet34 creates a ResNet-34 model.
func ResNet34(path *nn.Path, numClasses int64) nn.FuncT {
	return resnet(path, numClasses, 3, 4, 6, 3)
}

// ResNet34 creates a ResNet-34 model without final fully connfected layer.
func ResNet34NoFinalLayer(path *nn.Path) nn.FuncT {
	return resnet(path, 0, 3, 4, 6, 3)
}

// ResNet50 creates a ResNet-50 model.
func ResNet50(path *nn.Path, numClasses int64) ts.ModuleT {
	return bottleneckResnet(path, numClasses, 3, 4, 6, 3)
}

// ResNet50 creates a ResNet-50 model without final fully connfected layer.
func ResNet50NoFinalLayer(path *nn.Path) ts.ModuleT {
	return bottleneckResnet(path, 0, 3, 4, 6, 3)
}

// ResNet101 creates a ResNet-101 model.
func ResNet101(path *nn.Path, numClasses int64) ts.ModuleT {
	return bottleneckResnet(path, numClasses, 3, 4, 23, 3)
}

// ResNet101 creates a ResNet-101 model without final fully connfected layer.
func ResNet101NoFinalLayer(path *nn.Path) ts.ModuleT {
	return bottleneckResnet(path, 0, 3, 4, 23, 3)
}

// ResNet152 creates a ResNet-152 model.
func ResNet152(path *nn.Path, numClasses int64) ts.ModuleT {
	return bottleneckResnet(path, numClasses, 3, 8, 36, 3)
}

// ResNet150 creates a ResNet-150 model without final fully connfected layer.
func ResNet150NoFinalLayer(path *nn.Path) ts.ModuleT {
	return bottleneckResnet(path, 0, 3, 8, 36, 3)
}
