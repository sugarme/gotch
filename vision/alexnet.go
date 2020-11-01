package vision

import (
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

// AlexNet implementation
// https://arxiv.org/abs/1404.5997

func anConv2d(p *nn.Path, cIn, cOut, ksize, padding, stride int64) *nn.Conv2D {
	config := nn.DefaultConv2DConfig()
	config.Stride = []int64{stride, stride}
	config.Padding = []int64{padding, padding}

	return nn.NewConv2D(p, cIn, cOut, ksize, config)
}

func anMaxPool2d(xs *ts.Tensor, ksize, stride int64) *ts.Tensor {
	return xs.MustMaxPool2d([]int64{ksize, ksize}, []int64{stride, stride}, []int64{0, 0}, []int64{1, 1}, false, false)
}

func features(p *nn.Path) ts.ModuleT {
	seq := nn.SeqT()
	seq.Add(anConv2d(p.Sub("0"), 3, 64, 11, 2, 4))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		tmp1 := xs.MustRelu(false)
		res := anMaxPool2d(tmp1, 3, 2)
		tmp1.MustDrop()
		return res
	}))

	seq.Add(anConv2d(p.Sub("3"), 64, 192, 5, 1, 2))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		tmp1 := xs.MustRelu(false)
		res := anMaxPool2d(tmp1, 3, 2)
		tmp1.MustDrop()
		return res
	}))

	seq.Add(anConv2d(p.Sub("6"), 192, 384, 3, 1, 1))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	seq.Add(anConv2d(p.Sub("8"), 384, 256, 3, 1, 1))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	seq.Add(anConv2d(p.Sub("10"), 256, 256, 3, 1, 1))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		tmp1 := xs.MustRelu(false)
		res := anMaxPool2d(tmp1, 3, 2)
		tmp1.MustDrop()
		return res
	}))

	return seq
}

func classifier(p *nn.Path, nclasses int64) ts.ModuleT {
	seq := nn.SeqT()

	seq.AddFnT(nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		return ts.MustDropout(xs, 0.5, train)
	}))

	seq.Add(nn.NewLinear(p.Sub("1"), 256*6*6, 4096, nn.DefaultLinearConfig()))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	seq.AddFnT(nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		return ts.MustDropout(xs, 0.5, train)
	}))

	seq.Add(nn.NewLinear(p.Sub("4"), 4096, 4096, nn.DefaultLinearConfig()))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	seq.Add(nn.NewLinear(p.Sub("6"), 4096, nclasses, nn.DefaultLinearConfig()))

	return seq
}

func AlexNet(p *nn.Path, nclasses int64) ts.ModuleT {
	seq := nn.SeqT()

	seq.Add(features(p.Sub("features")))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		tmp1 := xs.MustAdaptiveAvgPool2d([]int64{6, 6}, false)
		res := tmp1.FlatView()
		tmp1.MustDrop()
		return res
	}))

	seq.Add(classifier(p.Sub("classifier"), nclasses))

	return seq
}
