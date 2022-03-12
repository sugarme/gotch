package vision

// SqueezeNet implementation.

import (
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

func snMaxPool2D(xs *ts.Tensor) *ts.Tensor {
	return xs.MustMaxPool2d([]int64{3, 3}, []int64{2, 2}, []int64{0, 0}, []int64{1, 1}, true, false)
}

func fire(p *nn.Path, cIn int64, cSqueeze int64, cExp1 int64, cExp3 int64) ts.ModuleT {

	cfg3 := nn.DefaultConv2DConfig()
	cfg3.Padding = []int64{1, 1}

	squeeze := nn.NewConv2D(p.Sub("squeeze"), cIn, cSqueeze, 1, nn.DefaultConv2DConfig())
	exp1 := nn.NewConv2D(p.Sub("expand1x1"), cSqueeze, cExp1, 1, nn.DefaultConv2DConfig())
	exp3 := nn.NewConv2D(p.Sub("expand3x3"), cSqueeze, cExp3, 3, cfg3)

	// NOTE: train will not be used
	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		tmp1 := xs.Apply(squeeze)
		tmp2 := tmp1.MustRelu(true)

		exp1Tmp := tmp2.Apply(exp1)
		exp1Ts := exp1Tmp.MustRelu(true)

		exp3Tmp := tmp2.Apply(exp3)
		exp3Ts := exp3Tmp.MustRelu(true)

		return ts.MustCat([]ts.Tensor{*exp1Ts, *exp3Ts}, 1)
	})
}

func squeezenet(p *nn.Path, v1_0 bool, nclasses int64) ts.ModuleT {
	fp := p.Sub("features")
	cp := p.Sub("classifier")

	initialConvConfig := nn.DefaultConv2DConfig()
	initialConvConfig.Stride = []int64{2, 2}

	finalConvConfig := nn.DefaultConv2DConfig()
	finalConvConfig.Stride = []int64{1, 1}

	features := nn.SeqT()

	if v1_0 {
		features.Add(nn.NewConv2D(fp.Sub("0"), 3, 96, 7, initialConvConfig))

		features.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
			return xs.MustRelu(false)
		}))

		features.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
			return snMaxPool2D(xs)
		}))

		features.Add(fire(fp.Sub("3"), 96, 16, 64, 64))

		features.Add(fire(fp.Sub("4"), 128, 16, 64, 64))

		features.Add(fire(fp.Sub("5"), 128, 32, 128, 128))

		features.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
			return snMaxPool2D(xs)
		}))

		features.Add(fire(fp.Sub("7"), 256, 32, 128, 128))

		features.Add(fire(fp.Sub("8"), 256, 48, 192, 192))

		features.Add(fire(fp.Sub("9"), 384, 48, 192, 192))

		features.Add(fire(fp.Sub("10"), 384, 64, 256, 256))

		features.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
			return snMaxPool2D(xs)
		}))

		features.Add(fire(fp.Sub("12"), 512, 64, 256, 256))

	} else {
		features.Add(nn.NewConv2D(fp.Sub("0"), 3, 64, 3, initialConvConfig))

		features.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
			return xs.MustRelu(false)
		}))

		features.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
			return snMaxPool2D(xs)
		}))

		features.Add(fire(fp.Sub("3"), 64, 16, 64, 64))

		features.Add(fire(fp.Sub("4"), 128, 16, 64, 64))

		features.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
			return snMaxPool2D(xs)
		}))

		features.Add(fire(fp.Sub("6"), 128, 32, 128, 128))

		features.Add(fire(fp.Sub("7"), 256, 32, 128, 128))

		features.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
			return snMaxPool2D(xs)
		}))

		features.Add(fire(fp.Sub("9"), 256, 48, 192, 192))

		features.Add(fire(fp.Sub("10"), 384, 48, 192, 192))

		features.Add(fire(fp.Sub("11"), 384, 64, 256, 256))

		features.Add(fire(fp.Sub("12"), 512, 64, 256, 256))
	}

	features.AddFnT(nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		return ts.MustDropout(xs, 0.5, train)
	}))

	features.Add(nn.NewConv2D(cp.Sub("1"), 512, nclasses, 1, finalConvConfig))

	features.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		tmp1 := xs.MustRelu(false)
		tmp2 := tmp1.MustAdaptiveAvgPool2d([]int64{1, 1}, false)
		tmp1.MustDrop()
		res := tmp2.FlatView()
		tmp2.MustDrop()
		return res
	}))

	return features
}

func SqueezeNetV1_0(p *nn.Path, nclasses int64) ts.ModuleT {
	return squeezenet(p, true, nclasses)
}

func SqueezeNetV1_1(p *nn.Path, nclasses int64) ts.ModuleT {
	return squeezenet(p, false, nclasses)
}
