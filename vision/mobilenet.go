package vision

// MobileNet V2 implementation.
// https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

// Conv2D + BatchNorm2D + ReLU6
func cbr(p *nn.Path, cIn, cOut, ks, stride, g int64) ts.ModuleT {
	config := nn.DefaultConv2DConfig()
	config.Stride = []int64{stride, stride}
	pad := (ks - 1) / 2
	config.Padding = []int64{pad, pad}
	config.Groups = g
	config.Bias = false

	seq := nn.SeqT()

	seq.Add(nn.NewConv2D(p.Sub("0"), cIn, cOut, ks, config))

	seq.Add(nn.BatchNorm2D(p.Sub("1"), cOut, nn.DefaultBatchNormConfig()))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		tmp := xs.MustRelu(false)
		res := tmp.MustClampMax(ts.FloatScalar(6.0), true)
		return res
	}))

	return seq
}

// Inverted Residual block.
func inv(p *nn.Path, cIn, cOut, stride, er int64) ts.ModuleT {
	cHidden := er * cIn
	seq := nn.SeqT()

	id := 0
	if er != 1 {
		seq.Add(cbr(p.Sub(fmt.Sprintf("%v", id)), cIn, cHidden, 1, 1, 1))
		id += 1
	}

	seq.Add(cbr(p.Sub(fmt.Sprintf("%v", id)), cHidden, cHidden, 3, stride, cHidden))

	configNoBias := nn.DefaultConv2DConfig()
	configNoBias.Bias = false
	seq.Add(nn.NewConv2D(p.Sub(fmt.Sprintf("%v", id+1)), cHidden, cOut, 1, configNoBias))

	seq.Add(nn.BatchNorm2D(p.Sub(fmt.Sprintf("%v", id+2)), cOut, nn.DefaultBatchNormConfig()))

	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		ys := xs.ApplyT(seq, train)
		if stride == 1 && cIn == cOut {
			res := ys.MustAdd(xs, true)
			return res
		} else {
			return ys
		}
	})
}

var invertedResidualSettings [][]int64 = [][]int64{
	{1, 16, 1, 1},
	{6, 24, 2, 2},
	{6, 32, 3, 2},
	{6, 64, 4, 2},
	{6, 96, 3, 1},
	{6, 160, 3, 2},
	{6, 320, 1, 1},
}

func MobileNetV2(p *nn.Path, nclasses int64) ts.ModuleT {
	fp := p.Sub("features")
	cp := p.Sub("classifier")
	cIn := int64(32)

	features := nn.SeqT()

	features.Add(cbr(fp.Sub("0"), 3, cIn, 3, 2, 1))

	layerId := 1
	for _, l := range invertedResidualSettings {
		er := l[0]
		cOut := l[1]
		n := l[2]
		stride := l[3]

		for i := 0; i < int(n); i++ {
			s := stride
			if i > 0 {
				s = 1
			}
			path := fp.Sub(fmt.Sprintf("%v", layerId))
			features.Add(inv(path.Sub("conv"), cIn, cOut, s, er))

			cIn = cOut
			layerId += 1
		}
	}

	features.Add(cbr(fp.Sub(fmt.Sprintf("%v", layerId)), cIn, 1280, 1, 1, 1))

	classifier := nn.SeqT()

	classifier.AddFnT(nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		return ts.MustDropout(xs, 0.5, train)
	}))

	classifier.Add(nn.NewLinear(cp.Sub("1"), 1280, nclasses, nn.DefaultLinearConfig()))

	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		tmp1 := xs.ApplyT(features, train)

		tmp2 := tmp1.MustMeanDim([]int64{2}, false, gotch.Float, true)
		tmp3 := tmp2.MustMeanDim([]int64{2}, false, gotch.Float, true)

		res := tmp3.ApplyT(classifier, train)
		tmp3.MustDrop()

		return res
	})

}
