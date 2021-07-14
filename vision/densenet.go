package vision

// DenseNet implementation.
//
// See "Densely Connected Convolutional Networks", Huang et al 2016.
// https://arxiv.org/abs/1608.06993

import (
	"fmt"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func dnConv2d(p *nn.Path, cIn, cOut, ksize, padding, stride int64) *nn.Conv2D {
	config := nn.DefaultConv2DConfig()
	config.Stride = []int64{stride, stride}
	config.Padding = []int64{padding, padding}
	config.Bias = false

	return nn.NewConv2D(p, cIn, cOut, ksize, config)
}

type denseLayer struct {
	Conv1 *nn.Conv2D
	Bn1   *nn.BatchNorm
	Conv2 *nn.Conv2D
	Bn2   *nn.BatchNorm
}

func (l *denseLayer) ForwardT(xs *ts.Tensor, train bool) *ts.Tensor {
	ys1 := xs.ApplyT(l.Bn1, train)
	ys2 := ys1.MustRelu(true)
	ys3 := ys2.Apply(l.Conv1)
	ys2.MustDrop()
	ys4 := ys3.ApplyT(l.Bn2, train)
	ys3.MustDrop()
	ys5 := ys4.MustRelu(true)
	ys := ys5.Apply(l.Conv2)
	ys5.MustDrop()

	res := ts.MustCat([]ts.Tensor{*xs, *ys}, 1)
	ys.MustDrop()

	return res
}

func newDenseLayer(p *nn.Path, cIn, bnSize, growth int64) ts.ModuleT {
	cInter := bnSize * growth
	bn1 := nn.BatchNorm2D(p.Sub("norm1"), cIn, nn.DefaultBatchNormConfig())
	conv1 := dnConv2d(p.Sub("conv1"), cIn, cInter, 1, 0, 1)
	bn2 := nn.BatchNorm2D(p.Sub("norm2"), cInter, nn.DefaultBatchNormConfig())
	conv2 := dnConv2d(p.Sub("conv2"), cInter, growth, 3, 1, 1)

	return &denseLayer{
		Bn1:   bn1,
		Conv1: conv1,
		Bn2:   bn2,
		Conv2: conv2,
	}
}

func denseBlock(p *nn.Path, cIn, bnSize, growth, nlayers int64) ts.ModuleT {
	seq := nn.SeqT()
	for i := 0; i < int(nlayers); i++ {
		seq.Add(newDenseLayer(p.Sub(fmt.Sprintf("denselayer%v", 1+i)), cIn+(int64(i)*growth), bnSize, growth))
	}

	return seq
}

func transition(p *nn.Path, cIn, cOut int64) ts.ModuleT {
	seq := nn.SeqT()

	seq.Add(nn.BatchNorm2D(p.Sub("norm"), cIn, nn.DefaultBatchNormConfig()))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	seq.Add(dnConv2d(p.Sub("conv"), cIn, cOut, 1, 0, 1))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.AvgPool2DDefault(2, false)
	}))

	return seq
}

func densenet(p *nn.Path, cIn, bnSize int64, growth int64, blockConfig []int64, cOut int64) ts.ModuleT {
	fp := p.Sub("features")
	seq := nn.SeqT()

	seq.Add(dnConv2d(fp.Sub("conv0"), 3, cIn, 7, 3, 2))

	seq.Add(nn.BatchNorm2D(fp.Sub("norm0"), cIn, nn.DefaultBatchNormConfig()))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		tmp := xs.MustRelu(false)
		return tmp.MustMaxPool2d([]int64{3, 3}, []int64{2, 2}, []int64{1, 1}, []int64{1, 1}, false, true)
	}))

	nfeat := cIn

	for i, nlayers := range blockConfig {
		seq.Add(denseBlock(fp.Sub(fmt.Sprintf("denseblock%v", 1+i)), nfeat, bnSize, growth, nlayers))

		nfeat += nlayers * growth

		if i+1 != len(blockConfig) {
			seq.Add(transition(fp.Sub(fmt.Sprintf("transition%v", 1+i)), nfeat, nfeat/2))
			nfeat = nfeat / 2
		}
	}

	seq.Add(nn.BatchNorm2D(fp.Sub("norm5"), nfeat, nn.DefaultBatchNormConfig()))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		tmp1 := xs.MustRelu(false)
		tmp2 := tmp1.MustAvgPool2d([]int64{7, 7}, []int64{1, 1}, []int64{0, 0}, false, true, []int64{1}, true)
		res := tmp2.FlatView()
		tmp2.MustDrop()
		return res
	}))

	seq.Add(nn.NewLinear(p.Sub("classifier"), nfeat, cOut, nn.DefaultLinearConfig()))

	return seq
}

func DenseNet121(p *nn.Path, nclasses int64) ts.ModuleT {
	// path, cIn, bnSize, growth, blockConfig, cOut
	return densenet(p, 64, 4, 32, []int64{6, 12, 24, 16}, nclasses)
}

func DenseNet161(p *nn.Path, nclasses int64) ts.ModuleT {
	return densenet(p, 96, 4, 48, []int64{6, 12, 36, 24}, nclasses)
}

func DenseNet169(p *nn.Path, nclasses int64) ts.ModuleT {
	return densenet(p, 64, 4, 32, []int64{6, 12, 32, 32}, nclasses)
}

func DenseNet201(p *nn.Path, nclasses int64) ts.ModuleT {
	return densenet(p, 64, 4, 32, []int64{6, 12, 48, 32}, nclasses)
}
