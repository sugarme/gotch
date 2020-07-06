package vision

// InceptionV3

import (
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func convBn(p nn.Path, cIn, cOut, ksize, pad, stride int64) (retVal ts.ModuleT) {

	convConfig := nn.DefaultConv2DConfig()
	convConfig.Stride = []int64{stride, stride}
	convConfig.Padding = []int64{pad, pad}
	convConfig.Bias = false

	bnConfig := nn.DefaultBatchNormConfig()
	bnConfig.Eps = 0.001

	seq := nn.SeqT()

	convP := p.Sub("conv")
	seq.Add(nn.NewConv2D(&convP, cIn, cOut, ksize, convConfig))

	seq.Add(nn.BatchNorm2D(p.Sub("bn"), cOut, bnConfig))

	seq.AddFn(nn.NewFunc(func(xs ts.Tensor) ts.Tensor {
		return xs.MustRelu(false)
	}))

	return seq
}

func convBn2(p nn.Path, cIn, cOut int64, ksize []int64, pad []int64) (retVal ts.ModuleT) {
	convConfig := nn.DefaultConv2DConfig()
	convConfig.Padding = pad
	convConfig.Bias = false

	bnConfig := nn.DefaultBatchNormConfig()
	bnConfig.Eps = 0.001

	seq := nn.SeqT()

	seq.Add(nn.NewConv(p.Sub("conv"), cIn, cOut, ksize, convConfig).(nn.Conv2D))

	seq.Add(nn.BatchNorm2D(p.Sub("bn"), cOut, bnConfig))

	seq.AddFn(nn.NewFunc(func(xs ts.Tensor) ts.Tensor {
		return xs.MustRelu(false)
	}))

	return seq
}

func inMaxPool2D(xs ts.Tensor, ksize, stride int64) (retVal ts.Tensor) {
	return xs.MustMaxPool2D([]int64{ksize, ksize}, []int64{stride, stride}, []int64{0, 0}, []int64{1, 1}, false, false)
}

func inceptionA(p nn.Path, cIn, cPool int64) (retVal ts.ModuleT) {
	b1 := convBn(p.Sub("branch1x1"), cIn, 64, 1, 0, 1)
	b21 := convBn(p.Sub("branch5x5_1"), cIn, 48, 1, 0, 1)
	b22 := convBn(p.Sub("branch5x5_2"), 48, 64, 5, 2, 1)
	b31 := convBn(p.Sub("branch3x3dbl_1"), cIn, 64, 1, 0, 1)
	b32 := convBn(p.Sub("branch3x3dbl_2"), 64, 96, 3, 1, 1)
	b33 := convBn(p.Sub("branch3x3dbl_3"), 96, 96, 3, 1, 1)
	bpool := convBn(p.Sub("branch_pool"), cIn, cPool, 1, 0, 1)

	return nn.NewFuncT(func(xs ts.Tensor, train bool) ts.Tensor {
		b1Ts := xs.ApplyT(b1, train)

		b2Tmp := xs.ApplyT(b21, train)
		b2Ts := b2Tmp.ApplyT(b22, train)
		b2Tmp.MustDrop()

		b3Tmp1 := xs.ApplyT(b31, train)
		b3Tmp2 := b3Tmp1.ApplyT(b32, train)
		b3Tmp1.MustDrop()
		b3Ts := b3Tmp2.ApplyT(b33, train)
		b3Tmp2.MustDrop()

		bpoolTmp := xs.MustAvgPool2D([]int64{3, 3}, []int64{1, 1}, []int64{1, 1}, false, true, 9, false)
		bpoolTs := bpoolTmp.ApplyT(bpool, train)

		return ts.MustCat([]ts.Tensor{b1Ts, b2Ts, b3Ts, bpoolTs}, 1, true)
	})
}

// TODO: continue
