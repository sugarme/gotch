package vision

// InceptionV3

import (
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func convBn(p *nn.Path, cIn, cOut, ksize, pad, stride int64) ts.ModuleT {

	convConfig := nn.DefaultConv2DConfig()
	convConfig.Stride = []int64{stride, stride}
	convConfig.Padding = []int64{pad, pad}
	convConfig.Bias = false

	bnConfig := nn.DefaultBatchNormConfig()
	bnConfig.Eps = 0.001

	seq := nn.SeqT()

	convP := p.Sub("conv")
	seq.Add(nn.NewConv2D(convP, cIn, cOut, ksize, convConfig))

	seq.Add(nn.BatchNorm2D(p.Sub("bn"), cOut, bnConfig))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	return seq
}

func convBn2(p *nn.Path, cIn, cOut int64, ksize []int64, pad []int64) ts.ModuleT {
	convConfig := nn.DefaultConv2DConfig()
	convConfig.Padding = pad
	convConfig.Bias = false

	bnConfig := nn.DefaultBatchNormConfig()
	bnConfig.Eps = 0.001

	seq := nn.SeqT()

	seq.Add(nn.NewConv(p.Sub("conv"), cIn, cOut, ksize, convConfig).(*nn.Conv2D))

	seq.Add(nn.BatchNorm2D(p.Sub("bn"), cOut, bnConfig))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	return seq
}

func inMaxPool2D(xs *ts.Tensor, ksize, stride int64) *ts.Tensor {
	return xs.MustMaxPool2d([]int64{ksize, ksize}, []int64{stride, stride}, []int64{0, 0}, []int64{1, 1}, false, false)
}

func inceptionA(p *nn.Path, cIn, cPool int64) ts.ModuleT {
	b1 := convBn(p.Sub("branch1x1"), cIn, 64, 1, 0, 1)
	b21 := convBn(p.Sub("branch5x5_1"), cIn, 48, 1, 0, 1)
	b22 := convBn(p.Sub("branch5x5_2"), 48, 64, 5, 2, 1)
	b31 := convBn(p.Sub("branch3x3dbl_1"), cIn, 64, 1, 0, 1)
	b32 := convBn(p.Sub("branch3x3dbl_2"), 64, 96, 3, 1, 1)
	b33 := convBn(p.Sub("branch3x3dbl_3"), 96, 96, 3, 1, 1)
	bpool := convBn(p.Sub("branch_pool"), cIn, cPool, 1, 0, 1)

	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		b1Ts := xs.ApplyT(b1, train)

		b2Tmp := xs.ApplyT(b21, train)
		b2Ts := b2Tmp.ApplyT(b22, train)
		b2Tmp.MustDrop()

		b3Tmp1 := xs.ApplyT(b31, train)
		b3Tmp2 := b3Tmp1.ApplyT(b32, train)
		b3Tmp1.MustDrop()
		b3Ts := b3Tmp2.ApplyT(b33, train)
		b3Tmp2.MustDrop()

		bpoolTmp := xs.MustAvgPool2d([]int64{3, 3}, []int64{1, 1}, []int64{1, 1}, false, true, []int64{9}, false)
		bpoolTs := bpoolTmp.ApplyT(bpool, train)

		res := ts.MustCat([]ts.Tensor{*b1Ts, *b2Ts, *b3Ts, *bpoolTs}, 1)

		return res
	})
}

func inceptionB(p *nn.Path, cIn int64) ts.ModuleT {
	b1 := convBn(p.Sub("branch3x3"), cIn, 384, 3, 0, 2)
	b21 := convBn(p.Sub("branch3x3dbl_1"), cIn, 64, 1, 0, 1)
	b22 := convBn(p.Sub("branch3x3dbl_2"), 64, 96, 3, 1, 1)
	b23 := convBn(p.Sub("branch3x3dbl_3"), 96, 96, 3, 0, 2)

	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		b1Ts := xs.ApplyT(b1, train)

		b2Tmp1 := xs.ApplyT(b21, train)
		b2Tmp2 := b2Tmp1.ApplyT(b22, train)
		b2Tmp1.MustDrop()
		b2Ts := b2Tmp2.ApplyT(b23, train)
		b2Tmp2.MustDrop()

		bpoolTs := inMaxPool2D(xs, 3, 2)

		res := ts.MustCat([]ts.Tensor{*b1Ts, *b2Ts, *bpoolTs}, 1)

		return res
	})
}

func inceptionC(p *nn.Path, cIn int64, c7 int64) ts.ModuleT {

	b1 := convBn(p.Sub("branch1x1"), cIn, 192, 1, 0, 1)

	b21 := convBn(p.Sub("branch7x7_1"), cIn, c7, 1, 0, 1)
	b22 := convBn2(p.Sub("branch7x7_2"), c7, c7, []int64{1, 7}, []int64{0, 3})
	b23 := convBn2(p.Sub("branch7x7_3"), c7, 192, []int64{7, 1}, []int64{3, 0})

	b31 := convBn(p.Sub("branch7x7dbl_1"), cIn, c7, 1, 0, 1)
	b32 := convBn2(p.Sub("branch7x7dbl_2"), c7, c7, []int64{7, 1}, []int64{3, 0})
	b33 := convBn2(p.Sub("branch7x7dbl_3"), c7, c7, []int64{1, 7}, []int64{0, 3})
	b34 := convBn2(p.Sub("branch7x7dbl_4"), c7, c7, []int64{7, 1}, []int64{3, 0})
	b35 := convBn2(p.Sub("branch7x7dbl_5"), c7, 192, []int64{1, 7}, []int64{0, 3})

	bpool := convBn(p.Sub("branch_pool"), cIn, 192, 1, 0, 1)

	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		b1Ts := xs.ApplyT(b1, train)

		b2Tmp1 := xs.ApplyT(b21, train)
		b2Tmp2 := b2Tmp1.ApplyT(b22, train)
		b2Tmp1.MustDrop()
		b2Ts := b2Tmp2.ApplyT(b23, train)
		b2Tmp2.MustDrop()

		b3Tmp1 := xs.ApplyT(b31, train)
		b3Tmp2 := b3Tmp1.ApplyT(b32, train)
		b3Tmp1.MustDrop()
		b3Tmp3 := b3Tmp2.ApplyT(b33, train)
		b3Tmp2.MustDrop()
		b3Tmp4 := b3Tmp3.ApplyT(b34, train)
		b3Tmp3.MustDrop()
		b3Ts := b3Tmp4.ApplyT(b35, train)
		b3Tmp4.MustDrop()

		bpTmp1 := xs.MustAvgPool2d([]int64{3, 3}, []int64{1, 1}, []int64{1, 1}, false, true, []int64{9}, false)
		bpoolTs := bpTmp1.ApplyT(bpool, train)

		return ts.MustCat([]ts.Tensor{*b1Ts, *b2Ts, *b3Ts, *bpoolTs}, 1)
	})
}

func inceptionD(p *nn.Path, cIn int64) ts.ModuleT {

	b11 := convBn(p.Sub("branch3x3_1"), cIn, 192, 1, 0, 1)
	b12 := convBn(p.Sub("branch3x3_2"), 192, 320, 3, 0, 2)

	b21 := convBn(p.Sub("branch7x7x3_1"), cIn, 192, 1, 0, 1)
	b22 := convBn2(p.Sub("branch7x7x3_2"), 192, 192, []int64{1, 7}, []int64{0, 3})
	b23 := convBn2(p.Sub("branch7x7x3_3"), 192, 192, []int64{7, 1}, []int64{3, 0})
	b24 := convBn(p.Sub("branch7x7x3_4"), 192, 192, 3, 0, 2)

	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		b1Tmp := xs.ApplyT(b11, train)
		b1Ts := b1Tmp.ApplyT(b12, train)
		b1Tmp.MustDrop()

		b2Tmp1 := xs.ApplyT(b21, train)
		b2Tmp2 := b2Tmp1.ApplyT(b22, train)
		b2Tmp1.MustDrop()
		b2Tmp3 := b2Tmp2.ApplyT(b23, train)
		b2Tmp2.MustDrop()
		b2Ts := b2Tmp3.ApplyT(b24, train)
		b2Tmp3.MustDrop()

		bpoolTs := inMaxPool2D(xs, 3, 2)

		return ts.MustCat([]ts.Tensor{*b1Ts, *b2Ts, *bpoolTs}, 1)

	})
}

func inceptionE(p *nn.Path, cIn int64) ts.ModuleT {
	b1 := convBn(p.Sub("branch1x1"), cIn, 320, 1, 0, 1)

	b21 := convBn(p.Sub("branch3x3_1"), cIn, 384, 1, 0, 1)
	b22a := convBn2(p.Sub("branch3x3_2a"), 384, 384, []int64{1, 3}, []int64{0, 1})
	b22b := convBn2(p.Sub("branch3x3_2b"), 384, 384, []int64{3, 1}, []int64{1, 0})

	b31 := convBn(p.Sub("branch3x3dbl_1"), cIn, 448, 1, 0, 1)
	b32 := convBn(p.Sub("branch3x3dbl_2"), 448, 384, 3, 1, 1)
	b33a := convBn2(p.Sub("branch3x3dbl_3a"), 384, 384, []int64{1, 3}, []int64{0, 1})
	b33b := convBn2(p.Sub("branch3x3dbl_3b"), 384, 384, []int64{3, 1}, []int64{1, 0})

	bpool := convBn(p.Sub("branch_pool"), cIn, 192, 1, 0, 1)

	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		b1Ts := xs.ApplyT(b1, train)

		b2Tmp := xs.ApplyT(b21, train)
		b2aTs := b2Tmp.ApplyT(b22a, train)
		b2bTs := b2Tmp.ApplyT(b22b, train)
		b2Ts := ts.MustCat([]ts.Tensor{*b2aTs, *b2bTs}, 1)

		b3Tmp1 := xs.ApplyT(b31, train)
		b3Tmp2 := b3Tmp1.ApplyT(b32, train)
		b3Tmp1.MustDrop()
		b3aTs := b3Tmp2.ApplyT(b33a, train)
		b3bTs := b3Tmp2.ApplyT(b33b, train)
		b3Ts := ts.MustCat([]ts.Tensor{*b3aTs, *b3bTs}, 1)

		bpTmp1 := xs.MustAvgPool2d([]int64{3, 3}, []int64{1, 1}, []int64{1, 1}, false, true, []int64{9}, false)
		bpoolTs := bpTmp1.ApplyT(bpool, train)

		return ts.MustCat([]ts.Tensor{*b1Ts, *b2Ts, *b3Ts, *bpoolTs}, 1)
	})

}

func InceptionV3(p *nn.Path, nclasses int64) ts.ModuleT {
	seq := nn.SeqT()

	seq.Add(convBn(p.Sub("Conv2d_1a_3x3"), 3, 32, 3, 0, 2))
	seq.Add(convBn(p.Sub("Conv2d_2a_3x3"), 32, 32, 3, 0, 1))
	seq.Add(convBn(p.Sub("Conv2d_2b_3x3"), 32, 64, 3, 1, 1))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		tmp := xs.MustRelu(false)
		res := inMaxPool2D(tmp, 3, 2)
		tmp.MustDrop()
		return res
	}))

	seq.Add(convBn(p.Sub("Conv2d_3b_1x1"), 64, 80, 1, 0, 1))
	seq.Add(convBn(p.Sub("Conv2d_4a_3x3"), 80, 192, 3, 0, 1))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		tmp := xs.MustRelu(false)
		res := inMaxPool2D(tmp, 3, 2)
		tmp.MustDrop()
		return res
	}))

	seq.Add(inceptionA(p.Sub("Mixed_5b"), 192, 32))
	seq.Add(inceptionA(p.Sub("Mixed_5c"), 256, 64))
	seq.Add(inceptionA(p.Sub("Mixed_5d"), 288, 64))

	seq.Add(inceptionB(p.Sub("Mixed_6a"), 288))

	seq.Add(inceptionC(p.Sub("Mixed_6b"), 768, 128))
	seq.Add(inceptionC(p.Sub("Mixed_6c"), 768, 160))
	seq.Add(inceptionC(p.Sub("Mixed_6d"), 768, 160))
	seq.Add(inceptionC(p.Sub("Mixed_6e"), 768, 192))

	seq.Add(inceptionD(p.Sub("Mixed_7a"), 768))

	seq.Add(inceptionE(p.Sub("Mixed_7b"), 1280))
	seq.Add(inceptionE(p.Sub("Mixed_7c"), 2048))

	seq.AddFnT(nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		tmp1 := xs.MustAdaptiveAvgPool2d([]int64{1, 1}, false)
		tmp2 := ts.MustDropout(tmp1, 0.5, train)
		tmp1.MustDrop()
		res := tmp2.FlatView()
		return res
	}))

	seq.Add(nn.NewLinear(p.Sub("fc"), 2048, nclasses, nn.DefaultLinearConfig()))

	return seq
}
