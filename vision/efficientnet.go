package vision

import (
	"fmt"
	"math"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

const (
	batchNormMomentum float64 = 0.99
	batchNormEpsilon  float64 = 1e-3
)

type BlockArgs struct {
	KernelSize   int64
	NumRepeat    int64
	InputFilters int64
	OutputFilter int64
	ExpandRatio  int64
	SeRatio      float64
	Stride       int64
}

func ba(k, r, i, o, er int64, sr float64, s int64) *BlockArgs {
	return &BlockArgs{
		KernelSize:   k,
		NumRepeat:    r,
		InputFilters: i,
		OutputFilter: o,
		ExpandRatio:  er,
		SeRatio:      sr,
		Stride:       s,
	}
}

func blockArgs() (retVal []BlockArgs) {
	return []BlockArgs{
		*ba(3, 1, 32, 16, 1, 0.25, 1),
		*ba(3, 2, 16, 24, 6, 0.25, 2),
		*ba(5, 2, 24, 40, 6, 0.25, 2),
		*ba(3, 3, 40, 80, 6, 0.25, 2),
		*ba(5, 3, 80, 112, 6, 0.25, 1),
		*ba(5, 4, 112, 192, 6, 0.25, 2),
		*ba(3, 1, 192, 320, 6, 0.25, 1),
	}
}

type params struct {
	Width   float64
	Depth   float64
	Res     int64
	Dropout float64
}

func (p *params) roundRepeats(repeats int64) int64 {

	return int64(math.Ceil(p.Depth * float64(repeats)))
}

func (p *params) roundFilters(filters int64) int64 {
	var divisor int64 = 8
	filF := p.Width * float64(filters)
	filI := int64(filF + float64(divisor)/2.0)

	newFilters := int64(math.Max(float64(divisor), float64(filI/divisor*divisor)))

	if float64(newFilters) < (0.9 * filF) {
		newFilters += int64(divisor)
	}

	return newFilters
}

// Conv2D with same padding
func enConv2d(vs *nn.Path, i, o, k int64, c *nn.Conv2DConfig, train bool) ts.ModuleT {
	conv2d := nn.NewConv2D(vs, i, o, k, c)
	s := c.Stride

	return nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		size := xs.MustSize()
		ih := size[2]
		iw := size[3]
		oh := (ih + s[0] - 1) / s[0]
		ow := (iw + s[0] - 1) / s[0]

		var padH int64 = 0
		if (((oh - 1) * s[0]) + k - ih) > 0 {
			padH = ((oh - 1) * s[0]) + k - ih
		}
		var padW int64 = 0
		if (((ow - 1) * s[0]) + k - iw) > 0 {
			padW = ((ow - 1) * s[0]) + k - iw
		}

		var res *ts.Tensor
		if padW > 0 || padH > 0 {
			zeroP2D := xs.MustZeroPad2d(padW/2, padW-padW/2, padH/2, padH-padH/2, false)
			res = zeroP2D.ApplyT(conv2d, train)
			zeroP2D.MustDrop()
			return res
		} else {
			res = xs.ApplyT(conv2d, train)
			return res
		}
	})
}

func newParams(width, depth float64, res int64, dropout float64) *params {
	return &params{
		width,
		depth,
		res,
		dropout,
	}
}

func b0() *params {
	return newParams(1.0, 1.0, 224, 0.2)
}

func b1() *params {
	return newParams(1.0, 1.1, 240, 0.2)
}

func b2() *params {
	return newParams(1.1, 1.2, 260, 0.3)
}

func b3() *params {
	return newParams(1.2, 1.4, 300, 0.3)
}

func b4() *params {
	return newParams(1.4, 1.8, 380, 0.4)
}

func b5() *params {
	return newParams(1.6, 2.2, 456, 0.4)
}

func b6() *params {
	return newParams(1.8, 2.6, 528, 0.5)
}

func b7() *params {
	return newParams(2.0, 3.1, 600, 0.5)
}

func block(p *nn.Path, args BlockArgs) ts.ModuleT {

	inp := args.InputFilters
	oup := args.InputFilters * args.ExpandRatio
	finalOup := args.OutputFilter

	bn2d := nn.DefaultBatchNormConfig()
	bn2d.Momentum = 1.0 - batchNormMomentum
	bn2d.Eps = batchNormEpsilon

	convConfigNoBias := nn.DefaultConv2DConfig()
	convConfigNoBias.Bias = false

	depthwiseConvConfig := nn.DefaultConv2DConfig()
	depthwiseConvConfig.Stride = []int64{args.Stride, args.Stride}
	depthwiseConvConfig.Groups = oup
	depthwiseConvConfig.Bias = false

	expansion := nn.SeqT()
	if args.ExpandRatio != 1 {
		expansion.Add(enConv2d(p.Sub("_expand_conv"), inp, oup, 1, convConfigNoBias, false))
		expansion.Add(nn.BatchNorm2D(p.Sub("_bn0"), oup, bn2d))
		expansion.AddFn(nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
			return xs.Swish()
		}))
	}

	depthwiseConv := enConv2d(p.Sub("_depthwise_conv"), oup, oup, args.KernelSize, depthwiseConvConfig, false)
	depthwiseBn := nn.BatchNorm2D(p.Sub("_bn1"), oup, bn2d)

	// NOTE: args.SeRatio is optional float64. Default = 0
	var se *nn.SequentialT // se will be nil if args.SeRatio == 0
	if args.SeRatio > 0 {
		var nsc int64 = 1
		if (float64(inp) * args.SeRatio) > 1 {
			nsc = int64(float64(inp) * args.SeRatio)
		}

		se = nn.SeqT()
		se.Add(enConv2d(p.Sub("_se_reduce"), oup, nsc, 1, nn.DefaultConv2DConfig(), false))

		se.AddFn(nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
			return xs.Swish()
		}))

		se.Add(enConv2d(p.Sub("_se_expand"), nsc, oup, 1, nn.DefaultConv2DConfig(), false))
	}

	projectConv := enConv2d(p.Sub("_project_conv"), oup, finalOup, 1, convConfigNoBias, false)

	projectBn := nn.BatchNorm2D(p.Sub("_bn2"), finalOup, bn2d)

	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		var ys *ts.Tensor
		if args.ExpandRatio == 1 {
			ys = xs.MustShallowClone()
		} else {
			ys = xs.ApplyT(expansion, train)
		}

		ys1 := ys.ApplyT(depthwiseConv, false)
		ys.MustDrop()
		ys2 := ys1.ApplyT(depthwiseBn, train)
		ys1.MustDrop()
		ys3 := ys2.Swish()
		ys2.MustDrop()

		var ys4 *ts.Tensor
		// NOTE: args.SeRatio is optional value.
		if args.SeRatio == 0 {
			ys4 = ys3.MustShallowClone()
			ys3.MustDrop()
		} else {
			tmp1 := ys3.MustAdaptiveAvgPool2d([]int64{1, 1}, false)
			tmp2 := tmp1.ApplyT(se, train)
			tmp1.MustDrop()
			tmp3 := tmp2.MustSigmoid(true)
			ys4 = ys3.MustMul(tmp3, true)
			tmp3.MustDrop()
		}

		ys5 := ys4.ApplyT(projectConv, false)
		ys4.MustDrop()
		ys6 := ys5.ApplyT(projectBn, train)
		ys5.MustDrop()

		if args.Stride == 1 && inp == finalOup {
			return ys6.MustAdd(xs, true)
		} else {
			return ys6
		}
	})
}

func efficientnet(p *nn.Path, params *params, nclasses int64) ts.ModuleT {

	args := blockArgs()

	bn2dConfig := nn.DefaultBatchNormConfig()
	bn2dConfig.Momentum = 1.0 - batchNormMomentum
	bn2dConfig.Eps = batchNormEpsilon

	convConfigNoBias := nn.DefaultConv2DConfig()
	convConfigNoBias.Bias = false

	convS2Config := nn.DefaultConv2DConfig()
	convS2Config.Stride = []int64{2, 2}
	convS2Config.Bias = false

	outC := params.roundFilters(32)
	convStem := enConv2d(p.Sub("_conv_stem"), 3, outC, 3, convS2Config, false)
	bn0 := nn.BatchNorm2D(p.Sub("_bn0"), outC, bn2dConfig)

	blocks := nn.SeqT()
	blockP := p.Sub("_blocks")
	blockIdx := 0
	for _, arg := range args {
		a1 := arg
		a1.InputFilters = params.roundFilters(arg.InputFilters)
		a1.OutputFilter = params.roundFilters(arg.OutputFilter)

		blocks.Add(block(blockP.Sub(fmt.Sprintf("%v", blockIdx)), a1))
		blockIdx += 1

		a2 := a1
		a2.InputFilters = a1.OutputFilter
		a2.Stride = 1

		for i := 1; i < int(params.roundRepeats(a2.NumRepeat)); i++ {
			blocks.Add(block(blockP.Sub(fmt.Sprintf("%v", blockIdx)), a2))
			blockIdx += 1
		}
	}

	lastArg := args[len(args)-1]
	inChannels := params.roundFilters(lastArg.OutputFilter)
	outC = params.roundFilters(1280)

	convHead := enConv2d(p.Sub("_conv_head"), inChannels, outC, 1, convConfigNoBias, false)
	bn1 := nn.BatchNorm2D(p.Sub("_bn1"), outC, bn2dConfig)

	classifier := nn.SeqT()

	classifier.AddFnT(nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		return ts.MustDropout(xs, 0.2, train)
	}))

	classifier.Add(nn.NewLinear(p.Sub("_fc"), outC, nclasses, nn.DefaultLinearConfig()))

	return nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		tmp1 := xs.ApplyT(convStem, false)
		tmp2 := tmp1.ApplyT(bn0, train)
		tmp1.MustDrop()
		tmp3 := tmp2.Swish()
		tmp2.MustDrop()
		tmp4 := tmp3.ApplyT(blocks, train)
		tmp3.MustDrop()
		tmp5 := tmp4.ApplyT(convHead, false)
		tmp4.MustDrop()
		tmp6 := tmp5.ApplyT(bn1, train)
		tmp5.MustDrop()
		tmp7 := tmp6.Swish()
		tmp6.MustDrop()
		tmp8 := tmp7.MustAdaptiveAvgPool2d([]int64{1, 1}, false)
		tmp7.MustDrop()
		tmp9 := tmp8.MustSqueeze1(-1, true)
		tmp10 := tmp9.MustSqueeze1(-1, true)

		res := tmp10.ApplyT(classifier, train)
		tmp10.MustDrop()
		return res
	})

}

func EfficientNetB0(p *nn.Path, nclasses int64) ts.ModuleT {
	return efficientnet(p, b0(), nclasses)
}

func EfficientNetB1(p *nn.Path, nclasses int64) ts.ModuleT {
	return efficientnet(p, b1(), nclasses)
}

func EfficientNetB2(p *nn.Path, nclasses int64) ts.ModuleT {
	return efficientnet(p, b2(), nclasses)
}

func EfficientNetB3(p *nn.Path, nclasses int64) ts.ModuleT {
	return efficientnet(p, b3(), nclasses)
}

func EfficientNetB4(p *nn.Path, nclasses int64) ts.ModuleT {
	return efficientnet(p, b4(), nclasses)
}

func EfficientNetB5(p *nn.Path, nclasses int64) ts.ModuleT {
	return efficientnet(p, b5(), nclasses)
}

func EfficientNetB6(p *nn.Path, nclasses int64) ts.ModuleT {
	return efficientnet(p, b6(), nclasses)
}

func EfficientNetB7(p *nn.Path, nclasses int64) ts.ModuleT {
	return efficientnet(p, b7(), nclasses)
}
