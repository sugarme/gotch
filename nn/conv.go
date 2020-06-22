package nn

// N-dimensional convolution layers.

import (
	ts "github.com/sugarme/gotch/tensor"
)

type Conv1DConfig struct {
	Kval     int64
	Stride   []int64
	Padding  []int64
	Dilation []int64
	Groups   int64
	Bias     bool
	WsInit   Init
	BsInit   Init
}

type Conv2DConfig struct {
	Kval     int64
	Stride   []int64
	Padding  []int64
	Dilation []int64
	Groups   int64
	Bias     bool
	WsInit   Init
	BsInit   Init
}

type Conv3DConfig struct {
	Kval     int64
	Stride   []int64
	Padding  []int64
	Dilation []int64
	Groups   int64
	Bias     bool
	WsInit   Init
	BsInit   Init
}

// DefaultConvConfig create a default 1D ConvConfig
func DefaultConv1DConfig() Conv1DConfig {
	return Conv1DConfig{
		Stride:   []int64{1},
		Padding:  []int64{0},
		Dilation: []int64{1},
		Groups:   1,
		Bias:     true,
		WsInit:   NewKaimingUniformInit(),
		BsInit:   NewConstInit(float64(0.0)),
	}
}

// DefaultConvConfig2D creates a default 2D ConvConfig
func DefaultConv2DConfig() Conv2DConfig {
	return Conv2DConfig{
		Stride:   []int64{1, 1},
		Padding:  []int64{0, 0},
		Dilation: []int64{1, 1},
		Groups:   1,
		Bias:     true,
		WsInit:   NewKaimingUniformInit(),
		BsInit:   NewConstInit(float64(0.0)),
	}
}

type Conv1D struct {
	Ws     ts.Tensor
	Bs     ts.Tensor // optional
	Config Conv1DConfig
}

func NewConv1D(vs *Path, inDim, outDim int64, cfg Conv1DConfig) Conv1D {
	var conv Conv1D
	conv.Config = cfg
	if cfg.Bias {
		conv.Bs = vs.NewVar("bias", []int64{outDim}, cfg.BsInit)
	}
	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, cfg.Kval)
	conv.Ws = vs.NewVar("weight", weightSize, cfg.WsInit)

	return conv
}

type Conv2D struct {
	Ws     ts.Tensor
	Bs     ts.Tensor // optional
	Config Conv2DConfig
}

func NewConv2D(vs *Path, inDim, outDim int64, cfg Conv2DConfig) Conv2D {
	var conv Conv2D
	conv.Config = cfg
	if cfg.Bias {
		conv.Bs = vs.NewVar("bias", []int64{outDim}, cfg.BsInit)
	}
	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, cfg.Kval, cfg.Kval)
	conv.Ws = vs.NewVar("weight", weightSize, cfg.WsInit)

	return conv
}

type Conv3D struct {
	Ws     ts.Tensor
	Bs     ts.Tensor // optional
	Config Conv3DConfig
}

func NewConv3D(vs *Path, inDim, outDim int64, cfg Conv3DConfig) Conv3D {
	var conv Conv3D
	conv.Config = cfg
	if cfg.Bias {
		conv.Bs = vs.NewVar("bias", []int64{outDim}, cfg.BsInit)
	}
	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, cfg.Kval, cfg.Kval, cfg.Kval)
	conv.Ws = vs.NewVar("weight", weightSize, cfg.WsInit)

	return conv
}

// Implement Module for Conv1D, Conv2D, Conv3D:
// ============================================

func (c Conv1D) Forward(xs ts.Tensor) ts.Tensor {
	return ts.MustConv1D(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}

func (c Conv2D) Forward(xs ts.Tensor) ts.Tensor {
	return ts.MustConv2D(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}
func (c Conv3D) Forward(xs ts.Tensor) ts.Tensor {
	return ts.MustConv3D(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}
