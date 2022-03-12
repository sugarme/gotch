package nn

// A two dimension transposed convolution layer.

import (
	"log"

	"github.com/sugarme/gotch/ts"
)

type ConvTranspose1DConfig struct {
	Stride        []int64
	Padding       []int64
	OutputPadding []int64
	Dilation      []int64
	Groups        int64
	Bias          bool
	WsInit        Init
	BsInit        Init
}

type ConvTranspose2DConfig struct {
	Stride        []int64
	Padding       []int64
	OutputPadding []int64
	Dilation      []int64
	Groups        int64
	Bias          bool
	WsInit        Init
	BsInit        Init
}

type ConvTranspose3DConfig struct {
	Stride        []int64
	Padding       []int64
	OutputPadding []int64
	Dilation      []int64
	Groups        int64
	Bias          bool
	WsInit        Init
	BsInit        Init
}

// DefaultConvConfig create a default 1D ConvConfig
func DefaultConvTranspose1DConfig() *ConvTranspose1DConfig {
	return &ConvTranspose1DConfig{
		Stride:        []int64{1},
		Padding:       []int64{0},
		OutputPadding: []int64{0},
		Dilation:      []int64{1},
		Groups:        1,
		Bias:          true,
		WsInit:        NewKaimingUniformInit(),
		BsInit:        NewConstInit(float64(0.0)),
	}
}

type ConvTranspose1D struct {
	Ws     *ts.Tensor
	Bs     *ts.Tensor // optional
	Config *ConvTranspose1DConfig
}

func NewConvTranspose1D(vs *Path, inDim, outDim int64, ksizes []int64, cfg *ConvTranspose1DConfig) *ConvTranspose1D {
	if len(ksizes) != 1 {
		log.Fatalf("NewConvTranspose1D method call: Kernel size should be 1. Got %v\n", len(ksizes))
	}

	var (
		ws *ts.Tensor
		bs *ts.Tensor = ts.NewTensor()
	)

	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, ksizes...)
	ws = vs.MustNewVar("weight", weightSize, cfg.WsInit)

	if cfg.Bias {
		bs = vs.MustNewVar("bias", []int64{outDim}, cfg.BsInit)
	}

	return &ConvTranspose1D{
		Ws:     ws,
		Bs:     bs,
		Config: cfg,
	}
}

type ConvTranspose2D struct {
	Ws     *ts.Tensor
	Bs     *ts.Tensor // optional
	Config *ConvTranspose2DConfig
}

func NewConvTranspose2D(vs *Path, inDim, outDim int64, ksizes []int64, cfg *ConvTranspose2DConfig) *ConvTranspose2D {

	if len(ksizes) != 2 {
		log.Fatalf("NewConvTranspose2D method call: Kernel size should be 2. Got %v\n", len(ksizes))
	}

	var (
		ws *ts.Tensor
		bs *ts.Tensor = ts.NewTensor()
	)

	if cfg.Bias {
		bs = vs.MustNewVar("bias", []int64{outDim}, cfg.BsInit)
	}
	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, ksizes...)
	ws = vs.MustNewVar("weight", weightSize, cfg.WsInit)

	return &ConvTranspose2D{
		Ws:     ws,
		Bs:     bs,
		Config: cfg,
	}
}

type ConvTranspose3D struct {
	Ws     *ts.Tensor
	Bs     *ts.Tensor // optional
	Config *ConvTranspose3DConfig
}

func NewConvTranspose3D(vs *Path, inDim, outDim int64, ksizes []int64, cfg *ConvTranspose3DConfig) *ConvTranspose3D {
	if len(ksizes) != 3 {
		log.Fatalf("NewConvTranspose3D method call: Kernel size should be 3. Got %v\n", len(ksizes))
	}

	var (
		ws *ts.Tensor
		bs *ts.Tensor = ts.NewTensor()
	)

	if cfg.Bias {
		bs = vs.MustNewVar("bias", []int64{outDim}, cfg.BsInit)
	}
	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, ksizes...)
	ws = vs.MustNewVar("weight", weightSize, cfg.WsInit)

	return &ConvTranspose3D{
		Ws:     ws,
		Bs:     bs,
		Config: cfg,
	}
}

// Implement Module for Conv1D, Conv2D, Conv3D:
// ============================================

func (c *ConvTranspose1D) Forward(xs *ts.Tensor) *ts.Tensor {
	return ts.MustConvTranspose1d(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.OutputPadding, c.Config.Groups, c.Config.Dilation)
}

func (c *ConvTranspose2D) Forward(xs *ts.Tensor) *ts.Tensor {
	return ts.MustConvTranspose2d(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.OutputPadding, c.Config.Groups, c.Config.Dilation)
}
func (c *ConvTranspose3D) Forward(xs *ts.Tensor) *ts.Tensor {
	return ts.MustConvTranspose3d(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.OutputPadding, c.Config.Groups, c.Config.Dilation)
}
