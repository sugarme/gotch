package nn

// N-dimensional convolution layers.

import (
	"fmt"
	"reflect"

	ts "github.com/sugarme/gotch/tensor"
)

type Conv1DConfig struct {
	Stride   []int64
	Padding  []int64
	Dilation []int64
	Groups   int64
	Bias     bool
	WsInit   Init
	BsInit   Init
}

type Conv2DConfig struct {
	Stride   []int64
	Padding  []int64
	Dilation []int64
	Groups   int64
	Bias     bool
	WsInit   Init
	BsInit   Init
}

type Conv3DConfig struct {
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

func NewConv1D(vs *Path, inDim, outDim, k int64, cfg Conv1DConfig) Conv1D {
	var conv Conv1D
	conv.Config = cfg
	if cfg.Bias {
		conv.Bs = vs.NewVar("bias", []int64{outDim}, cfg.BsInit)
	}
	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, k)
	conv.Ws = vs.NewVar("weight", weightSize, cfg.WsInit)

	return conv
}

type Conv2D struct {
	Ws     ts.Tensor
	Bs     ts.Tensor // optional
	Config Conv2DConfig
}

func NewConv2D(vs Path, inDim, outDim int64, k int64, cfg Conv2DConfig) Conv2D {
	var conv Conv2D
	conv.Config = cfg
	if cfg.Bias {
		conv.Bs = vs.NewVar("bias", []int64{outDim}, cfg.BsInit)
	}
	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, k, k)
	conv.Ws = vs.NewVar("weight", weightSize, cfg.WsInit)

	return conv
}

type Conv3D struct {
	Ws     ts.Tensor
	Bs     ts.Tensor // optional
	Config Conv3DConfig
}

func NewConv3D(vs *Path, inDim, outDim, k int64, cfg Conv3DConfig) Conv3D {
	var conv Conv3D
	conv.Config = cfg
	if cfg.Bias {
		conv.Bs = vs.NewVar("bias", []int64{outDim}, cfg.BsInit)
	}
	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, k, k, k)
	conv.Ws = vs.NewVar("weight", weightSize, cfg.WsInit)

	return conv
}

type Conv interface{}

// func buildConvConfig(ksizes []int64, groups int64, bias bool, ws Init, bs Init) interface{} {
func buildConvConfig(ksizes []int64) interface{} {
	// Default values
	groups := int64(1)
	bias := true
	ws := NewKaimingUniformInit()
	bs := NewConstInit(0.0)

	switch len(ksizes) {
	case 1:
		return Conv1DConfig{
			Stride:   ksizes,
			Padding:  ksizes,
			Dilation: ksizes,
			Groups:   groups,
			Bias:     bias,
			WsInit:   ws,
			BsInit:   bs,
		}
	case 2:
		return Conv2DConfig{
			Stride:   ksizes,
			Padding:  ksizes,
			Dilation: ksizes,
			Groups:   groups,
			Bias:     bias,
			WsInit:   ws,
			BsInit:   bs,
		}
	case 3:
		return Conv3DConfig{
			Stride:   ksizes,
			Padding:  ksizes,
			Dilation: ksizes,
			Groups:   groups,
			Bias:     bias,
			WsInit:   ws,
			BsInit:   bs,
		}

	default:
		err := fmt.Errorf("Expected nd length from 1 to 3. Got %v\n", len(ksizes))
		panic(err)
	}
}

// NewConv is a generic builder to build Conv1D, Conv2D, Conv3D. It returns
// an interface Conv which might need a type assertion for further use.
func NewConv(vs Path, inDim, outDim int64, ksizes []int64, config interface{}) Conv {

	configT := reflect.TypeOf(config)

	switch {
	case len(ksizes) == 1 && configT.Name() == "Conv1DConfig":
		var conv Conv1D
		conv.Config = config.(Conv1DConfig)
		if config.(Conv1DConfig).Bias {
			conv.Bs = vs.NewVar("bias", []int64{outDim}, config.(Conv1DConfig).BsInit)
		}
		weightSize := []int64{outDim, int64(inDim / config.(Conv1DConfig).Groups)}
		weightSize = append(weightSize, ksizes...)
		conv.Ws = vs.NewVar("weight", weightSize, config.(Conv1DConfig).WsInit)
		return conv
	case len(ksizes) == 2 && configT.Name() == "Conv2DConfig":
		var conv Conv2D
		conv.Config = config.(Conv2DConfig)
		if config.(Conv2DConfig).Bias {
			conv.Bs = vs.NewVar("bias", []int64{outDim}, config.(Conv2DConfig).BsInit)
		}
		weightSize := []int64{outDim, int64(inDim / config.(Conv2DConfig).Groups)}
		weightSize = append(weightSize, ksizes...)
		conv.Ws = vs.NewVar("weight", weightSize, config.(Conv2DConfig).WsInit)
		return conv
	case len(ksizes) == 3 && configT.Name() == "Conv3DConfig":
		var conv Conv3D
		conv.Config = config.(Conv3DConfig)
		if config.(Conv3DConfig).Bias {
			conv.Bs = vs.NewVar("bias", []int64{outDim}, config.(Conv3DConfig).BsInit)
		}
		weightSize := []int64{outDim, int64(inDim / config.(Conv3DConfig).Groups)}
		weightSize = append(weightSize, ksizes...)
		conv.Ws = vs.NewVar("weight", weightSize, config.(Conv3DConfig).WsInit)
		return conv
	default:
		err := fmt.Errorf("Expected nd length from 1 to 3. Got %v\n", len(ksizes))
		panic(err)
	}
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

// Implement ModuleT for Conv1D, Conv2D, Conv3D:
// ============================================

// NOTE: `train` param won't be used, will be?

func (c Conv1D) ForwardT(xs ts.Tensor, train bool) ts.Tensor {
	return ts.MustConv1D(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}

func (c Conv2D) ForwardT(xs ts.Tensor, train bool) ts.Tensor {
	return ts.MustConv2D(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}
func (c Conv3D) ForwardT(xs ts.Tensor, train bool) ts.Tensor {
	return ts.MustConv3D(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}
