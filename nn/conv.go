package nn

// N-dimensional convolution layers.

import (
	"fmt"
	"reflect"

	"github.com/sugarme/gotch/ts"
)

// Conv1DConfig:
// ============

// Conv1DConfig is configuration struct for convolution 1D.
type Conv1DConfig struct {
	Stride   []int64
	Padding  []int64
	Dilation []int64
	Groups   int64
	Bias     bool
	WsInit   Init
	BsInit   Init
}

// Conv1DConfigOpt is option for Conv1DConfig.
type Conv1DConfigOpt func(*Conv1DConfig)

// withStride1D adds stride 1D option.
func WithStride1D(val int64) Conv1DConfigOpt {
	return func(cfg *Conv1DConfig) {
		cfg.Stride = []int64{val}
	}
}

// WithPadding1D adds padding 1D option.
func WithPadding1D(val int64) Conv1DConfigOpt {
	return func(cfg *Conv1DConfig) {
		cfg.Padding = []int64{val}
	}
}

// WithDilation1D adds dilation 1D option.
func WithDilation1D(val int64) Conv1DConfigOpt {
	return func(cfg *Conv1DConfig) {
		cfg.Dilation = []int64{val}
	}
}

func WithGroup1D(val int64) Conv1DConfigOpt {
	return func(cfg *Conv1DConfig) {
		cfg.Groups = val
	}
}

// WithBias1D adds bias 1D option.
func WithBias1D(val bool) Conv1DConfigOpt {
	return func(cfg *Conv1DConfig) {
		cfg.Bias = val
	}
}

// WithWsInit adds WsInit 1D option.
func WithWsInit1D(val Init) Conv1DConfigOpt {
	return func(cfg *Conv1DConfig) {
		cfg.WsInit = val
	}
}

// WithBsInit adds BsInit 1D option.
func WithBsInit1D(val Init) Conv1DConfigOpt {
	return func(cfg *Conv1DConfig) {
		cfg.BsInit = val
	}
}

// DefaultConvConfig create a default 1D ConvConfig
func DefaultConv1DConfig() *Conv1DConfig {
	return &Conv1DConfig{
		Stride:   []int64{1},
		Padding:  []int64{0},
		Dilation: []int64{1},
		Groups:   1,
		Bias:     true,
		WsInit:   NewKaimingUniformInit(),
		BsInit:   NewConstInit(float64(0.0)),
	}
}

// NewConv1DConfig creates Conv1DConfig.
func NewConv1DConfig(opts ...Conv1DConfigOpt) *Conv1DConfig {
	cfg := DefaultConv1DConfig()
	for _, o := range opts {
		o(cfg)
	}

	return cfg
}

// Conv2DConfig:
// ============

// Conv2DConfig is configuration for convolution 2D.
type Conv2DConfig struct {
	Stride   []int64
	Padding  []int64
	Dilation []int64
	Groups   int64
	Bias     bool
	WsInit   Init
	BsInit   Init
}

// Conv2DConfigOpt is option type for Conv2DConfig.
type Conv2DConfigOpt func(*Conv2DConfig)

// WithStride2D adds stride 2D option.
func WithStride2D(val int64) Conv2DConfigOpt {
	return func(cfg *Conv2DConfig) {
		cfg.Stride = []int64{val, val}
	}
}

// WithPadding2D adds padding 2D option.
func WithPadding2D(val int64) Conv2DConfigOpt {
	return func(cfg *Conv2DConfig) {
		cfg.Padding = []int64{val, val}
	}
}

// WithDilation2D adds dilation 2D option.
func WithDilation2D(val int64) Conv2DConfigOpt {
	return func(cfg *Conv2DConfig) {
		cfg.Dilation = []int64{val, val}
	}
}

// WithGroup2D adds group 2D option.
func WithGroup2D(val int64) Conv2DConfigOpt {
	return func(cfg *Conv2DConfig) {
		cfg.Groups = val
	}
}

// WithBias2D adds bias 2D option.
func WithBias2D(val bool) Conv2DConfigOpt {
	return func(cfg *Conv2DConfig) {
		cfg.Bias = val
	}
}

// WithWsInit2D adds WsInit 2D option.
func WithWsInit2D(val Init) Conv2DConfigOpt {
	return func(cfg *Conv2DConfig) {
		cfg.WsInit = val
	}
}

// WithBsInit adds BsInit 2D option.
func WithBsInit2D(val Init) Conv2DConfigOpt {
	return func(cfg *Conv2DConfig) {
		cfg.BsInit = val
	}
}

// DefaultConvConfig2D creates a default 2D ConvConfig
func DefaultConv2DConfig() *Conv2DConfig {
	return &Conv2DConfig{
		Stride:   []int64{1, 1},
		Padding:  []int64{0, 0},
		Dilation: []int64{1, 1},
		Groups:   1,
		Bias:     true,
		WsInit:   NewKaimingUniformInit(),
		BsInit:   NewConstInit(float64(0.0)),
	}
}

// NewConv2DConfig creates Conv2DConfig.
func NewConv2DConfig(opts ...Conv2DConfigOpt) *Conv2DConfig {
	cfg := DefaultConv2DConfig()
	for _, o := range opts {
		o(cfg)
	}

	return cfg
}

// Conv3DConfig:
// =============

// Conv3DConfig is configuration struct for convolution 3D.
type Conv3DConfig struct {
	Stride   []int64
	Padding  []int64
	Dilation []int64
	Groups   int64
	Bias     bool
	WsInit   Init
	BsInit   Init
}

// Conv3DConfigOpt is option type for Conv3DConfig.
type Conv3DConfigOpt func(*Conv3DConfig)

// WithStride3D adds stride 3D option.
func WithStride3D(val int64) Conv3DConfigOpt {
	return func(cfg *Conv3DConfig) {
		cfg.Stride = []int64{val, val, val}
	}
}

// WithPadding3D adds padding 3D option.
func WithPadding3D(val int64) Conv3DConfigOpt {
	return func(cfg *Conv3DConfig) {
		cfg.Padding = []int64{val, val, val}
	}
}

// WithDilation3D adds dilation 3D option.
func WithDilation3D(val int64) Conv3DConfigOpt {
	return func(cfg *Conv3DConfig) {
		cfg.Dilation = []int64{val, val, val}
	}
}

// WithGroup3D adds group 3D option.
func WithGroup3D(val int64) Conv3DConfigOpt {
	return func(cfg *Conv3DConfig) {
		cfg.Groups = val
	}
}

// WithBias3D adds bias 3D option.
func WithBias3D(val bool) Conv3DConfigOpt {
	return func(cfg *Conv3DConfig) {
		cfg.Bias = val
	}
}

// WithWsInit3D adds WsInit 3D option.
func WithWsInit3D(val Init) Conv3DConfigOpt {
	return func(cfg *Conv3DConfig) {
		cfg.WsInit = val
	}
}

// WithBsInit adds BsInit 3D option.
func WithBsInit3D(val Init) Conv3DConfigOpt {
	return func(cfg *Conv3DConfig) {
		cfg.BsInit = val
	}
}

// DefaultConvConfig3D creates a default 3D ConvConfig
func DefaultConv3DConfig() *Conv3DConfig {
	return &Conv3DConfig{
		Stride:   []int64{1, 1, 1},
		Padding:  []int64{0, 0, 0},
		Dilation: []int64{1, 1, 1},
		Groups:   1,
		Bias:     true,
		WsInit:   NewKaimingUniformInit(),
		BsInit:   NewConstInit(float64(0.0)),
	}
}

// NewConv3DConfig creates Conv3DConfig.
func NewConv3DConfig(opts ...Conv3DConfigOpt) *Conv3DConfig {
	cfg := DefaultConv3DConfig()
	for _, o := range opts {
		o(cfg)
	}

	return cfg
}

// Conv1D is convolution 1D struct.
type Conv1D struct {
	Ws     *ts.Tensor
	Bs     *ts.Tensor // optional
	Config *Conv1DConfig
}

// NewConv1D creates Conv1D struct.
func NewConv1D(vs *Path, inDim, outDim, k int64, cfg *Conv1DConfig) *Conv1D {
	var (
		ws *ts.Tensor
		bs *ts.Tensor = ts.NewTensor()
	)
	if cfg.Bias {
		bs = vs.MustNewVar("bias", []int64{outDim}, cfg.BsInit)
	}
	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, k)
	ws = vs.MustNewVar("weight", weightSize, cfg.WsInit)

	return &Conv1D{
		Ws:     ws,
		Bs:     bs,
		Config: cfg,
	}
}

// Conv2D is convolution 2D struct.
type Conv2D struct {
	Ws     *ts.Tensor
	Bs     *ts.Tensor // optional
	Config *Conv2DConfig
}

// NewConv2D creates new Conv2D.
func NewConv2D(vs *Path, inDim, outDim int64, k int64, cfg *Conv2DConfig) *Conv2D {
	var (
		ws *ts.Tensor
		bs *ts.Tensor = ts.NewTensor()
	)
	if cfg.Bias {
		bs = vs.MustNewVar("bias", []int64{outDim}, cfg.BsInit)
	}
	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, k, k)
	ws = vs.MustNewVar("weight", weightSize, cfg.WsInit)

	return &Conv2D{
		Ws:     ws,
		Bs:     bs,
		Config: cfg,
	}
}

// Conv3D is convolution 3D struct.
type Conv3D struct {
	Ws     *ts.Tensor
	Bs     *ts.Tensor // optional
	Config *Conv3DConfig
}

// NewConv3D creates new Conv3D struct.
func NewConv3D(vs *Path, inDim, outDim, k int64, cfg *Conv3DConfig) *Conv3D {
	var (
		ws *ts.Tensor
		bs *ts.Tensor = ts.NewTensor()
	)
	if cfg.Bias {
		bs = vs.MustNewVar("bias", []int64{outDim}, cfg.BsInit)
	}
	weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
	weightSize = append(weightSize, k, k, k)
	ws = vs.MustNewVar("weight", weightSize, cfg.WsInit)

	return &Conv3D{
		Ws:     ws,
		Bs:     bs,
		Config: cfg,
	}
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
func NewConv(vs *Path, inDim, outDim int64, ksizes []int64, config interface{}) Conv {

	configT := reflect.TypeOf(config)
	var (
		ws *ts.Tensor
		bs *ts.Tensor = ts.NewTensor()
	)

	switch {
	case len(ksizes) == 1 && configT.String() == "*nn.Conv1DConfig":
		cfg := config.(*Conv1DConfig)
		if cfg.Bias {
			bs = vs.MustNewVar("bias", []int64{outDim}, cfg.BsInit)
		}
		weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
		weightSize = append(weightSize, ksizes...)
		ws = vs.MustNewVar("weight", weightSize, cfg.WsInit)
		return &Conv1D{
			Ws:     ws,
			Bs:     bs,
			Config: cfg,
		}
	case len(ksizes) == 2 && configT.String() == "*nn.Conv2DConfig":
		cfg := config.(*Conv2DConfig)
		if cfg.Bias {
			bs = vs.MustNewVar("bias", []int64{outDim}, cfg.BsInit)
		}
		weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
		weightSize = append(weightSize, ksizes...)
		ws = vs.MustNewVar("weight", weightSize, cfg.WsInit)
		return &Conv2D{
			Ws:     ws,
			Bs:     bs,
			Config: cfg,
		}
	case len(ksizes) == 3 && configT.String() == "*nn.Conv3DConfig":
		cfg := config.(*Conv3DConfig)
		if cfg.Bias {
			bs = vs.MustNewVar("bias", []int64{outDim}, cfg.BsInit)
		}
		weightSize := []int64{outDim, int64(inDim / cfg.Groups)}
		weightSize = append(weightSize, ksizes...)
		ws = vs.MustNewVar("weight", weightSize, cfg.WsInit)
		return &Conv3D{
			Ws:     ws,
			Bs:     bs,
			Config: cfg,
		}
	default:
		err := fmt.Errorf("Expected nd length from 1 to 3. Got %v - configT name: '%v'\n", len(ksizes), configT.String())
		panic(err)
	}
}

// Implement Module for Conv1D, Conv2D, Conv3D:
// ============================================

func (c *Conv1D) Forward(xs *ts.Tensor) *ts.Tensor {
	return ts.MustConv1d(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}

func (c *Conv2D) Forward(xs *ts.Tensor) *ts.Tensor {
	return ts.MustConv2d(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}
func (c *Conv3D) Forward(xs *ts.Tensor) *ts.Tensor {
	return ts.MustConv3d(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}

// Implement ModuleT for Conv1D, Conv2D, Conv3D:
// ============================================

// NOTE: `train` param won't be used, will be?

func (c *Conv1D) ForwardT(xs *ts.Tensor, train bool) *ts.Tensor {
	return ts.MustConv1d(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}

func (c *Conv2D) ForwardT(xs *ts.Tensor, train bool) *ts.Tensor {
	return ts.MustConv2d(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}
func (c *Conv3D) ForwardT(xs *ts.Tensor, train bool) *ts.Tensor {
	return ts.MustConv3d(xs, c.Ws, c.Bs, c.Config.Stride, c.Config.Padding, c.Config.Dilation, c.Config.Groups)
}
