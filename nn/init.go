package nn

import (
	"fmt"
	"log"
	"math"
	"strings"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

type Init interface {
	// creates a new tensor with specified initiation
	InitTensor(dims []int64, device gotch.Device, dtypeOpt ...gotch.DType) (retVal *ts.Tensor)

	// re-initializes (in-place) an existing tensor with the specified initiation
	Set(tensor *ts.Tensor)
}

// constInit:
// ==========

type constInit struct {
	value float64
}

var _ Init = new(constInit)

func NewConstInit(v float64) constInit {
	return constInit{v}
}

func (c constInit) InitTensor(dims []int64, device gotch.Device, dtypeOpt ...gotch.DType) (retVal *ts.Tensor) {
	dtype := gotch.DefaultDType
	if len(dtypeOpt) > 0 {
		dtype = dtypeOpt[0]
	}

	var err error
	switch {
	case c.value == 0.0:
		retVal = ts.MustZeros(dims, dtype, device)
	case c.value == 1.0:
		retVal = ts.MustOnes(dims, dtype, device)
	default:
		data := make([]float64, ts.FlattenDim(dims))
		for i := range data {
			data[i] = c.value
		}
		retVal, err = ts.NewTensorFromData(data, dims)
		if err != nil {
			log.Fatalf("constInit - InitTensor method call error: %v\n", err)
		}
	}

	return retVal
}

func (c constInit) Set(tensor *ts.Tensor) {
	var err error
	scalarVal := ts.FloatScalar(c.value)
	if err != nil {
		log.Fatalf("constInit - Set method call error: %v\n", err)
	}

	tensor.Fill_(scalarVal)
}

// randnInit :
// ===========
type randnInit struct {
	mean  float64
	stdev float64
}

var _ Init = new(randnInit)

func NewRandnInit(mean, stdev float64) randnInit {
	return randnInit{mean, stdev}
}

func (r randnInit) InitTensor(dims []int64, device gotch.Device, dtypeOpt ...gotch.DType) (retVal *ts.Tensor) {
	dtype := gotch.DefaultDType
	if len(dtypeOpt) > 0 {
		dtype = dtypeOpt[0]
	}

	// if r.mean == 0 && math.Abs(r.stdev-1) <= math.SmallestNonzeroFloat64 {
	if r.mean == 0 {
		return ts.MustRandn(dims, dtype, device)
	}

	initTs := ts.MustRandn(dims, dtype, device)
	return initTs.MustMulScalar(ts.FloatScalar(r.stdev), true).MustAddScalar(ts.FloatScalar(r.mean), true)
}

func (r randnInit) Set(tensor *ts.Tensor) {
	dims, err := tensor.Size()
	if err != nil {
		log.Fatalf("randInit - Set method call error: %v\n", err)
	}

	initTs := r.InitTensor(dims, tensor.MustDevice())
	tensor.Copy_(initTs)
	initTs.MustDrop()
}

// uniformInit :
// =============

type uniformInit struct {
	lo float64
	up float64
}

var _ Init = new(uniformInit)

func NewUniformInit(lo, up float64) uniformInit {
	return uniformInit{lo, up}
}

func (u uniformInit) InitTensor(dims []int64, device gotch.Device, dtypeOpt ...gotch.DType) (retVal *ts.Tensor) {
	dtype := gotch.DefaultDType
	if len(dtypeOpt) > 0 {
		dtype = dtypeOpt[0]
	}

	var err error
	retVal = ts.MustZeros(dims, dtype, device)
	retVal.Uniform_(u.lo, u.up)
	if err != nil {
		log.Fatalf("uniformInit - InitTensor method call error: %v\n", err)
	}
	return retVal
}

func (u uniformInit) Set(tensor *ts.Tensor) {
	tensor.Uniform_(u.lo, u.up)
}

// kaiminguniformInit :
// ====================
type KaimingOptions struct {
	NegativeSlope float64
	Mode          string
	NonLinearity  string
}

type KaimingOption func(*KaimingOptions)

func DefaultKaimingOptions() *KaimingOptions {
	return &KaimingOptions{
		NegativeSlope: 0.01,
		Mode:          "fanIn",
		NonLinearity:  "leaky_relu",
	}
}

func WithKaimingMode(v string) KaimingOption {
	if v != "fanIn" && v != "fanOut" {
		panic("Mode must be either 'fanIn' or 'fanOut'.")
	}
	return func(opt *KaimingOptions) {
		opt.Mode = v
	}
}

func WithKaimingNonLinearity(v string) KaimingOption {
	return func(opt *KaimingOptions) {
		opt.NonLinearity = v
	}
}

func WithKaimingNegativeSlope(v float64) KaimingOption {
	return func(opt *KaimingOptions) {
		opt.NegativeSlope = v
	}
}

func NewKaimingOptions(opts ...KaimingOption) *KaimingOptions {
	options := DefaultKaimingOptions()
	for _, opt := range opts {
		opt(options)
	}

	return options
}

type kaimingUniformInit struct {
	NegativeSlope float64
	Mode          string
	NonLinearity  string
}

var _ Init = new(kaimingUniformInit)

func NewKaimingUniformInit(opts ...KaimingOption) *kaimingUniformInit {
	o := DefaultKaimingOptions()
	for _, opt := range opts {
		opt(o)
	}

	return &kaimingUniformInit{
		NegativeSlope: o.NegativeSlope,
		Mode:          o.Mode,
		NonLinearity:  o.NonLinearity,
	}
}

func (k *kaimingUniformInit) InitTensor(dims []int64, device gotch.Device, dtypeOpt ...gotch.DType) (retVal *ts.Tensor) {
	dtype := gotch.DefaultDType
	if len(dtypeOpt) > 0 {
		dtype = dtypeOpt[0]
	}

	fanIn, _, err := CalculateFans(dims)
	if err != nil {
		panic(err)
	}

	gain, err := calculateGain(k.NonLinearity, k.NegativeSlope) // default non-linearity="leaky_relu", negative_slope=0.01
	if err != nil {
		err = fmt.Errorf("kaimingUniformInit.InitTensor() failed: %v\n", err)
		panic(err)
	}

	std := gain / math.Sqrt(float64(fanIn)) // default using fanIn

	// Calculate uniform bounds from standard deviation
	bound := math.Sqrt(3.0) * std

	retVal = ts.MustZeros(dims, dtype, device)
	retVal.Uniform_(-bound, bound)

	return retVal
}

// product calculates product by multiplying elements
func product(dims []int64) (retVal int64) {
	for i, v := range dims {
		if i == 0 {
			retVal = v
		} else {
			retVal = retVal * v
		}
	}

	return retVal
}

func factorial(n int64) (result int64) {
	if n > 0 {
		result = n * factorial(n-1)
		return result
	}
	return 1
}

func (k kaimingUniformInit) Set(tensor *ts.Tensor) {
	dims, err := tensor.Size()
	if err != nil {
		log.Fatalf("uniformInit - Set method call error: %v\n", err)
	}

	fanIn, _, err := CalculateFans(dims)
	if err != nil {
		panic(err)
	}

	gain, err := calculateGain(k.NonLinearity, k.NegativeSlope) // default non-linearity="leaky_relu", negative_slope=0.01
	if err != nil {
		err = fmt.Errorf("kaimingUniformInit.Set() failed: %v\n", err)
		panic(err)
	}

	std := gain / math.Sqrt(float64(fanIn)) // default using fanIn

	// Calculate uniform bounds from standard deviation
	bound := math.Sqrt(3.0) * std

	tensor.Uniform_(-bound, bound)
}

// glorotInit :
// ====================
type glorotNInit struct{}

func NewGlorotNInit() glorotNInit {
	return glorotNInit{}
}

func (gl glorotNInit) InitTensor(dims []int64, device gotch.Device) (retVal *ts.Tensor) {
	// TODO: implement

	return
}

func (gl glorotNInit) Set(tensor *ts.Tensor) {
	// TODO: implement
}

// KaimingUniform:
// ===============
// Base on Pytorch:
// https://github.com/pytorch/pytorch/blob/98f40af7e3133e042454efab668a842c4d01176e/torch/nn/init.py#L284
func calculateFan(shape []int64) (fan map[string]int64, err error) {
	if len(shape) < 2 {
		err = fmt.Errorf("calculateFan() failed: fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
		return
	}

	fan = make(map[string]int64)

	numInputFmap := shape[1]
	numOutputFmap := shape[0]
	var receptiveFieldSize int64 = 1
	if len(shape) > 2 {
		// calculate product
		for _, s := range shape[2:] {
			receptiveFieldSize *= int64(s)
		}
	}

	fan["fanIn"] = numInputFmap * receptiveFieldSize
	fan["fanOut"] = numOutputFmap * receptiveFieldSize

	return fan, nil
}

// CalculateFans calculates fan-in and fan-out based on tensor shape.
func CalculateFans(shape []int64) (fanIn, fanOut int64, err error) {
	fan, err := calculateFan(shape)
	return fan["fanIn"], fan["fanOut"], err
}

// Return the recommended gain value for the given nonlinearity function.
// Default fn should be `leaky_relu`
func calculateGain(fn string, paramOpt ...float64) (float64, error) {
	linearFns := []string{"linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"}

	negativeSlope := 0.01
	if len(paramOpt) > 0 {
		negativeSlope = paramOpt[0]
	}

	fn = strings.ToLower(fn)
	if contains(linearFns, fn) || fn == "sigmoid" {
		return 1, nil
	}

	switch fn {
	case "tanh":
		return 5.0 / 3.0, nil
	case "relu":
		return math.Sqrt(2.0), nil
	case "leaky_relu": // default fn
		return math.Sqrt(2.0 / (1 + math.Pow(negativeSlope, 2))), nil
	case "selu":
		return 3.0 / 4, nil // Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
	default:
		err := fmt.Errorf("calculateGain() failed: unsupported non-linearity function %q\n", fn)
		return -1, err
	}
}

func contains(items []string, item string) bool {
	for _, i := range items {
		if item == i {
			return true
		}
	}
	return false
}
