package nn

import (
	"log"
	"math"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

type Init interface {
	// creates a new tensor with specified initiation
	InitTensor(dims []int64, device gotch.Device) (retVal *ts.Tensor)

	// re-initializes (in-place) an existing tensor with the specified initiation
	Set(tensor *ts.Tensor)
}

// constInit:
// ==========

type constInit struct {
	value float64
}

func NewConstInit(v float64) constInit {
	return constInit{v}
}

func (c constInit) InitTensor(dims []int64, device gotch.Device) (retVal *ts.Tensor) {
	var err error
	kind := gotch.Float
	switch {
	case c.value == 0.0:
		retVal = ts.MustZeros(dims, kind, device)
	case c.value == 1.0:
		retVal = ts.MustOnes(dims, kind, device)
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

func NewRandnInit(mean, stdev float64) randnInit {
	return randnInit{mean, stdev}
}

func (r randnInit) InitTensor(dims []int64, device gotch.Device) (retVal *ts.Tensor) {
	// if r.mean == 0 && math.Abs(r.stdev-1) <= math.SmallestNonzeroFloat64 {
	if r.mean == 0 {
		return ts.MustRandn(dims, gotch.Float, device)
	}

	initTs := ts.MustRandn(dims, gotch.Float, device)
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

func NewUniformInit(lo, up float64) uniformInit {
	return uniformInit{lo, up}
}

func (u uniformInit) InitTensor(dims []int64, device gotch.Device) (retVal *ts.Tensor) {
	var err error
	kind := gotch.Float
	retVal = ts.MustZeros(dims, kind, device)
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

type kaimingUniformInit struct{}

func NewKaimingUniformInit() kaimingUniformInit {
	return kaimingUniformInit{}
}

func (k kaimingUniformInit) InitTensor(dims []int64, device gotch.Device) (retVal *ts.Tensor) {
	var fanIn int64
	if len(dims) == 0 {
		log.Fatalf("KaimingUniformInit method call: dims (%v) should have length >= 1", dims)
	} else if len(dims) == 1 {
		fanIn = factorial(dims[0])
	} else {
		fanIn = product(dims[1:])
	}

	bound := math.Sqrt(1.0 / float64(fanIn))
	kind := gotch.Float
	retVal = ts.MustZeros(dims, kind, device)
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

	var fanIn int64
	if len(dims) == 0 {
		log.Fatalf("KaimingUniformInit Set method call: Tensor (%v) should have length >= 1", tensor.MustSize())
	} else if len(dims) == 1 {
		fanIn = factorial(dims[0])
	} else {
		fanIn = product(dims[1:])
	}

	bound := math.Sqrt(1.0 / float64(fanIn))
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
