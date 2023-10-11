package main

import (
	"fmt"
	"math/rand"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

const (
	ImageDimNN    int64 = 784
	HiddenNodesNN int64 = 128
	LabelNN       int64 = 10

	BatchSize int64 = 3000

	epochsNN = 200
	LrNN     = 1e-3
)

type model struct {
	fc  *nn.Linear
	act nn.Func
}

func newModel(vs *nn.VarStore) *model {
	fc := nn.NewLinear(vs.Root(), ImageDimNN, HiddenNodesNN, nn.DefaultLinearConfig())
	act := nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	})

	return &model{
		fc:  fc,
		act: act,
	}
}

func (m *model) Forward(x *ts.Tensor) *ts.Tensor {
	fc := m.fc.Forward(x)
	act := m.act.Forward(fc)

	return act
}

func newData() []float32 {
	n := int(BatchSize * ImageDimNN)
	data := make([]float32, n)
	for i := 0; i < n; i++ {
		data[i] = rand.Float32()
	}

	return data
}

func main() {
	epochs := 4000

	// device := gotch.CPU
	device := gotch.CudaIfAvailable()
	vs := nn.NewVarStore(device)
	m := newModel(vs)

	for i := 0; i < epochs; i++ {
		// input := ts.MustOfSlice(newData()).MustView([]int64{BatchSize, ImageDimNN}, true).MustTo(device, true)
		input := ts.MustRandn([]int64{BatchSize, ImageDimNN}, gotch.Float, device)

		ts.NoGrad(func() {
			_ = m.Forward(input)
		})

		if i%10 == 0 {
			fmt.Printf("=================== Epoch %03d completed========================\n", i)
		}
	}

	ts.CleanUp()
}
