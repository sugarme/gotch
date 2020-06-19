package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func testOptimizer() {

	var data []float64
	for i := 0; i < 15; i++ {
		data = append(data, float64(i))
	}
	xs, err := ts.NewTensorFromData(data, []int64{int64(len(data)), 1})
	if err != nil {
		log.Fatal(err)
	}

	ys := xs.MustMul1(ts.FloatScalar(0.42)).MustAdd1(ts.FloatScalar(1.337))

	vs := nn.NewVarStore(gotch.CPU)

	cfg := nn.LinearConfig{
		WsInit: nn.NewConstInit(0.001),
		BsInit: nn.NewConstInit(0.001),
		Bias:   true,
	}

	// fmt.Printf("Number of trainable variables: %v\n", vs.Len())
	linear := nn.NewLinear(vs.Root(), 1, 1, cfg)
	// fmt.Printf("Trainable variables at app: %v\n", vs.TrainableVariable())

	loss := xs.Apply(linear).MustMseLoss(ys, ts.ReductionMean.ToInt())
	initialLoss := loss.MustView([]int64{-1}).MustFloat64Value([]int64{0})
	fmt.Printf("Initial Loss: %.3f\n", initialLoss)

	opt, err := nn.DefaultSGDConfig().Build(vs, 1e-2)
	if err != nil {
		log.Fatal("Failed building SGD optimizer")
	}

	for i := 0; i < 50; i++ {
		// loss = xs.Apply(linear)
		loss = linear.Forward(xs)
		loss = loss.MustMseLoss(ys, ts.ReductionMean.ToInt())

		fmt.Printf("Loss: %.3f\n", loss.MustView([]int64{-1}).MustFloat64Value([]int64{0}))

		opt.BackwardStep(loss)

		fmt.Printf("Bs: %.3f - Bs Grad: %.3f\n", linear.Bs.MustView([]int64{-1}).MustFloat64Value([]int64{0}), linear.Bs.MustGrad().MustFloat64Value([]int64{0}))
		fmt.Printf("Ws: %.3f - Ws Grad: %.3f\n", linear.Ws.MustView([]int64{-1}).MustFloat64Value([]int64{0}), linear.Ws.MustGrad().MustFloat64Value([]int64{0}))

	}

}

func main() {
	testOptimizer()
}
