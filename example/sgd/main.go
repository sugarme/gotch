package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func myModule(p nn.Path, dim int64) ts.Module {
	x1 := p.Zeros("x1", []int64{dim})
	x2 := p.Zeros("x1", []int64{dim})

	return nn.NewFunc(func(xs ts.Tensor) ts.Tensor {
		return xs.MustMul(x1).MustAdd(xs.MustExp().MustMul(x2))
	})

}

func main() {

	vs := nn.NewVarStore(gotch.CPU)

	m := myModule(vs.Root(), 7)

	opt, err := nn.DefaultSGDConfig().Build(vs, 1e-2)
	if err != nil {
		log.Fatal(err)
	}

	for i := 0; i < 50; i++ {
		xs := ts.MustZeros([]int64{7}, gotch.Float.CInt(), gotch.CPU.CInt())
		ys := ts.MustZeros([]int64{7}, gotch.Float.CInt(), gotch.CPU.CInt())

		loss := m.Forward(xs).MustSub(ys).MustPow(ts.IntScalar(2)).MustSum(gotch.Float.CInt())

		opt.BackwardStep(loss)

		fmt.Printf("Loss: %v\n", loss.MustView([]int64{-1}).MustFloat64Value([]int64{0}))

	}

}
