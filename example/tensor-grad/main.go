package main

import (
	"fmt"

	"github.com/sugarme/gotch/ts"
)

func main() {
	x := ts.TensorFrom([]float64{2.0})
	x = x.MustSetRequiresGrad(true, false)
	x.ZeroGrad()

	xy := ts.TensorFrom([]float64{2.0})
	xz := ts.TensorFrom([]float64{3.0})

	y := x.MustMul(xy, false)
	z := x.MustMul(xz, false)

	y.Backward()
	xgrad := x.MustGrad(false)
	xgrad.Print() // [2.0]
	z.Backward()
	xgrad = x.MustGrad(false)
	xgrad.Print() // [5.0] due to accumulated 2.0 + 3.0

	isGradEnabled := ts.MustGradSetEnabled(false)
	fmt.Printf("Previous GradMode enabled state: %v\n", isGradEnabled)
	isGradEnabled = ts.MustGradSetEnabled(true)
	fmt.Printf("Previous GradMode enabled state: %v\n", isGradEnabled)

}
