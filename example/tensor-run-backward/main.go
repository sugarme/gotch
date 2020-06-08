package main

import (
	"fmt"
	"log"

	wrapper "github.com/sugarme/gotch/wrapper"
)

func main() {
	x := wrapper.TensorFrom([]float64{2.0})
	x = x.MustSetRequiresGrad(true)
	x.ZeroGrad()

	xmul := wrapper.TensorFrom([]float64{3.0})
	xadd := wrapper.TensorFrom([]float64{5.0})

	x1 := x.MustMul(xmul)
	x2 := x1.MustMul(xmul)
	x3 := x2.MustMul(xmul)

	y := x3.MustAdd(xadd)

	inputs := []wrapper.Tensor{x}

	dy_over_dx, err := wrapper.RunBackward([]wrapper.Tensor{y}, inputs, true, true)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("dy_over_dx length: %v\n", len(dy_over_dx))

	// dy_over_dx1 := dy_over_dx[0]
	// err = dy_over_dx1.Backward()
	// if err != nil {
	// log.Fatalf("Errors:\n, %v", err)
	// }

	dy_over_dx[0].MustBackward()

	x.MustGrad().Print()

}

/* // Compute a second order derivative using run_backward.
 * let mut x = Tensor::from(42.0).set_requires_grad(true);
 * let y = &x * &x * &x + &x + &x * &x;
 * x.zero_grad();
 * let dy_over_dx = Tensor::run_backward(&[y], &[&x], true, true);
 * assert_eq!(dy_over_dx.len(), 1);
 * let dy_over_dx = &dy_over_dx[0];
 * dy_over_dx.backward();
 * let dy_over_dx2 = x.grad();
 * assert_eq!(f64::from(&dy_over_dx2), 254.0); */
