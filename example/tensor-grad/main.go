package main

import (
	// "fmt"
	// "log"

	wrapper "github.com/sugarme/gotch/wrapper"
)

func main() {
	x := wrapper.TensorFrom([]float64{2.0})
	x = x.MustSetRequiresGrad(true)
	x.ZeroGrad()

	xy := wrapper.TensorFrom([]float64{2.0})
	xz := wrapper.TensorFrom([]float64{3.0})

	y := x.MustMul(xy)
	z := x.MustMul(xz)

	y.Backward()
	xgrad := x.MustGrad()
	xgrad.Print()
	z.Backward()
	xgrad = x.MustGrad()
	xgrad.Print()

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
