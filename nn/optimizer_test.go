package nn_test

import (
	"testing"
)

// Commented out due to https://github.com/sugarme/gotch/issues/128
// func TestOptimizer(t *testing.T) {
// 	// Set the random seed to make this reproducible.
// 	libtch.ManualSeed(1)
// 	x := ts.MustArangeStart(ts.IntScalar(1), ts.IntScalar(15), gotch.Float, gotch.CPU).MustView([]int64{-1, 1}, true)
// 	// y = x * 0.42 + 1.337
// 	y := x.MustMulScalar(ts.FloatScalar(0.42), false).MustAddScalar(ts.FloatScalar(1.337), false)
// 
// 	vs := nn.NewVarStore(gotch.CPU)
// 	path := vs.Root()
// 
// 	cfg := &nn.LinearConfig{
// 		WsInit: nn.NewConstInit(0.0),
// 		BsInit: nn.NewConstInit(0.0),
// 		Bias:   true,
// 	}
// 	model := nn.NewLinear(path, 1, 1, cfg)
// 
// 	lr := 1e-2
// 	opt, err := nn.DefaultSGDConfig().Build(vs, lr)
// 	if err != nil {
// 		t.Errorf("Failed building SGD optimizer")
// 	}
// 
// 	initialLoss := x.ApplyT(model, true).MustMseLoss(y, 1, true).Float64Values(true)[0]
// 	wantLoss := float64(1.0)
// 	if initialLoss < wantLoss {
// 		t.Errorf("Expect initial loss > %v, got %v", wantLoss, initialLoss)
// 	}
// 
// 	// Optimization loop
// 	for i := 0; i < 50; i++ {
// 		logits := model.Forward(x)
// 		loss := logits.MustMseLoss(y, 1, true)
// 		if i%10 == 0 {
// 			fmt.Printf("Loss: %.3f\n", loss.MustView([]int64{-1}, false).MustFloat64Value([]int64{0}))
// 		}
// 
// 		loss.MustRequiresGrad_(true)
// 		opt.BackwardStep(loss)
// 		loss.MustDrop()
// 	}
// 
// 	loss := x.Apply(model).MustMseLoss(y, 1, true)
// 	opt.BackwardStep(loss)
// 
// 	loss = x.Apply(model).MustMseLoss(y, 1, true)
// 	finalLoss := loss.Float64Values()[0]
// 	fmt.Printf("Final loss: %v\n", finalLoss)
// 
// 	if finalLoss > 0.25 {
// 		t.Errorf("Expect initial loss < 0.25, got %v", finalLoss)
// 	}
// }

// see https://github.com/pytorch/pytorch/blob/9b203f667ac096db9f5f5679ac3e3d7931c34d36/test/test_nn.py#L2308
func TestClipGradNorm(t *testing.T) {
	// TODO.
	// vs := nn.NewVarStore(gotch.CPU)
	// path := vs.Root()
	// l := nn.NewLinear(path, 10, 10, nn.DefaultLinearConfig())
	// maxNorm := 2.0
}

// see https://github.com/pytorch/pytorch/blob/9b203f667ac096db9f5f5679ac3e3d7931c34d36/test/test_nn.py#L2364
func TestClipGradValue(t *testing.T) {
	// TODO
}
