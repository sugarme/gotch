package nn_test

/*
 * import (
 *   // "reflect"
 *   "fmt"
 *   "log"
 *   "testing"
 *
 *   "github.com/sugarme/gotch"
 *   "github.com/sugarme/gotch/nn"
 *   ts "github.com/sugarme/gotch/tensor"
 * )
 *
 * func TestOptimizer(t *testing.T) {
 *
 *   var data []float32
 *   for i := 0; i < 15; i++ {
 *     data = append(data, float32(i))
 *   }
 *   xs, err := ts.NewTensorFromData(data, []int64{int64(len(data)), 1})
 *   if err != nil {
 *     log.Fatal(err)
 *   }
 *
 *   ys := xs.MustMul1(ts.FloatScalar(0.42), false).MustAdd1(ts.FloatScalar(1.337), false)
 *
 *   vs := nn.NewVarStore(gotch.CPU)
 *
 *   optCfg := nn.DefaultSGDConfig()
 *   opt, err := optCfg.Build(vs, 1e-2)
 *   if err != nil {
 *     t.Errorf("Failed building SGD optimizer")
 *   }
 *
 *   cfg := nn.LinearConfig{
 *     WsInit: nn.NewConstInit(0.0),
 *     BsInit: nn.NewConstInit(0.0),
 *     Bias:   true,
 *   }
 *
 *   linear := nn.NewLinear(vs.Root(), 1, 1, cfg)
 *
 *   logits := xs.Apply(linear)
 *   loss := logits.MustMseLoss(ys, ts.ReductionMean.ToInt(), true)
 *
 *   initialLoss := loss.MustView([]int64{-1}, false).MustFloat64Value([]int64{0})
 *
 *   wantLoss := float64(1.0)
 *
 *   if initialLoss < wantLoss {
 *     t.Errorf("Expect initial loss > %v, got %v", wantLoss, initialLoss)
 *   }
 *
 *   for i := 0; i < 50; i++ {
 *     loss = xs.Apply(linear).MustMseLoss(ys, ts.ReductionMean.ToInt(), true)
 *
 *     opt.BackwardStep(loss)
 *     fmt.Printf("Loss: %.3f\n", loss.MustView([]int64{-1}, false).MustFloat64Value([]int64{0}))
 *   }
 *
 *   loss = xs.Apply(linear).MustMseLoss(ys, ts.ReductionMean.ToInt(), true)
 *   finalLoss := loss.Values()[0]
 *   fmt.Printf("Final loss: %v\n", finalLoss)
 *
 *   if finalLoss > 0.25 {
 *     t.Errorf("Expect initial loss < 0.25, got %v", finalLoss)
 *   }
 * } */
