package nn_test

import (
	// "reflect"
	// "fmt"
	"math"
	"testing"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
)

func TestLambdaLR(t *testing.T) {
	vs := nn.NewVarStore(gotch.CPU)
	opt, err := nn.DefaultAdamConfig().Build(vs, 0.001)
	if err != nil {
		t.Error(err)
	}

	ld1 := func(epoch interface{}) float64 {
		return float64(epoch.(int) / 30)
	}

	var s *nn.LRScheduler
	s = nn.NewLambdaLR(opt, []nn.LambdaFn{ld1}).Build()

	wants := []float64{
		0.0,   // epoch < 30 -> 0 * lr = 0 * 0.0
		0.001, // 30 <= epoch < 60 -> 1 * 0.001 = 0.001
		0.002, // 60 <= epoch < 90 ->  2 * 0.001 = 0.002
		0.003, // 90 <= epoch  ->  3 * 0.001 = 0.003
	}
	i := 0
	for epoch := 0; epoch < 100; epoch++ {
		if (epoch+1)%30 == 0 && epoch > 0 {
			i += 1
		}

		s.Step()
		want := wants[i]
		got := opt.GetLRs()[0]
		if got != want {
			t.Errorf("Epoch %d: Want %v - Got %v", epoch, want, got)
		}
	}
}

func TestMultiplicativeLR(t *testing.T) {
	vs := nn.NewVarStore(gotch.CPU)
	opt, err := nn.DefaultAdamConfig().Build(vs, 1)
	if err != nil {
		t.Error(err)
	}

	ld1 := func(epoch interface{}) float64 {
		e := float64(epoch.(int))
		return math.Pow(2, e) // 2 ** epoch
	}

	var s *nn.LRScheduler
	s = nn.NewMultiplicativeLR(opt, []nn.LambdaFn{ld1}).Build()

	wants := []float64{
		2,     // 2^1 * 1 = 2
		8,     // 2^2 * 2 = 8
		64,    // 2^3 * 8 = 64
		1024,  // 2^4 * 64 = 1024
		32768, // 2^5 *1024 = 32768
	}
	for epoch := 0; epoch < 5; epoch++ {
		s.Step()
		want := wants[epoch]
		got := opt.GetLRs()[0]
		if got != want {
			t.Errorf("Epoch %d: Want %v - Got %v", epoch, want, got)
		}
	}
}

func TestStepLR(t *testing.T) {
	vs := nn.NewVarStore(gotch.CPU)
	opt, err := nn.DefaultAdamConfig().Build(vs, 0.05)
	if err != nil {
		t.Error(err)
	}

	var s *nn.LRScheduler
	s = nn.NewStepLR(opt, 30, 0.1).Build()

	wants := []float64{
		0.05,    // initial LR -> 0.05
		0.005,   // 30 <= epoch < 60 -> 0.05 * gamma = 0.005
		0.0005,  // 60 <= epoch < 90 -> 0.005 * gamma = 0.0005
		0.00005, // 90 <= epoch < 120 -> 0.0005 * gamma = 0.00005
	}
	i := 0
	for epoch := 0; epoch < 100; epoch++ {
		s.Step()
		if (epoch+1)%30 == 0 && epoch > 0 {
			i += 1
		}
		want := wants[i]
		got := opt.GetLRs()[0]
		if got != want {
			t.Errorf("Epoch %d: Want %v - Got %v", epoch, want, got)
		}
	}
}

func TestMultiStepLR(t *testing.T) {
	vs := nn.NewVarStore(gotch.CPU)
	opt, err := nn.DefaultAdamConfig().Build(vs, 0.05)
	if err != nil {
		t.Error(err)
	}

	var s *nn.LRScheduler
	s = nn.NewMultiStepLR(opt, []int{31, 81}, 0.1).Build()

	wants := []float64{
		0.05,   // initial LR -> 0.05
		0.005,  // 30 <= epoch < 80 -> 0.05 * gamma = 0.005
		0.0005, // 80 <= epoch -> 0.005 * gamma = 0.0005
	}
	i := 0
	for epoch := 0; epoch < 100; epoch++ {
		s.Step()
		if contain(epoch, []int{30, 80}) {
			i += 1
		}
		want := wants[i]
		got := opt.GetLRs()[0]
		if got != want {
			t.Errorf("Epoch %d: Want %v - Got %v", epoch, want, got)
		}
	}
}

func TestExponentialLR(t *testing.T) {
	vs := nn.NewVarStore(gotch.CPU)
	opt, err := nn.DefaultAdamConfig().Build(vs, 0.05)
	if err != nil {
		t.Error(err)
	}

	var s *nn.LRScheduler
	s = nn.NewStepLR(opt, 30, 0.1).Build()

	wants := []float64{
		0.05,    // initial LR -> 0.05
		0.005,   // epoch 1 -> 0.05 * gamma = 0.005
		0.0005,  // epoch 2 -> 0.005 * gamma = 0.0005
		0.00005, // epoch 3 -> 0.0005 * gamma = 0.00005
	}
	i := 0
	for epoch := 0; epoch < 3; epoch++ {
		s.Step()
		want := wants[i]
		got := opt.GetLRs()[0]
		if got != want {
			t.Errorf("Epoch %d: Want %v - Got %v", epoch, want, got)
		}
	}
}

func TestCosineAnnealingLR(t *testing.T) {
	vs := nn.NewVarStore(gotch.CPU)
	// model := NewLinear(vs.Root(), 10, 2, DefaultLinearConfig())
	opt, err := nn.DefaultSGDConfig().Build(vs, 1.0)
	if err != nil {
		t.Error(err)
	}

	var s *nn.LRScheduler
	steps := 10
	s = nn.NewCosineAnnealingLR(opt, steps, 0.0).Build()

	for epoch := 0; epoch < 5; epoch++ {
		opt.SetLRs([]float64{1.0})
		// s := NewCosineAnnealingLR(opt, steps, 0.0).Build()
		for idx := 0; idx < steps; idx++ {
			s.Step()
			t.Logf("LR: %0.10f\n", opt.GetLRs())
		}

		t.Logf("Reset scheduler. \n")
		opt.ResetStepCount()
		s = nn.NewCosineAnnealingLR(opt, steps, 0.0).Build()
	}

	// t.Log(model)
}

func contain(item int, list []int) bool {
	for _, i := range list {
		if i == item {
			return true
		}
	}

	return false
}

func TestCyclicLR(t *testing.T) {
	vs := nn.NewVarStore(gotch.CPU)
	model := nn.NewLinear(vs.Root(), 10, 2, nn.DefaultLinearConfig())
	opt, err := nn.DefaultSGDConfig().Build(vs, 1.0)
	if err != nil {
		t.Error(err)
	}

	var s *nn.LRScheduler
	baseLRs := []float64{0.001}
	maxLRs := []float64{0.1}
	s = nn.NewCyclicLR(opt, baseLRs, maxLRs, nn.WithCyclicStepSizeUp(5), nn.WithCyclicMode("triangular")).Build()

	var lrs []float64
	for i := 0; i < 100; i++ {
		opt.Step()
		lr := opt.GetLRs()[0]
		lrs = append(lrs, lr)
		t.Logf("batch %2d: lr %0.4f\n", i, lr)
		s.Step()
	}
	// t.Logf("Lrs: %+v\n", lrs)
	t.Log(model)
}

func TestCosineAnnealingWarmRestarts(t *testing.T) {
	vs := nn.NewVarStore(gotch.CPU)
	model := nn.NewLinear(vs.Root(), 2, 1, nn.DefaultLinearConfig())
	opt, err := nn.DefaultSGDConfig().Build(vs, 0.1)
	if err != nil {
		t.Error(err)
	}

	var s *nn.LRScheduler
	t0 := 10
	tMult := 1
	etaMin := 0.001
	lastEpoch := -1
	s = nn.NewCosineAnnealingWarmRestarts(opt, t0, nn.WithTMult(tMult), nn.WithEtaMin(etaMin), nn.WithCosineAnnealingLastEpoch(lastEpoch)).Build()

	var lrs []float64
	for i := 0; i < 100; i++ {
		s.Step()
		lr := opt.GetLRs()[0]
		lrs = append(lrs, lr)
		t.Logf("batch %2d: lr %0.4f\n", i, lr)
	}
	// t.Logf("Lrs: %+v\n", lrs)
	t.Log(model)
}

func TestOneCycleLR(t *testing.T) {
	vs := nn.NewVarStore(gotch.CPU)
	model := nn.NewLinear(vs.Root(), 2, 1, nn.DefaultLinearConfig())
	opt, err := nn.DefaultSGDConfig().Build(vs, 0.1)
	if err != nil {
		t.Error(err)
	}

	var s *nn.LRScheduler
	maxLR := 0.1
	stepsPerEpoch := 10
	epochs := 10
	annealStrategy := "linear"

	s = nn.NewOneCycleLR(opt, maxLR, nn.WithOneCycleStepsPerEpoch(stepsPerEpoch), nn.WithOneCycleEpochs(epochs), nn.WithOneCycleAnnealStrategy(annealStrategy)).Build()

	var lrs []float64
	for i := 0; i < 100; i++ {
		opt.Step()
		lr := opt.GetLRs()[0]
		lrs = append(lrs, lr)
		t.Logf("batch %2d: lr %0.4f\n", i, lr)
		s.Step()
	}
	// t.Logf("Lrs: %+v\n", lrs)
	t.Log(model)
}
