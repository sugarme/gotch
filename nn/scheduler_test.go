package nn

import (
	// "reflect"
	// "fmt"
	"testing"

	"github.com/sugarme/gotch"
)

// func TestLambdaLRScheduler_Step(t *testing.T) {
func TestLambdaLR(t *testing.T) {
	vs := NewVarStore(gotch.CPU)
	opt, err := DefaultAdamConfig().Build(vs, 0.001)
	if err != nil {
		t.Error(err)
	}

	ld1 := func(epoch interface{}) float64 {
		return float64(epoch.(int) / 30)
	}

	var s *LRScheduler
	s = NewLambdaLR(opt, []LambdaFn{ld1}, 100).Build()

	wants := []float64{
		0.001, // initial LR
		0.001, // epoch 30s/30 = 1 * 0.001
		0.002, // epoch 60s/30 = 2 * 0.001 = 0.002
		0.006, // epoch 90s/30 = 3 * 0.002 = 0.006
	}
	i := 0
	for epoch := 0; epoch < 100; epoch++ {
		if epoch%30 == 0 && epoch > 0 {
			s.Step(epoch)
			i += 1
		}

		want := wants[i]
		got := opt.GetLRs()[0]
		if got != want {
			t.Errorf("Epoch %d: Want %v - Got %v", epoch, want, got)
		}
	}
}
