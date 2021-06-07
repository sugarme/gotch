package nn

import (
	// "fmt"
	"log"
)

type scheduler interface {
	SetLRs(epochOpt ...int)
	Build() *LRScheduler
}

// LRScheduler is a scheduler to update optimizer learning rates.
type LRScheduler struct {
	scheduler scheduler
}

// Step updates optimizer learning rate.
func (s *LRScheduler) Step(epochOpt ...int) {
	s.scheduler.SetLRs(epochOpt...)
}

type LambdaFn func(in interface{}) float64

// LamdaLR is a LRScheduler configuration to build LRScheduler.
// It generates new learning rate for each parameter group: new lr = initial lr x Lambda function.
// When lastEpoch = -1, it sets initial lrs as lrs.
type LambdaLR struct {
	opt       *Optimizer
	lrLambdas []LambdaFn // length should be equal to length of optimizer param groups.
	lastEpoch int
}

// NewLambdaLRScheduler creates a new LambdaLRScheduler.
func NewLambdaLR(opt *Optimizer, ldFns []LambdaFn, lastEpoch int) *LambdaLR {
	ngroup := opt.ParamGroupNum()
	if int(ngroup) != len(ldFns) {
		log.Fatalf("Number of lambda functions (%d) is not equal to number of optimizer groups (%d)", len(ldFns), ngroup)
	}
	return &LambdaLR{opt, ldFns, lastEpoch}
}

// Build implements scheduler interface.
func (l *LambdaLR) Build() *LRScheduler {
	return &LRScheduler{l}
}

// SetLRs implements scheduler interface.
func (l *LambdaLR) SetLRs(epochOpt ...int) {
	epoch := -1
	if len(epochOpt) > 0 {
		epoch = epochOpt[0]
	}

	switch epoch {
	case -1:
		l.lastEpoch += 1
	default:
		l.lastEpoch = epoch
	}

	var newLRs []float64
	lrs, err := l.opt.opt.GetLearningRates()
	if err != nil {
		log.Fatal(err)
	}

	for i, lr := range lrs {
		lambda := l.lrLambdas[i](l.lastEpoch)
		newLR := lr
		if lambda > 0 {
			newLR = lr * lambda
		}
		newLRs = append(newLRs, newLR)
	}

	l.opt.SetLRs(newLRs)
}
