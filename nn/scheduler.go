package nn

import (
	"fmt"
	"log"
	"math"
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

// LamdaLR calculates new learning rate for each parameter group by applying
// Lambda function to the corresponding INITIAL learning rate.
type LambdaLR struct {
	opt        *Optimizer
	lrLambdas  []LambdaFn // length should be 1 or equal to length of optimizer param groups.
	initialLRs []float64
}

// NewLambdaLRS creates a new LambdaLRS.
func NewLambdaLR(opt *Optimizer, ldFns []LambdaFn) *LambdaLR {
	ngroup := opt.ParamGroupNum()
	initialLRs := opt.GetLRs()
	var funcs []LambdaFn = make([]LambdaFn, ngroup)
	switch len(ldFns) {
	case 1:
		// Apply Lambda function to all param groups
		for i := 0; i < ngroup; i++ {
			funcs[i] = ldFns[0]
		}
	case ngroup:
		funcs = ldFns
	default:
		log.Fatalf("Number of lambda functions (%d) is not equal to number of optimizer groups (%d)", len(ldFns), ngroup)
	}
	return &LambdaLR{opt, ldFns, initialLRs}
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

	var newLRs []float64
	switch epoch {
	case -1, 0:
		newLRs = l.initialLRs
	default:
		for i, lr := range l.initialLRs {
			lambda := l.lrLambdas[i](epoch)
			newLR := lr * lambda
			newLRs = append(newLRs, newLR)
		}
	}

	l.opt.SetLRs(newLRs)
}

// MultiplicativeLR calculates new learning rates for each optimizer para groups
// by applying corresponding Lambda function to the CURRENT learning rate.
type MultiplicativeLR struct {
	opt        *Optimizer
	lrLambdas  []LambdaFn // length should be 1 or equal to length of optimizer param groups.
	initialLRs []float64
}

// NewMultiplicativeLR creates a new MultiplicativeLR.
func NewMultiplicativeLR(opt *Optimizer, ldFns []LambdaFn) *MultiplicativeLR {
	ngroup := opt.ParamGroupNum()
	initialLRs := opt.GetLRs()
	var funcs []LambdaFn = make([]LambdaFn, ngroup)
	switch len(ldFns) {
	case 1:
		// Apply Lambda function to all param groups
		for i := 0; i < ngroup; i++ {
			funcs[i] = ldFns[0]
		}
	case ngroup:
		funcs = ldFns
	default:
		log.Fatalf("Number of lambda functions (%d) is not equal to number of optimizer groups (%d)", len(ldFns), ngroup)
	}
	return &MultiplicativeLR{opt, ldFns, initialLRs}
}

// Build implements scheduler interface.
func (m *MultiplicativeLR) Build() *LRScheduler {
	return &LRScheduler{m}
}

// SetLRs implements scheduler interface.
func (l *MultiplicativeLR) SetLRs(epochOpt ...int) {
	epoch := -1
	if len(epochOpt) > 0 {
		epoch = epochOpt[0]
	}

	var newLRs []float64
	lrs, err := l.opt.opt.GetLearningRates()
	if err != nil {
		log.Fatal(err)
	}

	switch epoch {
	case -1, 0:
		newLRs = lrs
	default:
		for i, lr := range lrs {
			lambda := l.lrLambdas[i](epoch)
			newLR := lr * lambda
			newLRs = append(newLRs, newLR)
		}
	}

	l.opt.SetLRs(newLRs)
}

// StepLR decays the learning rates of each optimizer parameter group by gamma every
// step size epochs.
//
// NOTE. Such decay can happen simultaneously with other changes to the learning rate
// from outside this scheduler.
type StepLR struct {
	opt        *Optimizer
	stepSize   int
	gamma      float64
	initialLRs []float64
}

// NewStepLR creates a new StepLR.
func NewStepLR(opt *Optimizer, stepSize int, gamma float64) *StepLR {
	initialLRs := opt.GetLRs()
	return &StepLR{opt, stepSize, gamma, initialLRs}
}

// Build implements scheduler interface.
func (s *StepLR) Build() *LRScheduler {
	return &LRScheduler{s}
}

// SetLRs implements scheduler interface.
func (s *StepLR) SetLRs(epochOpt ...int) {
	epoch := -1
	if len(epochOpt) > 0 {
		epoch = epochOpt[0]
	}

	var newLRs []float64
	lrs, err := s.opt.opt.GetLearningRates()
	if err != nil {
		log.Fatal(err)
	}

	switch {
	case epoch == -1, epoch == 0, epoch%s.stepSize != 0:
		newLRs = lrs
	default:
		for _, lr := range lrs {
			newLR := floatRound(lr*s.gamma, 10)
			newLRs = append(newLRs, newLR)
		}
	}

	s.opt.SetLRs(newLRs)
}

// floatRound rounds float64 value to a specified precision.
// Modified from: https://stackoverflow.com/questions/18390266
func floatRound(input float64, precision int) float64 {
	roundFactor := math.Pow(10, float64(precision))
	up := input * roundFactor
	round := int(up + math.Copysign(0.5, up))

	return float64(round) / roundFactor
}

// StepLR decays the learning rates of each optimizer parameter group by gamm once
// the number of epochs reaches one of the milestones.
//
// NOTE. Such decay can happen simultaneously with other changes to the learning rate
// from outside this scheduler.
type MultiStepLR struct {
	opt        *Optimizer
	milestones []int
	gamma      float64
	initialLRs []float64
}

// NewStepLR creates a new StepLR.
func NewMultiStepLR(opt *Optimizer, milestones []int, gamma float64) *MultiStepLR {
	initialLRs := opt.GetLRs()
	return &MultiStepLR{opt, milestones, gamma, initialLRs}
}

// Build implements scheduler interface.
func (ms *MultiStepLR) Build() *LRScheduler {
	return &LRScheduler{ms}
}

// SetLRs implements scheduler interface.
func (ms *MultiStepLR) SetLRs(epochOpt ...int) {
	epoch := -1
	if len(epochOpt) > 0 {
		epoch = epochOpt[0]
	}

	var newLRs []float64
	lrs, err := ms.opt.opt.GetLearningRates()
	if err != nil {
		log.Fatal(err)
	}

	switch {
	case !contain(epoch, ms.milestones):
		newLRs = lrs
	default:
		for _, lr := range lrs {
			newLR := floatRound(lr*ms.gamma, 10)
			newLRs = append(newLRs, newLR)
		}
	}

	ms.opt.SetLRs(newLRs)
}

func contain(item int, list []int) bool {
	for _, i := range list {
		if i == item {
			return true
		}
	}

	return false
}

// ExponentialLR decays the learning rates of each optimizer parameter group by gamma every
// epochs.
type ExponentialLR struct {
	opt        *Optimizer
	gamma      float64
	initialLRs []float64
}

// NewExponentialLR creates a new ExponentialLR.
func NewExponentialLR(opt *Optimizer, gamma float64) *ExponentialLR {
	initialLRs := opt.GetLRs()
	return &ExponentialLR{opt, gamma, initialLRs}
}

// Build implements scheduler interface.
func (e *ExponentialLR) Build() *LRScheduler {
	return &LRScheduler{e}
}

// SetLRs implements scheduler interface.
func (e *ExponentialLR) SetLRs(epochOpt ...int) {
	epoch := -1
	if len(epochOpt) > 0 {
		epoch = epochOpt[0]
	}

	var newLRs []float64
	lrs, err := e.opt.opt.GetLearningRates()
	if err != nil {
		log.Fatal(err)
	}

	switch {
	case epoch == -1, epoch == 0:
		newLRs = lrs
	default:
		for _, lr := range lrs {
			newLR := floatRound(lr*e.gamma, 10)
			newLRs = append(newLRs, newLR)
		}
	}

	e.opt.SetLRs(newLRs)
}

// CosineAnnealingLR set the learning rates of each optimizer parameter group by using
// a cosine annealing schedule where eta max is set to initial learning rate and Tcur
// is the number of epochs since the last restart in SGDR (Stochastic Gradient Descent with Warm Restarts).
//
// NOTE. this implements only the cosine annealing part of SGDR, and not the starts.
// Ref.
// - https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR
// - https://arxiv.org/abs/1608.03983
type CosineAnnealingLR struct {
	opt        *Optimizer
	tmax       int     // maximal number of iteration
	etaMin     float64 // Minimum learning rate. Default = 0
	initialLRs []float64
	stepCount  int
	lastEpoch  int
}

// NewConsineAnnealingLR creates a new ConsineAnnealingLR.
func NewCosineAnnealingLR(opt *Optimizer, tmax int, etaMin float64, lastEpochOpt ...int) *CosineAnnealingLR {
	lastEpoch := -1
	if len(lastEpochOpt) > 0 {
		lastEpoch = lastEpochOpt[0]
	}
	stepCount := 0
	opt.ResetStepCount()
	initialLRs := opt.GetLRs()
	fmt.Printf("Scheduler initialLRs: %v\n", initialLRs)
	return &CosineAnnealingLR{opt, tmax, etaMin, initialLRs, stepCount, lastEpoch}
}

// Build implements scheduler interface.
func (ca *CosineAnnealingLR) Build() *LRScheduler {
	s := &LRScheduler{ca}
	s.Step()
	return s
}

// SetLRs implements scheduler interface.
func (ca *CosineAnnealingLR) SetLRs(epochOpt ...int) {
	switch len(epochOpt) {
	case 0:
		ca.lastEpoch += 1
	default:
		ca.lastEpoch = epochOpt[0]
	}

	var newLRs []float64
	lrs, err := ca.opt.opt.GetLearningRates()
	if err != nil {
		log.Fatal(err)
	}

	switch {
	case ca.lastEpoch == 0:
		newLRs = ca.initialLRs
	case (ca.lastEpoch-1-ca.tmax)%(2*ca.tmax) == 0:
		// fmt.Printf("Epoch %d: case 2(%v)\n", ca.lastEpoch, (ca.lastEpoch-1-ca.tmax)%(2*ca.tmax))
		for i, lr := range lrs {
			// group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
			newLR := lr + (ca.initialLRs[i]-ca.etaMin)*(1-math.Cos(math.Pi/float64(ca.tmax)))/2
			newLRs = append(newLRs, newLR)
		}
	default:
		// fmt.Printf("Epoch %d: case 3(%v)\n", ca.lastEpoch, (ca.lastEpoch-1-ca.tmax)%(2*ca.tmax))
		for _, lr := range lrs {
			//(1 + math.cos(math.pi * self.last_epoch / self.T_max))
			dividend := 1 + math.Cos(math.Pi*float64(ca.lastEpoch)/float64(ca.tmax))

			// (1 + math.cos(math.pi * (self.last_ca.lastEpoch - 1) / self.T_max)) * (group['lr'] - self.eta_min) + self.eta_min
			divisor := (1 + math.Cos(math.Pi*(float64(ca.lastEpoch-1)/float64(ca.tmax))))
			newLR := (dividend/divisor)*(lr-ca.etaMin) + ca.etaMin
			newLRs = append(newLRs, newLR)
		}
	}

	ca.opt.SetLRs(newLRs)
	ca.stepCount += 1
	// fmt.Printf("Epoch %d - LRs: %0.6f\n", ca.lastEpoch, ca.opt.GetLRs())
}
