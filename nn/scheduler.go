package nn

import (
	// "fmt"
	"fmt"
	"log"
	"math"
)

type SchedulerOptions struct {
	// Metrics   map[string]interface{}
	Loss      float64 // Usually metrics is loss value
	LastEpoch int
}
type SchedulerOption func(*SchedulerOptions)

func defaultSchedulerOptions() *SchedulerOptions {
	return &SchedulerOptions{
		// Metrics:   make(map[string]interface{}, 0),
		Loss:      math.Inf(1),
		LastEpoch: -1,
	}
}

func WithLastEpoch(epoch int) SchedulerOption {
	return func(o *SchedulerOptions) {
		o.LastEpoch = epoch
	}
}

func WithLoss(loss float64) SchedulerOption {
	return func(o *SchedulerOptions) {
		o.Loss = loss
	}
}

type scheduler interface {
	SetLRs(opts ...SchedulerOption)
	Build() *LRScheduler
}

// LRScheduler is a scheduler to update optimizer learning rates.
type LRScheduler struct {
	scheduler scheduler
}

// Step updates optimizer learning rate.
func (s *LRScheduler) Step(opts ...SchedulerOption) {
	s.scheduler.SetLRs(opts...)
}

type LambdaFn func(in interface{}) float64

// LamdaLR calculates new learning rate for each parameter group by applying
// Lambda function to the corresponding INITIAL learning rate.
type LambdaLR struct {
	opt        *Optimizer
	lrLambdas  []LambdaFn // length should be 1 or equal to length of optimizer param groups.
	initialLRs []float64
	stepCount  int
	lastEpoch  int
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

	return &LambdaLR{
		opt:        opt,
		lrLambdas:  ldFns,
		initialLRs: initialLRs,
		stepCount:  0,
		lastEpoch:  -1,
	}
}

// Build implements scheduler interface.
func (l *LambdaLR) Build() *LRScheduler {
	s := &LRScheduler{l}
	s.Step()
	return s
}

// SetLRs implements scheduler interface.
func (l *LambdaLR) SetLRs(opts ...SchedulerOption) {
	options := defaultSchedulerOptions()
	for _, o := range opts {
		o(options)
	}
	switch options.LastEpoch {
	case -1:
		l.lastEpoch += 1
	default:
		l.lastEpoch = options.LastEpoch
	}

	var newLRs []float64
	switch l.lastEpoch {
	case 0:
		newLRs = l.initialLRs
	default:
		for i, lr := range l.initialLRs {
			lambda := l.lrLambdas[i](l.lastEpoch)
			newLR := lr * lambda
			newLRs = append(newLRs, newLR)
		}
	}

	l.opt.SetLRs(newLRs)
	l.stepCount += 1
}

// MultiplicativeLR calculates new learning rates for each optimizer para groups
// by applying corresponding Lambda function to the CURRENT learning rate.
type MultiplicativeLR struct {
	opt        *Optimizer
	lrLambdas  []LambdaFn // length should be 1 or equal to length of optimizer param groups.
	initialLRs []float64
	stepCount  int
	lastEpoch  int
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
	return &MultiplicativeLR{
		opt:        opt,
		lrLambdas:  ldFns,
		initialLRs: initialLRs,
		stepCount:  0,
		lastEpoch:  -1,
	}
}

// Build implements scheduler interface.
func (m *MultiplicativeLR) Build() *LRScheduler {
	s := &LRScheduler{m}
	s.Step()
	return s
}

// SetLRs implements scheduler interface.
func (m *MultiplicativeLR) SetLRs(opts ...SchedulerOption) {
	options := defaultSchedulerOptions()
	for _, o := range opts {
		o(options)
	}
	switch options.LastEpoch {
	case -1:
		m.lastEpoch += 1
	default:
		m.lastEpoch = options.LastEpoch
	}

	var newLRs []float64
	lrs, err := m.opt.opt.GetLearningRates()
	if err != nil {
		log.Fatal(err)
	}

	switch m.lastEpoch {
	case 0:
		newLRs = m.initialLRs
	default:
		for i, lr := range lrs {
			lambda := m.lrLambdas[i](m.lastEpoch)
			newLR := lr * lambda
			newLRs = append(newLRs, newLR)
		}
	}

	m.opt.SetLRs(newLRs)
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
	stepCount  int
	lastEpoch  int
}

// NewStepLR creates a new StepLR.
func NewStepLR(opt *Optimizer, stepSize int, gamma float64) *StepLR {
	initialLRs := opt.GetLRs()
	return &StepLR{
		opt:        opt,
		stepSize:   stepSize,
		gamma:      gamma,
		initialLRs: initialLRs,
		stepCount:  0,
		lastEpoch:  -1,
	}
}

// Build implements scheduler interface.
func (s *StepLR) Build() *LRScheduler {
	sc := &LRScheduler{s}
	sc.Step()
	return sc
}

// SetLRs implements scheduler interface.
func (s *StepLR) SetLRs(opts ...SchedulerOption) {
	options := defaultSchedulerOptions()
	for _, o := range opts {
		o(options)
	}
	switch options.LastEpoch {
	case -1:
		s.lastEpoch += 1
	default:
		s.lastEpoch = options.LastEpoch
	}

	var newLRs []float64
	lrs, err := s.opt.opt.GetLearningRates()
	if err != nil {
		log.Fatal(err)
	}

	switch {
	case s.lastEpoch == 0, s.lastEpoch%s.stepSize != 0:
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
	stepCount  int
	lastEpoch  int
}

// NewStepLR creates a new StepLR.
func NewMultiStepLR(opt *Optimizer, milestones []int, gamma float64) *MultiStepLR {
	initialLRs := opt.GetLRs()
	return &MultiStepLR{
		opt:        opt,
		milestones: milestones,
		gamma:      gamma,
		initialLRs: initialLRs,
		stepCount:  0,
		lastEpoch:  -1,
	}
}

// Build implements scheduler interface.
func (ms *MultiStepLR) Build() *LRScheduler {
	s := &LRScheduler{ms}
	s.Step()
	return s
}

// SetLRs implements scheduler interface.
func (ms *MultiStepLR) SetLRs(opts ...SchedulerOption) {
	options := defaultSchedulerOptions()
	for _, o := range opts {
		o(options)
	}
	switch options.LastEpoch {
	case -1:
		ms.lastEpoch += 1
	default:
		ms.lastEpoch = options.LastEpoch
	}

	var newLRs []float64
	lrs, err := ms.opt.opt.GetLearningRates()
	if err != nil {
		log.Fatal(err)
	}

	switch {
	case !contain(ms.lastEpoch, ms.milestones):
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
	stepCount  int
	lastEpoch  int
}

// NewExponentialLR creates a new ExponentialLR.
func NewExponentialLR(opt *Optimizer, gamma float64) *ExponentialLR {
	initialLRs := opt.GetLRs()
	return &ExponentialLR{
		opt:        opt,
		gamma:      gamma,
		initialLRs: initialLRs,
		stepCount:  0,
		lastEpoch:  -1,
	}
}

// Build implements scheduler interface.
func (e *ExponentialLR) Build() *LRScheduler {
	s := &LRScheduler{e}
	s.Step()
	return s
}

// SetLRs implements scheduler interface.
func (e *ExponentialLR) SetLRs(opts ...SchedulerOption) {
	options := defaultSchedulerOptions()
	for _, o := range opts {
		o(options)
	}
	switch options.LastEpoch {
	case -1:
		e.lastEpoch += 1
	default:
		e.lastEpoch = options.LastEpoch
	}

	var newLRs []float64
	lrs, err := e.opt.opt.GetLearningRates()
	if err != nil {
		log.Fatal(err)
	}

	switch {
	case e.lastEpoch == 0:
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
func NewCosineAnnealingLR(opt *Optimizer, tmax int, etaMin float64) *CosineAnnealingLR {
	opt.ResetStepCount()
	initialLRs := opt.GetLRs()
	return &CosineAnnealingLR{
		opt:        opt,
		tmax:       tmax,
		etaMin:     etaMin,
		initialLRs: initialLRs,
		stepCount:  0,
		lastEpoch:  -1,
	}
}

// Build implements scheduler interface.
func (ca *CosineAnnealingLR) Build() *LRScheduler {
	s := &LRScheduler{ca}
	s.Step()
	return s
}

// SetLRs implements scheduler interface.
func (ca *CosineAnnealingLR) SetLRs(opts ...SchedulerOption) {
	options := defaultSchedulerOptions()
	for _, o := range opts {
		o(options)
	}
	switch options.LastEpoch {
	case -1:
		ca.lastEpoch += 1
	default:
		ca.lastEpoch = options.LastEpoch
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
		for i, lr := range lrs {
			// group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
			newLR := lr + (ca.initialLRs[i]-ca.etaMin)*(1-math.Cos(math.Pi/float64(ca.tmax)))/2
			newLRs = append(newLRs, newLR)
		}
	default:
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
}

// ReduceLROnPlateau reduces learning rate when a metric has stopped improving.
// Models often benefit from reducing the learning rate by a factor
// of 2-10 once learning stagnates. This scheduler reads a metrics
// quantity and if no improvement is seen for a 'patience' number
// of epochs, the learning rate is reduced.
type ReduceLROnPlateau struct {
	opt *Optimizer

	// One of `min` or `max`. In `min` mode, lr will be reduced
	// when the quantiy monitored has stopped DECREASING. In `max`
	// mode, it will be reduced when the quantity mornitored has stopped
	// INCREASING. Default = "min"
	mode string

	// Factor by which the learning rate will be reduced (new LR = lr * factor).
	// Default = 0.1
	factor float64

	// Number of epochs with no improvement after which learning rate
	// will be reduced. E.g., if patience = 2, then we will ignore the first
	// 2 epochs with no improvement, and wil only decrease the LR after the 3rd epoch
	// if the loss still hasn't improved then.
	// Default: 10
	patience int

	// If "true", prints a message to stdout for each update.
	// Default = false
	verbose bool

	// Threshold for measuring the new optimum to only focus on
	// significant changes.
	// Default = 1e-4
	threshold float64

	// One of `rel`, `abs`
	// - `rel`: dynamicThreshold = best * (1 + threshold) in `max` mode
	// or bet * (1 - threshold) in `min` mode
	// - `abs`: dynamicThreshold = best + threshold in `max` mode or
	// best - threshold in `min` mode.
	// Default = `rel`
	thresholdMode string

	// Number of epochs to wait before resuming normal operation after
	// LR has been reduced.
	// Default = 0
	cooldown int

	// Default = 0
	cooldownCounter int

	// A lower bound on the learning rate of all optimizer param groups.
	// If length = 1, it applies to all param groups, otherwise, it should
	// have the same legnth as optimizer learning groups.
	// Default = []float64{0}
	minLRs []float64

	// Minimal decay applied to LR. If the difference between new and old LR
	// is smaller than eps, then update is ignored.
	// Default = 1e-8
	eps float64

	// Default = modeWorse (either inf or -inf)
	best float64

	// Default = 0
	numBadEpochs int

	// The worse value for the chosen mode
	// Default = inf if mode="min" or -inf if mode="max"
	modeWorse float64

	// Default = 0
	lastEpoch int
}

type ReduceLROnPlateauOptions struct {
	Mode          string
	Factor        float64
	Patience      int
	Verbose       bool
	Threshold     float64
	ThresholdMode string
	MinLRs        []float64
	Cooldown      int
	Eps           float64
}

type ReduceLROnPlateauOption func(*ReduceLROnPlateauOptions)

func defaultReduceLROnPlateauOptions() *ReduceLROnPlateauOptions {
	return &ReduceLROnPlateauOptions{
		Mode:          "min",
		Factor:        0.1,
		Patience:      10,
		Verbose:       false,
		Threshold:     1e-4,
		ThresholdMode: "rel",
		Cooldown:      0,
		MinLRs:        []float64{0.0},
		Eps:           1e-8,
	}
}

func WithReduceOnPlateauMode(mode string) ReduceLROnPlateauOption {
	return func(o *ReduceLROnPlateauOptions) {
		o.Mode = mode
	}
}

func WithReduceOnPlateauFactor(factor float64) ReduceLROnPlateauOption {
	return func(o *ReduceLROnPlateauOptions) {
		o.Factor = factor
	}
}

func WithReduceOnPlateauPatience(patience int) ReduceLROnPlateauOption {
	return func(o *ReduceLROnPlateauOptions) {
		o.Patience = patience
	}
}

func WithReduceOnPlateauVerbose(verbose bool) ReduceLROnPlateauOption {
	return func(o *ReduceLROnPlateauOptions) {
		o.Verbose = verbose
	}
}

func WithReduceOnPlateauThreshold(threshold float64) ReduceLROnPlateauOption {
	return func(o *ReduceLROnPlateauOptions) {
		o.Threshold = threshold
	}
}

func WithReduceOnPlateauThresholdMode(thresholdMode string) ReduceLROnPlateauOption {
	return func(o *ReduceLROnPlateauOptions) {
		o.ThresholdMode = thresholdMode
	}
}

func WithReduceOnPlateauEps(eps float64) ReduceLROnPlateauOption {
	return func(o *ReduceLROnPlateauOptions) {
		o.Eps = eps
	}
}

func WithReduceOnPlateauMinLRs(minLRs []float64) ReduceLROnPlateauOption {
	return func(o *ReduceLROnPlateauOptions) {
		o.MinLRs = minLRs
	}
}

func WithReduceOnPlateauCooldown(cooldown int) ReduceLROnPlateauOption {
	return func(o *ReduceLROnPlateauOptions) {
		o.Cooldown = cooldown
	}
}

func NewReduceLROnPlateau(opt *Optimizer, opts ...ReduceLROnPlateauOption) *ReduceLROnPlateau {
	options := defaultReduceLROnPlateauOptions()
	for _, o := range opts {
		o(options)
	}

	// Validate input parameters
	if options.Mode != "min" && options.Mode != "max" {
		log.Fatalf("Invalid 'mode'. Mode should be either 'min' or 'max', got %v\n", options.Mode)
	}
	if options.Factor >= 1.0 {
		log.Fatalf("Factor should be < 1.0. Got %v\n", options.Factor)
	}

	if options.ThresholdMode != "rel" && options.ThresholdMode != "abs" {
		log.Fatalf("Invalide threshold mode. Should be 'rel' or 'abs'. Got %v\n", options.ThresholdMode)
	}

	var modeWorse float64
	switch options.Mode {
	case "min":
		modeWorse = math.Inf(1) // inf
	case "max":
		modeWorse = math.Inf(-1) // -inf
	}

	ngroup := opt.ParamGroupNum()
	var minLRs []float64 = make([]float64, ngroup)
	switch len(options.MinLRs) {
	case 1:
		for i := 0; i < ngroup; i++ {
			minLRs[i] = options.MinLRs[0]
		}
	case ngroup:
		minLRs = options.MinLRs
	default:
		log.Fatalf("MinLRs should have length of 1 or the same length as optimizer param groups. Got %v\n", len(options.MinLRs))
	}

	return &ReduceLROnPlateau{
		opt:             opt,
		mode:            options.Mode,
		factor:          options.Factor,
		patience:        options.Patience,
		verbose:         options.Verbose,
		threshold:       options.Threshold,
		thresholdMode:   options.ThresholdMode,
		cooldown:        options.Cooldown,
		cooldownCounter: 0,
		minLRs:          minLRs,
		eps:             options.Eps,

		best:         modeWorse,
		numBadEpochs: 0,
		modeWorse:    modeWorse,
		lastEpoch:    0,
	}
}

// Reset number of bad epochs counter and cooldown counter
func (s *ReduceLROnPlateau) reset() {
	s.best = s.modeWorse
	s.cooldownCounter = 0
	s.numBadEpochs = 0
}

func (s *ReduceLROnPlateau) inCooldown() bool {
	return s.cooldownCounter > 0
}

// Evaluates whether the metrics (loss) is better than current (best) value.
func (s *ReduceLROnPlateau) isBetter(a, best float64) bool {
	switch {
	case s.mode == "min" && s.thresholdMode == "rel":
		relEpsilon := 1.0 - s.threshold
		return a < best*relEpsilon

	case s.mode == "min" && s.thresholdMode == "abs":
		return a < best-s.threshold

	case s.mode == "max" && s.thresholdMode == "rel":
		relEpsilon := s.threshold + 1
		return a > best*relEpsilon
	default: // mode == "max" && thresholdMode == "abs"
		return a > best+s.threshold
	}
}

func (s *ReduceLROnPlateau) reduceLRs(epoch int) {
	oldLRs := s.opt.GetLRs()

	var newLRs []float64 = oldLRs
	for i, oldLR := range oldLRs {
		newLR := floatMax(oldLR*s.factor, s.minLRs[i])
		if oldLR-newLR > s.eps {
			newLRs[i] = newLR
			if s.verbose {
				fmt.Printf("Epoch %06d: Reducing learning rate of param groups %d to %0.4e\n", epoch, i, newLR)
			}
		}
	}

	s.opt.SetLRs(newLRs)
}

// SetLRs implements scheduler interface.
func (s *ReduceLROnPlateau) SetLRs(opts ...SchedulerOption) {
	options := defaultSchedulerOptions()
	for _, o := range opts {
		o(options)
	}

	switch options.LastEpoch {
	case -1:
		s.lastEpoch += 1
	default:
		s.lastEpoch = options.LastEpoch
	}

	switch s.isBetter(options.Loss, s.best) {
	case true:
		s.best = options.Loss
		s.numBadEpochs = 0
	case false:
		s.numBadEpochs += 1
	}

	if s.inCooldown() {
		s.cooldownCounter -= 1
		s.numBadEpochs = 0 // ignore any bad epochs in cooldown
	}

	if s.numBadEpochs > s.patience {
		s.reduceLRs(s.lastEpoch)
		s.cooldownCounter = s.cooldown
		s.numBadEpochs = 0
	}
}

// Build implements scheduler interface.
func (s *ReduceLROnPlateau) Build() *LRScheduler {
	return &LRScheduler{s}
}

func floatMax(v1, v2 float64) float64 {
	if v1 >= v2 {
		return v1
	}
	return v2
}

// CyclicLR sets the learning rate of each parameter group according to
// cyclical learning rate policy (CLR). The policy cycles the learning
// rate between two boundaries with a constant frequency, as detailed in
// the paper `Cyclical Learning Rates for Training Neural Networks`_.
// The distance between the two boundaries can be scaled on a per-iteration
// or per-cycle basis.
//
// Cyclical learning rate policy changes the learning rate after every batch.
// `Step()` should be called after a batch has been used for training.
// This class has three built-in policies, as put forth in the paper:
// - "triangular": A basic triangular cycle without amplitude scaling.
// - "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
// - "exp_range": A cycle that scales initial amplitude by :math:`\text{gamma}^{\text{cycle iterations}}`
// at each cycle iteration.
//
// Source:
// - Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
// - bckenstler/CLR: https://github.com/bckenstler/CLR
type CyclicLR struct {
	// optimizer (Optimizer): Wrapped optimizer.
	opt *Optimizer

	// base_lr (float or list): Initial learning rate which is the
	// lower boundary in the cycle for each parameter group.
	initialLRs []float64

	// Upper learning rate boundaries in the cycle
	// for each parameter group. Functionally,
	// it defines the cycle amplitude (max_lr - base_lr).
	// The lr at any cycle is the sum of base_lr
	// and some scaling of the amplitude; therefore
	// max_lr may not actually be reached depending on
	// scaling function.
	maxLRs []float64

	// Number of training iterations in the
	// increasing half of a cycle.
	// Default: 2000
	stepSizeUp int

	// Number of training iterations in the
	// decreasing half of a cycle. If stepSizeDown is -1,
	// it is set to stepSizeUp.
	// Default: -1
	stepSizeDown int

	// One of {triangular, triangular2, exp_range}.
	// Values correspond to policies detailed above.
	// If scaleFn is not None, this argument is ignored.
	// Default: 'triangular'
	mode string

	// Constant in 'expRange' scaling function:
	// gamma**(cycle iterations)
	// Default: 1.0
	gamma float64

	// Custom scaling policy defined by a single
	// argument lambda function, where
	// 0 <= scaleFn(x) <= 1 for all x >= 0.
	// If specified, then 'mode' is ignored.
	// Default: nil
	scaleFn func(x float64) float64

	// One of {'cycle', 'iterations'}.
	// Defines whether scale_fn is evaluated on
	// cycle number or cycle iterations (training
	// iterations since start of cycle).
	// Default: 'cycle'
	scaleMode string

	// If `true`, momentum is cycled inversely
	// to learning rate between 'baseMomentum' and 'maxMomentum'.
	// Default: true
	cycleMomentum bool

	// Lower momentum boundaries in the cycle
	// for each parameter group. Note that momentum is cycled inversely
	// to learning rate; at the peak of a cycle, momentum is
	// 'baseMomentum' and learning rate is 'maxLR'.
	// Default: 0.8
	baseMomentums []float64

	// Upper momentum boundaries in the cycle
	// for each parameter group. Functionally,
	// it defines the cycle amplitude (maxMomentum - baseMomentum).
	// The momentum at any cycle is the difference of maxMomentum
	// and some scaling of the amplitude; therefore
	// baseMomentum may not actually be reached depending on
	// scaling function. Note that momentum is cycled inversely
	// to learning rate; at the start of a cycle, momentum is 'maxMomentum'
	// and learning rate is 'baseLR'
	// Default: 0.9
	maxMomentums []float64

	// The index of the last batch. This parameter is used when
	// resuming a training job. Since `Step()` should be invoked after each
	// batch instead of after each epoch, this number represents the total
	// number of *batches* computed, not the total number of epochs computed.
	// When lastEpoch=-1, the schedule is started from the beginning.
	// Default: -1
	lastEpoch int

	totalSize int
	stepRatio float64
}

type CyclicOptions struct {
	StepSizeUp    int                     // 2000
	StepSizeDown  int                     // -1
	Mode          string                  // "triangular"
	Gamma         float64                 // 1.0
	ScaleFn       func(x float64) float64 // nil
	ScaleMode     string                  // "cycle"
	CycleMomentum bool                    // true
	BaseMomentum  float64                 // 0.8
	MaxMomentum   float64                 // 0.9
	LastEpoch     int                     // -1
}

type CyclicOption func(*CyclicOptions)

func defaultCyclicOptions() *CyclicOptions {
	return &CyclicOptions{
		StepSizeUp:    2000,
		StepSizeDown:  -1,
		Mode:          "triangular",
		Gamma:         1.0,
		ScaleFn:       nil,
		ScaleMode:     "cycle",
		CycleMomentum: true,
		BaseMomentum:  0.8,
		MaxMomentum:   0.9,
		LastEpoch:     -1,
	}
}

func WithCyclicStepSizeUp(v int) CyclicOption {
	return func(o *CyclicOptions) {
		o.StepSizeUp = v
	}
}

func WithCyclicStepSizeDown(v int) CyclicOption {
	return func(o *CyclicOptions) {
		o.StepSizeDown = v
	}
}

func WithCyclicMode(v string) CyclicOption {
	return func(o *CyclicOptions) {
		o.Mode = v
	}
}

func WithCyclicGamma(v float64) CyclicOption {
	return func(o *CyclicOptions) {
		o.Gamma = v
	}
}

func WithCyclicScaleFn(v func(x float64) float64) CyclicOption {
	return func(o *CyclicOptions) {
		o.ScaleFn = v
	}
}

func WithCyclicScaleMode(v string) CyclicOption {
	return func(o *CyclicOptions) {
		o.ScaleMode = v
	}
}

func WithCyclicCycleMomentum(v bool) CyclicOption {
	return func(o *CyclicOptions) {
		o.CycleMomentum = v
	}
}

func WithCyclicBaseMomentum(v float64) CyclicOption {
	return func(o *CyclicOptions) {
		o.BaseMomentum = v
	}
}

func WithCyclicMaxMomentum(v float64) CyclicOption {
	return func(o *CyclicOptions) {
		o.MaxMomentum = v
	}
}

func WithCyclicLastEpoch(v int) CyclicOption {
	return func(o *CyclicOptions) {
		o.LastEpoch = v
	}
}

func NewCyclicLR(opt *Optimizer, baseLRs, maxLRs []float64, opts ...CyclicOption) *CyclicLR {
	options := defaultCyclicOptions()
	for _, o := range opts {
		o(options)
	}

	var cyc *CyclicLR = new(CyclicLR)

	initialLRs := formatParam(opt, baseLRs, "baseLRs")
	if options.LastEpoch == -1 {
		opt.SetLRs(initialLRs)
	}
	cyc.initialLRs = initialLRs

	cyc.opt = opt
	cyc.maxLRs = formatParam(opt, maxLRs, "maxLRs")

	var stepSizeDown int
	switch options.StepSizeDown {
	case -1:
		stepSizeDown = options.StepSizeUp
	default:
		stepSizeDown = options.StepSizeDown
	}
	cyc.stepSizeUp = options.StepSizeUp
	cyc.stepSizeDown = stepSizeDown

	totalSize := stepSizeDown + options.StepSizeUp
	stepRatio := float64(options.StepSizeUp) / float64(totalSize)
	cyc.totalSize = totalSize
	cyc.stepRatio = stepRatio

	if !strContain([]string{"triangular", "triangular2", "exp_range"}, options.Mode) && options.ScaleFn == nil {
		log.Fatalf("Invalide 'mode': %v and scale function is nil\n", options.Mode)
	}
	cyc.mode = options.Mode
	cyc.gamma = options.Gamma

	switch options.ScaleFn {
	case nil:
		switch cyc.mode {
		case "triangular":
			cyc.scaleFn = func(x float64) float64 {
				return 1.0
			}
			cyc.scaleMode = "cycle"

		case "triangular2":
			cyc.scaleFn = func(x float64) float64 {
				return 1 / (math.Pow(2.0, (x - 1.0)))
			}
			cyc.scaleMode = "cycle"
		case "ex_range":
			cyc.scaleFn = func(x float64) float64 {
				return math.Pow(cyc.gamma, x)
			}
			cyc.scaleMode = "iterations"
		}

	default:
		cyc.scaleFn = options.ScaleFn
		cyc.scaleMode = options.ScaleMode
	}

	cyc.cycleMomentum = options.CycleMomentum
	if cyc.cycleMomentum {
		// if optimizer doesn't have momentum, throw error
		// TODO. type casting optimizer.config and check
		cyc.baseMomentums = formatParam(opt, []float64{options.BaseMomentum}, "baseMomentum")
		if options.LastEpoch == -1 {
			opt.SetMomentum(options.BaseMomentum)
		}
		cyc.maxMomentums = formatParam(opt, []float64{options.MaxMomentum}, "maxMomentum")
	}

	return cyc
}

func strContain(items []string, item string) bool {
	for _, i := range items {
		if i == item {
			return true
		}
	}

	return false
}

func formatParam(opt *Optimizer, param []float64, paramName string) []float64 {
	ngroup := opt.ParamGroupNum()
	var paramOut []float64 = make([]float64, ngroup)
	switch len(param) {
	case 1:
		for i := 0; i < ngroup; i++ {
			paramOut[i] = param[0]
		}
	case ngroup:
		paramOut = param
	default:
		log.Fatalf("Length of %s should be either 1 or equal to number of param groups. Got %v\n", paramName, len(param))
	}

	return paramOut
}

// SetLRs implements scheduler interface.
//
// It calculates the learning rate at batch index. This function treats
// `lastEpoch` as the last batch index.
// NOTE. If `cycleMomentum` is ``true``, this function has a side effect of
// updating the optimizer's momentum.
func (cyc *CyclicLR) SetLRs(opts ...SchedulerOption) {
	options := defaultSchedulerOptions()
	for _, o := range opts {
		o(options)
	}
	switch options.LastEpoch {
	case -1:
		cyc.lastEpoch += 1
	default:
		cyc.lastEpoch = options.LastEpoch
	}

	cycle := math.Floor(1.0 + float64(cyc.lastEpoch)/float64(cyc.totalSize))
	x := 1.0 + float64(cyc.lastEpoch)/float64(cyc.totalSize) - cycle

	var scaleFactor float64
	switch {
	case x <= cyc.stepRatio:
		scaleFactor = x / cyc.stepRatio
	default:
		scaleFactor = (x - 1.0) / (cyc.stepRatio - 1.0)
	}

	ngroup := cyc.opt.ParamGroupNum()
	var newLRs []float64 = make([]float64, ngroup)
	for i := 0; i < ngroup; i++ {
		baseLR := cyc.initialLRs[i]
		maxLR := cyc.maxLRs[i]
		baseHeight := (maxLR - baseLR) * scaleFactor
		var newLR float64
		switch cyc.scaleMode {
		case "cycle":
			newLR = baseLR + baseHeight*cyc.scaleFn(cycle)
		default:
			newLR = baseLR + baseHeight*cyc.scaleFn(float64(cyc.lastEpoch))
		}

		newLRs[i] = newLR
	}

	// Update optimizer learning rates.
	cyc.opt.SetLRs(newLRs)

	// Update optimizer momentum.
	// NOTE. for now, we just assuming there's 1 param group and will update momentum for such param group.
	if cyc.cycleMomentum {
		var momentum float64
		baseMomentum, maxMomentum := cyc.baseMomentums[0], cyc.maxMomentums[0]
		baseHeight := (maxMomentum - baseMomentum) * scaleFactor
		switch cyc.scaleMode {
		case "cycle":
			momentum = maxMomentum - baseHeight*cyc.scaleFn(cycle)
		default:
			momentum = maxMomentum - baseHeight*cyc.scaleFn(float64(cyc.lastEpoch))
		}
		cyc.opt.SetMomentum(momentum)
	}
}

// Build implements scheduler interface.
func (cyc *CyclicLR) Build() *LRScheduler {
	return &LRScheduler{cyc}
}
