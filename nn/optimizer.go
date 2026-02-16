package nn

// Optimizers to be used for gradient-descent based training.

import (
	"fmt"
	"log"
	"math"

	"github.com/sugarme/gotch/ts"
)

// Optimizer is a struct object to run gradient descent.
type Optimizer struct {
	varstore *VarStore
	opt      *ts.COptimizer
	// variablesInOptimizer uint8
	variablesInOptimizer map[string]struct{}
	config               interface{}
	stepCount            int
}

// OptimizerConfig defines Optimizer configurations. These configs can be used to build optimizer.
type OptimizerConfig interface {
	buildCOpt(lr float64) (*ts.COptimizer, error)

	// Build builds an optimizer with the specified learning rate handling variables stored in `vs`.
	//
	// NOTE: Build is a 'default' method. It can be called by wrapping
	// 'DefaultBuild' function
	// E.g. AdamOptimizerConfig struct have a method to fullfil `Build` method of
	// OptimizerConfig by wrapping `DefaultBuild` like
	// (config AdamOptimizerConfig) Build(vs VarStore, lr float64) (retVal Optimizer, err error){
	//		return defaultBuild(config, vs, lr)
	// }
	Build(vs *VarStore, lr float64) (*Optimizer, error)
}

// defaultBuild is `default` Build method for OptimizerConfig interface
func defaultBuild(config OptimizerConfig, vs *VarStore, lr float64) (*Optimizer, error) {
	opt, err := config.buildCOpt(lr)
	if err != nil {
		return nil, err
	}

	names := make(map[string]struct{})
	for name, v := range vs.vars {
		if v.Trainable {
			if err = opt.AddParameter(v.Tensor, v.Group); err != nil {
				err = fmt.Errorf("Optimizer defaultBuild - AddParameter failed: %w\n", err)
				return nil, err
			}
		}
		names[name] = struct{}{}
	}

	return &Optimizer{
		varstore:             vs,
		opt:                  opt,
		variablesInOptimizer: names,
		config:               config,
		stepCount:            0,
	}, nil
}

// SGD Optimizer:
//===============

// SGDConfig holds parameters for building the SGD (Stochastic Gradient Descent) optimizer.
type SGDConfig struct {
	Momentum  float64
	Dampening float64
	Wd        float64
	Nesterov  bool
}

// DefaultSGDConfig creates SGDConfig with default values.
func DefaultSGDConfig() *SGDConfig {
	return &SGDConfig{
		Momentum:  0.0,
		Dampening: 0.0,
		Wd:        0.0,
		Nesterov:  false,
	}
}

// NewSGD creates the configuration for a SGD optimizer with specified values
func NewSGDConfig(momentum, dampening, wd float64, nesterov bool) *SGDConfig {
	return &SGDConfig{
		Momentum:  momentum,
		Dampening: dampening,
		Wd:        wd,
		Nesterov:  nesterov,
	}
}

// Implement OptimizerConfig interface for SGDConfig
func (c *SGDConfig) buildCOpt(lr float64) (*ts.COptimizer, error) {
	return ts.Sgd(lr, c.Momentum, c.Dampening, c.Wd, c.Nesterov)
}

func (c *SGDConfig) Build(vs *VarStore, lr float64) (*Optimizer, error) {
	return defaultBuild(c, vs, lr)
}

// Adam optimizer:
// ===============

type AdamConfig struct {
	Beta1 float64
	Beta2 float64
	Wd    float64
}

// DefaultAdamConfig creates AdamConfig with default values
func DefaultAdamConfig() *AdamConfig {
	return &AdamConfig{
		Beta1: 0.9,
		Beta2: 0.999,
		Wd:    0.0,
	}
}

// NewAdamConfig creates AdamConfig with specified values
func NewAdamConfig(beta1, beta2, wd float64) *AdamConfig {
	return &AdamConfig{
		Beta1: beta1,
		Beta2: beta2,
		Wd:    wd,
	}
}

// Implement OptimizerConfig interface for AdamConfig
func (c *AdamConfig) buildCOpt(lr float64) (*ts.COptimizer, error) {
	return ts.Adam(lr, c.Beta1, c.Beta2, c.Wd)
}

func (c *AdamConfig) Build(vs *VarStore, lr float64) (*Optimizer, error) {
	return defaultBuild(c, vs, lr)
}

// AdamW optimizer:
// ===============

type AdamWConfig struct {
	Beta1 float64
	Beta2 float64
	Wd    float64
}

// DefaultAdamWConfig creates AdamWConfig with default values
func DefaultAdamWConfig() *AdamWConfig {
	return &AdamWConfig{
		Beta1: 0.9,
		Beta2: 0.999,
		Wd:    0.01,
	}
}

// NewAdamWConfig creates AdamWConfig with specified values
func NewAdamWConfig(beta1, beta2, wd float64) *AdamWConfig {
	return &AdamWConfig{
		Beta1: beta1,
		Beta2: beta2,
		Wd:    wd,
	}
}

// Implement OptimizerConfig interface for AdamWConfig
func (c *AdamWConfig) buildCOpt(lr float64) (*ts.COptimizer, error) {
	return ts.AdamW(lr, c.Beta1, c.Beta2, c.Wd)
}

// Build builds AdamW optimizer
func (c *AdamWConfig) Build(vs *VarStore, lr float64) (*Optimizer, error) {
	return defaultBuild(c, vs, lr)
}

// RMSProp optimizer:
// ===============

type RMSPropConfig struct {
	Alpha    float64
	Eps      float64
	Wd       float64
	Momentum float64
	Centered bool
}

// DefaultAdamConfig creates AdamConfig with default values
func DefaultRMSPropConfig() *RMSPropConfig {
	return &RMSPropConfig{
		Alpha:    0.99,
		Eps:      1e-8,
		Wd:       0.0,
		Momentum: 0.0,
		Centered: false,
	}
}

// NewRMSPropConfig creates RMSPropConfig with specified values
func NewRMSPropConfig(alpha, eps, wd, momentum float64, centered bool) *RMSPropConfig {
	return &RMSPropConfig{
		Alpha:    alpha,
		Eps:      eps,
		Wd:       wd,
		Momentum: momentum,
		Centered: centered,
	}
}

// Implement OptimizerConfig interface for RMSPropConfig
func (c *RMSPropConfig) buildCOpt(lr float64) (*ts.COptimizer, error) {
	return ts.RmsProp(lr, c.Alpha, c.Eps, c.Wd, c.Momentum, c.Centered)
}

func (c *RMSPropConfig) Build(vs *VarStore, lr float64) (*Optimizer, error) {
	return defaultBuild(c, vs, lr)
}

// Optimizer methods:
// ==================

func (opt *Optimizer) addMissingVariables() {
	type param struct {
		tensor *ts.Tensor
		group  uint
	}
	trainables := make(map[string]param)
	for name, v := range opt.varstore.vars {
		if v.Trainable {
			trainables[name] = param{tensor: v.Tensor, group: v.Group}
		}
	}
	missingVariables := len(trainables) - len(opt.variablesInOptimizer)
	if missingVariables > 0 {
		log.Println("INFO: Optimizer.addMissingVariables()...")
		for name, x := range trainables {
			if _, ok := opt.variablesInOptimizer[name]; !ok {
				opt.opt.AddParameter(x.tensor, x.group)
				opt.variablesInOptimizer[name] = struct{}{}
			}
		}
	}
}

// ZeroGrad zeroes the gradient for the tensors tracked by this optimizer.
func (opt *Optimizer) ZeroGrad() error {
	if err := opt.opt.ZeroGrad(); err != nil {
		err = fmt.Errorf("Optimizer.ZeroGrad() failed: %w\n", err)
		return err
	}
	return nil
}

// MustZeroGrad zeroes the gradient for the tensors tracked by this optimizer.
func (opt *Optimizer) MustZeroGrad() {
	err := opt.ZeroGrad()
	if err != nil {
		log.Fatal(err)
	}
}

// Clips gradient value at some specified maximum value.
func (opt *Optimizer) ClipGradValue(max float64) {
	opt.varstore.Lock()
	defer opt.varstore.Unlock()

	for _, v := range opt.varstore.vars {
		if v.Trainable {
			// v.Tensor.MustGrad().Clamp_(ts.FloatScalar(-max), ts.FloatScalar(max))
			gradTs := v.Tensor.MustGrad(false)
			gradTs.Clamp_(ts.FloatScalar(-max), ts.FloatScalar(max))
		}
	}
}

// Step performs an optimization step, updating the tracked tensors based on their gradients.
func (opt *Optimizer) Step() error {
	err := opt.opt.Step()
	if err != nil {
		err = fmt.Errorf("Optimizer.Step() failed: %w\n", err)
		return err
	}
	opt.stepCount += 1

	return nil
}

// MustStep performs an optimization step, updating the tracked tensors based on their gradients.
func (opt *Optimizer) MustStep() {
	err := opt.Step()
	if err != nil {
		log.Fatal(err)
	}
}

// ResetStepCount set step count to zero.
func (opt *Optimizer) ResetStepCount() {
	opt.stepCount = 0
}

// StepCount get current step count.
func (opt *Optimizer) StepCount() int {
	return opt.stepCount
}

// BackwardStep applies a backward step pass, update the gradients, and performs an optimization step.
func (opt *Optimizer) BackwardStep(loss *ts.Tensor) error {
	// PyTorch 2.10.0: Ensure gradients enabled before backward pass
	// optimizer.step() disables them and doesn't restore, so we must manage state
	ts.MustGradSetEnabled(true)

	err := opt.opt.ZeroGrad()
	if err != nil {
		err = fmt.Errorf("Optimizer.BackwardStep() failed: %w\n", err)
		return err
	}

	loss.MustBackward()
	err = opt.opt.Step()
	if err != nil {
		err = fmt.Errorf("Optimizer.BackwardStep() failed: %w\n", err)
		return err
	}

	// PyTorch 2.10.0: step() disables gradients, so re-enable them
	// BackwardStep is a training operation, gradients must remain enabled
	ts.MustGradSetEnabled(true)

	return nil
}

// MustBackwardStep applies a backward step pass, update the gradients, and performs an optimization step.
func (opt *Optimizer) MustBackwardStep(loss *ts.Tensor) {
	err := opt.BackwardStep(loss)
	if err != nil {
		log.Fatal(err)
	}
}

// BackwardStepClip applies a backward step pass, update the gradients, and performs an optimization step.
//
// The gradients are clipped based on `max` before being applied.
func (opt *Optimizer) BackwardStepClip(loss *ts.Tensor, max float64) error {
	err := opt.opt.ZeroGrad()
	if err != nil {
		err = fmt.Errorf("Optimizer.BackwardStepClip() failed: %w\n", err)
		return err
	}
	loss.MustBackward()
	opt.ClipGradValue(max)
	err = opt.opt.Step()
	if err != nil {
		err = fmt.Errorf("Optimizer.BackwardStepClip() failed: %w\n", err)
		return err
	}
	return nil
}

// MustBackwardStepClip applies a backward step pass, update the gradients, and performs an optimization step.
//
// The gradients are clipped based on `max` before being applied.
func (opt *Optimizer) MustBackwardStepClip(loss *ts.Tensor, max float64) {
	err := opt.BackwardStepClip(loss, max)
	if err != nil {
		log.Fatal(err)
	}
}

type ClipOpts struct {
	NormType         float64
	ErrorIfNonFinite bool
}

type ClipOpt func(*ClipOpts)

func defaultClipOpts() *ClipOpts {
	return &ClipOpts{
		NormType:         2.0,
		ErrorIfNonFinite: false, // will switch to "true" in the future.
	}
}

func WithNormType(v float64) ClipOpt {
	return func(o *ClipOpts) {
		o.NormType = v
	}
}

func WithErrorIfNonFinite(v bool) ClipOpt {
	return func(o *ClipOpts) {
		o.ErrorIfNonFinite = v
	}
}

// / Clips gradient L2 norm over all trainable parameters.
//
// The norm is computed over all gradients together, as if they were
// concatenated into a single vector.
//
// / Args:
// - max: max norm of the gradient
// - o.NormType. Type of the used p-norm, can be "inf" for infinity norm. Default= 2.0
// - o.ErrorIfNonFinite bool. If true, throw error if total norm of the gradients from paramters is "nan", "inf" or "-inf". Default=false
// Returns: total norm of the parameters (viewed as a single vector)
// ref. https://github.com/pytorch/pytorch/blob/cb4aeff7d8e4c70bb638cf159878c5204d0cc2da/torch/nn/utils/clip_grad.py#L59
func (opt *Optimizer) ClipGradNorm(max float64, opts ...ClipOpt) error {
	o := defaultClipOpts()
	for _, option := range opts {
		option(o)
	}

	opt.varstore.Lock()
	defer opt.varstore.Unlock()
	parameters := opt.varstore.TrainableVariables()
	if len(parameters) == 0 {
		// return ts.MustOfSlice([]float64{0.0}), nil
		return nil
	}

	var (
		norms     []*ts.Tensor
		totalNorm *ts.Tensor
	)

	device := opt.varstore.device

	// FIXME. What about mixed-precision?
	dtype := parameters[0].DType()

	if o.NormType == math.Inf(1) {
		for _, v := range opt.varstore.vars {
			n := v.Tensor.MustGrad(false).MustDetach(true).MustAbs(true).MustMax(true).MustTo(device, true)
			norms = append(norms, n)
		}
		// total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
		totalNorm = ts.MustStack(norms, 0).MustMax(true)
	} else {
		for _, v := range opt.varstore.vars {
			// x := v.Tensor.MustGrad(false).MustNorm(true)

			// NOTE. tensor.Norm() is going to be deprecated. So use linalg_norm
			// Ref. https://pytorch.org/docs/stable/generated/torch.linalg.norm.html#torch.linalg.norm
			x := v.Tensor.MustGrad(false).MustDetach(true).MustLinalgNorm(ts.FloatScalar(o.NormType), nil, false, dtype, true)
			norms = append(norms, x)
		}
	}

	// totalNorm = ts.MustStack(norms, 0).MustNorm(true).MustAddScalar(ts.FloatScalar(1e-6), true)
	// total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
	totalNorm = ts.MustStack(norms, 0).MustLinalgNorm(ts.FloatScalar(o.NormType), nil, false, dtype, true)
	for _, x := range norms {
		x.MustDrop()
	}

	totalNormVal := totalNorm.Float64Values(true)[0]
	//  if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
	if o.ErrorIfNonFinite && (math.IsNaN(totalNormVal) || math.IsInf(totalNormVal, 1)) {
		err := fmt.Errorf("The total norm of order (%v) for gradients from 'parameters' is non-finite, so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, set option.ErrorIfNonFinite= false", o.NormType)
		return err
	}

	// clip_coef = max_norm / (total_norm + 1e-6)
	// clipCoefTs := ts.TensorFrom([]float64{max}).MustDiv(totalNorm, true)
	clipCoef := max / (totalNormVal + 1e-6)
	// NOTE: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
	// avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
	// when the gradients do not reside in CPU memory.
	// clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
	if clipCoef > 1.0 {
		clipCoef = 1.0
	}
	for _, v := range opt.varstore.vars {
		if v.Trainable {
			// p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
			// v.Tensor.MustGrad(false).MustDetach(true).MustMulScalar_(ts.FloatScalar(clipCoef))
			v.Tensor.MustGrad(false).MustMulScalar_(ts.FloatScalar(clipCoef))
		}
	}

	return nil
}

// BackwardStepClipNorm applies a backward step pass, update the gradients, and performs an optimization step.
//
// The gradients L2 norm is clipped based on `max`.
func (opt *Optimizer) BackwardStepClipNorm(loss *ts.Tensor, max float64, opts ...ClipOpt) error {
	err := opt.opt.ZeroGrad()
	if err != nil {
		err := fmt.Errorf("Optimizer.BackwardStepClipNorm() failed: %w\n", err)
		return err
	}
	err = loss.Backward()
	if err != nil {
		err := fmt.Errorf("Optimizer.BackwardStepClipNorm() failed: %w\n", err)
		return err
	}

	err = opt.ClipGradNorm(max, opts...)
	if err != nil {
		err := fmt.Errorf("Optimizer.BackwardStepClipNorm() failed: %w\n", err)
		return err
	}

	err = opt.Step()
	if err != nil {
		err := fmt.Errorf("Optimizer.BackwardStepClipNorm() failed: %w\n", err)
		return err
	}

	return nil
}

// MustBackwardStepClipNorm applies a backward step pass, update the gradients, and performs an optimization step.
//
// The gradients L2 norm is clipped based on `max`.
func (opt *Optimizer) MustBackwardStepClipNorm(loss *ts.Tensor, max float64, opts ...ClipOpt) {
	err := opt.BackwardStepClipNorm(loss, max, opts...)
	if err != nil {
		log.Fatal(err)
	}
}

// SetLR sets the optimizer learning rate.
//
// NOTE. it sets a SINGLE value of learning rate for all parameter groups.
// Most of the time, there's one parameter group.
func (opt *Optimizer) SetLR(lr float64) {
	err := opt.opt.SetLearningRate(lr)
	if err != nil {
		log.Fatalf("Optimizer - SetLR  method call error: %v\n", err)
	}
}

func (opt *Optimizer) GetLRs() []float64 {
	lrs, err := opt.opt.GetLearningRates()
	if err != nil {
		log.Fatalf("Optimizer - GetLRs  method call error: %v\n", err)
	}

	return lrs
}

// SetLRs sets learning rates for ALL parameter groups respectively.
func (opt *Optimizer) SetLRs(lrs []float64) {
	err := opt.opt.SetLearningRates(lrs)
	if err != nil {
		log.Fatalf("Optimizer - SetLRs  method call error: %v\n", err)
	}
}

// SetMomentum sets the optimizer momentum.
func (opt *Optimizer) SetMomentum(m float64) {
	err := opt.opt.SetMomentum(m)
	if err != nil {
		log.Fatalf("Optimizer - SetMomentum  method call error: %v\n", err)
	}
}

func (opt *Optimizer) ParamGroupNum() int {
	ngroup, err := opt.opt.ParamGroupNum()
	if err != nil {
		log.Fatalf("Optimizer - ParamGroupNum  method call error: %v\n", err)
	}

	return int(ngroup)
}

func (opt *Optimizer) AddParamGroup(tensors []*ts.Tensor) {
	err := opt.opt.AddParamGroup(tensors)
	if err != nil {
		log.Fatalf("Optimizer - ParamGroupNum  method call error: %v\n", err)
	}
}
