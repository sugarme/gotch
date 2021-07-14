package nn

// Optimizers to be used for gradient-descent based training.

import (
	"fmt"
	"log"

	ts "github.com/sugarme/gotch/tensor"
)

// Optimizer is a struct object to run gradient descent.
type Optimizer struct {
	opt *ts.COptimizer
	// variables            Variables // having embedded sync.Mutex
	variablesInOptimizer uint8
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
func defaultBuild(config OptimizerConfig, vs *VarStore, lr float64) (retVal *Optimizer, err error) {

	opt, err := config.buildCOpt(lr)
	if err != nil {
		return retVal, err
	}

	if len(vs.Vars.TrainableVariables) > 0 {
		for _, v := range vs.Vars.TrainableVariables {
			if err = opt.AddParameter(v.Tensor, v.Group); err != nil {
				err = fmt.Errorf("Optimizer defaultBuild - AddParameter failed: %w\n", err)
				return nil, err
			}
		}
	}

	return &Optimizer{
		opt: opt,
		// variables:            vs.Vars,
		variablesInOptimizer: uint8(len(vs.Vars.TrainableVariables)),
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

	// missingVariables := len(opt.variables.TrainableVariables) - int(opt.variablesInOptimizer)
	//
	// if missingVariables > 0 {
	// var tensors []ts.Tensor
	// for _, t := range opt.variables.TrainableVariables[opt.variablesInOptimizer:] {
	// tensor := t.MustShallowClone()
	// tensor.Detach_()
	// tensors = append(tensors, tensor)
	// }
	//
	// opt.opt.AddParameters(tensors)
	// opt.variablesInOptimizer = uint8(len(opt.variables.TrainableVariables))
	// }

}

// ZeroGrad zeroes the gradient for the tensors tracked by this optimizer.
func (opt *Optimizer) ZeroGrad() {
	opt.addMissingVariables()
	if err := opt.opt.ZeroGrad(); err != nil {
		log.Fatalf("Optimizer - ZeroGrad method call error: %v\n", err)
	}
}

// Clips gradient value at some specified maximum value.
func (opt *Optimizer) ClipGradValue(max float64) {

	// opt.variables.mutex.Lock()
	// defer opt.variables.mutex.Unlock()

	// for _, tensor := range opt.variables.TrainableVariables {
	// tensor.MustGrad().Clamp_(ts.FloatScalar(-max), ts.FloatScalar(max))
	// }
}

// Step performs an optimization step, updating the tracked tensors based on their gradients.
func (opt *Optimizer) Step() {
	opt.addMissingVariables()
	err := opt.opt.Step()
	if err != nil {
		log.Fatalf("Optimizer - Step method call error: %v\n", err)
	}
	opt.stepCount += 1
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
func (opt *Optimizer) BackwardStep(loss *ts.Tensor) {
	opt.addMissingVariables()
	err := opt.opt.ZeroGrad()
	if err != nil {
		log.Fatalf("Optimizer - BackwardStep method call - ZeroGrad error: %v\n", err)
	}
	loss.MustBackward()
	err = opt.opt.Step()
	if err != nil {
		log.Fatalf("Optimizer - BackwardStep  method call - Step() error: %v\n", err)
	}
}

// BackwardStepClip applies a backward step pass, update the gradients, and performs an optimization step.
//
// The gradients are clipped based on `max` before being applied.
func (opt *Optimizer) BackwardStepClip(loss *ts.Tensor, max float64) {
	opt.addMissingVariables()
	err := opt.opt.ZeroGrad()
	if err != nil {
		log.Fatalf("Optimizer - BackwardStepClip method call - ZeroGrad error: %v\n", err)
	}
	loss.MustBackward()
	opt.ClipGradValue(max)
	err = opt.opt.Step()
	if err != nil {
		log.Fatalf("Optimizer - BackwardStepClip  method call - Step() error: %v\n", err)
	}
}

/// TODO. Clips gradient L2 norm over all trainable parameters.
//
// The norm is computed over all gradients together, as if they were
// concatenated into a single vector.
func (opt *Optimizer) ClipGradNorm(max float64) {
	// TODO.
	log.Fatalf("Not implemented yet!")
}

// TODO. Applies a backward step pass, update the gradients, and performs an optimization step.
//
// The gradients L2 norm is clipped based on `max`.
func (opt *Optimizer) BackwardStepClipNorm(loss *ts.Tensor, max float64) {
	// TODO.
	log.Fatalf("Not implemented yet!")
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

func (opt *Optimizer) AddParamGroup(tensors []ts.Tensor) {
	err := opt.opt.AddParamGroup(tensors)
	if err != nil {
		log.Fatalf("Optimizer - ParamGroupNum  method call error: %v\n", err)
	}
}
