package nn

// Optimizers to be used for gradient-descent based training.

import (
	// "github.com/sugarme/gotch"
	"log"

	ts "github.com/sugarme/gotch/tensor"
)

// Optimizer is a struct object to run gradient descent.
type Optimizer struct {
	opt                  ts.COptimizer
	variables            Variables // having embedded sync.Mutex
	variablesInOptimizer uint8
	config               interface{}
}

// OptimizerConfig defines Optimizer configurations. These configs can be used to build optimizer.
type OptimizerConfig interface {
	BuildCOpt(lr float64) (retVal ts.COptimizer, err error)

	// Build builds an optimizer with the specified learning rate handling variables stored in `vs`.
	//
	// NOTE: Build is a 'default' method. It can be called by wrapping
	// 'DefaultBuild' function
	// E.g. AdamOptimizerConfig struct have a method to fullfil `Build` method of
	// OptimizerConfig by wrapping `DefaultBuild` like
	// (config AdamOptimizerConfig) Build(vs VarStore, lr float64) (retVal Optimizer, err error){
	//		return defaultBuild(config, vs, lr)
	// }
	Build(vs VarStore, lr float64) (retVal Optimizer, err error)
}

// defaultBuild is `default` Build method for OptimizerConfig interface
func defaultBuild(config OptimizerConfig, vs VarStore, lr float64) (retVal Optimizer, err error) {

	opt, err := config.BuildCOpt(lr)
	if err != nil {
		return retVal, err
	}

	vs.variables.mutex.Lock()
	defer vs.variables.mutex.Unlock()

	if err = opt.AddParameters(vs.variables.TrainableVariable); err != nil {
		return retVal, err
	}

	return Optimizer{
		opt:                  opt,
		variables:            vs.variables,
		variablesInOptimizer: uint8(len(vs.variables.TrainableVariable)),
		config:               config,
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
func DefaultSGDConfig() SGDConfig {
	return SGDConfig{
		Momentum:  0.0,
		Dampening: 0.0,
		Wd:        0.0,
		Nesterov:  false,
	}
}

// NewSGD creates the configuration for a SGD optimizer with specified values
func NewSGDConfig(momentum, dampening, wd float64, nesterov bool) (retVal SGDConfig) {
	return SGDConfig{
		Momentum:  momentum,
		Dampening: dampening,
		Wd:        wd,
		Nesterov:  nesterov,
	}
}

// Implement OptimizerConfig interface for SGDConfig
func (c SGDConfig) BuildCOpt(lr float64) (retVal ts.COptimizer, err error) {
	return ts.Sgd(lr, c.Momentum, c.Dampening, c.Wd, c.Nesterov)
}

func (c SGDConfig) Build(vs VarStore, lr float64) (retVal Optimizer, err error) {
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
func DefaultAdamConfig() AdamConfig {
	return AdamConfig{
		Beta1: 0.9,
		Beta2: 0.999,
		Wd:    0.0,
	}
}

// NewAdamConfig creates AdamConfig with specified values
func NewAdamConfig(beta1, beta2, wd float64) AdamConfig {
	return AdamConfig{
		Beta1: beta1,
		Beta2: beta2,
		Wd:    wd,
	}
}

// Implement OptimizerConfig interface for AdamConfig
func (c AdamConfig) BuildCOpt(lr float64) (retVal ts.COptimizer, err error) {
	return ts.Adam(lr, c.Beta1, c.Beta2, c.Wd)
}

func (c AdamConfig) Build(vs VarStore, lr float64) (retVal Optimizer, err error) {
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
func DefaultRMSPropConfig() RMSPropConfig {
	return RMSPropConfig{
		Alpha:    0.99,
		Eps:      1e-8,
		Wd:       0.0,
		Momentum: 0.0,
		Centered: false,
	}
}

// NewRMSPropConfig creates RMSPropConfig with specified values
func NewRMSPropConfig(alpha, eps, wd, momentum float64, centered bool) RMSPropConfig {
	return RMSPropConfig{
		Alpha:    alpha,
		Eps:      eps,
		Wd:       wd,
		Momentum: momentum,
		Centered: centered,
	}
}

// Implement OptimizerConfig interface for RMSPropConfig
func (c RMSPropConfig) BuildCOpt(lr float64) (retVal ts.COptimizer, err error) {
	return ts.RmsProp(lr, c.Alpha, c.Eps, c.Wd, c.Momentum, c.Centered)
}

func (c RMSPropConfig) Build(vs VarStore, lr float64) (retVal Optimizer, err error) {
	return defaultBuild(c, vs, lr)
}

// Optimizer methods:
// ==================
func (opt *Optimizer) addMissingVariables() {

	opt.variables.mutex.Lock()
	defer opt.variables.mutex.Unlock()

	missingVariables := len(opt.variables.TrainableVariable) - int(opt.variablesInOptimizer)

	if missingVariables > 0 {
		opt.opt.AddParameters(opt.variables.TrainableVariable[opt.variablesInOptimizer:])
		opt.variablesInOptimizer = uint8(len(opt.variables.TrainableVariable))
	}

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

	opt.variables.mutex.Lock()
	defer opt.variables.mutex.Unlock()

	for _, tensor := range opt.variables.TrainableVariable {
		tensor.MustGrad().Clamp_(ts.FloatScalar(-max), ts.FloatScalar(max))
	}
}

// Step performs an optimization step, updating the tracked tensors based on their gradients.
func (opt *Optimizer) Step() {
	opt.addMissingVariables()
	err := opt.opt.Step()
	if err != nil {
		log.Fatalf("Optimizer - Step method call error: %v\n", err)
	}
}

// BackwardStep applies a backward step pass, update the gradients, and performs an optimization step.
func (opt *Optimizer) BackwardStep(loss ts.Tensor) {
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
func (opt *Optimizer) BackwardStepClip(loss ts.Tensor, max float64) {
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

// SetLR sets the optimizer learning rate.
func (opt *Optimizer) SetLR(lr float64) {
	err := opt.opt.SetLearningRate(lr)
	if err != nil {
		log.Fatalf("Optimizer - SetLR  method call error: %v\n", err)
	}
}

// SetMomentum sets the optimizer momentum.
func (opt *Optimizer) SetMomentum(m float64) {
	err := opt.opt.SetMomentum(m)
	if err != nil {
		log.Fatalf("Optimizer - SetMomentum  method call error: %v\n", err)
	}
}
