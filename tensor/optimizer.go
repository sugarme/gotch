package tensor

import (
	"log"

	lib "github.com/sugarme/gotch/libtch"
)

type COptimizer struct {
	coptimizer lib.Coptimizer
}

// Adam returns Adam optimizer
func Adam(lr, beta1, beta2, weightDecay float64) (*COptimizer, error) {
	coptimizer := lib.AtoAdam(lr, beta1, beta2, weightDecay)

	if err := TorchErr(); err != nil {
		return nil, err
	}

	return &COptimizer{coptimizer}, nil
}

// AdamW returns AdamW optimizer
func AdamW(lr, beta1, beta2, weightDecay float64) (*COptimizer, error) {
	coptimizer := lib.AtoAdamW(lr, beta1, beta2, weightDecay)

	if err := TorchErr(); err != nil {
		return nil, err
	}

	return &COptimizer{coptimizer}, nil
}

// RmsProp returns RMSProp optimizer
func RmsProp(lr, alpha, eps, wd, momentum float64, centered bool) (*COptimizer, error) {
	var centeredCInt int
	switch centered {
	case true:
		centeredCInt = 1
	case false:
		centeredCInt = 0
	}

	coptimizer := lib.AtoRmsProp(lr, alpha, eps, wd, momentum, centeredCInt)
	if err := TorchErr(); err != nil {
		return nil, err
	}

	return &COptimizer{coptimizer}, nil
}

// Sgd returns SGD optimizer
func Sgd(lr, momentum, dampening, wd float64, nesterov bool) (*COptimizer, error) {
	var nesterovCInt int
	switch nesterov {
	case true:
		nesterovCInt = 1
	case false:
		nesterovCInt = 0
	}

	coptimizer := lib.AtoSgd(lr, momentum, dampening, wd, nesterovCInt)
	if err := TorchErr(); err != nil {
		return nil, err
	}

	return &COptimizer{coptimizer}, nil
}

// AddParameters adds parameters as a slice of tensors to optimizer
func (co *COptimizer) AddParameters(tensors []Tensor) error {
	var ctensors []lib.Ctensor
	for _, t := range tensors {
		ctensors = append(ctensors, t.ctensor)
	}

	ntensors := len(tensors)

	// NOTE. temporary switch back as param group not updated yet!
	lib.AtoAddParametersOld(co.coptimizer, ctensors, ntensors)

	return TorchErr()
}

// AddParameter adds a single parameter to parameter group.
func (co *COptimizer) AddParameter(param *Tensor, group uint) error {
	lib.AtoAddParameter(co.coptimizer, param.ctensor, group)

	return TorchErr()
}

// SetLeanringRate sets learning rate for the optimizer
func (co *COptimizer) SetLearningRate(lr float64) error {
	lib.AtoSetLearningRate(co.coptimizer, lr)

	return TorchErr()
}

// GetLeanringRates get learning rates for the optimizer
func (co *COptimizer) GetLearningRates() ([]float64, error) {
	lrs := lib.AtoGetLearningRates(co.coptimizer)

	if err := TorchErr(); err != nil {
		return nil, err
	}

	return lrs, nil
}

func (co *COptimizer) SetLearningRates(lrs []float64) error {
	lib.AtoSetLearningRates(co.coptimizer, lrs)
	if err := TorchErr(); err != nil {
		return err
	}

	return nil
}

func (co *COptimizer) ParamGroupNum() (int64, error) {
	ngroup := lib.AtoParamGroupNum(co.coptimizer)

	if err := TorchErr(); err != nil {
		return -1, err
	}

	return ngroup, nil
}

func (co *COptimizer) AddParamGroup(tensors []Tensor) error {
	var ctensors []lib.Ctensor
	for _, t := range tensors {
		ctensors = append(ctensors, t.ctensor)
	}

	ntensors := len(tensors)

	lib.AtoAddParamGroup(co.coptimizer, ctensors, ntensors)
	return TorchErr()
}

// SetMomentum sets a momentum for the optimizer
func (co *COptimizer) SetMomentum(m float64) error {
	lib.AtoSetMomentum(co.coptimizer, m)

	return TorchErr()
}

// ZeroGrad sets gradients to zero
func (co *COptimizer) ZeroGrad() error {
	lib.AtoZeroGrad(co.coptimizer)

	return TorchErr()
}

// Steps proceeds optimizer
func (co *COptimizer) Step() error {
	lib.AtoStep(co.coptimizer)

	return TorchErr()
}

// Drop removes optimizer and frees up memory.
func (co *COptimizer) Drop() {
	lib.AtoFree(co.coptimizer)

	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}
}
