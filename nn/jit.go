package nn

import (
	"io"
	"log"
	"strings"

	ts "github.com/sugarme/gotch/tensor"
)

// TrainableCModule is a trainable version of JIT Pytorch module
//
// These modules can be created via TorchScript python API.
// See: https://pytorch.org/docs/stable/jit.html
type TrainableCModule struct {
	Inner *ts.CModule
}

// TrainableCModuleLoad loads a PyTorch saved JIT module from a file and adds
// tensors (weights) to `varstore` so that module can be trained.
func TrainableCModuleLoad(p *Path, file string) (*TrainableCModule, error) {
	inner, err := ts.ModuleLoadOnDevice(file, p.Device())
	if err != nil {
		return nil, err
	}

	namedTensors, err := inner.NamedParameters()
	if err != nil {
		return nil, err
	}

	// Add named tensors (weights) to varstore
	for _, namedTensor := range namedTensors {
		name := strings.ReplaceAll(namedTensor.Name, ".", "_")
		requiresGrad := namedTensor.Tensor.MustRequiresGrad()
		// NOTE: return is a newly created and added tensor in varstore.
		// This tensor is different from input named tensor.
		// If not using, just ignore it. Drop it, will drop tensor at varstore.
		_ = p.Add(name, namedTensor.Tensor, requiresGrad)

		// Clean-up named tensors.
		namedTensor.Tensor.MustDrop()
	}

	return &TrainableCModule{inner}, nil
}

func TrainableCModuleLoadData(p *Path, stream io.Reader) (*TrainableCModule, error) {
	inner, err := ts.ModuleLoadDataOnDevice(stream, p.Device())
	if err != nil {
		return nil, err
	}
	namedTensors, err := inner.NamedParameters()
	if err != nil {
		return nil, err
	}

	// Add named tensors (weights) to varstore
	for _, namedTensor := range namedTensors {
		name := strings.ReplaceAll(namedTensor.Name, ".", "_")
		requiresGrad := namedTensor.Tensor.MustRequiresGrad()
		// NOTE: return is a newly created and added tensor in varstore.
		// This tensor is different from input named tensor.
		// If not using, just ignore it. Drop it, will drop tensor at varstore.
		_ = p.Add(name, namedTensor.Tensor, requiresGrad)

		// Clean-up named tensors.
		namedTensor.Tensor.MustDrop()
	}

	return &TrainableCModule{inner}, nil
}

// Save saves TrainableCModule to specified file.
func (m *TrainableCModule) Save(file string) error {
	return m.Inner.Save(file)
}

// ForwardT implements ModuleT for TrainableCModule.
// NOTE: train parameter will not be used.
func (m *TrainableCModule) ForwardT(x *ts.Tensor, train bool) *ts.Tensor {
	retVal, err := m.Inner.ForwardTs([]ts.Tensor{*x})
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// SetTrain set TrainableCModule to train mode
func (m *TrainableCModule) SetTrain() {
	m.Inner.SetTrain()
}

// SetEval set TrainableCModule to inference mode
func (m *TrainableCModule) SetEval() {
	m.Inner.SetEval()
}
