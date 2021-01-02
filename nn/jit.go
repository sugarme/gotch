package nn

import (
	ts "github.com/sugarme/gotch/tensor"
)

// TrainableCModule is a trainable version of JIT Pytorch module
//
// These modules can be created via TorchScript python API.
// See: https://pytorch.org/docs/stable/jit.html
type TrainableCModule struct {
	Inner ts.CModule
}

// TrainableModuleLoad loads a PyTorch saved JIT module from a file and adds
// tensors (weights) to `varstore` so that module can be trained.
/*
 * func TrainableModuleLoad(p *Path, file string) (*TrainableCModule, error) {
 *
 *   inner, err := ts.ModuleLoadOnDevice(file, p.Device())
 *   if err != nil {
 *     return nil, err
 *   }
 * }
 *  */
