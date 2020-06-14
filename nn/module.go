package nn

import (
	ts "github.com/sugarme/gotch/tensor"
)

// Module interface is a container with only one method `Forward`
//
// The following is `module` concept from Pytorch documenation:
// Base class for all neural network modules. Your models should also subclass this class.
// Modules can also contain other Modules, allowing to nest them in a tree structure.
// You can assign the submodules as regular attributes. Submodules assigned in this way will
// be registered, and will have their parameters converted too when you call .cuda(), etc.
type Module interface {
	// ModuleT
	Forward(xs ts.Tensor) ts.Tensor
}

// ModuleT is a `Module` with an additional train parameter
// The train parameter is commonly used to have different behavior
// between training and evaluation. E.g. When using dropout or batch-normalization.
type ModuleT interface {
	ForwardT(xs ts.Tensor, train bool) ts.Tensor
}

// DefaultModuleT implements default method `BatchAccuracyForLogits`.
// NOTE: when creating a struct that implement `ModuleT`, it should
// include `DefaultModule` so that the 'default' methods `BatchAccuracyForLogits`
// is automatically implemented.
// Concept taken from Rust language trait **Default Implementation**
// Ref: https://doc.rust-lang.org/1.22.1/book/second-edition/ch10-02-traits.html
//
// Example:
//
// type FooModule struct{
// 		DefaultModuleT
// 		OtherField string
// }
type DefaultModuleT struct{}

func (dmt *DefaultModuleT) BatchAccuracyForLogits(xs, ys ts.Tensor, batchSize int) float64 {

	var (
		sumAccuracy float64 = 0.0
		sampleCount float64 = 0.0
	)

	// TODO: implement Iter2...

	return sumAccuracy / sampleCount
}

// TODO: should we include tensor in `Module` and `ModuleT` interfaces???
// I.e.:
// type Module interface{
//     t.Tensor
//     Forward(xs ts.Tensor) ts.Tensor
// }
