package nn

// A sequential layer used to chain multiple layers and closures.

import (
	ts "github.com/sugarme/gotch/tensor"
)

// Sequential is a layer (container) that combines multiple other layers.
type Sequential struct {
	layers []ts.Module
}

// Seq creates a new empty sequential layer
func Seq() Sequential {
	return Sequential{layers: make([]ts.Module, 0)}
}

// Sequential methods:
//====================

// Len returns number of sub-layers embedded in this layer
func (s *Sequential) Len() (retVal int64) {
	return int64(len(s.layers))
}

// IsEmpty returns true if this layer does not have any sub-layers.
func (s *Sequential) IsEmpty() (retVal bool) {
	return len(s.layers) == 0
}

// Add appends a layer after all the current layers.
func (s *Sequential) Add(l ts.Module) {

	s.layers = append(s.layers, l)
}

// AddFn appends a closure after all the current layers.
//
// NOTE: fn should have signature `func(t ts.Tensor) ts.Tensor`
// and it implements Module interface
func (s *Sequential) AddFn(fn interface{}) {

	s.Add(fn.(ts.Module))
}

// ForwardAll applies the forward pass and returns the output for each layer.
func (s *Sequential) ForwardAll(xs ts.Tensor, opts ...uint8) (retVal []ts.Tensor) {

	var n uint8 = uint8(len(s.layers))
	if len(opts) > 0 {
		n = opts[0]
	}

	if s.IsEmpty() {
		return []ts.Tensor{xs.MustShallowClone()}
	}

	for i := 0; i < int(n); i++ {
		retVal = append(retVal, s.layers[i].Forward(xs))
	}

	return retVal
}

func ForwardAllWithN(n uint8) func() uint8 {
	return func() uint8 {
		return n
	}
}

// Implement Module interface for Sequential:
// ==========================================
func (s Sequential) Forward(xs ts.Tensor) (retVal ts.Tensor) {
	if s.IsEmpty() {
		return xs.MustShallowClone()
	}

	// forward sequentially
	var currTs ts.Tensor = xs
	for i := 0; i < len(s.layers); i++ {
		currTs = s.layers[i].Forward(currTs)
	}

	return currTs
}
