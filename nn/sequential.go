package nn

// A sequential layer used to chain multiple layers and closures.

import (
	ts "github.com/sugarme/gotch/tensor"
	// "reflect"
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
func (s Sequential) Len() (retVal int64) {
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
func (s *Sequential) AddFn(fn ts.Module) {

	s.Add(fn)
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

// WithUint8 returns an uint8 value option
func WithUint8(n uint8) func() uint8 {
	return func() uint8 {
		return n
	}
}

// Implement Module interface for Sequential:
// ==========================================

// Forward implements Module interface for Sequential
func (s *Sequential) Forward(xs ts.Tensor) (retVal ts.Tensor) {
	if s.IsEmpty() {
		return xs.MustShallowClone()
	}

	// forward sequentially
	outs := make([]ts.Tensor, len(s.layers))
	for i := 0; i < len(s.layers); i++ {
		if i == 0 {
			outs[0] = s.layers[i].Forward(xs)
			defer outs[0].MustDrop()
		} else if i == len(s.layers)-1 {
			return s.layers[i].Forward(outs[i-1])
		} else {
			outs[i] = s.layers[i].Forward(outs[i-1])
			defer outs[i].MustDrop()
		}
	}

	return
}

// SequentialT is a sequential layer combining new layers with support for a training mode.
type SequentialT struct {
	layers []ts.ModuleT
}

/// SeqT creates a new empty sequential layer.
func SeqT() SequentialT {
	return SequentialT{
		layers: make([]ts.ModuleT, 0),
	}
}

// SequentialT methods:
//=====================

// Len returns number of sub-layers embedded in this layer
func (s *SequentialT) Len() (retVal int64) {
	return int64(len(s.layers))
}

// IsEmpty returns true if this layer does not have any sub-layers.
func (s *SequentialT) IsEmpty() (retVal bool) {
	return len(s.layers) == 0
}

// Implement ModuleT interface for SequentialT:
// ==========================================
/*
 * func (s SequentialT) Forward(xs ts.Tensor) (retVal ts.Tensor) {
 *   if s.IsEmpty() {
 *     return xs.MustShallowClone()
 *   }
 *
 *   // forward sequentially
 *   var currTs ts.Tensor = xs
 *   for i := 0; i < len(s.layers); i++ {
 *     currTs = s.layers[i].Forward(currTs)
 *   }
 *
 *   return currTs
 * }
 *  */
func (s SequentialT) ForwardT(xs ts.Tensor, train bool) (retVal ts.Tensor) {
	if s.IsEmpty() {
		return xs.MustShallowClone()
	}

	// forward sequentially
	var currTs ts.Tensor = xs
	for i := 0; i < len(s.layers); i++ {
		currTs = s.layers[i].ForwardT(currTs, train)
	}

	return currTs
}

// Add appends a layer after all the current layers.
func (s *SequentialT) Add(l ts.ModuleT) {
	s.layers = append(s.layers, l)
}

// AddFn appends a closure after all the current layers.
//
// NOTE: fn should have signature `func(t ts.Tensor) ts.Tensor`
// and it implements Module interface
func (s *SequentialT) AddFn(fn ts.ModuleT) {

	s.Add(fn)
}

// AddFn appends a closure after all the current layers.
//
// NOTE: fn should have signature `func(t ts.Tensor, train bool) ts.Tensor`
// and it implements Module interface
func (s *SequentialT) AddFnT(fn ts.ModuleT) {

	s.Add(fn)
}

// ForwardAll applies the forward pass and returns the output for each layer.
func (s *SequentialT) ForwardAllT(xs ts.Tensor, train bool, opts ...uint8) (retVal []ts.Tensor) {

	var n uint8 = uint8(len(s.layers))
	if len(opts) > 0 {
		n = opts[0]
	}

	if s.IsEmpty() {
		return []ts.Tensor{xs.MustShallowClone()}
	}

	currTs := xs
	for i := 0; i < int(n); i++ {
		res := s.layers[i].ForwardT(currTs, train)
		retVal = append(retVal, res)
		currTs = res
	}

	return retVal
}

// ForwardWith is a handler function to implement Module interface for
// any (anonymous) function it wraps.
//
// Ref. https://stackoverflow.com/a/42182987
// NOTE: Specifically, `ForwardWith` is used to wrap anonymous function
// as input parameter of `AddFn` Sequential method.
type ForwardWith func(ts.Tensor) ts.Tensor

func (fw ForwardWith) Forward(xs ts.Tensor) ts.Tensor {
	return fw(xs)
}

type ForwardTWith func(ts.Tensor, bool) ts.Tensor

func (fw ForwardTWith) ForwardT(xs ts.Tensor, train bool) ts.Tensor {
	return fw(xs, train)
}
