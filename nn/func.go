package nn

// Layers defined by closure

import (
	"github.com/sugarme/gotch/ts"
)

type Func struct {
	f func(*ts.Tensor) *ts.Tensor
}

func NewFunc(fn func(*ts.Tensor) *ts.Tensor) (retVal Func) {
	return Func{f: fn}
}

// Implement Module interface for Func:
// ====================================
func (fn Func) Forward(xs *ts.Tensor) (retVal *ts.Tensor) {
	return fn.f(xs)
}

// ForwardT implements ModuleT for Func object as well.
//
// NOTE: train param will not be used.
func (fn Func) ForwardT(xs *ts.Tensor, train bool) (retVal *ts.Tensor) {
	return fn.f(xs)
}

type FuncT struct {
	f func(*ts.Tensor, bool) *ts.Tensor
}

func NewFuncT(fn func(*ts.Tensor, bool) *ts.Tensor) (retVal FuncT) {
	return FuncT{f: fn}
}

// Implement Module interface for Func:
// ====================================
func (fn FuncT) ForwardT(xs *ts.Tensor, train bool) (retVal *ts.Tensor) {
	return fn.f(xs, train)
}
