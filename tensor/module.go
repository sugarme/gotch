package tensor

// Module interface is a container with only one method `Forward`
//
// The following is `module` concept from Pytorch documenation:
// Base class for all neural network modules. Your models should also subclass this class.
// Modules can also contain other Modules, allowing to nest them in a tree structure.
// You can assign the submodules as regular attributes. Submodules assigned in this way will
// be registered, and will have their parameters converted too when you call .cuda(), etc.
type Module interface {
	// ModuleT
	Forward(xs *Tensor) *Tensor
}

// ModuleT is a `Module` with an additional train parameter
// The train parameter is commonly used to have different behavior
// between training and evaluation. E.g. When using dropout or batch-normalization.
type ModuleT interface {
	// Forward(xs Tensor) Tensor
	ForwardT(xs *Tensor, train bool) *Tensor
}

/*
 * // DefaultModuleT implements default method `BatchAccuracyForLogits`.
 * // NOTE: when creating a struct that implement `ModuleT`, it should
 * // include `DefaultModule` so that the 'default' methods `BatchAccuracyForLogits`
 * // is automatically implemented.
 * // Concept taken from Rust language trait **Default Implementation**
 * // Ref: https://doc.rust-lang.org/1.22.1/book/second-edition/ch10-02-traits.html
 * //
 * // Example:
 * //
 * // type FooModule struct{
 * // 		DefaultModuleT
 * // 		OtherField string
 * // }
 * type DefaultModuleT struct{}
 *
 * func (dmt DefaultModuleT) Forward(xs Tensor) (retVal Tensor) {
 *   // TODO: implement
 *
 *   return
 * }
 *
 * // Implement Module interface
 * func (dmt DefaultModuleT) ForwardT(xs Tensor, train bool) (retVal Tensor) {
 *   // TODO: implement
 *
 *   return dmt.Forward(xs)
 * }
 *  */

// NOTE: this func has been moved to `nn/sequential` as `NoGradGuard`
// seem not working in Go and the function needs to add varstore variable
// parameter. Hence, it is moved to `nn` to avoid cycle reference.
/*
 * // BatchAccuracyForLigits calculate accuracy in batch.
 * //
 * // TODO: It would be nice if it is one method an object that implements ModuleT
 * // interface.
 * func BatchAccuracyForLogits(m ModuleT, xs, ys Tensor, d gotch.Device, batchSize int) (retVal float64) {
 *
 *   var (
 *     sumAccuracy float64 = 0.0
 *     sampleCount float64 = 0.0
 *   )
 *
 *   _ = MustGradSetEnabled(false)
 *
 *   iter2 := MustNewIter2(xs, ys, int64(batchSize))
 *   for {
 *     item, ok := iter2.Next()
 *     if !ok {
 *       break
 *     }
 *
 *     size := float64(item.Data.MustSize()[0])
 *     bImages := item.Data.MustTo(d, true)
 *     bLabels := item.Label.MustTo(d, true)
 *
 *     logits := m.ForwardT(bImages, false)
 *     acc := logits.AccuracyForLogits(bLabels)
 *     sumAccuracy += acc.Values()[0] * size
 *     sampleCount += size
 *
 *     bImages.MustDrop()
 *     bLabels.MustDrop()
 *     acc.MustDrop()
 *   }
 *
 *   _ = MustGradSetEnabled(true)
 *
 *   return sumAccuracy / sampleCount
 *
 * }
 *  */

// Tensor methods for Module and ModuleT:
// ======================================

// Apply forwards tensor itself through a module.
func (ts *Tensor) Apply(m Module) (retVal *Tensor) {
	return m.Forward(ts)
}

// Apply forwards tensor itself through a module T.
func (ts *Tensor) ApplyT(m ModuleT, train bool) (retVal *Tensor) {
	return m.ForwardT(ts, train)
}

// ApplyOpt forwards a tensor itself through a module if given, shallow-copies
// the tensor otherwise.
func (ts *Tensor) ApplyOpt(opts ...ModuleOption) (retVal *Tensor) {

	switch {
	case len(opts) > 0:
		m := opts[0]()
		return m.Forward(ts)
	default:
		return ts.MustShallowClone()
	}
}

type ModuleOption func() Module

func WithModule(m Module) ModuleOption {
	return func() Module {
		return m
	}
}

// ApplyOptT forwards a tensor itself through a module T if given, shallow-copies
// the tensor otherwise.
func (ts *Tensor) ApplyOptT(train bool, opts ...ModuleTOption) (retVal *Tensor) {

	switch {
	case len(opts) > 0:
		m := opts[0]()
		return m.ForwardT(ts, train)
	default:
		return ts.MustShallowClone()
	}
}

type ModuleTOption func() ModuleT

func WithModuleT(m ModuleT) ModuleTOption {
	return func() ModuleT {
		return m
	}
}
