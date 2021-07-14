package nn

import (
	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

type lossFnOptions struct {
	ClassWeights []float64
	Reduction    int64 // 0: "None", 1: "mean", 2: "sum"
	IgnoreIndex  int64
	PosWeight    int64 // index of the weight attributed to positive class. Used in BCELoss
}

type LossFnOption func(*lossFnOptions)

func WithLossFnWeights(vals []float64) LossFnOption {
	return func(o *lossFnOptions) {
		o.ClassWeights = vals
	}
}
func WithLossFnReduction(val int64) LossFnOption {
	return func(o *lossFnOptions) {
		o.Reduction = val
	}
}
func WithLossFnIgnoreIndex(val int64) LossFnOption {
	return func(o *lossFnOptions) {
		o.IgnoreIndex = val
	}
}

func WithLossFnPosWeight(val int64) LossFnOption {
	return func(o *lossFnOptions) {
		o.PosWeight = val
	}
}

func defaultLossFnOptions() *lossFnOptions {
	return &lossFnOptions{
		ClassWeights: nil,
		Reduction:    1, // "mean"
		IgnoreIndex:  -100,
		PosWeight:    -1,
	}
}

// CrossEntropyLoss calculates cross entropy loss.
// Ref. https://github.com/pytorch/pytorch/blob/15be189f0de4addf4f68d18022500f67617ab05d/torch/nn/functional.py#L2012
// - logits: tensor of shape [B, C, H, W] corresponding the raw output of the model.
// - target: ground truth tensor of shape [B, 1, H, W]
// - posWeight: scalar representing the weight attributed to positive class.
// This is especially useful for an imbalanced dataset
func CrossEntropyLoss(logits, target *ts.Tensor, opts ...LossFnOption) *ts.Tensor {
	options := defaultLossFnOptions()
	for _, o := range opts {
		o(options)
	}

	var ws *ts.Tensor
	device := logits.MustDevice()
	dtype := logits.DType()
	if len(options.ClassWeights) > 0 {
		ws = ts.MustOfSlice(options.ClassWeights).MustTotype(dtype, true).MustTo(device, true)
	} else {
		ws = ts.NewTensor()
	}
	reduction := options.Reduction
	ignoreIndex := options.IgnoreIndex

	logSm := logits.MustLogSoftmax(-1, gotch.Float, false)
	loss := logSm.MustNllLoss(target, ws, reduction, ignoreIndex, true)
	ws.MustDrop()

	return loss
}

// BCELoss calculates a binary cross entropy loss.
//
// - logits: tensor of shape [B, C, H, W] corresponding the raw output of the model.
// - target: ground truth tensor of shape [B, 1, H, W]
// - posWeight: scalar representing the weight attributed to positive class.
// This is especially useful for an imbalanced dataset
func BCELoss(logits, target *ts.Tensor, opts ...LossFnOption) *ts.Tensor {
	options := defaultLossFnOptions()
	for _, o := range opts {
		o(options)
	}

	var ws *ts.Tensor
	device := logits.MustDevice()
	dtype := logits.DType()
	if len(options.ClassWeights) > 0 {
		ws = ts.MustOfSlice(options.ClassWeights).MustTotype(dtype, true).MustTo(device, true)
	} else {
		ws = ts.NewTensor()
	}
	reduction := options.Reduction

	var posWeight *ts.Tensor
	if options.PosWeight >= 0 {
		posWeight = ts.MustOfSlice([]int64{options.PosWeight})
	} else {
		posWeight = ts.NewTensor()
	}

	loss := logits.MustSqueeze(false).MustBinaryCrossEntropyWithLogits(target, ws, posWeight, reduction, true)
	return loss
}
