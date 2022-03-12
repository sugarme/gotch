package ts

// Other tensor methods

import (
	"github.com/sugarme/gotch"
)

// CrossEntropyForLogits computes the cross-entropy loss based on some logits and targets.
func (ts *Tensor) CrossEntropyForLogits(targets *Tensor) (retVal *Tensor) {
	weight := NewTensor()
	reduction := int64(1) // Mean of loss
	ignoreIndex := int64(-100)

	logSm := ts.MustLogSoftmax(-1, gotch.Float, false)
	return logSm.MustNllLoss(targets, weight, reduction, ignoreIndex, true)
}

// AccuracyForLogits returns the average accuracy for some given logits assuming that
// targets represent ground-truth.
func (ts *Tensor) AccuracyForLogits(targets *Tensor) (retVal *Tensor) {
	argmax := ts.MustArgmax([]int64{-1}, false, false)
	eq1 := argmax.MustEqTensor(targets, true)
	return eq1.MustTotype(gotch.Float, true).MustMean(gotch.Float, true)
}

func (ts *Tensor) MaxPool2DDefault(ksize int64, del bool) (retVal *Tensor) {
	return ts.MustMaxPool2d([]int64{ksize, ksize}, []int64{ksize, ksize}, []int64{0, 0}, []int64{1, 1}, false, del)
}
