package tensor

// Other tensor methods

import (
	"github.com/sugarme/gotch"
)

// CrossEntropyForLogits computes the cross-entropy loss based on some logits and targets.
func (ts Tensor) CrossEntropyForLogits(targets Tensor) (retVal Tensor) {
	return ts.MustLogSoftmax(-1, gotch.Float.CInt(), true).MustNllLoss(targets, true)
}

// AccuracyForLogits returns the average accuracy for some given logits assuming that
// targets represent ground-truth.
func (ts Tensor) AccuracyForLogits(targets Tensor) (retVal Tensor) {
	return ts.MustArgmax(-1, false, true).MustEq1(targets).MustTotype(gotch.Float, true).MustMean(gotch.Float.CInt(), true)
}

// TODO: continue
