package tensor

// Other tensor methods

import (
	"github.com/sugarme/gotch"
)

// CrossEntropyForLogits computes the cross-entropy loss based on some logits and targets.
func (ts Tensor) CrossEntropyForLogits(targets Tensor) (retVal Tensor) {
	// return ts.MustLogSoftmax(-1, gotch.Float.CInt(), true).MustNllLoss(targets, true)

	logSm := ts.MustLogSoftmax(-1, gotch.Float.CInt(), true)
	return logSm.MustNllLoss(targets, true)
}

// AccuracyForLogits returns the average accuracy for some given logits assuming that
// targets represent ground-truth.
func (ts Tensor) AccuracyForLogits(targets Tensor) (retVal Tensor) {
	argmax := ts.MustArgmax(-1, false, true)
	eq1 := argmax.MustEq1(targets, true)
	return eq1.MustTotype(gotch.Float, true).MustMean(gotch.Float.CInt(), true)
}

func (ts Tensor) MaxPool2DDefault(ksize int64, del bool) (retVal Tensor) {
	return ts.MustMaxPool2D([]int64{ksize, ksize}, []int64{ksize, ksize}, []int64{0, 0}, []int64{1, 1}, false, del)
}

// TODO: continue
