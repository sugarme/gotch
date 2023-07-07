package ts

// Other tensor methods

// CrossEntropyForLogits computes the cross-entropy loss based on some logits and targets.
func (ts *Tensor) CrossEntropyForLogits(targets *Tensor) (retVal *Tensor) {
	weight := NewTensor()
	reduction := int64(1) // Mean of loss
	ignoreIndex := int64(-100)

	dtype := ts.DType()
	logSm := ts.MustLogSoftmax(-1, dtype, false)
	return logSm.MustNllLoss(targets, weight, reduction, ignoreIndex, true)
}

// AccuracyForLogits returns the average accuracy for some given logits assuming that
// targets represent ground-truth.
func (ts *Tensor) AccuracyForLogits(targets *Tensor) (retVal *Tensor) {
	argmax := ts.MustArgmax([]int64{-1}, false, false)
	eq1 := argmax.MustEqTensor(targets, true)
	dtype := ts.DType()
	return eq1.MustTotype(dtype, true).MustMean(dtype, true)
}

func (ts *Tensor) MaxPool2DDefault(ksize int64, del bool) (retVal *Tensor) {
	return ts.MustMaxPool2d([]int64{ksize, ksize}, []int64{ksize, ksize}, []int64{0, 0}, []int64{1, 1}, false, del)
}
