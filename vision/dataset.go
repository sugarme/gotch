package vision

// A simple dataset structure shared by various computer vision datasets.

import (
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

type Dataset struct {
	TrainImages ts.Tensor
	TrainLabels ts.Tensor
	TestImages  ts.Tensor
	TestLabels  ts.Tensor
	Labels      int64
}

// Dataset Methods:
//=================

// TrainIter creates an iterator of Iter type for train images and labels
func (ds Dataset) TrainIter(batchSize int64) (retVal nn.Iter2) {
	return nn.MustNewIter2(ds.TrainImages, ds.TrainLabels, batchSize)

}

// TestIter creates an iterator of Iter type for test images and labels
func (ds Dataset) TestIter(batchSize int64) (retVal nn.Iter2) {
	return nn.MustNewIter2(ds.TestImages, ds.TestLabels, batchSize)
}
