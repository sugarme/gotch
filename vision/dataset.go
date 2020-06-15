package vision

// A simple dataset structure shared by various computer vision datasets.

import (
	ts "github.com/sugarme/gotch/tensor"
)

type Dataset struct {
	TrainImages ts.Tensor
	TrainLabels ts.Tensor
	TestImages  ts.Tensor
	TestLabels  ts.Tensor
	Labels      int64
}

// TODO: implement methods
