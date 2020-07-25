package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/nn"
)

const (
	LearningRate float64 = 0.01
	HiddenSize   int64   = 256
	SeqLen       int64   = 180
	BatchSize    int64   = 256
	Epochs       int64   = 100
	SamplingLen  int64   = 1024
)

func sample(data ts.TextData, lstm nn.LSTM, linear nn.Linear, device gotch.Device) (retVal string) {

	return
}
