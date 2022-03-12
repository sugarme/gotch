package aug

import (
	"github.com/sugarme/gotch/ts"
)

// Normalize normalizes a tensor image with mean and standard deviation.
// Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
// channels, this transform will normalize each channel of the input
// ``torch.*Tensor`` i.e.,
// ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
// .. note::
// This transform acts out of place, i.e., it does not mutate the input tensor.
// Args:
// - mean (sequence): Sequence of means for each channel.
// - std (sequence): Sequence of standard deviations for each channel.
type Normalize struct {
	mean []float64 // should be from 0 to 1
	std  []float64 // should be > 0 and <= 1
}

type normalizeOptions struct {
	mean []float64
	std  []float64
}

type NormalizeOption func(*normalizeOptions)

// Mean and SD can be calculated for specific dataset as follow:
/*
	mean = 0.0
	meansq = 0.0
	count = 0

	for index, data in enumerate(train_loader):
			mean = data.sum()
			meansq = meansq + (data**2).sum()
			count += np.prod(data.shape)

	total_mean = mean/count
	total_var = (meansq/count) - (total_mean**2)
	total_std = torch.sqrt(total_var)
	print("mean: " + str(total_mean))
	print("std: " + str(total_std))
*/

// For example. ImageNet dataset has RGB mean and standard error:
// meanVals := []float64{0.485, 0.456, 0.406}
// sdVals := []float64{0.229, 0.224, 0.225}
func defaultNormalizeOptions() *normalizeOptions {
	return &normalizeOptions{
		mean: []float64{0, 0, 0},
		std:  []float64{1, 1, 1},
	}
}

func WithNormalizeStd(std []float64) NormalizeOption {
	return func(o *normalizeOptions) {
		o.std = std
	}
}

func WithNormalizeMean(mean []float64) NormalizeOption {
	return func(o *normalizeOptions) {
		o.mean = mean
	}
}

func newNormalize(opts ...NormalizeOption) *Normalize {
	p := defaultNormalizeOptions()
	for _, o := range opts {
		o(p)
	}

	return &Normalize{
		mean: p.mean,
		std:  p.std,
	}
}

func (n *Normalize) Forward(x *ts.Tensor) *ts.Tensor {
	fx := Byte2FloatImage(x)

	out := normalize(fx, n.mean, n.std)

	bx := Float2ByteImage(out)
	fx.MustDrop()
	out.MustDrop()

	return bx
}

func WithNormalize(opts ...NormalizeOption) Option {
	n := newNormalize(opts...)
	return func(o *Options) {
		o.normalize = n
	}
}
