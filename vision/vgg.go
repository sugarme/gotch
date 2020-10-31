package vision

// VGG models

import (
	"fmt"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

// NOTE: each list element contains multiple convolutions with some specified number
// of features followed by a single max-pool layer.
func layersA() [][]int64 {
	return [][]int64{
		{64},
		{128},
		{256, 256},
		{512, 512},
		{512, 512},
	}
}

func layersB() [][]int64 {
	return [][]int64{
		{64, 64},
		{128, 128},
		{256, 256},
		{512, 512},
		{512, 512},
	}
}

func layersD() [][]int64 {
	return [][]int64{
		{64, 64},
		{128, 128},
		{256, 256, 256},
		{512, 512, 512},
		{512, 512, 512},
	}
}

func layersE() [][]int64 {
	return [][]int64{
		{64, 64},
		{128, 128},
		{256, 256, 256, 256},
		{512, 512, 512, 512},
		{512, 512, 512, 512},
	}
}

func vggConv2d(path *nn.Path, cIn, cOut int64) *nn.Conv2D {

	config := nn.DefaultConv2DConfig()
	config.Stride = []int64{1, 1}
	config.Padding = []int64{1, 1}

	return nn.NewConv2D(path, cIn, cOut, 3, config)
}

func vgg(path *nn.Path, config [][]int64, nclasses int64, batchNorm bool) *nn.SequentialT {

	c := path.Sub("classifier")
	seq := nn.SeqT()
	f := path.Sub("features")
	var cIn int64 = 3

	for _, channels := range config {
		for _, cOut := range channels {
			l := seq.Len()
			seq.Add(vggConv2d(f.Sub(fmt.Sprintf("%v", l)), cIn, cOut))

			if batchNorm {
				bnLen := seq.Len()
				seq.Add(nn.BatchNorm2D(f.Sub(fmt.Sprintf("%v", bnLen)), cOut, nn.DefaultBatchNormConfig()))
			}

			seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
				return xs.MustRelu(false)
			}))

			cIn = cOut
		} // end of inner For loop

		seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
			return xs.MaxPool2DDefault(2, false)
		}))

	} // end of outer For loop

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.FlatView()
	}))

	seq.Add(nn.NewLinear(c.Sub(fmt.Sprint("0")), 512*7*7, 4096, nn.DefaultLinearConfig()))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	seq.AddFn(nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		return ts.MustDropout(xs, 0.5, train)
	}))

	seq.Add(nn.NewLinear(c.Sub(fmt.Sprint("3")), 4096, 4096, nn.DefaultLinearConfig()))

	seq.AddFn(nn.NewFunc(func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustRelu(false)
	}))

	seq.AddFn(nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		return ts.MustDropout(xs, 0.5, train)
	}))

	seq.Add(nn.NewLinear(c.Sub(fmt.Sprint("6")), 4096, nclasses, nn.DefaultLinearConfig()))

	return seq
}

func VGG11(path *nn.Path, nclasses int64) *nn.SequentialT {
	return vgg(path, layersA(), nclasses, false)
}

func VGG11BN(path *nn.Path, nclasses int64) *nn.SequentialT {
	return vgg(path, layersA(), nclasses, true)
}

func VGG13(path *nn.Path, nclasses int64) *nn.SequentialT {
	return vgg(path, layersB(), nclasses, false)
}

func VGG13BN(path *nn.Path, nclasses int64) *nn.SequentialT {
	return vgg(path, layersB(), nclasses, true)
}

func VGG16(path *nn.Path, nclasses int64) *nn.SequentialT {
	return vgg(path, layersD(), nclasses, false)
}

func VGG16BN(path *nn.Path, nclasses int64) *nn.SequentialT {
	return vgg(path, layersD(), nclasses, true)
}

func VGG19(path *nn.Path, nclasses int64) *nn.SequentialT {
	return vgg(path, layersE(), nclasses, false)
}

func VGG19BN(path *nn.Path, nclasses int64) *nn.SequentialT {
	return vgg(path, layersE(), nclasses, true)
}
