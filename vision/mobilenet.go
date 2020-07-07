package vision

// MobileNet V2 implementation.
// https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html

import (
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

// Conv2D + BatchNorm2D + ReLU6
func cbr(p nn.Path, cIn, cOut, ks, stride, g int64) (retVal ts.ModuleT) {
	config := nn.DefaultConv2DConfig()
	config.Stride = []int64{stride, stride}
	pad := (ks - 1) / 2
	config.Padding = []int64{pad, pad}
	config.Groups = g
	config.Bias = false

	seq := nn.SeqT()
	seq.Add(nn.NewConv2D(p.Sub("0"), cIn, cOut, ks, config))

	return seq
}
