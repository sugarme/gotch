package aug

import (
	"fmt"
	"log"

	ts "github.com/sugarme/gotch/tensor"
)

// Ref. https://github.com/pytorch/vision/blob/f1d734213af65dc06e777877d315973ba8386080/torchvision/transforms/functional_tensor.py

type ColorJitter struct {
	brightness []float64
	contrast   []float64
	saturation []float64
	hue        []float64
}

type colorOptions struct {
	brightness []float64
	contrast   []float64
	saturation []float64
	hue        []float64
}

type ColorOption func(*colorOptions)

func defaultColorOptions() *colorOptions {
	return &colorOptions{
		brightness: []float64{0, 0},
		contrast:   []float64{0, 0},
		saturation: []float64{0, 0},
		hue:        []float64{0, 0},
	}
}

func WithColorBrightness(v []float64) ColorOption {
	b := checkOption(v)
	return func(o *colorOptions) {
		o.brightness = b
	}
}

func WithColorContrast(v []float64) ColorOption {
	c := checkOption(v)
	return func(o *colorOptions) {
		o.contrast = c
	}
}

func WithColorSaturation(v []float64) ColorOption {
	s := checkOption(v)
	return func(o *colorOptions) {
		o.saturation = s
	}
}

func WithColorHue(vals []float64) ColorOption {
	if len(vals) > 2 {
		err := fmt.Errorf("Expected 1-2 values. Got %v\n", len(vals))
		log.Fatal(err)
	}
	for _, v := range vals {
		if v < -0.5 || v > 0.5 {
			err := fmt.Errorf("Expected hue color option from [-0.5, 0.5]. Got %v\n", v)
			log.Fatal(err)
		}
	}

	return func(o *colorOptions) {
		o.hue = vals
	}
}

func checkOption(vals []float64) []float64 {
	if len(vals) > 2 {
		err := fmt.Errorf("Expected 1-2 values. Got %v\n", len(vals))
		log.Fatal(err)
	}
	for _, v := range vals {
		if v < 0 {
			err := fmt.Errorf("Expected non-zero value. Got %v\n", v)
			log.Fatal(err)
		}
	}

	return vals
}

func newColorJitter(opts ...ColorOption) *ColorJitter {
	options := defaultColorOptions()
	for _, o := range opts {
		o(options)
	}

	return &ColorJitter{
		brightness: getParam(options.brightness),
		contrast:   getParam(options.contrast),
		saturation: getParam(options.saturation),
		hue:        getParam(options.hue, "hue"),
	}
}

func WithColorJitter(opts ...ColorOption) Option {

	c := newColorJitter(opts...)

	return func(o *Options) {
		o.colorJitter = c
	}
}

func getParam(vals []float64, nameOpt ...string) []float64 {
	name := ""
	if len(nameOpt) > 0 {
		name = nameOpt[0]
	}

	var input []float64
	if len(vals) == 2 {
		if vals[0] < vals[1] {
			input = vals
		} else {
			input = []float64{vals[1], vals[0]}
		}
	} else {
		input = vals
	}

	out := make([]float64, 2)

	// if value is 0 or (1., 1.) for brightness/contrast/saturation
	// or (0., 0.) for hue, do nothing
	switch name {
	case "hue":
		switch len(input) {
		case 2:
			if input[0] == 0 && input[1] == 0 {
				return nil
			} else {
				return input
			}
		default: // 1 value
			v := input[0]
			if v < 0 {
				out[0] = v
				out[1] = -v
			} else {
				out[0] = -v
				out[1] = v
			}
			return out
		}
	default:
		switch len(input) {
		case 2:
			if (input[0] == 1 && input[1] == 1) || (input[0] == 0 && input[1] == 0) {
				return nil
			} else {
				return input
			}

		default: // 1 value
			v := input[0]
			if v == 0 || v == 1.0 {
				return nil
			}
			center := 1.0
			v1 := center - v
			v2 := center + v
			if v1 < 0 {
				out[0] = 0
			} else {
				out[0] = v1
			}
			out[1] = v2
			return out
		}
	}
}

// Forward implement ts.Module by randomly picking one of brightness, contrast,
// staturation or hue function to transform input image tensor.
// NOTE. input image dtype must be `uint8(Byte)`
func (c *ColorJitter) Forward(x *ts.Tensor) *ts.Tensor {
	// 1. Brightness
	var bOut *ts.Tensor
	if c.brightness == nil {
		bOut = x.MustShallowClone()
	} else {
		bfactor := randVal(c.brightness[0], c.brightness[1])
		bOut = adjustBrightness(x, bfactor)
	}
	// 2. Contrast
	var cOut *ts.Tensor
	if c.contrast == nil {
		cOut = bOut.MustShallowClone()
		bOut.MustDrop()
	} else {
		cfactor := randVal(c.contrast[0], c.contrast[1])
		cOut = adjustContrast(bOut, cfactor)
		bOut.MustDrop()
	}
	// 3. Saturation
	var sOut *ts.Tensor
	if c.saturation == nil {
		sOut = cOut.MustShallowClone()
		cOut.MustDrop()
	} else {
		sfactor := randVal(c.saturation[0], c.saturation[1])
		sOut = adjustSaturation(cOut, sfactor)
		cOut.MustDrop()
	}
	// 4. Hue
	var hOut *ts.Tensor
	if c.hue == nil {
		hOut = sOut.MustShallowClone()
		sOut.MustDrop()
	} else {
		hfactor := randVal(c.hue[0], c.hue[1])
		hOut = adjustHue(sOut, hfactor)
		sOut.MustDrop()
	}

	bx := Float2ByteImage(hOut)
	hOut.MustDrop()

	return bx
}
