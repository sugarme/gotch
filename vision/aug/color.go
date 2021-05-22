package aug

import (
	"math/rand"
	"time"

	ts "github.com/sugarme/gotch/tensor"
)

// Ref. https://github.com/pytorch/vision/blob/f1d734213af65dc06e777877d315973ba8386080/torchvision/transforms/functional_tensor.py

type ColorJitter struct {
	brightness float64
	contrast   float64
	saturation float64
	hue        float64
}

func defaultColorJitter() *ColorJitter {
	return &ColorJitter{
		brightness: 1.0,
		contrast:   1.0,
		saturation: 1.0,
		hue:        0.0,
	}
}

func (c *ColorJitter) setBrightness(brightness float64) {
	c.brightness = brightness
}

func (c *ColorJitter) setContrast(contrast float64) {
	c.contrast = contrast
}

func (c *ColorJitter) setSaturation(sat float64) {
	c.saturation = sat
}

func (c *ColorJitter) setHue(hue float64) {
	c.hue = hue
}

// Forward implement ts.Module by randomly picking one of brightness, contrast,
// staturation or hue function to transform input image tensor.
func (c *ColorJitter) Forward(x *ts.Tensor) *ts.Tensor {
	rand.Seed(time.Now().UnixNano())
	idx := rand.Intn(4)
	switch idx {
	case 0:
		v := randVal(getMinMax(c.brightness))
		return adjustBrightness(x, v)
	case 1:
		v := randVal(getMinMax(c.contrast))
		return adjustContrast(x, v)
	case 2:
		v := randVal(getMinMax(c.saturation))
		return adjustSaturation(x, v)
	case 3:
		v := randVal(0, c.hue)
		return adjustHue(x, v)
	default:
		panic("Shouldn't reach here.")
	}
}

func WithColorJitter(brightness, contrast, sat, hue float64) Option {
	c := defaultColorJitter()
	c.setBrightness(brightness)
	c.setContrast(contrast)
	c.setSaturation(sat)
	c.setHue(hue)

	return func(o *Options) {
		o.colorJitter = c
	}
}
