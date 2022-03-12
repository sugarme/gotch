package aug

import (
	"math/rand"
	"time"

	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

// Transformer is an interface that can transform an image tensor.
type Transformer interface {
	Transform(x *ts.Tensor) *ts.Tensor
}

// Augment is a struct composes of augmentation functions to implement Transformer interface.
type Augment struct {
	augments *nn.Sequential
}

// Transform implements Transformer interface for Augment struct.
func (a *Augment) Transform(image *ts.Tensor) *ts.Tensor {
	out := a.augments.Forward(image)
	return out
}

type Options struct {
	rotate                *RotateModule
	randRotate            *RandRotateModule
	resize                *ResizeModule
	colorJitter           *ColorJitter
	gaussianBlur          *GaussianBlur
	randomHFlip           *RandomHorizontalFlip
	randomVFlip           *RandomVerticalFlip
	randomCrop            *RandomCrop
	centerCrop            *CenterCrop
	randomCutout          *RandomCutout
	randomPerspective     *RandomPerspective
	randomAffine          *RandomAffine
	randomGrayscale       *RandomGrayscale
	randomSolarize        *RandomSolarize
	randomPosterize       *RandomPosterize
	randomInvert          *RandomInvert
	randomAutocontrast    *RandomAutocontrast
	randomAdjustSharpness *RandomAdjustSharpness
	randomEqualize        *RandomEqualize
	downSample            *DownSample
	zoomIn                *ZoomIn
	zoomOut               *ZoomOut
	normalize             *Normalize
}

func defaultOption() *Options {
	return &Options{
		rotate:                nil,
		randRotate:            nil,
		resize:                nil,
		colorJitter:           nil,
		gaussianBlur:          nil,
		randomHFlip:           nil,
		randomVFlip:           nil,
		randomCrop:            nil,
		centerCrop:            nil,
		randomCutout:          nil,
		randomPerspective:     nil,
		randomAffine:          nil,
		randomGrayscale:       nil,
		randomSolarize:        nil,
		randomPosterize:       nil,
		randomInvert:          nil,
		randomAutocontrast:    nil,
		randomAdjustSharpness: nil,
		randomEqualize:        nil,
		downSample:            nil,
		zoomIn:                nil,
		zoomOut:               nil,
		normalize:             nil,
	}
}

type Option func(o *Options)

// Compose creates a new Augment struct by adding augmentation methods.
func Compose(opts ...Option) (Transformer, error) {
	augOpts := defaultOption()
	for _, opt := range opts {
		if opt != nil {
			opt(augOpts)
		}
	}

	var augs *nn.Sequential = nn.Seq()

	if augOpts.rotate != nil {
		augs.Add(augOpts.rotate)
	}

	if augOpts.randRotate != nil {
		augs.Add(augOpts.randRotate)
	}

	if augOpts.resize != nil {
		augs.Add(augOpts.resize)
	}

	if augOpts.colorJitter != nil {
		augs.Add(augOpts.colorJitter)
	}

	if augOpts.gaussianBlur != nil {
		augs.Add(augOpts.gaussianBlur)
	}

	if augOpts.randomHFlip != nil {
		augs.Add(augOpts.randomHFlip)
	}

	if augOpts.randomVFlip != nil {
		augs.Add(augOpts.randomVFlip)
	}

	if augOpts.randomCrop != nil {
		augs.Add(augOpts.randomCrop)
	}

	if augOpts.centerCrop != nil {
		augs.Add(augOpts.centerCrop)
	}

	if augOpts.randomCutout != nil {
		augs.Add(augOpts.randomCutout)
	}

	if augOpts.randomPerspective != nil {
		augs.Add(augOpts.randomPerspective)
	}

	if augOpts.randomAffine != nil {
		augs.Add(augOpts.randomAffine)
	}

	if augOpts.randomGrayscale != nil {
		augs.Add(augOpts.randomGrayscale)
	}

	if augOpts.randomSolarize != nil {
		augs.Add(augOpts.randomSolarize)
	}

	if augOpts.randomPosterize != nil {
		augs.Add(augOpts.randomPosterize)
	}

	if augOpts.randomInvert != nil {
		augs.Add(augOpts.randomInvert)
	}

	if augOpts.randomAutocontrast != nil {
		augs.Add(augOpts.randomAutocontrast)
	}

	if augOpts.randomAdjustSharpness != nil {
		augs.Add(augOpts.randomAdjustSharpness)
	}

	if augOpts.randomEqualize != nil {
		augs.Add(augOpts.randomEqualize)
	}

	if augOpts.normalize != nil {
		augs.Add(augOpts.normalize)
	}

	if augOpts.downSample != nil {
		augs.Add(augOpts.downSample)
	}

	if augOpts.zoomIn != nil {
		augs.Add(augOpts.zoomIn)
	}

	if augOpts.zoomOut != nil {
		augs.Add(augOpts.zoomOut)
	}

	return &Augment{augs}, nil
}

// OneOf randomly return one transformer from list of transformers
// with a specific p value.
func OneOf(pvalue float64, tfOpts ...Option) Option {
	tfsNum := len(tfOpts)
	if tfsNum < 1 {
		return nil
	}

	randP := randPvalue()
	if randP >= pvalue {
		return nil
	}

	rand.Seed(time.Now().UnixNano())
	idx := rand.Intn(tfsNum)

	return tfOpts[idx]
}
