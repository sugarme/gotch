package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/vision"
	"github.com/sugarme/gotch/vision/aug"
)

func main() {

	// roundTrip()
	tOne()

}

func roundTrip() {
	img, err := vision.Load("./bb.png")
	if err != nil {
		panic(err)
	}

	fmt.Printf("%i", img)
	fimg := aug.Byte2FloatImage(img)
	fmt.Printf("%i", fimg)

	bimg := aug.Float2ByteImage(fimg)
	fmt.Printf("%i", bimg)

	err = vision.Save(bimg, "./bimg.png")
	if err != nil {
		log.Fatal(err)
	}
}

func tOne() {
	img, err := vision.Load("bb.png")
	if err != nil {
		panic(err)
	}

	// device := gotch.CudaIfAvailable()
	device := gotch.CPU
	imgTs := img.MustTo(device, true)
	// h := imgTs.MustSize()[1]
	// w := imgTs.MustSize()[2]

	// t, err := aug.Compose(aug.WithRandomAutocontrast(1.0))
	// t, err := aug.Compose(aug.WithRandomSolarize(aug.WithSolarizeThreshold(125), aug.WithSolarizePvalue(1.0)))
	// t, err := aug.Compose(aug.WithRandomAdjustSharpness(aug.WithSharpnessPvalue(1.0), aug.WithSharpnessFactor(10)))

	// Rotate
	// t, err := aug.Compose(aug.WithRandRotate(15, 15))
	// t, err := aug.Compose(aug.WithRotate(15))
	// t, err := aug.Compose(aug.WithRandomAffine(aug.WithAffineDegree([]int64{0, 15})))

	// DownSample
	// t, err := aug.Compose(aug.WithResize(h/2, w/2))
	// t, err := aug.Compose(aug.WithResize(320, 320))
	// ZoomOut
	// t, err := aug.Compose(aug.WithZoomOut(0.3))

	// t, err := aug.Compose(aug.WithRandomPosterize(aug.WithPosterizeBits(2), aug.WithPosterizePvalue(1.0)))

	// t, err := aug.Compose(aug.WithRandomPerspective(aug.WithPerspectiveScale(0.6), aug.WithPerspectivePvalue(1.0)))
	// t, err := aug.Compose(aug.WithNormalize(aug.WithNormalizeMean([]float64{0.485, 0.456, 0.406}), aug.WithNormalizeStd([]float64{0.229, 0.224, 0.225})))
	// t, err := aug.Compose(aug.WithRandomInvert(1.0))
	// t, err := aug.Compose(aug.WithRandomGrayscale(1.0))
	// t, err := aug.Compose(aug.WithRandomVFlip(1.0))
	// t, err := aug.Compose(aug.WithRandomHFlip(1.0))
	// t, err := aug.Compose(aug.WithRandomEqualize(1.0))
	// t, err := aug.Compose(aug.WithRandomCutout(aug.WithCutoutValue([]int64{124, 96, 255}), aug.WithCutoutScale([]float64{0.01, 0.1}), aug.WithCutoutRatio([]float64{0.5, 0.5}), aug.WithCutoutPvalue(1.0)))
	t, err := aug.Compose(aug.WithRandomCutout(aug.WithCutoutValue([]int64{127, 127, 127}), aug.WithCutoutRatio([]float64{0.01, 0.2}), aug.WithCutoutPvalue(1.0)))
	// t, err := aug.Compose(aug.WithRandomCutout(aug.WithCutoutScale([]float64{0.3, 0.3}), aug.WithCutoutPvalue(1.0)))

	// t, err := aug.Compose(aug.WithCenterCrop([]int64{320, 320}))
	// t, err := aug.Compose(aug.WithRandomAutocontrast())
	// t, err := aug.Compose(aug.WithColorJitter(aug.WithColorBrightness([]float64{1.3, 1.3})))
	// t, err := aug.Compose(aug.WithColorJitter(aug.WithColorSaturation([]float64{1.3, 1.3})))
	// t, err := aug.Compose(aug.WithColorJitter(aug.WithColorContrast([]float64{1.3, 1.3})))
	// t, err := aug.Compose(aug.WithColorJitter(aug.WithColorHue([]float64{0.3})))
	// t, err := aug.Compose(aug.WithGaussianBlur([]int64{5, 5}, []float64{1.0, 2.0}))
	// t, err := aug.Compose(aug.WithRandomAffine(aug.WithAffineDegree([]int64{0, 15}), aug.WithAffineShear([]float64{0, 15})))
	// t, err := aug.Compose(aug.WithRandomAffine(aug.WithAffineDegree([]int64{0, 15}), aug.WithAffineTranslate([]float64{0.0, 0.1})))

	out := t.Transform(imgTs)
	fname := fmt.Sprintf("./bb-transformed.jpg")
	err = vision.Save(out, fname)
	if err != nil {
		panic(err)
	}
	imgTs.MustDrop()
	out.MustDrop()
}

func tMany() {
	n := 360
	for i := 1; i <= n; i++ {
		img, err := vision.Load("./bb.png")
		if err != nil {
			panic(err)
		}

		// device := gotch.CudaIfAvailable()
		device := gotch.CPU
		imgTs := img.MustTo(device, true)

		t, err := aug.Compose(
			aug.WithResize(200, 200),
			aug.WithRandomVFlip(0.5),
			aug.WithRandomHFlip(0.5),
			aug.WithRandomCutout(),
			aug.OneOf(
				0.3,
				aug.WithColorJitter(aug.WithColorBrightness([]float64{0.3})),
				aug.WithRandomGrayscale(1.0),
			),
			aug.OneOf(
				0.3,
				aug.WithGaussianBlur([]int64{5, 5}, []float64{1.0, 2.0}),
				aug.WithRandomAffine(),
			),
		)
		if err != nil {
			panic(err)
		}

		out := t.Transform(imgTs)
		fname := fmt.Sprintf("./output/bb-%03d.png", i)
		err = vision.Save(out, fname)
		if err != nil {
			panic(err)
		}
		imgTs.MustDrop()
		out.MustDrop()

		fmt.Printf("%03d/%v completed.\n", i, n)
	}

}
