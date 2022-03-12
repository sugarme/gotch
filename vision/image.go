package vision

// Utility functions to manipulate images.

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// (height, width, channel) -> (channel, height, width)
func hwcToCHW(tensor *ts.Tensor) *ts.Tensor {
	retVal, err := tensor.Permute([]int64{2, 0, 1}, false)
	if err != nil {
		log.Fatalf("hwcToCHW error: %v\n", err)
	}
	return retVal
}

func chwToHWC(tensor *ts.Tensor) *ts.Tensor {
	retVal, err := tensor.Permute([]int64{1, 2, 0}, false)
	if err != nil {
		log.Fatalf("hwcToCHW error: %v\n", err)
	}
	return retVal
}

// Load loads an image from a file.
//
// On success returns a tensor of shape [channel, height, width].
func Load(path string) (*ts.Tensor, error) {
	var tensor *ts.Tensor
	tensor, err := ts.LoadHwc(path)
	if err != nil {
		return nil, err
	}

	loadedTs := hwcToCHW(tensor)
	tensor.MustDrop()

	return loadedTs, nil
}

// Save saves an image to a file.
//
// This expects as input a tensor of shape [channel, height, width].
// The image format is based on the filename suffix, supported suffixes
// are jpg, png, tga, and bmp.
// The tensor input should be of kind UInt8 with values ranging from
// 0 to 255.
func Save(tensor *ts.Tensor, path string) error {
	t, err := tensor.Totype(gotch.Uint8, false) // false to keep the input tensor
	if err != nil {
		err = fmt.Errorf("Save - Tensor.Totype() error: %v\n", err)
		return err
	}

	shape, err := t.Size()
	if err != nil {
		err = fmt.Errorf("Save - Tensor.Size() error: %v\n", err)
		return err
	}

	var tsCHW, tsHWC *ts.Tensor
	switch {
	case len(shape) == 4 && shape[0] == 1:
		tsCHW = t.MustSqueezeDim(int64(0), true)
		chwTs := chwToHWC(tsCHW)
		tsCHW.MustDrop()
		tsHWC = chwTs.MustTo(gotch.CPU, true)
	case len(shape) == 3:
		chwTs := t.MustTo(gotch.CPU, true)
		tsHWC = chwToHWC(chwTs)
		chwTs.MustDrop()
	default:
		err = fmt.Errorf("Unexpected size (%v) for image tensor.\n", len(shape))
		return err
	}

	if err = ts.SaveHwc(tsHWC, path); err != nil {
		return err
	}

	tsHWC.MustDrop()
	return nil
}

// Resize resizes an image.
//
// This expects as input a tensor of shape [channel, height, width] and returns
// a tensor of shape [channel, out_h, out_w].
func Resize(t *ts.Tensor, outW int64, outH int64) (*ts.Tensor, error) {
	hwcTs := chwToHWC(t)
	tmpTs, err := ts.ResizeHwc(hwcTs, outW, outH)
	if err != nil {
		return nil, err
	}
	hwcTs.MustDrop()

	tsCHW := hwcToCHW(tmpTs)
	tmpTs.MustDrop()

	return tsCHW, nil
}

func resizePreserveAspectRatioHWC(t *ts.Tensor, outW int64, outH int64) (*ts.Tensor, error) {
	tsSize, err := t.Size()
	if err != nil {
		err = fmt.Errorf("resizePreserveAspectRatioHWC - ts.Size() method call err: %v\n", err)
		return nil, err
	}

	h := tsSize[1]
	w := tsSize[0]

	switch (w * outH) == (h * outW) {
	case true: // same ratio
		tmpTs, err := ts.ResizeHwc(t, outW, outH)
		if err != nil {
			err = fmt.Errorf("resizePreserveAspectRatioHWC - ts.ResizeHwc() method call err: %v\n", err)
			return nil, err
		}
		tsCHW := hwcToCHW(tmpTs)
		tmpTs.MustDrop()

		return tsCHW, nil

	case false:
		ratioW := float64(outW) / float64(h)
		ratioH := float64(outH) / float64(w)
		ratio := math.Max(ratioW, ratioH)

		resizeW := int64(ratio * float64(h))
		resizeH := int64(ratio * float64(w))

		resizeW = maxInt64(resizeW, outW)
		resizeH = maxInt64(resizeH, outH)

		tmpTs, err := ts.ResizeHwc(t, resizeW, resizeH)
		tsCHW := hwcToCHW(tmpTs)
		tmpTs.MustDrop()

		var tensorW *ts.Tensor
		if resizeW == outW {
			tensorW = tsCHW.MustShallowClone()
		} else {
			tensorW, err = tsCHW.Narrow(2, (resizeW-outW)/2, outW, false)
			if err != nil {
				err = fmt.Errorf("resizePreserveAspectRatioHWC - ts.Narrow() method call err: %v\n", err)
				return nil, err
			}
		}

		switch int64(resizeH) == outH {
		case true:
			tsCHW.MustDrop()
			return tensorW, nil
		case false:
			tensorH, err := tsCHW.Narrow(1, (resizeH-outH)/2, outH, true)
			if err != nil {
				err = fmt.Errorf("resizePreserveAspectRatioHWC - ts.Narrow() method call err: %v\n", err)
				return nil, err
			}

			tensorW.MustDrop()
			return tensorH, nil

		default:
			err = fmt.Errorf("Shouldn't reach here")
			return nil, err
		}

	default:
		err = fmt.Errorf("Shouldn't reach here")
		return nil, err
	}
}

// ResizePreserveAspectRatio resizes an image, preserve the aspect ratio by taking a center crop.
//
// This expects as input a tensor of shape [channel, height, width] and returns
func ResizePreserveAspectRatio(t *ts.Tensor, outW int64, outH int64) (*ts.Tensor, error) {
	hwcTs := chwToHWC(t)
	resizedTs, err := resizePreserveAspectRatioHWC(hwcTs, outW, outH)
	if err != nil {
		return nil, err
	}
	hwcTs.MustDrop()
	return resizedTs, nil
}

// LoadAndResize loads and resizes an image, preserve the aspect ratio by taking a center crop.
func LoadAndResize(path string, outW int64, outH int64) (*ts.Tensor, error) {
	tensor, err := ts.LoadHwc(path)
	if err != nil {
		return nil, err
	}

	resizedTs, err := resizePreserveAspectRatioHWC(tensor, outW, outH)
	if err != nil {
		return nil, err
	}
	tensor.MustDrop()

	return resizedTs, nil
}

// LoadDir loads all the images in a directory.
func LoadDir(dir string, outW int64, outH int64) (*ts.Tensor, error) {
	var filePaths []string // "dir/filename.ext"
	var tensors []ts.Tensor
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		err = fmt.Errorf("LoadDir - Read directory error: %v\n", err)
		return nil, err
	}
	for _, f := range files {
		filePaths = append(filePaths, fmt.Sprintf("%v%v", dir, f.Name()))
	}

	for _, path := range filePaths {
		tensor, err := LoadAndResize(path, outW, outH)
		if err != nil {
			err = fmt.Errorf("LoadDir - LoadAndResize method call error: %v\n", err)
			return nil, err
		}
		tensors = append(tensors, *tensor)
	}

	stackedTs, err := ts.Stack(tensors, 0)
	if err != nil {
		return nil, err
	}

	for i := 0; i < len(tensors); i++ {
		tensors[i].MustDrop()
	}

	return stackedTs, nil
}

func maxInt64(v1, v2 int64) int64 {
	if v1 > v2 {
		return v1
	}

	return v2
}
