package vision

// Utility functions to manipulate images.

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	// "path/filepath"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// (height, width, channel) -> (channel, height, width)
func hwcToCHW(tensor ts.Tensor) (retVal ts.Tensor) {
	var err error
	retVal, err = tensor.Permute([]int64{2, 0, 1})
	if err != nil {
		log.Fatalf("hwcToCHW error: %v\n", err)
	}
	return retVal
}

func chwToHWC(tensor ts.Tensor) (retVal ts.Tensor) {
	var err error
	retVal, err = tensor.Permute([]int64{1, 2, 0})
	if err != nil {
		log.Fatalf("hwcToCHW error: %v\n", err)
	}
	return retVal
}

// Load loads an image from a file.
//
// On success returns a tensor of shape [channel, height, width].
func Load(path string) (retVal ts.Tensor, err error) {
	var tensor ts.Tensor
	tensor, err = ts.LoadHwc(path)
	if err != nil {
		return retVal, err
	}

	retVal = hwcToCHW(tensor)
	return retVal, nil
}

// Save saves an image to a file.
//
// This expects as input a tensor of shape [channel, height, width].
// The image format is based on the filename suffix, supported suffixes
// are jpg, png, tga, and bmp.
// The tensor input should be of kind UInt8 with values ranging from
// 0 to 255.
func Save(tensor ts.Tensor, path string) (err error) {
	t, err := tensor.Totype(gotch.Uint8)
	if err != nil {
		err = fmt.Errorf("Save - Tensor.Totype() error: %v\n", err)
		return err
	}

	shape, err := t.Size()
	if err != nil {
		err = fmt.Errorf("Save - Tensor.Size() error: %v\n", err)
		return err
	}

	switch {
	case len(shape) == 4 && shape[0] == 1:
		return ts.SaveHwc(chwToHWC(t.MustSqueeze1(int64(0)).MustTo(gotch.CPU)), path)
	case len(shape) == 3:
		return ts.SaveHwc(chwToHWC(t.MustTo(gotch.CPU)), path)
	default:
		err = fmt.Errorf("Unexpected size (%v) for image tensor.\n", len(shape))
		return err
	}
}

// Resize resizes an image.
//
// This expects as input a tensor of shape [channel, height, width] and returns
// a tensor of shape [channel, out_h, out_w].
func Resize(t ts.Tensor, outW int64, outH int64) (retVal ts.Tensor, err error) {
	tmpTs, err := ts.ResizeHwc(t, outW, outH)
	if err != nil {
		return retVal, err
	}
	retVal = hwcToCHW(tmpTs)

	return retVal, nil
}

// TODO: implement
func resizePreserveAspectRatioHWC(t ts.Tensor, outW int64, outH int64) (retVal ts.Tensor, err error) {
	tsSize, err := t.Size()
	if err != nil {
		err = fmt.Errorf("resizePreserveAspectRatioHWC - ts.Size() method call err: %v\n", err)
		return retVal, err
	}
	w := tsSize[0]
	h := tsSize[1]
	if w*outH == h*outW {
		tmpTs, err := ts.ResizeHwc(t, outW, outH)
		if err != nil {
			err = fmt.Errorf("resizePreserveAspectRatioHWC - ts.ResizeHwc() method call err: %v\n", err)
			return retVal, err
		}
		return hwcToCHW(tmpTs), nil
	} else {

		ratioW := float64(outW) / float64(w)
		ratioH := float64(outH) / float64(h)
		ratio := math.Max(ratioW, ratioH)

		resizeW := int64(ratio) * h
		resizeH := int64(ratio) * w

		resizeW = int64(math.Max(float64(resizeW), float64(outW)))
		resizeH = int64(math.Max(float64(resizeH), float64(outH)))
		tmpTs, err := ts.ResizeHwc(t, resizeW, resizeH)

		tensor := hwcToCHW(tmpTs)

		var tensorW ts.Tensor
		var tensorH ts.Tensor
		if resizeW != outW {
			tensorW, err = tensor.Narrow(2, (resizeW-outW)/2, outW)
			if err != nil {
				err = fmt.Errorf("resizePreserveAspectRatioHWC - ts.Narrow() method call err: %v\n", err)
				return retVal, err
			}
		}

		if resizeH == outH {
			retVal = tensorW
		} else {
			tensorH, err = tensor.Narrow(2, (resizeH-outH)/2, outH)
			if err != nil {
				err = fmt.Errorf("resizePreserveAspectRatioHWC - ts.Narrow() method call err: %v\n", err)
				return retVal, err
			}
			retVal = tensorH
		}

		return retVal, nil
	}
}

// ResizePreserveAspectRatio resizes an image, preserve the aspect ratio by taking a center crop.
//
// This expects as input a tensor of shape [channel, height, width] and returns
func ResizePreserveAspectRatio(t ts.Tensor, outW int64, outH int64) (retVal ts.Tensor, err error) {
	return resizePreserveAspectRatioHWC(chwToHWC(t), outW, outH)
}

// LoadAndResize loads and resizes an image, preserve the aspect ratio by taking a center crop.
func LoadAndResize(path string, outW int64, outH int64) (retVal ts.Tensor, err error) {
	tensor, err := ts.LoadHwc(path)
	if err != nil {
		return retVal, err
	}

	return resizePreserveAspectRatioHWC(tensor, outW, outH)
}

// LoadDir loads all the images in a directory.
func LoadDir(dir string, outW int64, outH int64) (retVal ts.Tensor, err error) {
	var filePaths []string // "dir/filename.ext"
	var tensors []ts.Tensor
	// TODO: implement it
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		err = fmt.Errorf("LoadDir - Read directory error: %v\n", err)
		return retVal, err
	}
	for _, f := range files {
		filePaths = append(filePaths, fmt.Sprintf("%v%v", dir, f.Name()))
	}

	for _, path := range filePaths {
		tensor, err := LoadAndResize(path, outW, outH)
		if err != nil {
			err = fmt.Errorf("LoadDir - LoadAndResize method call error: %v\n", err)
			return retVal, err
		}
		tensors = append(tensors, tensor)
	}

	return ts.Stack(tensors, 0)
}
