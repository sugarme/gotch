package vision

// Utility functions to manipulate images.

import (
	"fmt"
	"log"
	"path/filepath"

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
	// TODO: implement

	return
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

// TODO: should we need this func???
func visitDirs(dir string, files []string) (err error) {

	return nil
}

// LoadDir loads all the images in a directory.
func LoadDir(path string, outW int64, outH int64) (retVal ts.Tensor, err error) {
	// var files []string
	// TODO: implement it

	return
}
