package wrapper

import "C"

import (
	// "unsafe"

	lib "github.com/sugarme/gotch/libtch"
)

// LoadHwc returns a tensor of shape [width, height, channels] on success.
func LoadHwc(path string) (retVal Tensor, err error) {

	ctensor := lib.AtLoadImage(path)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor}
	return retVal, nil
}

// SaveHwc save an image from tensor. It expects a tensor of shape [width,
// height, channels]
func SaveHwc(ts Tensor, path string) (err error) {

	lib.AtSaveImage(ts.ctensor, path)
	return TorchErr()
}

// ResizeHwc expects a tensor of shape [width, height, channels].
// On success returns a tensor of shape [width, height, channels].
func ResizeHwc(ts Tensor, outWidth, outHeight int64) (retVal Tensor, err error) {

	ctensor := lib.AtResizeImage(ts.ctensor, outWidth, outHeight)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}
	retVal = Tensor{ctensor}

	return retVal, nil
}
