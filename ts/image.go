package ts

import "C"

import (
	// "unsafe"

	lib "github.com/sugarme/gotch/libtch"
)

// LoadHwc returns a tensor of shape [height, width, channels] on success.
func LoadHwc(path string) (*Tensor, error) {

	ctensor := lib.AtLoadImage(path)
	err := TorchErr()
	if err != nil {
		return nil, err
	}

	return &Tensor{ctensor}, nil
}

// SaveHwc save an image from tensor. It expects a tensor of shape [height,
// width, channels]
func SaveHwc(ts *Tensor, path string) error {

	lib.AtSaveImage(ts.ctensor, path)
	return TorchErr()
}

// ResizeHwc expects a tensor of shape [height, width, channels].
// On success returns a tensor of shape [height, width, channels].
func ResizeHwc(ts *Tensor, outWidth, outHeight int64) (*Tensor, error) {

	ctensor := lib.AtResizeImage(ts.ctensor, outWidth, outHeight)
	err := TorchErr()
	if err != nil {
		return nil, err
	}

	return &Tensor{ctensor}, nil
}
