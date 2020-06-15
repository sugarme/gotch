package vision

// The MNIST hand-written digit dataset.
//
// The files can be obtained from the following link:
// http://yann.lecun.com/exdb/mnist/

import (
	"encoding/binary"
	"io"
	"log"
	"os"
	"path/filepath"

	ts "github.com/sugarme/gotch/tensor"
)

// Image holds the pixel intensities of an image.
// 255 is foreground (black), 0 is background (white).
type RawImage []byte

const numLabels = 10
const pixelRange = 255

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	// Width of the input tensor / picture
	Width = 28
	// Height of the input tensor / picture
	Height = 28
)

func readLabels(r io.Reader, e error) (retVal ts.Tensor) {
	if e != nil {
		log.Fatalf("readLabels errors: %v\n", e)
	}

	var (
		magic int32
		n     int32
		err   error
	)

	// Check magic number
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		log.Fatalf("readLabels - binary.Read error: %v\n", err)
	}
	if magic != labelMagic {
		log.Fatal(os.ErrInvalid)
	}

	// Now decode number
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		log.Fatalf("readLabels - binary.Read error: %v\n", err)
	}

	// label is a digit number range 0 - 9
	labels := make([]uint8, n)
	for i := 0; i < int(n); i++ {
		var l uint8
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			log.Fatalf("readLabels - binary.Read error: %v\n", err)
		}
		labels[i] = l
	}

	retVal, err = ts.OfSlice(labels)
	if err != nil {
		log.Fatalf("readLabels - ts.OfSlice error: %v\n", err)
	}

	return retVal
}

func readImages(r io.Reader, e error) (retVal ts.Tensor) {
	if e != nil {
		log.Fatalf("readLabels errors: %v\n", e)
	}

	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
		err   error
	)

	// Check magic number
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		log.Fatalf("readImages - binary.Read error: %v\n", err)
	}

	if magic != imageMagic {
		log.Fatalf("readImages - incorrect imageMagic: %v\n", err) // err is os.ErrInvalid
	}

	// Now, decode image
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		log.Fatalf("readImages - binary.Read error: %v\n", err)
	}
	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		log.Fatalf("readImages - binary.Read error: %v\n", err)
	}
	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		log.Fatalf("readImages - binary.Read error: %v\n", err)
	}

	imgs := make([]RawImage, n)
	m := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgs[i] = make(RawImage, m)
		m_, err := io.ReadFull(r, imgs[i])
		if err != nil {
			log.Fatalf("readImages - io.ReadFull error: %v\n", err)
		}
		if m_ != int(m) {
			log.Fatalf("readImages - image matrix size mismatched error: %v\n", os.ErrInvalid)
		}
	}

	retVal, err = ts.NewTensorFromData(imgs, []int64{int64(n), int64(nrow * ncol)})
	if err != nil {
		log.Fatalf("readImages - ts.NewTensorFromData error: %v\n", err)
	}

	return retVal
}

// LoadMNISTDir loads all MNIST data from a given directory to Dataset
func LoadMNISTDir(dir string) (retVal Dataset) {
	const (
		trainLabels = "train-labels-idx1-ubyte"
		trainImages = "train-images-idx3-ubyte"
		testLabels  = "t10k-labels-idx1-ubyte"
		testImages  = "t10k-images-idx3-ubyte"
	)

	trainLabelsFile := filepath.Join(dir, trainLabels)
	trainImagesFile := filepath.Join(dir, trainImages)
	testLabelsFile := filepath.Join(dir, testLabels)
	testImagesFile := filepath.Join(dir, testImages)

	trainImagesTs := readImages(os.Open(trainImagesFile))
	trainLabelsTs := readLabels(os.Open(trainLabelsFile))
	testImagesTs := readImages(os.Open(testImagesFile))
	testLabelsTs := readLabels(os.Open(testLabelsFile))

	return Dataset{
		TrainImages: trainImagesTs,
		TrainLabels: trainLabelsTs,
		TestImages:  testImagesTs,
		TestLabels:  testLabelsTs,
		Labels:      10,
	}
}
