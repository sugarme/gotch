package vision

// The MNIST hand-written digit dataset.
//
// The files can be obtained from the following link:
// http://yann.lecun.com/exdb/mnist/

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// readInt32 read 4 bytes and convert to MSB first (big endian) interger.
func readInt32(f *os.File) (retVal int, err error) {
	buf := make([]byte, 4)
	n, err := f.Read(buf)
	switch {
	case err != nil:
		return 0, err
	case n != 4:
		err = fmt.Errorf("Invalid format: %v", f.Name())
		return 0, err
	}

	// flip to big endian
	var v int = 0
	for _, i := range buf {
		v = v*256 + int(i)
	}

	return v, nil
}

// checkMagicNumber checks the magic number located at the first 4 bytes of
// mnist files.
func checkMagicNumber(f *os.File, wantNumber int) (err error) {
	gotNumber, err := readInt32(f)
	if err != nil {
		return err
	}

	if gotNumber != wantNumber {
		err = fmt.Errorf("incorrect magic number: got %v want %v\n", gotNumber, wantNumber)
		return err
	}

	return nil
}

func readLabels(filename string) *ts.Tensor {

	f, err := os.Open(filename)
	if err != nil {
		log.Fatalf("readLabels errors: %v\n", err)
	}
	defer f.Close()

	if err = checkMagicNumber(f, 2049); err != nil {
		log.Fatal(err)
	}

	samples, err := readInt32(f)
	if err != nil {
		log.Fatal(err)
	}

	var data []uint8 = make([]uint8, samples)
	len, err := f.Read(data)
	if err != nil || len != samples {
		err = fmt.Errorf("invalid format %v", f.Name())
		log.Fatal(err)
	}

	labelsTs, err := ts.OfSlice(data)
	if err != nil {
		err = fmt.Errorf("create label tensor err.")
		log.Fatal(err)
	}

	return labelsTs.MustTotype(gotch.Int64, true)
}

func readImages(filename string) *ts.Tensor {
	f, err := os.Open(filename)
	if err != nil {
		log.Fatalf("readImages errors: %v\n", err)
	}
	defer f.Close()

	if err = checkMagicNumber(f, 2051); err != nil {
		log.Fatal(err)
	}

	samples, err := readInt32(f)
	if err != nil {
		log.Fatal(err)
	}

	rows, err := readInt32(f)
	if err != nil {
		log.Fatal(err)
	}
	cols, err := readInt32(f)
	if err != nil {
		log.Fatal(err)
	}

	dataLen := samples * rows * cols
	var data []uint8 = make([]uint8, dataLen)
	len, err := f.Read(data)
	if err != nil || len != dataLen {
		err = fmt.Errorf("invalid format %v", f.Name())
		log.Fatal(err)
	}

	imagesTs, err := ts.OfSlice(data)
	if err != nil {
		err = fmt.Errorf("create images tensor err.")
		log.Fatal(err)
	}

	return imagesTs.MustView([]int64{int64(samples), int64(rows * cols)}, true).MustTotype(gotch.Float, true).MustDivScalar(ts.FloatScalar(255.0), true)
}

// LoadMNISTDir loads all MNIST data from a given directory to Dataset
func LoadMNISTDir(dir string) *Dataset {
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

	trainImagesTs := readImages(trainImagesFile)
	trainLabelsTs := readLabels(trainLabelsFile)
	testImagesTs := readImages(testImagesFile)
	testLabelsTs := readLabels(testLabelsFile)

	return &Dataset{
		TrainImages: trainImagesTs,
		TrainLabels: trainLabelsTs,
		TestImages:  testImagesTs,
		TestLabels:  testLabelsTs,
		Labels:      10,
	}
}
