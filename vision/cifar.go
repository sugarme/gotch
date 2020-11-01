package vision

// The CIFAR-10 dataset.
//
// The files can be downloaded from the following page:
// https://www.cs.toronto.edu/~kriz/cifar.html
// The binary version of the dataset is used.

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

const (
	cfW            int64 = 32
	cfH            int64 = 32
	cfC            int64 = 3
	bytesPerImage  int64 = cfW*cfH*cfC + 1
	samplesPerFile int64 = 10000
)

func readFile(filename string) (imagesTs *ts.Tensor, labelsTs *ts.Tensor) {
	f, err := os.Open(filename)
	if err != nil {
		log.Fatalf("readImages errors: %v\n", err)
	}
	defer f.Close()

	dataLen := int(samplesPerFile * bytesPerImage)
	var data []uint8 = make([]uint8, dataLen)
	len, err := f.Read(data)
	if err != nil || len != dataLen {
		err = fmt.Errorf("invalid format %v", f.Name())
		log.Fatal(err)
	}

	content, err := ts.OfSlice(data)
	if err != nil {
		err = fmt.Errorf("create images tensor err.")
		log.Fatal(err)
	}

	images := ts.MustZeros([]int64{samplesPerFile, cfC, cfH, cfW}, gotch.Float, gotch.CPU)
	labels := ts.MustZeros([]int64{samplesPerFile}, gotch.Int64, gotch.CPU)

	for idx := 0; idx < int(samplesPerFile); idx++ {
		contentOffset := int(bytesPerImage) * idx

		labelContentTs := content.Idx(ts.NewSelect(int64(contentOffset)))
		selectLabelTs := labels.Idx(ts.NewSelect(int64(idx)))
		selectLabelTs.Copy_(labelContentTs)
		labelContentTs.MustDrop()

		tmp1 := content.MustNarrow(0, int64(1+contentOffset), int64(bytesPerImage-1), false)
		tmp2 := tmp1.MustView([]int64{cfC, cfH, cfW}, true)
		tmp3 := tmp2.MustTo(gotch.CPU, true)

		// NOTE: tensor indexing operations return view on the same memory
		// images.Idx(ts.NewSelect(int64(idx))).Copy_(tmp3)
		images.Idx(ts.NewSelect(int64(idx))).MustView([]int64{cfC, cfH, cfW}, false).Copy_(tmp3)
		tmp3.MustDrop()
	}

	tmp1 := images.MustTotype(gotch.Float, true)
	imagesTs = tmp1.MustDiv1(ts.FloatScalar(255.0), true)

	labelsTs = labels

	return imagesTs, labelsTs
}

func CFLoadDir(dir string) *Dataset {

	dirAbs, err := filepath.Abs(dir)
	if err != nil {
		log.Fatal(err)
	}

	testImages, testLabels := readFile(fmt.Sprintf("%v/test_batch.bin", dirAbs))

	var trainImages []ts.Tensor
	var trainLabels []ts.Tensor

	trainFiles := []string{
		"data_batch_1.bin",
		"data_batch_2.bin",
		"data_batch_3.bin",
		"data_batch_4.bin",
		"data_batch_5.bin",
	}

	for _, f := range trainFiles {
		img, l := readFile(fmt.Sprintf("%v/%v", dirAbs, f))
		trainImages = append(trainImages, *img)
		trainLabels = append(trainLabels, *l)
	}

	return &Dataset{
		TrainImages: ts.MustCat(trainImages, 0),
		TrainLabels: ts.MustCat(trainLabels, 0),
		TestImages:  testImages,
		TestLabels:  testLabels,
		Labels:      10,
	}
}
