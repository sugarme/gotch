package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"path/filepath"
	"sort"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

const (
	saveDir             string  = "../../data/yolo"
	configName          string  = "yolo-v3.cfg"
	confidenceThreshold float64 = 0.5
	nmsThreshold        float64 = 0.4
)

var (
	model     string
	imageFile string
)

type Bbox struct {
	xmin            float64
	ymin            float64
	xmax            float64
	ymax            float64
	confidence      float64
	classIndex      uint
	classConfidence float64
}

type ByConfBbox []Bbox

// Implement sort.Interface for []Bbox on Bbox.confidence:
// =====================================================
func (bb ByConfBbox) Len() int           { return len(bb) }
func (bb ByConfBbox) Less(i, j int) bool { return bb[i].confidence < bb[j].confidence }
func (bb ByConfBbox) Swap(i, j int)      { bb[i], bb[j] = bb[j], bb[i] }

// Intersection over union of two bounding boxes.
func Iou(b1, b2 Bbox) (retVal float64) {
	b1Area := (b1.xmax - b1.xmin + 1.0) * (b1.ymax - b1.ymin + 1.0)
	b2Area := (b2.xmax - b2.xmin + 1.0) * (b2.ymax - b2.ymin + 1.0)

	iXmin := math.Max(b1.xmin, b2.xmin)
	iXmax := math.Min(b1.xmax, b2.xmax)
	iYmin := math.Max(b1.ymin, b2.ymin)
	iYmax := math.Min(b1.ymax, b2.ymax)

	iArea := math.Max((iXmax-iXmin+1.0), 0.0) * math.Max((iYmax-iYmin+1.0), 0)

	return (iArea) / (b1Area + b2Area - iArea)
}

// Assuming x1 <= x2 and y1 <= y2
func drawRect(t *ts.Tensor, x1, x2, y1, y2 int64) {
	color := ts.MustOfSlice([]float64{0.0, 0.0, 1.0}).MustView([]int64{3, 1, 1}, true)

	// NOTE: `narrow` will create a tensor (view) that share same storage with
	// original one.
	tmp1 := t.MustNarrow(2, x1, x2-x1, false)
	tmp2 := tmp1.MustNarrow(1, y1, y2-y1, true)
	tmp2.Copy_(color)
	tmp2.MustDrop()
	color.MustDrop()
}

func drawLabel(t *ts.Tensor, text []string, x, y int64) {
	device, err := t.Device()
	if err != nil {
		log.Fatal(err)
	}
	label := textToImageTs(text).MustTo(device, true)

	labelSize := label.MustSize()
	height := labelSize[1]
	width := labelSize[2]

	imageSize := t.MustSize()
	lenY := height
	if lenY > imageSize[1] {
		lenY = imageSize[1] - y
	}

	lenX := width
	if lenX > imageSize[2] {
		lenX = imageSize[2] - x
	}

	// NOTE: `narrow` will create a tensor (view) that share same storage with
	// original one.

	tmp1 := t.MustNarrow(2, x, lenX, false)
	tmp2 := tmp1.MustNarrow(1, y, lenY, true)
	tmp2.Copy_(label)
	tmp2.MustDrop()
	label.MustDrop()
}

func report(pred *ts.Tensor, img *ts.Tensor, w int64, h int64) *ts.Tensor {
	size2, err := pred.Size2()
	if err != nil {
		log.Fatal(err)
	}
	npreds := size2[0]
	predSize := size2[1]

	nclasses := uint(predSize - 5)

	// The bounding boxes grouped by (maximum) class index.
	var bboxes [][]Bbox = make([][]Bbox, int(nclasses))

	// Extract the bounding boxes for which confidence is above the threshold.
	for index := 0; index < int(npreds); index++ {
		predIdx := pred.MustGet(index)
		var predVals []float64 = predIdx.Float64Values()
		predIdx.MustDrop()

		confidence := predVals[4]
		if confidence > confidenceThreshold {
			classIndex := 0
			for i := 0; i < int(nclasses); i++ {
				if predVals[5+i] > predVals[5+classIndex] {
					classIndex = i
				}
			}

			if predVals[classIndex+5] > 0.0 {
				bbox := Bbox{
					xmin:            predVals[0] - (predVals[2] / 2.0),
					ymin:            predVals[1] - (predVals[3] / 2.0),
					xmax:            predVals[0] + (predVals[2] / 2.0),
					ymax:            predVals[1] + (predVals[3] / 2.0),
					confidence:      confidence,
					classIndex:      uint(classIndex),
					classConfidence: predVals[5+classIndex],
				}

				bboxes[classIndex] = append(bboxes[classIndex], bbox)
			}
		}

	}

	// Perform non-maximum suppression.
	var bboxesRes [][]Bbox
	for _, bboxesForClass := range bboxes {
		// 1. Sort by confidence
		sort.Sort(ByConfBbox(bboxesForClass))

		// 2.
		var currentIndex = 0
		for index := 0; index < len(bboxesForClass); index++ {
			drop := false
			for predIndex := 0; predIndex < currentIndex; predIndex++ {
				iou := Iou(bboxesForClass[predIndex], bboxesForClass[index])
				if iou > nmsThreshold {
					drop = true
					break
				}
			}

			if !drop {
				// swap
				bboxesForClass[currentIndex], bboxesForClass[index] = bboxesForClass[index], bboxesForClass[currentIndex]
				currentIndex += 1
			}
		}
		// 3. Truncate at currentIndex (exclusive)
		if currentIndex < len(bboxesForClass) {
			bboxesForClass = append(bboxesForClass[:currentIndex])
		}

		bboxesRes = append(bboxesRes, bboxesForClass)
	}

	// Annotate the original image and print boxes information.
	size3, err := img.Size3()
	if err != nil {
		log.Fatal(err)
	}
	initialH := size3[1]
	initialW := size3[2]

	imageTmp := img.MustTotype(gotch.Float, false)
	image := imageTmp.MustDiv1(ts.FloatScalar(255.0), true)

	var wRatio float64 = float64(initialW) / float64(w)
	var hRatio float64 = float64(initialH) / float64(h)

	for classIndex, bboxesForClass := range bboxesRes {
		for _, b := range bboxesForClass {
			fmt.Printf("%v: %v\n", CocoClasses[classIndex], b)

			xmin := min(max(int64(b.xmin*wRatio), 0), (initialW - 1))
			ymin := min(max(int64(b.ymin*hRatio), 0), (initialH - 1))
			xmax := min(max(int64(b.xmax*wRatio), 0), (initialW - 1))
			ymax := min(max(int64(b.ymax*hRatio), 0), (initialH - 1))

			// draw rect
			drawRect(image, xmin, xmax, ymin, min(ymax, ymin+2))
			drawRect(image, xmin, xmax, max(ymin, ymax-2), ymax)
			drawRect(image, xmin, min(xmax, xmin+2), ymin, ymax)
			drawRect(image, max(xmin, xmax-2), xmax, ymin, ymax)

			label := fmt.Sprintf("%v; %.3f\n", CocoClasses[classIndex], b.confidence)
			drawLabel(image, []string{label}, xmin, ymin-15)
		}
	}

	imgTmp := image.MustMul1(ts.FloatScalar(255.0), true)
	retVal := imgTmp.MustTotype(gotch.Uint8, true)

	return retVal
}

func init() {
	flag.StringVar(&model, "model", "../../data/yolo/yolo-v3.pt", "Yolo model weights file")
	flag.StringVar(&imageFile, "image", "../../data/yolo/bondi.jpg", "image file to infer")
}

func main() {

	flag.Parse()
	configPath, err := filepath.Abs(configName)
	if err != nil {
		log.Fatal(err)
	}

	modelPath, err := filepath.Abs(model)
	if err != nil {
		log.Fatal(err)
	}

	imagePath, err := filepath.Abs(imageFile)
	if err != nil {
		log.Fatal(err)
	}

	var darknet *Darknet = ParseConfig(configPath)

	vs := nn.NewVarStore(gotch.CPU)
	model := darknet.BuildModel(vs.Root())

	err = vs.Load(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Yolo weights loaded.")

	originalImage, err := vision.Load(imagePath)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Image file loaded")

	netHeight := darknet.Height()
	netWidth := darknet.Width()

	imgClone := originalImage.MustShallowClone().MustDetach(false)

	imageTs, err := vision.Resize(imgClone, netWidth, netHeight)
	if err != nil {
		log.Fatal(err)
	}

	imgTmp1 := imageTs.MustUnsqueeze(0, true)
	imgTmp2 := imgTmp1.MustTotype(gotch.Float, true)
	img := imgTmp2.MustDivScalar(ts.FloatScalar(255.0), true)
	predictTmp := model.ForwardT(img, false)

	predictions := predictTmp.MustSqueeze(true)

	imgRes := report(predictions, originalImage, netWidth, netHeight)

	savePath, err := filepath.Abs(saveDir)
	if err != nil {
		log.Fatal(err)
	}

	inputFile := filepath.Base(imagePath)
	saveFile := fmt.Sprintf("%v/yolo_%v", savePath, inputFile)
	err = vision.Save(imgRes, saveFile)
	if err != nil {
		log.Fatal(err)
	}
}

func max(v1, v2 int64) (retVal int64) {
	if v1 > v2 {
		return v1
	}

	return v2
}

func min(v1, v2 int64) (retVal int64) {
	if v1 < v2 {
		return v1
	}

	return v2
}
