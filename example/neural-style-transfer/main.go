package main

// This is inspired by the Neural Style tutorial from PyTorch.org
//   https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

const (
	StyleWeight  float64 = 1e6
	LearningRate float64 = 1e-1
	TotalSteps   int64   = 3000
)

var (
	StyleIndexes   []uint = []uint{0, 2, 5, 7, 10}
	ContentIndexes []int  = []int{7}

	model   string
	content string
	style   string
)

func gramMatrix(m ts.Tensor) (retVal ts.Tensor) {
	sizes, err := m.Size4()
	if err != nil {
		log.Fatal(err)
	}

	var (
		a int64 = sizes[0]
		b int64 = sizes[1]
		c int64 = sizes[2]
		d int64 = sizes[3]
	)

	mview := m.MustView([]int64{a * b, c * d}, false)
	mviewT := mview.MustT(false)
	gram := mview.MustMatMul(mviewT, true)
	mviewT.MustDrop()

	return gram.MustDiv1(ts.IntScalar(a*b*c*d), true)
}

func styleLoss(m1 ts.Tensor, m2 ts.Tensor) (retVal ts.Tensor) {
	gram1 := gramMatrix(m1)
	// m1.MustDrop()
	gram2 := gramMatrix(m2)
	// m2.MustDrop()
	loss := gram1.MustMseLoss(gram2, ts.ReductionMean.ToInt(), true)
	gram2.MustDrop()
	return loss
}

func init() {
	flag.StringVar(&model, "model", "../../data/neural-style-transfer/vgg16.pt", "VGG16 model file")
	flag.StringVar(&content, "content", "../../data/neural-style-transfer/content.jpg", "Content image file to test")
	flag.StringVar(&style, "style", "../../data/neural-style-transfer/style.jpg", "Style image file to save")
}

func main() {

	flag.Parse()

	modelPath, err := filepath.Abs(model)
	if err != nil {
		log.Fatal(err)
	}

	contentPath, err := filepath.Abs(content)
	if err != nil {
		log.Fatal(err)
	}

	stylePath, err := filepath.Abs(style)
	if err != nil {
		log.Fatal(err)
	}

	cuda := gotch.CudaBuilder(0)
	device := cuda.CudaIfAvailable()

	// device := gotch.CPU
	netVS := nn.NewVarStore(device)
	in := vision.NewImageNet()
	net := vision.VGG16(netVS.Root(), in.ClassCount())

	fmt.Printf("nclasses: %v\n", in.ClassCount())

	err = netVS.Load(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	netVS.Freeze()

	styleImage, err := in.LoadImage(stylePath)
	if err != nil {
		log.Fatal(err)
	}

	usStyle := styleImage.MustUnsqueeze(0, true)
	styleImg := usStyle.MustTo(device, true)

	fmt.Printf("styleImg size: %v\n", styleImg.MustSize())

	contentImage, err := in.LoadImage(contentPath)
	if err != nil {
		log.Fatal(err)
	}

	usContent := contentImage.MustUnsqueeze(0, true)
	contentImg := usContent.MustTo(device, true)

	var maxIndex uint = 0
	for _, i := range StyleIndexes {
		if i > maxIndex {
			maxIndex = i
		}
	}

	maxLayer := uint8(maxIndex + 1)

	fmt.Printf("max layer: %v\n", maxLayer)

	styleLayers := net.ForwardAllT(styleImg, false, maxLayer)
	contentLayers := net.ForwardAllT(contentImg, false, maxLayer)

	vs := nn.NewVarStore(device)
	path := vs.Root()
	inputVar := path.VarCopy("img", contentImg)
	opt, err := nn.DefaultAdamConfig().Build(vs, LearningRate)
	if err != nil {
		log.Fatal(err)
	}

	styleWeight := ts.FloatScalar(StyleWeight)
	for stepIdx := 1; stepIdx <= int(TotalSteps); stepIdx++ {
		inputLayers := net.ForwardAllT(inputVar, false, maxLayer)

		// var sLoss ts.Tensor
		sLoss := ts.MustZeros([]int64{1}, gotch.Float.CInt(), device.CInt())
		cLoss := ts.MustZeros([]int64{1}, gotch.Float.CInt(), device.CInt())
		for _, idx := range StyleIndexes {
			l := styleLoss(inputLayers[idx], styleLayers[idx])
			sLoss = sLoss.MustAdd(l, true)
			l.MustDrop()
		}
		for _, idx := range ContentIndexes {
			l := inputLayers[idx].MustMseLoss(contentLayers[idx], ts.ReductionMean.ToInt(), true)
			cLoss = cLoss.MustAdd(l, true)
			l.MustDrop()
		}

		for _, t := range inputLayers {
			t.MustDrop()
		}

		lossMul := sLoss.MustMul1(styleWeight, true)
		loss := lossMul.MustAdd(cLoss, true)
		opt.BackwardStep(loss)

		if (stepIdx % 10) == 0 {
			clone := inputVar.MustShallowClone()
			img := clone.MustDetach()
			// clone.MustDrop()
			err := in.SaveImage(img, fmt.Sprintf("../../data/neural-style-transfer/out%v.jpg", stepIdx))
			if err != nil {
				log.Fatal(err)
			}
			img.MustDrop()
		}

		fmt.Printf("Step %v ... Done. Loss %10.1f\n", stepIdx, loss.Values()[0])
		cLoss.MustDrop()
		loss.MustDrop()
	}

}
