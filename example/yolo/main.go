package main

import (
	"flag"
	"fmt"
	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
	"log"
	"path/filepath"
)

const configName = "yolo-v3.cfg"

var (
	model string
	image string
)

func init() {
	flag.StringVar(&model, "model", "../../data/yolo/yolo-v3.pt", "Yolo model weights file")
	flag.StringVar(&image, "image", "../../data/yolo/bondi.jpg", "image file to infer")
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

	imagePath, err := filepath.Abs(image)
	if err != nil {
		log.Fatal(err)
	}

	var darknet Darknet = ParseConfig(configPath)

	fmt.Printf("darknet number of parameters: %v\n", len(darknet.Parameters))
	fmt.Printf("darknet number of blocks: %v\n", len(darknet.Blocks))

	vs := nn.NewVarStore(gotch.CPU)
	model := darknet.BuildModel(vs.Root())
	fmt.Printf("Model: %v\n", model)
	fmt.Printf("Image path: %v\n", imagePath)

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
	fmt.Printf("Image shape: %v\n", originalImage.MustSize())

	netHeight := darknet.Height()
	netWidth := darknet.Width()

	fmt.Printf("net Height: %v\n", netHeight)
	fmt.Printf("net Width: %v\n", netWidth)

	imageTs, err := vision.Resize(originalImage, netWidth, netHeight)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("imageTs shape: %v\n", imageTs.MustSize())

	imgTmp1 := imageTs.MustUnsqueeze(0, true)
	imgTmp2 := imgTmp1.MustTotype(gotch.Float, true)
	img := imgTmp2.MustDiv1(ts.FloatScalar(255.0), true)
	predictTmp := model.ForwardT(img, false)
	// predictions := predictTmp.MustSqueeze(true)
	fmt.Printf("predictTmp shape: %v\n", predictTmp.MustSize())

}
