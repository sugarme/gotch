package main

// This example illustrates how to use pre-trained vision models.
// model to get the imagenet label for some image.

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/gotch/vision"
)

var (
	model string
	image string
)

func init() {
	flag.StringVar(&model, "model", "../../data/pretrained/resnet18.pt", "Model weights for inference")
	flag.StringVar(&image, "image", "../../data/pretrained/koala.jpg", "Image file to get imagenet label")
}

func main() {
	flag.Parse()

	imagePath, err := filepath.Abs(image)
	if err != nil {
		log.Fatal(err)
	}
	modelPath, err := filepath.Abs(model)
	if err != nil {
		log.Fatal(err)
	}

	in := vision.NewImageNet()

	// Load the image file and resize it to the usual imagenet dimension of 224x224.
	imageTs, err := in.LoadImageAndResize224(imagePath)
	if err != nil {
		log.Fatal(err)
	}

	// Create the model and load the weights from the file.
	_, modelFile := filepath.Split(modelPath)
	modelName := strings.TrimSuffix(modelFile, filepath.Ext(modelFile))

	// Create the model and load the weights from the file.
	vs := nn.NewVarStore(gotch.CPU)
	var net ts.ModuleT
	switch modelName {
	case "resnet18":
		net = vision.ResNet18(vs.Root(), in.ClassCount())
		err = vs.Load(modelPath)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("ResNet18 weights loaded.")
	case "vgg16":
		net = vision.VGG16(vs.Root(), in.ClassCount())
		err = vs.Load(modelPath)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("VGG16 weights loaded.")
	case "alexnet":
		net = vision.AlexNet(vs.Root(), in.ClassCount())
		err = vs.Load(modelPath)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("AlexNet weights loaded.")
	case "squeezenet-v1_1":
		net = vision.SqueezeNetV1_1(vs.Root(), in.ClassCount())
		err = vs.Load(modelPath)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("SqueezeNetV1_1 weights loaded.")
	case "mobilenet-v2":
		net = vision.MobileNetV2(vs.Root(), in.ClassCount())
		err = vs.Load(modelPath)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("MobileNetV2 weights loaded.")
	case "inception-v3":
		net = vision.InceptionV3(vs.Root(), in.ClassCount())
		err = vs.Load(modelPath)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("InceptionV3 weights loaded.")
	case "efficientnet-b4":
		net = vision.EfficientNetB4(vs.Root(), in.ClassCount())
		err = vs.Load(modelPath)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("EfficientNetB4 weights loaded.")
	default:
		log.Fatalf("Invalid model name (%v)\n", modelName)
	}

	// Apply the forward pass of the model to get the logits.
	input := imageTs.MustUnsqueeze(0, true)
	logits := net.ForwardT(input, false)

	// Convert to probability
	pval := logits.MustSoftmax(-1, gotch.Float, true)

	// Print the top 5 categories for this image.
	top5 := in.Top(pval, int64(5))

	for _, i := range top5 {
		fmt.Printf("%-80v %5.2f%%\n", i.Label, i.Pvalue*100)
	}
}
