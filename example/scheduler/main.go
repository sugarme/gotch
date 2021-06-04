package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/vision"
	// ts "github.com/sugarme/gotch/tensor"
)

func main() {

	vs := nn.NewVarStore(gotch.CPU)
	model := vision.EfficientNetB4(vs.Root(), 1000)
	vs.Load("../../data/pretrained/efficientnet-b4.pt")

	adamConfig := nn.DefaultAdamConfig()
	o, err := adamConfig.Build(vs, 0.001)
	if err != nil {
		log.Fatal(err)
	}

	ngroup := o.ParamGroupNum()
	lrs := o.GetLRs()

	fmt.Printf("Number of param groups: %v\n", ngroup)
	fmt.Printf("Learning rates: %+v\n", lrs)

	log.Print(model)

}
