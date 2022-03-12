package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

func main() {

	device := gotch.CPU
	vs := nn.NewVarStore(device)
	// model := vision.EfficientNetB4(vs.Root(), 1000)
	// vs.Load("../../data/pretrained/efficientnet-b4.pt")

	model := newNet(vs.Root())
	adamConfig := nn.DefaultAdamConfig()
	o, err := adamConfig.Build(vs, 0.001)
	if err != nil {
		log.Fatal(err)
	}

	ngroup := o.ParamGroupNum()
	lrs := o.GetLRs()

	fmt.Printf("Number of param groups: %v\n", ngroup)
	fmt.Printf("Learning rates: %+v\n", lrs)

	newLRs := []float64{0.005}
	o.SetLRs(newLRs)
	fmt.Printf("New LRs: %+v\n", o.GetLRs())

	zerosTs := ts.MustZeros([]int64{2, 2}, gotch.Float, device)
	onesTs := ts.MustOnes([]int64{3, 5}, gotch.Float, device)

	o.AddParamGroup([]ts.Tensor{*zerosTs, *onesTs})
	fmt.Printf("New num of param groups: %v\n", o.ParamGroupNum())

	fmt.Printf("New LRs: %+v\n", o.GetLRs())

	// Set new lrs
	newLRs = []float64{0.0003, 0.0006}
	o.SetLRs(newLRs)
	fmt.Printf("New LRs: %+v\n", o.GetLRs())

	log.Print(model)
}

type Net struct {
	conv1 *nn.Conv2D
	conv2 *nn.Conv2D
	fc    *nn.Linear
}

func newNet(vs *nn.Path) *Net {
	conv1 := nn.NewConv2D(vs, 1, 16, 2, nn.DefaultConv2DConfig())
	conv2 := nn.NewConv2D(vs, 16, 10, 2, nn.DefaultConv2DConfig())
	fc := nn.NewLinear(vs, 10, 10, nn.DefaultLinearConfig())

	return &Net{
		conv1,
		conv2,
		fc,
	}
}

func (n Net) ForwardT(xs *ts.Tensor, train bool) *ts.Tensor {
	xs = xs.MustView([]int64{-1, 1, 8, 8}, false)

	outC1 := xs.Apply(n.conv1)
	outMP1 := outC1.MaxPool2DDefault(2, true)
	defer outMP1.MustDrop()

	outC2 := outMP1.Apply(n.conv2)
	outMP2 := outC2.MaxPool2DDefault(2, true)
	outView2 := outMP2.MustView([]int64{-1, 10}, true)
	defer outView2.MustDrop()

	outFC := outView2.Apply(n.fc)
	return outFC.MustRelu(true)
}
