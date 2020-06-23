package main

import (
	"fmt"
	"log"
	"math"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

func main() {
	// noSeq()
	withSeq()
	// noSeq2Layers()

	// seqNoVarStore()
}

func noSeq() {
	ds := vision.LoadMNISTDir("../../data/mnist")

	wsInit := nn.NewKaimingUniformInit()
	ws := wsInit.InitTensor([]int64{10, 784}, gotch.CPU).MustT(true)

	bound := 1.0 / math.Sqrt(float64(784))
	bsInit := nn.NewUniformInit(-bound, bound)
	bs := bsInit.InitTensor([]int64{10}, gotch.CPU)

	for i := 0; i < 2000; i++ {
		mul := ds.TrainImages.MustMatMul(ws, false)
		logits := mul.MustAdd(bs, true)
		loss := logits.AccuracyForLogits(ds.TrainLabels)

		fmt.Printf("Epoch %v\t Loss: %.3f\n", i, loss.Values()[0])
		loss.MustDrop()
	}

}

func withSeq() {
	seq := nn.Seq()
	vs := nn.NewVarStore(gotch.CPU)
	// seq.Add(nn.NewLinear(vs.Root(), 784, 10, *nn.DefaultLinearConfig()))
	seq.Add(nn.NewLinear(vs.Root(), 784, 128, *nn.DefaultLinearConfig()))
	seq.Add(nn.NewLinear(vs.Root(), 128, 10, *nn.DefaultLinearConfig()))

	opt, err := nn.DefaultAdamConfig().Build(vs, 1e-2)
	if err != nil {
		log.Fatal(err)
	}

	ds := vision.LoadMNISTDir("../../data/mnist")

	for i := 0; i < 2000; i++ {
		logits := seq.Forward(ds.TrainImages)
		loss := logits.CrossEntropyForLogits(ds.TrainLabels)
		opt.BackwardStep(loss)

		testLogits := seq.Forward(ds.TestImages)
		testAccuracy := testLogits.AccuracyForLogits(ds.TestLabels)

		fmt.Printf("Epoch: %v \t Loss: %.3f \t Test accuracy: %.2f%%\n", i, loss.Values()[0], testAccuracy.Values()[0]*100)

		loss.MustDrop()
		testAccuracy.MustDrop()
	}

}

func noSeq2Layers() {
	ds := vision.LoadMNISTDir("../../data/mnist")

	wsInit := nn.NewKaimingUniformInit()
	ws1 := wsInit.InitTensor([]int64{1024, 784}, gotch.CPU).MustT(true)
	ws2 := wsInit.InitTensor([]int64{10, 1024}, gotch.CPU).MustT(true)

	bound1 := 1.0 / math.Sqrt(float64(784))
	bsInit1 := nn.NewUniformInit(-bound1, bound1)
	bs1 := bsInit1.InitTensor([]int64{1024}, gotch.CPU)

	bound2 := 1.0 / math.Sqrt(float64(1024))
	bsInit2 := nn.NewUniformInit(-bound2, bound2)
	bs2 := bsInit2.InitTensor([]int64{10}, gotch.CPU)

	for i := 0; i < 2000; i++ {
		mul1 := ds.TrainImages.MustMatMul(ws1, false)
		out1 := mul1.MustAdd(bs1, true)

		mul2 := out1.MustMatMul(ws2, true)
		logits := mul2.MustAdd(bs2, true)

		loss := logits.AccuracyForLogits(ds.TrainLabels)

		fmt.Printf("Epoch %v\t Loss: %.3f\n", i, loss.Values()[0])
		loss.MustDrop()
	}
}

func seqNoVarStore() {

	ds := vision.LoadMNISTDir("../../data/mnist")

	wsInit := nn.NewKaimingUniformInit()
	ws1 := wsInit.InitTensor([]int64{1024, 784}, gotch.CPU).MustT(true)
	ws2 := wsInit.InitTensor([]int64{10, 1024}, gotch.CPU).MustT(true)

	bound1 := 1.0 / math.Sqrt(float64(784))
	bsInit1 := nn.NewUniformInit(-bound1, bound1)
	bs1 := bsInit1.InitTensor([]int64{1024}, gotch.CPU)

	bound2 := 1.0 / math.Sqrt(float64(1024))
	bsInit2 := nn.NewUniformInit(-bound2, bound2)
	bs2 := bsInit2.InitTensor([]int64{10}, gotch.CPU)

	l1 := Linear{&ws1, &bs1}
	l2 := Linear{&ws2, &bs2}

	seq := Seq()
	seq.Add(l1)
	seq.Add(l2)
	// seq.Add1(l1)
	// seq.Add2(l2)

	for i := 0; i < 2000; i++ {
		logits := seq.Forward(ds.TrainImages)

		logits.MustDrop()
	}

}

type Linear struct {
	Ws *ts.Tensor
	Bs *ts.Tensor
}

func (l Linear) Forward(xs ts.Tensor) ts.Tensor {
	mul := xs.MustMatMul(*l.Ws, false)
	return mul.MustAdd(*l.Bs, true)
}

type Sequential struct {
	layers []ts.Module
	l1     ts.Module
	l2     ts.Module
}

func Seq() Sequential {
	return Sequential{layers: make([]ts.Module, 0)}
}

// Len returns number of sub-layers embedded in this layer
func (s *Sequential) Len() (retVal int64) {
	return int64(len(s.layers))
}

// IsEmpty returns true if this layer does not have any sub-layers.
func (s *Sequential) IsEmpty() (retVal bool) {
	return len(s.layers) == 0
}

// Add appends a layer after all the current layers.
func (s *Sequential) Add(l ts.Module) {

	s.layers = append(s.layers, l)
}

func (s *Sequential) Add1(l ts.Module) {
	s.l1 = l
}

func (s *Sequential) Add2(l ts.Module) {
	s.l2 = l
}

func (s Sequential) Forward(xs ts.Tensor) (retVal ts.Tensor) {
	if s.IsEmpty() {
		return xs.MustShallowClone()
	}

	// forward sequentially
	outs := make([]ts.Tensor, len(s.layers))
	for i := 0; i < len(s.layers); i++ {
		if i == 0 {
			outs[0] = s.layers[i].Forward(xs)
			defer outs[0].MustDrop()
		} else if i == len(s.layers)-1 {
			return s.layers[i].Forward(outs[i-1])
		} else {
			outs[i+1] = s.layers[i].Forward(outs[i-1])
			defer outs[i+1].MustDrop()
		}
	}

	return

	// out1 := s.l1.Forward(xs)
	// defer out1.MustDrop()
	//
	// return s.l2.Forward(out1)

}
