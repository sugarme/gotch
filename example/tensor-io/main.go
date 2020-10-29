package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/tensor"
)

func main() {

	ts, err := tensor.OfSlice([]float64{1.3, 29.7})
	if err != nil {
		panic(err)
	}

	path := "file.pt"
	ts.MustSave(path)

	loadedTs := tensor.MustLoad(path)

	loadedTs.Print()

	ts1 := tensor.MustOfSlice([]float64{1.3, 29.7})
	ts2 := tensor.MustOfSlice([]float64{2.1, 31.2})

	var namedTensors []tensor.NamedTensor = []tensor.NamedTensor{
		{Name: "ts1", Tensor: ts1},
		{Name: "ts2", Tensor: ts2},
	}

	pathMulti := "file_multi.pt"

	// err = tensor.SaveMulti(namedTensors, pathMulti)
	// if err != nil {
	// panic(err)
	// }
	err = tensor.SaveMultiNew(namedTensors, pathMulti)
	if err != nil {
		panic(err)
	}

	var data []tensor.NamedTensor

	data = tensor.MustLoadMulti(pathMulti)

	for _, v := range data {
		v.Tensor.Print()
	}

	device := gotch.NewCuda()

	data = tensor.MustLoadMultiWithDevice(pathMulti, device)
	for _, v := range data {
		v.Tensor.Print()
	}

	tsString := ts.MustToString(80)

	fmt.Printf("Tensor String: \n%v\n", tsString)

	imagePath := "mnist-sample.png"

	imageTs, err := tensor.LoadHwc(imagePath)
	if err != nil {
		panic(err)
	}

	err = imageTs.Save("mnist-tensor-saved.png")
	if err != nil {
		panic(err)
	}

}
