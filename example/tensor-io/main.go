package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

func main() {

	x, err := ts.OfSlice([]float64{1.3, 29.7})
	if err != nil {
		panic(err)
	}

	path := "file.pt"
	x.MustSave(path)

	loadedTs := ts.MustLoad(path)

	loadedTs.Print()

	ts1 := ts.MustOfSlice([]float64{1.3, 29.7})
	ts2 := ts.MustOfSlice([]float64{2.1, 31.2})

	var namedTensors []ts.NamedTensor = []ts.NamedTensor{
		{Name: "ts1", Tensor: ts1},
		{Name: "ts2", Tensor: ts2},
	}

	pathMulti := "file_multi.pt"

	// err = ts.SaveMulti(namedTensors, pathMulti)
	// if err != nil {
	// panic(err)
	// }
	err = ts.SaveMultiNew(namedTensors, pathMulti)
	if err != nil {
		panic(err)
	}

	var data []ts.NamedTensor

	data = ts.MustLoadMulti(pathMulti)

	for _, v := range data {
		v.Tensor.Print()
	}

	device := gotch.NewCuda()

	data = ts.MustLoadMultiWithDevice(pathMulti, device)
	for _, v := range data {
		v.Tensor.Print()
	}

	tsString := x.MustToString(80)

	fmt.Printf("Tensor String: \n%v\n", tsString)

	imagePath := "mnist-sample.png"

	imageTs, err := ts.LoadHwc(imagePath)
	if err != nil {
		panic(err)
	}

	err = imageTs.Save("mnist-tensor-saved.png")
	if err != nil {
		panic(err)
	}

}
