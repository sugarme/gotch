package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	wrapper "github.com/sugarme/gotch/wrapper"
)

func main() {

	ts, err := wrapper.OfSlice([]float64{1.3, 29.7})
	if err != nil {
		panic(err)
	}

	path := "file.pt"
	ts.MustSave(path)

	loadedTs := wrapper.MustLoad(path)

	loadedTs.Print()

	ts1 := wrapper.MustOfSlice([]float64{1.3, 29.7})
	ts2 := wrapper.MustOfSlice([]float64{2.1, 31.2})

	var namedTensors []wrapper.NamedTensor = []wrapper.NamedTensor{
		{Name: "ts1", Tensor: ts1},
		{Name: "ts2", Tensor: ts2},
	}

	pathMulti := "file_multi.pt"

	err = wrapper.SaveMulti(namedTensors, pathMulti)
	if err != nil {
		panic(err)
	}

	var data []wrapper.NamedTensor

	data = wrapper.MustLoadMulti(pathMulti)

	for _, v := range data {
		v.Tensor.Print()
	}

	device := gotch.NewCuda()

	data = wrapper.MustLoadMultiWithDevice(pathMulti, device)
	for _, v := range data {
		v.Tensor.Print()
	}

	tsString := ts.MustToString(80)

	fmt.Printf("Tensor String: \n%v\n", tsString)

	imagePath := "mnist-sample.png"

	imageTs, err := wrapper.LoadHwc(imagePath)
	if err != nil {
		panic(err)
	}

	err = imageTs.Save("mnist-tensor-saved.png")
	if err != nil {
		panic(err)
	}

}
