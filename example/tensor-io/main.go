package main

import (
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

	_ = wrapper.MustLoadMulti(pathMulti)

}
