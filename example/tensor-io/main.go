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
}
