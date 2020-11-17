package main

import (
	"fmt"
	"log"

	ts "github.com/sugarme/gotch/tensor"
)

func main() {
	filepath := "../../data/convert-model/bert/model.npz"

	namedTensors, err := ts.ReadNpz(filepath)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Num of named tensor: %v\n", len(namedTensors))
	/*
	 *   for _, nt := range namedTensors {
	 *     // fmt.Printf("%q\n", nt.Name)
	 *     if nt.Name == "bert.encoder.layer.1.attention.output.LayerNorm.weight" {
	 *       fmt.Printf("%0.3f", nt.Tensor)
	 *     }
	 *   }
	 *  */

	// fmt.Printf("%v", namedTensors[70].Tensor)

	outputFile := "../../data/convert-model/bert/model.gt"
	err = ts.SaveMulti(namedTensors, outputFile)
	if err != nil {
		log.Fatal(err)
	}

}
