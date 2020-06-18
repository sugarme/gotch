package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
)

func main() {

	vs := nn.NewVarStore(gotch.CPU)

	fmt.Printf("Is VarStore emptry? %v\n ", vs.IsEmpty())

	path := vs.Root()

	init := nn.NewKaimingUniformInit()

	init.InitTensor([]int64{1, 4}, gotch.CPU).Print()

	path.NewVar("layer1", []int64{1, 10}, nn.NewKaimingUniformInit())

	fmt.Printf("Is VarStore emptry? %v\n ", vs.IsEmpty())

}
