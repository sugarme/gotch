package main

import (
	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
)

func main() {

	vs := nn.NewVarStore(gotch.CPU)

	path := vs.Root()

	l := nn.NewLinear(path, 4, 3, nn.DefaultLinearConfig())

	l.Bs.Print()
}
