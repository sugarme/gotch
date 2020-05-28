package main

import (
	"fmt"

	tensor "github.com/sugarme/gotch/tensor"
)

func main() {
	_, err := tensor.FnOfSlice()
	if err != nil {
		fmt.Println(err)
	}

}
