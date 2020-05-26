package main

import (
	"fmt"
	"reflect"
	"unsafe"

	t "github.com/sugarme/gotch/torch"
)

type Tensor struct {
	c_tensor *t.C_tensor
}

func FnOfSlice(data []float64) (retVal Tensor, err error) {
	dataLen := len(data)
	dat := unsafe.Pointer(data)

	c_tensor := t.AtTensorOfData(dat, int64(dataLen), 1, 7, 7)

	retVal = Tensor{c_tensor}

	return retVal, nil
}

func main() {

	t := t.NewTensor()

	fmt.Printf("Type of t: %v\n", reflect.TypeOf(t))
}
