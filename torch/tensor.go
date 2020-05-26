package torch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import (
	"fmt"
	"reflect"
)

type C_tensor struct {
	_private []uint8
}

func NewTensor() {
	t := C.at_new_tensor()
	fmt.Printf("Tensor Type: %v\n", reflect.TypeOf(t).Kind())
	fmt.Printf("Tensor Value: %v\n", t)
}
