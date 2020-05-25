package torch

//#include <stdbool.h>
//#include "torch_api.h"
import "C"

type C_tensor struct {
	_private []uint8
}

func NewTensor() *C_tensor {
	return C.at_new_tensor()
}
