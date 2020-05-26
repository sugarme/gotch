// NOTE: functions in this file would be automatically generated
// and named as `c-generated.go`
package libtch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

func Atc_cuda_device_count() int {
	return C.atc_cuda_device_count()
}
