package gorch

import (
	"log"

	lib "github.com/sugarme/gotch/libtch"
)

type Device struct {
	Name  string
	Value int
}

type Cuda Device

var (
	CPU  Device = Device{Name: "CPU", Value: -1}
	CUDA Cuda   = Cuda{Name: "CUDA", Value: 0}
)

func CudaBuilder(v uint) Device {
	// TODO: fully initiate cuda here
	return Device{Name: "CUDA", Value: int(v)}
}

// Cuda methods:
// =============

// DeviceCount returns the number of GPU that can be used.
func (cu Cuda) DeviceCount() int64 {
	cInt := lib.Atc_cuda_device_count()
	return int64(cInt)
}

/*
 *
 * // CudnnIsAvailable returns true if cuda support is available
 * func (cu Cuda) IsAvailable() bool {
 *   return lib.Atc_cuda_is_available()
 * }
 *
 * // CudnnIsAvailable return true if cudnn support is available
 * func (cu Cuda) CudnnIsAvailable() bool {
 *   return lib.Atc_cudnn_is_available()
 * }
 *
 * // CudnnSetBenchmark sets cudnn benchmark mode
 * //
 * // When set cudnn will try to optimize the generators during the first network
 * // runs and then use the optimized architecture in the following runs. This can
 * // result in significant performance improvements.
 * func (cu Cuda) CudnnSetBenchmark(b bool) {
 *   switch b {
 *   case true:
 *     lib.Atc_set_benchmark_cudnn(1)
 *   case false:
 *     lib.Act_cuda_benchmark_cudd(0)
 *   }
 * } */

// Device methods:
//================

func (d Device) CInt() CInt {
	switch {
	case d.Name == "CPU":
		return -1
	case d.Name == "CUDA":
		// TODO: create a function to retrieve cuda_index
		var deviceIndex int = d.Value
		return CInt(deviceIndex)
	default:
		log.Fatal("Not reachable")
		return 0
	}
}

func (d Device) OfCInt(v CInt) Device {
	switch {
	case v == -1:
		return Device{Name: "CPU", Value: 1}
	case v >= 0:
		return CudaBuilder(uint(v))
	default:
		log.Fatalf("Unexpected device %v", v)
	}
	return Device{}
}

// CudaIfAvailable returns a GPU device if available, else default to CPU
func (d Device) CudaIfAvailable() Device {
	switch {
	// case CUDA.IsAvailable():
	// return CudaBuilder(0)
	default:
		return CPU
	}
}
