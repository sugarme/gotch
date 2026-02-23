package gotch

import (
	"log"

	lib "github.com/sugarme/gotch/libtch"
)

type Device struct {
	Name  string
	Value int
}

type Cuda Device
type Mps Device

var (
	CPU  Device = Device{Name: "CPU", Value: -1}
	MPS  Mps    = Mps{Name: "MPS", Value: -2}
	CUDA Cuda   = Cuda{Name: "CUDA", Value: 0}
)

func CudaBuilder(v uint) Device {
	// TODO: fully initiate cuda here
	return Device{Name: "CUDA", Value: int(v)}
}

// NewCuda creates a cuda device (default) if available
// If will be panic if cuda is not available.
func NewCuda() Device {
	var d Cuda
	if !d.IsAvailable() {
		log.Fatalf("Cuda is not available.")
	}

	return CudaBuilder(0)
}

// MPS methods:
// ============

// IsAvailable returns true if MPS (Metal Performance Shaders) support is available
func (m Mps) IsAvailable() bool {
	return lib.AtcMpsIsAvailable()
}

// CurrentAllocatedMemory returns the number of bytes currently allocated on the MPS device.
func (m Mps) CurrentAllocatedMemory() uint64 {
	return lib.AtcMpsCurrentAllocatedMemory()
}

// EmptyCache releases all cached MPS memory back to the system.
func (m Mps) EmptyCache() {
	lib.AtcMpsEmptyCache()
}

// Synchronize waits for all MPS operations to complete.
func (m Mps) Synchronize() {
	lib.AtcMpsSynchronize()
}

// Cuda methods:
// =============

// DeviceCount returns the number of GPU that can be used.
func (cu Cuda) DeviceCount() int64 {
	cInt := lib.AtcCudaDeviceCount()
	return int64(cInt)
}

// CudnnIsAvailable returns true if cuda support is available
func (cu Cuda) IsAvailable() bool {
	return lib.AtcCudaIsAvailable()
}

// CudnnIsAvailable return true if cudnn support is available
func (cu Cuda) CudnnIsAvailable() bool {
	return lib.AtcCudnnIsAvailable()
}

// CudnnSetBenchmark sets cudnn benchmark mode
//
// When set cudnn will try to optimize the generators during the first network
// runs and then use the optimized architecture in the following runs. This can
// result in significant performance improvements.
func (cu Cuda) CudnnSetBenchmark(b bool) {
	switch b {
	case true:
		lib.AtcSetBenchmarkCudnn(1)
	case false:
		lib.AtcSetBenchmarkCudnn(0)
	}
}

// Device methods:
//================

func (d Device) CInt() CInt {
	switch {
	case d.Name == "CPU":
		return -1
	case d.Name == "MPS":
		return -2
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
		return Device{Name: "CPU", Value: -1}
	case v == -2:
		return Device{Name: "MPS", Value: -2}
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
	case CUDA.IsAvailable():
		return CudaBuilder(0)
	default:
		return CPU
	}
}

// IsCuda returns whether device is a Cuda device
func (d Device) IsCuda() bool {
	if d.Name == "CPU" {
		return false
	}

	return true
}

// CudaIfAvailable returns a GPU device if available, else CPU.
func CudaIfAvailable() Device {
	switch {
	case CUDA.IsAvailable():
		return CudaBuilder(0)
	default:
		return CPU
	}
}

// MPSIfAvailable returns an MPS device if available, else CPU.
func MPSIfAvailable() Device {
	switch {
	case MPS.IsAvailable():
		return Device{Name: "MPS", Value: -2}
	default:
		return CPU
	}
}

// BestAvailableDevice returns the best available device (CUDA > MPS > CPU).
func BestAvailableDevice() Device {
	switch {
	case CUDA.IsAvailable():
		return CudaBuilder(0)
	case MPS.IsAvailable():
		return Device{Name: "MPS", Value: -2}
	default:
		return CPU
	}
}
