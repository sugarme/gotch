//go:build gotch_gpu

package ts_test

import (
	"testing"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

func TestCudaCurrentDevice(t *testing.T) {
	cudaIdx, err := ts.CudaCurrentDevice()
	if err != nil {
		t.Error(err)
	}

	t.Logf("current CUDA index: %v\n", cudaIdx) // should be 0 if having 1 GPU device

	x := ts.MustZeros([]int64{2, 3, 4}, gotch.Float, gotch.CudaIfAvailable())
	currentCudaIndex := x.MustDevice().Value
	t.Logf("x current cuda index: %v\n", currentCudaIndex) // 0

	previousCudaIndex, err := ts.CudaSetDevice(currentCudaIndex)
	if err != nil {
		t.Error(err)
	}
	t.Logf("Cuda index BEFORE set: %v\n", previousCudaIndex) // 0

	cudaIdxAfter, err := ts.CudaCurrentDevice()
	if err != nil {
		t.Error(err)
	}
	t.Logf("Cuda index AFTER set: %v\n", cudaIdxAfter) // 0
}
