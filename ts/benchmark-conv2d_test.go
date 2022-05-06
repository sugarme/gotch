package ts_test

import (
	"testing"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// GOMAXPROCS=8 go test -bench=BenchmarkConv2d -benchtime=100x -run=^a | tee op-conv-bench.txt
// benchstat op-conv-bench.txt
func BenchmarkConv2dCPU(b *testing.B) {
	// var shape []int64 = []int64{64, 3, 224, 224}
	var shape []int64 = []int64{32, 64, 64, 64}
	device := gotch.CPU
	x := ts.MustRandn(shape, gotch.Float, device)
	// kDims := []int64{1, 3, 3, 3}
	kDims := []int64{1, 64, 3, 3}
	kernelTemplate := []int64{
		1, 1, 1,
		1, -8, 1,
		1, 1, 1,
	}
	var kernelData []int64
	for i := 0; i < int(kDims[0]*kDims[1]); i++ {
		kernelData = append(kernelData, kernelTemplate...)
	}
	weight := ts.MustOfSlice(kernelData).MustView(kDims, true).MustTotype(gotch.Float, true).MustTo(device, true)

	stride := []int64{1, 1}
	padding := []int64{0, 0}
	dilation := []int64{1, 1}
	for i := 0; i < b.N; i++ {
		out, err := ts.Conv2d(x, weight, ts.NewTensor(), stride, padding, dilation, 1)
		if err != nil {
			panic(err)
		}
		out.MustDrop()
	}
}

// GOMAXPROCS=8 go test -bench=BenchmarkConv2d -benchtime=100x -run=^a | tee op-conv-bench.txt
// benchstat op-conv-bench.txt
func BenchmarkConv2dCUDA(b *testing.B) {
	// var shape []int64 = []int64{64, 3, 224, 224}
	var shape []int64 = []int64{32, 64, 64, 64}
	device := gotch.CudaIfAvailable()
	x := ts.MustRandn(shape, gotch.Float, device)
	// kDims := []int64{1, 3, 3, 3}
	kDims := []int64{1, 64, 3, 3}
	kernelTemplate := []int64{
		1, 1, 1,
		1, -8, 1,
		1, 1, 1,
	}
	var kernelData []int64
	for i := 0; i < int(kDims[0]*kDims[1]); i++ {
		kernelData = append(kernelData, kernelTemplate...)
	}
	weight := ts.MustOfSlice(kernelData).MustView(kDims, true).MustTotype(gotch.Float, true).MustTo(device, true)

	stride := []int64{1, 1}
	padding := []int64{0, 0}
	dilation := []int64{1, 1}
	for i := 0; i < b.N; i++ {
		out, err := ts.Conv2d(x, weight, ts.NewTensor(), stride, padding, dilation, 1)
		if err != nil {
			panic(err)
		}
		out.MustDrop()
	}
}
