package tensor

// #include "stdlib.h"
import "C"

import (
	"fmt"
	"log"
	"unsafe"

	// "github.com/sugarme/gotch"
	lib "github.com/sugarme/gotch/libtch"
)

// NOTE. This is a temporarily patched to make it run.
// TODO. make change at generator for []Tensor input

func (ts Tensor) Lstm(hxData []Tensor, paramsData []Tensor, hasBiases bool, numLayers int64, dropout float64, train bool, bidirectional bool, batchFirst bool) (output, h, c Tensor, err error) {

	// NOTE: `atg_lstm` will create 3 consecutive Ctensors in memory of C land. The first
	// Ctensor will have address given by `ctensorPtr1` here.
	// The next pointers can be calculated based on `ctensorPtr1`
	ctensorPtr1 := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	ctensorPtr2 := (*lib.Ctensor)(unsafe.Pointer(uintptr(unsafe.Pointer(ctensorPtr1)) + unsafe.Sizeof(ctensorPtr1)))
	ctensorPtr3 := (*lib.Ctensor)(unsafe.Pointer(uintptr(unsafe.Pointer(ctensorPtr2)) + unsafe.Sizeof(ctensorPtr1)))

	var chxData []lib.Ctensor
	for _, t := range hxData {
		chxData = append(chxData, t.ctensor)
	}

	var cparamsData []lib.Ctensor
	for _, t := range paramsData {
		cparamsData = append(cparamsData, t.ctensor)
	}

	var chasBiases int32 = 0
	if hasBiases {
		chasBiases = 1
	}
	var ctrain int32 = 0
	if train {
		ctrain = 1
	}
	var cbidirectional int32 = 0
	if bidirectional {
		cbidirectional = 1
	}
	var cbatchFirst int32 = 0
	if batchFirst {
		cbatchFirst = 1
	}

	lib.AtgLstm(ctensorPtr1, ts.ctensor, chxData, len(hxData), cparamsData, len(paramsData), chasBiases, numLayers, dropout, ctrain, cbidirectional, cbatchFirst)
	err = TorchErr()
	if err != nil {
		return output, h, c, err
	}

	return Tensor{ctensor: *ctensorPtr1}, Tensor{ctensor: *ctensorPtr2}, Tensor{ctensor: *ctensorPtr3}, nil

}

func (ts Tensor) MustLstm(hxData []Tensor, paramsData []Tensor, hasBiases bool, numLayers int64, dropout float64, train bool, bidirectional bool, batchFirst bool) (output, h, c Tensor) {
	output, h, c, err := ts.Lstm(hxData, paramsData, hasBiases, numLayers, dropout, train, bidirectional, batchFirst)

	if err != nil {
		log.Fatal(err)
	}

	return output, h, c
}

func (ts Tensor) Gru(hx Tensor, paramsData []Tensor, hasBiases bool, numLayers int64, dropout float64, train bool, bidirectional bool, batchFirst bool) (output, h Tensor, err error) {

	// NOTE: `atg_gru` will create 2 consecutive Ctensors in memory of C land.
	// The first Ctensor will have address given by `ctensorPtr1` here.
	// The next pointer can be calculated based on `ctensorPtr1`
	ctensorPtr1 := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	ctensorPtr2 := (*lib.Ctensor)(unsafe.Pointer(uintptr(unsafe.Pointer(ctensorPtr1)) + unsafe.Sizeof(ctensorPtr1)))

	var cparamsData []lib.Ctensor
	for _, t := range paramsData {
		cparamsData = append(cparamsData, t.ctensor)
	}

	var chasBiases int32 = 0
	if hasBiases {
		chasBiases = 1
	}
	var ctrain int32 = 0
	if train {
		ctrain = 1
	}
	var cbidirectional int32 = 0
	if bidirectional {
		cbidirectional = 1
	}
	var cbatchFirst int32 = 0
	if batchFirst {
		cbatchFirst = 1
	}

	lib.AtgGru(ctensorPtr1, ts.ctensor, hx.ctensor, cparamsData, len(paramsData), chasBiases, numLayers, dropout, ctrain, cbidirectional, cbatchFirst)
	err = TorchErr()
	if err != nil {
		return output, h, err
	}

	return Tensor{ctensor: *ctensorPtr1}, Tensor{ctensor: *ctensorPtr2}, nil

}

func (ts Tensor) MustGru(hx Tensor, paramsData []Tensor, hasBiases bool, numLayers int64, dropout float64, train bool, bidirectional bool, batchFirst bool) (output, h Tensor) {
	output, h, err := ts.Gru(hx, paramsData, hasBiases, numLayers, dropout, train, bidirectional, batchFirst)
	if err != nil {
		log.Fatal(err)
	}

	return output, h
}

func (ts Tensor) TopK(k int64, dim int64, largest bool, sorted bool) (ts1 Tensor, ts2 Tensor, err error) {

	// NOTE: `lib.AtgTopk` will return 2 tensors in C memory. First tensor pointer
	// is given by ctensorPtr1
	ctensorPtr1 := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	ctensorPtr2 := (*lib.Ctensor)(unsafe.Pointer(uintptr(unsafe.Pointer(ctensorPtr1)) + unsafe.Sizeof(ctensorPtr1)))
	var clargest int32 = 0
	if largest {
		clargest = 1
	}
	var csorted int32 = 0
	if sorted {
		csorted = 1
	}

	lib.AtgTopk(ctensorPtr1, ts.ctensor, k, dim, clargest, csorted)
	err = TorchErr()
	if err != nil {
		return ts1, ts2, err
	}

	return Tensor{ctensor: *ctensorPtr1}, Tensor{ctensor: *ctensorPtr2}, nil
}

func (ts Tensor) MustTopK(k int64, dim int64, largest bool, sorted bool) (ts1 Tensor, ts2 Tensor) {

	ts1, ts2, err := ts.TopK(k, dim, largest, sorted)
	if err != nil {
		log.Fatal(err)
	}

	return ts1, ts2
}

// NOTE. `NLLLoss` is a version of `NllLoss` in tensor-generated
// with default weight, reduction and ignoreIndex
func (ts Tensor) NLLLoss(target Tensor, del bool) (retVal Tensor, err error) {
	ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))
	if del {
		defer ts.MustDrop()
	}

	reduction := int64(1) // Mean of loss
	ignoreIndex := int64(-100)
	// defer C.free(unsafe.Pointer(ptr))

	lib.AtgNllLoss(ptr, ts.ctensor, target.ctensor, nil, reduction, ignoreIndex)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	retVal = Tensor{ctensor: *ptr}

	return retVal, nil
}

func (ts Tensor) MustNLLLoss(target Tensor, del bool) (retVal Tensor) {
	retVal, err := ts.NLLLoss(target, del)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

// NOTE: the following 9 APIs are missing from `tensor-generated.go` with
// pattern of **return tensor pointer**: `tensor *atg_FUNCTION_NAME()`.
// The returning tensor pointer actually is the FIRST element of a vector
// of C tensor pointers. Next pointer will be calculated from the first.
// In C land, verifying a valid pointer is to check whether it points to **NULL**.
//
// tensor *atg_align_tensors(tensor *tensors_data, int tensors_len);
// tensor *atg_broadcast_tensors(tensor *tensors_data, int tensors_len);
// tensor *atg_chunk(tensor self, int64_t chunks, int64_t dim);
// tensor *atg_meshgrid(tensor *tensors_data, int tensors_len);
// tensor *atg_nonzero_numpy(tensor self);
// tensor *atg_split(tensor self, int64_t split_size, int64_t dim);
// tensor *atg_split_with_sizes(tensor self, int64_t *split_sizes_data, int split_sizes_len, int64_t dim);
// tensor *atg_unbind(tensor self, int64_t dim);
// tensor *atg_where(tensor condition);

// Split splits tensor into chunks
//
// Parameters:
//  - splitSize – size of a single chunk or list of sizes for each chunk
//  - dim – dimension along which to split the tensor.
// Ref. https://pytorch.org/docs/stable/generated/torch.split.html
func (ts Tensor) Split(splitSize, dim int64) (retVal []Tensor, err error) {

	ctensorsPtr := lib.AtgSplit(ts.ctensor, splitSize, dim)
	if err = TorchErr(); err != nil {
		return retVal, err
	}

	// NOTE: ctensorsPtr is a c-pointer to a vector of tensors. The first
	// C tensor is the `ctensorsPtr` value. The next pointer will be
	// calculated from there. The vector of tensors will end if the calculated
	// pointer value is `null`.
	currentPtr := ctensorsPtr
	retVal = append(retVal, Tensor{ctensor: *currentPtr})
	for {
		// calculate the next pointer value
		nextPtr := (*lib.Ctensor)(unsafe.Pointer(uintptr(unsafe.Pointer(currentPtr)) + unsafe.Sizeof(currentPtr)))
		if *nextPtr == nil {
			break
		}

		retVal = append(retVal, Tensor{ctensor: *nextPtr})
		currentPtr = nextPtr
	}

	return retVal, nil
}

func (ts Tensor) MustSplit(splitSize, dim int64, del bool) (retVal []Tensor) {
	if del {
		defer ts.MustDrop()
	}

	retVal, err := ts.Split(splitSize, dim)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}
