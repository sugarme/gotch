package libtch

// NOTE. This file is a patch of missing auto-generated APIs in `c-generated.go`

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import "unsafe"

// NOTE: 9 patches for pattern of **return tensor pointer**: `tensor *atg_FUNCTION_NAME()`:
// tensor *atg_align_tensors(tensor *tensors_data, int tensors_len);
// tensor *atg_broadcast_tensors(tensor *tensors_data, int tensors_len);
// tensor *atg_chunk(tensor self, int64_t chunks, int64_t dim);
// tensor *atg_meshgrid(tensor *tensors_data, int tensors_len);
// tensor *atg_nonzero_numpy(tensor self);
// tensor *atg_split(tensor self, int64_t split_size, int64_t dim);
// tensor *atg_split_with_sizes(tensor self, int64_t *split_sizes_data, int split_sizes_len, int64_t dim);
// tensor *atg_unbind(tensor self, int64_t dim);
// tensor *atg_where(tensor condition);

// tensor *atg_align_tensors(tensor *tensors_data, int tensors_len);
func AtgAlignTensors(tensorsData []Ctensor, tensorsLen int) *Ctensor {

	ctensorsDataPtr := (*Ctensor)(unsafe.Pointer(&tensorsData[0]))
	ctensorsLen := *(*C.int)(unsafe.Pointer(&tensorsLen))
	return C.atg_align_tensors(ctensorsDataPtr, ctensorsLen)
}

// tensor *atg_broadcast_tensors(tensor *tensors_data, int tensors_len);
func AtgBroadcastTensors(tensorsData []Ctensor, tensorsLen int) *Ctensor {

	ctensorsDataPtr := (*Ctensor)(unsafe.Pointer(&tensorsData[0]))
	ctensorsLen := *(*C.int)(unsafe.Pointer(&tensorsLen))
	return C.atg_broadcast_tensors(ctensorsDataPtr, ctensorsLen)
}

// tensor *atg_chunk(tensor self, int64_t chunks, int64_t dim);
func AtgChunk(self Ctensor, chunks int64, dim int64) *Ctensor {

	cchunks := *(*C.int64_t)(unsafe.Pointer(&chunks))
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	return C.atg_chunk(self, cchunks, cdim)
}

// tensor *atg_meshgrid(tensor *tensors_data, int tensors_len);
func AtgMeshgrid(tensorsData []Ctensor, tensorsLen int) *Ctensor {

	ctensorsDataPtr := (*Ctensor)(unsafe.Pointer(&tensorsData[0]))
	ctensorsLen := *(*C.int)(unsafe.Pointer(&tensorsLen))
	return C.atg_meshgrid(ctensorsDataPtr, ctensorsLen)
}

// tensor *atg_nonzero_numpy(tensor self);
func AtgNonzeroNumpy(self Ctensor) *Ctensor {
	return C.atg_nonzero_numpy(self)
}

// tensor *atg_split(tensor self, int64_t split_size, int64_t dim);
func AtgSplit(self Ctensor, splitSize int64, dim int64) *Ctensor {

	csplitSize := *(*C.int64_t)(unsafe.Pointer(&splitSize))
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))

	return C.atg_split(self, csplitSize, cdim)
}

// tensor *atg_split_with_sizes(tensor self, int64_t *split_sizes_data, int split_sizes_len, int64_t dim);
func AtgSplitWithSizes(self Ctensor, splitSizesData []int64, splitSizesLen int, dim int64) *Ctensor {

	csplitSizesDataPtr := (*C.int64_t)(unsafe.Pointer(&splitSizesData[0]))
	csplitSizesLen := *(*C.int)(unsafe.Pointer(&splitSizesLen))
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))

	return C.atg_split_with_sizes(self, csplitSizesDataPtr, csplitSizesLen, cdim)
}

// tensor *atg_unbind(tensor self, int64_t dim);
func AtgUnbind(self Ctensor, dim int64) *Ctensor {

	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	return C.atg_unbind(self, cdim)
}

// tensor *atg_where(tensor condition);
func AtgWhere(condition Ctensor) *Ctensor {
	return C.atg_where(condition)
}
