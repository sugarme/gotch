// NOTE: this is a sample for OCaml generated code for `c-generated.go`
package libtch

//#include "stdbool.h"
//#include "torch_api.h"
import "C"

import (
	"unsafe"
)

// void atg_eq1(tensor *, tensor self, tensor other);
func AtgEq1(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_eq1(ptr, self, other)
}

// void atg_matmul(tensor *, tensor self, tensor other);
func AtgMatmul(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_matmul(ptr, self, other)
}

// void atg_to(tensor *, tensor self, int device);
func AtgTo(ptr *Ctensor, self Ctensor, device int) {
	cdevice := *(*C.int)(unsafe.Pointer(&device))
	C.atg_to(ptr, self, cdevice)
}

// int at_device(tensor);
func AtDevice(ts Ctensor) int {
	cint := C.at_device(ts)
	return *(*int)(unsafe.Pointer(&cint))
}

// void atg_grad(tensor *, tensor self);
func AtgGrad(ptr *Ctensor, self Ctensor) {
	C.atg_grad(ptr, self)
}

// void atg_detach_(tensor *, tensor self);
func AtgDetach_(ptr *Ctensor, self Ctensor) {
	C.atg_detach_(ptr, self)
}

// void atg_zero_(tensor *, tensor self);
func AtgZero_(ptr *Ctensor, self Ctensor) {
	C.atg_zero_(ptr, self)
}

// void atg_set_requires_grad(tensor *, tensor self, int r);
func AtgSetRequiresGrad(ptr *Ctensor, self Ctensor, r int) {
	cr := *(*C.int)(unsafe.Pointer(&r))
	C.atg_set_requires_grad(ptr, self, cr)
}

// void atg_mul(tensor *, tensor self, tensor other);
func AtgMul(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_mul(ptr, self, other)
}

// void atg_mul_(tensor *, tensor self, tensor other);
func AtgMul_(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_mul_(ptr, self, other)
}

// void atg_mul1(tensor *, tensor self, scalar other);
func AtgMul1(ptr *Ctensor, self Ctensor, other Cscalar) {
	C.atg_mul1(ptr, self, other)
}

// void atg_add(tensor *, tensor self, tensor other);
func AtgAdd(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_add(ptr, self, other)
}

// void atg_add_(tensor *, tensor self, tensor other);
func AtgAdd_(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_add_(ptr, self, other)
}

// id atg_add1(tensor *, tensor self, scalar other);
func AtgAdd1(ptr *Ctensor, self Ctensor, other Cscalar) {
	C.atg_add1(ptr, self, other)
}

// void atg_totype(tensor *, tensor self, int scalar_type);
func AtgTotype(ptr *Ctensor, self Ctensor, scalar_type int32) {
	cscalar_type := *(*C.int)(unsafe.Pointer(&scalar_type))
	C.atg_totype(ptr, self, cscalar_type)
}

// void atg_unsqueeze(tensor *, tensor self, int64_t dim);
func AtgUnsqueeze(ptr *Ctensor, self Ctensor, dim int64) {
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	C.atg_unsqueeze(ptr, self, cdim)
}

// void atg_select(tensor *, tensor self, int64_t dim, int64_t index);
func AtgSelect(ptr *Ctensor, self Ctensor, dim int64, index int64) {
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	cindex := *(*C.int64_t)(unsafe.Pointer(&index))
	C.atg_select(ptr, self, cdim, cindex)
}

// void atg_narrow(tensor *, tensor self, int64_t dim, int64_t start, int64_t length);
func AtgNarrow(ptr *Ctensor, self Ctensor, dim int64, start int64, length int64) {
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	cstart := *(*C.int64_t)(unsafe.Pointer(&start))
	clength := *(*C.int64_t)(unsafe.Pointer(&length))
	C.atg_narrow(ptr, self, cdim, cstart, clength)
}

// void atg_index_select(tensor *, tensor self, int64_t dim, tensor index);
func AtgIndexSelect(ptr *Ctensor, self Ctensor, dim int64, index Ctensor) {
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	C.atg_index_select(ptr, self, cdim, index)
}

// void atg_zeros(tensor *, int64_t *size_data, int size_len, int options_kind, int options_device);
func AtgZeros(ptr *Ctensor, sizeData []int64, sizeLen int, optionsKind, optionsDevice int32) {
	// just get pointer of the first element of the shape(sizeData)
	csizeDataPtr := (*C.int64_t)(unsafe.Pointer(&sizeData[0]))
	csizeLen := *(*C.int)(unsafe.Pointer(&sizeLen))
	coptionsKind := *(*C.int)(unsafe.Pointer(&optionsKind))
	coptionsDevice := *(*C.int)(unsafe.Pointer(&optionsDevice))

	C.atg_zeros(ptr, csizeDataPtr, csizeLen, coptionsKind, coptionsDevice)
}

// void atg_ones(tensor *, int64_t *size_data, int size_len, int options_kind, int options_device);
func AtgOnes(ptr *Ctensor, sizeData []int64, sizeLen int, optionsKind, optionsDevice int32) {
	// just get pointer of the first element of the shape(sizeData)
	csizeDataPtr := (*C.int64_t)(unsafe.Pointer(&sizeData[0]))
	csizeLen := *(*C.int)(unsafe.Pointer(&sizeLen))
	coptionsKind := *(*C.int)(unsafe.Pointer(&optionsKind))
	coptionsDevice := *(*C.int)(unsafe.Pointer(&optionsDevice))

	C.atg_ones(ptr, csizeDataPtr, csizeLen, coptionsKind, coptionsDevice)
}

// void atg_uniform_(tensor *, tensor self, double from, double to);
func AtgUniform_(ptr *Ctensor, self Ctensor, from float64, to float64) {
	cfrom := *(*C.double)(unsafe.Pointer(&from))
	cto := *(*C.double)(unsafe.Pointer(&to))

	C.atg_uniform_(ptr, self, cfrom, cto)
}

// void atg_zeros_like(tensor *, tensor self);
func AtgZerosLike(ptr *Ctensor, self Ctensor) {
	C.atg_zeros_like(ptr, self)
}

// void atg_fill_(tensor *, tensor self, scalar value);
func AtgFill_(ptr *Ctensor, self Ctensor, value Cscalar) {
	C.atg_fill_(ptr, self, value)
}

// void atg_randn_like(tensor *, tensor self);
func AtgRandnLike(ptr *Ctensor, self Ctensor) {
	C.atg_rand_like(ptr, self)
}

// void atg_log_softmax(tensor *, tensor self, int64_t dim, int dtype);
func AtgLogSoftmax(ptr *Ctensor, self Ctensor, dim int64, dtype int32) {
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	cdtype := *(*C.int)(unsafe.Pointer(&dtype))

	C.atg_log_softmax(ptr, self, cdim, cdtype)
}

// void atg_nll_loss(tensor *, tensor self, tensor target, tensor weight, int64_t reduction, int64_t ignore_index);
func AtgNllLoss(ptr *Ctensor, self Ctensor, target Ctensor, weight Ctensor, reduction int64, ignoreIndex int64) {
	creduction := *(*C.int64_t)(unsafe.Pointer(&reduction))
	cignoreIndex := *(*C.int64_t)(unsafe.Pointer(&ignoreIndex))

	C.atg_nll_loss(ptr, self, target, weight, creduction, cignoreIndex)
}

// void atg_argmax(tensor *, tensor self, int64_t dim, int keepdim);
func AtgArgmax(ptr *Ctensor, self Ctensor, dim int64, keepDim int) {
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	ckeepDim := *(*C.int)(unsafe.Pointer(&keepDim))

	C.atg_argmax(ptr, self, cdim, ckeepDim)
}

// void atg_mean(tensor *, tensor self, int dtype);
func AtgMean(ptr *Ctensor, self Ctensor, dtype int32) {
	cdtype := *(*C.int)(unsafe.Pointer(&dtype))

	C.atg_mean(ptr, self, cdtype)
}

// void atg_permute(tensor *, tensor self, int64_t *dims_data, int dims_len);
func AtgPermute(ptr *Ctensor, self Ctensor, dims []int64, dimLen int) {
	// just get pointer of the first element of the shape
	cdimsPtr := (*C.int64_t)(unsafe.Pointer(&dims[0]))
	cdimLen := *(*C.int)(unsafe.Pointer(&dimLen))

	C.atg_permute(ptr, self, cdimsPtr, cdimLen)
}

// void atg_squeeze1(tensor *, tensor self, int64_t dim);
func AtgSqueeze1(ptr *Ctensor, self Ctensor, dim int64) {
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))

	C.atg_squeeze1(ptr, self, cdim)
}

// void atg_squeeze_(tensor *, tensor self);
func AtgSqueeze_(ptr *Ctensor, self Ctensor) {
	C.atg_squeeze_(ptr, self)
}

// void atg_stack(tensor *, tensor *tensors_data, int tensors_len, int64_t dim);
func AtgStack(ptr *Ctensor, tensorsData []Ctensor, tensorsLen int, dim int64) {
	tensorsDataPtr := (*Ctensor)(unsafe.Pointer(&tensorsData[0]))
	ctensorsLen := *(*C.int)(unsafe.Pointer(&tensorsLen))
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))

	C.atg_stack(ptr, tensorsDataPtr, ctensorsLen, cdim)
}

// void atg_mm(tensor *, tensor self, tensor mat2);
func AtgMm(ptr *Ctensor, self Ctensor, mat2 Ctensor) {
	C.atg_mm(ptr, self, mat2)
}

// void atg_view(tensor *, tensor self, int64_t *size_data, int size_len);
func AtgView(ptr *Ctensor, self Ctensor, sizeData []int64, sizeLen int) {
	sizeDataPtr := (*C.int64_t)(unsafe.Pointer(&sizeData[0]))
	csizeLen := *(*C.int)(unsafe.Pointer(&sizeLen))

	C.atg_view(ptr, self, sizeDataPtr, csizeLen)
}

// void atg_div1(tensor *, tensor self, scalar other);
func AtgDiv1(ptr *Ctensor, self Ctensor, other Cscalar) {
	C.atg_div1(ptr, self, other)
}

// void atg_div(tensor *, tensor self, tensor other);
func AtgDiv(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_div(ptr, self, other)
}

// void atg_randperm(tensor *, int64_t n, int options_kind, int options_device);
func AtgRandperm(ptr *Ctensor, n int64, optionKind int32, optionDevice int32) {
	cn := *(*C.int64_t)(unsafe.Pointer(&n))
	coptionKind := *(*C.int)(unsafe.Pointer(&optionKind))
	coptionDevice := *(*C.int)(unsafe.Pointer(&optionDevice))

	C.atg_randperm(ptr, cn, coptionKind, coptionDevice)
}

// void atg_clamp_(tensor *, tensor self, scalar min, scalar max);
func AtgClamp_(ptr *Ctensor, self Ctensor, min Cscalar, max Cscalar) {
	C.atg_clamp_(ptr, self, min, max)
}

// void atg_clamp(tensor *, tensor self, scalar min, scalar max);
func AtgClamp(ptr *Ctensor, self Ctensor, min Cscalar, max Cscalar) {
	C.atg_clamp(ptr, self, min, max)
}

// void atg_relu(tensor *, tensor self);
func AtgRelu(ptr *Ctensor, self Ctensor) {
	C.atg_relu(ptr, self)
}

// void atg_relu_(tensor *, tensor self);
func AtgRelu_(ptr *Ctensor, self Ctensor) {
	C.atg_relu_(ptr, self)
}

// void atg_t(tensor *, tensor self);
func AtgT(ptr *Ctensor, self Ctensor) {
	C.atg_t(ptr, self)
}

// void atg_t_(tensor *, tensor self);
func AtgT_(ptr *Ctensor, self Ctensor) {
	C.atg_t_(ptr, self)
}

// void atg_mse_loss(tensor *, tensor self, tensor target, int64_t reduction);
func AtgMseLoss(ptr *Ctensor, self Ctensor, target Ctensor, reduction int) {
	creduction := *(*C.int64_t)(unsafe.Pointer(&reduction))

	C.atg_mse_loss(ptr, self, target, creduction)
}

// void atg_exp(tensor *, tensor self);
func AtgExp(ptr *Ctensor, self Ctensor) {
	C.atg_exp(ptr, self)
}

// void atg_exp_(tensor *, tensor self);
func AtgExp_(ptr *Ctensor, self Ctensor) {
	C.atg_exp_(ptr, self)
}

// void atg_pow(tensor *, tensor self, scalar exponent);
func AtgPow(ptr *Ctensor, self Ctensor, exponent Cscalar) {
	C.atg_pow(ptr, self, exponent)
}

// void atg_sum(tensor *, tensor self, int dtype);
func AtgSum(ptr *Ctensor, self Ctensor, dtype int32) {
	cdtype := *(*C.int)(unsafe.Pointer(&dtype))

	C.atg_sum(ptr, self, cdtype)
}

// void atg_sub(tensor *, tensor self, tensor other);
func AtgSub(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_sub(ptr, self, other)
}

// void atg_sub1(tensor *, tensor self, scalar other);
func AtgSub1(ptr *Ctensor, self Ctensor, other Cscalar) {
	C.atg_sub1(ptr, self, other)
}

// void atg_sub_(tensor *, tensor self, tensor other);
func AtgSub_(ptr *Ctensor, self Ctensor, other Ctensor) {
	C.atg_sub_(ptr, self, other)
}

// void atg_conv1d(tensor *, tensor input, tensor weight, tensor bias, int64_t *stride_data, int stride_len, int64_t *padding_data, int padding_len, int64_t *dilation_data, int dilation_len, int64_t groups);
func AtgConv1d(ptr *Ctensor, input Ctensor, weight Ctensor, bias Ctensor, strideData []int64, strideLen int, paddingData []int64, paddingLen int, dilationData []int64, dilationLen int, groups int64) {
	cstrideDataPtr := (*C.int64_t)(unsafe.Pointer(&strideData[0]))
	cstrideLen := *(*C.int)(unsafe.Pointer(&strideLen))
	cpaddingDataPtr := (*C.int64_t)(unsafe.Pointer(&paddingData[0]))
	cpaddingLen := *(*C.int)(unsafe.Pointer(&paddingLen))
	cdilationDataPtr := (*C.int64_t)(unsafe.Pointer(&dilationData[0]))
	cdilationLen := *(*C.int)(unsafe.Pointer(&dilationLen))
	cgroups := *(*C.int64_t)(unsafe.Pointer(&groups))

	C.atg_conv1d(ptr, input, weight, bias, cstrideDataPtr, cstrideLen, cpaddingDataPtr, cpaddingLen, cdilationDataPtr, cdilationLen, cgroups)
}

// void atg_conv2d(tensor *, tensor input, tensor weight, tensor bias, int64_t *stride_data, int stride_len, int64_t *padding_data, int padding_len, int64_t *dilation_data, int dilation_len, int64_t groups);
func AtgConv2d(ptr *Ctensor, input Ctensor, weight Ctensor, bias Ctensor, strideData []int64, strideLen int, paddingData []int64, paddingLen int, dilationData []int64, dilationLen int, groups int64) {
	cstrideDataPtr := (*C.int64_t)(unsafe.Pointer(&strideData[0]))
	cstrideLen := *(*C.int)(unsafe.Pointer(&strideLen))
	cpaddingDataPtr := (*C.int64_t)(unsafe.Pointer(&paddingData[0]))
	cpaddingLen := *(*C.int)(unsafe.Pointer(&paddingLen))
	cdilationDataPtr := (*C.int64_t)(unsafe.Pointer(&dilationData[0]))
	cdilationLen := *(*C.int)(unsafe.Pointer(&dilationLen))
	cgroups := *(*C.int64_t)(unsafe.Pointer(&groups))

	C.atg_conv2d(ptr, input, weight, bias, cstrideDataPtr, cstrideLen, cpaddingDataPtr, cpaddingLen, cdilationDataPtr, cdilationLen, cgroups)
}

// void atg_conv3d(tensor *, tensor input, tensor weight, tensor bias, int64_t *stride_data, int stride_len, int64_t *padding_data, int padding_len, int64_t *dilation_data, int dilation_len, int64_t groups);
func AtgConv3d(ptr *Ctensor, input Ctensor, weight Ctensor, bias Ctensor, strideData []int64, strideLen int, paddingData []int64, paddingLen int, dilationData []int64, dilationLen int, groups int64) {
	cstrideDataPtr := (*C.int64_t)(unsafe.Pointer(&strideData[0]))
	cstrideLen := *(*C.int)(unsafe.Pointer(&strideLen))
	cpaddingDataPtr := (*C.int64_t)(unsafe.Pointer(&paddingData[0]))
	cpaddingLen := *(*C.int)(unsafe.Pointer(&paddingLen))
	cdilationDataPtr := (*C.int64_t)(unsafe.Pointer(&dilationData[0]))
	cdilationLen := *(*C.int)(unsafe.Pointer(&dilationLen))
	cgroups := *(*C.int64_t)(unsafe.Pointer(&groups))

	C.atg_conv3d(ptr, input, weight, bias, cstrideDataPtr, cstrideLen, cpaddingDataPtr, cpaddingLen, cdilationDataPtr, cdilationLen, cgroups)
}

// void atg_max_pool2d(tensor *, tensor self, int64_t *kernel_size_data, int kernel_size_len, int64_t *stride_data, int stride_len, int64_t *padding_data, int padding_len, int64_t *dilation_data, int dilation_len, int ceil_mode);
func AtgMaxPool2d(ptr *Ctensor, self Ctensor, kernelSizeData []int64, kernelSizeLen int, strideData []int64, strideLen int, paddingData []int64, paddingLen int, dilationData []int64, dilationLen int, ceilMode int) {

	ckernelSizeDataPtr := (*C.int64_t)(unsafe.Pointer(&kernelSizeData[0]))
	ckernelSizeLen := *(*C.int)(unsafe.Pointer(&kernelSizeLen))
	cstrideDataPtr := (*C.int64_t)(unsafe.Pointer(&strideData[0]))
	cstrideLen := *(*C.int)(unsafe.Pointer(&strideLen))
	cpaddingDataPtr := (*C.int64_t)(unsafe.Pointer(&paddingData[0]))
	cpaddingLen := *(*C.int)(unsafe.Pointer(&paddingLen))
	cdilationDataPtr := (*C.int64_t)(unsafe.Pointer(&dilationData[0]))
	cdilationLen := *(*C.int)(unsafe.Pointer(&dilationLen))
	cceilMode := *(*C.int)(unsafe.Pointer(&ceilMode))

	C.atg_max_pool2d(ptr, self, ckernelSizeDataPtr, ckernelSizeLen, cstrideDataPtr, cstrideLen, cpaddingDataPtr, cpaddingLen, cdilationDataPtr, cdilationLen, cceilMode)
}

// void atg_dropout(tensor *, tensor input, double p, int train);
func AtgDropout(ptr *Ctensor, input Ctensor, p float64, train int) {
	cp := *(*C.double)(unsafe.Pointer(&p))
	ctrain := *(*C.int)(unsafe.Pointer(&train))

	C.atg_dropout(ptr, input, cp, ctrain)
}

// void atg_dropout_(tensor *, tensor self, double p, int train);
func AtgDropout_(ptr *Ctensor, self Ctensor, p float64, train int) {
	cp := *(*C.double)(unsafe.Pointer(&p))
	ctrain := *(*C.int)(unsafe.Pointer(&train))

	C.atg_dropout_(ptr, self, cp, ctrain)
}

// void atg_conv_transpose1d(tensor *, tensor input, tensor weight, tensor bias, int64_t *stride_data, int stride_len, int64_t *padding_data, int padding_len, int64_t *output_padding_data, int output_padding_len, int64_t groups, int64_t *dilation_data, int dilation_len);
func AtgConvTranspose1d(ptr *Ctensor, input Ctensor, weight Ctensor, bias Ctensor, strideData []int64, strideLen int, paddingData []int64, paddingLen int, outputPaddingData []int64, outputPaddingLen int, dilationData []int64, dilationLen int, groups int64) {
	cstrideDataPtr := (*C.int64_t)(unsafe.Pointer(&strideData[0]))
	cstrideLen := *(*C.int)(unsafe.Pointer(&strideLen))
	cpaddingDataPtr := (*C.int64_t)(unsafe.Pointer(&paddingData[0]))
	cpaddingLen := *(*C.int)(unsafe.Pointer(&paddingLen))
	coutputPaddingDataPtr := (*C.int64_t)(unsafe.Pointer(&outputPaddingData[0]))
	coutputPaddingLen := *(*C.int)(unsafe.Pointer(&outputPaddingLen))
	cdilationDataPtr := (*C.int64_t)(unsafe.Pointer(&dilationData[0]))
	cdilationLen := *(*C.int)(unsafe.Pointer(&dilationLen))
	cgroups := *(*C.int64_t)(unsafe.Pointer(&groups))

	C.atg_conv_transpose1d(ptr, input, weight, bias, cstrideDataPtr, cstrideLen, cpaddingDataPtr, cpaddingLen, coutputPaddingDataPtr, coutputPaddingLen, cgroups, cdilationDataPtr, cdilationLen)
}

// void atg_conv_transpose2d(tensor *, tensor input, tensor weight, tensor bias, int64_t *stride_data, int stride_len, int64_t *padding_data, int padding_len, int64_t *output_padding_data, int output_padding_len, int64_t groups, int64_t *dilation_data, int dilation_len);
func AtgConvTranspose2d(ptr *Ctensor, input Ctensor, weight Ctensor, bias Ctensor, strideData []int64, strideLen int, paddingData []int64, paddingLen int, outputPaddingData []int64, outputPaddingLen int, dilationData []int64, dilationLen int, groups int64) {
	cstrideDataPtr := (*C.int64_t)(unsafe.Pointer(&strideData[0]))
	cstrideLen := *(*C.int)(unsafe.Pointer(&strideLen))
	cpaddingDataPtr := (*C.int64_t)(unsafe.Pointer(&paddingData[0]))
	cpaddingLen := *(*C.int)(unsafe.Pointer(&paddingLen))
	coutputPaddingDataPtr := (*C.int64_t)(unsafe.Pointer(&outputPaddingData[0]))
	coutputPaddingLen := *(*C.int)(unsafe.Pointer(&outputPaddingLen))
	cdilationDataPtr := (*C.int64_t)(unsafe.Pointer(&dilationData[0]))
	cdilationLen := *(*C.int)(unsafe.Pointer(&dilationLen))
	cgroups := *(*C.int64_t)(unsafe.Pointer(&groups))

	C.atg_conv_transpose2d(ptr, input, weight, bias, cstrideDataPtr, cstrideLen, cpaddingDataPtr, cpaddingLen, coutputPaddingDataPtr, coutputPaddingLen, cgroups, cdilationDataPtr, cdilationLen)
}

// void atg_conv_transpose3d(tensor *, tensor input, tensor weight, tensor bias, int64_t *stride_data, int stride_len, int64_t *padding_data, int padding_len, int64_t *output_padding_data, int output_padding_len, int64_t groups, int64_t *dilation_data, int dilation_len);
func AtgConvTranspose3d(ptr *Ctensor, input Ctensor, weight Ctensor, bias Ctensor, strideData []int64, strideLen int, paddingData []int64, paddingLen int, outputPaddingData []int64, outputPaddingLen int, dilationData []int64, dilationLen int, groups int64) {
	cstrideDataPtr := (*C.int64_t)(unsafe.Pointer(&strideData[0]))
	cstrideLen := *(*C.int)(unsafe.Pointer(&strideLen))
	cpaddingDataPtr := (*C.int64_t)(unsafe.Pointer(&paddingData[0]))
	cpaddingLen := *(*C.int)(unsafe.Pointer(&paddingLen))
	coutputPaddingDataPtr := (*C.int64_t)(unsafe.Pointer(&outputPaddingData[0]))
	coutputPaddingLen := *(*C.int)(unsafe.Pointer(&outputPaddingLen))
	cdilationDataPtr := (*C.int64_t)(unsafe.Pointer(&dilationData[0]))
	cdilationLen := *(*C.int)(unsafe.Pointer(&dilationLen))
	cgroups := *(*C.int64_t)(unsafe.Pointer(&groups))

	C.atg_conv_transpose3d(ptr, input, weight, bias, cstrideDataPtr, cstrideLen, cpaddingDataPtr, cpaddingLen, coutputPaddingDataPtr, coutputPaddingLen, cgroups, cdilationDataPtr, cdilationLen)
}

// void atg_lstm(tensor *, tensor input, tensor *hx_data, int hx_len, tensor *params_data, int params_len, int has_biases, int64_t num_layers, double dropout, int train, int bidirectional, int batch_first);
func AtgLstm(ptr *Ctensor, input Ctensor, hxData []Ctensor, hxLen int, paramsData []Ctensor, paramsLen int, hasBiases int, numLayers int64, dropout float64, train int, bidirectional int, batchFirst int) {

	chxDataPtr := (*Ctensor)(unsafe.Pointer(&hxData[0]))
	chxLen := *(*C.int)(unsafe.Pointer(&hxLen))
	cparamsDataPtr := (*Ctensor)(unsafe.Pointer(&paramsData[0]))
	cparamsLen := *(*C.int)(unsafe.Pointer(&paramsLen))
	chasBiases := *(*C.int)(unsafe.Pointer(&hasBiases))
	cnumLayers := *(*C.int64_t)(unsafe.Pointer(&numLayers))
	cdropout := *(*C.double)(unsafe.Pointer(&dropout))
	ctrain := *(*C.int)(unsafe.Pointer(&train))
	cbidirectional := *(*C.int)(unsafe.Pointer(&bidirectional))
	cbatchFirst := *(*C.int)(unsafe.Pointer(&batchFirst))

	C.atg_lstm(ptr, input, chxDataPtr, chxLen, cparamsDataPtr, cparamsLen, chasBiases, cnumLayers, cdropout, ctrain, cbidirectional, cbatchFirst)
}

// void atg_gru(tensor *, tensor input, tensor hx, tensor *params_data, int params_len, int has_biases, int64_t num_layers, double dropout, int train, int bidirectional, int batch_first);
func AtgGru(ptr *Ctensor, input Ctensor, hx Ctensor, paramsData []Ctensor, paramsLen int, hasBiases int, numLayers int64, dropout float64, train int, bidirectional int, batchFirst int) {

	cparamsDataPtr := (*Ctensor)(unsafe.Pointer(&paramsData[0]))
	cparamsLen := *(*C.int)(unsafe.Pointer(&paramsLen))
	chasBiases := *(*C.int)(unsafe.Pointer(&hasBiases))
	cnumLayers := *(*C.int64_t)(unsafe.Pointer(&numLayers))
	cdropout := *(*C.double)(unsafe.Pointer(&dropout))
	ctrain := *(*C.int)(unsafe.Pointer(&train))
	cbidirectional := *(*C.int)(unsafe.Pointer(&bidirectional))
	cbatchFirst := *(*C.int)(unsafe.Pointer(&batchFirst))

	C.atg_gru(ptr, input, hx, cparamsDataPtr, cparamsLen, chasBiases, cnumLayers, cdropout, ctrain, cbidirectional, cbatchFirst)
}

// void atg_randn(tensor *, int64_t *size_data, int size_len, int options_kind, int options_device);
func AtgRandn(ptr *Ctensor, sizeData []int64, sizeLen int, optionsKind int32, optionsDevice int32) {

	csizeDataPtr := (*C.int64_t)(unsafe.Pointer(&sizeData[0]))
	csizeLen := *(*C.int)(unsafe.Pointer(&sizeLen))
	coptionKind := *(*C.int)(unsafe.Pointer(&optionsKind))
	coptionDevice := *(*C.int)(unsafe.Pointer(&optionsDevice))

	C.atg_randn(ptr, csizeDataPtr, csizeLen, coptionKind, coptionDevice)
}

// void atg_embedding(tensor *, tensor weight, tensor indices, int64_t padding_idx, int scale_grad_by_freq, int sparse);
func AtgEmbedding(ptr *Ctensor, weight Ctensor, indices Ctensor, paddingIdx int64, scaleGradByFreq int, sparse int) {

	cpaddingIdx := *(*C.int64_t)(unsafe.Pointer(&paddingIdx))
	cscaleGradByFreq := *(*C.int)(unsafe.Pointer(&scaleGradByFreq))
	csparse := *(*C.int)(unsafe.Pointer(&sparse))

	C.atg_embedding(ptr, weight, indices, cpaddingIdx, cscaleGradByFreq, csparse)
}

// void atg_randint(tensor *, int64_t high, int64_t *size_data, int size_len, int options_kind, int options_device);
func AtgRandint(ptr *Ctensor, high int64, sizeData []int64, sizeLen int, optionsKind int32, optionsDevice int32) {

	chigh := *(*C.int64_t)(unsafe.Pointer(&high))
	csizeDataPtr := (*C.int64_t)(unsafe.Pointer(&sizeData[0]))
	csizeLen := *(*C.int)(unsafe.Pointer(&sizeLen))
	coptionKind := *(*C.int)(unsafe.Pointer(&optionsKind))
	coptionDevice := *(*C.int)(unsafe.Pointer(&optionsDevice))

	C.atg_randint(ptr, chigh, csizeDataPtr, csizeLen, coptionKind, coptionDevice)
}

// void atg_layer_norm(tensor *, tensor input, int64_t *normalized_shape_data, int normalized_shape_len, tensor weight, tensor bias, double eps, int cudnn_enable);
func AtgLayerNorm(ptr *Ctensor, input Ctensor, normalizedShapeData []int64, normalizedShapeLen int, weight Ctensor, bias Ctensor, eps float64, cudnnEnable int) {

	cnormalizedShapeDataPtr := (*C.int64_t)(unsafe.Pointer(&normalizedShapeData[0]))
	cnormalizedShapeLen := *(*C.int)(unsafe.Pointer(&normalizedShapeLen))
	ceps := *(*C.double)(unsafe.Pointer(&eps))
	ccudnnEnable := *(*C.int)(unsafe.Pointer(&cudnnEnable))

	C.atg_layer_norm(ptr, input, cnormalizedShapeDataPtr, cnormalizedShapeLen, weight, bias, ceps, ccudnnEnable)
}

// void atg_batch_norm(tensor *, tensor input, tensor weight, tensor bias, tensor running_mean, tensor running_var, int training, double momentum, double eps, int cudnn_enabled);
func AtgBatchNorm(ptr *Ctensor, input Ctensor, weight Ctensor, bias Ctensor, runningMean Ctensor, runningVar Ctensor, training int, momentum float64, eps float64, cudnnEnable int) {

	ctraining := *(*C.int)(unsafe.Pointer(&training))
	cmomentum := *(*C.double)(unsafe.Pointer(&momentum))
	ceps := *(*C.double)(unsafe.Pointer(&eps))
	ccudnnEnable := *(*C.int)(unsafe.Pointer(&cudnnEnable))

	C.atg_batch_norm(ptr, input, weight, bias, runningMean, runningVar, ctraining, cmomentum, ceps, ccudnnEnable)
}

// void atg_cat(tensor *, tensor *tensors_data, int tensors_len, int64_t dim);
func AtgCat(ptr *Ctensor, tensorsData []Ctensor, tensorsLen int, dim int64) {
	tensorsDataPtr := (*Ctensor)(unsafe.Pointer(&tensorsData[0]))
	ctensorsLen := *(*C.int)(unsafe.Pointer(&tensorsLen))
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))

	C.atg_cat(ptr, tensorsDataPtr, ctensorsLen, cdim)
}

// void atg_topk(tensor *, tensor self, int64_t k, int64_t dim, int largest, int sorted);
func AtgTopk(ptr *Ctensor, self Ctensor, k int64, dim int64, largest int, sorted int) {
	ck := *(*C.int64_t)(unsafe.Pointer(&k))
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))
	clargest := *(*C.int)(unsafe.Pointer(&largest))
	csorted := *(*C.int)(unsafe.Pointer(&sorted))

	C.atg_topk(ptr, self, ck, cdim, clargest, csorted)
}

// void atg_adaptive_avg_pool2d(tensor *, tensor self, int64_t *output_size_data, int output_size_len);
func AtgAdaptiveAvgPool2d(ptr *Ctensor, self Ctensor, outputSizeData []int64, outputSizeLen int) {
	outputSizeDataPtr := (*C.int64_t)(unsafe.Pointer(&outputSizeData[0]))
	coutputSizeLen := *(*C.int)(unsafe.Pointer(&outputSizeLen))

	C.atg_adaptive_avg_pool2d(ptr, self, outputSizeDataPtr, coutputSizeLen)
}
