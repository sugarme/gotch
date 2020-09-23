# NOTES ON WRITING WRAPPER FUNCTIONS


## Function Input Arguments

### `tensor` -> `t *C_tensor`

```c
void at_print(tensor);
```

```go
func AtPrint(t *C_tensor) {
	c_tensor := (C.tensor)((*t).private)
	C.at_print(c_tensor)
}
```

### C pointer e.g `int64_t *` -> `ptr unsafe.Pointer`

In function body, `cPtr := (*C.long)(ptr)`

```c
void at_shape(tensor, int64_t *);
```

```go
func AtShape(t *C_tensor, ptr unsafe.Pointer) {
	c_tensor := (C.tensor)((*t).private)
	c_ptr := (*C.long)(ptr)
	C.at_shape(c_tensor, c_ptr)
}
```

### C types e.g `size_t ndims` -> equivalent Go types `ndims uint`

In function body, `c_ndims := *(*C.size_t)(unsafe.Pointer(&ndims))`

```c
tensor at_tensor_of_data(void *vs, int64_t *dims, size_t ndims, size_t element_size_in_bytes, int type);
```

```go
func AtTensorOfData(vs unsafe.Pointer, dims []int64, ndims uint, elt_size_in_bytes uint, kind int) *C_tensor {

    // 1. Unsafe pointer
	c_dims := (*C.int64_t)(unsafe.Pointer(&dims[0]))
	c_ndims := *(*C.size_t)(unsafe.Pointer(&ndims))
	c_elt_size_in_bytes := *(*C.size_t)(unsafe.Pointer(&elt_size_in_bytes))
	c_kind := *(*C.int)(unsafe.Pointer(&kind))

    // 2. Call C function
	t := C.at_tensor_of_data(vs, c_dims, c_ndims, c_elt_size_in_bytes, c_kind)

    // 3. Form return value
	return &C_tensor{private: unsafe.Pointer(t)}
}
```


## Function Return

### `void *CFUNC(...)`

```c
void *at_data_ptr(tensor);
```

```go
func AtDataPtr(t *C_tensor) unsafe.Pointer {
	c_tensor := (C.tensor)((*t).private)
	return C.at_data_ptr(c_tensor)
}
```

### `tensor` -> `*C_tensor`

then in the return of function body

```go
    // Call C function
    t := C.FUNCTION_TO_CALL(...)
    // Return
	return &C_tensor{private: unsafe.Pointer(t)}
```

### `tensor *CFUNC(...)`

The pattern of **return tensor pointer**: `tensor *atg_FUNCTION_NAME()`.
The returning tensor pointer actually is the FIRST element of a vector of C tensor pointers. 
Next pointer will be calculated from the first. In C land, verifying a valid pointer is 
to check whether it points to **NULL**.

```c

tensor *atg_split(tensor self, int64_t split_size, int64_t dim);

```

```go

// Wrapper
func AtgSplit(self Ctensor, splitSize int64, dim int64) *Ctensor {

	csplitSize := *(*C.int64_t)(unsafe.Pointer(&splitSize))
	cdim := *(*C.int64_t)(unsafe.Pointer(&dim))

	return C.atg_split(self, csplitSize, cdim)
}

// API

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


```


### C types e.g. `C_ulong` -> Go equivalent types `uint64`

then in the return of function body

```go

	c_result := C.FUNCTION_CALL(...)
	return *(*uint64)(unsafe.Pointer(&c_result))

```

### C type pointers e.g. `char *FUNCTION()` --> `*C.char`

then just return the C function call. 

```c
char *get_and_reset_last_err(); // thread-local
```

```go
func GetAndResetLastErr() *C.char{
   return C.get_and_reset_last_err()
}
```


## Multiple Tensors Created In C Land Memory

- When there are multiple Ctensor created in C land memory. A first Ctensor
    pointer will be created and given to the C function. It will create
    consecutive Ctensor(s) based on this pointer. The next pointer(s) can be
    calulated based on this pointer and its size.

- Example: **lstm** function

    + **C function**

    ```C
        void atg_lstm(tensor *, tensor input, tensor *hx_data, int hx_len, tensor *params_data, int params_len, int has_biases, int64_t num_layers, double dropout, int train, int bidirectional, int batch_first);
    ```

    + **Go wrapper function**

    ```go
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
    ```

    + **Go API function**

    ```go
    func (ts Tensor) LSTM(hxData []Tensor, paramsData []Tensor, hasBiases bool, numLayers int64, dropout float64, train bool, bidirectional bool, batchFirst bool) (output, h, c Tensor, err error) {

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

        chasBiases := 0
        if hasBiases {
            chasBiases = 1
        }
        ctrain := 0
        if train {
            ctrain = 1
        }
        cbidirectional := 0
        if bidirectional {
            cbidirectional = 1
        }
        cbatchFirst := 0
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
    ```


