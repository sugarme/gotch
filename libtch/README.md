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

### `void *`

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

### C types e.g. `C_ulong` -> Go equivalent types `uint64`

then in the return of function body

```go

	c_result := C.FUNCTION_CALL(...)
	return *(*uint64)(unsafe.Pointer(&c_result))

```



