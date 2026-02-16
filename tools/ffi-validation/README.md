# FFI Conversion Validation Tool

This tool validates that C-to-Go type conversions are done correctly in gotch's FFI layer.

## Purpose

During the PyTorch 2.10.0 upgrade, we discovered critical bugs in how C types were converted to Go types. This tool documents which conversion methods are safe and which have bugs.

## Running the Tool

```bash
# From gotch root directory
go run tools/ffi-validation/main.go

# Or using Make
make ffi-validate
```

## What It Tests

The tool tests conversions for common C types:
- `C.int` → Go `int` (⚠️ UNSAFE with pointer casting)
- `C.bool` → Go `bool` (✓ SAFE)
- `C.double` → Go `float64` (✓ SAFE)
- `C.int64_t` → Go `int64` (✓ SAFE)
- `C.int32_t` → Go `int32` (✓ SAFE)

## The Bug

On 64-bit systems:
- `C.int` is 4 bytes
- Go `int` is 8 bytes

Using unsafe pointer conversion reads 8 bytes from a 4-byte value, picking up garbage memory:

```go
// WRONG - reads garbage memory
result := *(*int)(unsafe.Pointer(&cVal))  // Returns 4294967297 for input 1

// CORRECT - proper type conversion
result := int(cVal)  // Returns 1 for input 1
```

## Bugs Fixed

This bug affected:
1. `libtch/tensor.go:AtGradSetEnabled()` (line 410-414)
   - Returned garbage instead of previous gradient state
   - Caused gradient state persistence failures

2. `libtch/tensor.go:AtDevice()` (line 104-106)
   - Returned garbage instead of device ID
   - Could cause device selection issues

## When to Run

Run this validation:
- After modifying FFI code in `libtch/`
- When adding new C API bindings
- To verify conversions are correct on your platform

## Example Output

```
Testing C.int -> Go int:
  ✓ Input: 0  Result: 0
  ✗ Input: 1  Unsafe: 4294967297 (WRONG)  Direct: 1 (CORRECT)

  ⚠️  WARNING: Unsafe pointer conversion corrupts 3/5 values
  Fix: Use int(cVal) instead of *(*int)(unsafe.Pointer(&cVal))
```

## Recommendation

**Always use direct type conversion for C return values:**

```go
// ✓ GOOD
func AtDevice(ts Ctensor) int {
    cint := C.at_device(ts)
    return int(cint)  // Direct conversion
}

// ✗ BAD
func AtDevice(ts Ctensor) int {
    cint := C.at_device(ts)
    return *(*int)(unsafe.Pointer(&cint))  // Unsafe pointer - BUGGY
}
```
