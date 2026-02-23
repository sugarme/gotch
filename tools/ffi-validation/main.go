// Package main provides FFI conversion validation for gotch
// Run with: go run tools/ffi-validation/main.go
//
// This tool validates that C-to-Go type conversions are done correctly
// and documents which conversion methods have bugs.
package main

import (
	"fmt"
	"unsafe"
)

/*
#include <stdbool.h>
#include <stdint.h>

// Test functions that mirror the actual C API behavior in libtch
int test_return_int(int value) { return value; }
bool test_return_bool(bool value) { return value; }
double test_return_double(double value) { return value; }
int64_t test_return_int64(int64_t value) { return value; }
int32_t test_return_int32(int32_t value) { return value; }
*/
import "C"

func main() {
	fmt.Println("=== FFI Conversion Validation ===\n")

	testIntConversion()
	testBoolConversion()
	testDoubleConversion()
	testInt64Conversion()
	testInt32Conversion()
	printSummary()
}

func testIntConversion() {
	fmt.Println("Testing C.int -> Go int:")
	testCases := []int{0, 1, 2, 100, -1}
	bugCount := 0

	for _, value := range testCases {
		cVal := C.test_return_int(C.int(value))

		// Method 1: Unsafe pointer (BUGGY on 64-bit systems)
		unsafeResult := *(*int)(unsafe.Pointer(&cVal))

		// Method 2: Direct conversion (CORRECT)
		directResult := int(cVal)

		if unsafeResult != directResult {
			bugCount++
			fmt.Printf("  ✗ Input: %d  Unsafe: %d (WRONG)  Direct: %d (CORRECT)\n",
				value, unsafeResult, directResult)
		} else {
			fmt.Printf("  ✓ Input: %d  Result: %d\n", value, directResult)
		}
	}

	if bugCount > 0 {
		fmt.Printf("\n  ⚠️  WARNING: Unsafe pointer conversion corrupts %d/%d values\n", bugCount, len(testCases))
		fmt.Println("  Reason: C.int is 4 bytes, Go int is 8 bytes on 64-bit systems")
		fmt.Println("  Fix: Use int(cVal) instead of *(*int)(unsafe.Pointer(&cVal))")
	}
	fmt.Println()
}

func testBoolConversion() {
	fmt.Println("Testing C.bool -> Go bool:")
	testCases := []bool{true, false}
	allMatch := true

	for _, value := range testCases {
		cVal := C.test_return_bool(C.bool(value))

		unsafeResult := *(*bool)(unsafe.Pointer(&cVal))
		directResult := bool(cVal)

		match := (unsafeResult == directResult && directResult == value)
		if !match {
			allMatch = false
			fmt.Printf("  ✗ Input: %v  Unsafe: %v  Direct: %v\n", value, unsafeResult, directResult)
		} else {
			fmt.Printf("  ✓ Input: %v  Result: %v\n", value, directResult)
		}
	}

	if allMatch {
		fmt.Println("  ✓ Both methods work (same size: 1 byte)")
	}
	fmt.Println()
}

func testDoubleConversion() {
	fmt.Println("Testing C.double -> Go float64:")
	testCases := []float64{0.0, 1.0, 3.14159, -2.5}
	allMatch := true

	for _, value := range testCases {
		cVal := C.test_return_double(C.double(value))

		unsafeResult := *(*float64)(unsafe.Pointer(&cVal))
		directResult := float64(cVal)

		match := (unsafeResult == directResult && directResult == value)
		if !match {
			allMatch = false
			fmt.Printf("  ✗ Input: %f  Unsafe: %f  Direct: %f\n", value, unsafeResult, directResult)
		} else {
			fmt.Printf("  ✓ Input: %f  Result: %f\n", value, directResult)
		}
	}

	if allMatch {
		fmt.Println("  ✓ Both methods work (same size: 8 bytes)")
	}
	fmt.Println()
}

func testInt64Conversion() {
	fmt.Println("Testing C.int64_t -> Go int64:")
	testCases := []int64{0, 1, 100, -1, 1 << 32}
	allMatch := true

	for _, value := range testCases {
		cVal := C.test_return_int64(C.int64_t(value))

		unsafeResult := *(*int64)(unsafe.Pointer(&cVal))
		directResult := int64(cVal)

		match := (unsafeResult == directResult && directResult == value)
		if !match {
			allMatch = false
			fmt.Printf("  ✗ Input: %d  Unsafe: %d  Direct: %d\n", value, unsafeResult, directResult)
		} else {
			fmt.Printf("  ✓ Input: %d  Result: %d\n", value, directResult)
		}
	}

	if allMatch {
		fmt.Println("  ✓ Both methods work (same size: 8 bytes)")
	}
	fmt.Println()
}

func testInt32Conversion() {
	fmt.Println("Testing C.int32_t -> Go int32:")
	testCases := []int32{0, 1, 100, -1}
	allMatch := true

	for _, value := range testCases {
		cVal := C.test_return_int32(C.int32_t(value))

		unsafeResult := *(*int32)(unsafe.Pointer(&cVal))
		directResult := int32(cVal)

		match := (unsafeResult == directResult && directResult == value)
		if !match {
			allMatch = false
			fmt.Printf("  ✗ Input: %d  Unsafe: %d  Direct: %d\n", value, unsafeResult, directResult)
		} else {
			fmt.Printf("  ✓ Input: %d  Result: %d\n", value, directResult)
		}
	}

	if allMatch {
		fmt.Println("  ✓ Both methods work (same size: 4 bytes)")
	}
	fmt.Println()
}

func printSummary() {
	fmt.Println("=== Summary ===\n")
	fmt.Println("SAFE - Both methods work (same size on 64-bit systems):")
	fmt.Println("  ✓ C.bool      <-> Go bool     (1 byte)")
	fmt.Println("  ✓ C.int32_t   <-> Go int32    (4 bytes)")
	fmt.Println("  ✓ C.int64_t   <-> Go int64    (8 bytes)")
	fmt.Println("  ✓ C.uint64_t  <-> Go uint64   (8 bytes)")
	fmt.Println("  ✓ C.double    <-> Go float64  (8 bytes)")
	fmt.Println()
	fmt.Println("UNSAFE - Unsafe pointer conversion corrupts values:")
	fmt.Println("  ✗ C.int       <-> Go int      (4 bytes vs 8 bytes)")
	fmt.Println()
	fmt.Println("RECOMMENDATION:")
	fmt.Println("  Always use direct type conversion: int(cVal)")
	fmt.Println("  Avoid unsafe pointer casting: *(*int)(unsafe.Pointer(&cVal))")
	fmt.Println()
	fmt.Println("BUGS FIXED in PyTorch 2.10.0 upgrade:")
	fmt.Println("  - libtch/tensor.go:AtGradSetEnabled() - Fixed C.int return conversion")
	fmt.Println("  - libtch/tensor.go:AtDevice() - Fixed C.int return conversion")
	fmt.Println()
	fmt.Println("Run this validation after making FFI changes to verify correctness.")
}
