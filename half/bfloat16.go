package half

import (
	"math"
	"math/bits"
)

// A 16-bit floating point type implementing the bfloat16 format.
// Ref. https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
// https://github.com/starkat99/half-rs/tree/main/src/bfloat

// The bfloat16 - Google 'brain' floating point format is a truncated 16-bit version of the IEEE 754 standard binary32.
// bfloat16 has approximately the same dynamic range as float32 (8 bits -> 3.4 × 10^38) by having a lower precision than float16.
// While float16 has a precision of 10 bits, bfloat16 has a precision of only 7 bits.
//
// +------------+------------------------+----------------------------+
// | 1-bit sign | 8-bit exponent (range) | 7-bit fraction (precision) |
// +------------+------------------------+----------------------------+
type BFloat16 uint16

// Ref.https://github.com/starkat99/half-rs/blob/cabfc74e2a48b44b4556780f9d1550dd50a708be/src/bfloat/convert.rs#L5C1-L24C1
func Float32ToBFloat16(value float32) uint16 {
	// convert to raw bytes
	x := math.Float32bits(value)

	// Check for NaN
	if (x & 0x7FFF_FFFF) > 0x7F80_0000 {
		// keep high part of current mantissa but also set most significant mantissa bit
		return uint16((x >> 16) | 0x0040)
	}

	// Round and shift
	var roundBit uint32 = 0x0000_8000
	if ((x & roundBit) != 0) && ((x & (3*roundBit - 1)) != 0) {
		return uint16(x>>16) + 1
	} else {
		return uint16(x >> 16)
	}
}

func Float64ToBFloat16(value float64) uint16 {
	// Convert o raw bytes, truncating the last 32-bits of mantissa
	// that precision will always be lost on half-precision
	val := math.Float64bits(value)
	x := uint32(val >> 32)

	// Extract IEEE754 components
	sign := x & 0x8000_0000
	exp := x & 0x7FF0_0000
	man := x & 0x000F_FFFF

	// Check for all exponent bit being set, which is Infinity or NaN
	if exp == 0x7FF0_0000 {
		// Set mantissa MSB for NaN  and also keep shifted mantissa bits.
		// Also check the last 32 bits.
		var nanBit uint32 = 0x0040
		if man == 0 && (uint32(val) == 0) {
			nanBit = 0
		}

		return uint16((sign >> 16) | 0x7F80 | nanBit | (man >> 13))
	}

	// The number is normalized, start assembling half precision version
	halfSign := sign >> 16

	// Unbias the exponent, then bias for bfloat16 precision
	unbiasedExp := (int64(exp>>20) - 1023)
	halfExp := unbiasedExp + 127

	// Check for exponent overflow, return +infinity
	if halfExp >= 0xFF {
		return uint16(halfSign | 0x7F80)
	}

	// Check for underflow
	if halfExp <= 0 {
		// Check mantissa for what we can do
		if 7-halfExp > 21 {
			// No rounding possibility, so this is a full underflow, return signed zero
			return uint16(halfSign)
		}

		// Don't forget about hidden leading mantissa bit when assembling mantissa
		man = man | 0x0010_0000
		halfMan := man >> (14 - halfExp)

		// Check for rounding
		var roundBit uint32 = 1 << (13 - halfExp)
		if ((man & roundBit) != 0) && ((man & (3*roundBit - 1)) != 0) {
			halfMan += 1
		}

		// No exponent for subnormals
		return uint16(halfSign | halfMan)
	}

	// Rebias the exponent
	halfExp1 := uint32(halfExp) << 7
	halfMan1 := man >> 13

	// Check for rounding
	var roundBit1 uint32 = 0x0000_1000

	if ((man & roundBit1) != 0) && ((man & (3*roundBit1 - 1)) != 0) {
		// Round it
		return uint16((halfSign | halfExp1 | halfMan1) + 1)
	} else {
		return uint16(halfSign | halfExp1 | halfMan1)
	}
}

func BFloat16ToFloat32(i uint16) float32 {
	// If NaN, keep current mantissa but also set most significant mantissa bit
	if i&0x7FFF > 0x7F80 {
		return math.Float32frombits((uint32(i) | 0x0040) << 16)
	} else {
		return math.Float32frombits(uint32(i) << 16)
	}
}

func BFloat16ToFloat64(i uint16) float64 {
	// Check for signed zero
	if i&0x7FFF == 0 {
		return math.Float64frombits(uint64(i) << 48)
	}

	halfSign := uint64(i & 0x8000)
	halfExp := uint64(i & 0x7F80)
	halfMan := uint64(i & 0x007F)

	// Check for an infinity or NaN when all exponent bits set
	if halfExp == 0x7F80 {
		// Check for signed infinity if mantissa is zero
		if halfMan == 0 {
			return math.Float64frombits((halfSign << 48) | 0x7FF0_0000_0000_0000)
		} else {
			// NaN, keep current mantissa but also set most significant mantissa bit
			return math.Float64frombits((halfSign << 48) | 0x7FF8_0000_0000_0000 | (halfMan << 45))
		}
	}

	// Calculate double-precision components with adjusted exponent
	sign := halfSign << 48

	// Unbias exponent
	unbiasedExp := (int64(halfExp) >> 7) - 127

	// Check for subnormals, which will be normalized by adjusting exponent
	if halfExp == 0 {
		// Calculate how much to adjust the exponent by
		// leading zeros uint16
		e := bits.LeadingZeros16(uint16(halfMan)) - 9

		// Rebias and adjust exponent
		exp := (uint64(1023-127-e) << 52)
		man := (halfMan << (46 + e)) & 0xF_FFFF_FFFF_FFFF

		return math.Float64frombits(sign | exp | man)
	}

	// Rebias exponent for a normalized normal
	exp := uint64(unbiasedExp+1023) << 52
	man := (halfMan & 0x007F) << 45

	return math.Float64frombits(sign | exp | man)
}
