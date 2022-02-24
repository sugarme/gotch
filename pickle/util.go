package pickle

import "io"

// Converts the bits representation of a Half Float (16 bits) number to
// an IEEE 754 float representation (32 bits)
// From http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
func FloatBits16to32(u16 uint16) uint32 {
	return mantissaTable[offsetTable[u16>>10]+(uint32(u16)&0x3ff)] + exponentTable[u16>>10]
}

var mantissaTable [2048]uint32
var exponentTable [64]uint32
var offsetTable [64]uint32

func init() {
	initMantissaTable()
	initExponentTable()
	initOffsetTable()
}

func initMantissaTable() {
	mantissaTable[0] = 0
	for i := uint32(1); i < 1024; i++ {
		mantissaTable[i] = convertMantissa(i)
	}
	for i := uint32(1024); i < 2048; i++ {
		mantissaTable[i] = 0x38000000 + ((i - 1024) << 13)
	}
}

func initExponentTable() {
	exponentTable[0] = 0
	exponentTable[31] = 0x47800000
	exponentTable[32] = 0x80000000
	exponentTable[63] = 0xC7800000
	for i := uint32(1); i < 31; i++ {
		exponentTable[i] = i << 23
	}
	for i := uint32(33); i < 63; i++ {
		exponentTable[i] = 0x80000000 + (i-32)<<23
	}
}

func initOffsetTable() {
	offsetTable[0] = 0
	offsetTable[32] = 0
	for i := uint32(1); i < 31; i++ {
		offsetTable[i] = 1024
	}
	for i := uint32(32); i < 64; i++ {
		offsetTable[i] = 1024
	}
}

func convertMantissa(i uint32) uint32 {
	var m uint32 = i << 13  // zero pad mantissa bits
	var e uint32 = 0        // zero exponent
	for m&0x00800000 != 0 { // while not normalized
		e -= 0x00800000 // decrement exponent (1 << 23)
		m <<= 1         // shift mantissa
	}
	m &= ^uint32(0x00800000) // clear leading 1 bit
	e += 0x38800000          // adjust bias ((127-14)<<23)
	return m | e             // return combined number
}

type LimitedBufferReader struct {
	r              io.Reader
	scalarSize     int
	remainingBytes int
	buf            []byte
	bufIndex       int
}

func NewLimitedBufferReader(
	r io.Reader,
	dataSize, scalarSize, bufferSize int,
) *LimitedBufferReader {
	size := bufferSize * scalarSize
	return &LimitedBufferReader{
		r:              r,
		scalarSize:     scalarSize,
		remainingBytes: scalarSize * dataSize,
		buf:            make([]byte, size),
		bufIndex:       size,
	}
}

func (br *LimitedBufferReader) HasNext() bool {
	return br.remainingBytes != 0
}

func (br *LimitedBufferReader) ReadNext() ([]byte, error) {
	if br.remainingBytes == 0 {
		return nil, io.EOF
	}
	if br.bufIndex == len(br.buf) {
		br.bufIndex = 0
		if br.remainingBytes < len(br.buf) {
			br.buf = br.buf[0:br.remainingBytes]
		}
		_, err := br.r.Read(br.buf)
		if err != nil {
			return nil, err
		}
	}
	result := br.buf[br.bufIndex : br.bufIndex+br.scalarSize]
	br.bufIndex += br.scalarSize
	br.remainingBytes -= br.scalarSize
	return result, nil
}
