package ts

import (
	"bytes"
	"fmt"
	"log"
	"reflect"
	"strconv"

	"github.com/sugarme/gotch"
)

func (ts *Tensor) ValueGo() interface{} {
	dtype := ts.DType()
	numel := ts.Numel()
	var dst interface{}
	switch dtype {
	case gotch.Uint8:
		dst = make([]uint8, numel)
	case gotch.Int8:
		dst = make([]int8, numel)
	case gotch.Int16:
		dst = make([]int16, numel)
	case gotch.Int:
		dst = make([]int32, numel)
	case gotch.Int64:
		dst = make([]int64, numel)
	case gotch.Float:
		dst = make([]float32, numel)
	case gotch.Double:
		dst = make([]float64, numel)
	case gotch.Bool:
		dst = make([]bool, numel)
	default:
		err := fmt.Errorf("Unsupported type: `dst` type: %v, tensor DType: %v", dtype, ts.DType())
		log.Fatal(err)
	}
	err := ts.CopyData(dst, ts.Numel())
	if err != nil {
		log.Fatal(err)
	}

	// convert []int32 -> int
	if reflect.TypeOf(dst).String() == "[]int32" {
		dst = sliceInt32ToInt(dst.([]int32))
	}

	return dst
}

func shapeToSize(shape []int64) int {
	n := 1
	for _, v := range shape {
		if v == 0 {
			continue
		}
		n = n * int(v)
	}
	return n
}

func shapeToNumels(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= int(d)
	}
	return n
}

func shapeToStrides(shape []int) []int {
	numel := shapeToNumels(shape)
	var strides []int
	for _, v := range shape {
		numel /= int(v)
		strides = append(strides, numel)
	}

	return strides
}

func toSliceInt(in []int64) []int {
	out := make([]int, len(in))
	for i := 0; i < len(in); i++ {
		out[i] = int(in[i])
	}
	return out
}

func maxInt(a, b int) int {
	if a >= b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}

func toSlice(input interface{}) []interface{} {
	vlen := reflect.ValueOf(input).Len()
	out := make([]interface{}, vlen)
	for i := 0; i < vlen; i++ {
		out[i] = reflect.ValueOf(input).Index(i).Interface()
	}

	return out
}

func sliceInterface(data interface{}, start, end int) []interface{} {
	return toSlice(data)[start:end]
}

// Implement Format interface for Tensor:
// ======================================
var (
	fmtByte  = []byte("%")
	precByte = []byte(".")
	fmtFlags = [...]rune{'+', '-', '#', ' ', '0'}

	ufVec    = []byte("Vector")
	ufMat    = []byte("Matrix")
	ufTensor = []byte("Tensor")
)

// fmtState is a custom formatter for Tensor that implements fmt.State interface
type fmtState struct {
	fmt.State
	verb      rune          // format verb
	flat      bool          // whether to print tensor in flatten format
	meta      bool          // whether to print out meta data
	ext       bool          // whether to print full tensor data (no truncation)
	pad       []byte        // padding space
	w         int           // width - total calculated space to print out tensor values
	p         int           // precision for float dtype
	base      int           // integer counting base for integer dtype cases
	htrunc    []byte        // horizontal truncation symbol space
	vtrunc    []byte        // vertical truncation symbol space
	rows      int           // total rows
	cols      int           // total columns
	printRows int           // rows to print
	printCols int           // columns to print
	shape     []int         // shape of tensor to print out
	buf       *bytes.Buffer // memory to hold formated tensor data to print
}

func newFmtState(s fmt.State, verb rune, shape []int) *fmtState {
	w, _ := s.Width()
	p, _ := s.Precision()

	return &fmtState{
		State:  s,
		verb:   verb,
		flat:   s.Flag('-'),
		meta:   s.Flag('+'),
		ext:    s.Flag('#'),
		w:      w,
		p:      p,
		htrunc: []byte("..., "),
		vtrunc: []byte("...,\n"),
		shape:  shape,
		buf:    bytes.NewBuffer(make([]byte, 0)),
	}
}

// originalFmt returns original format.
func (f *fmtState) originalFmt() string {
	// write format symbol and verbs
	buf := bytes.NewBuffer(fmtByte) // '%'
	for _, flag := range fmtFlags {
		if f.Flag(int(flag)) {
			buf.WriteRune(flag)
		}
	}

	// write width format
	if w, ok := f.Width(); ok {
		buf.WriteString(strconv.Itoa(w))
	}

	// write precision verb
	if p, ok := f.Precision(); ok {
		buf.Write(precByte)
		buf.WriteString(strconv.Itoa(p))
	}

	buf.WriteRune(f.verb)

	return buf.String()
}

// cleanFmt returns a start of the format.
func (f *fmtState) initFmt() string {
	buf := bytes.NewBuffer(fmtByte)

	// write width format
	if w, ok := f.Width(); ok {
		buf.WriteString(strconv.Itoa(w))
	}

	// write precision verb
	if p, ok := f.Precision(); ok {
		buf.Write(precByte)
		buf.WriteString(strconv.Itoa(p))
	}

	buf.WriteRune(f.verb)

	return buf.String()
}

// cast casts tensor data to the formatter.
func (f *fmtState) cast(ts *Tensor) {
	// rows and columns
	if ts.Dim() == 1 {
		f.rows = 1
		f.cols = int(ts.Numel())
	} else {
		shape := ts.MustSize()
		f.rows = int(shape[len(shape)-2])
		f.cols = int(shape[len(shape)-1])
	}

	// printRows and printCols
	switch {
	case f.flat && f.ext:
		f.printCols = int(ts.Numel())
	case f.flat:
		f.printCols = 10
	case f.ext:
		f.printCols = f.cols
		f.printRows = f.rows
	default:
		f.printCols = minInt(f.cols, 6)
		f.printRows = minInt(f.rows, 6)
	}
}

// fmtVerb formats verbs.
func (f *fmtState) fmtVerb(ts *Tensor) {
	if f.verb == 'H' { // print out only header.
		f.meta = true
		return
	}

	// var typ T
	typ := ts.DType()

	switch typ.String() {
	case "float32", "float64":
		switch f.verb {
		case 'f', 'e', 'E', 'G', 'b':
			// accepted. Do nothing
		default:
			f.verb = 'g'
		}

	case "uint8", "int8", "int16", "int32", "int64":
		switch f.verb {
		case 'b':
			f.base = 2
		case 'd':
			f.base = 10
		case 'o':
			f.base = 8
		case 'x', 'X':
			f.base = 16
		default:
			f.base = 10
			f.verb = 'd'
		}
	case "bool":
		f.verb = 't'
	default:
		f.verb = 'v'
	}
}

// computeWidth computes a width that can fit for every element.
func (f *fmtState) computeWidth(values interface{}) {
	format := f.initFmt()
	vlen := reflect.ValueOf(values).Len()
	f.w = 0
	for i := 0; i < vlen; i++ {
		val := reflect.ValueOf(values).Index(i)
		w, _ := fmt.Fprintf(f.buf, format, val)

		if w > f.w {
			f.w = w
		}
		f.buf.Reset()
	}
}

// makePad prepares white spaces for print-out format.
func (f *fmtState) makePad() {
	f.pad = make([]byte, maxInt(f.w, 2))
	for i := range f.pad {
		f.pad[i] = ' ' // one white space
	}
}

func (f *fmtState) writeHTrunc() {
	f.Write(f.htrunc)
}

func (f *fmtState) writeVTrunc() {
	f.Write(f.vtrunc)
}

// Format implements fmt.Formatter interface so that we can use
// fmt.Print... and verbs to print out Tensor value in different formats.
func (ts *Tensor) Format(s fmt.State, verb rune) {
	shape := toSliceInt(ts.MustSize())
	strides := shapeToStrides(shape)
	device := ts.MustDevice()
	dtype := ts.DType().String()
	defined := ts.MustDefined()
	if verb == 'i' {
		fmt.Fprintf(
			s,
			"\nTENSOR INFO:\n\tShape:\t\t%v\n\tDType:\t\t%v\n\tDevice:\t\t%v\n\tDefined:\t%v\n",
			shape,
			dtype,
			device,
			defined,
		)
		return
	}

	data := ts.ValueGo()

	f := newFmtState(s, verb, shape)
	f.computeWidth(data)
	f.makePad()
	f.cast(ts)

	// Tensor meta data
	if f.meta {
		switch ts.Dim() {
		case 1:
			f.Write(ufVec)
		case 2:
			f.Write(ufMat)
		default:
			f.Write(ufTensor)
			fmt.Fprintf(f, ": Dim=%d, ", ts.Dim())
		}
		fmt.Fprintf(f, "Shape=%v, Strides=%v\n", shape, strides)
	}

	if f.verb == 'H' {
		return
	}

	if f.flat {
		// TODO.
		// writeFlatTensor()
		log.Printf("WARNING: f.writeFlatTensor() NotImplemedted.\n")
		return
	}

	// 0d (scalar)
	if len(shape) == 0 {
		fmt.Printf("%v", data)
	}

	// 1d (slice)
	if len(shape) == 1 {
		values := sliceInterface(data, 0, shape[0])
		f.writeVector(values)
		return
	}

	// 2d (matrix)
	if len(shape) == 2 {
		vlen := shape[0] * shape[1]
		values := sliceInterface(data, 0, vlen)
		f.writeMatrix(values, shape)
		return
	}

	// >= 3d (tensor)
	f.Write([]byte("\n"))
	f.writeTensor(ts, data)
}

// writeTensor writes input tensor in specified format.
func (f *fmtState) writeTensor(ts *Tensor, values interface{}) {
	shape := toSliceInt(ts.MustSize())
	strides := shapeToStrides(shape)
	size := shapeToNumels(shape)
	mSize := shape[len(shape)-1] * shape[len(shape)-2]
	var (
		offset   int
		printOne = false
	)

	for i := 0; i < size; i += int(mSize) {
		dims := make([]int, len(strides)-2)
		for n := 0; n < len(strides[:len(strides)-2]); n++ {
			stride := strides[n]
			var dim int = offset
			for _, s := range strides[:n] {
				dim = dim % s
			}
			dim = dim / stride
			dims[n] = dim
		}

		var (
			vlimit      = f.printCols
			shouldPrint = true
			printVTrunc bool
			conds       []bool
		)
		for i, val := range dims {
			maxDim := shape[i]
			if (val < vlimit/2) || (val >= maxDim-vlimit/2) {
				conds = append(conds, true)
				shouldPrint = shouldPrint && true
			} else {
				conds = append(conds, false)
				shouldPrint = shouldPrint && false

				printVTrunc = true
			}
		} // inner for
		if shouldPrint {
			dimsLabel := "("
			for _, d := range dims {
				dimsLabel += fmt.Sprintf("%v, ", d)
			}
			dimsLabel += ".,.) =\n"
			f.Write([]byte(dimsLabel))

			// Print matrix [H, W]
			data := sliceInterface(values, offset, offset+mSize)
			f.writeMatrix(data, shape[len(shape)-2:])

			printOne = false
		}

		if printVTrunc && !printOne {
			// NOTE. vertical truncation at > 2D level
			vtrunc := fmt.Sprintf("...,\n\n")
			f.Write([]byte(vtrunc))
			printOne = true
		}
		offset += mSize
	} // outer for
}

func (f *fmtState) writeMatrix(data []interface{}, shape []int) {
	n := shapeToNumels(shape)
	dataLen := len(data)
	if dataLen != n {
		log.Fatalf("mismatched: slice data has %v elements - shape: %v\n", dataLen, n)
	}
	if len(shape) != 2 {
		log.Fatal("Shape must have length of 2.\n")
	}

	stride := int(shape[1])
	currIdx := 0
	nextIdx := stride
	truncatedRows := shape[0] - f.printCols
	for row := 0; row < int(shape[0]); row++ {
		var slice []interface{}
		switch {
		case row < f.printCols/2: // First part
			slice = data[currIdx:nextIdx]
			f.writeVector(slice)
			currIdx = nextIdx
			nextIdx += stride
		case row == f.printCols/2: // Truncated sign
			if f.printCols != f.cols { // truncated mode
				f.writeVTrunc()
			} else { // full mode
				// Do nothing
			}
			currIdx = nextIdx
			nextIdx += stride
		case row > f.printCols/2 && row < f.printCols/2+truncatedRows: // Skip part
			currIdx = nextIdx
			nextIdx += stride
		case row >= f.printCols/2+truncatedRows: // Second part
			slice = data[currIdx:nextIdx]
			f.writeVector(slice)
			currIdx = nextIdx
			nextIdx += stride
		}
	}

	// 1 line between matrix
	f.Write([]byte("\n"))

}

func (f *fmtState) writeVector(data []interface{}) {
	format := f.initFmt()
	vlen := len(data)
	for col := 0; col < vlen; col++ {
		if f.cols <= f.printCols || (col < f.printCols/2 || (col >= f.cols-f.printCols/2)) {
			el := data[col]
			// TODO: more format options here
			w, _ := fmt.Fprintf(f.buf, format, el)
			f.Write(f.buf.Bytes())
			f.Write(f.pad[:f.w-w]) // prepad
			f.Write(f.pad[:2])     // pad
			f.buf.Reset()
		} else if col == f.printCols/2 {
			f.writeHTrunc()
		}
	}

	f.Write([]byte("\n"))
}

// Print prints tensor meta data to stdout.
func (ts *Tensor) Info() {
	fmt.Printf("%i", ts)
}
