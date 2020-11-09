package tensor

import (
	"bytes"
	"fmt"
	"log"
	"reflect"
	"strconv"
	"unsafe"

	"github.com/sugarme/gotch"
)

var fmtByte = []byte("%")
var precByte = []byte(".")

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
	// fmt.Println(dst)
	return dst
}
func (ts *Tensor) ToSlice() reflect.Value {
	// Create a 1-dimensional slice of the base large enough for the data and
	// copy the data in.
	shape := ts.MustSize()
	dt := ts.DType()
	n := int(numElements(shape))
	var (
		slice reflect.Value
		typ   reflect.Type
	)
	if dt.String() == "String" {
		panic("Unsupported 'String' type")
	} else {
		gtyp, err := gotch.ToGoType(dt)
		if err != nil {
			log.Fatal(err)
		}
		typ = reflect.SliceOf(gtyp)
		slice = reflect.MakeSlice(typ, n, n)
		data := ts.ValueGo()
		slice = reflect.ValueOf(data)
	}
	// Now we have the data in place in the base slice we can add the
	// dimensions. We want to walk backwards through the shape. If the shape is
	// length 1 or 0 then we're already done.
	if len(shape) == 0 {
		return slice.Index(0)
	}
	if len(shape) == 1 {
		return slice
	}
	// We have a special case if the tensor has no data. Our backing slice is
	// empty, but we still want to create slices following the shape. In this
	// case only the final part of the shape will be 0 and we want to recalculate
	// n at this point ignoring that 0.
	// For example if our shape is 3 * 2 * 0 then n will be zero, but we still
	// want 6 zero length slices to group as follows.
	// {{} {}} {{} {}} {{} {}}
	if n == 0 {
		n = int(numElements(shape[:len(shape)-1]))
	}
	for i := len(shape) - 2; i >= 0; i-- {
		underlyingSize := typ.Elem().Size()
		typ = reflect.SliceOf(typ)
		subsliceLen := int(shape[i+1])
		if subsliceLen != 0 {
			n = n / subsliceLen
		}
		// Just using reflection it is difficult to avoid unnecessary
		// allocations while setting up the sub-slices as the Slice function on
		// a slice Value allocates. So we end up doing pointer arithmetic!
		// Pointer() on a slice gives us access to the data backing the slice.
		// We insert slice headers directly into this data.
		data := unsafe.Pointer(slice.Pointer())
		nextSlice := reflect.MakeSlice(typ, n, n)
		for j := 0; j < n; j++ {
			// This is equivalent to nSlice[j] = slice[j*subsliceLen: (j+1)*subsliceLen]
			setSliceInSlice(nextSlice, j, sliceHeader{
				Data: unsafe.Pointer(uintptr(data) + (uintptr(j*subsliceLen) * underlyingSize)),
				Len:  subsliceLen,
				Cap:  subsliceLen,
			})
		}

		fmt.Printf("nextSlice length: %v\n", nextSlice.Len())
		fmt.Printf("%v\n\n", nextSlice)

		slice = nextSlice
	}
	return slice
}

// setSliceInSlice sets slice[index] = content.
func setSliceInSlice(slice reflect.Value, index int, content sliceHeader) {
	const sliceSize = unsafe.Sizeof(sliceHeader{})
	// We must cast slice.Pointer to uninptr & back again to avoid GC issues.
	// See https://github.com/google/go-cmp/issues/167#issuecomment-546093202
	*(*sliceHeader)(unsafe.Pointer(uintptr(unsafe.Pointer(slice.Pointer())) + (uintptr(index) * sliceSize))) = content
}
func numElements(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

// It isn't safe to use reflect.SliceHeader as it uses a uintptr for Data and
// this is not inspected by the garbage collector
type sliceHeader struct {
	Data unsafe.Pointer
	Len  int
	Cap  int
}

// Format implements fmt.Formatter interface so that we can use
// fmt.Print... and verbs to print out Tensor value in different formats.
func (ts *Tensor) Format(s fmt.State, c rune) {
	shape := ts.MustSize()
	device := ts.MustDevice()
	dtype := ts.DType()
	if c == 'i' {
		fmt.Fprintf(s, "\nTENSOR INFO:\n\tShape:\t\t%v\n\tDType:\t\t%v\n\tDevice:\t\t%v\n\tDefined:\t%v\n", shape, dtype, device, ts.MustDefined())
		return
	}

	data := ts.ValueGo()

	f := newFmtState(s, c, shape)
	f.setWidth(data)
	f.makePad()

	// 0d (scalar)
	if len(shape) == 0 {
		fmt.Printf("%v", data)
	}

	// 1d (slice)
	if len(shape) == 1 {
		f.writeSlice(data)
		return
	}

	// 2d (matrix)
	if len(shape) == 2 {
		f.writeMatrix(data, shape)
		return
	}

	// >= 3d (tensor)
	mSize := int(shape[len(shape)-2] * shape[len(shape)-1])
	mShape := shape[len(shape)-2:]
	dims := shape[:len(shape)-2]
	var rdims []int64
	for d := len(dims) - 1; d >= 0; d-- {
		rdims = append(rdims, dims[d])
	}

	f.writeTensor(0, rdims, 0, mSize, mShape, data, "")
}

// fmtState is a struct that implements fmt.State interface
type fmtState struct {
	fmt.State
	c     rune   // format verb
	pad   []byte // padding
	w     int    // width
	p     int    // precision
	shape []int64
	buf   *bytes.Buffer
}

func newFmtState(s fmt.State, c rune, shape []int64) *fmtState {
	w, _ := s.Width()
	p, _ := s.Precision()

	return &fmtState{
		State: s,
		c:     c,
		w:     w,
		p:     p,
		shape: shape,
		buf:   bytes.NewBuffer(make([]byte, 0)),
	}
}

// writeTensor iterates recursively through a reversed shape of tensor starting from axis 3 and
// and prints out matrices (of last two dims size).
func (f *fmtState) writeTensor(d int, dims []int64, offset int, mSize int, mShape []int64, data interface{}, mName string) {

	offsetSize := product(dims[:d]) * mSize
	for i := 0; i < int(dims[d]); i++ {
		name := fmt.Sprintf("%v,%v", i+1, mName)
		if d >= len(dims)-1 { // last dim, let's print out
			// write matrix name
			nameStr := fmt.Sprintf("(%v.,.) =\n", name)
			f.Write([]byte(nameStr))
			// write matrix values
			slice := reflect.ValueOf(data).Slice(offset, offset+mSize).Interface()
			f.writeMatrix(slice, mShape)
		} else { // recursively loop
			f.writeTensor(d+1, dims, offset, mSize, mShape, data, name)
		}

		// update offset
		offset += offsetSize
	}
}

func product(dims []int64) int {
	var p int = 1
	if len(dims) == 0 {
		return p
	}

	for _, d := range dims {
		p = p * int(d)
	}

	return p
}

func (f *fmtState) writeMatrix(data interface{}, shape []int64) {
	n := shapeToSize(shape)
	dataLen := reflect.ValueOf(data).Len()
	if dataLen != n {
		log.Fatalf("mismatched: slice data has %v elements - shape: %v\n", dataLen, n)
	}
	if len(shape) != 2 {
		log.Fatal("Shape must have length of 2.\n")
	}

	stride := int(shape[1])
	currIdx := 0
	nextIdx := stride
	for i := 0; i < int(shape[0]); i++ {
		slice := reflect.ValueOf(data).Slice(currIdx, nextIdx)
		f.writeSlice(slice.Interface())
		currIdx = nextIdx
		nextIdx += stride
	}

	// 1 line between matrix
	f.Write([]byte("\n"))

}

func (f *fmtState) writeSlice(data interface{}) {
	format := f.cleanFmt()
	dataLen := reflect.ValueOf(data).Len()
	for i := 0; i < dataLen; i++ {
		el := reflect.ValueOf(data).Index(i).Interface()

		// TODO: more format options here
		w, _ := fmt.Fprintf(f.buf, format, el)
		f.Write(f.buf.Bytes())
		f.Write(f.pad[:f.w-w]) // prepad
		f.Write(f.pad[:2])     // pad
		f.buf.Reset()
	}

	f.Write([]byte("\n"))
}

func (f *fmtState) cleanFmt() string {
	buf := bytes.NewBuffer(fmtByte)

	// width
	if w, ok := f.Width(); ok {
		buf.WriteString(strconv.Itoa(w))
	}

	// precision
	if p, ok := f.Precision(); ok {
		buf.Write(precByte)
		buf.WriteString(strconv.Itoa(p))
	}

	buf.WriteRune(f.c)
	return buf.String()
}

func (f *fmtState) makePad() {
	f.pad = make([]byte, maxInt(f.w, 4))
	for i := range f.pad {
		f.pad[i] = ' '
	}
}

// setWidth determines maximal width from input data and set to `w` field
func (f *fmtState) setWidth(data interface{}) {
	format := f.cleanFmt()
	f.w = 0
	for i := 0; i < reflect.ValueOf(data).Len(); i++ {
		el := reflect.ValueOf(data).Index(i).Interface()
		w, _ := fmt.Fprintf(f.buf, format, el)
		if w > f.w {
			f.w = w
		}
		f.buf.Reset()
	}
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

func maxInt(a, b int) int {
	if a >= b {
		return a
	}
	return b
}
