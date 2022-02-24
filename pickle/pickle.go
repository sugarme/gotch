package pickle

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/big"
	"os"
	"strconv"
	"strings"
)

// This file implements Python Pickle Machinery.
//
// Pickle creates portable serialized representations of Python objects.
// See module `copyreg` for a mechanism for registering custom picklers
// See module `pickletools` source for extensive comments
// Ref. https://github.com/python/cpython/blob/main/Lib/pickle.py

// See more:...
// https://docs.python.org/3/library/pickle.html
// https://docs.python.org/3/library/pickletools.html
// https://pytorch.org/tutorials/beginner/saving_loading_models.html
// https://github.com/pytorch/pytorch/blob/master/torch/serialization.py

// Pikle Version:
// ==============
// FormatVersion = "4.0"
// CompatibleFormats [
// 	"1.0": Original protocol 0
// 	"1.1": Protocol 0 with INST added
// 	"1.2": Original protocol 1
// 	"1.3": Protocol 1 with BINFLOAT added
// 	"2.0": Protocol 2
// 	"3.0": Protocol 3
// 	"4.0": Protocol 4
//	"5.0": Protocol 5
// ]

const HighestProtocol byte = 5 // The highest protocol number pickle currently knows how to read
var DefaultProtocol byte = 4   // The protocol pickle currently used to write by default.

// Error formatter:
// ================
func pickleError(msg string) error {
	err := fmt.Errorf("PicklingError: %s", msg)
	return err
}

func picklingError(msg string) error {
	err := fmt.Errorf("Unpickable Object: %s", msg)
	return err
}

func unpicklingError(msg string) error {
	err := fmt.Errorf("UnpicklingError: %s", msg)
	return err
}

// Stop implements error interface. It is raised by `Unpickler.LoadStop()`
// in response to the STOP opcode, passing the object that is the result of unpickling.
type Stop struct {
	value interface{} // TODO. specific type
}

func newStop(value interface{}) Stop { return Stop{value} }
func (s Stop) Error() string         { return "STOP" }

var _ error = Stop{}

// Pickle opcodes:
// ==============
// See pickletools.py for extensive docs.

var (
	MARK            rune = '(' // push special markobject on stack
	STOP            rune = '.' // every pickle ends with STOP
	POP             rune = '0' // discard topmost stack item
	POP_MARK        rune = '1' // discard stack top through topmost markobject
	DUP             rune = '2' // duplicate top stack item
	FLOAT           rune = 'F' // push float object; decimal string argument
	INT             rune = 'I' // push integer or bool; decimal string argument
	BININT          rune = 'J' // push four-byte signed int
	BININT1         rune = 'K' // push 1-byte unsigned int
	LONG            rune = 'L' // push long; decimal string argument
	BININT2         rune = 'M' // push 2-byte unsigned int
	NONE            rune = 'N' // push None
	PERSID          rune = 'P' // push persistent object; id is taken from string arg
	BINPERSID       rune = 'Q' //  "       "         "  ;  "  "   "     "  stack
	REDUCE          rune = 'R' // apply callable to argtuple, both on stack
	STRING          rune = 'S' // push string; NL-terminated string argument
	BINSTRING       rune = 'T' // push string; counted binary string argument
	SHORT_BINSTRING rune = 'U' //  "     "   ;    "      "       "      " < 256 bytes
	UNICODE         rune = 'V' // push Unicode string; raw-unicode-escaped'd argument
	BINUNICODE      rune = 'X' //   "     "       "  ; counted UTF-8 string argument
	APPEND          rune = 'a' // append stack top to list below it
	BUILD           rune = 'b' // call __setstate__ or __dict__.update()
	GLOBAL          rune = 'c' // push self.find_class(modname, name); 2 string args
	DICT            rune = 'd' // build a dict from stack items
	EMPTY_DICT      rune = '}' // push empty dict
	APPENDS         rune = 'e' // extend list on stack by topmost stack slice
	GET             rune = 'g' // push item from memo on stack; index is string arg
	BINGET          rune = 'h' //   "    "    "    "   "   "  ;   "    " 1-byte arg
	INST            rune = 'i' // build & push class instance
	LONG_BINGET     rune = 'j' // push item from memo on stack; index is 4-byte arg
	LIST            rune = 'l' // build list from topmost stack items
	EMPTY_LIST      rune = ']' // push empty list
	OBJ             rune = 'o' // build & push class instance
	PUT             rune = 'p' // store stack top in memo; index is string arg
	BINPUT          rune = 'q' //   "     "    "   "   " ;   "    " 1-byte arg
	LONG_BINPUT     rune = 'r' //   "     "    "   "   " ;   "    " 4-byte arg
	SETITEM         rune = 's' // add key+value pair to dict
	TUPLE           rune = 't' // build tuple from topmost stack items
	EMPTY_TUPLE     rune = ')' // push empty tuple
	SETITEMS        rune = 'u' // modify dict by adding topmost key+value pairs
	BINFLOAT        rune = 'G' // push float; arg is 8-byte float encoding

	// TRUE            rune = 'I01\n'  // not an opcode; see INT docs in pickletools.py
	// FALSE           rune = 'I00\n'  // not an opcode; see INT docs in pickletools.py

	// Protocol 2

	PROTO    rune = '\x80' // identify pickle protocol
	NEWOBJ   rune = '\x81' // build object by applying cls.__new__ to argtuple
	EXT1     rune = '\x82' // push object from extension registry; 1-byte index
	EXT2     rune = '\x83' // ditto, but 2-byte index
	EXT4     rune = '\x84' // ditto, but 4-byte index
	TUPLE1   rune = '\x85' // build 1-tuple from stack top
	TUPLE2   rune = '\x86' // build 2-tuple from two topmost stack items
	TUPLE3   rune = '\x87' // build 3-tuple from three topmost stack items
	NEWTRUE  rune = '\x88' // push True
	NEWFALSE rune = '\x89' // push False
	LONG1    rune = '\x8a' // push long from < 256 bytes
	LONG4    rune = '\x8b' // push really big long

	tuplesize2code []rune = []rune{EMPTY_TUPLE, TUPLE1, TUPLE2, TUPLE3}

	// Protocol 3 (Python 3.x)

	BINBYTES       rune = 'B' // push bytes; counted binary string argument
	SHORT_BINBYTES rune = 'C' //  "     "   ;    "      "       "      " < 256 bytes

	// Protocol 4

	SHORT_BINUNICODE rune = '\x8c' // push short string; UTF-8 length < 256 bytes
	BINUNICODE8      rune = '\x8d' // push very long string
	BINBYTES8        rune = '\x8e' // push very long bytes string
	EMPTY_SET        rune = '\x8f' // push empty set on the stack
	ADDITEMS         rune = '\x90' // modify set by adding topmost stack items
	FROZENSET        rune = '\x91' // build frozenset from topmost stack items
	NEWOBJ_EX        rune = '\x92' // like NEWOBJ but work with keyword only arguments
	STACK_GLOBAL     rune = '\x93' // same as GLOBAL but using names on the stacks
	MEMOIZE          rune = '\x94' // store top of the stack in memo
	FRAME            rune = '\x95' // indicate the beginning of a new frame

	// Protocol 5

	BYTEARRAY8      rune = '\x96' // push bytearray
	NEXT_BUFFER     rune = '\x97' // push next out-of-band buffer
	READONLY_BUFFER rune = '\x98' // make top of stack readonly
)

// Unpickling Machinery:
// =====================

type Unpickler struct {
	proto        byte            // protocol version of the pickle
	reader       io.Reader       // binary file reader
	currentFrame *bytes.Reader   // buffer frame reader
	stack        []interface{}   // keeps marked objects
	metaStack    [][]interface{} // keeps stacks of marked objects

	// data structure that remembers which objects the pickler/unpickler has already seen
	// so that shared or recursive objects are pickled/unpickled by reference and not by value
	// This property is useful when re-using picklers/unpicklers.
	memo map[int]interface{}

	FindClass      func(module, name string) (interface{}, error) // function to determine data type
	PersistentLoad func(interface{}) (interface{}, error)         // function how to load pickled objects by its id.

	GetExtension     func(code int) (interface{}, error)
	NextBufferFunc   func() (interface{}, error)
	MakeReadOnlyFunc func(interface{}) (interface{}, error)
}

// NewUnpickler creates a new Unpickler.
func NewUnpickler(r io.Reader) Unpickler {
	return Unpickler{
		reader: r,
		memo:   make(map[int]interface{}, 0),
	}
}

// read reads n bytes from reader.
func (up *Unpickler) read(n int) ([]byte, error) {
	data := make([]byte, n)
	if up.currentFrame != nil {
		nbytes, err := io.ReadFull(up.currentFrame, data)

		switch {
		case err != nil && err != io.EOF && err != io.ErrUnexpectedEOF:
			return nil, err

		case nbytes == 0 && n != 0: // remaining data
			up.currentFrame = nil
			nbytes, err := io.ReadFull(up.reader, data)
			return data[0:nbytes], err

		case nbytes < n:
			err := fmt.Errorf("Unpickler.read() failed: pickle exhausted before end of frame")
			return nil, err

		default:
			return data[0:nbytes], nil
		}
	}

	nbytes, err := io.ReadFull(up.reader, data)
	return data[0:nbytes], err
}

// readOne reads 1 byte.
func (up *Unpickler) readOne() (byte, error) {
	data, err := up.read(1)
	if err != nil {
		return 0, err
	}

	return data[0], nil
}

// readLine reads one line of data.
func (up *Unpickler) readLine() ([]byte, error) {
	if up.currentFrame != nil {
		line, err := readLine(up.currentFrame)
		if err != nil {
			if err == io.EOF && len(line) == 0 {
				up.currentFrame = nil
				return readLine(up.reader)
			}

			return nil, err
		}

		if len(line) == 0 {
			err := fmt.Errorf("Unpickler.readLine() failed: no data.")
			return nil, err
		}
		if line[len(line)-1] != '\n' {
			err := fmt.Errorf("Unpickler.readLine() failed: pickle exhausted before end of frame.")
			return nil, err
		}

		return line, nil
	}

	return readLine(up.reader)
}

// readLine reads one line of data. Line ends by '\n' byte.
func readLine(r io.Reader) ([]byte, error) {
	bufferSize := 64 // just set buffer line = 64. One might change it.
	line := make([]byte, 0, bufferSize)

	buf := make([]byte, 1)
	for {
		nbytes, err := r.Read(buf)

		if nbytes != 1 {
			return line, err
		}

		line = append(line, buf[0])
		if buf[0] == '\n' || err != nil {
			return line, err
		}
	}
}

// loadFrame loads new data to currentFrame. It throws error if currentFrame is not empty.
func (up *Unpickler) loadFrame(frameSize int) error {
	buf := make([]byte, frameSize)
	// Throw error if current frame is not empty
	if up.currentFrame != nil {
		nbytes, err := up.currentFrame.Read(buf)
		if nbytes > 0 || err == nil {
			err := unpicklingError("beginning of a new frame before end of a current frame")
			return err
		}
	}

	// now, load data to currentFrame
	_, err := io.ReadFull(up.reader, buf)
	if err != nil {
		return err
	}
	up.currentFrame = bytes.NewReader(buf)

	return nil
}

// append appends an object to stack.
func (up *Unpickler) append(obj interface{}) {
	up.stack = append(up.stack, obj)
}

// stackPop pops an object out of stack.
func (up *Unpickler) stackPop() (interface{}, error) {
	obj, err := up.stackLast()
	if err != nil {
		return nil, err
	}

	up.stack = up.stack[:len(up.stack)-1]

	return obj, nil
}

// stackLast get last object in stack.
func (up *Unpickler) stackLast() (interface{}, error) {
	if len(up.stack) == 0 {
		err := fmt.Errorf("Unpickler.stackLast() failed: stack is empty.")
		return nil, err
	}

	last := up.stack[len(up.stack)-1]

	return last, nil
}

// metaStackPop pop a stack out from metaStack.
func (up *Unpickler) metaStackPop() ([]interface{}, error) {
	stack, err := up.metaStackLast()
	if err != nil {
		return nil, err
	}

	up.metaStack = up.metaStack[:len(up.metaStack)-1]

	return stack, nil
}

// metaStackLast get last stack in metaStack.
func (up *Unpickler) metaStackLast() ([]interface{}, error) {
	if len(up.metaStack) == 0 {
		err := fmt.Errorf("Unpickler.metaStackLast() failed: metaStack is empty.")
		return nil, err
	}

	last := up.metaStack[len(up.metaStack)-1]

	return last, nil
}

// popMark pops all objects those have been pushed to the stack (after last mask).
func (up *Unpickler) popMark() ([]interface{}, error) {
	objects := up.stack
	newStack, err := up.metaStackPop()
	if err != nil {
		return nil, err
	}
	up.stack = newStack

	return objects, nil
}

func (up *Unpickler) findClass(module, name string) (interface{}, error) {
	switch module {
	case "collections":
		switch name {
		case "OrderedDict":
			return &OrderedDictClass{}, nil
		}

	case "__builtin__":
		switch name {
		case "object":
			return &ObjectClass{}, nil
		}
	case "copy_reg":
		switch name {
		case "_reconstructor":
			return &Reconstructor{}, nil
		}
	}
	if up.FindClass != nil {
		return up.FindClass(module, name)
	}
	return NewGenericClass(module, name), nil
}

func (up *Unpickler) persistentLoad(pid interface{}) error {
	err := unpicklingError("Unpickler.persistentLoad() failed: unsupported persistent id encountered.")
	return err
}

// Construct dispatch table:
// =========================
// See https://en.wikipedia.org/wiki/Dispatch_table
// dispatch table is a table of pointers to functions/methods.

// unpickle dispatch table
var upDispatch [math.MaxUint8]func(*Unpickler) error

// loadProto reads pickle protocol version.
func loadProto(up *Unpickler) error {
	proto, err := up.readOne()
	if err != nil {
		return err
	}
	if proto < 0 || proto >= HighestProtocol {
		err := fmt.Errorf("loadProto() failed: unsupported pickle protocol (%d)", proto)
		return err
	}

	up.proto = proto

	return nil
}

// loadFrame loads new frame.
func loadFrame(up *Unpickler) error {
	buf, err := up.read(8)
	if err != nil {
		return err
	}
	frameSize := binary.LittleEndian.Uint64(buf)
	if frameSize > math.MaxUint64 {
		err := fmt.Errorf("loadFrame() failed: frame size > sys.maxsize %v", frameSize)
		return err
	}

	return up.loadFrame(int(frameSize))
}

// loadPersIds load persistent object to stack.
func loadPersId(up *Unpickler) error {
	if up.PersistentLoad == nil {
		err := fmt.Errorf("loadPersId() failed: unsupported persistent Id encountered.")
		return err
	}

	line, err := up.readLine()
	if err != nil {
		return err
	}

	pid := string(line[:len(line)-1])
	obj, err := up.PersistentLoad(pid)
	if err != nil {
		err = fmt.Errorf("loadPersId() failed: %w", err)
		return err
	}

	up.append(obj)

	return nil
}

func loadBinPersId(up *Unpickler) error {
	if up.PersistentLoad == nil {
		err := fmt.Errorf("loadPersId() failed: unsupported persistent Id encountered.")
		return err
	}

	pid, err := up.stackPop()
	if err != nil {
		return err
	}

	obj, err := up.PersistentLoad(pid)
	if err != nil {
		err = fmt.Errorf("loadBinPersId() failed: %w", err)
		return err
	}

	up.append(obj)

	return nil
}

// loads nil object
func loadNone(up *Unpickler) error {
	up.append(nil)
	return nil
}

// loads a bool object with value = false.
func loadFalse(up *Unpickler) error {
	up.append(false)
	return nil
}

// loads a bool object with value = true
func loadTrue(up *Unpickler) error {
	up.append(true)
	return nil
}

// loads object of type int (can be integer, bool or decimal string value)
func loadInt(up *Unpickler) error {
	line, err := up.readLine()
	if err != nil {
		err = fmt.Errorf("loadInt() failed: %w", err)
		return err
	}

	data := string(line[:len(line)-1])
	switch {
	case len(data) == 2 && data[0] == '0' && data[1] == '0':
		up.append(false)
		return nil

	case len(data) == 2 && data[0] == '0' && data[1] == '1':
		up.append(true)
		return nil

	default:
		val, err := strconv.Atoi(data)
		if err != nil {
			err = fmt.Errorf("loadInt() failed: %w", err)
		}
		up.append(val)
		return nil
	}
}

// load 4 bytes of uint.
func loadBinInt(up *Unpickler) error {
	buf, err := up.read(4)
	if err != nil {
		err = fmt.Errorf("loadBinInt() failed: %w", err)
		return err
	}

	uval := binary.LittleEndian.Uint32(buf)
	val := int(uval)
	if buf[3]&0x80 != 0 {
		val = -(int(^uval) + 1)
	}
	up.append(val)

	return nil
}

// loads one byte of uint.
func loadBinInt1(up *Unpickler) error {
	b, err := up.readOne()
	if err != nil {
		err = fmt.Errorf("loadBinInt1() failed: %w", err)
		return err
	}
	up.append(int(b))

	return nil
}

// loads 2 bytes of uint.
func loadBinInt2(up *Unpickler) error {
	buf, err := up.read(2)
	if err != nil {
		err = fmt.Errorf("loadBinInt2() failed: %w", err)
		return err
	}

	val := int(binary.LittleEndian.Uint16(buf))
	up.append(val)

	return nil
}

// load long; decimal string argument.
func loadLong(up *Unpickler) error {
	line, err := up.readLine()
	if err != nil {
		err = fmt.Errorf("loadLong() failed: %w", err)
		return nil
	}

	// last byte is string dtype.
	if len(line) == 1 {
		err = fmt.Errorf("loadLong() failed: invalid long data")
	}
	data := line[:len(line)-1]
	if data[len(data)-1] == 'L' {
		data = data[0 : len(data)-1]
	}

	val, err := strconv.ParseInt(string(data), 10, 64)
	if err != nil {
		// check for overflow, if so, swap to larger range.
		if numErr, ok := err.(*strconv.NumError); ok && numErr.Err == strconv.ErrRange {
			bigInt, ok := new(big.Int).SetString(string(data), 10)
			if !ok {
				err = fmt.Errorf("loadLong() failed: invalid long data")
				return err
			}

			up.append(bigInt)
			return nil
		}

		err = fmt.Errorf("loadLong() failed: %w", err)
		return err
	}
	up.append(int(val))

	return nil
}

// loads long interger of less than 256 bytes.
func loadLong1(up *Unpickler) error {
	len, err := up.readOne()
	if err != nil {
		err = fmt.Errorf("loadLong1() failed: %w", err)
		return err
	}

	buf, err := up.read(int(len))
	if err != nil {
		err = fmt.Errorf("loadLong1() failed: %w", err)
		return err
	}

	val := decodeLong(buf)
	up.append(val)

	return nil
}

// loads object of really big long integer.
func loadLong4(up *Unpickler) error {
	buf, err := up.read(4)
	if err != nil {
		err = fmt.Errorf("loadLong4() failed: %w", err)
		return err
	}

	len := decodeInt32(buf)
	if len < 0 {
		err = fmt.Errorf("loadLong4() failed: LONG pickle has negative byte count")
	}
	data, err := up.read(len)
	if err != nil {
		err = fmt.Errorf("loadLong4() failed: %w", err)
		return err
	}
	val := decodeLong(data)
	up.append(val)

	return nil
}

// loads float object or decimal string argument.
func loadFloat(up *Unpickler) error {
	line, err := up.readLine()
	if err != nil {
		err = fmt.Errorf("loadFloat() failed: %w", err)
		return err
	}

	val, err := strconv.ParseFloat(string(line[:len(line)-1]), 64)
	if err != nil {
		err = fmt.Errorf("loadFloat() failed: %w", err)
		return err
	}
	up.append(val)

	return nil
}

// loads float object of 8-byte encoding.
func loadBinFloat(up *Unpickler) error {
	buf, err := up.read(8)
	if err != nil {
		err = fmt.Errorf("loadBinFloat() failed: %w", err)
		return err
	}

	val := math.Float64frombits(binary.BigEndian.Uint64(buf))
	up.append(val)

	return nil
}

// loads object of string value.
func loadString(up *Unpickler) error {
	line, err := up.readLine()
	if err != nil {
		err = fmt.Errorf("loadString() failed: %w", err)
		return err
	}

	data := line[:len(line)-1]

	// strip outermost quotes
	if len(data) >= 2 && data[0] == data[len(data)-1] && (data[0] == '\'' || data[0] == '"') {
		data = data[1 : len(data)-1]
	} else {
		err = unpicklingError("the STRING opcode argument must be quoted.")
		err = fmt.Errorf("loadString() failed: %w", err)
		return err
	}
	up.append(data)

	return nil
}

// loads object of counted binary string.
func loadBinString(up *Unpickler) error {
	// Deprecated BINSTRING uses signed 32-bit length
	buf, err := up.read(4)
	if err != nil {
		err = fmt.Errorf("loadBinString() failed: %w", err)
		return err
	}

	len := decodeInt32(buf)
	if len < 0 {
		err = unpicklingError("loadBinString() failed: BINSTRING pickle has negative byte count.")
		return err
	}

	data, err := up.read(len)
	if err != nil {
		err = fmt.Errorf("loadBinString() failed: %w", err)
		return err
	}

	val := string(data)
	up.append(val)

	return nil
}

// loads object of bytes
func loadBinBytes(up *Unpickler) error {
	buf, err := up.read(4)
	if err != nil {
		err := fmt.Errorf("loadBinBytes() failed: %w", err)
		return err
	}

	len := int(binary.LittleEndian.Uint32(buf))
	buf, err = up.read(len)
	if err != nil {
		err := fmt.Errorf("loadBinBytes() failed: %w", err)
		return err
	}
	up.append(buf)

	return nil
}

// loads object of Unicode string value (raw-unicode-escaped).
func loadUnicode(up *Unpickler) error {
	line, err := up.readLine()
	if err != nil {
		err := fmt.Errorf("loadUnicode() failed: %w", err)
		return err
	}
	val := string(line[:len(line)-1])
	up.append(val)

	return nil
}

// loads objects of Unicode string (counted UTF-8 string)
func loadBinUnicode(up *Unpickler) error {
	buf, err := up.read(4)
	if err != nil {
		err = fmt.Errorf("loadBinUnicode() failed: %w", err)
		return err
	}

	len := int(binary.LittleEndian.Uint32(buf))
	buf, err = up.read(len)
	if err != nil {
		err = fmt.Errorf("loadBinUnicode() failed: %w", err)
		return err
	}
	val := string(buf)
	up.append(val)

	return nil
}

// loads a object of very long string value.
func loadBinUnicode8(up *Unpickler) error {
	buf, err := up.read(8)
	if err != nil {
		err = fmt.Errorf("loadBinUnicode8() failed: %w", err)
		return err
	}

	len := int(binary.LittleEndian.Uint64(buf))
	if len > math.MaxInt64 {
		err = unpicklingError("loadBinUnicode8() failed: BINUNICODE8 exceeds system's maximum size")
		return err
	}
	buf, err = up.read(len)
	if err != nil {
		err = fmt.Errorf("loadBinUnicode8() failed: %w", err)
		return err
	}
	val := string(buf)
	up.append(val)

	return nil
}

// loads object of very long bytes string value.
func loadBinBytes8(up *Unpickler) error {
	buf, err := up.read(8)
	if err != nil {
		err = fmt.Errorf("loadBinBytes8() failed: %w", err)
		return err
	}

	len := binary.LittleEndian.Uint64(buf)
	if len > math.MaxInt64 {
		err = unpicklingError("loadBinBytes8() failed: BINBYTES8 exceeds system's maximum size")
		return err
	}
	buf, err = up.read(int(len))
	if err != nil {
		err = fmt.Errorf("loadBinBytes8() failed: %w", err)
		return err
	}
	up.append(buf)

	return nil
}

func loadByteArray8(up *Unpickler) error {
	buf, err := up.read(8)
	if err != nil {
		err = fmt.Errorf("loadBinBytes8() failed: %w", err)
		return err
	}

	len := binary.LittleEndian.Uint64(buf)
	if len > math.MaxInt64 {
		err = unpicklingError("loadBinBytes8() failed: BINBYTES8 exceeds system's maximum size.")
		return err
	}
	buf, err = up.read(int(len))
	if err != nil {
		err = fmt.Errorf("loadBinBytes8() failed: %w", err)
		return err
	}

	val := NewByteArrayFromSlice(buf)
	up.append(val)

	return nil
}

// loads next out-of-band buffer.
func loadNextBuffer(up *Unpickler) error {
	if up.NextBufferFunc == nil {
		err := fmt.Errorf("loadNextBuffer() failed: Pickle stream refers to out-of-band data but NextBufferFunc was not given")
		return err
	}

	buf, err := up.NextBufferFunc()
	if err != nil {
		err = fmt.Errorf("loadNextBuffer() failed: %w", err)
		return err
	}

	up.append(buf)

	return nil
}

// makes top of stack readonly.
func loadReadOnlyBuffer(up *Unpickler) error {
	if up.MakeReadOnlyFunc == nil {
		return nil
	}

	buf, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadReadOnlyBuffer() failed: %w", err)
		return err
	}

	buf, err = up.MakeReadOnlyFunc(buf)
	if err != nil {
		err = fmt.Errorf("loadReadOnlyBuffer() failed: %w", err)
		return err
	}
	up.append(buf)

	return nil
}

// loads counted binary string object (< 256 bytes).
func loadShortBinString(up *Unpickler) error {
	len, err := up.readOne()
	if err != nil {
		err = fmt.Errorf("loadShortBinString() failed: %w", err)
		return err
	}
	data, err := up.read(int(len))
	if err != nil {
		err = fmt.Errorf("loadShortBinString() failed: %w", err)
		return err
	}
	up.append(string(data))

	return nil
}

// loads bytes object with counted binary string < 256 bytes.
func loadShortBinBytes(up *Unpickler) error {
	len, err := up.readOne()
	if err != nil {
		err = fmt.Errorf("loadShortBinBytes() failed: %w", err)
		return err
	}
	buf, err := up.read(int(len))
	if err != nil {
		err = fmt.Errorf("loadShortBinBytes() failed: %w", err)
		return err
	}
	up.append(buf)

	return nil
}

// load short string object; UTF-8 length < 256 bytes.
func loadShortBinUnicode(up *Unpickler) error {
	len, err := up.readOne()
	if err != nil {
		err = fmt.Errorf("loadShortBinUnicode() failed: %w", err)
		return err
	}

	buf, err := up.read(int(len))
	if err != nil {
		err = fmt.Errorf("loadShortBinUnicode() failed: %w", err)
		return err
	}
	up.append(string(buf))

	return nil
}

// loads tuple from last-mark stack objects.
func loadTuple(up *Unpickler) error {
	objects, err := up.popMark()
	if err != nil {
		err = fmt.Errorf("loadTuple() failed: %w", err)
		return err
	}
	val := NewTupleFromSlice(objects)
	up.append(val)

	return nil
}

// load empty tuple.
func loadEmptyTuple(up *Unpickler) error {
	t := NewTupleFromSlice([]interface{}{})
	up.append(t)

	return nil
}

// load one tuple from stack top.
func loadTuple1(up *Unpickler) error {
	obj, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadTuple() failed: %w", err)
		return err
	}
	val := NewTupleFromSlice([]interface{}{obj})
	up.append(val)

	return nil
}

// load 2-tuple object from 2 topmost stack objects.
func loadTuple2(up *Unpickler) error {
	obj2, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadTuple2() failed: %w", err)
		return err
	}
	obj1, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadTuple2() failed: %w", err)
		return err
	}
	val := NewTupleFromSlice([]interface{}{obj1, obj2})
	up.append(val)

	return nil
}

// loads 3-tuple object from 3 most stack objects
func loadTuple3(up *Unpickler) error {
	objects := make([]interface{}, 3)
	var (
		err error
		n   int = 0
	)
	for i := 2; i >= 0; i-- {
		objects[n], err = up.stackPop()
		if err != nil {
			err = fmt.Errorf("loadTuple3() failed: %w", err)
		}
	}
	val := NewTupleFromSlice(objects)
	up.append(val)

	return nil
}

// loads empty list object.
func loadEmptyList(up *Unpickler) error {
	up.append(NewList())

	return nil
}

// loads empty dict.
func loadEmptyDict(up *Unpickler) error {
	up.append(NewDict())
	return nil
}

// loads empty set on the stack.
func loadEmptySet(up *Unpickler) error {
	up.append(NewSet())
	return nil
}

// loads frozenset from topmost stack objects.
func loadFrozenSet(up *Unpickler) error {
	objects, err := up.popMark()
	if err != nil {
		err = fmt.Errorf("loadFrozenSet() failed: %w", err)
		return err
	}
	up.append(NewFrozenSetFromSlice(objects))
	return nil
}

// loads list from topmost stack objects
func loadList(up *Unpickler) error {
	objects, err := up.popMark()
	if err != nil {
		err = fmt.Errorf("loadList() failed: %w", err)
		return err
	}
	up.append(NewListFromSlice(objects))
	return nil
}

// loads a dict from stack objects
func loadDict(up *Unpickler) error {
	objects, err := up.popMark()
	if err != nil {
		err = fmt.Errorf("loadDict() failed: %w", err)
		return err
	}
	d := NewDict()
	objectsLen := len(objects)
	for i := 0; i < objectsLen; i += 2 {
		d.Set(objects[i], objects[i+1])
	}
	up.append(d)
	return nil
}

// loads class instance.
func loadInst(up *Unpickler) error {
	line, err := up.readLine()
	if err != nil {
		err = fmt.Errorf("loadInst() failed: %w", err)
		return err
	}
	module := string(line[0 : len(line)-1])

	line, err = up.readLine()
	if err != nil {
		err = fmt.Errorf("loadInst() failed: %w", err)
		return err
	}
	name := string(line[0 : len(line)-1])

	class, err := up.findClass(module, name)
	if err != nil {
		err = fmt.Errorf("loadInst() failed: %w", err)
		return err
	}

	args, err := up.popMark()
	if err != nil {
		err = fmt.Errorf("loadInst() failed: %w", err)
		return err
	}

	return up.instantiate(class, args)
}

// loads class instance
func loadObj(up *Unpickler) error {
	// Stack is ... markobject classobject arg1 arg2 ...
	args, err := up.popMark()
	if err != nil {
		err = fmt.Errorf("loadObj() failed: %w", err)
		return err
	}
	if len(args) == 0 {
		return fmt.Errorf("OBJ class missing")
	}
	class := args[0]
	args = args[1:len(args)]
	return up.instantiate(class, args)
}

// instantiates a object based on input dtype and arguments.
func (up *Unpickler) instantiate(class interface{}, args []interface{}) error {
	var err error
	var value interface{}
	switch ct := class.(type) {
	case Callable:
		value, err = ct.Call(args...)
	case PyNewable:
		value, err = ct.PyNew(args...)
	default:
		return fmt.Errorf("cannot instantiate %#v", class)
	}

	if err != nil {
		err = fmt.Errorf("instantiate() failed: %w", err)
		return err
	}
	up.append(value)
	return nil
}

// loads object by applying cls.__new__ to argtuple
func loadNewObj(up *Unpickler) error {
	args, err := up.stackPop()
	if err != nil {
		return err
	}
	argsTuple, argsOk := args.(*Tuple)
	if !argsOk {
		err := fmt.Errorf("NEWOBJ args must be *Tuple")
		err = fmt.Errorf("loadNewObj() failed: %w", err)
		return err
	}

	rawClass, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadNewObj() failed: %w", err)
		return err
	}
	class, classOk := rawClass.(PyNewable)
	if !classOk {
		err := fmt.Errorf("NEWOBJ requires a PyNewable object: %#v", rawClass)
		err = fmt.Errorf("loadNewObj() failed: %w", err)
		return err
	}

	result, err := class.PyNew(*argsTuple...)
	if err != nil {
		return err
	}
	up.append(result)
	return nil
}

// like NEWOBJ but work with keyword only arguments
func loadNewObjEx(up *Unpickler) error {
	kwargs, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadNewObjEx() failed: %w", err)
		return err
	}

	args, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadNewObjEx() failed: %w", err)
		return err
	}
	argsTuple, argsOk := args.(*Tuple)
	if !argsOk {
		err := fmt.Errorf("NEWOBJ_EX args must be *Tuple")
		err = fmt.Errorf("loadNewObjEx() failed: %w", err)
		return err
	}

	rawClass, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadNewObjEx() failed: %w", err)
		return err
	}
	class, classOk := rawClass.(PyNewable)
	if !classOk {
		err := fmt.Errorf("NEWOBJ_EX requires a PyNewable object")
		err = fmt.Errorf("loadNewObjEx() failed: %w", err)
		return err
	}

	allArgs := []interface{}(*argsTuple)
	allArgs = append(allArgs, kwargs)

	result, err := class.PyNew(allArgs...)
	if err != nil {
		err = fmt.Errorf("loadNewObjEx() failed: %w", err)
		return err
	}
	up.append(result)
	return nil
}

// loads 'self.find_class(module, name)'; 2 string args.
// It decodes "module" and "name" of the object from binary file
// and find object class, then push to stack.
//
// NOTE. Pytorch rebuilds tensor (legacy) triggers from here
// with: module "torch._utils" - name "_rebuild_tensor" or "_rebuild_tensor_v2"
// rebuild tensor based on '_rebuild_tensor_v2' hook may break in the future.
// ref. https://github.com/pytorch/pytorch/blob/c2255c36ec121fdb998ce3db8deb7508c814b567/torch/_utils.py#L138
func loadGlobal(up *Unpickler) error {
	line, err := up.readLine()
	if err != nil {
		err = fmt.Errorf("loadGlobal() failed: %w", err)
		return err
	}
	module := string(line[0 : len(line)-1])

	line, err = up.readLine()
	if err != nil {
		err = fmt.Errorf("loadGlobal() failed: %w", err)
		return err
	}
	name := string(line[0 : len(line)-1])

	class, err := up.findClass(module, name)
	if err != nil {
		err = fmt.Errorf("loadGlobal() failed: %w", err)
		return err
	}
	up.append(class)
	return nil
}

// same as GLOBAL but using names on the stacks
func loadStackGlobal(up *Unpickler) error {
	rawName, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadStackGlobal() failed: %w", err)
		return err
	}
	name, nameOk := rawName.(string)
	if !nameOk {
		err := fmt.Errorf("STACK_GLOBAL requires str name: %#v", rawName)
		err = fmt.Errorf("loadStackGlobal() failed: %w", err)
		return err
	}

	rawModule, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadStackGlobal() failed: %w", err)
		return err
	}
	module, moduleOk := rawModule.(string)
	if !moduleOk {
		err := fmt.Errorf("STACK_GLOBAL requires str module: %#v", rawModule)
		err = fmt.Errorf("loadStackGlobal() failed: %w", err)
		return err
	}

	class, err := up.findClass(module, name)
	if err != nil {
		err = fmt.Errorf("loadStackGlobal() failed: %w", err)
		return err
	}
	up.append(class)

	return nil
}

// loads object from extension registry; 1-byte index
func opExt1(up *Unpickler) error {
	if up.GetExtension == nil {
		err := fmt.Errorf("unsupported extension code encountered")
		err = fmt.Errorf("loadStackGlobal() failed: %w", err)
		return err
	}
	i, err := up.readOne()
	if err != nil {
		err = fmt.Errorf("loadStackGlobal() failed: %w", err)
		return err
	}
	obj, err := up.GetExtension(int(i))
	if err != nil {
		err = fmt.Errorf("loadStackGlobal() failed: %w", err)
		return err
	}
	up.append(obj)

	return nil
}

// ditto, but 2-byte index
func opExt2(up *Unpickler) error {
	if up.GetExtension == nil {
		err := fmt.Errorf("unsupported extension code encountered")
		err = fmt.Errorf("opExt2() failed: %w", err)
		return err
	}
	buf, err := up.read(2)
	if err != nil {
		err = fmt.Errorf("opExt2() failed: %w", err)
		return err
	}
	code := int(binary.LittleEndian.Uint16(buf))
	obj, err := up.GetExtension(code)
	if err != nil {
		err = fmt.Errorf("opExt2() failed: %w", err)
		return err
	}
	up.append(obj)

	return nil
}

// ditto, but 4-byte index
func opExt4(up *Unpickler) error {
	if up.GetExtension == nil {
		err := fmt.Errorf("unsupported extension code encountered")
		err = fmt.Errorf("opExt4() failed: %w", err)
		return err
	}
	buf, err := up.read(4)
	if err != nil {
		err = fmt.Errorf("opExt4() failed: %w", err)
		return err
	}
	code := int(binary.LittleEndian.Uint32(buf))
	obj, err := up.GetExtension(code)
	if err != nil {
		err = fmt.Errorf("opExt4() failed: %w", err)
		return err
	}
	up.append(obj)

	return nil
}

// apply callable to argtuple, both on stack
func loadReduce(up *Unpickler) error {
	args, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadReduce() failed: %w", err)
		return err
	}
	argsTuple, argsOk := args.(*Tuple)
	if !argsOk {
		err := fmt.Errorf("REDUCE args must be *Tuple")
		err = fmt.Errorf("loadReduce() failed: %w", err)
		return err
	}

	function, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadReduce() failed: %w", err)
		return err
	}
	callable, callableOk := function.(Callable)
	if !callableOk {
		err := fmt.Errorf("REDUCE requires a Callable object: %#v", function)
		err = fmt.Errorf("loadReduce() failed: %w", err)
		return err
	}

	result, err := callable.Call(*argsTuple...)
	if err != nil {
		err = fmt.Errorf("loadReduce() failed: %w", err)
		return err
	}
	up.append(result)

	return nil
}

// discards topmost stack item
func loadPop(up *Unpickler) error {
	if len(up.stack) == 0 {
		_, err := up.popMark()
		err = fmt.Errorf("loadPop() failed: %w", err)
		return err
	}
	up.stack = up.stack[:len(up.stack)-1]

	return nil
}

// discards stack top through topmost markobject
func loadPopMark(up *Unpickler) error {
	_, err := up.popMark()
	if err != nil {
		err = fmt.Errorf("loadPopMark() failed: %w", err)
		return err
	}

	return nil
}

// duplicate top stack item
func loadDup(up *Unpickler) error {
	item, err := up.stackLast()
	if err != nil {
		err = fmt.Errorf("loadDup() failed: %w", err)
		return err
	}
	up.append(item)

	return nil
}

// loads object from memo on stack; index is string arg
func loadGet(up *Unpickler) error {
	line, err := up.readLine()
	if err != nil {
		err = fmt.Errorf("loadGet() failed: %w", err)
		return err
	}
	i, err := strconv.Atoi(string(line[:len(line)-1]))
	if err != nil {
		err = fmt.Errorf("loadGet() failed: %w", err)
		return err
	}
	up.append(up.memo[i])

	return nil
}

// loads object from memo on stack; index is 1-byte arg
func loadBinGet(up *Unpickler) error {
	i, err := up.readOne()
	if err != nil {
		err = fmt.Errorf("loadBinGet() failed: %w", err)
		return err
	}
	up.append(up.memo[int(i)])

	return nil
}

// load object from memo on stack; index is 4-byte arg
func loadLongBinGet(up *Unpickler) error {
	buf, err := up.read(4)
	if err != nil {
		err = fmt.Errorf("loadBinGet() failed: %w", err)
		return err
	}
	i := int(binary.LittleEndian.Uint32(buf))

	up.append(up.memo[i])

	return nil
}

// store stack top in memo; index is string arg
func loadPut(up *Unpickler) error {
	line, err := up.readLine()
	if err != nil {
		err = fmt.Errorf("loadPut() failed: %w", err)
		return err
	}
	i, err := strconv.Atoi(string(line[:len(line)-1]))
	if err != nil {
		err = fmt.Errorf("loadPut() failed: %w", err)
		return err
	}
	if i < 0 {
		err := fmt.Errorf("negative PUT argument")
		err = fmt.Errorf("loadPut() failed: %w", err)
		return err
	}
	up.memo[i], err = up.stackLast()
	if err != nil {
		err = fmt.Errorf("loadPut() failed: %w", err)
		return err
	}

	return nil
}

// store stack top in memo; index is 1-byte arg
func loadBinPut(up *Unpickler) error {
	i, err := up.readOne()
	if err != nil {
		err = fmt.Errorf("loadBinPut() failed: %w", err)
		return err
	}
	up.memo[int(i)], err = up.stackLast()
	if err != nil {
		err = fmt.Errorf("loadBinPut() failed: %w", err)
	}

	return nil
}

// stores stack top in memo; index is 4-byte arg
func loadLongBinPut(up *Unpickler) error {
	buf, err := up.read(4)
	if err != nil {
		err = fmt.Errorf("loadLongBinPut() failed: %w", err)
		return err
	}
	i := int(binary.LittleEndian.Uint32(buf))
	up.memo[i], err = up.stackLast()
	if err != nil {
		err = fmt.Errorf("loadLongBinPut() failed: %w", err)
		return err
	}

	return nil
}

// stores top of the stack in memo
func loadMemoize(up *Unpickler) error {
	value, err := up.stackLast()
	if err != nil {
		err = fmt.Errorf("loadMemoize() failed: %w", err)
		return err
	}
	up.memo[len(up.memo)] = value

	return nil
}

// appends stack top to list below it
func loadAppend(up *Unpickler) error {
	value, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadAppend() failed: %w", err)
		return err
	}
	obj, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadAppend() failed: %w", err)
		return err
	}
	list, listOk := obj.(ListAppender)
	if !listOk {
		err := fmt.Errorf("APPEND requires ListAppender")
		err = fmt.Errorf("loadAppend() failed: %w", err)
		return err
	}
	list.Append(value)
	up.append(list)

	return nil
}

// extends list on stack by topmost stack slice
func loadAppends(up *Unpickler) error {
	items, err := up.popMark()
	if err != nil {
		err = fmt.Errorf("loadAppends() failed: %w", err)
		return err
	}
	obj, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadAppends() failed: %w", err)
		return err
	}
	list, listOk := obj.(ListAppender)
	if !listOk {
		err := fmt.Errorf("APPEND requires List")
		err = fmt.Errorf("loadAppends() failed: %w", err)
		return err
	}
	for _, item := range items {
		list.Append(item)
	}
	up.append(list)

	return nil
}

// adds key+value pair to dict
func loadSetItem(up *Unpickler) error {
	value, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadSetItem() failed: %w", err)
		return err
	}
	key, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadSetItem() failed: %w", err)
		return err
	}
	obj, err := up.stackLast()
	if err != nil {
		err = fmt.Errorf("loadSetItem() failed: %w", err)
		return err
	}
	dict, dictOk := obj.(DictSetter)
	if !dictOk {
		err := fmt.Errorf("SETITEM requires DictSetter")
		err = fmt.Errorf("loadSetItem() failed: %w", err)
		return err
	}
	dict.Set(key, value)

	return nil
}

// modifies dict by adding topmost key+value pairs
func loadSetItems(up *Unpickler) error {
	items, err := up.popMark()
	if err != nil {
		err = fmt.Errorf("loadSetItems() failed: %w", err)
		return err
	}
	obj, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadSetItems() failed: %w", err)
		return err
	}
	dict, dictOk := obj.(DictSetter)
	if !dictOk {
		err := fmt.Errorf("SETITEMS requires DictSetter")
		err = fmt.Errorf("loadSetItems() failed: %w", err)
		return err
	}
	itemsLen := len(items)
	for i := 0; i < itemsLen; i += 2 {
		dict.Set(items[i], items[i+1])
	}
	up.append(dict)

	return nil
}

// modifies set by adding topmost stack items
func loadAddItems(up *Unpickler) error {
	items, err := up.popMark()
	if err != nil {
		err = fmt.Errorf("loadAddItems() failed: %w", err)
		return err
	}
	obj, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadAddItems() failed: %w", err)
		return err
	}
	set, setOk := obj.(SetAdder)
	if !setOk {
		err := fmt.Errorf("ADDITEMS requires SetAdder")
		err = fmt.Errorf("loadAddItems() failed: %w", err)
		return err
	}
	for _, item := range items {
		set.Add(item)
	}
	up.append(set)

	return nil
}

// calls __setstate__ or __dict__.update()
func loadBuild(up *Unpickler) error {
	state, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadBuild() failed: %w", err)
		return err
	}
	inst, err := up.stackLast()
	if err != nil {
		err = fmt.Errorf("loadBuild() failed: %w", err)
		return err
	}
	if obj, ok := inst.(PyStateSettable); ok {
		return obj.PySetState(state)
	}

	var slotState interface{}
	if tuple, ok := state.(*Tuple); ok && tuple.Len() == 2 {
		state = tuple.Get(0)
		slotState = tuple.Get(1)
	}

	if stateDict, ok := state.(*Dict); ok {
		instPds, instPdsOk := inst.(PyDictSettable)
		if !instPdsOk {
			err := fmt.Errorf("BUILD requires a PyDictSettable instance: %#v", inst)
			err = fmt.Errorf("loadBuild() failed: %w", err)
			return err
		}
		for _, entry := range *stateDict {
			err := instPds.PyDictSet(entry.Key, entry.Value)
			if err != nil {
				err = fmt.Errorf("loadBuild() failed: %w", err)
				return err
			}
		}
	}

	if slotStateDict, ok := slotState.(*Dict); ok {
		instSa, instOk := inst.(PyAttrSettable)
		if !instOk {
			err := fmt.Errorf("BUILD requires a PyAttrSettable instance: %#v", inst)
			err = fmt.Errorf("loadBuild() failed: %w", err)
			return err
		}

		for _, entry := range *slotStateDict {
			sk, keyOk := entry.Key.(string)
			if !keyOk {
				err := fmt.Errorf("BUILD requires string slot state keys")
				err = fmt.Errorf("loadBuild() failed: %w", err)
				return err
			}
			err := instSa.PySetAttr(sk, entry.Value)
			if err != nil {
				err = fmt.Errorf("loadBuild() failed: %w", err)
				return err
			}
		}
	}

	return nil
}

// loads special markobject on stack
func loadMark(up *Unpickler) error {
	up.metaStack = append(up.metaStack, up.stack)
	up.stack = make([]interface{}, 0)

	return nil
}

// every pickle ends with STOP
func loadStop(up *Unpickler) error {
	value, err := up.stackPop()
	if err != nil {
		err = fmt.Errorf("loadStop() failed: %w", err)
		return err
	}

	return Stop{value: value}
}

// initUnpickleDispatch creates a dispatch table for unpickling machinery.
func initUnpicklerDispatch() {
	upDispatch[PROTO] = loadProto
	upDispatch[FRAME] = loadFrame
	upDispatch[PERSID] = loadPersId
	upDispatch[BINPERSID] = loadBinPersId
	upDispatch[NONE] = loadNone
	upDispatch[NEWFALSE] = loadFalse
	upDispatch[NEWTRUE] = loadTrue
	upDispatch[INT] = loadInt
	upDispatch[BININT] = loadBinInt
	upDispatch[BININT1] = loadBinInt1
	upDispatch[BININT2] = loadBinInt2
	upDispatch[LONG] = loadLong
	upDispatch[LONG1] = loadLong1
	upDispatch[LONG4] = loadLong4
	upDispatch[FLOAT] = loadFloat
	upDispatch[BINFLOAT] = loadBinFloat
	upDispatch[STRING] = loadString
	upDispatch[BINSTRING] = loadBinString
	upDispatch[BINBYTES] = loadBinBytes
	upDispatch[UNICODE] = loadUnicode
	upDispatch[BINUNICODE] = loadBinUnicode
	upDispatch[BINUNICODE8] = loadBinUnicode8
	upDispatch[BINBYTES8] = loadBinBytes8
	upDispatch[BYTEARRAY8] = loadByteArray8
	upDispatch[NEXT_BUFFER] = loadNextBuffer
	upDispatch[READONLY_BUFFER] = loadReadOnlyBuffer
	upDispatch[SHORT_BINSTRING] = loadShortBinString
	upDispatch[SHORT_BINBYTES] = loadShortBinBytes
	upDispatch[SHORT_BINUNICODE] = loadShortBinUnicode
	upDispatch[TUPLE] = loadTuple
	upDispatch[EMPTY_TUPLE] = loadEmptyTuple
	upDispatch[TUPLE1] = loadTuple1
	upDispatch[TUPLE2] = loadTuple2
	upDispatch[TUPLE3] = loadTuple3
	upDispatch[EMPTY_LIST] = loadEmptyList
	upDispatch[EMPTY_DICT] = loadEmptyDict
	upDispatch[EMPTY_SET] = loadEmptySet
	upDispatch[FROZENSET] = loadFrozenSet
	upDispatch[LIST] = loadList
	upDispatch[DICT] = loadDict
	upDispatch[INST] = loadInst
	upDispatch[OBJ] = loadObj
	upDispatch[NEWOBJ] = loadNewObj
	upDispatch[NEWOBJ_EX] = loadNewObjEx
	upDispatch[GLOBAL] = loadGlobal
	upDispatch[STACK_GLOBAL] = loadStackGlobal
	upDispatch[EXT1] = opExt1
	upDispatch[EXT2] = opExt2
	upDispatch[EXT4] = opExt4
	upDispatch[REDUCE] = loadReduce
	upDispatch[POP] = loadPop
	upDispatch[POP_MARK] = loadPopMark
	upDispatch[DUP] = loadDup
	upDispatch[GET] = loadGet
	upDispatch[BINGET] = loadBinGet
	upDispatch[LONG_BINGET] = loadLongBinGet
	upDispatch[PUT] = loadPut
	upDispatch[BINPUT] = loadBinPut
	upDispatch[LONG_BINPUT] = loadLongBinPut
	upDispatch[MEMOIZE] = loadMemoize
	upDispatch[APPEND] = loadAppend
	upDispatch[APPENDS] = loadAppends
	upDispatch[SETITEM] = loadSetItem
	upDispatch[SETITEMS] = loadSetItems
	upDispatch[ADDITEMS] = loadAddItems
	upDispatch[BUILD] = loadBuild
	upDispatch[MARK] = loadMark
	upDispatch[STOP] = loadStop
}

func decodeInt32(buf []byte) int {
	uval := binary.LittleEndian.Uint32(buf)
	val := int(uval)
	if buf[3]&0x80 != 0 {
		val = -(int(^uval) + 1)
	}

	return val
}

func decodeLong(data []byte) interface{} {
	if len(data) == 0 {
		return nil
	}

	// determine whether most-significant bit (MSB) is set
	isMsbSet := data[len(data)-1]&0x80 != 0

	if len(data) > 8 {
		bInt := new(big.Int)
		for i := len(data) - 1; i >= 0; i-- {
			bInt = bInt.Lsh(bInt, 8) // left shift 8 bits
			if isMsbSet {
				bInt = bInt.Or(bInt, big.NewInt(int64(^data[i])))
			} else {
				bInt = bInt.Or(bInt, big.NewInt(int64(data[i])))
			}
		} // for

		if isMsbSet {
			bInt = bInt.Add(bInt, big.NewInt(1))
			bInt = bInt.Neg(bInt)
		}

		return bInt

	} // if

	var val, bitMask uint64
	for i := len(data) - 1; i >= 0; i-- {
		val = (val << 8) | uint64(data[i])
		bitMask = (bitMask << 8) | 0xFF
	}

	if isMsbSet {
		return -(int(^val & bitMask))
	}

	return int(val)
}

func init() {
	initUnpicklerDispatch()
}

// Load decodes objects by loading through unpickling machinery.
func (up *Unpickler) Load() (interface{}, error) {
	up.metaStack = make([][]interface{}, 0)
	up.stack = make([]interface{}, 0)
	up.proto = 0

	for {
		opcode, err := up.readOne()
		if err != nil {
			return nil, err
		}

		opFunc := upDispatch[opcode]
		if opFunc == nil {
			err := fmt.Errorf("Unpickler.Load() failed:unknown opcode: 0x%x '%c'", opcode, opcode)
			return nil, err
		}

		err = opFunc(up)
		if err != nil {
			if p, ok := err.(Stop); ok {
				return p.value, nil
			}

			err := fmt.Errorf("Unpickler.Load() failed: %w", err)
			return nil, err
		}
	}
}

// Load unpickles a pickled file.
func Load(filename string) (interface{}, error) {
	f, err := os.Open(filename)
	if err != nil {
		err := fmt.Errorf("Load() failed: %w", err)
		return nil, err
	}
	defer f.Close()

	up := NewUnpickler(f)

	return up.Load()
}

// Loads unpicles a string.
func Loads(s string) (interface{}, error) {
	sr := strings.NewReader(s)
	up := NewUnpickler(sr)

	return up.Load()
}
