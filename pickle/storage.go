package pickle

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/sugarme/gotch"
)

// This file implements Pytorch storage data types.
// ref: https://github.com/pytorch/pytorch/blob/c2255c36ec121fdb998ce3db8deb7508c814b567/torch/storage.py
/*
torch.double: 'DoubleStorage',
torch.float: 'FloatStorage',
torch.half: 'HalfStorage',
torch.long: 'LongStorage',
torch.int: 'IntStorage',
torch.int16: 'ShortStorage',
torch.int8: 'CharStorage',
torch.uint8: 'ByteStorage',
torch.bool: 'BoolStorage',
torch.bfloat16: 'BFloat16Storage',
torch.cdouble: 'ComplexDoubleStorage',
torch.cfloat: 'ComplexFloatStorage',
torch.qint8: 'QInt8Storage',
torch.qint32: 'QInt32Storage',
torch.quint8: 'QUInt8Storage',
torch.quint4x2: 'QUInt4x2Storage',
torch.quint2x4: 'QUInt2x4Storage',
*/

// StorageClass defines interface for types to be used in Storage.
type StorageClass interface {
	New(size int, location string) Storage
}

// Storage define Storage interface.
type Storage interface {
	SetFromFile(r io.Reader) error
	SetFromFileWithSize(r io.Reader, size int) error
	DType() gotch.DType
	GetData() interface{}
	Device() gotch.Device
}

// BaseStorage represents a base storage.
type BaseStorage struct {
	Size     int
	Location string
}

// HalfStorage:
// ============

type HalfStorageClass struct{}

var _ StorageClass = &HalfStorageClass{}

func (s *HalfStorageClass) New(size int, location string) Storage {
	return &HalfStorage{
		BaseStorage: BaseStorage{Size: size, Location: location},
		Data:        nil,
	}
}

type HalfStorage struct {
	BaseStorage
	Data []float32
}

var _ Storage = &HalfStorage{}

func (s *HalfStorage) SetFromFile(r io.Reader) error {
	return setFromFile(s, r)
}

func (s *HalfStorage) SetFromFileWithSize(r io.Reader, size int) error {
	data := make([]float32, size)
	br := NewLimitedBufferReader(r, size, 2, 512)
	for i := 0; i < size; i++ {
		bytes, err := br.ReadNext()
		if err != nil {
			return err
		}
		u16 := binary.LittleEndian.Uint16(bytes)
		data[i] = math.Float32frombits(FloatBits16to32(u16))
	}
	s.Data = data
	return nil
}

func (s *HalfStorage) GetData() interface{} {
	return s.Data
}

func (s *HalfStorage) DType() gotch.DType {
	return gotch.Float
}

func (s *HalfStorage) Device() gotch.Device {
	switch s.Location {
	case "cuda":
		return gotch.CudaIfAvailable()
	default:
		return gotch.CPU
	}
}

// FloatStorage:
// =============

type FloatStorageClass struct{}

var _ StorageClass = &FloatStorageClass{}

func (s *FloatStorageClass) New(size int, location string) Storage {
	return &FloatStorage{
		BaseStorage: BaseStorage{Size: size, Location: location},
		Data:        nil,
	}
}

type FloatStorage struct {
	BaseStorage
	Data []float32
}

var _ Storage = &FloatStorage{}

func (s *FloatStorage) SetFromFile(r io.Reader) error {
	return setFromFile(s, r)
}

func (s *FloatStorage) SetFromFileWithSize(r io.Reader, size int) error {
	data := make([]float32, size)
	br := NewLimitedBufferReader(r, size, 4, 512)
	for i := 0; i < size; i++ {
		bytes, err := br.ReadNext()
		if err != nil {
			return err
		}
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(bytes))
	}
	s.Data = data
	return nil
}

func (s *FloatStorage) GetData() interface{} {
	return s.Data
}

func (s *FloatStorage) DType() gotch.DType {
	return gotch.Float
}

func (s *FloatStorage) Device() gotch.Device {
	switch s.Location {
	case "cuda":
		return gotch.CudaIfAvailable()
	default:
		return gotch.CPU
	}
}

// DoubleStorage:
// ==============

type DoubleStorageClass struct{}

var _ StorageClass = &DoubleStorageClass{}

func (s *DoubleStorageClass) New(size int, location string) Storage {
	return &DoubleStorage{
		BaseStorage: BaseStorage{Size: size, Location: location},
		Data:        nil,
	}
}

type DoubleStorage struct {
	BaseStorage
	Data []float64
}

var _ Storage = &DoubleStorage{}

func (s *DoubleStorage) SetFromFile(r io.Reader) error {
	return setFromFile(s, r)
}

func (s *DoubleStorage) SetFromFileWithSize(r io.Reader, size int) error {
	data := make([]float64, size)
	br := NewLimitedBufferReader(r, size, 8, 512)
	for i := 0; i < size; i++ {
		bytes, err := br.ReadNext()
		if err != nil {
			return err
		}
		data[i] = math.Float64frombits(binary.LittleEndian.Uint64(bytes))
	}
	s.Data = data
	return nil
}

func (s *DoubleStorage) GetData() interface{} {
	return s.Data
}

func (s *DoubleStorage) DType() gotch.DType {
	return gotch.Double
}

func (s *DoubleStorage) Device() gotch.Device {
	switch s.Location {
	case "cuda":
		return gotch.CudaIfAvailable()
	default:
		return gotch.CPU
	}
}

// CharStorage:
// ============

type CharStorageClass struct{}

var _ StorageClass = &CharStorageClass{}

func (s *CharStorageClass) New(size int, location string) Storage {
	return &CharStorage{
		BaseStorage: BaseStorage{Size: size, Location: location},
		Data:        nil,
	}
}

type CharStorage struct {
	BaseStorage
	Data []int8
}

var _ Storage = &CharStorage{}

func (s *CharStorage) SetFromFile(r io.Reader) error {
	return setFromFile(s, r)
}

func (s *CharStorage) SetFromFileWithSize(r io.Reader, size int) error {
	data := make([]int8, size)
	br := NewLimitedBufferReader(r, size, 1, 512)
	for i := 0; i < size; i++ {
		bytes, err := br.ReadNext()
		if err != nil {
			return err
		}
		data[i] = int8(bytes[0])
	}
	s.Data = data
	return nil
}

func (s *CharStorage) GetData() interface{} {
	return s.Data
}

func (s *CharStorage) DType() gotch.DType {
	return gotch.Int8
}

func (s *CharStorage) Device() gotch.Device {
	switch s.Location {
	case "cuda":
		return gotch.CudaIfAvailable()
	default:
		return gotch.CPU
	}
}

// ShortStorage:
// =============

type ShortStorageClass struct{}

var _ StorageClass = &ShortStorageClass{}

func (s *ShortStorageClass) New(size int, location string) Storage {
	return &ShortStorage{
		BaseStorage: BaseStorage{Size: size, Location: location},
		Data:        nil,
	}
}

type ShortStorage struct {
	BaseStorage
	Data []int16
}

var _ Storage = &ShortStorage{}

func (s *ShortStorage) SetFromFile(r io.Reader) error {
	return setFromFile(s, r)
}

func (s *ShortStorage) SetFromFileWithSize(r io.Reader, size int) error {
	data := make([]int16, size)
	br := NewLimitedBufferReader(r, size, 2, 512)
	for i := 0; i < size; i++ {
		bytes, err := br.ReadNext()
		if err != nil {
			return err
		}
		data[i] = int16(binary.LittleEndian.Uint16(bytes))
	}
	s.Data = data
	return nil
}

func (s *ShortStorage) GetData() interface{} {
	return s.Data
}

func (s *ShortStorage) DType() gotch.DType {
	return gotch.Int16
}

func (s *ShortStorage) Device() gotch.Device {
	switch s.Location {
	case "cuda":
		return gotch.CudaIfAvailable()
	default:
		return gotch.CPU
	}
}

// IntStorage:
// ===========

type IntStorageClass struct{}

var _ StorageClass = &IntStorageClass{}

func (s *IntStorageClass) New(size int, location string) Storage {
	return &IntStorage{
		BaseStorage: BaseStorage{Size: size, Location: location},
		Data:        nil,
	}
}

type IntStorage struct {
	BaseStorage
	Data []int32
}

var _ Storage = &IntStorage{}

func (s *IntStorage) SetFromFile(r io.Reader) error {
	return setFromFile(s, r)
}

func (s *IntStorage) SetFromFileWithSize(r io.Reader, size int) error {
	data := make([]int32, size)
	br := NewLimitedBufferReader(r, size, 4, 512)
	for i := 0; i < size; i++ {
		bytes, err := br.ReadNext()
		if err != nil {
			return err
		}
		data[i] = int32(binary.LittleEndian.Uint32(bytes))
	}
	s.Data = data
	return nil
}

func (s *IntStorage) GetData() interface{} {
	return s.Data
}

func (s *IntStorage) DType() gotch.DType {
	return gotch.Int
}

func (s *IntStorage) Device() gotch.Device {
	switch s.Location {
	case "cuda":
		return gotch.CudaIfAvailable()
	default:
		return gotch.CPU
	}
}

// LongStorage:
// ============

type LongStorageClass struct{}

var _ StorageClass = &LongStorageClass{}

func (s *LongStorageClass) New(size int, location string) Storage {
	return &LongStorage{
		BaseStorage: BaseStorage{Size: size, Location: location},
		Data:        nil,
	}
}

type LongStorage struct {
	BaseStorage
	Data []int64
}

var _ Storage = &LongStorage{}

func (s *LongStorage) SetFromFile(r io.Reader) error {
	return setFromFile(s, r)
}

func (s *LongStorage) SetFromFileWithSize(r io.Reader, size int) error {
	data := make([]int64, size)
	br := NewLimitedBufferReader(r, size, 8, 512)
	for i := 0; i < size; i++ {
		bytes, err := br.ReadNext()
		if err != nil {
			return err
		}
		data[i] = int64(binary.LittleEndian.Uint64(bytes))
	}
	s.Data = data
	return nil
}

func (s *LongStorage) GetData() interface{} {
	return s.Data
}

func (s *LongStorage) DType() gotch.DType {
	return gotch.Int64
}

func (s *LongStorage) Device() gotch.Device {
	switch s.Location {
	case "cuda":
		return gotch.CudaIfAvailable()
	default:
		return gotch.CPU
	}
}

// ByteStorage:
// ============

type ByteStorageClass struct{}

var _ StorageClass = &ByteStorageClass{}

func (s *ByteStorageClass) New(size int, location string) Storage {
	return &ByteStorage{
		BaseStorage: BaseStorage{Size: size, Location: location},
		Data:        nil,
	}
}

type ByteStorage struct {
	BaseStorage
	Data []uint8
}

var _ Storage = &ByteStorage{}

func (s *ByteStorage) SetFromFile(r io.Reader) error {
	return setFromFile(s, r)
}

func (s *ByteStorage) SetFromFileWithSize(r io.Reader, size int) error {
	data := make([]uint8, size)
	br := NewLimitedBufferReader(r, size, 1, 512)
	for i := 0; i < size; i++ {
		bytes, err := br.ReadNext()
		if err != nil {
			return err
		}
		data[i] = bytes[0]
	}
	s.Data = data
	return nil
}

func (s *ByteStorage) GetData() interface{} {
	return s.Data
}

func (s *ByteStorage) DType() gotch.DType {
	return gotch.Uint8
}

func (s *ByteStorage) Device() gotch.Device {
	switch s.Location {
	case "cuda":
		return gotch.CudaIfAvailable()
	default:
		return gotch.CPU
	}
}

// BoolStorage:
// ============

type BoolStorageClass struct{}

var _ StorageClass = &BoolStorageClass{}

func (s *BoolStorageClass) New(size int, location string) Storage {
	return &BoolStorage{
		BaseStorage: BaseStorage{Size: size, Location: location},
		Data:        nil,
	}
}

type BoolStorage struct {
	BaseStorage
	Data []bool
}

var _ Storage = &BoolStorage{}

func (s *BoolStorage) SetFromFile(r io.Reader) error {
	return setFromFile(s, r)
}

func (s *BoolStorage) SetFromFileWithSize(r io.Reader, size int) error {
	data := make([]bool, size)
	br := NewLimitedBufferReader(r, size, 1, 512)
	for i := 0; i < size; i++ {
		bytes, err := br.ReadNext()
		if err != nil {
			return err
		}
		data[i] = bytes[0] == 1
	}
	s.Data = data
	return nil
}

func (s *BoolStorage) GetData() interface{} {
	return s.Data
}

func (s *BoolStorage) DType() gotch.DType {
	return gotch.Float
}

func (s *BoolStorage) Device() gotch.Device {
	switch s.Location {
	case "cuda":
		return gotch.CudaIfAvailable()
	default:
		return gotch.CPU
	}
}

func setFromFile(s Storage, r io.Reader) error {
	sizeBuf := make([]byte, 8)
	_, err := r.Read(sizeBuf)
	if err != nil {
		return err
	}
	size := int(binary.LittleEndian.Uint64(sizeBuf))
	return s.SetFromFileWithSize(r, size)
}

// StorageTensor:
//===============
type StorageTensor struct {
	Source        Storage
	StorageOffset int64
	Size          []int64
	Stride        []int64
	RequiresGrad  bool
}

// Rebuild Tensor:
// ===============
// ref. https://github.com/pytorch/pytorch/blob/c2255c36ec121fdb998ce3db8deb7508c814b567/torch/_utils.py#L132
// ref. def _rebuild_tensor(storage, storage_offset, size, stride):

type RebuildTensor struct{}

var _ Callable = &RebuildTensor{}

func (r *RebuildTensor) Call(args ...interface{}) (interface{}, error) {
	if len(args) != 4 {
		return nil, fmt.Errorf("RebuildTensor.Call() failed. Expected 4 args, got %d: %#v", len(args), args)
	}

	storage, storageOk := args[0].(Storage)
	storageOffset, storageOffsetOk := args[1].(int)
	size, sizeOk := args[2].(*Tuple)
	stride, strideOk := args[3].(*Tuple)
	if !storageOk || !storageOffsetOk || !sizeOk || !strideOk {
		return nil, fmt.Errorf("RebuildTensor.Call() unexpected args: %#v", args)
	}

	tensor := &StorageTensor{
		Source:        storage,
		StorageOffset: int64(storageOffset),
		RequiresGrad:  false,
	}
	var err error
	tensor.Size, err = tupleToInt64Slice(size)
	if err != nil {
		return nil, err
	}
	tensor.Stride, err = tupleToInt64Slice(stride)
	if err != nil {
		return nil, err
	}
	return tensor, nil
}

// RebuildTensorV2 represents a struct to rebuild tensor back from pickle object.
type RebuildTensorV2 struct{}

var _ Callable = &RebuildTensorV2{}

func (r *RebuildTensorV2) Call(args ...interface{}) (interface{}, error) {
	if len(args) != 6 {
		return nil, fmt.Errorf("RebuildTensorV2 unexpected args: %#v", args)
	}

	storage, storageOk := args[0].(Storage)
	storageOffset, storageOffsetOk := args[1].(int)
	size, sizeOk := args[2].(*Tuple)
	stride, strideOk := args[3].(*Tuple)
	requiresGrad, requiresGradOk := args[4].(bool)
	// arg[5] "backward hooks" is unused
	if !storageOk || !storageOffsetOk || !sizeOk || !strideOk ||
		!requiresGradOk {
		return nil, fmt.Errorf("RebuildTensorV2 unexpected args: %#v", args)
	}

	tensor := &StorageTensor{
		Source:        storage,
		StorageOffset: int64(storageOffset),
		RequiresGrad:  requiresGrad,
	}
	var err error
	tensor.Size, err = tupleToInt64Slice(size)
	if err != nil {
		return nil, err
	}
	tensor.Stride, err = tupleToInt64Slice(stride)
	if err != nil {
		return nil, err
	}
	return tensor, nil
}

// Rebuild Parameter:
// ==================
// RebuildTensor represents a struct to rebuild tensor back from pickle object.
// Ref. https://github.com/pytorch/pytorch/blob/c2255c36ec121fdb998ce3db8deb7508c814b567/torch/_utils.py#L240
type RebuildParameter struct{}

var _ Callable = &RebuildParameter{}

func (r *RebuildParameter) Call(args ...interface{}) (interface{}, error) {
	if len(args) != 3 { // data(*StorageTensor), requires_grad, backward_hooks
		return nil, fmt.Errorf("RebuildParameter unexpected 3 args, got %d: %#v", len(args), args)
	}

	tensor, ok := args[0].(*StorageTensor)
	if !ok {
		err := fmt.Errorf("RebuildParameter.Call() failed: unexpected arg: %#v\n", args)
		return nil, err
	}

	return tensor, nil
}

func tupleToInt64Slice(tuple *Tuple) ([]int64, error) {
	length := tuple.Len()
	slice := make([]int64, length)
	for i := 0; i < length; i++ {
		value, ok := tuple.Get(i).(int)
		if !ok {
			return nil, fmt.Errorf("tuple of ints expected: %#v", tuple)
		}
		slice[i] = int64(value)
	}
	return slice, nil
}

// Rebuild Sparse Tensor:
//=======================
// ref. https://github.com/pytorch/pytorch/blob/c2255c36ec121fdb998ce3db8deb7508c814b567/torch/_utils.py#L178
type RebuildSparseTensor struct{}

var _ Callable = &RebuildSparseTensor{}

func (r *RebuildSparseTensor) Call(args ...interface{}) (interface{}, error) {
	// TODO.
	panic("RebuildSparseTensor.Call(): NotImplementedError")
}

// Rebuild Sparse CSR Tensor:
// ==========================
// Ref. https://github.com/pytorch/pytorch/blob/c2255c36ec121fdb998ce3db8deb7508c814b567/torch/_utils.py#L187
type RebuildSparseCsrTensor struct{}

var _ Callable = &RebuildSparseCsrTensor{}

func (r *RebuildSparseCsrTensor) Call(args ...interface{}) (interface{}, error) {
	// TODO.
	panic("RebuildSparseCsrTensor.Call(): NotImplementedError")
}

// Rebuild Device Tensor From Numpy:
// =================================
// Ref. https://github.com/pytorch/pytorch/blob/c2255c36ec121fdb998ce3db8deb7508c814b567/torch/_utils.py#L197
type RebuildDeviceTensorFromNumpy struct{}

var _ Callable = &RebuildDeviceTensorFromNumpy{}

func (r *RebuildDeviceTensorFromNumpy) Call(args ...interface{}) (interface{}, error) {
	// TODO.
	panic("RebuildDeviceTensorFromNumpy.Call(): NotImplementedError")
}

// Rebuild Meta Tensor No Storage:
// ===============================
// Ref. https://github.com/pytorch/pytorch/blob/c2255c36ec121fdb998ce3db8deb7508c814b567/torch/_utils.py#L208
type RebuildMetaTensorNoStorage struct{}

var _ Callable = &RebuildMetaTensorNoStorage{}

func (r *RebuildMetaTensorNoStorage) Call(args ...interface{}) (interface{}, error) {
	// TODO.
	panic("RebuildMetaTensorNoStorage.Call(): NotImplementedError")
}

// Rebuild QTensor:
// ================
// Ref. https://github.com/pytorch/pytorch/blob/c2255c36ec121fdb998ce3db8deb7508c814b567/torch/_utils.py#L214
type RebuildQtensor struct{}

var _ Callable = &RebuildQtensor{}

func (r *RebuildQtensor) Call(args ...interface{}) (interface{}, error) {
	// TODO.
	panic("RebuildQtensor.Call(): NotImplementedError")
}
