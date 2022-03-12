package pickle

// Ref.
// https://docs.python.org/3/library/pickle.html
// https://docs.python.org/3/library/pickletools.html
// https://github.com/python/cpython/blob/main/Lib/pickle.py (****real code here****)
// https://pytorch.org/tutorials/beginner/saving_loading_models.html
// https://github.com/pytorch/pytorch/blob/master/torch/serialization.py

import (
	"archive/tar"
	"archive/zip"
	"errors"
	"fmt"
	"io"
	"log"
	"math/big"
	"os"
	"path"
	"reflect"
	"sort"

	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

const hexMagicNumber = "1950a86a20f9469cfc6c"
const protocolVersion = 1001

var ErrInvalidMagicNumber = errors.New("invalid pytorch magic number")
var ErrInvalidProtocolVersion = errors.New("invalid pytorch protocol version")

// Encode encodes model using pickling machinery.
// Output pickled model can be loads with Python Pytorch as `torch.load("pytorch_model.bin")`
//
// TODO. implement pickling part so that model can be exported and load with Python Pytorch.
// See https://github.com/python/cpython/blob/b0de6299a840a397d4fe3e6c98159d9f258d3295/Lib/pickle.py#L407
func Encode(model ts.Module, outputFile string) error {
	panic("NotImplementedError")
}

// Decode decodes pickled data created by 'torch.save()' with Python Pytorch
// and rebuilds named tensor weights.
func Decode(filename string) (map[string]*ts.Tensor, error) {
	newUnpickler := func(r io.Reader) Unpickler {
		return NewUnpickler(r)
	}
	result, err := LoadWithUnpickler(filename, newUnpickler)
	if err != nil {
		err := fmt.Errorf("Decode() failed: %w", err)
		return nil, err
	}

	// Rebuild tensors from Storage tensors
	namedTensors := make(map[string]*ts.Tensor)
	dictResult, isOrderedDict := result.(*OrderedDict)
	if !isOrderedDict {
		err := fmt.Errorf("Decode() failed: expected 'OrderedDict' type, got %v\n", reflect.TypeOf(result).String())
		return nil, err
	}
	for name, item := range dictResult.Map {
		sx, isStorageTensor := item.Value.(*StorageTensor)
		if !isStorageTensor {
			err := fmt.Errorf("Decode() failed: expected 'StorageTensor' type, got %v\n", reflect.TypeOf(item.Value).String())
			return nil, err
		}

		data := sx.Source.GetData()
		size := sx.Size
		dtype := sx.Source.DType()
		device := sx.Source.Device()
		stride := sx.Stride
		storageOffset := sx.StorageOffset

		// log.Printf("%q - %q - shape: %v - stride: %v - storageOffset: %v\n", name, sx.Source.Device().Name, sx.Size, sx.Stride, storageOffset)
		// log.Printf("data: %v\n", data)

		// Dealing with Pytorch `..._tracked` variables.
		if reflect.ValueOf(data).Len() == 0 {
			log.Printf("INFO: skip weigth %q with zero data length.\n", name.(string))
			continue
		}

		// TODO. should we just skip them?
		if reflect.ValueOf(data).Len() == 1 && len(size) == 0 {
			size = []int64{1}
			stride = []int64{1}
		}

		x := ts.MustOfSlice(data).MustAsStrided(size, stride, []int64{storageOffset}, true).MustTotype(dtype, true).MustTo(device, true)
		if sx.RequiresGrad {
			x.MustRequiresGrad_(sx.RequiresGrad)
		}

		namedTensors[name.(string)] = x
	}

	return namedTensors, nil
}

// LoadWithUnpickler is like Load, but it accepts a newUnpickler function which
// is used to create new customized pickle.Unpickler instances.
func LoadWithUnpickler(filename string, newUnpickler func(r io.Reader) Unpickler) (interface{}, error) {
	if !isZipFile(filename) {
		return loadLegacyFile(filename, newUnpickler)
	}
	return loadZipFile(filename, newUnpickler)
}

func loadZipFile(filename string, newUnpickler func(r io.Reader) Unpickler) (interface{}, error) {
	// Open a zip archive for reading.
	r, err := zip.OpenReader(filename)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	fileRecords := make(map[string]*zip.File, len(r.File))
	for _, f := range r.File {
		_, recordName := path.Split(f.Name)
		fileRecords[recordName] = f
	}

	if _, isTorchScript := fileRecords["constants.pkl"]; isTorchScript {
		return nil, fmt.Errorf("TorchScript is not supported")
	}

	dataFile, hasDataFile := fileRecords["data.pkl"]
	if !hasDataFile {
		return nil, fmt.Errorf("data.pkl not found in zip file")
	}
	df, err := dataFile.Open()
	if err != nil {
		return nil, err
	}
	defer df.Close()

	loadedStorages := make(map[string]Storage)

	u := newUnpickler(df)
	u.FindClass = makePickleFindClass(u.FindClass)
	u.PersistentLoad = func(savedId interface{}) (interface{}, error) {
		tuple, tupleOk := savedId.(*Tuple)
		if !tupleOk || tuple.Len() == 0 {
			return nil, fmt.Errorf("PersistentLoad: non-empty tuple expected, got %#v", savedId)
		}
		typename, typenameOk := tuple.Get(0).(string)
		if !typenameOk {
			return nil, fmt.Errorf("PersistentLoad: cannot get typename")
		}
		if typename != "storage" {
			return nil, fmt.Errorf("unknown typename for PersistentLoad, expected 'storage' but got '%s'", typename)
		}
		if tuple.Len() < 5 {
			return nil, fmt.Errorf("PersistentLoad: unexpected storage data length")
		}
		dataType, dataTypeOk := tuple.Get(1).(StorageClass)
		key, keyOk := tuple.Get(2).(string)
		location, locationOk := tuple.Get(3).(string)
		size, sizeOk := tuple.Get(4).(int)
		if !dataTypeOk || !keyOk || !locationOk || !sizeOk {
			return nil, fmt.Errorf("PersistentLoad: unexpected data types")
		}
		storage, storageExists := loadedStorages[key]
		if !storageExists {
			storage, err = loadTensor(dataType, size, location, key, fileRecords)
			if err != nil {
				return nil, err
			}
			loadedStorages[key] = storage
		}
		return storage, nil
	}
	return u.Load()
}

func loadTensor(
	dataType StorageClass,
	size int,
	location, key string,
	zipFileRecords map[string]*zip.File,
) (Storage, error) {
	file, fileOk := zipFileRecords[key]
	if !fileOk {
		return nil, fmt.Errorf("cannot find zip record '%s'", key)
	}
	f, err := file.Open()
	if err != nil {
		return nil, err
	}
	defer f.Close()

	storage := dataType.New(size, location)
	err = storage.SetFromFileWithSize(f, size)
	return storage, err
}

func loadLegacyFile(filename string, newUnpickler func(r io.Reader) Unpickler) (interface{}, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	tr := tar.NewReader(f)
	for {
		_, err := tr.Next()
		switch err {
		case nil:
			// TODO: ...
			panic("legacy load from tar not implemented")
		case io.EOF:
			break // End of archive
		case tar.ErrHeader, io.ErrUnexpectedEOF:
			_, err = f.Seek(0, io.SeekStart)
			if err != nil {
				return nil, err
			}
			return loadLegacyNoTar(f, newUnpickler)
		default:
			return nil, err
		}
	}
}

func loadLegacyNoTar(f *os.File, newUnpickler func(r io.Reader) Unpickler) (interface{}, error) {
	if err := readAndCheckMagicNumber(f); err != nil {
		return nil, err
	}
	if err := readAndChecProtocolVersion(f); err != nil {
		return nil, err
	}
	if _, err := unpickle(f); err != nil { // sys info
		return nil, err
	}

	deserializedObjects := make(map[string]Storage)

	u := newUnpickler(f)
	u.FindClass = makePickleFindClass(u.FindClass)
	u.PersistentLoad = func(savedId interface{}) (interface{}, error) {
		tuple, tupleOk := savedId.(*Tuple)
		if !tupleOk || tuple.Len() == 0 {
			return nil, fmt.Errorf("PersistentLoad: non-empty tuple expected, got %#v", savedId)
		}
		typename, typenameOk := tuple.Get(0).(string)
		if !typenameOk {
			return nil, fmt.Errorf("PersistentLoad: cannot get typename")
		}

		// fmt.Printf("typename: %s\n", typename)

		switch typename {
		case "storage":
			if tuple.Len() < 6 {
				return nil, fmt.Errorf(
					"PersistentLoad: unexpected storage data length")
			}
			dataType, dataTypeOk := tuple.Get(1).(StorageClass)
			rootKey, rootKeyOk := tuple.Get(2).(string)
			location, locationOk := tuple.Get(3).(string)
			size, sizeOk := tuple.Get(4).(int)
			viewMetadata := tuple.Get(5)
			if !dataTypeOk || !rootKeyOk || !locationOk || !sizeOk {
				return nil, fmt.Errorf("PersistentLoad: unexpected data types")
			}

			// fmt.Printf("dtype: %v - rootKey: %v - device: %v - size %v - viewMetaData: %v\n", reflect.TypeOf(dataType), rootKey, location, size, viewMetadata)

			storage, storageExists := deserializedObjects[rootKey]
			if !storageExists {
				storage = dataType.New(size, location)
				deserializedObjects[rootKey] = storage
			}
			switch vm := viewMetadata.(type) {
			case nil:
				return storage, nil
			case []interface{}:
				if len(vm) != 3 {
					return nil, fmt.Errorf(
						"PersistentLoad: unexpected view metadata length")
				}
				panic("viewMetadata not implemented")
				// TODO: ...
				// view_key, offset, view_size = view_metadata
				// if view_key not in deserialized_objects:
				//     deserialized_objects[view_key] = storage[offset:offset + view_size]
				// return deserialized_objects[view_key]
			default:
				return nil, fmt.Errorf("PersistentLoad: unexpected view metadata type")
			}
		case "module":
			if tuple.Len() < 2 {
				return nil, fmt.Errorf("PersistentLoad: unexpected module data length")
			}
			return tuple.Get(1), nil
		default:
			return nil, fmt.Errorf("Unexpected saved ID type: %s", typename)
		}
	}

	result, err := u.Load()
	if err != nil {
		return nil, err
	}

	rawStorageKeys, err := unpickle(f)
	if err != nil {
		return nil, err
	}
	storageKeys, err := makeStorageKeys(rawStorageKeys)
	if err != nil {
		return nil, err
	}

	for _, key := range storageKeys {
		storageObj, ok := deserializedObjects[key]
		if !ok {
			return nil, fmt.Errorf("storage object not found for key '%s'", key)
		}

		err = storageObj.SetFromFile(f)
		if err != nil {
			return nil, err
		}
	}

	return result, nil
}

func makeStorageKeys(obj interface{}) ([]string, error) {
	list, ok := obj.(*List)
	if !ok {
		return nil, fmt.Errorf("invalid storage keys data")
	}
	keys := make([]string, len(*list))
	for i, rawKey := range *list {
		key, keyOk := rawKey.(string)
		if !keyOk {
			return nil, fmt.Errorf("invalid storage key")
		}
		keys[i] = key
	}
	return keys, nil
}

func readAndCheckMagicNumber(r io.Reader) error {
	obj, err := unpickle(r)
	if err != nil {
		return err
	}
	if n, ok := obj.(*big.Int); !ok || n.Text(16) != hexMagicNumber {
		return ErrInvalidMagicNumber
	}
	return nil
}

func readAndChecProtocolVersion(r io.Reader) error {
	obj, err := unpickle(r)
	if err != nil {
		return err
	}
	if n, ok := obj.(int); !ok || n != protocolVersion {
		return ErrInvalidProtocolVersion
	}
	return nil
}

func unpickle(r io.Reader) (interface{}, error) {
	u := NewUnpickler(r)
	return u.Load()
}

func isZipFile(filename string) bool {
	r, err := zip.OpenReader(filename)
	if err != nil {
		return false
	}
	r.Close()
	return true
}

func makePickleFindClass(fallback func(module, name string) (interface{}, error)) func(module, name string) (interface{}, error) {
	return func(module, name string) (interface{}, error) {
		switch module + "." + name {
		case "torch._utils._rebuild_tensor":
			return &RebuildTensor{}, nil
		case "torch._utils._rebuild_tensor_v2":
			return &RebuildTensorV2{}, nil
		case "torch._utils._rebuild_parameter":
			return &RebuildParameter{}, nil
		case "torch._utils._sparse_tensor":
			return &RebuildSparseTensor{}, nil
		case "torch._utils._rebuild_sparse_csr_tensor":
			return &RebuildSparseCsrTensor{}, nil
		case "torch._utils._rebuild_device_tensor_from_numpy":
			return &RebuildDeviceTensorFromNumpy{}, nil
		case "torch._utils._rebuild_meta_tensor_no_storage":
			return &RebuildMetaTensorNoStorage{}, nil
		case "torch._utils._rebuild_qtensor":
			return &RebuildQtensor{}, nil

		case "torch.FloatStorage":
			return &FloatStorageClass{}, nil
		case "torch.HalfStorage":
			return &HalfStorageClass{}, nil
		case "torch.DoubleStorage":
			return &DoubleStorageClass{}, nil
		case "torch.CharStorage":
			return &CharStorageClass{}, nil
		case "torch.ShortStorage":
			return &ShortStorageClass{}, nil
		case "torch.IntStorage":
			return &IntStorageClass{}, nil
		case "torch.LongStorage":
			return &LongStorageClass{}, nil
		case "torch.ByteStorage":
			return &ByteStorageClass{}, nil
		case "torch.BoolStorage":
			return &BoolStorageClass{}, nil
		case "torch.nn.backends.thnn._get_thnn_function_backend":
			// this is for historical pickle deserilaization, it is not used otherwise
			return getThnnFunctionBackend{}, nil
		default:
			if fallback == nil {
				return nil, fmt.Errorf("class not found: %s %s", module, name)
			}
			return fallback(module, name)
		}
	}
}

// LoadAll finds and loads all weights from varstore.
// It will throw err if one of weights from varstore cannot find from loaded pretrained model.
func LoadAll(vs *nn.VarStore, modelFile string) error {
	weights, err := Decode(modelFile)
	if err != nil {
		err = fmt.Errorf("LoadAll() failed: %w", err)
		return err
	}

	var namedTensors []ts.NamedTensor
	for n, x := range weights {
		namedTs := ts.NamedTensor{
			Name:   n,
			Tensor: x,
		}

		namedTensors = append(namedTensors, namedTs)
	}

	err = vs.LoadWeights(namedTensors)
	if err != nil {
		return err
	}

	for _, x := range weights {
		x.MustDrop()
	}

	return nil
}

// LoadPartial finds and loads weights for varstore.
// It returns list of unfound weight names.
func LoadPartial(vs *nn.VarStore, modelFile string) ([]string, error) {
	weights, err := Decode(modelFile)
	if err != nil {
		err = fmt.Errorf("LoadPartial() failed: %w", err)
		return nil, err
	}

	var namedTensors []ts.NamedTensor
	for n, x := range weights {
		namedTs := ts.NamedTensor{
			Name:   n,
			Tensor: x,
		}

		namedTensors = append(namedTensors, namedTs)
	}

	var missingVariables []string

	missingVariables, err = vs.LoadWeightsPartial(namedTensors)
	if err != nil {
		return nil, err
	}

	for _, x := range weights {
		x.MustDrop()
	}

	return missingVariables, nil
}

// LoadInfo loads pretrained weights and prints out name and shape of weights.
func LoadInfo(modelFile string) error {
	weights, err := Decode(modelFile)
	if err != nil {
		err = fmt.Errorf("LoadInfo() failed: %w", err)
		return err
	}

	layers := make([]string, 0, len(weights))
	for tsName := range weights {
		layers = append(layers, tsName)
	}
	sort.Strings(layers)
	for _, l := range layers {
		var x *ts.Tensor
		for tsName, tsVal := range weights {
			if tsName == l {
				x = tsVal
				break
			}
		}
		fmt.Printf("%s - %+v\n", l, x.MustSize())
	}

	fmt.Printf("Num of variables: %v\n", len(weights))

	for _, x := range weights {
		x.MustDrop()
	}

	return nil
}
