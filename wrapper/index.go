package wrapper

// Indexing operations for tensor
// It defines a `i` indexing operation. This can be used in various scenarios.
//
// Usage:
// Using an integer index returns slice obtained by selecting elements with
// specified index. Negative values can be used for the index, and `..` can
// be used to get all the indexes from a given dimension.
//
// ```
//  ts := wrapper.OfSlice(int[1,2,3,4,5,6])
//	ts.View((2,3))
//	t := ts.i(1)
//	t = ts.i(.., -2)
//```
//
// Indexes like `..1`, `1..` or `1..2` can be used to narrow a dimension.
//
// ```
//  ts := wrapper.OfSlice(int[1,2,3,4,5,6])
//	ts.View((2,3))
//	t := ts.i((..,1..))
//	t.Size() 								// [2,2]
//	t = t.Contiguous()
//	tsSlice := t.View(-1) 	// [2,3,5,6]
//	t := ts.i((..1, ..))
//	t.Size() 								// [1,3]
//	t = t.Contiguous()
// 	t.View(-1)							// [1,2,3]
//	t = ts.i((.., 1..2))
// 	t.Size()								// [2,1]
// 	t = t.Contiguous()
//	t = t.View(-1)					// [2,5]
//	t = ts.i((.., 1..=2))
//	t.Size()								// [2,2]
//	t = t.Contiguous()
//	t.View(-1)							// [2,3,5,6]
// ```
//
// `NewAxis` index can be used to insert a dimension.
//
// 	```
//  ts := wrapper.OfSlice(int[1,2,3,4,5,6])
//	ts.View((2,3))
// 	t := ts.i((NewAxis,))
//	t.Size()									// [1,2,3]
// 	t = ts.i((..,..,NewAxis))
// 	t.Size()									// [2,3,1]
//	```
//
// Unlike NumPy, the `i` operation does not support advanced indexing.
// The result can be different from NumPy with same set of arguments.
// For example, `tensor.i(..1, []int{0,3}, []int{2,1,3})` does narrowing
// on first dimension, and index selection on second and third dimensions.
// The analogous NumPy indexing `array[:1, [0, 3], [2, 1, 3]]` throws
// shape mismatch error due to advanced indexing rule. Another distinction
// is that `i` guarantees the input and result tensor shares the same
// underlying storage, while NumPy may copy the tensor in certain scenarios.

import (
	"fmt"
	"log"
	"reflect"

	"github.com/sugarme/gotch"
)

type NewAxis struct{}

// TensorIndexer is an interface which defines method `From`
// for any type to fulfill to become an tensor indexer
type TensorIndexer interface {
	From(interface{}) TensorIndexer
}

// Below is a list of all types that implement `TensorIndexer`
// So that they can act as tensor indexer.
// type Select struct{}
// type Narrow struct {
// bound int64
// }
// type IndexSelect struct {
// tensor wrapper.Tensor
// }
// type InsertNewAxis struct{}
//
// // Implementing `TensorIndexer`
// func (sel Select) From(index interface{}) TensorIndexer {
// return sel.Select(index.(int64))
// }
// func (sel Select) new(index int64) Select {
// return Select{index: index}
// }

type SelectFn func(int64) TensorIndexer
type NarrowFn func(from int64, to int64) TensorIndexer
type IndexSelectFn func(ts Tensor) TensorIndexer
type InsertNewAxisFn func() TensorIndexer

func (sel SelectFn) From(index int64) TensorIndexer {
	return sel(index)
}

// TODO: implement `TensorIndexer` for the rest

// NOTE: all the below variables will have `TensorIndexer` trait.
// In other words, they are `TensorIndexer` type.
//
// Alternatively, we can create a enum-like of TensorIndexer using map.
// E.g. TensorIndexers =  map[string]interface{}
// TensorIndexers["Select"]  = SelectFn
// TensorIndexers["Narrow"]  = NarrowFn
// TensorIndexers["IndexSelect"]  = IndexSelectFn
// TensorIndexers["InsertNewAxis"]  = InsertNewAxisFn
var (
	Select        SelectFn
	Narrow        NarrowFn
	IndexSelect   IndexSelectFn
	InsertNewAxis InsertNewAxisFn
)

type IndexOp interface {
	Idx(index interface{}) Tensor
}

// implement IndexOp for Tensor:
// =============================

// Idx implements `IndexOp` interface for Tensor
//
// NOTE:
// - `index`: expects type `TensorIndexer` or `[]TensorIndexer`
func (ts *Tensor) Idx(index interface{}) (retVal Tensor) {

	// indexTyp := reflect.TypeOf(index)
	indexVal := reflect.ValueOf(index)

	var indexes []TensorIndexer

	switch indexVal.Kind().String() { // TODO: double check whether it 'Interface' or 'TensorIndexer'???
	case "TensorIndexer": // T: A
		indexes = append(indexes, index.(TensorIndexer))
	case "Slice": // T: []TensorIndexer
		switch len(index.([]TensorIndexer)) {
		case 1: // T: [A]
			idxA := index.([]TensorIndexer)[0]
			indexes = append(indexes, idxA)
		case 2: // T: [A, B]
			idxA := index.([]TensorIndexer)[0]
			idxB := index.([]TensorIndexer)[1]
			indexes = append(indexes, idxA, idxB)
		case 3: // T: [A, B, C]
			idxA := index.([]TensorIndexer)[0]
			idxB := index.([]TensorIndexer)[1]
			idxC := index.([]TensorIndexer)[2]
			indexes = append(indexes, idxA, idxB, idxC)
		case 4: // T: [A, B, C, D]
			idxA := index.([]TensorIndexer)[0]
			idxB := index.([]TensorIndexer)[1]
			idxC := index.([]TensorIndexer)[2]
			idxD := index.([]TensorIndexer)[3]
			indexes = append(indexes, idxA, idxB, idxC, idxD)
		case 5: // T: [A, B, C, D, E]
			idxA := index.([]TensorIndexer)[0]
			idxB := index.([]TensorIndexer)[1]
			idxC := index.([]TensorIndexer)[2]
			idxD := index.([]TensorIndexer)[3]
			idxE := index.([]TensorIndexer)[4]
			indexes = append(indexes, idxA, idxB, idxC, idxD, idxE)
		case 6: // T: [A, B, C, D, E, F]
			idxA := index.([]TensorIndexer)[0]
			idxB := index.([]TensorIndexer)[1]
			idxC := index.([]TensorIndexer)[2]
			idxD := index.([]TensorIndexer)[3]
			idxE := index.([]TensorIndexer)[4]
			idxF := index.([]TensorIndexer)[5]
			indexes = append(indexes, idxA, idxB, idxC, idxD, idxE, idxF)
		case 7: // T: [A, B, C, D, E, F, G]
			idxA := index.([]TensorIndexer)[0]
			idxB := index.([]TensorIndexer)[1]
			idxC := index.([]TensorIndexer)[2]
			idxD := index.([]TensorIndexer)[3]
			idxE := index.([]TensorIndexer)[4]
			idxF := index.([]TensorIndexer)[5]
			idxG := index.([]TensorIndexer)[6]
			indexes = append(indexes, idxA, idxB, idxC, idxD, idxE, idxF, idxG)
		default:
			log.Fatalf("Invalid input 'index' slice length (%v) - max is 7\n", len(index.([]TensorIndexer)))
		}
	default:
		log.Fatalf("Invalid 'index' type (%v) - Expected type 'TensorIndexer' or '[]TensorIndexer'\n.", indexVal.Kind().String())
	}

	return ts.mustIndexer(indexes)
}

// Tensor Methods:
// ===============
func (ts Tensor) indexer(indexSpec []TensorIndexer) (retVal Tensor, err error) {

	// Make sure number of non-newaxis is not exceed number of dimensions
	var nonNewAxis []TensorIndexer
	for _, ti := range indexSpec {
		if reflect.ValueOf(ti).String() != "InsertNewAxis" {
			nonNewAxis = append(nonNewAxis, ti)
		}
	}
	tsShape, err := ts.Size()
	if err != nil {
		return retVal, err
	}
	tsLen := len(tsShape)
	if len(nonNewAxis) > tsLen {
		err = fmt.Errorf("Too many indices for tensor of dimension %v\n", tsLen)
		return retVal, err
	}

	// Make sure tensor conforms the format
	for _, spec := range indexSpec {
		// If `spec` is `IndexSelectFn` function and either
		if reflect.ValueOf(spec).String() == "IndexSelectFn" {

			// 1. its input tensor has dimension > 1, throw error.
			f, err := NewFunc(spec)
			if err != nil {
				err = fmt.Errorf("Indexer Func Error: %v\n", err)
				return retVal, err
			}
			//  list of `spec` function input parameters.
			inArgs := f.Info().InArgs
			tsVal := inArgs[0] // reflect.Value
			inputTensor := reflect.ValueOf(tsVal).Interface().(Tensor)
			inputTensorShape, err := inputTensor.Size()
			if err != nil {
				err = fmt.Errorf("Indexer Func Error: %v\n", err)
				return retVal, err
			}
			if len(inputTensorShape) != 1 {
				err = fmt.Errorf("Multi-dimenstional tensor is not supported for indexing.")
				return retVal, err
			}

			// 2. its input tensor has an unsupported dtype
			if inputTensor.DType() != gotch.Int64 ||
				inputTensor.DType() != gotch.Int16 ||
				inputTensor.DType() != gotch.Int8 ||
				inputTensor.DType() != gotch.Int {

				err = fmt.Errorf("The dtype of tensor used as indices must be one of: 'int64', 'int16', 'int8', 'int'. \n")
				return retVal, err
			}
		}
	}

	// Now, apply indexing from left to right.
	var (
		currTensor Tensor = ts.MustShallowClone()
		currIdx    int64  = 0
	)

	// `spec` is a function type implements `TensorIndexer`
	for _, spec := range indexSpec {
		var (
			nextTensor Tensor
			nextIdx    int64
		)

		// get info of `spec` function
		f, err := NewFunc(spec)
		if err != nil {
			err = fmt.Errorf("Indexer Func Error: %v\n", err)
			return retVal, err
		}
		//  list of `spec` function input parameters.
		inArgs := f.Info().InArgs

		// Now, specific indexOp depending on `TensorIndexer` func
		switch reflect.ValueOf(spec).Kind().String() {
		case "InsertNewAxis":
			nextTensor, err = currTensor.Unsqueeze(currIdx)
			if err != nil {
				return retVal, err
			}
			nextIdx = currIdx + 1
		case "SelectFn": // 1 param of `(index int64)`
			indexVal := inArgs[0]
			index := reflect.ValueOf(indexVal).Interface().(int64)
			nextTensor, err = currTensor.Select(currIdx, index) // TODO: double-check is `*index` or `index`
			if err != nil {
				return retVal, err
			}
			nextIdx = currIdx // not advanced because select() squeezes dimension
		case "NarrowFn": // 2 params: `(start, end int64)`
			// TODO: implement for `Unbounded`, `Included`, `Excluded` ranges
			// NOTE: for now, just implement (Included(start), Excluded(end))` case
			start := reflect.ValueOf(inArgs[0]).Interface().(int64)
			end := reflect.ValueOf(inArgs[1]).Interface().(int64)
			nextTensor, err = currTensor.Narrow(currIdx, start, end-start)
			if err != nil {
				return retVal, err
			}
			nextIdx = currIdx + 1
		case "IndexSelectFn": // 1 param `(indexTensor Tensor)`
			indexTensor := reflect.ValueOf(inArgs[0]).Interface().(Tensor)
			indexTensor, err = indexTensor.ToDevice(currTensor.Device())
			if err != nil {
				return retVal, err
			}
			nextTensor, err = currTensor.IndexSelect(currIdx, indexTensor)
			if err != nil {
				return retVal, err
			}
			nextIdx = currIdx + 1
		}

		currTensor = nextTensor
		currIdx = nextIdx
	}

	retVal = currTensor

	return
}

func (ts Tensor) mustIndexer(indexSpec []TensorIndexer) (retVal Tensor) {
	retVal, err := ts.indexer(indexSpec)
	if err != nil {
		panic(err)
	}
	return retVal
}
