package ts

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

// NOTE: select, narrow and indexing operations (except when using a LongTensor index) return views onto the same memory.
// https://discuss.pytorch.org/t/does-select-and-narrow-return-a-view-or-copy/289

import (
	"fmt"
	"log"
	"reflect"

	"github.com/sugarme/gotch"
)

type NewAxis struct{}

// TensorIndexer is an interface which defines method `From`
// for any type to fulfill to become an tensor indexer
type TensorIndexer interface{}
type Select struct{ Index int64 }
type Narrow struct {
	Start int64
	End   int64
}
type IndexSelect struct{ Index *Tensor }
type InsertNewAxis struct{}

// NewSelect creates an tensor indexer with given index.
// `index` must be in range of tensor dimension. E.g. tensor shape [2,8]
// will have size = 2, hence `index` should be in range from [0,2)
func NewSelect(index int64) Select {
	return Select{index}
}

func NewNarrow(start, end int64) Narrow {
	return Narrow{Start: start, End: end}
}

func NewIndexSelect(ts *Tensor) IndexSelect {
	return IndexSelect{Index: ts}
}

func NewInsertNewAxis() InsertNewAxis {
	return InsertNewAxis{}
}

func NewSliceIndex(sl []int64) IndexSelect {
	ts := MustOfSlice(sl)

	return IndexSelect{Index: ts}
}

// type SelectFn func(int64)
// type NarrowFn func(from int64, to int64)
// type IndexSelectFn func(ts Tensor)
// type InsertNewAxisFn func()
//
// var (
// // Select        SelectFn
// // Narrow        NarrowFn
// // IndexSelect   IndexSelectFn
// // InsertNewAxis InsertNewAxisFn
// )

type IndexOp interface {
	Idx(index interface{}) Tensor
}

// implement IndexOp for Tensor:
// =============================

// Idx implements `IndexOp` interface for Tensor
//
// NOTE:
// - `index`: expects type `TensorIndexer` or `[]TensorIndexer`
func (ts *Tensor) Idx(index interface{}) (retVal *Tensor) {

	// indexTyp := reflect.TypeOf(index)
	indexVal := reflect.ValueOf(index)

	var indexes []TensorIndexer

	switch indexVal.Kind().String() {
	case "struct": // T: A
		indexes = append(indexes, index.(TensorIndexer))
	case "slice": // T: []TensorIndexer
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
func (ts *Tensor) indexer(indexSpec []TensorIndexer) (retVal *Tensor, err error) {

	// Make sure number of non-newaxis is not exceed number of dimensions
	var numNewAxis int = 0
	for _, ti := range indexSpec {
		if reflect.TypeOf(ti).Name() == "InsertNewAxis" {
			numNewAxis += 1
		}
	}

	tsShape, err := ts.Size()
	if err != nil {
		return retVal, err
	}
	tsLen := len(tsShape)
	if len(indexSpec) > tsLen+numNewAxis {
		err = fmt.Errorf("Too many indices for tensor of dimension %v\n", tsLen)
		return retVal, err
	}

	// Make sure tensor conforms the format
	for _, spec := range indexSpec {
		// If `spec` is `IndexSelect` type and
		if reflect.TypeOf(spec).Name() == "IndexSelect" {
			if reflect.ValueOf(spec).Kind() == reflect.Struct {
				inputTensor := reflect.ValueOf(spec).FieldByName("Index").Interface().(*Tensor)

				// 1. Either its input tensor has dimension > 1, throw error.
				inputTensorShape, err := inputTensor.Size()
				if err != nil {
					err = fmt.Errorf("Indexer Func Error: %v\n", err)
					return retVal, err
				}
				if len(inputTensorShape) != 1 {
					err = fmt.Errorf("Multi-dimenstional tensor is not supported for indexing.")
					return retVal, err
				}

				// 2. Or its input tensor has an unsupported dtype
				if inputTensor.DType() != gotch.Int64 &&
					inputTensor.DType() != gotch.Int16 &&
					inputTensor.DType() != gotch.Int8 &&
					inputTensor.DType() != gotch.Int {

					err = fmt.Errorf("The dtype of tensor used (%v) as indices must be one of: 'int64', 'int16', 'int8', 'int'. \n", inputTensor.DType())
					return retVal, err
				}
			}
		}
	}

	// Now, apply indexing from left to right.
	var (
		currTensor *Tensor = ts.MustShallowClone()
		currIdx    int64   = 0
		nextTensor *Tensor
		nextIdx    int64
	)

	// `spec` is a function type implements `TensorIndexer`
	for _, spec := range indexSpec {

		switch reflect.TypeOf(spec).Name() {
		case "InsertNewAxis":
			nextTensor, err = currTensor.Unsqueeze(currIdx, true)
			if err != nil {
				return retVal, err
			}
			nextIdx = currIdx + 1
		case "Select": // 1 field: `Index`
			index := reflect.ValueOf(spec).FieldByName("Index").Interface().(int64)
			nextTensor, err = currTensor.Select(currIdx, index, true) // TODO: double-check is `*index` or `index`
			if err != nil {
				return retVal, err
			}
			nextIdx = currIdx // not advanced because select() squeezes dimension
		case "Narrow": // 2 fields: `(Start, End int64)`
			// TODO: implement for `Unbounded`, `Included`, `Excluded` ranges
			// NOTE: for now, just implement (Included(start), Excluded(end))` case
			start := reflect.ValueOf(spec).FieldByName("Start").Interface().(int64)
			end := reflect.ValueOf(spec).FieldByName("End").Interface().(int64)
			nextTensor, err = currTensor.Narrow(currIdx, start, end-start, true)
			if err != nil {
				return retVal, err
			}
			nextIdx = currIdx + 1
		case "IndexSelect": // 1 field `(Index *Tensor)`
			indexTensor := reflect.ValueOf(spec).FieldByName("Index").Interface().(*Tensor)
			device, err := currTensor.Device()
			if err != nil {
				return retVal, err
			}
			indexTensor, err = indexTensor.To(device, true)
			if err != nil {
				return retVal, err
			}
			nextTensor, err = currTensor.IndexSelect(currIdx, indexTensor, true)
			if err != nil {
				return retVal, err
			}
			nextIdx = currIdx + 1
		} // end of switch

		currTensor = nextTensor
		currIdx = nextIdx
	}

	retVal = currTensor
	return retVal, nil
}

func (ts *Tensor) mustIndexer(indexSpec []TensorIndexer) (retVal *Tensor) {
	retVal, err := ts.indexer(indexSpec)
	if err != nil {
		panic(err)
	}
	return retVal
}
