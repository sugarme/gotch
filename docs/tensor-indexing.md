# Tensor Operations

## Tensor Indexing

### Insert New axis

```go
    
    import(
        "github.com/sugarme/gotch"
        ts "github.com/sugarme/gotch/tensor"
    )
    ...

	tensor := ts.MustArange1(ts.IntScalar(0), ts.IntScalar(2*3), gotch.Int64, gotch.CPU).MustView([]int64{2, 3}, true)

	var idxs1 []ts.TensorIndexer = []ts.TensorIndexer{
		ts.NewInsertNewAxis(),
	}

	result1 := tensor.Idx(idxs1) // [1, 2, 3]
    

	var idxs2 []ts.TensorIndexer = []ts.TensorIndexer{
		ts.NewNarrow(0, tensor.MustSize()[0]),
		ts.NewInsertNewAxis(),
	}

	result2 := tensor.Idx(idxs2) // [2, 1, 3]

	var idxs3 []ts.TensorIndexer = []ts.TensorIndexer{
		ts.NewNarrow(0, tensor.MustSize()[0]),
		ts.NewNarrow(0, tensor.MustSize()[1]),
		ts.NewInsertNewAxis(),
	}

	result3 := tensor.Idx(idxs3) // [2, 3, 1]

```
