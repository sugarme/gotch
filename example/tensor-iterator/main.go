package main

import (
	"fmt"
	"reflect"

	wrapper "github.com/sugarme/gotch/wrapper"
)

func main() {
	data := [][]int64{
		{1, 1, 1, 2, 2, 2, 3, 3},
		{1, 1, 1, 2, 2, 2, 4, 4},
	}
	shape := []int64{16}

	ts, err := wrapper.NewTensorFromData(data, shape)
	if err != nil {
		panic(err)
	}

	it, err := ts.Iter(reflect.Float64)
	if err != nil {
		panic(err)
	}

	for i := 0; i < int(it.Len); i++ {
		v := it.Next()
		fmt.Println(v)
	}

}
