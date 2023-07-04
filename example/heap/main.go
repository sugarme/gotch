package main

import (
	"fmt"
	"runtime/metrics"
	// "math/rand"
	// "time"
	// "github.com/sugarme/gotch"
	// "github.com/sugarme/gotch/ts"
)

// Run with: `go build  -gcflags="-m=3"`

//go:noinline
func main() {
	s := []metrics.Sample{{Name: "/gc/stack/starting-size:bytes"}}
	metrics.Read(s)
	fmt.Printf("Initial stack size: %d\n", s[0].Value.Uint64())

	// x, err := ts.Randn([]int64{2, 3, 224, 224}, gotch.Float, gotch.CPU)
	// if err != nil {
	// panic(err)
	// }
	// fmt.Printf("x: %v\n", x.Name())

	// x := ts.MustOfSlice([]float32{1, 2, 3})
	// fmt.Printf("x: %v\n", x.Name())

	x := new(foo)
	// x := newFoo()
	// x := &foo{
	// name: "foo",
	// f:    &foo1{foo1Name: "foo1"},
	// }

	fmt.Printf("x: %q\n", x.name)

	// b := new(ts.Tensor)
	// fmt.Printf("b: %v\n", b)

	// time.Sleep(time.Second * 2)
}

type foo struct {
	data [1e4]interface{} // 10_000 * 4 = 40_000 bytes
	name string
	// f    *foo1
}

type foo1 struct {
	foo1Name string
}

func newFoo() *foo {
	return new(foo)
}

// func newData() []float32 {
// // n := 3 * 224 * 224 * 12
// n := 3
// data := make([]float32, n)
// for i := 0; i < n; i++ {
// data[i] = rand.Float32()
// }
//
// return data
// }
