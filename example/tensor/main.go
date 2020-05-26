package main

import (
	"fmt"
	"reflect"

	t "github.com/sugarme/gotch/torch"
)

func main() {

	t := t.NewTensor()

	fmt.Printf("Type of t: %v\n", reflect.TypeOf(t))
}
