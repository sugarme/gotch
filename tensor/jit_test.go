package tensor_test

import (
	"fmt"
	"reflect"
	"testing"

	ts "github.com/sugarme/gotch/tensor"
)

func roundTrip(v interface{}, t *testing.T) {

	val := ts.NewIValue(v)
	cval, err := val.ToCIValue()
	fmt.Printf("cval at test:%v\n", cval)
	if err != nil {
		t.Logf("Error while converting to CIValue from Go\n")
	}

	val2, err := ts.IValueFromC(cval)
	if err != nil {
		t.Logf("Error while converting to IValue from C: %v\n", err)
	}

	if !reflect.DeepEqual(val, val2) {
		t.Errorf("Expected ivalue (%v) equal to ivalue2 (%v)\n", val, val2)
	}
}

func TestIValue(t *testing.T) {

	roundTrip(nil, t)

	roundTrip(int64(45), t)
	roundTrip(false, t)
	roundTrip(true, t)

	roundTrip("Hello", t)
	roundTrip([]int64{3, 4}, t)
	roundTrip([]float64{3.1, 4.1}, t)
}
