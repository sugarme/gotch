package tensor_test

import (
	"reflect"
	"testing"

	ts "github.com/sugarme/gotch/tensor"
)

func roundTrip(v interface{}, t *testing.T) {

	val := ts.NewIValue(v)
	cval, err := val.ToCIValue()
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

	roundTrip([]bool{true, false}, t)

	roundTrip("Hello", t)
	roundTrip([]int64{3, 4}, t)
	roundTrip([]float64{3.1, 4.1}, t)

	roundTrip([]string{"Abc", "DEF"}, t)
	// roundTrip([]int{23, 32}, t)

	roundTrip(map[int64]int64{12: 3, 14: 5}, t)
	// roundTrip(map[float32]float64{12.3: 3.3, 14.3: 5.3}, t)
}

func TestModuleLoad(t *testing.T) {
	foo, err := ts.ModuleLoad("foo1.pt")
	if err != nil {
		t.Error(err)
	}

	ts1 := ts.TensorFrom([]int64{42})
	ts2 := ts.TensorFrom([]int64{1337})

	res, err := foo.ForwardTs([]ts.Tensor{ts1, ts2})
	if err != nil {
		t.Error(err)
	}
	got := int(res.Values()[0])

	want := 1421

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Expected value: %v\n", want)
		t.Errorf("Got value: %v\n", got)
	}

}
