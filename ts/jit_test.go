package ts_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/gotch/ts"
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

// NOTE: comment out for Travis CI
// Uncomment to test locally
/*
 * func TestIValue(t *testing.T) {
 *
 *   roundTrip(nil, t)
 *
 *   roundTrip(int64(45), t)
 *   roundTrip(false, t)
 *   roundTrip(true, t)
 *
 *   roundTrip([]bool{true, false}, t)
 *
 *   roundTrip("Hello", t)
 *   roundTrip([]int64{3, 4}, t)
 *   roundTrip([]float64{3.1, 4.1}, t)
 *
 *   roundTrip([]string{"Abc", "DEF"}, t)
 *   // roundTrip([]int{23, 32}, t)
 *
 *   roundTrip(map[int64]int64{12: 3, 14: 5}, t)
 *   // roundTrip(map[float32]float64{12.3: 3.3, 14.3: 5.3}, t)
 * }
 *  */

func TestModuleForwardTs(t *testing.T) {
	foo, err := ts.ModuleLoad("foo1.gt")
	if err != nil {
		t.Error(err)
	}

	ts1 := ts.TensorFrom([]int64{42})
	ts2 := ts.TensorFrom([]int64{1337})

	res, err := foo.ForwardTs([]ts.Tensor{*ts1, *ts2})
	if err != nil {
		t.Error(err)
	}
	got := int(res.Float64Values()[0])

	want := 1421

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Expected value: %v\n", want)
		t.Errorf("Got value: %v\n", got)
	}

}

func TestModuleForwardIValue(t *testing.T) {
	foo, err := ts.ModuleLoad("foo2.gt")
	if err != nil {
		t.Error(err)
	}

	ts1 := ts.TensorFrom([]int64{42})
	ts2 := ts.TensorFrom([]int64{1337})

	iv1 := ts.NewIValue(*ts1)
	iv2 := ts.NewIValue(*ts2)

	got, err := foo.ForwardIs([]ts.IValue{*iv1, *iv2})
	if err != nil {
		t.Error(err)
	}

	expectedTs1 := ts.TensorFrom([]int64{1421})
	expectedTs2 := ts.TensorFrom([]int64{-1295})
	want := ts.NewIValue([]ts.Tensor{*expectedTs1, *expectedTs2})

	if !reflect.DeepEqual(want.Name(), got.Name()) {
		t.Errorf("Expected Ivalue Name: %v\n", want.Name())
		t.Errorf("Got Ivalue Name: %v\n", got.Name())
	}

	if !reflect.DeepEqual(want.Kind(), got.Kind()) {
		t.Errorf("Expected Ivalue Kind: %v\n", want.Kind())
		t.Errorf("Got Ivalue Kind: %v\n", got.Kind())
	}

	// TODO: compare IValue value
	// NOTE: due to their different pointer values so we need to
	// extract their value and compare
	/* if !reflect.DeepEqual(want, got) {
	 *   t.Errorf("Expected value: %v\n", want)
	 *   t.Errorf("Got value: %v\n", got)
	 * } */

}
