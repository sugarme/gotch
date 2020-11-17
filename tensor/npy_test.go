package tensor_test

import (
	"reflect"
	"testing"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func TestNpyHeaderParse(t *testing.T) {

	h1 := "{'descr': '<f8', 'fortran_order': False, 'shape': (128,), }"
	want1 := ts.NewNpyHeader(gotch.Double, false, []int64{128})

	testParse(t, want1, h1)

	h2 := "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128), }"
	want2 := ts.NewNpyHeader(gotch.Float, true, []int64{256, 1, 128})

	testParse(t, want2, h2)

	h3, err := ts.ParseNpyHeader(h1)
	if err != nil {
		t.Error(err)
	}
	testToString(t, h1, h3)

	h4, err := ts.ParseNpyHeader(h2)
	if err != nil {
		t.Error(err)
	}
	testToString(t, h2, h4)

	h5 := ts.NewNpyHeader(gotch.Int64, false, []int64{})
	want5 := "{'descr': '<i8', 'fortran_order': False, 'shape': (), }"
	testToString(t, want5, h5)
}

func testParse(t *testing.T, want *ts.NpyHeader, headerStr string) {
	got, err := ts.ParseNpyHeader(headerStr)
	if err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(want, got) {
		t.Errorf("want: %+v\n", want)
		t.Errorf("got: %+v\n", got)
	}
}

func testToString(t *testing.T, want string, h *ts.NpyHeader) {
	got, err := h.ToString()
	if err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(want, got) {
		t.Errorf("want: %+v\n", want)
		t.Errorf("got: %+v\n", got)
	}
}
