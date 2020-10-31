package nn_test

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

func TestVarStoreEntry(t *testing.T) {
	vs := nn.NewVarStore(gotch.CPU)
	root := vs.Root()
	e1 := root.Entry("key")
	t1 := e1.OrZeros([]int64{3, 1, 4})
	e2 := root.Entry("key")
	t2 := e2.OrZeros([]int64{1, 5, 9})

	wantT1 := []int64{3, 1, 4}
	wantT2 := []int64{3, 1, 4}

	gotT1 := t1.MustSize()
	gotT2 := t2.MustSize()

	if !reflect.DeepEqual(wantT1, gotT1) {
		t.Errorf("Expected t1 size: %v\n", wantT1)
		t.Errorf("Got t1 size: %v\n", gotT1)
	}

	if !reflect.DeepEqual(wantT2, gotT2) {
		t.Errorf("Expected t2 size: %v\n", wantT2)
		t.Errorf("Got t2 size: %v\n", gotT2)
	}
}

// NOTE: comment out for working on Travis.
// uncomment to test locally

func TestSaveLoad(t *testing.T) {
	filename := "vsload.test"
	filenameAbs, err := filepath.Abs(filename)
	if err != nil {
		panic(err)
	}

	add := func(vs *nn.Path) (*ts.Tensor, *ts.Tensor) {
		subA := vs.Sub("a")
		subB := subA.Sub("b")
		v := subB.Ones("t2", []int64{3})
		u := vs.Zeros("t1", []int64{4})

		wa := vs.Sub("a")
		wb := wa.Sub("b")
		wc := wb.Sub("ccc")
		_ = wc.Ones("t123", []int64{3})
		_ = wc.Ones("t123", []int64{3})

		return u, v
	}

	vs1 := nn.NewVarStore(gotch.CPU)
	vs2 := nn.NewVarStore(gotch.CPU)

	u1, v1 := add(vs1.Root())
	u2, v2 := add(vs2.Root())

	ts.NoGrad(func() {
		u1.Add1_(ts.FloatScalar(42.0))
		v1.Mul1_(ts.FloatScalar(2.0))
	})

	wantU1 := float64(42.0)
	wantV1 := float64(2.0)
	wantU2 := float64(0.0)
	wantV2 := float64(1.0)

	gotU1 := u1.MustMean(gotch.Float, false).Float64Values()[0]
	gotV1 := v1.MustMean(gotch.Float, false).Float64Values()[0]
	gotU2 := u2.MustMean(gotch.Float, false).Float64Values()[0]
	gotV2 := v2.MustMean(gotch.Float, false).Float64Values()[0]

	if !reflect.DeepEqual(wantU1, gotU1) {
		t.Errorf("Expected u1: %v\n", wantU1)
		t.Errorf("Got u1: %v\n", gotU1)
	}
	if !reflect.DeepEqual(wantV1, gotV1) {
		t.Errorf("Expected v1: %v\n", wantV1)
		t.Errorf("Got v1: %v\n", gotV1)
	}

	if !reflect.DeepEqual(wantU2, gotU2) {
		t.Errorf("Expected u2: %v\n", wantU2)
		t.Errorf("Got u2: %v\n", gotU2)
	}
	if !reflect.DeepEqual(wantV2, gotV2) {
		t.Errorf("Expected v2: %v\n", wantV2)
		t.Errorf("Got v2: %v\n", gotV2)
	}

	err = vs1.Save(filenameAbs)
	if err != nil {
		panic(err)
	}

	err = vs2.Load(filenameAbs)
	if err != nil {
		panic(err)
	}

	wantU2 = float64(42.0)
	wantV2 = float64(2.0)
	gotU2 = u2.MustMean(gotch.Float, false).Float64Values()[0]
	gotV2 = v2.MustMean(gotch.Float, false).Float64Values()[0]

	if !reflect.DeepEqual(wantU1, gotU1) {
		t.Errorf("Expected u1: %v\n", wantU1)
		t.Errorf("Got u1: %v\n", gotU1)
	}
	if !reflect.DeepEqual(wantU2, gotU2) {
		t.Errorf("Expected u2: %v\n", wantU2)
		t.Errorf("Got u2: %v\n", gotU2)
	}
	if !reflect.DeepEqual(wantV2, gotV2) {
		t.Errorf("Expected v2: %v\n", wantV2)
		t.Errorf("Got v2: %v\n", gotV2)
	}

	err = os.Remove(filenameAbs)
	if err != nil {
		t.Errorf("Failed deleting varstore saved file: %v\n", filenameAbs)
	}
}
