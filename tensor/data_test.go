package tensor_test

import (
	// "fmt"
	"io/ioutil"
	"log"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func TestTextData_NewTextData(t *testing.T) {
	// Create text file to test
	filename := "/tmp/test.txt"
	filePath, err := filepath.Abs(filename)
	if err != nil {
		log.Fatal(err)
	}

	txt := `héllo`
	// txt := "h\xC3\xA9llo"
	err = ioutil.WriteFile(filePath, []byte(txt), 0644)
	if err != nil {
		log.Fatal(err)
	}

	textData, err := ts.NewTextData(filename)
	if err != nil {
		log.Fatal(err)
	}

	wantData := []float64{0, 1, 2, 3, 3, 4}
	gotData := textData.CloneData().Float64Values()

	if !reflect.DeepEqual(wantData, gotData) {
		t.Errorf("Want data: %v\n", wantData)
		t.Errorf("Got data: %v\n", gotData)
	}

	wantLabelLen := int64(5)
	gotLabelLen := textData.Labels()

	if !reflect.DeepEqual(wantLabelLen, gotLabelLen) {
		t.Errorf("Want label len: %v\n", wantLabelLen)
		t.Errorf("Got label len: %v\n", gotLabelLen)
	}

	wantChar := rune(195)
	gotChar := textData.LabelForChar(int64(1))

	if !reflect.DeepEqual(wantChar, gotChar) {
		t.Errorf("Want Char: %q\n", wantChar)
		t.Errorf("Got Char: %q\n", gotChar)
	}
}

func TestTextDataIter(t *testing.T) {

	filename := "/tmp/test.txt"
	filePath, err := filepath.Abs(filename)
	if err != nil {
		log.Fatal(err)
	}

	txt := "01234567890123456789"
	// txt := `hello world`
	err = ioutil.WriteFile(filePath, []byte(txt), 0644)
	if err != nil {
		log.Fatal(err)
	}

	textData, err := ts.NewTextData(filename)
	if err != nil {
		log.Fatal(err)
	}

	iter := textData.IterShuffle(2, 5) // (seqLen, batchSize)
	// fmt.Printf("indexesLen: %v\n", iter.IndexesLen)
	// fmt.Printf("data: %v\n", iter.Data.Int64Values())
	// fmt.Printf("Indexes: %v\n", iter.Indexes.Int64Values())

	for {
		xs, ok := iter.Next()
		if !ok {
			break
		}

		size := xs.MustSize()
		idxCol := ts.NewNarrow(0, size[0]) // column
		idxCol1 := ts.NewNarrow(0, 1)      // first column
		// idxNextEl := ts.NewSelect(1)
		col1 := xs.Idx([]ts.TensorIndexer{idxCol, idxCol1})
		// nextEl := xs.Idx([]ts.TensorIndexer{idxNextEl})
		// col1PlusOne := ts.MustStack([]ts.Tensor{col1, nextEl}, 0)
		// col1Fmod := col1PlusOne.MustFmod(ts.IntScalar(10), false)
		col1Fmod := col1.MustFmod(ts.IntScalar(10), false)

		// t.Errorf("col1 shape: %v\n", col1Fmod.MustSize())

		idxCol2 := ts.NewNarrow(1, 2)
		col2 := xs.Idx([]ts.TensorIndexer{idxCol, idxCol2})
		// t.Errorf("col2 shape: %v\n", col2.MustSize())

		pow := col1Fmod.MustSub(col2, true).MustPow(ts.IntScalar(2), true)
		sum := pow.MustSum(gotch.Float, true)

		// Will pass if there's no panic
		vals := sum.Int64Values()
		t.Logf("sum: %v\n", vals)
	}

}
