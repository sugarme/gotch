package tensor_test

import (
	ts "github.com/sugarme/gotch/tensor"
	"io/ioutil"
	"log"
	"path/filepath"
	"reflect"
	"testing"
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
