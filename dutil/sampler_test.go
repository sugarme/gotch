package dutil_test

import (
	// "fmt"
	"reflect"
	"testing"

	"github.com/sugarme/gotch/dutil"
)

func TestSequentialSampler(t *testing.T) {
	s := dutil.NewSequentialSampler(3)
	want := []int{0, 1, 2}
	got := s.Sample()

	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %+v\n", want)
		t.Errorf("Got: %+v\n", got)
	}
}

func TestRandomSampler(t *testing.T) {
	// Default Optional (size and replacement)
	s, err := dutil.NewRandomSampler(10)
	if err != nil {
		t.Errorf("Unexpected error. Got: %v\n", err)
	}

	want := 1
	got := s.BatchSize() // NOTE. BatchSize is always 1 (for SequentialSampler and RandomSampler)
	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %+v\n", want)
		t.Errorf("Got: %+v\n", got)
	}

	// Replacement Opt
	s1, err := dutil.NewRandomSampler(3, dutil.WithReplacement(true))
	if err != nil {
		t.Errorf("Unexpected error. Got: %v\n", err)
	}

	indices := s1.Sample()
	if isDup(indices) {
		t.Errorf("Unexpected duplicated elements. Got: %+v\n", indices)
	}

	// Size option
	size := 3
	s2, err := dutil.NewRandomSampler(10, dutil.WithSize(size))
	if err != nil {
		t.Errorf("Unexpected error. Got: %v\n", err)
	}
	indices = s2.Sample()

	if len(indices) != size {
		t.Errorf("Want size: %v\n", size)
		t.Errorf("Got size: %v\n", len(indices))
	}
}

func isDup(input []int) bool {
	dmap := make(map[int]bool)

	for _, key := range input {
		if _, ok := dmap[key]; ok {
			return true
		}

		dmap[key] = true
	}

	return false
}

func TestNewBatchSampler(t *testing.T) {
	// Valid
	_, err := dutil.NewBatchSampler(10, 3, true)
	if err != nil {
		t.Errorf("Unexpected error. Got: %v\n", err)
	}

	// Invalid batch size
	_, err = dutil.NewBatchSampler(10, 11, true)
	if err == nil {
		t.Errorf("Expected invalid batch size error.")
	}
	_, err = dutil.NewBatchSampler(10, 0, true)
	if err == nil {
		t.Errorf("Expected invalid batch size error.")
	}
}

func TestBatchSampler_BatchSize(t *testing.T) {
	batchSize := 5
	s, err := dutil.NewBatchSampler(10, batchSize, true)
	if err != nil {
		t.Errorf("Unexpected error. Got: %v\n", err)
	}

	got := s.BatchSize()

	if !reflect.DeepEqual(batchSize, got) {
		t.Errorf("Want batch size: %v\n", batchSize)
		t.Errorf("Got batch size: %v\n", got)
	}
}

func TestBatchSampler_Sample(t *testing.T) {
	s1, err := dutil.NewBatchSampler(10, 3, true)
	if err != nil {
		t.Errorf("Unexpected error. Got: %v\n", err)
	}

	indices := s1.Sample()
	want1 := 9
	got1 := len(indices)

	if !reflect.DeepEqual(want1, got1) {
		t.Errorf("Want indices length: %v\n", want1)
		t.Errorf("Got indices length: %v\n", got1)
	}

	// Shuffle
	n := 1000
	s2, err := dutil.NewBatchSampler(n, 3, false, true)
	if err != nil {
		t.Errorf("Unexpected error. Got: %v\n", err)
	}

	want2 := seq(n)
	got2 := s2.Sample()
	if reflect.DeepEqual(want2, got2) {
		t.Errorf("Want indices: %+v\n", want2)
		t.Errorf("Got indices: %+v\n", got2)
	}
}

func seq(n int) []int {
	var s []int
	for i := 0; i < n; i++ {
		s = append(s, i)
	}
	return s
}
