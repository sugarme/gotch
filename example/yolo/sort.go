package main

import (
	"sort"
)

type sortable struct {
	values     []interface{}
	comparator Comparator
}

// Implement `Interface` interface of Go package `sort` for sortable struct:
// =========================================================================

func (s sortable) Len() (retVal int) {
	return len(s.values)
}

func (s sortable) Swap(idx1, idx2 int) {
	s.values[idx1], s.values[idx2] = s.values[idx2], s.values[idx1]
}

func (s sortable) Less(idx1, idx2 int) (retVal bool) {
	return s.comparator.Compare(s.values[idx1], s.values[idx2]) < 0
}

// Sort sorts values in-place.
func Sort(values []interface{}, comparator Comparator) {
	sort.Sort(sortable{values, comparator})
}
