package ts

// include/c10/core/Layout.h
type Layout int8

const (
	Strided    Layout = iota // 0
	Sparse                   // 1
	SparseCsr                // 2
	Mkldnn                   // 3
	SparseCsc                // 4
	SparseBsr                // 5
	SparseBsc                // 6
	NumOptions               // 7
)
