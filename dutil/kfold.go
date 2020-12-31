package dutil

import (
	"fmt"
	"math/rand"
	"sort"
)

// KFold represents a struct helper to
// split data into partitions.
type KFold struct {
	n       int
	nfolds  int
	shuffle bool
	//seed int
}

// Fold represents a partitions with
// 2 fields: indexes of train set and test set.
type Fold struct {
	Train []int
	Test  []int
}

type KFoldOptions struct {
	NFolds  int  // number of folds
	Shuffle bool // whether suffling before splitting
}

type KFoldOption func(*KFoldOptions)

func NewKFoldOptions(options ...KFoldOption) KFoldOptions {
	opts := KFoldOptions{
		NFolds:  5,
		Shuffle: false,
	}

	for _, o := range options {
		o(&opts)
	}

	return opts
}

func WithNFolds(nfolds int) KFoldOption {
	return func(o *KFoldOptions) {
		o.NFolds = nfolds
	}
}

func WithKFoldShuffle(shuffle bool) KFoldOption {
	return func(o *KFoldOptions) {
		o.Shuffle = shuffle
	}
}

// NewKFold creates a new KFold struct.
func NewKFold(n int, opt ...KFoldOption) (*KFold, error) {
	opts := NewKFoldOptions(opt...)

	if opts.NFolds < 2 {
		err := fmt.Errorf("nfolds must be at least 2. Got: %v\n", opts.NFolds)
		return nil, err
	}

	if opts.NFolds > n {
		err := fmt.Errorf("nfolds cannot be greater than number of samples (%v). Got: %v\n", n, opts.NFolds)
		return nil, err
	}

	return &KFold{
		n:       n,
		nfolds:  opts.NFolds,
		shuffle: opts.Shuffle,
	}, nil
}

func (kf *KFold) Split() []Fold {
	odd := kf.n % kf.nfolds
	nsamples := kf.n - odd
	fsize := nsamples / kf.nfolds
	var indices []int

	allIndices := rand.Perm(kf.n)
	// Drop last odd-time elements
	indices = allIndices[:nsamples]

	if !kf.shuffle {
		sort.Ints(indices)
	}

	// Split to train, test sets
	var (
		folds [][]int
		fold  []int
	)
	for i := 0; i < nsamples; i++ {
		fold = append(fold, i)
		if len(fold) == fsize {
			folds = append(folds, fold)
			fold = []int{}
		}
	}

	var splits []Fold
	for i := 0; i < kf.nfolds; i++ {
		test := folds[i]
		var trainFolds [][]int
		trainFolds = append(trainFolds, folds[:i]...)
		trainFolds = append(trainFolds, folds[i+1:]...)
		var train []int
		for _, f := range trainFolds {
			train = append(train, f...)
		}

		split := Fold{
			Test:  values(indices, test),
			Train: values(indices, train),
		}
		splits = append(splits, split)
	}

	return splits
}

func contains(data []int, item int) bool {
	for _, el := range data {
		if el == item {
			return true
		}
	}

	return false
}

func values(data []int, keys []int) []int {
	var vals []int
	for _, k := range keys {
		v := data[k]
		vals = append(vals, v)
	}
	return vals
}
