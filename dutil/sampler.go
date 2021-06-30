package dutil

import (
	"fmt"
	"math/rand"
	"time"
)

// Sampler represents an interface to draw sample
// from a dataset by implementing `Sample` method to
// generate a slice of indices of data samples.
type Sampler interface {
	Sample() []int
	BatchSize() int
}

// SequentialSampler represents a method to
// draw sample by its order in the dataset.
type SequentialSampler struct {
	n         int // number of samples
	batchSize int // always = 1
}

// NewSequentialSampler create a new SequentialSampler
//
// n : number of samples in dataset
func NewSequentialSampler(n int) *SequentialSampler {
	return &SequentialSampler{n, 1}
}

// Sample implements Sampler interface.
func (s *SequentialSampler) Sample() []int {
	var indices []int
	for i := 0; i < s.n; i++ {
		indices = append(indices, i)
	}

	return indices
}

func (s *SequentialSampler) BatchSize() int {
	return s.batchSize
}

// RandomSampler represents a method to draw
// a sample randomly from dataset.
type RandomSampler struct {
	n           int  // number of samples
	size        int  // size of sampling
	replacement bool // whether replacement or not
	batchSize   int  // always = 1
}

type RandOptions struct {
	Size        int
	Replacement bool
}

type RandOption func(*RandOptions)

func NewRandOptions(options ...RandOption) RandOptions {
	opts := RandOptions{
		Size:        0,
		Replacement: false,
	}

	for _, o := range options {
		o(&opts)
	}

	return opts
}

func WithSize(size int) RandOption {
	return func(o *RandOptions) {
		o.Size = size
	}
}

func WithReplacement(replacement bool) RandOption {
	return func(o *RandOptions) {
		o.Replacement = replacement
	}
}

// NewRandomSampler creates a new RandomSampler.
//
// n : number of samples in dataset
// size: Optional (default=n). Size of sampling.
// replacement: Optional (default=false). Whether not repeated or repeated samples.
func NewRandomSampler(n int, opt ...RandOption) (*RandomSampler, error) {

	opts := NewRandOptions(opt...)

	if opts.Size > n {
		err := fmt.Errorf("Sampling size can not be greater than number of samples.")
		return nil, err
	}

	size := n // default sampling size = all samples
	if opts.Size != 0 {
		size = opts.Size
	}

	return &RandomSampler{
		n:           n,
		size:        size,
		replacement: opts.Replacement,
		batchSize:   1,
	}, nil
}

// Sample implements Sampler interface.
func (s *RandomSampler) Sample() []int {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	var indices []int

	if !s.replacement {
		for i := 0; i < s.size; i++ {
			idx := r.Intn(s.n)
			indices = append(indices, idx)
		}
		return indices
	}

	if s.size == s.n {
		indices = r.Perm(s.n)
		return indices
	}

	// Random range with fixed length
	var max, min int
	for {
		max = r.Intn(s.n)
		min = max - s.size
		if min >= 0 {
			break
		}
	}

	// Random permutation in a range [min, max)
	// ref. https://stackoverflow.com/questions/35354800
	indices = r.Perm(max - min)
	for i := range indices {
		indices[i] += min
	}

	return indices
}

// BatchSize implements Sampler interface.
// It's always return 1.
func (s *RandomSampler) BatchSize() int {
	return s.batchSize
}

func intRange(n int) []int {
	var r []int
	for i := 0; i < n; i++ {
		r = append(r, i)
	}
	return r
}

// BatchSampler constructs a way to draw batches of samples.
type BatchSampler struct {
	n         int
	batchSize int
	shuffle   bool
	dropLast  bool
}

// NewBatchSampler creates a new BatchSampler.
func NewBatchSampler(n, batchSize int, dropLast bool, shuffleOpt ...bool) (*BatchSampler, error) {
	if batchSize > n || batchSize < 1 {
		err := fmt.Errorf("Invalid batch size: batch size must be equal or greater than 1 and less or equal to number of samples(%v). Got %v", n, batchSize)
		return nil, err
	}

	shuffle := false
	if len(shuffleOpt) > 0 {
		shuffle = shuffleOpt[0]
	}

	return &BatchSampler{
		n:         n,
		batchSize: batchSize,
		shuffle:   shuffle,
		dropLast:  dropLast,
	}, nil
}

// Sample implements Sampler interface
func (s *BatchSampler) Sample() []int {
	var (
		batch   []int
		batches []int
	)

	var indices []int
	switch s.shuffle {
	case false:
		for i := 0; i < s.n; i++ {
			indices = append(indices, i)
		}
	case true:
		// random permutation
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		indices = r.Perm(s.n)
	}

	for _, i := range indices {
		batch = append(batch, i)
		if len(batch) == s.batchSize {
			batches = append(batches, batch...)
			batch = []int{}
		}
	}
	if !s.dropLast {
		batches = append(batches, batch...)
	}

	return batches
}

// BatchSize returns batch size.
func (s *BatchSampler) BatchSize() int {
	return s.batchSize
}
