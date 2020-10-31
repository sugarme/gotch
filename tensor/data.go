package tensor

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/sugarme/gotch"
)

// Iter2 is an iterator over a pair of tensors which have the same first dimension
// size.
// The typical use case is to iterate over batches. Each batch is a pair
// containing a (potentially random) slice of each of the two input
// tensors.
type Iter2 struct {
	xs                   *Tensor
	ys                   *Tensor
	batchIndex           int64
	batchSize            int64
	totalSize            int64
	device               gotch.Device
	returnSmallLastBatch bool
}

// NewIter2 returns a new iterator.
//
// This takes as input two tensors which first dimension must match. The
// returned iterator can be used to range over mini-batches of data of
// specified size.
// An error is returned if `xs` and `ys` have different first dimension
// sizes.
//
// # Arguments
//
// * `xs` - the features to be used by the model.
// * `ys` - the targets that the model attempts to predict.
// * `batch_size` - the size of batches to be returned.
func NewIter2(xs, ys *Tensor, batchSize int64) (*Iter2, error) {
	var (
		iter *Iter2
		err  error
	)

	totalSize := xs.MustSize()[0]
	if ys.MustSize()[0] != totalSize {
		err = fmt.Errorf("Different dimension for the two inputs: %v - %v", xs.MustSize(), ys.MustSize())
		return nil, err
	}

	// xsClone, err := xs.ZerosLike(false)
	// if err != nil {
	// log.Fatal(err)
	// }
	// xsClone.Copy_(xs)
	//
	// ysClone, err := ys.ZerosLike(false)
	// if err != nil {
	// log.Fatal(err)
	// }
	// ysClone.Copy_(ys)

	iter = &Iter2{
		xs: xs.MustShallowClone(),
		ys: ys.MustShallowClone(),
		// xs:                   xsClone,
		// ys:                   ysClone,
		batchIndex:           0,
		batchSize:            batchSize,
		totalSize:            totalSize,
		returnSmallLastBatch: false,
	}

	return iter, nil
}

// MustNewIter2 returns a new iterator.
//
// This takes as input two tensors which first dimension must match. The
// returned iterator can be used to range over mini-batches of data of
// specified size.
// Panics if `xs` and `ys` have different first dimension sizes.
//
// # Arguments
//
// * `xs` - the features to be used by the model.
// * `ys` - the targets that the model attempts to predict.
// * `batch_size` - the size of batches to be returned.
func MustNewIter2(xs, ys *Tensor, batchSize int64) *Iter2 {
	iter, err := NewIter2(xs, ys, batchSize)

	if err != nil {
		log.Fatal(err)
	}

	return iter
}

// Shuffle shuffles the dataset.
//
// The iterator would still run over the whole dataset but the order in
// which elements are grouped in mini-batches is randomized.
func (it *Iter2) Shuffle() {
	index := MustRandperm(it.totalSize, gotch.Int64, gotch.CPU)

	it.xs = it.xs.MustIndexSelect(0, index, true)
	it.ys = it.ys.MustIndexSelect(0, index, true)

	index.MustDrop()
}

// ToDevice transfers the mini-batches to a specified device.
func (it *Iter2) ToDevice(device gotch.Device) *Iter2 {
	it.device = device
	return it
}

// ReturnSmallLastBatch when set, returns the last batch even if smaller than the batch size.
func (it *Iter2) ReturnSmallLastBatch() *Iter2 {
	it.returnSmallLastBatch = true
	return it
}

type Iter2Item struct {
	Data  *Tensor
	Label *Tensor
}

// Next implements iterator for Iter2
func (it *Iter2) Next() (item Iter2Item, ok bool) {
	start := it.batchIndex * it.batchSize
	size := it.batchSize
	if it.totalSize-start < it.batchSize {
		size = it.totalSize - start
	}

	if (size <= 0) || (!it.returnSmallLastBatch && size < it.batchSize) {
		// err = fmt.Errorf("Last small batch error")
		return item, false
	} else {
		it.batchIndex += 1

		// Indexing
		narrowIndex := NewNarrow(start, start+size)

		return Iter2Item{
			Data:  it.xs.Idx(narrowIndex),
			Label: it.ys.Idx(narrowIndex),
		}, true
	}
}

func (it *Iter2) Drop() {
	it.xs.MustDrop()
	it.ys.MustDrop()
}

// TextData represent text data in tensor of runes (uint8)
// and its corresponding string
type TextData struct {
	Data         *Tensor // frequency (occurence) of byte value from input text
	CharForLabel []rune  // unique rune values from input text
}

// TextDataIter is a text data interator
type TextDataIter struct {
	Data       *Tensor
	SeqLen     int64
	BatchIndex int64
	BatchSize  int64
	Indexes    *Tensor
	IndexesLen int64
}

// NewTextData creates a text dataset from a file
//
// It reads text input from file to `[]byte` buffer
// - Loops over each byte
// - first byte will be labelled `0`
// - next byte if exist will be labelled with existing label (index), otherwise
// will labelled with new label(index)
// Data: tensor of labels
// CharForLabel: []rune (unique runes from text input)
func NewTextData(filename string) (*TextData, error) {
	filePath, err := filepath.Abs(filename)
	if err != nil {
		return nil, err
	}

	r, err := os.Open(filePath)

	buffer, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}

	var labelForChar map[byte]uint8 = make(map[byte]uint8, 0)
	var charForLabel []rune
	var dataIndexes []uint8

	for _, runeVal := range buffer {
		if len(labelForChar) == 0 {
			labelForChar[runeVal] = 0
			dataIndexes = append(dataIndexes, 0)
			charForLabel = append(charForLabel, rune(runeVal))
		} else {
			label, ok := labelForChar[runeVal]
			if !ok {
				newLabel := uint8(len(labelForChar))
				labelForChar[runeVal] = newLabel
				dataIndexes = append(dataIndexes, newLabel)
				charForLabel = append(charForLabel, rune(runeVal))
			} else {
				dataIndexes = append(dataIndexes, label)
			}
		}
	}

	data := MustOfSlice(dataIndexes)

	return &TextData{
		Data:         data,
		CharForLabel: charForLabel,
	}, nil
}

// Labels returns the number of different `character` (rune) used by the dataset.
func (td *TextData) Labels() (retVal int64) {
	return int64(len(td.CharForLabel))
}

// Data returns a shallow copy of the data.
func (td *TextData) CloneData() *Tensor {
	return td.Data.MustShallowClone()
}

// LabelForChar returns a corresponding `char` (rune) for
// specified label input
func (td *TextData) LabelForChar(label int64) rune {
	return td.CharForLabel[int(label)]
}

// IterShuffle returns a batch iterator over the dataset.
// Each sample is made of seq_len characters.
func (td *TextData) IterShuffle(seqLen int64, batchSize int64) *TextDataIter {

	indexesLen := td.Data.MustSize()[0] - seqLen + 1

	return &TextDataIter{
		Data:       td.Data.MustShallowClone(),
		SeqLen:     seqLen,
		BatchIndex: 0,
		BatchSize:  batchSize,
		Indexes:    MustRandperm(indexesLen, gotch.Int64, gotch.CPU),
		IndexesLen: indexesLen,
	}
}

// Next implements iterator for TextDataIter
func (tdi *TextDataIter) Next() (*Tensor, bool) {
	start := tdi.BatchIndex * tdi.BatchSize
	size := min(tdi.BatchSize, tdi.IndexesLen-start)

	if size < tdi.BatchSize {
		return nil, false
	}

	tdi.BatchIndex += 1

	narrowIdx := NewNarrow(start, start+size)
	indexesTs := tdi.Indexes.Idx(narrowIdx)

	indexes := indexesTs.Int64Values()
	indexesTs.MustDrop()

	var batch []Tensor

	for _, idx := range indexes {
		narrowIdx := NewNarrow(idx, idx+tdi.SeqLen)
		idxTs := tdi.Data.Idx(narrowIdx)
		batch = append(batch, *idxTs)
	}

	retVal := MustStack(batch, 0)

	// Delete intermediate tensors
	for _, xs := range batch {
		xs.MustDrop()
	}

	return retVal, true
}

func min(v1, v2 int64) int64 {
	if v1 < v2 {
		return v1
	}
	return v2
}
