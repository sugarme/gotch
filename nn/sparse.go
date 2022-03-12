package nn

// Sparse layers

import (
	"github.com/sugarme/gotch/ts"
)

// Configuration option for an embedding layer.
type EmbeddingConfig struct {
	Sparse          bool
	ScaleGradByFreq bool
	WsInit          Init
	PaddingIdx      int64
}

func DefaultEmbeddingConfig() *EmbeddingConfig {
	return &EmbeddingConfig{
		Sparse:          false,
		ScaleGradByFreq: false,
		WsInit:          NewRandnInit(0.0, 1.0),
		PaddingIdx:      -1,
	}
}

// An embedding layer.
//
// An embedding layer acts as a simple lookup table that stores embeddings.
// This is commonly used to store word embeddings.
type Embedding struct {
	Ws     *ts.Tensor
	config *EmbeddingConfig
}

// NewEmbedding creates a new Embedding
func NewEmbedding(vs *Path, numEmbeddings int64, embeddingDim int64, config *EmbeddingConfig) *Embedding {
	return &Embedding{
		Ws:     vs.MustNewVar("weight", []int64{numEmbeddings, embeddingDim}, config.WsInit),
		config: config,
	}
}

// Implement Module, ModuleT interfaces for Embedding:
// =========================================

// Forward implements Module interface for Embedding
func (e *Embedding) Forward(xs *ts.Tensor) *ts.Tensor {
	return ts.MustEmbedding(e.Ws, xs, e.config.PaddingIdx, e.config.ScaleGradByFreq, e.config.Sparse)
}

// ForwardT implements ModuleT interface for Embedding
func (e *Embedding) ForwardT(xs *ts.Tensor, train bool) *ts.Tensor {
	return ts.MustEmbedding(e.Ws, xs, e.config.PaddingIdx, e.config.ScaleGradByFreq, e.config.Sparse)
}
