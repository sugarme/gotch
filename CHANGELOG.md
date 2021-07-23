# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [Nofix]
- ctype `long` caused compiling error in MacOS as noted on [#44]. Not working on linux box.

## [0.3.14]
- Fixed temporary fix huge number of learning group returned from C at `libtch/tensor.go AtoGetLearningRates`
- Fixed incorrect `nn.AdamWConfig` and some documentation.
- Fixed - reworked on `vision.ResNet` and `vision.DenseNet` to fix incorrect layers and memory leak
- Changed `dutil.DataLoader.Reset()` to reshuffle when resetting DataLoader if flag is true
- Changed `dutil.DataLoader.Next()`. Deleted case batch size == 1 to make consistency by always returning items in a slice `[]element dtype` even with batchsize = 1.
- Added `nn.CrossEntropyLoss` and `nn.BCELoss` 
- Fixed `tensor.ForwardIs` return `Tuple` and `TensorList` instead of always returning `TensorList`
- Changed exporting augment options and make ColorJitter forward output dtype `uint8` for chaining with other augment options.
- [#45] Fixed `init/RandInt` incorrect initialization 
- [#48] Fixed `init/RandInit` when init with mean = 0.0.

## [0.3.13]
- Fixed multiple memory leakage at `vision/image.go`
- Fixed memory leakage at `dutil/dataloader.go`
- Fixed multiple memory leakage at `efficientnet.go`
- Added `dataloader.Len()` method
- Fixed deleting input tensor inside function at `tensor/other.go`  `tensor.CrossEntropyForLogits` and `tensor.AccuracyForLogits`
- Added warning to `varstore.LoadPartial` when mismatched tensor shapes between source and varstore.
- Fixed incorrect message mismatched tensor shape at `nn.Varstore.Load`
- Fixed incorrect y -> x at `vision/aug/affine.go` getParam func
- Fixed double free tensor at `vision/aug/function.go` Equalize func.
- Changed `vision/aug` all input image should be `uint8` (Byte) dtype and transformed output has the same dtype (uint8) so that `Compose()` can compose any transformer options.
- Fixed wrong result of `aug.RandomAdjustSharpness`
- Fixed memory leak at `aug/function.getAffineGrid`
- Changed `vision/aug` and correct ColorJitter
- Changed `vision/aug` and correct Resize
- Changed `dutil/sampler` to accept batchsize from 1.
- Fixed double free in `vision/image.go/resizePreserveAspectRatio`

## [0.3.12]

Skip this tag

## [0.3.11]

Same as [0.3.10]

## [0.3.10]
- Update installation at README.md
- [#38] fixed JIT model
- Added Optimizer Learning Rate Schedulers
- Added AdamW Optimizer

## [0.3.9]
- [#24], [#26]: fixed memory leak.
- [#30]: fixed varstore.Save() randomly panic - segmentfault
- [#32]: nn.Seq Forward return nil tensor if length of layers = 1
- [#36]: resolved image augmentation

## [0.3.8]

### Fixed
- [#20]: fixed IValue.Value() method return `[]interface{}` instead of `[]Tensor`

## [0.3.7]

### Added
- Added trainable JIT Module APIs and example/jit-train. Now, a Python Pytorch model `.pt` can be loaded then continue training/fine-tuning in Go.

## [0.3.6]

### Added
- Added `dutil` sub-package that serves Pytorch  `DataSet` and `DataLoader` concepts

## [0.3.5]

### Added
- Added function `gotch.CudaIfAvailable()`. NOTE that: `device := gotch.NewCuda().CudaIfAvailable()` will throw error if CUDA is not available. 

### Changed
- Switched back to install libtorch inside gotch library as go init() function is triggered after cgo called.

## [0.3.4]

### Added
- [#4] Automatically download and install Libtorch and setup environment variables.

## [0.3.2]

### Added
- [#6]: Go native tensor print using `fmt.Formatter` interface. Now, a tensor can be printed out like: `fmt.Printf("%.3f", tensor)` (for float type)

## [0.3.3]

### Fixed
- nn/sequential: fixed missing case number of layers = 1 causing panic
- nn/varstore: fixed(nn/varstore): fixed nil pointer at LoadPartial due to not break loop

## [0.3.1]

### Changed
- Changed to use `map[string]*Tensor` at `nn/varstore.go`
- Changed to use `*Path` argument of `NewLayerNorm` method at `nn/layer-norm.go`
- Lots of clean-up return variables i.e. retVal, err

## [0.3.0]

### Changed
- Updated to Pytorch C++ APIs v1.7.0
- Switched back to `lib.AtoAddParametersOld` as the `ato_add_parameters` has not been implemented correctly. Using the updated API will cause optimizer stops working.

## [0.2.0]

### Changed
- Convert all APIs to using **Pointer Receiver**

### Added
- Added drawing image label at `example/yolo` example
- Added some example images and README files for `example/yolo` and `example/neural-style-transfer`

## [0.1.10]

### Added
- Added `tensor.SaveMultiNew`

## [0.1.9]

### Changed
- Reverse changes [#10] to original.

## [0.1.8]

### Changed
- [#10]: `ts.Drop()` and `ts.MustDrop()` now can call multiple times without panic


[#10]: https://github.com/sugarme/gotch/issues/10
[#6]: https://github.com/sugarme/gotch/issues/6
[#4]: https://github.com/sugarme/gotch/issues/4
[#20]: https://github.com/sugarme/gotch/issues/20
[#24]: https://github.com/sugarme/gotch/issues/24
[#26]: https://github.com/sugarme/gotch/issues/26
[#30]: https://github.com/sugarme/gotch/issues/30
[#32]: https://github.com/sugarme/gotch/issues/32
[#44]: https://github.com/sugarme/gotch/issues/44
[#45]: https://github.com/sugarme/gotch/issues/45
[#48]: https://github.com/sugarme/gotch/issues/48
