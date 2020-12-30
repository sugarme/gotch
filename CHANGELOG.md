# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- [#...]: Fix a bug with...

### Changed
- [#...]: 

### Added
- [#...]: 


## [0.1.8]

### Changed
- [#10]: `ts.Drop()` and `ts.MustDrop()` now can call multiple times without panic

## [0.1.9]

### Changed
- Reverse changes [#10] to original.

## [0.1.10]

### Added
- Added `tensor.SaveMultiNew`


## [0.2.0]

### Changed
- Convert all APIs to using **Pointer Receiver**

### Added
- Added drawing image label at `example/yolo` example
- Added some example images and README files for `example/yolo` and `example/neural-style-transfer`

## [0.3.0]

### Changed
- Updated to Pytorch C++ APIs v1.7.0
- Switched back to `lib.AtoAddParametersOld` as the `ato_add_parameters` has not been implemented correctly. Using the updated API will cause optimizer stops working.

## [0.3.1]

### Changed
- Changed to use `map[string]*Tensor` at `nn/varstore.go`
- Changed to use `*Path` argument of `NewLayerNorm` method at `nn/layer-norm.go`
- Lots of clean-up return variables i.e. retVal, err

## [0.3.2]

### Added
- [#6]: Go native tensor print using `fmt.Formatter` interface. Now, a tensor can be printed out like: `fmt.Printf("%.3f", tensor)` (for float type)

## [0.3.3]

### Fixed
- nn/sequential: fixed missing case number of layers = 1 causing panic
- nn/varstore: fixed(nn/varstore): fixed nil pointer at LoadPartial due to not break loop

# [0.3.4]

### Added
- [#4] Automatically download and install Libtorch and setup environment variables.

[#10]: https://github.com/sugarme/gotch/issues/10
[#6]: https://github.com/sugarme/gotch/issues/6
[#4]: https://github.com/sugarme/gotch/issues/4

