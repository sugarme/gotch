# Gotch [![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)[![Go.Dev reference](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white&style=flat-square)](https://pkg.go.dev/github.com/sugarme/gotch?tab=doc) 


## Overview

`gotch` creates a thin wrapper to Pytorch C++ APIs (Libtorch) to make use of its already optimized C++ tensor APIs (3039) and dynamic graph computation with CUDA support and provides idiomatic Go APIs for developing and implementing Deep Learning in Go.

**Some features are**
- [x] Comprehensive Pytorch tensor APIs (2525)
- [x] Fully featured Pytorch dynamic graph computation
- [x] JIT interface to run model trained/saved using PyTorch Python API
- [x] Load pretrained Pytorch models and run inference
- [x] Pure Go APIs to build and train neural network models with both CPU and GPU support
- [x] Most recent image models
- [ ] NLP Language models - [Transformer](https://github.com/sugarme/transformer) in separate package built with **gotch** and [pure Go Tokenizer](https://github.com/sugarme/tokenizer).

`gotch` is in active development mode and may have API breaking changes. Feel free to pull request, report issues or discuss any concerns. All contributions are welcome. 

`gotch` current version is **v0.9.1**

## Dependencies

- **Libtorch** C++ v2.10.0 library of [Pytorch](https://pytorch.org/)
- Clang-17/Clang++-17 compilers

## Installation

- Default CUDA version is `11.8` if CUDA is available otherwise using CPU version.
- Default Pytorch C++ API version is `2.10.0`

**NOTE**: `libtorch` will be installed at **`/usr/local/lib`**

### CPU

#### Step 1: Setup libtorch (skip this step if a valid libtorch already installed in your machine!)

```bash
    wget https://github.com/sugarme/gotch/releases/download/v0.9.0/setup-libtorch.sh
    chmod +x setup-libtorch.sh
    export CUDA_VER=cpu && bash setup-libtorch.sh
```

**Update Environment**: in Debian/Ubuntu, add/update the following lines to `.bashrc` file

```bash
    export GOTCH_LIBTORCH="/usr/local/lib/libtorch"
    export LIBRARY_PATH="$LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
    export CPATH="$CPATH:$GOTCH_LIBTORCH/lib:$GOTCH_LIBTORCH/include:$GOTCH_LIBTORCH/include/torch/csrc/api/include"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
```

#### Step 2: Setup gotch

```bash
    wget https://github.com/sugarme/gotch/releases/download/v0.9.0/setup-gotch.sh
    chmod +x setup-gotch.sh
    export CUDA_VER=cpu && export GOTCH_VER=v0.9.1 && bash setup-gotch.sh
```

### GPU

**NOTE**: make sure your machine has working CUDA. 
- Check version: `nvidia-smi`
- [Install nvidia driver here](https://www.nvidia.com/Download/Find.aspx?lang=en)
- [Install CUDA here](https://developer.nvidia.com/cuda-downloads)
- [Install CuDNN here](https://developer.nvidia.com/rdp/cudnn-download#)

#### Step 1: Setup libtorch (skip this step if a valid libtorch already installed in your machine!)

```bash
    wget https://github.com/sugarme/gotch/releases/download/v0.9.0/setup-libtorch.sh
    chmod +x setup-libtorch.sh

    export CUDA_VER=11.8 && bash setup-libtorch.sh
```

**Update Environment**: in Debian/Ubuntu, add/update the following lines to `.bashrc` file

```bash
    export GOTCH_LIBTORCH="/usr/local/lib/libtorch"
    export LIBRARY_PATH="$LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
    export CPATH="$CPATH:$GOTCH_LIBTORCH/lib:$GOTCH_LIBTORCH/include:$GOTCH_LIBTORCH/include/torch/csrc/api/include"
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GOTCH_LIBTORCH/lib:/usr/lib64-nvidia:/usr/local/cuda-${CUDA_VERSION}/lib64"
```

#### Step 2: Setup gotch

```bash
    wget https://github.com/sugarme/gotch/releases/download/v0.9.0/setup-gotch.sh
    chmod +x setup-gotch.sh
    export CUDA_VER=11.8 && export GOTCH_VER=v0.9.1 && bash setup-gotch.sh
```

## Examples

### Basic tensor operations

```go
import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

func basicOps() {

xs := ts.MustRand([]int64{3, 5, 6}, gotch.Float, gotch.CPU)
fmt.Printf("%8.3f\n", xs)
fmt.Printf("%i", xs)

/*
(1,.,.) =
   0.391     0.055     0.638     0.514     0.757     0.446  
   0.817     0.075     0.437     0.452     0.077     0.492  
   0.504     0.945     0.863     0.243     0.254     0.640  
   0.850     0.132     0.763     0.572     0.216     0.116  
   0.410     0.660     0.156     0.336     0.885     0.391  

(2,.,.) =
   0.952     0.731     0.380     0.390     0.374     0.001  
   0.455     0.142     0.088     0.039     0.862     0.939  
   0.621     0.198     0.728     0.914     0.168     0.057  
   0.655     0.231     0.680     0.069     0.803     0.243  
   0.853     0.729     0.983     0.534     0.749     0.624  

(3,.,.) =
   0.734     0.447     0.914     0.956     0.269     0.000  
   0.427     0.034     0.477     0.535     0.440     0.972  
   0.407     0.945     0.099     0.184     0.778     0.058  
   0.482     0.996     0.085     0.605     0.282     0.671  
   0.887     0.029     0.005     0.216     0.354     0.262  



TENSOR INFO:
        Shape:          [3 5 6]
        DType:          float32
        Device:         {CPU 1}
        Defined:        true
*/

// Basic tensor operations
ts1 := ts.MustArange(ts.IntScalar(6), gotch.Int64, gotch.CPU).MustView([]int64{2, 3}, true)
defer ts1.MustDrop()
ts2 := ts.MustOnes([]int64{3, 4}, gotch.Int64, gotch.CPU)
defer ts2.MustDrop()

mul := ts1.MustMatmul(ts2, false)
defer mul.MustDrop()

fmt.Printf("ts1:\n%2d", ts1)
fmt.Printf("ts2:\n%2d", ts2)
fmt.Printf("mul tensor (ts1 x ts2):\n%2d", mul)

/*
ts1:
 0   1   2  
 3   4   5  

ts2:
 1   1   1   1  
 1   1   1   1  
 1   1   1   1  

mul tensor (ts1 x ts2):
 3   3   3   3  
12  12  12  12  
*/


// In-place operation
ts3 := ts.MustOnes([]int64{2, 3}, gotch.Float, gotch.CPU)
fmt.Printf("Before:\n%v", ts3)
ts3.MustAddScalar_(ts.FloatScalar(2.0))
fmt.Printf("After (ts3 + 2.0):\n%v", ts3)

/*
Before:
1  1  1  
1  1  1  

After (ts3 + 2.0):
3  3  3  
3  3  3  
*/
}
```

### Simplified Convolutional neural network

```go
import (
    "fmt"

    "github.com/sugarme/gotch"
    "github.com/sugarme/gotch/nn"
    "github.com/sugarme/gotch/ts"
)

type Net struct {
    conv1 *nn.Conv2D
    conv2 *nn.Conv2D
    fc    *nn.Linear
}

func newNet(vs *nn.Path) *Net {
    conv1 := nn.NewConv2D(vs, 1, 16, 2, nn.DefaultConv2DConfig())
    conv2 := nn.NewConv2D(vs, 16, 10, 2, nn.DefaultConv2DConfig())
    fc := nn.NewLinear(vs, 10, 10, nn.DefaultLinearConfig())

    return &Net{
        conv1,
        conv2,
        fc,
    }
}

func (n Net) ForwardT(xs *ts.Tensor, train bool) *ts.Tensor {
    xs = xs.MustView([]int64{-1, 1, 8, 8}, false)

    outC1 := xs.Apply(n.conv1)
    outMP1 := outC1.MaxPool2DDefault(2, true)
    defer outMP1.MustDrop()

    outC2 := outMP1.Apply(n.conv2)
    outMP2 := outC2.MaxPool2DDefault(2, true)
    outView2 := outMP2.MustView([]int64{-1, 10}, true)
    defer outView2.MustDrop()

    outFC := outView2.Apply(n.fc)
    return outFC.MustRelu(true)
}

func main() {

    vs := nn.NewVarStore(gotch.CPU)
    net := newNet(vs.Root())

    xs := ts.MustOnes([]int64{8, 8}, gotch.Float, gotch.CPU)

    logits := net.ForwardT(xs, false)
    fmt.Printf("Logits: %0.3f", logits)
}

//Logits: 0.000  0.000  0.000  0.225  0.321  0.147  0.000  0.207  0.000  0.000
```

## Play with `gotch` on Google Colab or locally

- [Tensor Initiation](example/basic) <a href="https://colab.research.google.com/github/sugarme/nb/blob/master/tensor/tensor-initiation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- [Tensor Indexing](example/basic) <a href="https://colab.research.google.com/github/sugarme/nb/blob/master/tensor/tensor-indexing.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- [MNIST](example/mnist) <a href="https://colab.research.google.com/github/sugarme/nb/blob/master/mnist/mnist.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- [YOLO v3 model infering](example/yolo) <a href="https://colab.research.google.com/github/sugarme/nb/blob/master/yolo/yolo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- [RNN model training](example/char-rnn)
- [CIFAR model training](example/cifar)
- [JIT ResNet18 Torch Script model load and inference](example/jit)
- [Neural style transfer](example/neural-style-transfer)
- [Image pretrained models - inference](example/pretrained-model)
- [Translation](example/translation)
- [Convert Pytorch Python model to Go](example/convert-model)
- [Load Python Pytorch JIT model then train/finetune in Go](example/jit-train)
- [Image augmentation](example/augmentation)

## Getting Started

- See [pkg.go.dev](https://pkg.go.dev/github.com/sugarme/gotch?tab=doc) for APIs detail.

For unit tests use:
- make test

## PyTorch 2.10.0 Upgrade Notes

This version includes critical fixes for PyTorch 2.10.0 compatibility:
- Fixed gradient state management (thread-local in PyTorch 2.10.0)
- Fixed FFI type conversion bugs (C.int → Go int)

To validate FFI conversions:
```bash
make ffi-validate
```

## License

`gotch` is Apache 2.0 licensed.

## Acknowledgement

- This project has been inspired and used many concepts from [tch-rs](https://github.com/LaurentMazare/tch-rs)
    Libtorch Rust binding. 
