# GoTch [![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)[![Go.Dev reference](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white&style=flat-square)](https://pkg.go.dev/github.com/sugarme/gotch?tab=doc)[![Travis CI](https://api.travis-ci.org/sugarme/gotch.svg?branch=master)](https://travis-ci.org/sugarme/gotch)[![Go Report Card](https://goreportcard.com/badge/github.com/sugarme/gotch)](https://goreportcard.com/report/github.com/sugarme/gotch) 


## Overview

- **GoTch** is a C++ Libtorch Go binding for developing and implementing deep learning projects in Go.
- This package is to create a thin wrapper of Libtorch to make use of its tensor APIs and CUDA support while implementing as much idiomatic Go as possible. 

## Dependencies

- **Libtorch** C++ v1.5.0 library of [Pytorch](https://pytorch.org/)


## Installation

- **CPU**

    Default values: `LIBTORCH_VER=1.5.1` and `GOTCH_VER=v0.1.7`

    ```bash
    go get -u github.com/sugarme/gotch@v0.1.7
    bash ${GOPATH}/pkg/mod/github.com/sugarme/gotch@v0.1.7/setup-cpu.sh

    ```

- **GPU**

    Default values: `LIBTORCH_VER=1.5.1`, `CUDA_VER=10.1` and `GOTCH_VER=v0.1.7`

    ```bash
    go get -u github.com/sugarme/gotch@v0.1.7
    bash ${GOPATH}/pkg/mod/github.com/sugarme/gotch@v0.1.7/setup-gpu.sh

    ```

## Examples

### Basic tensor operations

```go

import (
	"fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func basicOps() {

	// Initiate a tensor
	tensor := ts.MustArange1(ts.FloatScalar(0), ts.FloatScalar(12), gotch.Float, gotch.CPU).MustView([]int64{3, 4}, true)

	tensor.Print()
    //  0   1   2   3
    //  4   5   6   7
    //  8   9  10  11
    // [ CPUFloatType{3,4} ]

	fmt.Printf("tensor values: %v\n", tensor.Float64Values())
    //tensor values: [0 1 2 3 4 5 6 7 8 9 10 11]

	fmt.Printf("tensor dtype: %v\n", tensor.DType())
    //tensor dtype: float32

	fmt.Printf("tensor shape: %v\n", tensor.MustSize())
    //tensor shape: [3 4]

	fmt.Printf("tensor element number: %v\n", tensor.Numel())
    //tensor element number: 12

	// Delete a tensor (NOTE. tensor is created in C memory and will need to free up manually.)
	tensor.MustDrop()

	// Basic tensor operations
	ts1 := ts.MustArange(ts.IntScalar(6), gotch.Int64, gotch.CPU).MustView([]int64{2, 3}, true)
	defer ts1.MustDrop()
	ts2 := ts.MustOnes([]int64{3, 4}, gotch.Int64, gotch.CPU)
	defer ts2.MustDrop()

	mul := ts1.MustMatmul(ts2, false)
	defer mul.MustDrop()
	fmt.Println("ts1: ")
	ts1.Print()
	fmt.Println("ts2: ")
	ts2.Print()
	fmt.Println("mul tensor (ts1 x ts2): ")
	mul.Print()

    //ts1: 
    // 0  1  2
    // 3  4  5
    //[ CPULongType{2,3} ]
    //ts2: 
    // 1  1  1  1
    // 1  1  1  1
    // 1  1  1  1
    //[ CPULongType{3,4} ]
    //mul tensor (ts1 x ts2): 
    //  3   3   3   3
    // 12  12  12  12
    //[ CPULongType{2,4} ]


	// In-place operation
	ts3 := ts.MustOnes([]int64{2, 3}, gotch.Float, gotch.CPU)
	fmt.Println("Before:")
	ts3.Print()
	ts3.MustAdd1_(ts.FloatScalar(2.0))
	fmt.Printf("After (ts3 + 2.0): \n")
	ts3.Print()
	ts3.MustDrop()

    //Before:
    // 1  1  1
    // 1  1  1
    //[ CPUFloatType{2,3} ]
    //After (ts3 + 2.0): 
    // 3  3  3
    // 3  3  3
    //[ CPUFloatType{2,3} ]

}

```

### Simplified Convolutional neural network

```go

    import (
        "github.com/sugarme/gotch"
        "github.com/sugarme/gotch/nn"
        ts "github.com/sugarme/gotch/tensor"
    )

    type Net struct {
        conv1 nn.Conv2D
        conv2 nn.Conv2D
        fc    nn.Linear
    }

    func newNet(vs nn.Path) Net {
        conv1 := nn.NewConv2D(vs, 1, 16, 2, nn.DefaultConv2DConfig())
        conv2 := nn.NewConv2D(vs, 16, 10, 2, nn.DefaultConv2DConfig())
        fc := nn.NewLinear(vs, 10, 10, nn.DefaultLinearConfig())

        return Net{
            conv1,
            conv2,
            fc,
        }
    }

    func (n Net) ForwardT(xs ts.Tensor, train bool) (retVal ts.Tensor) {
        xs = xs.MustView([]int64{-1, 1, 8, 8}, false)

        outC1 := xs.Apply(n.conv1)
        outMP1 := outC1.MaxPool2DDefault(2, true)
        defer outMP1.MustDrop()

        outC2 := outMP1.Apply(n.conv2)
        outMP2 := outC2.MaxPool2DDefault(2, true)
        outView2 := outMP2.MustView([]int64{-1, 10}, true)
        defer outView2.MustDrop()

        outFC := outView2.Apply(&n.fc)

        return outFC.MustRelu(true)

    }

    func main() {

        vs := nn.NewVarStore(gotch.CPU)
        net := newNet(vs.Root())

        xs := ts.MustOnes([]int64{8, 8}, gotch.Float, gotch.CPU)

        logits := net.ForwardT(xs, false)
        logits.Print()
    }

    // 0.0000  0.0000  0.0000  0.2477  0.2437  0.0000  0.0000  0.0000  0.0000  0.0171
    //[ CPUFloatType{1,10} ]


```

- Real application examples can be found at [example folder](example/README.md) 

## Getting Started

- [Documentations](docs/README.md)

- See [pkg.go.dev](https://pkg.go.dev/github.com/sugarme/gotch?tab=doc) for detail APIs 


## License

GoTch is Apache 2.0 licensed.


## Acknowledgement

- This project has been inspired and used many concepts from [tch-rs](https://github.com/LaurentMazare/tch-rs)
    Libtorch Rust binding. 



