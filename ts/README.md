# BENCHMARK

## Convolution 2D

Ref.
1. https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
2. https://github.com/soumith/convnet-benchmarks

Benchmark tensor operation `conv2d` forward propagation:
- input shape: `[32, 64, 64, 64]`
- kernel:            `[64, 3, 3]`

```bash
goos: linux
goarch: amd64
pkg: github.com/sugarme/gotch/ts
cpu: Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz
BenchmarkConv2dCPU-8                 100          21198303 ns/op
BenchmarkConv2dCUDA-8                100           2201213 ns/op

CUDA 11.1
CuDNN 8.0.5
```

## `gotch`
```bash
name          time/op
Conv2dCPU-8   21.2ms ± 0%
Conv2dCUDA-8  2.20ms ± 0%
```

## Python `Pytorch` 1.11

```bash
conv2d-CPU(x):   56.7 ms
conv2d-CUDA(x):   38.0 ms
```

benchmark Python code below

```python
import torch
import timeit

x = torch.randn(32, 64, 64, 64)

def conv2dCPU(x):
    conv1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
    return conv1(x)

def conv2dCUDA(x):
    x = x.cuda()
    conv1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False).cuda()
    return conv1(x)

t0 = timeit.Timer(
    stmt='conv2dCPU(x)',
    setup='from __main__ import conv2dCPU',
    globals={'x': x})

t1 = timeit.Timer(
    stmt='conv2dCUDA(x)',
    setup='from __main__ import conv2dCUDA',
    globals={'x': x})

print(f'conv2d-CPU(x):  {t0.timeit(100) / 100 * 1e3:>5.1f} ms')
print(f'conv2d-CUDA(x):  {t1.timeit(100) / 100 * 1e3:>5.1f} ms')
```
