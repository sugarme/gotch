package torch

// #cgo CXXFLAGS: -std=c++14 -I${SRCDIR} -O3 -Wall -g -Wno-sign-compare -Wno-unused-function -I/Library/Developer/CommandLineTools/usr/include/c++/v1 -I/usr/local/include -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include
// #cgo LDFLAGS: -lstdc++ -L/opt/libtorch/lib  -ltorch -lcaffe2 -lc10
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
import "C"
