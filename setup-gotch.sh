#!/bin/bash

GOTCH_VERSION="${GOTCH_VER:-v0.7.0}"
CUDA_VERSION="${CUDA_VER:-11.3}"

if [ -z $GOPATH ]; then
  GOPATH="$HOME/go"
fi
GOTCH_PATH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION"

# Install gotch
#==============
echo "GOPATH:'$GOPATH'"
echo "GOTCH_VERSION: '$GOTCH_VERSION'"
echo "CUDA_VERSION: '$CUDA_VERSION'"

cwd=$(pwd)
GOTCH_TEST_DIR="/tmp/gotch-test"
if [ -d $GOTCH_TEST_DIR ]; then
  sudo rm -rf $GOTCH_TEST_DIR
fi
mkdir $GOTCH_TEST_DIR
cd $GOTCH_TEST_DIR
go mod init "github.com/sugarme/gotch-test"
go get -d "github.com/sugarme/gotch@$GOTCH_VERSION" 
rm -rf $GOTCH_TEST_DIR
cd $cwd

# Setup gotch for CUDA or non-CUDA device:
#=========================================
GOTCH_LIB_FILE="$GOTCH_PATH/libtch/lib.go"
if [ -f $GOTCH_LIB_FILE ]
then
  echo "$GOTCH_LIB_FILE existing. Deleting..."
  sudo rm $GOTCH_LIB_FILE
fi

# Create files for CUDA or non-CUDA device
if [ $CUDA_VERSION == "cpu" ]; then
  echo "creating $GOTCH_LIB_FILE for CPU"
  sudo tee -a $GOTCH_LIB_FILE > /dev/null <<EOT
package libtch

// #cgo CFLAGS: -I${SRCDIR} -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CFLAGS: -I/usr/local/include
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
// #cgo LDFLAGS: -lstdc++ -ltorch -lc10 -ltorch_cpu -L/lib64
// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR} -g -O3
// #cgo CFLAGS: -I${SRCDIR}/libtorch/lib -I${SRCDIR}/libtorch/include -I${SRCDIR}/libtorch/include/torch/csrc/api/include -I${SRCDIR}/libtorch/include/torch/csrc
// #cgo LDFLAGS: -L${SRCDIR}/libtorch/lib
// #cgo CXXFLAGS: -I${SRCDIR}/libtorch/lib -I${SRCDIR}/libtorch/include -I${SRCDIR}/libtorch/include/torch/csrc/api/include -I${SRCDIR}/libtorch/include/torch/csrc
import "C"
EOT
else
  echo "creating $GOTCH_LIB_FILE for GPU"
  sudo tee -a $GOTCH_LIB_FILE > /dev/null <<EOT
package libtch

// #cgo LDFLAGS: -lstdc++ -ltorch -lc10 -ltorch_cpu
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn -lcaffe2_nvrtc -lnvrtc-builtins -lnvrtc -lnvToolsExt -lc10_cuda -ltorch_cuda
// #cgo CFLAGS: -I${SRCDIR} -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR} -g -O3
import "C"
EOT
fi

sudo ldconfig
