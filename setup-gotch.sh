#!/bin/bash

GOTCH_VERSION="${GOTCH_VER:-latest}"
LIBTORCH_VERSION="${LIBTORCH_VER:-1.7.0}"
CUDA_VERSION="${CUDA_VER:-10.1}"

if [ $CUDA_VERSION == "cpu" ]
then
  CU_VERSION="cpu"
else
  CU_VERSION="cu${CUDA_VERSION//./}"
fi

# Install gotch
#==============
cwd=$(pwd)
mkdir /tmp/gotch-test
cd /tmp/gotch-test
go mod init "github.com/sugarme/gotch-test"
go get "github.com/sugarme/gotch@$GOTCH_VERSION" 
rm -rf /tmp/gotch-test
cd $cwd

if [ $CUDA_VERSION == "cpu" ]
then
  # prepare C lib for CPU version
  sudo rm $GOTCH/libtch/lib.go
  sudo cp $GOTCH/libtch/lib.go.cpu $GOTCH/libtch/lib.go
  sudo mv $GOTCH/libtch/dummy_cuda_dependency.cpp $GOTCH/libtch/dummy_cuda_dependency.cpp.gpu
  sudo mv $GOTCH/libtch/fake_cuda_dependency.cpp.cpu $GOTCH/libtch/fake_cuda_dependency.cpp
fi

