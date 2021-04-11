#!/bin/bash

GOTCH_VERSION="${GOTCH_VER:-v0.3.8}"
CUDA_VERSION="${CUDA_VER:-10.1}"
GOTCH_PATH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION"

# Install gotch
#==============
echo "GOPATH:'$GOPATH'"
echo "GOTCH_VERSION: '$GOTCH_VERSION'"
echo "CUDA_VERSION: '$CUDA_VERSION'"

cwd=$(pwd)
sudo rm -rf /tmp/gotch-test
mkdir /tmp/gotch-test
cd /tmp/gotch-test
go mod init "github.com/sugarme/gotch-test"
go get "github.com/sugarme/gotch@$GOTCH_VERSION" 
rm -rf /tmp/gotch-test
cd $cwd

if [ $CUDA_VERSION=="cpu" ]
then
  # prepare C lib for CPU version
  sudo rm $GOTCH_PATH/libtch/lib.go
  sudo cp $GOTCH_PATH/libtch/lib.go.cpu $GOTCH_PATH/libtch/lib.go
  sudo mv $GOTCH_PATH/libtch/dummy_cuda_dependency.cpp $GOTCH_PATH/libtch/dummy_cuda_dependency.cpp.gpu
  sudo mv $GOTCH_PATH/libtch/fake_cuda_dependency.cpp.cpu $GOTCH_PATH/libtch/fake_cuda_dependency.cpp
fi


