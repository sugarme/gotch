#!/bin/bash

# Env
GOTCH_VERSION="${GOTCH_VER:-v0.3.0}"
LIBTORCH_VERSION="${LIBTORCH_VER:-1.7.0}"

GOTCH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION"
LIBTORCH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION/libtch/libtorch"
LIBRARY_PATH="$LIBTORCH/lib"
CPATH="$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc:$LIBTORCH/include/torch/csrc/api/include"
LD_LIBRARY_PATH="$LIBTORCH/lib"

# Precompiled libtorch
sudo rm -rf $LIBTORCH
sudo mkdir -p $LIBTORCH
wget -O /tmp/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip
sudo unzip /tmp/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip -d $GOTCH/libtch

# Update .bashrc
FILE="$HOME/.bashrc"
LN_LIBTORCH="export LIBTORCH=$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION/libtch/libtorch"
LN_LIBRARY_PATH="export LIBRARY_PATH=$LIBTORCH/lib"
LN_CPATH="export CPATH=$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc:$LIBTORCH/include/torch/csrc/api/include"
LN_LD_LIBRARY_PATH="export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH"
sudo grep -xqF -- "$LN_LIBTORCH" "$FILE" || sudo echo "$LN_LIBTORCH" >> "$FILE"
sudo grep -xqF -- "$LN_LIBRARY_PATH" "$FILE" || sudo echo "$LN_LIBRARY_PATH" >> "$FILE"
sudo grep -xqF -- "$LN_CPATH" "$FILE" || sudo echo "$LN_CPATH" >> "$FILE"
sudo grep -xqF -- "$LN_LD_LIBRARY_PATH" "$FILE" || sudo echo "$LN_LD_LIBRARY_PATH" >> "$FILE"

sudo rm $GOTCH/libtch/lib.go
sudo cp $GOTCH/libtch/lib.go.cpu $GOTCH/libtch/lib.go
sudo mv $GOTCH/libtch/dummy_cuda_dependency.cpp $GOTCH/libtch/dummy_cuda_dependency.cpp.gpu
sudo mv $GOTCH/libtch/fake_cuda_dependency.cpp.cpu $GOTCH/libtch/fake_cuda_dependency.cpp

source "$FILE"
