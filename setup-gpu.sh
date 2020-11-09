#!/bin/bash

GOTCH_VERSION="${GOTCH_VER:-v0.3.2}"
LIBTORCH_VERSION="${LIBTORCH_VER:-1.7.0}"
CUDA_VERSION="${CUDA_VER:-10.1}"
CU_VERSION="${CUDA_VERSION//./}"

GOTCH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION"
LIBTORCH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION/libtch/libtorch"
LIBRARY_PATH="$LIBTORCH/lib"
CPATH="$CPATH:$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBTORCH/lib:/usr/lib64-nvidia:/usr/local/cuda-${CUDA_VERSION}/lib64"

sudo rm -rf $LIBTORCH
sudo mkdir -p $LIBTORCH


wget -O /tmp/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu${CU_VERSION}.zip https://download.pytorch.org/libtorch/cu${CU_VERSION}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu${CU_VERSION}.zip
sudo unzip /tmp/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu${CU_VERSION}.zip -d $GOTCH/libtch

# update .bashrc
FILE="$HOME/.bashrc"
LN_LIBTORCH="export LIBTORCH=$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION/libtch/libtorch"
LN_LIBRARY_PATH="export LIBRARY_PATH=$LIBTORCH/lib"
LN_CPATH="export CPATH=$CPATH:$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include"
LN_LD_LIBRARY_PATH="export LD_LIBRARY_PATH=$LIBTORCH/lib:/usr/lib64-nvidia:/usr/local/cuda-${CUDA_VERSION}/lib64"
sudo grep -xqF -- "$LN_LIBTORCH" "$FILE" || sudo echo "$LN_LIBTORCH" >> "$FILE"
sudo grep -xqF -- "$LN_LIBRARY_PATH" "$FILE" || sudo echo "$LN_LIBRARY_PATH" >> "$FILE"
sudo grep -xqF -- "$LN_CPATH" "$FILE" || sudo echo "$LN_CPATH" >> "$FILE"
sudo grep -xqF -- "$LN_LD_LIBRARY_PATH" "$FILE" || sudo echo "$LN_LD_LIBRARY_PATH" >> "$FILE"

source "$FILE"
