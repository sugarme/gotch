#!/bin/bash

GOTCH_VERSION="${GOTCH_VER:-v0.2.1}"
LIBTORCH_VERSION="${LIBTORCH_VER:-1.7.0}"
CUDA_VERSION="${CUDA_VER:-10.1}"
CU_VERSION="${CUDA_VERSION//./}"

export GOTCH="$HOME/projects/sugarme/gotch"
export LIBTORCH="$HOME/projects/sugarme/gotch/libtch/libtorch"
export LIBRARY_PATH="$LIBTORCH/lib"
export CPATH="$CPATH:$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBTORCH/lib:/usr/lib64-nvidia:/usr/local/cuda-${CUDA_VERSION}/lib64"

sudo rm -rf $LIBTORCH
sudo mkdir -p $LIBTORCH

wget -O /tmp/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu${CU_VERSION}.zip https://download.pytorch.org/libtorch/cu${CU_VERSION}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu${CU_VERSION}.zip
sudo unzip /tmp/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu${CU_VERSION}.zip -d $GOTCH/libtch
