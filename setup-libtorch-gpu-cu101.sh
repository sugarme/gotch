#!/bin/bash

export GOTCH_VERSION="v0.1.1"
export LIBTORCH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION/libtch"
export LIBRARY_PATH=$LIBTORCH/lib
export CPATH=$CPATH:$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBTORCH/lib:/usr/lib64-nvidia:/usr/local/cuda-10.1/lib64

sudo rm -rf $LIBTORCH
sudo mkdir -p $LIBTORCH
wget -O /tmp/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu101.zip https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu101.zip
sudo unzip /tmp/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu101.zip -d $LIBTORCH

# update .bashrc
FILE='$HOME/.bashrc'
LN_LIBTORCH = 'export LIBTORCH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION/libtch"'
LN_LIBRARY_PATH = 'export LIBRARY_PATH=$LIBTORCH/lib'
LN_CPATH = 'export CPATH=$CPATH:$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include'
LN_LD_LIBRARY_PATH = 'export LD_LIBRARY_PATH=$LIBTORCH/lib:/usr/lib64-nvidia:/usr/local/cuda-10.1/lib64'
sudo grep -qF -- "$LN_LIBTORCH" "$FILE" || sudo echo "$LN_LIBTORCH" >> "$FILE"
sudo grep -qF -- "$LN_LIBRARY_PATH" "$FILE" || sudo echo "$LN_LIBRARY_PATH" >> "$FILE"
sudo grep -qF -- "$LN_CPATH" "$FILE" || sudo echo "$LN_CPATH" >> "$FILE"
sudo grep -qF -- "$LN_LD_LIBRARY_PATH" "$FILE" || sudo echo "$LN_LD_LIBRARY_PATH" >> "$FILE"
