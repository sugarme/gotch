#!/bin/bash

# Env
export GOTCH_VERSION="v0.1.2"
export LIBTORCH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION/libtch"
export LIBRARY_PATH=$LIBTORCH/lib
export CPATH=$CPATH:$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc:$LIBTORCH/include/torch/csrc/api/include
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Precompiled libtorch
wget -O /tmp/libtorch-cxx11-abi-shared-with-deps-1.5.1+cpu.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcpu.zip
sudo unzip /tmp/libtorch-cxx11-abi-shared-with-deps-1.5.1+cpu.zip -d $LIBTORCH

# Update .bashrc
export FILE='$HOME/.bashrc'
export LN_LIBTORCH='export LIBTORCH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION/libtch"'
export LN_LIBRARY_PATH='export LIBRARY_PATH=$LIBTORCH/lib'
export LN_CPATH='export CPATH=$CPATH:$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc:$LIBTORCH/include/torch/csrc/api/include'
export LN_LD_LIBRARY_PATH = 'export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH'
sudo grep -qF -- "$LN_LIBTORCH" "$FILE" || sudo echo "$LN_LIBTORCH" >> "$FILE"
sudo grep -qF -- "$LN_LIBRARY_PATH" "$FILE" || sudo echo "$LN_LIBRARY_PATH" >> "$FILE"
sudo grep -qF -- "$LN_CPATH" "$FILE" || sudo echo "$LN_CPATH" >> "$FILE"
sudo grep -qF -- "$LN_LD_LIBRARY_PATH" "$FILE" || sudo echo "$LN_LD_LIBRARY_PATH" >> "$FILE"

# Update CPU config
sudo rm $LIBTORCH/lib.go
sudo cp $LIBTORCH/lib.go.cpu $LIBTORCH/lib.go
sudo mv $LIBTORCH/dummy_cuda_dependency.cpp $LIBTORCH/dummy_cuda_dependency.cpp.gpu
sudo mv $LIBTORCH/fake_cuda_dependency.cpp.cpu $LIBTORCH/fake_cuda_dependency.cpp
